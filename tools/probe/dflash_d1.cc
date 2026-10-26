// dflash_d1 - spec 19 §6 D1 (plan 19b Task 3), the engine's half: the DFlash drafter on the card beside the loaded
// target drafts at teacher-forced anchors of tap dumps (tools/oracle/dump_taps.py's format - 19a's P0 dumps, or the
// engine's own once 19c writes them), and every draft's readback is written as one JSON line for
// tools/oracle/dflash_d1.py, which runs tools/oracle/dflash_ref.py on the SAME taps and holds the two to D1's bars
// (top-16 sets tie-aware, scores cosine >= 0.9999 bf16 / >= 0.999 int8, walk ids equal at T = 0).
//
//   dflash_d1 <target snapshot> <drafter> <out.jsonl> [--format int8|bf16|int4] [--head int8|bf16]
//             [--draft-vocab 32k|64k|128k [--draft-vocab-ids FILE]] [--anchors N] [--k K] [--max-len L]
//             [--draft-q raw|capped] <dump.taps.safetensors>...
//
// Per dump (monotonic anchors, one ring): the projection list fills the ring with the dump's taps (8 positions a
// replay, from max(tap_from, first anchor - 2047)), and at each anchor p (N anchors evenly over the dump's
// continuation, dflash_accept.py's range n_prompt .. n - 8) the draft list runs at (p, ids[p], K, T 0). A line:
//   {"dump": name, "p": p, "anchor": id, "k": K, "ids": [...], "chosen": [...], "cand": [[16]...], "unary": [[16]...],
//    "scores": [[16]...]}
// The first line is the run's: {"run": {"format", "head", "draft_vocab", "vocab_used", "v_ids": [...] (the draft
// vocabulary's ids, when one is used), "launches": [draft, project], "load": the drafter's load line}}.
//
// Spec 23f: a Muse-Glimmer target (loader::is_muse_checkpoint) runs Muse-Glimmer-30B-assistant through
// runtime::dflash::MuseDrafter instead - the target's embedding and head from loader::load_muse_glimmer with ONE layer
// (layers_override 1: no 17 GB of layers for a drafter test), --format int4|int8|bf16 (default int4), --k 1..15
// (default 15), --draft-q raw|capped (default raw: the readback's unary / scores are the raw head logits, or
// 20 tanh(z x 0.196 / 20)); the projection 16 positions a replay from max(tap_from, first anchor - 2048) (the
// window |q - k| <= 2048); anchors over n_prompt .. n - K - 1; every draft the whole 16-row block (MuseDrafter), its
// first K rows written. The run line adds "form": "muse" and "q".
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/dflash_loader.h"
#include "loader/draft_vocab.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "loader/muse_loader.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/dflash/muse_drafter.h"

namespace {

struct Dump {
  std::string name;
  uint32_t n = 0, n_prompt = 0, tap_from = 0;
  std::vector<uint32_t> layers;
  std::vector<int32_t> ids;
  std::vector<std::vector<uint16_t>> taps;   // per tap layer [n - tap_from][hidden]
};

Dump read_dump(const std::string& path, uint32_t hidden) {
  loader::MappedFile f(path);
  uint64_t hlen = 0;
  std::memcpy(&hlen, f.data(), 8);
  const common::json::Value h =
      common::json::parse(std::string_view(reinterpret_cast<const char*>(f.data()) + 8, hlen));
  const common::json::Value& md = h.at("__metadata__");
  Dump d;
  d.name = md.at("name").str();
  d.n = uint32_t(std::stoul(md.at("n").str()));
  d.n_prompt = uint32_t(std::stoul(md.at("n_prompt").str()));
  d.tap_from = uint32_t(std::stoul(md.at("tap_from").str()));
  std::string ls = md.at("layers").str();
  for (size_t at = 0; at < ls.size();) {
    const size_t c = ls.find(',', at);
    d.layers.push_back(uint32_t(std::stoul(ls.substr(at, c - at))));
    at = c == std::string::npos ? ls.size() : c + 1;
  }
  const size_t data0 = 8 + hlen;
  for (const auto& [name, t] : loader::SafetensorsSet::parse_header(f.data(), f.size())) {
    const uint8_t* p = f.data() + data0 + t.begin;
    if (name == "ids") {
      d.ids.resize((t.end - t.begin) / 4);
      std::memcpy(d.ids.data(), p, t.end - t.begin);
    }
  }
  d.taps.resize(d.layers.size());
  for (const auto& [name, t] : loader::SafetensorsSet::parse_header(f.data(), f.size()))
    for (size_t i = 0; i < d.layers.size(); ++i)
      if (name == "tap.L" + std::to_string(d.layers[i])) {
        if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[1] != hidden || t.shape[0] != d.n - d.tap_from)
          throw std::runtime_error(path + ": " + name + " is not bf16 [n - tap_from][hidden]");
        d.taps[i].resize(t.end - t.begin);
        std::memcpy(d.taps[i].data(), f.data() + data0 + t.begin, t.end - t.begin);
      }
  if (d.ids.size() != d.n) throw std::runtime_error(path + ": ids is not [n]");
  return d;
}

std::string arr(const uint32_t* v, uint32_t n) {
  std::string s = "[";
  for (uint32_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}
std::string farr(const float* v, uint32_t n) {
  std::string s = "[";
  char b[32];
  for (uint32_t i = 0; i < n; ++i) {
    std::snprintf(b, sizeof b, "%.9g", double(v[i]));
    s += (i ? "," : "") + std::string(b);
  }
  return s + "]";
}

// Spec 23f: anchors evenly over [lo, hi], at most `anchors` (dflash_accept.py's anchors_of).
std::vector<uint32_t> anchors_over(uint32_t lo, uint32_t hi, uint32_t anchors) {
  std::vector<uint32_t> ps;
  const uint32_t cnt = std::min(anchors, hi - lo + 1);
  for (uint32_t i = 0; i < cnt; ++i) ps.push_back(lo + uint32_t(uint64_t(hi - lo) * i / std::max(1u, cnt - 1)));
  ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
  return ps;
}

// Spec 23f: the Muse-Glimmer arm (the header's note).
int run_muse(l0::Context& ctx, const std::string& snap, const std::string& drafter, const std::string& out_path,
             loader::DraftFormat fmt, loader::LmHeadForm head, uint32_t anchors, uint32_t K, uint32_t max_len,
             runtime::dflash::MuseQ qf, const std::vector<std::string>& dumps) {
  namespace rd = runtime::dflash;
  const uint32_t layers = loader::muse_checkpoint_desc(snap).layers;
  const model::MuseGlimmerDesc one = loader::muse_checkpoint_desc(snap, 1);
  std::vector<l0::Context*> devs = {&ctx};
  const loader::MuseWeights m = loader::load_muse_glimmer(devs, snap, max_len, model::MusePlacement::one(one),
                                                          head == loader::LmHeadForm::Int8, 1);
  const loader::MuseDevicePart& part = m.parts.at(0);
  const model::DFlashTargetShape shape{"Muse-Glimmer-30B", layers, m.desc.hidden, part.lm_head->shape.N};
  rd::MuseDrafter dr(ctx, loader::load_drafter(ctx, drafter, shape, *part.embed, fmt), max_len, K, qf);
  l0::Mem control(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(control.ptr(), 0, sizeof(runtime::Control));
  dr.attach(ctx, *part.embed, *part.lm_head, control);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto replay = [&](l0::CmdList& list) {
    q.execute(list, &fence);
    fence.wait();
  };
  std::ofstream out(out_path);
  out << "{\"run\": {\"form\": \"muse\", \"q\": \"" << rd::muse_q_name(qf) << "\", \"format\": \""
      << loader::draft_format_name(fmt) << "\", \"head\": \"" << loader::lm_head_form_name(head)
      << "\", \"draft_vocab\": 0, \"vocab_used\": " << shape.vocab_rows << ", \"k\": " << K << ", \"launches\": ["
      << dr.draft_step()->kernel_count << ", " << dr.project_step().kernel_count << "], \"load\": \""
      << dr.weights().line << "\", \"v_ids\": []}}\n";
  const uint32_t H = dr.desc().hidden, ntap = uint32_t(dr.tap_layers().size()), M = kernels::dflash::kMuseM;
  double t_draft = 0;
  size_t n_draft = 0;
  for (const std::string& path : dumps) {
    const Dump d = read_dump(path, H);
    if (d.layers != dr.tap_layers()) throw std::runtime_error(path + ": tap layers differ from the drafter's");
    if (d.n < K + 2 || d.n_prompt + K + 1 > d.n) continue;
    const std::vector<uint32_t> ps = anchors_over(std::max(d.n_prompt, d.tap_from + 1), d.n - K - 1, anchors);
    dr.reset(imm);   // a cold ring per dump
    const uint32_t win = dr.window();
    uint32_t next_pos = std::max(d.tap_from, ps.front() > win ? ps.front() - win : 0u);
    for (uint32_t p : ps) {
      while (next_pos < p) {   // project [next_pos, min(p, next_pos + 16))
        const uint32_t n = std::min(M, p - next_pos);
        std::vector<uint16_t> rows(size_t(M) * ntap * H, 0);
        for (uint32_t r = 0; r < n; ++r)
          for (uint32_t i = 0; i < ntap; ++i)
            std::memcpy(&rows[(size_t(r) * ntap + i) * H], &d.taps[i][size_t(next_pos + r - d.tap_from) * H], H * 2);
        imm.copy(dr.taps().ptr(), rows.data(), rows.size() * 2);
        replay(dr.projection(next_pos, n));
        next_pos += n;
      }
      runtime::Control* c = control.as<runtime::Control>();
      c->pos = p;
      c->n_active = 1;
      c->cur_token[0] = uint32_t(d.ids[p]);
      const auto t0 = std::chrono::steady_clock::now();
      const rd::MuseDraftOut& o = dr.draft(replay, K);
      t_draft += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++n_draft;
      std::string cand = "[", un = "[", sc = "[";
      for (uint32_t j = 0; j < K; ++j) {
        cand += (j ? "," : "") + arr(o.cand[j], 16);
        un += (j ? "," : "") + farr(o.unary[j], 16);
        sc += (j ? "," : "") + farr(o.scores[j], 16);
      }
      out << "{\"dump\": \"" << d.name << "\", \"p\": " << p << ", \"anchor\": " << d.ids[p] << ", \"k\": " << K
          << ", \"ids\": " << arr(o.ids, K) << ", \"chosen\": " << arr(o.chosen, K) << ", \"cand\": " << cand
          << "], \"unary\": " << un << "], \"scores\": " << sc << "]}\n";
    }
    std::printf("%s: %zu anchors\n", d.name.c_str(), ps.size());
  }
  std::printf("dflash_d1 (Muse-Glimmer, drafter %s, head %s, q %s): %zu drafts, %.3f ms a draft list (host-timed, one "
              "replay, fence included); %s\n", loader::draft_format_name(fmt), loader::lm_head_form_name(head),
              rd::muse_q_name(qf), n_draft, n_draft ? 1e3 * t_draft / double(n_draft) : 0.0, out_path.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: dflash_d1 <target> <drafter> <out.jsonl> [options] <dump>...\n");
    return 2;
  }
  const std::string target = argv[1], drafter = argv[2], out_path = argv[3];
  loader::DraftFormat fmt = loader::DraftFormat::Int8;
  bool fmt_given = false;
  loader::LmHeadForm head = loader::LmHeadForm::Int8;
  uint32_t dv = 0, anchors = 16, K = 0, max_len = 8192;
  runtime::dflash::MuseQ qf = runtime::dflash::MuseQ::Raw;
  std::string dv_ids;
  std::vector<std::string> dumps;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--format") {
      if (!loader::parse_draft_format(next(), fmt)) throw std::runtime_error("--format int8|bf16|int4");
      fmt_given = true;
    } else if (a == "--draft-q") {
      if (!runtime::dflash::parse_muse_q(next(), qf)) throw std::runtime_error("--draft-q raw|capped");
    } else if (a == "--head") {
      if (!loader::parse_lm_head_form(next(), head)) throw std::runtime_error("--head int8|bf16");
    } else if (a == "--draft-vocab") {
      if (!loader::parse_draft_vocab(next(), dv)) throw std::runtime_error("--draft-vocab 32k|64k|128k");
    } else if (a == "--draft-vocab-ids") {
      dv_ids = next();
    } else if (a == "--anchors") {
      anchors = uint32_t(std::stoul(next()));
    } else if (a == "--k") {
      K = uint32_t(std::stoul(next()));
    } else if (a == "--max-len") {
      max_len = uint32_t(std::stoul(next()));
    } else {
      dumps.push_back(a);
    }
  }
  const std::string snap = loader::resolve_snapshot(target);
  l0::Context ctx(0);
  if (loader::is_muse_checkpoint(snap)) {   // spec 23f
    if (dv) throw std::runtime_error("--draft-vocab is not built for Muse-Glimmer's drafter");
    if (!fmt_given) fmt = loader::DraftFormat::Int4G64;
    if (K == 0) K = kernels::dflash::kMuseR;
    if (K < 1 || K > kernels::dflash::kMuseR) throw std::runtime_error("--k is 1..15 for Muse-Glimmer");
    return run_muse(ctx, snap, drafter, out_path, fmt, head, anchors, K, max_len, qf, dumps);
  }
  if (K == 0) K = kernels::dflash::kR;
  if (K < 1 || K > kernels::dflash::kR) throw std::runtime_error("--k is 1..7");
  if (fmt == loader::DraftFormat::Int4G64) throw std::runtime_error("--format int4 is Muse-Glimmer's (spec 23f)");
  loader::DraftVocabSpec dvs;
  if (dv) {
    dvs.size = dv;
    dvs.added = loader::added_token_ids_file(snap + "tokenizer.json");
    if (!dv_ids.empty()) dvs.ranked = loader::read_ranked_ids(dv_ids);
  }
  // A draft vocabulary needs the MTP head loaded (loader::load's rule, spec 8 §11; 19c relaxes it for a drafter).
  const loader::LoadedModel m = loader::load(ctx, snap, max_len, dv != 0, head, dvs);
  const loader::DFlashWeights w = loader::load_drafter(ctx, drafter, m, fmt);
  const runtime::dflash::DFlashTarget t = runtime::dflash::dflash_target(m, dv != 0);
  runtime::dflash::DFlashBuffers b(ctx, w.desc, w.target_hidden, t.head_rows(), max_len);
  l0::Mem control(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(control.ptr(), 0, sizeof(runtime::Control));
  runtime::CapturedStep draft = runtime::dflash::build_dflash_draft(ctx, w, t, b, control);
  runtime::CapturedStep project = runtime::dflash::build_dflash_project(ctx, w, b);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::ofstream out(out_path);
  {
    std::string v_ids = "[]";
    if (m.draft_vocab) v_ids = arr(m.draft_vocab->host_ids.data(), m.draft_vocab->size());
    out << "{\"run\": {\"format\": \"" << loader::draft_format_name(fmt) << "\", \"head\": \""
        << loader::lm_head_form_name(head) << "\", \"draft_vocab\": " << dv << ", \"vocab_used\": " << t.vocab_used
        << ", \"k\": " << K << ", \"launches\": [" << draft.kernel_count << ", " << project.kernel_count
        << "], \"load\": \"" << w.line << "\", \"v_ids\": " << v_ids << "}}\n";
  }
  const uint32_t H = w.desc.hidden, ntap = uint32_t(w.desc.taps.size()), M = kernels::dflash::kM;
  double t_draft = 0;
  size_t n_draft = 0;
  for (const std::string& path : dumps) {
    const Dump d = read_dump(path, H);
    if (d.layers != w.desc.taps) throw std::runtime_error(path + ": tap layers differ from the drafter's");
    if (d.n < 8 || d.n_prompt + 8 > d.n) continue;
    // anchors evenly over n_prompt .. n - 8 (dflash_accept.py's anchors_of with K 7), at most `anchors`
    std::vector<uint32_t> ps;
    const uint32_t lo = std::max(d.n_prompt, d.tap_from + 1), hi = d.n - 8;
    const uint32_t cnt = std::min(anchors, hi - lo + 1);
    for (uint32_t i = 0; i < cnt; ++i) ps.push_back(lo + uint32_t(uint64_t(hi - lo) * i / std::max(1u, cnt - 1)));
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    b.zero(imm);   // a cold ring per dump
    const uint32_t win = w.desc.max_window();
    uint32_t next_pos = std::max(d.tap_from, ps.front() > win - 1 ? ps.front() - (win - 1) : 0u);
    // A dump's taps start at tap_from: positions before it are in neither side's context (dflash_d1.py's reference
    // builds its context from tap_from too), so the two still draft from the same context.
    for (uint32_t p : ps) {
      while (next_pos < p) {   // project [next_pos, min(p, next_pos + 8))
        const uint32_t n = std::min(M, p - next_pos);
        std::vector<uint16_t> rows(size_t(M) * ntap * H, 0);
        for (uint32_t r = 0; r < n; ++r)
          for (uint32_t i = 0; i < ntap; ++i)
            std::memcpy(&rows[(size_t(r) * ntap + i) * H], &d.taps[i][size_t(next_pos + r - d.tap_from) * H], H * 2);
        imm.copy(b.taps.ptr(), rows.data(), rows.size() * 2);
        b.ctl()->proj_pos = next_pos;
        b.ctl()->proj_n = n;
        q.execute(project.list, &fence);
        fence.wait();
        next_pos += n;
      }
      runtime::Control* c = control.as<runtime::Control>();
      c->pos = p;
      c->n_active = 1;
      c->cur_token[0] = uint32_t(d.ids[p]);
      runtime::dflash::set_draft_params(*b.ctl(), w.desc, K, 0.0f, 0);
      const auto t0 = std::chrono::steady_clock::now();
      q.execute(draft.list, &fence);
      fence.wait();
      t_draft += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++n_draft;
      const runtime::dflash::DFlashDraftOut& o = *b.readback();
      std::string cand = "[", un = "[", sc = "[";
      for (uint32_t j = 0; j < K; ++j) {
        cand += (j ? "," : "") + arr(o.cand[j], 16);
        un += (j ? "," : "") + farr(o.unary[j], 16);
        sc += (j ? "," : "") + farr(o.scores[j], 16);
      }
      out << "{\"dump\": \"" << d.name << "\", \"p\": " << p << ", \"anchor\": " << d.ids[p] << ", \"k\": " << o.k
          << ", \"ids\": " << arr(o.ids, K) << ", \"chosen\": " << arr(o.chosen, K) << ", \"cand\": " << cand
          << "], \"unary\": " << un << "], \"scores\": " << sc << "]}\n";
    }
    std::printf("%s: %zu anchors\n", d.name.c_str(), ps.size());
  }
  std::printf("dflash_d1: %zu drafts, %.3f ms a draft list (host-timed, one replay, fence included); %s\n", n_draft,
              n_draft ? 1e3 * t_draft / double(n_draft) : 0.0, out_path.c_str());
  return 0;
}
