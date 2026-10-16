// dflash_d1 - spec 19 §6 D1 (plan 19b Task 3), the engine's half: the DFlash drafter on the card beside the loaded
// target drafts at teacher-forced anchors of tap dumps (tools/oracle/dump_taps.py's format - 19a's P0 dumps, or the
// engine's own once 19c writes them), and every draft's readback is written as one JSON line for
// tools/oracle/dflash_d1.py, which runs tools/oracle/dflash_ref.py on the SAME taps and holds the two to D1's bars
// (top-16 sets tie-aware, scores cosine >= 0.9999 bf16 / >= 0.999 int8, walk ids equal at T = 0).
//
//   dflash_d1 <target snapshot> <drafter> <out.jsonl> [--format int8|bf16] [--head int8|bf16]
//             [--draft-vocab 32k|64k|128k [--draft-vocab-ids FILE]] [--anchors N] [--k K] [--max-len L]
//             <dump.taps.safetensors>...
//
// Per dump (monotonic anchors, one ring): the projection list fills the ring with the dump's taps (8 positions a
// replay, from max(tap_from, first anchor - 2047)), and at each anchor p (N anchors evenly over the dump's
// continuation, dflash_accept.py's range n_prompt .. n - 8) the draft list runs at (p, ids[p], K, T 0). A line:
//   {"dump": name, "p": p, "anchor": id, "k": K, "ids": [...], "chosen": [...], "cand": [[16]...], "unary": [[16]...],
//    "scores": [[16]...]}
// The first line is the run's: {"run": {"format", "head", "draft_vocab", "vocab_used", "v_ids": [...] (the draft
// vocabulary's ids, when one is used), "launches": [draft, project], "load": the drafter's load line}}.
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
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: dflash_d1 <target> <drafter> <out.jsonl> [options] <dump>...\n");
    return 2;
  }
  const std::string target = argv[1], drafter = argv[2], out_path = argv[3];
  loader::DraftFormat fmt = loader::DraftFormat::Int8;
  loader::LmHeadForm head = loader::LmHeadForm::Int8;
  uint32_t dv = 0, anchors = 16, K = 7, max_len = 8192;
  std::string dv_ids;
  std::vector<std::string> dumps;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--format") {
      if (!loader::parse_draft_format(next(), fmt)) throw std::runtime_error("--format int8|bf16");
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
  if (K < 1 || K > kernels::dflash::kR) throw std::runtime_error("--k is 1..7");
  const std::string snap = loader::resolve_snapshot(target);
  l0::Context ctx(0);
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
