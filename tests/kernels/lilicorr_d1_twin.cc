// lilicorr_d1_twin - spec 26b Task 6, D1's host half: the LiLiCorr drafter's CPU twin (tests/kernels/lilicorr_ref.h in
// Engine numerics - the card's arithmetic: bf16 GEMV inputs, a bf16 ring, the loader's own device forms of the weights)
// drafting at teacher-forced anchors of tap dumps, written as tools/probe/dflash_d1 --model lilicorr writes the card's
// drafts - the same JSONL - so tools/oracle/dflash_d1.py holds the twin to dflash_ref.py's LiLiCorr form on the same taps
// with D1's bars. On the Mac it runs over 26a's tiny pair (tools/oracle/bolt_make_tiny.py's Bolt + lilicorr_make_tiny.py's
// drafter, the dumps of bolt_ref.py --taps): the pipeline and the twin's agreement on a real target's taps before the
// card exists; on the box the same over the real drafter and 25a's dumps (CPU, minutes a dump).
//
//   lilicorr_d1_twin <target dir> <drafter dir> <out.jsonl> [--format exact|int8|int4] [--head int8|bf16] [--anchors N]
//                    [--reference] <dump.taps.safetensors>...
//
// --reference: the twin in Reference numerics (fp32 activations over fp32 W' x g, the fp32 head) - the D1 comparator's
// own arithmetic up to summation order, which separates a pipeline error (it fails too) from the Engine numerics' bf16
// roundings on a given model (it passes; the format arm is then the reference's: write --format exact).
//
// <target dir>: a Bolt checkpoint (only its embedding and lm_head are read - `<trunk>embeddings.weight`,
// `language_model.lm_head.weight`, bf16 [vocab][hidden]); the drafter by loader::LiliCheckpoint. Per dump a cold ring; the
// projection 8 positions at a time from max(tap_from, first anchor - (window - 1)); anchors evenly over n_prompt .. n - 8;
// every draft the whole 8-row block, all 7 slots walked at T 0 (dflash_d1.cc's loop, line for line).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/json.h"
#include "kernels/lilicorr_fixture_io.h"
#include "kernels/lilicorr_ref.h"
#include "loader/bolt_repack.h"
#include "loader/lilicorr_repack.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace {

struct Dump {   // tools/probe/dflash_d1.cc's read_dump (dump_taps.py's format)
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
  const common::json::Value h = common::json::parse(std::string_view(reinterpret_cast<const char*>(f.data()) + 8, hlen));
  const common::json::Value& md = h.at("__metadata__");
  Dump d;
  d.name = md.at("name").str();
  d.n = uint32_t(std::stoul(md.at("n").str()));
  d.n_prompt = uint32_t(std::stoul(md.at("n_prompt").str()));
  d.tap_from = uint32_t(std::stoul(md.at("tap_from").str()));
  const std::string ls = md.at("layers").str();
  for (size_t at = 0; at < ls.size();) {
    const size_t c = ls.find(',', at);
    d.layers.push_back(uint32_t(std::stoul(ls.substr(at, c - at))));
    at = c == std::string::npos ? ls.size() : c + 1;
  }
  const size_t data0 = 8 + hlen;
  d.taps.resize(d.layers.size());
  for (const auto& [name, t] : loader::SafetensorsSet::parse_header(f.data(), f.size())) {
    const uint8_t* p = f.data() + data0 + t.begin;
    if (name == "ids") {
      d.ids.resize((t.end - t.begin) / 4);
      std::memcpy(d.ids.data(), p, t.end - t.begin);
    }
    for (size_t i = 0; i < d.layers.size(); ++i)
      if (name == "tap.L" + std::to_string(d.layers[i])) {
        if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[1] != hidden || t.shape[0] != d.n - d.tap_from)
          throw std::runtime_error(path + ": " + name + " is not bf16 [n - tap_from][hidden]");
        d.taps[i].resize((t.end - t.begin) / 2);
        std::memcpy(d.taps[i].data(), p, t.end - t.begin);
      }
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
  try {
    if (argc < 5) {
      std::fprintf(stderr, "usage: lilicorr_d1_twin <target dir> <drafter dir> <out.jsonl> [options] <dump>...\n");
      return 2;
    }
    const std::string target = argv[1], drafter = argv[2], out_path = argv[3];
    std::string fmt = "int8", head = "int8";
    uint32_t anchors = 16;
    bool reference = false;
    std::vector<std::string> dumps;
    for (int i = 4; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
        return argv[++i];
      };
      if (a == "--format")
        fmt = next();
      else if (a == "--head")
        head = next();
      else if (a == "--reference")
        reference = true;
      else if (a == "--anchors")
        anchors = uint32_t(std::stoul(next()));
      else
        dumps.push_back(a);
    }
    loader::DraftFormat df;
    if (!loader::parse_lilicorr_draft_format(fmt, df)) throw std::runtime_error("--format exact|int8|int4");
    if (head != "int8" && head != "bf16") throw std::runtime_error("--head int8|bf16");
    dfref::threads = std::max(1u, std::thread::hardware_concurrency());
    const loader::LiliCheckpoint ck(drafter);
    const model::DFlashDesc& desc = ck.desc();
    // the target's embedding and head
    const std::string tdir = loader::resolve_snapshot(target);
    const loader::SafetensorsSet set(tdir.back() == '/' ? tdir : tdir + "/", true);
    const std::string trunk = loader::bolt_trunk_prefix(set);
    auto tensor = [&](const std::string& name) {
      const loader::TensorInfo& t = set.tensors().at(name);
      if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[1] != desc.hidden)
        throw std::runtime_error(name + " is not bf16 [vocab][" + std::to_string(desc.hidden) + "]");
      const uint16_t* p = reinterpret_cast<const uint16_t*>(set.data(t));
      return std::vector<uint16_t>(p, p + (t.end - t.begin) / 2);
    };
    const dfref::Numerics nm = reference ? dfref::Numerics::Reference : dfref::Numerics::Engine;
    if (reference && df != loader::DraftFormat::Exact) throw std::runtime_error("--reference is the exact arm's");
    const llfix::Form form = reference ? llfix::Form::Reference : df == loader::DraftFormat::Exact ? llfix::Form::Exact
                             : df == loader::DraftFormat::Int8 ? llfix::Form::Int8
                                                               : llfix::Form::Int4;
    const uint32_t win = desc.max_window();
    llref::Weights w = llfix::drafter_weights(ck, form, win >= 128 ? 64 : win / 2, 0);
    const std::vector<uint16_t> head_rows = tensor(loader::kBoltLmHead);
    llfix::set_target(w, desc.mask_id, tensor(trunk + "embeddings.weight"), head_rows,
                      reference ? llfix::HeadForm::Reference : head == "int8" ? llfix::HeadForm::Int8 : llfix::HeadForm::Bf16,
                      uint32_t(head_rows.size() / desc.hidden));
    const llref::Dims& D = w.d;
    std::ofstream out(out_path);
    out << "{\"run\": {\"form\": \"lilicorr\", \"twin\": true, \"head_input\": \"" << (reference ? "fp32" : "bf16")
        << "\", \"format\": \"" << loader::draft_format_name(df)
        << "\", \"head\": \"" << head << "\", \"draft_vocab\": 0, \"vocab_used\": " << D.used << ", \"k\": 7, \"load\": \""
        << "lilicorr_d1_twin (tests/kernels/lilicorr_ref.h, " << (reference ? "Reference" : "Engine")
        << " numerics)\", \"v_ids\": []}}\n";
    const uint32_t H = D.H, M = D.M, K = D.S, KC = D.k, ntap = D.taps;
    size_t n_draft = 0;
    for (const std::string& path : dumps) {
      const Dump d = read_dump(path, D.th);
      if (d.layers != desc.taps) throw std::runtime_error(path + ": tap layers differ from the drafter's");
      if (d.n < M || d.n_prompt + M > d.n) continue;
      std::vector<uint32_t> ps;
      const uint32_t lo = std::max(d.n_prompt, d.tap_from + 1), hi = d.n - M;
      const uint32_t cnt = std::min(anchors, hi - lo + 1);
      for (uint32_t i = 0; i < cnt; ++i) ps.push_back(lo + uint32_t(uint64_t(hi - lo) * i / std::max(1u, cnt - 1)));
      ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
      dfref::Ring ring = dfref::make_ring(D);
      const std::vector<float> rope = dfref::rope_table(D.theta, D.HD, d.n + 2 * M);
      std::vector<float> anchor(H, 0.0f);
      uint32_t next_pos = std::max(d.tap_from, ps.front() > win - 1 ? ps.front() - (win - 1) : 0u);
      for (uint32_t p : ps) {
        while (next_pos < p) {
          const uint32_t n = std::min(M, p - next_pos);
          std::vector<uint16_t> rows(size_t(n) * ntap * D.th);
          for (uint32_t r = 0; r < n; ++r)
            for (uint32_t i = 0; i < ntap; ++i)
              std::memcpy(&rows[(size_t(r) * ntap + i) * D.th], &d.taps[i][size_t(next_pos + r - d.tap_from) * D.th],
                          size_t(D.th) * 2);
          anchor = llref::project(w, nm, rows.data(), n, next_pos, rope, ring);
          next_pos += n;
        }
        const llref::DraftOut o =
            llref::draft(w, nm, ring, rope, anchor, true, uint32_t(d.ids[p]), p, K);
        std::string cand = "[", lp = "[", lg = "[", lat = "[";
        for (uint32_t s = 0; s < K; ++s) {
          cand += (s ? "," : "") + arr(&o.cand[size_t(s) * KC], KC);
          lp += (s ? "," : "") + farr(&o.lp[size_t(s) * KC], KC);
          lg += (s ? "," : "") + farr(&o.logit[size_t(s) * KC], KC);
          lat += s ? ",[" : "[";
          for (uint32_t i = 0; i < KC; ++i) lat += (i ? "," : "") + farr(&o.lattice[(size_t(s) * KC + i) * KC], KC);
          lat += "]";
        }
        out << "{\"dump\": \"" << d.name << "\", \"p\": " << p << ", \"anchor\": " << d.ids[p] << ", \"k\": " << K
            << ", \"ids\": " << arr(o.walk.ids.data(), K) << ", \"chosen\": " << arr(o.walk.chosen.data(), K)
            << ", \"cand\": " << cand << "], \"lp\": " << lp << "], \"logit\": " << lg << "], \"lattice\": " << lat
            << "], \"anchor_row\": " << farr(anchor.data(), H) << "}\n";
        ++n_draft;
      }
      std::printf("%s: %zu anchors\n", d.name.c_str(), ps.size());
    }
    std::printf("lilicorr_d1_twin (drafter %s, head %s): %zu drafts; %s\n", loader::draft_format_name(df), head.c_str(),
                n_draft, out_path.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lilicorr_d1_twin: %s\n", e.what());
    return 1;
  }
}
