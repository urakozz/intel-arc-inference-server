// lfm2moe_golden_test - spec 24 §7 F3 (decode): LFM2.5-24B-A2B's engine against tools/oracle/lfm2moe_ref.py (spec
// 24a) - one golden prompt x 32 greedy tokens - with the per-layer routing, conv-state and residual diagnostics.
// kolibri_golden_test's copy (tests/golden/kolibri_golden_test.cc), LFM2's semantics.
//
// **The golden files** (lfm2moe_ref.py `run`; tools/box_validate/lfm2moe_oracle.sh):
//   <oracle dir>/<prompt>.ids (or tests/golden/prompts/<prompt>.ids)   BOS first
//   <oracle dir>/<prompt>.golden.safetensors
//     logits          F32  [Lr][128000]   the last Lr rows of T + gen (row T + j: the forward that consumed tokens[j])
//     tokens          I32  [gen]          the greedy continuation (gen = 32)
//     h.L<l>          BF16 [A][2048]      the residual after layer l, the last A rows of T + gen
//     conv.L<l>       BF16 [2][2048]      conv layers: Bx of positions T - 2, T - 1 (the ring's two slots read next)
//     route.ids.L<l>  I32  [T + gen][4]   MoE layers: the selection in torch's rank order
//     route.w.L<l>    F32  [T + gen][4]   the renormalised weights (bf16 values)
//     route.gap.L<l>  F32  [T + gen]      sel(4th) - sel(5th) (0: an exact tie, undetermined)
//
// **The gate** (golden_common.h's tie-aware rule): ingest the ids one replay at a time, then for j = 0..31 the
// engine's sampled id against the golden decision row T - 1 + j (determined rows exact; a row whose top-1 is not
// unique accepts any member of the argmax set), then feed the GOLDEN token.
// **The routing** (spec 24 §7 F3): at every (row, MoE layer) the engine's route row against route.ids as SETS (torch's
// order among exact ties is its own). A set difference is a NEAR-TIE when the reference's gap is within tie_tol()
// (B70_LFM_TIE_TOL: 4e-3, RULED 2026-10-16 - one bf16 ulp of the logit carried through the sigmoid; spec 24 §12) and
// the sets differ by one expert; an exact 0 gap is undetermined; any other difference FAILS. Matching sets: the
// weights per expert, worst bf16 ulps printed.
// **The conv state** after the prompt: each conv layer's ring slots (T - 2) % 16, (T - 1) % 16 against conv.L<l> -
// exact fraction and cosine printed, held to the h bars below (Ruling: the engine's int4 GEMV and the reference's
// dequantised bf16 linear round differently, so Bx is not bitwise past the first linear).
// **The residual** h.L<l> per (row, layer): cosine per row; per layer median >= 0.9998 and min >= 0.99 (PROPOSED).
//
// **F3 on prefill (spec 24d)**: with `prefill` (or `prefill:<chunk>`) the prompt goes through Lfm2MoeEngine::prefill
// (chunks of 2048, or <chunk>) instead of one replay per id; the gate is the same rule from row T - 1 on (the
// prefill's head chooses the first token: decode's head binaries over the last row); the routing compares the last
// chunk's route rows; the h taps are the prefill's (row T - 1 of each layer) and then the decode steps'. The bars are
// the decode gate's (the h bars, B70_LFM_TIE_TOL): prefill's linears round through bf16-dequantised weights in a DPAS
// order, the reference through torch's bf16 linear, so neither path is bitwise the reference's.
//
// argv: <checkpoint> <oracle dir> <prompt> [bf16] [eager] [pp2 | pp2:<split>] [copy | peer] [prefill | prefill:<chunk>]
// Exit 77 (SKIP) when the checkpoint or the golden set is absent, or the placement cannot run.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "kernels/lfm2moe_kernels.h"
#include "runtime/lfm2moe_rig.h"
#include "runtime/control.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

#ifndef B70_SOURCE_DIR
#define B70_SOURCE_DIR "."   // tests/CMakeLists.txt defines the source tree (the committed prompts' home)
#endif

namespace {

namespace rl = runtime::lfm2moe;
namespace kr = kernels::lfm2moe::route;
constexpr uint32_t kMaxLen = 16384, kGen = 32;
float tie_tol() {
  const char* e = std::getenv("B70_LFM_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 4e-3f;   // RULED (header)
}
constexpr double kHMedian = 0.9998, kHMin = 0.99;   // PROPOSED (header)

struct RouteCount {
  uint64_t rows = 0, exact = 0, ties = 0, undetermined = 0, bad = 0;
  int worst_w = 0;
};

int ulps16(float a, float b) {
  const uint16_t x = common::f32_to_bf16(a), y = common::f32_to_bf16(b);
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(x) - key(y));
}

void compare_route(const uint32_t* row, const int32_t* want_ids, const float* want_w, float gap, RouteCount& c,
                   char& first_diff, const std::string& pname, uint32_t r, uint32_t layer) {
  ++c.rows;
  std::vector<uint32_t> g(row + kr::kIds, row + kr::kIds + 4), w(4);
  for (uint32_t k = 0; k < 4; ++k) w[k] = uint32_t(want_ids[k]);
  std::vector<uint32_t> gs = g, ws = w;
  std::sort(gs.begin(), gs.end());
  std::sort(ws.begin(), ws.end());
  if (gs == ws) {
    ++c.exact;
    for (uint32_t k = 0; k < 4; ++k)
      for (uint32_t j = 0; j < 4; ++j)
        if (g[k] == w[j]) {
          float gw;
          std::memcpy(&gw, &row[kr::kW + k], 4);
          c.worst_w = std::max(c.worst_w, ulps16(gw, want_w[j]));
        }
    return;
  }
  if (!first_diff) {
    std::printf("    %s L%u: first differing row %u (reference gap %.3e)\n", pname.c_str(), layer, r, double(gap));
    first_diff = 1;
  }
  std::vector<uint32_t> only_g, only_w;
  std::set_difference(gs.begin(), gs.end(), ws.begin(), ws.end(), std::back_inserter(only_g));
  std::set_difference(ws.begin(), ws.end(), gs.begin(), gs.end(), std::back_inserter(only_w));
  if (gap == 0.0f) {
    ++c.undetermined;
    return;
  }
  if (gap <= tie_tol() && only_g.size() == 1 && only_w.size() == 1) {
    ++c.ties;
    return;
  }
  ++c.bad;
  if (c.bad <= 10)
    std::printf("    %s L%u row %u: expert set differs beyond a near-tie (reference gap %.3e, %zu experts differ)\n",
                pname.c_str(), layer, r, double(gap), only_g.size());
}

double median(std::vector<double> v) {
  if (v.empty()) return 1.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-lfm2moe";
  const std::string pname = argc > 3 ? argv[3] : "lfm2moe_short";
  lfm2moe_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  o.attn = rl::lfm_attn();
  bool prefill = false;   // spec 24d: the prompt through Lfm2MoeEngine::prefill
  uint32_t chunk = 0;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "prefill") {
      prefill = true;
    } else if (a.rfind("prefill:", 0) == 0) {
      prefill = true;
      chunk = uint32_t(std::strtoul(a.c_str() + 8, nullptr, 10));
    } else if (a == "bf16") {
      o.int8_head = false;
    } else if (a == "int8") {
      o.int8_head = true;
    } else if (a == "eager") {
      o.attn = rl::LfmAttn::Eager;
    } else if (a == "pp2") {
      o.devices = 2;
    } else if (a.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10));
    } else if (a == "copy" || a == "peer") {
      o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    } else {
      std::fprintf(stderr, "lfm2moe_golden_test: unknown flag %s (bf16, int8, eager, pp2[:split], copy, peer, "
                   "prefill[:chunk])\n",
                   a.c_str());
      return 2;
    }
  }
  const std::string snap = lfm2moe_rig::lfm2moe_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no LFM2.5-24B-A2B checkpoint at '%s'\n", arg.c_str());
    return 77;
  }
  const std::string gpath = gdir + "/" + pname + ".golden.safetensors";
  std::string ids_path = gdir + "/" + pname + ".ids";
  if (!golden::exists(ids_path)) ids_path = std::string(B70_SOURCE_DIR) + "/tests/golden/prompts/" + pname + ".ids";
  if (!golden::exists(gpath) || !golden::exists(ids_path)) {
    std::printf("SKIP: %s or %s is absent (tools/box_validate/lfm2moe_oracle.sh)\n", gpath.c_str(), ids_path.c_str());
    return 77;
  }
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp2 needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Lfm2MoeDesc pre = loader::lfm2moe_checkpoint_desc(snap);
  const std::vector<uint32_t> ids = lfm2moe_rig::read_ids(ids_path, pre.vocab);
  CHECK(!ids.empty());
  // max_len: 16384, or the prompt + 32 generated rounded up to 256 when longer (lfm2moe_32k), within the trained
  // context (the final review: a fixed 16384 made the 32k prompt throw at position 16384).
  o.max_len = std::max<uint32_t>(kMaxLen, (uint32_t(ids.size()) + kGen + 255) / 256 * 256);
  if (o.max_len > pre.trained_max_len) {
    std::printf("SKIP: %zu ids + %u exceed the trained context %u\n", ids.size(), kGen, pre.trained_max_len);
    return 77;
  }
  if (o.devices == 1 && !lfm2moe_rig::fits_one_card(pre, o.int8_head, o.max_len, prefill)) {
    std::printf("SKIP: %s does not fit one card\n", snap.c_str());
    return 77;
  }
  lfm2moe_rig::Rig rig;
  lfm2moe_rig::build(rig, snap, o);
  rl::Lfm2MoeEngine& eng = *rig.eng;
  const model::Lfm2MoeDesc& d = eng.model().desc;
  std::printf("%s: %u layers, %s head, %s attention, fuse %s, %u device(s)%s, %zu launches a token, near-tie "
              "tolerance %.1e\n", snap.c_str(), d.layers, o.int8_head ? "int8" : "bf16", rl::lfm_attn_name(eng.attention()),
              rl::lfm_fuse_name(eng.fuse()).c_str(), eng.devices(),
              eng.devices() == 2 ? (std::string(", split ") + std::to_string(eng.split()) + " " +
                                    runtime::pp_handoff_name(eng.handoff())).c_str() : "",
              eng.launches(), double(tie_tol()));

  golden::Golden g(gpath);
  const uint32_t T = uint32_t(ids.size()), rows = T + kGen;
  const uint32_t lr = uint32_t(g.dim("logits", 2, 0));
  CHECK_EQ(g.dim("logits", 2, 1), uint64_t(d.vocab));
  CHECK(lr >= kGen + 1 && lr <= rows);
  const uint32_t logits_row0 = rows - lr;
  const float* glog = g.f32("logits", size_t(lr) * d.vocab);
  const int32_t* gtok = g.i32("tokens", kGen);
  std::printf("\n================ %s: %u prompt ids, %u generated ================\n", pname.c_str(), T, kGen);
  RouteCount rc;
  std::vector<char> first(d.layers, 0);
  std::vector<std::vector<double>> hcos(d.layers);
  std::vector<double> sa, sb;
  auto check_row = [&](uint32_t row) {
    const std::vector<uint32_t> r = eng.read_routes();
    uint32_t mi = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_dense(l)) continue;
      const std::string ls = std::to_string(l);
      const int32_t* gi = g.i32("route.ids.L" + ls, size_t(rows) * 4) + size_t(row) * 4;
      const float* gw = g.f32("route.w.L" + ls, size_t(rows) * 4) + size_t(row) * 4;
      const float gap = g.f32("route.gap.L" + ls, rows)[row];
      compare_route(r.data() + size_t(mi++) * rl::kRouteWords, gi, gw, gap, rc, first[l], pname, row, l);
    }
    const std::vector<uint16_t> tap = eng.read_debug_h();
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string n = "h.L" + std::to_string(l);
      const uint32_t A = uint32_t(g.dim(n, 2, 0)), h0 = rows - A;
      if (row < h0) continue;
      const uint16_t* want = g.bf16(n, size_t(A) * d.hidden) + size_t(row - h0) * d.hidden;
      hcos[l].push_back(golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb).cos);
    }
  };
  eng.reset();
  if (prefill) {
    // Spec 24d, F3 on prefill: the prompt in one Lfm2MoeEngine::prefill (chunks of `chunk`); the routing on the rows
    // the last chunk's route buffer holds; the h taps at the last prompt row (the prefill's tap: row T - 1 of each
    // layer). The 32 teacher-forced steps after it are decode's, as before.
    eng.prefill(ids, chunk);
    const uint32_t c = chunk ? chunk : rl::kPfC;
    const uint32_t row0 = ((T - 1) / c) * c;   // the last chunk's first position
    const std::vector<uint32_t> r = eng.read_prefill_routes();
    for (uint32_t t = row0; t < T; ++t)
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (d.is_dense(l)) continue;
        const std::string ls = std::to_string(l);
        const int32_t* gi = g.i32("route.ids.L" + ls, size_t(rows) * 4) + size_t(t) * 4;
        const float* gw = g.f32("route.w.L" + ls, size_t(rows) * 4) + size_t(t) * 4;
        const float gap = g.f32("route.gap.L" + ls, rows)[t];
        compare_route(r.data() + rl::pf_route_at(l) / 4 + size_t(t - row0) * rl::kRouteWords, gi, gw, gap, rc, first[l],
                      pname, t, l);
      }
    const std::vector<uint16_t> tap = eng.read_debug_h();
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string n = "h.L" + std::to_string(l);
      const uint32_t A = uint32_t(g.dim(n, 2, 0)), h0 = rows - A;
      if (T - 1 < h0) continue;
      const uint16_t* want = g.bf16(n, size_t(A) * d.hidden) + size_t(T - 1 - h0) * d.hidden;
      hcos[l].push_back(golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb).cos);
    }
    std::printf("  prefill: %u ids in chunks of %u (%zu launches), routing on rows %u..%u, h taps at row %u\n", T, c,
                eng.prefill_launches(), row0, T - 1, T - 1);
  }
  for (uint32_t t = 0; t < T && !prefill; ++t) {
    eng.ingest({ids[t]});
    check_row(t);
  }
  // The conv state after the prompt: ring slots (T - 2) % 16 and (T - 1) % 16 against conv.L<l> (oldest first).
  double conv_worst = 1.0;
  size_t conv_exact = 0, conv_total = 0;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_attention(l) || T < 2) continue;
    const std::vector<uint16_t> ring = eng.read_conv_ring(d.conv_index(l));
    const uint16_t* want = g.bf16("conv.L" + std::to_string(l), size_t(2) * d.hidden);
    for (uint32_t s = 0; s < 2; ++s) {
      const uint16_t* got = ring.data() + size_t((T - 2 + s) % 16) * d.hidden;
      const double c = golden::compare_bf16(got, want + size_t(s) * d.hidden, d.hidden, sa, sb).cos;
      conv_worst = std::min(conv_worst, c);
      for (uint32_t k = 0; k < d.hidden; ++k) conv_exact += got[k] == want[size_t(s) * d.hidden + k];
      conv_total += d.hidden;
    }
  }
  uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
  int first_bad = -1;
  double logit_min = 1.0;
  for (uint32_t j = 0; j < kGen; ++j) {
    const uint32_t row = T - 1 + j;
    const uint32_t got = eng.pending();
    CHECK(row >= logits_row0);
    const float* want = glog + size_t(row - logits_row0) * d.vocab;
    const std::vector<float> lg = eng.read_logits();
    logit_min = std::min(logit_min, golden::compare_f32(lg.data(), want, d.vocab, sa, sb).cos);
    const golden::GoldenDecision dec = golden::golden_decision(want, d.vocab, d.vocab_used);
    if (dec.determined()) {
      ++det;
      if (got == dec.set[0]) ++det_ok;
      else if (first_bad < 0) first_bad = int(j);
    } else {
      ++ties;
      if (dec.contains(got)) ++tie_ok;
      else if (first_bad < 0) first_bad = int(j);
    }
    if (got != uint32_t(gtok[j])) {
      std::printf("  step %u: engine %u, golden %u (%s)\n", j, got, uint32_t(gtok[j]), dec.determined() ? "determined" : "tie row");
      golden::print_top5("engine", lg.data(), d.vocab, d.vocab_used);
      golden::print_top5("golden", want, d.vocab, d.vocab_used);
    }
    eng.ingest({uint32_t(gtok[j])});   // teacher-forced: row T + j
    check_row(T + j);
  }
  const bool gate_ok = det_ok == det && tie_ok == ties;
  std::printf("  gate: %u / %u determined rows exact, %u / %u tie rows in the argmax set%s; logits worst cosine %.6f\n",
              det_ok, det, tie_ok, ties, gate_ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(),
              logit_min);
  std::printf("  conv state after the prompt: %zu / %zu values exact, worst cosine %.6f\n", conv_exact, conv_total,
              conv_worst);
  bool h_ok = true;
  std::printf("  h (median / min cosine per layer):");
  for (uint32_t l = 0; l < d.layers; ++l) {
    const double med = median(hcos[l]);
    const double mn = hcos[l].empty() ? 1.0 : *std::min_element(hcos[l].begin(), hcos[l].end());
    std::printf(" %u:%.5f/%.4f", l, med, mn);
    h_ok = h_ok && med >= kHMedian && mn >= kHMin;
  }
  std::printf("\nrouting: %llu (row, layer) (%llu exact sets, %llu near-ties within %.1e, %llu undetermined (gap 0), "
              "%llu differing; matching weights worst %d bf16 ulps)\n", (unsigned long long)rc.rows,
              (unsigned long long)rc.exact, (unsigned long long)rc.ties, double(tie_tol()),
              (unsigned long long)rc.undetermined, (unsigned long long)rc.bad, rc.worst_w);
  CHECK(gate_ok);
  CHECK_EQ(rc.bad, uint64_t(0));
  CHECK(h_ok);
  CHECK(conv_worst >= kHMin);
  std::puts("lfm2moe_golden_test OK");
  return 0;
}
