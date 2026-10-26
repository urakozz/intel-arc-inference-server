// bolt_golden_test - spec 25 §9 B3 (decode): Nemotron 3.5 Bolt's engine against tools/oracle/bolt_ref.py (spec 25a, the
// engine-format reference: the checkpoint's int4 dequantised) - one golden prompt x 32 greedy tokens - with the per-block
// routing, SSM-state, conv-state and residual diagnostics. kolibri_golden_test's shape, Bolt's semantics.
//
// **The golden files** (bolt_ref.py `run`; tools/box_validate/bolt_oracle.sh):
//   <oracle dir>/<prompt>.ids (or tests/golden/prompts/<prompt>.ids)   no BOS (the template writes none)
//   <oracle dir>/<prompt>.golden.safetensors
//     logits          F32  [Lr][131072]   the last Lr rows of T + gen (row T + j: the forward that consumed tokens[j])
//     tokens          I32  [gen]          the greedy continuation (gen = 32)
//     h.L<l>          BF16 [A][4096]      the residual after block l, the last A rows of T + gen
//     route.ids.L<l>  I32  [T + gen][9]   MoE blocks: the selected ids ASCENDING (the set)
//     route.w.L<l>    F32  [T + gen][9]   the weights at those ids (fp32, never rounded)
//     route.gap.L<l>  F32  [T + gen]      sel(9th) - sel(10th) (0: an exact tie, undetermined)
//     ssm.L<l>        F32  [128][64][128] Mamba blocks: the SSM state after the prompt
//     conv.L<l>       BF16 [3][10240]     Mamba blocks: the last three RAW xBC columns before the first generated row
//
// **The gate** (golden_common.h's tie-aware rule): ingest the ids one replay at a time, then for j = 0..31 the engine's
// sampled id against the golden decision row T - 1 + j (determined rows exact; a row whose top-1 is not unique accepts any
// member of the argmax set), then feed the GOLDEN token.
// **The routing** (spec 25 §9 B3, Decision 11): at every (row, MoE block) the engine's route row against route.ids as SETS.
// A set difference is a NEAR-TIE when the reference's gap is within tie_tol() (B70_BOLT_TIE_TOL: 1e-3, PROPOSED until 25a's
// gap distribution is measured on the real weights - row 57) and the sets differ by one expert; an exact 0 gap is
// undetermined (torch's topk keeps either id of a tie: 25a); any other difference FAILS. Matching sets: w per expert, the
// worst fp32 ulps printed (25a: torch's router GEMM is row-count dependent and its sum of the nine s has no fixed order -
// w is never compared bitwise).
// **The SSM state and the conv columns** after the prompt: cosine per block >= 0.999 (PROPOSED) and the exact fraction
// printed - not bitwise (the engine's int4 GEMV ((q - 8) x scale in fp32) and the reference's bf16-dequantised torch linear
// round differently from the first in_proj on: 24c's finding).
// **The residual** h.L<l> per (row, block): cosine per row; per block median >= 0.9998 and min >= 0.99 (PROPOSED).
//
// **B3 on prefill (spec 25d)**: with `prefill` (or `prefill:<chunk>`, a multiple of 64) the prompt goes through
// BoltEngine::prefill instead of one decode replay per id (the first generated id is the prefill head's: decode's head
// binaries over the last row); the routing compares the last chunk's route rows; the h taps are the prefill's (row T - 1
// of each block) and then the decode steps'; the SSM state and the conv columns are the prefill's. The bars are the decode
// gate's (PROPOSED): prefill's linears round through bf16-dequantised weights in a DPAS order and its SSM is the SSD form -
// both further from the reference than decode's chain, which the bars are meant to absorb (Kolibri's precedent).
//
// argv: <checkpoint> <oracle dir> <prompt> [bf16] [eager] [b16] [layers:<N>] [pp2 | pp2:<split>] [copy | peer]
//       [prefill | prefill:<chunk>]
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
#include "kernels/bolt_kernels.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/bolt_rig.h"
#include "runtime/control.h"

#ifndef B70_SOURCE_DIR
#define B70_SOURCE_DIR "."   // tests/CMakeLists.txt defines the source tree (the committed prompts' home)
#endif

namespace {

namespace rb = runtime::bolt;
namespace kr = kernels::bolt::route;
constexpr uint32_t kMaxLen = 16384, kGen = 32;
float tie_tol() {
  const char* e = std::getenv("B70_BOLT_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 1e-3f;   // PROPOSED (header; Decision 11)
}
constexpr double kHMedian = 0.9998, kHMin = 0.99, kStateCos = 0.999;   // PROPOSED (header)

struct RouteCount {
  uint64_t rows = 0, exact = 0, ties = 0, undetermined = 0, bad = 0;
  int64_t worst_w = 0;
};

int64_t ulps32(float a, float b) {
  int32_t x, y;
  std::memcpy(&x, &a, 4);
  std::memcpy(&y, &b, 4);
  const int64_t kx = x < 0 ? -int64_t(x & 0x7FFFFFFF) : x, ky = y < 0 ? -int64_t(y & 0x7FFFFFFF) : y;
  return std::llabs(kx - ky);
}

void compare_route(const uint32_t* row, const int32_t* want_ids, const float* want_w, float gap, RouteCount& c,
                   char& first_diff, const std::string& pname, uint32_t r, uint32_t layer) {
  ++c.rows;
  const uint32_t K = kernels::bolt::kTopK;
  std::vector<uint32_t> g(row + kr::kIds, row + kr::kIds + K), w(K);
  for (uint32_t k = 0; k < K; ++k) w[k] = uint32_t(want_ids[k]);
  std::vector<uint32_t> gs = g, ws = w;
  std::sort(gs.begin(), gs.end());
  std::sort(ws.begin(), ws.end());
  if (gs == ws) {
    ++c.exact;
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t j = 0; j < K; ++j)
        if (g[k] == w[j]) {
          float gw;
          std::memcpy(&gw, &row[kr::kW + k], 4);
          c.worst_w = std::max(c.worst_w, ulps32(gw, want_w[j]));
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
double cosine_f32(const float* a, const float* b, size_t n, size_t& exact) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
    exact += std::memcmp(&a[i], &b[i], 4) == 0;
  }
  return na == 0 && nb == 0 ? 1.0 : dot / std::sqrt(na * nb + 1e-300);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-bolt";
  const std::string pname = argc > 3 ? argv[3] : "bolt_short";
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  o.attn = rb::bolt_attn();
  o.state = rb::bolt_state();
  bool devices_given = false;
  bool prefill = false;   // spec 25d: the prompt through BoltEngine::prefill
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
      o.attn = rb::BoltAttn::Eager;
    } else if (a == "b16") {
      o.state = rb::BoltState::Bf16;
    } else if (a.rfind("layers:", 0) == 0) {
      o.layers = uint32_t(std::strtoul(a.c_str() + 7, nullptr, 10));
    } else if (a == "pp2") {
      o.devices = 2;
      devices_given = true;
    } else if (a.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      devices_given = true;
      o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10));
    } else if (a == "copy" || a == "peer") {
      o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    } else {
      std::fprintf(stderr, "bolt_golden_test: unknown flag %s (bf16, int8, eager, b16, layers:N, pp2[:split], copy, peer, "
                   "prefill[:chunk])\n", a.c_str());
      return 2;
    }
  }
  const std::string snap = bolt_rig::bolt_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no Nemotron 3.5 Bolt checkpoint at '%s'\n", arg.c_str());
    return 77;
  }
  const std::string gpath = gdir + "/" + pname + ".golden.safetensors";
  std::string ids_path = gdir + "/" + pname + ".ids";
  if (!golden::exists(ids_path)) ids_path = std::string(B70_SOURCE_DIR) + "/tests/golden/prompts/" + pname + ".ids";
  if (!golden::exists(gpath) || !golden::exists(ids_path)) {
    std::printf("SKIP: %s or %s is absent (tools/box_validate/bolt_oracle.sh)\n", gpath.c_str(), ids_path.c_str());
    return 77;
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snap, o.layers);
  const std::vector<uint32_t> ids = bolt_rig::read_ids(ids_path, pre.vocab);
  CHECK(!ids.empty());
  // max_len: 16384, or the prompt + 32 generated rounded up to 256 when longer (bolt_32k), within the trained context.
  o.max_len = std::max<uint32_t>(kMaxLen, (uint32_t(ids.size()) + kGen + 255) / 256 * 256);
  if (o.max_len > pre.max_pos) {
    std::printf("SKIP: %zu ids + %u exceed the trained context %u\n", ids.size(), kGen, pre.max_pos);
    return 77;
  }
  if (!devices_given && !bolt_rig::fits(pre, o, prefill)) o.devices = 2;   // the real model: two cards (Decision 4)
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  if (!bolt_rig::fits(pre, o, prefill)) {
    std::printf("SKIP: %s does not fit the placement (%u card(s))\n", snap.c_str(), o.devices);
    return 77;
  }
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& eng = *rig.eng;
  const model::BoltDesc& d = eng.model().desc;
  std::printf("%s: %u blocks, %s head, %s attention, fuse %s, %s state, %u device(s)%s, %zu launches a token, near-tie "
              "tolerance %.1e\n", snap.c_str(), d.layers, o.int8_head ? "int8" : "bf16", rb::bolt_attn_name(eng.attention()),
              rb::bolt_fuse_name(eng.fuse()).c_str(), rb::bolt_state_name(eng.state()), eng.devices(),
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
  const uint32_t K = kernels::bolt::kTopK;
  std::printf("\n================ %s: %u prompt ids, %u generated ================\n", pname.c_str(), T, kGen);
  RouteCount rc;
  std::vector<char> first(d.layers, 0);
  std::vector<std::vector<double>> hcos(d.layers);
  std::vector<double> sa, sb;
  // One (row, MoE block)'s route row against the golden set.
  auto check_route = [&](const uint32_t* r, uint32_t row, uint32_t l) {
    const std::string ls = std::to_string(l);
    const int32_t* gi = g.i32("route.ids.L" + ls, size_t(rows) * K) + size_t(row) * K;
    const float* gw = g.f32("route.w.L" + ls, size_t(rows) * K) + size_t(row) * K;
    const float gap = g.f32("route.gap.L" + ls, rows)[row];
    compare_route(r, gi, gw, gap, rc, first[l], pname, row, l);
  };
  auto check_taps = [&](uint32_t row) {
    const std::vector<uint16_t> tap = eng.read_debug_h();
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string n = "h.L" + std::to_string(l);
      const uint32_t A = uint32_t(g.dim(n, 2, 0)), h0 = rows - A;
      if (row < h0) continue;
      const uint16_t* want = g.bf16(n, size_t(A) * d.hidden) + size_t(row - h0) * d.hidden;
      hcos[l].push_back(golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb).cos);
    }
  };
  auto check_row = [&](uint32_t row) {
    const std::vector<uint32_t> r = eng.read_routes();
    uint32_t mi = 0;
    for (uint32_t l = 0; l < d.layers; ++l)
      if (d.is_moe(l)) check_route(r.data() + size_t(mi++) * rb::kRouteWords, row, l);
    check_taps(row);
  };
  eng.reset();
  if (prefill) {
    // Spec 25d: the prompt in one BoltEngine::prefill (chunks of `chunk`, every chunk on the 64-grid from position 0); the
    // routing on the rows the last chunk's route buffer holds; the h taps at the last prompt row (the prefill's tap).
    const uint32_t c = chunk ? chunk : rb::kPfC;
    rb::check_prefill_chunk(c);
    eng.prefill(ids, c);
    const std::vector<uint32_t> r = eng.read_prefill_routes();
    const uint32_t row0 = ((T - 1) / c) * c;
    for (uint32_t t = row0; t < T; ++t)
      for (uint32_t l = 0; l < d.layers; ++l)
        if (d.is_moe(l)) check_route(r.data() + rb::pf_route_at(l) / 4 + size_t(t - row0) * rb::kRouteWords, t, l);
    check_taps(T - 1);
    std::printf("  prefill: %u ids in chunks of %u (%zu launches), routing on rows %u..%u, h taps at row %u\n", T, c,
                eng.prefill_launches(), row0, T - 1, T - 1);
  } else {
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      check_row(t);
    }
  }
  // The SSM state and the conv columns after the prompt.
  double state_worst = 1.0, conv_worst = 1.0;
  size_t state_exact = 0, state_total = 0, conv_exact = 0, conv_total = 0;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l)) continue;
    const std::string ls = std::to_string(l);
    const std::vector<float> st = eng.read_ssm_state(d.mamba_index(l));
    const float* want_s = g.f32("ssm.L" + ls, st.size());
    state_worst = std::min(state_worst, cosine_f32(st.data(), want_s, st.size(), state_exact));
    state_total += st.size();
    if (T < 3) continue;
    const std::vector<uint16_t> ring = eng.read_conv_ring(d.mamba_index(l));
    const uint16_t* want = g.bf16("conv.L" + ls, size_t(3) * d.m_conv_dim);
    for (uint32_t s = 0; s < 3; ++s) {
      const uint16_t* got = ring.data() + size_t((T - 3 + s) % 16) * d.m_conv_dim;
      conv_worst = std::min(conv_worst, golden::compare_bf16(got, want + size_t(s) * d.m_conv_dim, d.m_conv_dim, sa, sb).cos);
      for (uint32_t k = 0; k < d.m_conv_dim; ++k) conv_exact += got[k] == want[size_t(s) * d.m_conv_dim + k];
      conv_total += d.m_conv_dim;
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
  std::printf("  SSM state after the prompt: %zu / %zu values exact, worst block cosine %.6f (bar %.3f, PROPOSED)\n",
              state_exact, state_total, state_worst, kStateCos);
  std::printf("  conv columns after the prompt: %zu / %zu values exact, worst cosine %.6f\n", conv_exact, conv_total,
              conv_worst);
  bool h_ok = true;
  std::printf("  h (median / min cosine per block):");
  for (uint32_t l = 0; l < d.layers; ++l) {
    const double med = median(hcos[l]);
    const double mn = hcos[l].empty() ? 1.0 : *std::min_element(hcos[l].begin(), hcos[l].end());
    std::printf(" %u:%.5f/%.4f", l, med, mn);
    h_ok = h_ok && med >= kHMedian && mn >= kHMin;
  }
  std::printf("\nrouting: %llu (row, block) (%llu exact sets, %llu near-ties within %.1e, %llu undetermined (gap 0), %llu "
              "differing; matching weights worst %lld fp32 ulps)\n", (unsigned long long)rc.rows,
              (unsigned long long)rc.exact, (unsigned long long)rc.ties, double(tie_tol()),
              (unsigned long long)rc.undetermined, (unsigned long long)rc.bad, (long long)rc.worst_w);
  CHECK(gate_ok);
  CHECK_EQ(rc.bad, uint64_t(0));
  CHECK(h_ok);
  CHECK(state_worst >= kStateCos);
  CHECK(conv_worst >= kStateCos);
  std::puts("bolt_golden_test OK");
  return 0;
}
