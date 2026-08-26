// probe_gemv: the production GEMV at every production shape, both layouts,
// S in {1,2,4,8,16}; GB/s of weight bytes (int4 nibbles + f16 scales), and a
// correctness check against the CPU reference at every configuration.
// Decision rules (spec 1 §4.2): layout = higher GB/s summed over the five
// int4 shapes; S per shape = smallest S within 3% of that shape's best.
//
// The weights are random, not a uniform fill: the B70 compresses device-local
// memory losslessly, so a repeated word reads back far above the 608 GB/s
// theoretical peak (doc 01). Model weights are incompressible; so is this.
//
// `probe_gemv --loads` runs a SECOND, separate thing in the same binary: spec
// 1.7 §3's P1 load-path battery (`run_loads` below). With no argument this
// program is byte-for-byte the instrument that produced
// docs/probe-gemv-2026-08-24.md and its output format is unchanged, because
// that output is a committed record and the battery must not perturb it.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "common/int4.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

namespace {
const double kPeak = 600.0;     // GB/s, doc 01's roofline denominator
const double kMeasured = 590.0; // GB/s, what probe_bw reads through this launch path

// --- the P1 load-path battery ----------------------------------------------
// One row per (shape, variant). The variants are compiled from
// tools/probe/probe_gemv_loads.cl, which states the message-count model each
// one moves; `msgs` below is that model's number and is printed beside the
// measurement so a reader can check the arithmetic against the result.
//
// `base` is gemv.cl verbatim (every PGL_* knob at 0) and every other row is
// held BYTE-IDENTICAL to it: all of these variants keep the accumulation order
// exactly, so a differing bit is a bug in the variant and not a tolerance
// question. That is this battery's TDD equivalent, together with the
// pre-registered per-shape predictions in the task report.
struct LoadVariant {
  const char* tag;
  uint32_t L;         // layout
  bool s_l0best;      // false: the S the engine binds. true: the S layout 0 picks for itself
  const char* msgs;   // LSC messages per subgroup per k-group (weights + scales + activations)
  const char* what;
};
const LoadVariant kVariants[] = {
    {"base",     1, false, "10",     "gemv.cl verbatim - the control"},
    {"xwide",    1, false, "6",      "activations 8x16 B -> 4x32 B"},
    {"deq",      1, false, "10",     "xor+shift dequant (ALU only)"},
    {"pf1",      1, false, "10+1pf", "prefetch() 1 k-group ahead"},
    {"pf2",      1, false, "10+1pf", "prefetch() 2 ahead (bestla's distance)"},
    {"pf4",      1, false, "10+1pf", "prefetch() 4 ahead"},
    {"pfbuf",    1, false, "10",     "register double-buffer (+8 u32/lane)"},
    {"ballast",  1, false, "10",     "pfbuf's register-pressure control"},
    {"l0base",   0, false, "17",     "GPTQ-native, 8 strided 64 B loads"},
    {"l0b2d",    0, false, "10",     "+ 2D block read 8r16 (one message)"},
    {"l0b2dx",   0, false, "6",      "+ 2D block read and wide activations"},
    {"l0b2d16",  0, false, "9.5",    "+ 2D block read 16r16 (two k-groups)"},
    {"l0base",   0, true,  "17",     "GPTQ-native at layout 0's own best S"},
    {"l0b2d",    0, true,  "10",     "+ 2D block read 8r16, same S"},
};
struct LoadShape {
  uint32_t K, N, S_prod, S_l0;
  bool blk2d16;
  const char* name;
};
// The six shapes the engine binds, at the S it binds them at (model/qwen35.cc),
// and the S layout 0 picks for itself in docs/probe-gemv-2026-08-24.md.
const LoadShape kShapes[] = {
    {6144, 5120, 16, 4, true, "out/o_proj"},   {5120, 14336, 1, 2, true, "q‖k‖v"},
    {5120, 16384, 1, 16, true, "qkv‖z"},       {5120, 34816, 4, 8, true, "gate‖up"},
    {17408, 5120, 16, 4, false, "down"},       {5120, 248320, 1, 1, true, "lm_head int4"},
};

std::string pgl_name(const char* tag, uint32_t K, uint32_t N, uint32_t S, uint32_t L) {
  return "pgl_" + std::string(tag) + "_M1_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_S" + std::to_string(S) + "_L" + std::to_string(L);
}

int run_loads(l0::Context& ctx, l0::Queue& q, l0::Fence& f) {
  std::puts("# probe_gemv --loads - the P1 load-path battery (spec 1.7 §3)\n");
  std::puts("Same arithmetic, same bytes, different message counts. `msgs` is the LSC");
  std::puts("messages one subgroup issues per k-group (64 K elements, 544 weight bytes):");
  std::puts("weights + scales + activations. Every row is held byte-identical to `base`,");
  std::puts("which is src/kernels/gemv.cl verbatim. Timing is the house 8-replay/drop-3");
  std::puts("median over 40 launches cycling NB weight copies past the 24 MB L2.\n");
  bool all_ok = true;
  std::vector<double> warm_first;   // the discarded warm-up config, kept as ramp evidence
  std::vector<double> base_first;  // GB/s of the `base` row, per shape, for the drift control
  for (const LoadShape& sh : kShapes) {
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    std::printf("\n## %s - %u×%u, %.2f MB of weights\n\n", sh.name, sh.K, sh.N,
                double(w.bytes()) / (1u << 20));
    // WARM-UP, discarded. The 8-replay/drop-3 inside `time_list` covers the
    // ramp WITHIN a configuration; it does not cover the device ramping down
    // while the host computes this shape's CPU reference, which takes seconds
    // of GPU idle. Measured 2026-08-26: without this the first recorded
    // configuration of a shape read 355 / 475 / 524 GB/s where the same binary
    // read 534 / 541 / 538 once the device was warm - +50.4% / +13.8% / +2.7%.
    // That is a clock ramp, not a variant, and it would have been attributed to
    // whichever variant happened to be first.
    const std::string warm = pgl_name("base", sh.K, sh.N, sh.S_prod, 1);
    const double warm_gbps =
        double(w.bytes()) /
        (run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, warm.c_str()}, 40).us_per_launch * 1e3);
    std::printf("| variant | L | S | msgs | µs | GB/s | %% of 590 | Δ%% vs base | bytes vs base | what |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
    std::vector<float> base_out;
    double base_gbps = 0;
    warm_first.push_back(warm_gbps);
    for (const LoadVariant& v : kVariants) {
      if (v.s_l0best && sh.S_l0 == sh.S_prod) continue;      // no second S to run
      if (!std::strcmp(v.tag, "l0b2d16") && !sh.blk2d16) continue;  // odd k-groups per slice
      const uint32_t S = v.s_l0best ? sh.S_l0 : sh.S_prod;
      const std::string name = pgl_name(v.tag, sh.K, sh.N, S, v.L);
      GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, S, v.L, name.c_str()}, 40);
      const double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
      const double err = max_abs_err(r.out, ref);
      const char* bytes = "-";
      if (base_out.empty()) {
        base_out = r.out;
        base_gbps = gbps;
        base_first.push_back(gbps);
        bytes = "(the control)";
        if (err > tol) { all_ok = false; bytes = "**WRONG vs CPU ref**"; }
      } else if (S != sh.S_prod) {
        // A different split-K width reorders the sum (S partials folded by the
        // host here, as `prep` does in situ), so this row is held to the
        // reference tolerance, not to bit-identity.
        if (err > tol) { all_ok = false; bytes = "**WRONG vs CPU ref**"; }
        else bytes = "ref ok (S reorders the sum)";
      } else {
        size_t diff = 0;
        for (size_t i = 0; i < base_out.size(); ++i)
          if (std::memcmp(&base_out[i], &r.out[i], sizeof(float)) != 0) ++diff;
        if (diff == 0) {
          bytes = "identical";
        } else {
          all_ok = false;
          bytes = "**DIFFER**";
          std::printf("<!-- %s: %zu of %zu cells differ, max abs err vs ref %.3g (tol %.3g) -->\n",
                      name.c_str(), diff, base_out.size(), err, tol);
        }
      }
      std::printf("| %s | %u | %u | %s | %.1f | %.0f | %.0f%% | %+.1f%% | %s | %s |\n", v.tag, v.L,
                  S, v.msgs, r.us_per_launch, gbps, 100.0 * gbps / kMeasured,
                  100.0 * (gbps / base_gbps - 1.0), bytes, v.what);
      std::fflush(stdout);
    }
  }
  // Drift control (the Task-A lesson): re-measure every `base` row at the end
  // of the battery on the same instrument in the same process. A battery whose
  // control has moved has not measured its variants either.
  // The ramp control. `warm` is the discarded warm-up configuration and `base`
  // is the identical binary run immediately after it: the gap between them is
  // how much of the device's clock ramp this battery would have charged to
  // whichever variant came first.
  std::puts("\n## ramp control - the discarded warm-up beside the recorded `base`\n");
  std::printf("| shape | warm-up (discarded) GB/s | recorded `base` GB/s | ramp |\n|---|---|---|---|\n");
  for (size_t i = 0; i < base_first.size(); ++i)
    std::printf("| %s | %.0f | %.0f | %+.1f%% |\n", kShapes[i].name, warm_first[i], base_first[i],
                100.0 * (base_first[i] / warm_first[i] - 1.0));

  std::puts("\n## drift control - `base` re-measured after the whole battery\n");
  std::printf("| shape | GB/s at battery start | GB/s at battery end | Δ%% |\n|---|---|---|---|\n");
  double worst = 0;
  for (size_t i = 0; i < sizeof(kShapes) / sizeof(kShapes[0]); ++i) {
    const LoadShape& sh = kShapes[i];
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    const std::string name = pgl_name("base", sh.K, sh.N, sh.S_prod, 1);
    // Same discarded warm-up as the battery: regenerating the random weights
    // above is seconds of GPU idle, and without this the control measures the
    // ramp back out of it instead of the drift it is here for.
    run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, name.c_str()}, 40);
    GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, name.c_str()}, 40);
    const double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
    const double d = 100.0 * (gbps / base_first[i] - 1.0);
    worst = std::max(worst, d < 0 ? -d : d);
    std::printf("| %s | %.0f | %.0f | %+.2f%% |\n", sh.name, base_first[i], gbps, d);
    std::fflush(stdout);
  }
  std::printf("\nworst |Δ| over the six controls: **%.2f%%** (the standing per-cell "
              "repeatability of this instrument is a median 0.21%%)\n", worst);
  std::printf("\nbyte-identity and CPU-reference check: %s\n",
              all_ok ? "every row passed" : "**A ROW FAILED - see the WRONG/DIFFER cells above**");
  return all_ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  if (argc > 1 && std::string(argv[1]) == "--loads") return run_loads(ctx, q, f);
  const double peak = kPeak;
  struct Shape { uint32_t K, N; const char* name; };
  // Names use U+2016 (‖), not '|': this table is pasted into docs/ as markdown.
  const Shape shapes[] = {{6144, 5120, "out/o_proj"}, {5120, 14336, "q‖k‖v"}, {5120, 16384, "qkv‖z"},
                          {5120, 34816, "gate‖up"}, {17408, 5120, "down"}};
  const uint32_t Ss[] = {1, 2, 4, 8, 16};
  double layout_sum[2] = {0, 0};
  bool all_ok = true;  // any row over tolerance makes the probe exit non-zero
  std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, double>> best;  // (L, shape idx) -> S -> GB/s

  std::printf("| shape | K×N | L | S | µs | GB/s | %% of 600 | max abs err | tol |\n|---|---|---|---|---|---|---|---|---|\n");
  for (uint32_t si = 0; si < 5; ++si) {
    const Shape& sh = shapes[si];
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42 + si);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    for (uint32_t L = 0; L < 2; ++L)
      for (uint32_t S : Ss) {
        GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, S, L}, 40);
        double err = max_abs_err(r.out, ref);
        if (err > tol) all_ok = false;
        double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
        best[{L, si}][S] = gbps;
        std::printf("| %s | %u×%u | %u | %u | %.1f | %.0f | %.0f%% | %.2g | %.2g |%s\n", sh.name, sh.K, sh.N, L, S,
                    r.us_per_launch, gbps, 100.0 * gbps / peak, err, tol, err <= tol ? "" : "  **WRONG**");
        std::fflush(stdout);
      }
  }
  // lm_head, bf16
  {
    std::vector<uint16_t> w_rm = random_bf16(size_t(248320) * 5120, 3, -0.05f, 0.05f);
    common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), 5120, 248320);
    std::vector<uint16_t> x = random_bf16(5120, 5);
    GemvResult r = run_gemv_bf16(ctx, q, f, w, x, 1, 10);
    double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
    std::printf("| lm_head bf16 | 5120×248320 | B | 1 | %.1f | %.0f | %.0f%% | - | - |\n", r.us_per_launch, gbps, 100.0 * gbps / peak);
  }
  std::puts("\nDecision:");
  for (uint32_t L = 0; L < 2; ++L) {
    for (uint32_t si = 0; si < 5; ++si) {
      double b = 0; for (auto& e : best[{L, si}]) b = std::max(b, e.second);
      layout_sum[L] += b;
    }
    std::printf("- layout %u: sum of best GB/s over shapes = %.0f\n", L, layout_sum[L]);
  }
  const uint32_t Lwin = layout_sum[1] > layout_sum[0] ? 1 : 0;
  std::printf("- canonical layout: %u\n", Lwin);
  for (uint32_t si = 0; si < 5; ++si) {
    double b = 0; for (auto& e : best[{Lwin, si}]) b = std::max(b, e.second);
    uint32_t pick = 16;
    for (uint32_t S : Ss) if (best[{Lwin, si}][S] >= 0.97 * b) { pick = S; break; }
    std::printf("- %s (%u×%u): S = %u\n", shapes[si].name, shapes[si].K, shapes[si].N, pick);
  }
  return all_ok ? 0 : 1;
}
