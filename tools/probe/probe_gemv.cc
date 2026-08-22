// probe_gemv: the production GEMV at every production shape, both layouts,
// S in {1,2,4,8,16}; GB/s of weight bytes (int4 nibbles + f16 scales), and a
// correctness check against the CPU reference at every configuration.
// Decision rules (spec 1 §4.2): layout = higher GB/s summed over the five
// int4 shapes; S per shape = smallest S within 3% of that shape's best.
//
// The weights are random, not a uniform fill: the B70 compresses device-local
// memory losslessly, so a repeated word reads back far above the 608 GB/s
// theoretical peak (doc 01). Model weights are incompressible; so is this.
#include <algorithm>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>
#include "common/int4.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  const double peak = 600.0;  // GB/s, doc 01
  struct Shape { uint32_t K, N; const char* name; };
  const Shape shapes[] = {{5120, 5120, "out/o_proj"}, {5120, 14336, "q|k|v"}, {5120, 16384, "qkv|z"},
                          {5120, 34816, "gate|up"}, {17408, 5120, "down"}};
  const uint32_t Ss[] = {1, 2, 4, 8, 16};
  double layout_sum[2] = {0, 0};
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
  return 0;
}
