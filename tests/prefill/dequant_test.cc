// pf_dequant_tile: every int4 tile layout the loader produces, in both output
// orientations, BIT-EXACT against common::Int4Gptq::at rounded once to bf16.
// The same fp32 expression on both sides, so a differing bit is a bug and
// never a tolerance question -- the discipline
// tests/loader/dequant_fixture_test.cc established across languages, applied
// here across devices.
#include <cstdio>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "dequant_harness.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  // K and N cover complete group-64 and column-16 tiles in both layouts.
  constexpr uint32_t K = 256;
  constexpr uint32_t N = 64;
  const common::Int4Gptq weights = common::Int4Gptq::random(K, N, 7);
  std::vector<uint16_t> kn(size_t(K) * N), nk(size_t(K) * N);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) {
      const uint16_t value = common::f32_to_bf16(weights.at(k, n));
      kn[size_t(k) * N + n] = value;
      nk[size_t(n) * K + k] = value;
    }

  for (uint32_t layout = 0; layout < 2; ++layout)
    for (uint32_t transposed = 0; transposed < 2; ++transposed) {
      const std::vector<uint16_t> got =
          run_dequant(ctx, q, f, weights, K, N, layout, transposed);
      const std::vector<uint16_t>& ref = transposed ? nk : kn;
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n) {
          const size_t index = transposed ? size_t(n) * K + k : size_t(k) * N + n;
          if (got[index] != ref[index]) {
            const uint32_t word = weights.qweight[size_t(k / 8) * N + n];
            const uint16_t scale = weights.scales[size_t(k / 64) * N + n];
            std::fprintf(stderr,
                         "pf_dequant_tile L%u T%u mismatch at k=%u n=%u: "
                         "got=0x%04X ref=0x%04X (word=0x%08X nibble q=%u "
                         "scale=0x%04X=%.9g)\n",
                         layout, transposed, k, n, got[index], ref[index], word,
                         (word >> (4 * (k % 8))) & 0xFu, scale,
                         double(common::f16_to_f32(scale)));
            return 1;
          }
        }
      std::printf("pf_dequant_tile L%u T%u: %zu elements bit-exact\n",
                  layout, transposed, ref.size());
    }
  return 0;
}
