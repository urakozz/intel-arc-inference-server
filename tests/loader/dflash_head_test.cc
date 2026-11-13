// Spec 19v2 C3: candidate head quantisation preserves the loaded int8 head's effective weights on its int4 grid.
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "loader/dflash_repack.h"

int main() {
  constexpr unsigned K = 256, N = 32;
  std::vector<int8_t> q(size_t(K) * N);
  std::vector<float> scales(N);
  for (unsigned n = 0; n < N; ++n) {
    scales[n] = 0.0007f + float(n) * 0.00013f;
    for (unsigned k = 0; k < K; ++k) {
      const size_t at = (size_t(n / 16) * (K / 16) + k / 16) * 256 + (k % 16) * 16 + n % 16;
      q[at] = n ? int8_t(int((k * 37 + n * 19) % 255) - 127) : 0;
    }
  }
  const auto h = loader::dflash_candidate_head(q.data(), scales.data(), K, N);
  CHECK(h.kind == model::WeightKind::Int4 && h.shape().layout == 1 && h.shape().S == 1);
  CHECK(h.K == K && h.N == N);
  CHECK_EQ(h.bytes(), size_t(N / 16) * (K / 64) * 136 * 4);
  CHECK(h.bf16.empty() && h.i8.empty() && h.scales.empty());
  for (unsigned n = 0; n < N; ++n)
    for (unsigned k = 0; k < K; ++k) {
      const auto* tile = h.int4.data() + (size_t(n / 16) * (K / 64) + k / 64) * 136;
      const uint16_t sh = uint16_t(tile[128 + n % 16 / 2] >> (16 * (n % 2)));
      const float s = common::f16_to_f32(sh);
      const unsigned word = tile[((k % 64) / 8) * 16 + n % 16];
      const int code = int((word >> (4 * (k % 8))) & 15u) - 8;
      const size_t at = (size_t(n / 16) * (K / 16) + k / 16) * 256 + (k % 16) * 16 + n % 16;
      const float original = float(q[at]) * scales[n];
      if (!n) CHECK(s == 0 && code == 0);
      else CHECK(s > 0 && std::fabs(original - float(code) * s) <= 0.51f * s);
    }
  CHECK(h.int4 == loader::dflash_candidate_head(q.data(), scales.data(), K, N).int4);
  std::puts("dflash_head_test OK: exact int8 source, int4 grid error, zero row, layout and determinism");
}
