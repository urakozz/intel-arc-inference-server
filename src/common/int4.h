#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "common/bf16.h"

namespace common {
// GPTQ layout 0 (spec 1 §6.3): qweight[K/8][N] u32, nibble i of word (r, n)
// is the weight at k = r*8 + i, column n; scales f16 [K/64][N]; symmetric,
// zero point 8 (stored as 7 in the checkpoint's v1 qzeros, which we drop).
struct Int4Gptq {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;

  static constexpr uint32_t kGroup = 64;

  float at(uint32_t k, uint32_t n) const {
    uint32_t word = qweight[(k / 8) * N + n];
    int q = int((word >> (4 * (k % 8))) & 0xFu);
    return float(q - 8) * f16_to_f32(scales[(k / kGroup) * N + n]);
  }
  size_t bytes() const { return qweight.size() * 4 + scales.size() * 2; }

  // Uniform random nibbles, scales in [0.02, 0.08]: typical magnitude of a
  // 27B checkpoint's group scales, so outputs are O(1) for unit inputs.
  static Int4Gptq random(uint32_t K, uint32_t N, uint32_t seed) {
    Int4Gptq w; w.K = K; w.N = N;
    std::mt19937 rng(seed);
    w.qweight.resize(size_t(K / 8) * N);
    for (auto& v : w.qweight) v = rng();
    w.scales.resize(size_t(K / kGroup) * N);
    std::uniform_real_distribution<float> sd(0.02f, 0.08f);
    for (auto& s : w.scales) s = f32_to_f16(sd(rng));
    return w;
  }
};
}  // namespace common
