#pragma once
#include <cstdint>
#include <random>
#include <vector>

// The rotation signs D_K of spec 5 (R_K = D_K blockdiag(H_1024) / 32): one
// mt19937 seeded with K, one bit per k, 1 meaning -1. Host-generated once;
// the kernels and every CPU mirror read the same vector.
namespace runtime::prefill {

inline std::vector<float> int8_signs(uint32_t K) {
  std::mt19937 rng(K);
  std::vector<float> d(K);
  for (float& v : d) v = (rng() & 1u) ? -1.0f : 1.0f;
  return d;
}

inline std::vector<uint32_t> int8_sign_bits(uint32_t K) {
  const std::vector<float> d = int8_signs(K);
  std::vector<uint32_t> b(K / 32, 0u);
  for (uint32_t k = 0; k < K; ++k)
    if (d[k] < 0.0f) b[k / 32] |= 1u << (k % 32);
  return b;
}

}  // namespace runtime::prefill
