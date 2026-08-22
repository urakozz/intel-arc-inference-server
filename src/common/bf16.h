#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace common {
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
// Round-to-nearest-even; NaN inputs are not expected and not handled.
inline uint16_t f32_to_bf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t((u + rounding) >> 16);
}
// IEEE half <-> float, normal range only (scales are ~1e-3..1; never subnormal).
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  uint32_t u;
  if (exp == 0) u = sign << 31;                                   // zero / subnormal -> 0
  else if (exp == 31) u = (sign << 31) | 0x7F800000u | (man << 13);
  else u = (sign << 31) | ((exp + 112u) << 23) | (man << 13);
  float f; std::memcpy(&f, &u, 4); return f;
}
inline uint16_t f32_to_f16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = u & 0x7FFFFFu;
  if (exp <= 0) return uint16_t(sign);                            // flush to zero
  if (exp >= 31) return uint16_t(sign | 0x7C00u);
  uint32_t half = sign | (uint32_t(exp) << 10) | (man >> 13);
  uint32_t rem = man & 0x1FFFu;                                   // RNE on the 13 dropped bits
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return uint16_t(half);
}

// Canonical bf16 weight layout (spec 1 §9.2): tiles [n_tile][k_octet][8 k][16 n].
// A subgroup reads one 256 B tile per 8 k with a single block read.
struct Bf16Tiled {
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> data;
  static Bf16Tiled from_rowmajor(const uint16_t* w, uint32_t K, uint32_t N) {
    Bf16Tiled t; t.K = K; t.N = N;
    t.data.resize(size_t(N) * K);
    const uint32_t K8 = K / 8;
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        t.data[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16] = w[size_t(n) * K + k];
    return t;
  }
  size_t bytes() const { return data.size() * 2; }
};
}  // namespace common
