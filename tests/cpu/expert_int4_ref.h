#pragma once
// Scalar fp32 twin of cpu::expert_forward (src/cpu/expert_int4.h), from the unpacked codes: no SIMD, no packing, the
// card's epilogue (bf16 gate / up, bf16 silu, bf16 h). Its group sums run in plain k order; the kernel's run in
// 8 / 16 lanes, so the two differ by fp32 accumulation-order noise (and, rarely, one bf16 rounding of a gate / up).
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace cpu_ref {

// codes [N][K] 0..15, scales [N][K/64] f16 words, x fp32 [K]: column n's dot, the card's order over groups.
inline float col_dot(const uint8_t* codes, const uint16_t* scales, const float* x, uint32_t K, uint32_t n) {
  float acc = 0.f;
  for (uint32_t g = 0; g < K / 64; ++g) {
    float gacc = 0.f;
    for (uint32_t k = g * 64; k < g * 64 + 64; ++k) gacc += float(int(codes[size_t(n) * K + k]) - 8) * x[k];
    const float t = gacc * common::f16_to_f32(scales[size_t(n) * (K / 64) + g]);
    acc += t;
  }
  return acc;
}
inline float rb(float f) { return common::bf16_to_f32(common::f32_to_bf16(f)); }

struct Result {
  std::vector<float> h, out;   // [M][inter], [M][hidden]
};
inline Result expert(const uint8_t* gu_codes, const uint16_t* gu_scales, const uint8_t* dn_codes, const uint16_t* dn_scales,
                     const uint16_t* x_bf16, uint32_t M, uint32_t hidden, uint32_t inter) {
  Result r;
  r.h.resize(size_t(M) * inter);
  r.out.resize(size_t(M) * hidden);
  for (uint32_t m = 0; m < M; ++m) {
    std::vector<float> x(hidden);
    for (uint32_t k = 0; k < hidden; ++k) x[k] = common::bf16_to_f32(x_bf16[size_t(m) * hidden + k]);
    for (uint32_t i = 0; i < inter; ++i) {
      const float g = rb(col_dot(gu_codes, gu_scales, x.data(), hidden, i));
      const float u = rb(col_dot(gu_codes, gu_scales, x.data(), hidden, inter + i));
      const float s = rb(g / (1.0f + std::exp(-g)));
      r.h[size_t(m) * inter + i] = rb(s * u);
    }
    for (uint32_t n = 0; n < hidden; ++n)
      r.out[size_t(m) * hidden + n] = col_dot(dn_codes, dn_scales, &r.h[size_t(m) * inter], inter, n);
  }
  return r;
}

}  // namespace cpu_ref
