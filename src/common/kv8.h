#pragma once
// Spec 12 (the operator's `rotkv` ruling, spec 12 §8): the int8 KV cache's arithmetic on
// the host - the ONE home of the scheme's definition outside src/kernels/kv8.cl, which
// repeats every statement below and must be edited together with this file.
//
//   R = H_256 · diag(s) / 16            Sylvester H (natural order), s = the probe's signs
//   rotate(x)   = x R    : y[j] = s[j] · FWHT(x)[j] / 16
//   unrotate(y) = y R^T  : x[i] = FWHT(s ⊙ y)[i] / 16
//
// R is the rotation tools/oracle/kv_int8_probe.py measured (`hadamard(256, 0)`: torch's
// Generator seeded 0, randint(0, 2, (256,)) * 2 - 1), so the engine and the 12a probe
// rotate identically; kSignWords is that vector, bit j of word j / 32 set when s[j] = -1
// (tests/common/kv8_test.cc pins the bit string the probe printed).
//
// Per (position, kv head), K and V alike, after rotation:
//   amax = max_i |y[i]|                 (fp32; max is exact in any order)
//   s16  = f16_rne(amax / 127)          (the stored scale; correctly rounded divide)
//   q[i] = s16 == 0 ? 0 : clamp(rint(y[i] / f16f(s16)), -127, 127)
//   deq  = float(q) · f16f(s16)         (exact in fp32: 7 + 11 significant bits)
// The quantiser divides by the ROUNDED scale, so |y - deq| <= f16f(s16) / 2 + (the clamp,
// only when s16 rounded below amax / 127, by at most 2^-11 of amax).
//
// The FWHT is the canonical in-place one, stages h = 1, 2, ..., 128 ascending, the pair
// (i, i + h) with i & h == 0 becoming (a + c, a - c). Every op is one fp32 rounding in a
// fixed order and the sign and the 1/16 are exact, so the device's rotation is bitwise
// this one (kv8_kernels_test).
#include <cmath>
#include <cstdint>
#include <cstring>

namespace common::kv8 {

constexpr uint32_t kHeadDim = 256;
constexpr int kQMax = 127;
constexpr float kInvSqrtN = 0.0625f;   // 1/16 = 1/sqrt(256), exact

// bit j of word j / 32: s[j] = -1. The probe's hadamard(256, 0) signs (127 of them -1).
constexpr uint32_t kSignWords[8] = {0xA197D809u, 0xA6850592u, 0xB205A73Eu, 0x20AFA769u,
                                    0x45C2CF64u, 0xBEAEAACBu, 0x75F39ADDu, 0x2385A70Cu};

inline bool sign_neg(uint32_t j) { return (kSignWords[j >> 5] >> (j & 31u)) & 1u; }

inline uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// Exact sign flip: the sign bit, not a multiply (identical result, no rounding either way).
inline float flip(float v, bool neg) { return neg ? bits_f32(f32_bits(v) ^ 0x80000000u) : v; }

// IEEE binary16, round to nearest even, subnormals kept (NOT common::f32_to_f16, which
// flushes them: a tiny row's scale must still dequantise what was quantised). Finite input;
// magnitudes >= 65504 saturate to 65504 (a scale never gets there: amax would be 8.3e6).
inline uint16_t f32_to_f16_rne(float f) {
  uint32_t u = f32_bits(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  u &= 0x7FFFFFFFu;
  if (u >= 0x477FE000u) return uint16_t(sign | 0x7BFFu);          // >= 65504: saturate
  if (u < 0x38800000u)                                             // < 2^-14: subnormal half
    return uint16_t(sign | uint32_t(std::nearbyint(bits_f32(u) * 16777216.0f)));   // x 2^24, exact
  const uint32_t e = (u >> 23) - 112u;                             // -127 + 15
  const uint32_t m = u & 0x7FFFFFu;
  uint32_t h = (e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;      // a carry rolls into e: correct
  return uint16_t(sign | h);
}
// binary16 -> fp32, exact (finite inputs; the scales are never inf or NaN).
inline float f16f(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0 ? float(m) * 5.9604644775390625e-08f      // m x 2^-24, exact
                         : bits_f32(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}

// In place, ascending stages.
inline void fwht256(float* v) {
  for (uint32_t h = 1; h < kHeadDim; h <<= 1)
    for (uint32_t i = 0; i < kHeadDim; ++i)
      if ((i & h) == 0) {
        const float a = v[i], c = v[i + h];
        v[i] = a + c;
        v[i + h] = a - c;
      }
}

// y = x R. `out` may alias `x`.
inline void rotate(const float* x, float* out) {
  float v[kHeadDim];
  std::memcpy(v, x, sizeof v);
  fwht256(v);
  for (uint32_t j = 0; j < kHeadDim; ++j) out[j] = flip(v[j], sign_neg(j)) * kInvSqrtN;
}

// x = y R^T. `out` may alias `y`.
inline void unrotate(const float* y, float* out) {
  float v[kHeadDim];
  for (uint32_t j = 0; j < kHeadDim; ++j) v[j] = flip(y[j], sign_neg(j));
  fwht256(v);
  for (uint32_t i = 0; i < kHeadDim; ++i) out[i] = v[i] * kInvSqrtN;
}

// One rotated row -> 256 int8 + its fp16 scale (returned).
inline uint16_t quantise(const float* y, int8_t* q) {
  float amax = 0.0f;
  for (uint32_t i = 0; i < kHeadDim; ++i) amax = std::fmax(amax, std::fabs(y[i]));
  const uint16_t s16 = f32_to_f16_rne(amax / 127.0f);
  const float sf = f16f(s16);
  for (uint32_t i = 0; i < kHeadDim; ++i) {
    float r = sf == 0.0f ? 0.0f : std::nearbyint(y[i] / sf);
    r = std::fmin(std::fmax(r, -127.0f), 127.0f);
    q[i] = int8_t(r);
  }
  return s16;
}

inline float dequant(int8_t q, uint16_t s16) { return float(q) * f16f(s16); }

// The writer's whole chain for one K or V row as the cache's bf16 value would hold it
// (`x` = f32(rne_bf16(.)), exactly what the bf16 cache stores): rotate, quantise.
inline uint16_t encode(const float* x, int8_t* q) {
  float y[kHeadDim];
  rotate(x, y);
  return quantise(y, q);
}

}  // namespace common::kv8
