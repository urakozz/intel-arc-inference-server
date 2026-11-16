#pragma once
// The bf16 eager decode attention chain on the host: transformers' eager path, torch's CPU fp32 softmax
// bit for bit. Shared by the eager references of Kolibri-1 (kolibri_ref.h) and
// Qwen3.8-Flash-Next (qwen4exp_ref.h, qwen4exp_pf_ref.h).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/bf16.h"
#include "kernels/moe_ref.h"   // f32 / rf

namespace eager_attn_ref {

using moe_ref::f32;
using moe_ref::rf;

// --- eager decode attention (the host chain) ---------------------------------------------------
// transformers' bf16 eager path, one q head over
// keys 0..len-1, every rounding point at the reference's place:
//
//   s_p = rne(f32(rne(Σ_d q_d·k_pd)) · scale)    the bf16 GEMM's one rounding, then `* scaling`
//                                               on the bf16 tensor (fp32 inside, rounded)
//   p_p = rne(e_p · (1 / Σ e))  e_p = exp(s_p - max)   torch's fp32 softmax on the CPU
//         (SoftMaxKernel.cpp _vec_softmax_lastdim at AVX2: Sleef_expf8_u10, the 8-lane sum,
//         one reciprocal), then .to(bf16)
//   o_d = Σ_p p_p·v_pd                           the bf16 GEMM: fp32 sum, rounded once
//
// What is bitwise against torch, and why (measured against a torch dump of the eager path):
//   * the softmax: exact. exp_torch() is Sleef's expf u10 step for step (fma as the AVX2 build
//     has it), the sum is reduce_all's order (8 lane accumulators, then the permute tree), the
//     reciprocal and the multiply are torch's. Probed on 22,760 probabilities: 0 fp32
//     differences (std::exp: 2,101; a sequential sum: 20,069).
//   * the score's rounding points: exact. The dot's ACCUMULATION ORDER is ours (one fma chain,
//     d ascending; every product of two bf16 values is exact in fp32, so fma = mul + add and
//     the chain is contraction-proof); torch's GEMM order is its own and differs between the
//     prompt pass and a decode step. The two round apart only where the fp32 dot sits within
//     its accumulation error of a bf16 rounding boundary (~2^-16 x a few per score).
//   * P·V: same - our order (block sums ascending, each an fma chain), one rounding.
// The device kernel computes exactly these functions in exactly these orders (exp_torch is
// in the kernel too), so kernel against this reference is bitwise.
constexpr float kAttnScale = 0.08838834764831845f;   // float(128 ** -0.5): torch's scalar cast

// Sleef_expf8_u10 (sleefsimdsp.c xexpf at AVX2 + FMA): torch's Vectorized<float>::exp, which is
// what the fp32 softmax evaluates (torch.exp itself goes to MKL's vsExp and is NOT this).
inline float exp_torch(float d) {
  const float kRLn2 = 1.442695040888963407359924681001892137426645954152985934135449406931f;
  const float kL2U = 0.693145751953125f, kL2L = 1.428606765330187045e-06f;
  const int q = int(std::nearbyint(d * kRLn2));   // cvtps2dq: nearest-even
  const float qf = float(q);
  float s = std::fma(qf, -kL2U, d);
  s = std::fma(qf, -kL2L, s);
  float u = 0.000198527617612853646278381f;
  u = std::fma(u, s, 0.00139304355252534151077271f);
  u = std::fma(u, s, 0.00833336077630519866943359f);
  u = std::fma(u, s, 0.0416664853692054748535156f);
  u = std::fma(u, s, 0.166666671633720397949219f);
  u = std::fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + std::fma(ss, u, s);
  const auto pow2i = [](int e) {   // vpow2i_vf_vi2: exponent field only (e in -75 .. 75 here)
    const uint32_t b = uint32_t(e + 127) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  if (d < -104.0f) return 0.0f;   // before ldexp: q would leave pow2i's range
  if (100.0f < d) return INFINITY;
  u = u * pow2i(q >> 1);
  return u * pow2i(q - (q >> 1));
}

inline float eager_score(const float* q, const uint16_t* k, uint32_t hd) {
  float a = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) a = std::fma(q[d], f32(k[d]), a);
  return rf(rf(a) * kAttnScale);
}

// p[0..len) from s[0..len) (bf16 values). The sum is reduce_all's at a row of >= 8 entries;
// a row shorter than 8 is summed as the reference's PROMPT pass sums it (padded with exact
// zeros to the prompt's length, >= 8 for every golden prompt) - a decode-only row of < 8 keys
// would be sequential in torch, which the engine never meets after a prompt of >= 8 tokens.
inline void eager_softmax(const float* s, uint32_t len, float* p) {
  float mx = -INFINITY;
  for (uint32_t i = 0; i < len; ++i) mx = std::max(mx, s[i]);
  float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < len; ++i) {
    p[i] = exp_torch(s[i] - mx);
    lane[i % 8] += p[i];   // lane i % 8's chain, ascending i (0 + e = e exactly)
  }
  const float sum = ((lane[0] + lane[4]) + (lane[2] + lane[6])) + ((lane[1] + lane[5]) + (lane[3] + lane[7]));
  const float inv = 1.0f / sum;
  for (uint32_t i = 0; i < len; ++i) p[i] = rf(p[i] * inv);
}

// A 32-bit integer hash (the families' synthetic test inputs).
inline uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

}  // namespace eager_attn_ref
