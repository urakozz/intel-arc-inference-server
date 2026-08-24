#pragma once
// CPU references for src/kernels/prep.cl - the three between-GEMV kernels.
//
// These are not "approximately the same maths": they are the *same op chain in
// the same order*, so the device output can be compared **bit for bit**. Every
// line below has a twin in prep.cl and the two must be edited together.
//
// The rounding discipline (plan 3, Task 2 preamble) - torch rounds per op, so
// the engine does too:
//   * a GEMV's split-K partials are summed in fp32 and rounded to bf16 ONCE -
//     that bf16 value is the linear's output in the reference;
//   * the residual add is bf16-in/bf16-out (`rne(f32(a) + f32(b))`);
//   * norms widen bf16 -> fp32, multiply by the fp32 `(1 + w)` weight and round
//     the result back to bf16 (the reference's `type_as(x)`);
//   * inside-op accumulation (the variance sum) stays fp32 and is NOT matched
//     term-for-term against torch - but it IS matched term-for-term against the
//     kernel, which is what makes these comparisons exact.
//
// Two places where the *spelling* matters, and is deliberately identical here
// and in prep.cl:
//
//   1. **`1.0f / sqrt(x)`, never `rsqrt`.** OpenCL's `rsqrt` (and the host's)
//      is a ~2 ulp approximation with no cross-implementation guarantee, so a
//      bit-exact comparison through it is not available. Correctly rounded
//      `sqrt` followed by a correctly rounded divide is (controller ruling,
//      2026-08-25; docs/12-kernels.md, "Rounding discipline").
//   2. **The square-accumulate is an explicit `fma`.** `sum += v*v` may or may
//      not be contracted into a fused multiply-add by either compiler; writing
//      the fusion explicitly on both sides removes the question. The variance
//      tree is otherwise plain fp32 adds.
//
// **The variance tree order** (identical text in prep.cl): the work-group's
// WG lanes each accumulate their own strided slice - lane i takes k = i,
// i+WG, i+2·WG, … in ascending k - into one fp32 register with `fma`, write it
// to SLM, and then a fixed pairwise tree collapses SLM: for stride = WG/2,
// WG/4, …, 1, lane i < stride does `red[i] += red[i + stride]`, with a barrier
// after every step. So for WG = 256 the shape is 256 → 128 → 64 → … → 1.
// Nothing here is data-dependent, so two replays of the captured list produce
// the same bits, and so does the reference.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace prep_ref {

// The device dimensions these kernels are compiled for (mirrors prep.cl).
constexpr uint32_t kWgRes = 256, kWgSilu = 256, kWgGated = 128;
constexpr uint32_t kSiluS = 4;              // split-K slices of gate||up
constexpr uint32_t kSiluN = 17408;          // intermediate size
constexpr uint32_t kSiluFusedN = 34816;     // gate||up, interleaved in 16-column blocks
constexpr uint32_t kGatedS = 1;             // split-K slices of qkv||z
constexpr uint32_t kGatedHeads = 48, kHeadDim = 128;
constexpr uint32_t kQkvzN = 16384, kZOff = 10240;
constexpr uint32_t kGatedOutN = kGatedHeads * kHeadDim;  // 6144

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }

// silu(x) = x / (1 + exp(-x)) - plain `exp`, matching prep.cl's plain `exp`
// (not `native_exp`). This is the ONE op in the file whose result is allowed to
// differ between host and device: OpenCL permits 3 ulp on fp32 `exp`. Callers
// keep it as the last factor so everything before it stays comparable exactly.
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }

// prep_res_norm: residual add + RMSNorm, in place on `resid`.
//   mixer_b = rne(Σ_s partials[s][m][k])            (skipped when S_PREV == 0)
//   r_b     = rne(f32(resid[m][k]) + f32(mixer_b))  (S_PREV == 0: r_b = resid)
//   resid[m][k] = r_b
//   rstd    = 1 / sqrt(mean_k(f32(r_b)²) + 1e-6)    (tree order above)
//   x_out   = rne(f32(r_b) · rstd · norm_w[k])
inline void res_norm(const float* partials, uint16_t* resid, const float* norm_w, uint16_t* x_out,
                     uint32_t M, uint32_t K, uint32_t S_PREV) {
  std::vector<float> row(K);
  std::vector<float> red(kWgRes);
  for (uint32_t m = 0; m < M; ++m) {
    uint16_t* rp = resid + size_t(m) * K;
    for (uint32_t k = 0; k < K; ++k) {
      uint16_t r_b;
      if (S_PREV == 0) {
        r_b = rp[k];
      } else {
        float acc = 0.f;
        for (uint32_t s = 0; s < S_PREV; ++s) acc += partials[(size_t(s) * M + m) * K + k];
        r_b = rne(f32(rp[k]) + f32(rne(acc)));
      }
      rp[k] = r_b;
      row[k] = f32(r_b);
    }
    // Variance, in the kernel's tree order (see the header comment).
    for (uint32_t i = 0; i < kWgRes; ++i) {
      float s = 0.f;
      for (uint32_t k = i; k < K; k += kWgRes) s = std::fma(row[k], row[k], s);
      red[i] = s;
    }
    for (uint32_t stride = kWgRes / 2; stride > 0; stride >>= 1)
      for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
    const float mean = red[0] / float(K);
    const float rstd = 1.0f / std::sqrt(mean + 1e-6f);
    for (uint32_t k = 0; k < K; ++k)
      x_out[size_t(m) * K + k] = rne(row[k] * rstd * norm_w[k]);
  }
}

// prep_silu_mul: gate||up (interleaved in 16-column blocks) -> silu(gate)·up.
//   gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16
//   g_b = rne(Σ_s partials[s][m][gflat]) ;  u_b likewise
//   s_b = rne(silu_f32(f32(g_b)))
//   x_out[m][k] = rne(f32(s_b) · f32(u_b))
inline void silu_mul(const float* partials, uint16_t* x_out, uint32_t M) {
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < kSiluN; ++k) {
      const size_t gflat = size_t(k / 16) * 32 + (k % 16), uflat = gflat + 16;
      float ga = 0.f, ua = 0.f;
      for (uint32_t s = 0; s < kSiluS; ++s) {
        const size_t base = (size_t(s) * M + m) * kSiluFusedN;
        ga += partials[base + gflat];
        ua += partials[base + uflat];
      }
      const uint16_t g_b = rne(ga), u_b = rne(ua);
      const uint16_t s_b = rne(silu_f32(f32(g_b)));
      x_out[size_t(m) * kSiluN + k] = rne(f32(s_b) * f32(u_b));
    }
}

// prep_gated_head: Qwen3_5RMSNormGated over one GDN v-head (doc 03).
//   o_b = rne(gdn_o[m][h][i])                       (recurrence output -> bf16)
//   z_b = rne(Σ_s qkvz[s][m][kZOff + h·128 + i])
//   var = mean_i(f32(o_b)²)                         (128-lane tree, order above)
//   n_b = rne(f32(o_b) · (1 / sqrt(var + 1e-6)))
//   t_b = rne(f32(gated_w[i]) · f32(n_b))           (plain w, no +1)
//   x_out[m][h·128+i] = rne(f32(t_b) · silu_f32(f32(z_b)))
// The silu factor is deliberately the LAST op: everything up to and including
// `t_b` is exactly reproducible, only the final product carries `exp`'s slack.
inline void gated_head(const float* qkvz, const float* gdn_o, const uint16_t* gated_w,
                       uint16_t* x_out, uint32_t M) {
  std::vector<uint16_t> o_b(kHeadDim), z_b(kHeadDim);
  std::vector<float> red(kHeadDim);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t h = 0; h < kGatedHeads; ++h) {
      for (uint32_t i = 0; i < kHeadDim; ++i) {
        o_b[i] = rne(gdn_o[(size_t(m) * kGatedHeads + h) * kHeadDim + i]);
        float za = 0.f;
        for (uint32_t s = 0; s < kGatedS; ++s)
          za += qkvz[(size_t(s) * M + m) * kQkvzN + kZOff + h * kHeadDim + i];
        z_b[i] = rne(za);
        const float o_f = f32(o_b[i]);
        red[i] = o_f * o_f;   // one term per lane: a plain fp32 multiply, no fma
      }
      for (uint32_t stride = kWgGated / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
      const float var = red[0] / float(kHeadDim);
      const float rstd = 1.0f / std::sqrt(var + 1e-6f);
      for (uint32_t i = 0; i < kHeadDim; ++i) {
        const uint16_t n_b = rne(f32(o_b[i]) * rstd);
        const uint16_t t_b = rne(f32(gated_w[i]) * f32(n_b));
        x_out[size_t(m) * kGatedOutN + h * kHeadDim + i] =
            rne(f32(t_b) * silu_f32(f32(z_b[i])));
      }
    }
}

}  // namespace prep_ref
