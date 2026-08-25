#pragma once
// CPU reference for src/kernels/gdn_step.cl - one GDN layer's decode step:
// depthwise conv1d update + SiLU, l2norm, and the gated delta-rule recurrence.
//
// This is not "the same maths": it is **the same op chain in the same order**,
// including every bf16 rounding and every reduction tree, so the device can be
// held to a few ulp rather than to a hand-waved tolerance. Every block below
// has a twin in gdn_step.cl and the two must be edited together. The maths
// itself is docs/03-models.md, "Layer math - verified in the modeling file"
// (the GDN block), which wins over any other text.
//
// ---------------------------------------------------------------------------
// The tile mapping (stated identically in gdn_step.cl and docs/12-kernels.md)
// ---------------------------------------------------------------------------
// The kernel's grid is (48 heads, 4 chunks); work-group `(h, c)` owns state
// columns `[32c, 32c+32)` of head `h` and is 256 work-items = **16 subgroups of
// 16 lanes** (SIMD16). With `lid = get_local_id(0)`, `s = lid / 16` (subgroup)
// and `l = lid % 16` (lane), work-item `(s, l)` owns the 8x2 tile
//
//     k-rows   8s .. 8s+7                      (subgroup s owns k-band s)
//     columns  32c + l   and   32c + l + 16    (lane l owns two of the 32)
//
// = 16 fp32 of state per work-item, 16 x 16 x 16 = 4096 = 128 k x 32 v per
// work-group. Columns are independent under the rank-1 update, so the update
// needs no communication at all; only the two contractions do.
//
// Both contractions therefore reduce **across the 16 subgroups** for a fixed
// column, and the reduction order is a property of the k-bands alone - it does
// not depend on `c` or on `l`, which is why this reference can walk whole heads
// and still reproduce the kernel's arithmetic exactly:
//
//   * `kv[v] = Σ_k S[k][v]·kf[k]` - band `b` contributes
//     `Σ_{j=0..7} S[8b+j][v]·kf[8b+j]` accumulated in ascending `j` with `fma`;
//     the 16 band partials then collapse with a fixed pairwise tree, for
//     `stride = 8, 4, 2, 1`: `p[b] += p[b + stride]` for `b < stride`.
//   * `o[v] = Σ_k qf[k]·S[k][v]` - identical shape, with `qf` in place of `kf`.
//
// The l2norm sums are a second, differently shaped tree: the kernel gives lane
// `i < 128` the term `f32(q_b[i])²` (one term per lane, so a plain multiply and
// no `fma`) and lane `128+i` the k term, then collapses each 128-wide array
// with `stride = 64, 32, …, 1`: `r[i] += r[i + stride]`.
//
// ---------------------------------------------------------------------------
// Rounding discipline (plan 3 Task 2's preamble, applied op by op)
// ---------------------------------------------------------------------------
//   * `raw_b` - the qkv linear's output, rounded to bf16 once. It is also what
//     the conv ring stores, exactly as the reference's `conv_states` hold the
//     *input* sequence, not the convolved one.
//   * the conv accumulates in fp32 over widened bf16 inputs and fp32 weights,
//     taps ascending (t = 0 oldest … t = 3 = the current token), explicit `fma`
//     on both sides so neither compiler's contraction default matters;
//   * `x_b = rne(silu_f32(conv))` - the activation's bf16 output;
//   * l2norm sums squares of the widened bf16 in fp32 (tree above),
//     `inv = 1.0f / sqrt(sum + 1e-6f)` - **never `rsqrt`** - and rounds the
//     normalised value to bf16; `q` is then widened and scaled by `1/√128` in
//     fp32, matching the reference's `query = query * scale` after `.float()`;
//   * `a`/`b` are rounded to bf16 (they are the a‖b linear's outputs) and β, g
//     are computed from them in fp32 - the reference's `.float()` path;
//   * the recurrence is pure fp32 and rounds nothing;
//   * `gdn_o` is written fp32: `prep_gated_head` (Task 2) does that rounding.
//
// `1.0f / sqrt(x)` rather than `rsqrt(x)`, and explicit `fma`, are the same two
// spellings prep_ref.h pins, for the same reason (docs/12-kernels.md,
// "Rounding discipline"). Every kernel is now built with
// `-cl-fp32-correctly-rounded-divide-sqrt` (cmake/ocloc.cmake), so `/` and
// `sqrt` agree with the host bit for bit; `exp` and `log1p` do not - OpenCL
// allows 3 and 2 ulp - and that slack is the whole reason this comparison has a
// tolerance at all.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace gdn_ref {

// The device dimensions this kernel is compiled for (mirrors gdn_step.cl).
constexpr uint32_t kHeads = 48;          // v-heads = work-groups in x
constexpr uint32_t kChunks = 4;          // state-column chunks = work-groups in y
constexpr uint32_t kDim = 128;           // head dim, k and v alike
constexpr uint32_t kQkvzN = 16384;       // qkv‖z row width (z at 10240, not read here)
constexpr uint32_t kConvRows = 10240;    // the qkv channels the depthwise conv covers
constexpr uint32_t kConvTaps = 4;
constexpr uint32_t kQOff = 0, kKOff = 2048, kVOff = 4096;   // flat qkv channel bases
constexpr uint32_t kRing = 16;           // conv ring depth (>= M + 3)
constexpr uint32_t kNegAOff = 40960;     // gdn_small, in floats: 163840 / 4
constexpr uint32_t kDtBiasOff = 41008;   //                       164032 / 4
constexpr uint32_t kAbStride = 128;      // ab_out row: a at [0,48), b at [48,96)
constexpr uint32_t kBOff = 48;
constexpr float kQScale = 0.08838834764831845f;   // 1/sqrt(128), applied to q in fp32
constexpr uint32_t kBands = 16, kBandK = 8;       // the kv/o tree: 16 bands of 8 k-rows

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }

// silu(x) = x / (1 + exp(-x)) - plain `exp`, matching gdn_step.cl's plain `exp`.
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
// torch's softplus threshold, spelled identically on both sides: above 20 the
// function is its own argument to far more than fp32 can hold.
inline float softplus_f32(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// One decode step of one GDN layer, over every head. `conv_ring` and `state`
// are updated in place; `gdn_o` is written. `pos` is the absolute position of
// token 0 of this step and `n_act` the tokens in it (n_act <= M).
//
//   qkvz  fp32 [1][M][16384]  the qkv‖z GEMV's split-K partials (S = 1)
//   ab    fp32 [M][128]       a at [0,48), b at [48,96)
//   small fp32                the layer's GDN block: conv[10240][4] at 0,
//                             negA[48] at kNegAOff, dt_bias[48] at kDtBiasOff
//   ring  bf16 [16][10240]    this layer's conv ring, slot-major
//   state fp32 [48][128][128] this layer's S, k-major rows and v columns
//   o     fp32 [M][48][128]
inline void step(uint32_t pos, uint32_t n_act, uint32_t M, const float* qkvz, const float* ab,
                 const float* small, uint16_t* ring, float* state, float* o_out) {
  // --- conv1d update + SiLU, every qkv channel, every active token ----------
  // One channel at a time with a 4-wide sliding window, exactly as one kernel
  // work-item walks its channel: the window's three older slots come from the
  // ring (positions < pos) and then from this step's own raw values, and a
  // position below zero contributes 0 - the reference's zero-initialised conv
  // state. The ring write is the RAW value, before the conv and before silu.
  std::vector<uint16_t> xb(size_t(M) * kConvRows);   // the conv+silu outputs, bf16
  for (uint32_t ch = 0; ch < kConvRows; ++ch) {
    const float* w = small + size_t(ch) * kConvTaps;
    float win[kConvTaps];
    for (uint32_t j = 3; j >= 1; --j) {              // window slots 0,1,2 = pos-3,-2,-1
      const int64_t p = int64_t(pos) - int64_t(j);
      win[3 - j] = p < 0 ? 0.0f : f32(ring[size_t(uint64_t(p) % kRing) * kConvRows + ch]);
    }
    for (uint32_t m = 0; m < n_act; ++m) {
      const uint16_t raw_b = rne(qkvz[size_t(m) * kQkvzN + ch]);   // gdn_step.cl: raw_b
      ring[size_t((pos + m) % kRing) * kConvRows + ch] = raw_b;
      win[3] = f32(raw_b);
      float acc = 0.0f;
      for (uint32_t t = 0; t < kConvTaps; ++t) acc = std::fma(w[t], win[t], acc);   // t ascending
      xb[size_t(m) * kConvRows + ch] = rne(silu_f32(acc));         // gdn_step.cl: xs[which][m][i]
      win[0] = win[1];
      win[1] = win[2];
      win[2] = win[3];
    }
  }

  // --- per head: scalars, l2norm, recurrence --------------------------------
  std::vector<float> qf(kDim), kf(kDim), rq(kDim), rk(kDim);
  std::vector<float> kv(kDim), dl(kDim), ov(kDim);
  for (uint32_t h = 0; h < kHeads; ++h) {
    const uint32_t kh = h / 3;                       // repeat_interleave(·, 3): v-head -> k-head
    float* S = state + size_t(h) * kDim * kDim;      // S[k * 128 + v]
    const float negA = small[kNegAOff + h], dt_bias = small[kDtBiasOff + h];

    for (uint32_t m = 0; m < n_act; ++m) {
      // Head scalars, fp32 from the a‖b linear's bf16 outputs.
      const uint16_t a_b = rne(ab[size_t(m) * kAbStride + h]);          // gdn_step.cl: a_b
      const uint16_t b_b = rne(ab[size_t(m) * kAbStride + kBOff + h]);  // gdn_step.cl: b_b
      const float g = negA * softplus_f32(f32(a_b) + dt_bias);     // negA = -exp(A_log), baked
      const float beta = 1.0f / (1.0f + std::exp(-f32(b_b)));
      const float decay = std::exp(g);

      const uint16_t* qb = &xb[size_t(m) * kConvRows + kQOff + kh * kDim];
      const uint16_t* kb = &xb[size_t(m) * kConvRows + kKOff + kh * kDim];
      const uint16_t* vb = &xb[size_t(m) * kConvRows + kVOff + h * kDim];

      // l2norm: one term per lane (a plain multiply, no fma), then the 128-wide
      // pairwise tree 128 -> 64 -> … -> 1, for q and k side by side.
      for (uint32_t i = 0; i < kDim; ++i) {
        const float qv = f32(qb[i]), kvv = f32(kb[i]);
        rq[i] = qv * qv;
        rk[i] = kvv * kvv;
      }
      for (uint32_t stride = kDim / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) {
          rq[i] += rq[i + stride];
          rk[i] += rk[i + stride];
        }
      const float inv_q = 1.0f / std::sqrt(rq[0] + 1e-6f);
      const float inv_k = 1.0f / std::sqrt(rk[0] + 1e-6f);
      for (uint32_t i = 0; i < kDim; ++i) {
        qf[i] = f32(rne(f32(qb[i]) * inv_q)) * kQScale;   // gdn_step.cl: qf_s, scaled after the round
        kf[i] = f32(rne(f32(kb[i]) * inv_k));             // gdn_step.cl: kf_s, not scaled
      }

      // The recurrence, pure fp32 (docs/03-models.md): decay, contract, delta,
      // rank-1 update, contract again. Nothing here rounds.
      for (uint32_t k = 0; k < kDim; ++k)
        for (uint32_t v = 0; v < kDim; ++v) S[k * kDim + v] *= decay;

      for (uint32_t v = 0; v < kDim; ++v) {
        float p[kBands];
        for (uint32_t b = 0; b < kBands; ++b) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < kBandK; ++j) {
            const uint32_t k = b * kBandK + j;
            acc = std::fma(S[k * kDim + v], kf[k], acc);
          }
          p[b] = acc;
        }
        for (uint32_t stride = kBands / 2; stride > 0; stride >>= 1)
          for (uint32_t b = 0; b < stride; ++b) p[b] += p[b + stride];
        kv[v] = p[0];
      }
      for (uint32_t v = 0; v < kDim; ++v) dl[v] = (f32(vb[v]) - kv[v]) * beta;
      for (uint32_t k = 0; k < kDim; ++k)
        for (uint32_t v = 0; v < kDim; ++v)
          S[k * kDim + v] = std::fma(kf[k], dl[v], S[k * kDim + v]);

      for (uint32_t v = 0; v < kDim; ++v) {
        float p[kBands];
        for (uint32_t b = 0; b < kBands; ++b) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < kBandK; ++j) {
            const uint32_t k = b * kBandK + j;
            acc = std::fma(qf[k], S[k * kDim + v], acc);
          }
          p[b] = acc;
        }
        for (uint32_t stride = kBands / 2; stride > 0; stride >>= 1)
          for (uint32_t b = 0; b < stride; ++b) p[b] += p[b + stride];
        ov[v] = p[0];
      }
      for (uint32_t v = 0; v < kDim; ++v)
        o_out[(size_t(m) * kHeads + h) * kDim + v] = ov[v];
    }
  }
}

}  // namespace gdn_ref
