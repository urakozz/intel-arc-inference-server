#pragma once
// CPU reference for src/kernels/attn.cl - the three kernels that make up one
// full-attention layer's decode step: `attn_prep`, `attn_decode`, `attn_reduce`.
// 16 of the model's 64 layers run this trio.
//
// As with gdn_ref.h, this is not "the same maths": it is **the same op chain in
// the same order**, every bf16 rounding and every reduction/merge order
// included, so the device can be held to a few ulp rather than to a hand-waved
// tolerance. Every block below has a twin in attn.cl and the two must be edited
// together. The maths itself is docs/03-models.md, "Layer math - verified in the
// modeling file" (the full-attention block), which wins over any other text.
//
// ---------------------------------------------------------------------------
// Column map of the fused qkv linear (doc 03; model::Qwen35's `LinearId::Qkv`)
// ---------------------------------------------------------------------------
// `q_proj ‖ k_proj ‖ v_proj` = 12288 + 1024 + 1024 = 14336 columns, split-K
// S = 2. Inside q_proj the 24 heads are **interleaved per head**, not two
// halves: head `h` is `[h·512, h·512+256)` and its gate is the next 256.
// k-head `j` is at `12288 + j·256`, v-head `j` at `13312 + j·256`.
//
// ---------------------------------------------------------------------------
// The orders, stated identically in attn.cl and docs/12-kernels.md
// ---------------------------------------------------------------------------
//  1. **The RMSNorm sum of squares** (`attn_prep`): the work-group is 256 lanes
//     and the head is 256 wide, so lane `i` contributes exactly one term,
//     `f32(x_b[i])²` - a plain multiply, no `fma` - and the 256-wide array
//     collapses with a fixed pairwise tree: for `stride = 128, 64, …, 1`,
//     `red[i] += red[i + stride]`.
//  2. **The score dot** (`attn_decode`): subgroup `s` of the 16 owns one
//     position; its lane `l` accumulates the 16 elements `d = l + 16·t`,
//     `t = 0..15` **ascending**, with an explicit `fma`, into `dot_red[16s + l]`.
//     The 16 lane partials then collapse with a fixed pairwise tree: for
//     `stride = 8, 4, 2, 1`, `dot_red[16s + l] += dot_red[16s + l + stride]`.
//     `dot_red[16s]` × 1/16 is the score.
//  3. **The online softmax wave** (`attn_decode`): a block of `kBlock` = 64
//     positions is walked in 4 waves of 16. The wave is 16 because the
//     work-group has 16 subgroups; the wave *count* is `kBlock / 16`, which is
//     the one thing spec 1.5's lever L5 changed. Given the wave's scores
//     `sc[0..15]` (−INF where the causal bound masks the position),
//         nmx  = max(mx, sc[0], sc[1], …, sc[15])        (ascending s)
//         resc = exp(mx − nmx)                            (0 when mx = −INF)
//         w[s] = exp(sc[s] − nmx) ;  ssum = Σ_s w[s]      (ascending s)
//         sm   = fma(sm, resc, ssum)
//         mx   = nmx
//     and, per dim `d`,
//         t      = Σ_s fma(w[s], f32(kv_v[p_s][j][d]), t) (ascending s)
//         acc[d] = fma(acc[d], resc, t)
//     A wave whose every position is masked **and** with no earlier valid
//     position leaves `nmx = −INF`; that case is skipped whole (`resc = 1`,
//     `w = 0`), because `exp(−INF − (−INF))` is a NaN and nothing else here is.
//  4. **The block merge** (`attn_reduce`): blocks are merged in **ascending
//     block order**, `b = 0 … nb−1` with `nb = (pos + m)/kBlock + 1`:
//         nmx = max(mx, bmx) ; a = exp(mx − nmx) ; bs = exp(bmx − nmx)
//         sm  = fma(sm, a, bsm·bs)
//         acc[d] = fma(acc[d], a, bacc[d]·bs)
//         mx  = nmx
//     Every one of those `nb` blocks has at least its own first position inside
//     the causal bound (`kBlock·b ≤ pos + m`), so `bmx` is finite and `nmx` is never
//     −INF. Blocks `≥ nb` are never read - which is what lets `attn_decode`'s
//     per-block early-out leave them untouched.
//
// ---------------------------------------------------------------------------
// Rounding discipline (plan 3 Task 2's preamble, applied op by op)
// ---------------------------------------------------------------------------
//   * `x_b = rne_bf16(Σ_s partials[s][…])` - the qkv linear's bf16 output,
//     summed in ascending slice order and rounded once;
//   * the norm widens to fp32, sums squares in fp32, uses `1.0f / sqrt(mean +
//     1e-6f)` - **never `rsqrt`** - multiplies by the fp32 `(1 + w)` weight the
//     loader baked (loader/small_layout.h) and rounds back to bf16, then widens
//     again: `nrm[i] = f32(rne_bf16(f32(x_b[i]) · rstd · w[i]))`;
//   * **RoPE runs on that fp32 widened normalised value.** `attn_q` keeps the
//     fp32 result; the k written to the cache is `rne_bf16` of it. torch reaches
//     the same value through bf16 tensor ops and therefore rounds once more,
//     inside the cos/sin multiply-add - a **≤ 1-op difference**, taken
//     deliberately (controller ruling 2026-08-25) and recorded in
//     docs/12-kernels.md;
//   * `attn_gate[m][h][i] = f32(rne_bf16(partials_gate))` - the linear's
//     rounding, widened for the sigmoid that comes at the very end;
//   * `kv_v` is just `rne_bf16(partials)`: v is never normed and never roped;
//   * the whole softmax - scores, `mx`, `sm`, `acc`, the merge - is fp32 and
//     rounds nothing. `attn_part` is fp32 for the same reason;
//   * the final chain is the reference's: eager attention output cast to bf16,
//     then `attn_output * sigmoid(gate)` as a bf16 × fp32 → fp32 product cast
//     back - `rne_bf16(f32(rne_bf16(acc/sm)) · sigmoid_f32(gate))`.
//
// `1.0f / sqrt(x)` rather than `rsqrt(x)`, and explicit `fma`, are the two
// spellings prep_ref.h and gdn_ref.h pin, for the same reason
// (docs/12-kernels.md, "Rounding discipline"). Every kernel is built with
// `-cl-fp32-correctly-rounded-divide-sqrt` (cmake/ocloc.cmake), so `/` and
// `sqrt` agree with the host bit for bit. **`exp` does not** - OpenCL allows
// 3 ulp - and that slack, in the softmax and in the final sigmoid, is the whole
// reason `attn_out` carries a tolerance while `kv_k` / `kv_v` do not.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/bf16.h"

namespace attn_ref {

// The device dimensions this trio is compiled for (mirrors attn.cl).
constexpr uint32_t kQHeads = 24;        // full-attention q-heads
constexpr uint32_t kKvHeads = 4;        // k/v heads; GQA 6:1
constexpr uint32_t kGqa = 6;            // q-heads per kv-head
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kQkvN = 14336;       // q‖gate (12288) ‖ k (1024) ‖ v (1024)
constexpr uint32_t kQkvS = 2;           // qkv split-K slices (model::Qwen35's table)
constexpr uint32_t kKOff = 12288, kVOff = 13312;
constexpr uint32_t kRotHalf = 32, kRotDim = 64;   // partial RoPE: dims 0..63
constexpr uint32_t kQNormOff = 0;       // FA small block, in floats: 0 / 4
constexpr uint32_t kKNormOff = 256;     //                         1024 / 4
// KV positions per attn_decode work-group - attn.cl's ATTN_BLOCK, which spec
// 1.5's lever L5 took from 256 to 64. Everything block-shaped in this file is
// DERIVED from it (`kWaves` below, `nblocks = max_len / kBlock` in both
// kernels, `nb = (pos + m)/kBlock + 1` in the merge), so the reference follows
// the kernel's blocking by construction and cannot be left modelling the old
// one. Only the *wave* is fixed at 16 positions: it is the work-group's
// subgroup count, not a function of the block.
constexpr uint32_t kBlock = 64;
constexpr uint32_t kWaveP = 16;                   // positions per wave = subgroups
constexpr uint32_t kWaves = kBlock / kWaveP;      // 4 waves of 16 positions
constexpr uint32_t kLanes = 16, kPerLane = 16;    // SIMD16: 16 lanes x 16 elements
constexpr float kScale = 0.0625f;       // 1/sqrt(256)
constexpr uint32_t kPartStride = 258;   // {mx, sm, acc[256]} per (qh, block, m)
constexpr uint32_t kOutN = 6144;        // 24 x 256

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
// sigmoid(x) = 1 / (1 + exp(-x)) - plain `exp`, matching attn.cl's plain `exp`.
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float ninf() { return -std::numeric_limits<float>::infinity(); }

// --------------------------------------------------------------------------
// attn_prep - grid (28, M), work-group 256. Work-groups 0..23 are q-heads,
// 24..27 are kv-heads; the loop below walks them in that order because the
// kernel's work-groups are independent and the order is immaterial.
//
//   partials  fp32 [2][M][14336]   the qkv GEMV's split-K partials (S = 2)
//   fa_small  fp32                 q_norm (1+w)[256] at 0, k_norm at 256
//   rope      fp32 [max_len][2][32] cos at [p][0][i], sin at [p][1][i]
//   attn_q    fp32 [M][24][256]    normed + roped, fp32
//   attn_gate fp32 [M][24][256]    f32(rne_bf16(gate columns))
//   kv_k/kv_v bf16 [max_len][4][256] each - this layer's slice
// --------------------------------------------------------------------------
inline void prep(uint32_t pos, uint32_t n_act, uint32_t M, const float* partials,
                 const float* fa_small, const float* rope, float* attn_q, float* attn_gate,
                 uint16_t* kv_k, uint16_t* kv_v) {
  std::vector<float> red(kHeadDim), nrm(kHeadDim);
  for (uint32_t wg = 0; wg < kQHeads + kKvHeads; ++wg) {
    const bool is_q = wg < kQHeads;
    const uint32_t h = is_q ? wg : wg - kQHeads;                 // q-head or kv-head
    const float* nw = fa_small + (is_q ? kQNormOff : kKNormOff);
    for (uint32_t m = 0; m < n_act; ++m) {
      const size_t base = is_q ? size_t(h) * 2 * kHeadDim
                               : size_t(kKOff) + size_t(h) * kHeadDim;
      auto qkv_sum = [&](size_t col) {
        float v = 0.0f;
        for (uint32_t s = 0; s < kQkvS; ++s)
          v += partials[(size_t(s) * M + m) * kQkvN + col];
        return v;
      };

      // The linear's bf16 output, then the norm's sum of squares: one term per
      // lane (a plain multiply, no fma) and the 256 -> 1 pairwise tree.
      for (uint32_t i = 0; i < kHeadDim; ++i) {
        const float xf = f32(rne(qkv_sum(base + i)));
        red[i] = xf * xf;
      }
      for (uint32_t stride = kHeadDim / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
      const float rstd = 1.0f / std::sqrt(red[0] / float(kHeadDim) + 1e-6f);   // never rsqrt
      for (uint32_t i = 0; i < kHeadDim; ++i)
        nrm[i] = f32(rne(f32(rne(qkv_sum(base + i))) * rstd * nw[i]));

      // Partial RoPE over dims 0..63, pairs (i, i+32), on the fp32 widened
      // normalised value. rotate_half over the 64-slice:
      //   out_i      = x_i·cos_i     − x_{i+32}·sin_i
      //   out_{i+32} = x_{i+32}·cos_i + x_i·sin_i
      // One rounded product plus one fma, spelled identically in attn.cl.
      // Dims 64..255 pass through untouched.
      const float* cs = rope + size_t(pos + m) * 2 * kRotHalf;
      for (uint32_t i = 0; i < kHeadDim; ++i) {
        float outv;
        if (i < kRotHalf) {
          const float c = cs[i], s = cs[kRotHalf + i];
          const float t = nrm[i + kRotHalf] * s;
          outv = std::fma(nrm[i], c, -t);
        } else if (i < kRotDim) {
          const uint32_t ii = i - kRotHalf;
          const float c = cs[ii], s = cs[kRotHalf + ii];
          const float t = nrm[ii] * s;
          outv = std::fma(nrm[i], c, t);
        } else {
          outv = nrm[i];
        }
        if (is_q)
          attn_q[(size_t(m) * kQHeads + h) * kHeadDim + i] = outv;
        else
          kv_k[(size_t(pos + m) * kKvHeads + h) * kHeadDim + i] = rne(outv);
      }

      if (is_q) {
        for (uint32_t i = 0; i < kHeadDim; ++i)
          attn_gate[(size_t(m) * kQHeads + h) * kHeadDim + i] =
              f32(rne(qkv_sum(base + kHeadDim + i)));
      } else {
        // v is never normed and never roped: the linear's rounding, and done.
        for (uint32_t i = 0; i < kHeadDim; ++i)
          kv_v[(size_t(pos + m) * kKvHeads + h) * kHeadDim + i] =
              rne(qkv_sum(size_t(kVOff) + size_t(h) * kHeadDim + i));
      }
    }
  }
}

// --------------------------------------------------------------------------
// attn_decode - grid (4 kv-heads, max_len/kBlock blocks), work-group 256.
// The per-block early-out and the wave scheme are order 2 and 3 of the header.
// --------------------------------------------------------------------------
inline void decode(uint32_t pos, uint32_t n_act, uint32_t M, uint32_t max_len,
                   const float* attn_q, const uint16_t* kv_k, const uint16_t* kv_v,
                   float* attn_part) {
  const uint32_t nblocks = max_len / kBlock;
  std::vector<float> acc(kHeadDim);
  for (uint32_t j = 0; j < kKvHeads; ++j) {
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
      const uint32_t bstart = blk * kBlock;
      if (bstart >= pos + n_act) continue;     // the kernel's per-BLOCK early-out
      for (uint32_t qhl = 0; qhl < kGqa; ++qhl) {
        const uint32_t qh = j * kGqa + qhl;
        for (uint32_t m = 0; m < n_act; ++m) {
          const float* q = attn_q + (size_t(m) * kQHeads + qh) * kHeadDim;
          float mx = ninf(), sm = 0.0f;
          for (uint32_t d = 0; d < kHeadDim; ++d) acc[d] = 0.0f;

          for (uint32_t w = 0; w < kWaves; ++w) {
            // The wave's 16 scores. Subgroup s owns position bstart + 16w + s;
            // its 16 lanes each take 16 elements, d = l + 16t ascending.
            float sc[kWaveP];
            for (uint32_t s = 0; s < kWaveP; ++s) {
              const uint32_t p = bstart + w * kWaveP + s;
              if (p > pos + m) { sc[s] = ninf(); continue; }     // causal mask
              const uint16_t* krow = kv_k + (size_t(p) * kKvHeads + j) * kHeadDim;
              float dot_red[kLanes];
              for (uint32_t l = 0; l < kLanes; ++l) {
                float a = 0.0f;
                for (uint32_t t = 0; t < kPerLane; ++t) {
                  const uint32_t d = l + kLanes * t;             // ascending t
                  a = std::fma(q[d], f32(krow[d]), a);
                }
                dot_red[l] = a;
              }
              for (uint32_t stride = kLanes / 2; stride > 0; stride >>= 1)
                for (uint32_t l = 0; l < stride; ++l) dot_red[l] += dot_red[l + stride];
              sc[s] = dot_red[0] * kScale;
            }

            // The online update, ascending s. `sc` becomes the weights in place,
            // exactly as attn.cl does it (one 16-float array, not two).
            float nmx = mx;
            for (uint32_t s = 0; s < kWaveP; ++s) nmx = std::fmax(nmx, sc[s]);
            float resc;
            if (nmx > ninf()) {
              resc = std::exp(mx - nmx);              // mx = -INF -> 0; mx = nmx -> 1
              float ssum = 0.0f;
              for (uint32_t s = 0; s < kWaveP; ++s) {
                sc[s] = std::exp(sc[s] - nmx);        // masked -> exp(-INF) = 0
                ssum += sc[s];
              }
              sm = std::fma(sm, resc, ssum);
              mx = nmx;
            } else {
              resc = 1.0f;                            // nothing valid yet: skip whole
              for (uint32_t s = 0; s < kWaveP; ++s) sc[s] = 0.0f;
            }

            for (uint32_t d = 0; d < kHeadDim; ++d) {
              float t = 0.0f;
              for (uint32_t s = 0; s < kWaveP; ++s) {
                const uint32_t p = bstart + w * kWaveP + s;
                // The masked slot's weight is already 0; the load is skipped
                // anyway so an unwritten KV slot can never turn 0·x into a NaN.
                const float vf =
                    p <= pos + m ? f32(kv_v[(size_t(p) * kKvHeads + j) * kHeadDim + d]) : 0.0f;
                t = std::fma(sc[s], vf, t);
              }
              acc[d] = std::fma(acc[d], resc, t);
            }
          }

          float* out = attn_part + ((size_t(qh) * nblocks + blk) * M + m) * kPartStride;
          out[0] = mx;
          out[1] = sm;
          for (uint32_t d = 0; d < kHeadDim; ++d) out[2 + d] = acc[d];
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// attn_reduce - grid (24 q-heads, M), work-group 256. Order 4 of the header,
// then the final gated chain.
// --------------------------------------------------------------------------
inline void reduce(uint32_t pos, uint32_t n_act, uint32_t M, uint32_t max_len,
                   const float* attn_part, const float* attn_gate, uint16_t* attn_out) {
  const uint32_t nblocks = max_len / kBlock;
  std::vector<float> acc(kHeadDim);
  for (uint32_t h = 0; h < kQHeads; ++h) {
    for (uint32_t m = 0; m < n_act; ++m) {
      const uint32_t nb = (pos + m) / kBlock + 1;
      float mx = ninf(), sm = 0.0f;
      for (uint32_t d = 0; d < kHeadDim; ++d) acc[d] = 0.0f;
      for (uint32_t b = 0; b < nb; ++b) {          // ascending block order
        const float* p = attn_part + ((size_t(h) * nblocks + b) * M + m) * kPartStride;
        const float bmx = p[0], bsm = p[1];
        const float nmx = std::fmax(mx, bmx);
        const float a = std::exp(mx - nmx);        // mx = -INF on the first block -> 0
        const float bs = std::exp(bmx - nmx);
        sm = std::fma(sm, a, bsm * bs);
        for (uint32_t d = 0; d < kHeadDim; ++d) acc[d] = std::fma(acc[d], a, p[2 + d] * bs);
        mx = nmx;
      }
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        const float o = acc[d] / sm;
        const float g = attn_gate[(size_t(m) * kQHeads + h) * kHeadDim + d];
        attn_out[size_t(m) * kOutN + size_t(h) * kHeadDim + d] =
            rne(f32(rne(o)) * sigmoid_f32(g));
      }
    }
  }
}

}  // namespace attn_ref
