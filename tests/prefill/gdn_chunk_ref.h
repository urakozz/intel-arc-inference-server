#pragma once
// CPU reference for `gdn_chunk` - the WY-representation chunked gated delta
// rule, in the same eight stages the nine `pf_gdn_*` kernels run, with every
// bf16 rounding and every reduction order stated.
//
// This is not "the same maths": it is **the same op chain in the same order**,
// so the device can be held to a few ulp - and in several places to bit
// equality - rather than to a hand-waved tolerance. Every block below has a
// twin in `src/kernels/prefill/pf_gdn_conv.cl`, `pf_gdn_wy.cl` or
// `pf_gdn_scan.cl`, **and the files must be edited together**. This is the same
// rule, for the same reason, that `tests/kernels/gdn_ref.h` states against
// `gdn_step.cl`.
//
// ---------------------------------------------------------------------------
// The algorithm, written out once (plan 6b's standalone section, verbatim)
// ---------------------------------------------------------------------------
// Transcribed from the FLA reference vLLM runs (`chunk.py:23-82` is the driver;
// `cumsum.py:27-71`, `chunk_scaled_dot_kkt.py:46-112`, `solve_tril.py:38-100`,
// `wy_fast.py:30-115`, `chunk_delta_h.py:88-317`, `chunk_o.py:84-137` are the
// six stages). Intra-chunk size 64 (`utils.py:31` FLA_CHUNK_SIZE; Intel's CuTe
// kernel agrees, `gdn_attn_utils.h:8` chunk_size_xe2 = 64).
//
// For one v-head `h`, one 64-chunk covering positions p_0 … p_{L-1} (L <= 64),
// with the state S in R^{128x128} k-major (S[k][x] - decode's `gdn_state`
// layout, gdn_step.cl:22) **as it stands at the chunk's first position**:
//
//   g[i]      = negA[h] * softplus(f32(rne(a[p_i])) + dt_bias[h])       fp32, <= 0
//   beta[i]   = 1 / (1 + exp(-f32(rne(b[p_i]))))                        fp32
//   gc[i]     = SUM_{j<=i} g[j]                                         fp32, j ascending
//   gl        = gc[L-1]
//   q[i][k]   = f32(qf_b[p_i][k]) * Q_SCALE    (bf16 in xb; Q_SCALE = 1/sqrt(128))
//   k[i][k]   = f32(kf_b[p_i][k])              (bf16 in xb)
//   v[i][x]   = f32(xb_v[p_i][x])              (bf16 in xb, conv+SiLU output)
//
//   A[i][j]   = beta[i] * (SUM_k k[i][k]*k[j][k]) * exp(gc[i] - gc[j])  for i > j, else 0
//   T         = (I + A)^-1                                              unit lower triangular
//               ^^^ NOTE THE SIGN. FLA stores A positive
//               (chunk_scaled_dot_kkt.py) and NEGATES it on the way into the
//               solve (`solve_tril.py:82`: b_A = -tl.where(m_A, b_A, 0)).
//               Plan 6b's algebra section writes (I - A)^-1, which is a
//               transcription error in the plan: at L = 2 the recurrence needs
//               the coefficient on Delta_0 inside vn_1 to be
//               -beta_1 exp(g_1) (k_1 . k_0) = -A[1][0], and (I - A)^-1 gives
//               +A[1][0]. Derived by hand, then confirmed against the FLA line.
//   vb[j][x]  = rne( v[j][x] * beta[j] )                         Q1  bf16 (wy_fast:88)
//   kb[j][k]  = rne( k[j][k] * beta[j] * exp(gc[j]) )            Q2  bf16 (wy_fast:110)
//   u[i][x]   = rne( SUM_{j<=i} T[i][j] * f32(vb[j][x]) )        Q3  bf16 (wy_fast:89)
//   w[i][k]   = rne( SUM_{j<=i} T[i][j] * f32(kb[j][k]) )        Q4  bf16 (wy_fast:112)
//   A2[i][j]  = (SUM_k q[i][k]*k[j][k]) * exp(gc[i] - gc[j])            for j <= i, else 0
//
//   # the sequential step. S is the chunk-start state for BOTH lines below.
//   vn[i][x]  = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x]                    fp32
//   o[i][x]   = ( SUM_k q[i][k] * S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j] * vn[j][x]
//   vs[i][x]  = vn[i][x] * exp(gl - gc[i])
//   S[k][x]  <- S[k][x] * exp(gl) + SUM_i k[i][k] * vs[i][x]                   i ascending
//
// Three properties are load-bearing:
//   * **A2's mask includes the diagonal** (chunk_o.py:123, `>=`) and **A's does
//     not** (chunk_scaled_dot_kkt.py:108, `>`). Getting the pair backwards is an
//     off-by-one a 64-position test catches and a 1-position test does not.
//   * **`o` reads the chunk-start S, not the updated one** (chunk_o.py is a
//     separate kernel over chunk_delta_h's per-chunk snapshots). Computing vn
//     and o before touching S is what removes FLA's [C/64][48][128][128] fp32
//     `h` buffer (100,663,296 B at C = 2048, derived) entirely.
//   * **Q_SCALE is folded into q at read**, not applied as FLA's trailing
//     `* scale` (chunk_o.py:137). Algebraically the same for both terms of `o`,
//     and it is what makes our `q` decode's `qf_s` (gdn_step.cl:312).
//
// Two deliberate deviations from FLA, both plan 6b ruling R8: its `solve_tril`
// rounds T to bf16 (chunk.py:47-49) and its `chunk_delta_h` rounds `v_new`
// before the S update (chunk_delta_h.py:274). Both stay **fp32** here.
//
// **Q1-Q4 are the four bf16 roundings the chunked form ADDS and for which
// decode has no twin**; every other rounding is one of gdn_step.cl's P1-P8.
// They are kept because they are the reference vLLM runs, and they are the
// dominant term of the numerics band
// (docs/prefill-l1-preregistration-2026-09-05.md §2.2).
//
// ---------------------------------------------------------------------------
// Reduction orders, stated once and mirrored by the kernels
// ---------------------------------------------------------------------------
//   * `pf_gdn_A`'s k·k and `pf_gdn_A2`'s q·k use gdn_step.cl:52-65's band tree:
//     band `b` accumulates 8 terms in ASCENDING kk with explicit `fma`, then
//     the 16 band partials collapse with stride = 8, 4, 2, 1. Those two kernels
//     keep decode's tile, so the reference keeps decode's tree.
//   * **the scan's two 128-term contractions (`w · S` and `q · S`) are ONE
//     ascending-k `fma` chain per output** - `asc_dot` below, not `band_dot`.
//     Ruling A25 rewrote `pf_gdn_scan` to give each work-item an output tile of
//     8 with a private ascending-k accumulation (the band tree cost 640
//     barriers per 64-position sub-chunk and 0.64 TFLOP/s), and this file
//     mirrors the kernel it is a reference for. **Consequence worth stating
//     plainly: on these two sums the CPU reference's order is now closer to the
//     chunked kernel's than to decode's `gdn_step`** - which is the right way
//     round, because case 1 grades `gdn_chunk` against a same-order fp32
//     reference while case 2 grades it against decode on the device, and
//     keeping those two comparisons distinct is the whole point of having both.
//     The pre-registration for the re-measured band is
//     docs/prefill-gdn-scan-2026-09-05.md §1.3.
//   * every triangular sum (`u`, `w`, and o's A2 term) runs `j` ASCENDING with
//     explicit `fma` and one accumulator.
//   * the cumulative gate is an ascending fp32 running sum inside a 64-chunk,
//     restarting at every chunk boundary (`tl.cumsum`'s order).
//   * `1.0f / sqrt(x)`, never `rsqrt`; explicit `fma`; plain `exp`, never
//     `native_exp`. Every kernel is built with
//     `-cl-fp32-correctly-rounded-divide-sqrt`, so `/` and `sqrt` agree with
//     the host bit for bit; `exp` (3 ulp) and `log1p` (2 ulp) do not, and that
//     slack is why the bars against this file that are not bit-exact are the
//     ones with a transcendental in the chain.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace gdn_chunk_ref {

// Mirrors pf_gdn_conv.cl / pf_gdn_wy.cl / pf_gdn_scan.cl, which mirror
// model::Qwen35.
constexpr uint32_t kHeads = 48, kKHeads = 16, kDim = 128, kCT = 64;
constexpr uint32_t kConvRows = 10240, kConvTaps = 4, kQkvzN = 16384;
constexpr uint32_t kQOff = 0, kKOff = 2048, kVOff = 4096;
constexpr uint32_t kRing = 16, kAbStride = 128, kBOff = 48;
constexpr uint32_t kNegAOff = 40960, kDtBiasOff = 41008;   // gdn_small, in floats
constexpr float kQScale = 0.08838834764831845f;            // 1/sqrt(128)
constexpr uint32_t kBands = 16, kBandK = 8;                // the 128-term dot's tree

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
inline float softplus_f32(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

inline uint32_t nchunks(uint32_t C) { return (C + kCT - 1) / kCT; }

// gdn_step.cl:52-65's band tree over 128 terms. `a` and `b` are the two
// operands as functions of k; the caller supplies them already widened to fp32.
inline float band_dot(const float* a, const float* b_stride_base, size_t b_stride) {
  float p[kBands];
  for (uint32_t band = 0; band < kBands; ++band) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < kBandK; ++j) {
      const uint32_t k = band * kBandK + j;
      acc = std::fma(a[k], b_stride_base[size_t(k) * b_stride], acc);
    }
    p[band] = acc;
  }
  for (uint32_t stride = kBands / 2; stride > 0; stride >>= 1)
    for (uint32_t band = 0; band < stride; ++band) p[band] += p[band + stride];
  return p[0];
}

// pf_gdn_scan.cl's contraction after ruling A25: ONE accumulator, k ascending
// over all 128, explicit `fma`. Used by `scan` below and by nothing else.
inline float asc_dot(const float* a, const float* b_stride_base, size_t b_stride) {
  float acc = 0.0f;
  for (uint32_t k = 0; k < kDim; ++k)
    acc = std::fma(a[k], b_stride_base[size_t(k) * b_stride], acc);
  return acc;
}

// ---------------------------------------------------------------------------
// Stage 1 - pf_gdn_seed + pf_gdn_conv (P1, P2, P3)
// ---------------------------------------------------------------------------
//   qkvz  fp32 [C][16384]     the layer's qkv‖z GEMV output, S = 1
//   small fp32                conv[10240][4] at 0
//   ring  bf16 [16][10240]    updated in place: the chunk's LAST min(C,3)
//                             positions only, which is exactly what the next
//                             chunk's seed and decode's gdn_step read
//   xb    bf16 [C][10240]     conv + SiLU output
inline void conv(uint32_t pos, uint32_t C, const float* qkvz, const float* small,
                 uint16_t* ring, uint16_t* xb) {
  for (uint32_t ch = 0; ch < kConvRows; ++ch) {
    const float* w = small + size_t(ch) * kConvTaps;
    float win[kConvTaps];
    for (uint32_t j = 3; j >= 1; --j) {          // window slots 0,1,2 = pos-3,-2,-1
      const int64_t p = int64_t(pos) - int64_t(j);
      win[3 - j] = p < 0 ? 0.0f : f32(ring[size_t(uint64_t(p) % kRing) * kConvRows + ch]);
    }
    for (uint32_t m = 0; m < C; ++m) {
      const uint16_t raw_b = rne(qkvz[size_t(m) * kQkvzN + ch]);                     // P1
      win[3] = f32(raw_b);
      float acc = 0.0f;
      for (uint32_t t = 0; t < kConvTaps; ++t) acc = std::fma(w[t], win[t], acc);    // P2
      xb[size_t(m) * kConvRows + ch] = rne(silu_f32(acc));                           // P3
      win[0] = win[1];
      win[1] = win[2];
      win[2] = win[3];
    }
    for (uint32_t t = 0; t < 3; ++t) {           // the ring writeback
      const int64_t m = int64_t(C) - 3 + int64_t(t);
      if (m >= 0)
        ring[size_t((uint64_t(pos) + uint64_t(m)) % kRing) * kConvRows + ch] =
            rne(qkvz[size_t(m) * kQkvzN + ch]);
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2 - pf_gdn_l2norm, IN PLACE over xb (P4, P5, P6, P7)
// The stored q word is the ROUNDED, UNSCALED one; Q_SCALE is applied by the
// consumers in fp32, which is what makes it decode's `qf_s`.
// ---------------------------------------------------------------------------
inline void l2norm(uint32_t C, uint16_t* xb) {
  std::vector<float> red(kDim);
  for (uint32_t m = 0; m < C; ++m) {
    for (uint32_t wg = 0; wg < 2 * kKHeads; ++wg) {
      const uint32_t kh = wg < kKHeads ? wg : wg - kKHeads;
      const uint32_t base = (wg < kKHeads ? kQOff : kKOff) + kh * kDim;
      uint16_t* row = xb + size_t(m) * kConvRows + base;
      for (uint32_t i = 0; i < kDim; ++i) {
        const float t = f32(row[i]);
        red[i] = t * t;                          // one term per lane: plain multiply (P4)
      }
      for (uint32_t stride = kDim / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
      const float inv = 1.0f / std::sqrt(red[0] + 1e-6f);                            // P5
      for (uint32_t i = 0; i < kDim; ++i) row[i] = rne(f32(row[i]) * inv);      // P6 / P7
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 3 - pf_gdn_gate (P8 + the intra-chunk inclusive cumsum)
//   g_cum fp32 [C][48]   the CUMULATIVE gc, restarting at every 64-boundary
//   beta  fp32 [C][48]
// ---------------------------------------------------------------------------
inline void gate(uint32_t C, const float* ab, const float* small, float* g_cum,
                 float* beta) {
  for (uint32_t h = 0; h < kHeads; ++h) {
    const float negA = small[kNegAOff + h], dt_bias = small[kDtBiasOff + h];
    for (uint32_t t = 0; t < nchunks(C); ++t) {
      const uint32_t L = std::min(kCT, C - t * kCT);
      float acc = 0.0f;                          // restarts every 64-chunk
      for (uint32_t i = 0; i < L; ++i) {
        const uint32_t m = t * kCT + i;
        const uint16_t a_b = rne(ab[size_t(m) * kAbStride + h]);                     // P8
        const uint16_t b_b = rne(ab[size_t(m) * kAbStride + kBOff + h]);
        acc += negA * softplus_f32(f32(a_b) + dt_bias);
        g_cum[size_t(m) * kHeads + h] = acc;
        beta[size_t(m) * kHeads + h] = 1.0f / (1.0f + std::exp(-f32(b_b)));
      }
    }
  }
}

// --- the per-(head, chunk) staging every stage below shares -----------------
// q is read SCALED (Q_SCALE in fp32, after the bf16 round), k and v raw.
struct Tile {
  float q[kCT][kDim], k[kCT][kDim], v[kCT][kDim], gc[kCT], beta[kCT];
  uint32_t L;
};

inline void stage(uint32_t C, uint32_t h, uint32_t t, const uint16_t* xb, const float* g_cum,
                  const float* beta, Tile& s) {
  const uint32_t kh = h / 3;                     // repeat_interleave(., 3)
  s.L = std::min(kCT, C - t * kCT);
  for (uint32_t i = 0; i < s.L; ++i) {
    const uint32_t m = t * kCT + i;
    const uint16_t* row = xb + size_t(m) * kConvRows;
    for (uint32_t d = 0; d < kDim; ++d) {
      s.q[i][d] = f32(row[kQOff + kh * kDim + d]) * kQScale;
      s.k[i][d] = f32(row[kKOff + kh * kDim + d]);
      s.v[i][d] = f32(row[kVOff + h * kDim + d]);
    }
    if (g_cum) s.gc[i] = g_cum[size_t(m) * kHeads + h];
    if (beta) s.beta[i] = beta[size_t(m) * kHeads + h];
  }
}

inline size_t tri_base(uint32_t t, uint32_t h) {
  return ((size_t(t) * kHeads + h) * kCT) * kCT;
}

// ---------------------------------------------------------------------------
// Stage 4 - pf_gdn_A.  A[i][j] = beta[i] * (k_i . k_j) * exp(gc[i]-gc[j]), i > j
// ---------------------------------------------------------------------------
inline void mat_A(uint32_t C, const uint16_t* xb, const float* g_cum, const float* beta,
                  float* A) {
  Tile s;
  for (uint32_t t = 0; t < nchunks(C); ++t)
    for (uint32_t h = 0; h < kHeads; ++h) {
      stage(C, h, t, xb, g_cum, beta, s);
      float* At = A + tri_base(t, h);
      for (uint32_t i = 0; i < kCT; ++i)
        for (uint32_t j = 0; j < kCT; ++j) {
          float val = 0.0f;
          if (i < s.L && j < s.L && i > j)
            val = s.beta[i] * band_dot(s.k[i], &s.k[j][0], 1) *
                  std::exp(s.gc[i] - s.gc[j]);
          At[size_t(i) * kCT + j] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 5 - pf_gdn_solve.  T = (I + A)^-1, IN PLACE over A. `A` is NEGATED at
// staging, where FLA negates it (solve_tril.py:82) - see the sign note in the
// algorithm block above. The substitution is then the plain one, `i` ascending:
// T[i][j] = As[i][j] + SUM_{j<l<i} As[i][l]*T[l][j] with As = -A, `l` ascending
// with explicit fma. T[i][i] = 1, T[i][j] = 0 for j > i, and rows >= L stay
// zero - INCLUDING the diagonal, which the kernel must not write past L. That
// convention is asserted by the test.
// ---------------------------------------------------------------------------
inline void solve(uint32_t C, float* A) {
  float row[kCT];
  for (uint32_t t = 0; t < nchunks(C); ++t)
    for (uint32_t h = 0; h < kHeads; ++h) {
      float* T = A + tri_base(t, h);
      const uint32_t L = std::min(kCT, C - t * kCT);
      // -A on the strictly-lower half, +0.0f elsewhere (solve_tril.py:82).
      for (uint32_t i = 0; i < kCT; ++i)
        for (uint32_t j = 0; j < kCT; ++j)
          T[size_t(i) * kCT + j] = i > j ? -T[size_t(i) * kCT + j] : 0.0f;
      for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
          row[j] = 0.0f;
          if (j < i) {
            float acc = T[size_t(i) * kCT + j];
            for (uint32_t l = j + 1; l < i; ++l)
              acc = std::fma(T[size_t(i) * kCT + l], T[size_t(l) * kCT + j], acc);
            row[j] = acc;
          } else if (j == i) {
            row[j] = 1.0f;
          }
        }
        for (uint32_t j = 0; j <= i; ++j) T[size_t(i) * kCT + j] = row[j];
      }
    }
}

// ---------------------------------------------------------------------------
// Stage 6 - pf_gdn_wu.  Q1/Q2 then Q3/Q4; `j` ascending, one RNE each.
//   w, u  bf16 [C][48][128], indexed ((m * 48 + h) * 128 + x)
// ---------------------------------------------------------------------------
inline void wu(uint32_t C, const uint16_t* xb, const float* T, const float* g_cum,
               const float* beta, uint16_t* w, uint16_t* u) {
  Tile s;
  std::vector<float> vb(size_t(kCT) * kDim), kb(size_t(kCT) * kDim);
  for (uint32_t t = 0; t < nchunks(C); ++t)
    for (uint32_t h = 0; h < kHeads; ++h) {
      stage(C, h, t, xb, g_cum, beta, s);
      const float* Tt = T + tri_base(t, h);
      for (uint32_t j = 0; j < s.L; ++j) {
        const float eg = std::exp(s.gc[j]);
        for (uint32_t d = 0; d < kDim; ++d) {
          // Round ONCE, at the end of each expression, as wy_fast.py:88 and
          // :110 do with `.to(dtype)`.
          vb[size_t(j) * kDim + d] = f32(rne(s.v[j][d] * s.beta[j]));                // Q1
          kb[size_t(j) * kDim + d] = f32(rne(s.k[j][d] * s.beta[j] * eg));           // Q2
        }
      }
      for (uint32_t i = 0; i < s.L; ++i) {
        const uint32_t m = t * kCT + i;
        for (uint32_t d = 0; d < kDim; ++d) {
          float au = 0.0f, aw = 0.0f;
          for (uint32_t j = 0; j <= i; ++j) {                       // j ascending
            au = std::fma(Tt[size_t(i) * kCT + j], vb[size_t(j) * kDim + d], au);
            aw = std::fma(Tt[size_t(i) * kCT + j], kb[size_t(j) * kDim + d], aw);
          }
          u[(size_t(m) * kHeads + h) * kDim + d] = rne(au);                          // Q3
          w[(size_t(m) * kHeads + h) * kDim + d] = rne(aw);                          // Q4
        }
      }
    }
}

// ---------------------------------------------------------------------------
// Stage 7 - pf_gdn_A2.  A2[i][j] = (q_i . k_j) * exp(gc[i]-gc[j]), j <= i.
// NOTE THE DIAGONAL: `<=`, where mat_A uses `>`.
// ---------------------------------------------------------------------------
inline void mat_A2(uint32_t C, const uint16_t* xb, const float* g_cum, float* A2) {
  Tile s;
  for (uint32_t t = 0; t < nchunks(C); ++t)
    for (uint32_t h = 0; h < kHeads; ++h) {
      stage(C, h, t, xb, g_cum, nullptr, s);
      float* At = A2 + tri_base(t, h);
      for (uint32_t i = 0; i < kCT; ++i)
        for (uint32_t j = 0; j < kCT; ++j) {
          float val = 0.0f;
          if (i < s.L && j < s.L && j <= i)
            val = band_dot(s.q[i], &s.k[j][0], 1) * std::exp(s.gc[i] - s.gc[j]);
          At[size_t(i) * kCT + j] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 8 - pf_gdn_scan. The sequential chunk-to-chunk state scan.
//   state fp32 [48][128][128] k-major, updated in place
//   o     fp32 [C][48][128]   written (P10: fp32; pf_gated_head rounds it)
// `vn` and `o` are computed from the CHUNK-START state; only then is S touched.
// TWO reassociations relative to decode live here, both named and both
// mirrored above:
//   * decode interleaves the decay with the rank-1 update per position
//     (gdn_step.cl:318-349); the chunk applies exp(gl) once and adds
//     SUM_i k_i (x) vs_i.
//   * ruling A25: `w · S` and `q · S` are ONE ascending-k fma chain
//     (`asc_dot`), not decode's 16-band tree - the kernel gives each work-item
//     an output tile of 8 with a private accumulator, which is what took the
//     barrier count per 64-position sub-chunk from 640 to 3.
// ---------------------------------------------------------------------------
inline void scan(uint32_t C, const uint16_t* xb, const uint16_t* w, const uint16_t* u,
                 const float* A2, const float* g_cum, float* state, float* o) {
  Tile s;
  std::vector<float> vn(size_t(kCT) * kDim), wf(kDim), expg(kCT);
  for (uint32_t h = 0; h < kHeads; ++h) {
    float* S = state + size_t(h) * kDim * kDim;      // S[k * 128 + x]
    for (uint32_t t = 0; t < nchunks(C); ++t) {
      stage(C, h, t, xb, g_cum, nullptr, s);
      const float* At = A2 + tri_base(t, h);
      for (uint32_t i = 0; i < s.L; ++i) expg[i] = std::exp(s.gc[i]);
      const float gl = s.gc[s.L - 1];

      // vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x]
      for (uint32_t i = 0; i < s.L; ++i) {
        const uint32_t m = t * kCT + i;
        for (uint32_t d = 0; d < kDim; ++d)
          wf[d] = f32(w[(size_t(m) * kHeads + h) * kDim + d]);
        for (uint32_t x = 0; x < kDim; ++x)
          vn[size_t(i) * kDim + x] =
              f32(u[(size_t(m) * kHeads + h) * kDim + x]) - asc_dot(wf.data(), S + x, kDim);
      }
      // o[i][x] = (SUM_k q[i][k]*S[k][x]) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
      for (uint32_t i = 0; i < s.L; ++i) {
        const uint32_t m = t * kCT + i;
        for (uint32_t x = 0; x < kDim; ++x) {
          float acc = asc_dot(s.q[i], S + x, kDim) * expg[i];
          for (uint32_t j = 0; j <= i; ++j)                          // j ascending
            acc = std::fma(At[size_t(i) * kCT + j], vn[size_t(j) * kDim + x], acc);
          o[(size_t(m) * kHeads + h) * kDim + x] = acc;              // P10: fp32
        }
      }
      // S <- S * exp(gl) + SUM_i k_i (x) (vn_i * exp(gl - gc[i])), i ascending
      const float dl = std::exp(gl);
      for (uint32_t k = 0; k < kDim; ++k)
        for (uint32_t x = 0; x < kDim; ++x) S[size_t(k) * kDim + x] *= dl;
      for (uint32_t i = 0; i < s.L; ++i) {
        const float sc = std::exp(gl - s.gc[i]);
        for (uint32_t k = 0; k < kDim; ++k) {
          const float kfv = s.k[i][k];
          for (uint32_t x = 0; x < kDim; ++x)
            S[size_t(k) * kDim + x] =
                std::fma(kfv, vn[size_t(i) * kDim + x] * sc, S[size_t(k) * kDim + x]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The whole chunk, stages 1-8 - what `runtime::prefill::gdn_chunk` runs, minus
// the tenth launch (`pf_gated_head`, which has its own reference in
// tests/kernels/prep_ref.h).
// ---------------------------------------------------------------------------
inline void chunk(uint32_t pos, uint32_t C, const float* qkvz, const float* ab,
                  const float* small, uint16_t* ring, float* state, float* o) {
  std::vector<uint16_t> xb(size_t(C) * kConvRows);
  std::vector<float> g_cum(size_t(C) * kHeads), beta(size_t(C) * kHeads);
  std::vector<float> A(size_t(nchunks(C)) * kHeads * kCT * kCT);
  std::vector<float> A2(A.size());
  std::vector<uint16_t> w(size_t(C) * kHeads * kDim), u(w.size());

  conv(pos, C, qkvz, small, ring, xb.data());
  l2norm(C, xb.data());
  gate(C, ab, small, g_cum.data(), beta.data());
  mat_A(C, xb.data(), g_cum.data(), beta.data(), A.data());
  solve(C, A.data());
  wu(C, xb.data(), A.data(), g_cum.data(), beta.data(), w.data(), u.data());
  mat_A2(C, xb.data(), g_cum.data(), A2.data());
  scan(C, xb.data(), w.data(), u.data(), A2.data(), g_cum.data(), state, o);
}

}  // namespace gdn_chunk_ref
