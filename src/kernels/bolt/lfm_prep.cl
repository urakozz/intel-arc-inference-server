// lfm_prep.cl - spec 24c: LFM2.5-24B-A2B's one-launch residual norm, the one family of its between-GEMV kernels
// still built (tools/probe/probe_bolt_step's norm stand-in; the other three and the fused router arm were never
// compiled after LFM2.5 was dropped and went 2026-11-15):
//
//   lfm_res_norm(h_in, slices, w, h_out, x)            NORM_S    grid (NORM_WGS, M),     WG 256
//
// The reference is 24a's tools/oracle/lfm2moe_ref.py (spec 24 §2); tests/kernels/lfm2moe_ref.h repeats every
// chain below - EDIT THE TWO TOGETHER. The rounding discipline is prep.cl's: a GEMV's split-K partials summed in
// fp32 from 0.0f and rounded to bf16 once, every elementwise op widened to fp32 and rounded.
//
// **The norm** (spec 24 §2.2, §4.4; plan 24c Review Focus 3): ONE launch a norm site. Every work-group folds the
// whole row - r[k] = rne(f32(h_in[k]) + f32(rne(Σ_s slices[s][m][k]))) (NORM_S > 0) or h_in[k] (NORM_S == 0, the
// `_NF` sites after lfm_moe_down folded) - reduces Σ r² by ONE fixed tree (256 lanes, lane i an fma chain over k = i,
// i + 256, ... ascending, then 128 .. 1 pairwise: the same bits in every work-group), and writes only its own chunk
// of h_out (NORM_S > 0) and of x = rne(f32(rne(r · rstd)) · w), rstd = 1 / sqrt(Σ / K + 1e-5) (never rsqrt).
// The residual PING-PONGS: h_in and h_out are different buffers (the capture alternates them per fold site), so
// no work-group reads a chunk another one is rewriting.
//
// **No sub-group function anywhere in this file.**
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (prep.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
// One element of the two-rounding RMSNorm: rne(f32(rne(r · rstd)) · w).
inline ushort lfm_norm_elem(float r, float rstd, float w) { return rne_bf16(rf(r * rstd) * w); }

// ---------------------------------------------------------------------------------------------------------------
#ifdef NORM_S
#if !defined(K) || !defined(NORM_WGS) || !defined(NORM_EPS)
#error "lfm_res_norm: K, NORM_WGS and NORM_EPS must be defined"
#endif
#define WG_NORM 256
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
#if K % WG_NORM != 0
#error "lfm_res_norm: K is whole 256-lane rows"
#endif

// Row m's folded value at k (the slices summed from 0.0f, ascending s).
inline float lfm_fold(__global const ushort* restrict h_in, __global const float* restrict slices, uint m, uint k) {
#if NORM_S > 0
  float v = 0.0f;
  for (uint s = 0; s < NORM_S; ++s) v += slices[((size_t)s * M + m) * K + k];
  return bf16f(rne_bf16(bf16f(h_in[(size_t)m * K + k]) + bf16f(rne_bf16(v))));
#else
  (void)slices;
  return bf16f(h_in[(size_t)m * K + k]);
#endif
}

__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void lfm_res_norm(__global const ushort* restrict h_in, __global const float* restrict slices,
                           __global const float* restrict norm_w, __global ushort* restrict h_out,
                           __global ushort* restrict x_out
                           ) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float rs[K];          // the folded row (bf16 values), then (router WGs) the normalised row
  __local float red[WG_NORM];
  float acc = 0.0f;
  for (uint k = lid; k < K; k += WG_NORM) {
    const float v = lfm_fold(h_in, slices, m, k);
    rs[k] = v;
    acc = fma(v, v, acc);
  }
  red[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_NORM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)K + NORM_EPS);
  const uint k0 = wg * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_NORM) {
#if NORM_S > 0
    h_out[(size_t)m * K + k] = rne_bf16(rs[k]);
#else
    (void)h_out;
#endif
    x_out[(size_t)m * K + k] = lfm_norm_elem(rs[k], rstd, norm_w[k]);
  }
}
#endif  // NORM_S
