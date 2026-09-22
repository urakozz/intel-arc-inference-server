// pf_prep.cl -- the prefill path's residual-add + RMSNorm pair and its
// SiLU-mul, at **runtime `M`** and **S = 1**. Three entry points in one file,
// prep.cl's pattern; variants are `-D` and `M` is NOT one of them.
//
// This file is `src/kernels/prep.cl:209-335` transcribed with exactly three
// classes of edit, and nothing else:
//
//   1. `#ifndef M / #define M 1` is gone. Every entry point takes `uint
//      m_count` as its LAST argument, and every index expression that read `M`
//      reads `m_count`: `partials[((size_t)s * m_count + m) * K + k]`,
//      `sumsq[(size_t)g * m_count + m]`. The compiled binary is therefore
//      independent of the chunk width, which is what interfaces.md's
//      "no per-M compiled variants on the prefill path" asks for.
//   2. `SILU_S` is **1**, not decode's 8 (plan 6b ruling R1). The prefill
//      linears run without split-K, so `prep_silu_mul`'s `for (s < SILU_S)`
//      loop -- which is what made the widened decode kernel move eight times
//      the traffic it needed to -- folds one slice.
//   3. `m = get_group_id(1)` stays and no mask is added: the grid's y extent
//      IS `M` (the tile rule is `ceil(M/1)` with tile 1), so there is no tail
//      tile here to mask. `m_count` is passed only so the strides above can be
//      computed, and `pf_res_fold` is the one entry that truly needs it -- the
//      `sumsq` rectangle is `[FOLD_G][m_count]`.
//
// **Everything else is the numerics contract and is untouched**: both bf16 RNE
// steps, the `fma` square-accumulate, `1.0f / sqrt(x)` (never `rsqrt`), the
// `red[WG]` pairwise tree with a barrier after every step, and stage B's
// ascending-`g` fold. `tests/prefill/pf_prep_test.cc` holds the M = 1 output
// **bit-identical** to the decode binaries and the M = 64 output bit-identical
// to `tests/kernels/prep_ref.h` driven at runtime M -- which is only available
// because the per-element chain and both tree orders are the same text.
// prep.cl's own header carries the full argument for each of them; it is not
// repeated here, it is *inherited*, and the two files must be edited together.

#ifndef K
#define K 5120            /* the residual row length (kHidden) */
#endif
#ifndef S_PREV
#define S_PREV 0          /* 0 = no mixer to fold; 1 = the prefill GEMM's one slice */
#endif
#if S_PREV > 1
#error "pf_prep: the prefill path is S = 1 by plan 6b ruling R1 -- a wider S_PREV means a producer that still emits decode's split-K rectangle, which at M = C would be a multi-TB buffer (R1's arithmetic). Fix the producer, not this guard."
#endif

// **Reading `partials` as bf16 was measured and rejected** (parity program
// S2(b); `.superpowers/sdd/s2-epilogue-fusion-report.md`). At S_PREV = 1 this
// kernel's first act on `partials` IS `rne_bf16`, so a producer that stored the
// rounded word would be bitwise free here -- and it was, and it saved 1.7 ms on
// the `norm` row while costing the producing GEMM 9.3. Left as a note rather
// than a variant so that nobody re-derives it; the reason is on the producer's
// side and is recorded in pf_gemm.cl.

#ifndef FOLD_G
#define FOLD_G 20
#endif
#ifndef NORM_G
#define NORM_G 20
#endif
#ifndef NORM_WGS
#define NORM_WGS 20
#endif

#define WG_FOLD 256
#define WG_NORM 256
#define FOLD_CHUNK ((K + FOLD_G - 1) / FOLD_G)
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
#define WG_SILU 256

// gate||up's shapes are the model's and never vary. SILU_S is 1 by ruling R1
// (decode's copy of GateUp.S is 8); the guard below is what stops a
// copy-and-paste from re-introducing decode's split-K on this path.
#define SILU_S 1
#if SILU_S != 1
#error "pf_prep: the prefill path folds ONE slice (plan 6b ruling R1)"
#endif
#define SILU_N 17408
#define SILU_FUSED_N 34816
#define SILU_CHUNK 4096   /* 5 work-groups cover 17408; the last does 1024 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even. Same add-and-shift as common::f32_to_bf16
// and prep.cl's; NaN is not expected and not handled.
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// silu(x) = x / (1 + exp(-x)), plain `exp` (not `native_exp`), kept as the LAST
// factor so everything before it stays bit-comparable with the host.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------
// pf_res_fold -- stage A: fold the previous linear's ONE fp32 slice into the
// residual stream and reduce this chunk's Sigma x^2. Grid (FOLD_G, M), WG 256.
//
//   mixer_b = rne_bf16(partials[m][k])                   (skipped if S_PREV==0)
//   r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))
//   resid[m][k] = r_b
//   sumsq[g][m] = tree_{lanes}(Sigma_{k in chunk g} f32(r_b)^2)
__attribute__((reqd_work_group_size(WG_FOLD, 1, 1)))
__kernel void pf_res_fold(__global const float* restrict partials,
                          __global ushort* restrict resid,
                          __global float* restrict sumsq, uint m_count) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_FOLD];
  __global ushort* rp = resid + (size_t)m * K;

  const uint k0 = g * FOLD_CHUNK;
  uint k1 = k0 + FOLD_CHUNK;
  if (k1 > K) k1 = K;

  float acc2 = 0.f;
  for (uint k = k0 + lid; k < k1; k += WG_FOLD) {
#if S_PREV == 0
    const ushort r_b = rp[k];
#else
    float acc = 0.f;
    for (uint s = 0; s < S_PREV; ++s) acc += partials[((size_t)s * m_count + m) * K + k];
    const ushort r_b = rne_bf16(bf16f(rp[k]) + bf16f(rne_bf16(acc)));
#endif
    rp[k] = r_b;
    const float v = bf16f(r_b);
    acc2 = fma(v, v, acc2);
  }
  // A lane with no element in this chunk contributes an exact 0.0f, so the tree
  // below is the same shape for every g whatever FOLD_CHUNK is.
  red[lid] = acc2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_FOLD / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sumsq[(size_t)g * m_count + m] = red[0];
}

// ---------------------------------------------------------------------------
// pf_norm_finish -- stage B: the global Sigma, the rms, and the normalised row.
// Grid (NORM_WGS, M), WG 256.
//
//   rstd        = 1 / sqrt(Sigma_{g<NORM_G} sumsq[g][m] / K + 1e-6)
//   x_out[m][k] = rne_bf16(f32(resid[m][k]) . rstd . norm_w[k])
//
// Every lane of every work-group folds the NORM_G partials itself in the same
// ascending-`g` order, so the grid carries no cross-work-group dependency.
// **Nothing in this kernel reads `partials`, so `S` cannot change its cost**:
// its traffic at M = C is exactly decode's widened, and the S = 1 correction
// that shrinks `pf_res_fold` and `pf_silu_mul` does not touch it.
__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void pf_norm_finish(__global const float* restrict sumsq,
                             __global const ushort* restrict resid,
                             __global const float* restrict norm_w,
                             __global ushort* restrict x_out, uint m_count) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);

  float total = 0.f;
  for (uint g = 0; g < NORM_G; ++g) total += sumsq[(size_t)g * m_count + m];
  const float mean = total / (float)K;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);

  const __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = w * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_NORM)
    x_out[(size_t)m * K + k] = rne_bf16(bf16f(rp[k]) * rstd * norm_w[k]);
}

// ---------------------------------------------------------------------------
// pf_silu_mul -- the MLP's activation over gate||up, interleaved in 16-column
// blocks (the loader's `cols_interleave16`, docs/13-loader.md). Grid (5, M),
// WG 256; the last chunk covers 1024 and the `k < k1` bound makes that safe.
//
//   gflat = (k/16).32 + k%16 ;  uflat = gflat + 16
//   g_b = rne_bf16(partials[m][gflat]) ;  u_b likewise      (ONE slice: R1)
//   s_b = rne_bf16(silu_f32(f32(g_b)))
//   x_out[m][k] = rne_bf16(f32(s_b) . f32(u_b))
__attribute__((reqd_work_group_size(WG_SILU, 1, 1)))
__kernel void pf_silu_mul(__global const float* restrict partials,
                          __global ushort* restrict x_out, uint m_count) {
  const uint m = get_group_id(1);
  const uint k0 = get_group_id(0) * SILU_CHUNK;
  uint k1 = k0 + SILU_CHUNK;
  if (k1 > SILU_N) k1 = SILU_N;

  for (uint k = k0 + get_local_id(0); k < k1; k += WG_SILU) {
    const size_t gflat = (size_t)(k / 16) * 32 + (k % 16);
    const size_t uflat = gflat + 16;
    float ga = 0.f, ua = 0.f;
    for (uint s = 0; s < SILU_S; ++s) {
      const size_t base = ((size_t)s * m_count + m) * SILU_FUSED_N;
      ga += partials[base + gflat];
      ua += partials[base + uflat];
    }
    const ushort g_b = rne_bf16(ga), u_b = rne_bf16(ua);
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
    x_out[(size_t)m * SILU_N + k] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}
