// prep.cl - the three kernels that sit between the GEMVs of a decode step:
// residual add + RMSNorm, SiLU·mul over gate‖up, and the GDN gated head norm.
// Three entry points in one file; variants are `-D`.
//
// **Rounding discipline** (plan 3 Task 2; docs/12-kernels.md). The oracle is
// torch, and torch rounds per op: a linear's output is bf16, and every
// elementwise op widens to fp32 internally and rounds its result back to bf16.
// These kernels do the same, so the engine's residual stream stays comparable
// to the golden tensors op for op:
//   * a GEMV's split-K partials are summed in fp32 and rounded to bf16 ONCE -
//     that is the linear's bf16 output (`rne_bf16(Σ_s partials)`);
//   * the residual add is bf16-in / bf16-out;
//   * a norm widens to fp32, multiplies by the fp32 `(1 + w)` weight and rounds
//     back to bf16 (the reference's `type_as(x)`);
//   * inside-op accumulation (the variance sum) stays fp32 and is *not* matched
//     to torch term for term - that drift is what the golden gate absorbs.
//
// Two spellings are load-bearing, and `tests/kernels/prep_ref.h` repeats them
// verbatim so the CPU reference is **bit-exact**, not merely close:
//
//   1. **`1.0f / sqrt(x)`, never `rsqrt(x)`.** `rsqrt` is a ~2 ulp
//      approximation with no cross-implementation guarantee; correctly rounded
//      `sqrt` plus a correctly rounded divide is reproducible on the host
//      (controller ruling 2026-08-25).
//   2. **The square-accumulate is an explicit `fma`.** `sum += v*v` may or may
//      not be contracted into a fused multiply-add by either compiler; writing
//      the fusion explicitly on both sides removes the question.
//
// **The variance tree order** (identical text in prep_ref.h): the work-group's
// WG lanes each accumulate their own strided slice - lane i takes k = i,
// i+WG, i+2·WG, … in ascending k - into one fp32 register with `fma`, write it
// to SLM, and then a fixed pairwise tree collapses SLM: for stride = WG/2,
// WG/4, …, 1, lane i < stride does `red[i] += red[i + stride]`, with a barrier
// after every step. So for WG = 256 the shape is 256 → 128 → 64 → … → 1.
// Nothing here is data-dependent, so two replays of the captured list produce
// the same bits, and so does the reference.

#ifndef M
#define M 1
#endif
#ifndef K
#define K 5120            /* prep_res_norm's row length (kHidden) */
#endif
#ifndef S_PREV
#define S_PREV 0          /* prep_res_norm's split-K slice count; 0 = no mixer */
#endif

#define WG_RES 256
#define WG_SILU 256
#define WG_GATED 128

// prep_silu_mul's shapes are the model's and never vary (M is its whole variant
// space): gate‖up is 2 × 17408 columns interleaved in 16-column blocks, split-K
// S = 4 (docs/13-loader.md, model::Qwen35's table).
#define SILU_S 4
#define SILU_N 17408
#define SILU_FUSED_N 34816
#define SILU_CHUNK 4096   /* 5 work-groups cover 17408; the last does 1024 */

// prep_gated_head's shapes, likewise fixed: qkv‖z is 16384 wide with z at
// column 10240, split-K S = 1; 48 v-heads of 128 (model::Qwen35).
#define GATED_S 1
#define GATED_HEADS 48
#define HEAD_DIM 128
#define QKVZ_N 16384
#define Z_OFF 10240
#define GATED_OUT_N (GATED_HEADS * HEAD_DIM)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 → bf16, round-to-nearest-even. Same add-and-shift as common::f32_to_bf16;
// NaN is not expected and not handled.
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// silu(x) = x / (1 + exp(-x)), plain `exp` (not `native_exp`). This is the one
// op whose result may differ from the host's by a rounding: OpenCL allows 3 ulp
// on fp32 `exp`. Both callers keep it as the LAST factor so that everything
// before it stays bit-comparable.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------
// prep_res_norm - residual add + RMSNorm, one work-group of 256 per token.
//
//   mixer_b = rne_bf16(Σ_s partials[s][m][k])            (skipped if S_PREV==0)
//   r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))  (S_PREV==0: r_b = resid)
//   resid[m][k] = r_b                                    (the residual stream)
//   rstd    = 1 / sqrt(mean_k(f32(r_b)²) + 1e-6)         (tree order above)
//   x_out[m][k] = rne_bf16(f32(r_b) · rstd · norm_w[k])
//
// The whole row is staged in SLM as fp32 (K·4 = 20 KB at K = 5120) so phase 3
// re-reads it from SLM instead of re-widening `resid` from DRAM, and the
// variance never leaves the work-group. Grid = (1, M), so `m` is group id 1.
__attribute__((reqd_work_group_size(WG_RES, 1, 1)))
__kernel void prep_res_norm(__global const float* restrict partials,
                            __global ushort* restrict resid,
                            __global const float* restrict norm_w,
                            __global ushort* restrict x_out) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float row[K];
  __local float red[WG_RES];
  __global ushort* rp = resid + (size_t)m * K;

  for (uint k = lid; k < K; k += WG_RES) {
#if S_PREV == 0
    const ushort r_b = rp[k];
#else
    float acc = 0.f;
    for (uint s = 0; s < S_PREV; ++s) acc += partials[((size_t)s * M + m) * K + k];
    const ushort r_b = rne_bf16(bf16f(rp[k]) + bf16f(rne_bf16(acc)));
#endif
    rp[k] = r_b;
    row[k] = bf16f(r_b);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float acc2 = 0.f;
  for (uint k = lid; k < K; k += WG_RES) acc2 = fma(row[k], row[k], acc2);
  red[lid] = acc2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_RES / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = red[0] / (float)K;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);

  for (uint k = lid; k < K; k += WG_RES)
    x_out[(size_t)m * K + k] = rne_bf16(row[k] * rstd * norm_w[k]);
}

// ---------------------------------------------------------------------------
// prep_silu_mul - the MLP's activation, over gate‖up interleaved in 16-column
// blocks (the loader's `cols_interleave16`, docs/13-loader.md):
//
//   gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16
//   g_b = rne_bf16(Σ_s partials[s][m][gflat]) ;  u_b likewise
//   s_b = rne_bf16(silu_f32(f32(g_b)))
//   x_out[m][k] = rne_bf16(f32(s_b) · f32(u_b))
//
// No reduction, no SLM, no barrier: one output per iteration, `SILU_CHUNK`
// outputs per work-group. Grid = (5, M) at N = 17408 - the last chunk covers
// 1024 and the `k < k1` bound is what makes that safe.
__attribute__((reqd_work_group_size(WG_SILU, 1, 1)))
__kernel void prep_silu_mul(__global const float* restrict partials,
                            __global ushort* restrict x_out) {
  const uint m = get_group_id(1);
  const uint k0 = get_group_id(0) * SILU_CHUNK;
  uint k1 = k0 + SILU_CHUNK;
  if (k1 > SILU_N) k1 = SILU_N;

  for (uint k = k0 + get_local_id(0); k < k1; k += WG_SILU) {
    const size_t gflat = (size_t)(k / 16) * 32 + (k % 16);
    const size_t uflat = gflat + 16;
    float ga = 0.f, ua = 0.f;
    for (uint s = 0; s < SILU_S; ++s) {
      const size_t base = ((size_t)s * M + m) * SILU_FUSED_N;
      ga += partials[base + gflat];
      ua += partials[base + uflat];
    }
    const ushort g_b = rne_bf16(ga), u_b = rne_bf16(ua);
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
    x_out[(size_t)m * SILU_N + k] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}

// ---------------------------------------------------------------------------
// prep_gated_head - `Qwen3_5RMSNormGated` over one GDN v-head (doc 03: norm →
// cast → ×w → ×silu(z.float()) → cast). One work-group of 128 per (head, token),
// so lane i owns channel i and the variance tree is exactly the 128 lanes:
//
//   o_b = rne_bf16(gdn_o[m][h][i])                    (recurrence output → bf16)
//   z_b = rne_bf16(Σ_s qkvz_partials[s][m][Z_OFF + h·128 + i])
//   var = mean_i(f32(o_b)²)                           (128 → 64 → … → 1)
//   n_b = rne_bf16(f32(o_b) · (1 / sqrt(var + 1e-6)))
//   t_b = rne_bf16(f32(gated_w[i]) · f32(n_b))        (plain w, NO +1 - the one
//                                                      norm in the model without
//                                                      it, and the one whose
//                                                      weight stays bf16)
//   x_out[m][h·128+i] = rne_bf16(f32(t_b) · silu_f32(f32(z_b)))
//
// The silu factor is deliberately the LAST op: everything through `t_b` is
// bit-reproducible on the host, only the final product carries `exp`'s slack.
// Grid = (48, M). One term per lane means no `fma` here - a plain multiply.
__attribute__((reqd_work_group_size(WG_GATED, 1, 1)))
__kernel void prep_gated_head(__global const float* restrict qkvz_partials,
                              __global const float* restrict gdn_o,
                              __global const ushort* restrict gated_w,
                              __global ushort* restrict x_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[HEAD_DIM];

  const ushort o_b = rne_bf16(gdn_o[((size_t)m * GATED_HEADS + h) * HEAD_DIM + i]);
  float za = 0.f;
  for (uint s = 0; s < GATED_S; ++s)
    za += qkvz_partials[((size_t)s * M + m) * QKVZ_N + Z_OFF + h * HEAD_DIM + i];
  const ushort z_b = rne_bf16(za);

  const float o_f = bf16f(o_b);
  red[i] = o_f * o_f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_GATED / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float var = red[0] / (float)HEAD_DIM;
  const float rstd = 1.0f / sqrt(var + 1e-6f);

  const ushort n_b = rne_bf16(o_f * rstd);
  const ushort t_b = rne_bf16(bf16f(gated_w[i]) * bf16f(n_b));
  x_out[(size_t)m * GATED_OUT_N + h * HEAD_DIM + i] =
      rne_bf16(bf16f(t_b) * silu_f32(bf16f(z_b)));
}
