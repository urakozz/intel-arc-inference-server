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

// The two-stage split of prep_res_norm (spec 1.5 lever L1). `FOLD_G` is stage
// A's work-group count and therefore the number of fp32 partial sums-of-squares
// it writes; `NORM_G` is how many of them stage B folds (it must equal the
// FOLD_G of the stage-A launch that fed it) and `NORM_WGS` is stage B's own
// work-group count. All three are in the compiled variant's NAME
// (kernels::prep_res_fold_variant / prep_norm_finish_variant), so a host that
// asks for a grid the binary was not built for names a file that does not
// exist and throws at capture rather than reducing the wrong number of slices.
#ifndef FOLD_G
#define FOLD_G 20
#endif
#ifndef NORM_G
#define NORM_G 20
#endif
#ifndef NORM_WGS
#define NORM_WGS 20
#endif

#define WG_RES 256
#define WG_FOLD 256
#define WG_NORM 256
// Ceiling division: a K that is not a multiple of the group count leaves the
// last chunk short, and the `k < k1` bound is what makes that safe (the same
// device prep_silu_mul's ragged 1024-wide last chunk uses).
#define FOLD_CHUNK ((K + FOLD_G - 1) / FOLD_G)
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
#define WG_SILU 256
#define WG_GATED 128

// prep_silu_mul's shapes are the model's and never vary (M is its whole variant
// space): gate‖up is 2 × 17408 columns interleaved in 16-column blocks, split-K
// S = 4 (docs/13-loader.md, model::Qwen35's table).
// SILU_S is a *copy* of GateUp.S and this file cannot see the table it copies.
// runtime::Capture::check_sizes (src/runtime/capture.cc) asserts the pair at
// capture time, so a retune in model/qwen35.cc throws there rather than making
// this loop sum the wrong number of slices in silence.
#define SILU_S 4
#define SILU_N 17408
#define SILU_FUSED_N 34816
#define SILU_CHUNK 4096   /* 5 work-groups cover 17408; the last does 1024 */

// prep_gated_head's shapes, likewise fixed: qkv‖z is 16384 wide with z at
// column 10240, split-K S = 1; 48 v-heads of 128 (model::Qwen35).
// GATED_S is a *copy* of QkvZ.S; runtime::Capture::check_sizes
// (src/runtime/capture.cc) asserts the pair at capture time - see SILU_S above.
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
// prep_res_fold / prep_norm_finish - prep_res_norm in two launches (spec 1.5
// lever L1). Same maths, one Σ apart, and 320 subgroups instead of 16.
//
// **Why two kernels.** RMSNorm's mean is over the whole row, so a single kernel
// cannot split the row across work-groups: the work-group IS the reduction
// domain, and at M = 1 that is ONE work-group - 16 subgroups on one Xe-core,
// measured at 22.3 µs and 17.0 GB/s in situ (docs/12 `prep_res_norm` →
// Measured). What the measurement says is scarce is the **subgroup**
// (docs/15 §L2: 8 → 32 → 128 subgroups took the `a‖b` GEMV 48.8 → 13.1 →
// 5.3 µs, while spreading the same 8 over 4× the work-groups bought zero). A
// grid-wide reduction needs a second launch, and a second launch is what this
// pair is.
//
//   stage A  prep_res_fold(partials, resid, sumsq)      grid (FOLD_G, M)
//   stage B  prep_norm_finish(sumsq, resid, norm_w, x)  grid (NORM_WGS, M)
//
// **The numerics contract, which is the whole point of this shape.** Stage A
// walks its chunk with the SAME per-element chain the single-WG kernel uses -
// the same s = 0…S_PREV-1 partial order, the same two bf16 RNE steps - and the
// per-element chain is independent of who runs it. So **`resid` out of one
// launch, given the same inputs, is bit-identical** to prep_res_norm's, which
// `prep_test` asserts. The ONE thing that changes is the global Σx² tree:
// instead of 256 lane accumulators over the whole row collapsed by one pairwise
// tree, it is FOLD_G chunk trees (each 256 → 128 → … → 1 over lanes holding
// their own chunk's terms) summed by stage B in ascending `g`.
//
// **That does NOT mean the engine's tensors are unchanged, and the first draft
// of this comment claimed it did.** The reordered Σ moves `x` in the last bf16
// ulp somewhere in a real row; the next GEMV consumes that `x`; 60-odd layers
// amplify it. Measured, before and after this lever on the same box: the golden
// gate's per-layer tap cosines **do move, in both directions** - `code`'s worst
// tap 0.829 → 0.914, `prose`'s 0.99945 → 0.99481 - while the token ids stayed
// 96/96 element-exact. The arbiter is the gate, not a tolerance and not a
// bit-identity argument, and the standing rule is that every lever which
// reorders a sum runs it **before as well as after** (docs/14, "The diagnostics
// move with every lever"; docs/12 `prep_res_fold` → the numerics contract).
//
// Nothing here is data-dependent, there is no atomic, FOLD_G is fixed at
// compile time and the `g` loop is in index order, so two replays of the
// captured list still produce the same bits.

// Stage A: fold the previous GEMV's split-K partials into the residual stream
// and reduce this chunk's Σx².
//
//   mixer_b = rne_bf16(Σ_s partials[s][m][k])            (skipped if S_PREV==0)
//   r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))
//   resid[m][k] = r_b
//   sumsq[g][m] = tree_{lanes}(Σ_{k in chunk g} f32(r_b)²)
//
// No SLM row staging: at FOLD_CHUNK = 256 and WG_FOLD = 256 a lane owns exactly
// one element, so the value it squares is the one it just wrote and the 20 KB
// `row[K]` array the single-WG kernel needs is gone with it. That drops this
// kernel's SLM from 21 KB to 1 KB, which is the other half of what lets 20
// work-groups be resident at once.
__attribute__((reqd_work_group_size(WG_FOLD, 1, 1)))
__kernel void prep_res_fold(__global const float* restrict partials,
                            __global ushort* restrict resid,
                            __global float* restrict sumsq) {
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
    for (uint s = 0; s < S_PREV; ++s) acc += partials[((size_t)s * M + m) * K + k];
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
  if (lid == 0) sumsq[(size_t)g * M + m] = red[0];
}

// Stage B: the global Σ, the rms, and the normalised row.
//
//   rstd        = 1 / sqrt(Σ_{g=0}^{NORM_G-1} sumsq[g][m] / K + 1e-6)
//   x_out[m][k] = rne_bf16(f32(resid[m][k]) · rstd · norm_w[k])
//
// **Every lane of every work-group folds the NORM_G partials itself**, in the
// same ascending-`g` order, so every work-group computes bit-identical `rstd`
// from bit-identical inputs and the grid is free of any cross-work-group
// dependency. NORM_G is 20 floats: 80 bytes, the same 80 bytes for every lane
// of the subgroup, and one cache line after the first lane touches it. That is
// what buys the second half of the lever - the rescale phase is the OTHER
// serialised pass over the row (10 KB of `resid` + 20 KB of `norm_w` + 10 KB of
// `x` through one Xe-core), and NORM_WGS spreads it exactly as stage A spreads
// the fold. NORM_WGS = 1 is the plan's literal design and is kept compiled so
// the two can be measured against each other.
//
// `resid` is re-widened from DRAM here rather than handed over in SLM - SLM
// does not survive a launch boundary. The bytes are the ones stage A just
// wrote, so they are L2-warm, and `bf16f(resid[k])` is by construction the same
// float the single-WG kernel kept in `row[k]`.
__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void prep_norm_finish(__global const float* restrict sumsq,
                               __global const ushort* restrict resid,
                               __global const float* restrict norm_w,
                               __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);

  float total = 0.f;
  for (uint g = 0; g < NORM_G; ++g) total += sumsq[(size_t)g * M + m];
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
