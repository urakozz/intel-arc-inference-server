// pf_attn.cl -- the two kernels of the COMPOSED prefill attention path
// (ruling A14): the causal row softmax between the two inherited GEMMs, and the
// output gate that `attn_reduce` used to carry.
//
//   S = Q K^T          gemm_bf16_batched, L = one kv group (6 q-heads)
//   P = softmax(S)     THIS FILE, pf_softmax_causal
//   O = P V            gemm_bf16_batched, L = 6
//   out = O . sigmoid(gate)   THIS FILE, pf_attn_gate
//
// **Label: L1 functional, untuned.** Plan 6d owns the tuned version and its
// 85-100 ms/chunk pre-registration; this file exists so `Engine::prefill` has a
// correct attention at all. What is deliberately NOT done here: no head tiling
// beyond one kv group, no fusion of the three passes over `S`, no online
// (single-pass) softmax, no SLM staging of a row. Every one of those is a
// performance lever and this is a correctness stage.
//
// ---------------------------------------------------------------------------
// pf_softmax_causal -- grid (C rows, LH head slots), work-group 256
// ---------------------------------------------------------------------------
//   s   fp32   [LH][C][ld]   the QK^T GEMM's output; `ld` is the row pitch
//                            (PrefillScratch::pf_s is sized [kSHeads][kC][max_len])
//   p   bf16   [LH][C][ld]   softmax(s), the PV GEMM's A operand
//
// Row `m` of head slot `l` is the query at ABSOLUTE position `pos + m`, so the
// causal bound is `j <= pos + m` -- `nvalid = pos + m + 1` columns. Columns
// `[nvalid, npad)` are written as an exact +0.0 bf16 and never read as scores:
// the PV GEMM's `K` is `npad` (sycl-tla wants a multiple of 8) and a zero
// weight is what makes those padded columns contribute nothing. **That is the
// one reason this kernel writes past `nvalid` at all**, and it is why `npad`
// is an argument rather than derived here.
//
// **Three passes over `s`, and the arithmetic is deliberate**: max, then the
// sum of exponentials, then the normalised store. Traffic is identical to the
// two-pass-plus-rescale form (14 B/element either way) and this one rounds to
// bf16 exactly ONCE, which is the interfaces.md discipline. The score scale
// 1/16 = 1/sqrt(256) is attn.cl's own (doc 03) and is applied to the fp32 GEMM
// output before the max, so the max, the exponent and the store all see the
// same scaled value -- decode scales inside `attn_decode` before its own max
// for the same reason.
//
// Reduction orders, fixed and stated so the host reference can copy them:
//   * lane `t` folds columns `t, t + 256, t + 512, ...` ascending, into a
//     private accumulator (fmax for the max, `+` for the sum);
//   * the 256 lane partials collapse with the pairwise tree
//     `stride = 128, 64, ..., 1`, a barrier after every step -- pf_prep.cl's
//     tree, character for character.
//
// The one op OpenCL does not correctly round here is `exp` (3 ulp), which is
// why the composed path's bar is a tolerance and not an ulp count.
//
// ---------------------------------------------------------------------------
// pf_attn_gate -- grid (24 q-heads, C), work-group 256
// ---------------------------------------------------------------------------
// `attn_reduce`'s tail, and nothing else of it: the composed path has already
// divided by the softmax denominator (P is normalised), so what is left is the
// gate. The chain is attn.cl's, transcribed:
//
//     gate = f32(rne_bf16(qkv_partials[m][h.512 + 256 + d]))
//     out[m][h.256 + d] = rne_bf16( f32(rne_bf16(o)) . sigmoid_f32(gate) )
//
// **The gate is read straight out of `qkv_partials`** rather than staged into a
// buffer of its own. `attn_prep` copies it into `attn_gate` only because
// decode's `attn_reduce` runs after `attn_part` has been built and cannot reach
// the partials any more; on the prefill path `partials` is still the qkv
// linear's output when this kernel runs (the o_proj GEMM, which overwrites it,
// is the next launch but one), so the copy is dead weight. That is why
// `PrefillScratch` has no `attn_gate` field and why `pf_attn_prep_q16` does not
// write one.
//
// `o` is fp32 [24][C][256] with a runtime per-head stride, because the PV GEMM
// writes each q-head's C rows contiguously at `ldc = 256`.

#define WG_SM 256
#define WG_GATE 256

#define Q_HEADS 24
#define HD 256
#define QKV_N 14336      /* q||gate (12288) || k (1024) || v (1024) */
#define OUT_N 6144       /* 24 x 256, the o_proj input row */

// 1/sqrt(head_dim) with head_dim = 256 (doc 03). attn.cl spells the same
// constant `SCALE` and tests/kernels/attn_ref.h `kScale`.
#define ATTN_SCALE (1.0f / 16.0f)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// attn_ref::sigmoid_f32 -- plain `exp`, never `native_exp`, and the reciprocal
// spelled as a divide so -cl-fp32-correctly-rounded-divide-sqrt binds it.
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

__attribute__((reqd_work_group_size(WG_SM, 1, 1)))
__kernel void pf_softmax_causal(__global const float* restrict s,
                                __global ushort* restrict p, uint pos, uint npad,
                                uint ld, uint stride_l) {
  const uint m = get_group_id(0);
  const uint l = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_SM];

  const size_t base = (size_t)l * stride_l + (size_t)m * ld;
  const uint nvalid = pos + m + 1;          // columns [0, pos + m]

  // Pass 1 -- the row max over the causal prefix.
  float mx = -INFINITY;
  for (uint j = lid; j < nvalid; j += WG_SM) mx = fmax(mx, s[base + j] * ATTN_SCALE);
  red[lid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_SM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] = fmax(red[lid], red[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rowmax = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);             // red[] is about to be reused

  // Pass 2 -- the denominator.
  float sum = 0.0f;
  for (uint j = lid; j < nvalid; j += WG_SM) sum += exp(s[base + j] * ATTN_SCALE - rowmax);
  red[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_SM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / red[0];

  // Pass 3 -- the normalised weights, one bf16 rounding, zeros past the bound.
  for (uint j = lid; j < npad; j += WG_SM)
    p[base + j] = j < nvalid ? rne_bf16(exp(s[base + j] * ATTN_SCALE - rowmax) * inv)
                             : (ushort)0;
}

__attribute__((reqd_work_group_size(WG_GATE, 1, 1)))
__kernel void pf_attn_gate(__global const float* restrict o,
                           __global const float* restrict qkv_partials,
                           __global ushort* restrict out, uint stride_h) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  const float ov = o[(size_t)h * stride_h + (size_t)m * HD + d];
  const float g =
      bf16f(rne_bf16(qkv_partials[(size_t)m * QKV_N + (size_t)h * 2 * HD + HD + d]));
  out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(ov)) * sigmoid_f32(g));
}
