// pf_attn_prep.cl -- the widened `attn_prep`: q/k RMSNorm, partial RoPE, and
// the chunk's K/V written into the cache at absolute positions. Grid
// (24 q-heads + 4 kv-heads, C), work-group 256; work-item `i` owns dim `i`, so
// the norm's reduction domain is exactly the work-group.
//
// `src/kernels/attn.cl:307-393` transcribed with three edits and nothing else:
//
//   1. `#ifndef M / #define M 1` is gone, and **so is every use of it**. There
//      were two: `qkv_sum`'s split-K stride, which disappears at QKV_S = 1
//      (plan 6b ruling R1), and the `if (n_act > M) n_act = M;` clamp, which
//      existed because the compiled variant was the ceiling. There is no
//      compiled ceiling here -- every output stride this kernel writes is
//      M-independent (`attn_q`/`attn_gate` are `[.][24*256]`, the KV caches are
//      indexed by absolute position, and the partials row base is
//      `m * QKV_N`), which is why this file needs no `-D M` at all and no
//      `m_count` argument either.
//   2. `QKV_S` is **1**, not decode's 2. Same ruling: the prefill linears run
//      unsplit, so the fold before the linear's bf16 rounding is one term.
//   3. `ATTN_BLOCK` / `MAXLEN` and their `#error` guards are dropped.
//      `attn_prep` expands neither (docs: the define is inert in that binary
//      and only present because it shares a file with the two kernels whose
//      buffer stride depends on it); this file does not share.
//
// `if (m >= n_act) return;` is KEPT. It is the kernel's own contract -- the
// caller may round the grid up -- and it is uniform across the work-group.
//
// The rounding discipline is attn.cl's, unchanged and inherited: the norm
// widens to fp32, sums squares in fp32 (the 256-lane pairwise tree), uses
// `1.0f / sqrt(mean + 1e-6f)` and never `rsqrt`, multiplies by the fp32
// `(1 + w)` weight the loader baked, rounds back to bf16 and widens again;
// RoPE runs on that fp32 value as one rounded product plus one `fma`;
// `attn_q` keeps the fp32 result and the cached k is `rne_bf16` of it; `kv_v`
// is `rne_bf16(partials)` and is never normed or roped.
//
// **`attn_q` and `attn_gate` are fp32 here** because plan 6b's L1 route feeds
// decode's unmodified `attn_decode`/`attn_reduce`, which read them as fp32.
// Interfaces.md ruling A9 makes q **bf16** on the final prefill path, where
// plan 6d's composed attention (ruling A14) needs DPAS operands. That is a
// change of consumer, not of this kernel's arithmetic: the value written is the
// same fp32 `outv`, and A9's extra rounding is one `rne_bf16` at the store.

#ifndef CTRL_POS
#error "pf_attn_prep: CTRL_POS must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef CTRL_NACT
#error "pf_attn_prep: CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef QNORM_OFF
#error "pf_attn_prep: QNORM_OFF must be defined (loader/small_layout.h kFaOffQNorm / 4)"
#endif
#ifndef KNORM_OFF
#error "pf_attn_prep: KNORM_OFF must be defined (loader/small_layout.h kFaOffKNorm / 4)"
#endif

// The model's dimensions (model::Qwen35); none of them is a variant.
#define Q_HEADS 24        /* full-attention q-heads */
#define KV_HEADS 4        /* k/v heads */
#define HD 256            /* head dim, q, k and v alike */
#define QKV_N 14336       /* q||gate (12288) || k (1024) || v (1024) */
#define K_OFF 12288
#define V_OFF 13312
#define ROT_HALF 32       /* partial_rotary_factor 0.25 of 256 -> 64 dims, 32 pairs */
#define ROT_DIM 64

#define QKV_S 1
#if QKV_S != 1
#error "pf_attn_prep: the prefill path folds ONE slice (plan 6b ruling R1)"
#endif

#define WG_PREP 256

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// The fold before the linear's bf16 rounding. attn.cl sums `QKV_S` slices at
// stride `s * M`; at QKV_S = 1 there is one slice, the stride is unreachable
// and the row base is simply `m * QKV_N` -- which is the ONE place `M` was used
// in this kernel besides the `n_act` clamp, and why this file compiles without
// knowing the chunk width. A one-term ascending sum is the same value as
// decode's when its slices 1.. are +0.0f, which is exactly how
// `tests/prefill/pf_attn_test.cc` drives the bit-identity comparison.
inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  return p[(size_t)m * QKV_N + col];
}

__attribute__((reqd_work_group_size(WG_PREP, 1, 1)))
__kernel void pf_attn_prep(__global const uint* restrict ctrl,
                           __global const float* restrict qkv_partials,
                           __global const float* restrict fa_small,
                           __global const float* restrict rope,
                           __global float* restrict attn_q,
                           __global float* restrict attn_gate,
                           __global ushort* restrict kv_k,
                           __global ushort* restrict kv_v) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[WG_PREP];   // the norm's sum of squares
  __local float nrm[HD];        // the normalised head, fp32 -- RoPE needs dim i+-32

  const uint pos = ctrl[CTRL_POS];
  const uint n_act = ctrl[CTRL_NACT];
  if (m >= n_act) return;       // uniform across the work-group

  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;                  // q-head, or kv-head j
  __global const float* restrict nw = fa_small + (is_q ? QNORM_OFF : KNORM_OFF);
  const size_t base = is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD;

  const float xf = bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + i)));
  red[i] = xf * xf;                                        // one term per lane
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_PREP / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = red[0] / (float)HD;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);             // never rsqrt
  nrm[i] = bf16f(rne_bf16(xf * rstd * nw[i]));
  barrier(CLK_LOCAL_MEM_FENCE);

  // Partial RoPE, rotate_half over the 64-slice (doc 03: non-interleaved
  // halves, dims 64..255 untouched):
  //   out_i      = x_i.cos_i     - x_{i+32}.sin_i
  //   out_{i+32} = x_{i+32}.cos_i + x_i.sin_i
  __global const float* restrict cs = rope + (size_t)(pos + m) * 2 * ROT_HALF;
  float outv;
  if (i < ROT_HALF) {
    const float c = cs[i], s = cs[ROT_HALF + i];
    const float t = nrm[i + ROT_HALF] * s;
    outv = fma(nrm[i], c, -t);
  } else if (i < ROT_DIM) {
    const uint ii = i - ROT_HALF;
    const float c = cs[ii], s = cs[ROT_HALF + ii];
    const float t = nrm[ii] * s;
    outv = fma(nrm[i], c, t);
  } else {
    outv = nrm[i];
  }

  if (is_q) {
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = outv;
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + HD + i)));
  } else {
    const size_t slot = ((size_t)(pos + m) * KV_HEADS + h) * HD + i;
    kv_k[slot] = rne_bf16(outv);
    kv_v[slot] = rne_bf16(qkv_sum(qkv_partials, m, V_OFF + (size_t)h * HD + i));
  }
}
