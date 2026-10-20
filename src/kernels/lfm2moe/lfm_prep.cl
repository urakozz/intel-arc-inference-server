// lfm_prep.cl - spec 24c: LFM2.5-24B-A2B's between-GEMV kernels. Four families, each compiled only when its
// defines are given (one binary per family and shape; every define in the binary's name):
//
//   lfm_res_norm(h_in, slices, w, h_out, x)            NORM_S    grid (NORM_WGS, M),     WG 256
//   lfm_res_norm (ROUTER_FUSED: the arm `router`)     + (rw, logits_r)  grid (NORM_WGS + 4, M)
//   lfm_silu_mul(slices, out)                          SILU_I    grid (ceil(I / 256), M), WG 256
//   lfm_attn_prep(ctrl, slices, qkn, rope, attn_q, kv_k, kv_v)   QKV_S   grid (Q + 2 KV heads, M), WG 64
//   lfm_embed_gather(ctrl, tiles, h)                   EMBED_V   grid (1, M),            WG 256
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
// **The arm `router`** (ROUTER_FUSED, spec 24 §4.4 "P0's fused arm g"): work-groups NORM_WGS .. NORM_WGS + 3 compute
// the identical normalised row into SLM and then 16 router columns each in lfm_router's order (lfm_moe.cl:
// 8 lanes of 256 consecutive k per column, ascending, then the stride 4 / 2 / 1 tree) - one launch fewer per MoE
// layer, the logits bitwise lfm_router's on the same row.
//
// **No sub-group function anywhere in this file**, so it also builds as OpenCL 1.2 for an indicative run on the
// Mac's GPU (tools/mac/clrun/lfm2moe_run.cc).
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
#ifdef ROUTER_FUSED
#if !defined(ROUTER_E) || ROUTER_E != 64 || K % 64 != 0
#error "lfm_res_norm _R: the router is 64 rows (4 work-groups of 16 columns) over whole k-octet spans"
#endif
#define NORM_GROUPS (NORM_WGS + ROUTER_E / 16)
#define RSPAN (K / 8)
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
#ifdef ROUTER_FUSED
                           , __global const ushort* restrict rw, __global float* restrict logits_r
#endif
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
#ifdef ROUTER_FUSED
  if (wg >= NORM_WGS) {   // uniform: a router work-group (the arm `router`)
    for (uint k = lid; k < K; k += WG_NORM) rs[k] = bf16f(lfm_norm_elem(rs[k], rstd, norm_w[k]));
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float rr[8][16];
    const uint lane = lid % 16, j = lid / 16;   // lanes 0..127: column lane of the n-tile, k-span j
    const uint nt = wg - NORM_WGS;
    if (lid < 128) {
      float a = 0.0f;
      const __global ushort* restrict t = rw + (size_t)nt * (K / 8) * 128;
      for (uint k = j * RSPAN; k < (j + 1) * RSPAN; ++k)
        a += bf16f(t[((size_t)(k / 8) * 8 + k % 8) * 16 + lane]) * rs[k];
      rr[j][lane] = a;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 4; stride > 0; stride >>= 1) {
      if (lid < 128 && j < stride) rr[j][lane] += rr[j + stride][lane];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < 16) logits_r[(size_t)m * ROUTER_E + nt * 16 + lane] = rr[0][lane];
    return;
  }
#endif
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

// ---------------------------------------------------------------------------------------------------------------
// lfm_silu_mul - prep_silu_mul's chain at LFM2's dense intermediate (prep.cl refuses 11776): gate||up interleaved
// in 16-column blocks (common::cols_interleave16), SILU_S split-K slices:
//   g_b = rne(Σ_s slices[s][m][gflat]),  u_b likewise;  out[m][i] = rne(f32(rne(silu(f32(g_b)))) · f32(u_b))
#ifdef SILU_I
#ifndef SILU_S
#error "lfm_silu_mul: SILU_S (the gate||up GEMV's split-K) must be defined"
#endif
#define WG_SILU 256
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
__attribute__((reqd_work_group_size(WG_SILU, 1, 1)))
__kernel void lfm_silu_mul(__global const float* restrict slices, __global ushort* restrict out) {
  const uint m = get_group_id(1);
  const uint i = get_group_id(0) * WG_SILU + get_local_id(0);
  if (i >= SILU_I) return;
  const size_t FN = (size_t)2 * SILU_I;
  const size_t gflat = (size_t)(i / 16) * 32 + (i % 16), uflat = gflat + 16;
  float ga = 0.0f, ua = 0.0f;
  for (uint s = 0; s < SILU_S; ++s) {
    const size_t base = ((size_t)s * M + m) * FN;
    ga += slices[base + gflat];
    ua += slices[base + uflat];
  }
  const ushort g_b = rne_bf16(ga), u_b = rne_bf16(ua);
  out[(size_t)m * SILU_I + i] = rne_bf16(bf16f(rne_bf16(silu_f32(bf16f(g_b)))) * bf16f(u_b));
}
#endif  // SILU_I

// ---------------------------------------------------------------------------------------------------------------
// lfm_attn_prep - kol_attn_prep (src/kernels/kolibri/kol_prep.cl) at head_dim 64, eps 1e-5, RoPE on EVERY attention
// layer (spec 24 §4.2; Kolibri rotates only its sliding layers) over a growing cache. Grid (Q_HEADS + 2 KV_HEADS, M),
// work-group HD (64), work-item d owns dim d of one head: work-groups 0..Q-1 q heads, Q..Q+KV-1 k heads, the last
// KV the v heads.
//
//   x_b   = rne(Σ_{s < QKV_S} slices[s][m][col])                      the linear's bf16 output
//   rstd  = 1 / sqrt(tree_d(fma(x_b, x_b, 0)) / 64 + 1e-5)            q, k: per head (64-lane tree)
//   y     = rne(f32(rne(x_b · rstd)) · w_qk[d])                       q_layernorm / k_layernorm, plain w
//   out   = rne(f32(rne(y · cos)) + f32(rne(rot(y) · sin)))           rot(y)[d] = d < 32 ? -y[d + 32] : y[d - 32]
//   q -> attn_q[m][h][d] (fp32 of a bf16 value); k -> kv_k[pos + m][j][d]; v = x_b -> kv_v[pos + m][j][d]
//
// `qkn` is the layer's q_norm [64] then k_norm [64] (fp32: loader::lfm::small_block's q_norm / k_norm, adjacent);
// `rope` the table [max_len][2][32] (cos then sin, bf16 values: loader::lfm_rope_table).
#ifdef QKV_S
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "lfm_attn_prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HD) || !defined(PREP_EPS)
#error "lfm_attn_prep: Q_HEADS, KV_HEADS, HD and PREP_EPS must be defined"
#endif
#define QKV_N ((Q_HEADS + 2 * KV_HEADS) * HD)
#define HALF (HD / 2)
inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
}
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void lfm_attn_prep(__global const uint* restrict ctrl, __global const float* restrict slices,
                            __global const float* restrict qkn, __global const float* restrict rope,
                            __global float* restrict attn_q, __global ushort* restrict kv_k,
                            __global ushort* restrict kv_v) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float xs[HD];
  __local float red[HD];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform across the work-group

  const uint kind = wg < Q_HEADS ? 0u : (wg < Q_HEADS + KV_HEADS ? 1u : 2u);   // q, k, v
  const uint h = kind == 0 ? wg : (kind == 1 ? wg - Q_HEADS : wg - Q_HEADS - KV_HEADS);
  const size_t col = (size_t)(kind == 0 ? 0 : (kind == 1 ? Q_HEADS * HD : (Q_HEADS + KV_HEADS) * HD)) +
                     (size_t)h * HD + d;
  const float xb = rf(qkv_sum(slices, m, col));
  const size_t row = (size_t)(pos + m);
  if (kind == 2) {   // v: the linear's output, no norm, no rotation
    kv_v[(row * KV_HEADS + h) * HD + d] = rne_bf16(xb);
    return;
  }
  red[d] = fma(xb, xb, 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = HD / 2; stride > 0; stride >>= 1) {
    if (d < stride) red[d] += red[d + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)HD + PREP_EPS);
  const float y = bf16f(lfm_norm_elem(xb, rstd, qkn[(kind == 0 ? 0 : HD) + d]));
  xs[d] = y;
  barrier(CLK_LOCAL_MEM_FENCE);
  const __global float* restrict cs = rope + row * 2 * HALF;
  const uint ii = d % HALF;
  const float c = cs[ii], sn = cs[HALF + ii];
  const float rot = d < HALF ? -xs[d + HALF] : xs[d - HALF];
  const ushort out = rne_bf16(rf(y * c) + rf(rot * sn));
  if (kind == 0)
    attn_q[((size_t)m * Q_HEADS + h) * HD + d] = bf16f(out);
  else
    kv_k[(row * KV_HEADS + h) * HD + d] = out;
}
#endif  // QKV_S

// ---------------------------------------------------------------------------------------------------------------
// lfm_embed_gather - row cur_token[m] of the bf16 embedding out of gemv_bf16's tiles {K, EMBED_V} (the ONE copy
// that also serves the bf16 head, spec 24 §4.4), into the residual h[m]: element (k, n) of the [vocab][K] tensor
// sits at ((n / 16) x K/8 + k / 8) x 8 + k % 8) x 16 + n % 16 (loader::lfm::lfm_embed_row_index). An id >= EMBED_V
// reads row EMBED_V - 1 (never past the allocation; the host refuses such ids before a replay).
#ifdef EMBED_V
#if !defined(CTRL_CUR) || !defined(CTRL_NACT) || !defined(K)
#error "lfm_embed_gather: CTRL_CUR / CTRL_NACT and K must be defined"
#endif
#define WG_EMBED 256
__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void lfm_embed_gather(__global const uint* restrict ctrl, __global const ushort* restrict tiles,
                               __global ushort* restrict h) {
  const uint m = get_group_id(1);
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
  uint id = ctrl[CTRL_CUR + m];
  if (id >= EMBED_V) id = EMBED_V - 1;
  const size_t base = (size_t)(id / 16) * (K / 8);
  for (uint k = get_local_id(0); k < K; k += WG_EMBED)
    h[(size_t)m * K + k] = tiles[((base + k / 8) * 8 + k % 8) * 16 + id % 16];
}
#endif  // EMBED_V
