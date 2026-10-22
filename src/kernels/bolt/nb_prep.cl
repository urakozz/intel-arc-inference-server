// nb_prep.cl - spec 25c: Nemotron 3.5 Bolt's between-GEMV kernels. Four families, each compiled only when its defines
// are given (one binary per family and shape; every define in the binary's name):
//
//   nb_res_norm(h_in, slices, slices2, w, h_out, x)        NORM_S   grid (NORM_WGS, M), WG 256
//   nb_res_norm (ROUTER_FUSED: the arm `router`)          + (rw, logits_r, n_exp)  grid (NORM_WGS + n_exp / 16, M)
//   nb_sh_act(slices, lat, sh)                             ACT_S    grid (25, M), WG 256
//   nb_attn_prep(ctrl, slices, attn_q, kv_k, kv_v)         QKV_S    grid (18, M), WG 256
//   nb_embed_gather(ctrl, tiles, h)                        EMBED_V  grid (1, M), WG 256
//
// The reference is 25a's tools/oracle/bolt_ref.py (spec 25 §2); tests/kernels/bolt_ref.h repeats every chain below - EDIT
// THE TWO TOGETHER. The rounding discipline: a GEMV's split-K partials summed from 0.0f in ascending slice order and
// rounded to bf16 once, every elementwise op in fp32, rounded where the reference rounds.
//
// **The norm site** (spec 25 §2.1, §6.1; plan 25c Review Focus 4): ONE launch a site, and no block folds its own output -
// the next norm folds whatever split-K slices are pending: NORM_S > 0 with NORM_S2 == 0 one source (an out_proj /
// o_proj: r = rne(h + rne(Σ_s slices))), NORM_S2 > 0 the MoE block's two (fc2 then shared_down at the slices' offset the
// capture gives slices2: r = rne(h + rne(rne(Σ fc2) + rne(Σ shared_down))) - spec 25 §2.4's three roundings), NORM_S ==
// 0 none (`_NF`: block 0's norm over the gathered embedding). Every work-group folds the WHOLE row and reduces Σ r² by ONE
// fixed tree - the HALVING tree over the 4096 squares (lane l holds k = l + 256 j, j < 16: in-lane j vs j + 8, + 4, + 2,
// + 1, then 128 .. 1 across the lanes - v[:n/2] + v[n/2:] at every level), the same bits in every work-group - and writes
// only its own chunk of h_out (NORM_S > 0) and of x = rne((r · rstd) · w), rstd = 1 / sqrt(Σ / 4096 + 1e-5) (never
// rsqrt): NemotronHRMSNorm's ONE rounding. The residual PING-PONGS: h_in and h_out are different buffers (the capture
// alternates them per fold site), so no work-group reads a chunk another one is rewriting.
//
// **The arm `router`** (ROUTER_FUSED, spec 25 §6.1, Decision 16): work-groups NORM_WGS .. NORM_WGS + n_exp / 16 - 1 compute
// the identical normalised row into SLM and then 16 router columns each in nb_router's order (nb_moe.cl: 8 lanes of 512
// consecutive k per column, ascending, then the stride 4 / 2 / 1 tree) - one launch fewer per MoE block, the logits bitwise
// nb_router's on the same row.
//
// **No sub-group function anywhere in this file**, so it also builds as OpenCL 1.2 for an indicative run on the Mac's
// GPU (tools/mac/clrun/bolt_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#include "nb_common.h"

// ---------------------------------------------------------------------------------------------------------------
#ifdef NORM_S
#if !defined(K) || !defined(NORM_WGS) || !defined(NORM_EPS) || !defined(NORM_S2)
#error "nb_res_norm: K, NORM_WGS, NORM_EPS and NORM_S2 must be defined"
#endif
#define WG_NORM 256
#define NORM_PER (K / WG_NORM)          /* 16 at K 4096 */
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
#if K % WG_NORM != 0 || (NORM_PER & (NORM_PER - 1)) != 0
#error "nb_res_norm: K is a power-of-two number of 256-lane rows (the halving tree)"
#endif
#if NORM_S == 0 && NORM_S2 != 0
#error "nb_res_norm: a second source needs a first"
#endif
#ifdef ROUTER_FUSED
#if K % 64 != 0
#error "nb_res_norm _R: the router is 16-column tiles over whole k-octet spans"
#endif
#define RSPAN (K / 8)
#endif

// Row m's folded value at k (each source's slices summed from 0.0f, ascending s).
inline float nb_fold(__global const ushort* restrict h_in, __global const float* restrict s1,
                     __global const float* restrict s2, uint m, uint k) {
#if NORM_S > 0
  float a = 0.0f;
  for (uint s = 0; s < NORM_S; ++s) a += s1[((size_t)s * M + m) * K + k];
#if NORM_S2 > 0
  float b = 0.0f;
  for (uint s = 0; s < NORM_S2; ++s) b += s2[((size_t)s * M + m) * K + k];
  const float mixer = rf(rf(a) + rf(b));
  return rf(bf16f(h_in[(size_t)m * K + k]) + mixer);
#else
  (void)s2;
  return rf(bf16f(h_in[(size_t)m * K + k]) + rf(a));
#endif
#else
  (void)s1;
  (void)s2;
  return bf16f(h_in[(size_t)m * K + k]);
#endif
}

__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void nb_res_norm(__global const ushort* restrict h_in, __global const float* restrict slices,
                          __global const float* restrict slices2, __global const float* restrict norm_w,
                          __global ushort* restrict h_out, __global ushort* restrict x_out
#ifdef ROUTER_FUSED
                          , __global const ushort* restrict rw, __global float* restrict logits_r, uint n_exp
#endif
                          ) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float rs[K];          // the folded row (bf16 values), then (router WGs) the normalised row
  __local float red[WG_NORM];
#ifdef ROUTER_FUSED
  __local float rr[8][16];      // the router work-groups' 8 k-span partials of 16 columns
#endif
  float sq[NORM_PER];
  for (uint j = 0; j < NORM_PER; ++j) {
    const uint k = lid + j * WG_NORM;
    const float v = nb_fold(h_in, slices, slices2, m, k);
    rs[k] = v;
    sq[j] = v * v;
  }
  for (uint hv = NORM_PER / 2; hv > 0; hv >>= 1)
    for (uint j = 0; j < hv; ++j) sq[j] = sq[j] + sq[j + hv];
  red[lid] = sq[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_NORM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] = red[lid] + red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)K + NORM_EPS);
#ifdef ROUTER_FUSED
  if (wg >= NORM_WGS) {   // uniform: a router work-group (the arm `router`)
    for (uint k = lid; k < K; k += WG_NORM) rs[k] = rf((rs[k] * rstd) * norm_w[k]);
    barrier(CLK_LOCAL_MEM_FENCE);
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
    if (lid < 16) logits_r[(size_t)m * n_exp + nt * 16 + lane] = rr[0][lane];
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
    x_out[(size_t)m * K + k] = rne_bf16((rs[k] * rstd) * norm_w[k]);
  }
}
#endif  // NORM_S

// ---------------------------------------------------------------------------------------------------------------
// nb_sh_act - fc1_latent || shared_experts.up_proj's ONE GEMV (N 6400: columns [0, 1024) the latent, [1024, 6400) the
// shared expert's up), ACT_S split-K slices:
//   lat[m][c]  = rne(Σ_s slices[s][m][c])                     c < 1024 - the routed experts' input (spec 25 §2.4)
//   sh[m][i]   = relu2(rne(Σ_s slices[s][m][1024 + i]))       relu2(u) = rne(relu(u)²) on bf16 (torch.square(F.relu))
#ifdef ACT_S
#define ACT_L 1024
#define ACT_SH 5376
#define ACT_N (ACT_L + ACT_SH)
#define WG_ACT 256
__attribute__((reqd_work_group_size(WG_ACT, 1, 1)))
__kernel void nb_sh_act(__global const float* restrict slices, __global ushort* restrict lat, __global ushort* restrict sh) {
  const uint m = get_group_id(1);
  const uint c = get_group_id(0) * WG_ACT + get_local_id(0);
  if (c >= ACT_N) return;
  float v = 0.0f;
  for (uint s = 0; s < ACT_S; ++s) v += slices[((size_t)s * M + m) * ACT_N + c];
  if (c < ACT_L) {
    lat[(size_t)m * ACT_L + c] = rne_bf16(v);
  } else {
    float u = rf(v);
    u = u > 0.0f ? u : 0.0f;
    sh[(size_t)m * ACT_SH + (c - ACT_L)] = rne_bf16(u * u);
  }
}
#endif  // ACT_S

// ---------------------------------------------------------------------------------------------------------------
// nb_attn_prep - the NoPE attention's prep (spec 25 §2.3): q / k / v = rne(Σ_s q||k||v slices) - no bias, no q / k norm,
// no position encoding. q -> attn_q[m][h][d] as the fp32 of the bf16 value (mg_attn's input, Decision 15); k / v -> the
// growing cache rows pos + m, [rows][2][128] (mg_attn's layout: Decision 15's check). Grid (ceil(4608 / 256), M).
#ifdef QKV_S
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "nb_attn_prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#define PREP_Q 4096
#define PREP_KV 256
#define PREP_N (PREP_Q + 2 * PREP_KV)
#define WG_PREP 256
__attribute__((reqd_work_group_size(WG_PREP, 1, 1)))
__kernel void nb_attn_prep(__global const uint* restrict ctrl, __global const float* restrict slices,
                           __global float* restrict attn_q, __global ushort* restrict kv_k,
                           __global ushort* restrict kv_v) {
  const uint m = get_group_id(1);
  const uint c = get_group_id(0) * WG_PREP + get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act || c >= PREP_N) return;
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += slices[((size_t)s * M + m) * PREP_N + c];
  const size_t row = (size_t)(pos + m);
  if (c < PREP_Q)
    attn_q[(size_t)m * PREP_Q + c] = rf(v);
  else if (c < PREP_Q + PREP_KV)
    kv_k[row * PREP_KV + (c - PREP_Q)] = rne_bf16(v);
  else
    kv_v[row * PREP_KV + (c - PREP_Q - PREP_KV)] = rne_bf16(v);
}
#endif  // QKV_S

// ---------------------------------------------------------------------------------------------------------------
// nb_embed_gather - row cur_token[m] of the bf16 embedding out of gemv_bf16's tiles {K, EMBED_V} into the residual h[m]:
// element (k, n) of the [vocab][K] tensor sits at ((n / 16) x K/8 + k / 8) x 8 + k % 8) x 16 + n % 16
// (loader::bolt::embed_row_index). An id >= EMBED_V reads row EMBED_V - 1 (never past the allocation; the host refuses
// such ids before a replay).
#ifdef EMBED_V
#if !defined(CTRL_CUR) || !defined(CTRL_NACT) || !defined(K)
#error "nb_embed_gather: CTRL_CUR / CTRL_NACT and K must be defined"
#endif
#define WG_EMBED 256
__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void nb_embed_gather(__global const uint* restrict ctrl, __global const ushort* restrict tiles,
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
