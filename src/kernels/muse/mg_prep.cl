// mg_prep.cl - spec 23c: Muse-Glimmer-30B's between-GEMV kernels. Five families, each compiled only when its
// defines are given (one binary per family and shape; src/kernels/muse_kernels.h names them):
//
//   mg_embed(ctrl, embed, resid, sumsq)                         EMBED_G   grid (1, M),            WG 256
//   mg_norm(sumsq, resid, w, x_out)                             NORM_G    grid (NORM_WGS, M),     WG 256
//   mg_final_norm(sumsq, resid, w, x_out)                       NORM_G    (the same binary)
//   mg_post_add(sumsq_sub, sub, w_post, resid, sumsq_out)       POST_G    grid (POST_G, M),       WG 256
//   mg_attn_prep(ctrl, partials, rope, attn_q, gate_out,        QGKV_N    grid (2 Q + 2 KV, M),   WG 128
//                kv_k, kv_v)
//   mg_silu_mul(partials, h)                                    SILU_I    grid (ceil(I / 256), M), WG 256
//
// The reference is transformers 5.19.0's muse_glimmer (spec 23 §1.1); tests/kernels/muse_ref.h repeats every chain
// below - edit the two together. The rounding discipline is prep.cl's: a linear's split-K partials summed in fp32
// and rounded to bf16 once, every elementwise op widened to fp32 and rounded where the reference rounds.
//
// **Muse's norms are not Kolibri's** (plan 23c Review Focus 1): the four decoder norms are CENTERED -
// `x * rsqrt(ms + eps) * (1 + w)` in fp32, ONE rounding (M:122-136) - where kol_prep.cl rounds x̂ first and
// multiplies by plain w. eps 1e-5 for input / pre-feedforward (mg_norm, NORM_EPS), 1e-8 for the two post norms
// (mg_post_add, POST_EPS). The final norm is plain w with `pow(ms + 1e-5, -0.5)` (M:101-119), and the embedding's
// norm is weight-less (M:428-436). `pow(x, -0.5)`, `rsqrt` and `1 / sqrt` are the same fp32 value on torch's CPU
// (23a, 10^7 draws): every rstd here is the correctly rounded `1 / sqrt` (-cl-fp32-correctly-rounded-divide-sqrt is
// the project default, cmake/ocloc.cmake).
//
// **The Σ² trees** are prep.cl prep_res_fold's: G chunks of ceil(K / G) columns, lane i of chunk g takes
// k = g·chunk + i, + 256, ... with fma, then 256 -> 1 pairwise; a norm adds the G sums ascending. At hidden 6656,
// G = 26 (chunks of exactly 256). mg_embed and mg_post_add write those sums for the next norm; prep_res_fold ..._Z
// writes a sub-block's (o_proj's, down's) own row and sums for mg_post_add (the sandwich: the post norms normalise
// the sub-block's OWN bf16 output before the add).
//
// **No sub-group function anywhere in this file**, so it also builds as OpenCL 1.2 for an indicative run on the
// Mac's GPU (tools/mac/clrun/muse_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#ifndef K
#define K 6656              /* hidden */
#endif
#define WG_ROW 256

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (prep.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
// One element of the centered norm: (x · rstd) · (1 + w), one rounding.
inline ushort mg_centered(float x, float rstd, float w) { return rne_bf16((x * rstd) * (1.0f + w)); }

// Sleef_expf8_u10 (sleefsimdsp.c xexpf, AVX2 + FMA) - k2_ref::exp_torch / k2_attn_eager.cl's, the same lines:
// torch's Vectorized<float>::exp, what its CPU sigmoid / SiLU evaluate on bf16 tensors.
inline float exp_torch(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

// ---------------------------------------------------------------------------------------
// mg_embed - the gathered bf16 row through the weight-less norm (M:451-453); its output IS the residual stream
// (M:504), so it writes `resid` and the per-chunk Σ² layer 0's mg_norm reads. One work-group per row:
//   id      = ctrl[CTRL_CUR + m]; the image / video placeholders (IMG_TOKEN, VID_TOKEN) embed as row 0, as the
//             reference's text path does (M:1010-1013: `llm_input_ids[multimodal_mask] = 0`)
//   rstd    = 1 / sqrt(Σ_g chunk_g(row²) / K + 1e-5)          the raw row's chunk trees, ascending
//   resid   = rne(f32(row) · rstd)                             one rounding (with_scale False)
//   sumsq_g = chunk_g(resid²)                                  [EMBED_G][M]
// An id >= VOCAB writes nothing and sets Control::debug_flag (embed_gather.cl's rule).
#ifdef EMBED_G
#if !defined(CTRL_CUR) || !defined(CTRL_DEBUG) || !defined(VOCAB) || !defined(IMG_TOKEN) || !defined(VID_TOKEN)
#error "mg_embed: CTRL_CUR, CTRL_DEBUG, VOCAB, IMG_TOKEN and VID_TOKEN must be defined"
#endif
#if M > 8
#error "mg_embed: M > 8 exceeds Control::cur_token[]"
#endif
#define EMBED_EPS 1e-5f
#define EMBED_CHUNK ((K + EMBED_G - 1) / EMBED_G)
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_embed(__global uint* restrict ctrl, __global const ushort* restrict embed,
                       __global ushort* restrict resid, __global float* restrict sumsq) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_ROW];
  __local float tot[1];
  uint id = ctrl[CTRL_CUR + m];
  if (id == IMG_TOKEN || id == VID_TOKEN) id = 0u;
  if (id >= VOCAB) {   // uniform: an engine bug, not a token - report, write nothing
    if (lid == 0) ctrl[CTRL_DEBUG] = 0xE0000000u | m;
    return;
  }
  const __global ushort* row = embed + (size_t)id * K;
  __global ushort* rp = resid + (size_t)m * K;
  if (lid == 0) tot[0] = 0.0f;
  for (uint g = 0; g < EMBED_G; ++g) {   // pass 1: the raw row's chunk trees, added ascending
    const uint k0 = g * EMBED_CHUNK;
    uint k1 = k0 + EMBED_CHUNK;
    if (k1 > K) k1 = K;
    float acc2 = 0.f;
    for (uint k = k0 + lid; k < k1; k += WG_ROW) {
      const float v = bf16f(row[k]);
      acc2 = fma(v, v, acc2);
    }
    red[lid] = acc2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG_ROW / 2; stride > 0; stride >>= 1) {
      if (lid < stride) red[lid] += red[lid + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) tot[0] += red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(tot[0] / (float)K + EMBED_EPS);
  for (uint g = 0; g < EMBED_G; ++g) {   // pass 2: the normed row (the residual) and its chunk trees
    const uint k0 = g * EMBED_CHUNK;
    uint k1 = k0 + EMBED_CHUNK;
    if (k1 > K) k1 = K;
    float acc2 = 0.f;
    for (uint k = k0 + lid; k < k1; k += WG_ROW) {
      const ushort y = rne_bf16(bf16f(row[k]) * rstd);
      rp[k] = y;
      const float v = bf16f(y);
      acc2 = fma(v, v, acc2);
    }
    red[lid] = acc2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG_ROW / 2; stride > 0; stride >>= 1) {
      if (lid < stride) red[lid] += red[lid + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sumsq[(size_t)g * M + m] = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // EMBED_G

// ---------------------------------------------------------------------------------------
// mg_norm - input_layernorm / pre_feedforward_layernorm (centered, NORM_EPS) over the residual's chunk sums:
//   rstd    = 1 / sqrt(Σ_{g ascending} sumsq[g][m] / K + NORM_EPS)
//   x[m][k] = rne((f32(resid[m][k]) · rstd) · (1 + w[k]))
// mg_final_norm - model.norm (MuseGlimmerRMSNorm with weight): plain w, eps 1e-5:
//   x[m][k] = rne((f32(resid[m][k]) · rstd) · w[k])
// `w` is the checkpoint's bf16 weight widened to fp32 (loader::MuseLayer::norms, MuseDevicePart::final_norm).
#ifdef NORM_G
#if !defined(NORM_WGS) || !defined(NORM_EPS)
#error "mg_norm: NORM_WGS and NORM_EPS must be defined"
#endif
#define FINAL_EPS 1e-5f
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
inline float mg_rstd(__global const float* restrict sumsq, uint m, float eps) {
  float total = 0.f;
  for (uint g = 0; g < NORM_G; ++g) total += sumsq[(size_t)g * M + m];
  return 1.0f / sqrt(total / (float)K + eps);
}
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_norm(__global const float* restrict sumsq, __global const ushort* restrict resid,
                      __global const float* restrict norm_w, __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const float rstd = mg_rstd(sumsq, m, NORM_EPS);
  const __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = w * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_ROW) x_out[(size_t)m * K + k] = mg_centered(bf16f(rp[k]), rstd, norm_w[k]);
}
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_final_norm(__global const float* restrict sumsq, __global const ushort* restrict resid,
                            __global const float* restrict norm_w, __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const float rstd = mg_rstd(sumsq, m, FINAL_EPS);
  const __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = w * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_ROW) x_out[(size_t)m * K + k] = rne_bf16((bf16f(rp[k]) * rstd) * norm_w[k]);
}
#endif  // NORM_G

// ---------------------------------------------------------------------------------------
// mg_post_add - the sandwich's post norm and the residual add, fused with the next norm's chunk trees. `sub` is
// the sub-block's bf16 output [M][K] (o_proj's or down's row) and `sumsq_sub` its POST_G chunk sums (prep.cl's
// prep_res_fold ..._Z wrote both); work-group g owns chunk g:
//   rstd_s = 1 / sqrt(Σ_{g ascending} sumsq_sub[g][m] / K + POST_EPS)        eps 1e-8 (post_norm_eps)
//   n      = rne((f32(sub[m][k]) · rstd_s) · (1 + w[k]))                     post_*_layernorm(sub-block)
//   r      = rne(f32(resid[m][k]) + f32(n));  resid[m][k] = r                 residual + ...
//   sumsq_out[g][m] = chunk_g(r²)                                             for the next mg_norm
#ifdef POST_G
#ifndef POST_EPS
#error "mg_post_add: POST_EPS must be defined"
#endif
#define POST_CHUNK ((K + POST_G - 1) / POST_G)
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_post_add(__global const float* restrict sumsq_sub, __global const ushort* restrict sub,
                          __global const float* restrict norm_w, __global ushort* restrict resid,
                          __global float* restrict sumsq_out) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_ROW];
  float total = 0.f;
  for (uint q = 0; q < POST_G; ++q) total += sumsq_sub[(size_t)q * M + m];
  const float rstd = 1.0f / sqrt(total / (float)K + POST_EPS);
  __global ushort* rp = resid + (size_t)m * K;
  const __global ushort* sp = sub + (size_t)m * K;
  const uint k0 = g * POST_CHUNK;
  uint k1 = k0 + POST_CHUNK;
  if (k1 > K) k1 = K;
  float acc2 = 0.f;
  for (uint k = k0 + lid; k < k1; k += WG_ROW) {
    const ushort n = mg_centered(bf16f(sp[k]), rstd, norm_w[k]);
    const ushort r = rne_bf16(bf16f(rp[k]) + bf16f(n));
    rp[k] = r;
    const float v = bf16f(r);
    acc2 = fma(v, v, acc2);
  }
  red[lid] = acc2;   // a lane with no element contributes an exact 0.0f (prep_res_fold's tree)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_ROW / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sumsq_out[(size_t)g * M + m] = red[0];
}
#endif  // POST_G

// ---------------------------------------------------------------------------------------
// mg_attn_prep - q / gate / k / v out of the fused q||gate||k||v GEMV (8704 columns: q 4096, gate 4096, k 256,
// v 256 - model::MuseGlimmerDesc::qgkv_off_*). Grid (2 Q_HEADS + 2 KV_HEADS, M), work-group HD (128), work-item d
// owns dim d of one 128-column group: groups [0, Q) q heads, [Q, 2Q) the gate's, then KV k heads, then KV v heads.
//
//   x_b   = rne(Σ_{s < QGKV_S} partials[s][m][col])                   the linear's bf16 output
//   q, k: rstd = 1 / sqrt(tree_d(fma(x_b, x_b, 0)) / 128 + 1e-5)      the weight-less qk_norm, per head
//         y = rne(x_b · rstd)                                          q̂ / k̂ (bf16)
//         q: y = rne(f32(y) · 3.87f)                                    x qk_scale_factor: a second rounding
//         SLIDING: y = rne(f32(rne(y · cos)) + f32(rne(rot(y) · sin)))  rot(y)[d] = d < 64 ? -y[d+64] : y[d-64]
//         q -> attn_q[m][h][d] (fp32 of a bf16 value); k -> kv_k[row][j][d]
//   v:    kv_v[row][j][d] = x_b
//   gate: gate_out[m][h·128 + d] = rne(sigmoid(f32(x_b)))             torch.sigmoid(gate_proj(xn)), bf16 (M:358)
//   row = SLIDING ? (pos + m) & (RING - 1) : pos + m                   the ring / the growing NoPE cache
//
// No weights: Muse's q / k norms are weight-less (with_scale False, M:312). `rope` is the table
// [max_len][2][64] (runtime::muse::rope_table; read only when SLIDING). The full layers are NoPE: no rotary,
// whatever the position (plan 23c Review Focus 5).
#ifdef QGKV_N
#ifndef QGKV_S
#error "mg_attn_prep: QGKV_S (the fused GEMV's split-K) must be defined"
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "mg_attn_prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HD) || !defined(SLIDING) || !defined(RING) || !defined(QK_SCALE)
#error "mg_attn_prep: Q_HEADS, KV_HEADS, HD, SLIDING, RING and QK_SCALE must be defined"
#endif
#if QGKV_N != (2 * Q_HEADS + 2 * KV_HEADS) * HD
#error "mg_attn_prep: q||gate||k||v is (2 Q_HEADS + 2 KV_HEADS) x HD wide"
#endif
#if SLIDING && (RING == 0 || (RING & (RING - 1)) != 0)
#error "mg_attn_prep: the sliding ring is a power of two"
#endif
#define QK_EPS 1e-5f
#define HALF (HD / 2)
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void mg_attn_prep(__global const uint* restrict ctrl, __global const float* restrict partials,
                           __global const float* restrict rope, __global float* restrict attn_q,
                           __global ushort* restrict gate_out, __global ushort* restrict kv_k,
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

  // 0 q, 1 gate, 2 k, 3 v; columns are wg x HD + d in every case (the four parts are whole heads, in order).
  const uint kind = wg < Q_HEADS ? 0u : (wg < 2 * Q_HEADS ? 1u : (wg < 2 * Q_HEADS + KV_HEADS ? 2u : 3u));
  const uint h = kind == 0 ? wg : (kind == 1 ? wg - Q_HEADS : (kind == 2 ? wg - 2 * Q_HEADS : wg - 2 * Q_HEADS - KV_HEADS));
  const size_t col = (size_t)wg * HD + d;
  float acc = 0.0f;
  for (uint s = 0; s < QGKV_S; ++s) acc += partials[((size_t)s * M + m) * QGKV_N + col];
  const float xb = rf(acc);
  if (kind == 1) {
    gate_out[((size_t)m * Q_HEADS + h) * HD + d] = rne_bf16(sigmoid_t(xb));
    return;
  }
  const uint p = pos + m;
#if SLIDING
  const size_t row = (size_t)(p & (RING - 1));
#else
  const size_t row = (size_t)p;
#endif
  if (kind == 3) {
    kv_v[(row * KV_HEADS + h) * HD + d] = rne_bf16(xb);
    return;
  }
  red[d] = fma(xb, xb, 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = HD / 2; stride > 0; stride >>= 1) {
    if (d < stride) red[d] += red[d + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)HD + QK_EPS);
  float y = rf(xb * rstd);
  if (kind == 0) y = rf(y * QK_SCALE);
#if SLIDING
  xs[d] = y;
  barrier(CLK_LOCAL_MEM_FENCE);
  const __global float* restrict cs = rope + (size_t)p * 2 * HALF;
  const uint ii = d % HALF;
  const float c = cs[ii], sn = cs[HALF + ii];
  const float rot = d < HALF ? -xs[d + HALF] : xs[d - HALF];
  const ushort out = rne_bf16(rf(y * c) + rf(rot * sn));
#else
  (void)rope;
  (void)xs;
  const ushort out = rne_bf16(y);
#endif
  if (kind == 0)
    attn_q[((size_t)m * Q_HEADS + h) * HD + d] = bf16f(out);
  else
    kv_k[(row * KV_HEADS + h) * HD + d] = out;
}
#endif  // QGKV_N

// ---------------------------------------------------------------------------------------
// mg_silu_mul - the MLP's activation over gate || up CONCATENATED (up at column SILU_I; loader 23b - not Qwen's
// interleave16, so prep.cl's prep_silu_mul does not apply), from the GEMV's SILU_S split-K partials:
//   g = rne(Σ_s partials[s][m][i]);  u = rne(Σ_s partials[s][m][SILU_I + i])
//   h[m][i] = rne(f32(rne(silu(g))) · u)                                act_fn(gate_proj(x)) * up_proj(x), M:154
// SiLU through exp_torch (torch's Vectorized<float>::exp).
#ifdef SILU_I
#ifndef SILU_S
#error "mg_silu_mul: SILU_S (the gate||up GEMV's split-K) must be defined"
#endif
#define SILU_N (2 * SILU_I)
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_silu_mul(__global const float* restrict partials, __global ushort* restrict h) {
  const uint i = get_group_id(0) * WG_ROW + get_local_id(0);
  const uint m = get_group_id(1);
  if (i >= SILU_I) return;
  float ga = 0.0f, ua = 0.0f;
  for (uint s = 0; s < SILU_S; ++s) {
    const size_t base = ((size_t)s * M + m) * SILU_N;
    ga += partials[base + i];
    ua += partials[base + SILU_I + i];
  }
  const float g = rf(ga), u = rf(ua);
  h[(size_t)m * SILU_I + i] = rne_bf16(rf(silu_t(g)) * u);
}
#endif  // SILU_I
