// kol_prep.cl - spec 20c: Kolibri-1's between-GEMV kernels. Three families, each compiled only when
// its defines are given (one binary per family and shape):
//
//   kol_norm_finish(sumsq, resid, w, x)                    NORM_G   grid (NORM_WGS, M), WG 256
//   kol_post_add(sumsq_a, a, w, resid, sumsq_out)          POST_G   grid (POST_G, M),   WG 256
//   kol_attn_prep(ctrl, partials, qkn, rope, attn_q,       QKV_N    grid (Q + 2 KV heads, M), WG 128
//                 kv_k, kv_v)
//
// The reference is tools/oracle/kolibri_ref.py (spec 20a), op for op; tests/kernels/kolibri_ref.h
// repeats every chain below - edit the two together. The rounding discipline is prep.cl's: a linear's
// split-K partials summed in fp32 and rounded to bf16 once, every elementwise op widened to fp32 and
// rounded. Kolibri's norm is PLAIN w with x̂ rounded first: `w * bf16(x * rsqrt(mean(x²) + eps))` -
// two roundings, not prep.cl's one.
//
// **The sandwich** (spec 20 §1, plan 20c Review Focus 1): x += post_attn_norm(attn(...)) and
// x += post_ffn_norm(moe(...)) - the post norms normalise the sub-block's OWN bf16 output before the
// add. So a sub-block's output is never folded into the residual directly: prep.cl's prep_res_fold
// with ZERO_RESID writes it (o_proj's partials) or re-reads it (the MoE output, SP0) into its own row
// and reduces its Σ², and kol_post_add normalises it, adds, and reduces the new residual's Σ² for the
// next norm.
//
// **No sub-group function anywhere in this file**, so it also builds as OpenCL 1.2 for an indicative
// run on the Mac's GPU (tools/mac/clrun/kolibri_run.cc).
#ifndef M
#define M 1
#endif
#ifndef K
#define K 2560              /* hidden */
#endif
#define KOL_EPS 1e-6f

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (prep.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
// One element of Kolibri's RMSNorm: rne(f32(rne(x · rstd)) · w).
inline ushort kol_norm_elem(float x, float rstd, float w) { return rne_bf16(rf(x * rstd) * w); }

// ---------------------------------------------------------------------------------------
// kol_norm_finish - stage B after prep.cl's prep_res_fold (stage A, unchanged):
//   rstd        = 1 / sqrt(Σ_{g ascending} sumsq[g][m] / K + 1e-6)      never rsqrt
//   x[m][k]     = rne(f32(rne(f32(resid[m][k]) · rstd)) · w[k])
#ifdef NORM_G
#ifndef NORM_WGS
#error "kol_norm_finish: NORM_WGS must be defined"
#endif
#define WG_NORM 256
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void kol_norm_finish(__global const float* restrict sumsq, __global const ushort* restrict resid,
                              __global const float* restrict norm_w, __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  float total = 0.f;
  for (uint g = 0; g < NORM_G; ++g) total += sumsq[(size_t)g * M + m];
  const float rstd = 1.0f / sqrt(total / (float)K + KOL_EPS);
  const __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = w * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_NORM)
    x_out[(size_t)m * K + k] = kol_norm_elem(bf16f(rp[k]), rstd, norm_w[k]);
}
#endif  // NORM_G

// ---------------------------------------------------------------------------------------
// kol_post_add - the sandwich's post norm and the residual add, fused with the next norm's stage A.
// `a` is the sub-block's bf16 output [M][K] and `sumsq_a` its POST_G chunk sums (prep_res_fold
// ..._Z or ..._SP0 wrote both); work-group g owns chunk g (K / POST_G columns):
//   rstd_a = 1 / sqrt(Σ_{g ascending} sumsq_a[g][m] / K + 1e-6)
//   n      = rne(f32(rne(f32(a[m][k]) · rstd_a)) · w[k])                post_*_norm(sub-block)
//   r      = rne(f32(resid[m][k]) + f32(n));  resid[m][k] = r              x += ...
//   sumsq_out[g][m] = prep_res_fold's chunk tree over f32(r)²            (same chunking, same tree)
#ifdef POST_G
#define WG_POST 256
#define POST_CHUNK ((K + POST_G - 1) / POST_G)
__attribute__((reqd_work_group_size(WG_POST, 1, 1)))
__kernel void kol_post_add(__global const float* restrict sumsq_a, __global const ushort* restrict a,
                           __global const float* restrict norm_w, __global ushort* restrict resid,
                           __global float* restrict sumsq_out) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_POST];
  float total = 0.f;
  for (uint q = 0; q < POST_G; ++q) total += sumsq_a[(size_t)q * M + m];
  const float rstd = 1.0f / sqrt(total / (float)K + KOL_EPS);
  __global ushort* rp = resid + (size_t)m * K;
  const __global ushort* ap = a + (size_t)m * K;
  const uint k0 = g * POST_CHUNK;
  uint k1 = k0 + POST_CHUNK;
  if (k1 > K) k1 = K;
  float acc2 = 0.f;
  for (uint k = k0 + lid; k < k1; k += WG_POST) {
    const ushort n = kol_norm_elem(bf16f(ap[k]), rstd, norm_w[k]);
    const ushort r = rne_bf16(bf16f(rp[k]) + bf16f(n));
    rp[k] = r;
    const float v = bf16f(r);
    acc2 = fma(v, v, acc2);
  }
  red[lid] = acc2;   // a lane with no element contributes an exact 0.0f (prep_res_fold's tree)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_POST / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sumsq_out[(size_t)g * M + m] = red[0];
}
#endif  // POST_G

// ---------------------------------------------------------------------------------------
// kol_attn_prep - q / k / v out of the fused q||k||v GEMV: the per-head RMSNorm of q and k, RoPE in the
// sliding layers only (the full layers are NoPE), the cache writes. Grid (Q_HEADS + 2 KV_HEADS, M),
// work-group HD (128), work-item d owns dim d of one head: work-groups 0..Q-1 are q heads, Q.. Q+KV-1
// k heads, the last KV the v heads.
//
//   x_b   = rne(Σ_{s < QKV_S} partials[s][m][col])                    the linear's bf16 output
//   rstd  = 1 / sqrt(tree_d(fma(x_b, x_b, 0)) / 128 + 1e-6)              q, k: per head (128-lane tree)
//   y     = rne(f32(rne(x_b · rstd)) · w_qk[d])                          q_norm / k_norm, plain w
//   SLIDING: y = rne(f32(rne(y · cos)) + f32(rne(rot(y) · sin)))        rot(y)[d] = d < 64 ? -y[d+64] : y[d-64]
//   q -> attn_q[m][h][d] (fp32 of a bf16 value); k -> kv_k[row][j][d]; v = x_b -> kv_v[row][j][d]
//   row = SLIDING ? (pos + m) & (RING - 1) : pos + m                     the ring / the growing cache
//
// `qkn` is the layer's q_norm [128] then k_norm [128] (fp32, loader::KolLayer::norms + 4 hidden);
// `rope` the RoPE table [max_len][2][64] (read only when SLIDING).
#ifdef QKV_N
#ifndef QKV_S
#error "kol_attn_prep: QKV_S (the fused GEMV's split-K) must be defined"
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "kol_attn_prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HD) || !defined(SLIDING) || !defined(RING)
#error "kol_attn_prep: Q_HEADS, KV_HEADS, HD, SLIDING and RING must be defined"
#endif
#if QKV_N != (Q_HEADS + 2 * KV_HEADS) * HD
#error "kol_attn_prep: q||k||v is (Q_HEADS + 2 KV_HEADS) x HD wide"
#endif
#if SLIDING && (RING == 0 || (RING & (RING - 1)) != 0)
#error "kol_attn_prep: the sliding ring is a power of two"
#endif
#define HALF (HD / 2)
inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
}
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void kol_attn_prep(__global const uint* restrict ctrl, __global const float* restrict partials,
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
  const float xb = rf(qkv_sum(partials, m, col));
  const uint p = pos + m;
#if SLIDING
  const size_t row = (size_t)(p & (RING - 1));
#else
  const size_t row = (size_t)p;
#endif
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
  const float rstd = 1.0f / sqrt(red[0] / (float)HD + KOL_EPS);
  const float y = bf16f(kol_norm_elem(xb, rstd, qkn[(kind == 0 ? 0 : HD) + d]));
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
#endif  // QKV_N
