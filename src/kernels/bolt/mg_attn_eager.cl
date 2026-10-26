// mg_attn_eager.cl - spec 23c: Muse-Glimmer-30B's EAGER decode attention (B70_MUSE_ATTN=eager; spec 18 §10.1's
// switch): the reference's bf16 eager chain (transformers 5.19.0 eager_attention_forward in bf16, then the output
// gate, M:263-269 / M:357-358) on the device, over a SLIDING window of 2048 keys through the 4096-slot ring or a FULL
// (NoPE) layer's growing cache. ONE launch a layer (k2 / Kolibri's eager forms are four): eager exists for the A/B
// against the reference's rounding points, not for speed.
//
//   mg_attn_eager(ctrl, attn_q, kv_k, kv_v, gate, attn_s, attn_out, stride)     grid (Q_HEADS, M), WG 128
//     attn_q     fp32 [M][Q_HEADS][128]           mg_attn_prep's q (bf16 values)
//     kv_k/kv_v  bf16 [rows][KV_HEADS][128]       this layer's ring (rows = RING) or cache
//     gate       bf16 [M][Q_HEADS x 128]          mg_attn_prep's bf16(sigmoid(g))
//     attn_s     fp32 [M][Q_HEADS][stride]        row m's scores, then probabilities, at i = p - lo_m
//     attn_out   bf16 [M][Q_HEADS x 128]          o_proj's input
//
// **The keys.** Row m (query position hi = pos + m) sees keys [lo, hi]: lo = max(0, hi - (WINDOW - 1)) in a sliding
// layer (2048 keys INCLUDING the query: i - 2048 < j <= i, MU:93-102), 0 in a full one. Key p is row p & (RING - 1)
// of the ring (sliding) or row p of the cache (full). A row's score index is i = p - lo (the cached decode pass's
// row: its cache holds exactly the visible keys), so the softmax's 8 lanes take i % 8.
//
// **The op chain** (k2_attn_eager.cl's, its account of what is bitwise against torch holds here; tests/kernels/
// muse_ref.h attention_eager is the host twin, itself bitwise to transformers' eager_attention_forward on the
// fixture - scores, probabilities and outputs):
//   score    s_i = rne(f32(rne(Σ_d q_d·k_pd)) · SCALE)   one fma chain, d ascending (exact products)
//   softmax  torch's fp32 softmax over the row (Sleef exp, reduce_all's 8-lane sum, one reciprocal), then bf16
//   P·V      o_d = Σ_i p_i·v_pd in blocks of eager_ppw(span) keys from i = 0 (v2's ppw over the step's span from
//            row 0's lo): per block an fma chain ascending, the blocks added ascending
//   out      rne(f32(rne(o)) · f32(gate))                 the output gate (M:358)
//
// Plain OpenCL C on purpose (no sub-group operations, no block reads): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "mg_attn_eager: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TGT) || !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(WINDOW) || !defined(RING)
#error "mg_attn_eager: TGT, Q_HEADS, KV_HEADS, WINDOW and RING must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "mg_attn_eager: q-heads must be a multiple of kv-heads"
#endif
#if M < 1 || M > 16
#error "mg_attn_eager: M is 1..16"
#endif
#if WINDOW != 0 && (RING < WINDOW + M || (RING & (RING - 1)) != 0)
#error "mg_attn_eager: a sliding layer's ring is a power of two holding the window and the step's rows"
#endif
#define SLIDING (WINDOW != 0)
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define WG HD
#define SSCALE 0.08838834764831845f   /* float(128 ** -0.5): torch casts the scalar to fp32 */
#define LANES 8                       /* reduce_all's Vectorized<float> width at AVX2 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }

// Sleef_expf8_u10 (sleefsimdsp.c xexpf, AVX2 + FMA) - k2_ref::exp_torch / k2_attn_eager.cl's, the same lines.
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

inline uint eager_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}
// The first visible key of the query at position `hi`.
inline uint key_lo(uint hi) {
#if SLIDING
  return hi + 1 > WINDOW ? hi + 1 - WINDOW : 0u;
#else
  (void)hi;
  return 0u;
#endif
}
inline size_t key_row(uint p) {
#if SLIDING
  return (size_t)(p & (RING - 1));
#else
  return (size_t)p;
#endif
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void mg_attn_eager(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                            __global const ushort* restrict kv_k, __global const ushort* restrict kv_v,
                            __global const ushort* restrict gate, __global float* restrict attn_s,
                            __global ushort* restrict attn_out, const uint stride) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint j = h / GQA;
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform
  const uint hi = pos + m, lo = key_lo(hi);
  const uint len = hi - lo + 1;
  __global float* restrict row = attn_s + ((size_t)m * Q_HEADS + h) * stride;
  __local float qv[HD];
  __local float red[WG];
  qv[lid] = attn_q[((size_t)m * Q_HEADS + h) * HD + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // scores: work-item = key, one fma chain over d ascending
  float mx = -INFINITY;
  for (uint i = lid; i < len; i += WG) {
    __global const ushort* restrict kr = kv_k + (key_row(lo + i) * KV_HEADS + j) * HD;
    float a = 0.0f;
    for (uint d = 0; d < HD; ++d) a = fma(qv[d], bf16f(kr[d]), a);
    const float s = rf(rf(a) * SSCALE);
    row[i] = s;
    mx = fmax(mx, s);
  }
  red[lid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint w = WG / 2; w > 0; w >>= 1) {   // max: exact in any order
    if (lid < w) red[lid] = fmax(red[lid], red[lid + w]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float gmx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < len; i += WG) row[i] = exp_torch(row[i] - gmx);   // each work-item its own keys
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);   // this work-group reads its own e back
  if (lid < LANES) {   // reduce_all's lane chains: lane j sums e_j, e_{j+8}, ... ascending
    float acc = 0.0f;
    for (uint i = lid; i < len; i += LANES) acc += row[i];
    red[lid] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sum = ((red[0] + red[4]) + (red[2] + red[6])) + ((red[1] + red[5]) + (red[3] + red[7]));
  const float inv = 1.0f / sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < len; i += WG) row[i] = rf(row[i] * inv);
  barrier(CLK_GLOBAL_MEM_FENCE);

  // P·V: work-item d = dim d; blocks of eager_ppw(the step's span from row 0's lo) from i = 0, ascending
  const uint blk = eager_ppw(pos + n_act - key_lo(pos));
  float o = 0.0f;
  for (uint b0 = 0; b0 < len; b0 += blk) {
    const uint b1 = min(len, b0 + blk);
    float acc = 0.0f;
    for (uint i = b0; i < b1; ++i) acc = fma(row[i], bf16f(kv_v[(key_row(lo + i) * KV_HEADS + j) * HD + lid]), acc);
    o += acc;
  }
  const size_t oi = (size_t)m * OUT_N + (size_t)h * HD + lid;
  attn_out[oi] = rne_bf16(bf16f(rne_bf16(o)) * bf16f(gate[oi]));
}
