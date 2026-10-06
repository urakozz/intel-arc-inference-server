// kol_attn_eager.cl - Kolibri-1's EAGER decode attention (B70_KOLIBRI_ATTN=eager; spec 18 §10.1's
// switch, spec 20c): the reference's bf16 eager chain (tools/oracle/kolibri_ref.py attention() =
// transformers' eager path in bf16) on the device - k2_attn_eager.cl's four launches without K2's
// gate, over a SLIDING window through the ring or a FULL layer's growing cache. k2_attn_eager.cl and
// its binaries are untouched.
//
//   kol_attn_eager_score  (ctrl, attn_q, kv_k, attn_s, stride)       grid (KV_HEADS, TGT)
//   kol_attn_eager_softmax(ctrl, attn_s, stride)                     grid (Q_HEADS, M)
//   kol_attn_eager_pv     (ctrl, attn_s, kv_v, attn_part, stride)    grid (KV_HEADS, TGT)
//   kol_attn_eager_reduce (ctrl, attn_part, attn_out)                grid (Q_HEADS, M)
//     every work-group 128 work-items
//     attn_q     fp32 [M][Q_HEADS][128]           kol_attn_prep's q (bf16 values)
//     kv_k/kv_v  bf16 [rows][KV_HEADS][128]       this layer's ring (rows = RING) or cache
//     attn_s     fp32 [M][Q_HEADS][stride]        row m's scores, then probabilities, at i = p - lo_m
//     attn_part  fp32 [Q_HEADS][TGT][M][130]      words 2.. hold a block's P·V sums
//     attn_out   bf16 [M][Q_HEADS x 128]          o_proj's input
//
// **The keys.** Row m (query position hi = pos + m) sees keys [lo, hi]: lo = max(0, hi - (WINDOW - 1))
// in a sliding layer (513 keys INCLUDING the query: i - 513 < j <= i), 0 in a full one. Key p is
// row p & (RING - 1) of the ring (sliding) or row p of the cache (full). The step's keys are the span
// [lo_0, pos + n_act) cut into TGT blocks of eager_ppw(span) keys (v2's ppw); a row's score index is
// i = p - lo_m - the cached decode pass's row (its cache holds exactly the visible keys), so the
// softmax's 8 lanes take i % 8. tests/kernels/kolibri_ref.h attention_eager is the host twin.
//
// **The op chain** (k2_attn_eager.cl's, its account of what is bitwise against torch holds here):
//   score    s_i = rne(f32(rne(Σ_d q_d·k_pd)) · SCALE)   one fma chain, d ascending (exact products)
//   softmax  torch's fp32 softmax over the row (Sleef exp, reduce_all's 8-lane sum), then bf16
//   P·V      o_d = Σ_i p_i·v_pd: per block an fma chain ascending, the blocks added ascending
//   out      rne(o)                                       (no gate: Kolibri's attention has none)
//
// Known, ulp-level: the reference's PROMPT pass indexes a sliding row from key 0 (masked keys
// included), so a prompt row past position 512 sums its lanes over other indices than this decode
// row does (spec 20 §11 "known deviations").
//
// Plain OpenCL C on purpose (no sub-group operations, no block reads): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "kol_attn_eager: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TGT) || !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(WINDOW) || !defined(RING)
#error "kol_attn_eager: TGT, Q_HEADS, KV_HEADS, WINDOW and RING must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "kol_attn_eager: q-heads must be a multiple of kv-heads"
#endif
#if M < 1 || M > 4
#error "kol_attn_eager: M is 1..4"
#endif
#if WINDOW != 0 && (RING < WINDOW + M || (RING & (RING - 1)) != 0)
#error "kol_attn_eager: a sliding layer's ring is a power of two holding the window and the step's rows"
#endif
#define SLIDING (WINDOW != 0)
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define PART (HD + 2)
#define WG HD
#define NPR (M * GQA)
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
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
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

// Scores: work-item = key (strided over the block), every (row, q head) dot of the kv head.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_eager_score(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                                   __global const ushort* restrict kv_k, __global float* restrict attn_s,
                                   const uint stride) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (n_act == 0) return;
  const uint klo = key_lo(pos), end = pos + n_act;
  const uint s = eager_ppw(end - klo);
  const uint bstart = klo + blk * s;
  if (bstart >= end) return;   // uniform
  const uint bend = min(bstart + s, end);

  __local float qpack[NPR * HD];   // [m][qhl][128]
  for (uint i = lid; i < NPR * HD; i += WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint p = bstart + lid; p < bend; p += WG) {
    __global const ushort* restrict kr = kv_k + (key_row(p) * KV_HEADS + j) * HD;
    float a[NPR];
    for (uint pr = 0; pr < NPR; ++pr) a[pr] = 0.0f;
    for (uint d0 = 0; d0 < HD; d0 += 8) {
      const ushort8 k8 = vload8(0, kr + d0);
      const float kf[8] = {bf16f(k8.s0), bf16f(k8.s1), bf16f(k8.s2), bf16f(k8.s3),
                           bf16f(k8.s4), bf16f(k8.s5), bf16f(k8.s6), bf16f(k8.s7)};
      for (uint pr = 0; pr < NPR; ++pr)
        for (uint t = 0; t < 8; ++t) a[pr] = fma(qpack[pr * HD + d0 + t], kf[t], a[pr]);   // d ascending
    }
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      const uint hi = pos + m, lo = key_lo(hi);
      if (m >= n_act || p > hi || p < lo) continue;
      attn_s[((size_t)m * Q_HEADS + j * GQA + qhl) * stride + (p - lo)] = rf(rf(a[pr]) * SSCALE);
    }
  }
}

// The softmax of one row, in place: scores in, bf16-valued probabilities out.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_eager_softmax(__global const uint* restrict ctrl, __global float* restrict attn_s,
                                     const uint stride) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  if (m >= n_rows(ctrl)) return;
  const uint hi = pos + m;
  const uint len = hi - key_lo(hi) + 1;
  __global float* restrict row = attn_s + ((size_t)m * Q_HEADS + h) * stride;
  __local float red[WG];

  float mx = -INFINITY;
  for (uint i = lid; i < len; i += WG) mx = fmax(mx, row[i]);
  red[lid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint w = WG / 2; w > 0; w >>= 1) {   // max: exact in any order
    if (lid < w) red[lid] = fmax(red[lid], red[lid + w]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float gmx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = lid; i < len; i += WG) row[i] = exp_torch(row[i] - gmx);
  barrier(CLK_GLOBAL_MEM_FENCE);   // this work-group reads its own e back

  if (lid < LANES) {   // reduce_all's lane chains: lane j sums e_j, e_{j+8}, ... ascending
    float acc = 0.0f;
    for (uint i = lid; i < len; i += LANES) acc += row[i];
    red[lid] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sum = ((red[0] + red[4]) + (red[2] + red[6])) + ((red[1] + red[5]) + (red[3] + red[7]));
  const float inv = 1.0f / sum;
  for (uint i = lid; i < len; i += WG) row[i] = rf(row[i] * inv);
}

// P·V over one key block: work-item d = dim d, every (row, q head) of the kv head; the block's sums to
// attn_part words 2.. (fma chains, keys ascending).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_eager_pv(__global const uint* restrict ctrl, __global const float* restrict attn_s,
                                __global const ushort* restrict kv_v, __global float* restrict attn_part,
                                const uint stride) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint d = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (n_act == 0) return;
  const uint klo = key_lo(pos), end = pos + n_act;
  const uint s = eager_ppw(end - klo);
  const uint bstart = klo + blk * s;
  if (bstart >= end) return;   // uniform
  const uint bend = min(bstart + s, end);

  float acc[NPR];
  for (uint pr = 0; pr < NPR; ++pr) acc[pr] = 0.0f;
  for (uint p = bstart; p < bend; ++p) {
    const float v = bf16f(kv_v[(key_row(p) * KV_HEADS + j) * HD + d]);
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      const uint hi = pos + m, lo = key_lo(hi);
      if (m >= n_act || p > hi || p < lo) continue;   // uniform
      acc[pr] = fma(attn_s[((size_t)m * Q_HEADS + j * GQA + qhl) * stride + (p - lo)], v, acc[pr]);
    }
  }
  for (uint pr = 0; pr < NPR; ++pr) {
    const uint m = pr / GQA, qhl = pr % GQA;
    if (m >= n_act) continue;
    attn_part[(((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART + 2 + d] = acc[pr];
  }
}

// Row m's blocks (every block of the step: a block below its window holds an exact 0) added
// ascending, rounded once.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_eager_reduce(__global const uint* restrict ctrl, __global const float* restrict attn_part,
                                    __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (m >= n_act) return;
  const uint klo = key_lo(pos);
  const uint s = eager_ppw(pos + n_act - klo);
  const uint nb = (pos + m + 1 - klo + s - 1) / s;   // the blocks up to row m's own last key
  float o = 0.0f;
  for (uint b = 0; b < nb; ++b) o += attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(o);
}
