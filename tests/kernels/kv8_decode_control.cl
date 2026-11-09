// Frozen int8 decode arithmetic from 800783f2, for 3f bitwise comparisons.
// Only the test/probe load override is conditional; the zero-kind reference uses
// the original source statements, without including the production reader.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#define KV8_DEC 1
#define KV8_PREP 0
#define KV8_GATE 0
#ifndef KV8_PROBE_KIND
#define KV8_PROBE_KIND 0
#endif
// The model's dimensions (model::Qwen35; FA heads as attn.cl bakes them).
#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define OUT_N 6144        /* 24 x 256 */
#define QKV_N 14336       /* q||gate (12288) || k (1024) || v (1024) */
#define K_OFF 12288
#define V_OFF 13312
#define KV8_WG 256        /* every KV8_PREP / KV8_DEC reduce / KV8_GATE work-group: one head */

// The probe's signs (src/common/kv8.h kSignWords).
__constant uint KV8_SIGNS[8] = {0xA197D809u, 0xA6850592u, 0xB205A73Eu, 0x20AFA769u,
                                0x45C2CF64u, 0xBEAEAACBu, 0x75F39ADDu, 0x2385A70Cu};

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// s[j] applied as the sign bit: exact, and the same bits as a multiply by -1.
inline float kv8_flip(float v, uint j) {
  return as_float(as_uint(v) ^ (((KV8_SIGNS[j >> 5] >> (j & 31u)) & 1u) << 31));
}

// binary16 <-> fp32 in integer arithmetic (kv8.h's f32_to_f16_rne / f16f, op for op): no
// cl_khr_fp16 needed, subnormals kept, >= 65504 saturates. A scale is >= 0 and finite.
inline ushort kv8_f16_rne(float f) {
  uint u = as_uint(f);
  const uint sign = (u >> 16) & 0x8000u;
  u &= 0x7FFFFFFFu;
  if (u >= 0x477FE000u) return (ushort)(sign | 0x7BFFu);
  if (u < 0x38800000u) return (ushort)(sign | (uint)rint(as_float(u) * 16777216.0f));
  const uint e = (u >> 23) - 112u;
  const uint m = u & 0x7FFFFFu;
  uint h = (e << 10) | (m >> 13);
  const uint rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
  return (ushort)(sign | h);
}
inline float kv8_f16f(ushort h) {
  const uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0u ? (float)m * 5.9604644775390625e-08f
                          : as_float(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}

#if KV8_PREP || KV8_DEC || KV8_GATE
// The 256-point FWHT across a 256-work-item work-group, work-item i owning element i,
// staged through `buf` (256 floats of SLM). Every work-item must call it (barriers). It
// starts and ends on a barrier-safe footing: the first write follows the caller's last
// read only through the trailing barrier of a previous call or the caller's own barrier.
inline float kv8_fwht(float v, uint i, __local float* buf) {
  for (uint h = 1u; h < (uint)HD; h <<= 1) {
    buf[i] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float o = buf[i ^ h];
    v = (i & h) ? (o - v) : (v + o);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return v;
}
inline float kv8_rotate(float x, uint i, __local float* buf) {
  return kv8_flip(kv8_fwht(x, i, buf), i) * 0.0625f;
}
inline float kv8_unrotate(float y, uint i, __local float* buf) {
  return kv8_fwht(kv8_flip(y, i), i, buf) * 0.0625f;
}
#endif

#if KV8_DEC
// ---------------------------------------------------------------------------
// attn_decode_v2_kv8 / attn_reduce_v2_kv8 - attn_v2.cl's pair (its header is the
// contract: per-row stride ppw(L), grid (4, TGT), partials [24][TGT][M][258], verify rows
// bitwise the M = 1 steps), with the loads replaced and the reduce's un-rotation added.
//
//   attn_decode_v2_kv8(ctrl, attn_q, kv_k, k_scale, kv_v, v_scale, attn_part)
//   attn_reduce_v2_kv8(ctrl, attn_part, attn_gate, attn_out)
//
// K: sub-group s owns position p; lane l's 16 elements are dims l + 16 t (two uchar8
//    sub-group block reads, attn_v2.cl's order), dequantised as float(q8) * f16f(scale[p]).
// V: work-item d's 16 values of the wave, dim d of positions p0 .. p0 + 15, plain loads
//    (lanes read 16 consecutive bytes), zero past `last` (the 2D read's surface height in
//    the bf16 kernel), dequantised the same way.
// The online softmax and its orders are attn_v2.cl's, unchanged.
// ---------------------------------------------------------------------------
#ifndef M
#define M 1
#endif
#ifndef CTRL_POS
#error "kv8 dec: CTRL_POS must be defined"
#endif
#ifndef CTRL_NACT
#error "kv8 dec: CTRL_NACT must be defined"
#endif
#ifndef TGT
#error "kv8 dec: TGT must be defined (runtime::DecodeScratch::kAttnV2Blocks)"
#endif
#ifndef BATCH_ROWS
#if M < 1 || M > 4
#error "kv8 dec: M is 1..4"
#endif
#elif M < 1 || M > 8
#error "kv8 dec BATCH_ROWS: M is 1..8"
#endif
#define WAVE_P 16
#define SG 16
#define PER_LANE 16
#define PART 258
#define NPR (M * GQA)
#define SSCALE 0.0625f       /* attn_v2's shipped EXP2 = 0 form: exp and 1/16 */
#define EXPF(x) exp(x)

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

#if KV8_PROBE_KIND
#include "../../tools/probe/kv8_decode_loads.cl"
#else
// K stays int8 in registers with its row's scale beside it (16 bytes + one float per lane,
// where 16 dequantised floats would be 16 GRF); the score dot dequantises at use, and since
// float(q8) * scale is exact the operand is the same value either way.
inline void load_k8(__global const char* restrict kv_k, __global const ushort* restrict k_scale,
                    uint p, uint j, bool ok, char* kq, float* ksc) {
  if (ok) {   // uniform within the sub-group
    const size_t r = (size_t)p * KV_HEADS + j;
    __global const uchar* row = (__global const uchar*)(kv_k + r * HD);
    *ksc = kv8_f16f(k_scale[r]);
    const uchar8 a = intel_sub_group_block_read_uc8(row);
    const uchar8 b = intel_sub_group_block_read_uc8(row + 128);
    kq[0] = as_char(a.s0);  kq[1] = as_char(a.s1);  kq[2] = as_char(a.s2);  kq[3] = as_char(a.s3);
    kq[4] = as_char(a.s4);  kq[5] = as_char(a.s5);  kq[6] = as_char(a.s6);  kq[7] = as_char(a.s7);
    kq[8] = as_char(b.s0);  kq[9] = as_char(b.s1);  kq[10] = as_char(b.s2); kq[11] = as_char(b.s3);
    kq[12] = as_char(b.s4); kq[13] = as_char(b.s5); kq[14] = as_char(b.s6); kq[15] = as_char(b.s7);
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kq[t] = 0;
    *ksc = 0.0f;
  }
}

inline void load_v8(__global const char* restrict kv_v, __global const ushort* restrict v_scale,
                    uint p0, uint j, uint lid, uint last, float* vreg) {
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint p = p0 + s;
    if (p <= last) {
      const size_t r = (size_t)p * KV_HEADS + j;
      vreg[s] = (float)kv_v[r * HD + lid] * kv8_f16f(v_scale[r]);
    } else {
      vreg[s] = 0.0f;   // never loaded: an unwritten slot cannot turn 0 x into a NaN
    }
  }
}

#endif

__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_q,
#ifndef BATCH_ROWS
                                 __global const char* restrict kv_k,
                                 __global const ushort* restrict k_scale,
                                 __global const char* restrict kv_v,
                                 __global const ushort* restrict v_scale,
#else
                                 __global const char* restrict kv_k_all,
                                 __global const ushort* restrict k_scale_all,
                                 __global const char* restrict kv_v_all,
                                 __global const ushort* restrict v_scale_all,
#endif
                                 __global float* restrict attn_part
#ifdef BATCH_ROWS
                                 , __global const uint* restrict rows
#endif
                                 ) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;

#ifndef BATCH_ROWS
  const uint pos = ctrl[CTRL_POS];
#endif
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

#ifndef BATCH_ROWS
  const uint s0 = v2_ppw(pos + 1);
  const uint s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
  if (!live0 && !live1) return;
#else
  uint rp[M], rs[M], rb[M];   // attn_v2.cl's BATCH_ROWS build, line for line
  bool live = false;
  for (uint m = 0; m < M; ++m) {
    rp[m] = m < n_act ? rows[m * BROW_WORDS + BROW_POS] : 0u;
    rs[m] = m < n_act ? rows[m * BROW_WORDS + BROW_SLOT] : 0u;
    rb[m] = m < n_act ? rows[m * BROW_WORDS + BROW_KVB] : 0u;
    if (m < n_act && blk * v2_ppw(rp[m] + 1) <= rp[m]) live = true;
  }
  if (!live) return;
#endif

  __local float qpack[NPR * HD];
  __local float sc_x[2][NPR][WAVE_P];
  for (uint i = lid; i < NPR * HD; i += KV8_WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx[NPR], sm[NPR], acc[NPR];
  for (uint q = 0; q < NPR; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
  uint wc = 0;

#ifndef BATCH_ROWS
  for (uint g = 0; g < 2; ++g) {
    const uint mlo = g ? msplit : 0u, mhi = g ? n_act : msplit;
    if (mlo >= mhi) continue;
    const uint s = g ? s1 : s0;
    const uint end = pos + mhi;
#else
  for (uint mlo = 0, mhi = 0; mlo < n_act; mlo = mhi) {
    const uint s = v2_ppw(rp[mlo] + 1);
    mhi = mlo + 1;
    while (mhi < n_act && rs[mhi] == rs[mlo] && v2_ppw(rp[mhi] + 1) == s) ++mhi;
    const uint end = rp[mhi - 1] + 1;
    const size_t r0 = (size_t)rb[mlo] * KV_HEADS;   // the slot's first (position, kv head) row
    __global const char* restrict kv_k = kv_k_all + r0 * HD;
    __global const ushort* restrict k_scale = k_scale_all + r0;
    __global const char* restrict kv_v = kv_v_all + r0 * HD;
    __global const ushort* restrict v_scale = v_scale_all + r0;
#endif
    const uint bstart = blk * s;
    if (bstart >= end) continue;
    const uint bend = min(bstart + s, end);
    const uint last = end - 1;
    const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

    char kq[PER_LANE];
    float ksc, vreg[WAVE_P];
    load_k8(kv_k, k_scale, bstart + sgid, j, bstart + sgid <= last, kq, &ksc);
    load_v8(kv_v, v_scale, bstart, j, lid, last, vreg);
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;
        float qv[PER_LANE];
        const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD]);
        const uint8 qb = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD + 128]);
        qv[0] = as_float(qa.s0); qv[1] = as_float(qa.s1); qv[2] = as_float(qa.s2);
        qv[3] = as_float(qa.s3); qv[4] = as_float(qa.s4); qv[5] = as_float(qa.s5);
        qv[6] = as_float(qa.s6); qv[7] = as_float(qa.s7);
        qv[8] = as_float(qb.s0); qv[9] = as_float(qb.s1); qv[10] = as_float(qb.s2);
        qv[11] = as_float(qb.s3); qv[12] = as_float(qb.s4); qv[13] = as_float(qb.s5);
        qv[14] = as_float(qb.s6); qv[15] = as_float(qb.s7);
        float a = 0.0f;
        for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], (float)kq[t] * ksc, a);   // exact operand
        for (uint stride = SG / 2; stride > 0; stride >>= 1)
          a += intel_sub_group_shuffle_down(a, a, stride);
        if (lane == 0) sc_x[buf][pr][sgid] = a;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;
#ifndef BATCH_ROWS
        const uint bound = pos + m;
#else
        const uint bound = rp[m];
#endif
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
        const float nmx = fmax(mx[pr], sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = EXPF(mx[pr] - nmx);
          wl = EXPF(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f, tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) {
          const float ws = intel_sub_group_shuffle(wl, s2);
          ssum += ws;
          tsum = fma(ws, vreg[s2], tsum);
        }
        if (nmx > -INFINITY) {
          sm[pr] = fma(sm[pr], resc, ssum);
          mx[pr] = nmx;
        }
        acc[pr] = fma(acc[pr], resc, tsum);
      }
      if (w + 1 < nwaves) {
        load_k8(kv_k, k_scale, p0 + WAVE_P + sgid, j, p0 + WAVE_P + sgid <= last, kq, &ksc);
        load_v8(kv_v, v_scale, p0 + WAVE_P, j, lid, last, vreg);
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out =
          attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + lid] = acc[pr];
      if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
    }
  }
}

// attn_reduce_v2's merge, then o = acc / sm un-rotated across the work-group (work-item
// d = dim d), then attn.cl's gated chain on the un-rotated value.
__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void attn_reduce_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_part,
                                 __global const float* restrict attn_gate,
                                 __global ushort* restrict attn_out
#ifdef BATCH_ROWS
                                 , __global const uint* restrict rows
#endif
                                 ) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  __local float xch[HD];
#ifndef BATCH_ROWS
  const uint pos = ctrl[CTRL_POS];
#endif
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
#ifndef BATCH_ROWS
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
#else
  const uint pm = rows[m * BROW_WORDS + BROW_POS];
  uint nb = pm / v2_ppw(pm + 1) + 1;
#endif
  if (nb > TGT) nb = TGT;
  for (uint b = d; b < nb; b += KV8_WG) {
    __global const float* restrict p = attn_part + (((size_t)h * TGT + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = EXPF(mx - nmx);
    const float bs = EXPF(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const float o = kv8_unrotate(acc / sm, d, xch);
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * sigmoid_f32(g));
}
#endif  // KV8_DEC

// Scale widening is tested independently over the entire finite fp16 domain.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kv8_decode_scale_probe(__global const ushort* p, __global float* out, uint count) {
  const uint i = get_global_id(0);
  if (i < count) {
#if KV8_PROBE_KIND
    out[i] = probe_scale(p, i);
#else
    out[i] = kv8_f16f(p[i]);
#endif
  }
}
