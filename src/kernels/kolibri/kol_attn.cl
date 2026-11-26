// kol_attn.cl - spec 20c: Kolibri-1's decode attention, spec 10's v2 (src/kernels/attn_v2.cl)
// carried to head_dim 128, here at GQA 12 without a gate, over a SLIDING window through
// the ring (WINDOW 513, RING 4096) or a FULL (NoPE) layer's growing cache (WINDOW 0). attn_v2.cl is
// untouched.
//
//   kol_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)    grid (KV_HEADS, TGT), WG 128
//   kol_attn_reduce(ctrl, attn_part, attn_out)              grid (Q_HEADS, M),    WG 128
//     attn_q     fp32 [M][Q_HEADS][128]     kol_attn_prep's q (bf16 values)
//     kv_k/kv_v  bf16 [rows][KV_HEADS][128] this layer's ring (rows = RING) or cache (rows = max_len)
//     attn_part  fp32 [Q_HEADS][TGT][M][130]   {mx, sm, acc[128]} per block
//     attn_out   bf16 [M][Q_HEADS x 128]    o_proj's input
//
// **What is v2's at head_dim 128, unchanged:** a work-group of 128 = 8 sub-groups x 16 lanes, work-item d
// owning dim d of every q head of the kv head; waves of 16 positions, sub-group s owning two; a K row
// one sub-group block read; the score dot 8 fmas per lane then the 16-lane shuffle tree; the online
// softmax in fp32 with `exp`; the ascending orders of the sums and of the block merge; the V 2D block
// read; a position past row m's bound scores -INF, weight exactly 0.
//
// **What is Kolibri's:**
//   * GQA 12 (NPR = M x 12 (q, row) pairs per wave); no gate - attn_out = rne(acc / sm).
//   * the keys of row m: [lo_m, hi_m], hi_m = pos + m, lo_m = hi_m - 512 (clamped at 0) in a sliding
//     layer, 0 in a full one; a key below lo_m also scores -INF.
//   * FULL: the blocks are v2's - ppw(L) over L = pos + m + 1 keys from 0, rows = positions.
//   * SLIDING: the blocks are ABSOLUTE 64-key blocks [base + 64 b, + 64) from base = lo_0 rounded
//     down to 64 (at most 10 blocks for 513 + 3 keys, <= TGT); key p is ring row p & (RING - 1).
//     RING is a multiple of 64, so a 64-aligned block - and every 16-key wave in it - is 16 / 64
//     contiguous ring rows: no 2D read ever crosses the ring's end (plan 20c Review Focus 2).
//   * the reduce merges only blocks that saw a key of the row (bmx > -INF): a block wholly below
//     lo_m holds {-INF, 0, 0}.
//
// Flash keeps the probabilities in fp32 (one rounding at the end); the reference rounds scores and
// probabilities to bf16 (eager, kol_attn_eager.cl behind B70_KOLIBRI_ATTN=eager). Card only:
// sub-group shuffles, reductions and 2D block reads.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// KV8 (PERF-102, `--kv-cache int8`; undefined on every other command line): a FULL layer's int8 cache (kv8_scheme.h at
// head_dim 128, written by ink_prep.cl's INK_KV8 build): kv_k / kv_v int8 [rows][KV_HEADS][128] and their fp16 scales
// k_scale / v_scale [rows][KV_HEADS] after them in the argument list. attn_q is the ROTATED q. K and V are read exactly
// (q8 x scale in fp32, kv8.cl's decode rule) into the same arithmetic as the bf16 build's values; the reduce un-rotates
// each head's output before the gate. Every other value and order is the build without it.
#ifndef KV8
#define KV8 0
#endif
#if KV8
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "kol_attn: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TGT) || !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(WINDOW) || !defined(RING)
#error "kol_attn: TGT, Q_HEADS, KV_HEADS, WINDOW and RING must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "kol_attn: q-heads must be a multiple of kv-heads"
#endif
#if M < 1 || (!REL_BIAS && M > 4) || (REL_BIAS && M > 8)
#error "kol_attn: unsupported row count"
#endif
#if REL_BIAS && !defined(REL_DIM)
#error "kol_attn: REL_DIM must accompany the relative bias extent"
#endif
#define SLIDING (WINDOW != 0)
// ATTN_SPLIT (Inkling's verify rows, M > 1): a pair's wave weights (the bias, the max, the exps, their sum) computed by
// one sub-group - pair pr by sub-group pr % NSG - into local memory, not by all eight; each sub-group then takes its
// dims' P V from them. The same values in the same order: bitwise the unsplit form.
#ifndef ATTN_SPLIT
#define ATTN_SPLIT 0
#endif
#if KV8 && (SLIDING || !GATE)
#error "kol_attn KV8: the full layers' cache only (the rings stay bf16), with the output gate (Inkling's form)"
#endif
#if SLIDING && ((RING % 64) != 0 || RING < WINDOW + M || (WINDOW + M + 63) / 64 + 1 > TGT)
#error "kol_attn: the ring is whole 64-key blocks holding the window, and the window's blocks fit TGT"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define WAVE_P 16
#define SG 16
#define NSG (HD / SG)          /* 8 sub-groups */
#define PPS (WAVE_P / NSG)     /* 2 positions of a wave per sub-group */
#define PER_LANE (HD / SG)     /* 8 elements of a 128-dim dot per lane */
#define PART (HD + 2)
#define WG HD
#define NPR (M * GQA)
#define SSCALE 0.08838834764831845f   /* 1 / sqrt(128) */

#include "bf16.h"
#if KV8
#define KV8_N HD
#include "kv8_scheme.h"
#define KV_T char
#define KREG_T float
#define KVAL(x) (x)
#else
#define KV_T ushort
#define KREG_T ushort
#define KVAL(x) bf16f(x)
#endif
#if KV8
#define KV8_ARG(x) x,
#define KV8_VARG(x) x, v4, vsc,
#define KV8_VDECL uint v4[8]; float vsc[2];
#else
#define KV8_ARG(x)
#define KV8_VARG(x)
#define KV8_VDECL
#endif

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}
inline uint key_lo(uint hi) {
#if SLIDING
  return hi + 1 > WINDOW ? hi + 1 - WINDOW : 0u;
#else
  (void)hi;
  return 0u;
#endif
}
inline uint key_row(uint p) {
#if SLIDING
  return p & (RING - 1);
#else
  return p;
#endif
}

// One K row's 8 elements for this lane: dim lane + 16 t, t ascending.
#if KV8
inline void load_k(__global const char* restrict kv_k, __global const ushort* restrict k_scale, uint p, uint j, bool ok,
                   float* kreg) {
  if (ok) {   // uniform within the sub-group
    const uchar8 a = intel_sub_group_block_read_uc8((__global const uchar*)(kv_k + ((size_t)p * KV_HEADS + j) * HD));
    const float s = vload_half((size_t)p * KV_HEADS + j, (__global const half*)k_scale);   // exact (kv8_f16f's value)
    kreg[0] = (float)as_char(a.s0) * s; kreg[1] = (float)as_char(a.s1) * s; kreg[2] = (float)as_char(a.s2) * s;
    kreg[3] = (float)as_char(a.s3) * s; kreg[4] = (float)as_char(a.s4) * s; kreg[5] = (float)as_char(a.s5) * s;
    kreg[6] = (float)as_char(a.s6) * s; kreg[7] = (float)as_char(a.s7) * s;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = 0.0f;
  }
}
#else
inline void load_k(__global const ushort* restrict kv_k, uint p, uint j, bool ok, ushort* kreg) {
  if (ok) {   // uniform within the sub-group: p is the sub-group's position
    const ushort8 a = intel_sub_group_block_read_us8(kv_k + ((size_t)key_row(p) * KV_HEADS + j) * HD);
    kreg[0] = a.s0; kreg[1] = a.s1; kreg[2] = a.s2; kreg[3] = a.s3;
    kreg[4] = a.s4; kreg[5] = a.s5; kreg[6] = a.s6; kreg[7] = a.s7;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = (ushort)0;
  }
}
#endif

// The wave's 16 V values of dim `lid` (sub-group sgid holds dims 16 sgid .. +15), position p0 + s at
// register s; outside [first, last] 0. p0 is 16-aligned, so the 16 rows are contiguous (in the ring:
// RING is whole 64-row blocks). The surface: the ring's RING rows, or the cache's last + 1. KV8: one block read of 16
// codes a position, times its scale.
#if KV8
// Two waves (32 positions, p0 32-aligned at the even wave) of this lane's dim in one transform read - uint z holds
// positions 4 z .. 4 z + 3 - and their 32 scales one a lane (vsc[h]: position p0 + 16 h + lane), loaded at the even
// wave; the odd wave reads the second half. The values are the one-position reader's: q8 x f16(scale), 0 past last.
inline void load_v(__global const char* restrict kv_v, __global const ushort* restrict v_scale, __private uint* v4, __private float* vsc,
                   uint p0, uint j, uint sgid, uint first, uint last, float* vreg) {
  (void)first;   // 0: a full layer
  const uint odd = (p0 >> 4) & 1u, lane = get_sub_group_local_id();
  if (odd == 0u) {   // uniform
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((__global void*)(kv_v + (size_t)j * HD), HD, (int)last + 1,
                                                        KV_HEADS * HD, (int2)((int)(SG * sgid), (int)p0), v4);
    for (uint h = 0; h < 2u; ++h) {
      const uint p = p0 + 16u * h + lane;
      vsc[h] = p <= last ? vload_half((size_t)p * KV_HEADS + j, (__global const half*)v_scale) : 0.0f;
    }
  }
  const float sc = odd ? vsc[1] : vsc[0];
#pragma unroll
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint w = odd ? v4[4u + (s >> 2)] : v4[s >> 2];
    vreg[s] = (float)as_char((uchar)(w >> (8u * (s & 3u)))) * intel_sub_group_shuffle(sc, s);
  }
}
#else
inline void load_v(__global const ushort* restrict kv_v, uint p0, uint j, uint sgid, uint first, uint last,
                   ushort* vreg) {
  ushort vb[WAVE_P];
#if SLIDING
  const int height = RING;
#else
  const int height = (int)last + 1;
#endif
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD), HD * 2, height,
                                             KV_HEADS * HD * 2, (int2)((int)(SG * sgid), (int)key_row(p0)), vb);
  for (uint s = 0; s < WAVE_P; ++s) vreg[s] = (p0 + s <= last && p0 + s >= first) ? vb[s] : (ushort)0;
}
#endif

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void kol_attn_decode(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                              __global const KV_T* restrict kv_k, __global const KV_T* restrict kv_v,
#if KV8
                              __global const ushort* restrict k_scale, __global const ushort* restrict v_scale,
#endif
                              __global float* restrict attn_part
#if REL_BIAS
                              , __global const float* restrict relative_q,
                                __global const float* restrict relative_table
#endif
                              ) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

#if SLIDING
  // One group of rows, absolute 64-key blocks from base.
  const uint base = key_lo(pos) & ~63u;
  const uint s0 = 64u, s1 = 64u;
  const uint msplit = n_act;
  const bool live0 = base + blk * 64u < pos + n_act;
  const bool live1 = false;
#else
  const uint base = 0u;
  const uint s0 = v2_ppw(pos + 1);
  const uint s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
#endif
  if (!live0 && !live1) return;

  __local float qpack[NPR * HD];              // [m][qhl][128]
  __local float sc_x[2][NPR][WAVE_P];         // the wave's raw dots, double buffered
  for (uint i = lid; i < NPR * HD; i += WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if ATTN_SPLIT
  __local float smx[NPR], ssm[NPR], srs[NPR], swl[NPR][WAVE_P];   // each pair's running max and sum, its wave's weights
  float acc[NPR];
  for (uint q = 0; q < NPR; ++q) acc[q] = 0.0f;
  if (lid < NPR) { smx[lid] = -INFINITY; ssm[lid] = 0.0f; }
#else
  float mx[NPR], sm[NPR], acc[NPR];
  for (uint q = 0; q < NPR; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
#endif
  uint wc = 0;

  for (uint g = 0; g < 2; ++g) {
    const uint mlo = g ? msplit : 0u, mhi = g ? n_act : msplit;
    if (mlo >= mhi) continue;
    const uint s = g ? s1 : s0;
    const uint end = pos + mhi;
    const uint bstart = base + blk * s;
    if (bstart >= end) continue;
    const uint bend = min(bstart + s, end);
    const uint last = end - 1;
    const uint first = key_lo(pos + mlo);   // the group's lowest visible key
    const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

    KV8_VDECL
    KREG_T kreg[PPS][PER_LANE], vreg[WAVE_P];
    for (uint pp = 0; pp < PPS; ++pp) {
      const uint p = bstart + sgid + NSG * pp;
      load_k(kv_k, KV8_ARG(k_scale) p, j, p <= last && p >= first, kreg[pp]);
    }
    load_v(kv_v, KV8_VARG(v_scale) bstart, j, sgid, first, last, vreg);
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD]);
        float qv[PER_LANE];
        qv[0] = as_float(qa.s0); qv[1] = as_float(qa.s1); qv[2] = as_float(qa.s2);
        qv[3] = as_float(qa.s3); qv[4] = as_float(qa.s4); qv[5] = as_float(qa.s5);
        qv[6] = as_float(qa.s6); qv[7] = as_float(qa.s7);
        for (uint pp = 0; pp < PPS; ++pp) {
          float a = 0.0f;
          for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], KVAL(kreg[pp][t]), a);
          for (uint stride = SG / 2; stride > 0; stride >>= 1)
            a += intel_sub_group_shuffle_down(a, a, stride);
          if (lane == 0) sc_x[buf][pr][sgid + NSG * pp] = a;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#if ATTN_SPLIT
      for (uint pr = sgid; pr < NPR; pr += NSG) {   // the pair's owner sub-group
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const float mxp = smx[pr];
        const uint bound = pos + m, lo = key_lo(bound);
        const uint p = p0 + lane;
#if REL_BIAS
        float mine = -INFINITY;
        if (p <= bound && p >= lo) {
          mine = sc_x[buf][pr][lane] * SSCALE;
          const uint distance = bound - p;
          if (distance < REL_BIAS) {
            const __global float* r = relative_q + ((size_t)m * Q_HEADS + j * GQA + pr % GQA) * REL_DIM;
            float bias = 0.0f;
#pragma unroll
            for (uint d = 0; d < REL_DIM; ++d) bias = fma(r[d], relative_table[d * REL_BIAS + distance], bias);
            mine += bias;
          }
        }
#else
        const float mine = (p <= bound && p >= lo) ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
#endif
        const float nmx = fmax(mxp, sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = exp(mxp - nmx);
          wl = exp(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) ssum += intel_sub_group_shuffle(wl, s2);   // ascending s
        swl[pr][lane] = wl;
        if (lane == 0) {
          srs[pr] = resc;
          if (nmx > -INFINITY) {
            ssm[pr] = fma(ssm[pr], resc, ssum);
            smx[pr] = nmx;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        float tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) tsum = fma(swl[pr][s2], KVAL(vreg[s2]), tsum);   // ascending s
        acc[pr] = fma(acc[pr], srs[pr], tsum);
      }
#else
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint bound = pos + m, lo = key_lo(bound);
        const uint p = p0 + lane;
#if REL_BIAS
        float mine = -INFINITY;
        if (p <= bound && p >= lo) {
          mine = sc_x[buf][pr][lane] * SSCALE;
          const uint distance = bound - p;
          if (distance < REL_BIAS) {
            const __global float* r = relative_q + ((size_t)m * Q_HEADS + j * GQA + pr % GQA) * REL_DIM;
            float bias = 0.0f;
#pragma unroll
            for (uint d = 0; d < REL_DIM; ++d) bias = fma(r[d], relative_table[d * REL_BIAS + distance], bias);
            mine += bias;
          }
        }
#else
        const float mine = (p <= bound && p >= lo) ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
#endif
        const float nmx = fmax(mx[pr], sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = exp(mx[pr] - nmx);
          wl = exp(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f, tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) {   // ascending s
          const float ws = intel_sub_group_shuffle(wl, s2);
          ssum += ws;
          tsum = fma(ws, KVAL(vreg[s2]), tsum);
        }
        if (nmx > -INFINITY) {
          sm[pr] = fma(sm[pr], resc, ssum);
          mx[pr] = nmx;
        }
        acc[pr] = fma(acc[pr], resc, tsum);
      }
#endif
      if (w + 1 < nwaves) {
        for (uint pp = 0; pp < PPS; ++pp) {
          const uint p = p0 + WAVE_P + sgid + NSG * pp;
          load_k(kv_k, KV8_ARG(k_scale) p, j, p <= last && p >= first, kreg[pp]);
        }
        load_v(kv_v, KV8_VARG(v_scale) p0 + WAVE_P, j, sgid, first, last, vreg);
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out = attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + lid] = acc[pr];
#if ATTN_SPLIT
      if (lid == 0) { out[0] = smx[pr]; out[1] = ssm[pr]; }
#else
      if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
#endif
    }
  }
}

// kol_attn_reduce - grid (Q_HEADS, M), work-group 128, work-item d = dim d: row m's blocks merged
// ascending (a block that saw none of the row's keys - bmx -INF - is skipped), rne(acc / sm).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_reduce(__global const uint* restrict ctrl, __global const float* restrict attn_part,
                              __global ushort* restrict attn_out
#if SINK_TOK
                              , __global const float* restrict sinks
#endif
#if GATE
                              , __global const float* restrict gate
#endif
                              ) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
#if SLIDING
  uint nb = (pos + m - (key_lo(pos) & ~63u)) / 64u + 1u;
#else
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
#endif
  if (nb > TGT) nb = TGT;
  for (uint b = d; b < nb; b += WG) {
    __global const float* restrict p = attn_part + (((size_t)h * TGT + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#if SINK_TOK
  float mx = sinks[(size_t)m * SINK_PITCH + h], sm = 1.0f, acc = 0.0f;
#else
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
#endif
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    if (!(bmx > -INFINITY)) continue;   // a block below the row's window
    const float bacc = attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = exp(mx - nmx);
    const float bs = exp(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
#if KV8 && GATE
  __local float rot[HD];
  const size_t out = (size_t)m * OUT_N + (size_t)h * HD + d;
  attn_out[out] = rne_bf16(kv8n_unrotate(acc / sm, d, rot) / (1.0f + exp(-gate[out])));
#elif GATE
  const size_t out = (size_t)m * OUT_N + (size_t)h * HD + d;
  attn_out[out] = rne_bf16((acc / sm) / (1.0f + exp(-gate[out])));
#else
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(acc / sm);
#endif
}
