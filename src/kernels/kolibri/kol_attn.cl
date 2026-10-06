// kol_attn.cl - spec 20c: Kolibri-1's decode attention, spec 10's v2 (src/kernels/attn_v2.cl) as
// k2_attn.cl carries it to head_dim 128, here at GQA 12 without a gate, over a SLIDING window through
// the ring (WINDOW 513, RING 4096) or a FULL (NoPE) layer's growing cache (WINDOW 0). attn_v2.cl and
// k2_attn.cl are untouched.
//
//   kol_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)    grid (KV_HEADS, TGT), WG 128
//   kol_attn_reduce(ctrl, attn_part, attn_out)              grid (Q_HEADS, M),    WG 128
//     attn_q     fp32 [M][Q_HEADS][128]     kol_attn_prep's q (bf16 values)
//     kv_k/kv_v  bf16 [rows][KV_HEADS][128] this layer's ring (rows = RING) or cache (rows = max_len)
//     attn_part  fp32 [Q_HEADS][TGT][M][130]   {mx, sm, acc[128]} per block
//     attn_out   bf16 [M][Q_HEADS x 128]    o_proj's input
//
// **What is k2_attn.cl's, unchanged:** a work-group of 128 = 8 sub-groups x 16 lanes, work-item d
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
#if M < 1 || M > 4
#error "kol_attn: M is 1..4"
#endif
#define SLIDING (WINDOW != 0)
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

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

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
inline void load_k(__global const ushort* restrict kv_k, uint p, uint j, bool ok, ushort* kreg) {
  if (ok) {   // uniform within the sub-group: p is the sub-group's position
    const ushort8 a = intel_sub_group_block_read_us8(kv_k + ((size_t)key_row(p) * KV_HEADS + j) * HD);
    kreg[0] = a.s0; kreg[1] = a.s1; kreg[2] = a.s2; kreg[3] = a.s3;
    kreg[4] = a.s4; kreg[5] = a.s5; kreg[6] = a.s6; kreg[7] = a.s7;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = (ushort)0;
  }
}

// The wave's 16 V values of dim `lid` (sub-group sgid holds dims 16 sgid .. +15), position p0 + s at
// register s; outside [first, last] 0. p0 is 16-aligned, so the 16 rows are contiguous (in the ring:
// RING is whole 64-row blocks). The surface: the ring's RING rows, or the cache's last + 1.
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

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void kol_attn_decode(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                              __global const ushort* restrict kv_k, __global const ushort* restrict kv_v,
                              __global float* restrict attn_part) {
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

  float mx[NPR], sm[NPR], acc[NPR];
  for (uint q = 0; q < NPR; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
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

    ushort kreg[PPS][PER_LANE], vreg[WAVE_P];
    for (uint pp = 0; pp < PPS; ++pp) {
      const uint p = bstart + sgid + NSG * pp;
      load_k(kv_k, p, j, p <= last && p >= first, kreg[pp]);
    }
    load_v(kv_v, bstart, j, sgid, first, last, vreg);
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
          for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], bf16f(kreg[pp][t]), a);
          for (uint stride = SG / 2; stride > 0; stride >>= 1)
            a += intel_sub_group_shuffle_down(a, a, stride);
          if (lane == 0) sc_x[buf][pr][sgid + NSG * pp] = a;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint bound = pos + m, lo = key_lo(bound);
        const uint p = p0 + lane;
        const float mine = (p <= bound && p >= lo) ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
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
          tsum = fma(ws, bf16f(vreg[s2]), tsum);
        }
        if (nmx > -INFINITY) {
          sm[pr] = fma(sm[pr], resc, ssum);
          mx[pr] = nmx;
        }
        acc[pr] = fma(acc[pr], resc, tsum);
      }
      if (w + 1 < nwaves) {
        for (uint pp = 0; pp < PPS; ++pp) {
          const uint p = p0 + WAVE_P + sgid + NSG * pp;
          load_k(kv_k, p, j, p <= last && p >= first, kreg[pp]);
        }
        load_v(kv_v, p0 + WAVE_P, j, sgid, first, last, vreg);
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out = attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + lid] = acc[pr];
      if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
    }
  }
}

// kol_attn_reduce - grid (Q_HEADS, M), work-group 128, work-item d = dim d: row m's blocks merged
// ascending (a block that saw none of the row's keys - bmx -INF - is skipped), rne(acc / sm).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void kol_attn_reduce(__global const uint* restrict ctrl, __global const float* restrict attn_part,
                              __global ushort* restrict attn_out) {
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
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
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
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(acc / sm);
}
