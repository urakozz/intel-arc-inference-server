// attn_v2.cl - decode attention at depth, v2 (spec 10 §2; plan 10b). The pair
// `attn_decode_v2` + `attn_reduce_v2` replaces attn.cl's `attn_decode` +
// `attn_reduce` when the capture selects it (B70_DECODE_ATTN, src/runtime/capture.cc).
// `attn_prep` (attn.cl) is shared and unchanged: v2 reads the same attn_q / attn_gate
// and the same KV caches, and writes the same attn_out.
//
// It is plan 10a's winner `P0_L2_D1_Q1_E1` (tools/probe/probe_decode_attn.cl, record
// docs/probe-decode-attn-2026-09-28.md), shipped in its exp form `P0_L2_D1_Q1` (see EXP2
// below), with two changes the probe did not have:
//
//  1. **The stride is per ROW and a function of that row's key count alone.** Row m of
//     a step attends to L_m = pos + m + 1 positions and walks them in blocks of
//         ppw(L) = max(64, roundup64(ceil(L / TGT))),   TGT = 32
//     positions per work-group (at most TGT blocks for any L). The probe used one
//     stride per launch from pos + n_active, so an M = 4 verify row m and the M = 1
//     step at pos + m could block the same keys differently (a last-ulp reassociation).
//     Here row m of any launch sees exactly the blocks, waves and orders of the M = 1
//     step at pos + m, so spec 8's verify rows are bitwise the plain steps (tested by
//     tests/kernels/attn_v2_test.cc). ppw is a step function of L that moves every
//     2048 keys, so the rows of one launch (L spans <= 3) have at most two strides:
//     rows [0, msplit) at s0 and [msplit, n_active) at s1. A work-group runs the two
//     row groups one after the other (the second only when s1 != s0, i.e. for at most
//     three positions in 2048), each over its own positions.
//     No Control word and no cached value: the stride is recomputed from `pos` every
//     step, so a restored snapshot (spec 7) or a rewound pos needs nothing.
//  2. **The grid is (4 kv heads, TGT) and the partial table [24][TGT][M][258]**, not
//     MAXLEN / 64 blocks: no row needs more than TGT work-groups, so v2 has no
//     max_len-dependent code (one binary per M serves every max_len) and no idle grid.
//     The V 2D block read takes the surface height from `last + 1`, not MAXLEN, so a
//     wave's rows past the last written position read as zero and never past the cache.
//
// The wave (DOT 1 of the probe): grid (4, TGT), work-group 256 = 16 sub-groups of 16,
// sub-group s owns position p0 + s of a 16-position wave. K by two sub-group block
// reads per row (lane l element t = dim l + 16 t - attn.cl's order), V by one 2D block
// read per sub-group per wave (16 positions x 16 dims, lane = dim). One K/V wave feeds
// all 6 x M (q head, row) pairs: the score dot in attn.cl's per-lane fma order and its
// pairwise tree (stride 8, 4, 2, 1) by shuffles, q read from SLM by block reads; ONE
// barrier per wave publishes the raw dots (double-buffered SLM by a wave counter that
// runs across the two row groups); lane s computes position s's weight and the weights
// are broadcast in ascending s for the sum and the V fma (attn.cl's orders).
// EXP2=1: exp2 with 1/16 * log2(e) folded into the score scale (spec 10 lever 4,
// approved 2026-09-28 subject to the gates), max and merges in the log2 domain. It fails
// the golden gate (cjk row 9 flips, spec 10 §8), so the shipped build is EXP2=0:
// attn.cl's `exp` and 1/16 scale.
//
// Masking: row m's bound is pos + m. A position past it scores -INF, weight exactly 0,
// which adds +-0 to the sums - so a longer walk (the verify's group end) is neutral to
// row m, which is the bitwise claim above. Whole-wave-masked rows keep (mx, sm, acc).
//
//   attn_decode_v2(ctrl, attn_q, kv_k, kv_v, attn_part)   grid (4, TGT), WG 256
//   attn_reduce_v2(ctrl, attn_part, attn_gate, attn_out)  grid (24, M),  WG 256
//     attn_part  fp32 [24][TGT][M][258]  {mx (log2 domain if EXP2), sm, acc[256]}
//     the rest as attn.cl's header.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#ifndef CTRL_POS
#error "attn_v2: CTRL_POS must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef CTRL_NACT
#error "attn_v2: CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef TGT
#error "attn_v2: TGT must be defined (src/kernels/CMakeLists.txt; runtime::DecodeScratch::kAttnV2Blocks)"
#endif

// Spec 15c: another model's heads come in as -DFA_Q_HEADS / -DFA_KV_HEADS (the binary
// is `_Q<q>KV<kv>`, kernels::fa_suffix); unset, the five lines below are Qwen3.8's,
// token for token. Ornith 16 / 2: GQA 8, so NPR = 8 x M (q, row) pairs per wave and
// 8 KB x M of SLM for qpack.
#ifndef FA_Q_HEADS
#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define OUT_N 6144
#else
#ifndef FA_KV_HEADS
#error "attn_v2: FA_Q_HEADS needs FA_KV_HEADS"
#endif
#define Q_HEADS FA_Q_HEADS
#define KV_HEADS FA_KV_HEADS
#define GQA (FA_Q_HEADS / FA_KV_HEADS)
#define HD 256
#define OUT_N (FA_Q_HEADS * HD)
#if FA_Q_HEADS % FA_KV_HEADS != 0
#error "attn_v2: q-heads must be a multiple of kv-heads"
#endif
#endif
#define WAVE_P 16
#define SG 16
#define PER_LANE 16
#define PART 258
#define WG 256
#define NPR (M * GQA)
#ifndef EXP2
#error "attn_v2: EXP2 must be defined (src/kernels/CMakeLists.txt, B70_ATTN_V2_EXP2)"
#endif
#if EXP2
#define SSCALE (0.0625f * M_LOG2E_F)   /* 1/sqrt(256) * log2(e) */
#define EXPF(x) exp2(x)
#else
#define SSCALE 0.0625f
#define EXPF(x) exp(x)
#endif

// M = 5..8: spec 19a's verify probe (B70_VERIFY_M8). qpack is 6 KB x M (48 KB at 8, under
// the 128 KB a work-group may hold) and mx / sm / acc hold 6 x M floats per lane; the rows
// of one launch still span at most two strides (ppw moves every 2048 keys).
#if M < 1 || M > 8
#error "attn_v2: M is 1..8"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// Positions per work-group for a row attending to `len` keys (header, change 1).
inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

// One K row's 16 elements for this lane: dim lane + 16 t, t ascending.
inline void load_k(__global const ushort* restrict kv_k, uint p, uint j, bool ok, ushort* kreg) {
  if (ok) {   // uniform within the sub-group: p is the sub-group's position
    __global const ushort* row = kv_k + ((size_t)p * KV_HEADS + j) * HD;
    const ushort8 a = intel_sub_group_block_read_us8(row);
    const ushort8 b = intel_sub_group_block_read_us8(row + 128);
    kreg[0] = a.s0; kreg[1] = a.s1; kreg[2] = a.s2; kreg[3] = a.s3;
    kreg[4] = a.s4; kreg[5] = a.s5; kreg[6] = a.s6; kreg[7] = a.s7;
    kreg[8] = b.s0; kreg[9] = b.s1; kreg[10] = b.s2; kreg[11] = b.s3;
    kreg[12] = b.s4; kreg[13] = b.s5; kreg[14] = b.s6; kreg[15] = b.s7;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = (ushort)0;
  }
}

// The wave's 16 V values of dim `lid`, position p0 + s at register s; past `last` 0.
inline void load_v(__global const ushort* restrict kv_v, uint p0, uint j, uint sgid, uint last,
                   ushort* vreg) {
  ushort vb[WAVE_P];
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD), HD * 2,
                                             last + 1, KV_HEADS * HD * 2,
                                             (int2)((int)(SG * sgid), (int)p0), vb);
  for (uint s = 0; s < WAVE_P; ++s) vreg[s] = p0 + s <= last ? vb[s] : (ushort)0;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode_v2(__global const uint* restrict ctrl,
                             __global const float* restrict attn_q,
                             __global const ushort* restrict kv_k,
                             __global const ushort* restrict kv_v,
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

  // The (at most two) row groups: rows [0, msplit) at s0, [msplit, n_act) at s1.
  const uint s0 = v2_ppw(pos + 1);
  const uint s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  // Uniform early-out: this block holds no position of either group.
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
  if (!live0 && !live1) return;

  __local float qpack[NPR * HD];              // [m][qhl][256], 6 KB x M
  __local float sc_x[2][NPR][WAVE_P];         // the wave's raw dots, double buffered
  for (uint i = lid; i < NPR * HD; i += WG) {
    const uint pr = i / HD, d = i % HD;       // pr = m * GQA + qhl
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx[NPR], sm[NPR], acc[NPR];
  for (uint q = 0; q < NPR; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
  uint wc = 0;                                // waves run by this work-group: the sc_x buffer

  for (uint g = 0; g < 2; ++g) {
    const uint mlo = g ? msplit : 0u, mhi = g ? n_act : msplit;
    if (mlo >= mhi) continue;                 // uniform
    const uint s = g ? s1 : s0;
    const uint end = pos + mhi;               // positions [0, end) are this group's keys
    const uint bstart = blk * s;
    if (bstart >= end) continue;              // uniform
    const uint bend = min(bstart + s, end);
    const uint last = end - 1;
    const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

    ushort kreg[PER_LANE], vreg[WAVE_P];
    load_k(kv_k, bstart + sgid, j, bstart + sgid <= last, kreg);
    load_v(kv_v, bstart, j, sgid, last, vreg);
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
      // Scores: sub-group sgid owns position p0 + sgid; attn.cl's per-lane fma order,
      // then its pairwise tree by shuffles; lane 0 holds the dot.
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
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
        for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], bf16f(kreg[t]), a);
        for (uint stride = SG / 2; stride > 0; stride >>= 1)
          a += intel_sub_group_shuffle_down(a, a, stride);
        if (lane == 0) sc_x[buf][pr][sgid] = a;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // The online update, per row: lane s owns position p0 + s.
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint bound = pos + m;
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
        const float nmx = fmax(mx[pr], sub_group_reduce_max(mine));   // max: exact in any order
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = EXPF(mx[pr] - nmx);
          wl = EXPF(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f, tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) {   // ascending s, as attn.cl
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
        load_k(kv_k, p0 + WAVE_P + sgid, j, p0 + WAVE_P + sgid <= last, kreg);
        load_v(kv_v, p0 + WAVE_P, j, sgid, last, vreg);
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

// attn_reduce_v2 - grid (24 q heads, M), work-group 256, work-item d = dim d: attn.cl's
// attn_reduce over row m's nb = (pos + m) / ppw(pos + m + 1) + 1 <= TGT partials,
// ascending, merged as the decode kernel's EXPF.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void attn_reduce_v2(__global const uint* restrict ctrl,
                             __global const float* restrict attn_part,
                             __global const float* restrict attn_gate,
                             __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;                    // uniform
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
  if (nb > TGT) nb = TGT;                    // never taken (header); an overrun guard
  for (uint b = d; b < nb; b += WG) {
    __global const float* restrict p = attn_part + (((size_t)h * TGT + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {            // ascending block order
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);         // finite: block b starts at or before pos + m
    const float a = EXPF(mx - nmx);          // mx = -INF on the first block -> 0
    const float bs = EXPF(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const float o = acc / sm;
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * sigmoid_f32(g));
}
