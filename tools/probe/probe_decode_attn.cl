// probe_decode_attn.cl - spec 10 P1: decode attention at depth, the lever probe.
//
// Plan: docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md (Task 2).
// Record: docs/probe-decode-attn-2026-09-28.md.
//
// PROBE kernels. No runtime path binds them; src/kernels/attn.cl is untouched. The
// control (arm 0) is attn.cl itself, compiled at the probe's M (tools/probe/CMakeLists.txt).
//
// The same maths, buffers and orders as attn.cl's attn_decode + attn_reduce (read its
// header first), with the levers of spec 10 §2 as -D knobs:
//
//   PPW       positions per work-group: 64, 256, 512, 1024, 2048 - or 0: derived ON THE
//             DEVICE from pos, ppw = max(64, roundup64(ceil((pos + n_act) / tgt))), with
//             tgt (live work-groups per kv head) read from the control word CTRL_TGT. The
//             grid is sized for the smallest ppw (64), work-groups past the context exit.
//   LOAD      0: attn.cl's loads (one ushort per lane per K element and per V row, 32 B
//             messages); 1: K by two 1D sub-group block reads per row (256 B messages,
//             lane l element t is still dim l + 16 t, so the dot's order is attn.cl's),
//             V as 0; 2: as 1, and V by one 2D block read per wave per sub-group
//             (16 positions x 16 dims, lane = dim = lid, register = position).
//   DOT       0: attn.cl's structure (m outer, per q-head SLM tree with 4 + 1 barriers);
//             1: m inner (K/V loaded once per wave for all 6 x M rows), the score tree by
//             sub-group shuffles (the same pairwise adds, so lane 0 holds attn.cl's bits),
//             ONE barrier per wave publishing all 6 x M scores, and the softmax weights
//             computed lane-parallel (lane s owns position s) then broadcast in ascending
//             s - every add and fma in attn.cl's order, so PPW 64 is bitwise attn.cl.
//   PREFETCH  0/1 (DOT 1 only): the next wave's K/V registers loaded before this wave's
//             arithmetic (a register double buffer).
//   REDUCE    0: a separate pda_reduce launch over the partials (attn_reduce's merge);
//             1: the last work-group of each kv head to finish (a device counter) merges
//             that head's 6 x M rows in the same fixed ascending order, so the result does
//             not depend on which work-group is last; the counter is reset by that WG.
//   QBLK      0/1 (DOT 1 only): the q operand of the score dot by two SLM sub-group block
//             reads per row (lane l element i = dim l + 16 i, the same order) instead of
//             16 scalar SLM reads.
//   EXP2      0/1: exp2 with 1/16 * log2(e) folded into the score scale (spec 10 lever 4,
//             spec 6c's approved form); mx and the merge are then in the log2 domain.
//             Not bitwise attn.cl.
//   M         rows in flight (1..4).
// (256-GRF mode is a compile option, not a define: the tag's G field, CMakeLists.txt.)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#ifndef MAXLEN
#define MAXLEN 131072
#endif
#ifndef PPW
#define PPW 64
#endif
#ifndef LOAD
#define LOAD 0
#endif
#ifndef DOT
#define DOT 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef REDUCE
#define REDUCE 0
#endif
#ifndef QBLK
#define QBLK 0
#endif
#ifndef EXP2
#define EXP2 0
#endif
#if EXP2
#define SSCALE (0.0625f * M_LOG2E_F)
#define EXPF(x) exp2(x)
#else
#define SSCALE 0.0625f
#define EXPF(x) exp(x)
#endif
#define CTRL_POS 0
#define CTRL_NACT 1
#define CTRL_TGT 19   /* runtime::Control::pad[0]: the probe's target work-groups per kv head */

#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define OUT_N 6144
#define SCALE 0.0625f
#define WAVE_P 16
#define SG 16
#define PER_LANE 16
#define PART 258
#define WG 256
#define PMIN (PPW ? PPW : 64)
#define NBLOCKS (MAXLEN / PMIN)
#define NPR (M * GQA)

#if PPW % 64 != 0
#error "PPW must be 0 or a multiple of 64"
#endif
#if QBLK && !DOT
#error "QBLK is a DOT 1 lever"
#endif
#if PREFETCH && !DOT
#error "PREFETCH is a DOT 1 lever"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// Positions per work-group for this launch. Uniform: every input is a control word.
inline uint ppw_of(__global const uint* restrict ctrl, uint end) {
#if PPW
  return PPW;
#else
  uint tgt = ctrl[CTRL_TGT];
  if (tgt == 0) tgt = 1;
  uint p = (end + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return max(64u, p);
#endif
}

// One K row's 16 elements for this lane: dim lane + 16 t, t ascending.
inline void load_k(__global const ushort* restrict kv_k, uint p, uint j, uint lane, bool ok,
                   ushort* kreg) {
#if LOAD == 0
  for (uint t = 0; t < PER_LANE; ++t)
    kreg[t] = ok ? kv_k[((size_t)p * KV_HEADS + j) * HD + lane + SG * t] : (ushort)0;
#else
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
#endif
}

// The wave's 16 V values of dim `lid`, position p0 + s at register s. A position past
// `last` is 0 (the unwritten-slot NaN guard of attn.cl, at the point of load).
inline void load_v(__global const ushort* restrict kv_v, uint p0, uint j, uint lid, uint sgid,
                   uint last, ushort* vreg) {
#if LOAD == 2
  ushort vb[WAVE_P];
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD), HD * 2,
                                             MAXLEN, KV_HEADS * HD * 2,
                                             (int2)((int)(SG * sgid), (int)p0), vb);
  for (uint s = 0; s < WAVE_P; ++s) vreg[s] = p0 + s <= last ? vb[s] : (ushort)0;
#else
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint ps = p0 + s;
    vreg[s] = ps <= last ? kv_v[((size_t)ps * KV_HEADS + j) * HD + lid] : (ushort)0;
  }
#endif
}

// attn_reduce's merge for (q head h, row m), work-item d = dim d, reading nb partials
// in ascending order; the output chain is attn_reduce's.
inline void merge_row(__global const float* restrict attn_part,
                      __global const float* restrict attn_gate, __global ushort* restrict attn_out,
                      __local float* hmx, __local float* hsm, uint h, uint m, uint nb, uint d) {
  for (uint b = d; b < nb; b += WG) {
    __global const float* restrict p = attn_part + (((size_t)h * NBLOCKS + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[(((size_t)h * NBLOCKS + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = EXPF(mx - nmx);
    const float bs = EXPF(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);   // hmx/hsm are reused by the next row
  const float o = acc / sm;
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * sigmoid_f32(g));
}

// ---------------------------------------------------------------------------
// pda_decode - grid (4 kv heads, NBLOCKS), work-group 256 = 16 sub-groups x 16.
//   pda_decode(ctrl, attn_q, kv_k, kv_v, attn_part, attn_gate, attn_out, counters)
// attn_gate / attn_out / counters are read only by REDUCE 1.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pda_decode(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                         __global const ushort* restrict kv_k, __global const ushort* restrict kv_v,
                         __global float* restrict attn_part,
                         __global const float* restrict attn_gate,
                         __global ushort* restrict attn_out, __global uint* counters) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;
  const uint end = pos + n_act;             // positions [0, end) exist this step
  const uint ppw = ppw_of(ctrl, end);
  const uint bstart = blk * ppw;
  if (bstart >= end) return;                // the early-out, uniform
  const uint bend = min(bstart + ppw, end);
  const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

#if DOT == 0
  // ---- attn.cl's structure, verbatim but for PPW and LOAD -----------------------
  __local float qpack[GQA * HD];
  __local float dot_red[WG];
  for (uint m = 0; m < n_act; ++m) {
    for (uint qhl = 0; qhl < GQA; ++qhl)
      qpack[qhl * HD + lid] = attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    float mx[GQA], sm[GQA], acc[GQA];
    for (uint q = 0; q < GQA; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
    const uint bound = pos + m;
    for (uint w = 0; w < nwaves; ++w) {
      const uint p0 = bstart + w * WAVE_P;
      const uint p = p0 + sgid;
      ushort kreg[PER_LANE], vreg[WAVE_P];
      load_k(kv_k, p, j, lane, p <= bound, kreg);
      load_v(kv_v, p0, j, lid, sgid, bound, vreg);
      for (uint qhl = 0; qhl < GQA; ++qhl) {
        float a = 0.0f;
        if (p <= bound)
          for (uint t = 0; t < PER_LANE; ++t)
            a = fma(qpack[qhl * HD + lane + SG * t], bf16f(kreg[t]), a);
        dot_red[lid] = a;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = SG / 2; stride > 0; stride >>= 1) {
          if (lane < stride) dot_red[sgid * SG + lane] += dot_red[sgid * SG + lane + stride];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        float sc[WAVE_P];
        for (uint s = 0; s < WAVE_P; ++s)
          sc[s] = p0 + s <= bound ? dot_red[s * SG] * SSCALE : -INFINITY;
        float nmx = mx[qhl];
        for (uint s = 0; s < WAVE_P; ++s) nmx = fmax(nmx, sc[s]);
        float resc;
        if (nmx > -INFINITY) {
          resc = EXPF(mx[qhl] - nmx);
          float ssum = 0.0f;
          for (uint s = 0; s < WAVE_P; ++s) { sc[s] = EXPF(sc[s] - nmx); ssum += sc[s]; }
          sm[qhl] = fma(sm[qhl], resc, ssum);
          mx[qhl] = nmx;
        } else {
          resc = 1.0f;
          for (uint s = 0; s < WAVE_P; ++s) sc[s] = 0.0f;
        }
        float tsum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) tsum = fma(sc[s], bf16f(vreg[s]), tsum);
        acc[qhl] = fma(acc[qhl], resc, tsum);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
    for (uint qhl = 0; qhl < GQA; ++qhl) {
      __global float* restrict out =
          attn_part + (((size_t)(j * GQA + qhl) * NBLOCKS + blk) * M + m) * PART;
      out[2 + lid] = acc[qhl];
      if (lid == 0) { out[0] = mx[qhl]; out[1] = sm[qhl]; }
    }
  }
#else
  // ---- DOT 1: m inner, shuffle tree, one barrier per wave ------------------------
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
  const uint last = end - 1;                  // the last written position this step
  ushort kreg[PER_LANE], vreg[WAVE_P];
  load_k(kv_k, bstart + sgid, j, lane, bstart + sgid <= last, kreg);
  load_v(kv_v, bstart, j, lid, sgid, last, vreg);
  for (uint w = 0; w < nwaves; ++w) {
    const uint p0 = bstart + w * WAVE_P;
    const uint buf = w & 1u;
#if PREFETCH
    ushort kn[PER_LANE], vn[WAVE_P];
    if (w + 1 < nwaves) {
      load_k(kv_k, p0 + WAVE_P + sgid, j, lane, p0 + WAVE_P + sgid <= last, kn);
      load_v(kv_v, p0 + WAVE_P, j, lid, sgid, last, vn);
    }
#endif
    // Scores: sub-group sgid owns position p0 + sgid. attn.cl's per-lane fma order,
    // then its pairwise tree (stride 8, 4, 2, 1) by shuffles; lane 0 holds the dot.
    for (uint pr = 0; pr < NPR; ++pr) {
      float a = 0.0f;
#if QBLK
      float qv[PER_LANE];
      {
        const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD]);
        const uint8 qb = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD + 128]);
        qv[0] = as_float(qa.s0); qv[1] = as_float(qa.s1); qv[2] = as_float(qa.s2);
        qv[3] = as_float(qa.s3); qv[4] = as_float(qa.s4); qv[5] = as_float(qa.s5);
        qv[6] = as_float(qa.s6); qv[7] = as_float(qa.s7);
        qv[8] = as_float(qb.s0); qv[9] = as_float(qb.s1); qv[10] = as_float(qb.s2);
        qv[11] = as_float(qb.s3); qv[12] = as_float(qb.s4); qv[13] = as_float(qb.s5);
        qv[14] = as_float(qb.s6); qv[15] = as_float(qb.s7);
      }
      for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], bf16f(kreg[t]), a);
#else
      for (uint t = 0; t < PER_LANE; ++t)
        a = fma(qpack[pr * HD + lane + SG * t], bf16f(kreg[t]), a);
#endif
      for (uint stride = SG / 2; stride > 0; stride >>= 1)
        a += intel_sub_group_shuffle_down(a, a, stride);
      if (lane == 0) sc_x[buf][pr][sgid] = a;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // The online update, per row: lane s owns position p0 + s.
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA;
      if (m >= n_act) break;                  // uniform
      const uint bound = pos + m;
      const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
      float nmx = mx[pr];
      nmx = fmax(nmx, sub_group_reduce_max(mine));   // max is exact in any order
      float resc, wl;
      if (nmx > -INFINITY) {
        resc = EXPF(mx[pr] - nmx);
        wl = EXPF(mine - nmx);
      } else {
        resc = 1.0f;
        wl = 0.0f;
      }
      float ssum = 0.0f, tsum = 0.0f;
      for (uint s = 0; s < WAVE_P; ++s) {     // ascending s, as attn.cl
        const float ws = intel_sub_group_shuffle(wl, s);
        ssum += ws;
        tsum = fma(ws, bf16f(vreg[s]), tsum);
      }
      if (nmx > -INFINITY) {
        sm[pr] = fma(sm[pr], resc, ssum);
        mx[pr] = nmx;
      }
      acc[pr] = fma(acc[pr], resc, tsum);
    }
#if PREFETCH
    if (w + 1 < nwaves) {
      for (uint t = 0; t < PER_LANE; ++t) kreg[t] = kn[t];
      for (uint s = 0; s < WAVE_P; ++s) vreg[s] = vn[s];
    }
#else
    if (w + 1 < nwaves) {
      load_k(kv_k, p0 + WAVE_P + sgid, j, lane, p0 + WAVE_P + sgid <= last, kreg);
      load_v(kv_v, p0 + WAVE_P, j, lid, sgid, last, vreg);
    }
#endif
  }
  for (uint pr = 0; pr < NPR; ++pr) {
    const uint m = pr / GQA, qhl = pr % GQA;
    if (m >= n_act) break;
    __global float* restrict out =
        attn_part + (((size_t)(j * GQA + qhl) * NBLOCKS + blk) * M + m) * PART;
    out[2 + lid] = acc[pr];
    if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
  }
#endif

#if REDUCE == 1
  // The last work-group of kv head j to finish merges its 6 x n_act rows.
  __local uint is_last;
  __local float hmx[NBLOCKS], hsm[NBLOCKS];
  const uint nlive = (end + ppw - 1) / ppw;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    const uint old = atomic_fetch_add_explicit((volatile __global atomic_uint*)&counters[j], 1u,
                                               memory_order_acq_rel, memory_scope_device);
    is_last = old == nlive - 1 ? 1u : 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (is_last == 0u) return;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  for (uint m = 0; m < n_act; ++m) {
    const uint nb = (pos + m) / ppw + 1;
    for (uint qhl = 0; qhl < GQA; ++qhl)
      merge_row(attn_part, attn_gate, attn_out, hmx, hsm, j * GQA + qhl, m, nb, lid);
  }
  if (lid == 0)
    atomic_store_explicit((volatile __global atomic_uint*)&counters[j], 0u, memory_order_relaxed,
                          memory_scope_device);
#endif
}

// ---------------------------------------------------------------------------
// pda_reduce - grid (24 q heads, M), work-group 256: attn_reduce, with nb from ppw.
// dbg[0] receives the partial count row (h 0, m 0) merged: the arm's own counter for
// the partial table's traffic.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void pda_reduce(__global const uint* restrict ctrl, __global const float* restrict attn_part,
                         __global const float* restrict attn_gate,
                         __global ushort* restrict attn_out, __global uint* restrict dbg) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[NBLOCKS], hsm[NBLOCKS];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
  const uint ppw = ppw_of(ctrl, pos + n_act);
  uint nb = (pos + m) / ppw + 1;
  if (nb > NBLOCKS) nb = NBLOCKS;
  if (h == 0 && m == 0 && d == 0) dbg[0] = nb;
  merge_row(attn_part, attn_gate, attn_out, hmx, hsm, h, m, nb, d);
}
