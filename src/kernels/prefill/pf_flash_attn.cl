// pf_flash_attn - spec 6: the fused bf16 flash attention of prefill.
// This is plan 6a's winning arm
// pfa_KT64_R16_H6_Q0 (tools/probe/probe_flash_attn.cl), promoted with its text
// unchanged except for the entry name, then (spec 6c, docs/12-kernels.md ("How the prefill kernels got here")
// levers 1-2) moved to 8 rows per work-group, with arm KT64_R8_Q2's `exp2` behind EXP2;
// built with KT=64 RPW=8 HPW=6 QREG=0 EXP2=<see CMakeLists.txt> and
// -cl-intel-256-GRF-per-thread. WG = 16 x 6 lanes, grid (ceil(C / 8), 4, 1).
//
//   Q   bf16 [C][24][256]    pf_q; q-head h = 6 j + l for kv head j
//   Kc  bf16 [depth][4][256] one layer's K cache, depth = pos + C
//   Vc  bf16 [depth][4][256] one layer's V cache
//   O   fp32 [24][rows][256] pf_o's layout, rows = pad256(C)
//
// Each sub-group owns 8 query rows of one head. O is 16 float8 (8 rows x 256 dims) in
// registers, S for one KV tile is KT/16 float8 with row in the component and key in the
// lane, which is the bf16 DPAS A-operand layout, so P goes to PV with no data movement.
// Loads are pf_gemm.cl's verified idioms: K^T via the TRANSB transposed 32-bit read, V
// via the !TRANSB VNNI-transform read, Q via the 16-bit A read (8 rows instead of 32).
//
// Numerics (not negotiable, the correctness bar depends on them): fp32 scores times
// ATTN_SCALE; the mask covers key <= pos + row AND key < depth; fp32 online softmax; P =
// exp(s - m) rounded to bf16 as the PV operand; O / l at the end. EXP2=0 uses `exp`,
// never `native_exp` (spec 6 §3.1). EXP2=1 is the research form approved on
// 2026-09-27: scores times ATTN_SCALE * log2(e), so s, the running max m and the masked
// -INF all live in the log2 domain, and `exp2` (one base-2 math.exp in IGC; `exp`'s
// range-split temporaries are what spill 77 GRF). EXP2=1 moves the rounding point and
// measured K3a l0 0.999951245 < composed 0.999959889 (spec 6 §9); it was enabled
// it anyway (default EXP2=1, K3a tol 1e-5 on l0). RPW 8 alone is bitwise neutral (K3a numbers identical to RPW 16).
// Rows past C in the last 8-row block are computed (finite); rows of blocks wholly past
// C are not written.
//
// 3l (2): SPLIT=1 builds the split-K form for short chunks deep in the cache (the same
// tile loop; SPLIT unset is the token stream above, unchanged). Grid (ceil(C / 8), kv,
// (GQA / HPW) x n): work-group span sp walks keys [sp * span, sp * span + span) only - the
// last span to `last`, its rows' causal end - so every span starts at a key below pos,
// which every row sees. O is the partials area [n][24][c8][256] (c8 = C rounded up to 8),
// unnormalised; ML holds each row's running max m [n][24][c8], then its sum l, the same.
// pf_flash_merge (FLASH_MERGE=1) folds the spans into pf_o: M = max m_s, w_s = EXPF(m_s - M),
// O = (sum w_s O_s) * (1 / sum w_s l_s), spans ascending.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#ifndef KT
#error "KT (KV positions per tile, 32 or 64) must be defined"
#endif
#if KT % 32
#error "KT must be a multiple of 32 (the PV loop takes key atoms in pairs)"
#endif
#if RPW % 8
#error "RPW must be a multiple of 8"
#endif
// Unset FA_Q_HEADS / FA_KV_HEADS (every Qwen3.8 and Agnes command line): Qwen3.8's 24 q /
// 4 kv heads, GQA 6, token for token. Spec 15d: another model's heads, named
// `_Q<q>KV<kv>` (Ornith 16 / 2: GQA 8, built at HPW 8 so grid.z stays 1).
#ifndef FA_Q_HEADS
#define FA_QH 24
#define FA_KVH 4
#define FA_GQA 6u
#else
#ifndef FA_KV_HEADS
#error "pf_flash_attn: FA_Q_HEADS needs FA_KV_HEADS"
#endif
#define FA_QH FA_Q_HEADS
#define FA_KVH FA_KV_HEADS
#define FA_GQA ((uint)(FA_Q_HEADS / FA_KV_HEADS))
#if FA_Q_HEADS % FA_KV_HEADS != 0 || (FA_Q_HEADS / FA_KV_HEADS) % HPW != 0
#error "pf_flash_attn: q-heads must be a multiple of kv-heads, and the GQA group of HPW"
#endif
#endif
#define SG 16
#define HD 256
#define NDA (HD / 16)                 /* 16 dim-atoms of O */
#define NKA (KT / 16)                 /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)           /* sub-groups per work-group */
#define ATTN_SCALE (1.0f / 16.0f)
#ifndef EXP2
#error "EXP2 (0: exp, 1: exp2 with ATTN_SCALE * log2(e) folded) must be defined"
#endif
#if EXP2
#define SCORE_SCALE (ATTN_SCALE * M_LOG2E_F)   /* scores in the log2 domain */
#define EXPF(x) exp2(x)
#else
#define SCORE_SCALE ATTN_SCALE
#define EXPF(x) exp(x)
#endif
// FA_SKELETON=1 (probe-only, spec 38a Task 1: tools/probe's pfas_* arms of the PAIR and WIDE forms): EXPF and the
// sub-group max / sum of the softmax become identities, so an arm times its form's loads, DPAS, SLM traffic and
// barriers without the softmax's arithmetic. Its output is not attention. Unset everywhere else.
#ifndef FA_SKELETON
#define FA_SKELETON 0
#endif
#if FA_SKELETON
#undef EXPF
#define EXPF(x) (x)
#define FA_RMAX(x) (x)
#define FA_RADD(x) (x)
#else
#define FA_RMAX(x) sub_group_reduce_max(x)
#define FA_RADD(x) sub_group_reduce_add(x)
#endif

#include "bf16.h"

#ifndef SPLIT
#define SPLIT 0
#endif
// KV_PAGED (plan 19v3b Task 0: the KV page in positions, runtime::kKvPage; undefined on every other command line): Kc /
// Vc are the layer's whole cache and a last argument `pages` maps the KV pages - key p is cache row
// pages[p / KV_PAGED] x KV_PAGED + p % KV_PAGED. A tile of KT keys starts at a multiple of KT and KT divides
// KV_PAGED, so a tile is in one page: its K and V reads take that page as the surface (rows to the page's end or to
// `depth`, the zeros past it as before). Every value and order is the contiguous kernel's. Plan 31 Task 12: the r8 form
// with GATE_EPI too (Ornith's slots).
#if defined(KV_PAGED) && (KV_PAGED % KT != 0 || (GATE_EPI && SPLIT) || FLASH_MERGE)
#error "pf_flash_attn KV_PAGED: whole tiles a page; the r8 (gated or not) / split forms only"
#endif
#ifndef GATE_EPI
#define GATE_EPI 0
#endif
#if GATE_EPI
// The output gate fused into the epilogue (MoE prefill phase 3): pf_attn.cl's pf_attn_gate, statement for statement -
// out[m][h.256 + d] = rne_bf16(f32(rne_bf16(o)) . sigmoid_f32(f32(rne_bf16(gate)))), the gate read from qkv_partials.
#define QKV_N ((2 * FA_Q_HEADS + 2 * FA_KV_HEADS) * HD)
#define OUT_N (FA_Q_HEADS * HD)
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }
#endif
#ifndef FLASH_MERGE
#define FLASH_MERGE 0
#endif
#ifndef PAIR
#define PAIR 0
#endif
#ifndef WIDE
#define WIDE 0
#endif
#if FA_SKELETON && !PAIR && !WIDE
#error "pf_flash_attn FA_SKELETON: the PAIR or WIDE form only"
#endif
#if WIDE
// WIDE=1 (spec 38, PERF-112): 32 rows of a head over four sub-groups, O split by dim quarters - see the header.
#include "prefill/pf_flash_wide.h"
#elif PAIR
// PAIR=1 (long-context prefill): two sub-groups share 16 query rows of one head, so every K and V operand feeds two
// row blocks' DPAS instead of one (the token stream above reads 68 KB of operands per 128 DPAS a 64-key tile, this
// form 43 KB). Who computes what changes; no value, operand or order does:
//   S       sub-group k of the pair computes the scores of key atoms 2k, 2k + 1 for both row blocks (the same DPAS
//           chain over the 16 dim atoms); the row maxima meet through SLM (barrier 1), mnew = fmax(m, fmax(max of
//           atoms 0-1, max of atoms 2-3)) - a max of the same set.
//   l       psum keeps its order ((p0 + p1) + p2) + p3: sub-group 0 publishes p0 + p1, sub-group 1 adds p2, p3, reduces
//           and owns l; it hands l over once, after the walk.
//   P V     both publish their P atoms in SLM (barrier 2); sub-group k owns dim atoms 8k .. 8k + 7 of O for both row
//           blocks over all four key atoms in order.
// A work-group is HPW pairs (one per q-head of its group) on the same 16 rows, so every sub-group walks the same
// tiles and the barriers are uniform. Row block 0 may walk one tile more than in the 8-row form, wholly masked
// for it: corr 1 and P 0, so its numbers stay the same. Grid (ceil(C / 16), kv-heads, GQA / HPW).
#if KT != 64 || RPW != 16 || SPLIT || QREG || defined(KV_PAGED)
#error "pf_flash_attn PAIR: KT 64, RPW 16, no SPLIT / QREG / KV_PAGED"
#endif
#ifndef PAIR_QSLM
#define PAIR_QSLM 0
#endif
__attribute__((reqd_work_group_size(SG * 2 * HPW, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_flash_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows
#if GATE_EPI
                  , __global const float* restrict G, __global ushort* restrict Out
#endif
                  ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint k = s & 1u, hp = s >> 1;                              // key half / dim half; the pair
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * FA_GQA + get_group_id(2) * HPW + hp;          // q head
  const uint r0 = get_group_id(0) * 16u;                           // first of 16 rows
  if (r0 >= C) return;                                             // uniform over the work-group
  const uint depth = pos + C;
  const int q_w = FA_QH * HD * 2, q_h = (int)C, q_p = FA_QH * HD * 2;
  const int kt_w = FA_KVH * HD * 2, kt_h = (int)depth, kt_p = FA_KVH * HD * 2;
  const int v_w = FA_KVH * HD * 2, v_h = (int)depth, v_p = FA_KVH * HD * 2;
  __local float smax[HPW * 32];      // [pair][k][16 rows]: the tile's row maxima of key half k
  __local float sa[HPW * 256];       // [pair][rb][r * 16 + lane]: sub-group 0's p0 + p1
  __local ushort sp[HPW * 1024];     // [pair][rb * 4 + b][r * 16 + lane]: P, bf16
  __local float sl[HPW * 16];        // [pair][16 rows]: l after the walk
#if PAIR_QSLM
  // PAIR_QSLM=n (probe-only, spec 38a Task 1: does SLM have read bandwidth beside L1's?): Q of dim atoms 0 .. 2n - 1
  // is read once into SLM before the walk and from there every tile; the same values, so bitwise. Unset elsewhere.
  __local ushort sq[HPW * PAIR_QSLM * 512];   // [pair][rb][dim-atom pair][256]
  for (uint i = k; i < 2u * PAIR_QSLM; i += 2u) {
    const uint rb = i / PAIR_QSLM, kp = i % PAIR_QSLM;
    ushort t[16];
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                              (int2)((int)(h * HD + 32u * kp), (int)(r0 + 8u * rb)), t);
    intel_sub_group_block_write_us8(sq + (hp * 2u * PAIR_QSLM + i) * 256u, vload8(0, t));
    intel_sub_group_block_write_us8(sq + (hp * 2u * PAIR_QSLM + i) * 256u + 128u, vload8(0, t + 8));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  float8 o[2][NDA / 2];
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] = (float8)(0.0f);
  float8 m[2] = {(float8)(-INFINITY), (float8)(-INFINITY)}, lsum[2] = {(float8)(0.0f), (float8)(0.0f)};
  const uint last = min(depth, pos + r0 + 16u);
  for (uint t0 = 0; t0 < last; t0 += KT) {
    float8 sacc[2][2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) sacc[rb][0] = sacc[rb][1] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2) {
      short8 qa[2][2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
#if PAIR_QSLM
        if (kk < 2u * PAIR_QSLM) {
          __local const ushort* const qs = sq + (hp * 2u * PAIR_QSLM + rb * PAIR_QSLM + kk / 2u) * 256u;
          qa[rb][0] = as_short8(intel_sub_group_block_read_us8(qs));
          qa[rb][1] = as_short8(intel_sub_group_block_read_us8(qs + 128u));
          continue;
        }
#endif
        ushort t[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                  (int2)((int)(h * HD + 16u * kk), (int)(r0 + 8u * rb)), t);
        qa[rb][0] = as_short8(vload8(0, t));
        qa[rb][1] = as_short8(vload8(0, t + 8));
      }
#pragma unroll
      for (uint kq = 0; kq < 2u; ++kq)
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          uint kb[8];
          intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
              (__global void*)Kc, kt_w, kt_h, kt_p,
              (int2)((int)((j * HD + 16u * (kk + kq)) / 2u), (int)(t0 + 16u * (2u * k + bb))), kb);
#pragma unroll
          for (uint rb = 0; rb < 2u; ++rb)
            sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kq], as_int8(vload8(0, kb)), sacc[rb][bb]);
        }
    }
    // scale, mask, this half's row maxima (component r = row r0 + 8 rb + r; lane = key t0 + 16 (2k + bb) + l)
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      const uint rr = r0 + 8u * rb;
      float8 tmax = (float8)(-INFINITY), red;
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint key = t0 + 16u * (2u * k + bb) + l;
#define MASK(r) sacc[rb][bb].s##r = (key <= pos + rr + r && key < depth) ? sacc[rb][bb].s##r * SCORE_SCALE : -INFINITY;
        MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
        tmax = fmax(tmax, sacc[rb][bb]);
      }
#define RED(r) red.s##r = FA_RMAX(tmax.s##r);
      RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
      if (l == 0) vstore8(red, 0, smax + hp * 32u + k * 16u + 8u * rb);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 corr[2], mnew[2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      mnew[rb] = fmax(m[rb], fmax(vload8(0, smax + hp * 32u + 8u * rb), vload8(0, smax + hp * 32u + 16u + 8u * rb)));
      corr[rb] = EXPF(m[rb] - mnew[rb]);
      float8 part = (float8)(0.0f);
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        sacc[rb][bb] = EXPF(sacc[rb][bb] - mnew[rb]);   // p, kept for sub-group 1's sum
        part += sacc[rb][bb];
        short8 pa;
#define CV(r) pa.s##r = as_short(rne_bf16(sacc[rb][bb].s##r));
        CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
        intel_sub_group_block_write_us8(sp + hp * 1024u + (rb * 4u + 2u * k + bb) * 128u, as_ushort8(pa));
      }
      if (k == 0) intel_sub_group_block_write8((__local uint*)(sa + hp * 256u + rb * 128u), as_uint8(part));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 1) {
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        float8 psum = as_float8(intel_sub_group_block_read8((__local const uint*)(sa + hp * 256u + rb * 128u)));
        psum += sacc[rb][0];
        psum += sacc[rb][1];
        float8 rsum;
#define SUM(r) rsum.s##r = FA_RADD(psum.s##r);
        SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
        lsum[rb] = lsum[rb] * corr[rb] + rsum;
      }
    }
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      m[rb] = mnew[rb];
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] *= corr[rb];
    }
    short8 pa[2][NKA];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
      for (uint b = 0; b < NKA; ++b)
        pa[rb][b] = as_short8(intel_sub_group_block_read_us8(sp + hp * 1024u + (rb * 4u + b) * 128u));
    // O += P V over this sub-group's dim atoms 8k + dd, key atoms in order
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; dd += 2) {
        uint vb[32];
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, v_w, v_h, v_p, (int2)((int)(j * HD + 16u * (8u * k + dd)), (int)(t0 + 16u * b)), vb);
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
#pragma unroll
            for (uint rb = 0; rb < 2u; ++rb)
              o[rb][dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                  pa[rb][b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[rb][dd + c]);
      }
  }
  if (k == 1 && l == 0) {
    vstore8(lsum[0], 0, sl + hp * 16u);
    vstore8(lsum[1], 0, sl + hp * 16u + 8u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    const uint rr = r0 + 8u * rb;
    if (rr >= C) break;                                            // a block wholly past C is not written
    const float8 inv = 1.0f / vload8(0, sl + hp * 16u + 8u * rb);
#if GATE_EPI
    const int g_w = QKV_N * 4, o_w = OUT_N * 2;
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      const uint da = 8u * k + dd;
      const float8 w = o[rb][dd] * inv;
      float8 gt;
      intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)G, g_w, (int)C, g_w,
                                                (int2)((int)(h * 2u * HD + HD + 16u * da), (int)rr), (__private uint*)&gt);
      ushort8 ov;
#define GATE(r) ov.s##r = rne_bf16(bf16f(rne_bf16(w.s##r)) * sigmoid_f32(bf16f(rne_bf16(gt.s##r))));
      GATE(0) GATE(1) GATE(2) GATE(3) GATE(4) GATE(5) GATE(6) GATE(7)
#undef GATE
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Out, o_w, (int)C, o_w,
                                                 (int2)((int)(h * HD + 16u * da), (int)rr), (__private ushort*)&ov);
    }
#else
    __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      float8 w = o[rb][dd] * inv;
      intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                                 (int2)((int)(16u * (8u * k + dd)), (int)rr), (__private uint*)&w);
    }
#endif
  }
}
#elif !FLASH_MERGE
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#if SPLIT
__kernel void pf_flash_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows, __global float* restrict ML, uint span
#ifdef KV_PAGED
                  , __global const uint* restrict pages
#endif
                  ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                  // kv head
  const uint hgs = FA_GQA / HPW;                                   // head groups per kv head
  const uint sp = get_group_id(2) / hgs, nsp = get_num_groups(2) / hgs;   // this span, the spans
  const uint h = j * FA_GQA + (get_group_id(2) % hgs) * HPW + s / (RPW / 8u);  // q head
#else
__kernel void pf_flash_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows
#if GATE_EPI
                  , __global const float* restrict G, __global ushort* restrict Out
#endif
#ifdef KV_PAGED
                  , __global const uint* restrict pages
#endif
                  ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * FA_GQA + get_group_id(2) * HPW + s / (RPW / 8u);  // q head
#endif
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));   // first of 8 rows
  if (r0 >= C) return;                                             // no barriers below
  const uint depth = pos + C;
  const int q_w = FA_QH * HD * 2, q_h = (int)C, q_p = FA_QH * HD * 2;
  const int kt_w = FA_KVH * HD * 2, kt_h = (int)depth, kt_p = FA_KVH * HD * 2;   // K as dwords: 512 wide
  const int v_w = FA_KVH * HD * 2, v_h = (int)depth, v_p = FA_KVH * HD * 2;

  short8 qa[NDA];                      // Q, 8 rows x 256 dims, as 16 A operands
  // t[c*8 + r] = Q[r0 + r][h*256 + 16 (kk + c) + lane], the 8-row form of pf_gemm's A read.
#define LOAD_Q()                                                                      \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; kk += 2) {                            \
    ushort t[16];                                                                     \
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,       \
                                              (int2)((int)(h * HD + 16u * kk), (int)r0), t); \
    qa[kk] = as_short8(vload8(0, t));                                                 \
    qa[kk + 1] = as_short8(vload8(0, t + 8));                                         \
  }
#if QREG
  LOAD_Q();
#endif
  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);          // keys this SG's rows can see
#if SPLIT
  const uint kend = sp + 1u == nsp ? last : sp * span + span;   // span starts are < pos
  for (uint t0 = sp * span; t0 < kend; t0 += KT) {
#else
  for (uint t0 = 0; t0 < last; t0 += KT) {
#endif
#if !QREG
    LOAD_Q();
#endif
#ifdef KV_PAGED
    // The tile's page: its rows from Kp / Vp, the tile's keys at t0 - pgs there, ph rows readable.
    const uint pgs = t0 / KV_PAGED * KV_PAGED;
    const size_t ppr = (size_t)pages[t0 / KV_PAGED] * KV_PAGED * FA_KVH * HD;
    __global const ushort* const Kp = Kc + ppr;
    __global const ushort* const Vp = Vc + ppr;
    const int ph = (int)min(depth - pgs, (uint)KV_PAGED);
#endif
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        uint kb[8];
#ifndef KV_PAGED
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
#else
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kp, kt_w, ph, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 - pgs + 16u * b)), kb);
#endif
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
      }
    // scale, mask, running max per row (component r = row r0 + r; lane = key t0 + 16 b + l)
    float8 tmax = (float8)(-INFINITY);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const uint key = t0 + 16u * b + l;
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && key < depth) ? sacc[b].s##r * SCORE_SCALE : -INFINITY;
      MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
      tmax = fmax(tmax, sacc[b]);
    }
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
    const float8 corr = EXPF(m - mnew);          // m = -INF on the first tile: EXPF(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - mnew);     // masked: EXPF(-INF) = 0
      psum += p;
#define CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));
      CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
    }
    float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
    SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    // O += P V: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI)
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA; dd += 2) {
        uint vb[32];   // 32 keys x 32 dims: [c*16 + jj], c = dim half, jj = key pair
#ifndef KV_PAGED
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, v_w, v_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(t0 + 16u * b)), vb);
#else
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vp, v_w, ph, v_p, (int2)((int)(j * HD + 16u * dd), (int)(t0 - pgs + 16u * b)), vb);
#endif
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
            o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);
      }
  }
#if SPLIT
  // the span's unnormalised O and its m, l (row r0 + r in component r, every lane alike)
  const uint c8 = (C + 7u) & ~7u;
  const size_t slot = (size_t)(sp * FA_QH + h) * c8;
  __global float* Op = O + slot * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd)
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Op, HD * 4, (int)c8, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&o[dd]);
  if (l == 0) {
    vstore8(m, 0, ML + slot + r0);
    vstore8(lsum, 0, ML + (size_t)nsp * FA_QH * c8 + slot + r0);
  }
}
#else
  const float8 inv = 1.0f / lsum;
#if GATE_EPI
  // O / l gated straight into the o_proj input row [C][OUT_N] bf16; pf_o is not written. Rows past C are clipped by
  // the surfaces' height C (the gate reads 0 there and nothing is stored).
  const int g_w = QKV_N * 4, o_w = OUT_N * 2;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    const float8 w = o[dd] * inv;
    float8 gt;
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)G, g_w, (int)C, g_w,
                                              (int2)((int)(h * 2u * HD + HD + 16u * dd), (int)r0), (__private uint*)&gt);
    ushort8 ov;
#define GATE(r) ov.s##r = rne_bf16(bf16f(rne_bf16(w.s##r)) * sigmoid_f32(bf16f(rne_bf16(gt.s##r))));
    GATE(0) GATE(1) GATE(2) GATE(3) GATE(4) GATE(5) GATE(6) GATE(7)
#undef GATE
    intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Out, o_w, (int)C, o_w,
                                               (int2)((int)(h * HD + 16u * dd), (int)r0), (__private ushort*)&ov);
  }
#else
  // O / l, into pf_o's layout [24][rows][256] fp32
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
#endif  // GATE_EPI
}
#endif  // SPLIT
#endif  // !FLASH_MERGE

#if FLASH_MERGE
// pf_flash_merge(Op, ML, O, C, rows, n): grid (C, q-heads), one work-item a dim. GATE_EPI (Ornith): pf_attn_gate's
// statement on the merged value, into the o_proj input row Out [C][OUT_N]; O is not written.
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void pf_flash_merge(__global const float* restrict Op, __global const float* restrict ML,
                             __global float* restrict O, uint C, uint rows, uint nsp
#if GATE_EPI
                             , __global const float* restrict G, __global ushort* restrict Out
#endif
                             ) {
  const uint r = get_group_id(0), h = get_group_id(1), d = get_local_id(0);
  const uint c8 = (C + 7u) & ~7u;
  const size_t stride = (size_t)FA_QH * c8, at = (size_t)h * c8 + r;
  float mx = -INFINITY;
  for (uint sp = 0; sp < nsp; ++sp) mx = fmax(mx, ML[sp * stride + at]);
  float acc = 0.0f, lsum = 0.0f;
  for (uint sp = 0; sp < nsp; ++sp) {
    const float w = EXPF(ML[sp * stride + at] - mx);
    lsum += w * ML[(nsp + sp) * stride + at];
    acc += w * Op[(sp * stride + at) * HD + d];
  }
#if GATE_EPI
  const float ov = acc * (1.0f / lsum);
  const float g = bf16f(rne_bf16(G[(size_t)r * QKV_N + (size_t)h * 2 * HD + HD + d]));
  Out[(size_t)r * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(ov)) * sigmoid_f32(g));
#else
  O[((size_t)h * rows + r) * HD + d] = acc * (1.0f / lsum);
#endif
}
#endif  // FLASH_MERGE
