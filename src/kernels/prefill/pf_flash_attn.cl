// pf_flash_attn - spec 6: the fused bf16 flash attention of prefill.
// Spec: docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md.
// Plan: docs/superpowers/plans/2026-09-25-spec6b-flash-attn-integration-and-128k.md.
// Record: docs/probes/probe-flash-attn-2026-09-25.md. This is plan 6a's winning arm
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
// never `native_exp` (spec 6 §3.1). EXP2=1 is the research form the operator approved on
// 2026-09-27: scores times ATTN_SCALE * log2(e), so s, the running max m and the masked
// -INF all live in the log2 domain, and `exp2` (one base-2 math.exp in IGC; `exp`'s
// range-split temporaries are what spill 77 GRF). EXP2=1 moves the rounding point and
// measured K3a l0 0.999951245 < composed 0.999959889 (spec 6 §9); the operator enabled
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

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}

#ifndef SPLIT
#define SPLIT 0
#endif
#ifndef GATE_EPI
#define GATE_EPI 0
#endif
#if GATE_EPI
// The output gate fused into the epilogue (MoE prefill phase 3): pf_attn.cl's pf_attn_gate, statement for statement -
// out[m][h.256 + d] = rne_bf16(f32(rne_bf16(o)) . sigmoid_f32(f32(rne_bf16(gate)))), the gate read from qkv_partials.
#define QKV_N ((2 * FA_Q_HEADS + 2 * FA_KV_HEADS) * HD)
#define OUT_N (FA_Q_HEADS * HD)
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }
#endif
#ifndef FLASH_MERGE
#define FLASH_MERGE 0
#endif
#if !FLASH_MERGE
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#if SPLIT
__kernel void pf_flash_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows, __global float* restrict ML, uint span) {
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
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        uint kb[8];
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
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
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, v_w, v_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(t0 + 16u * b)), vb);
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
// pf_flash_merge(Op, ML, O, C, rows, n): grid (C, q-heads), one work-item a dim.
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void pf_flash_merge(__global const float* restrict Op, __global const float* restrict ML,
                             __global float* restrict O, uint C, uint rows, uint nsp) {
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
  O[((size_t)h * rows + r) * HD + d] = acc * (1.0f / lsum);
}
#endif  // FLASH_MERGE
