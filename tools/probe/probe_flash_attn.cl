// probe_flash_attn - spec 6 P1: fused bf16 flash attention, one arm per -D set.
// Contract and layouts: docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md.
// Record: docs/probe-flash-attn-2026-09-25.md.
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
// ATTN_SCALE; the mask covers key <= pos + row AND key < depth; fp32 online softmax
// with `exp`, never `native_exp`; P = exp(s - m) rounded to bf16 as the PV operand;
// O / l at the end.
//
// PROBE-ONLY: no runtime path binds `pfa`.
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
#define SG 16
#define HD 256
#define NDA (HD / 16)                 /* 16 dim-atoms of O */
#define NKA (KT / 16)                 /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)           /* sub-groups per work-group */
#define ATTN_SCALE (1.0f / 16.0f)

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pfa(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * 6u + get_group_id(2) * HPW + s / (RPW / 8u);  // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));   // first of 8 rows
  if (r0 >= C) return;                                             // no barriers below
  const uint depth = pos + C;
  const int q_w = 24 * HD * 2, q_h = (int)C, q_p = 24 * HD * 2;
  const int kt_w = 4 * HD * 2, kt_h = (int)depth, kt_p = 4 * HD * 2;   // K as dwords: 512 wide
  const int v_w = 4 * HD * 2, v_h = (int)depth, v_p = 4 * HD * 2;

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
  for (uint t0 = 0; t0 < last; t0 += KT) {
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
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && key < depth) ? sacc[b].s##r * ATTN_SCALE : -INFINITY;
      MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
      tmax = fmax(tmax, sacc[b]);
    }
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
    const float8 corr = exp(m - mnew);          // m = -INF on the first tile: exp(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - mnew);     // masked: exp(-INF) = 0
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
  // O / l, into pf_o's layout [24][rows][256] fp32
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
}
