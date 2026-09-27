// pfa_levers_research - the lever arms of docs/research-flash-prefill-2026-09-27.md.
// PROBE-ONLY and NOT BUILT: no CMake target, no runtime binding. Same entry (`pfa`),
// arguments, layouts and launch as tools/probe/probe_flash_attn.cl, so it runs through
// the unchanged probe_flash_attn harness: copy it over probe_flash_attn.cl and list the
// arms ("KT RPW HPW QREG") in tools/probe/CMakeLists.txt's PFA foreach and in `kArms`.
// QREG here is a LEVER BITMASK, not "Q held in registers" (Q is always re-read per tile):
//   bit0 DM   per-element causal mask only on tiles that cross the diagonal / depth
//   bit1 E2   native_exp2 with ATTN_SCALE*log2(e) folded (IGC emits one math.exp; `exp`
//             emits two plus a range-split sequence, which is what spills 77 GRF)
//   bit2 LR   FA-4 conditional O rescale (only when some row max grew by > 8, log2 units)
//   bit3 CVT  hardware float->bf16 RNE (intel_convert_bfloat16_as_ushort)
//   bit4 RV   reverse row-block order (heaviest causal work dispatched first)
//   bit5 DS   deferred row sums (per-lane partials, one reduction after the loop)
//   bit6 KL   K via transpose_32b_32r8x1c; its register layout is NOT the one assumed
//             below -- measured cos 0.15..0.24, FAIL. Kept only as that record.
// Measured winner: KT=64 RPW=8 HPW=6 QREG=2 (exp2 + 6-sub-group work-groups).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef QREG
#define QREG 0
#endif
#define DM ((QREG >> 0) & 1)
#define E2 (((QREG >> 1) & 1) ? 2 : 0)
#define LR ((QREG >> 2) & 1)
#define CVT ((QREG >> 3) & 1)
#define RV ((QREG >> 4) & 1)
#define DS ((QREG >> 5) & 1)
#define KL ((QREG >> 6) & 1)
#define QS 0
#define SG 16
#define HD 256
#define NDA (HD / 16)
#define NKA (KT / 16)
#define SGS (HPW * RPW / 8)
#define ATTN_SCALE (1.0f / 16.0f)
#define LOG2E 1.4426950408889634f
#define LRT 8.0f

inline ushort rne_bf16(float f) {
#if CVT
  return intel_convert_bfloat16_as_ushort(f);
#else
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
#endif
}
#if E2 == 0
#define EXPF(x) exp(x)
#define SC ATTN_SCALE
#elif E2 == 1
#define EXPF(x) exp2(x)
#define SC (ATTN_SCALE * LOG2E)
#else
#define EXPF(x) native_exp2(x)
#define SC (ATTN_SCALE * LOG2E)
#endif

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pfa(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);
#if RV
  const uint gx = get_num_groups(0) - 1u - get_group_id(0);
#else
  const uint gx = get_group_id(0);
#endif
  const uint h = j * 6u + get_group_id(2) * HPW + s / (RPW / 8u);
  const uint r0 = gx * RPW + 8u * (s % (RPW / 8u));
  if (r0 >= C) return;
  const uint depth = pos + C;
  const int q_w = 24 * HD * 2, q_h = (int)C, q_p = 24 * HD * 2;
  const int kt_w = 4 * HD * 2, kt_h = (int)depth, kt_p = 4 * HD * 2;
  const int v_w = 4 * HD * 2, v_h = (int)depth, v_p = 4 * HD * 2;

  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);
  for (uint t0 = 0; t0 < last; t0 += KT) {
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#if QS
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2) {
      short8 q0, q1;
      {
        ushort t[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                  (int2)((int)(h * HD + 16u * kk), (int)r0), t);
        q0 = as_short8(vload8(0, t));
        q1 = as_short8(vload8(0, t + 8));
      }
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        uint kb[8], kc[8];
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * (kk + 1)) / 2u), (int)(t0 + 16u * b)), kc);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(q0, as_int8(vload8(0, kb)), sacc[b]);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(q1, as_int8(vload8(0, kc)), sacc[b]);
      }
    }
#else
    short8 qa[NDA];
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2) {
      ushort t[16];
      intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                (int2)((int)(h * HD + 16u * kk), (int)r0), t);
      qa[kk] = as_short8(vload8(0, t));
      qa[kk + 1] = as_short8(vload8(0, t + 8));
    }
#if KL
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; b += 2) {
        uint kb[16];   // assumed: kb[0..7] keys +0..15, kb[8..15] keys +16..31
        intel_sub_group_2d_block_read_transpose_32b_32r8x1c(
            (__global void*)Kc, kt_w, kt_h, kt_p,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
        sacc[b + 1] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(1, kb)), sacc[b + 1]);
      }
#else
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
#endif
#endif
    float8 tmax = (float8)(-INFINITY);
#if DM
    const bool edge = (t0 + KT > pos + r0 + 1u) || (t0 + KT > depth);  // SG-uniform
    if (edge) {
#endif
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        const uint key = t0 + 16u * b + l;
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && key < depth) ? sacc[b].s##r * SC : -INFINITY;
        MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
        tmax = fmax(tmax, sacc[b]);
      }
#if DM
    } else {
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        sacc[b] *= SC;
        tmax = fmax(tmax, sacc[b]);
      }
    }
#endif
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
#if LR
    // FA-4 style: keep the stale max unless some row grew by more than LRT (log2 units
    // when E2 > 0). P <= 2^LRT then, harmless in fp32 and bf16. Uniform over the SG.
    const int8 grow = isgreater(mnew, m + (float8)(LRT));
    const int anyg = grow.s0 | grow.s1 | grow.s2 | grow.s3 | grow.s4 | grow.s5 | grow.s6 | grow.s7;
    const bool resc = sub_group_any(anyg != 0) || t0 == 0;
    if (!resc) mnew = m;
#endif
    const float8 corr = EXPF(m - mnew);
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - mnew);
      psum += p;
#define CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));
      CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
    }
#if DS
    const float8 rsum = psum;          // per-lane partial; reduced once after the loop
#else
    float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
    SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
#endif
#if LR
    if (resc) {
      lsum = lsum * corr + rsum;
#pragma unroll
      for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    } else {
      lsum += rsum;
    }
#else
    lsum = lsum * corr + rsum;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
#endif
    m = mnew;
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA; dd += 2) {
        uint vb[32];
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
#if DS
#define SUM(r) lsum.s##r = sub_group_reduce_add(lsum.s##r);
  SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
#endif
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
}
