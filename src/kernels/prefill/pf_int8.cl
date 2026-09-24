// pf_int8 - spec 5's int8 prefill linears
// (docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md): the h8 path of
// docs/probe-w4a8-2026-09-23.md sections 14-15, promoted from
// tools/probe/probe_w8a8.cl with the arithmetic unchanged. One source, built
// per family by a define:
//   PF_QUANT_HAD  (NBLK)       pf_quant_had    bf16 x -> int8 (x R_K), per-token scale
//   PF_REQUANT_ROT (LAYOUT)    pf_requant_rot  int4 g64 -> per-channel int8 of (W R_K)
//                              pf_colmax_rot   max |W R_K| per column (load-time scales)
//   PF_GEMM_I8   (SILU_EPI)    pf_gemm_i8      i8 x i8, scales in the epilogue only
// R_K = D_K blockdiag(H_1024) / 32 over K. D_K comes from
// runtime/prefill/int8_signs.h. The quantiser holds block element 16 j + lane
// and the requant holds 64 lane + j; both run the canonical Sylvester FWHT on
// the natural index with ascending stages, so they rotate identically
// (pf_int8_test's cross case pins it).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

#ifdef PF_QUANT_HAD
#ifndef NBLK
#error "pf_quant_had: NBLK (= K / 1024) must be defined"
#endif
// One sub-group per 1024-block, NBLK sub-groups per token row. Lane l's value j
// is element 16 j + l of the block (sub-group block reads). FWHT stages
// h = 1..8 are lane shuffles and 16..512 register butterflies, ascending.
__attribute__((reqd_work_group_size(16 * NBLK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_quant_had(__global const ushort* restrict x, __global const float* restrict sgn,
                           __global char* restrict xq, __global float* restrict xs, uint ldx) {
  const uint K = NBLK * 1024u;
  const uint m = get_group_id(0);
  const uint b = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  __global const ushort* src = x + (size_t)m * ldx + b * 1024u;
  __local float red[NBLK];
  float v[64];
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    const ushort8 u = intel_sub_group_block_read_us8(src + r * 128u);
    const float8 sg = as_float8(intel_sub_group_block_read8(
        (__global const uint*)(sgn + b * 1024u + r * 128u)));
    v[8u * r + 0u] = bf16f(u.s0) * sg.s0; v[8u * r + 1u] = bf16f(u.s1) * sg.s1;
    v[8u * r + 2u] = bf16f(u.s2) * sg.s2; v[8u * r + 3u] = bf16f(u.s3) * sg.s3;
    v[8u * r + 4u] = bf16f(u.s4) * sg.s4; v[8u * r + 5u] = bf16f(u.s5) * sg.s5;
    v[8u * r + 6u] = bf16f(u.s6) * sg.s6; v[8u * r + 7u] = bf16f(u.s7) * sg.s7;
  }
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
#pragma unroll
  for (uint hj = 1u; hj < 64u; hj <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & hj) == 0u) {
        const float a = v[j], c = v[j + hj];
        v[j] = a + c;
        v[j + hj] = a - c;
      }
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 64u; ++j) {
    v[j] *= (1.0f / 32.0f);
    amax = fmax(amax, fabs(v[j]));
  }
  amax = sub_group_reduce_max(amax);
  if (l == 0u) red[b] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax = 0.0f;
  for (uint i = 0; i < (uint)NBLK; ++i) amax = fmax(amax, red[i]);
  const float scale = amax / 127.0f;
  const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
  if (b == 0u && l == 0u) xs[m] = scale;
  __global uchar* dst = (__global uchar*)(xq + (size_t)m * K + b * 1024u);
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    uchar8 o;
    o.s0 = (uchar)convert_char_sat_rte(v[8u * r + 0u] * iv);
    o.s1 = (uchar)convert_char_sat_rte(v[8u * r + 1u] * iv);
    o.s2 = (uchar)convert_char_sat_rte(v[8u * r + 2u] * iv);
    o.s3 = (uchar)convert_char_sat_rte(v[8u * r + 3u] * iv);
    o.s4 = (uchar)convert_char_sat_rte(v[8u * r + 4u] * iv);
    o.s5 = (uchar)convert_char_sat_rte(v[8u * r + 5u] * iv);
    o.s6 = (uchar)convert_char_sat_rte(v[8u * r + 6u] * iv);
    o.s7 = (uchar)convert_char_sat_rte(v[8u * r + 7u] * iv);
    intel_sub_group_block_write_uc8(dst + r * 128u, o);
  }
}
#endif  // PF_QUANT_HAD
