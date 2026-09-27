// pf_bf16_slab.cl - one 1024-column slab of a bf16 linear, untiled for pf_gemm (spec 8,
// plan 8b Task 3: the MTP head's KV fill during prefill).
//
//   pf_bf16_slab(w, out, n0)
//     w    bf16, gemv_bf16's tiled layout (common::repack_bf16_tiled):
//          w[((n/16) * K/8 + k/8) * 128 + (k%8) * 16 + n%16] = W[n][k]
//     out  bf16 [K][1024] row-major: out[k][j] = W[n0 + j][k] -- pf_gemm's B operand, the
//          same slab pf_dequant_slab writes for an int4 linear
//     n0   the slab's first column, a multiple of 1024
//
// Grid (1024 / 16 column tiles, K / 8), one subgroup of 16 per work-group: work-group
// (t, k8) block-reads the 128-element block of tile n0/16 + t at k8 - lane l gets
// W[n][k8*8 + i] for its column n in element i - and scatters it to 8 slab rows. A pure
// copy: the slab is the weight's bits.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef K
#error "pf_bf16_slab: K must be defined"
#endif
#define NS 1024
#define SG 16
#define K8 (K / 8)

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1)))
__kernel void pf_bf16_slab(__global const ushort* restrict w, __global ushort* restrict out,
                           const uint n0) {
  const uint lane = get_sub_group_local_id();
  const uint t = get_group_id(0);
  const uint k8 = get_group_id(1);
  const uint nt = n0 / SG + t;
  const ushort8 v = intel_sub_group_block_read_us8(w + ((size_t)nt * K8 + k8) * 128);
  __global ushort* restrict o = out + (size_t)(k8 * 8) * NS + t * SG + lane;
  o[0 * NS] = v.s0;
  o[1 * NS] = v.s1;
  o[2 * NS] = v.s2;
  o[3 * NS] = v.s3;
  o[4 * NS] = v.s4;
  o[5 * NS] = v.s5;
  o[6 * NS] = v.s6;
  o[7 * NS] = v.s7;
}
