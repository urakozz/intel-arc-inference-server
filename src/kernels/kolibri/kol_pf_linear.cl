// kol_pf_linear.cl - spec 20d: one bf16 slab of a Kolibri-1 bf16 attention linear (decision 2's bf16 arm)
// for the prefill GEMM (pf_gemm.cl's pf_gemm_T0, unchanged). The int4 arm's slabs are 18c's
// k2_pf_dequant_slab at Kolibri's shapes (layout 0, the same contract); this is its bf16 twin.
//
// Why not prefill/pf_bf16_slab.cl (spec 8's): it writes exactly NS = 1024 columns from a multiple of 1024,
// and o_proj is N 2560 = 1024 + 1024 + 512 - its last slab would read 512 columns past the weight. This
// is that kernel's copy with k2_pf_dequant_slab's two changes:
//
//   1. the slab width `ns` is a runtime argument (the output pitch; the grid is ns / 16), so a linear is
//      walked in slabs of 1024 and ONE tail slab of pad256(N - n0) columns (runtime::kolibri::pf_slab_width);
//   2. a column n >= N (the tail's padding up to the GEMM's 256-column tile) is written as exact zeros,
//      so the GEMM's padded output columns are exact zeros and land in the partials row's own padding.
//
//   kol_pf_bf16_slab(w, out, n0, ns)    grid (ns / 16, K / 8), WG 16
//     w    bf16, gemv_bf16's tiled layout (common::repack_bf16_tiled):
//          w[((n/16) * K/8 + k/8) * 128 + (k%8) * 16 + n%16] = W[n][k]
//     out  bf16 [K][ns] row-major: out[k][c] = W[n0 + c][k] (0 for n0 + c >= N) - pf_gemm's B operand
//
// A pure copy: the slab is the weight's bits. One sub-group block read under cl_intel_subgroups (as
// pf_bf16_slab.cl), the plain loads it equals otherwise - so it runs on the Mac (tools/mac/clrun).
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define KOL_SG16 __attribute__((intel_reqd_sub_group_size(16)))
#else
#define KOL_SG16
#endif
#ifndef K
#error "kol_pf_bf16_slab: K must be defined"
#endif
#ifndef N
#error "kol_pf_bf16_slab: N must be defined"
#endif
#define SG 16
#define K8 (K / 8)
#if K % 8 != 0 || N % SG != 0
#error "kol_pf_bf16_slab: K must be whole k-octets, N whole 16-column tiles"
#endif

__attribute__((reqd_work_group_size(SG, 1, 1)))
KOL_SG16
__kernel void kol_pf_bf16_slab(__global const ushort* restrict w, __global ushort* restrict out,
                               const uint n0, const uint ns) {
  const uint lane = get_local_id(0);
  const uint t = get_group_id(0);               // 0 .. ns/16 - 1, within the slab
  const uint k8 = get_group_id(1);
  const uint n_local = t * SG + lane;
  __global ushort* restrict o = out + (size_t)(k8 * 8) * ns + n_local;
  if (n0 + t * SG >= N) {                       // uniform per work-group (N % 16 == 0)
    for (uint i = 0; i < 8; ++i) o[(size_t)i * ns] = (ushort)0;
    return;
  }
  const uint nt = n0 / SG + t;
  const __global ushort* src = w + ((size_t)nt * K8 + k8) * 128;
#ifdef cl_intel_subgroups
  const ushort8 v = intel_sub_group_block_read_us8(src);
#else
  const ushort8 v = (ushort8)(src[0 * SG + lane], src[1 * SG + lane], src[2 * SG + lane], src[3 * SG + lane],
                              src[4 * SG + lane], src[5 * SG + lane], src[6 * SG + lane], src[7 * SG + lane]);
#endif
  o[0 * (size_t)ns] = v.s0;
  o[1 * (size_t)ns] = v.s1;
  o[2 * (size_t)ns] = v.s2;
  o[3 * (size_t)ns] = v.s3;
  o[4 * (size_t)ns] = v.s4;
  o[5 * (size_t)ns] = v.s5;
  o[6 * (size_t)ns] = v.s6;
  o[7 * (size_t)ns] = v.s7;
}
