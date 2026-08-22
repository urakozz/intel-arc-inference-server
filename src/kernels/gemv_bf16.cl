// gemv_bf16.cl - bf16 weights x bf16 activations, M in [1,8]. Used for lm_head
// (N = 248320, 2.5 GB read per token) and the padded a||b projection (N = 128).
// Same lane-per-n structure as gemv.cl. Weights are in the canonical bf16 tile
// layout [n_tile][k_octet][8 k][16 n], so one intel_sub_group_block_read_us8
// per 8 k streams 256 contiguous bytes per subgroup. No split-K: lm_head has
// 15520 subgroups and fills the device on N alone.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define K8 (K / 8)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv_bf16(__global const ushort* restrict w,
                        __global const ushort* restrict x,
                        __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  __global const ushort* tile = w + (size_t)n_tile * K8 * 128;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint k8 = 0; k8 < K8; ++k8, tile += 128) {
    ushort8 wv = intel_sub_group_block_read_us8(tile);
    for (int m = 0; m < M; ++m) {
      ushort8 xv = vload8(0, x + (size_t)m * K + k8 * 8);
      float a = acc[m];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[m] = a;
    }
  }
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m];
}
