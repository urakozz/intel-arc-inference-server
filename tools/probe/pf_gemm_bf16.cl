// pf_gemm_bf16 (P-A). PROBE-ONLY: our own OpenCL C bf16 DPAS GEMM on the
// Level Zero list -- the go/no-go candidate for dropping sycl-tla from the
// prefill path. Design pre-registered BEFORE this file was written:
// docs/probe-prefill-vllm-parity-2026-09-14.md addendum §A4. No runtime path
// binds this kernel and no existing kernel is touched.
//
// C[M][N] fp32 = A[M][K] bf16 . B[K][N] bf16 (lda = K, ldb = N, ldc = N) --
// gemm_bf16's exact contract (src/runtime/prefill/gemm.h) and the layout
// pf_dequant_tile already writes (src/kernels/prefill/dequant.cl, LAYOUT 0 /
// TRANSPOSED 0): no repack, no second copy, no producer changes.
//
// Work-group tile 256(m) x 256(n) x 32(k); 32 subgroups in an 8(m) x 4(n)
// arrangement; subgroup tile 32(m) x 64(n) x 32(k); DPAS atom 8x16x16
// (intel_sub_group_bf16_bf16_matrix_mad_k16 -> dpas.8x8, M=8 K=16 N=16,
// sub-group 16). These are sycl-tla's own numbers (xe_gemm_config.h,
// MainloopXeL1Staged<2>), kept because they are the only tile whose rate on
// this part is measured (§A4.2 of the addendum argues each choice).
//
// Requirements, met by every production shape (derived, §A4.1): M % 256 == 0,
// N % 256 == 0, K % 32 == 0; every pitch (K*2, N*2, N*4 bytes) a multiple of
// 64 B. A tail-masked variant is a production item, not part of this probe.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16    // sub-group size
#define WG_M 256 // work-group tile, m
#define WG_N 256 // work-group tile, n
#define WG_K 32  // work-group tile, k
#define SG_ROWS 8  // subgroups along m
#define SG_COLS 4  // subgroups along n

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm_bf16(__global const ushort* restrict A,
                           __global const ushort* restrict B,
                           __global float* restrict C,
                           uint M, uint K, uint N) {
  const uint gx = get_group_id(0);  // M axis: consecutive ids share a B column-block
  const uint gy = get_group_id(1);  // N axis
  const uint m0 = gx * WG_M;
  const uint n0 = gy * WG_N;
  const uint s = get_sub_group_id();  // 0..31
  const uint sm = s >> 2;             // 0..7: subgroup row, owns m0+32*sm .. +31
  const uint sn = s & 3u;             // 0..3: subgroup col, owns n0+64*sn .. +63

  const uint lda2 = K * 2u, ldb2 = N * 2u, ldc4 = N * 4u;
  const uint num_k_tiles = K / WG_K;

  // C accumulators: 32(m) x 64(n) fp32 per subgroup, 16 lanes -> 128
  // elements/lane = 16 float8. acc[a][b]: a = m-atom (0..3, 8 rows each),
  // b = n-atom (0..3, 16 cols each).
  float8 acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);

  // --- prologue: prefetch k-tiles 0 and 1 (2-deep cooperative prefetch) ----
  // A: 256 rows x 32 k -- subgroup s takes 8 rows (8s .. 8s+7).
  // B: 32 k-rows x 256 n -- subgroup s takes 8 rows within a 32-column strip.
#pragma unroll
  for (int pf = 0; pf < 2; ++pf) {
    const int k_pf = pf * WG_K;
    intel_sub_group_2d_block_prefetch_16b_8r16x2c(
        (__global void*)A, (int)lda2, (int)M, (int)lda2,
        (int2)(k_pf, (int)(m0 + 8u * s)));
    intel_sub_group_2d_block_prefetch_16b_8r16x2c(
        (__global void*)B, (int)ldb2, (int)K, (int)ldb2,
        (int2)((int)(n0 + 32u * (s & 7u)), k_pf + (int)(8u * (s >> 3))));
  }

  for (uint t = 0; t < num_k_tiles; ++t) {
    const uint k0 = t * WG_K;

    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    // --- load A: 32(m) x 32(k), one message ---
    // afrag[c*32 + r] = A[m0+32sm+r][k0+16c+lane]: lane = k column,
    // register = m row (sycl-tla's XE_LOAD_2D<16,32,32,16>).
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c(
        (__global void*)A, (int)lda2, (int)M, (int)lda2,
        (int2)((int)k0, (int)(m0 + 32u * sm)), afrag);

    // --- load B: 32(k) x 64(n), two VNNI-transform messages ---
    // bfrag[h][c*16+j] = (B[k0+2j+1][n0+64sn+32h+16c+lane] << 16) |
    //                     B[k0+2j][...] -- even k in the low half.
    uint bfrag[2][32];
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)B, (int)ldb2, (int)K, (int)ldb2,
        (int2)((int)(n0 + 64u * sn), (int)k0), bfrag[0]);
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)B, (int)ldb2, (int)K, (int)ldb2,
        (int2)((int)(n0 + 64u * sn + 32u), (int)k0), bfrag[1]);

    // --- prefetch k-tile t+2, if it exists ---
    if (t + 2 < num_k_tiles) {
      const int k_pf = (int)((t + 2) * WG_K);
      intel_sub_group_2d_block_prefetch_16b_8r16x2c(
          (__global void*)A, (int)lda2, (int)M, (int)lda2,
          (int2)(k_pf, (int)(m0 + 8u * s)));
      intel_sub_group_2d_block_prefetch_16b_8r16x2c(
          (__global void*)B, (int)ldb2, (int)K, (int)ldb2,
          (int2)((int)(n0 + 32u * (s & 7u)), k_pf + (int)(8u * (s >> 3))));
    }

    // --- 32 dpas per k-tile, b-outer a-inner: consecutive dpas share the B
    // operand (src1), so IGC can fuse four at a time into its DPAS macro.
    // The short8 for (m-atom a, k-step ks): afrag[ks*32 + 8a .. +7].
    // The int8 for (n-atom b = 2h+c, k-step ks): bfrag[h][c*16 + 8ks .. +7].
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[4];
#pragma unroll
      for (int a = 0; a < 4; ++a)
        af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int h = b >> 1, c = b & 1;
        const int8 bv = as_int8(vload8(0, bfrag[h] + (uint)(c * 16 + 8 * ks)));
#pragma unroll
        for (int a = 0; a < 4; ++a)
          acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }

    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }

  // --- epilogue: 16 stores per subgroup, 8 rows x 64 B each, no C read ---
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c(
          (__global void*)C, (int)ldc4, (int)M, (int)ldc4,
          (int2)((int)(n0 + 64u * sn + 16u * b), (int)(m0 + 32u * sm + 8u * a)),
          (__private uint*)&acc[a][b]);
}
