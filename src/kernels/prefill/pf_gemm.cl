// pf_gemm - the prefill GEMM on the Level Zero list (spec 2.1 §3.1). It is probe P-A's
// pf_gemm_bf16 (tools/probe/pf_gemm_bf16.cl, measured 156.53 TFLOP/s and bitwise equal to
// sycl-tla, docs/probe-prefill-vllm-parity-2026-09-14.md "Phase 2 results") with the tile,
// the 32-dpas k-tile body, the 2D block loads, the 2-deep prefetch and the split barrier
// UNCHANGED, generalised in three ways:
//   * lda / ldb / ldc are runtime arguments (the probe fixed them to K, N, N);
//   * a batch index l = get_group_id(2) offsets A, B and C by strideA/B/C elements
//     (one launch per batched GEMM, as sycl-tla's gemm_bf16_batched);
//   * TRANSB=1 reads B stored [N][K] (K contiguous: the KV cache) with transposed 32-bit
//     loads instead of the VNNI-transform loads of a [K][N] B.
//
//   C[l][m][n] fp32 = Σ_k A[l][m][k] · B[l][k][n],   0 <= m < M, 0 <= n < N, 0 <= k < K
//   A: bf16, element (m, k) at A + l·strideA + m·lda + k
//   B: bf16, TRANSB 0: (k, n) at B + l·strideB + k·ldb + n ;  TRANSB 1: (k, n) at B + l·strideB + n·ldb + k
//   C: fp32, (m, n) at C + l·strideC + m·ldc + n
//
// Requirements (runtime::prefill::gemm_l0 asserts every one before launching):
//   M % 256 == 0 and N % 256 == 0 -- the CALLER pads (spec §3.1: the padding is computed
//   and never read); K % 32 == 0; every base (after the batch offset) 64-byte aligned;
//   lda·2, ldb·2, ldc·4 multiples of 16 bytes; lda >= K; ldb >= N (T0) or >= K (T1);
//   ldc >= N. The `width` of every 2D block descriptor below is the LOGICAL matrix width
//   in bytes and the `pitch` is the leading dimension, so a read never clamps inside the
//   matrix and never reaches past it.
//
// TRANSB's B fragment. The DPAS B operand for one k-step is, per lane n, eight 32-bit words
// holding (B[2t][n], B[2t+1][n]) pairs, even k in the low half. With B stored [N][K]
// (K contiguous) those pairs are ADJACENT 16-bit words, so the [N][K] surface read as
// 32-bit [N][K/2] already holds every operand word; a TRANSPOSED 2D block load
// (intel_sub_group_2d_block_read_transpose_32b_16r8x1c: 16 rows x 8 dwords, lane = row =
// n, registers = dword columns = k-pairs) delivers lane n's 8 k-pairs = one k half-tile of
// one n-atom. Eight of them (four n-atoms x two k-halves) cover the subgroup's 64 n for a
// whole 32-k tile, against two transform loads on the T0 side. (S0's build gate found the
// spec's originally-assumed 16-dword-wide read, `..._16r16x1c`, undeclared on this driver;
// only the 8-dword row is exposed, so each n-atom's fragment is two 8-dword loads instead
// of one 16-dword load -- same coverage, same k order.) Same dpas instructions, same k
// order, so the two variants are expected to be bitwise equal (spec §8 risk 1 if not).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#ifndef TRANSB
#error "pf_gemm: TRANSB must be defined (0: B is [K][N]; 1: B is [N][K])"
#endif
#define SG 16
#define WG_M 256
#define WG_N 256
#define WG_K 32

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm(__global const ushort* restrict A_base,
                      __global const ushort* restrict B_base,
                      __global float* restrict C_base,
                      uint M, uint K, uint N, uint lda, uint ldb, uint ldc,
                      ulong strideA, ulong strideB, ulong strideC) {
  const uint gx = get_group_id(0);   // M axis: consecutive ids share a B column block
  const uint gy = get_group_id(1);   // N axis
  const ulong l = get_group_id(2);   // batch
  __global const ushort* restrict A = A_base + l * strideA;
  __global const ushort* restrict B = B_base + l * strideB;
  __global float* restrict C = C_base + l * strideC;
  const uint m0 = gx * WG_M;
  const uint n0 = gy * WG_N;
  const uint s = get_sub_group_id();   // 0..31
  const uint sm = s >> 2;              // 0..7: rows m0 + 32 sm .. +31
  const uint sn = s & 3u;              // 0..3: cols n0 + 64 sn .. +63

  // 2D block descriptors: (width bytes, height rows, pitch bytes).
  const int a_w = (int)(K * 2u), a_h = (int)M, a_p = (int)(lda * 2u);
#if TRANSB
  const int b_w = (int)(K * 2u), b_h = (int)N, b_p = (int)(ldb * 2u);   // [N][K] seen as [N][K/2] dwords
#else
  const int b_w = (int)(N * 2u), b_h = (int)K, b_p = (int)(ldb * 2u);   // [K][N]
#endif
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);
  const uint num_k_tiles = K / WG_K;

  float8 acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);

  // --- prologue: prefetch k-tiles 0 and 1. A: subgroup s takes rows m0+8s..+7 of the
  // 256 x 32 tile. B (T0): 8 k-rows x 32 n per subgroup over the 32 x 256 tile.
  // B (T1): the tile is 256 n-rows x 32 k; subgroup s takes rows n0+8s..+7.
#pragma unroll
  for (int pf = 0; pf < 2; ++pf) {
    const int k_pf = pf * WG_K;
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p,
                                                  (int2)(k_pf, (int)(m0 + 8u * s)));
#if TRANSB
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)B, b_w, b_h, b_p,
                                                  (int2)(k_pf, (int)(n0 + 8u * s)));
#else
    intel_sub_group_2d_block_prefetch_16b_8r16x2c(
        (__global void*)B, b_w, b_h, b_p,
        (int2)((int)(n0 + 32u * (s & 7u)), k_pf + (int)(8u * (s >> 3))));
#endif
  }

  for (uint t = 0; t < num_k_tiles; ++t) {
    const uint k0 = t * WG_K;
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    // A: 32 m x 32 k, one message. afrag[c*32 + r] = A[m0+32sm+r][k0+16c+lane].
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)A, a_w, a_h, a_p,
                                               (int2)((int)k0, (int)(m0 + 32u * sm)), afrag);

#if TRANSB
    // B: eight transposed dword loads, two per n-atom b. S0's build gate found
    // intel_sub_group_2d_block_read_transpose_32b_16r16x1c undeclared on this driver
    // (ocloc: "did you mean intel_sub_group_2d_block_read_transpose_32b_16r8x1c?" --
    // only the 8-dword-wide row is exposed); each n-atom's 16-dword fragment is instead
    // two 8-dword transposed loads, one per k half-tile ks -- exactly the split the dpas
    // loop below already reads by (same k order, same lane, so still bitwise-equality
    // candidate; spec §8 risk 1). bfrag[b][8ks + j] holds, for lane n = n0+64sn+16b+lane,
    // the pair (B[k0+16ks+2j][n], B[k0+16ks+2j+1][n]), even k low.
    uint bfrag[4][16];
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)B, b_w, b_h, b_p,
            (int2)((int)(k0 / 2u) + 8 * ks, (int)(n0 + 64u * sn + 16u * b)), bfrag[b] + 8 * ks);
#else
    // B: 32 k x 64 n, two VNNI-transform messages.
    // bfrag[h][c*16 + j] = (B[k0+2j+1][n0+64sn+32h+16c+lane] << 16) | B[k0+2j][...].
    uint bfrag[2][32];
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)B, b_w, b_h, b_p, (int2)((int)(n0 + 64u * sn), (int)k0), bfrag[0]);
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)B, b_w, b_h, b_p, (int2)((int)(n0 + 64u * sn + 32u), (int)k0), bfrag[1]);
#endif

    if (t + 2 < num_k_tiles) {   // prefetch k-tile t+2
      const int k_pf = (int)((t + 2) * WG_K);
      intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p,
                                                    (int2)(k_pf, (int)(m0 + 8u * s)));
#if TRANSB
      intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)B, b_w, b_h, b_p,
                                                    (int2)(k_pf, (int)(n0 + 8u * s)));
#else
      intel_sub_group_2d_block_prefetch_16b_8r16x2c(
          (__global void*)B, b_w, b_h, b_p,
          (int2)((int)(n0 + 32u * (s & 7u)), k_pf + (int)(8u * (s >> 3))));
#endif
    }

    // 32 dpas per k-tile, b-outer a-inner (four consecutive dpas share src1).
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[4];
#pragma unroll
      for (int a = 0; a < 4; ++a)
        af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < 4; ++b) {
#if TRANSB
        const int8 bv = as_int8(vload8(0, bfrag[b] + (uint)(8 * ks)));
#else
        const int h = b >> 1, c = b & 1;
        const int8 bv = as_int8(vload8(0, bfrag[h] + (uint)(c * 16 + 8 * ks)));
#endif
#pragma unroll
        for (int a = 0; a < 4; ++a)
          acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }

  // epilogue: 16 stores per subgroup, 8 rows x 64 B each, no C read.
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c(
          (__global void*)C, c_w, c_h, c_p,
          (int2)((int)(n0 + 64u * sn + 16u * b), (int)(m0 + 32u * sm + 8u * a)),
          (__private uint*)&acc[a][b]);
}
