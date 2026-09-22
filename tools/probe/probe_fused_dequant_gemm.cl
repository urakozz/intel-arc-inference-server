// probe_fused_dequant_gemm - the S1 probe's candidate kernels
// (docs/superpowers/specs/2026-09-22-fused-dequant-gemm-probe-design.md §5).
//
// PROBE-ONLY. `src/kernels/prefill/pf_gemm.cl` and `pf_dequant_slab.cl` are NOT
// touched by this file; no runtime path binds these entry points.
//
// This is `src/kernels/prefill/pf_gemm.cl`'s TRANSB=0 kernel with **only the B
// path replaced**: the 256x256x32 tile, the 512-work-item group, the 8x4
// subgroup raster, the A 2D block load, the 2-deep A prefetch, the 16 float8
// accumulators, the 32 `dpas.8x8` per k-tile in b-outer/a-inner order, the
// split barrier and the 16-store epilogue are transcribed verbatim. B no longer
// arrives as bf16 from a dequant slab; it is built in-kernel from the layout-0
// int4 source.
//
//   C[m][n] fp32 = Σ_k A[m][k] · Bq[k][n],  Bq[k][n] = rne_bf16((float)qm8 · scale)
//   A:  bf16 [M][lda]
//   QW: u32  [K/8][N]   nibble i of word (r, n) is the weight at k = r*8 + i
//   SC: f16  [K/64][N]  the group scale of (k/64, n)
//   C:  fp32 [M][ldc]
//
// **The arithmetic is `pf_dequant_slab.cl`'s, statement for statement** (§2 of
// the design fixes it): the same `^ 0x88888888`, the same `((int)(u << (28 -
// 4i))) >> 28` sign extension, the same `(float)qm8 * scale` with `scale` the
// group's f16 widened to f32, and the same single `rne_bf16`. Nothing is
// reassociated and no rounding point moves, so C is expected to be BITWISE
// equal to the two-pass path and the gate is a memcmp.
//
// The k order is the production kernel's, unchanged. `bfrag` element j of
// n-atom b is the VNNI pair (B[k0+2j][n], B[k0+2j+1][n]) with the even k in the
// low half; k0+2j lives in qweight row k0/8 + (j>>2) at nibble 2*(j&3), and
// k0+2j+1 at nibble 2*(j&3)+1. A 32-deep k-tile therefore needs exactly FOUR
// qweight rows per column, and (because 32 | k0 and the group is 64 wide) ONE
// scale per column - it cannot straddle two groups.
//
// FUSED = 1  registers only. Each subgroup loads the 4 qweight rows x 64 of its
//            own columns (four `..._32b_4r16x1c` messages, 1 KB, against the
//            two 2 KB VNNI transform loads it replaces) and dequantises
//            straight into its B fragment registers. No SLM, no extra barrier.
//            The 8 m-subgroups that share an n-column block each repeat the
//            work: the dequant ALU is 8x redundant per work-group, which is the
//            risk §5 names.
// FUSED = 2  SLM staged, double buffered. The work-group dequantises each
//            k-tile's 32k x 256n B ONCE - subgroup s takes n-atom s>>1 and
//            k-half s&1, two qweight rows of 16 columns - into SLM, while the
//            dpas block consumes the previous tile from the other buffer. The
//            SLM image is exactly the register image `intel_sub_group_block_-
//            read8` returns (j-major, lane-minor), so the dpas operand is
//            bit-for-bit the fragment the transform load delivered.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#ifndef FUSED
#error "probe_fused_dequant_gemm: FUSED must be defined (1 registers-only, 2 SLM staged)"
#endif
#if FUSED != 1 && FUSED != 2
#error "probe_fused_dequant_gemm: FUSED must be 1 or 2"
#endif

#define SG 16
#define WG_M 256
#define WG_N 256
#define WG_K 32

// pf_dequant_slab.cl's rne_bf16, transcribed verbatim. Round-to-nearest-even on
// the f32->bf16 truncation point: `rounding` is 0x7FFF plus the destination's
// low bit, so a tie rounds to the even word. NOT a truncation (the failure mode
// the oneDNN path showed, docs/prefill-parity-2026-09-20.md).
inline ushort rne_bf16(float f) {
  const uint u = as_uint(f);
  const uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// One VNNI pair: nibbles 2p (even k, low half) and 2p+1 (odd k, high half) of
// the already-xored word `u`, at this column's group scale. The two shifts are
// pf_dequant_slab's `28 - 4*i` at i = 2p and i = 2p+1.
inline uint pf_pair(uint u, int p, float scale) {
  const int lo = ((int)(u << (28 - 8 * p))) >> 28;
  const int hi = ((int)(u << (24 - 8 * p))) >> 28;
  return ((uint)rne_bf16((float)hi * scale) << 16) | (uint)rne_bf16((float)lo * scale);
}

// Two qweight rows (16 k) of one column -> the 8 VNNI pairs of one k half-tile,
// in the register order `intel_sub_group_2d_block_read_transform_16b_32r16x2c`
// produces and the dpas below reads.
inline uint8 pf_dequant8(uint w0, uint w1, float scale) {
  const uint u0 = w0 ^ 0x88888888u;
  const uint u1 = w1 ^ 0x88888888u;
  uint8 v;
  v.s0 = pf_pair(u0, 0, scale);
  v.s1 = pf_pair(u0, 1, scale);
  v.s2 = pf_pair(u0, 2, scale);
  v.s3 = pf_pair(u0, 3, scale);
  v.s4 = pf_pair(u1, 0, scale);
  v.s5 = pf_pair(u1, 1, scale);
  v.s6 = pf_pair(u1, 2, scale);
  v.s7 = pf_pair(u1, 3, scale);
  return v;
}

#if FUSED == 2
// 16 n-atoms x 16 VNNI pairs x 16 lanes of uint = one k-tile's 32k x 256n B
// fragment set for the whole work-group, laid out so that n-atom a's k-half ks
// is `intel_sub_group_block_read8(BSLM + a*256 + ks*128)`.
#define SLM_TILE_U32 4096
#endif

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pfd_gemm(__global const ushort* restrict A,
                       __global const uint* restrict QW,
                       __global const ushort* restrict SC,
                       __global float* restrict C,
                       uint M, uint K, uint N, uint lda, uint ldc) {
  const uint gx = get_group_id(0);   // M axis
  const uint gy = get_group_id(1);   // N axis
  const uint m0 = gx * WG_M;
  const uint n0 = gy * WG_N;
  const uint s = get_sub_group_id();   // 0..31
  const uint sm = s >> 2;              // 0..7: rows m0 + 32 sm .. +31
  const uint sn = s & 3u;              // 0..3: cols n0 + 64 sn .. +63

  const int a_w = (int)(K * 2u), a_h = (int)M, a_p = (int)(lda * 2u);
  const int q_w = (int)(N * 4u), q_h = (int)(K / 8u), q_p = (int)(N * 4u);
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);
  const uint num_k_tiles = K / WG_K;

#if FUSED == 2
  __local uint BSLM[2 * SLM_TILE_U32];
  const uint p_atom = s >> 1;                 // 0..15: this subgroup's n-atom
  const uint p_half = s & 1u;                 // 0..1: which 16-k half of the tile
  const uint p_n = n0 + 16u * p_atom;
  __local uint* const p_dst = BSLM + p_atom * 256u + p_half * 128u;
#endif

  float8 acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);

  // --- prologue: prefetch k-tiles 0 and 1. A exactly as pf_gemm does. QW is a
  // [K/8][N] u32 surface: the work-group's 4 rows x 256 dwords are covered by
  // subgroups 0..15 taking one 4r x 16-dword block each (s and s+16 coincide).
#pragma unroll
  for (int pf = 0; pf < 2; ++pf) {
    const int k_pf = pf * WG_K;
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p,
                                                  (int2)(k_pf, (int)(m0 + 8u * s)));
    intel_sub_group_2d_block_prefetch_32b_4r16x1c(
        (__global void*)QW, q_w, q_h, q_p,
        (int2)((int)(n0 + 16u * (s & 15u)), k_pf / 8));
  }

#if FUSED == 1
  const uint nb = n0 + 64u * sn;   // this subgroup's four n-atoms
  float sc[4];
#else
  // Stage k-tile 0 into buffer 0 before the first wait.
  {
    uint qw2[2];
    intel_sub_group_2d_block_read_32b_2r16x1c((__global void*)QW, q_w, q_h, q_p,
                                              (int2)((int)p_n, (int)(2u * p_half)), qw2);
    const float sc0 = (float)as_half(intel_sub_group_block_read_us(SC + (size_t)p_n));
    intel_sub_group_block_write8(p_dst, pf_dequant8(qw2[0], qw2[1], sc0));
  }
  intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif

  for (uint t = 0; t < num_k_tiles; ++t) {
    const uint k0 = t * WG_K;
#if FUSED == 1
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#else
    const uint cur = t & 1u;
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#endif

    // A: 32 m x 32 k, one message. afrag[c*32 + r] = A[m0+32sm+r][k0+16c+lane].
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)A, a_w, a_h, a_p,
                                               (int2)((int)k0, (int)(m0 + 32u * sm)), afrag);

#if FUSED == 1
    // B source: four qweight rows (k0 .. k0+31) of this subgroup's 64 columns,
    // one 4r x 16-dword message per n-atom -- 1 KB against the 4 KB of bf16 the
    // two transform loads read today.
    uint qw[4][4];
#pragma unroll
    for (int b = 0; b < 4; ++b)
      intel_sub_group_2d_block_read_32b_4r16x1c((__global void*)QW, q_w, q_h, q_p,
                                                (int2)((int)(nb + 16u * (uint)b), (int)(k0 / 8u)),
                                                qw[b]);
    // One scale per column per 64-k group: a 32-deep tile never straddles two.
    if ((t & 1u) == 0u) {
      const size_t so = (size_t)(t >> 1) * N + nb;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        sc[b] = (float)as_half(intel_sub_group_block_read_us(SC + so + 16u * (uint)b));
    }
#endif

    if (t + 2 < num_k_tiles) {   // prefetch k-tile t+2
      const int k_pf = (int)((t + 2) * WG_K);
      intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p,
                                                    (int2)(k_pf, (int)(m0 + 8u * s)));
      intel_sub_group_2d_block_prefetch_32b_4r16x1c(
          (__global void*)QW, q_w, q_h, q_p,
          (int2)((int)(n0 + 16u * (s & 15u)), k_pf / 8));
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
#if FUSED == 1
        const int8 bv = as_int8(pf_dequant8(qw[b][2 * ks], qw[b][2 * ks + 1], sc[b]));
#else
        const int8 bv = as_int8(intel_sub_group_block_read8(
            BSLM + cur * SLM_TILE_U32 + (4u * sn + (uint)b) * 256u + (uint)ks * 128u));
#endif
#pragma unroll
        for (int a = 0; a < 4; ++a)
          acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }

#if FUSED == 1
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#else
    // Stage k-tile t+1 into the other buffer. It sits AFTER the dpas block so
    // the qweight load's latency and the dequant ALU overlap the matrix math;
    // it sits BEFORE the arrive so the arrive/wait pair that follows publishes
    // it (a write placed after the arrive would not be covered).
    if (t + 1 < num_k_tiles) {
      const uint k1 = k0 + WG_K;
      uint qw2[2];
      intel_sub_group_2d_block_read_32b_2r16x1c(
          (__global void*)QW, q_w, q_h, q_p,
          (int2)((int)p_n, (int)(k1 / 8u + 2u * p_half)), qw2);
      const float sc1 =
          (float)as_half(intel_sub_group_block_read_us(SC + (size_t)(k1 / 64u) * N + p_n));
      intel_sub_group_block_write8((cur ^ 1u) ? p_dst + SLM_TILE_U32 : p_dst,
                                   pf_dequant8(qw2[0], qw2[1], sc1));
    }
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif
  }
#if FUSED == 2
  intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);   // pair the last arrive
#endif

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
