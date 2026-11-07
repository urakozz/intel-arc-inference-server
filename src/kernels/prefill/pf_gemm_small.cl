// pf_gemm_small - pf_gemm (pf_gemm.cl, TRANSB 0, plain and SILU_EPI) for a chunk of few rows
// (open-work 3a). pf_gemm gives every 256 x 256 output tile one work-group of 32 sub-groups
// (8 m x 4 n, each 32 m x 64 n), so a 1024-column slab is 4 work-groups for any chunk up to 256
// rows: 4 of the 32 Xe cores, each doing a whole 256-row tile's DPAS for the few rows the chunk
// has. Here a sub-group owns NA row blocks of 8 (8 NA rows) x NB 16-column atoms, a work-group
// SN such sub-groups side by side along n, and the grid covers only pad(M, 8 NA) rows - so a
// short chunk's slab is many small work-groups spread over every core, each doing only the
// rows that exist.
//
// **Bit-identical to pf_gemm.** Each accumulator (one 8-row block x one 16-column atom) starts
// at zero and takes the same DPAS chain pf_gemm's does - per 32-k tile, the k16 half ks = 0
// then 1, intel_sub_group_bf16_bf16_matrix_mad_k16 on the same A rows and B columns - and a
// DPAS output row depends only on its own A row (spec 28's property R), so neither the row
// block's neighbours nor the tile shape change a value. The epilogues are pf_gemm's, the SiLU
// chain copied verbatim from pf_gemm.cl / pf_prep.cl (the numerics contract; edit together).
// tests/prefill/pf_gemm_test.cc holds the equality at every built variant.
//
// Requirements (runtime::prefill::gemm_l0 asserts them): N % (16 NB SN) == 0, K % 32 == 0,
// 64-byte aligned bases, 16-byte multiple pitches. M is the rows the grid covers (a multiple
// of 8 NA); rows past the chunk's are padding, as pf_gemm's are. SILU_EPI needs NB even (a
// sub-group's atoms pair as gate, up - pf_gemm.cl's parity argument, the sub-group's n base a
// multiple of 32).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#ifndef NA
#error "pf_gemm_small: NA (8-row blocks per sub-group: 1, 2 or 4) must be defined"
#endif
#ifndef NB
#error "pf_gemm_small: NB (16-column atoms per sub-group: 2 or 4) must be defined"
#endif
#ifndef SN
#error "pf_gemm_small: SN (sub-groups per work-group, along n) must be defined"
#endif
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
#ifndef NFAST
#define NFAST 0   /* 1: grid dimension 0 walks the column tiles, 1 the row tiles */
#endif
#if NA != 1 && NA != 2 && NA != 4
#error "pf_gemm_small: NA is 1, 2 or 4"
#endif
#if NB != 2 && NB != 4
#error "pf_gemm_small: NB is 2 or 4"
#endif

#define SG 16
#define WG_K 32
#define R (8 * NA)          /* a sub-group's rows */
#define CN (16 * NB)        /* a sub-group's columns */

#if SILU_EPI
// **Copied verbatim from pf_prep.cl** (pf_gemm.cl carries the same copy): both bf16 RNE
// steps, plain `exp` (never `native_exp`), silu the LAST factor.
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

// The A fragment of one 32-k tile: R rows x 32 k, afrag[c R + r] = A[mb + r][k0 + 16 c + lane]
// (pf_gemm's 32r16x2c read at R = 32).
inline void read_a(__global const ushort* A, int a_w, int a_h, int a_p, int k0, int mb, __private ushort* afrag) {
#if NA == 4
  intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#elif NA == 2
  intel_sub_group_2d_block_read_16b_16r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#else
  intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#endif
}
inline void prefetch(__global const ushort* A, int a_w, int a_h, int a_p, __global const ushort* B,
                     int b_w, int b_h, int b_p, int k, int mb, int nb) {
#if NA == 4
  intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#elif NA == 2
  intel_sub_group_2d_block_prefetch_16b_16r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#else
  intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#endif
#pragma unroll
  for (int h = 0; h < NB / 2; ++h)
    intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)B, b_w, b_h, b_p,
                                                   (int2)(nb + 32 * h, k));
}

__attribute__((reqd_work_group_size(SG * SN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm_small(__global const ushort* restrict A, __global const ushort* restrict B,
                            __global float* restrict C_base, uint M, uint K, uint N, uint lda,
                            uint ldb, uint ldc) {
  const uint s = get_sub_group_id();
#if NFAST
  const uint mb = get_group_id(1) * R;
  const uint nb = get_group_id(0) * (CN * SN) + CN * s;
#else
  const uint mb = get_group_id(0) * R;
  const uint nb = get_group_id(1) * (CN * SN) + CN * s;
#endif
#if SILU_EPI
  __global ushort* restrict X = (__global ushort*)C_base;   // bf16 x, `ldc` carries ldx
#else
  __global float* restrict C = C_base;
#endif
  const int a_w = (int)(K * 2u), a_h = (int)M, a_p = (int)(lda * 2u);
  const int b_w = (int)(N * 2u), b_h = (int)K, b_p = (int)(ldb * 2u);
  const uint num_k_tiles = K / WG_K;

  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);

  prefetch(A, a_w, a_h, a_p, B, b_w, b_h, b_p, 0, (int)mb, (int)nb);
  if (num_k_tiles > 1u) prefetch(A, a_w, a_h, a_p, B, b_w, b_h, b_p, WG_K, (int)mb, (int)nb);

  for (uint t = 0; t < num_k_tiles; ++t) {
    const uint k0 = t * WG_K;
#if SN > 1
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif
    ushort afrag[2 * R];
    read_a(A, a_w, a_h, a_p, (int)k0, (int)mb, afrag);
    // B: 32 k x CN n, NB / 2 VNNI-transform messages (pf_gemm's).
    // bfrag[h][c*16 + j] = (B[k0+2j+1][nb+32h+16c+lane] << 16) | B[k0+2j][...].
    uint bfrag[NB / 2][32];
#pragma unroll
    for (int h = 0; h < NB / 2; ++h)
      intel_sub_group_2d_block_read_transform_16b_32r16x2c(
          (__global void*)B, b_w, b_h, b_p, (int2)((int)(nb + 32u * (uint)h), (int)k0), bfrag[h]);
    if (t + 2 < num_k_tiles)
      prefetch(A, a_w, a_h, a_p, B, b_w, b_h, b_p, (int)((t + 2) * WG_K), (int)mb, (int)nb);

    // pf_gemm's k16 halves, the same DPAS per accumulator in the same order.
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[NA];
#pragma unroll
      for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * R + 8 * a)));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const int h = b >> 1, c = b & 1;
        const int8 bv = as_int8(vload8(0, bfrag[h] + (uint)(c * 16 + 8 * ks)));
#pragma unroll
        for (int a = 0; a < NA; ++a)
          acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }
#if SN > 1
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#endif
  }

#if SILU_EPI
  // pf_gemm's SiLU epilogue: atom 2p is a gate block, 2p + 1 its up block; x column = gate
  // column / 2 + lane.
  const int x_w = (int)(N * 2u), x_h = (int)M, x_p = (int)(ldc * 2u);
#define PF_SILU_ROW(r)                                             \
  do {                                                             \
    const ushort g_b = rne_bf16(gv.s##r), u_b = rne_bf16(uv.s##r); \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));             \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                     \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a) {
#pragma unroll
    for (int p = 0; p < NB / 2; ++p) {
      const float8 gv = acc[a][2 * p], uv = acc[a][2 * p + 1];
      const uint ng = nb + 32u * (uint)p;
      ushort xv[8];
      PF_SILU_ROW(0); PF_SILU_ROW(1); PF_SILU_ROW(2); PF_SILU_ROW(3);
      PF_SILU_ROW(4); PF_SILU_ROW(5); PF_SILU_ROW(6); PF_SILU_ROW(7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p,
                                                 (int2)((int)(ng >> 1), (int)(mb + 8u * a)), xv);
    }
  }
#undef PF_SILU_ROW
#else
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c(
          (__global void*)C, c_w, c_h, c_p, (int2)((int)(nb + 16u * b), (int)(mb + 8u * a)),
          (__private uint*)&acc[a][b]);
#endif
}
