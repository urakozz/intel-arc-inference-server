// pf_moe_gemm - spec 15d: the GROUPED expert GEMM of a prefill chunk (spec 15 §4.4): ONE
// launch covers every expert's tiles. Work-group (tile, n-block) reads its tile's weight
// block and first sorted row from the tile table pf_moe.cl's pf_moe_sort wrote, and
// computes that tile's TM = 32 rows x 256 columns of its expert's linear. A padding tile
// (block NONE), and a tile whose block is outside this launch's batch [b0, b1), returns at
// once - there is no barrier in this kernel, so the early return is legal - which is how
// the grid stays a function of C alone (the host never reads a count).
//
// Two forms, by define, both pf_gemm.cl / pf_int8.cl's verified mainloops with only the
// addressing changed (A from the tile's first row, B from the tile's expert):
//
//   I8 = 0   bf16 x bf16, fp32 accumulation: pf_gemm.cl's sub-group tile (32 m x 64 n,
//            4 x 4 float8), its 2D block A read and VNNI-transform B read, its 32-k dpas
//            body. 4 sub-groups per work-group. B = pf_moe_dequant_*'s bf16 blocks,
//            expert e at (e - b0) x K x N, row-major [K][N].
//   I8 = 1   int8 x int8, int32 accumulation (spec 5's h8, pf_int8.cl's pf_gemm_i8): its
//            sub-group tile (32 m x 32 n, 4 x 2 int8), its 64-k body; scales only in the
//            epilogue (acc x ws[n] x xs[row]). 8 sub-groups per work-group. A = the
//            gathered pf_quant_had rows, xs their per-token scales; B = pf_requant_rot's
//            VNNI-4 u32 [K/4][ldb] over the layer's whole expert array, expert e's N
//            columns at u32 column (e - b0) x N; ws the per-column scales of that array.
//
// Epilogues: SILU_EPI = 1 is gate||up (16-column gate/up interleave inside each expert
// block, common::cols_interleave16: a sub-group's atom pair is (gate, up)) and writes
// h = bf16 [rows][N/2] with prep_silu_mul's chain, verbatim from pf_gemm.cl / pf_int8.cl;
// SILU_EPI = 0 (bf16 only) is down and writes y = rne(acc) bf16 [rows][N] - the one
// rounding every consumer of a linear's output applies first (pf_moe_combine's d_b).
//
// **Row independence (spec 15d Review Focus 1).** Output row r of a tile is Σ_k A[r][k]
// B[k][n] through the same dpas instruction sequence whatever the other 31 rows hold:
// DPAS reduces along K only, a row's accumulator never meets another row's, and the
// epilogue is per element (its scale is the row's own xs). So permuting the rows inside
// a tile, or moving a row to another tile of the same expert, leaves every row's output
// bitwise unchanged - tests/kernels/pf_moe_test.cc checks it on the card; the host
// reference (pf_moe_ref.h) is row-wise by construction and its test pins the contract.
//
// **Bounds.** Every tile is a whole TM rows of its own expert (pf_moe_sort pads each
// expert to a multiple of TM), so the 2D block surfaces are exactly one tile tall and the
// reads and writes stay inside the tile - no read relies on out-of-surface zero fill and
// no write on out-of-surface discard. Padding rows hold zeros (pf_moe_gather) and their
// outputs are never read.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !defined(K) || !defined(N) || !defined(TM)
#error "pf_moe_gemm: K, N (one expert block) and TM must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif
#ifndef I8
#define I8 0
#endif
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
#if I8 && !SILU_EPI
#error "pf_moe_gemm: the int8 (h8) form is built for gate||up only (spec 15d: down's K is not whole 1024-k rotation blocks)"
#endif
#if TM != 32
#error "pf_moe_gemm: TM must be 32 - one sub-group's row tile is the whole expert tile"
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define WG_N 256
#if N % WG_N != 0
#error "pf_moe_gemm: N must be whole 256-column work-group blocks"
#endif
#if I8
#define SGS 8          /* 8 sub-groups x 32 columns */
#define KSTEP 64       /* pf_gemm_i8's CHUNK */
#else
#define SGS 4          /* 4 sub-groups x 64 columns */
#define KSTEP 32       /* pf_gemm's WG_K */
#endif
#if K % KSTEP != 0
#error "pf_moe_gemm: K must be whole k-steps"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// pf_gemm.cl's / pf_int8.cl's chain, verbatim (the numerics contract with pf_prep.cl).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
#if SILU_EPI
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

// A: bf16 [rows][K] (I8 0) or int8 [rows][K] read as 16-bit pairs (I8 1).
// xs: fp32 [rows], I8 only.  B, ws, ldb: see the header (ws / ldb I8 only).
// out: SILU_EPI 1 -> bf16 h [rows][N/2]; 0 -> bf16 y [rows][N].
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_moe_gemm(__global const uint* restrict tiles, __global const ushort* restrict A,
                          __global const float* restrict xs, __global const uint* restrict B,
                          __global const float* restrict ws, __global ushort* restrict out,
                          uint b0, uint b1, uint ldb) {
  const uint tile = get_group_id(0);
  const uint e = tiles[2 * tile];
  if (e == NONE || e < b0 || e >= b1) return;      // uniform; no barrier below
  const uint row0 = tiles[2 * tile + 1];
  const uint n0 = get_group_id(1) * WG_N;
  const uint s = get_sub_group_id();

#if !I8
  (void)xs; (void)ws; (void)ldb;
  __global const ushort* restrict Ab = A + (size_t)row0 * K;
  __global const ushort* restrict Bb = (__global const ushort*)B + (size_t)(e - b0) * K * N;
  const int a_w = K * 2, a_h = TM, a_p = K * 2;
  const int b_w = N * 2, b_h = K, b_p = N * 2;
  const uint nsg = n0 + 64u * s;                   // this sub-group's 64 columns

  float8 acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);

  for (uint k0 = 0; k0 < K; k0 += KSTEP) {
    // A: 32 m x 32 k, one message. afrag[c*32 + r] = A[row0 + r][k0 + 16c + lane].
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)Ab, a_w, a_h, a_p,
                                               (int2)((int)k0, 0), afrag);
    // B: 32 k x 64 n, two VNNI-transform messages (pf_gemm.cl's !TRANSB read).
    uint bfrag[2][32];
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)Bb, b_w, b_h, b_p, (int2)((int)nsg, (int)k0), bfrag[0]);
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(
        (__global void*)Bb, b_w, b_h, b_p, (int2)((int)(nsg + 32u), (int)k0), bfrag[1]);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int h = b >> 1, c = b & 1;
        const int8 bv = as_int8(vload8(0, bfrag[h] + (uint)(c * 16 + 8 * ks)));
#pragma unroll
        for (int a = 0; a < 4; ++a)
          acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }
  }

#if SILU_EPI
  // pf_gemm.cl's SILU epilogue: atom b = 2p is gate, 2p + 1 its up (nsg is a multiple of
  // 64 inside the expert block, so the parity holds), x column of gate column g is g / 2.
  __global ushort* restrict X = out + (size_t)row0 * (N / 2);
  const int x_w = N, x_h = TM, x_p = N;            // N/2 bf16 = N bytes
#define PF_SILU_ROW(r)                                             \
  do {                                                             \
    const ushort g_b = rne_bf16(gv.s##r), u_b = rne_bf16(uv.s##r); \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));             \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                     \
  } while (0)
#pragma unroll
  for (int a = 0; a < 4; ++a) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const float8 gv = acc[a][2 * p], uv = acc[a][2 * p + 1];
      const uint ng = nsg + 32u * (uint)p;
      ushort xv[8];
      PF_SILU_ROW(0); PF_SILU_ROW(1); PF_SILU_ROW(2); PF_SILU_ROW(3);
      PF_SILU_ROW(4); PF_SILU_ROW(5); PF_SILU_ROW(6); PF_SILU_ROW(7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p,
                                                 (int2)((int)(ng >> 1), 8 * a), xv);
    }
  }
#undef PF_SILU_ROW
#else
  // y = rne(acc): acc[a][b].sR is row 8a + R, column nsg + 16b + lane.
  __global ushort* restrict Y = out + (size_t)row0 * N;
  const int y_w = N * 2, y_h = TM, y_p = N * 2;
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float8 v = acc[a][b];
      ushort yv[8];
      yv[0] = rne_bf16(v.s0); yv[1] = rne_bf16(v.s1); yv[2] = rne_bf16(v.s2); yv[3] = rne_bf16(v.s3);
      yv[4] = rne_bf16(v.s4); yv[5] = rne_bf16(v.s5); yv[6] = rne_bf16(v.s6); yv[7] = rne_bf16(v.s7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Y, y_w, y_h, y_p,
                                                 (int2)((int)(nsg + 16u * (uint)b), 8 * a), yv);
    }
#endif

#else  // I8
  __global const ushort* restrict Ab = A + (size_t)row0 * (K / 2);   // K int8 per row
  const int x_w = K, x_h = TM, x_p = K;
  __global const uint* restrict Bb = B + (size_t)(e - b0) * N;      // this expert's columns
  const int b_w = N * 4, b_h = K / 4, b_p = (int)(ldb * 4u);
  const uint nb = n0 + 32u * s;                    // this sub-group's 32 columns

  int8 acc[4][2];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b] = (int8)(0);

  for (uint k0 = 0; k0 < K; k0 += KSTEP) {
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)Ab, x_w, x_h, x_p,
                                               (int2)((int)(k0 / 2u), 0), afrag);
    uint wv[2][16];
#pragma unroll
    for (int b = 0; b < 2; ++b)
      intel_sub_group_2d_block_read_32b_16r16x1c((__global void*)Bb, b_w, b_h, b_p,
                                                 (int2)((int)(nb + 16u * (uint)b), (int)(k0 / 4u)),
                                                 wv[b]);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int8 bv = as_int8(vload8(0, wv[b] + (uint)(8 * ks)));
#pragma unroll
        for (int a = 0; a < 4; ++a)
          acc[a][b] = intel_sub_group_i8_i8_matrix_mad_k32(af[a], bv, acc[a][b]);
      }
    }
  }

  // pf_int8.cl's SILU epilogue: atom 0 at nb (a multiple of 32) is gate, atom 1 its up.
  __global const float* restrict wse = ws + (size_t)e * N;
  const float wg = as_float(intel_sub_group_block_read((__global const uint*)(wse + nb)));
  const float wu = as_float(intel_sub_group_block_read((__global const uint*)(wse + nb + 16u)));
  __global ushort* restrict X = out + (size_t)row0 * (N / 2);
  const int o_w = N, o_h = TM, o_p = N;
#define PF_SILU_ROW_I8(r)                                                     \
  do {                                                                        \
    const float xsr = xs[row0 + 8u * (uint)a + r];                            \
    const float g = (float)acc[a][0].s##r * (wg * xsr);                       \
    const float u = (float)acc[a][1].s##r * (wu * xsr);                       \
    const ushort g_b = rne_bf16(g), u_b = rne_bf16(u);                        \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));                        \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                                \
  } while (0)
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    ushort xv[8];
    PF_SILU_ROW_I8(0); PF_SILU_ROW_I8(1); PF_SILU_ROW_I8(2); PF_SILU_ROW_I8(3);
    PF_SILU_ROW_I8(4); PF_SILU_ROW_I8(5); PF_SILU_ROW_I8(6); PF_SILU_ROW_I8(7);
    intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, o_w, o_h, o_p,
                                               (int2)((int)(nb >> 1), 8 * a), xv);
  }
#undef PF_SILU_ROW_I8
#endif
}
