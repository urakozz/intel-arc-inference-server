// gemm_sm_wide.cl - the small-M GEMM past 16 rows (spec 32e, plan 32e Task 2): M = 32 / 64 activation rows against
// the int4 decode weights through bf16 DPAS, with gemm_sm's per-row arithmetic under another thread shape.
//
// gemm_sm (gemm_sm.cl) holds every row's activations of a group beside two accumulator sets in one sub-group, which
// spills past 16 rows. The wide form keeps gemm_sm's arguments in gemm_sm's order (w, scales, x, out), its output layout
// out[(s M + m) N + n] (one slice under SPLITK 1) and its grid, so a capture site swaps the variant name, the entry
// point and the work-group size only. Two thread shapes (plan 34a Task 1 measured them; kernels::SmallMTuning names
// which a shape binds at which M):
//   RC (row chunk)  the work-group gains a row-chunk dimension: sub-group (column c, K-slice ks, chunk rc) runs rows
//                   RC rc .. RC rc + RC - 1 against the same 16 weight columns as its siblings (each chunk is
//                   gemm_sm at RC rows; the siblings' weight reads hit the same lines). RC 32, 256 GRF.
//   SWAP            one sub-group for every row: the group's four B operands dequantised first and kept, then row
//                   block by row block - its A operands, four DPAS into ONE group accumulator, the scale step. 128 GRF.
// Either way each 8-row block's A operands of k16 steps 0, 1 and 2, 3 come from two 2D block reads of the rows
// (8 rows x 16 k x 2 blocks; x in elements, width and pitch 2K bytes, height the sub-group's rows) that the DPAS
// read in place - no row loads and moves (spec 34).
//
// Compile-time knobs (src/kernels/CMakeLists.txt, from kernels::small_m_tuning):
//   M, K, N, S   rows (a multiple of 8, 17..64), the weight shape, the output's split-K slices (S | K/64)
//   LAYOUT       0: GPTQ qweight [K/8][N] u32 + f16 scales [K/64][N]; 1: 544-byte tiles (common/repack.h)
//   SPLITK, DEQ, BLK2D   gemm_sm's (the row's knobs, unchanged)
//   RC           the row-chunk shape's rows a sub-group (8 | RC, RC | M); SWAP 1: the swap shape (RC = M)
//   PF_G         0 or a weight prefetch distance in this sub-group's own 64-k groups (chunk 0 issues it)
//
// **Property R (spec 32e; held by tests/kernels/gemm_sm_test.cc):** a DPAS output element is one chain of bf16 x bf16
// products added in k order; which 8-row block or sub-group a row sits in does not enter it. What does - the k
// pairing and order, the zero accumulator per g64 group, acc += group x scale in fp32, S and the SLM reduce in slice
// order - is gemm_sm's. So row m at 32 or 64 rows is bitwise row m of gemm_sm at 16 rows of the same shape.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#error "gemm_sm_wide: M"
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#ifndef SPLITK
#define SPLITK 0
#endif
#ifndef DEQ
#define DEQ 0
#endif
#ifndef BLK2D
#define BLK2D 0
#endif
#ifndef SWAP
#define SWAP 0
#endif
#ifndef PF_G
#define PF_G 0
#endif

#if SWAP
#define RC M
#endif
#ifndef RC
#error "gemm_sm_wide: RC or SWAP"
#endif
#if SPLITK == 0
#define NSGN 4
#define KSL 1
#else
#define NSGN 2
#define KSL 8
#endif

#define SG 16
#define WG (SG * NSGN * KSL)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define G_PER_SG (G_PER_S / KSL)
#define TILE_U32 136
#define RCN (M / RC)          /* row chunks (sub-groups per column sub-group and K-slice) */
#define RBC (RC / 8)          /* row blocks of 8 a sub-group holds */

#if M < 17 || M > 64
#error "gemm_sm_wide: M in [17, 64] (gemm_sm takes M <= 16)"
#endif
#if (M % RC) != 0 || (RC % 8) != 0
#error "gemm_sm_wide: RC | M and 8 | RC"
#endif
#if (K % GROUP) != 0 || (G % S) != 0 || (G_PER_S % KSL) != 0
#error "gemm_sm_wide: S | K/64 and KSL | K/64/S"
#endif
#if (N % (SG * NSGN)) != 0
#error "gemm_sm_wide: N must be a multiple of 16 x NSGN"
#endif
#if SPLITK && S != 1
#error "gemm_sm_wide: the in-work-group split-K writes one slice (S 1)"
#endif
#if LAYOUT > 1
#error "gemm_sm_wide: int4 layouts 0 and 1"
#endif
#if BLK2D && LAYOUT != 0
#error "gemm_sm_wide: BLK2D is layout 0 only"
#endif

// Component r of a float8 (r is a compile-time constant once the loops unroll).
inline float f8(float8 v, int r) {
  switch (r) {
    case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
    case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7;
  }
}

// gemm_sm's dequant: one int4 nibble (index i of the word: k = 8 j + i) as the bf16 integer q - 8, in the low 16 bits
// (lo) or the high 16 bits (hi) of the result.
#if DEQ == 0
inline int nib_s(uint word, uint i) { return ((int)((word ^ 0x88888888u) << (28u - 4u * i))) >> 28; }
inline uint nib_lo(uint word, uint i) { return as_uint((float)nib_s(word, i)) >> 16; }
inline uint nib_hi(uint word, uint i) { return as_uint((float)nib_s(word, i)) & 0xFFFF0000u; }
#else
inline float nib_f(uint word, uint i) {
  return as_float((0x4300u | ((word >> (4u * i)) & 0xFu)) << 16) - 136.0f;
}
inline uint nib_lo(uint word, uint i) { return as_uint(nib_f(word, i)) >> 16; }
inline uint nib_hi(uint word, uint i) { return as_uint(nib_f(word, i)) & 0xFFFF0000u; }
#endif
// Nibbles 2p (low k) and 2p + 1 as one bf16 pair: the DPAS B operand's per-lane dword.
inline int pair_bf16(uint word, uint p) {
  return (int)(nib_hi(word, 2u * p + 1u) | nib_lo(word, 2u * p));
}

// Prefetch of group gp's weights for this sub-group's 16 columns.
inline void prefetch_group(__global const uint* w, uint n_tile, uint gp) {
  gp = min(gp, (uint)(G - 1));
#if LAYOUT == 0
  intel_sub_group_2d_block_prefetch_32b_4r16x1c((__global void*)w, (int)(N * 4u), (int)(K / 8u),
                                                (int)(N * 4u), (int2)((int)(n_tile * SG), (int)(gp * 8u)));
  intel_sub_group_2d_block_prefetch_32b_4r16x1c((__global void*)w, (int)(N * 4u), (int)(K / 8u),
                                                (int)(N * 4u), (int2)((int)(n_tile * SG), (int)(gp * 8u + 4u)));
#else
  prefetch(w + ((size_t)n_tile * G + gp) * TILE_U32, TILE_U32);
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG * RCN, 1, 1)))
__kernel void gemm_sm_wide(__global const uint* restrict w,
                         __global const half* restrict scales,
                         __global const ushort* restrict x,
                         __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint col_sg = sg % NSGN;
  const uint ks = (sg / NSGN) % KSL;
  const uint rc = sg / (NSGN * KSL);
  const uint n_tile = get_group_id(0) * NSGN + col_sg;
  const uint n0 = n_tile * SG;
  const uint n = n0 + lane;
  const uint s = get_group_id(1);
  const uint gbase = s * G_PER_S + ks;
  const uint r0 = rc * RC;
  __global const ushort* xr = x + (size_t)r0 * K;

  float8 acc[RBC];
  for (int rb = 0; rb < RBC; ++rb) acc[rb] = (float8)0.0f;
  for (uint i = 0; i < G_PER_SG; ++i) {
    const uint g = gbase + i * KSL;
#if PF_G
    if (rc == 0) prefetch_group(w, n_tile, g + PF_G * KSL);
#endif
    // ---- the group's eight u32 rows of this lane's column and its scale (gemm_sm's reads) ----------
    uint wv[8];
    float scale;
#if LAYOUT == 0
#if BLK2D
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)w, (int)(N * 4u), (int)(K / 8u),
                                              (int)(N * 4u), (int2)((int)n0, (int)(g * 8u)), wv);
#else
    for (int j = 0; j < 8; ++j) wv[j] = w[(size_t)(g * 8 + j) * N + n];
#endif
    scale = (float)scales[(size_t)g * N + n];
#else
    __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
    const uint8 blk = intel_sub_group_block_read8(tile);
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
    scale = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tile + 128)));
#endif
#if !SWAP
    // [row block][8 d + r]: row rb * 8 + r of k16 step d, this lane's k - two 2D reads a block (steps 0, 1 and 2, 3)
    // on the chunk's rows (base xr, height RC)
    ushort xa[RBC][32];
    for (int rb = 0; rb < RBC; ++rb) {
      intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)xr, (int)(K * 2u), (int)RC, (int)(K * 2u),
                                                (int2)((int)(g * GROUP), rb * 8), &xa[rb][0]);
      intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)xr, (int)(K * 2u), (int)RC, (int)(K * 2u),
                                                (int2)((int)(g * GROUP + 32u), rb * 8), &xa[rb][16]);
    }
    float8 gacc[RBC];
    for (int rb = 0; rb < RBC; ++rb) gacc[rb] = (float8)0.0f;
    for (int d = 0; d < 4; ++d) {
      const uint w0 = wv[2 * d], w1 = wv[2 * d + 1];
      int8 b;
      b.s0 = pair_bf16(w0, 0); b.s1 = pair_bf16(w0, 1); b.s2 = pair_bf16(w0, 2); b.s3 = pair_bf16(w0, 3);
      b.s4 = pair_bf16(w1, 0); b.s5 = pair_bf16(w1, 1); b.s6 = pair_bf16(w1, 2); b.s7 = pair_bf16(w1, 3);
      for (int rb = 0; rb < RBC; ++rb)
        gacc[rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(d, xa[rb])), b, gacc[rb]);
    }
    for (int rb = 0; rb < RBC; ++rb) acc[rb] += gacc[rb] * scale;
#else
    int8 b[4];
    for (int d = 0; d < 4; ++d) {
      const uint w0 = wv[2 * d], w1 = wv[2 * d + 1];
      b[d].s0 = pair_bf16(w0, 0); b[d].s1 = pair_bf16(w0, 1); b[d].s2 = pair_bf16(w0, 2); b[d].s3 = pair_bf16(w0, 3);
      b[d].s4 = pair_bf16(w1, 0); b[d].s5 = pair_bf16(w1, 1); b[d].s6 = pair_bf16(w1, 2); b[d].s7 = pair_bf16(w1, 3);
    }
    for (int rb = 0; rb < RBC; ++rb) {
      // read - two DPAS - read - two DPAS (both reads first overflows 128 GRF); the DPAS read them in place
      float8 gacc = (float8)0.0f;
      {
        ushort a01[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)xr, (int)(K * 2u), (int)RC, (int)(K * 2u),
                                                  (int2)((int)(g * GROUP), rb * 8), a01);
        gacc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, a01)), b[0], gacc);
        gacc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(1, a01)), b[1], gacc);
      }
      {
        ushort a23[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)xr, (int)(K * 2u), (int)RC, (int)(K * 2u),
                                                  (int2)((int)(g * GROUP + 32u), rb * 8), a23);
        gacc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, a23)), b[2], gacc);
        gacc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(1, a23)), b[3], gacc);
      }
      acc[rb] += gacc * scale;
    }
#endif
  }

  // ---- the end of the K walk: acc[rb].s(r) = row r0 + rb*8 + r, column n ---------------------------
#if SPLITK == 1
  __local float red[KSL][NSGN][M][SG];
  for (int rb = 0; rb < RBC; ++rb)
    for (int r = 0; r < 8; ++r) red[ks][col_sg][r0 + rb * 8 + r][lane] = f8(acc[rb], r);
  barrier(CLK_LOCAL_MEM_FENCE);
  // gemm_sm's reduce (slice 0 + 1 + ... in order); the KSL x RCN sub-groups of a column share the rows
  for (uint e = rc * KSL + ks; e < M; e += KSL * RCN) {
    float v = red[0][col_sg][e][lane];
    for (uint j = 1; j < KSL; ++j) v += red[j][col_sg][e][lane];
    out[(size_t)e * N + n] = v;
  }
#else
  for (int rb = 0; rb < RBC; ++rb)
    for (int r = 0; r < 8; ++r) out[((size_t)s * M + r0 + rb * 8 + r) * N + n] = f8(acc[rb], r);
#endif
}
