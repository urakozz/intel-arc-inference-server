// gemm_sm.cl - the small-M decode / verify GEMM (spec 28 §2, plan 28b): M = 2..16 activation rows
// against the decode weights through bf16 DPAS, one dequant per weight tile for every row.
//
// The production form of plan 28a's stand-in (tools/probe/probe_small_m.cl, arm `bf16 wB`): the
// orientation, type and S 28a chose (spec 28 §9) - weights on the DPAS B operand, bf16, the shape's
// production split-K S. It reads exactly the bytes gemv / gemv_i8w read and writes exactly the
// output they write, with the same kernel arguments in the same order, so a capture site swaps the
// variant name, the entry point, the work-group and the grid and nothing else:
//
//   gemm_sm      int4 g64 symmetric, LAYOUT 0 (GPTQ qweight [K/8][N] u32 + f16 scales [K/64][N]) or
//                LAYOUT 1 (544-byte tiles, common/repack.h) - gemv.cl's (w, scales, x, out); slice s
//                writes out[(s M + m) N + n], the layout prep_res_fold / prep_silu_mul / attn_prep /
//                gdn_step read.
//   gemm_sm_i8w  LAYOUT 2: the int8 weights [N/16][K/16][16 k][16 n] + fp32 column scales [N]
//                (loader/lm_head_int8.h; the int8 lm_head) -
//                gemv_i8w.cl's (w, scales, x, out); writes out[m N + n], scaled.
//
// Compile-time knobs (src/kernels/CMakeLists.txt, from kernels::small_m_tuning - one row per shape):
//   M, K, N, S   rows (1..16), the weight shape, the output's split-K slices (S | K/64; 1 for LAYOUT 2
//                and for SPLITK 1)
//   SPLITK       0: grid split-K - dimension 1 of the grid is the slice, four 16-column sub-groups a
//                work-group (gemv's 64 work-items, grid (N/64, S)); 1: the in-work-group split-K
//                (spec 28 §2.1) - 2 column sub-groups x 8 K-slice sub-groups (256 work-items, grid
//                (N/32, 1)), groups interleaved across the slices, the slices reduced through SLM in
//                slice order; the output is then ONE slice (S must be 1)
//   DEQ          0: xor + arithmetic shift (gemv's GEMV_DEQ_SHIFT); 1: the magic-exponent form - the
//                nibble OR'd into the mantissa of bf16 0x4300 (= 128 + q) and 136 subtracted in fp32.
//                Both produce the exact bf16 integer q - 8, so the DPAS inputs - and the output - are
//                bit-identical; DEQ is a speed knob only
//   BLK2D        LAYOUT 0: the weights by one 2D block read (gemv's GEMV_BLOCK2D cell)
//   (256 GRF is the ocloc option -cl-intel-256-GRF-per-thread on the variant's line, not a define)
//
// **Property R (spec 28 §2, held by tests/kernels/gemm_sm_test.cc at every built M):** a DPAS output
// row depends only on its own A row; the per-group order (4 x k16 DPAS from a zero accumulator, then
// acc += group x scale in fp32; LAYOUT 2: one DPAS chain over all K, the column scale at the end),
// the row block of 8, S and the work-group shape do not depend on M. So row m's output is bitwise
// the same at every M >= 2 of a variant row and whatever the other rows hold. Rows past M are zero
// A rows and never stored. Against gemv (M = 1) rows differ by accumulation order (spec 28 §3, N2).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

#ifndef M
#error "gemm_sm: M"
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
#define RB ((M + 7) / 8)          /* row blocks of 8 */
#define ROWS (RB * 8)

#if M < 1 || M > 16
#error "gemm_sm: M in [1, 16]"
#endif
#if (K % GROUP) != 0 || (G % S) != 0 || (G_PER_S % KSL) != 0
#error "gemm_sm: S | K/64 and KSL | K/64/S"
#endif
#if (N % (SG * NSGN)) != 0
#error "gemm_sm: N must be a multiple of 16 x NSGN"
#endif
#if SPLITK && S != 1
#error "gemm_sm: the in-work-group split-K writes one slice (S 1)"
#endif
#if LAYOUT == 2 && S != 1
#error "gemm_sm: the int8 form has no grid split-K (gemv_i8w's contract)"
#endif
#if BLK2D && LAYOUT != 0
#error "gemm_sm: BLK2D is layout 0 only"
#endif
#if DEQ && LAYOUT == 2
#error "gemm_sm: DEQ is an int4 knob"
#endif

// Component r of a float8 (r is a compile-time constant once the loops unroll).
inline float f8(float8 v, int r) {
  switch (r) {
    case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
    case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7;
  }
}

#if LAYOUT == 2
// A small integer (|v| <= 127) as bf16 bits: exact, the float's low 16 bits are zero.
inline uint int_bf16_hi(int v) { return as_uint((float)v) & 0xFFFF0000u; }
inline uint int_bf16_lo(int v) { return as_uint((float)v) >> 16; }
#else
// One int4 nibble (index i of the word: k = 8 j + i) as the bf16 integer q - 8, in the low 16 bits
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
#endif

// The A operand of k16 step d for row block rb: lane l holds row (rb*8 + r)'s element 16 d + l in .s(r).
#define A_OPERAND(a, xa, rb, d)                                                                    \
  a.s0 = xa[(rb) * 8 + 0][d]; a.s1 = xa[(rb) * 8 + 1][d]; a.s2 = xa[(rb) * 8 + 2][d];              \
  a.s3 = xa[(rb) * 8 + 3][d]; a.s4 = xa[(rb) * 8 + 4][d]; a.s5 = xa[(rb) * 8 + 5][d];              \
  a.s6 = xa[(rb) * 8 + 6][d]; a.s7 = xa[(rb) * 8 + 7][d];

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG, 1, 1)))
#if LAYOUT == 2
__kernel void gemm_sm_i8w(__global const uchar* restrict w,
                          __global const float* restrict scales,
                          __global const ushort* restrict x,
                          __global float* restrict out) {
#else
__kernel void gemm_sm(__global const uint* restrict w,
                      __global const half* restrict scales,
                      __global const ushort* restrict x,
                      __global float* restrict out) {
#endif
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint col_sg = sg % NSGN;
  const uint ks = sg / NSGN;
  const uint n_tile = get_group_id(0) * NSGN + col_sg;
  const uint n0 = n_tile * SG;
  const uint n = n0 + lane;
  const uint s = get_group_id(1);
  const uint gbase = s * G_PER_S + ks;

  float8 acc[RB];
  for (int rb = 0; rb < RB; ++rb) acc[rb] = (float8)0.0f;

  for (uint i = 0; i < G_PER_SG; ++i) {
    const uint g = gbase + i * KSL;
    // ---- the activation rows of this group: [row][k16 step], lane l = element 16 d + l -------
    ushort xa[ROWS][4];
    for (int r = 0; r < ROWS; ++r) {
      ushort4 v = (ushort4)0;
      if (r < M) v = intel_sub_group_block_read_us4(x + (size_t)r * K + g * GROUP);
      xa[r][0] = v.s0; xa[r][1] = v.s1; xa[r][2] = v.s2; xa[r][3] = v.s3;
    }
#if LAYOUT == 2
    // ---- int8: four 256-byte [16 k][16 n] tiles, k16 steps 4 g .. 4 g + 3 -------------------
    __global const uchar* tile = w + ((size_t)n_tile * (K / 16) + g * 4u) * 256u;
    for (int d = 0; d < 4; ++d) {
      const char16 c = as_char16(intel_sub_group_block_read_uc16(tile + d * 256));
      int8 b;
      b.s0 = (int)(int_bf16_hi(c.s1) | int_bf16_lo(c.s0));
      b.s1 = (int)(int_bf16_hi(c.s3) | int_bf16_lo(c.s2));
      b.s2 = (int)(int_bf16_hi(c.s5) | int_bf16_lo(c.s4));
      b.s3 = (int)(int_bf16_hi(c.s7) | int_bf16_lo(c.s6));
      b.s4 = (int)(int_bf16_hi(c.s9) | int_bf16_lo(c.s8));
      b.s5 = (int)(int_bf16_hi(c.sb) | int_bf16_lo(c.sa));
      b.s6 = (int)(int_bf16_hi(c.sd) | int_bf16_lo(c.sc));
      b.s7 = (int)(int_bf16_hi(c.sf) | int_bf16_lo(c.se));
      for (int rb = 0; rb < RB; ++rb) {
        short8 a;
        A_OPERAND(a, xa, rb, d)
        acc[rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc[rb]);
      }
    }
#else
    // ---- int4: the group's eight u32 rows of this lane's column and its scale -----------------
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
    // Dequantised once into the B operand, used by every row block; the group's DPAS chain from
    // zero, then the g64 scale in fp32 (the checkpoint's semantics - no scale folded into bf16).
    float8 gacc[RB];
    for (int rb = 0; rb < RB; ++rb) gacc[rb] = (float8)0.0f;
    for (int d = 0; d < 4; ++d) {
      const uint w0 = wv[2 * d], w1 = wv[2 * d + 1];
      int8 b;
      b.s0 = pair_bf16(w0, 0); b.s1 = pair_bf16(w0, 1); b.s2 = pair_bf16(w0, 2); b.s3 = pair_bf16(w0, 3);
      b.s4 = pair_bf16(w1, 0); b.s5 = pair_bf16(w1, 1); b.s6 = pair_bf16(w1, 2); b.s7 = pair_bf16(w1, 3);
      for (int rb = 0; rb < RB; ++rb) {
        short8 a;
        A_OPERAND(a, xa, rb, d)
        gacc[rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, gacc[rb]);
      }
    }
    for (int rb = 0; rb < RB; ++rb) acc[rb] += gacc[rb] * scale;
#endif
  }

  // ---- the end of the K walk: acc[rb].s(r) = row rb*8 + r, column n ---------------------------
#if SPLITK == 1
  __local float red[KSL][NSGN][ROWS][SG];
  for (int rb = 0; rb < RB; ++rb)
    for (int r = 0; r < 8; ++r) red[ks][col_sg][rb * 8 + r][lane] = f8(acc[rb], r);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint e = ks; e < M; e += KSL) {
    float v = red[0][col_sg][e][lane];
    for (uint j = 1; j < KSL; ++j) v += red[j][col_sg][e][lane];
#if LAYOUT == 2
    v *= scales[n];
#endif
    out[((size_t)s * M + e) * N + n] = v;
  }
#else
#if LAYOUT == 2
  const float cs = scales[n];
#endif
  for (int rb = 0; rb < RB; ++rb)
    for (int r = 0; r < 8; ++r) {
      const int m = rb * 8 + r;
      if (m < M) {
#if LAYOUT == 2
        out[(size_t)m * N + n] = f8(acc[rb], r) * cs;
#else
        out[((size_t)s * M + m) * N + n] = f8(acc[rb], r);
#endif
      }
    }
#endif
}
