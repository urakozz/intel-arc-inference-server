// probe_small_m.cl - spec 28 P0's stand-in small-M GEMMs (plan 28a): M = 1..16 activation
// rows against Qwen3.8's decode weights through DPAS, one dequant per weight tile.
//
// PROBE kernels: no runtime path binds `psm` / `psm_quant`, no production file is touched.
// They read exactly the bytes gemv / gemv_i8w read (int4 g64 symmetric in layout 0 or 1,
// the int8 lm_head's tiled bytes) and write exactly the partials gemv writes
// (out[(s M + m) N + n]; the head's out[m N + n], already scaled), so the control and the
// candidate are like for like.
//
// Compile-time knobs (tools/probe/CMakeLists.txt builds one binary per set):
//   M, K, N, S   rows (1..16), the weight shape, grid split-K slices (S | K/64)
//   LAYOUT       0: GPTQ qweight [K/8][N] u32 + f16 scales [K/64][N]; 1: 544-byte tiles
//                (common/repack.h); 2: the int8 head [N/16][K/16][16 k][16 n] + fp32 [N]
//   TYPE         1: bf16 DPAS (k16) - the weight dequantised to the bf16 integer q - 8 (int4)
//                or q (int8), exact; 2: int DPAS (k32) on int8 activations (psm_quant's per-row
//                scale) - i8 x i4 for int4, i8 x i8 for the head
//   ORIENT       0 (wB): weights on the B operand - a sub-group owns 16 output columns, one
//                per lane, the rows are the A operand in blocks of 8 (spec 28 §2's sketch);
//                1 (wA): weights on the A operand, 8 output columns per DPAS row, the tokens
//                on B, one per lane - M <= 16 in one DPAS column block (oneDNN's, §2.1)
//   SPLITK       0 (grid): dimension 1 of the grid is the slice, as gemv; 1 (wg): NSGN
//                column sub-groups x KSL K-slice sub-groups per work-group, slices
//                interleaved by group, reduced through SLM in slice order (§2.1)
//   NSGN, KSL    the work-group shape (grid: NSGN 4, KSL 1 - gemv's 64 work-items)
//   DEQ          0: xor + arithmetic shift (gemv's GEMV_DEQ_SHIFT); 1: the magic-exponent
//                form - the nibble OR'd into the mantissa of bf16 0x4300 (= 128 + q, exact)
//                and 136 subtracted in fp32 (exact)
//   PF_G         0 or a prefetch distance in this sub-group's own 64-k groups (2D block
//                prefetch for layout 0, prefetch() for the tiled layouts)
//   BLK2D        layout 0, wB: the weights by one 2D block read (gemv's GEMV_BLOCK2D)
//   I4A_SWAP     wA i8 x i4: nibble order inside an A byte (0: low nibble = even k)
//
// Order of operations (property R: a row's output depends only on its own row, and the
// per-row arithmetic is the same at every M): per 64-k group, the group's DPAS chain from a
// zero accumulator (4 x k16 bf16, 2 x k32 int), then acc += group * scale in fp32; the head
// chains every DPAS into one accumulator and applies the column scale at the end. Rows past
// M (wB) or token lanes past M (wA) are zero and never stored.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

#ifndef M
#error "probe_small_m: M"
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#ifndef TYPE
#define TYPE 1
#endif
#ifndef ORIENT
#define ORIENT 0
#endif
#ifndef SPLITK
#define SPLITK 0
#endif
#ifndef DEQ
#define DEQ 0
#endif
#ifndef PF_G
#define PF_G 0
#endif
#ifndef BLK2D
#define BLK2D 0
#endif
#ifndef I4A_SWAP
#define I4A_SWAP 0
#endif
#if SPLITK == 0
#undef NSGN
#undef KSL
#define NSGN 4
#define KSL 1
#endif
#ifndef NSGN
#define NSGN 2
#endif
#ifndef KSL
#define KSL 8
#endif

#define SG 16
#define WG (SG * NSGN * KSL)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define G_PER_SG (G_PER_S / KSL)
#define TILE_U32 136
#define RB ((M + 7) / 8)          /* wB: row blocks of 8 */
#define ROWS (RB * 8)

#if M < 1 || M > 16
#error "probe_small_m: M in [1, 16]"
#endif
#if (K % GROUP) != 0 || (G % S) != 0 || (G_PER_S % KSL) != 0
#error "probe_small_m: S | K/64 and KSL | K/64/S"
#endif
#if (N % (SG * NSGN)) != 0
#error "probe_small_m: N must be a multiple of 16 x NSGN"
#endif
#if LAYOUT == 2 && S != 1
#error "probe_small_m: the int8 head has no grid split-K (gemv_i8w's contract)"
#endif
#if BLK2D && (LAYOUT != 0 || ORIENT != 0)
#error "probe_small_m: BLK2D is layout 0, wB"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// Component r of a float8 (r is a compile-time constant once the loops unroll).
inline float f8(float8 v, int r) {
  switch (r) {
    case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
    case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7;
  }
}
// A small integer (|v| <= 127) as bf16 bits: exact, and the float's low 16 bits are zero.
inline uint int_bf16_hi(int v) { return as_uint((float)v) & 0xFFFF0000u; }
inline uint int_bf16_lo(int v) { return as_uint((float)v) >> 16; }

// One int4 nibble (index i of the word: k = 8r + i) as the bf16 integer q - 8, in the low
// 16 bits (lo) or the high 16 bits (hi) of the result.
#if DEQ == 0
inline int nib_s(uint word, uint i) { return ((int)((word ^ 0x88888888u) << (28u - 4u * i))) >> 28; }
inline uint nib_lo(uint word, uint i) { return int_bf16_lo(nib_s(word, i)); }
inline uint nib_hi(uint word, uint i) { return int_bf16_hi(nib_s(word, i)); }
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

#if TYPE == 1
#define ACC_T float8
#else
#define ACC_T int8
#endif

// ---------------------------------------------------------------------------------------
// The activation quantiser for TYPE 2 (pw4a8_quant's rule, tools/probe/probe_w4a8.cl): one
// work-group per row, scale = max|x| / 127, q = RNE(x / scale) saturated. In every binary
// so a TYPE 2 arm times quantiser + GEMM from one module.
#define QWG 256
__attribute__((reqd_work_group_size(QWG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void psm_quant(__global const ushort* restrict x, __global char* restrict xq,
                        __global float* restrict xs) {
  const uint m = get_group_id(0);
  const uint lid = get_local_id(0);
  __global const ushort* row = x + (size_t)m * K;
  __local float red[QWG / SG];
  float amax = 0.0f;
  for (uint k = lid; k < K; k += QWG) amax = fmax(amax, fabs(bf16f(row[k])));
  amax = sub_group_reduce_max(amax);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = 0; i < QWG / SG; ++i) amax = fmax(amax, red[i]);
  const float scale = amax / 127.0f;
  if (lid == 0) xs[m] = scale;
  for (uint k = lid; k < K; k += QWG)
    xq[(size_t)m * K + k] = scale == 0.0f ? (char)0 : convert_char_sat_rte(bf16f(row[k]) / scale);
}

// ---------------------------------------------------------------------------------------
// Prefetch of group gp's weights for this sub-group's 16 columns.
inline void prefetch_group(__global const uchar* w, uint n_tile, uint gp) {
#if PF_G
  gp = min(gp, (uint)(G - 1));
#if LAYOUT == 0
  intel_sub_group_2d_block_prefetch_32b_4r16x1c((__global void*)w, (int)(N * 4u), (int)(K / 8u),
                                                (int)(N * 4u), (int2)((int)(n_tile * SG), (int)(gp * 8u)));
  intel_sub_group_2d_block_prefetch_32b_4r16x1c((__global void*)w, (int)(N * 4u), (int)(K / 8u),
                                                (int)(N * 4u), (int2)((int)(n_tile * SG), (int)(gp * 8u + 4u)));
#elif LAYOUT == 1
  prefetch((__global const uint*)w + ((size_t)n_tile * G + gp) * TILE_U32, TILE_U32);
#else
  prefetch(w + ((size_t)n_tile * (K / 16) + gp * 4u) * 256u, 1024);
#endif
#endif
}

// The int4 group scale(s) this lane needs, as fp32.
#if LAYOUT == 0
#define SCALE_AT(sc, g, n) ((float)((__global const half*)(sc))[(size_t)(g) * N + (n)])
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void psm(__global const uchar* restrict w,       // int4 words / tiles, or int8 head bytes
                  __global const uchar* restrict sc,      // L0: f16 [K/64][N]; L2: fp32 [N]; L1: unused
                  __global const uchar* restrict x,       // TYPE 1: bf16 [M][K]; TYPE 2: int8 [M][K]
                  __global const float* restrict xs,      // TYPE 2: fp32 [M] row scales
                  __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint col_sg = sg % NSGN;
  const uint ks = sg / NSGN;
  const uint n_tile = get_group_id(0) * NSGN + col_sg;
  const uint n0 = n_tile * SG;
  const uint s = get_group_id(1);
  const uint gbase = s * G_PER_S + ks;
  __global const uint* w32 = (__global const uint*)w;

#if ORIENT == 0
  // ============================== wB: weights on B ======================================
  const uint n = n0 + lane;
  float8 acc[RB];
  for (int rb = 0; rb < RB; ++rb) acc[rb] = (float8)0.0f;
#if LAYOUT == 2 && TYPE == 2
  int8 iacc[RB];
  for (int rb = 0; rb < RB; ++rb) iacc[rb] = (int8)0;
#endif
  for (uint i = 0; i < G_PER_SG; ++i) {
    const uint g = gbase + i * KSL;
#if PF_G
    prefetch_group(w, n_tile, g + PF_G * KSL);
#endif
    // ---- the activation rows of this group, A operands --------------------------------
#if TYPE == 1
    ushort xa[ROWS][4];   // [row][k16 step]: element lane of the step
    for (int r = 0; r < ROWS; ++r) {
      ushort4 v = (ushort4)0;
      if (r < M) v = intel_sub_group_block_read_us4((__global const ushort*)x + (size_t)r * K + g * GROUP);
      xa[r][0] = v.s0; xa[r][1] = v.s1; xa[r][2] = v.s2; xa[r][3] = v.s3;
    }
#else
    ushort xa[ROWS][2];   // [row][k32 step]: bytes k0 + 2 lane, k0 + 2 lane + 1
    for (int r = 0; r < ROWS; ++r) {
      ushort2 v = (ushort2)0;
      if (r < M)
        v = intel_sub_group_block_read_us2((__global const ushort*)(x + (size_t)r * K + g * GROUP));
      xa[r][0] = v.s0; xa[r][1] = v.s1;
    }
#endif
    // ---- the weights: B operands, dequantised once for every row block -----------------
#if LAYOUT == 2
    __global const uchar* tile = w + ((size_t)n_tile * (K / 16) + g * 4u) * 256u;
#if TYPE == 1
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
        a.s0 = xa[rb * 8 + 0][d]; a.s1 = xa[rb * 8 + 1][d]; a.s2 = xa[rb * 8 + 2][d]; a.s3 = xa[rb * 8 + 3][d];
        a.s4 = xa[rb * 8 + 4][d]; a.s5 = xa[rb * 8 + 5][d]; a.s6 = xa[rb * 8 + 6][d]; a.s7 = xa[rb * 8 + 7][d];
        acc[rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc[rb]);
      }
    }
#else
    for (int h = 0; h < 2; ++h) {
      const int4 lo = as_int4(intel_sub_group_block_read_uc16(tile + (2 * h) * 256));
      const int4 hi = as_int4(intel_sub_group_block_read_uc16(tile + (2 * h + 1) * 256));
      const int8 b = (int8)(lo, hi);
      for (int rb = 0; rb < RB; ++rb) {
        short8 a;
        a.s0 = xa[rb * 8 + 0][h]; a.s1 = xa[rb * 8 + 1][h]; a.s2 = xa[rb * 8 + 2][h]; a.s3 = xa[rb * 8 + 3][h];
        a.s4 = xa[rb * 8 + 4][h]; a.s5 = xa[rb * 8 + 5][h]; a.s6 = xa[rb * 8 + 6][h]; a.s7 = xa[rb * 8 + 7][h];
        iacc[rb] = intel_sub_group_i8_i8_matrix_mad_k32(a, b, iacc[rb]);
      }
    }
#endif
#else   // int4
    uint wv[8];
    float scale;
#if LAYOUT == 0
#if BLK2D
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)w32, (int)(N * 4u), (int)(K / 8u),
                                              (int)(N * 4u), (int2)((int)n0, (int)(g * 8u)), wv);
#else
    for (int j = 0; j < 8; ++j) wv[j] = w32[(size_t)(g * 8 + j) * N + n];
#endif
    scale = SCALE_AT(sc, g, n);
#else
    __global const uint* tile = w32 + ((size_t)n_tile * G + g) * TILE_U32;
    const uint8 blk = intel_sub_group_block_read8(tile);
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
    scale = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tile + 128)));
#endif
#if TYPE == 1
    float8 gacc[RB];
    for (int rb = 0; rb < RB; ++rb) gacc[rb] = (float8)0.0f;
    for (int d = 0; d < 4; ++d) {
      const uint w0 = wv[2 * d], w1 = wv[2 * d + 1];
      int8 b;
      b.s0 = pair_bf16(w0, 0); b.s1 = pair_bf16(w0, 1); b.s2 = pair_bf16(w0, 2); b.s3 = pair_bf16(w0, 3);
      b.s4 = pair_bf16(w1, 0); b.s5 = pair_bf16(w1, 1); b.s6 = pair_bf16(w1, 2); b.s7 = pair_bf16(w1, 3);
      for (int rb = 0; rb < RB; ++rb) {
        short8 a;
        a.s0 = xa[rb * 8 + 0][d]; a.s1 = xa[rb * 8 + 1][d]; a.s2 = xa[rb * 8 + 2][d]; a.s3 = xa[rb * 8 + 3][d];
        a.s4 = xa[rb * 8 + 4][d]; a.s5 = xa[rb * 8 + 5][d]; a.s6 = xa[rb * 8 + 6][d]; a.s7 = xa[rb * 8 + 7][d];
        gacc[rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, gacc[rb]);
      }
    }
    for (int rb = 0; rb < RB; ++rb) acc[rb] += gacc[rb] * scale;
#else
    int8 giacc[RB];
    for (int rb = 0; rb < RB; ++rb) giacc[rb] = (int8)0;
    for (int h = 0; h < 2; ++h) {
      const int4 b = as_int4((uint4)(wv[4 * h], wv[4 * h + 1], wv[4 * h + 2], wv[4 * h + 3]) ^ 0x88888888u);
      for (int rb = 0; rb < RB; ++rb) {
        short8 a;
        a.s0 = xa[rb * 8 + 0][h]; a.s1 = xa[rb * 8 + 1][h]; a.s2 = xa[rb * 8 + 2][h]; a.s3 = xa[rb * 8 + 3][h];
        a.s4 = xa[rb * 8 + 4][h]; a.s5 = xa[rb * 8 + 5][h]; a.s6 = xa[rb * 8 + 6][h]; a.s7 = xa[rb * 8 + 7][h];
        giacc[rb] = intel_sub_group_i8_i4_matrix_mad_k32(a, b, giacc[rb]);
      }
    }
    for (int rb = 0; rb < RB; ++rb) acc[rb] += convert_float8(giacc[rb]) * scale;
#endif
#endif
  }
#if LAYOUT == 2 && TYPE == 2
  for (int rb = 0; rb < RB; ++rb) acc[rb] = convert_float8(iacc[rb]);
#endif
  // ---- the end of the K walk: per-lane values acc[rb].s(r) = row rb*8 + r, column n ----
#define ACC_N (RB * 8)
#if SPLITK == 1
  __local float red[KSL][NSGN][ACC_N][SG];
  for (int rb = 0; rb < RB; ++rb)
    for (int r = 0; r < 8; ++r) red[ks][col_sg][rb * 8 + r][lane] = f8(acc[rb], r);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint e = ks; e < ACC_N; e += KSL) {
    if (e >= M) break;
    float v = red[0][col_sg][e][lane];
    for (uint j = 1; j < KSL; ++j) v += red[j][col_sg][e][lane];
#if TYPE == 2
    v *= xs[e];
#endif
#if LAYOUT == 2
    v *= ((__global const float*)sc)[n];
#endif
    out[((size_t)s * M + e) * N + n] = v;
  }
#else
  for (int rb = 0; rb < RB; ++rb)
    for (int r = 0; r < 8; ++r) {
      const int m = rb * 8 + r;
      if (m < M) {
        float v = f8(acc[rb], r);
#if TYPE == 2
        v *= xs[m];
#endif
#if LAYOUT == 2
        v *= ((__global const float*)sc)[n];
#endif
        out[((size_t)s * M + m) * N + n] = v;
      }
    }
#endif

#else
  // ============================== wA: weights on A ======================================
  // Lane = token t. Output columns n0 + cb*8 + r for cb = 0, 1 (two DPAS row blocks).
  const uint t = lane;
  const uint tl = min(t, (uint)(M - 1));   // a valid row to load; lanes >= M are zeroed
  float8 acc[2] = {(float8)0.0f, (float8)0.0f};
#if LAYOUT == 2 && TYPE == 2
  int8 iacc[2] = {(int8)0, (int8)0};
#endif
  for (uint i = 0; i < G_PER_SG; ++i) {
    const uint g = gbase + i * KSL;
#if PF_G
    prefetch_group(w, n_tile, g + PF_G * KSL);
#endif
#if TYPE == 1
    // B operands: the token's 64 k as four k16 bf16-pair fragments.
    int8 b[4];
    {
      __global const uint* xr = (__global const uint*)(x + ((size_t)tl * K + g * GROUP) * 2);
      for (int d = 0; d < 4; ++d) {
        int8 v = as_int8(vload8(0, xr + d * 8));
        b[d] = t < M ? v : (int8)0;
      }
    }
#else
    int8 b[2];
    {
      __global const uint* xr = (__global const uint*)(x + (size_t)tl * K + g * GROUP);
      for (int h = 0; h < 2; ++h) {
        int8 v = as_int8(vload8(0, xr + h * 8));
        b[h] = t < M ? v : (int8)0;
      }
    }
#endif
#if LAYOUT == 2
    __global const uchar* tile = w + ((size_t)n_tile * (K / 16) + g * 4u) * 256u;
#if TYPE == 1
    for (int d = 0; d < 4; ++d) {
      // lane l: k = (g*4 + d)*16 + l, the 16 columns' bytes contiguous
      const char16 c = as_char16(vload16(0, tile + d * 256 + lane * 16));
      short8 a0, a1;
      a0.s0 = int_bf16_lo(c.s0); a0.s1 = int_bf16_lo(c.s1); a0.s2 = int_bf16_lo(c.s2); a0.s3 = int_bf16_lo(c.s3);
      a0.s4 = int_bf16_lo(c.s4); a0.s5 = int_bf16_lo(c.s5); a0.s6 = int_bf16_lo(c.s6); a0.s7 = int_bf16_lo(c.s7);
      a1.s0 = int_bf16_lo(c.s8); a1.s1 = int_bf16_lo(c.s9); a1.s2 = int_bf16_lo(c.sa); a1.s3 = int_bf16_lo(c.sb);
      a1.s4 = int_bf16_lo(c.sc); a1.s5 = int_bf16_lo(c.sd); a1.s6 = int_bf16_lo(c.se); a1.s7 = int_bf16_lo(c.sf);
      acc[0] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, b[d], acc[0]);
      acc[1] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, b[d], acc[1]);
    }
#else
    for (int h = 0; h < 2; ++h) {
      // lane l: k = k0 + 2l (low byte) and k0 + 2l + 1 (high byte), k0 = (g*4 + 2h)*16
      const uint kk = 2u * lane;
      __global const uchar* tt = tile + (2 * h + kk / 16) * 256 + (kk % 16) * 16;
      const uchar16 lo = vload16(0, tt), hi = vload16(0, tt + 16);
      const ushort16 sh = convert_ushort16(lo) | (convert_ushort16(hi) << (ushort)8);
      const short8 a0 = as_short8(sh.lo), a1 = as_short8(sh.hi);
      iacc[0] = intel_sub_group_i8_i8_matrix_mad_k32(a0, b[h], iacc[0]);
      iacc[1] = intel_sub_group_i8_i8_matrix_mad_k32(a1, b[h], iacc[1]);
    }
#endif
#else   // int4
    float8 sc0, sc1;
    {
#if LAYOUT == 0
      const float16 f = convert_float16(vload16(0, (__global const half*)sc + (size_t)g * N + n0));
#else
      const float16 f = convert_float16(vload16(0, (__global const half*)(w32 + ((size_t)n_tile * G + g) * TILE_U32 + 128)));
#endif
      sc0 = f.lo; sc1 = f.hi;
    }
#if LAYOUT == 0
#define WROW(j) (w32 + (size_t)(g * 8u + (j)) * N + n0)
#else
#define WROW(j) (w32 + ((size_t)n_tile * G + g) * TILE_U32 + (j) * 16u)
#endif
#if TYPE == 1
    float8 g0 = (float8)0.0f, g1 = (float8)0.0f;
    const uint ni = lane & 7u;             // this lane's nibble: k = 16d + lane -> word row 2d + lane/8
    for (int d = 0; d < 4; ++d) {
      const uint16 wr = vload16(0, WROW(2u * d + (lane >> 3)));
      short8 a0, a1;
      a0.s0 = nib_lo(wr.s0, ni); a0.s1 = nib_lo(wr.s1, ni); a0.s2 = nib_lo(wr.s2, ni); a0.s3 = nib_lo(wr.s3, ni);
      a0.s4 = nib_lo(wr.s4, ni); a0.s5 = nib_lo(wr.s5, ni); a0.s6 = nib_lo(wr.s6, ni); a0.s7 = nib_lo(wr.s7, ni);
      a1.s0 = nib_lo(wr.s8, ni); a1.s1 = nib_lo(wr.s9, ni); a1.s2 = nib_lo(wr.sa, ni); a1.s3 = nib_lo(wr.sb, ni);
      a1.s4 = nib_lo(wr.sc, ni); a1.s5 = nib_lo(wr.sd, ni); a1.s6 = nib_lo(wr.se, ni); a1.s7 = nib_lo(wr.sf, ni);
      g0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, b[d], g0);
      g1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, b[d], g1);
    }
    acc[0] += g0 * sc0;
    acc[1] += g1 * sc1;
#else
    int8 g0 = (int8)0, g1 = (int8)0;
    const uint bs = 8u * (lane & 3u);      // k = 32h + 2 lane -> word row 4h + lane/4, byte lane%4
    for (int h = 0; h < 2; ++h) {
      const uint16 wr = vload16(0, WROW(4u * h + (lane >> 2))) ^ (uint16)0x88888888u;
      const uint16 by = (wr >> bs) & 0xFFu;
#if I4A_SWAP
      const uint16 bb = ((by & 0xFu) << 4) | (by >> 4);
#else
      const uint16 bb = by;
#endif
      const char16 c = as_char16(convert_uchar16(bb));
      g0 = intel_sub_group_i4_i8_matrix_mad_k32(c.lo, b[h], g0);
      g1 = intel_sub_group_i4_i8_matrix_mad_k32(c.hi, b[h], g1);
    }
    acc[0] += convert_float8(g0) * sc0;
    acc[1] += convert_float8(g1) * sc1;
#endif
#undef WROW
#endif
  }
#if LAYOUT == 2 && TYPE == 2
  acc[0] = convert_float8(iacc[0]);
  acc[1] = convert_float8(iacc[1]);
#endif
  // ---- per-lane values acc[cb].s(r) = token t, column n0 + cb*8 + r --------------------
#if SPLITK == 1
  __local float red[KSL][NSGN][16][SG];
  for (int e = 0; e < 16; ++e) red[ks][col_sg][e][lane] = f8(acc[e / 8], e % 8);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint e = ks; e < 16; e += KSL) {
    float v = red[0][col_sg][e][lane];
    for (uint j = 1; j < KSL; ++j) v += red[j][col_sg][e][lane];
    if (t < M) {
#if TYPE == 2
      v *= xs[t];
#endif
#if LAYOUT == 2
      v *= ((__global const float*)sc)[n0 + e];
#endif
      out[((size_t)s * M + t) * N + n0 + e] = v;
    }
  }
#else
  if (t < M) {
    float16 v = (float16)(acc[0], acc[1]);
#if TYPE == 2
    v *= xs[t];
#endif
#if LAYOUT == 2
    v *= vload16(0, (__global const float*)sc + n0);
#endif
    vstore16(v, 0, out + ((size_t)s * M + t) * N + n0);
  }
#endif
#endif
}
