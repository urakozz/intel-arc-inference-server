// probe_moe_prefill.cl - PROBE kernels of plan 2026-11-12 (Ornith MoE prefill) Task 1: the grouped expert GEMM read
// straight from the int4 layout-1 expert blocks (no per-chunk bf16 weight pass). No runtime path binds them; the
// probe's control runs the production binaries (pf_moe_dequant_* + pf_moe_gemm), which is what makes the memcmp a
// gate. One source, the form by define:
//
//   FORM 1  bf16, each sub-group converts its own B columns in registers (pf_gemm_q4.cl's layout-1 read and its B
//           value statement for statement), work-group = WGM x WGN sub-groups of 8 NA rows x 16 NB columns
//   FORM 2  bf16, the work-group's B slab of one 64-k group converted ONCE into a double-buffered SLM image in the
//           VNNI register layout, shared by its WGM row sub-groups
//   FORM 3  W4A8: int8 activations x the int4 weights as the native s4 DPAS operand (intel_sub_group_i8_i4_matrix_
//           mad_k32: the nibble xor 8 is the whole conversion, exact), int32 per 64-k group, then the group's f16
//           weight scale (XSG 0: x the per-row scale in the epilogue) or the weight scale x the row's per-group
//           activation scale (XSG 1) in fp32 per group
//
// pmp_gemm(items, A, xs, W, out, ldx): work-group (item, n-block) computes item.rows rows (a multiple of 32, from the
// item's first sorted row) x WG columns of block item.e. items[4 i] = (block or NONE, first row, rows, -). A sub-group
// whose rows start past the item's end skips its DPAS (FORM 2: it still converts its share of B).
//
// CVT (FORMS 1-2), the B value rne((q - 8) x scale) as bf16 - every form the same value:
//   0  pf_gemm_q4.cl's q4_pair verbatim (sign-extending shifts, int -> float, x scale, software rne)
//   1  the same float, rounded by the hardware f32 -> bf16 conversion (RNE; cl_intel_bfloat16_conversions)
//   2  the magic exponent: as_float(0x41800000 | q << 19) = 16 + q, fma(that, scale, -24 scale) = (q - 8) scale
//      exactly (both products exact: 4-bit x 11-bit significands), then the hardware conversion
//   4  (phase 3 Task 1, FORM 1 only) the EXACT integer (q - 8) as bf16, no scale: the same magic exponent, 16 + q - 24
//      = q - 8 exactly, and the bf16 bits are the float's top half. The group's scale is applied to the group's DPAS
//      partial sum instead (below), so the weight the model sees is the checkpoint's integer exactly.
// EPI 1: gate||up's SiLU epilogue (pf_moe_gemm.cl's chain, verbatim) -> h bf16 [rows][N/2]; EPI 0: down,
// y = rne(acc) bf16 [rows][N]; EPI 2 (FORM 3, and CVT 4): the fp32 result before any epilogue -> [rows][N], the check.
//
// CVT 4's group loop (pf_moe_w4a8.cl's, with a bf16 A): a zeroed fp32 tile per 64-k group takes the group's four k16
// DPAS steps, then acc = fma(tile, scale[group][column], acc) - the scale is per lane (lane = column). The tile and
// A for the whole group are live together, so the register budget allows NA 4 (32 rows) only.
//
// OPT bit 8 (FORM 1, EPI 1; `o8` in the plan): A is read through the sorted row table instead of the gathered xg -
// `A` is the chunk's x [C][K] and the `xs` argument carries row_tok. One block read of a row's 32 k a message (the DPAS
// A operand needs lane = k, row = register, which a 2D block read gives only for consecutive rows); a NONE row reads
// zeros, rows past the item's last are clamped to it (their results are dropped by the store surface).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

#if !defined(K) || !defined(N) || !defined(FORM) || !defined(NA) || !defined(NB) || !defined(WGM) || !defined(WGN)
#error "probe_moe_prefill: K, N, FORM, NA, NB, WGM and WGN must be defined"
#endif
#ifndef EPI
#define EPI 0
#endif
#ifndef CVT
#define CVT 0
#endif
#ifndef XSG
#define XSG 0
#endif
// OPT bits: 1 the grid's n-blocks on dimension 0 (an item's work-groups dispatched together, so its A rows are read
// once into L2 instead of once per n-block); 2 FORM 2's slab through sub-group block reads / writes of SLM (one
// message for a lane's 8 dwords instead of 8).
#ifndef OPT
#define OPT 0
#endif
#define NFIRST (OPT & 1)
#define SLMIO ((OPT >> 1) & 1)
// OPT bit 4, DIAGNOSTIC: no DPAS (one add keeps the operands live) - prices the loads and the conversion alone.
#if OPT & 4
#define BF16_MAD(a, b, c) ((c) + (float8)((float)((int)(a).s0 + (b).s0)))
#else
#define BF16_MAD(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16((a), (b), (c))
#endif
#if SLMIO
#define SLAB_WRITE(p, v) intel_sub_group_block_write8((__local uint*)(p), (v));
#define SLAB_READ(p) intel_sub_group_block_read8((__local const uint*)(p))
#else
#define SLAB_WRITE(p, v)                                                              \
  {                                                                                   \
    __local uint* d_ = (p) + lane;                                                    \
    d_[0] = (v).s0; d_[16] = (v).s1; d_[32] = (v).s2; d_[48] = (v).s3;                \
    d_[64] = (v).s4; d_[80] = (v).s5; d_[96] = (v).s6; d_[112] = (v).s7;              \
  }
inline uint8 slab_read(__local const uint* p, uint lane) {
  p += lane;
  uint8 v;
  v.s0 = p[0]; v.s1 = p[16]; v.s2 = p[32]; v.s3 = p[48];
  v.s4 = p[64]; v.s5 = p[80]; v.s6 = p[96]; v.s7 = p[112];
  return v;
}
#define SLAB_READ(p) slab_read((p), lane)
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define GROUP 64
#define TILE_U32 136
#define R (8 * NA)               /* a sub-group's rows */
#define CN (16 * NB)             /* a sub-group's columns */
#define WG_N (CN * WGN)          /* a work-group's columns */
#define NSG (WGM * WGN)
#define G (K / GROUP)
#define BLK_U32 ((size_t)(N / 16) * (K / GROUP) * TILE_U32)
#if N % WG_N != 0 || K % GROUP != 0
#error "probe_moe_prefill: N whole work-group blocks, K whole 64-k groups"
#endif
#define INDEXED_A (((OPT) & 8) && EPI == 1)
#if ((OPT) & 8) && FORM != 1
#error "probe_moe_prefill: the indexed A (OPT bit 8) is FORM 1"
#endif
#if CVT == 4 && FORM != 1
#error "probe_moe_prefill: CVT 4 (exact integer weights) is FORM 1"
#endif
#if CVT == 4 && NA != 4
#error "probe_moe_prefill: CVT 4 holds a whole 64-k group of A and a tile per atom: NA 4"
#endif
#if EPI == 1 && NB % 2
#error "probe_moe_prefill: the SiLU epilogue pairs atoms (NB even)"
#endif
#if FORM == 3 && NA != 4
#error "probe_moe_prefill: W4A8 sub-groups are 32 rows (one 2D read of 32 int8 rows)"
#endif
#if FORM != 3 && NA != 4 && NA != 8
#error "probe_moe_prefill: bf16 sub-groups are 32 or 64 rows"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

#if FORM != 3
// One VNNI pair (k 2p low, 2p + 1 high) of column word w at the column's group scale.
inline uint q4_pair(uint w, int p, float scale) {
#if CVT == 0
  const uint u = w ^ 0x88888888u;
  const int lo = ((int)(u << (28 - 8 * p))) >> 28;
  const int hi = ((int)(u << (24 - 8 * p))) >> 28;
  return ((uint)rne_bf16((float)hi * scale) << 16) | (uint)rne_bf16((float)lo * scale);
#elif CVT == 1
  const uint u = w ^ 0x88888888u;
  const int lo = ((int)(u << (28 - 8 * p))) >> 28;
  const int hi = ((int)(u << (24 - 8 * p))) >> 28;
  return ((uint)intel_convert_bfloat16_as_ushort((float)hi * scale) << 16) |
         (uint)intel_convert_bfloat16_as_ushort((float)lo * scale);
#elif CVT == 3
  // Inline vISA: the signed nibble pair (q - 8: bfe on the xored word, sign-extended) into one register pair
  // interleaved lo / hi, int -> f32, then ONE mixed-mode multiply by the scale with a bf16 destination - the
  // hardware rounds the exact product (q - 8) x scale (a 15-bit significand) once, to nearest even: the same bf16
  // as rne_bf16((float)(q - 8) * scale), written straight into the VNNI pair (lo half = even k).
  const uint u = w ^ 0x88888888u;
  uint r;
#define Q4V(LO, HI)                                                                     \
  __asm__("{\n"                                                                         \
          ".decl PR_BF v_type=G type=BF num_elts=32 alias=<%0,0>\n"                     \
          ".decl PU_D v_type=G type=D num_elts=16 alias=<%1,0>\n"                       \
          ".decl PS_F v_type=G type=F num_elts=16 alias=<%2,0>\n"                       \
          ".decl PT_D v_type=G type=D num_elts=32 align=GRF\n"                          \
          ".decl PT_F v_type=G type=F num_elts=32 alias=<PT_D,0>\n"                     \
          "bfe (M1, 16) PT_D(0,0)<2> 0x4:d " #LO ":d PU_D(0,0)<1;1,0>\n"                \
          "bfe (M1, 16) PT_D(0,1)<2> 0x4:d " #HI ":d PU_D(0,0)<1;1,0>\n"                \
          "mov (M1, 32) PT_F(0,0)<1> PT_D(0,0)<1;1,0>\n"                                \
          "mul (M1, 32) PR_BF(0,0)<1> PT_F(0,0)<1;1,0> PS_F(0,0)<1;2,0>\n"              \
          "}\n"                                                                         \
          : "=rw"(r)                                                                    \
          : "rw"(u), "rw"(scale))
  if (p == 0) Q4V(0, 4);
  else if (p == 1) Q4V(8, 12);
  else if (p == 2) Q4V(16, 20);
  else Q4V(24, 28);
#undef Q4V
  return r;
#elif CVT == 4
  (void)scale;   // the integer, not the product: the scale is applied to the group's partial sum
  const uint blo = 8 * p <= 19 ? (w << (19 - 8 * p)) : (w >> (8 * p - 19));
  const uint bhi = 8 * p + 4 <= 19 ? (w << (15 - 8 * p)) : (w >> (8 * p + 4 - 19));
  const float flo = as_float((blo & 0x00780000u) | 0x41800000u) - 24.0f;   // 16 + q - 24 = q - 8, exact
  const float fhi = as_float((bhi & 0x00780000u) | 0x41800000u) - 24.0f;
  return (as_uint(fhi) & 0xFFFF0000u) | (as_uint(flo) >> 16);   // small integers: the low 16 bits are zero
#elif CVT == 9
  (void)scale;   // DIAGNOSTIC: no conversion (the words as B bits) - prices the conversion; not a B value
  return w ^ (0x01010101u * (uint)p);
#else
  const float c = -24.0f * scale;
  const uint blo = 8 * p <= 19 ? (w << (19 - 8 * p)) : (w >> (8 * p - 19));
  const uint bhi = 8 * p + 4 <= 19 ? (w << (15 - 8 * p)) : (w >> (8 * p + 4 - 19));
  const float flo = as_float((blo & 0x00780000u) | 0x41800000u);
  const float fhi = as_float((bhi & 0x00780000u) | 0x41800000u);
  return ((uint)intel_convert_bfloat16_as_ushort(fma(fhi, scale, c)) << 16) |
         (uint)intel_convert_bfloat16_as_ushort(fma(flo, scale, c));
#endif
}
// Two words (16 k) of one column -> one k16 half's 8 VNNI pairs (the slab GEMM's transform-read register image).
inline int8 q4_half(uint w0, uint w1, float scale) {
  uint8 v;
  v.s0 = q4_pair(w0, 0, scale);
  v.s1 = q4_pair(w0, 1, scale);
  v.s2 = q4_pair(w0, 2, scale);
  v.s3 = q4_pair(w0, 3, scale);
  v.s4 = q4_pair(w1, 0, scale);
  v.s5 = q4_pair(w1, 1, scale);
  v.s6 = q4_pair(w1, 2, scale);
  v.s7 = q4_pair(w1, 3, scale);
  return as_int8(v);
}
#endif

// The 136-u32 layout-1 tile of column atom n / 16 and group g of block e.
inline __global const uint* q4_tile(__global const uint* W, uint e, uint n, uint g) {
  return W + (size_t)e * BLK_U32 + ((size_t)(n / 16) * G + g) * TILE_U32;
}

__attribute__((reqd_work_group_size(SG * NSG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pmp_gemm(__global const uint* restrict items, __global const ushort* restrict A,
                       __global const float* restrict xs, __global const uint* restrict W,
                       __global void* restrict out, uint ldx) {
  const uint it = get_group_id(NFIRST ? 1 : 0);
  const uint e = items[4 * it];
  if (e == NONE) return;                              // uniform for the work-group
  const uint row0 = items[4 * it + 1];
  const uint rows = items[4 * it + 2];
  const uint s = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint sm = s / WGN, sn = s % WGN;
  const uint mb = row0 + R * sm;                      // this sub-group's first row
  const bool live = R * sm < rows;
  const uint nwg = get_group_id(NFIRST ? 0 : 1) * WG_N;
  const uint nb = nwg + CN * sn;                      // this sub-group's first column
  (void)xs; (void)ldx; (void)lane;

#if FORM == 1 || FORM == 2
  // ---- bf16 ---------------------------------------------------------------------------------------------------
#if FORM == 1
  if (!live) return;                                  // no barrier in FORM 1
#endif
#if INDEXED_A
  __global const uint* restrict rt = (__global const uint*)xs;   // row_tok
  const uint row_last = row0 + rows - 1u;
#else
  __global const ushort* restrict Ab = A + (size_t)mb * K;
  const int a_w = K * 2, a_h = R, a_p = K * 2;
#endif
  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);

#if FORM == 1
  uint wq[NB][8];
  float sc[NB];
#define LOAD_GROUP(gg, WQ, SC)                                                        \
  _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                    \
    __global const uint* tl = q4_tile(W, e, nb + 16u * (uint)b, (gg));                \
    const uint8 v = intel_sub_group_block_read8(tl);                                  \
    WQ[b][0] = v.s0; WQ[b][1] = v.s1; WQ[b][2] = v.s2; WQ[b][3] = v.s3;               \
    WQ[b][4] = v.s4; WQ[b][5] = v.s5; WQ[b][6] = v.s6; WQ[b][7] = v.s7;               \
    SC[b] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tl + 128))); \
  }
  LOAD_GROUP(0u, wq, sc)
  for (uint g = 0; g < G; ++g) {
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < G) { LOAD_GROUP(g + 1, wn, sn_) }
#if CVT == 4
    ushort afrag[2][NA / 4][64];   // the group's A: 32 rows x 64 k
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int r = 0; r < NA / 4; ++r) {
        const uint k0 = g * GROUP + (uint)h * 32u;
        intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                   a_w - (int)(2u * k0), 32, a_p, (int2)(0, 0), afrag[h][r]);
      }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      int8 bv[4];   // the four k16 steps of the group, exact integers
#pragma unroll
      for (int s4 = 0; s4 < 4; ++s4) bv[s4] = q4_half(wq[b][2 * s4], wq[b][2 * s4 + 1], sc[b]);
#pragma unroll
      for (int a = 0; a < NA; ++a) {
        float8 tile = (float8)(0.0f);
#pragma unroll
        for (int s4 = 0; s4 < 4; ++s4) {
          const short8 af = as_short8(vload8(0, afrag[s4 / 2][a / 4] + (uint)((s4 % 2) * 32 + 8 * (a % 4))));
          tile = BF16_MAD(af, bv[s4], tile);
        }
        acc[a][b] = fma(tile, (float8)(sc[b]), acc[a][b]);
      }
    }
#else
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint k0 = g * GROUP + (uint)h * 32u;
#if INDEXED_A
      ushort rv[NA][8][2];   // row (a, r)'s k0 .. k0 + 31 as two k16 steps, lane = k
#pragma unroll
      for (int a = 0; a < NA; ++a)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const uint t = rt[min(mb + 8u * (uint)a + (uint)r, row_last)];
          const ushort2 v = t == NONE ? (ushort2)(0) : intel_sub_group_block_read_us2(A + (size_t)t * K + k0);
          rv[a][r][0] = v.s0;
          rv[a][r][1] = v.s1;
        }
#else
      ushort afrag[NA / 4][64];
#pragma unroll
      for (int r = 0; r < NA / 4; ++r)
        intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                   a_w - (int)(2u * k0), 32, a_p, (int2)(0, 0), afrag[r]);
#endif
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        short8 af[NA];
#if INDEXED_A
#pragma unroll
        for (int a = 0; a < NA; ++a)
          af[a] = (short8)((short)rv[a][0][ks], (short)rv[a][1][ks], (short)rv[a][2][ks], (short)rv[a][3][ks],
                           (short)rv[a][4][ks], (short)rv[a][5][ks], (short)rv[a][6][ks], (short)rv[a][7][ks]);
#else
#pragma unroll
        for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(ks * 32 + 8 * (a % 4))));
#endif
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          const int8 bv = q4_half(wq[b][4 * h + 2 * ks], wq[b][4 * h + 2 * ks + 1], sc[b]);
#pragma unroll
          for (int a = 0; a < NA; ++a) acc[a][b] = BF16_MAD(af[a], bv, acc[a][b]);
        }
      }
    }
#endif
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
      for (int j = 0; j < 8; ++j) wq[b][j] = wn[b][j];
      sc[b] = sn_[b];
    }
  }
#undef LOAD_GROUP
#else   // FORM 2
  // The group's B slab: WG_N / 16 atoms x 4 k16 halves = NCH chunks of 8 u32 x 16 lanes, chunk c = atom 4 + half.
#define NCH ((WG_N / 16) * 4)
#define PER ((NCH + NSG - 1) / NSG)
  __local uint slab[2][NCH * 128];
  uint cw[PER][2];
  float cs[PER];
#define LOAD_CHUNKS(gg)                                                              \
  _Pragma("unroll") for (int i = 0; i < PER; ++i) {                                   \
    const uint c = s + NSG * (uint)i;                                                 \
    if (c < NCH) {                                                                    \
      __global const uint* tl = q4_tile(W, e, nwg + 16u * (c / 4u), (gg));            \
      const uint2 w2 = intel_sub_group_block_read2(tl + 32u * (c % 4u));             \
      cw[i][0] = w2.s0;                                                               \
      cw[i][1] = w2.s1;                                                               \
      cs[i] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tl + 128))); \
    }                                                                                 \
  }
#define CONVERT_CHUNKS(buf)                                                          \
  _Pragma("unroll") for (int i = 0; i < PER; ++i) {                                   \
    const uint c = s + NSG * (uint)i;                                                 \
    if (c < NCH) {                                                                    \
      const uint8 v = as_uint8(q4_half(cw[i][0], cw[i][1], cs[i]));                   \
      SLAB_WRITE(slab[buf] + c * 128u, v)                                             \
    }                                                                                 \
  }
  LOAD_CHUNKS(0u)
  CONVERT_CHUNKS(0)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint g = 0; g < G; ++g) {
    if (g + 1 < G) { LOAD_CHUNKS(g + 1) }
    if (live) {
      __local const uint* bs = slab[g & 1];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint k0 = g * GROUP + (uint)h * 32u;
        ushort afrag[NA / 4][64];
#pragma unroll
        for (int r = 0; r < NA / 4; ++r)
          intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                     a_w - (int)(2u * k0), 32, a_p, (int2)(0, 0), afrag[r]);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          short8 af[NA];
#pragma unroll
          for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(ks * 32 + 8 * (a % 4))));
#pragma unroll
          for (int b = 0; b < NB; ++b) {
            const int8 bv = as_int8(SLAB_READ(bs + ((sn * NB + (uint)b) * 4u + 2u * (uint)h + (uint)ks) * 128u));
#pragma unroll
            for (int a = 0; a < NA; ++a) acc[a][b] = BF16_MAD(af[a], bv, acc[a][b]);
          }
        }
      }
    }
    if (g + 1 < G) { CONVERT_CHUNKS((g + 1) & 1) }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#undef LOAD_CHUNKS
#undef CONVERT_CHUNKS
  if (!live) return;
#endif

#if EPI == 1
  __global ushort* restrict X = (__global ushort*)out + (size_t)mb * (N / 2);
  // The surface ends at the item's last row: a 64-row sub-group whose second half is past the item would
  // otherwise write the next item's rows (the hardware drops writes outside the surface).
  const int x_w = N, x_h = min((int)R, (int)(rows - R * sm)), x_p = N;
#define PF_SILU_ROW(r)                                             \
  do {                                                             \
    const ushort g_b = rne_bf16(gv.s##r), u_b = rne_bf16(uv.s##r); \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));             \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                     \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int p = 0; p < NB / 2; ++p) {
      const float8 gv = acc[a][2 * p], uv = acc[a][2 * p + 1];
      const uint ng = nb + 32u * (uint)p;
      ushort xv[8];
      PF_SILU_ROW(0); PF_SILU_ROW(1); PF_SILU_ROW(2); PF_SILU_ROW(3);
      PF_SILU_ROW(4); PF_SILU_ROW(5); PF_SILU_ROW(6); PF_SILU_ROW(7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p, (int2)((int)(ng >> 1), 8 * a), xv);
    }
#undef PF_SILU_ROW
#elif EPI == 2
  __global float* restrict Cf = (__global float*)out + (size_t)mb * N;   // the fp32 result, the check's
  const int c_w = N * 4, c_h = min((int)R, (int)(rows - R * sm)), c_p = N * 4;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Cf, c_w, c_h, c_p,
                                                 (int2)((int)(nb + 16u * (uint)b), 8 * a), (__private uint*)&acc[a][b]);
#else
  __global ushort* restrict Y = (__global ushort*)out + (size_t)mb * N;
  // The surface ends at the item's last row: a 64-row sub-group whose second half is past the item would
  // otherwise write the next item's rows (the hardware drops writes outside the surface).
  const int y_w = N * 2, y_h = min((int)R, (int)(rows - R * sm)), y_p = N * 2;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float8 v = acc[a][b];
      ushort yv[8];
      yv[0] = rne_bf16(v.s0); yv[1] = rne_bf16(v.s1); yv[2] = rne_bf16(v.s2); yv[3] = rne_bf16(v.s3);
      yv[4] = rne_bf16(v.s4); yv[5] = rne_bf16(v.s5); yv[6] = rne_bf16(v.s6); yv[7] = rne_bf16(v.s7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Y, y_w, y_h, y_p,
                                                 (int2)((int)(nb + 16u * (uint)b), 8 * a), yv);
    }
#endif

#else   // FORM 3: W4A8
  // ---- W4A8 ---------------------------------------------------------------------------------------------------
  if (!live) return;
  __global const uchar* restrict Ab = (__global const uchar*)A + (size_t)mb * K;   // int8 rows
  const int a_w = K, a_h = R, a_p = K;
  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);
  uint wq[NB][8];
  float sc[NB];
#define LOAD_GROUP(gg, WQ, SC)                                                        \
  _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                    \
    __global const uint* tl = q4_tile(W, e, nb + 16u * (uint)b, (gg));                \
    const uint8 v = intel_sub_group_block_read8(tl);                                  \
    WQ[b][0] = v.s0; WQ[b][1] = v.s1; WQ[b][2] = v.s2; WQ[b][3] = v.s3;               \
    WQ[b][4] = v.s4; WQ[b][5] = v.s5; WQ[b][6] = v.s6; WQ[b][7] = v.s7;               \
    SC[b] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tl + 128))); \
  }
  LOAD_GROUP(0u, wq, sc)
  for (uint g = 0; g < G; ++g) {
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < G) { LOAD_GROUP(g + 1, wn, sn_) }
    const uint k0 = g * GROUP;
    // 32 rows x 64 int8 k, one message: afrag[c 32 + r] = the k32 half c of row r, 2 int8 a lane.
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + k0), a_w - (int)k0, a_h, a_p, (int2)(0, 0),
                                               afrag);
#if XSG
    float8 xg[NA];   // the rows' activation scales of this group (uniform across lanes)
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int r = 0; r < 8; ++r) xg[a][r] = xs[(size_t)(mb + 8u * (uint)a + (uint)r) * G + g];
#endif
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const int4 b0 = as_int4(vload4(0, wq[b]) ^ (uint4)(0x88888888u));
      const int4 b1 = as_int4(vload4(1, wq[b]) ^ (uint4)(0x88888888u));
#pragma unroll
      for (int a = 0; a < NA; ++a) {
        int8 t = intel_sub_group_i8_i4_matrix_mad_k32(as_short8(vload8(0, afrag + (uint)(8 * a))), b0, (int8)(0));
        t = intel_sub_group_i8_i4_matrix_mad_k32(as_short8(vload8(0, afrag + (uint)(32 + 8 * a))), b1, t);
#if XSG
        acc[a][b] = fma(convert_float8(t), (float8)(sc[b]) * xg[a], acc[a][b]);
#else
        acc[a][b] = fma(convert_float8(t), (float8)(sc[b]), acc[a][b]);
#endif
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
      for (int j = 0; j < 8; ++j) wq[b][j] = wn[b][j];
      sc[b] = sn_[b];
    }
  }
#undef LOAD_GROUP
#if !XSG
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    float8 xr;
#pragma unroll
    for (int r = 0; r < 8; ++r) xr[r] = xs[mb + 8u * (uint)a + (uint)r];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] *= xr;
  }
#endif
#if EPI == 2
  __global float* restrict Cf = (__global float*)out + (size_t)mb * N;
  const int c_w = N * 4, c_h = R, c_p = N * 4;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Cf, c_w, c_h, c_p,
                                                 (int2)((int)(nb + 16u * (uint)b), 8 * a), (__private uint*)&acc[a][b]);
#elif EPI == 1
  __global ushort* restrict X = (__global ushort*)out + (size_t)mb * (N / 2);
  const int x_w = N, x_h = R, x_p = N;
#define PF_SILU_ROW(r)                                             \
  do {                                                             \
    const ushort g_b = rne_bf16(gv.s##r), u_b = rne_bf16(uv.s##r); \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));             \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                     \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int p = 0; p < NB / 2; ++p) {
      const float8 gv = acc[a][2 * p], uv = acc[a][2 * p + 1];
      const uint ng = nb + 32u * (uint)p;
      ushort xv[8];
      PF_SILU_ROW(0); PF_SILU_ROW(1); PF_SILU_ROW(2); PF_SILU_ROW(3);
      PF_SILU_ROW(4); PF_SILU_ROW(5); PF_SILU_ROW(6); PF_SILU_ROW(7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p, (int2)((int)(ng >> 1), 8 * a), xv);
    }
#undef PF_SILU_ROW
#else
  __global ushort* restrict Y = (__global ushort*)out + (size_t)mb * N;
  const int y_w = N * 2, y_h = R, y_p = N * 2;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float8 v = acc[a][b];
      ushort yv[8];
      yv[0] = rne_bf16(v.s0); yv[1] = rne_bf16(v.s1); yv[2] = rne_bf16(v.s2); yv[3] = rne_bf16(v.s3);
      yv[4] = rne_bf16(v.s4); yv[5] = rne_bf16(v.s5); yv[6] = rne_bf16(v.s6); yv[7] = rne_bf16(v.s7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Y, y_w, y_h, y_p,
                                                 (int2)((int)(nb + 16u * (uint)b), 8 * a), yv);
    }
#endif
#endif
}

// pmp_quant(x, q, xs, rows, kq): per row r of x bf16 [rows][K] (work-group = one row, 16 x 8 lanes), symmetric int8
// with one fp32 scale per row (kq = K) or per 64-k group (kq = 64): scale = max|x| / 127 (1 when the group is zero),
// q = rint(x / scale) clamped to [-127, 127]. W4A8's activation quantiser; timing and the integer check use it.
#if FORM == 3
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void pmp_quant(__global const ushort* restrict x, __global char* restrict q, __global float* restrict xs,
                        uint rows, uint kq) {
  __local float red[128];
  const uint r = get_group_id(0);
  const uint l = get_local_id(0);
  if (r >= rows) return;
  __global const ushort* xr = x + (size_t)r * K;
  for (uint k0 = 0; k0 < K; k0 += kq) {
    float m = 0.0f;
    for (uint k = k0 + l; k < k0 + kq; k += 128) m = fmax(m, fabs(bf16f(xr[k])));
    red[l] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint w = 64; w > 0; w >>= 1) {
      if (l < w) red[l] = fmax(red[l], red[l + w]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float sc = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = k0 + l; k < k0 + kq; k += 128)
      q[(size_t)r * K + k] = (char)clamp(rint(bf16f(xr[k]) / sc), -127.0f, 127.0f);
    if (l == 0) xs[(size_t)r * (K / kq) + k0 / kq] = sc;
  }
}
#endif
