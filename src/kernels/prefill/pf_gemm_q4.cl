// pf_gemm_q4 - pf_gemm_small (pf_gemm_small.cl) reading the int4 weights directly: no bf16 slab
// (open-work 3b). After 3a a short chunk's five Qwen3.8 linears cost ~220 ms, ~100 of it
// pf_dequant_slab writing every weight as bf16 whatever the row count. Here each sub-group loads
// its columns' int4 words and group scales and converts them in registers, one launch for the
// linear's whole N; nothing else of pf_gemm_small changes.
//
// **Bit-identical to the slab walk** (pf_dequant_slab, then pf_gemm / pf_gemm_small). The B
// value is pf_dequant_slab's statement for statement - the same `^ 0x88888888`, the same
// `((int)(u << (28 - 4 i))) >> 28` sign extension, `(float)qm8 * scale` with the f16 group scale
// widened, the same single rne_bf16 - and the B fragment is the register image the slab's VNNI
// transform read returns (element j of a k16 half = the pair (k 2j, k 2j + 1), the even k low;
// the 2026-09-22 probe's pf_dequant8, measured bitwise over 71 M outputs). Each accumulator takes
// pf_gemm_small's DPAS chain (per 32-k tile the k16 half 0 then 1, from zero), so the output is
// pf_gemm's. tests/prefill/pf_gemm_test.cc holds the equality.
//
// The cost moves, it does not vanish: the conversion runs once per ROW TILE (every sub-group
// converts its own columns), so it grows with the rows while the slab's is fixed - a win for few
// rows only (docs/probes/probe-fused-small-chunk-2026-11-08.md).
//
// Weights (LAYOUT, loader/small_layout.h's int4 forms):
//   0  GPTQ-native: QW u32 [K/8][N] (nibble i of word (r, n) is k = 8 r + i), SC f16 [K/64][N]
//   1  tiled: per 16-column tile nt and 64-k group g, 136 u32 at (nt G + g) 136 - word j of
//      lane l at j 16 + l (k = 64 g + 8 j + i), then 16 f16 scales; SC unused
// One 64-k group per iteration (one scale per column; K % 64 == 0): its 8 words a column are
// loaded one group ahead into registers, and layout 0's are 2D-prefetched PFD groups ahead.
//
// Requirements (runtime::prefill::gemm_l0_q4 asserts them): N % (16 NB SN) == 0, K % 64 == 0,
// 64-byte aligned bases, 16-byte multiple pitches. M is the rows the grid covers (a multiple of
// 8 NA). SILU_EPI needs NB even (atoms pair as gate, up - pf_gemm.cl's parity argument).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#ifndef NA
#error "pf_gemm_q4: NA (8-row blocks per sub-group: 1, 2 or 4) must be defined"
#endif
#ifndef NB
#error "pf_gemm_q4: NB (16-column atoms per sub-group: 1, 2 or 4) must be defined"
#endif
#ifndef SN
#error "pf_gemm_q4: SN (sub-groups per work-group, along n) must be defined"
#endif
#ifndef LAYOUT
#error "pf_gemm_q4: LAYOUT (0 GPTQ-native, 1 tiled) must be defined"
#endif
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
#ifndef PFD
#define PFD 2   /* layout 0: the 2D prefetch distance, in 64-k groups */
#endif
#if NA != 1 && NA != 2 && NA != 4
#error "pf_gemm_q4: NA is 1, 2 or 4"
#endif
#if NB != 1 && NB != 2 && NB != 4
#error "pf_gemm_q4: NB is 1, 2 or 4"
#endif
#if SILU_EPI && (NB % 2)
#error "pf_gemm_q4: the SiLU epilogue pairs atoms (NB even)"
#endif

#define SG 16
#define WG_K 32
#define GROUP 64
#define TILE_U32 136
#define R (8 * NA)          /* a sub-group's rows */
#define CN (16 * NB)        /* a sub-group's columns */

// **Copied verbatim from pf_dequant_slab.cl** (and pf_prep.cl's, the SiLU chain's): round to
// nearest even at the f32 -> bf16 point.
inline ushort rne_bf16(float f) {
  const uint u = as_uint(f);
  const uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
#if SILU_EPI
// **Copied verbatim from pf_prep.cl** (pf_gemm.cl / pf_gemm_small.cl carry the same copy): both
// bf16 RNE steps, plain `exp` (never `native_exp`), silu the LAST factor.
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

// One VNNI pair of the already-xored word u: nibble 2p (even k, low half) and 2p + 1 (odd k, high
// half) at the column's group scale - pf_dequant_slab's `28 - 4 i` at i = 2p and 2p + 1.
inline uint q4_pair(uint u, int p, float scale) {
  const int lo = ((int)(u << (28 - 8 * p))) >> 28;
  const int hi = ((int)(u << (24 - 8 * p))) >> 28;
  return ((uint)rne_bf16((float)hi * scale) << 16) | (uint)rne_bf16((float)lo * scale);
}
// Two words (16 k) of one column -> one k16 half's 8 VNNI pairs, in the register order
// intel_sub_group_2d_block_read_transform_16b_32r16x2c gives the slab GEMM.
inline int8 q4_half(uint w0, uint w1, float scale) {
  const uint u0 = w0 ^ 0x88888888u, u1 = w1 ^ 0x88888888u;
  uint8 v;
  v.s0 = q4_pair(u0, 0, scale);
  v.s1 = q4_pair(u0, 1, scale);
  v.s2 = q4_pair(u0, 2, scale);
  v.s3 = q4_pair(u0, 3, scale);
  v.s4 = q4_pair(u1, 0, scale);
  v.s5 = q4_pair(u1, 1, scale);
  v.s6 = q4_pair(u1, 2, scale);
  v.s7 = q4_pair(u1, 3, scale);
  return as_int8(v);
}

// The A fragment of one 32-k tile: R rows x 32 k (pf_gemm_small.cl's read_a).
inline void read_a(__global const ushort* A, int a_w, int a_h, int a_p, int k0, int mb, __private ushort* afrag) {
#if NA == 4
  intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#elif NA == 2
  intel_sub_group_2d_block_read_16b_16r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#else
  intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k0, mb), afrag);
#endif
}
inline void prefetch_a(__global const ushort* A, int a_w, int a_h, int a_p, int k, int mb) {
#if NA == 4
  intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#elif NA == 2
  intel_sub_group_2d_block_prefetch_16b_16r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#else
  intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, a_w, a_h, a_p, (int2)(k, mb));
#endif
}

// One 64-k group g of atom b (columns n .. n + 15): the 8 words a column (word j = k 64 g + 8 j
// .. + 7) and the group scale.
inline void load_group(__global const uint* QW, __global const half* SC, uint K, uint N, uint g, uint n,
                       __private uint* w, __private float* scale) {
#if LAYOUT == 0
  intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)QW, (int)(N * 4u), (int)(K / 8u), (int)(N * 4u),
                                            (int2)((int)n, (int)(g * 8u)), w);
  *scale = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)SC + (size_t)g * N + n));
#else
  __global const uint* tile = QW + ((size_t)(n / SG) * (K / GROUP) + g) * TILE_U32;
  const uint8 v = intel_sub_group_block_read8(tile);
  w[0] = v.s0; w[1] = v.s1; w[2] = v.s2; w[3] = v.s3;
  w[4] = v.s4; w[5] = v.s5; w[6] = v.s6; w[7] = v.s7;
  *scale = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tile + 128)));
#endif
}

__attribute__((reqd_work_group_size(SG * SN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm_q4(__global const ushort* restrict A, __global const uint* restrict QW,
                         __global const half* restrict SC, __global float* restrict C_base, uint M,
                         uint K, uint N, uint lda, uint ldc) {
  const uint s = get_sub_group_id();
  // Column tiles on grid dimension 0, row tiles on 1 (pf_gemm_small.cl's order and its reason).
  const uint mb = get_group_id(1) * R;
  const uint nb = get_group_id(0) * (CN * SN) + CN * s;
#if SILU_EPI
  __global ushort* restrict X = (__global ushort*)C_base;   // bf16 x, `ldc` carries ldx
#else
  __global float* restrict C = C_base;
#endif
  const int a_w = (int)(K * 2u), a_h = (int)M, a_p = (int)(lda * 2u);
  const uint num_groups = K / GROUP;

  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);

  prefetch_a(A, a_w, a_h, a_p, 0, (int)mb);
  prefetch_a(A, a_w, a_h, a_p, WG_K, (int)mb);
#if LAYOUT == 0
#pragma unroll
  for (int d = 1; d <= PFD; ++d)
    if ((uint)d < num_groups)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)QW, (int)(N * 4u), (int)(K / 8u),
                                                      (int)(N * 4u), (int2)((int)(nb + 16u * b), 8 * d));
#endif
  // Group 0's words, then each iteration loads the next group's before converting this one's.
  uint wq[NB][8];
  float sc[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) load_group(QW, SC, K, N, 0u, nb + 16u * b, wq[b], &sc[b]);

  for (uint g = 0; g < num_groups; ++g) {
#if SN > 1
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < num_groups) {
#pragma unroll
      for (int b = 0; b < NB; ++b) load_group(QW, SC, K, N, g + 1, nb + 16u * b, wn[b], &sn_[b]);
    }
#if LAYOUT == 0
    if (g + 1 + PFD < num_groups)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)QW, (int)(N * 4u), (int)(K / 8u),
                                                      (int)(N * 4u),
                                                      (int2)((int)(nb + 16u * b), (int)((g + 1 + PFD) * 8u)));
#endif
#pragma unroll
    for (int h = 0; h < 2; ++h) {   // the group's two 32-k tiles, in k order
      const uint k0 = g * GROUP + (uint)h * WG_K;
      ushort afrag[2 * R];
      // A's k moves in the BASE (k0 bf16 = 64-byte multiples), not the block's x coordinate: with
      // the coordinate, IGC 2.41's layout-1 build read k 0..63 in every group (the read's payload
      // kept its first x; the layout-0 build, with 2D weight reads beside it, did not) - found by
      // probe_fused_small's bitwise gate, 2026-11-08.
      read_a(A + k0, a_w - (int)(2u * k0), a_h, a_p, 0, (int)mb, afrag);
      if (g + 1 < num_groups)
        prefetch_a(A + k0 + GROUP, a_w - (int)(2u * (k0 + GROUP)), a_h, a_p, 0, (int)mb);
      // pf_gemm's k16 halves, the same DPAS per accumulator in the same order.
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        short8 af[NA];
#pragma unroll
        for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * R + 8 * a)));
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          const int8 bv = q4_half(wq[b][4 * h + 2 * ks], wq[b][4 * h + 2 * ks + 1], sc[b]);
#pragma unroll
          for (int a = 0; a < NA; ++a)
            acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
        }
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
      for (int j = 0; j < 8; ++j) wq[b][j] = wn[b][j];
      sc[b] = sn_[b];
    }
#if SN > 1
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#endif
  }

#if SILU_EPI
  // pf_gemm's SiLU epilogue: atom 2p is a gate block, 2p + 1 its up block; x column = gate
  // column / 2 + lane.
  // x is N / 2 bf16 columns: N bytes wide (pf_gemm_small's N * 2 bound is a slab's, inside the
  // row pitch; here N is the whole linear's and the width must not pass the pitch).
  const int x_w = (int)N, x_h = (int)M, x_p = (int)(ldc * 2u);
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
