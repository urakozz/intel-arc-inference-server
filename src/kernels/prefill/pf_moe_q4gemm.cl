// pf_moe_q4gemm - plan 2026-11-12 (Ornith MoE prefill) Task 3: the grouped expert GEMM of a prefill chunk read
// straight from the int4 layout-1 expert blocks. pf_moe_gemm.cl's bf16 form needed every block of the layer
// converted to bf16 in a separate pass each chunk (pf_moe_dequant_*: 4.9 ms a layer and chunk, the whole array
// written and read back); here each sub-group converts its own B columns in registers as the k loop reaches them,
// so only the int4 bytes cross memory. Probe: tools/probe/probe_moe_prefill (docs/probes/probe-moe-prefill-
// 2026-11-12.md), arm F1.a8b2m1n8.c2 - the record has the other forms measured and why this one.
//
// pf_moe_q4gemm(items, A, W, W_sh, out): grid (N / WG_N, ibound) - the n-blocks on dimension 0, so a work item's
// work-groups are dispatched together and read its A rows into L2 once. Work-group (n-block, item) computes the
// item's rows (a whole number of 32-row tiles, <= MI = 8 NA) x WG_N = 16 NB WGN columns of block items[4 i]:
// pf_moe_sort_mw.cl's work items (block, first row, rows, -), NONE past the used ones (the work-group returns at
// once; no barrier here). A sub-group is 8 NA rows x 16 NB columns; it skips its DPAS when its rows start past the
// item's end, and its stores stop at the item's last row (the 2D surface ends there: the hardware drops a write
// outside it), so no item writes another's rows.
//
// **Bitwise pf_moe_gemm over pf_moe_dequant_*'s bf16 blocks.** The B value is pf_moe_dequant's rne((q - 8) x scale)
// - here as fma(16 + q, scale, -24 scale) (16 + q from the magic exponent 0x41800000 | q << 19, both products exact
// for a 4-bit q and an f16 scale's 11-bit significand, so the fma's one rounding is the exact (q - 8) scale, the
// value pf_moe_dequant rounds) then the hardware's f32 -> bf16 conversion (round to nearest even, = rne_bf16 for
// these finite values) - in the register image the VNNI transform read gives the slab GEMM (element j of a k16
// half = (k 2j, k 2j + 1), even k low; pf_gemm_q4.cl's argument). Each accumulator takes pf_moe_gemm's DPAS chain:
// per 32-k tile the k16 half 0 then 1, k ascending from zero, no split-K. The epilogues are pf_moe_gemm.cl's,
// verbatim: SILU_EPI 1 gate||up (atom 2p gate, 2p + 1 its up) -> h bf16 [rows][N/2]; 0 down -> y = rne(acc)
// [rows][N]. Row independence holds as pf_moe_gemm's (a row's chain never meets another row's).
//
// The expert address: block e < EXPERTS at W + e x BLK (the layer's block base and the id - q4_pf_moe.cl's
// Q4_EXPERT_* line, spec 22's hook), the shared expert (e == EXPERTS) at W_sh (Ornith passes W + EXPERTS x BLK, its
// 257th block; Flash-Next its separate int4 block, or null for Intel's bf16 shared expert: those items return and the
// host runs that one block through q4_pf_dequant_*_shb + pf_moe_gemm).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !defined(K) || !defined(N) || !defined(EXPERTS)
#error "pf_moe_q4gemm: K, N (one expert block) and EXPERTS must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
#ifndef NA
#define NA 8            /* 8-row blocks a sub-group: 64 rows (half the conversion per DPAS of 32) */
#endif
#ifndef NB
#define NB 2            /* 16-column atoms a sub-group */
#endif
#ifndef WGN
#define WGN 8           /* sub-groups a work-group, along n */
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define GROUP 64
#define TILE_U32 136
#define R (8 * NA)
#define CN (16 * NB)
#define WG_N (CN * WGN)
#define G (K / GROUP)
#define BLK_U32 ((size_t)(N / 16) * (K / GROUP) * TILE_U32)
#if N % WG_N != 0 || K % GROUP != 0
#error "pf_moe_q4gemm: N whole work-group blocks, K whole 64-k groups"
#endif
#if NA != 4 && NA != 8
#error "pf_moe_q4gemm: a sub-group is 32 or 64 rows (whole 32-row A reads)"
#endif
#if SILU_EPI && NB % 2
#error "pf_moe_q4gemm: the SiLU epilogue pairs atoms (NB even)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// pf_moe_gemm.cl's chain, verbatim (the numerics contract with pf_prep.cl).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
#if SILU_EPI
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

// One VNNI pair (k 2p low, 2p + 1 high) of column word w (nibble i = k 8 j + i, unxored) at the column's scale.
inline uint q4_pair(uint w, int p, float scale) {
  const float c = -24.0f * scale;
  const uint blo = 8 * p <= 19 ? (w << (19 - 8 * p)) : (w >> (8 * p - 19));
  const uint bhi = 8 * p + 4 <= 19 ? (w << (15 - 8 * p)) : (w >> (8 * p + 4 - 19));
  const float flo = as_float((blo & 0x00780000u) | 0x41800000u);   // 16 + q
  const float fhi = as_float((bhi & 0x00780000u) | 0x41800000u);
  return ((uint)intel_convert_bfloat16_as_ushort(fma(fhi, scale, c)) << 16) |
         (uint)intel_convert_bfloat16_as_ushort(fma(flo, scale, c));
}
// Two words (16 k) of one column -> one k16 half's 8 VNNI pairs.
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

__attribute__((reqd_work_group_size(SG * WGN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_moe_q4gemm(__global const uint* restrict items, __global const ushort* restrict A,
                            __global const uint* restrict W, __global const uint* restrict W_sh,
                            __global ushort* restrict out) {
  const uint it = get_group_id(1);
  const uint e = items[4 * it];
  if (e == NONE || (e == EXPERTS && W_sh == 0)) return;   // uniform; no barrier below (no W_sh: the shared block's
                                                          // bf16 form runs the batch path, Flash-Next's Intel form)
  const uint row0 = items[4 * it + 1];
  const uint rows = items[4 * it + 2];
  const uint s = get_sub_group_id();
  const uint nb = get_group_id(0) * WG_N + CN * s;    // this sub-group's first column
  const uint mb = row0;                               // its first row (one sub-group a column block)
  __global const uint* restrict Wb = e == EXPERTS ? W_sh : W + (size_t)e * BLK_U32;
  __global const ushort* restrict Ab = A + (size_t)mb * K;
  const int a_w = K * 2, a_p = K * 2;
  // The A reads are one 32-row block each, the row offset in the BASE and the k offset in the base too (IGC 2.41
  // kept a 2D read's first coordinate across a loop when it moved in the coordinate: pf_gemm_q4.cl, 2026-11-08).
  // Task 4: a 64-row sub-group over a 32-row item (an expert's odd last tile) runs its first 32 rows' DPAS only -
  // uniform across the work-group; the rows it skips are past the item and never stored.
  const bool full = NA == 4 || rows >= 64u;

  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);

  uint wq[NB][8];
  float sc[NB];
#define LOAD_GROUP(gg, WQ, SC)                                                                     \
  _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                 \
    __global const uint* tl = Wb + ((size_t)((nb + 16u * (uint)b) / 16u) * G + (gg)) * TILE_U32;  \
    const uint8 v = intel_sub_group_block_read8(tl);                                               \
    WQ[b][0] = v.s0; WQ[b][1] = v.s1; WQ[b][2] = v.s2; WQ[b][3] = v.s3;                            \
    WQ[b][4] = v.s4; WQ[b][5] = v.s5; WQ[b][6] = v.s6; WQ[b][7] = v.s7;                            \
    SC[b] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tl + 128)));     \
  }
  LOAD_GROUP(0u, wq, sc)
  for (uint g = 0; g < G; ++g) {
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < G) { LOAD_GROUP(g + 1, wn, sn_) }
#pragma unroll
    for (int h = 0; h < 2; ++h) {   // the group's two 32-k tiles, in k order
      const uint k0 = g * GROUP + (uint)h * 32u;
      ushort afrag[NA / 4][64];
#pragma unroll
      for (int r = 0; r < NA / 4; ++r)
        if (r == 0 || full)
          intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                     a_w - (int)(2u * k0), 32, a_p, (int2)(0, 0), afrag[r]);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        short8 af[NA];
#pragma unroll
        for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(ks * 32 + 8 * (a % 4))));
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          const int8 bv = q4_half(wq[b][4 * h + 2 * ks], wq[b][4 * h + 2 * ks + 1], sc[b]);
#pragma unroll
          for (int a = 0; a < NA; ++a)
            if (a < 4 || full) acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
        }
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

  // The surface ends at the item's last row: a 64-row sub-group over a 32-row item writes only its first half.
  const int live_rows = min((int)R, (int)rows);
#if SILU_EPI
  __global ushort* restrict X = out + (size_t)mb * (N / 2);
  const int x_w = N, x_h = live_rows, x_p = N;
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
  __global ushort* restrict Y = out + (size_t)mb * N;
  const int y_w = N * 2, y_h = live_rows, y_p = N * 2;
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
}
