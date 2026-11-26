// pf_moe_q4gemm - plan 2026-11-12 (Ornith MoE prefill) Task 3: the grouped expert GEMM of a prefill chunk read
// straight from the int4 layout-1 expert blocks. pf_moe_gemm.cl's bf16 form needed every block of the layer
// converted to bf16 in a separate pass each chunk (pf_moe_dequant_*: 4.9 ms a layer and chunk, the whole array
// written and read back); here each sub-group converts its own B columns in registers as the k loop reaches them,
// so only the int4 bytes cross memory. Probe: tools/probe/probe_moe_prefill arm F1.a8b2m1n8.c2 - the probe measured the other forms and why this one.
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
#if defined(MS) && MS > 1
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#endif

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
// E1 register plan (research 2026-11-18, probe_moe_prefill's q4.* arms): NODB 1 loads a group's weights at its top
// instead of holding the next group's in registers; PF_W > 0 then prefetches the weight tiles PF_W groups ahead into
// L1, PF_A 1 the next 32-k step's A; AK16 1 reads A per k16 half. The same words, the same DPAS chain.
#ifndef NODB
#define NODB 0
#endif
#ifndef PF_W
#define PF_W 0
#endif
#ifndef PF_A
#define PF_A 0
#endif
#ifndef AK16
#define AK16 0
#endif
// SH_X 1 (Ornith's _SHX binary; Flash-Next's _SHX gate||up binaries, one card's and a tp card's, PERF-27): the shared
// expert's A rows are read from the chunk's x itself - its sorted rows are the tokens in order (row Rs + t is token t,
// Rs = hdr[1], pf_moe.cl's / q4_pf_moe.cl's H_SHARED_ROW), so the gather skips them (pf_moe_gather_routed,
// q4_pf_gather_routed). The same words. Its reads stay inside x: an item ends at most at Rs + C rounded up to whole
// 32-row tiles, and x has the chunk cap's rows (a multiple of 256).
#ifndef SH_X
#define SH_X 0
#endif
// MS > 1 (Inkling's NVFP4 / MXFP8 binaries): MS sub-groups along m (an item of MS x 64 rows) share one decode of the
// work-group's columns. Each 64-k group, the work-group's (16-column atom, k16 half) units are decoded once, each
// sub-group taking its share, into local memory as the DPAS B register image, then every sub-group runs its rows'
// DPAS from there. The decode decided the time (the FP codes take ~10 ALU ops a value, read by 64 rows only); the
// same B words, the same chain.
#ifndef MS
#define MS 1
#endif
#if MS > 1 && (WFMT == 0 || (NB * 4) % MS != 0)
#error "pf_moe_q4gemm: MS needs the FP formats, 64-row sub-groups and whole decode shares"
#endif
#if (PF_W || AK16) && !NODB
#error "pf_moe_q4gemm: PF_W and AK16 build on NODB"
#endif
#if PF_W
enum LSC_LDCC { LSC_LDCC_DEFAULT, LSC_LDCC_L1UC_L3UC, LSC_LDCC_L1UC_L3C, LSC_LDCC_L1C_L3UC, LSC_LDCC_L1C_L3C };
void __builtin_IB_lsc_prefetch_global_uint(const __global uint* base, int off, enum LSC_LDCC c);
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define GROUP 64
#include "expert_fp.h"
#define R (8 * NA)
#define CN (16 * NB)
#define WG_N (CN * WGN)
#define G (K / GROUP)
#define BLK_U32 ((size_t)(N / 16) * (K / GROUP) * TILE_U32 + EXPERT_OUTER_WORDS)
#if N % WG_N != 0 || K % GROUP != 0
#error "pf_moe_q4gemm: N whole work-group blocks, K whole 64-k groups"
#endif
#if NA != 4 && NA != 8
#error "pf_moe_q4gemm: a sub-group is 32 or 64 rows (whole 32-row A reads)"
#endif
#if SILU_EPI && NB % 2
#error "pf_moe_q4gemm: the SiLU epilogue pairs atoms (NB even)"
#endif

#include "bf16.h"
// pf_moe_gemm.cl's chain, verbatim (the numerics contract with pf_prep.cl).
#if SILU_EPI
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

#if WFMT == 0
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

#else
inline int8 fp_half(const __global uint* tile, uint lane, uint k) {
  const float scale = fp_tile_scale(tile, lane, k);
  uint pairs[8];
#pragma unroll
  for (uint p = 0; p < 8; ++p) {
    const uint i = 2 * p;
    const uint word = FP_TILE_WORD(tile, lane, k + i);
    const ushort lo = rne_bf16(fp_weight(word, i % FP_VALUES_PER_WORD, scale));
    const ushort hi = rne_bf16(fp_weight(word, (i + 1) % FP_VALUES_PER_WORD, scale));
    pairs[p] = (uint)lo | ((uint)hi << 16);
  }
  return as_int8(vload8(0, pairs));
}
#endif

#if MS > 1
__attribute__((reqd_work_group_size(SG * WGN * MS, 1, 1)))
#else
__attribute__((reqd_work_group_size(SG * WGN, 1, 1)))
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_moe_q4gemm(__global const uint* restrict items, __global const ushort* restrict A,
                            __global const uint* restrict W, __global const uint* restrict W_sh,
                            __global ushort* restrict out
#if SH_X
                            , __global const ushort* restrict XC, __global const uint* restrict hdr
#endif
                            ) {
  const uint it = get_group_id(1);
  const uint e = items[4 * it];
  if (e == NONE || (e == EXPERTS && W_sh == 0)) return;   // uniform; no barrier below (no W_sh: the shared block's
                                                          // bf16 form runs the batch path, Flash-Next's Intel form)
  if (e != EXPERTS && W == 0) return;   // no W: the shared block's items only (l0-int8's bf16 shared expert, open-work 16)
#if MS > 1
  const uint s = get_sub_group_id(), sm = s / WGN, sn = s % WGN;
  const uint row0 = items[4 * it + 1] + R * sm;       // this sub-group's 64 rows of the item
  const uint rows = items[4 * it + 2] > R * sm ? items[4 * it + 2] - R * sm : 0u;   // 0: decode only
  const uint nb = get_group_id(0) * WG_N + CN * sn;
  const uint mb = row0;
#else
  const uint row0 = items[4 * it + 1];
  const uint rows = items[4 * it + 2];
  const uint s = get_sub_group_id();
  const uint nb = get_group_id(0) * WG_N + CN * s;    // this sub-group's first column
  const uint mb = row0;                               // its first row (one sub-group a column block)
#endif
#if Q4_EXPERT_TABLE
  // the _TBL binaries (Flash-Next's expert tier, plan 21-QFN-v2a Task 8): W is the layer's table row, block e's address
  // at W[e] (u64; VRAM or pinned host memory)
  __global const uint* restrict Wb =
      e == EXPERTS ? W_sh : (__global const uint*)(intptr_t)((__global const ulong*)W)[e];
#else
  __global const uint* restrict Wb = e == EXPERTS ? W_sh : W + (size_t)e * BLK_U32;
#endif
  __global const ushort* restrict Ab = A + (size_t)mb * K;
#if SH_X
  if (e == EXPERTS) Ab = XC + (size_t)(mb - hdr[1]) * K;
#endif
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

#if WFMT != 0
  const uint fp_lane = get_sub_group_local_id();
#if MS > 1
  // Unit u = (atom u / 4, k16 half u % 4) of the work-group's WGN NB atoms; sub-group s decodes u = s + MS WGN i.
  // Three buffers and a split barrier a group: a sub-group decodes group g + 1, arrives, runs group g's DPAS, then
  // waits, so one sub-group's decode runs beside the others' DPAS. Group g + 1's buffer was last read by group
  // g - 2's DPAS, which every sub-group finished before it arrived at g - 1 (and the wait at g - 1 saw that).
#define UNITS (WGN * NB * 4)
#define UPS (UNITS / (MS * WGN))
#define FPH (16 / FP_VALUES_PER_WORD)
  __local uint slm[3][UNITS][128];   // lane l's VNNI word p at [p * 16 + l]
  const uint n_wg = get_group_id(0) * WG_N / 16u;
  uint uw[UPS][FPH], us[UPS];
#define UNIT_LOAD(gg)                                                                              \
  _Pragma("unroll") for (uint i = 0; i < UPS; ++i) {                                               \
    const uint u = s + MS * WGN * i, k = 16u * (u % 4u);                                           \
    const __global uint* tl = Wb + ((size_t)(n_wg + u / 4u) * G + (gg)) * TILE_U32;                \
    _Pragma("unroll") for (uint w = 0; w < FPH; ++w) uw[i][w] = FP_TILE_WORD(tl, fp_lane, k + w * FP_VALUES_PER_WORD); \
    us[i] = tl[32 * FP_CODE_BITS + (k / FP_BLOCK_K) * 4 + fp_lane / 4];                            \
  }
  UNIT_LOAD(0u)
#if WFMT == 1
  // NVFP4: a code byte's two E2M1 values as bf16 bits (exact), lo | hi << 16. An E2M1 x E4M3 product has at most 6
  // significant bits, so the bf16 of fp_weight's value is the product's truncation: one table read a pair.
  __local uint e2m1_pair[256];
  for (uint i = get_local_id(0); i < 256; i += get_local_size(0)) {
    const float lo = fp_weight(i, 0, 1.f), hi = fp_weight(i, 1, 1.f);
    e2m1_pair[i] = (as_uint(lo) >> 16) | (as_uint(hi) & 0xFFFF0000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
#endif
  for (uint g = 0; g < G; ++g) {
#if MS > 1
    __local uint (*buf)[128] = slm[g % 3u];
    for (uint dg = g == 0 ? 0u : g + 1; dg <= g + 1 && dg < G; ++dg) {   // group 0 first, then g + 1
    __local uint (*dst)[128] = slm[dg % 3u];
#pragma unroll
    for (uint i = 0; i < UPS; ++i) {
      const uint u = s + MS * WGN * i;
      const uint byte = (us[i] >> (8 * (fp_lane % 4))) & 255u;
#if WFMT == 1
      const float scale = fp_e4m3(byte);
      uint img[8];
#pragma unroll
      for (uint p = 0; p < 8; ++p) {
        const uint pair = e2m1_pair[(uw[i][p / 4] >> (8 * (p % 4))) & 255u];
        const float lo = as_float(pair << 16) * scale, hi = as_float(pair & 0xFFFF0000u) * scale;
        img[p] = (as_uint(lo) >> 16) | (as_uint(hi) & 0xFFFF0000u);
      }
      intel_sub_group_block_write8(dst[u], vload8(0, img));
#else
      const float scale = as_float(byte ? byte << 23 : 0x00400000u);
      uint img[8];
#pragma unroll
      for (uint p = 0; p < 8; ++p) {   // fp_half's pair p, from the registers
        const uint word = uw[i][(2 * p) / FP_VALUES_PER_WORD];
        const ushort lo = rne_bf16(fp_weight(word, (2 * p) % FP_VALUES_PER_WORD, scale));
        const ushort hi = rne_bf16(fp_weight(word, (2 * p + 1) % FP_VALUES_PER_WORD, scale));
        img[p] = (uint)lo | ((uint)hi << 16);
      }
      intel_sub_group_block_write8(dst[u], vload8(0, img));
#endif
    }
    if (dg + 1 < G) { UNIT_LOAD(dg + 1) }
    if (dg == 0) barrier(CLK_LOCAL_MEM_FENCE);
    }
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    if (rows == 0) {
      intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
      continue;
    }
#endif
#else
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
#if PF_W
  // A lane's 64-byte line of a 544-byte tile (lanes 0-8 cover it at either 32-byte phase), PF_W groups ahead, into L1
  // (OpenCL's prefetch() lowers to an L3-only message).
  const uint pf_off = 16u * min(get_sub_group_local_id(), 8u);
#define PREFETCH_GROUP(gg)                                                                                         \
  _Pragma("unroll") for (int b = 0; b < NB; ++b) __builtin_IB_lsc_prefetch_global_uint(                            \
      Wb + ((size_t)((nb + 16u * (uint)b) / 16u) * G + (gg)) * TILE_U32 + pf_off, 0, LSC_LDCC_L1C_L3C);
  for (uint d = 1; d < PF_W && d < G; ++d) { PREFETCH_GROUP(d) }
#endif
#if NODB
  for (uint g = 0; g < G; ++g) {
#if PF_W
    if (g + PF_W < G) { PREFETCH_GROUP(g + PF_W) }
#endif
    LOAD_GROUP(g, wq, sc)
#else
  LOAD_GROUP(0u, wq, sc)
  for (uint g = 0; g < G; ++g) {
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < G) { LOAD_GROUP(g + 1, wn, sn_) }
#endif
#endif
#pragma unroll
    for (int h = 0; h < 2; ++h) {   // the group's two 32-k tiles, in k order
      const uint k0 = g * GROUP + (uint)h * 32u;
#if !AK16
      ushort afrag[NA / 4][64];
#pragma unroll
      for (int r = 0; r < NA / 4; ++r)
        if (r == 0 || full)
          intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                     a_w - (int)(2u * k0), 32, a_p, (int2)(0, 0), afrag[r]);
#endif
#if PF_A
      if (k0 + 32u < K) {   // the next 32-k step's A
#pragma unroll
        for (int r = 0; r < NA / 4; ++r)
          if (r == 0 || full)
            intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)(Ab + (size_t)(32 * r) * K + k0 + 32u),
                                                           a_w - (int)(2u * k0) - 64, 32, a_p, (int2)(0, 0));
      }
#endif
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        short8 af[NA];
#if AK16
        ushort afrag[NA / 4][32];   // this k16 half's A: 32 rows x 16 k a block (x 16 ks: the surface stays >= 64 B)
#pragma unroll
        for (int r = 0; r < NA / 4; ++r)
          if (r == 0 || full)
            intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)(Ab + (size_t)(32 * r) * K + k0),
                                                       a_w - (int)(2u * k0), 32, a_p, (int2)(16 * ks, 0), afrag[r]);
#pragma unroll
        for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(8 * (a % 4))));
#else
#if INK_PREFILL
        // A 32-row item has no second fragment to read.
#pragma unroll
        for (int a = 0; a < NA; ++a)
          if (a < 4 || full)
            af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(ks * 32 + 8 * (a % 4))));
#else
#pragma unroll
        for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag[a / 4] + (uint)(ks * 32 + 8 * (a % 4))));
#endif
#endif
#pragma unroll
        for (int b = 0; b < NB; ++b) {
#if WFMT != 0 && MS > 1
          const int8 bv = as_int8(intel_sub_group_block_read8(buf[(NB * sn + (uint)b) * 4u + 2u * (uint)h + (uint)ks]));
#elif WFMT != 0
          const __global uint* tile = Wb + ((size_t)(nb / 16 + (uint)b) * G + g) * TILE_U32;
          const int8 bv = fp_half(tile, fp_lane, 32u * h + 16u * ks);
#else
          const int8 bv = q4_half(wq[b][4 * h + 2 * ks], wq[b][4 * h + 2 * ks + 1], sc[b]);
#endif
#pragma unroll
          for (int a = 0; a < NA; ++a)
            if (a < 4 || full) acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
        }
      }
    }
#if MS > 1
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#endif
#if WFMT == 0 && !NODB
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
      for (int j = 0; j < 8; ++j) wq[b][j] = wn[b][j];
      sc[b] = sn_[b];
    }
#endif
  }
#undef LOAD_GROUP
#undef PREFETCH_GROUP
#if MS > 1
#undef UNIT_LOAD
  if (rows == 0) return;   // past the barriers
#endif

#if WFMT == 1
  const float outer = as_float(Wb[BLK_U32 - 1]);
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] *= outer;
#endif

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
#elif defined(OUT_BF16_LANES)
  // Inkling's bf16-glue prompt pass: OUT_F32's per-lane stores with each sum rounded once to bf16 (the hardware's
  // round to nearest even). The 2D block-write epilogue below ran these shapes ~14 % slower than the fp32 stores.
  __global ushort* restrict Y = out + (size_t)mb * N;
  const uint lane = get_sub_group_local_id();
#define PF_BF16_ROW(r)                                                                 \
  if (8 * a + r < live_rows) Y[(size_t)(8 * a + r) * N + nb + 16u * (uint)b + lane] = intel_convert_bfloat16_as_ushort(v.s##r);
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float8 v = acc[a][b];
      PF_BF16_ROW(0) PF_BF16_ROW(1) PF_BF16_ROW(2) PF_BF16_ROW(3) PF_BF16_ROW(4) PF_BF16_ROW(5) PF_BF16_ROW(6) PF_BF16_ROW(7)
    }
#undef PF_BF16_ROW
#elif defined(OUT_F32)
  // Plan 21v2b Task 4 (a tensor-parallel card's down over its K half): the fp32 sums unrounded, [rows][N] float - the
  // two cards' partials are added before any rounding. Lane l owns column nb + 16 b + l; v.s<r> is row 8 a + r.
  __global float* restrict Y = (__global float*)out + (size_t)mb * N;
  const uint lane = get_sub_group_local_id();
#define PF_F32_ROW(r)                                                                  \
  if (8 * a + r < live_rows) Y[(size_t)(8 * a + r) * N + nb + 16u * (uint)b + lane] = v.s##r;
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float8 v = acc[a][b];
      PF_F32_ROW(0) PF_F32_ROW(1) PF_F32_ROW(2) PF_F32_ROW(3) PF_F32_ROW(4) PF_F32_ROW(5) PF_F32_ROW(6) PF_F32_ROW(7)
    }
#undef PF_F32_ROW
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
