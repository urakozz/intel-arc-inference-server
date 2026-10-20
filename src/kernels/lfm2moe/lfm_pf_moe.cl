// lfm_pf_moe.cl - spec 24d: LFM2.5-24B-A2B's routed experts over a PREFILL chunk - everything of the MoE block but the
// router (24c's lfm_router at M = kPfC), the route (24c's lfm_route at M = kPfC, grid (1, C)) and the grouped GEMMs
// (spec 15d's prefill/pf_moe_gemm.cl, unchanged, at K 2048 N 3072 SiLU and K 1536 N 2048). One binary
// (kernels::lfm2moe::pf_moe_variant), five entry points:
//
//   lfm_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)    grid (1), WG PF_WG (64)
//   lfm_pf_gather(x, hdr, row_tok, xg)                            grid (rows), WG 64
//   lfm_pf_dequant_gu(w_gu, hdr, out, b0, b1)                     grid (2I / 16, D / 64, b1 - b0), WG 16
//   lfm_pf_dequant_dn(w_dn, hdr, out, b0, b1)                     grid (D / 16, I / 64, b1 - b0), WG 16
//   lfm_pf_combine(route, pair_row, y, h, C)                      grid (D / 256, C), WG 256
//
// Kolibri's kol_pf_moe.cl structure (spec 20d: the sorted layout, the header, the tmax bound, the determinism
// argument), carried to this model - why it is a new source and kol_pf_moe.cl / k2_pf_moe.cl / pf_moe.cl are untouched:
//
//   * no shared expert: no block E of a weight batch, no shared rows after the routed ones, no shared tiles
//   * 64 experts, ids staged as UCHAR (PF_E <= 255), one expert a lane (PF_WG >= PF_E)
//   * the combine is lfm_moe_down's (24c): the route row's slots in RANK order, each product ROUNDED to bf16
//     (t_k = rne(f32(y_k) · w_k): grouped_mm's), summed in fp32 without fma (FP_CONTRACT OFF), one rounding, then the
//     residual fold h = rne(h + y) in place - the next norm is `_NF`, as in decode
//   * the expert address is lfm_expert.h's (the ONE macro, spec 15 decision 3) - the layer's 64 blocks one allocation
//
// **The sorted layout.** Every (token, slot) pair gets one ROW, expert-major, ascending token within an expert; each
// expert's rows are padded to whole TM-row TILES. The tile table (block, first row) is padded to a fixed tmax with
// block NONE, so every grid is a function of C alone and the host reads no count:
//
//   tmax(C) = floor((C x PF_K + PF_E x (TM - 1)) / TM)          318 at C = 2048 (no shared-expert term)
//
// (runtime::lfm2moe::pf_tiles is its host home.) The header (u32, kernels::lfm2moe::pf_hdr): [0] tiles used, [1] 0
// (reserved: the shared-row word of Kolibri's header, absent here), [2] rows used = TM x tiles, [3] C, [4 + e] expert
// e's rows (unpadded). Padding rows are zeros (the gather) and their outputs are never read; a row past C is never
// routed (the sort reads route rows [0, C) only) - no padding token reaches an expert.
//
// **Determinism.** No atomic anywhere. The sort is one work-group; each lane counts and then scatters its expert's
// pairs walking the chunk in ascending (token, slot) order, so every row's place is a function of the route rows
// alone. Gather and dequant are copies. The combine sums in the route row's rank order exactly as lfm_moe_down does
// (tests/kernels/lfm2moe_pf_ref.h repeats every entry point; EDIT THE TWO TOGETHER).
//
// **Portability.** The dequant's block read under cl_intel_subgroups, the plain loads it equals otherwise (lfm_moe.cl's
// idiom); scales through vload_half. Every entry point runs on the Mac's OpenCL 1.2 GPU for an indicative check
// (tools/mac/clrun/lfm2moe_run.cc).
#pragma OPENCL FP_CONTRACT OFF
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define LFM_PF_SG16 __attribute__((intel_reqd_sub_group_size(16)))
#else
#define LFM_PF_SG16
#endif

#if !defined(PF_E) || !defined(PF_K) || !defined(PF_WG) || !defined(PF_D) || !defined(PF_INTER) || !defined(TM) || !defined(KC)
#error "lfm_pf_moe: PF_E, PF_K, PF_WG, PF_D, PF_INTER, TM and KC must be defined (src/kernels/CMakeLists.txt, spec 24d)"
#endif
#if PF_K != 4 || PF_E <= PF_K || PF_E > PF_WG || PF_E > 255
#error "lfm_pf_moe: 4 slots (the route row), PF_K < PF_E <= the lanes (PF_WG), ids as uchar"
#endif
#if (PF_WG & (PF_WG - 1)) != 0 || PF_WG < 16 || PF_WG > 256
#error "lfm_pf_moe: PF_WG is a power of two in [16, 256]"
#endif
#if PF_D % 256 != 0 || PF_INTER % 64 != 0 || PF_D % 64 != 0 || TM != 32
#error "lfm_pf_moe: PF_D whole 256-column combine groups, PF_D / PF_INTER whole 64-k groups, TM 32 (pf_moe_gemm's)"
#endif

#define NONE 0xFFFFFFFFu
#define RW 16                   /* the route row (lfm_moe.cl's RW, kernels::lfm2moe::route) */
#define R_IDS 0
#define R_W 4
#define H_TILES 0
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define WG_GATHER 64
#define WG_COMBINE 256
#define TILE_U32 136            /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define GU_BLK_U32 ((size_t)(2 * PF_INTER / 16) * (PF_D / 64) * TILE_U32)
#define DN_BLK_U32 ((size_t)(PF_D / 16) * (PF_INTER / 64) * TILE_U32)
#include "lfm_expert.h"

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (lfm_moe.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// ---------------------------------------------------------------------------------------
// lfm_pf_sort - one work-group of PF_WG lanes; lane l owns expert l (l < PF_E):
//
//   1. stage the chunk's ids into SLM as uchar (min(id, PF_E - 1): lfm_route never writes a larger one, and a clamped
//      id keeps every pair inside the table);
//   2. lane l counts its expert's pairs c_l = #{(t, k) : id[t][k] == l};
//   3. lane 0's serial prefix over e ascending of ceil(c_e / TM) - expert e's first tile;
//   4. lane l writes its expert's tiles, then walks (t, k) ascending and gives each pair the next row of its expert:
//      row_tok[row] = t, pair_row[t][k] = row; the padding rows get NONE;
//   5. every lane: a share of the NONE tail of the tile table; the counts into the header.
//
// No pair is lost: the counts and the scatter read the same SLM ids, and a token's slots name distinct experts
// (lfm_route's ranks are a permutation), so expert e's rows [r0, r0 + c_e) are exactly its pairs.
__attribute__((reqd_work_group_size(PF_WG, 1, 1)))
__kernel void lfm_pf_sort(__global const uint* restrict route, __global uint* restrict hdr,
                          __global uint* restrict tiles, __global uint* restrict row_tok,
                          __global uint* restrict pair_row, uint C, uint tmax) {
  __local uchar sid[KC * PF_K];
  __local uint cnts[PF_WG];
  __local uint toff[PF_E + 1];
  const uint l = get_local_id(0);
  const uint n = min(C, (uint)KC);       // the host bounds C by kPfC; never past the SLM
  for (uint i = l; i < n * PF_K; i += PF_WG) {
    const uint id = route[(size_t)(i / PF_K) * RW + R_IDS + i % PF_K];
    sid[i] = (uchar)min(id, (uint)(PF_E - 1));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cnt = 0;
  if (l < PF_E)
    for (uint i = 0; i < n * PF_K; ++i) cnt += (uint)sid[i] == l ? 1u : 0u;
  cnts[l] = cnt;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) {
    uint acc = 0;
    for (uint j = 0; j < PF_E; ++j) {
      toff[j] = acc;
      acc += (cnts[j] + TM - 1) / TM;
    }
    toff[PF_E] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint ntiles = toff[PF_E];
  if (l < PF_E) {
    const uint t0 = toff[l];
    const uint nt = toff[l + 1] - t0;
    const uint r0 = t0 * TM;
    for (uint i = 0; i < nt; ++i)
      if (t0 + i < tmax) {
        tiles[2 * (t0 + i)] = l;
        tiles[2 * (t0 + i) + 1] = r0 + i * TM;
      }
    uint p = 0;
    for (uint t = 0; t < n; ++t)
      for (uint k = 0; k < PF_K; ++k)
        if ((uint)sid[t * PF_K + k] == l) {
          row_tok[r0 + p] = t;
          pair_row[t * PF_K + k] = r0 + p;
          ++p;
        }
    for (; p < nt * TM; ++p) row_tok[r0 + p] = NONE;
    hdr[H_COUNT + l] = cnt;
  }
  for (uint i = ntiles + l; i < tmax; i += PF_WG) {
    tiles[2 * i] = NONE;
    tiles[2 * i + 1] = 0;
  }
  if (l == 0) {
    hdr[H_TILES] = ntiles;
    hdr[1] = 0u;
    hdr[H_ROWS] = ntiles * TM;
    hdr[H_C] = n;
  }
}

// ---------------------------------------------------------------------------------------
// lfm_pf_gather - the grouped GEMM's A operand in sorted order: row r of xg is row row_tok[r] of the chunk's normed
// activations (bf16 [C][PF_D]); a padding row (NONE) is zeros. Rows past hdr[H_ROWS] belong to no tile and are not
// touched. kol_pf_gather's text.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void lfm_pf_gather(__global const uint4* restrict x, __global const uint* restrict hdr,
                            __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = PF_D * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : x[(size_t)t * W + i];
}

// ---------------------------------------------------------------------------------------
// The bf16 B operand of pf_moe_gemm.cl: weight blocks [b0, b1) of one layer into out[b - b0] = bf16 [K][N] row-major
// - expert b's int4 layout-1 block (lfm_expert.h's address), kol_pf_dequant's arithmetic statement for statement (the
// same xor, sign-extension shift and single RNE of (q - 8) x scale). An expert with no row this chunk (hdr[H_COUNT +
// b] == 0) returns at once - no tile reads it.
inline void lfm_pf_tile8(__global const uint* restrict tile, uint lane, uint* wv) {
#ifdef cl_intel_subgroups
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
#else
  for (uint j = 0; j < 8; ++j) wv[j] = tile[j * 16 + lane];
#endif
}
#define LFM_PF_DEQUANT(NAME, KK, NN, ADDR)                                                    \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                             \
  LFM_PF_SG16                                                                                 \
  __kernel void NAME(__global const uint* restrict w, __global const uint* restrict hdr,      \
                     __global ushort* restrict out, uint b0, uint b1) {                       \
    const uint lane = get_local_id(0);                                                        \
    const uint n_tile = get_group_id(0);                                                      \
    const uint g = get_group_id(1);                                                           \
    const uint bl = get_group_id(2);                                                          \
    const uint b = b0 + bl;                                                                   \
    if (b >= b1 || b >= PF_E) return;                                                         \
    if (hdr[H_COUNT + b] == 0u) return;                                                       \
    __global ushort* o = out + (size_t)bl * (KK) * (NN) + n_tile * 16 + lane;                 \
    __global const uint* tile = ADDR(w, b) + ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32;   \
    uint wv[8];                                                                               \
    lfm_pf_tile8(tile, lane, wv);                                                             \
    const float scale = vload_half(lane, (const __global half*)(tile + 128));                 \
    for (int j = 0; j < 8; ++j) {                                                             \
      const uint u = wv[j] ^ 0x88888888u;                                                     \
      for (int i = 0; i < 8; ++i) {                                                           \
        const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;                                     \
        const uint k = g * 64u + (uint)j * 8u + (uint)i;                                      \
        o[(size_t)k * (NN)] = rne_bf16((float)qm8 * scale);                                   \
      }                                                                                       \
    }                                                                                         \
  }

// gate||up blocks are K = PF_D, N = 2 x PF_INTER (interleave16, what pf_moe_gemm's SiLU epilogue reads); down blocks
// K = PF_INTER, N = PF_D.
LFM_PF_DEQUANT(lfm_pf_dequant_gu, PF_D, 2 * PF_INTER, LFM_EXPERT_GU)
LFM_PF_DEQUANT(lfm_pf_dequant_dn, PF_INTER, PF_D, LFM_EXPERT_DN)
#undef LFM_PF_DEQUANT

// ---------------------------------------------------------------------------------------
// lfm_pf_combine - token t, hidden column n = 256 x grid.x + lane: lfm_moe_down's epilogue over the down GEMM's sorted
// rows. y holds rne(down) per row (pf_moe_gemm's plain epilogue rounds once, as lfm_moe_down's rne(red)), so:
//
//   acc = t_0;  for k = 1 .. 3 (the route row's slots: RANK order): acc = acc + t_k,  t_k = rne(f32(y_k) x w_k)
//   h[t][n] = rne(f32(h[t][n]) + f32(rne(acc)))                     the residual fold, in place (one work-item a value)
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void lfm_pf_combine(__global const uint* restrict route, __global const uint* restrict pair_row,
                             __global const ushort* restrict y, __global ushort* restrict h, uint C) {
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float acc = bf16f(rne_bf16(bf16f(y[(size_t)pair_row[t * PF_K] * PF_D + n]) * as_float(rr[R_W])));
  for (uint k = 1; k < PF_K; ++k) {
    const float tk = bf16f(rne_bf16(bf16f(y[(size_t)pair_row[t * PF_K + k] * PF_D + n]) * as_float(rr[R_W + k])));
    acc = acc + tk;
  }
  __global ushort* restrict hp = h + (size_t)t * PF_D + n;
  *hp = rne_bf16(bf16f(*hp) + bf16f(rne_bf16(acc)));
}
