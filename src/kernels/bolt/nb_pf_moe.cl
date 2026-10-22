// nb_pf_moe.cl - spec 25d: Nemotron 3.5 Bolt's routed latent experts over a PREFILL chunk - everything of the MoE block's
// routed half but the router and the route (25c's nb_router / nb_route at M = 2048, grid (E / 16, C) and (1, C): decode's
// arithmetic row by row) and the grouped GEMMs (nb_pf_moe_gemm.cl: up with the relu2 epilogue, down plain). One binary per
// I (kernels::bolt::pf_moe_variant), five entry points:
//
//   nb_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax, n_exp)       grid (1), WG 256
//   nb_pf_gather(lat, hdr, row_tok, xg)                                    grid (tmax x 32), WG 64
//   nb_pf_dequant_up(w, hdr, out, b0, b1, n_exp)                           grid (I / 16, 1024 / 64, b1 - b0), WG 16
//   nb_pf_dequant_dn(w, hdr, out, b0, b1, n_exp)                           grid (1024 / 16, I / 64, b1 - b0), WG 16
//   nb_pf_combine(route, pair_row, y, r, C)                                grid (1024 / 256, C), WG 256
//
// lfm_pf_moe.cl's design (spec 24d, itself Kolibri's kol_pf_moe.cl) carried to Bolt - why a family file and those are
// untouched (Decision 19): top-9 (the 48-word route row), up to 512 experts per layer with E a kernel ARGUMENT (n_exp: 240 ..
// 496 by block - one binary per I serves every E, 25c's ruling), ids staged as ushort, two experts a lane (256 lanes, the
// route's own arrangement), the experts in the 1024 latent (nb_expert.h's ONE address macro: a block's E experts one
// allocation, up then down, layout 1), no shared expert in this file (Bolt's runs on the hidden 4096 as dense linears), and
// the combine is decode's nb_moe_down epilogue - NOT folded into h (fc2_latent's GEMM takes r; the next norm folds fc2 and
// shared_down, decode's two-source fold).
//
// **The sorted layout.** Every (row, slot) pair of the chunk's C REAL rows gets one sorted row, expert-major, ascending
// (row, slot) inside an expert (stable by row - Review Focus 3); each expert's rows are padded to whole TM-row tiles; the
// tile table (expert, first sorted row) is padded to a fixed tmax with NONE, so every grid is a function of (C, E) and the
// host reads no count:
//
//   tmax(C, E) = floor((9 C + E x (TM - 1)) / TM)          1056 at C 2048, E 496 (runtime::bolt::pf_tiles is its host home)
//
// The header (u32, kernels::bolt::pf_hdr): [0] tiles used, [1] 0, [2] rows used = TM x tiles, [3] C, [4 + e] expert e's
// rows (unpadded; 0 for e >= n_exp). Padding rows are zeros (the gather) and their outputs are never read; a row past C is
// never routed (the sort reads route rows [0, C) only) - **no padding row reaches an expert** (vllm-xpu-kernels 8b95ea6).
//
// **Determinism.** No atomic anywhere: one work-group sorts; each lane counts and scatters its two experts' pairs walking
// the chunk in ascending (row, slot) order. Gather and dequant are copies (the dequant: moe.cl's / pf_dequant_slab's xor,
// sign-extension shift and single RNE of (q - 8) x scale; an expert with no row this chunk returns at once). The combine:
//   acc = 0;  for the nine slots in ASCENDING id (the route row's order): acc = acc + (f32(y_k) · w_k)
//   r[t][n] = rne(acc)                                     the product rounded to fp32, then the add (FP_CONTRACT OFF)
// - nb_moe_down's epilogue on the down GEMM's rows, which round once (rne(acc): decode's rf(red)).
// tests/kernels/bolt_pf_ref.h (sort, gather, dequant_up / _dn, combine) repeats every entry point - EDIT THE TWO TOGETHER.
//
// **Portability.** The dequant's block read under cl_intel_subgroups, the plain loads it equals otherwise (nb_moe.cl's
// idiom); scales through vload_half. Every entry point runs on the Mac's OpenCL 1.2 GPU (tools/mac/clrun/bolt_pf_run.cc).
#pragma OPENCL FP_CONTRACT OFF
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define NB_PF_SG16 __attribute__((intel_reqd_sub_group_size(16)))
#else
#define NB_PF_SG16
#endif

#include "nb_common.h"

#if !defined(MOE_I) || !defined(TM) || !defined(PF_KC)
#error "nb_pf_moe: MOE_I, TM and PF_KC must be defined (src/kernels/CMakeLists.txt, spec 25d)"
#endif
#if TM != 32 || MOE_I % 64 != 0 || PF_KC > 2048
#error "nb_pf_moe: TM 32 (nb_pf_moe_gemm's tile), I whole 64-k groups, at most 2048 rows a chunk"
#endif
#include "nb_expert.h"

#define NONE 0xFFFFFFFFu
#define RW 48                   /* the route row (nb_moe.cl's RW, kernels::bolt::route) */
#define R_IDS 0
#define R_W 9
#define TOPK 9
#define H_TILES 0
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define MAXE 512
#define SORT_WG 256
#define WG_GATHER 64
#define WG_COMBINE 256
#define NB_LAT 1024
#define TILE_U32 136

// ---------------------------------------------------------------------------------------------------------------------
// nb_pf_sort - lane l owns experts l and l + 256 (< n_exp):
//   1. stage the chunk's ids into SLM as ushort (min(id, n_exp - 1): nb_route never writes a larger one, and a clamped id
//      keeps every pair inside the table);
//   2. lane l counts its experts' pairs;
//   3. lane 0's serial prefix over e ascending of ceil(c_e / TM) - expert e's first tile;
//   4. lane l writes its experts' tiles, then walks (t, k) ascending for each and gives every pair the next row of its
//      expert: row_tok[row] = t, pair_row[t x 9 + k] = row; the padding rows get NONE;
//   5. every lane: a share of the NONE tail of the tile table; the counts and the header.
__attribute__((reqd_work_group_size(SORT_WG, 1, 1)))
__kernel void nb_pf_sort(__global const uint* restrict route, __global uint* restrict hdr, __global uint* restrict tiles,
                         __global uint* restrict row_tok, __global uint* restrict pair_row, uint C, uint tmax,
                         uint n_exp) {
  __local ushort sid[PF_KC * TOPK];
  __local uint cnts[MAXE];
  __local uint toff[MAXE + 1];
  const uint l = get_local_id(0);
  const uint n = min(C, (uint)PF_KC);   // the host bounds C by kPfC; never past the SLM
  const uint E = min(n_exp, (uint)MAXE);
  for (uint i = l; i < n * TOPK; i += SORT_WG) {
    const uint id = route[(size_t)(i / TOPK) * RW + R_IDS + i % TOPK];
    sid[i] = (ushort)min(id, E - 1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint e0 = l, e1 = l + SORT_WG;
  uint c0 = 0, c1 = 0;
  for (uint i = 0; i < n * TOPK; ++i) {
    const uint v = sid[i];
    c0 += v == e0 ? 1u : 0u;
    c1 += v == e1 ? 1u : 0u;
  }
  cnts[e0] = e0 < E ? c0 : 0u;
  cnts[e1] = e1 < E ? c1 : 0u;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) {
    uint acc = 0;
    for (uint e = 0; e < E; ++e) {
      toff[e] = acc;
      acc += (cnts[e] + TM - 1) / TM;
    }
    toff[E] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint ntiles = toff[E];
  for (uint which = 0; which < 2; ++which) {
    const uint e = which ? e1 : e0;
    if (e >= MAXE) continue;
    hdr[H_COUNT + e] = e < E ? cnts[e] : 0u;
    if (e >= E) continue;
    const uint t0 = toff[e], nt = toff[e + 1] - t0, r0 = t0 * TM;
    for (uint i = 0; i < nt; ++i)
      if (t0 + i < tmax) {
        tiles[2 * (t0 + i)] = e;
        tiles[2 * (t0 + i) + 1] = r0 + i * TM;
      }
    uint p = 0;
    for (uint i = 0; i < n * TOPK; ++i)
      if ((uint)sid[i] == e) {
        row_tok[r0 + p] = i / TOPK;
        pair_row[i] = r0 + p;
        ++p;
      }
    for (; p < nt * TM; ++p) row_tok[r0 + p] = NONE;
  }
  for (uint i = ntiles + l; i < tmax; i += SORT_WG) {
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

// ---------------------------------------------------------------------------------------------------------------------
// nb_pf_gather - the up GEMM's A operand in sorted order: row r of xg is row row_tok[r] of the chunk's latent (bf16
// [C][1024], nb_sh_act's lat); a padding row (NONE) is zeros. Rows past hdr[H_ROWS] belong to no tile: untouched.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void nb_pf_gather(__global const uint4* restrict lat, __global const uint* restrict hdr,
                           __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = NB_LAT * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : lat[(size_t)t * W + i];
}

// ---------------------------------------------------------------------------------------------------------------------
// The bf16 B operands of nb_pf_moe_gemm.cl: experts [b0, b1) of one layer into out[b - b0] = bf16 [K][N] row-major. An
// expert with no row this chunk (hdr[H_COUNT + b] == 0) returns at once - no tile reads it.
inline void nb_pf_tile8(__global const uint* restrict tile, uint lane, uint* wv) {
#ifdef cl_intel_subgroups
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
#else
  for (uint j = 0; j < 8; ++j) wv[j] = tile[j * 16 + lane];
#endif
}
#define NB_PF_DEQUANT(NAME, KK, NN, ADDR)                                                       \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                               \
  NB_PF_SG16                                                                                    \
  __kernel void NAME(__global const uint* restrict w, __global const uint* restrict hdr,        \
                     __global ushort* restrict out, uint b0, uint b1, uint n_exp) {             \
    const uint lane = get_local_id(0);                                                          \
    const uint n_tile = get_group_id(0);                                                        \
    const uint g = get_group_id(1);                                                             \
    const uint bl = get_group_id(2);                                                            \
    const uint b = b0 + bl;                                                                     \
    if (b >= b1 || b >= n_exp || b >= MAXE) return;                                             \
    if (hdr[H_COUNT + b] == 0u) return;                                                         \
    __global ushort* o = out + (size_t)bl * (KK) * (NN) + n_tile * 16 + lane;                   \
    __global const uint* tile = ADDR(w, b) + ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32;     \
    uint wv[8];                                                                                 \
    nb_pf_tile8(tile, lane, wv);                                                                \
    const float scale = vload_half(lane, (const __global half*)(tile + 128));                   \
    for (int j = 0; j < 8; ++j) {                                                               \
      const uint u = wv[j] ^ 0x88888888u;                                                       \
      for (int i = 0; i < 8; ++i) {                                                             \
        const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;                                      \
        const uint k = g * 64u + (uint)j * 8u + (uint)i;                                        \
        o[(size_t)k * (NN)] = rne_bf16((float)qm8 * scale);                                     \
      }                                                                                         \
    }                                                                                           \
  }
// up blocks K = 1024 (the latent), N = I; down blocks K = I, N = 1024.
NB_PF_DEQUANT(nb_pf_dequant_up, NB_LAT, MOE_I, NB_EXPERT_UP)
NB_PF_DEQUANT(nb_pf_dequant_dn, MOE_I, NB_LAT, NB_EXPERT_DN)
#undef NB_PF_DEQUANT

// ---------------------------------------------------------------------------------------------------------------------
// nb_pf_combine - token t, latent column n = 256 x grid.x + lane: nb_moe_down's epilogue over the down GEMM's sorted rows.
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void nb_pf_combine(__global const uint* restrict route, __global const uint* restrict pair_row,
                            __global const ushort* restrict y, __global ushort* restrict r, uint C) {
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float acc = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const float v = bf16f(y[(size_t)pair_row[t * TOPK + k] * NB_LAT + n]) * as_float(rr[R_W + k]);
    acc = acc + v;
  }
  r[(size_t)t * NB_LAT + n] = rne_bf16(acc);
}
