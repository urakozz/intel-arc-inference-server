// kol_pf_moe.cl - spec 20d: Kolibri-1's routed experts over a PREFILL chunk - everything of the MoE
// block but the router GEMV (prefill/pf_gemv_bf16.cl), the route (20c's kol_moe.cl kol_route, decode's
// binary on grid (1, C)) and the grouped GEMMs (spec 15d's prefill/pf_moe_gemm.cl, unchanged, at
// Kolibri's shapes). One binary (kernels::kolibri::pf_moe_variant), five entry points:
//
//   kol_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)        grid (1), WG PF_WG (256)
//   kol_pf_gather(x, hdr, row_tok, xg)                               grid (rows), WG 64
//   kol_pf_dequant_gu(w_gu, w_sh_gu, hdr, out, b0, b1)                grid (2I / 16, D / 64, b1 - b0), WG 16
//   kol_pf_dequant_dn(w_dn, w_sh_dn, hdr, out, b0, b1)                grid (D / 16, I / 64, b1 - b0), WG 16
//   kol_pf_moe_combine(route, hdr, pair_row, y, mo, C)               grid (D / 256, C), WG 256
//
// Why not 15d's pf_moe.cl at Kolibri's defines - the three things this file
// changes, the design (the sorted layout, the header, the tmax bound, the determinism argument) being
// theirs, restated below:
//
//   * 384 experts. A byte-id sort stages the chunk's ids as BYTES (PF_E <= 255) and gives each lane ONE
//     expert (PF_WG >= PF_E, 256 lanes at most). Here the ids are staged as ushort
//     and lane l owns experts l and l + 256 (PF_EPL 2; 20c's route has the same 512 slots on 256 lanes).
//   * The shared expert is int4 (our export quantises it): block
//     PF_E of a weight batch is dequantised from ITS OWN layout-1 block (w_sh) exactly as a routed one.
//   * Kolibri's combine (20c's kol_moe_down epilogue): the six routed terms in ascending id, each product
//     rounded to fp32 THEN added (no fma: FP_CONTRACT OFF - the reference's index_add_ of y.float() * w),
//     + the shared row, ONE rounding to bf16, into `mo` - NOT the residual (post_ffn_norm comes first:
//     the sandwich's prep_res_fold SP0 + kol_post_add follow).
//
// **The sorted layout** (pf_moe.cl's). Every (token, slot) pair gets one ROW, expert-major, ascending
// token within an expert; each expert's rows are padded to whole TM-row TILES; the shared expert (block
// PF_E) follows with the C tokens in order. The tile table (block, first row) is padded to a fixed tmax
// with block NONE, so every grid is a function of C alone and the host reads no count:
//
//   tmax(C) = floor((C x PF_K + PF_E x (TM - 1)) / TM) + ceil(C / TM)          820 at C = 2048
//
// (runtime::kolibri::pf_tiles is its host home.) The header (u32, kernels::kolibri::pf_hdr): [0] tiles
// used, [1] the shared expert's first row, [2] rows used = TM x tiles, [3] C, [4 + e] expert e's rows
// (unpadded), [4 + PF_E] the shared expert's (C).
//
// **Determinism.** No atomic anywhere. The sort is one work-group; each lane counts and then scatters
// its experts' pairs walking the chunk in ascending (token, slot) order, so every row's place is a
// function of the route rows alone. Gather, dequant and copy are copies. The combine sums in the route
// row's slot order - ASCENDING EXPERT ID - exactly as kol_moe_down does (tests/kernels/kolibri_pf_ref.h
// repeats it and checks it against 20c's kolibri_ref::combine and torch's fixture row).
//
// No sub-group function but the int4 dequant's block read; scales through vload_half.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#if !defined(PF_E) || !defined(PF_K) || !defined(PF_WG) || !defined(PF_EPL) || !defined(PF_D) || !defined(PF_INTER) || !defined(TM) || !defined(KC)
#error "kol_pf_moe: PF_E, PF_K, PF_WG, PF_EPL, PF_D, PF_INTER, TM and KC must be defined (src/kernels/CMakeLists.txt, spec 20d)"
#endif
#if PF_K < 1 || PF_K > 8 || PF_E <= PF_K || PF_E > PF_WG * PF_EPL || PF_E > 65535
#error "kol_pf_moe: 1 <= PF_K <= 8 (the route row), PF_K < PF_E <= the lanes' experts (PF_WG x PF_EPL), ids as ushort"
#endif
#if (PF_WG & (PF_WG - 1)) != 0 || PF_WG < 16 || PF_WG > 256
#error "kol_pf_moe: PF_WG is a power of two in [16, 256]"
#endif
#if PF_D % 256 != 0 || PF_INTER % 64 != 0 || TM != 32
#error "kol_pf_moe: PF_D whole 256-column combine groups, PF_INTER whole 64-k groups, TM 32 (pf_moe_gemm's)"
#endif

#define NONE 0xFFFFFFFFu
#define RW 32                   /* the route row (kol_moe.cl's RW, kernels::kolibri::route) */
#define R_IDS 0
#define R_W 8
#define H_TILES 0
#define H_SHARED_ROW 1
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define WG_GATHER 64
#define WG_COMBINE 256
#define TILE_U32 136            /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */

#include "bf16.h"

// ---------------------------------------------------------------------------------------
// kol_pf_sort - one work-group of PF_WG lanes; lane l owns experts l + q x PF_WG, q < PF_EPL (< PF_E):
//
//   1. stage the chunk's ids into SLM as ushort (min(id, PF_E - 1): kol_route never writes a larger
//      one, and a clamped id keeps every pair inside the table);
//   2. lane l counts its experts' pairs c_e = #{(t, k) : id[t][k] == e} (one pass for all of them);
//   3. lane 0's serial prefix over e ascending of ceil(c_e / TM) - expert e's first tile;
//   4. lane l writes its experts' tiles, then walks (t, k) ascending and gives each pair the next row
//      of its expert: row_tok[row] = t, pair_row[t][k] = row; the padding rows get NONE;
//   5. every lane: a share of the shared expert's tiles and rows, of the NONE tail of the tile table;
//      the counts into the header.
//
// No pair is lost: the counts and the scatter read the same SLM ids, and a token's slots name distinct
// experts (kol_route's ranks are a permutation), so expert e's rows [r0, r0 + c_e) are exactly its pairs.
// tests/kernels/kolibri_pf_ref.h's sort() is this walk.
__attribute__((reqd_work_group_size(PF_WG, 1, 1)))
__kernel void kol_pf_sort(__global const uint* restrict route, __global uint* restrict hdr,
                          __global uint* restrict tiles, __global uint* restrict row_tok,
                          __global uint* restrict pair_row, uint C, uint tmax) {
  __local ushort sid[KC * PF_K];
  __local uint cnts[PF_WG * PF_EPL];
  __local uint toff[PF_E + 1];
  const uint l = get_local_id(0);
  const uint n = min(C, (uint)KC);       // the host bounds C by kPfC; never past the SLM
  for (uint i = l; i < n * PF_K; i += PF_WG) {
    const uint id = route[(size_t)(i / PF_K) * RW + R_IDS + i % PF_K];
    sid[i] = (ushort)min(id, (uint)(PF_E - 1));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cnt[PF_EPL];
  for (uint q = 0; q < PF_EPL; ++q) cnt[q] = 0;
  for (uint i = 0; i < n * PF_K; ++i) {
    const uint v = (uint)sid[i];
    if (v % PF_WG == l) cnt[v / PF_WG] += 1u;   // v < PF_E <= PF_WG x PF_EPL
  }
  for (uint q = 0; q < PF_EPL; ++q) cnts[l + q * PF_WG] = cnt[q];
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
  const uint ts = toff[PF_E];                    // the shared expert's first tile
  const uint st = (n + TM - 1) / TM;             // its tiles
  const uint ntiles = ts + st;
  const uint rs = ts * TM;                       // its first row
  for (uint q = 0; q < PF_EPL; ++q) {
    const uint e = l + q * PF_WG;
    if (e >= PF_E) continue;
    const uint t0 = toff[e];
    const uint nt = toff[e + 1] - t0;
    const uint r0 = t0 * TM;
    for (uint i = 0; i < nt; ++i)
      if (t0 + i < tmax) {
        tiles[2 * (t0 + i)] = e;
        tiles[2 * (t0 + i) + 1] = r0 + i * TM;
      }
    uint p = 0;
    for (uint t = 0; t < n; ++t)
      for (uint k = 0; k < PF_K; ++k)
        if ((uint)sid[t * PF_K + k] == e) {
          row_tok[r0 + p] = t;
          pair_row[t * PF_K + k] = r0 + p;
          ++p;
        }
    for (; p < nt * TM; ++p) row_tok[r0 + p] = NONE;
    hdr[H_COUNT + e] = cnt[q];
  }
  for (uint i = l; i < st; i += PF_WG)
    if (ts + i < tmax) {
      tiles[2 * (ts + i)] = PF_E;                // the shared expert's block
      tiles[2 * (ts + i) + 1] = rs + i * TM;
    }
  for (uint i = l; i < st * TM; i += PF_WG) row_tok[rs + i] = i < n ? i : NONE;
  for (uint i = ntiles + l; i < tmax; i += PF_WG) {
    tiles[2 * i] = NONE;
    tiles[2 * i + 1] = 0;
  }
  if (l == 0) {
    hdr[H_TILES] = ntiles;
    hdr[H_SHARED_ROW] = rs;
    hdr[H_ROWS] = ntiles * TM;
    hdr[H_C] = n;
    hdr[H_COUNT + PF_E] = n;
  }
}

// ---------------------------------------------------------------------------------------
// kol_pf_gather - the grouped GEMM's A operand in sorted order: row r of xg is row row_tok[r] of the
// chunk's normed activations (bf16 [C][PF_D]); a padding row (NONE) is zeros. Rows past hdr[H_ROWS]
// belong to no tile and are not touched.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void kol_pf_gather(__global const uint4* restrict x, __global const uint* restrict hdr,
                            __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = PF_D * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : x[(size_t)t * W + i];
}

// ---------------------------------------------------------------------------------------
// The bf16 B operand of pf_moe_gemm.cl: weight blocks [b0, b1) of one layer into out[b - b0] = bf16
// [K][N] row-major. Block b < PF_E: the routed expert's int4 layout-1 block (loader/kolibri1_layout.h:
// block b at b x its stride), the layout-1 dequant's arithmetic statement for statement (the same xor,
// sign-extension shift and single RNE of (q - 8) x scale); an expert with no row this chunk
// (hdr[H_COUNT + b] == 0) returns at once - no tile reads it. Block PF_E: the shared expert's own
// layout-1 block (w_sh, the same layout at the same widths), every chunk.
#define KOL_PF_DEQUANT(NAME, KK, NN)                                                          \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                             \
  __attribute__((intel_reqd_sub_group_size(16)))                                              \
  __kernel void NAME(__global const uint* restrict w, __global const uint* restrict w_sh,     \
                     __global const uint* restrict hdr, __global ushort* restrict out,        \
                     uint b0, uint b1) {                                                      \
    const uint lane = get_local_id(0);                                                        \
    const uint n_tile = get_group_id(0);                                                      \
    const uint g = get_group_id(1);                                                           \
    const uint bl = get_group_id(2);                                                          \
    const uint b = b0 + bl;                                                                   \
    if (b >= b1 || b > PF_E) return;                                                          \
    __global ushort* o = out + (size_t)bl * (KK) * (NN) + n_tile * 16 + lane;                 \
    if (b < PF_E && hdr[H_COUNT + b] == 0u) return;                                           \
    const size_t blk = (size_t)((NN) / 16) * ((KK) / 64) * TILE_U32;                          \
    __global const uint* tile = (b == PF_E ? w_sh : w + (size_t)b * blk) +                    \
                                ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32;                \
    const uint8 bw = intel_sub_group_block_read8(tile);                                       \
    uint wv[8];                                                                               \
    wv[0] = bw.s0; wv[1] = bw.s1; wv[2] = bw.s2; wv[3] = bw.s3;                               \
    wv[4] = bw.s4; wv[5] = bw.s5; wv[6] = bw.s6; wv[7] = bw.s7;                               \
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

// gate||up blocks are K = PF_D, N = 2 x PF_INTER (interleave16, what pf_moe_gemm's SiLU epilogue reads);
// down blocks K = PF_INTER, N = PF_D.
KOL_PF_DEQUANT(kol_pf_dequant_gu, PF_D, 2 * PF_INTER)
KOL_PF_DEQUANT(kol_pf_dequant_dn, PF_INTER, PF_D)
#undef KOL_PF_DEQUANT

// ---------------------------------------------------------------------------------------
#if INK_PREFILL
// INK_GLUE_BF16 (lossy): the expert GEMMs' rows arrive as bf16 (OUT_BF16_LANES: each sum rounded once), not fp32.
#if INK_GLUE_BF16
#define EXPERT_T ushort
#define EXPERT_LOAD(x) bf16f(x)
#else
#define EXPERT_T float
#define EXPERT_LOAD(x) (x)
#endif
// The softcapped SwiGLU of the gate||up GEMM's fp32 rows (pf_moe_q4gemm's OUT_F32 form, atom 2p gate, 2p + 1 its
// up): h = rne(4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25)), the GEMM epilogue's former text. As a pass of its own
// it is ~5x cheaper: inside the GEMM the transcendentals ran at the GEMM's few 256-register threads.
// Four outputs a work-item (16-byte loads and an 8-byte store).
// SWIGLU_WG: its work-group (96; a `--parallel tp` card's 576 columns take 48).
#ifndef SWIGLU_WG
#define SWIGLU_WG 96
#endif
__attribute__((reqd_work_group_size(SWIGLU_WG, 1, 1)))
__kernel void ink_prefill_swiglu(__global const EXPERT_T* restrict gu, __global ushort* restrict h) {
  const uint i = 4 * get_global_id(0), r = get_global_id(1);
  const size_t at = (size_t)r * 2 * PF_INTER + i / 16 * 32 + i % 16;
#if INK_GLUE_BF16
  const ushort4 gb = vload4(0, gu + at), ub = vload4(0, gu + at + 16);
  const float4 g4 = (float4)(bf16f(gb.s0), bf16f(gb.s1), bf16f(gb.s2), bf16f(gb.s3));
  const float4 u4 = (float4)(bf16f(ub.s0), bf16f(ub.s1), bf16f(ub.s2), bf16f(ub.s3));
#else
  const float4 g4 = vload4(0, gu + at), u4 = vload4(0, gu + at + 16);
#endif
  ushort out[4];
#pragma unroll
  for (uint c = 0; c < 4; ++c) {
    const float g = c == 0 ? g4.s0 : c == 1 ? g4.s1 : c == 2 ? g4.s2 : g4.s3;
    const float u = c == 0 ? u4.s0 : c == 1 ? u4.s1 : c == 2 ? u4.s2 : u4.s3;
    const float a = tanh(g / 4.f) * 4.f, b = tanh(u / 25.f) * 25.f;
    out[c] = rne_bf16(a * (1.f / (1.f + exp(-g))) * b);
  }
  vstore4(vload4(0, out), 0, h + (size_t)r * PF_INTER + i);
}

#pragma OPENCL FP_CONTRACT OFF
inline float ink_expert_sum(__global const uint* route, __global const uint* pair_row,
                            __global const EXPERT_T* y, uint r, uint i) {
  float value = 0.f;
  for (uint j = 0; j < PF_K; ++j)
    value += EXPERT_LOAD(y[(size_t)pair_row[(size_t)r * PF_K + j] * PF_D + i]) * as_float(route[(size_t)r * RW + R_W + j]);
  return value;
}

// A channel walks one slice of rows and recomputes the three sums across
// the slice boundary. The visit commits the new history after all slices: raw keeps only the chunk's last three
// rows, the ones the commit reads.
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void ink_prefill_moe_combine(__global const uint* restrict route,
                                     __global const uint* restrict pair_row,
                                     __global const EXPERT_T* restrict y,
                                     __global const float* restrict taps,
                                     __global const float* restrict history,
                                     __global float* restrict raw,
                                     __global float* restrict residual,
                                     __global ushort* restrict draft_tap, uint C) {
  const uint i = get_global_id(0);
  const uint first = get_group_id(1) * TM;
  float h0, h1, h2;
  if (first == 0) {
    h0 = history[i]; h1 = history[PF_D + i]; h2 = history[2 * PF_D + i];
  } else {
    h0 = ink_expert_sum(route, pair_row, y, first - 3, i);
    h1 = ink_expert_sum(route, pair_row, y, first - 2, i);
    h2 = ink_expert_sum(route, pair_row, y, first - 1, i);
  }
  const float4 w = vload4(i, taps);
  for (uint r = first; r < min(first + TM, C); ++r) {
    const float row = ink_expert_sum(route, pair_row, y, r, i);
    float sum = 0.f;
    sum += h0 * w.s0;
    sum += h1 * w.s1;
    sum += h2 * w.s2;
    sum += row * w.s3;
    const size_t at = (size_t)r * PF_D + i;
#if INK_MOE_TP_SUM
    raw[at] = row;   // `--parallel tp`: the card's part of the sum; the all-reduce, then the convolution (ink_post_conv)
    continue;
#endif
    const float value = residual[at] + (row + sum) / sqrt(5.f);
    if (r + 3 >= C) raw[at] = row;
    residual[at] = value;
    if (draft_tap) draft_tap[(size_t)r * 4 * PF_D + i] = rne_bf16(value);
    h0 = h1;
    h1 = h2;
    h2 = row;
  }
}
#else
// kol_pf_moe_combine - token t, hidden column n = 256 x grid.x + lane: kol_moe_down's epilogue over the
// down GEMM's sorted rows. y holds rne(down) per row (pf_moe_gemm's plain epilogue rounds once, as
// kol_moe_down's rne(red)), so:
//
//   acc = 0;  for j = 0 .. PF_K-1 (the route row's slots: ASCENDING expert id):
//             acc = acc + f32(y[pair_row[t][j]][n]) x w_j        the product rounded to fp32, then the add
//   mo[t][n] = rne(acc + f32(y[shared row + t][n]))             the UNGATED shared expert, one rounding
//
// - decode's chain with the down sums read instead of computed; NOT folded into the residual.
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void kol_pf_moe_combine(__global const uint* restrict route, __global const uint* restrict hdr,
                                 __global const uint* restrict pair_row, __global const ushort* restrict y,
                                 __global ushort* restrict mo, uint C) {
#pragma OPENCL FP_CONTRACT OFF
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float acc = 0.0f;
  for (uint j = 0; j < PF_K; ++j) {
    const float tj = bf16f(y[(size_t)pair_row[t * PF_K + j] * PF_D + n]) * as_float(rr[R_W + j]);
    acc = acc + tj;
  }
  const float sh = bf16f(y[(size_t)(hdr[H_SHARED_ROW] + t) * PF_D + n]);
  mo[(size_t)t * PF_D + n] = rne_bf16(acc + sh);
}
#endif
