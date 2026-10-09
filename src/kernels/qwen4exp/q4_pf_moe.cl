// q4_pf_moe.cl - spec 21d: Qwen3.8-Flash-Next's routed experts over a PREFILL chunk - everything of the MoE block
// but the router GEMV (prefill/pf_gemv_bf16.cl at {2560, 528}), the route (21c's q4_moe.cl q4_route, decode's binary
// on grid (1, C)) and the grouped GEMMs (spec 15d's prefill/pf_moe_gemm.cl, unchanged, at this family's shapes). One
// binary (kernels::qwen4exp::pf_moe_variant), seven entry points:
//
//   q4_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)            grid (1), WG PF_WG (256)
//   q4_pf_gather(x, hdr, row_tok, xg)                                   grid (rows), WG 64
//   q4_pf_dequant_gu / _gu_shb(w_gu, w_sh, hdr, out, b0, b1)            grid (2I / 16, D / 64, b1 - b0), WG 16
//   q4_pf_dequant_dn / _dn_shb(w_dn, w_sh, hdr, out, b0, b1)            grid (D / 16, I / 64, b1 - b0), WG 16
//   q4_pf_moe_combine(route, hdr, pair_row, y, out, C)                  grid (D / 256, C), WG 256
//
// Kolibri's kol_pf_moe.cl (spec 20d) is the design - the sorted layout, the header, the tmax bound, the determinism
// argument, restated below - with this family's three differences:
//
//   * 512 experts, top 10. Ids up to 511 do not fit pf_moe_sort's uchar staging (prefill/pf_moe.cl:61, :117-124):
//     they are staged as ushort and lane l of 256 owns experts l and l + 256 (PF_EPL 2; 21c's q4_route has the
//     same 512 slots on 256 lanes). The route row is 21c's (ids at R_IDS in RANK order, weights at R_W, the shared
//     gate at R_SG: kernels::qwen4exp::route).
//   * The shared expert is block 512 of every weight batch in either form 21b loads: int4 layout-1 blocks (ours:
//     q4_pf_dequant_gu / _dn, the routed experts' dequant) or bf16 gemv_bf16 tiles (Intel's: the _shb entries COPY
//     the tiles into row-major [K][N], kol_pf_moe.cl's copy).
//   * The combine is 21c's q4_moe_down epilogue - the 10 routed terms in the route row's RANK order (grouped_mm's
//     topk slot order, spec 21 §12), each rne(rne(y_k) x w_k) then added (no fma: FP_CONTRACT OFF), r_b = rne(sum),
//     + sh_b = rne(rne(y_shared) x sg), one rounding - into `out` (the block's y), NOT into H: the next
//     q4_hc_combine_norm folds it, x the block's inject weights.
//
// **The sorted layout** (pf_moe.cl's). Every (token, slot) pair gets one ROW, expert-major, ascending token within an
// expert; each expert's rows are padded to whole TM-row TILES; the shared expert (block 512) follows with the C
// tokens in order. The tile table (block, first row) is padded to a fixed tmax with block NONE, so every grid is a
// function of C alone and the host reads no count:
//
//   tmax(C) = floor((C x PF_K + PF_E x (TM - 1)) / TM) + ceil(C / TM)          1200 at C = 2048
//
// (runtime::qwen4exp::pf_tiles is its host home.) The header (u32, kernels::qwen4exp::pf_hdr): [0] tiles used, [1]
// the shared expert's first row, [2] rows used = TM x tiles, [3] C, [4 + e] expert e's rows (unpadded), [4 + PF_E]
// the shared expert's (C).
//
// **Determinism.** No atomic anywhere. The sort is one work-group; each lane counts and then scatters its experts'
// pairs walking the chunk in ascending (token, slot) order, so every row's place is a function of the route rows
// alone. Gather, dequant and copy are copies. The combine sums in the route row's slot order exactly as q4_moe_down
// does (tests/kernels/qwen4exp_pf_ref.h repeats it and checks it against 21c's q4ref::down).
//
// **The expert address** (spec 15 decision 3, spec 22's hook): Q4_EXPERT_GU / Q4_EXPERT_DN below are the ONE place
// a routed expert's block is found from its id - the layer's block base and the id, never a computed layer offset -
// the same pair as q4_moe.cl's (no kernel in this tree #includes, so the two lines are repeated; spec 22 replaces
// both files' pair).
//
// **Portability.** No sub-group function but the dequant's block read, which tools/mac/opencl/intel_shim.h emulates
// exactly; scales through vload_half. Every entry point runs on the Mac's OpenCL 1.2 GPU for an indicative check
// (tools/mac/clrun/qwen4exp_run.cc).
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#if !defined(PF_E) || !defined(PF_K) || !defined(PF_WG) || !defined(PF_EPL) || !defined(PF_D) || !defined(PF_INTER) || !defined(TM) || !defined(KC)
#error "q4_pf_moe: PF_E, PF_K, PF_WG, PF_EPL, PF_D, PF_INTER, TM and KC must be defined (src/kernels/CMakeLists.txt, spec 21d)"
#endif
#if PF_K < 1 || PF_K > 15 || PF_E <= PF_K || PF_E > PF_WG * PF_EPL || PF_E > 65535
#error "q4_pf_moe: 1 <= PF_K <= 15 (the route row), PF_K < PF_E <= the lanes' experts (PF_WG x PF_EPL), ids as ushort"
#endif
#if (PF_WG & (PF_WG - 1)) != 0 || PF_WG < 16 || PF_WG > 256
#error "q4_pf_moe: PF_WG is a power of two in [16, 256]"
#endif
#if PF_D % 256 != 0 || PF_INTER % 64 != 0 || TM != 32
#error "q4_pf_moe: PF_D whole 256-column combine groups, PF_INTER whole 64-k groups, TM 32 (pf_moe_gemm's)"
#endif

#define NONE 0xFFFFFFFFu
#define RW 32                   /* the route row (q4_moe.cl's RW, kernels::qwen4exp::route) */
#define R_IDS 0
#define R_W 16
#define R_SG 26
#define H_TILES 0
#define H_SHARED_ROW 1
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define WG_GATHER 64
#define WG_COMBINE 256
#define TILE_U32 136            /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define GU_BLK_U32 ((size_t)(2 * PF_INTER / 16) * (PF_D / 64) * TILE_U32)
#define DN_BLK_U32 ((size_t)(PF_D / 16) * (PF_INTER / 64) * TILE_U32)

// ---- the expert address: spec 22 replaces exactly these two lines (and q4_moe.cl's pair) -------------------------
#define Q4_EXPERT_GU(base, id) ((base) + (size_t)(id) * GU_BLK_U32)
#define Q4_EXPERT_DN(base, id) ((base) + (size_t)(id) * DN_BLK_U32)
// --------------------------------------------------------------------------------------------------------------

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (q4_moe.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }

// ---------------------------------------------------------------------------------------
// q4_pf_sort - one work-group of PF_WG lanes; lane l owns experts l + q x PF_WG, q < PF_EPL (< PF_E):
//
//   1. stage the chunk's ids into SLM as ushort (min(id, PF_E - 1): q4_route never writes a larger one, and a
//      clamped id keeps every pair inside the table);
//   2. lane l counts its experts' pairs c_e = #{(t, k) : id[t][k] == e} (one pass for all of them);
//   3. lane 0's serial prefix over e ascending of ceil(c_e / TM) - expert e's first tile;
//   4. lane l writes its experts' tiles, then walks (t, k) ascending and gives each pair the next row of its expert:
//      row_tok[row] = t, pair_row[t][k] = row; the padding rows get NONE;
//   5. every lane: a share of the shared expert's tiles and rows, of the NONE tail of the tile table; the counts
//      into the header.
//
// No pair is lost: the counts and the scatter read the same SLM ids, and a token's slots name distinct experts
// (q4_route's ranks are a permutation), so expert e's rows [r0, r0 + c_e) are exactly its pairs.
// tests/kernels/qwen4exp_pf_ref.h's sort() is this walk.
__attribute__((reqd_work_group_size(PF_WG, 1, 1)))
__kernel void q4_pf_sort(__global const uint* restrict route, __global uint* restrict hdr,
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
// q4_pf_gather - the grouped GEMM's A operand in sorted order: row r of xg is row row_tok[r] of the chunk's block
// input (bf16 [C][PF_D]); a padding row (NONE) is zeros. Rows past hdr[H_ROWS] belong to no tile and are not
// touched. kol_pf_gather's text.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void q4_pf_gather(__global const uint4* restrict x, __global const uint* restrict hdr,
                           __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = PF_D * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : x[(size_t)t * W + i];
}

// ---------------------------------------------------------------------------------------
// The bf16 B operand of pf_moe_gemm.cl: weight blocks [b0, b1) of one layer into out[b - b0] = bf16 [K][N]
// row-major. Block b < PF_E: the routed expert's int4 layout-1 block (loader/qwen4exp_layout.h: Q4_EXPERT_GU /
// _DN), kol_pf_dequant's arithmetic statement for statement (the same xor, sign-extension shift and single RNE of
// (q - 8) x scale); an expert with no row this chunk (hdr[H_COUNT + b] == 0) returns at once - no tile reads it.
// Block PF_E: the shared expert - int4 layout-1 at w_sh (one block: q4_pf_dequant_gu / _dn) or bf16 gemv_bf16
// tiles copied (w_sh[((n/16) K/8 + k/8) 128 + (k%8) 16 + n%16] = W[n][k]: the _shb entries).
#define Q4_PF_DQ_INT4(TILE)                                                                   \
  const uint8 bw = intel_sub_group_block_read8(TILE);                                         \
  uint wv[8];                                                                                 \
  wv[0] = bw.s0; wv[1] = bw.s1; wv[2] = bw.s2; wv[3] = bw.s3;                                 \
  wv[4] = bw.s4; wv[5] = bw.s5; wv[6] = bw.s6; wv[7] = bw.s7;                                 \
  const float scale = vload_half(lane, (const __global half*)((TILE) + 128));                 \
  for (int j = 0; j < 8; ++j) {                                                               \
    const uint u = wv[j] ^ 0x88888888u;                                                       \
    for (int i = 0; i < 8; ++i) {                                                             \
      const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;                                       \
      const uint k = g * 64u + (uint)j * 8u + (uint)i;                                        \
      o[(size_t)k * nn] = rne_bf16((float)qm8 * scale);                                       \
    }                                                                                         \
  }
#define Q4_PF_DEQUANT(NAME, KK, NN, ADDR, SHB)                                                \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                             \
  __attribute__((intel_reqd_sub_group_size(16)))                                              \
  __kernel void NAME(__global const uint* restrict w, __global const void* restrict w_sh,     \
                     __global const uint* restrict hdr, __global ushort* restrict out,        \
                     uint b0, uint b1) {                                                      \
    const uint lane = get_local_id(0);                                                        \
    const uint n_tile = get_group_id(0);                                                      \
    const uint g = get_group_id(1);                                                           \
    const uint bl = get_group_id(2);                                                          \
    const uint b = b0 + bl;                                                                   \
    const size_t nn = (NN);                                                                   \
    if (b >= b1 || b > PF_E) return;                                                          \
    __global ushort* o = out + (size_t)bl * (KK) * (NN) + n_tile * 16 + lane;                 \
    if (b == PF_E) {                                                                          \
      if (SHB) {                                                                              \
        const __global ushort* sh = (const __global ushort*)w_sh +                            \
                                    ((size_t)n_tile * ((KK) / 8) + g * 8u) * 128 + lane;      \
        for (uint k = 0; k < 64; ++k) o[(size_t)(g * 64u + k) * nn] = sh[(k / 8) * 128 + (k % 8) * 16]; \
      } else {                                                                                \
        const __global uint* tile = (const __global uint*)w_sh +                              \
                                    ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32;            \
        Q4_PF_DQ_INT4(tile)                                                                   \
      }                                                                                       \
      return;                                                                                 \
    }                                                                                         \
    if (hdr[H_COUNT + b] == 0u) return;                                                       \
    __global const uint* tile = ADDR(w, b) + ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32;   \
    Q4_PF_DQ_INT4(tile)                                                                       \
  }

// gate||up blocks are K = PF_D, N = 2 x PF_INTER (interleave16, what pf_moe_gemm's SiLU epilogue reads); down
// blocks K = PF_INTER, N = PF_D.
Q4_PF_DEQUANT(q4_pf_dequant_gu, PF_D, 2 * PF_INTER, Q4_EXPERT_GU, 0)
Q4_PF_DEQUANT(q4_pf_dequant_dn, PF_INTER, PF_D, Q4_EXPERT_DN, 0)
Q4_PF_DEQUANT(q4_pf_dequant_gu_shb, PF_D, 2 * PF_INTER, Q4_EXPERT_GU, 1)
Q4_PF_DEQUANT(q4_pf_dequant_dn_shb, PF_INTER, PF_D, Q4_EXPERT_DN, 1)
#undef Q4_PF_DEQUANT
#undef Q4_PF_DQ_INT4

// ---------------------------------------------------------------------------------------
// q4_pf_moe_combine - token t, hidden column n = 256 x grid.x + lane: q4_moe_down's epilogue over the down GEMM's
// sorted rows. y holds rne(down) per row (pf_moe_gemm's plain epilogue rounds once, as q4_moe_down's rf(red)), so:
//
//   sum = 0;  for k = 0 .. PF_K-1 (the route row's slots: RANK order):
//             sum = sum + rf(f32(y[pair_row[t][k]][n]) x w_k)        the product rounded to bf16, then the fp32 add
//   out[t][n] = rne(f32(rne(sum)) + f32(rne(f32(y[shared row + t][n]) x sg)))
//
// - decode's chain with the down sums read instead of computed; NOT folded into H.
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void q4_pf_moe_combine(__global const uint* restrict route, __global const uint* restrict hdr,
                                __global const uint* restrict pair_row, __global const ushort* restrict y,
                                __global ushort* restrict out, uint C) {
#pragma OPENCL FP_CONTRACT OFF
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float sum = 0.0f;
  for (uint k = 0; k < PF_K; ++k) {
    const float tk = rf(bf16f(y[(size_t)pair_row[t * PF_K + k] * PF_D + n]) * as_float(rr[R_W + k]));
    sum = sum + tk;
  }
  const ushort r_b = rne_bf16(sum);
  const ushort sh_b = rne_bf16(bf16f(y[(size_t)(hdr[H_SHARED_ROW] + t) * PF_D + n]) * as_float(rr[R_SG]));
  out[(size_t)t * PF_D + n] = rne_bf16(bf16f(r_b) + bf16f(sh_b));
}
