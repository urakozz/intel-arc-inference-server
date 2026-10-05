// pf_moe.cl - spec 15d: the mixture-of-experts block of a PREFILL chunk, everything but
// the grouped GEMMs (those are pf_moe_gemm.cl). Spec 15 §4.4 and §9 ("prefill routing
// entirely on the device, with a tile table padded with -1 so the host never reads
// per-expert counts"). One binary per shape (`pf_moe_E<experts>_T<top_k>_D<hidden>_I<inter>`,
// kernels::pf_moe_variant), five entry points:
//
//   pf_moe_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)   grid (1),        WG EXPERTS
//   pf_moe_gather(x, hdr, row_tok, xg)                          grid (rows),     WG 64
//   pf_moe_gather_i8(xq, xs, hdr, row_tok, xg, xsg)             grid (rows),     WG 64
//   pf_moe_dequant_gu / _dn(w, hdr, out, b0)                    grid (N/16, K/64, nb), WG 16
//   pf_moe_combine(route, hdr, pair_row, y, resid, C)           grid (HIDDEN/256, C), WG 256
//
// Before them the chunk's walk runs the router || shared-gate GEMV over all C rows
// (pf_gemv_bf16.cl at N = router_n, the {16, 16} tiling decode's gemv_bf16 binds, so row
// m is bit-identical to decode's GEMV of the same x) and DECODE's moe_route (moe.cl,
// grid (1, C): it reads logits row m and writes route row m, nothing else), so prefill
// and decode route by the same kernel, op for op (spec 15c's formula, transformers 5.18.0).
//
// **The sorted layout.** Every (token, slot) pair of the chunk gets one ROW of the sorted
// space, expert-major, ascending token index within an expert (the order the scan below
// visits them - deterministic, no atomics). Each expert's rows are padded up to a whole
// number of TM-row TILES, so a tile belongs to exactly one expert and no tile's rows are
// another expert's; the shared expert (weight block EXPERTS) follows the routed experts,
// its rows the C tokens in order (row Rs + t is token t): a dense M = C GEMM inside the
// same grouped launch. The tile table lists (block, first row) per tile and is padded to
// a fixed `tmax` with block = NONE, so the GEMMs' grids are a function of C alone:
//
//   tmax(C) = floor((C x TOP_K + EXPERTS x (TM - 1)) / TM) + ceil(C / TM)
//
// (Σ_e ceil(c_e / TM) <= (Σ c_e + EXPERTS (TM - 1)) / TM with Σ c_e = C x TOP_K; the host's
// runtime::moe_prefill_tiles is the one home of the formula and passes it here.)
//
// The header (u32, kernels::pf_moe), written by pf_moe_sort, read by everything after it:
//   [H_TILES]       tiles used (<= tmax)          [H_SHARED_ROW]  Rs, the shared expert's row 0
//   [H_ROWS]        rows used = TM x tiles        [H_C]           C
//   [H_COUNT + e]   rows of expert e (unpadded), e < EXPERTS; [H_COUNT + EXPERTS] = C
//
// **Determinism.** No atomic anywhere. The sort is one work-group: lane e counts and then
// scatters expert e's pairs by walking the chunk in ascending (token, slot) order, so
// every row's position is a function of the route rows alone. Gather and dequant are
// copies. The combine sums each token's TOP_K terms in FIXED slot order - moe_down's chain
// (moe.cl), rounding for rounding - then the shared expert by its gate, then folds into
// the residual, exactly as decode's moe_down does, so prefill and decode combine by one
// rule (tests/kernels/pf_moe_ref.h repeats it and checks it against moe_ref::combine).
//
// **Portability.** No sub-group function but the block reads of pf_moe_dequant_*, which
// tools/mac/opencl/intel_shim.h emulates exactly; scales go through vload_half, so no
// cl_khr_fp16. Every entry point therefore also runs on the Mac's OpenCL 1.2 GPU for an
// indicative check (tools/mac/clrun/pf_moe_run.cc).
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#if !defined(EXPERTS) || !defined(TOP_K) || !defined(HIDDEN) || !defined(INTER) || !defined(TM) || !defined(KC)
#error "pf_moe: EXPERTS, TOP_K, HIDDEN, INTER, TM and KC must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif
#if TOP_K < 1 || TOP_K > 8
#error "pf_moe: TOP_K is 1..8 (moe.cl's route row holds 8 slots)"
#endif
#if EXPERTS < 16 || EXPERTS > 256 || EXPERTS % 16 != 0
#error "pf_moe: EXPERTS must be a multiple of 16 up to 256 (one lane each; ids fit a uchar)"
#endif
#if HIDDEN % 256 != 0 || INTER % 64 != 0 || HIDDEN % 64 != 0
#error "pf_moe: HIDDEN must be whole 256-column combine groups, HIDDEN and INTER whole k-groups"
#endif

#define NONE 0xFFFFFFFFu
#define SHARED EXPERTS          /* the shared expert's weight block (MoeDesc::shared_block) */
#define RW 32                   /* moe.cl's route row words (kernels::moe_route::kWords) */
#define R_IDS 0
#define R_W 8
#define R_SG 16
#define H_TILES 0
#define H_SHARED_ROW 1
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define WG_GATHER 64
#define WG_COMBINE 256
#define TILE_U32 136            /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define GU_K HIDDEN             /* gate||up block: K = hidden, N = 2 x inter */
#define GU_N (2 * INTER)
#define DN_K INTER              /* down block: K = inter, N = hidden */
#define DN_N HIDDEN
#define GU_BLK_U32 ((size_t)(GU_N / 16) * (GU_K / 64) * TILE_U32)
#define DN_BLK_U32 ((size_t)(DN_N / 16) * (DN_K / 64) * TILE_U32)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even (moe.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// ---------------------------------------------------------------------------
// pf_moe_sort - one work-group of EXPERTS lanes; lane e owns expert e.
//
//   1. stage the chunk's expert ids into SLM as bytes (min(id, EXPERTS - 1): moe_route
//      never writes a larger one, and a clamped id keeps every pair inside the table);
//   2. lane e counts its pairs: c_e = #{(t, k) : id[t][k] == e};
//   3. lane 0's serial prefix over e of ceil(c_e / TM) - expert e's first tile - and the
//      shared expert's tiles after the last routed one;
//   4. lane e writes its tiles, then walks (t, k) ascending and gives each pair the next
//      row: row_tok[row] = t, pair_row[t][k] = row; its padding rows get NONE;
//   5. every lane: a share of the shared expert's tiles and rows, of the NONE tail of the
//      tile table, and its count into the header.
//
// No pair is lost: the counts and the scatter read the same SLM bytes, and a token's
// slots name distinct experts (moe_route's ranks are a permutation), so expert e's rows
// [r0, r0 + c_e) are exactly its pairs. tests/kernels/pf_moe_ref.h is the same walk.
__attribute__((reqd_work_group_size(EXPERTS, 1, 1)))
__kernel void pf_moe_sort(__global const uint* restrict route, __global uint* restrict hdr,
                          __global uint* restrict tiles, __global uint* restrict row_tok,
                          __global uint* restrict pair_row, uint C, uint tmax) {
  __local uchar sid[KC * TOP_K];
  __local uint cnts[EXPERTS];
  __local uint toff[EXPERTS + 1];
  const uint e = get_local_id(0);
  const uint n = min(C, (uint)KC);       // the host bounds C by kC; never past the SLM
  for (uint i = e; i < n * TOP_K; i += EXPERTS) {
    const uint id = route[(size_t)(i / TOP_K) * RW + R_IDS + i % TOP_K];
    sid[i] = (uchar)min(id, (uint)(EXPERTS - 1));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cnt = 0;
  for (uint i = 0; i < n * TOP_K; ++i) cnt += (uint)sid[i] == e ? 1u : 0u;
  cnts[e] = cnt;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    uint acc = 0;
    for (uint j = 0; j < EXPERTS; ++j) {
      toff[j] = acc;
      acc += (cnts[j] + TM - 1) / TM;
    }
    toff[EXPERTS] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint ts = toff[EXPERTS];                 // the shared expert's first tile
  const uint st = (n + TM - 1) / TM;             // its tiles
  const uint ntiles = ts + st;
  const uint rs = ts * TM;                       // its first row
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
    for (uint k = 0; k < TOP_K; ++k)
      if ((uint)sid[t * TOP_K + k] == e) {
        row_tok[r0 + p] = t;
        pair_row[t * TOP_K + k] = r0 + p;
        ++p;
      }
  for (; p < nt * TM; ++p) row_tok[r0 + p] = NONE;
  for (uint i = e; i < st; i += EXPERTS)
    if (ts + i < tmax) {
      tiles[2 * (ts + i)] = SHARED;
      tiles[2 * (ts + i) + 1] = rs + i * TM;
    }
  for (uint i = e; i < st * TM; i += EXPERTS) row_tok[rs + i] = i < n ? i : NONE;
  for (uint i = ntiles + e; i < tmax; i += EXPERTS) {
    tiles[2 * i] = NONE;
    tiles[2 * i + 1] = 0;
  }
  hdr[H_COUNT + e] = cnt;
  if (e == 0) {
    hdr[H_TILES] = ntiles;
    hdr[H_SHARED_ROW] = rs;
    hdr[H_ROWS] = ntiles * TM;
    hdr[H_C] = n;
    hdr[H_COUNT + EXPERTS] = n;
  }
}

// ---------------------------------------------------------------------------
// pf_moe_gather / pf_moe_gather_i8 - the GEMMs' A operand in sorted order: row r of xg is
// row row_tok[r] of the chunk's normed activations (bf16 [C][HIDDEN]), or of its h8
// quantisation (int8 [C][HIDDEN] + one fp32 scale per row, pf_int8.cl's pf_quant_had);
// a padding row (NONE) is written as zeros. Rows past hdr[H_ROWS] belong to no tile and
// are not touched. Work-group = one row, 16-byte copies.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void pf_moe_gather(__global const uint4* restrict x, __global const uint* restrict hdr,
                            __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = HIDDEN * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : x[(size_t)t * W + i];
}

__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void pf_moe_gather_i8(__global const uint4* restrict xq, __global const float* restrict xs,
                               __global const uint* restrict hdr,
                               __global const uint* restrict row_tok, __global uint4* restrict xg,
                               __global float* restrict xsg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = HIDDEN / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : xq[(size_t)t * W + i];
  if (get_local_id(0) == 0) xsg[r] = t == NONE ? 0.0f : xs[t];
}

// ---------------------------------------------------------------------------
// pf_moe_dequant_gu / pf_moe_dequant_dn - the bf16 B operand of pf_moe_gemm.cl's bf16 form:
// weight blocks [b0, b0 + grid.z) of one layer (loader/moe_layout.h: layout-1 int4 g64,
// block b at b x stride) into out[b - b0] = bf16 [K][N] row-major. pf_dequant_slab.cl's
// arithmetic, statement for statement (the same xor, sign-extension shift and single RNE),
// with the scale read by vload_half (the same f16 -> f32 conversion as as_half). A routed
// expert with no row this chunk (hdr[H_COUNT + b] == 0) is skipped: no tile reads it.
#define PF_MOE_DEQUANT(NAME, KK, NN, BLK)                                                     \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                             \
  __attribute__((intel_reqd_sub_group_size(16)))                                              \
  __kernel void NAME(__global const uint* restrict w, __global const uint* restrict hdr,      \
                     __global ushort* restrict out, uint b0) {                                \
    const uint lane = get_local_id(0);                                                        \
    const uint n_tile = get_group_id(0);                                                      \
    const uint g = get_group_id(1);                                                           \
    const uint bl = get_group_id(2);                                                          \
    const uint b = b0 + bl;                                                                   \
    if (b < EXPERTS && hdr[H_COUNT + b] == 0u) return;                                        \
    __global const uint* tile = w + (size_t)b * BLK + ((size_t)n_tile * (KK / 64) + g) * TILE_U32; \
    const uint8 blk = intel_sub_group_block_read8(tile);                                      \
    uint wv[8];                                                                               \
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;                           \
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;                           \
    const float scale = vload_half(lane, (const __global half*)(tile + 128));                 \
    __global ushort* o = out + (size_t)bl * KK * NN + n_tile * 16 + lane;                     \
    for (int j = 0; j < 8; ++j) {                                                             \
      const uint u = wv[j] ^ 0x88888888u;                                                     \
      for (int i = 0; i < 8; ++i) {                                                           \
        const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;                                     \
        const uint k = g * 64u + (uint)j * 8u + (uint)i;                                      \
        o[(size_t)k * NN] = rne_bf16((float)qm8 * scale);                                     \
      }                                                                                       \
    }                                                                                         \
  }
PF_MOE_DEQUANT(pf_moe_dequant_gu, GU_K, GU_N, GU_BLK_U32)
PF_MOE_DEQUANT(pf_moe_dequant_dn, DN_K, DN_N, DN_BLK_U32)
#undef PF_MOE_DEQUANT

// ---------------------------------------------------------------------------
// pf_moe_combine - token t, hidden column n = 256 x grid.x + lane: moe_down's epilogue
// (moe.cl) over the down GEMM's rows. y holds rne(down) per sorted row (pf_moe_gemm's
// plain epilogue rounds once, as every consumer of a linear does first), so:
//
//   for k = 0 .. TOP_K-1 (ascending):  t_k = rne(f32(y[pair_row[t][k]][n]) x w_k);  sum += f32(t_k)
//   r_b  = rne(sum)
//   sh_b = rne(f32(y[Rs + t][n]) x s_b)
//   resid[t][n] = rne(f32(resid[t][n]) + f32(rne(f32(r_b) + f32(sh_b))))
//
// w_k and s_b are the route row's (moe_route's bf16 values held in fp32), so this is
// moe_down's chain with the down sums read instead of computed - the residual fold
// included, which is why the next norm folds nothing (ModelDesc::ffn_fold_s() == 0).
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void pf_moe_combine(__global const uint* restrict route, __global const uint* restrict hdr,
                             __global const uint* restrict pair_row,
                             __global const ushort* restrict y, __global ushort* restrict resid,
                             uint C) {
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float sum = 0.0f;
  for (uint k = 0; k < TOP_K; ++k) {
    const uint row = pair_row[t * TOP_K + k];
    const ushort d_b = y[(size_t)row * HIDDEN + n];
    sum += bf16f(rne_bf16(bf16f(d_b) * as_float(rr[R_W + k])));
  }
  const ushort r_b = rne_bf16(sum);
  const ushort ds_b = y[(size_t)(hdr[H_SHARED_ROW] + t) * HIDDEN + n];
  const ushort sh_b = rne_bf16(bf16f(ds_b) * as_float(rr[R_SG]));
  const ushort o_b = rne_bf16(bf16f(r_b) + bf16f(sh_b));
  __global ushort* restrict rp = resid + (size_t)t * HIDDEN + n;
  *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
}
