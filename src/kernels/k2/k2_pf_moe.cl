// k2_pf_moe.cl - spec 18c: K2-Horizon's routed experts over a PREFILL chunk - MoVA's value
// experts (64, top-4, no shared expert) and the MoE block (100, top-8, + 1 shared) -
// everything but the grouped GEMMs, which are spec 15d's pf_moe_gemm.cl at K2's shapes,
// unchanged (its tile table, its padding tiles, its row-independence argument). One binary
// per router (kernels::k2::pf_moe_variant / pf_mova_variant), entry points by family:
//
//   always      k2_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)   grid (1), WG PF_WG
//               k2_pf_gather(x, hdr, row_tok, xg)                         grid (rows), WG 64
//   PF_INTER    k2_pf_dequant_gu / _dn(w, hdr, out, b0)       grid (N/16, K/64, nb), WG 16
//   (MoE)       k2_pf_moe_combine(route, hdr, pair_row, y, resid, C)  grid (PF_D/256, C), WG 256
//   PF_VN       k2_pf_dequant_v(w, hdr, out, b0)                 grid (VN/16, D/64, nb), WG 16
//   (MoVA)      k2_pf_mova_combine(route, pair_row, y, kv_v, pos, C)  grid (VN/256, C), WG 256
//
// Why not pf_moe.cl itself (spec 15d's, Ornith's): it gives one sort lane per expert with
// EXPERTS a multiple of 16 (K2's MoE has 100), always appends the shared expert's tiles
// (MoVA has none) and combines by Ornith's chain (fp32 sum in rank order, a GATED shared
// expert); K2's reference chain is the ascending-id bf16 index_add_ (spec 18 §3, k2_moe.cl's
// header) with the shared expert ungated. So this is pf_moe.cl's design carried to K2 - the
// sorted layout, the header, the tmax bound and the determinism argument are its, restated
// below - and pf_moe.cl and its binaries are untouched.
//
// **Routing is decode's kernel.** Before these, the chunk's walk runs k2_moe.cl's k2_route
// over all C rows (grid (1, C); the route row m of logits row m, nothing else - LS = 1 makes
// the binary's M irrelevant): the MoE router's logits come from pf_gemv_bf16.cl at decode's
// {16, 16} gemv_bf16 tiling (row m bitwise decode's GEMV of the same x), MoVA's from the
// prefill attention GEMM's v_router columns. Same formula, same ties (lower id), same
// ascending slot order as decode, op for op.
//
// **The sorted layout** (pf_moe.cl's). Every (token, slot) pair gets one ROW, expert-major,
// ascending token within an expert; each expert's rows are padded to whole TM-row TILES so a
// tile is one expert's; with PF_SHARED the shared expert (block PF_E) follows, its rows the C
// tokens in order. The tile table (block, first row) is padded to a fixed `tmax` with block
// NONE, so every grid is a function of C alone and the host reads no count:
//
//   tmax(C) = floor((C x PF_K + PF_E x (TM - 1)) / TM) + (PF_SHARED ? ceil(C / TM) : 0)
//
// (runtime::k2::pf_tiles is the host's one home of it.) The header (u32, kernels::pf_moe's
// word indices): [0] tiles used, [1] the shared expert's first row (NONE without one), [2]
// rows used = TM x tiles, [3] C, [4 + e] expert e's rows (unpadded), [4 + PF_E] the shared
// expert's (C, or 0).
//
// **Determinism.** No atomic anywhere. The sort is one work-group: lane e counts and then
// scatters expert e's pairs walking the chunk in ascending (token, slot) order, so every
// row's place is a function of the route rows alone. Gather and dequant are copies. The
// combines sum each token's PF_K terms in its route row's slot order - ASCENDING EXPERT ID -
// rounding to bf16 after every add, exactly as decode's k2_moe_down / k2_mova_value do
// (tests/kernels/k2_pf_ref.h repeats both and checks them against k2_ref.h's decode chains).
//
// **Portability.** No sub-group function but the block reads of the dequants, which
// tools/mac/opencl/intel_shim.h emulates exactly; scales through vload_half (no
// cl_khr_fp16). Every entry point runs on the Mac's OpenCL 1.2 GPU for an indicative check
// (tools/mac/clrun/k2_run.cc).
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#if !defined(PF_E) || !defined(PF_K) || !defined(PF_WG) || !defined(PF_SHARED) || !defined(PF_D) || !defined(TM) || !defined(KC)
#error "k2_pf_moe: PF_E, PF_K, PF_WG, PF_SHARED, PF_D, TM and KC must be defined (src/kernels/CMakeLists.txt, spec 18c)"
#endif
#if PF_K < 1 || PF_K > 8 || PF_E <= PF_K || PF_E > 255
#error "k2_pf_moe: 1 <= PF_K <= 8 (the route row), PF_K < PF_E <= 255 (ids staged as bytes)"
#endif
#if (PF_WG & (PF_WG - 1)) != 0 || PF_WG < 16 || PF_WG > 256 || PF_WG < PF_E
#error "k2_pf_moe: PF_WG is a power of two in [16, 256] holding one lane per expert"
#endif
#if PF_D % 256 != 0 || PF_D % 64 != 0
#error "k2_pf_moe: PF_D (hidden) must be whole 256-column combine groups and 64-k groups"
#endif

#define NONE 0xFFFFFFFFu
#define RW 32                   /* the route row (k2_moe.cl's RW, kernels::k2::route) */
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

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (k2_moe.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// ---------------------------------------------------------------------------------------
// k2_pf_sort - one work-group of PF_WG lanes; lane e < PF_E owns expert e (pf_moe_sort's
// five steps; the lanes >= PF_E only share the shared expert's and the NONE tail's writes):
//
//   1. stage the chunk's ids into SLM as bytes (min(id, PF_E - 1): k2_route never writes a
//      larger one, and a clamped id keeps every pair inside the table);
//   2. lane e counts its pairs c_e = #{(t, k) : id[t][k] == e};
//   3. lane 0's serial prefix over e of ceil(c_e / TM) - expert e's first tile;
//   4. lane e writes its tiles, then walks (t, k) ascending and gives each pair the next row:
//      row_tok[row] = t, pair_row[t][k] = row; its padding rows get NONE;
//   5. every lane: a share of the shared expert's tiles and rows (PF_SHARED), of the NONE
//      tail of the tile table; the counts into the header.
//
// No pair is lost: the counts and the scatter read the same SLM bytes, and a token's slots
// name distinct experts (k2_route's ranks are a permutation), so expert e's rows
// [r0, r0 + c_e) are exactly its pairs. tests/kernels/k2_pf_ref.h's sort() is this walk.
__attribute__((reqd_work_group_size(PF_WG, 1, 1)))
__kernel void k2_pf_sort(__global const uint* restrict route, __global uint* restrict hdr,
                         __global uint* restrict tiles, __global uint* restrict row_tok,
                         __global uint* restrict pair_row, uint C, uint tmax) {
  __local uchar sid[KC * PF_K];
  __local uint cnts[PF_WG];
  __local uint toff[PF_E + 1];
  const uint e = get_local_id(0);
  const uint n = min(C, (uint)KC);       // the host bounds C by kC; never past the SLM
  for (uint i = e; i < n * PF_K; i += PF_WG) {
    const uint id = route[(size_t)(i / PF_K) * RW + R_IDS + i % PF_K];
    sid[i] = (uchar)min(id, (uint)(PF_E - 1));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cnt = 0;
  if (e < PF_E)
    for (uint i = 0; i < n * PF_K; ++i) cnt += (uint)sid[i] == e ? 1u : 0u;
  cnts[e] = cnt;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    uint acc = 0;
    for (uint j = 0; j < PF_E; ++j) {
      toff[j] = acc;
      acc += (cnts[j] + TM - 1) / TM;
    }
    toff[PF_E] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint ts = toff[PF_E];                    // the shared expert's first tile
#if PF_SHARED
  const uint st = (n + TM - 1) / TM;             // its tiles
#else
  const uint st = 0;
#endif
  const uint ntiles = ts + st;
  const uint rs = ts * TM;                       // its first row
  if (e < PF_E) {
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
    hdr[H_COUNT + e] = cnt;
  }
#if PF_SHARED
  for (uint i = e; i < st; i += PF_WG)
    if (ts + i < tmax) {
      tiles[2 * (ts + i)] = PF_E;                // the shared expert's block (K2Desc::shared_block)
      tiles[2 * (ts + i) + 1] = rs + i * TM;
    }
  for (uint i = e; i < st * TM; i += PF_WG) row_tok[rs + i] = i < n ? i : NONE;
#endif
  for (uint i = ntiles + e; i < tmax; i += PF_WG) {
    tiles[2 * i] = NONE;
    tiles[2 * i + 1] = 0;
  }
  if (e == 0) {
    hdr[H_TILES] = ntiles;
    hdr[H_SHARED_ROW] = PF_SHARED ? rs : NONE;
    hdr[H_ROWS] = ntiles * TM;
    hdr[H_C] = n;
    hdr[H_COUNT + PF_E] = PF_SHARED ? n : 0u;
  }
}

// ---------------------------------------------------------------------------------------
// k2_pf_gather - the grouped GEMM's A operand in sorted order: row r of xg is row
// row_tok[r] of the chunk's normed activations (bf16 [C][PF_D]); a padding row (NONE) is
// zeros. Rows past hdr[H_ROWS] belong to no tile and are not touched. pf_moe_gather's text.
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void k2_pf_gather(__global const uint4* restrict x, __global const uint* restrict hdr,
                           __global const uint* restrict row_tok, __global uint4* restrict xg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = PF_D * 2 / 16;
  for (uint i = get_local_id(0); i < W; i += WG_GATHER)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : x[(size_t)t * W + i];
}

// ---------------------------------------------------------------------------------------
// The bf16 B operand of pf_moe_gemm.cl's bf16 form: weight blocks [b0, b0 + grid.z) of one
// layer's flat layout-1 array (loader/k2_layout.h: block b at b x its stride) into
// out[b - b0] = bf16 [K][N] row-major. pf_moe.cl's PF_MOE_DEQUANT statement for statement
// (the same xor, sign-extension shift and single RNE; the scale by vload_half). A routed
// expert with no row this chunk (hdr[H_COUNT + b] == 0) is skipped: no tile reads it.
#define K2_PF_DEQUANT(NAME, KK, NN)                                                           \
  __attribute__((reqd_work_group_size(16, 1, 1)))                                             \
  __attribute__((intel_reqd_sub_group_size(16)))                                              \
  __kernel void NAME(__global const uint* restrict w, __global const uint* restrict hdr,      \
                     __global ushort* restrict out, uint b0) {                                \
    const uint lane = get_local_id(0);                                                        \
    const uint n_tile = get_group_id(0);                                                      \
    const uint g = get_group_id(1);                                                           \
    const uint bl = get_group_id(2);                                                          \
    const uint b = b0 + bl;                                                                   \
    if (b < PF_E && hdr[H_COUNT + b] == 0u) return;                                           \
    const size_t blk = (size_t)((NN) / 16) * ((KK) / 64) * TILE_U32;                          \
    __global const uint* tile = w + (size_t)b * blk + ((size_t)n_tile * ((KK) / 64) + g) * TILE_U32; \
    const uint8 bw = intel_sub_group_block_read8(tile);                                       \
    uint wv[8];                                                                               \
    wv[0] = bw.s0; wv[1] = bw.s1; wv[2] = bw.s2; wv[3] = bw.s3;                               \
    wv[4] = bw.s4; wv[5] = bw.s5; wv[6] = bw.s6; wv[7] = bw.s7;                               \
    const float scale = vload_half(lane, (const __global half*)(tile + 128));                 \
    __global ushort* o = out + (size_t)bl * (KK) * (NN) + n_tile * 16 + lane;                 \
    for (int j = 0; j < 8; ++j) {                                                             \
      const uint u = wv[j] ^ 0x88888888u;                                                     \
      for (int i = 0; i < 8; ++i) {                                                           \
        const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;                                     \
        const uint k = g * 64u + (uint)j * 8u + (uint)i;                                      \
        o[(size_t)k * (NN)] = rne_bf16((float)qm8 * scale);                                   \
      }                                                                                       \
    }                                                                                         \
  }

// ---------------------------------------------------------------------------------------
// The MoE block (PF_INTER): gate||up blocks are K = PF_D, N = 2 x PF_INTER (interleave16,
// what pf_moe_gemm's SiLU epilogue reads), down blocks K = PF_INTER, N = PF_D; the shared
// expert is block PF_E of both.
#ifdef PF_INTER
#if !PF_SHARED
#error "k2_pf_moe: the MoE block carries its shared expert (PF_SHARED 1)"
#endif
#if PF_INTER % 64 != 0
#error "k2_pf_moe: PF_INTER must be whole 64-k groups"
#endif
K2_PF_DEQUANT(k2_pf_dequant_gu, PF_D, 2 * PF_INTER)
K2_PF_DEQUANT(k2_pf_dequant_dn, PF_INTER, PF_D)

// k2_pf_moe_combine - token t, hidden column n = 256 x grid.x + lane: k2_moe_down's
// epilogue (k2_moe.cl) over the down GEMM's sorted rows. y holds rne(down) per row
// (pf_moe_gemm's plain epilogue rounds once, as k2_moe_down's d_b), so:
//
//   acc = 0;  for j = 0 .. PF_K-1 (the route row's slots: ASCENDING expert id):
//             acc = f32(rne(acc + f32(rne(f32(y[pair_row[t][j]][n]) x w_j))))
//   o_b = rne(acc + f32(y[Rs + t][n]))                     the UNGATED shared expert
//   resid[t][n] = rne(f32(resid[t][n]) + f32(o_b))         the decoder's residual add
//
// - decode's chain with the down sums read instead of computed, the residual fold included
// (so the next layer's norm folds nothing, S_PREV 0, as on decode).
__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void k2_pf_moe_combine(__global const uint* restrict route, __global const uint* restrict hdr,
                                __global const uint* restrict pair_row,
                                __global const ushort* restrict y, __global ushort* restrict resid,
                                uint C) {
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float acc = 0.0f;
  for (uint j = 0; j < PF_K; ++j) {
    const ushort d_b = y[(size_t)pair_row[t * PF_K + j] * PF_D + n];
    const ushort t_b = rne_bf16(bf16f(d_b) * as_float(rr[R_W + j]));
    acc = bf16f(rne_bf16(acc + bf16f(t_b)));
  }
  const ushort sh_b = y[(size_t)(hdr[H_SHARED_ROW] + t) * PF_D + n];
  const ushort o_b = rne_bf16(acc + bf16f(sh_b));
  __global ushort* restrict rp = resid + (size_t)t * PF_D + n;
  *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
}
#endif  // PF_INTER

// ---------------------------------------------------------------------------------------
// MoVA's value experts (PF_VN = kv heads x head dim): blocks K = PF_D, N = PF_VN, no shared
// expert. pf_moe_gemm's plain epilogue leaves v_e = rne(Σ V_e x) per sorted row; the combine
// is k2_mova_value's epilogue (k2_moe.cl), written into this layer's V cache at pos + t:
//
//   acc = 0;  for j ascending id:  a = rne(silu(f32(y[pair_row[t][j]][n])))
//                                  acc = f32(rne(acc + f32(rne(f32(a) x w_j))))
//   kv_v[pos + t][n] = rne(acc)                  (the KV row layout [max_len][kv heads][hd])
//
// This v is what the cache holds and every later attention reads (spec 18 §3: MoVA's routed
// mix IS the value) - the same rule as decode's (Review Focus 1).
#ifdef PF_VN
#if PF_SHARED
#error "k2_pf_moe: MoVA has no shared expert (PF_SHARED 0)"
#endif
#if PF_VN % 256 != 0
#error "k2_pf_moe: PF_VN must be whole 256-column combine groups"
#endif
K2_PF_DEQUANT(k2_pf_dequant_v, PF_D, PF_VN)

inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

__attribute__((reqd_work_group_size(WG_COMBINE, 1, 1)))
__kernel void k2_pf_mova_combine(__global const uint* restrict route,
                                 __global const uint* restrict pair_row,
                                 __global const ushort* restrict y, __global ushort* restrict kv_v,
                                 uint pos, uint C) {
  const uint n = get_group_id(0) * WG_COMBINE + get_local_id(0);
  const uint t = get_group_id(1);
  if (t >= C) return;
  __global const uint* restrict rr = route + (size_t)t * RW;
  float acc = 0.0f;
  for (uint j = 0; j < PF_K; ++j) {
    const ushort v_b = y[(size_t)pair_row[t * PF_K + j] * PF_VN + n];
    const ushort a_b = rne_bf16(silu_f32(bf16f(v_b)));
    const ushort t_b = rne_bf16(bf16f(a_b) * as_float(rr[R_W + j]));
    acc = bf16f(rne_bf16(acc + bf16f(t_b)));
  }
  kv_v[(size_t)(pos + t) * PF_VN + n] = rne_bf16(acc);
}
#endif  // PF_VN
#undef K2_PF_DEQUANT
