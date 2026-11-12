// pf_moe_sort_mw.cl - plan 2026-11-12 (Ornith MoE prefill) Task 2: the prefill MoE sort as a multi-work-group
// histogram + scan + scatter. It writes the SAME buffers as pf_moe.cl's pf_moe_sort and q4_pf_moe.cl's q4_pf_sort,
// byte for byte (the header, the tile table padded to tmax, row_tok, pair_row - their layout is restated in
// pf_moe.cl), from the same route rows, without the one-work-group serial walk (1.5 ms a layer and chunk at C = 2048)
// and without the SLM staging of the whole chunk (KC), so the chunk is not capped by the sort. One binary per
// (experts, top_k) (`pf_moe_sort_mw_E<experts>_T<top_k>`, kernels::pf_moe_sort_mw_variant), three entry points:
//
//   pf_moe_hist(route, cnt, C)                                   grid (nblk), WG LANES
//   pf_moe_scan(cnt, hdr, tiles, row_tok, C, tmax)                grid (1),    WG LANES
//   pf_moe_scatter(route, cnt, row_tok, pair_row, C)              grid (nblk), WG LANES
//
// nblk = ceil(C / TB) blocks of TB tokens; cnt is u32 [nblk][EXPERTS] scratch. LANES = min(EXPERTS, 256) and lane l
// owns experts l + LANES j, j < EPL (EXPERTS = LANES x EPL).
//
//   1. hist: block b's TB x TOP_K expert ids staged in SLM; lane l counts each of its experts' pairs in the block:
//      cnt[b][e] = #{(t, k) in block b : id[t][k] == e}.
//   2. scan: lane l sums each of its experts' counts c_e; lane 0 takes pf_moe_sort's serial prefix over e of
//      ceil(c_e / TM) (the same sum; r0(e) = TM x it); then lane l replaces cnt[b][e] by expert e's first row of
//      block b - r0(e) plus the pairs of e in the blocks before b - walking b ascending. It writes the tile table (each expert's tiles, the
//      shared expert's after the last routed one, NONE to tmax), the expert tails' padding rows (row_tok NONE), the
//      shared expert's rows (token t at Rs + t) and the header.
//   3. scatter: block b's ids staged again; lane l walks the block's pairs in ascending (token, slot) order and gives
//      each pair of its experts the next row of that expert: row_tok[row] = t, pair_row[t][k] = row.
//
// **Bitwise the one-work-group sort.** Inside an expert, rows run in ascending (token, slot) order: blocks are
// consecutive token ranges taken in ascending b (step 2's prefix), and step 3 walks a block ascending. That is
// exactly pf_moe_sort's walk (lane e scans (t, k) ascending), so every row lands where it did. The tiles, the
// padding rows, the shared rows and the header are the same formulas. No atomic anywhere: every count and row is a
// function of the route rows alone. Ids above EXPERTS - 1 are clamped as pf_moe_sort clamps them (moe_route never
// writes one). tests/kernels/pf_moe_ref.h's blocked walk is this, on the host.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#if !defined(EXPERTS) || !defined(TOP_K) || !defined(TM) || !defined(TB)
#error "pf_moe_sort_mw: EXPERTS, TOP_K, TM and TB must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif
#if EXPERTS < 16 || EXPERTS > 1024 || EXPERTS % 16 != 0
#error "pf_moe_sort_mw: EXPERTS is a multiple of 16 up to 1024 (ids staged as ushort)"
#endif
#if TOP_K < 1 || TOP_K > 15
#error "pf_moe_sort_mw: TOP_K is 1..15 (the route row's id slots)"
#endif
#define LANES (EXPERTS < 256 ? EXPERTS : 256)
#if EXPERTS % LANES != 0
#error "pf_moe_sort_mw: EXPERTS must be a whole number of 256-lane passes"
#endif
#define EPL (EXPERTS / LANES)
#define NONE 0xFFFFFFFFu
#define SHARED EXPERTS          /* the shared expert's weight block */
#define RW 32                   /* the route row's words; ids at word 0 (pf_moe.cl's and q4_pf_moe.cl's R_IDS) */
#define H_TILES 0
#define H_SHARED_ROW 1
#define H_ROWS 2
#define H_C 3
#define H_COUNT 4
#define BP (TB * TOP_K)         /* pairs a block */

// Block b's ids into SLM (clamped); returns the block's tokens.
inline uint stage_ids(__global const uint* restrict route, __local ushort* sid, uint b, uint C) {
  const uint t0 = b * TB;
  const uint nt = min((uint)TB, C - t0);
  for (uint i = get_local_id(0); i < nt * TOP_K; i += LANES) {
    const uint id = route[(size_t)(t0 + i / TOP_K) * RW + i % TOP_K];
    sid[i] = (ushort)min(id, (uint)(EXPERTS - 1));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return nt;
}

__attribute__((reqd_work_group_size(LANES, 1, 1)))
__kernel void pf_moe_hist(__global const uint* restrict route, __global uint* restrict cnt, uint C) {
  __local ushort sid[BP];
  const uint b = get_group_id(0);
  const uint l = get_local_id(0);
  const uint n = stage_ids(route, sid, b, C) * TOP_K;
  uint c[EPL];
  for (uint j = 0; j < EPL; ++j) c[j] = 0;
  for (uint i = 0; i < n; ++i) {
    const uint id = sid[i];
    for (uint j = 0; j < EPL; ++j) c[j] += id == l + LANES * j ? 1u : 0u;
  }
  for (uint j = 0; j < EPL; ++j) cnt[(size_t)b * EXPERTS + l + LANES * j] = c[j];
}

__attribute__((reqd_work_group_size(LANES, 1, 1)))
__kernel void pf_moe_scan(__global uint* restrict cnt, __global uint* restrict hdr, __global uint* restrict tiles,
                          __global uint* restrict row_tok, uint C, uint tmax) {
  __local uint cnts[EXPERTS];
  __local uint toff[EXPERTS + 1];
  const uint l = get_local_id(0);
  const uint nblk = (C + TB - 1) / TB;
  uint tot[EPL];
  for (uint j = 0; j < EPL; ++j) {   // expert e's total (independent loads, so they overlap)
    const uint e = l + LANES * j;
    uint run = 0;
#pragma unroll 8
    for (uint b = 0; b < nblk; ++b) run += cnt[(size_t)b * EXPERTS + e];
    tot[j] = run;
    cnts[e] = run;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) {   // pf_moe_sort's serial prefix of the experts' tiles
    uint acc = 0;
    for (uint e = 0; e < EXPERTS; ++e) {
      toff[e] = acc;
      acc += (cnts[e] + TM - 1) / TM;
    }
    toff[EXPERTS] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint ts = toff[EXPERTS];                 // the shared expert's first tile
  const uint st = (C + TM - 1) / TM;             // its tiles
  const uint ntiles = ts + st;
  const uint rs = ts * TM;                       // its first row
  for (uint j = 0; j < EPL; ++j) {
    const uint e = l + LANES * j;
    const uint t0 = toff[e];
    const uint nt = toff[e + 1] - t0;
    const uint r0 = t0 * TM;
    uint run = r0;                               // block b's first row of e
#pragma unroll 8
    for (uint b = 0; b < nblk; ++b) {
      const uint v = cnt[(size_t)b * EXPERTS + e];
      cnt[(size_t)b * EXPERTS + e] = run;
      run += v;
    }
    for (uint i = 0; i < nt; ++i)
      if (t0 + i < tmax) {
        tiles[2 * (t0 + i)] = e;
        tiles[2 * (t0 + i) + 1] = r0 + i * TM;
      }
    for (uint p = tot[j]; p < nt * TM; ++p) row_tok[r0 + p] = NONE;
    hdr[H_COUNT + e] = tot[j];
  }
  for (uint i = l; i < st; i += LANES)
    if (ts + i < tmax) {
      tiles[2 * (ts + i)] = SHARED;
      tiles[2 * (ts + i) + 1] = rs + i * TM;
    }
  for (uint i = l; i < st * TM; i += LANES) row_tok[rs + i] = i < C ? i : NONE;
  for (uint i = ntiles + l; i < tmax; i += LANES) {
    tiles[2 * i] = NONE;
    tiles[2 * i + 1] = 0;
  }
  if (l == 0) {
    hdr[H_TILES] = ntiles;
    hdr[H_SHARED_ROW] = rs;
    hdr[H_ROWS] = ntiles * TM;
    hdr[H_C] = C;
    hdr[H_COUNT + EXPERTS] = C;
  }
}

__attribute__((reqd_work_group_size(LANES, 1, 1)))
__kernel void pf_moe_scatter(__global const uint* restrict route, __global const uint* restrict cnt,
                             __global uint* restrict row_tok, __global uint* restrict pair_row, uint C) {
  __local ushort sid[BP];
  const uint b = get_group_id(0);
  const uint l = get_local_id(0);
  const uint n = stage_ids(route, sid, b, C) * TOP_K;
  const uint p0 = b * BP;                        // the block's first pair (t0 x TOP_K)
  uint row[EPL];
  for (uint j = 0; j < EPL; ++j) row[j] = cnt[(size_t)b * EXPERTS + l + LANES * j];
  for (uint i = 0; i < n; ++i) {
    const uint id = sid[i];
    for (uint j = 0; j < EPL; ++j)
      if (id == l + LANES * j) {
        row_tok[row[j]] = b * TB + i / TOP_K;
        pair_row[p0 + i] = row[j];
        ++row[j];
      }
  }
}
