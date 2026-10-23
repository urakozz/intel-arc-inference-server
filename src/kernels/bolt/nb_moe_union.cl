// nb_moe_union.cl - spec 26c (Decision 10): Nemotron 3.5 Bolt's routed experts at a verify's M rows with each expert block
// read ONCE for every row that chose it - the union-aware form of nb_moe.cl's nb_moe_up / nb_moe_down, bitwise the plain
// M-row form (the same products in the same order per row; only who reads the weights changes). Three families, each
// compiled only when its defines are given:
//
//   nb_moe_up_U(route, lat, w, a, n_exp)        MOE_I + UNION_UP  grid (M x 9 x I / 64, 1), WG 64 x UP_KS
//   nb_moe_down_U(route, a, w, d, n_exp)        MOE_I + UNION_DN  grid (M x 9 x 1024 / 64, 1), WG 64 x DN_KS
//   nb_moe_comb_U(route, d, r)                  UNION_COMB        grid (1024 / 256, M), WG 256
// (UNION_COUNT: the up and down entries take a last argument `cnt` u32 [2][512] and add one per work-group that read an
// expert's tiles - cnt[x] for up, cnt[512 + x] for down: the test-only build that proves every block is read once.)
//
// **The union schedule** (every work-group derives it alone, in SLM, from the M route rows - no host step, no extra
// launch): the M x 9 (row, slot) pairs in (row, slot) order with their expert ids (nb_route's nine ascending, distinct
// ids a row; clamped to n_exp - 1 as nb_moe.cl does); a pair is its expert's FIRST if no earlier pair has that id; the
// distinct experts are ranked by id (the rank of a first pair = the first pairs with a smaller id). Work-group set u
// (u < M x 9: the worst case, every pair a different expert) takes the expert of rank u - or exits, uniformly, when
// fewer experts were chosen - and every pair of that expert, in (row, slot) order ("stable by row"), at most M of them
// per pass (a row never holds an expert twice on a real route; more pairs, only on a garbage row, take further passes).
// The grid is fixed by M (one captured list per M); the work-groups past the union's size return at once.
//
// **The arithmetic per pair is nb_moe.cl's per (row, slot), verbatim:**
//   up    u = Σ over the k-slices (UP_KS, the same pairwise tree) of tile_dot over the slice's groups (the weight tile and
//         its f16 scale loaded ONCE, then dot8 against each pair's latent row: per pair the same chain as tile_dot's);
//         a[row][slot][i] = rne(relu(rne(u))²)
//   down  the same over a's row (DN_KS slices, the same tree); d[row][slot][n] = rne(Σ) - the bf16 VALUE nb_moe_down's
//         combine reads as rf(red) (rf(x) == bf16f(rne_bf16(x)))
//   comb  acc = 0; for k = 0 .. 8 (ascending id): acc = acc + (f32(d[row][k][n]) · w_k); r[row][n] = rne(acc) - the
//         combine of nb_moe_down, product then add, FP_CONTRACT OFF
// so a, r (and hence everything after) equal the plain M-row binaries' bits (tests/kernels/bolt_moe_union_test.cc on the
// card, tools/mac/clrun/nb_union_run.cc on the Mac at DN_KS 1). The plain form stays the reference and an arm
// (B70_BOLT_VERIFY_MOE=plain); nb_moe.cl is not edited (its 25c binaries keep their bytes).
//
// **The cost** (Decision 10): the plain M-row form reads 9 x M expert blocks a MoE block (7.2 GB at M 8 over the 40
// blocks); this one reads the union's u blocks (spec 26 §4: ~40 x u x 2.45 MB) plus one more launch (the combine) a MoE
// block. Plain OpenCL C with cl_intel_subgroups' block reads where the device has them (nb_moe.cl's portability), so the
// file builds as OpenCL 1.2 for the Mac.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#error "nb_moe_union: M (the verify's rows, 1 .. 8) must be defined"
#endif
#if M < 1 || M > 8
#error "nb_moe_union: M is 1 .. 8 (Control::cur_token's 8 ids)"
#endif
#include "nb_common.h"

#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 48                       /* route row words (kernels::bolt::route::kWords) */
#define R_IDS 0
#define R_W 9
#define RT_K 9                      /* top-9 */
#define NB_LAT 1024                 /* the latent */
#define NPAIR (M * RT_K)            /* the (row, slot) pairs of one verify */

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define NB_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define NB_SG16
#endif

#ifdef UNION_COUNT
#define CNT_ARG , __global volatile uint* restrict cnt
#else
#define CNT_ARG
#endif

#if defined(UNION_UP) || defined(UNION_DN)
// The union schedule (header): leaves in SLM the expert `*sx` of distinct rank u (or 0xFFFFFFFF: none - every lane sees
// the same value after the barrier) and its pairs pl[0 .. *np) in (row, slot) order (pair index row x 9 + slot).
inline void nb_union(__global const uint* restrict route, uint n_exp, uint u, uint lid, uint wg, __local uint* ids,
                     __local uint* first, __local uint* sx, __local uint* pl, __local uint* np) {
  for (uint p = lid; p < NPAIR; p += wg) ids[p] = min(route[(size_t)(p / RT_K) * RW + R_IDS + p % RT_K], n_exp - 1);
  if (lid == 0) *sx = 0xFFFFFFFFu;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lid; p < NPAIR; p += wg) {   // is pair p its expert's first?
    uint f = 1u;
    for (uint q = 0; q < p; ++q) f = ids[q] == ids[p] ? 0u : f;
    first[p] = f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lid; p < NPAIR; p += wg) {   // a first pair's rank among the distinct experts, by id
    if (!first[p]) continue;
    uint rank = 0;
    for (uint q = 0; q < NPAIR; ++q) rank += (first[q] && ids[q] < ids[p]) ? 1u : 0u;
    if (rank == u) *sx = ids[p];   // at most one first pair has rank u
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    uint n = 0;
    const uint x = *sx;
    if (x != 0xFFFFFFFFu)
      for (uint p = 0; p < NPAIR; ++p)
        if (ids[p] == x) pl[n++] = p;
    *np = n;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
#if defined(UNION_UP) || defined(UNION_DN)
#ifndef MOE_I
#error "nb_moe_union: MOE_I (the block's routed intermediate) must be defined"
#endif
#if !defined(UP_KS) || !defined(DN_KS)
#error "nb_moe_union: UP_KS and DN_KS must be defined (nb_moe.cl's: the same k-slices, the same trees)"
#endif
#define G_UP (NB_LAT / GROUP)       /* 16 k-groups of the latent */
#define NT_UP (MOE_I / SG)          /* up's n-tiles */
#define G_DN (MOE_I / GROUP)        /* down's k-groups */
#define NT_DN (NB_LAT / SG)         /* 64 */
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (4 * SG * DN_KS)
#include "nb_expert.h"
#if MOE_I % GROUP != 0 || NT_UP % 4 != 0
#error "nb_moe_union: I is whole k-groups and whole 4-tile blocks"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "nb_moe_union: UP_KS must be a power of two dividing 1024 / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "nb_moe_union: DN_KS must be a power of two dividing I / 64"
#endif

// nb_moe.cl's dot8 / load_tile8, verbatim (gemv.cl's LAYOUT 1 order) - EDIT THE TWO TOGETHER.
inline float dot8(uint word, ushort8 xv) {
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
}
inline void load_tile8(const __global uint* restrict tile, uint lane, uint* wv) {
#ifdef cl_intel_subgroups
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
#else
  for (uint j = 0; j < 8; ++j) wv[j] = tile[j * SG + lane];
#endif
}
// nb_moe.cl's tile_dot over up to M rows at once: the tile and its scale loaded once a group, then per row r < n exactly
// tile_dot's chain (gacc from 0 over the 8 words, acc += gacc · scale). Row r's activations start at x + xo[r].
inline void tile_dot_rows(const __global uint* restrict tile, __global const ushort* restrict x, __private const uint* xo,
                          uint n, uint g0, uint g1, uint lane, __private float* acc) {
  for (uint r = 0; r < M; ++r) acc[r] = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    for (uint r = 0; r < M; ++r) {
      if (r >= n) break;
      float gacc = 0.f;
      for (uint j = 0; j < 8; ++j) gacc += dot8(wv[j], vload8(0, x + xo[r] + g * GROUP + j * 8));
      acc[r] += gacc * scale;
    }
  }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
#ifdef UNION_UP
// Work-group (u, blk): expert of distinct rank u, n-tiles 4 blk .. 4 blk + 3; sub-group (tile_idx, q): one 16-column
// tile, K slice q, merged per pair by nb_moe_up's pairwise tree.
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
NB_SG16
__kernel void nb_moe_up_U(__global const uint* restrict route, __global const ushort* restrict lat,
                          __global const uint* restrict w, __global ushort* restrict a, uint n_exp CNT_ARG) {
  __local uint ids[NPAIR], first[NPAIR], pl[NPAIR], sx, np;
  __local float red[M][4 * UP_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint tile_idx = sg / UP_KS;
  const uint q = sg % UP_KS;
  const uint u = get_group_id(0) / (NT_UP / 4);
  const uint blk = get_group_id(0) % (NT_UP / 4);
  const uint n_tile = blk * 4 + tile_idx;
  nb_union(route, n_exp, u, lid, WG_UP, ids, first, &sx, pl, &np);
  const uint e = sx, n_all = np;
  if (e == 0xFFFFFFFFu) return;   // uniform: fewer distinct experts than u + 1
#ifdef UNION_COUNT
  if (lid == 0) atomic_inc(cnt + e);
#endif
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = NB_EXPERT_UP(w, e) + ((size_t)n_tile * G_UP + g0) * TILE_U32;
  for (uint base = 0; base < n_all; base += M) {
    const uint n = min((uint)M, n_all - base);
    uint xo[M];
    for (uint r = 0; r < M; ++r) xo[r] = (pl[base + min(r, n - 1)] / RT_K) * NB_LAT;
    float acc[M];
    tile_dot_rows(tile, lat, xo, n, g0, g0 + G_UP / UP_KS, lane, acc);
    for (uint r = 0; r < M; ++r) red[r][sg][lane] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
      if (q < stride)
        for (uint r = 0; r < M; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (q == 0)
      for (uint r = 0; r < n; ++r) {
        const uint p = pl[base + r];   // pair = row x 9 + slot: a's [row][slot] row
        float v = rf(red[r][sg][lane]);
        v = v > 0.0f ? v : 0.0f;
        a[(size_t)p * MOE_I + n_tile * SG + lane] = rne_bf16(v * v);
      }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next pass reuses red
  }
}
#endif  // UNION_UP

// ---------------------------------------------------------------------------------------------------------------
#ifdef UNION_DN
// Work-group (u, blk): expert of distinct rank u, latent n-tiles 4 blk .. 4 blk + 3; sub-group (tile_idx, q): one tile,
// K slice q of the expert's down block, merged per pair by nb_moe_down's tree; d[row][slot][n] = rne(Σ).
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
NB_SG16
__kernel void nb_moe_down_U(__global const uint* restrict route, __global const ushort* restrict a,
                            __global const uint* restrict w, __global ushort* restrict d, uint n_exp CNT_ARG) {
  __local uint ids[NPAIR], first[NPAIR], pl[NPAIR], sx, np;
  __local float red[M][4 * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint tile_idx = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint u = get_group_id(0) / (NT_DN / 4);
  const uint blk = get_group_id(0) % (NT_DN / 4);
  const uint n_tile = blk * 4 + tile_idx;
  nb_union(route, n_exp, u, lid, WG_DN, ids, first, &sx, pl, &np);
  const uint e = sx, n_all = np;
  if (e == 0xFFFFFFFFu) return;   // uniform
#ifdef UNION_COUNT
  if (lid == 0) atomic_inc(cnt + 512 + e);
#endif
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile = NB_EXPERT_DN(w, e) + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  for (uint base = 0; base < n_all; base += M) {
    const uint n = min((uint)M, n_all - base);
    uint xo[M];
    for (uint r = 0; r < M; ++r) xo[r] = pl[base + min(r, n - 1)] * MOE_I;
    float acc[M];
    tile_dot_rows(tile, a, xo, n, g0, g0 + G_DN / DN_KS, lane, acc);
    for (uint r = 0; r < M; ++r) red[r][sg][lane] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
      if (q < stride)
        for (uint r = 0; r < M; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (q == 0)
      for (uint r = 0; r < n; ++r) d[(size_t)pl[base + r] * NB_LAT + n_tile * SG + lane] = rne_bf16(red[r][sg][lane]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // UNION_DN

// ---------------------------------------------------------------------------------------------------------------
#ifdef UNION_COMB
// Work-item (n, m): nb_moe_down's combine over row m's nine down rows, ascending id (the route row's slot order).
#define WG_COMB 256
__attribute__((reqd_work_group_size(WG_COMB, 1, 1)))
__kernel void nb_moe_comb_U(__global const uint* restrict route, __global const ushort* restrict d,
                            __global ushort* restrict r) {
  const uint n = get_group_id(0) * WG_COMB + get_local_id(0);
  const uint m = get_group_id(1);
  if (n >= NB_LAT) return;
  const __global uint* restrict rr = route + (size_t)m * RW;
  float acc = 0.0f;
  for (uint k = 0; k < RT_K; ++k) {
    const float t = bf16f(d[((size_t)m * RT_K + k) * NB_LAT + n]) * as_float(rr[R_W + k]);
    acc = acc + t;
  }
  r[(size_t)m * NB_LAT + n] = rne_bf16(acc);
}
#endif  // UNION_COMB
