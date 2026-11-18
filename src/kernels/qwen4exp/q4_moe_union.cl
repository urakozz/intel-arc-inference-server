// q4_moe_union.cl - open-work item 48 (spec 28m §4.4 / §4.5): Qwen3.8-Flash-Next's 11-slot MoE block at a verify's M rows
// with each distinct expert read ONCE for every row that chose it, and the shared expert once for all M rows - bitwise
// q4_moe.cl's per-row form at the same M (the same products in the same order per row; only who reads the weights
// changes). Ported from the cancelled Bolt's nb_moe_union.cl (git e7937cec^) to gated SiLU experts. The two entry
// points keep q4_moe.cl's names, arguments, grids and work-group sizes, so a list binds this binary in place of the
// per-row one and launches the same commands:
//
//   q4_moe_gate_up(route, x, w_gu, w_sh_gu, h)     grid (11 x NT_UP / 4, M), WG 64 x UP_KS
//   q4_moe_down(route, h, w_dn, w_sh_dn, y [, land, ctr])   grid (HIDDEN / 16, M), WG 16 x 11 x DN_KS
// (MOE_TP_PART: q4_moe.cl's q4_moe_finish is bound from the per-row binary; this file has no finish.)
// (UNION_COUNT, test-only: both entries take a last argument `cnt` u32 [2][MOE_E + 1] and add one per read of an
// expert's tiles - gate||up cnt[e] once a work-group (the shared expert at cnt[MOE_E]), down cnt[MOE_E + 1 + e] once a
// (work-group, K slice); tests/kernels/qwen4exp_moe_union_test.cc proves each chosen expert is read once a verify.)
//
// **The union schedule** (every work-group derives it alone, in SLM, from the M route rows - no host step, no extra
// launch): the M x 10 (row, slot) pairs, pair p = row x 10 + slot, their ids clamped as q4_moe.cl does; a pair's owner
// is the first pair with its id; the distinct experts are ranked in the order of their first pairs; the pairs sorted
// by (rank, p) - so each expert's pairs are in row order. Set 0 is the shared expert (all M rows), set s >= 1 the
// expert of rank s - 1.
//   gate||up  work-group (set, blk): set = group y x 11 + group x / (NT_UP / 4) (11 M sets >= 10 M + 1), the sets past
//             the union exit at once; the 4 n-tiles of blk over all of K, each weight tile and scale loaded once a
//             k-group and each pair's own tile_dot chain against its row; h[row][slot] as q4_moe_gate_up writes it.
//   down      work-group (n_tile, 0) does the whole n-tile for every row (the work-groups of rows 1.. exit at once):
//             sub-group (xs, q) takes the sets xs, xs + 11, ... at K slice q, each pair's partial into SLM, then each
//             row's K-slice tree and q4_moe_down's epilogue (the combine; under MOE_TP_PART each slot's fp32 partial
//             and the send of the M chunks (n_tile, m) of this n-tile to the peer, tp_send's protocol per chunk).
// The weights are found only through Q4_EXPERT_GU / Q4_EXPERT_DN (q4_moe.cl's pair, the _TBL table row form included),
// so an expert in pinned host memory is read once over the link too.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#error "q4_moe_union: M (the verify's rows) must be defined"
#endif
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 32                       /* route row words (kernels::qwen4exp::route::kWords) */
#define R_IDS 0
#define R_W 16
#define R_SG 26

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define Q4_SG16 __attribute__((intel_reqd_sub_group_size(SG)))

#include "bf16.h"
inline float rf(float f) { return bf16f(rne_bf16(f)); }
// q4_moe.cl's exp_torch / silu_t, verbatim.
inline float exp_torch(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

#if !defined(MOE_E) || !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS) || \
    !defined(SHARED_BF16)
#error "q4_moe_union: MOE_E, MOE_K, HIDDEN, INTER, UP_KS, DN_KS and SHARED_BF16 must be defined"
#endif
#if MOE_K < 1 || MOE_K > 15 || M < 1 || M > 8
#error "q4_moe_union: MOE_K is 1..15 (the route row's ids), M 1..8"
#endif
#define SLOTS (MOE_K + 1)
#define SHARED_SLOT MOE_K
#define NPAIR (M * MOE_K)           /* the routed (row, slot) pairs of one verify */
#define G_UP (HIDDEN / GROUP)
#define NT_UP (2 * INTER / SG)
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32)
#define G_DN (INTER / GROUP)
#define NT_DN (HIDDEN / SG)
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32)
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * SLOTS * DN_KS)
#define O_UP (HIDDEN / 8)
#define O_DN (INTER / 8)
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0 || HIDDEN % SG != 0
#error "q4_moe_union: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0 || (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "q4_moe_union: UP_KS / DN_KS must be powers of two dividing the k-groups"
#endif

// ---- the expert address: q4_moe.cl's pair, repeated (no kernel #includes another .cl) ---------------------------
#if Q4_EXPERT_TABLE
#define Q4_EXPERT_GU(base, id) ((__global const uint*)(intptr_t)((__global const ulong*)(base))[(id)])
#define Q4_EXPERT_DN(base, id) ((__global const uint*)(intptr_t)((__global const ulong*)(base))[(id)])
#else
#define Q4_EXPERT_GU(base, id) ((base) + (size_t)(id) * UP_BLK_U32)
#define Q4_EXPERT_DN(base, id) ((base) + (size_t)(id) * DN_BLK_U32)
#endif

#ifdef UNION_COUNT
#define CNT_ARG , __global volatile uint* restrict cnt
#else
#define CNT_ARG
#endif

// q4_moe.cl's load_tile8, verbatim (its dot8 is unrolled into tile_dot_rows below).
inline void load_tile8(const __global uint* restrict tile, uint lane, uint* wv) {
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
}
// q4_moe.cl's tile_dot over n <= M rows at once: the tile and its scale loaded once a k-group and each word's 8 weights
// decoded once, then per row r < n exactly tile_dot's chain (per word dot8's sum from 0 over the 8 products in order,
// gacc from 0 over the 8 words, acc += gacc x scale). Row r's activations start at x + xo[r]. Every loop over the rows
// runs M times with a predicate, unrolled, so the row arrays stay in registers.
inline void tile_dot_rows(const __global uint* restrict tile, const __global ushort* restrict x, __private const uint* xo,
                          uint n, uint g0, uint g1, uint lane, __private float* acc) {
  #pragma unroll
  for (uint r = 0; r < M; ++r) acc[r] = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    float gacc[M];
    #pragma unroll
    for (uint r = 0; r < M; ++r) gacc[r] = 0.f;
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
      float f[8];
      #pragma unroll
      for (uint k = 0; k < 8; ++k) f[k] = (float)((int)((wv[j] >> (4 * k)) & 0xFu) - 8);
      #pragma unroll
      for (uint r = 0; r < M; ++r)
        if (r < n) {
          const ushort8 xv = vload8(0, x + xo[r] + g * GROUP + j * 8);
          float a = 0.f;
          a += f[0] * bf16f(xv.s0);
          a += f[1] * bf16f(xv.s1);
          a += f[2] * bf16f(xv.s2);
          a += f[3] * bf16f(xv.s3);
          a += f[4] * bf16f(xv.s4);
          a += f[5] * bf16f(xv.s5);
          a += f[6] * bf16f(xv.s6);
          a += f[7] * bf16f(xv.s7);
          gacc[r] += a;
        }
    }
    #pragma unroll
    for (uint r = 0; r < M; ++r)
      if (r < n) acc[r] += gacc[r] * scale;
  }
}
#if SHARED_BF16
// q4_moe.cl's bf16_dot over n <= M rows: each k-octet's weights loaded once, per row the same chain.
inline void bf16_dot_rows(const __global ushort* restrict tile, const __global ushort* restrict x, __private const uint* xo,
                          uint n, uint o0, uint o1, __private float* acc) {
  #pragma unroll
  for (uint r = 0; r < M; ++r) acc[r] = 0.f;
  for (uint o = o0; o < o1; ++o) {
    const ushort8 wv = intel_sub_group_block_read_us8(tile + (size_t)o * 128);
    #pragma unroll
    for (uint r = 0; r < M; ++r) {
      if (r >= n) continue;
      const ushort8 xv = vload8(0, x + xo[r] + o * 8);
      float a = acc[r];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[r] = a;
    }
  }
}
#define SH_T __global const ushort*
#else
#define SH_T __global const uint*
#endif

// The union schedule: ids[p], the pairs sorted by (rank, p) in pl[], expert rank d's pairs at pl[st[d] .. st[d + 1]),
// its id in ex[d], the distinct count in *nd. Every lane sees the same values after the last barrier.
inline void q4_union(__global const uint* restrict route, uint lid, uint wg, __local uint* ids, __local uint* own,
                     __local uint* pl, __local uint* st, __local uint* ex, __local uint* nd) {
  for (uint p = lid; p < NPAIR; p += wg)
    ids[p] = min(route[(size_t)(p / MOE_K) * RW + R_IDS + p % MOE_K], (uint)(MOE_E - 1));
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lid; p < NPAIR; p += wg) {   // the owner: the first pair with p's id
    uint f = p;
    for (uint q = 0; q < p; ++q)
      if (ids[q] == ids[p]) {
        f = q;
        break;
      }
    own[p] = f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lid; p < NPAIR; p += wg) {
    const uint f = own[p];
    uint rank = 0, before = 0, mine = 0;   // rank: first pairs before the owner; then the place among the sort
    for (uint q = 0; q < NPAIR; ++q) {
      const uint fq = own[q];
      rank += (q < f && fq == q) ? 1u : 0u;
      before += fq < f ? 1u : 0u;                 // pairs of an expert ranked lower (their owner comes earlier)
      mine += (fq == f && q < p) ? 1u : 0u;       // the expert's earlier pairs
    }
    pl[before + mine] = p;
    if (p == f) {
      st[rank] = before;
      ex[rank] = ids[p];
    }
  }
  if (lid == 0) {
    uint n = 0;
    for (uint q = 0; q < NPAIR; ++q) n += own[q] == q ? 1u : 0u;
    *nd = n;
    st[n] = NPAIR;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

// ---------------------------------------------------------------------------------------------------------------
// gate||up: work-group (set, blk) owns n-tiles 4 blk .. 4 blk + 3 of its set's expert; sub-group (tile_idx, q): one
// 16-column tile, K slice q, merged per pair by q4_moe_gate_up's pairwise tree.
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
Q4_SG16
__kernel void q4_moe_gate_up(__global const uint* restrict route, __global const ushort* restrict x,
                             __global const uint* restrict w, SH_T restrict w_sh, __global ushort* restrict h CNT_ARG) {
  __local uint ids[NPAIR], own[NPAIR], pl[NPAIR], st[NPAIR + 1], ex[NPAIR], nd;
  __local float red[M][4 * UP_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint tile_idx = sg / UP_KS;
  const uint q = sg % UP_KS;
  const uint set = get_group_id(1) * SLOTS + get_group_id(0) / (NT_UP / 4);
  const uint blk = get_group_id(0) % (NT_UP / 4);
  const uint n_tile = blk * 4 + tile_idx;
  q4_union(route, lid, WG_UP, ids, own, pl, st, ex, &nd);
  if (set > nd) return;   // uniform: fewer distinct experts than set
  const bool shared = set == 0;
  const uint d = set - 1;
  const uint e = shared ? 0u : ex[d];
  const uint p0 = shared ? 0u : st[d], n_all = shared ? M : st[d + 1] - st[d];
#ifdef UNION_COUNT
  if (lid == 0) atomic_inc(cnt + (shared ? MOE_E : e));
#endif
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = Q4_EXPERT_GU(w, e) + ((size_t)n_tile * G_UP + g0) * TILE_U32;
  for (uint base = 0; base < n_all; base += M) {   // more than M pairs only on a row that repeats an id
    const uint n = min((uint)M, n_all - base);
    uint xo[M];
    #pragma unroll
    for (uint r = 0; r < M; ++r) {
      const uint i = base + min(r, n - 1);
      xo[r] = (shared ? i : pl[p0 + i] / MOE_K) * HIDDEN;
    }
    float acc[M];
    if (!shared) {
      tile_dot_rows(tile, x, xo, n, g0, g0 + G_UP / UP_KS, lane, acc);
    } else {
#if SHARED_BF16
      const uint o0 = q * (O_UP / UP_KS);
      bf16_dot_rows(w_sh + (size_t)n_tile * O_UP * 128, x, xo, n, o0, o0 + O_UP / UP_KS, acc);
#else
      tile_dot_rows(w_sh + ((size_t)n_tile * G_UP + g0) * TILE_U32, x, xo, n, g0, g0 + G_UP / UP_KS, lane, acc);
#endif
    }
    #pragma unroll
    for (uint r = 0; r < M; ++r) red[r][sg][lane] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
      if (q < stride)
        #pragma unroll
        for (uint r = 0; r < M; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (q == 0 && (tile_idx & 1u) == 0) {
      const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
      for (uint r = 0; r < n; ++r) {
        const uint p = shared ? 0u : pl[p0 + base + r];
        const uint row = shared ? base + r : p / MOE_K, slot = shared ? SHARED_SLOT : p % MOE_K;
        const ushort g_b = rne_bf16(red[r][sg][lane]);
        const ushort u_b = rne_bf16(red[r][sg + UP_KS][lane]);
        h[((size_t)row * SLOTS + slot) * INTER + i] = rne_bf16(rf(silu_t(bf16f(g_b))) * bf16f(u_b));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next pass reuses red
  }
}

// MOE_TP_PART: q4_moe.cl's tensor-parallel half (DN_KS 1) - y fp32 [SLOTS][M][HIDDEN] and the chunks to the peer.
#if MOE_TP_PART
#ifndef RANK
#error "q4_moe_union MOE_TP_PART: RANK (the card)"
#endif
#if DN_KS != 1
#error "q4_moe_union MOE_TP_PART: DN_KS 1 (a card's 5 groups in one partial)"
#endif
#define TP_MROWS 8
#define DATA_WORDS (SLOTS * TP_MROWS * HIDDEN)
#define J_CHUNKS (NT_DN * TP_MROWS)
#include "tp_wire.h"
#define Y_T __global float* restrict
#else
#define Y_T __global ushort* restrict
#endif
// down + the combine, one 16-column n-tile of the hidden row for every row per work-group.
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
Q4_SG16
__kernel void q4_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                          __global const uint* restrict w, SH_T restrict w_sh, Y_T y
#if MOE_TP_PART
                          , __global uint* land, __global uint* ctr
#endif
                          CNT_ARG) {
  __local uint ids[NPAIR], own[NPAIR], pl[NPAIR], st[NPAIR + 1], ex[NPAIR], nd;
  __local float red[M * SLOTS * DN_KS][SG];   // [row][slot][K slice]
  if (get_group_id(1) != 0) return;   // uniform: work-group (n_tile, 0) does every row
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint xs = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  q4_union(route, lid, WG_DN, ids, own, pl, st, ex, &nd);
  const uint g0 = q * (G_DN / DN_KS);
  for (uint set = xs; set <= nd; set += SLOTS) {   // uniform across the sub-group
    const bool shared = set == 0;
    const uint d = set - 1;
    const uint e = shared ? 0u : ex[d];
    const uint p0 = shared ? 0u : st[d], n_all = shared ? M : st[d + 1] - st[d];
#ifdef UNION_COUNT
    if (lane == 0) atomic_inc(cnt + (MOE_E + 1) + (shared ? MOE_E : e));
#endif
    const __global uint* restrict tile = Q4_EXPERT_DN(w, e) + ((size_t)n_tile * G_DN + g0) * TILE_U32;
    for (uint base = 0; base < n_all; base += M) {
      const uint n = min((uint)M, n_all - base);
      uint xo[M], rs[M];   // the pair's h row and its [row][slot] place
      #pragma unroll
      for (uint r = 0; r < M; ++r) {
        const uint i = base + min(r, n - 1);
        if (shared) {
          rs[r] = i * SLOTS + SHARED_SLOT;
        } else {
          const uint p = pl[p0 + i];
          rs[r] = (p / MOE_K) * SLOTS + p % MOE_K;
        }
        xo[r] = rs[r] * INTER;
      }
      float acc[M];
      if (!shared) {
        tile_dot_rows(tile, h, xo, n, g0, g0 + G_DN / DN_KS, lane, acc);
      } else {
#if SHARED_BF16
        const uint o0 = q * (O_DN / DN_KS);
        bf16_dot_rows(w_sh + (size_t)n_tile * O_DN * 128, h, xo, n, o0, o0 + O_DN / DN_KS, acc);
#else
        tile_dot_rows(w_sh + ((size_t)n_tile * G_DN + g0) * TILE_U32, h, xo, n, g0, g0 + G_DN / DN_KS, lane, acc);
#endif
      }
      #pragma unroll
      for (uint r = 0; r < M; ++r)   // a constant trip count: acc and rs stay in registers
        if (r < n) red[rs[r] * DN_KS + q][lane] = acc[r];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < M * SLOTS * SG; i += WG_DN) {   // each (row, slot)'s K-slice tree, q4_moe_down's order
    const uint rs = i / SG, ln = i % SG;
    for (uint stride = DN_KS / 2; stride > 0; stride >>= 1)
      for (uint k = 0; k < stride; ++k) red[rs * DN_KS + k][ln] += red[rs * DN_KS + k + stride][ln];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#if MOE_TP_PART
  // tp_send for the chunks (n_tile, m), m < M: every slot's 16 words here and in the peer's landing, the stamp lines,
  // the flags - q4_moe_down's per-chunk protocol, the M chunks of this n-tile from one work-group.
  __local uint s_n[M];
  if (lid == 0)
    for (uint m = 0; m < M; ++m) s_n[m] = tp_next(ctr, n_tile, m);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < M * SLOTS * SG; i += WG_DN) {
    const uint m = i / (SLOTS * SG), slot = (i / SG) % SLOTS, ln = i % SG;
    const uint n = s_n[m];
    const float v = red[m * SLOTS + slot][ln];
    const size_t e = ((size_t)slot * M + m) * HIDDEN + n_tile * SG + ln;
    y[e] = v;
    land[(n & 1u) * DATA_WORDS + e] = as_uint(v);
  }
  for (uint i = lid; i < M * TP_TRAILER_STRIDE; i += WG_DN) {
    const uint m = i / TP_TRAILER_STRIDE;
    const uint n = s_n[m];
    land[TRAILER(n & 1u, n_tile * TP_MROWS + m) + i % TP_TRAILER_STRIDE] = n;
  }
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    for (uint m = 0; m < M; ++m) {
      const uint n = s_n[m];
      *(volatile __global uint*)(land + FLAG(n & 1u, n_tile * TP_MROWS + m)) = n;
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  }
#else
  for (uint i = lid; i < M * SG; i += WG_DN) {
    const uint m = i / SG, ln = i % SG;
    const __global uint* restrict rr = route + (size_t)m * RW;
    float sum = 0.0f;
    for (uint k = 0; k < MOE_K; ++k) {
      const float t = rf(rf(red[(m * SLOTS + k) * DN_KS][ln]) * as_float(rr[R_W + k]));
      sum = sum + t;
    }
    const ushort r_b = rne_bf16(sum);
    const ushort sh_b = rne_bf16(rf(red[(m * SLOTS + SHARED_SLOT) * DN_KS][ln]) * as_float(rr[R_SG]));
    y[(size_t)m * HIDDEN + n_tile * SG + ln] = rne_bf16(bf16f(r_b) + bf16f(sh_b));
  }
#endif
}
