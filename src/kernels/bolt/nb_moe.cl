// nb_moe.cl - spec 25c: Nemotron 3.5 Bolt's routed experts - its router, the fp32 sigmoid top-9 route, the latent
// relu2 experts (top-9 of 240 .. 496, I 1536 .. 2560, in the 1024 latent) with the ascending-id combine. Four families,
// each compiled only when its defines are given:
//
//   nb_router(x, w, logits, n_exp)                    ROUTER_K   grid (n_exp / 16, M), WG 128
//   nb_route(logits, bias, route, n_exp)              ROUTE_K    grid (1, M), WG 256
//   nb_moe_up(route, lat, w, a, n_exp)                MOE_I      grid (9 x I / 64, M), WG 64 x UP_KS
//   nb_moe_up (ROUTE_FUSED: the arm `route`)         (logits, bias, route, lat, w, a, n_exp)
//   nb_moe_down(route, a, w, r, n_exp)                MOE_I      grid (1024 / 16, M), WG 16 x 9 x DN_KS
//
// kol_moe.cl's structure (spec 20c: 256 lanes x 2 experts, the fp32 sigmoid router, the ascending-id fp32 combine) and
// lfm_moe.cl's family pattern (spec 24c) carried to Bolt's semantics (spec 25 §2.4; 25a's facts). E is a kernel argument
// (`n_exp`, 240 .. 496 by block, a multiple of 16, at most 512): it bounds loops and the grid and changes no rounding, so
// one binary serves every block; I (the expert block's shape) is a define. moe.cl / kol_moe.cl / lfm_moe.cl are untouched
// (their route rows hold 8 slots; Bolt's is top-9 - spec 25 §6.1).
//
//   * the router (nb_router): E bf16 rows over gemv_bf16's tiles into fp32 logits - per column 8 lanes of 512
//       consecutive k (ascending; bf16 x bf16 products are exact in fp32), the 8 partials by the stride 4 / 2 / 1 tree;
//       NEVER rounded (R:1044) - a family kernel (not gemv_bf16) so the arm `router` (nb_res_norm _R) repeats its order
//   * the route (R:1040-1052, n_group 1): s = sigmoid(l) in fp32 (torch's vectorised form: 1 / (exp(0 - l) + 1),
//       Sleef's exp - bitwise torch's on the fixture); sel = s + bias (fp32); rank = #{j < E : sel_j > sel_e or (sel_j ==
//       sel_e and j < e)} - EXACT ties to the lower id (torch keeps either: 25a), lanes >= E never rank; the selected set
//       is rank < 9; w = (s / (Σ + 1e-20)) x 5.0 with Σ the nine s added in ASCENDING id (25a: torch.sum's order has no
//       name - this is the twin's own fixed order; B3 compares w with an ulp tolerance)
//   * the experts (R:871-881, non-gated): u = rne(up(lat)); a = rne(relu(u)²) (relu2 on bf16); d = rne(down(a)) - the
//       layout-1 tile_dot of moe.cl, every K split merged by the fixed pairwise tree
//   * the combine (R:925-950, Decision 10): acc = 0; for the nine slots in ASCENDING id: acc = acc + (f32(d) · w) - the
//       product rounded to fp32, THEN the add (no fma: FP_CONTRACT OFF); r = rne(acc). Not folded: fc2_latent's GEMV takes
//       r and the next norm folds fc2 and shared_down (nb_prep.cl's two-source fold)
//
// The route row (u32 words per (layer, token); host mirror kernels::bolt::route), 48 words:
//   [R_IDS + j]   expert id of slot j, ASCENDING, j < 9                     (u32)
//   [R_W + j]     w_j at that id, fp32 (never rounded)                     (fp32 bits)
//   [R_SEL + r]   sel of rank r, r = 0 .. 9 (rank 9: the first expert not taken - the near-tie margin)   (fp32 bits)
//   [R_SUM]       Σ of the nine s (before + 1e-20)                         (fp32 bits)
//   [R_SELMIN]    the smallest selected sel (rank 8)                       (fp32 bits)
//
// **The arm `route`** (ROUTE_FUSED, spec 25 §6.1): every up work-group recomputes the 256-lane rank from the logits
// (the same functions), work-group 0 also writes the route row nb_moe_down reads - one launch fewer per MoE block,
// bitwise the same.
//
// **Portability** (moe.cl's): sub-group block reads under cl_intel_subgroups, the plain loads they equal otherwise,
// every cross-lane step through SLM - the file builds as OpenCL 1.2 for the Mac (bolt_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#include "nb_common.h"

#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 48                       /* route row words (kernels::bolt::route::kWords) */
#define R_IDS 0
#define R_W 9
#define R_SEL 18
#define R_SUM 28
#define R_SELMIN 29
#define RT_K 9                      /* top-9 */
#define RT_WG 256                   /* the route's lanes: lane l owns experts l and l + 256 */
#define RT_SLOTS 512
#define NB_LAT 1024                 /* the latent */

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define NB_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define NB_SG16
#endif

// ---------------------------------------------------------------------------------------------------------------
// nb_router - work-group = one 16-column n-tile of the router (n_exp / 16 of them), work-item (j, lane) = k-span j (k in
// [512 j, 512 j + 512)) of column lane: an ascending chain, then the 8 spans by the stride 4 / 2 / 1 tree.
#ifdef ROUTER_K
#if ROUTER_K % 64 != 0
#error "nb_router: whole k-octet spans"
#endif
#define RSPAN (ROUTER_K / 8)
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void nb_router(__global const ushort* restrict x, __global const ushort* restrict w,
                        __global float* restrict logits, uint n_exp) {
  const uint nt = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint lane = lid % 16, j = lid / 16;
  __local float rr[8][16];
  const __global ushort* restrict t = w + (size_t)nt * (ROUTER_K / 8) * 128;
  const __global ushort* restrict xr = x + (size_t)m * ROUTER_K;
  float a = 0.0f;
  for (uint k = j * RSPAN; k < (j + 1) * RSPAN; ++k)
    a += bf16f(t[((size_t)(k / 8) * 8 + k % 8) * 16 + lane]) * bf16f(xr[k]);
  rr[j][lane] = a;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 4; stride > 0; stride >>= 1) {
    if (j < stride) rr[j][lane] += rr[j + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (j == 0) logits[(size_t)m * n_exp + nt * 16 + lane] = rr[0][lane];
}
#endif  // ROUTER_K

// ---------------------------------------------------------------------------------------------------------------
// The route of one token on 256 lanes (lane l: experts l and l + 256; e >= n_exp never ranks). Leaves the ranks 0 .. 9's
// ids and sel in top_id / top_sel, s in ls, and the nine selected ids ASCENDING in sid; lane 0 writes the route row when
// `write_row`.
#if defined(ROUTE_K) || defined(MOE_I)
inline void nb_route_row(__global const float* restrict lr, __global const float* restrict bias, uint n_exp, uint lane,
                         __local float* lsel, __local float* ls, __local uint* top_id, __local float* top_sel,
                         __local uint* sid, __global uint* restrict r, bool write_row) {
  for (uint q = 0; q < RT_SLOTS / RT_WG; ++q) {
    const uint e = lane + q * RT_WG;
    if (e < n_exp) {
      const float s = nb_sigmoid(lr[e]);
      ls[e] = s;
      lsel[e] = s + bias[e];
    } else {
      ls[e] = 0.0f;
      lsel[e] = -INFINITY;
    }
  }
  if (lane <= RT_K) {   // a defined row even if NaNs break the ranking
    top_id[lane] = lane;
    top_sel[lane] = -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint q = 0; q < RT_SLOTS / RT_WG; ++q) {
    const uint e = lane + q * RT_WG;
    if (e >= n_exp) continue;
    const float sel = lsel[e];
    uint rank = 0;
    for (uint j = 0; j < n_exp; ++j) {
      const float o = lsel[j];
      rank += (o > sel || (o == sel && j < e)) ? 1u : 0u;
    }
    if (rank <= RT_K) {
      top_id[rank] = e;
      top_sel[rank] = sel;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0) {   // the selected ids (ranks 0 .. 8) re-ordered by id: slot j's is the one with exactly j smaller
    for (uint j = 0; j < RT_K; ++j) {
      uint id = 0u;
      for (uint k = 0; k < RT_K; ++k) {
        uint smaller = 0;
        for (uint k2 = 0; k2 < RT_K; ++k2) smaller += top_id[k2] < top_id[k] ? 1u : 0u;
        if (smaller == j) id = top_id[k];
      }
      sid[j] = id;
    }
    if (write_row) {
      float sum = 0.0f;
      for (uint j = 0; j < RT_K; ++j) sum = sum + ls[sid[j]];   // ascending id, sequential, fp32
      for (uint j = 0; j < RT_K; ++j) {
        r[R_IDS + j] = sid[j];
        r[R_W + j] = as_uint((ls[sid[j]] / (sum + 1e-20f)) * 5.0f);
      }
      for (uint k = 0; k <= RT_K; ++k) r[R_SEL + k] = as_uint(top_sel[k]);
      r[R_SUM] = as_uint(sum);
      r[R_SELMIN] = as_uint(top_sel[RT_K - 1]);
      for (uint k = R_SELMIN + 1; k < RW; ++k) r[k] = 0u;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#ifdef ROUTE_K
#if ROUTE_K != RT_K
#error "nb_route: top-9"
#endif
__attribute__((reqd_work_group_size(RT_WG, 1, 1)))
__kernel void nb_route(__global const float* restrict logits, __global const float* restrict bias,
                       __global uint* restrict route, uint n_exp) {
  const uint m = get_group_id(1);
  __local float lsel[RT_SLOTS], ls[RT_SLOTS];
  __local uint top_id[RT_K + 1], sid[RT_K];
  __local float top_sel[RT_K + 1];
  nb_route_row(logits + (size_t)m * n_exp, bias, n_exp, get_local_id(0), lsel, ls, top_id, top_sel, sid,
               route + (size_t)m * RW, true);
}
#endif  // ROUTE_K

// ---------------------------------------------------------------------------------------------------------------
// The routed experts: 9 slots (ascending id), expert e's up / down at NB_EXPERT_UP / NB_EXPERT_DN (nb_expert.h).
#ifdef MOE_I
#if !defined(UP_KS) || !defined(DN_KS)
#error "nb_moe: UP_KS and DN_KS must be defined"
#endif
#define G_UP (NB_LAT / GROUP)       /* 16 k-groups of the latent */
#define NT_UP (MOE_I / SG)          /* up's n-tiles */
#define G_DN (MOE_I / GROUP)        /* down's k-groups */
#define NT_DN (NB_LAT / SG)         /* 64 */
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * RT_K * DN_KS)
#include "nb_expert.h"
#if MOE_I % GROUP != 0 || NT_UP % 4 != 0
#error "nb_moe: I is whole k-groups and whole 4-tile blocks"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "nb_moe: UP_KS must be a power of two dividing 1024 / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "nb_moe: DN_KS must be a power of two dividing I / 64"
#endif
#if defined(ROUTE_FUSED) && WG_UP != RT_WG
#error "nb_moe_up _R: the up work-group must be the route's 256 lanes"
#endif

// moe.cl's dot8 / load_tile8 / tile_dot, verbatim (gemv.cl's LAYOUT 1 order).
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
inline float tile_dot(const __global uint* restrict tile, const __global ushort* restrict xr, uint g0, uint g1,
                      uint lane) {
  float acc = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    float gacc = 0.f;
    for (uint j = 0; j < 8; ++j) gacc += dot8(wv[j], vload8(0, xr + g * GROUP + j * 8));
    acc += gacc * scale;
  }
  return acc;
}

// up: work-group (slot, blk) owns n-tiles 4 blk .. 4 blk + 3 of its slot's expert; sub-group (tile_idx, q): one 16-column
// tile, K slice q, merged by the pairwise tree:  a[m][slot][i] = rne(relu(rne(Σ up))²)
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
NB_SG16
__kernel void nb_moe_up(
#ifdef ROUTE_FUSED
                        __global const float* restrict logits, __global const float* restrict bias,
#endif
                        __global uint* restrict route, __global const ushort* restrict lat,
                        __global const uint* restrict w, __global ushort* restrict a, uint n_exp) {
  __local float red[4 * UP_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint tile_idx = sg / UP_KS;
  const uint q = sg % UP_KS;
  const uint slot = get_group_id(0) / (NT_UP / 4);
  const uint blk = get_group_id(0) % (NT_UP / 4);
  const uint m = get_group_id(1);
  const uint n_tile = blk * 4 + tile_idx;
#ifdef ROUTE_FUSED
  // The arm `route`: nb_route's function on this work-group's 256 lanes; work-group 0 writes the row.
  __local float lsel[RT_SLOTS], ls[RT_SLOTS];
  __local uint top_id[RT_K + 1], sid[RT_K];
  __local float top_sel[RT_K + 1];
  nb_route_row(logits + (size_t)m * n_exp, bias, n_exp, lid, lsel, ls, top_id, top_sel, sid, route + (size_t)m * RW,
               get_group_id(0) == 0);
  uint e = sid[slot];
#else
  uint e = route[(size_t)m * RW + R_IDS + slot];
#endif
  e = min(e, n_exp - 1);   // never past the allocation, whatever the row holds
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = NB_EXPERT_UP(w, e) + ((size_t)n_tile * G_UP + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, lat + (size_t)m * NB_LAT, g0, g0 + G_UP / UP_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0) {
    float u = rf(red[sg][lane]);
    u = u > 0.0f ? u : 0.0f;
    a[((size_t)m * RT_K + slot) * MOE_I + n_tile * SG + lane] = rne_bf16(u * u);
  }
}

// down + the combine, one 16-column n-tile of the latent per work-group; sub-group (slot, q) reads slot `slot`'s
// activation row and its expert's down block, K slice q; then sub-group 0:
//   acc = 0;  for k = 0 .. 8 (ascending id):  acc = acc + (f32(rne(down_k)) · w_k);  r[m][n] = rne(acc)
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
NB_SG16
__kernel void nb_moe_down(__global const uint* restrict route, __global const ushort* restrict a,
                          __global const uint* restrict w, __global ushort* restrict r, uint n_exp) {
  __local float red[RT_K * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const __global uint* restrict rr = route + (size_t)m * RW;
  uint e = rr[R_IDS + slot];
  e = min(e, n_exp - 1);
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile = NB_EXPERT_DN(w, e) + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, a + ((size_t)m * RT_K + slot) * MOE_I, g0, g0 + G_DN / DN_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float acc = 0.0f;
    for (uint k = 0; k < RT_K; ++k) {
      const float t = rf(red[k * DN_KS][lane]) * as_float(rr[R_W + k]);
      acc = acc + t;
    }
    r[(size_t)m * NB_LAT + n_tile * SG + lane] = rne_bf16(acc);
  }
}
#endif  // MOE_I
