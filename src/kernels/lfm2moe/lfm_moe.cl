// lfm_moe.cl - spec 24c: LFM2.5-24B-A2B's routed experts - its router, the sigmoid-bf16 route with
// renormalisation, the 64 x 1536 top-4 MoE block (no shared expert). Three families, each compiled only when its
// defines are given:
//
//   lfm_router(x, w, logits)                     ROUTER_E   grid (E / 16, M), WG 128
//   lfm_route(logits, bias, route)               ROUTE_E    grid (1, M), WG ROUTE_E (64)
//   lfm_moe_gate_up(route, x, w_gu, h)           MOE_E      grid (TOP_K x NT_UP / 4, M), WG 64 x UP_KS
//   lfm_moe_gate_up (ROUTE_FUSED: the arm `route`)  (logits, bias, route, x, w_gu, h)
//   lfm_moe_down(route, h, w_dn, h_res)          MOE_E      grid (HIDDEN / 16, M), WG 16 x TOP_K x DN_KS
//
// moe.cl's structure (spec 15c) and k2_moe.cl's route idiom (spec 18b), carried to LFM2's semantics (spec 24 §2.3,
// §2.4, §4.3; 24a's facts) - the layout-1 GEMV inner loop (tile_dot) and the fixed-order, atomic-free reductions
// are moe.cl's; what differs is why this is a new source and moe.cl / k2_moe.cl / kol_moe.cl are untouched:
//
//   * the router (lfm_router): 64 bf16 rows over gemv_bf16's tiles into fp32 logits, per column 8 lanes of 256
//       consecutive k (ascending; bf16 x bf16 products are exact in fp32) and the 8 partials by the stride 4 / 2 / 1
//       tree - a family kernel (not gemv_bf16) so the arm `router` (lfm_res_norm _R, lfm_prep.cl) runs the same
//       order inside the norm's launch and stays bitwise
//   * the route, every op in torch's dtype (M:183-194 - a FOURTH router chain):
//       l_b = rne(logit);  s_b = rne(sigmoid(l_b));  sel = s_b + bias (fp32: selection only)
//       rank = #{j : sel_j > sel_e or (sel_j == sel_e and j < e)}       EXACT ties to the lower id
//       Σ = ((s0 + s1) + s2) + s3 (fp32, rank order);  d = rne(rne(Σ) + 1e-6);  w_j = rne(s_j / d)
//   * the slot order is RANK order (descending sel) and the combine is grouped_mm's (24a: transformers 5.19.0
//       runs it): t_k = rne(f32(rne(down_k)) · w_k), y = rne(((t0 + t1) + t2) + t3) in fp32 - then the residual
//       fold h_res = rne(h_res + y) (moe.cl's last line; the next norm is `_NF`)
//   * no shared expert: four slots (moe.cl's SLOTS = TOP_K + 1 would read a fifth expert's bytes for nothing)
//
// The route row (u32 words per (layer, token); host mirror kernels::lfm2moe::route):
//   [R_IDS + j]  expert id of slot j, RANK order, j < 4                 (u32)
//   [R_W + j]    w_j, a bf16 value held in fp32                          (fp32 bits)
//   [R_SEL + j]  sel of ranks 0 .. 4 (rank 4: the first expert not taken - the near-tie margin)   (fp32 bits)
//   [R_SUM]      d = rne(rne(Σ s_b) + 1e-6)                              (fp32 bits, diagnostics)
//
// **The arm `route`** (ROUTE_FUSED, spec 24 §4.3 "P0's fused arm r"): every gate||up work-group recomputes the
// 64-lane rank from the logits (the same function), work-group 0 also writes the route row lfm_moe_down reads -
// one launch fewer per MoE layer, bitwise the same.
//
// **Portability** (moe.cl's): sub-group block reads under cl_intel_subgroups, the plain loads they equal
// otherwise, every cross-lane step through SLM - the file builds as OpenCL 1.2 for the Mac (lfm2moe_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 16                       /* route row words (kernels::lfm2moe::route::kWords) */
#define R_IDS 0
#define R_W 4
#define R_SEL 8
#define R_SUM 13
#define RK 4                        /* the route's TOP_K (the row holds 4 ids, 5 sel) */

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define LFM_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define LFM_SG16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------------------------------------------
// lfm_router - work-group = one 16-column n-tile of the router (E / 16 of them), work-item (j, lane) = k-span j
// (k in [256 j, 256 j + 256)) of column lane: an ascending chain, then the 8 spans by the stride 4 / 2 / 1 tree.
#ifdef ROUTER_E
#ifndef ROUTER_K
#error "lfm_router: ROUTER_K (hidden) must be defined"
#endif
#if ROUTER_E % 16 != 0 || ROUTER_K % 64 != 0
#error "lfm_router: whole 16-column n-tiles and whole k-octet spans"
#endif
#define RSPAN (ROUTER_K / 8)
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_router(__global const ushort* restrict x, __global const ushort* restrict w,
                         __global float* restrict logits) {
  const uint nt = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint lane = lid % 16, j = lid / 16;
  __local float rr[8][16];
  const __global ushort* restrict t = w + (size_t)nt * (ROUTER_K / 8) * 128;
  const __global ushort* restrict xr = x + (size_t)m * ROUTER_K;
  float a = 0.0f;
  for (uint k = j * RSPAN; k < (j + 1) * RSPAN; ++k) a += bf16f(t[((size_t)(k / 8) * 8 + k % 8) * 16 + lane]) * bf16f(xr[k]);
  rr[j][lane] = a;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 4; stride > 0; stride >>= 1) {
    if (j < stride) rr[j][lane] += rr[j + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (j == 0) logits[(size_t)m * ROUTER_E + nt * 16 + lane] = rr[0][lane];
}
#endif  // ROUTER_E

// ---------------------------------------------------------------------------------------------------------------
// The route of one token (64 lanes of a work-group: lane e owns expert e); lanes >= E take no part. Leaves the
// TOP_K ids (rank order) in top_id; lane 0 (write_row) writes the route row.
#if defined(ROUTE_E) || defined(MOE_E)
#ifndef LFM_E
#define LFM_E 64
#endif
inline void lfm_route_row(__global const float* restrict lr, __global const float* restrict bias, uint lane,
                          __local float* lsel, __local uint* top_id, __local float* top_s, __local float* top_sel,
                          __global uint* restrict r, bool write_row) {
  float s = 0.0f, sel = -INFINITY;
  if (lane < LFM_E) {
    s = rf(sigmoid_f32(rf(lr[lane])));
    sel = s + bias[lane];
    lsel[lane] = sel;
  }
  if (lane <= RK) {   // a defined row even if NaNs break the ranking
    top_id[lane] = lane;
    top_s[lane] = 0.0f;
    top_sel[lane] = -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < LFM_E) {
    uint rank = 0;
    for (uint j = 0; j < LFM_E; ++j) {
      const float q = lsel[j];
      rank += (q > sel || (q == sel && j < lane)) ? 1u : 0u;
    }
    if (rank <= RK) {
      top_id[rank] = lane;
      top_s[rank] = s;
      top_sel[rank] = sel;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (write_row && lane == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < RK; ++k) sum = sum + top_s[k];   // ((s0 + s1) + s2) + s3, rank order
    const float d = rf(rf(sum) + 1e-6f);
    for (uint k = 0; k < RK; ++k) {
      r[R_IDS + k] = top_id[k];
      r[R_W + k] = as_uint(rf(top_s[k] / d));
    }
    for (uint k = 0; k <= RK; ++k) r[R_SEL + k] = as_uint(top_sel[k]);
    r[R_SUM] = as_uint(d);
    for (uint k = R_SUM + 1; k < RW; ++k) r[k] = 0u;
  }
}
#endif

#ifdef ROUTE_E
#if ROUTE_E != 64
#error "lfm_route: 64 experts on 64 lanes"
#endif
__attribute__((reqd_work_group_size(ROUTE_E, 1, 1)))
__kernel void lfm_route(__global const float* restrict logits, __global const float* restrict bias,
                        __global uint* restrict route) {
  const uint m = get_group_id(1);
  __local float lsel[ROUTE_E];
  __local uint top_id[RK + 1];
  __local float top_s[RK + 1], top_sel[RK + 1];
  lfm_route_row(logits + (size_t)m * ROUTE_E, bias, get_local_id(0), lsel, top_id, top_s, top_sel,
                route + (size_t)m * RW, true);
}
#endif  // ROUTE_E

// ---------------------------------------------------------------------------------------------------------------
// The MoE block: TOP_K = 4 slots, the routed experts' layout-1 blocks by the route row's ids (rank order).
#ifdef MOE_E
#if !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS)
#error "lfm_moe: MOE_K, HIDDEN, INTER, UP_KS and DN_KS must be defined"
#endif
#if MOE_K != RK || MOE_E != LFM_E
#error "lfm_moe: the route row holds 4 slots of 64 experts"
#endif
#define G_UP (HIDDEN / GROUP)
#define NT_UP (2 * INTER / SG)
#define GU_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32)
#define G_DN (INTER / GROUP)
#define NT_DN (HIDDEN / SG)
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32)
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * MOE_K * DN_KS)
#include "lfm_expert.h"
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0 || HIDDEN % SG != 0
#error "lfm_moe: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "lfm_moe: UP_KS must be a power of two dividing HIDDEN / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "lfm_moe: DN_KS must be a power of two dividing INTER / 64"
#endif
#if defined(ROUTE_FUSED) && WG_UP < MOE_E
#error "lfm_moe _R: the gate||up work-group must hold a lane per expert"
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
inline float tile_dot(const __global uint* restrict tile, const __global ushort* restrict xr,
                      uint g0, uint g1, uint lane) {
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

// gate||up: work-group (slot, blk) owns n-tiles 4 blk .. 4 blk + 3 of its slot's expert (intermediate columns
// [32 blk, 32 blk + 32): tiles 0 / 2 gate, 1 / 3 the matching up, common::cols_interleave16); sub-group
// (tile_idx, q): one 16-column tile, K slice q, merged by the pairwise tree:
//   h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
LFM_SG16
__kernel void lfm_moe_gate_up(
#ifdef ROUTE_FUSED
                              __global const float* restrict logits, __global const float* restrict bias,
#endif
                              __global uint* restrict route, __global const ushort* restrict x,
                              __global const uint* restrict w, __global ushort* restrict h) {
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
  // The arm `route`: lfm_route's function on this work-group's first 64 lanes; work-group 0 writes the row.
  __local float lsel[MOE_E];
  __local uint top_id[RK + 1];
  __local float top_s[RK + 1], top_sel[RK + 1];
  lfm_route_row(logits + (size_t)m * MOE_E, bias, lid, lsel, top_id, top_s, top_sel, route + (size_t)m * RW,
                get_group_id(0) == 0);
  uint e = top_id[slot];
#else
  uint e = route[(size_t)m * RW + R_IDS + slot];
#endif
  e = min(e, (uint)(MOE_E - 1));   // never past the allocation, whatever the row holds
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = LFM_EXPERT_GU(w, e) + ((size_t)n_tile * G_UP + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, x + (size_t)m * HIDDEN, g0, g0 + G_UP / UP_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0 && (tile_idx & 1u) == 0) {
    const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
    const ushort g_b = rne_bf16(red[sg][lane]);
    const ushort u_b = rne_bf16(red[sg + UP_KS][lane]);
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
    h[((size_t)m * MOE_K + slot) * INTER + i] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}

// down + the combine + the residual fold, one 16-column n-tile of the hidden row per work-group; sub-group
// (slot, q) reads slot `slot`'s activation row and its expert's down block, K slice q; then sub-group 0:
//   t_k = rne(f32(rne(down_k)) · w_k);  y = rne(((t0 + t1) + t2) + t3) (fp32, rank order);  h_res = rne(h_res + y)
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
LFM_SG16
__kernel void lfm_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                           __global const uint* restrict w, __global ushort* restrict h_res) {
  __local float red[MOE_K * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const __global uint* restrict rr = route + (size_t)m * RW;
  uint e = rr[R_IDS + slot];
  e = min(e, (uint)(MOE_E - 1));
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile = LFM_EXPERT_DN(w, e) + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, h + ((size_t)m * MOE_K + slot) * INTER, g0, g0 + G_DN / DN_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float acc = bf16f(rne_bf16(bf16f(rne_bf16(red[0][lane])) * as_float(rr[R_W])));
    for (uint k = 1; k < MOE_K; ++k) {
      const float t = bf16f(rne_bf16(bf16f(rne_bf16(red[k * DN_KS][lane])) * as_float(rr[R_W + k])));
      acc = acc + t;
    }
    __global ushort* restrict rp = h_res + (size_t)m * HIDDEN + n_tile * SG + lane;
    *rp = rne_bf16(bf16f(*rp) + bf16f(rne_bf16(acc)));
  }
}
#endif  // MOE_E
