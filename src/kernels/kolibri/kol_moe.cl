// kol_moe.cl - spec 20c: Kolibri-1's routed experts - the 384-expert router and the MoE block (6 of
// 384 + the bf16 shared expert). Two families, each compiled only when its defines are given:
//
//   kol_route(logits, bias, route)                    ROUTE_E   grid (1, M), WG ROUTE_WG (256)
//   kol_moe_gate_up(route, x, w_gu, w_sh_gu, h)       MOE_E     grid (SLOTS x NT_UP / 4, M), WG 64 x UP_KS
//   kol_moe_down(route, h, w_dn, w_sh_dn, mo)         MOE_E     grid (HIDDEN / 16, M), WG 16 x SLOTS x DN_KS
//
// spec 15c's moe.cl / spec 18b's k2_moe.cl carried to Kolibri's semantics (tools/oracle/kolibri_ref.py,
// spec 20a; docs/probe-kolibri-2026-10-05.md "The router, exactly") - the layout-1 GEMV inner loop
// (tile_dot) and the fixed-order, atomic-free reductions are theirs; what differs is why this is a new
// source and moe.cl / k2_moe.cl are untouched:
//
//   * the router: logits fp32 (the bf16 GEMV's fp32 output, NEVER rounded to bf16 - GateLinear's
//       out_dtype fp32); sel = logit + bias (fp32, the bias widened) selects the top 6, ties to the
//       lower id; the weights are sigmoid(logit) of the selected - not of sel, not renormalised. 384
//       experts on 512 slots (the GEMV's zero rows) on 256 lanes, lane l owning experts l and l + 256:
//       a slot >= ROUTE_E is never ranked (sel -INF), whatever the real scores are (Review Focus 3).
//   * the slot order is ASCENDING EXPERT ID, and the combine is fp32: acc = 0; for j ascending:
//       acc = acc + f32(rne(down_j)) · w_j (the product rounded, THEN the add - no fma: the reference's
//       index_add_ of y.float() * w), + f32(rne(shared down)), ONE rounding to bf16.
//   * the shared expert is UNGATED and bf16 (spec 20 decision 4's proposal): slot SLOTS - 1 reads
//       gemv_bf16's tiles ([n_tile][k_octet][8 k][16 n], gate||up interleave16) instead of a layout-1
//       block - one launch serves the 6 routed and the shared expert.
//   * the block's output `mo` is NOT folded into the residual: post_ffn_norm comes first (the
//       sandwich; prep_res_fold SP0 then kol_post_add, kol_prep.cl) - k2_moe_down folds, this does not.
//
// The route row (u32 words per (layer, token); host mirror kernels::kolibri::route):
//   [R_IDS + j]   expert id of slot j, ASCENDING, j < TOP_K              (u32)
//   [R_W + j]     sigmoid(logit) of slot j, fp32 (never rounded)        (fp32 bits)
//   [R_SEL + j]   sel of slot j (logit + bias)                           (fp32 bits, diagnostics)
//   [R_NEXT]      sel of rank TOP_K - the first expert not taken        (the near-tie margin)
//   [R_LMIN]      the smallest logit among the selected                  (fp32 bits, diagnostics)
//
// **Portability**: sub-group block reads under cl_intel_subgroups, the plain loads they equal
// otherwise, every cross-lane step through SLM - so the file builds as OpenCL 1.2 for the indicative
// Mac run (tools/mac/clrun/kolibri_run.cc).
#ifndef M
#define M 1
#endif
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 32                       /* route row words (kernels::kolibri::route::kWords) */
#define R_IDS 0
#define R_W 8
#define R_SEL 16
#define R_NEXT 24
#define R_LMIN 25

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define KOL_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define KOL_SG16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// moe.cl's dot8 / tile_dot, verbatim.
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
// The shared slot: one gemv_bf16 tile column over k-octets [o0, o1): `tile` is the n_tile's first
// octet; lane `lane` takes column lane, a += w · x per k ascending (gemv_bf16's inner loop).
inline float bf16_dot(const __global ushort* restrict tile, const __global ushort* restrict xr,
                      uint o0, uint o1, uint lane) {
  float a = 0.f;
  for (uint o = o0; o < o1; ++o) {
#ifdef cl_intel_subgroups
    (void)lane;
    const ushort8 wv = intel_sub_group_block_read_us8(tile + (size_t)o * 128);
#else
    const __global ushort* t = tile + (size_t)o * 128;
    const ushort8 wv = (ushort8)(t[0 * SG + lane], t[1 * SG + lane], t[2 * SG + lane], t[3 * SG + lane],
                                 t[4 * SG + lane], t[5 * SG + lane], t[6 * SG + lane], t[7 * SG + lane]);
#endif
    const ushort8 xv = vload8(0, xr + o * 8);
    a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
    a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
    a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
    a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
  }
  return a;
}

// ---------------------------------------------------------------------------------------
// kol_route - one work-group per token, ROUTE_EPL experts per lane (lane l: l, l + WG, ...).
//   sel_e = logits[m][e] + bias[e]            e < ROUTE_E;  e >= ROUTE_E: -INF, never ranked
//   rank  = #{j < ROUTE_E : sel_j > sel_e or (sel_j == sel_e and j < e)}
//   lane 0: the TOP_K ids ascending; w_j = sigmoid(logits[m][id_j]) fp32; sel_j; next = sel of
//           rank TOP_K; the smallest selected logit
#ifdef ROUTE_E
#if !defined(ROUTE_K) || !defined(ROUTE_WG) || !defined(ROUTE_EPL) || !defined(ROUTE_LN)
#error "kol_route: ROUTE_K, ROUTE_WG, ROUTE_EPL and ROUTE_LN must be defined"
#endif
#if ROUTE_K < 1 || ROUTE_K > 8 || ROUTE_E < ROUTE_K + 1 || ROUTE_E > ROUTE_WG * ROUTE_EPL || ROUTE_LN != ROUTE_WG * ROUTE_EPL
#error "kol_route: 1 <= TOP_K <= 8, TOP_K < E <= the lanes' slots == the logits row (router_n)"
#endif
#define ROUTE_SLOTS (ROUTE_WG * ROUTE_EPL)
__attribute__((reqd_work_group_size(ROUTE_WG, 1, 1)))
__kernel void kol_route(__global const float* restrict logits, __global const float* restrict bias,
                        __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint lane = get_local_id(0);
  __local float lsel[ROUTE_SLOTS];
  __local uint top_id[ROUTE_K + 1];
  __local float top_sel[ROUTE_K + 1];
  const __global float* restrict lr = logits + (size_t)m * ROUTE_LN;
  for (uint q = 0; q < ROUTE_EPL; ++q) {
    const uint e = lane + q * ROUTE_WG;
    lsel[e] = e < ROUTE_E ? lr[e] + bias[e] : -INFINITY;
  }
  if (lane <= ROUTE_K) {   // a defined row even if NaNs break the ranking
    top_id[lane] = lane;
    top_sel[lane] = -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint q = 0; q < ROUTE_EPL; ++q) {
    const uint e = lane + q * ROUTE_WG;
    if (e >= ROUTE_E) continue;
    const float sel = lsel[e];
    uint rank = 0;
    for (uint j = 0; j < ROUTE_E; ++j) {
      const float s = lsel[j];
      rank += (s > sel || (s == sel && j < e)) ? 1u : 0u;
    }
    if (rank <= ROUTE_K) {
      top_id[rank] = e;
      top_sel[rank] = sel;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0) {
    __global uint* restrict r = route + (size_t)m * RW;
    float lmin = INFINITY;
    // The TOP_K ranks re-ordered by expert id: slot j's rank is the one with exactly j smaller ids.
    for (uint j = 0; j < 8; ++j) {
      uint id = 0u, wb = 0u, sb = 0u;
      if (j < ROUTE_K) {
        for (uint k = 0; k < ROUTE_K; ++k) {
          uint smaller = 0;
          for (uint k2 = 0; k2 < ROUTE_K; ++k2) smaller += top_id[k2] < top_id[k] ? 1u : 0u;
          if (smaller == j) {
            id = top_id[k];
            const float l = lr[id];
            wb = as_uint(sigmoid_f32(l));
            sb = as_uint(top_sel[k]);
            lmin = fmin(lmin, l);
          }
        }
      }
      r[R_IDS + j] = id;
      r[R_W + j] = wb;
      r[R_SEL + j] = sb;
    }
    r[R_NEXT] = as_uint(top_sel[ROUTE_K]);
    r[R_LMIN] = as_uint(lmin);
    for (uint k = R_LMIN + 1; k < RW; ++k) r[k] = 0u;
  }
}
#endif  // ROUTE_E

// ---------------------------------------------------------------------------------------
// The MoE block: SLOTS = TOP_K + 1 slots - the routed experts' layout-1 blocks (by the route row's
// ids) and the shared expert's bf16 tiles (the last slot).
#ifdef MOE_E
#if !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS)
#error "kol_moe: MOE_K, HIDDEN, INTER, UP_KS and DN_KS must be defined"
#endif
#if MOE_K < 1 || MOE_K > 8
#error "kol_moe: MOE_K is 1..8 (the route row holds 8 slots)"
#endif
#define SLOTS (MOE_K + 1)
#define SHARED_SLOT MOE_K
#define G_UP (HIDDEN / GROUP)
#define NT_UP (2 * INTER / SG)
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32)
#define G_DN (INTER / GROUP)
#define NT_DN (HIDDEN / SG)
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32)
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * SLOTS * DN_KS)
#define O_UP (HIDDEN / 8)          /* the shared gate||up tiles' k-octets */
#define O_DN (INTER / 8)
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0 || HIDDEN % SG != 0
#error "kol_moe: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "kol_moe: UP_KS must be a power of two dividing HIDDEN / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "kol_moe: DN_KS must be a power of two dividing INTER / 64"
#endif

// gate||up: work-group (slot, blk) owns n-tiles 4 blk .. 4 blk + 3 of its slot (intermediate columns
// [32 blk, 32 blk + 32)); sub-group (tile_idx, q): one 16-column tile, K slice q:
//   h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
KOL_SG16
__kernel void kol_moe_gate_up(__global const uint* restrict route, __global const ushort* restrict x,
                              __global const uint* restrict w, __global const ushort* restrict w_sh,
                              __global ushort* restrict h) {
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
  const __global ushort* restrict xr = x + (size_t)m * HIDDEN;
  float acc;
  if (slot < MOE_K) {   // uniform across the work-group
    uint e = route[(size_t)m * RW + R_IDS + slot];
    e = min(e, (uint)(MOE_E - 1));   // never past the allocation, whatever the row holds
    const uint g0 = q * (G_UP / UP_KS);
    const __global uint* restrict tile = w + (size_t)e * UP_BLK_U32 + ((size_t)n_tile * G_UP + g0) * TILE_U32;
    acc = tile_dot(tile, xr, g0, g0 + G_UP / UP_KS, lane);
  } else {
    const uint o0 = q * (O_UP / UP_KS);
    acc = bf16_dot(w_sh + (size_t)n_tile * O_UP * 128, xr, o0, o0 + O_UP / UP_KS, lane);
  }
  red[sg][lane] = acc;
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
    h[((size_t)m * SLOTS + slot) * INTER + i] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}

// down + the combine, one 16-column n-tile of the hidden row per work-group; sub-group (slot, q):
//   acc = 0;  for j = 0 .. TOP_K-1 (ascending id):  acc = acc + f32(rne(down_j)) · w_j   (no fma)
//   mo[m][n] = rne(acc + f32(rne(down_shared)))                       NOT folded into the residual
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
KOL_SG16
__kernel void kol_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                           __global const uint* restrict w, __global const ushort* restrict w_sh,
                           __global ushort* restrict mo) {
  __local float red[SLOTS * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const __global uint* restrict rr = route + (size_t)m * RW;
  const __global ushort* restrict hr = h + ((size_t)m * SLOTS + slot) * INTER;
  float acc;
  if (slot < MOE_K) {   // uniform across the sub-group
    uint e = rr[R_IDS + slot];
    e = min(e, (uint)(MOE_E - 1));
    const uint g0 = q * (G_DN / DN_KS);
    const __global uint* restrict tile = w + (size_t)e * DN_BLK_U32 + ((size_t)n_tile * G_DN + g0) * TILE_U32;
    acc = tile_dot(tile, hr, g0, g0 + G_DN / DN_KS, lane);
  } else {
    const uint o0 = q * (O_DN / DN_KS);
    acc = bf16_dot(w_sh + (size_t)n_tile * O_DN * 128, hr, o0, o0 + O_DN / DN_KS, lane);
  }
  red[sg][lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
#pragma OPENCL FP_CONTRACT OFF
    float sum = 0.0f;
    for (uint j = 0; j < MOE_K; ++j) {
      const float t = bf16f(rne_bf16(red[j * DN_KS][lane])) * as_float(rr[R_W + j]);
      sum = sum + t;
    }
    const float sh = bf16f(rne_bf16(red[SHARED_SLOT * DN_KS][lane]));
    mo[(size_t)m * HIDDEN + n_tile * SG + lane] = rne_bf16(sum + sh);
  }
}
#endif  // MOE_E
