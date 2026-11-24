// kol_moe.cl - spec 20c: Kolibri-1's routed experts - the 384-expert router and the MoE block (6 of
// 384 + the int4 shared expert). Two families, each compiled only when its defines are given:
//
//   kol_route(logits, bias, route)                    ROUTE_E   grid (1, M), WG ROUTE_WG (256)
//   kol_moe_gate_up(route, x, w_gu, w_sh_gu, h)       MOE_E     grid (SLOTS x NT_UP / 4, M), WG 64 x UP_KS
//   kol_moe_down(route, h, w_dn, w_sh_dn, mo)         MOE_E     grid (HIDDEN / 16, M), WG 16 x SLOTS x DN_KS
//
// spec 15c's moe.cl carried to Kolibri's semantics (tools/oracle/kolibri_ref.py,
// spec 20a) - the layout-1 GEMV inner loop
// (tile_dot) and the fixed-order, atomic-free reductions are theirs; what differs is why this is a new
// source and moe.cl is untouched:
//
//   * the router: logits fp32 (the bf16 GEMV's fp32 output, NEVER rounded to bf16 - GateLinear's
//       out_dtype fp32); sel = logit + bias (fp32, the bias widened) selects the top 6, ties to the
//       lower id; the weights are sigmoid(logit) of the selected - not of sel, not renormalised. 384
//       experts on 512 slots (the GEMV's zero rows) on 256 lanes, lane l owning experts l and l + 256:
//       a slot >= ROUTE_E is never ranked (sel -INF), whatever the real scores are (Review Focus 3).
//   * the slot order is ASCENDING EXPERT ID, and the combine is fp32: acc = 0; for j ascending:
//       acc = acc + f32(rne(down_j)) · w_j (the product rounded, THEN the add - no fma: the reference's
//       index_add_ of y.float() * w), + f32(rne(shared down)), ONE rounding to bf16.
//   * the shared expert is UNGATED and int4 (our AutoRound export
//       quantises it as every routed expert, g64 sym GPTQ): slot SLOTS - 1 reads ITS OWN layout-1 block
//       (w_sh, the routed blocks' layout at the same widths - shared_inter == moe_inter) through the same
//       tile_dot - one launch serves the 6 routed and the shared expert.
//   * the block's output `mo` is NOT folded into the residual: post_ffn_norm comes first (the
//       sandwich; prep_res_fold SP0 then kol_post_add, kol_prep.cl).
//
// The route row (u32 words per (layer, token); host mirror kernels::kolibri::route):
//   [R_IDS + j]   expert id of slot j, ASCENDING, j < TOP_K              (u32)
//   [R_W + j]     sigmoid(logit) of slot j, fp32 (never rounded)        (fp32 bits)
//   [R_SEL + j]   sel of slot j (logit + bias)                           (fp32 bits, diagnostics)
//   [R_NEXT]      sel of rank TOP_K - the first expert not taken        (the near-tie margin)
//   [R_LMIN]      the smallest logit among the selected                  (fp32 bits, diagnostics)
//
// Sub-group block reads, every cross-lane step through SLM.
#ifndef M
#define M 1
#endif
#define SG 16
#define GROUP 64
#include "expert_fp.h"
#define RW 32                       /* route row words (kernels::kolibri::route::kWords) */
#define R_IDS 0
#define R_W 8
#define R_SEL 16
#define R_NEXT 24
#define R_LMIN 25

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define KOL_SG16 __attribute__((intel_reqd_sub_group_size(SG)))

#include "bf16.h"
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

#if WFMT != 0
#include "expert_fp_gemv.h"
#else
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
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
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
#endif

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
#if INK_ROUTE_M1
// All logits of one row fit in a work-group, so the selection can consume
// the tiled BF16 GEMV result directly from local memory.
__attribute__((reqd_work_group_size(ROUTE_WG, 1, 1)))
KOL_SG16
__kernel void kol_route(__global const ushort* restrict weights, __global const ushort* restrict x,
                        __global const float* restrict bias, __global uint* restrict route) {
  const uint m = 0, lane = get_local_id(0);
  const uint tile = get_sub_group_id();
  __local float lsel[ROUTE_SLOTS], lr[ROUTE_SLOTS];
  __local uint top_id[ROUTE_K + 1];
  __local float top_sel[ROUTE_K + 1];
  const __global ushort* w = weights + (size_t)tile * (2304 / 8) * 128;
  float acc = 0.f;
  for (uint k8 = 0; k8 < 2304 / 8; ++k8, w += 128) {
    const ushort8 wv = intel_sub_group_block_read_us8(w);
    const ushort8 xv = vload8(0, x + k8 * 8);
    acc += bf16f(wv.s0) * bf16f(xv.s0); acc += bf16f(wv.s1) * bf16f(xv.s1);
    acc += bf16f(wv.s2) * bf16f(xv.s2); acc += bf16f(wv.s3) * bf16f(xv.s3);
    acc += bf16f(wv.s4) * bf16f(xv.s4); acc += bf16f(wv.s5) * bf16f(xv.s5);
    acc += bf16f(wv.s6) * bf16f(xv.s6); acc += bf16f(wv.s7) * bf16f(xv.s7);
  }
  lr[lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
#else
__attribute__((reqd_work_group_size(ROUTE_WG, 1, 1)))
__kernel void kol_route(__global const float* restrict logits, __global const float* restrict bias,
                        __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint lane = get_local_id(0);
  __local float lsel[ROUTE_SLOTS];
  __local uint top_id[ROUTE_K + 1];
  __local float top_sel[ROUTE_K + 1];
  const __global float* restrict lr = logits + (size_t)m * ROUTE_LN;
#endif
  for (uint q = 0; q < ROUTE_EPL; ++q) {
    const uint e = lane + q * ROUTE_WG;
#if ROUTE_RENORM
    lsel[e] = e < ROUTE_E ? sigmoid_f32(lr[e]) + bias[e] : -INFINITY;
#else
    lsel[e] = e < ROUTE_E ? lr[e] + bias[e] : -INFINITY;
#endif
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
#if ROUTE_RENORM
    float sum = 0.0f;
    for (uint k = 0; k < ROUTE_K; ++k) sum += as_float(r[R_W + k]);
    for (uint k = 0; k < ROUTE_K; ++k)
      r[R_W + k] = as_uint((as_float(r[R_W + k]) / (sum + 1e-20f)) * ROUTE_SCALE);
#endif
    r[R_NEXT] = as_uint(top_sel[ROUTE_K]);
    r[R_LMIN] = as_uint(lmin);
    for (uint k = R_LMIN + 1; k < RW; ++k) r[k] = 0u;
  }
}
#endif  // ROUTE_E

// ---------------------------------------------------------------------------------------
// The MoE block: SLOTS = TOP_K + 1 slots - the routed experts' layout-1 blocks (by the route row's
// ids) and the shared expert's own layout-1 block (the last slot).
#ifdef MOE_E
#if !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS)
#error "kol_moe: MOE_K, HIDDEN, INTER, UP_KS and DN_KS must be defined"
#endif
#if MOE_K < 1 || MOE_K > 8
#error "kol_moe: MOE_K is 1..8 (the route row holds 8 slots)"
#endif
#ifndef MOE_SHARED
#define MOE_SHARED 1
#endif
#define SLOTS (MOE_K + MOE_SHARED)
#define SHARED_SLOT MOE_K
#define G_UP (HIDDEN / GROUP)
#define NT_UP (2 * INTER / SG)
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32 + EXPERT_OUTER_WORDS)
#define G_DN (INTER / GROUP)
#define NT_DN (HIDDEN / SG)
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32 + EXPERT_OUTER_WORDS)
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * SLOTS * DN_KS)
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
                              __global const uint* restrict w, __global const uint* restrict w_sh,
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
  const __global uint* restrict blk_w = w_sh;   // the shared slot's block; a routed slot's by its id
  if (slot < MOE_K) {   // uniform across the work-group
    uint e = route[(size_t)m * RW + R_IDS + slot];
    e = min(e, (uint)(MOE_E - 1));   // never past the allocation, whatever the row holds
    blk_w = w + (size_t)e * UP_BLK_U32;
  }
  const uint g0 = q * (G_UP / UP_KS);
  const float acc = tile_dot(blk_w + ((size_t)n_tile * G_UP + g0) * TILE_U32, xr, g0, g0 + G_UP / UP_KS, lane);
  red[sg][lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0 && (tile_idx & 1u) == 0) {
    const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
#if defined(SWIGLU_CAP_G) && defined(SWIGLU_CAP_U)
    float g = red[sg][lane], u = red[sg + UP_KS][lane];
#if WFMT == 1
    const float outer = as_float(blk_w[UP_BLK_U32 - 1]);
    g *= outer;
    u *= outer;
#endif
    const float a = tanh(g / SWIGLU_CAP_G) * SWIGLU_CAP_G;
    const float b = tanh(u / SWIGLU_CAP_U) * SWIGLU_CAP_U;
    h[((size_t)m * SLOTS + slot) * INTER + i] = rne_bf16(a * sigmoid_f32(g) * b);
#else
#if WFMT == 1
    const float outer = as_float(blk_w[UP_BLK_U32 - 1]);
    const ushort g_b = rne_bf16(red[sg][lane] * outer);
    const ushort u_b = rne_bf16(red[sg + UP_KS][lane] * outer);
#else
    const ushort g_b = rne_bf16(red[sg][lane]);
    const ushort u_b = rne_bf16(red[sg + UP_KS][lane]);
#endif
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
    h[((size_t)m * SLOTS + slot) * INTER + i] = rne_bf16(bf16f(s_b) * bf16f(u_b));
#endif
  }
}

// down + the combine, one 16-column n-tile of the hidden row per work-group; sub-group (slot, q):
//   acc = 0;  for j = 0 .. TOP_K-1 (ascending id):  acc = acc + f32(rne(down_j)) · w_j   (no fma)
//   mo[m][n] = rne(acc + f32(rne(down_shared)))                       NOT folded into the residual
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
KOL_SG16
__kernel void kol_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                           __global const uint* restrict w, __global const uint* restrict w_sh,
#if INK_MOE_POST
                           __global float* restrict mo, __global const float* history,
                           __global const float* taps, __global float* residual, __global ushort* draft_tap) {
#elif MOE_OUT_F32
                           __global float* restrict mo) {
#else
                           __global ushort* restrict mo) {
#endif
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
  const __global uint* restrict blk_w = w_sh;   // the shared slot's block; a routed slot's by its id
  if (slot < MOE_K) {   // uniform across the sub-group
    uint e = rr[R_IDS + slot];
    e = min(e, (uint)(MOE_E - 1));
    blk_w = w + (size_t)e * DN_BLK_U32;
  }
  const uint g0 = q * (G_DN / DN_KS);
  const float acc = tile_dot(blk_w + ((size_t)n_tile * G_DN + g0) * TILE_U32, hr, g0, g0 + G_DN / DN_KS, lane);
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
      float dot = red[j * DN_KS][lane];
#if WFMT == 1
      const uint e = min(rr[R_IDS + j], (uint)(MOE_E - 1));
      dot *= as_float(w[(size_t)(e + 1) * DN_BLK_U32 - 1]);
#endif
#if MOE_OUT_F32
      const float t = dot * as_float(rr[R_W + j]);
#else
      const float t = bf16f(rne_bf16(dot)) * as_float(rr[R_W + j]);
#endif
      sum = sum + t;
    }
#if MOE_SHARED
    float sh = red[SHARED_SLOT * DN_KS][lane];
#if WFMT == 1
    sh *= as_float(w_sh[DN_BLK_U32 - 1]);
#endif
    sum += bf16f(rne_bf16(sh));
#endif
#if INK_MOE_POST
    const uint i = n_tile * SG + lane;
    const __global float* filter = taps + i * 4;
    float conv = 0.f;
    for (uint t = 0; t < 3; ++t) conv += filter[t] * history[t * HIDDEN + i];
    conv += filter[3] * sum;
    const float value = (sum + conv) / sqrt(5.f) + residual[i];
    mo[i] = sum;
    residual[i] = value;
    if (draft_tap) draft_tap[i] = rne_bf16(value);
#elif MOE_OUT_F32
    mo[(size_t)m * HIDDEN + n_tile * SG + lane] = sum;
#else
    mo[(size_t)m * HIDDEN + n_tile * SG + lane] = rne_bf16(sum);
#endif
  }
}
#endif  // MOE_E

#if MOE_UNION
#if M < 2 || M > 8 || MOE_E != 128 || MOE_SHARED || !WFMT || !MOE_OUT_F32 || DN_KS != 1
#error "kol_moe union requires 2 to 8 rows and 128 routed floating-point experts"
#endif
#if HIDDEN % 128 != 0 || !defined(SWIGLU_CAP_G) || !defined(SWIGLU_CAP_U)
#error "kol_moe union finish requires a 128-wide hidden tile and softcapped SwiGLU"
#endif
// Compact the distinct experts. Each packed nibble is the slot of this
// expert in one row, or 15 if that row did not select it.
__attribute__((reqd_work_group_size(MOE_E, 1, 1)))
__kernel void kol_moe_union_plan(__global const uint* route, __global uint2* plan) {
  const uint e = get_local_id(0);
  uint slots = 0;
#pragma unroll
  for (uint r = 0; r < M; ++r) {
    uint slot = 15;
    for (uint k = 0; k < MOE_K; ++k)
      if (route[r * RW + R_IDS + k] == e) slot = k;
    slots |= slot << (4 * r);
  }
  __local uint index[MOE_E], total;
  const bool live = slots != (0xffffffffu >> (32 - 4 * M));
  index[e] = live;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    uint n = 0;
    for (uint i = 0; i < MOE_E; ++i) {
      const uint active = index[i];
      index[i] = n;
      n += active;
    }
    total = n;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (live) plan[index[e]] = (uint2)(e, slots);
  if (e >= total && e < M * MOE_K) plan[e] = (uint2)(MOE_E, 0);
}

__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
KOL_SG16
__kernel void kol_moe_union_gate_up(__global const uint2* plan, __global const ushort* x,
                                   __global const uint* w, __global ushort* h) {
  const uint2 item = plan[get_group_id(1)];
  if (item.x >= MOE_E) return;
  const uint lid = get_local_id(0), lane = lid % SG, sg = lid / SG;
  const uint blk = get_group_id(0), tile_idx = sg / UP_KS, q = sg % UP_KS;
  const uint n_tile = blk * 4 + tile_idx, g0 = q * (G_UP / UP_KS);
  const __global uint* expert = w + (size_t)item.x * UP_BLK_U32;
  float acc[M] = {0};
  tile_dot_rows(expert + ((size_t)n_tile * G_UP + g0) * TILE_U32, x, item.y,
                HIDDEN, 0, g0, g0 + G_UP / UP_KS, lane, acc);
  __local float red[M][4 * UP_KS][SG];
#pragma unroll
  for (uint r = 0; r < M; ++r) red[r][sg][lane] = acc[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride; stride >>= 1) {
    if (q < stride)
#pragma unroll
      for (uint r = 0; r < M; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q != 0 || (tile_idx & 1u)) return;
#if WFMT == 1
  const float outer = as_float(expert[UP_BLK_U32 - 1]);
#else
  const float outer = 1.f;
#endif
  const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
#pragma unroll
  for (uint r = 0; r < M; ++r) {
    const uint slot = (item.y >> (4 * r)) & 15u;
    if (slot >= MOE_K) continue;
    const float g = red[r][sg][lane] * outer, u = red[r][sg + UP_KS][lane] * outer;
    const float a = tanh(g / SWIGLU_CAP_G) * SWIGLU_CAP_G;
    const float b = tanh(u / SWIGLU_CAP_U) * SWIGLU_CAP_U;
    h[((size_t)r * MOE_K + slot) * INTER + i] = rne_bf16(a * sigmoid_f32(g) * b);
  }
}

__attribute__((reqd_work_group_size(4 * SG, 1, 1)))
KOL_SG16
__kernel void kol_moe_union_down(__global const uint2* plan, __global const ushort* h,
                                __global const uint* w, __global float* expert_out) {
  const uint2 item = plan[get_group_id(1)];
  if (item.x >= MOE_E) return;
  const uint n_tile = get_group_id(0) * 4 + get_sub_group_id(), lane = get_sub_group_local_id();
  const __global uint* expert = w + (size_t)item.x * DN_BLK_U32;
  float acc[M] = {0};
  tile_dot_rows(expert + (size_t)n_tile * G_DN * TILE_U32, h, item.y,
                MOE_K * INTER, INTER, 0, G_DN, lane, acc);
#if WFMT == 1
  const float outer = as_float(expert[DN_BLK_U32 - 1]);
#else
  const float outer = 1.f;
#endif
#pragma unroll
  for (uint r = 0; r < M; ++r) {
    const uint slot = (item.y >> (4 * r)) & 15u;
    if (slot < MOE_K) expert_out[((size_t)r * MOE_K + slot) * HIDDEN + n_tile * SG + lane] = acc[r] * outer;
  }
}

// Preserve the per-row expert sum in slot order, then fold the short
// convolution and residual. Each channel sees all its rows and its history.
#pragma OPENCL FP_CONTRACT OFF
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void kol_moe_union_finish(__global const uint* route, __global const float* expert_out,
                                  __global const float* history, __global const float* taps,
                                  __global float* pending, __global float* residual, __global ushort* draft_tap) {
  const uint i = get_global_id(0);
  float rows[M] = {0};
#pragma unroll
  for (uint r = 0; r < M; ++r) {
    for (uint j = 0; j < MOE_K; ++j)
      rows[r] += expert_out[((size_t)r * MOE_K + j) * HIDDEN + i] * as_float(route[r * RW + R_W + j]);
    pending[(size_t)r * HIDDEN + i] = rows[r];
    float sum = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
      const int t = (int)r + (int)tap - 3;
      const float x = t < 0 ? history[(t + 3) * HIDDEN + i] : rows[t];
      sum += taps[i * 4 + tap] * x;
    }
    const size_t at = (size_t)r * HIDDEN + i;
    const float value = residual[at] + (rows[r] + sum) / sqrt(5.f);
    residual[at] = value;
    if (draft_tap) draft_tap[(size_t)r * 4 * HIDDEN + i] = rne_bf16(value);
  }
}
#endif
