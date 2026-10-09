// q4_moe.cl - spec 21c: Qwen3.8-Flash-Next's MoE block at 512 experts / top-10 / 640 (spec 21 §2.1, §4.4) - the
// router's softmax route and the 11-slot block (10 routed + the shared expert). Two families, each compiled only
// when its defines are given (kol_moe.cl's arrangement; moe.cl's semantics - Qwen3.5-MoE's softmax router - at
// widths moe.cl refuses: TOP_K <= 8, EXPERTS <= 256):
//
//   q4_route(logits, route)                        ROUTE_E   grid (1, M),                  WG 256 (2 experts a lane)
//   q4_moe_gate_up(route, x, w_gu, w_sh_gu, h)     MOE_E     grid (11 x NT_UP / 4, M),     WG 64 x UP_KS
//   q4_moe_down(route, h, w_dn, w_sh_dn, y)        MOE_E     grid (HIDDEN / 16, M),        WG 16 x 11 x DN_KS
//
// **The router** (M:961-970, Qwen4ExpTextTopKRouter; the router GEMV's fp32 row [528]: rows 0..511 the router,
// 512 the shared expert's gate, 513..527 zero): l = rne(logit) (F.linear's bf16 output); p = softmax(l) in fp32 -
// mx exact, ex = exp_torch(l - mx), the sum a pairwise tree over the 512 (stride 256 .. 1), p = ex / sum; the top 10
// by RANK = #{j : p_j > p_e or (p_j == p_e and j < e)} - EXACT ties to the lower id (torch's topk does not
// prefer the lower index: 21a's finding; those rows are undetermined for the routing gate); renormalised in fp32
// (s10 = sum_{k < 10} p_k, rank order) and rounded: w_k = rne(p_k / s10); the shared gate sg = rne(sigmoid(l_512)).
// The route row (u32 words per (layer, row); host mirror kernels::qwen4exp::route):
//   [R_IDS + k]  the expert of rank k, k < 10 - RANK order: the combine's order (grouped_mm sums the 10 weighted
//                rows in the router's topk slot order, 21a: spec 21 §12)
//   [R_W + k]    w_k (fp32 bits of a bf16 value)    [R_SG] sg (fp32 bits, bf16 value)
//   [R_P10] / [R_P11]   p of rank 9 and of rank 10 (the first not taken): the near-tie diagnostic
// **The block** (integrations/moe.py grouped_mm_experts_forward + M:981-992; tests/kernels/qwen4exp_ref.h):
//   slot k < 10: expert ids[k]'s layout-1 blocks; slot 10: the shared expert - int4 layout-1 blocks (ours, _SH4)
//   or bf16 gemv_bf16 tiles (Intel's interim checkpoint, _SHB: {2560, 1280} gate||up interleave16, {640, 2560})
//   h[m][slot][i] = rne(f32(rne(silu(f32(rne(sum gate))))) x f32(rne(sum up)))
//   y[m][n] = rne(f32(r_b) + f32(sh_b)):  r_b = rne(sum_k f32(rne(f32(rne(down_k)) x w_k))) in fp32, k ascending
//            (rank order, no fma: FP_CONTRACT OFF); sh_b = rne(f32(rne(down_10)) x sg)       NOT folded into H
//            (the next q4_hc_combine_norm folds it, x the block's inject weights)
//
// **The expert address** - the ONE place a routed expert's weights are found from its id (spec 15 decision 3):
// Q4_EXPERT_GU / Q4_EXPERT_DN below. Spec 22 (the expert-offload tier) replaces exactly these two lines with its
// indirection table, without touching the list or anything else here. (The plan's src/kernels/qwen4exp/q4_expert.h:
// no kernel source in this tree #includes - ocloc's include resolution is unproven blind - so the macro lives here.)
//
// **Portability**: sub-group block reads under cl_intel_subgroups, the plain loads they equal otherwise, every
// cross-lane step through SLM - the file builds as OpenCL 1.2 for the indicative Mac run.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 32                       /* route row words (kernels::qwen4exp::route::kWords) */
#define R_IDS 0
#define R_W 16
#define R_SG 26
#define R_P10 27
#define R_P11 28

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define Q4_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define Q4_SG16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
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
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

// ---------------------------------------------------------------------------------------------------------------
#ifdef ROUTE_E
#if !defined(ROUTE_K) || !defined(ROUTE_WG) || !defined(ROUTE_LN)
#error "q4_route: ROUTE_K, ROUTE_WG and ROUTE_LN must be defined"
#endif
#if ROUTE_K < 1 || ROUTE_K > 15 || ROUTE_E != 2 * ROUTE_WG || ROUTE_LN < ROUTE_E + 1
#error "q4_route: 1 <= TOP_K <= 15 (the route row's ids), two experts a lane, the row holds the shared gate"
#endif
__attribute__((reqd_work_group_size(ROUTE_WG, 1, 1)))
__kernel void q4_route(__global const float* restrict logits, __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint lane = get_local_id(0);
  __local float sl[ROUTE_E];     // the bf16-rounded logits, then the probabilities
  __local float red[ROUTE_E];    // the max, then the softmax sum tree
  __local uint top_id[ROUTE_K + 1];
  __local float top_p[ROUTE_K + 1];
  const __global float* restrict lr = logits + (size_t)m * ROUTE_LN;
  for (uint q = 0; q < 2; ++q) {
    const uint e = lane + q * ROUTE_WG;
    sl[e] = rf(lr[e]);
    red[e] = sl[e];
  }
  if (lane <= ROUTE_K) {   // a defined row even if NaNs break the ranking
    top_id[lane] = lane;
    top_p[lane] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = ROUTE_E / 2; stride > 0; stride >>= 1) {   // the max: exact in any order
    for (uint e = lane; e < stride; e += ROUTE_WG) red[e] = fmax(red[e], red[e + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float ex[2];
  for (uint q = 0; q < 2; ++q) {
    const uint e = lane + q * ROUTE_WG;
    ex[q] = exp_torch(sl[e] - mx);
    red[e] = ex[q];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = ROUTE_E / 2; stride > 0; stride >>= 1) {   // the sum: the pairwise tree, stride 256 .. 1
    for (uint e = lane; e < stride; e += ROUTE_WG) red[e] += red[e + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float sum = red[0];
  for (uint q = 0; q < 2; ++q) sl[lane + q * ROUTE_WG] = ex[q] / sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint q = 0; q < 2; ++q) {
    const uint e = lane + q * ROUTE_WG;
    const float pe = sl[e];
    uint rank = 0;
    for (uint jj = 0; jj < ROUTE_E; ++jj) {
      const float pj = sl[jj];
      rank += (pj > pe || (pj == pe && jj < e)) ? 1u : 0u;
    }
    if (rank <= ROUTE_K) {
      top_id[rank] = e;
      top_p[rank] = pe;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0) {
    __global uint* restrict r = route + (size_t)m * RW;
    float s10 = 0.0f;
    for (uint k = 0; k < ROUTE_K; ++k) s10 += top_p[k];
    for (uint k = 0; k < RW; ++k) r[k] = 0u;
    for (uint k = 0; k < ROUTE_K; ++k) {
      r[R_IDS + k] = top_id[k];
      r[R_W + k] = as_uint(rf(top_p[k] / s10));
    }
    r[R_SG] = as_uint(rf(sigmoid_t(rf(lr[ROUTE_E]))));
    r[R_P10] = as_uint(top_p[ROUTE_K - 1]);
    r[R_P11] = as_uint(top_p[ROUTE_K]);
  }
}
#endif  // ROUTE_E

// ---------------------------------------------------------------------------------------------------------------
#ifdef MOE_E
#if !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS) || !defined(SHARED_BF16)
#error "q4_moe: MOE_K, HIDDEN, INTER, UP_KS, DN_KS and SHARED_BF16 must be defined"
#endif
#if MOE_K < 1 || MOE_K > 15
#error "q4_moe: MOE_K is 1..15 (the route row's ids)"
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
#define O_UP (HIDDEN / 8)          /* the bf16 shared gate||up tiles' k-octets */
#define O_DN (INTER / 8)
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0 || HIDDEN % SG != 0
#error "q4_moe: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "q4_moe: UP_KS must be a power of two dividing HIDDEN / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "q4_moe: DN_KS must be a power of two dividing INTER / 64"
#endif

// ---- the expert address: spec 22 replaces exactly these two lines (its indirection table) -----------------------
#define Q4_EXPERT_GU(base, id) ((base) + (size_t)(id) * UP_BLK_U32)
#define Q4_EXPERT_DN(base, id) ((base) + (size_t)(id) * DN_BLK_U32)
// --------------------------------------------------------------------------------------------------------------

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
#if SHARED_BF16
// The bf16 shared slot: one gemv_bf16 tile column over k-octets [o0, o1) (kol_moe.cl's bf16_dot).
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
#define SH_T __global const ushort*
#else
#define SH_T __global const uint*
#endif

// gate||up: work-group (slot, blk) owns n-tiles 4 blk .. 4 blk + 3 of its slot (intermediate columns [32 blk,
// 32 blk + 32)); sub-group (tile_idx, q): one 16-column tile, K slice q, merged by the pairwise tree.
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
Q4_SG16
__kernel void q4_moe_gate_up(__global const uint* restrict route, __global const ushort* restrict x,
                             __global const uint* restrict w, SH_T restrict w_sh, __global ushort* restrict h) {
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
    const __global uint* restrict tile = Q4_EXPERT_GU(w, e) + ((size_t)n_tile * G_UP + g0) * TILE_U32;
    acc = tile_dot(tile, xr, g0, g0 + G_UP / UP_KS, lane);
  } else {
#if SHARED_BF16
    const uint o0 = q * (O_UP / UP_KS);
    acc = bf16_dot(w_sh + (size_t)n_tile * O_UP * 128, xr, o0, o0 + O_UP / UP_KS, lane);
#else
    const uint g0 = q * (G_UP / UP_KS);
    acc = tile_dot(w_sh + ((size_t)n_tile * G_UP + g0) * TILE_U32, xr, g0, g0 + G_UP / UP_KS, lane);
#endif
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
    h[((size_t)m * SLOTS + slot) * INTER + i] = rne_bf16(rf(silu_t(bf16f(g_b))) * bf16f(u_b));
  }
}

// down + the combine, one 16-column n-tile of the hidden row per work-group; sub-group (slot, q).
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
Q4_SG16
__kernel void q4_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                          __global const uint* restrict w, SH_T restrict w_sh, __global ushort* restrict y) {
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
    const __global uint* restrict tile = Q4_EXPERT_DN(w, e) + ((size_t)n_tile * G_DN + g0) * TILE_U32;
    acc = tile_dot(tile, hr, g0, g0 + G_DN / DN_KS, lane);
  } else {
#if SHARED_BF16
    const uint o0 = q * (O_DN / DN_KS);
    acc = bf16_dot(w_sh + (size_t)n_tile * O_DN * 128, hr, o0, o0 + O_DN / DN_KS, lane);
#else
    const uint g0 = q * (G_DN / DN_KS);
    acc = tile_dot(w_sh + ((size_t)n_tile * G_DN + g0) * TILE_U32, hr, g0, g0 + G_DN / DN_KS, lane);
#endif
  }
  red[sg][lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < MOE_K; ++k) {
      const float t = rf(rf(red[k * DN_KS][lane]) * as_float(rr[R_W + k]));
      sum = sum + t;
    }
    const ushort r_b = rne_bf16(sum);
    const ushort sh_b = rne_bf16(rf(red[SHARED_SLOT * DN_KS][lane]) * as_float(rr[R_SG]));
    y[(size_t)m * HIDDEN + n_tile * SG + lane] = rne_bf16(bf16f(r_b) + bf16f(sh_b));
  }
}
#endif  // MOE_E
