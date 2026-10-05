// k2_moe.cl - spec 18b: K2-Horizon's routed experts - its sigmoid router, the MoE block
// (100 experts top-8 + 1 shared) and MoVA's value experts (64, top-4). Three families,
// each compiled only when its defines are given:
//
//   k2_route(logits, bias, route)                ROUTE_E   grid (1, M), WG ROUTE_WG
//   k2_moe_gate_up(route, x, w, h)               MOE_E     grid (SLOTS x NT_UP / 4, M), WG 64 x UP_KS
//   k2_moe_down(route, h, w, resid)              MOE_E     grid (HIDDEN / 16, M), WG 16 x SLOTS x DN_KS
//   k2_mova_value(ctrl, route, x, w, kv_v)       MOVA_E    grid (MOVA_N / 16, M), WG 16 x MOVA_K x MOVA_KS
//
// These are spec 15c's moe.cl kernels (src/kernels/moe.cl) carried to K2's formula - the
// structure, the layout-1 GEMV inner loop (tile_dot, verbatim) and the fixed-order,
// atomic-free reductions are moe.cl's; what differs is spec 18 §3's semantics, which is
// why they are a new source and moe.cl (Ornith's binaries) is untouched:
//
//   * the router (modeling_k2_horizon.py:136-166 calc_router_weights, 560-579):
//       l_b  = rne(router GEMV)                     F.linear in bf16 (weight only)
//       s    = sigmoid(f32(l_b))                    fp32
//       sel  = s + f32(bias)                        the bias is used ONLY for selection
//       top  = the TOP_K largest sel                ties -> the lower expert id
//       w_k  = rne((s_k / Σ_k s_k) x 2.5)          normalised, THEN scaled, then bf16
//     The 100 routed experts sit on 128 lanes (the router GEMV's 128 rows, 28 of them zero
//     padding): a lane >= ROUTE_E has sel = -INF and is never ranked, so a padded row can
//     never be selected however negative the real scores are (Review Focus 4).
//   * the slot order is ASCENDING EXPERT ID, not rank: the reference accumulates with
//     index_add_ over `expert_hit` in ascending id (modeling_k2_horizon.py:186-192,
//     590-603) into a bf16 tensor, so every add rounds:
//       acc = 0;  for e ascending:  acc = rne(f32(acc) + f32(rne(f32(rne(expert_e)) · w_e)))
//     (moe.cl's Ornith combine sums in fp32 in rank order and rounds once - another chain.)
//   * the shared expert is UNGATED: MoE = rne(acc + f32(rne(shared down))) (line 607).
//   * MoVA: v = the same ascending bf16 chain over SiLU(V_e x) (combine_routed_experts with
//     activation=F.silu), written straight into the layer's V cache at pos.
//
// The route row (u32 words per (layer, token); the host mirror is kernels::k2::route,
// src/kernels/k2_kernels.h):
//   [R_IDS + j]   expert id of slot j, ASCENDING, j < TOP_K       (u32)
//   [R_W + j]     w of slot j, a bf16 value held in fp32           (fp32 bits)
//   [R_SEL + j]   sel of slot j (s + bias)                         (fp32 bits, diagnostics)
//   [R_NEXT]      sel of rank TOP_K - the first expert not taken  (the near-tie margin)
//   [R_SUM]       Σ s of the top-k, in rank order                  (fp32 bits)
//
// **Determinism.** No atomic; the rank is a count over SLM (a permutation, ties to the lower
// id); Σ s is summed by one lane in rank order; every K split merges by a fixed pairwise
// tree; the expert sums are in fixed slot order. Two replays give the same bits.
//
// **Portability** (moe.cl's): sub-group block reads under cl_intel_subgroups, the plain
// loads they equal otherwise, every cross-lane step through SLM - so the file builds as
// OpenCL 1.2 for the indicative Mac run (tools/mac/clrun/k2_run.cc).
#ifndef M
#define M 1
#endif
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define RW 32                       /* route row words (kernels::k2::route::kWords) */
#define R_IDS 0
#define R_W 8
#define R_SEL 16
#define R_NEXT 24
#define R_SUM 25

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define K2_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define K2_SG16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// moe.cl's dot8 and tile_dot, verbatim: 8 nibbles x 8 activations; one layout-1 tile's
// eight words for this lane; the group sum times the f16 scale into the running sum.
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

// ---------------------------------------------------------------------------------------
// k2_route - one work-group per token, one expert per lane (lane e owns expert e).
//
// `logits` is read as LS split-K slices of a [LS][M][LN] fp32 GEMV output from column LOFF:
// the MoE router's gemv_bf16 [M][128] (LN 128, LOFF 0, LS 1) or, for MoVA, the v_router
// columns of the fused attention GEMV's partials (LN 9280, LOFF 9216, LS = its S). `bias`
// is the layer's route block at the router's offset (fp32, loader/k2_layout.h).
//
//   l    = f32(rne(Σ_q logits[q][m][LOFF + e]))   ascending q    (e < ROUTE_E)
//   s    = sigmoid(l)
//   sel  = s + bias[e];   a lane e >= ROUTE_E: sel = -INF, never ranked
//   rank = #{j < ROUTE_E : sel_j > sel_e or (sel_j == sel_e and j < e)}
//   lane 0: Σ = Σ_{rank k < TOP_K} s_k (ascending k); slots = the TOP_K ids ascending;
//           w_j = rne((s_j / Σ) x ROUTE_SCALE)
#ifdef ROUTE_E
#if !defined(ROUTE_K) || !defined(ROUTE_WG) || !defined(ROUTE_LN) || !defined(ROUTE_LOFF) || !defined(ROUTE_LS)
#error "k2_route: ROUTE_K, ROUTE_WG, ROUTE_LN, ROUTE_LOFF and ROUTE_LS must be defined"
#endif
#ifndef ROUTE_SCALE
#error "k2_route: ROUTE_SCALE (router_scaling_factor, 2.5) must be defined"
#endif
#if ROUTE_K < 1 || ROUTE_K > 8 || ROUTE_E < ROUTE_K + 1 || ROUTE_E > ROUTE_WG
#error "k2_route: 1 <= TOP_K <= 8 (the route row), TOP_K < E <= the work-group"
#endif
#if (ROUTE_WG & (ROUTE_WG - 1)) != 0 || ROUTE_WG < 16 || ROUTE_WG > 256
#error "k2_route: ROUTE_WG is a power of two in [16, 256] (kernels::k2::route_wg)"
#endif
#if ROUTE_LOFF + ROUTE_E > ROUTE_LN
#error "k2_route: the router columns overrun the logits row"
#endif
__attribute__((reqd_work_group_size(ROUTE_WG, 1, 1)))
__kernel void k2_route(__global const float* restrict logits, __global const float* restrict bias,
                       __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint e = get_local_id(0);
  __local float lsel[ROUTE_WG];
  __local uint top_id[ROUTE_K + 1];
  __local float top_s[ROUTE_K + 1], top_sel[ROUTE_K + 1];

  const bool live = e < ROUTE_E;
  float s = 0.0f, sel = -INFINITY;
  if (live) {
    float acc = 0.0f;
    for (uint q = 0; q < ROUTE_LS; ++q) acc += logits[((size_t)q * M + m) * ROUTE_LN + ROUTE_LOFF + e];
    s = sigmoid_f32(bf16f(rne_bf16(acc)));
    sel = s + bias[e];
  }
  lsel[e] = sel;
  if (e <= ROUTE_K) {   // a defined row even if NaNs break the ranking
    top_id[e] = e;
    top_s[e] = 0.0f;
    top_sel[e] = -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (live) {
    uint rank = 0;
    for (uint j = 0; j < ROUTE_E; ++j) {
      const float q = lsel[j];
      rank += (q > sel || (q == sel && j < e)) ? 1u : 0u;
    }
    if (rank <= ROUTE_K) {
      top_id[rank] = e;
      top_s[rank] = s;
      top_sel[rank] = sel;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    __global uint* restrict r = route + (size_t)m * RW;
    float sum = 0.0f;
    for (uint k = 0; k < ROUTE_K; ++k) sum += top_s[k];   // rank order (torch.gather's)
    // The TOP_K ranks re-ordered by expert id: slot j's rank is the one with exactly j
    // smaller ids among the top (ids are distinct, so this is a permutation).
    for (uint j = 0; j < 8; ++j) {
      uint id = 0u, wb = 0u, sb = 0u;
      if (j < ROUTE_K) {
        for (uint k = 0; k < ROUTE_K; ++k) {
          uint smaller = 0;
          for (uint k2 = 0; k2 < ROUTE_K; ++k2) smaller += top_id[k2] < top_id[k] ? 1u : 0u;
          if (smaller == j) {
            const float t = top_s[k] / sum;
            const float w = t * ROUTE_SCALE;
            id = top_id[k];
            wb = as_uint(bf16f(rne_bf16(w)));
            sb = as_uint(top_sel[k]);
          }
        }
      }
      r[R_IDS + j] = id;
      r[R_W + j] = wb;
      r[R_SEL + j] = sb;
    }
    r[R_NEXT] = as_uint(top_sel[ROUTE_K]);
    r[R_SUM] = as_uint(sum);
    for (uint k = R_SUM + 1; k < RW; ++k) r[k] = 0u;
  }
}
#endif  // ROUTE_E

// ---------------------------------------------------------------------------------------
// The MoE block: gate||up of the TOP_K routed experts and the shared expert (slot TOP_K,
// block SHARED = MOE_E), SiLU x up fused; then down of all SLOTS slots, the ascending-id
// bf16 combine, the ungated shared expert, the residual fold. moe.cl's two kernels' grids,
// work-groups and K-split trees; the combine is K2's.
#ifdef MOE_E
#if !defined(MOE_K) || !defined(HIDDEN) || !defined(INTER) || !defined(UP_KS) || !defined(DN_KS)
#error "k2_moe: MOE_K, HIDDEN, INTER, UP_KS and DN_KS must be defined"
#endif
#if MOE_K < 1 || MOE_K > 8
#error "k2_moe: MOE_K is 1..8 (the route row holds 8 slots)"
#endif
#define SLOTS (MOE_K + 1)
#define SHARED MOE_E
#define G_UP (HIDDEN / GROUP)
#define NT_UP (2 * INTER / SG)
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32)
#define G_DN (INTER / GROUP)
#define NT_DN (HIDDEN / SG)
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32)
#define WG_UP (4 * SG * UP_KS)
#define WG_DN (SG * SLOTS * DN_KS)
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0 || HIDDEN % SG != 0
#error "k2_moe: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "k2_moe: UP_KS must be a power of two dividing HIDDEN / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "k2_moe: DN_KS must be a power of two dividing INTER / 64"
#endif

// gate||up: work-group (slot, blk) owns n-tiles 4 blk .. 4 blk + 3 of its slot's block
// (intermediate columns [32 blk, 32 blk + 32)); moe_gate_up's chain:
//   h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
K2_SG16
__kernel void k2_moe_gate_up(__global const uint* restrict route, __global const ushort* restrict x,
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

  uint e = slot < MOE_K ? route[(size_t)m * RW + R_IDS + slot] : (uint)SHARED;
  e = min(e, (uint)SHARED);   // never past the allocation, whatever the row holds
  const uint n_tile = blk * 4 + tile_idx;
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = w + (size_t)e * UP_BLK_U32 + ((size_t)n_tile * G_UP + g0) * TILE_U32;
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
    h[((size_t)m * SLOTS + slot) * INTER + i] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}

// down + combine + residual, one 16-column n-tile of the hidden row per work-group:
//   acc = 0;  for j = 0 .. TOP_K-1 (ascending expert id):
//             acc = f32(rne(acc + f32(rne(f32(rne(down_j)) · w_j))))
//   sh_b = rne(down_shared)                                the ungated shared expert
//   o_b  = rne(acc + f32(sh_b))                            routed + shared (line 607)
//   resid[m][n] = rne(f32(resid[m][n]) + f32(o_b))         the decoder's residual add
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
K2_SG16
__kernel void k2_moe_down(__global const uint* restrict route, __global const ushort* restrict h,
                          __global const uint* restrict w, __global ushort* restrict resid) {
  __local float red[SLOTS * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const __global uint* restrict rr = route + (size_t)m * RW;

  uint e = slot < MOE_K ? rr[R_IDS + slot] : (uint)SHARED;
  e = min(e, (uint)SHARED);
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile = w + (size_t)e * DN_BLK_U32 + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, h + ((size_t)m * SLOTS + slot) * INTER, g0, g0 + G_DN / DN_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float acc = 0.0f;
    for (uint j = 0; j < MOE_K; ++j) {
      const ushort d_b = rne_bf16(red[j * DN_KS][lane]);
      const ushort t_b = rne_bf16(bf16f(d_b) * as_float(rr[R_W + j]));
      acc = bf16f(rne_bf16(acc + bf16f(t_b)));
    }
    const ushort sh_b = rne_bf16(red[MOE_K * DN_KS][lane]);
    const ushort o_b = rne_bf16(acc + bf16f(sh_b));
    __global ushort* restrict rp = resid + (size_t)m * HIDDEN + n_tile * SG + lane;
    *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
  }
}
#endif  // MOE_E

// ---------------------------------------------------------------------------------------
// k2_mova_value - MoVA's routed value experts (K2HorizonMoVAAttention, lines 428-444): the
// MOVA_K selected experts' GEMVs (2560 -> 1024 each, layout-1 blocks, one launch over all
// of them), SiLU, the weighted ascending-id bf16 combine, written into this layer's V cache
// at row pos + m (the KV layout [max_len][KV_HEADS][HD]: column n of v is head n / HD).
// One 16-column n-tile per work-group; sub-group (slot, K slice):
//
//   v_e  = rne(Σ V_e x)                   the expert linear's bf16 output
//   a_e  = rne(silu(f32(v_e)))            activation=F.silu, bf16
//   acc  = 0;  for e ascending:  acc = f32(rne(acc + f32(rne(f32(a_e) · w_e))))
//   kv_v[pos + m][n] = rne(acc)
#ifdef MOVA_E
#if !defined(MOVA_K) || !defined(MOVA_D) || !defined(MOVA_N) || !defined(MOVA_KS)
#error "k2_mova: MOVA_K, MOVA_D (hidden), MOVA_N (kv heads x head dim) and MOVA_KS must be defined"
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_mova: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if MOVA_K < 1 || MOVA_K > 8 || MOVA_N % SG != 0 || MOVA_D % GROUP != 0
#error "k2_mova: 1 <= MOVA_K <= 8, MOVA_N whole n-tiles, MOVA_D whole k-groups"
#endif
#define G_V (MOVA_D / GROUP)
#define V_BLK_U32 ((size_t)(MOVA_N / SG) * G_V * TILE_U32)
#define WG_V (SG * MOVA_K * MOVA_KS)
#if (MOVA_KS & (MOVA_KS - 1)) != 0 || G_V % MOVA_KS != 0
#error "k2_mova: MOVA_KS must be a power of two dividing MOVA_D / 64"
#endif
__attribute__((reqd_work_group_size(WG_V, 1, 1)))
K2_SG16
__kernel void k2_mova_value(__global const uint* restrict ctrl, __global const uint* restrict route,
                            __global const ushort* restrict x, __global const uint* restrict w,
                            __global ushort* restrict kv_v) {
  __local float red[MOVA_K * MOVA_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / MOVA_KS;
  const uint q = sg % MOVA_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform across the work-group
  const __global uint* restrict rr = route + (size_t)m * RW;

  uint e = rr[R_IDS + slot];
  e = min(e, (uint)(MOVA_E - 1));
  const uint g0 = q * (G_V / MOVA_KS);
  const __global uint* restrict tile = w + (size_t)e * V_BLK_U32 + ((size_t)n_tile * G_V + g0) * TILE_U32;
  red[sg][lane] = tile_dot(tile, x + (size_t)m * MOVA_D, g0, g0 + G_V / MOVA_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = MOVA_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float acc = 0.0f;
    for (uint j = 0; j < MOVA_K; ++j) {
      const ushort v_b = rne_bf16(red[j * MOVA_KS][lane]);
      const ushort a_b = rne_bf16(silu_f32(bf16f(v_b)));
      const ushort t_b = rne_bf16(bf16f(a_b) * as_float(rr[R_W + j]));
      acc = bf16f(rne_bf16(acc + bf16f(t_b)));
    }
    kv_v[(size_t)(pos + m) * MOVA_N + n_tile * SG + lane] = rne_bf16(acc);
  }
}
#endif  // MOVA_E
