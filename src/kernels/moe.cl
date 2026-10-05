// moe.cl - spec 15c: the mixture-of-experts block of one decode step (spec 15 §4.1,
// §4.2 and §9's "shared expert as the ninth slot"). Three entry points in one file,
// as prep.cl's, so one binary per (M, shape) carries all three:
//
//   moe_route(logits, route)            grid (1, M),               WG EXPERTS (256)
//   moe_gate_up(route, x, w_gu, h)      grid (SLOTS x NT_UP/4, M), WG 64 x UP_KS
//   moe_down(route, h, w_dn, resid)     grid (HIDDEN / 16, M),     WG 16 x SLOTS x DN_KS
//
// Before them the decode list runs gemv_bf16 over the router || shared-gate rows
// (loader/moe_layout.h), writing `logits` fp32 [M][ROUTER_N]: the EXPERTS router
// logits, the shared expert's gate logit at column EXPERTS, zero padding. So a MoE
// block is FOUR launches - router GEMV, route, gate||up, down - with the expert ids
// produced and consumed on the device: nothing returns to the host and the captured
// list is fixed (spec 15 decision 3).
//
// **The reference this follows** (transformers 5.18.0, modeling_qwen3_5_moe.py,
// Qwen3_5MoeTopKRouter / Qwen3_5MoeSparseMoeBlock, with transformers' default
// `grouped_mm` experts - integrations/moe.py grouped_mm_experts_forward; 15a pins the
// oracle's settings), op by op, each torch op rounded where torch rounds it:
//
//   l_b   = rne(router GEMV)                      F.linear in bf16: the linear's output
//   p     = softmax(f32(l_b)) over EXPERTS        dtype=torch.float
//   top   = the TOP_K largest p, descending       torch.topk (sorted); ties -> lower id
//   w_k   = rne(p_k / Σ_k p_k)                    renormalised ALWAYS (this class has no
//                                                 norm_topk_prob switch), then .to(bf16)
//   gu    = expert gate||up GEMV -> g_b, u_b      bf16 linear outputs
//   a_b   = rne(rne(silu(f32(g_b))) · f32(u_b))   act_fn(gate) * up, both bf16 ops
//   d_b   = rne(expert down GEMV)                 bf16 linear output
//   t_k   = rne(f32(d_b,k) · f32(w_k))            proj_out * weights, bf16 x bf16
//   r_b   = rne(Σ_k f32(t_k))                     .view(T, k, H).sum(dim=1): fp32
//                                                 accumulation, slot order 0..k-1
//   s_b   = rne(sigmoid(f32(rne(gate GEMV))))     sigmoid(shared_expert_gate(x))
//   sh_b  = rne(f32(s_b) · f32(shared d_b))       its product with the shared output
//   o_b   = rne(f32(r_b) + f32(sh_b))             expert_output + shared_expert_output
//   resid = rne(f32(resid) + f32(o_b))            the decoder layer's residual add
//
// moe_down performs the last line itself - the "residual fold" - so the next
// prep_res_fold folds nothing (S_PREV 0; model::ModelDesc::ffn_fold_s).
//
// **Determinism.** No atomic anywhere. The sum over experts is in FIXED slot order
// (the route row's order: descending p, ties to the lower expert id, then the shared
// expert as slot TOP_K) through SLM, never in completion order; the softmax Σ is a
// fixed pairwise tree; every K split inside a work-group is merged by prep.cl's tree
// idiom. Two replays give the same bits.
//
// **Portability.** The two GEMV kernels use Intel's sub-group block reads when the
// compiler defines `cl_intel_subgroups` (ocloc for the B70) and plain per-lane loads
// of the same elements otherwise (lane l of a 16-lane group reads element l of each
// 16-element row: exactly what intel_sub_group_block_read8 returns), and every
// cross-lane step goes through SLM - so the file also builds as OpenCL 1.2 for an
// indicative host run (spec 15c, the Mac's OpenCL devices). moe_route uses no
// sub-group functions at all.
//
// The route row - u32 words per (layer, token), the host mirror is kernels::moe_route
// (kernels.h), read back by name in tests/kernels/moe_test.cc:
//   [R_IDS + k]  expert id of slot k, k < TOP_K             (u32)
//   [R_W + k]    w_k, a bf16 value held in fp32              (fp32 bits)
//   [R_SG]       s_b, the shared expert's gate              (fp32 bits, bf16 value)
//   [R_P + k]    p of slot k before renormalisation         (fp32 bits, diagnostics)
//   [R_P9]       p of rank TOP_K - the first expert NOT taken (R2's near-tie margin)
#ifndef M
#define M 1
#endif
#if !defined(EXPERTS) || !defined(TOP_K) || !defined(HIDDEN) || !defined(INTER) || !defined(ROUTER_N)
#error "moe: EXPERTS, TOP_K, HIDDEN, INTER and ROUTER_N must be defined (src/kernels/CMakeLists.txt)"
#endif
// The two in-work-group K splits (powers of 2) - they set the work-group sizes, so the
// host mirrors them (kernels::kMoeUpKs / kMoeDnKs) and CMake passes them explicitly.
#if !defined(UP_KS) || !defined(DN_KS)
#error "moe: UP_KS and DN_KS must be defined (src/kernels/CMakeLists.txt MOE_DEFINES)"
#endif

#define SLOTS (TOP_K + 1)           /* TOP_K routed + the shared expert */
#define SHARED EXPERTS              /* the shared expert's weight block (MoeDesc::shared_block) */
#define RW 32                       /* route row words (kernels::moe_route::kWords) */
#define R_IDS 0
#define R_W 8
#define R_SG 16
#define R_P 17
#define R_P9 25
#define SG 16
#define GROUP 64
#define TILE_U32 136                /* layout 1: 128 u32 of nibbles + 8 u32 of f16 scales */
#define G_UP (HIDDEN / GROUP)       /* gate||up k-groups: 32 at hidden 2048 */
#define NT_UP (2 * INTER / SG)      /* gate||up n-tiles per expert: 64 at INTER 512 */
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32)
#define G_DN (INTER / GROUP)        /* down k-groups: 8 */
#define NT_DN (HIDDEN / SG)         /* down n-tiles: 128 */
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32)
#define WG_ROUTE EXPERTS
#define WG_UP (4 * SG * UP_KS)      /* 4 n-tiles (2 gate + 2 up) x UP_KS K slices */
#define WG_DN (SG * SLOTS * DN_KS)  /* 1 n-tile x SLOTS x DN_KS K slices */

#if TOP_K < 1 || TOP_K > 8
#error "moe: TOP_K is 1..8 (the route row holds 8 slots)"
#endif
#if (EXPERTS & (EXPERTS - 1)) != 0 || EXPERTS < 16 || EXPERTS > 256
#error "moe: EXPERTS must be a power of two in [16, 256] (one per lane, a pairwise tree)"
#endif
#if ROUTER_N < EXPERTS + 1 || ROUTER_N % 16 != 0
#error "moe: ROUTER_N holds the EXPERTS router rows and the shared gate's row, padded to 16"
#endif
#if HIDDEN % GROUP != 0 || INTER % 32 != 0 || INTER % GROUP != 0
#error "moe: HIDDEN and INTER must be whole k-groups, INTER whole 32-column gate||up pairs"
#endif
#if (UP_KS & (UP_KS - 1)) != 0 || G_UP % UP_KS != 0
#error "moe: UP_KS must be a power of two dividing HIDDEN / 64"
#endif
#if (DN_KS & (DN_KS - 1)) != 0 || G_DN % DN_KS != 0
#error "moe: DN_KS must be a power of two dividing INTER / 64"
#endif

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define MOE_SG16 __attribute__((intel_reqd_sub_group_size(SG)))
#else
#define MOE_SG16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even (prep.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// prep.cl's silu, plain `exp`; the sigmoid as attn.cl spells it.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// gemv.cl's dot8, the plain spelling: 8 nibbles of one word times 8 activations.
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

// One layout-1 tile's eight nibble words for this lane: word j is the tile's
// element j x 16 + lane (common/repack.h: [k_octet j][lane l]).
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

// The int4 layout-1 GEMV of one 16-column n-tile over k-groups [g0, g1) of one
// expert block: gemv.cl's LAYOUT 1 inner loop, its order (8 dot8 into a group sum,
// then the group sum times the f16 scale into the running sum).
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

// ---------------------------------------------------------------------------
// moe_route - one work-group per token, one expert per lane (lane e owns expert e).
//
//   l   = f32(rne(logits[m][e]))                 the router linear's bf16 output
//   mx  = max_e l                                every lane scans SLM: exact, any order
//   ex  = exp(l - mx)
//   S   = Σ_e ex   - the pairwise tree: stride = EXPERTS/2 .. 1, red[e] += red[e+stride]
//   p   = ex / S
//   rank(e) = #{ j : p_j > p_e  or  (p_j == p_e and j < e) }   - ties to the lower id
//   rank < SLOTS: slot rank = (e, p)   (rank TOP_K is the first expert not taken)
//   lane 0: s8 = Σ_{k < TOP_K} p_k ascending k;  w_k = f32(rne(p_k / s8))
//           s_b = rne(sigmoid(f32(rne(logits[m][EXPERTS]))))
//
// The rank is the whole top-k: every lane counts its beaters over the SLM row (EXPERTS
// broadcast reads) and the ranks are a permutation - one barrier, no rounds.
// tests/kernels/moe_ref.h repeats this chain step for step.
__attribute__((reqd_work_group_size(WG_ROUTE, 1, 1)))
__kernel void moe_route(__global const float* restrict logits, __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint e = get_local_id(0);
  __local float sl[EXPERTS];     // the bf16-rounded logits, then the probabilities
  __local float red[EXPERTS];    // the softmax Σ tree
  __local uint top_id[SLOTS];
  __local float top_p[SLOTS];

  const __global float* restrict lrow = logits + (size_t)m * ROUTER_N;
  const float l = bf16f(rne_bf16(lrow[e]));
  sl[e] = l;
  if (e < SLOTS) {               // a defined row even if NaNs break the ranking
    top_id[e] = e;
    top_p[e] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = sl[0];
  for (uint j = 1; j < EXPERTS; ++j) mx = fmax(mx, sl[j]);
  const float ex = exp(l - mx);
  red[e] = ex;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = EXPERTS / 2; stride > 0; stride >>= 1) {
    if (e < stride) red[e] += red[e + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float p = ex / red[0];
  sl[e] = p;                     // every lane finished reading the logits (barriers above)
  barrier(CLK_LOCAL_MEM_FENCE);
  uint rank = 0;
  for (uint j = 0; j < EXPERTS; ++j) {
    const float q = sl[j];
    rank += (q > p || (q == p && j < e)) ? 1u : 0u;
  }
  if (rank < SLOTS) {
    top_id[rank] = e;
    top_p[rank] = p;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    __global uint* restrict r = route + (size_t)m * RW;
    float s8 = 0.0f;
    for (uint k = 0; k < TOP_K; ++k) s8 += top_p[k];
    for (uint k = 0; k < 8; ++k) {
      const bool live = k < TOP_K;
      r[R_IDS + k] = live ? top_id[k] : 0u;
      r[R_W + k] = live ? as_uint(bf16f(rne_bf16(top_p[k] / s8))) : 0u;
      r[R_P + k] = live ? as_uint(top_p[k]) : 0u;
    }
    r[R_SG] = as_uint(bf16f(rne_bf16(sigmoid_f32(bf16f(rne_bf16(lrow[EXPERTS]))))));
    r[R_P9] = as_uint(top_p[TOP_K]);
    for (uint k = R_P9 + 1; k < RW; ++k) r[k] = 0u;
  }
}

// ---------------------------------------------------------------------------
// moe_gate_up - gate||up of the TOP_K routed experts and the shared expert (slot
// TOP_K, weight block SHARED), SiLU x up fused. Work-group (slot, blk) owns 4 n-tiles
// of its slot's expert block: n-tiles 4 blk .. 4 blk + 3, i.e. intermediate columns
// [32 blk, 32 blk + 32) - tiles 0 / 2 are their gate halves, 1 / 3 the matching up
// halves (common::cols_interleave16). Sub-group s = (tile s / UP_KS, K slice s %
// UP_KS); the K slices merge by the fixed tree (stride UP_KS/2 .. 1), then the gate
// sub-groups pair with their up partner in SLM:
//
//   g_b = rne(Σ gate), u_b = rne(Σ up)                     the linears' bf16 outputs
//   h[m][slot][i] = rne(f32(rne(silu(f32(g_b)))) · f32(u_b))   prep_silu_mul's chain
//
// The expert id is read ONCE per work-group from the route row, and the weights at
// `id x UP_BLK_U32` - a wrong id or stride shows as one slot's h wrong
// (tests/kernels/moe_test.cc checks every slot against ITS expert).
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
MOE_SG16
__kernel void moe_gate_up(__global const uint* restrict route,
                          __global const ushort* restrict x,
                          __global const uint* restrict w,
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

  uint e = slot < TOP_K ? route[(size_t)m * RW + R_IDS + slot] : (uint)SHARED;
  e = min(e, (uint)SHARED);      // never past the allocation, whatever the row holds
  const uint n_tile = blk * 4 + tile_idx;
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile =
      w + (size_t)e * UP_BLK_U32 + ((size_t)n_tile * G_UP + g0) * TILE_U32;
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

// ---------------------------------------------------------------------------
// moe_down - down of all SLOTS slots for one 16-column n-tile of the hidden row, the
// weighted sum in fixed slot order, the shared expert's gate, the residual fold.
// Sub-group s = (slot s / DN_KS, K slice s % DN_KS) reads slot s / DN_KS's activation
// row h[m][slot] and its expert's down block; the K slices merge by the fixed tree;
// then sub-group 0 combines the SLOTS partials column by column (lane = column):
//
//   for k = 0 .. TOP_K-1 (ascending):  t = rne(f32(rne(down_k)) · w_k);  sum += f32(t)
//   r_b  = rne(sum)
//   sh_b = rne(f32(rne(down_shared)) · s_b)
//   resid[m][n] = rne(f32(resid[m][n]) + f32(rne(f32(r_b) + f32(sh_b))))
__attribute__((reqd_work_group_size(WG_DN, 1, 1)))
MOE_SG16
__kernel void moe_down(__global const uint* restrict route,
                       __global const ushort* restrict h,
                       __global const uint* restrict w,
                       __global ushort* restrict resid) {
  __local float red[SLOTS * DN_KS][SG];
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint slot = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  const __global uint* restrict rr = route + (size_t)m * RW;

  uint e = slot < TOP_K ? rr[R_IDS + slot] : (uint)SHARED;
  e = min(e, (uint)SHARED);
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile =
      w + (size_t)e * DN_BLK_U32 + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  red[sg][lane] =
      tile_dot(tile, h + ((size_t)m * SLOTS + slot) * INTER, g0, g0 + G_DN / DN_KS, lane);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < TOP_K; ++k) {
      const ushort d_b = rne_bf16(red[k * DN_KS][lane]);
      sum += bf16f(rne_bf16(bf16f(d_b) * as_float(rr[R_W + k])));
    }
    const ushort r_b = rne_bf16(sum);
    const ushort ds_b = rne_bf16(red[TOP_K * DN_KS][lane]);
    const ushort sh_b = rne_bf16(bf16f(ds_b) * as_float(rr[R_SG]));
    const ushort o_b = rne_bf16(bf16f(r_b) + bf16f(sh_b));
    __global ushort* restrict rp = resid + (size_t)m * HIDDEN + n_tile * SG + lane;
    *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
  }
}
