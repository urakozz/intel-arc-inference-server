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
// cross-lane step goes through SLM. moe_route uses no
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
#include "expert_fp.h"
#define G_UP (HIDDEN / GROUP)       /* gate||up k-groups: 32 at hidden 2048 */
#define NT_UP (2 * INTER / SG)      /* gate||up n-tiles per expert: 64 at INTER 512 */
#define UP_BLK_U32 ((size_t)NT_UP * G_UP * TILE_U32 + EXPERT_OUTER_WORDS)
#define G_DN (INTER / GROUP)        /* down k-groups: 8 */
#define NT_DN (HIDDEN / SG)         /* down n-tiles: 128 */
#define DN_BLK_U32 ((size_t)NT_DN * G_DN * TILE_U32 + EXPERT_OUTER_WORDS)
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

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define MOE_SG16 __attribute__((intel_reqd_sub_group_size(SG)))

#include "bf16.h"

// prep.cl's silu, plain `exp`; the sigmoid as attn.cl spells it.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

#if WFMT != 0
#include "expert_fp_gemv.h"
#else
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
  (void)lane;
  const uint8 b = intel_sub_group_block_read8(tile);
  wv[0] = b.s0; wv[1] = b.s1; wv[2] = b.s2; wv[3] = b.s3;
  wv[4] = b.s4; wv[5] = b.s5; wv[6] = b.s6; wv[7] = b.s7;
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

#endif

// EXPERT_DPAS 1 (the int4 experts): every dot on the bf16 DPAS, 8 rows a sub-group (rows past n zero) - the one-row
// kernels' single row and a union item's rows alike, so a verify row stays bitwise the plain step's (a row's sums do
// not depend on the operand's other rows). Each k16 step's codes once as the B operand (a code byte's two values
// c - 8 as bf16 bits from a 256-entry table, exact), the rows' activations the A operand, the k-group's four steps
// summed, then the group's scale (the lane's column) on the sums. Not tile_dot's order: the f32 judge.
#ifndef EXPERT_DPAS
#define EXPERT_DPAS 0
#endif
#if EXPERT_DPAS
#if WFMT != 0
#error "moe: EXPERT_DPAS is the int4 experts' form"
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define EXPERT_TABLE(q4_pair)                                                                       \
  __local uint q4_pair[256];                                                                       \
  for (uint i = get_local_id(0); i < 256; i += get_local_size(0))                                  \
    q4_pair[i] = (as_uint((float)((int)(i & 15u) - 8)) >> 16) |                                    \
                 (as_uint((float)((int)(i >> 4) - 8)) & 0xFFFF0000u);                              \
  barrier(CLK_LOCAL_MEM_FENCE);
// k-groups [g0, g1) of one 16-column tile for rows x + xo[r], r < n: the result's element r is row r's sum.
inline float8 tile_dpas(const __global uint* restrict tile, const __global ushort* restrict x, const uint* xo,
                        uint n, uint g0, uint g1, uint lane, __local const uint* q4_pair) {
  float8 acc = (float8)(0.f);
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    ushort xa[8][4];
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      ushort4 v = (ushort4)(0);
      if (r < M && r < n) v = intel_sub_group_block_read_us4(x + xo[r] + g * GROUP);
      xa[r][0] = v.s0; xa[r][1] = v.s1; xa[r][2] = v.s2; xa[r][3] = v.s3;
    }
    float8 c = (float8)(0.f);
#pragma unroll
    for (uint d = 0; d < 4; ++d) {
      short8 a;
      a.s0 = xa[0][d]; a.s1 = xa[1][d]; a.s2 = xa[2][d]; a.s3 = xa[3][d];
      a.s4 = xa[4][d]; a.s5 = xa[5][d]; a.s6 = xa[6][d]; a.s7 = xa[7][d];
      uint p[8];
#pragma unroll
      for (uint i = 0; i < 8; ++i) p[i] = q4_pair[(wv[2 * d + i / 4] >> (8 * (i % 4))) & 255u];
      c = intel_sub_group_bf16_bf16_matrix_mad_k16(a, as_int8(vload8(0, p)), c);
    }
    acc += c * scale;
  }
  return acc;
}
#endif

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
//
// ROUTE_SG 1 (prefill's moe_M1_..._R<pitch> binary only; tests/kernels/pf_moe_test.cc compares its every word with
// the form below): ONE SUB-GROUP a token, lane l holding experts 16 j + l - no SLM and no barrier. The max is exact
// in any order; the Σ tree's strides EXPERTS/2 .. 16 are the register pairs j, j + stride/16 and 8 .. 1 are
// shuffles, the same additions in the same order; the slots are taken one at a time, each the best untaken expert
// by (p descending, id ascending), which is the rank order above.
#ifndef ROUTE_SG
#define ROUTE_SG 0
#endif
#if ROUTE_SG
#if EXPERTS != 256 || TOP_K != 8
#error "moe: ROUTE_SG is built for 256 experts, top 8 (a lane's 16 experts, the route row's 8 slots)"
#endif
#define RPL (EXPERTS / SG)
__attribute__((reqd_work_group_size(SG, 1, 1))) MOE_SG16
__kernel void moe_route(__global const float* restrict logits, __global uint* restrict route) {
  const uint m = get_group_id(1);
  const uint lane = get_sub_group_local_id();
  const __global float* restrict lrow = logits + (size_t)m * ROUTER_N;
  float l[RPL], ex[RPL], red[RPL];
#pragma unroll
  for (uint j = 0; j < RPL; ++j) l[j] = bf16f(rne_bf16(lrow[SG * j + lane]));
  float mx = l[0];
#pragma unroll
  for (uint j = 1; j < RPL; ++j) mx = fmax(mx, l[j]);
#pragma unroll
  for (uint st = SG / 2; st > 0; st >>= 1) mx = fmax(mx, intel_sub_group_shuffle_xor(mx, st));
#pragma unroll
  for (uint j = 0; j < RPL; ++j) red[j] = ex[j] = exp(l[j] - mx);
#pragma unroll
  for (uint js = RPL / 2; js > 0; js >>= 1)
#pragma unroll
    for (uint j = 0; j < js; ++j) red[j] += red[j + js];
  float s = red[0];
#pragma unroll
  for (uint st = SG / 2; st > 0; st >>= 1) s = s + intel_sub_group_shuffle(s, (lane + st) & (SG - 1));
  const float S = sub_group_broadcast(s, 0);
  float p[RPL];
#pragma unroll
  for (uint j = 0; j < RPL; ++j) p[j] = ex[j] / S;

  uint taken = 0u;               // bit j: expert SG j + lane is in a slot
  uint top_id[SLOTS];
  float top_p[SLOTS];
#pragma unroll
  for (uint k = 0; k < SLOTS; ++k) {
    float bp = -1.0f;            // below every p; a NaN p is never taken (the form below leaves its slot defined)
    uint be = (uint)k;
#pragma unroll
    for (uint j = 0; j < RPL; ++j)
      if (!((taken >> j) & 1u) && p[j] > bp) {   // strict: the lower j (lower id) keeps a tie
        bp = p[j];
        be = SG * j + lane;
      }
#pragma unroll
    for (uint st = SG / 2; st > 0; st >>= 1) {
      const float op = intel_sub_group_shuffle_xor(bp, st);
      const uint oe = intel_sub_group_shuffle_xor(be, st);
      if (op > bp || (op == bp && oe < be)) {
        bp = op;
        be = oe;
      }
    }
    top_id[k] = be;
    top_p[k] = bp < 0.0f ? 0.0f : bp;
    if (bp >= 0.0f && be % SG == lane) taken |= 1u << (be / SG);
  }
  float s8 = 0.0f;
#pragma unroll
  for (uint k = 0; k < TOP_K; ++k) s8 += top_p[k];
  // lane l writes words l and l + 16 of the row
  uint lo = 0u, hi = 0u;
#pragma unroll
  for (uint k = 0; k < TOP_K; ++k) {
    if (lane == R_IDS + k) lo = top_id[k];
    if (lane == R_W + k) lo = as_uint(bf16f(rne_bf16(top_p[k] / s8)));
    if (lane + 16u == R_P + k) hi = as_uint(top_p[k]);
  }
  if (lane + 16u == R_SG) hi = as_uint(bf16f(rne_bf16(sigmoid_f32(bf16f(rne_bf16(lrow[EXPERTS]))))));
  if (lane + 16u == R_P9) hi = as_uint(top_p[TOP_K]);
  __global uint* restrict r = route + (size_t)m * RW;
  r[lane] = lo;
  r[lane + 16u] = hi;
}
#else
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
#endif

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
#if EXPERT_DPAS
  EXPERT_TABLE(q4_pair)
  const uint xo[1] = {0};
  red[sg][lane] = tile_dpas(tile, x + (size_t)m * HIDDEN, xo, 1, g0, g0 + G_UP / UP_KS, lane, q4_pair).s0;
#else
  red[sg][lane] = tile_dot(tile, x + (size_t)m * HIDDEN, g0, g0 + G_UP / UP_KS, lane);
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0 && (tile_idx & 1u) == 0) {
    const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
#if WFMT == 1
    const float outer = as_float(w[(size_t)(e + 1) * UP_BLK_U32 - 1]);
    const ushort g_b = rne_bf16(red[sg][lane] * outer);
    const ushort u_b = rne_bf16(red[sg + UP_KS][lane] * outer);
#else
    const ushort g_b = rne_bf16(red[sg][lane]);
    const ushort u_b = rne_bf16(red[sg + UP_KS][lane]);
#endif
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
#if EXPERT_DPAS
  EXPERT_TABLE(q4_pair)
  const uint xo[1] = {0};
  red[sg][lane] =
      tile_dpas(tile, h + ((size_t)m * SLOTS + slot) * INTER, xo, 1, g0, g0 + G_DN / DN_KS, lane, q4_pair).s0;
#else
  red[sg][lane] =
      tile_dot(tile, h + ((size_t)m * SLOTS + slot) * INTER, g0, g0 + G_DN / DN_KS, lane);
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride) red[sg][lane] += red[sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sg == 0) {
    float sum = 0.0f;
    for (uint k = 0; k < TOP_K; ++k) {
#if WFMT == 1
      const uint ek = min(rr[R_IDS + k], (uint)SHARED);
      const ushort d_b = rne_bf16(red[k * DN_KS][lane] * as_float(w[(size_t)(ek + 1) * DN_BLK_U32 - 1]));
#else
      const ushort d_b = rne_bf16(red[k * DN_KS][lane]);
#endif
      sum += bf16f(rne_bf16(bf16f(d_b) * as_float(rr[R_W + k])));
    }
    const ushort r_b = rne_bf16(sum);
#if WFMT == 1
    const ushort ds_b = rne_bf16(red[TOP_K * DN_KS][lane] * as_float(w[(size_t)(SHARED + 1) * DN_BLK_U32 - 1]));
#else
    const ushort ds_b = rne_bf16(red[TOP_K * DN_KS][lane]);
#endif
    const ushort sh_b = rne_bf16(bf16f(ds_b) * as_float(rr[R_SG]));
    const ushort o_b = rne_bf16(bf16f(r_b) + bf16f(sh_b));
    __global ushort* restrict rp = resid + (size_t)m * HIDDEN + n_tile * SG + lane;
    *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
  }
}

// ---------------------------------------------------------------------------
// MOE_UNION (PERF-120; undefined on every other command line): the row lists of 16 and 32 rows across sequences
// read each routed expert ONCE for (up to UR of) the rows routed to it - a 16-row step routes ~56 distinct experts a
// layer for its 128 per-row reads, a 32-row step ~84 for 256 (Ornith, depth 4096). Five launches instead of three:
//
//   moe_route(logits, route)                            as above, grid (1, M)
//   moe_union_plan(ctrl, route, plan)                   grid (1), WG EXPERTS
//   moe_gate_up_union(plan, x, w_gu, h)                 grid (MAX_ITEMS x NT_UP/4), WG 64 x UP_KS
//   moe_down_union(plan, h, w_dn, dn)                   grid (MAX_ITEMS x NT_DN/NTW), WG 16 x NTW x DN_KS
//   moe_finish(ctrl, route, dn, resid)                  grid (HIDDEN / 16, M), WG 16
//
// The plan lists, for every expert (the shared one last, every live row at slot TOP_K), its (row, slot) pairs among
// the list's n_active rows in (row, slot) order and its items - runs of at most UR of those pairs:
//   plan[0] the items, plan[1 + 2 i] item i's expert, plan[2 + 2 i] its first pair | its pairs << 16,
//   plan[1 + 2 MAX_ITEMS + p] pair p = row << 4 | slot.
// Work-group (item, blk) of gate||up / down is moe_gate_up's / moe_down's work-group for each of its pairs with the
// weight words read once a k-group: every pair's sums are tile_dot's, in its order (8 dot8 into a group sum, then the
// group sum times the scale into the running sum), its K slices merged by the same tree - so h and each slot's down
// output d_b are bitwise the per-row kernels'. moe_down_union writes d_b [M][SLOTS][HIDDEN] bf16 to `dn`; moe_finish is
// moe_down's combine (fixed slot order, the shared gate, the residual fold) over them. A row at or past n_active (the
// list's padding) is not read, written or folded. No atomic; no value depends on where a pair sits in the plan.
#ifdef MOE_UNION
#if WFMT != 0 || M > 32 || !defined(CTRL_NACT)
#error "moe MOE_UNION: the int4 experts at M <= 32 rows, CTRL_NACT"
#endif
#define UR 8                        /* pairs an item: a pass over the weight words */
#define NTW 8                       /* moe_down_union: n-tiles a work-group */
#define WG_DNU (SG * NTW * DN_KS)
#define MAX_ITEMS (M * TOP_K + (M + UR - 1) / UR)   /* every routed pair its own item, the shared expert's runs */
#define PLAN_PAIRS (1 + 2 * MAX_ITEMS)
#if NT_DN % NTW != 0
#error "moe MOE_UNION: NTW divides the down's n-tiles"
#endif

// An exclusive prefix sum over the work-group (lane order) and its total: sub-group scans, then the sub-groups' sums.
inline uint wg_scan(uint v, __local uint* sums, uint* total) {
  const uint sg = get_local_id(0) / SG, nsg = get_local_size(0) / SG;
  const uint before = sub_group_scan_exclusive_add(v);
  if (get_sub_group_local_id() == SG - 1) sums[sg] = before + v;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint base = 0, all = 0;
  for (uint s = 0; s < nsg; ++s) {
    base += s < sg ? sums[s] : 0u;
    all += sums[s];
  }
  barrier(CLK_LOCAL_MEM_FENCE);   // sums is the next scan's
  *total = all;
  return base + before;
}

__attribute__((reqd_work_group_size(WG_ROUTE, 1, 1)))
MOE_SG16
__kernel void moe_union_plan(__global const uint* restrict ctrl, __global const uint* restrict route,
                             __global uint* restrict plan) {
  __local uint ids[M * TOP_K];
  __local uint sums[WG_ROUTE / SG];
  const uint e = get_local_id(0);
  const uint n_act = min(ctrl[CTRL_NACT], (uint)M), np = n_act * TOP_K;
  for (uint i = e; i < np; i += WG_ROUTE) ids[i] = route[(size_t)(i / TOP_K) * RW + R_IDS + i % TOP_K];
  barrier(CLK_LOCAL_MEM_FENCE);
  uint c = 0;
  for (uint i = 0; i < np; ++i) c += ids[i] == e ? 1u : 0u;
  // the routed experts in id order, then the shared expert (every live row) after them
  uint n_items, n_pairs;
  const uint ioff = wg_scan((c + UR - 1) / UR, sums, &n_items);
  const uint poff = wg_scan(c, sums, &n_pairs);
  const uint sh_items = (n_act + UR - 1) / UR;
  if (e == 0) plan[0] = n_items + sh_items;
  __global uint* restrict pairs = plan + PLAN_PAIRS;
  for (uint i = 0, k = poff; k < poff + c; ++i)
    if (ids[i] == e) pairs[k++] = ((i / TOP_K) << 4) | (i % TOP_K);
  for (uint j = 0; j * UR < c; ++j) {
    plan[1 + 2 * (ioff + j)] = e;
    plan[2 + 2 * (ioff + j)] = (poff + j * UR) | (min(c - j * UR, (uint)UR) << 16);
  }
  for (uint m = e; m < n_act; m += WG_ROUTE) pairs[n_pairs + m] = (m << 4) | TOP_K;
  for (uint j = e; j < sh_items; j += WG_ROUTE) {
    plan[1 + 2 * (n_items + j)] = SHARED;
    plan[2 + 2 * (n_items + j)] = (n_pairs + j * UR) | (min(n_act - j * UR, (uint)UR) << 16);
  }
}

__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
MOE_SG16
__kernel void moe_gate_up_union(__global const uint* restrict plan,
                                __global const ushort* restrict x,
                                __global const uint* restrict w,
                                __global ushort* restrict h) {
  __local float red[UR][4 * UP_KS][SG];
  const uint item = get_group_id(0) / (NT_UP / 4);
  if (item >= plan[0]) return;
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint tile_idx = sg / UP_KS;
  const uint q = sg % UP_KS;
  const uint blk = get_group_id(0) % (NT_UP / 4);
  const uint e = plan[1 + 2 * item], p0 = plan[2 + 2 * item] & 0xFFFFu, nr = plan[2 + 2 * item] >> 16;
  const __global uint* restrict pairs = plan + PLAN_PAIRS + p0;
  const uint n_tile = blk * 4 + tile_idx;
  const uint g0 = q * (G_UP / UP_KS);
  const __global uint* restrict tile = w + (size_t)e * UP_BLK_U32 + ((size_t)n_tile * G_UP + g0) * TILE_U32;
  float acc[UR];
#if EXPERT_DPAS
  EXPERT_TABLE(q4_pair)
  uint xo[UR];
  for (uint r = 0; r < UR; ++r) xo[r] = r < nr ? (pairs[r] >> 4) * HIDDEN : 0u;
  vstore8(tile_dpas(tile, x, xo, nr, g0, g0 + G_UP / UP_KS, lane, q4_pair), 0, acc);
#else
  for (uint r = 0; r < UR; ++r) acc[r] = 0.f;
  for (uint g = g0; g < g0 + G_UP / UP_KS; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    for (uint r = 0; r < UR; ++r)
      if (r < nr) {
        const __global ushort* restrict xr = x + (size_t)(pairs[r] >> 4) * HIDDEN;
        float gacc = 0.f;
        for (uint j = 0; j < 8; ++j) gacc += dot8(wv[j], vload8(0, xr + g * GROUP + j * 8));
        acc[r] += gacc * scale;
      }
  }
#endif
  for (uint r = 0; r < nr; ++r) red[r][sg][lane] = acc[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = UP_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (uint r = 0; r < nr; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0 && (tile_idx & 1u) == 0) {
    const uint i = blk * 32 + (tile_idx / 2) * SG + lane;
    for (uint r = 0; r < nr; ++r) {
      const ushort g_b = rne_bf16(red[r][sg][lane]);
      const ushort u_b = rne_bf16(red[r][sg + UP_KS][lane]);
      const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
      h[((size_t)(pairs[r] >> 4) * SLOTS + (pairs[r] & 15u)) * INTER + i] = rne_bf16(bf16f(s_b) * bf16f(u_b));
    }
  }
}

__attribute__((reqd_work_group_size(WG_DNU, 1, 1)))
MOE_SG16
__kernel void moe_down_union(__global const uint* restrict plan,
                             __global const ushort* restrict h,
                             __global const uint* restrict w,
                             __global ushort* restrict dn) {
  __local float red[UR][NTW * DN_KS][SG];
  const uint item = get_group_id(0) / (NT_DN / NTW);
  if (item >= plan[0]) return;
  const uint lid = get_local_id(0);
  const uint lane = lid % SG;
  const uint sg = lid / SG;
  const uint t = sg / DN_KS;
  const uint q = sg % DN_KS;
  const uint n_tile = (get_group_id(0) % (NT_DN / NTW)) * NTW + t;
  const uint e = plan[1 + 2 * item], p0 = plan[2 + 2 * item] & 0xFFFFu, nr = plan[2 + 2 * item] >> 16;
  const __global uint* restrict pairs = plan + PLAN_PAIRS + p0;
  const uint g0 = q * (G_DN / DN_KS);
  const __global uint* restrict tile = w + (size_t)e * DN_BLK_U32 + ((size_t)n_tile * G_DN + g0) * TILE_U32;
  float acc[UR];
#if EXPERT_DPAS
  EXPERT_TABLE(q4_pair)
  uint xo[UR];
  for (uint r = 0; r < UR; ++r) xo[r] = r < nr ? ((pairs[r] >> 4) * SLOTS + (pairs[r] & 15u)) * INTER : 0u;
  vstore8(tile_dpas(tile, h, xo, nr, g0, g0 + G_DN / DN_KS, lane, q4_pair), 0, acc);
#else
  for (uint r = 0; r < UR; ++r) acc[r] = 0.f;
  for (uint g = g0; g < g0 + G_DN / DN_KS; ++g, tile += TILE_U32) {
    uint wv[8];
    load_tile8(tile, lane, wv);
    const float scale = vload_half(lane, (const __global half*)(tile + 128));
    for (uint r = 0; r < UR; ++r)
      if (r < nr) {
        const __global ushort* restrict hr = h + ((size_t)(pairs[r] >> 4) * SLOTS + (pairs[r] & 15u)) * INTER;
        float gacc = 0.f;
        for (uint j = 0; j < 8; ++j) gacc += dot8(wv[j], vload8(0, hr + g * GROUP + j * 8));
        acc[r] += gacc * scale;
      }
  }
#endif
  for (uint r = 0; r < nr; ++r) red[r][sg][lane] = acc[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (uint r = 0; r < nr; ++r) red[r][sg][lane] += red[r][sg + stride][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (uint r = 0; r < nr; ++r)
      dn[((size_t)(pairs[r] >> 4) * SLOTS + (pairs[r] & 15u)) * HIDDEN + n_tile * SG + lane] = rne_bf16(red[r][sg][lane]);
}

__attribute__((reqd_work_group_size(SG, 1, 1)))
MOE_SG16
__kernel void moe_finish(__global const uint* restrict ctrl,
                         __global const uint* restrict route,
                         __global const ushort* restrict dn,
                         __global ushort* restrict resid) {
  const uint lane = get_local_id(0);
  const uint n_tile = get_group_id(0);
  const uint m = get_group_id(1);
  if (m >= min(ctrl[CTRL_NACT], (uint)M)) return;
  const __global uint* restrict rr = route + (size_t)m * RW;
  const __global ushort* restrict dr = dn + (size_t)m * SLOTS * HIDDEN + n_tile * SG + lane;
  float sum = 0.0f;
  for (uint k = 0; k < TOP_K; ++k) sum += bf16f(rne_bf16(bf16f(dr[(size_t)k * HIDDEN]) * as_float(rr[R_W + k])));
  const ushort r_b = rne_bf16(sum);
  const ushort sh_b = rne_bf16(bf16f(dr[(size_t)TOP_K * HIDDEN]) * as_float(rr[R_SG]));
  const ushort o_b = rne_bf16(bf16f(r_b) + bf16f(sh_b));
  __global ushort* restrict rp = resid + (size_t)m * HIDDEN + n_tile * SG + lane;
  *rp = rne_bf16(bf16f(*rp) + bf16f(o_b));
}
#endif
