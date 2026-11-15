// verify_dpas_ablate.cl (3l; plan 33a Task 0): src/kernels/attn_verify_dpas.cl as on main 5bf86b58 (the PAIR / PVX
// split of 2026-11-10 included; the EAGER build, KV_PAGED and the tensor-parallel heads dropped) with one part of the
// work removed per define - TIMING ONLY, outputs are meaningless (with no define it is bitwise the production kernel
// at the same levers: tests/kernels/attn_verify_bitwise_test.cc --cand). Every removal is a function of the loop
// index (a register formed from the key and the atom), never a constant: IGC hoists a constant load.
//   ABL_K      1: no K loads                                       ABL_KCVT 1: K loaded, no int8 -> bf16 widening (KV8)
//   ABL_V      1: no V loads                                       ABL_VCVT 1: V loaded, no int8 -> bf16 widening (KV8)
//   ABL_SCALE  1: no scale loads (1.0, KV8)
//   ABL_QK     1: no q K^T DPAS (the scores gather the K registers)
//   ABL_SOFT   1: no row max / exp / sum (no sub-group reductions; p = s - m)
//   ABL_PV     1: no P V DPAS (O gathers P and the V registers)
//   ABL_OUT    1: no partial write (kept only behind a runtime-false test, so nothing upstream is dead code)
//   ABL_KONE / ABL_VONE 1: only row block 0's sub-groups load (and widen) K / V; the others take registers (not PAIR)
//   ABL_SDUP   1: (not PAIR, DS 2) sub-group ds == 1 of a row block skips its K loads and score DPAS and takes
//                 index-formed scores - the duplicated half of S at the 16-row line
//   ABL_SHALF  1: (not PAIR, DS 2, bf16) sub-group ds of a row block computes key atom ds of S only and takes
//                 index-formed scores for the other - option B's halved score chain a sub-group, without its exchange
// The bound arm of spec 33 §7: ABL_SDUP + ABL_VONE at 16 rows (what the PAIR split at KS 1 removes, none of its cost);
// ABL_VONE at 8 rows.
//
// One probe-only lift: PAIR is allowed at KS 1 (production #errors), with the exchange `sx` sized for the pair
// sub-groups only and the slice-merge buffer dropped - plan 33a's compile check of plan 33b's line
// (M=16 KS=1 DSPLIT=128 PAIR=1 PVX=8). It is compiled by tools/probe/verify_attn_depth.sh compile, not by CMake,
// and its outputs are not checked.
#ifndef ABL_K
#define ABL_K 0
#endif
#ifndef ABL_KCVT
#define ABL_KCVT 0
#endif
#ifndef ABL_V
#define ABL_V 0
#endif
#ifndef ABL_VCVT
#define ABL_VCVT 0
#endif
#ifndef ABL_SCALE
#define ABL_SCALE 0
#endif
#ifndef ABL_QK
#define ABL_QK 0
#endif
#ifndef ABL_SOFT
#define ABL_SOFT 0
#endif
#ifndef ABL_PV
#define ABL_PV 0
#endif
#ifndef ABL_OUT
#define ABL_OUT 0
#endif
#ifndef ABL_KONE
#define ABL_KONE 0
#endif
#ifndef ABL_VONE
#define ABL_VONE 0
#endif
#ifndef ABL_SDUP
#define ABL_SDUP 0
#endif
#ifndef ABL_SHALF
#define ABL_SHALF 0
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

#ifndef M
#error "attn_verify_dpas: M must be defined"
#endif
#ifndef CTRL_POS
#error "attn_verify_dpas: CTRL_POS must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef CTRL_NACT
#error "attn_verify_dpas: CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef TGT
#error "attn_verify_dpas: TGT must be defined (runtime::DecodeScratch::kAttnV2Blocks)"
#endif
#if !defined(KT) || !defined(PACK_H) || !defined(KS) || !defined(DSPLIT) || !defined(QSPLIT) || !defined(PSPLIT) || \
    !defined(KV8)
#error "attn_verify_dpas: KT, PACK_H, KS, DSPLIT, QSPLIT, PSPLIT and KV8 must all be defined"
#endif
#ifndef PAIR
#define PAIR 0
#endif
#ifndef PVX
#define PVX 0
#endif
#if !((M >= 2 && M <= 8) || M == 16)
#error "attn_verify_dpas: M is 2..8 (16 for the probe's record)"
#endif
#if KT != 32 && KT != 64
#error "attn_verify_dpas: KT is 32 or 64"
#endif
#if KS != 1 && KS != 2 && KS != 4 && KS != 8
#error "attn_verify_dpas: KS is 1, 2, 4 or 8"
#endif
#if DSPLIT != 256 && DSPLIT != 128
#error "attn_verify_dpas: DSPLIT is 256 or 128"
#endif

#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define SG 16
#define PART 258
#define NDA (HD / 16)                 /* dim atoms */
#define NKA (KT / 16)                 /* key atoms per tile */
#define DS (HD / DSPLIT)              /* sub-groups sharing a row block */
#define NDD (DSPLIT / 16)             /* dim atoms a sub-group owns */
#define SSCALE 0.0625f                /* attn_v2's shipped form: exp and 1/16 */
#if PACK_H
#define MB ((M + 7) / 8)
#define RB (GQA * MB)
#else
#define RB ((GQA * M + 7) / 8)
#endif
#define SPS (RB * DS + PVX)           /* sub-groups a slice */
#if PAIR && (KT != 32 || DSPLIT != 128)
#error "attn_verify_dpas: PAIR needs KT 32 and DSPLIT 128"
#endif
#if PVX && (!PAIR || SPS % 2)
#error "attn_verify_dpas: PVX needs PAIR and keeps a slice's sub-group count even"
#endif
#if PAIR && (ABL_KONE || ABL_VONE || ABL_SDUP)
#error "verify_dpas_ablate: ABL_KONE / ABL_VONE / ABL_SDUP are for the plain lines (PAIR reads V once a slice already)"
#endif
#if ABL_SHALF && (DS != 2 || NKA != 2 || KV8 || PAIR)
#error "verify_dpas_ablate: ABL_SHALF is for the bf16 16-row line (two sub-groups, two key atoms a row block)"
#endif
#if ABL_SDUP && DS != 2
#error "verify_dpas_ablate: ABL_SDUP needs two sub-groups a row block (DSPLIT 128)"
#endif
#define NSG (SPS * KS)
#define WG (SG * NSG)
#if WG > 1024
#error "attn_verify_dpas: more than 1024 lanes a work-group (RB x DS x KS sub-groups of 16)"
#endif
// SLM at the real sizes (the probe's KS 1 PAIR line: the exchange for the pair sub-groups only, no slice merge).
#define QS_BYTES (RB * NDA * 128 * 2 * (1 + QSPLIT))
#if PAIR
#define SX_SGS (KS == 1 ? RB * DS : NSG)
#define MERGE_BYTES (KS > 1 ? RB * 8 * HD * 4 : 0)
#define PAIR_BYTES (SX_SGS * 512 + KS * RB * 4 * 128 * 2 + KS * RB * 128 * 4 + KS * RB * 8 * 4 + RB * 8 * 4)
#else
#define MERGE_BYTES (KS > 1 ? RB * 8 * HD * 4 + RB * DS * 8 * 2 * 4 : 0)
#define PAIR_BYTES 0
#endif
#if QS_BYTES + MERGE_BYTES + PAIR_BYTES > 131072
#error "attn_verify_dpas: SLM over 128 KB (q staging + slice merge)"
#endif

#include "bf16.h"
#if KV8
// kv8.cl's binary16 -> fp32, op for op (src/common/kv8.h f16f).
inline float kv8_f16f(ushort h) {
  const uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0u ? (float)m * 5.9604644775390625e-08f
                          : as_float(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}
// An int8 as its exact bf16 bits.
inline uint kv8_bf(char c) { return as_uint((float)c) >> 16; }
// Four int8 (a dword, low byte first) as two dwords of exact bf16 pairs, low byte first.
inline uint2 kv8_bf4(uint w) {
  return (uint2)(kv8_bf(as_char((uchar)(w & 0xFFu))) | (kv8_bf(as_char((uchar)((w >> 8) & 0xFFu))) << 16),
                 kv8_bf(as_char((uchar)((w >> 16) & 0xFFu))) | (kv8_bf(as_char((uchar)(w >> 24))) << 16));
}
#if ABL_SCALE
#define KSC(key) 1.0f
#define VSC(key) 1.0f
#else
#define KSC(key) kv8_f16f(k_scale[(size_t)(key) * KV_HEADS + j])
#define VSC(key) kv8_f16f(v_scale[(size_t)(key) * KV_HEADS + j])
#endif
#endif

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

// Row block rb's A row r -> (q head in the group, row); `ok` false for an empty A row.
inline void row_of(uint rb, uint r, uint* qhl, uint* m, bool* ok) {
#if PACK_H
  *qhl = rb / MB;
  *m = (rb % MB) * 8u + r;
  *ok = *m < M;
#else
  const uint p = rb * 8u + r;
  *qhl = p / M;
  *m = p % M;
  *ok = p < GQA * M;
#endif
}

// The q operand's bf16 halves (QSPLIT): hi = rne(q), lo = rne(q - hi).
inline ushort q_hi(float q) { return rne_bf16(q); }
inline ushort q_lo(float q) { return rne_bf16(q - bf16f(rne_bf16(q))); }

#if KV8
#define KV_ARGS __global const char* restrict kv_k, __global const ushort* restrict k_scale, \
                __global const char* restrict kv_v, __global const ushort* restrict v_scale
#define ENTRY attn_verify_dpas_kv8
#else
#define KV_ARGS __global const ushort* restrict kv_k, __global const ushort* restrict kv_v
#define ENTRY attn_verify_dpas
#endif

// =============================================================================================
// The DPAS kernel.
// =============================================================================================
#define FOR8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)

// K of one dim atom kk (bf16: 16 keys x 16 dims, lane = key) from key row `y`; ABL_K / ABL_KONE: from the indices.
#define K_READ16(kb, kk, y)                                                                            \
  {                                                                                                   \
    if (ABL_K || (ABL_KONE && rb != 0u)) {                                                            \
      for (uint z = 0; z < 8u; ++z) kb[z] = (((y) + l) * 0x00010001u) ^ ((kk) << 4) ^ z;              \
    } else {                                                                                          \
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kv_k, KV_HEADS * HD * 2, (int)end, \
                                                          KV_HEADS * HD * 2,                          \
                                                          (int2)((int)((j * HD + 16u * (kk)) / 2u), (int)(y)), kb); \
    }                                                                                                 \
  }
// int8 K: 16 keys x 32 dims (dim pair kp) from key row `y`, the dwords before widening.
#define K_READ8(k8, kp, y)                                                                             \
  {                                                                                                   \
    if (ABL_K || (ABL_KONE && rb != 0u)) {                                                            \
      for (uint z = 0; z < 8u; ++z) k8[z] = (((y) + l) * 0x01010101u) ^ ((kp) << 4) ^ z;              \
    } else {                                                                                          \
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kv_k, KV_HEADS * HD, (int)end,     \
                                                          KV_HEADS * HD,                              \
                                                          (int2)((int)((j * HD + 32u * (kp)) / 4u), (int)(y)), k8); \
    }                                                                                                 \
  }
#if ABL_KCVT
#define K_WIDEN(kb, k8, h)                                                                             \
  _Pragma("unroll") for (uint z = 0; z < 4u; ++z) {                                                   \
    kb[2u * z] = k8[4u * (h) + z];                                                                    \
    kb[2u * z + 1u] = k8[4u * (h) + z] ^ k8[4u * (h) + ((z + 1u) & 3u)];                              \
  }
#else
#define K_WIDEN(kb, k8, h)                                                                             \
  _Pragma("unroll") for (uint z = 0; z < 4u; ++z) {                                                   \
    const uint2 pr = kv8_bf4(k8[4u * (h) + z]);                                                       \
    kb[2u * z] = pr.x;                                                                                \
    kb[2u * z + 1u] = pr.y;                                                                           \
  }
#endif

// S += q K^T for key atom `kbase` (16 keys from it, lane = key) of row block rb: 16 dim atoms ascending, hi then lo.
#if KV8
#define S_ATOM(sacc_b, kbase)                                                                          \
  _Pragma("unroll") for (uint kp = 0; kp < NDA / 2; ++kp) {                                           \
    uint k8[8];                                                                                       \
    K_READ8(k8, kp, kbase)                                                                            \
    _Pragma("unroll") for (uint h = 0; h < 2u; ++h) {                                                 \
      const uint kk = 2u * kp + h;                                                                    \
      uint kb[8];                                                                                     \
      K_WIDEN(kb, k8, h)                                                                              \
      S_DPAS(sacc_b, kk, kb)                                                                          \
    }                                                                                                 \
  }
#else
#define S_ATOM(sacc_b, kbase)                                                                          \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk) {                                               \
    uint kb[8];                                                                                       \
    K_READ16(kb, kk, kbase)                                                                           \
    S_DPAS(sacc_b, kk, kb)                                                                            \
  }
#endif
#if ABL_QK
#define S_DPAS(sacc_b, kk, kb) { sacc_b += as_float8(vload8(0, kb) & 0x3F80FFFFu) * 1e-30f; }
#elif QSPLIT
#define S_DPAS(sacc_b, kk, kb)                                                                         \
  {                                                                                                   \
    const short8 qa = as_short8(intel_sub_group_block_read_us8(&qs[(rb * NDA + (kk)) * 128]));        \
    const short8 ql = as_short8(intel_sub_group_block_read_us8(&qsl[(rb * NDA + (kk)) * 128]));       \
    sacc_b = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(vload8(0, kb)), sacc_b);            \
    sacc_b = intel_sub_group_bf16_bf16_matrix_mad_k16(ql, as_int8(vload8(0, kb)), sacc_b);            \
  }
#else
#define S_DPAS(sacc_b, kk, kb)                                                                         \
  {                                                                                                   \
    const short8 qa = as_short8(intel_sub_group_block_read_us8(&qs[(rb * NDA + (kk)) * 128]));        \
    sacc_b = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(vload8(0, kb)), sacc_b);            \
  }
#endif
// ABL_SDUP: the index-formed scores of key atom `kbase` (small, finite: the mask and exp stay real).
#define S_INDEX(kbase) (convert_float8((uint8)((kbase) + l) ^ (uint8)(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u)) * 1e-6f)
// s = S * 1/16 (KV8: (S * k_scale) * 1/16) for the keys kbase + lane, -INF past each row's bound.
#if KV8
#define S_MASK(sacc_b, kbase)                                                                          \
  {                                                                                                   \
    const int key = (int)((kbase) + l);                                                               \
    const float sk = (uint)key < end ? KSC(key) : 0.0f;                                               \
    _Pragma("unroll") for (uint r = 0; r < 8u; ++r)                                                   \
      sacc_b[r] = key <= bnd[r] ? (sacc_b[r] * sk) * SSCALE : -INFINITY;                              \
  }
#else
#define S_MASK(sacc_b, kbase)                                                                          \
  {                                                                                                   \
    const int key = (int)((kbase) + l);                                                               \
    _Pragma("unroll") for (uint r = 0; r < 8u; ++r) sacc_b[r] = key <= bnd[r] ? sacc_b[r] * SSCALE : -INFINITY; \
  }
#endif
// The partial write, or (ABL_OUT) none: behind a test the compiler cannot decide (pos is never ~0).
#define OUT_OK (!ABL_OUT || pos == 0xFFFFFFFFu)

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ENTRY(__global const uint* restrict ctrl, __global const float* restrict attn_q, KV_ARGS,
                    __global float* restrict attn_part) {
  const uint j = get_group_id(0), blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = get_sub_group_id(), l = get_sub_group_local_id();
  const uint k = sgid / SPS;                    // key slice
  const uint qi = sgid % SPS;                   // place in the slice
  const bool helper = qi >= RB * DS;            // PVX: a P V-only sub-group (no row block)
  const uint rb = helper ? 0u : qi / DS;        // row block
  const uint ds = qi % DS;                      // dim half (PAIR: key atom)

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;
  const uint s0 = v2_ppw(pos + 1), s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
  if (!live0 && !live1) return;                 // uniform over the work-group, before any barrier

  // q as bf16 A atoms in SLM: qs[(rb * 16 + kk) * 128 + r * 16 + lane] = q[row r of rb][16 kk + lane].
  __local ushort qs[RB * NDA * 128];
#if QSPLIT
  __local ushort qsl[RB * NDA * 128];
#endif
  for (uint i = lid; i < RB * NDA * 128; i += WG) {
    const uint b = i / (NDA * 128), kk = (i / 128) % NDA, r = (i / 16) % 8, e = i % 16;
    uint qhl, m;
    bool ok;
    row_of(b, r, &qhl, &m, &ok);
    const float qv = ok && m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + 16 * kk + e] : 0.0f;
    qs[i] = q_hi(qv);
#if QSPLIT
    qsl[i] = q_lo(qv);
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // This sub-group's rows: (q head, row) per component, the row's bound in each group (-1 = not in it).
  uint8 qhv;
  int8 bnd0, bnd1;
  uint8 mv;
  int8 okv;
#define ROWS(r)                                                                         \
  {                                                                                     \
    uint qq, mm;                                                                        \
    bool oo;                                                                            \
    row_of(rb, r, &qq, &mm, &oo);                                                       \
    oo = oo && mm < n_act;                                                              \
    qhv.s##r = qq;                                                                      \
    mv.s##r = mm;                                                                       \
    okv.s##r = oo ? 1 : 0;                                                              \
    bnd0.s##r = oo && mm < msplit ? (int)(pos + mm) : -1;                               \
    bnd1.s##r = oo && mm >= msplit ? (int)(pos + mm) : -1;                              \
  }
  FOR8(ROWS)
#undef ROWS

#if PAIR
#define NAM ((NDA + SPS - 1) / SPS)             /* the most dim atoms a sub-group owns */
  const uint a0 = (NDA * qi) / SPS, na = (NDA * (qi + 1u)) / SPS - a0;   // its dim atoms [a0, a0 + na)
  float8 o[RB][NAM];                            // [row block][owned atom]
#pragma unroll
  for (uint r2 = 0; r2 < RB; ++r2)
#pragma unroll
    for (uint a = 0; a < NAM; ++a) o[r2][a] = (float8)(0.0f);
  __local float sx[SX_SGS * 128];               // [sub-group][r * 16 + lane]: its key atom of S
  __local ushort spp[KS * RB * 4 * 128];        // [slice][row block][pa 0, pa 1, pl 0, pl 1][r * 16 + lane]
  __local float spc[KS * RB * 128];             // [slice][row block][r * 16 + lane]: the step's corr
#else
  float8 o[NDD];
#pragma unroll
  for (uint dd = 0; dd < NDD; ++dd) o[dd] = (float8)(0.0f);
#endif
  float8 mrow = (float8)(-INFINITY), lsum = (float8)(0.0f);

  for (uint g = 0; g < 2; ++g) {
    const bool live = g ? live1 : live0;
    if (!live) continue;                        // uniform
    const uint s = g ? s1 : s0;
    const uint end = pos + (g ? n_act : msplit);
    const int8 bnd = g ? bnd1 : bnd0;
    const uint bstart = blk * s;
    const uint bend = min(bstart + s, end);
    const uint ntile = (bend - bstart + KT - 1) / KT;
#if PAIR
    const uint nit = (ntile + KS - 1) / KS;     // uniform over the work-group: two barriers a step
    for (uint it = 0; it < nit; ++it) {
      const uint ti = k + it * KS;
      const uint t0 = bstart + ti * KT;
      const bool act = ti < ntile, qk = act && !helper;
#if PVX
      if (helper && act) {                      // the step's V rows into the cache while the pairs compute S
        for (uint c = qi - RB * DS; c < HD * (KV8 ? 1u : 2u) / 64u; c += PVX)
          intel_sub_group_2d_block_prefetch_16b_32r16x2c(
              (__global void*)((__global const char*)kv_v + (size_t)j * HD * (KV8 ? 1u : 2u)), HD * (KV8 ? 1 : 2),
              (int)end, KV_HEADS * HD * (KV8 ? 1 : 2), (int2)((int)(32u * c), (int)t0));
      }
#endif
      // ---- S's key atom ds, scaled and masked, swapped with the pair's other atom ----
      float8 so = (float8)(0.0f);
      if (qk) {
        S_ATOM(so, t0 + 16u * ds)
        S_MASK(so, t0 + 16u * ds)
        intel_sub_group_block_write8((__local uint*)&sx[sgid * 128u], as_uint8(so));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (qk) {
        const float8 sp = as_float8(intel_sub_group_block_read8((__local const uint*)&sx[(sgid ^ 1u) * 128u]));
        float8 sacc[NKA];
        sacc[0] = ds == 0u ? so : sp;
        sacc[1] = ds == 0u ? sp : so;
        float8 tmax = (float8)(-INFINITY);
#pragma unroll
        for (uint b = 0; b < NKA; ++b) tmax = fmax(tmax, sacc[b]);
#if KV8
        float svl[NKA];
#pragma unroll
        for (uint b = 0; b < NKA; ++b) {
          const uint key = t0 + 16u * b + l;
          svl[b] = key < end ? VSC(key) : 0.0f;
        }
#endif
#else
    for (uint ti = k; ti < ntile; ti += KS) {   // uniform per sub-group; no barrier inside
      const uint t0 = bstart + ti * KT;
      {
        // ---- S = q K^T over 16 dim atoms ----
        float8 sacc[NKA];
#pragma unroll
        for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#if ABL_SDUP
        if (ds == 1u) {
#pragma unroll
          for (uint b = 0; b < NKA; ++b) sacc[b] = S_INDEX(t0 + 16u * b);
        } else
#endif
        {
#if KV8
#pragma unroll
          for (uint b = 0; b < NKA; ++b) S_ATOM(sacc[b], t0 + 16u * b)
#else
#pragma unroll
          for (uint kk = 0; kk < NDA; ++kk) {   // main's order: q's atom read once for both key atoms
#if !ABL_QK
            const short8 qa = as_short8(intel_sub_group_block_read_us8(&qs[(rb * NDA + kk) * 128]));
#if QSPLIT
            const short8 ql = as_short8(intel_sub_group_block_read_us8(&qsl[(rb * NDA + kk) * 128]));
#endif
#endif
#pragma unroll
            for (uint b = 0; b < NKA; ++b) {
#if ABL_SHALF
              if (b != ds) continue;            // uniform per sub-group
#endif
              uint kb[8];
              K_READ16(kb, kk, t0 + 16u * b)
#if ABL_QK
              sacc[b] += as_float8(vload8(0, kb) & 0x3F80FFFFu) * 1e-30f;
#else
              sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(vload8(0, kb)), sacc[b]);
#if QSPLIT
              sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(ql, as_int8(vload8(0, kb)), sacc[b]);
#endif
#endif
            }
          }
#if ABL_SHALF
#pragma unroll
          for (uint b = 0; b < NKA; ++b)
            if (b != ds) sacc[b] = S_INDEX(t0 + 16u * b);
#endif
#endif
        }
        // ---- scale, mask, the row max (component r = A row r, lane = key t0 + 16 b + l) ----
        float8 tmax = (float8)(-INFINITY);
#if KV8
        float svl[NKA];
#endif
#pragma unroll
        for (uint b = 0; b < NKA; ++b) {
          S_MASK(sacc[b], t0 + 16u * b)
#if KV8
          const uint key = t0 + 16u * b + l;
          svl[b] = key < end ? VSC(key) : 0.0f;
#endif
          tmax = fmax(tmax, sacc[b]);
        }
#endif  // PAIR
        float8 mnew;
#if ABL_SOFT
        mnew = fmax(mrow, tmax);
#else
#define RED(r) mnew.s##r = fmax(mrow.s##r, sub_group_reduce_max(tmax.s##r));
        FOR8(RED)
#undef RED
#endif
        // corr / p with the whole-skip of a row that has seen no key yet (header)
        const int8 seen = isfinite(mnew);       // -1 where mnew is finite
#if ABL_SOFT
        const float8 corr = (float8)(1.0f);
#else
        const float8 corr = select((float8)(1.0f), exp(mrow - mnew), seen);
#endif
        float8 psum = (float8)(0.0f);
        short8 pa[NKA];
#if PSPLIT
        short8 pl[NKA];                         // P's lo half (header, PSPLIT)
#endif
#pragma unroll
        for (uint b = 0; b < NKA; ++b) {
#if ABL_SOFT
          const float8 p = sacc[b] - mnew;
#else
          const float8 p = select((float8)(0.0f), exp(sacc[b] - mnew), seen);
#endif
          psum += p;
#if KV8
          const float8 pv = p * svl[b];
#else
          const float8 pv = p;
#endif
#if PSPLIT
#define CV(r)                                                   \
  {                                                             \
    const ushort hi = rne_bf16(pv.s##r);                        \
    pa[b].s##r = as_short(hi);                                  \
    pl[b].s##r = as_short(rne_bf16(pv.s##r - bf16f(hi)));       \
  }
#else
#define CV(r) pa[b].s##r = as_short(rne_bf16(pv.s##r));
#endif
          FOR8(CV)
#undef CV
        }
        float8 rsum;
#if ABL_SOFT
        rsum = psum;
#else
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
        FOR8(SUM)
#undef SUM
#endif
        lsum = select(lsum, lsum * corr + rsum, seen);
        mrow = select(mrow, mnew, seen);
#if PAIR
        // publish the row block's P and corr (ds 0 the hi halves and corr, ds 1 the lo halves: equal in the pair)
        if (ds == 0u) {
          intel_sub_group_block_write_us8(&spp[((k * RB + rb) * 4u + 0u) * 128u], as_ushort8(pa[0]));
          intel_sub_group_block_write_us8(&spp[((k * RB + rb) * 4u + 1u) * 128u], as_ushort8(pa[1]));
          intel_sub_group_block_write8((__local uint*)&spc[(k * RB + rb) * 128u], as_uint8(corr));
        } else {
          intel_sub_group_block_write_us8(&spp[((k * RB + rb) * 4u + 2u) * 128u], as_ushort8(pl[0]));
          intel_sub_group_block_write_us8(&spp[((k * RB + rb) * 4u + 3u) * 128u], as_ushort8(pl[1]));
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (act) {
        // ---- O += P V: this sub-group's dim atoms, every row block ----
        short8 pr[RB][4];
#pragma unroll
        for (uint r2 = 0; r2 < RB; ++r2) {
#pragma unroll
          for (uint i = 0; i < 4u; ++i)
            pr[r2][i] = as_short8(intel_sub_group_block_read_us8(&spp[((k * RB + r2) * 4u + i) * 128u]));
          const float8 cr = as_float8(intel_sub_group_block_read8((__local const uint*)&spc[(k * RB + r2) * 128u]));
#pragma unroll
          for (uint a = 0; a < NAM; ++a) o[r2][a] *= cr;
        }
#pragma unroll
        for (uint a = 0; a < NAM; ++a) {
          if (a >= na) break;                   // uniform
          const uint dim0 = 16u * (a0 + a);
          uint vb[16];                          // 32 keys x 16 dims as key pairs (VNNI; rows past `end`: 0)
#if KV8
          uint v4[8];
          if (ABL_V) {
            for (uint i = 0; i < 8u; ++i) v4[i] = ((t0 + 4u * i) * 0x01010101u) ^ (dim0 + l);
          } else {
            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD, (int2)((int)(j * HD + dim0), (int)t0), v4);
          }
#pragma unroll
          for (uint i = 0; i < 8u; ++i) {
#if ABL_VCVT
            vb[2u * i] = v4[i];
            vb[2u * i + 1u] = v4[i] ^ v4[(i + 1u) & 7u];
#else
            const uint2 pw = kv8_bf4(v4[i]);
            vb[2u * i] = pw.x;
            vb[2u * i + 1u] = pw.y;
#endif
          }
#else
          if (ABL_V) {
            for (uint i = 0; i < 16u; ++i) vb[i] = ((t0 + 2u * i) * 0x00010001u) ^ (dim0 + l);
          } else {
            intel_sub_group_2d_block_read_transform_16b_32r16x1c(
                (__global void*)kv_v, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2, (int2)((int)(j * HD + dim0), (int)t0),
                vb);
          }
#endif
#pragma unroll
          for (uint r2 = 0; r2 < RB; ++r2) {
#if ABL_PV
            o[r2][a] += as_float8((as_uint8(convert_int8(pr[r2][0]) ^ convert_int8(pr[r2][3])) ^ vload8(0, vb) ^
                                   vload8(0, vb + 8u)) & 0x3F80FFFFu) * 1e-30f;
#else
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][0], as_int8(vload8(0, vb)), o[r2][a]);
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][1], as_int8(vload8(0, vb + 8u)), o[r2][a]);
#if PSPLIT
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][2], as_int8(vload8(0, vb)), o[r2][a]);
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][3], as_int8(vload8(0, vb + 8u)), o[r2][a]);
#endif
#endif
          }
        }
      }
#else
#pragma unroll
        for (uint dd = 0; dd < NDD; ++dd) o[dd] *= corr;
        // ---- O += P V over this sub-group's dims [ds DSPLIT, + DSPLIT) ----
#pragma unroll
        for (uint b = 0; b < NKA; b += 2)
#pragma unroll
          for (uint dd = 0; dd < NDD; dd += 2) {
            const uint dim0 = ds * DSPLIT + 16u * dd;
            uint vb[32];   // [c * 16 + jj]: c = dim half of 32 dims, jj = key pair of 32 keys
#if KV8
#pragma unroll
            for (uint c = 0; c < 2u; ++c) {
              // 32 keys x 16 dims of int8 V by the 8-bit VNNI-transform read (lane = dim; dword i = keys 4 i .. 4 i + 3,
              // rows past `end` read as 0), then each key pair as two exact bf16
              uint v4[8];
              if (ABL_V || (ABL_VONE && rb != 0u)) {
                for (uint i = 0; i < 8u; ++i) v4[i] = ((t0 + 16u * b + 4u * i) * 0x01010101u) ^ (dim0 + 16u * c + l);
              } else {
                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                    (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD,
                    (int2)((int)(j * HD + dim0 + 16u * c), (int)(t0 + 16u * b)), v4);
              }
#pragma unroll
              for (uint i = 0; i < 8u; ++i) {
#if ABL_VCVT
                vb[c * 16u + 2u * i] = v4[i];
                vb[c * 16u + 2u * i + 1u] = v4[i] ^ v4[(i + 1u) & 7u];
#else
                const uint2 pw = kv8_bf4(v4[i]);
                vb[c * 16u + 2u * i] = pw.x;
                vb[c * 16u + 2u * i + 1u] = pw.y;
#endif
              }
            }
#else
            if (ABL_V || (ABL_VONE && rb != 0u)) {
              for (uint i = 0; i < 32u; ++i) vb[i] = ((t0 + 16u * b + i) * 0x00010001u) ^ (dim0 + l);
            } else {
              intel_sub_group_2d_block_read_transform_16b_32r16x2c(
                  (__global void*)kv_v, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
                  (int2)((int)(j * HD + dim0), (int)(t0 + 16u * b)), vb);
            }
#endif
#if ABL_PV
#pragma unroll
            for (uint c = 0; c < 2u; ++c)
              o[dd + c] += as_float8((as_uint8(convert_int8(pa[b]) ^ convert_int8(pl[b + 1])) ^
                                      vload8(0, vb + c * 16u) ^ vload8(0, vb + c * 16u + 8u)) & 0x3F80FFFFu) * 1e-30f;
#else
#pragma unroll
            for (uint c = 0; c < 2u; ++c)
#pragma unroll
              for (uint kh = 0; kh < 2u; ++kh)
                o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    pa[b + kh], as_int8(vload8(0, vb + c * 16u + 8u * kh)), o[dd + c]);
#if PSPLIT
#pragma unroll
            for (uint c = 0; c < 2u; ++c)
#pragma unroll
              for (uint kh = 0; kh < 2u; ++kh)
                o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    pl[b + kh], as_int8(vload8(0, vb + c * 16u + 8u * kh)), o[dd + c]);
#endif
#endif
          }
      }
#endif  // PAIR
    }
  }

  // ---- the slice merge (slice order) and the partial write ----
#if PAIR
  // Split as P V was (header): the dim owners merge O for every row block, the pair's ds 0 its sums and maxima.
  __local float macc[KS > 1 ? RB * 8 * HD : 1]; // [rb][r][dim] (unused at the probe's KS 1)
  __local float mown[KS * RB * 8];              // [slice][rb][r]: the slice's own row maximum
  __local float msm[RB * 8];                    // [rb][r]: the merged row sum so far
  if (ds == 0u && !helper) {
#define PM(r) if (l == r) mown[(k * RB + rb) * 8u + r] = mrow.s##r;
    FOR8(PM)
#undef PM
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint k2 = 0; k2 < KS; ++k2) {
    if (k == k2) {
#if KS > 1
      if (k2 > 0) {
#pragma unroll
        for (uint r2 = 0; r2 < RB; ++r2) {
#define MRG(r)                                                                          \
  {                                                                                     \
    float ma = mown[r2 * 8u + r];                                                       \
    for (uint k3 = 1; k3 < k2; ++k3) ma = fmax(ma, mown[(k3 * RB + r2) * 8u + r]);      \
    const float mb = mown[(k2 * RB + r2) * 8u + r];                                     \
    const float nmx = fmax(ma, mb);                                                     \
    if (nmx > -INFINITY) {                                                              \
      const float ca = exp(ma - nmx), cb = exp(mb - nmx);                               \
      _Pragma("unroll") for (uint a = 0; a < NAM; ++a)                                  \
        if (a < na) {                                                                   \
          const float aa = macc[(r2 * 8u + r) * HD + 16u * (a0 + a) + l];               \
          o[r2][a].s##r = fma(aa, ca, o[r2][a].s##r * cb);                              \
        }                                                                               \
      if (r2 == rb && ds == 0u && !helper) {                                            \
        lsum.s##r = fma(msm[rb * 8u + r], ca, lsum.s##r * cb);                          \
        mrow.s##r = nmx;                                                                \
      }                                                                                 \
    }                                                                                   \
  }
          FOR8(MRG)
#undef MRG
        }
      }
      if (k2 + 1 < KS) {
#pragma unroll
        for (uint r2 = 0; r2 < RB; ++r2)
#pragma unroll
          for (uint a = 0; a < NAM; ++a)
            if (a < na) {
#define PUT(r) macc[(r2 * 8u + r) * HD + 16u * (a0 + a) + l] = o[r2][a].s##r;
              FOR8(PUT)
#undef PUT
            }
        if (ds == 0u && l == 0u && !helper) {
#define PUTS(r) msm[rb * 8u + r] = lsum.s##r;
          FOR8(PUTS)
#undef PUTS
        }
      } else
#endif
      {
#pragma unroll
        for (uint r2 = 0; r2 < RB; ++r2) {
#define OUT(r)                                                                          \
  {                                                                                     \
    uint qq, mm;                                                                        \
    bool oo;                                                                            \
    row_of(r2, r, &qq, &mm, &oo);                                                       \
    if (oo && mm < n_act && (mm < msplit ? live0 : live1) && OUT_OK) {                  \
      __global float* restrict out = attn_part + (((size_t)(j * GQA + qq) * TGT + blk) * M + mm) * PART; \
      _Pragma("unroll") for (uint a = 0; a < NAM; ++a)                                  \
        if (a < na) out[2 + 16u * (a0 + a) + l] = o[r2][a].s##r;                        \
      if (r2 == rb && ds == 0u && l == 0u && !helper) { out[0] = mrow.s##r; out[1] = lsum.s##r; } \
    }                                                                                   \
  }
          FOR8(OUT)
#undef OUT
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#else
#define OUT(r)                                                                          \
  if (okv.s##r && ((mv.s##r < msplit) ? live0 : live1) && OUT_OK) {                     \
    __global float* restrict out =                                                      \
        attn_part + (((size_t)(j * GQA + qhv.s##r) * TGT + blk) * M + mv.s##r) * PART;  \
    _Pragma("unroll") for (uint dd = 0; dd < NDD; ++dd) out[2 + ds * DSPLIT + 16u * dd + l] = o[dd].s##r; \
    if (ds == 0 && l == 0) { out[0] = mrow.s##r; out[1] = lsum.s##r; }                  \
  }
#if KS == 1
  FOR8(OUT)
#else
  __local float macc[RB * 8 * HD];              // [rb][r][dim]
  __local float mms[RB * DS * 8 * 2];           // [rb][ds][r]{m, sm}: one copy per dim half
  for (uint k2 = 0; k2 < KS; ++k2) {
    if (k == k2) {
      __local float* ms = &mms[((rb * DS + ds) * 8) * 2];
      if (k2 > 0) {                             // merge the slices before this one into mine
#define MRG(r)                                                                          \
  {                                                                                     \
    const float ma = ms[2 * r], sa = ms[2 * r + 1];                                     \
    const float nmx = fmax(ma, mrow.s##r);                                              \
    if (nmx > -INFINITY) {                                                              \
      const float ca = exp(ma - nmx), cb = exp(mrow.s##r - nmx);                        \
      _Pragma("unroll") for (uint dd = 0; dd < NDD; ++dd) {                                          \
        const float aa = macc[(rb * 8 + r) * HD + ds * DSPLIT + 16u * dd + l];          \
        o[dd].s##r = fma(aa, ca, o[dd].s##r * cb);                                      \
      }                                                                                 \
      lsum.s##r = fma(sa, ca, lsum.s##r * cb);                                          \
      mrow.s##r = nmx;                                                                  \
    }                                                                                   \
  }
        FOR8(MRG)
#undef MRG
      }
      if (k2 + 1 < KS) {
#define PUT(r)                                                                          \
  _Pragma("unroll") for (uint dd = 0; dd < NDD; ++dd) macc[(rb * 8 + r) * HD + ds * DSPLIT + 16u * dd + l] = o[dd].s##r; \
  if (l == 0) { ms[2 * r] = mrow.s##r; ms[2 * r + 1] = lsum.s##r; }
        FOR8(PUT)
#undef PUT
      } else {
        FOR8(OUT)
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#endif
#undef OUT
#endif  // PAIR
}
