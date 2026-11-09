// verify_dpas_ablate.cl (3l): src/kernels/attn_verify_dpas.cl at main 865da900 (its DPAS body; the EAGER build
// dropped) with one part of the work removed per define - TIMING ONLY, outputs are meaningless:
//   ABL_K      1: no K loads (registers from the key / atom)      ABL_KCVT 1: K loaded, no int8 -> bf16 widening (KV8)
//   ABL_V      1: no V loads (registers from the key / dim)       ABL_VCVT 1: V loaded, no int8 -> bf16 widening (KV8)
//   ABL_SCALE  1: no scale loads (1.0, KV8)
//   ABL_QK     1: no q K^T DPAS (the scores gather the K registers)
//   ABL_SOFT   1: no row max / exp / sum (no sub-group reductions; p = s - m)
//   ABL_PV     1: no P V DPAS (O gathers P and the V registers)
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
// ABL_KONE / ABL_VONE 1: only row block 0's sub-groups load (and widen) K / V; the others take registers
// ABL_PF d (candidate, not an ablation): the slice's tile d steps ahead prefetched into the cache, its K and V rows in
// 64-byte columns shared round-robin by the row blocks' sub-groups; ABL_PFV 1: V only, ABL_PFK 1: K only
#ifndef ABL_PF
#define ABL_PF 0
#endif
#ifndef ABL_PFK
#define ABL_PFK 1
#endif
#ifndef ABL_PFV
#define ABL_PFV 1
#endif
#ifndef ABL_KONE
#define ABL_KONE 0
#endif
#ifndef ABL_VONE
#define ABL_VONE 0
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
// ABL_SYNC (candidate): 1 a work-group barrier after every tile step (the row blocks' sub-groups walk the same
// tiles in step, so the K / V lines one brings are hits for the others), 2 the split barrier (arrive after the
// step's loads are consumed, wait before the next step's)
#ifndef ABL_SYNC
#define ABL_SYNC 0
#endif
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
#ifndef EAGER
#define EAGER 0
#endif
#ifdef FA_Q_HEADS
#error "attn_verify_dpas: Qwen3.8's 24 / 4 heads only (spec 28c §3); other heads keep attn_v2"
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
#define NSG (RB * DS * KS)
#define WG (SG * NSG)
#if WG > 1024
#error "attn_verify_dpas: more than 1024 lanes a work-group (RB x DS x KS sub-groups of 16)"
#endif
#define QS_BYTES (RB * NDA * 128 * 2 * (1 + QSPLIT))
#define MERGE_BYTES (KS > 1 ? RB * 8 * HD * 4 + RB * DS * 8 * 2 * 4 : 0)
#if QS_BYTES + MERGE_BYTES > 131072
#error "attn_verify_dpas: SLM over 128 KB (q staging + slice merge)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
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

// The slice merge / attn_reduce_v2's block merge, one value: (ma, sa, aa) <- (ma, sa, aa) + (mb, sb, ab).
inline void merge1(float* ma, float* sa, float* aa, float mb, float sb, float ab) {
  const float nmx = fmax(*ma, mb);
  if (nmx > -INFINITY) {
    const float a = exp(*ma - nmx), b = exp(mb - nmx);
    *sa = fma(*sa, a, sb * b);
    *aa = fma(*aa, a, ab * b);
    *ma = nmx;
  }
}

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

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ENTRY(__global const uint* restrict ctrl, __global const float* restrict attn_q, KV_ARGS,
                    __global float* restrict attn_part) {
  const uint j = get_group_id(0), blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = get_sub_group_id(), l = get_sub_group_local_id();
  const uint k = sgid / (RB * DS);              // key slice
  const uint rb = (sgid / DS) % RB;             // row block
  const uint ds = sgid % DS;                    // dim half

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

  float8 o[NDD];
#pragma unroll
  for (uint dd = 0; dd < NDD; ++dd) o[dd] = (float8)(0.0f);
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
#if ABL_SYNC
    const uint nit = (ntile + KS - 1) / KS;
    for (uint it = 0; it < nit; ++it) {
#if ABL_SYNC == 2
      if (it > 0) intel_work_group_barrier_wait(0);
#endif
      const uint ti = k + it * KS;
      if (ti < ntile) {
#else
    for (uint ti = k; ti < ntile; ti += KS) {   // uniform per sub-group; no barrier inside
#endif
      const uint t0 = bstart + ti * KT;
#if ABL_PF
      if (ti + ABL_PF * KS < ntile) {
        const uint tp = t0 + ABL_PF * KS * KT;
#define RB_BYTES (HD * (KV8 ? 1 : 2))                   /* one key's row of one kv head */
#define NPF (RB_BYTES / 64)                             /* 64-byte columns per row */
        for (uint c = rb; c < 2 * NPF; c += RB) {
          const bool isv = c >= NPF;
          if ((isv && !ABL_PFV) || (!isv && !ABL_PFK)) continue;
          intel_sub_group_2d_block_prefetch_16b_32r16x2c(
              (__global void*)((isv ? (__global const char*)kv_v : (__global const char*)kv_k) + (size_t)j * RB_BYTES),
              RB_BYTES, (int)end, KV_HEADS * RB_BYTES, (int2)((int)(32u * (c % NPF)), (int)tp));
        }
      }
#endif
      // ---- S = q K^T over 16 dim atoms ----
      float8 sacc[NKA];
#pragma unroll
      for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#pragma unroll
      for (uint kk = 0; kk < NDA; ++kk) {
        const short8 qa = as_short8(intel_sub_group_block_read_us8(&qs[(rb * NDA + kk) * 128]));
#if QSPLIT
        const short8 ql = as_short8(intel_sub_group_block_read_us8(&qsl[(rb * NDA + kk) * 128]));
#endif
#pragma unroll
        for (uint b = 0; b < NKA; ++b) {
          uint kb[8];
#if KV8
          const uint key = t0 + 16u * b + l;
          if (key < end) {
#if ABL_K
            const char16 c = as_char16((uint4)(key * 0x01010101u, kk, key ^ kk, key + kk));
#elif ABL_KONE
            if (rb != 0u) {
              for (uint z = 0; z < 8u; ++z) kb[z] = (key * 0x00010001u) ^ (kk << 4) ^ z;
              goto kdone;
            }
            const char16 c = vload16(0, kv_k + ((size_t)key * KV_HEADS + j) * HD + 16u * kk);
#else
            const char16 c = vload16(0, kv_k + ((size_t)key * KV_HEADS + j) * HD + 16u * kk);
#endif
#if ABL_KCVT
            const uint4 cw = as_uint4(c);
            kb[0] = cw.s0; kb[1] = cw.s1; kb[2] = cw.s2; kb[3] = cw.s3;
            kb[4] = cw.s0 ^ cw.s1; kb[5] = cw.s1 ^ cw.s2; kb[6] = cw.s2 ^ cw.s3; kb[7] = cw.s3 ^ cw.s0;
#else
            kb[0] = kv8_bf(c.s0) | (kv8_bf(c.s1) << 16);
            kb[1] = kv8_bf(c.s2) | (kv8_bf(c.s3) << 16);
            kb[2] = kv8_bf(c.s4) | (kv8_bf(c.s5) << 16);
            kb[3] = kv8_bf(c.s6) | (kv8_bf(c.s7) << 16);
            kb[4] = kv8_bf(c.s8) | (kv8_bf(c.s9) << 16);
            kb[5] = kv8_bf(c.sa) | (kv8_bf(c.sb) << 16);
            kb[6] = kv8_bf(c.sc) | (kv8_bf(c.sd) << 16);
            kb[7] = kv8_bf(c.se) | (kv8_bf(c.sf) << 16);
#endif
#if ABL_KONE
          kdone:;
#endif
          } else {
            for (uint z = 0; z < 8u; ++z) kb[z] = 0u;
          }
#elif ABL_K
          for (uint z = 0; z < 8u; ++z) kb[z] = ((t0 + 16u * b + l) * 0x00010001u) ^ (kk << 4) ^ z;
#else
          if (ABL_KONE && rb != 0u)
            for (uint z = 0; z < 8u; ++z) kb[z] = ((t0 + 16u * b + l) * 0x00010001u) ^ (kk << 4) ^ z;
          else
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (__global void*)kv_k, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
                (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
#endif
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
      // ---- scale, mask, the row max (component r = A row r, lane = key t0 + 16 b + l) ----
      float8 tmax = (float8)(-INFINITY);
#if KV8
      float svl[NKA];
#endif
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        const int key = (int)(t0 + 16u * b + l);
#if KV8
        const bool in = (uint)key < end;
#if ABL_SCALE
        const float sk = in ? 1.0f : 0.0f;
        svl[b] = in ? 1.0f : 0.0f;
#else
        const float sk = in ? kv8_f16f(k_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
        svl[b] = in ? kv8_f16f(v_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
#endif
#define MASK(r) sacc[b].s##r = key <= bnd.s##r ? (sacc[b].s##r * sk) * SSCALE : -INFINITY;
#else
#define MASK(r) sacc[b].s##r = key <= bnd.s##r ? sacc[b].s##r * SSCALE : -INFINITY;
#endif
        FOR8(MASK)
#undef MASK
        tmax = fmax(tmax, sacc[b]);
      }
      float8 mnew;
#if ABL_SOFT
      mnew = fmax(mrow, tmax);
#else
#define RED(r) mnew.s##r = fmax(mrow.s##r, sub_group_reduce_max(tmax.s##r));
      FOR8(RED)
#undef RED
#endif
      // corr / p with the whole-skip of a row that has seen no key yet (header)
      const int8 seen = isfinite(mnew);         // -1 where mnew is finite
#if ABL_SOFT
      const float8 corr = (float8)(1.0f);
#else
      const float8 corr = select((float8)(1.0f), exp(mrow - mnew), seen);
#endif
      float8 psum = (float8)(0.0f);
      short8 pa[NKA];
#if PSPLIT
      short8 pl[NKA];                           // P's lo half (header, PSPLIT)
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
#if ABL_VONE
            if (rb != 0u) {
              for (uint i = 0; i < 16u; ++i) vb[c * 16u + i] = ((t0 + 16u * b + i) * 0x00010001u) ^ (dim0 + l);
              continue;
            }
#endif
#if ABL_V
            for (uint i = 0; i < 8u; ++i) v4[i] = ((t0 + 16u * b + i) * 0x01010101u) ^ (dim0 + 16u * c + l);
#else
            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD,
                (int2)((int)(j * HD + dim0 + 16u * c), (int)(t0 + 16u * b)), v4);
#endif
#if ABL_VCVT
            for (uint i = 0; i < 8u; ++i) { vb[c * 16u + 2u * i] = v4[i]; vb[c * 16u + 2u * i + 1u] = v4[i] ^ v4[(i + 1u) & 7u]; }
#else
#pragma unroll
            for (uint i = 0; i < 8u; ++i) {
              const uint w = v4[i];
              vb[c * 16u + 2u * i] = kv8_bf(as_char((uchar)(w & 0xFFu))) | (kv8_bf(as_char((uchar)((w >> 8) & 0xFFu))) << 16);
              vb[c * 16u + 2u * i + 1u] =
                  kv8_bf(as_char((uchar)((w >> 16) & 0xFFu))) | (kv8_bf(as_char((uchar)(w >> 24))) << 16);
            }
#endif
          }
#elif ABL_V
          for (uint i = 0; i < 32u; ++i) vb[i] = ((t0 + 16u * b + i) * 0x00010001u) ^ (dim0 + l);
#else
          if (ABL_VONE && rb != 0u)
            for (uint i = 0; i < 32u; ++i) vb[i] = ((t0 + 16u * b + i) * 0x00010001u) ^ (dim0 + l);
          else
            intel_sub_group_2d_block_read_transform_16b_32r16x2c(
                (__global void*)kv_v, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
                (int2)((int)(j * HD + dim0), (int)(t0 + 16u * b)), vb);
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
#endif
#if PSPLIT && !ABL_PV
#pragma unroll
          for (uint c = 0; c < 2u; ++c)
#pragma unroll
            for (uint kh = 0; kh < 2u; ++kh)
              o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                  pl[b + kh], as_int8(vload8(0, vb + c * 16u + 8u * kh)), o[dd + c]);
#endif
        }
#if ABL_SYNC
      }
#if ABL_SYNC == 1
      barrier(CLK_LOCAL_MEM_FENCE);
#else
      intel_work_group_barrier_arrive(0);
#endif
    }
#if ABL_SYNC == 2
    if (nit > 0) intel_work_group_barrier_wait(0);
#endif
#else
    }
#endif
  }

  // ---- the slice merge (slice order) and the partial write ----
#define OUT(r)                                                                          \
  if (okv.s##r && ((mv.s##r < msplit) ? live0 : live1)) {                               \
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
}

