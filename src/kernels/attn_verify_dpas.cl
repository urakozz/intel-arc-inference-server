// attn_verify_dpas.cl - spec 28c: the verify step's attention on DPAS, in v2's blocking and partials.
// Spec: docs/superpowers/specs/2026-10-31-spec28c-verify-attention-by-shape-design.md (§2, §8).
// Plan: docs/superpowers/plans/2026-10-31-spec28c-verify-attn-build.md (Task 2).
//
// `attn_verify_dpas` (bf16 KV) / `attn_verify_dpas_kv8` (int8 KV, KV8 = 1) replace attn_v2.cl's
// `attn_decode_v2` / kv8.cl's `attn_decode_v2_kv8` in an M >= 2 verify list when the capture's rule
// (src/runtime/verify_attn.h) says so. They read the same attn_q and KV cache and write the SAME
// partials, so the reduce is the existing `attn_reduce_v2` (`attn_reduce_v2_kv8`) of the
// `attn_v2_M<M>_T32[_KV8]` binary - no new reduce:
//
//   attn_verify_dpas(ctrl, attn_q, kv_k, kv_v, attn_part)                         grid (4, TGT)
//   attn_verify_dpas_kv8(ctrl, attn_q, kv_k, k_scale, kv_v, v_scale, attn_part)   grid (4, TGT)
//     attn_part fp32 [24][TGT][M][258] {mx (natural log), sm, acc[256]} - attn_v2.cl's meaning
//
// ---------------------------------------------------------------------------------------------
// v2's blocking, exactly (attn_v2.cl's header, change 1 - Review Focus 1 of the plan)
// ---------------------------------------------------------------------------------------------
// Row m of a launch at pos attends to keys [0, pos + m]. Its stride is ppw(pos + m + 1); the rows
// of one launch have at most two strides: rows [0, msplit) at s0 and [msplit, n_act) at s1. For
// row group g (stride s, keys [0, end), end = pos + its last row + 1) work-group (j, blk) walks
// positions [blk s, min((blk + 1) s, end)) - the positions attn_decode_v2 walks for those rows - and
// writes the partial of (q head, blk, m) for every row of the group. attn_reduce_v2 reads row m's
// nb = (pos + m) / ppw(pos + m + 1) + 1 blocks, each of which starts at or before pos + m.
//
// ---------------------------------------------------------------------------------------------
// Inside a work-group
// ---------------------------------------------------------------------------------------------
// The 6 x M (q head, row) pairs of kv head j are the DPAS A rows, 8 to a row block (rb):
//   PACK_H = 1 (`h`, pf_flash_attn's packing): one q head's rows m = 8 i .. 8 i + 7 per row block,
//              RB = 6 ceil(M / 8) blocks (A rows past M are empty);
//   PACK_H = 0 (`g`): the pairs p = qhl M + m packed 8 to a block, RB = ceil(6 M / 8).
// Sub-group s: slice k = s / (RB DS), row block rb = (s / DS) % RB, dim half ds = s % DS, where
// DS = 256 / DSPLIT sub-groups share a row block's scores and own DSPLIT output dims each.
// A walk's tiles are KT keys anchored at the block start: tile i = [bstart + i KT, + KT) (KT divides
// 64, so a tile never crosses the block's end); slice k takes tiles i = k, k + KS, k + 2 KS, ...
// Per tile (pf_flash_attn.cl's idioms):
//   S (8 rows x 16 keys per key atom) = sum over 16 dim atoms of DPAS(q atom, K^T atom): q from SLM
//     (bf16, staged once), K^T by the transposed 32-bit 2D block read (bf16) or one 16-byte load of
//     the key's int8 row as exact bf16 pairs (KV8, pf_flash_attn_kv8's form), surface height `end`;
//   QSPLIT = 1: q = hi + lo, hi = rne_bf16(q), lo = rne_bf16(q - hi), two DPAS per atom (hi first) -
//     ~16 significant bits of q, against fp32 q in v2; QSPLIT = 0: hi alone (prefill's rounding);
//   s = S * 1/16 (KV8: (S * k_scale) * 1/16); a row's keys past its bound pos + m, and every key of a
//     row outside the group being walked, are -INF;
//   online softmax per row in fp32 with `exp` (spec 10 §8): mnew = max(m, max over the tile's keys);
//     when mnew = -INF the row has seen no key yet and the tile is skipped whole (corr 1, p 0 - as
//     v2's waves: exp(-INF - -INF) is the only NaN); else corr = exp(m - mnew), p = exp(s - mnew);
//     sm = sm corr + sum p (fp32 p), O = O corr + P V with P = rne_bf16(p) (KV8: rne_bf16(p v_scale))
//     as the DPAS A operand - S's layout, so P goes to PV with no data movement; V by the
//     VNNI-transform 2D read (KV8: the 8-bit transform read, key pairs widened to exact bf16).
// The KS slices of a row block merge in SLM in slice order 0, 1, ..., KS - 1 (attn_reduce_v2's
// formulas: nmx = max; a = exp(m_a - nmx), b = exp(m_b - nmx); sm = fma(sm_a, a, sm_b b),
// acc = fma(acc_a, a, acc_b b); both -INF: unchanged), the last slice writing the partial.
//
// ---------------------------------------------------------------------------------------------
// Property R (spec 28c N2a: a row bitwise equal at the same key count in any M >= 2 launch)
// ---------------------------------------------------------------------------------------------
// Row m's tiles, their slices and the merge order depend only on its own key count (its stride and
// block start); a tile past its bound but inside the group's walk is all -INF for it, which leaves
// its state bit for bit (corr = exp(0) = 1, p = 0, 0 x V = 0; or the whole-skip above); a DPAS output
// row depends only on its A row; the row max and sum are per row (one component of the sub-group
// reductions). So nothing in a row's result depends on M, its slot, or the other rows - PROVIDED one
// lever setting (KT, KS, DSPLIT, QSPLIT, GRF) serves every M of a KV form (Review Focus 2; one CMake
// line per form). PACK changes a row's slot only. Checked bitwise on the card, not assumed.
//
// ---------------------------------------------------------------------------------------------
// The levers (one define each; tools/probe sweeps them from this source) and what #errors
// ---------------------------------------------------------------------------------------------
//   M       2..8 (16: the probe's record only - no reduce or list exists there)
//   KT      32 | 64            keys per tile (16 is not built: the PV loop takes key atoms in pairs)
//   PACK_H  0 (g) | 1 (h)
//   KS      1 | 2 | 4 | 8      key slices per work-group
//   DSPLIT  256 | 128          output dims per sub-group
//   QSPLIT  0 | 1
//   KV8     0 | 1
//   EAGER   1: the eager build (the retired Mac runner used it) - plain loads, no sub-group
//           function at all: one work-item per (q head, row) runs the same walk (blocks, tiles,
//           slices, masks, the P rounding, the slice merge order, the partial layout) with a scalar
//           tile product (16-dim atoms ascending, hi then lo). Never a production line.
//   GRF     the ocloc option -cl-intel-256-GRF-per-thread on the command line (not a define).
// Work-group = 16 RB DS KS lanes: at most 512 (a 256-GRF work-group's 32 sub-groups); the probe
// builds the 1024-lane shapes at 128 GRF only. SLM: q (RB x 4 KB, x 2 at QSPLIT) + the slice merge
// (RB x 8 KB at KS > 1) <= 128 KB.
//
// Heads: Qwen3.8's 24 q / 4 kv (Agnes shares them) only; another model's heads keep v2 (the rule).
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
#if !defined(KT) || !defined(PACK_H) || !defined(KS) || !defined(DSPLIT) || !defined(QSPLIT) || !defined(KV8)
#error "attn_verify_dpas: KT, PACK_H, KS, DSPLIT, QSPLIT and KV8 must all be defined"
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

#if EAGER
// =============================================================================================
// EAGER: the same walk per (q head, row), one work-item each, plain loads (header).
// =============================================================================================
// K / V element i of key p (kv head j in scope); under KV8 the int8 value - the key's K scale goes on
// the score and its V scale into P, as in the DPAS body.
#if KV8
#define K_AT(p, i) ((float)kv_k[((size_t)(p) * KV_HEADS + j) * HD + (i)])
#define V_AT(p, i) ((float)kv_v[((size_t)(p) * KV_HEADS + j) * HD + (i)])
#else
#define K_AT(p, i) bf16f(kv_k[((size_t)(p) * KV_HEADS + j) * HD + (i)])
#define V_AT(p, i) bf16f(kv_v[((size_t)(p) * KV_HEADS + j) * HD + (i)])
#endif
// The eager work-group: 64 work-items whatever the levers (a Mac GPU's limit is 256); work-item i runs
// the pairs i, i + 64, ... (pair = qhl M + m).
#define EAGER_WG 64

__attribute__((reqd_work_group_size(EAGER_WG, 1, 1)))
__kernel void ENTRY(__global const uint* restrict ctrl, __global const float* restrict attn_q, KV_ARGS,
                    __global float* restrict attn_part) {
  const uint j = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;
  const uint s0 = v2_ppw(pos + 1), s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  for (uint pr = lid; pr < GQA * M; pr += EAGER_WG) {
    const uint qhl = pr / M, m = pr % M;
    if (m >= n_act) continue;
    const uint g = m < msplit ? 0u : 1u;
    const uint s = g ? s1 : s0;
    const uint end = pos + (g ? n_act : msplit);
    const uint bstart = blk * s;
    if (bstart >= end) continue;                 // v2 writes no partial for a group not live here
    const uint bend = min(bstart + s, end);
    const uint ntile = (bend - bstart + KT - 1) / KT;
    const uint bound = pos + m;
    const uint qh = j * GQA + qhl;
    __global const float* q = attn_q + ((size_t)m * Q_HEADS + qh) * HD;
    float tm = -INFINITY, tsm = 0.0f, tacc[HD];   // the merged state, slices 0 .. k - 1
    for (uint d = 0; d < HD; ++d) tacc[d] = 0.0f;
    for (uint k = 0; k < KS; ++k) {
      float rm = -INFINITY, rsm = 0.0f, racc[HD];
      for (uint d = 0; d < HD; ++d) racc[d] = 0.0f;
      for (uint i = k; i < ntile; i += KS) {
        const uint t0 = bstart + i * KT;
        float sc[KT], pr16[KT];
        float tmax = -INFINITY;
        for (uint t = 0; t < KT; ++t) {
          const uint key = t0 + t;
          float a = 0.0f;
          if (key < end) {
            for (uint kk = 0; kk < NDA; ++kk) {   // one DPAS atom: 16 dims, hi then lo
              float th = 0.0f;
              for (uint e = 0; e < 16; ++e)
                th = fma(bf16f(q_hi(q[16 * kk + e])), K_AT(key, 16 * kk + e), th);
              a += th;
#if QSPLIT
              float tl = 0.0f;
              for (uint e = 0; e < 16; ++e)
                tl = fma(bf16f(q_lo(q[16 * kk + e])), K_AT(key, 16 * kk + e), tl);
              a += tl;
#endif
            }
          }
#if KV8
          const float sk = key < end ? kv8_f16f(k_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
          sc[t] = key <= bound ? (a * sk) * SSCALE : -INFINITY;
#else
          sc[t] = key <= bound ? a * SSCALE : -INFINITY;
#endif
          tmax = fmax(tmax, sc[t]);
        }
        const float mnew = fmax(rm, tmax);
        if (!(mnew > -INFINITY)) continue;       // nothing seen yet: the tile is skipped whole
        const float corr = exp(rm - mnew);
        float psum = 0.0f;
        for (uint t = 0; t < KT; ++t) {
          const float p = exp(sc[t] - mnew);
          psum += p;
#if KV8
          const uint key = t0 + t;
          const float sv = key < end ? kv8_f16f(v_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
          pr16[t] = bf16f(rne_bf16(p * sv));
#else
          pr16[t] = bf16f(rne_bf16(p));
#endif
        }
        rsm = rsm * corr + psum;
        rm = mnew;
        for (uint d = 0; d < HD; ++d) {
          float o = racc[d] * corr;
          for (uint t = 0; t < KT; ++t) {
            const uint key = t0 + t;
            const float vv = key < end ? V_AT(key, d) : 0.0f;
            o = fma(pr16[t], vv, o);
          }
          racc[d] = o;
        }
      }
      if (k == 0) {
        tm = rm; tsm = rsm;
        for (uint d = 0; d < HD; ++d) tacc[d] = racc[d];
      } else {
        for (uint d = 0; d < HD; ++d) {
          float mm = tm, ss = tsm;
          merge1(&mm, &ss, &tacc[d], rm, rsm, racc[d]);
          if (d == HD - 1) { tm = mm; tsm = ss; }
        }
      }
    }
    __global float* restrict out = attn_part + (((size_t)qh * TGT + blk) * M + m) * PART;
    out[0] = tm;
    out[1] = tsm;
    for (uint d = 0; d < HD; ++d) out[2 + d] = tacc[d];
  }
}

#else  // !EAGER
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
    for (uint ti = k; ti < ntile; ti += KS) {   // uniform per sub-group; no barrier inside
      const uint t0 = bstart + ti * KT;
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
            const char16 c = vload16(0, kv_k + ((size_t)key * KV_HEADS + j) * HD + 16u * kk);
            kb[0] = kv8_bf(c.s0) | (kv8_bf(c.s1) << 16);
            kb[1] = kv8_bf(c.s2) | (kv8_bf(c.s3) << 16);
            kb[2] = kv8_bf(c.s4) | (kv8_bf(c.s5) << 16);
            kb[3] = kv8_bf(c.s6) | (kv8_bf(c.s7) << 16);
            kb[4] = kv8_bf(c.s8) | (kv8_bf(c.s9) << 16);
            kb[5] = kv8_bf(c.sa) | (kv8_bf(c.sb) << 16);
            kb[6] = kv8_bf(c.sc) | (kv8_bf(c.sd) << 16);
            kb[7] = kv8_bf(c.se) | (kv8_bf(c.sf) << 16);
          } else {
            for (uint z = 0; z < 8u; ++z) kb[z] = 0u;
          }
#else
          intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
              (__global void*)kv_k, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
              (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
#endif
          sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(vload8(0, kb)), sacc[b]);
#if QSPLIT
          sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(ql, as_int8(vload8(0, kb)), sacc[b]);
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
        const float sk = in ? kv8_f16f(k_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
        svl[b] = in ? kv8_f16f(v_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
#define MASK(r) sacc[b].s##r = key <= bnd.s##r ? (sacc[b].s##r * sk) * SSCALE : -INFINITY;
#else
#define MASK(r) sacc[b].s##r = key <= bnd.s##r ? sacc[b].s##r * SSCALE : -INFINITY;
#endif
        FOR8(MASK)
#undef MASK
        tmax = fmax(tmax, sacc[b]);
      }
      float8 mnew;
#define RED(r) mnew.s##r = fmax(mrow.s##r, sub_group_reduce_max(tmax.s##r));
      FOR8(RED)
#undef RED
      // corr / p with the whole-skip of a row that has seen no key yet (header)
      const int8 seen = isfinite(mnew);         // -1 where mnew is finite
      const float8 corr = select((float8)(1.0f), exp(mrow - mnew), seen);
      float8 psum = (float8)(0.0f);
      short8 pa[NKA];
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        const float8 p = select((float8)(0.0f), exp(sacc[b] - mnew), seen);
        psum += p;
#if KV8
        const float8 pv = p * svl[b];
#else
        const float8 pv = p;
#endif
#define CV(r) pa[b].s##r = as_short(rne_bf16(pv.s##r));
        FOR8(CV)
#undef CV
      }
      float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
      FOR8(SUM)
#undef SUM
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
            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD,
                (int2)((int)(j * HD + dim0 + 16u * c), (int)(t0 + 16u * b)), v4);
#pragma unroll
            for (uint i = 0; i < 8u; ++i) {
              const uint w = v4[i];
              vb[c * 16u + 2u * i] = kv8_bf(as_char((uchar)(w & 0xFFu))) | (kv8_bf(as_char((uchar)((w >> 8) & 0xFFu))) << 16);
              vb[c * 16u + 2u * i + 1u] =
                  kv8_bf(as_char((uchar)((w >> 16) & 0xFFu))) | (kv8_bf(as_char((uchar)(w >> 24))) << 16);
            }
          }
#else
          intel_sub_group_2d_block_read_transform_16b_32r16x2c(
              (__global void*)kv_v, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
              (int2)((int)(j * HD + dim0), (int)(t0 + 16u * b)), vb);
#endif
#pragma unroll
          for (uint c = 0; c < 2u; ++c)
#pragma unroll
            for (uint kh = 0; kh < 2u; ++kh)
              o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                  pa[b + kh], as_int8(vload8(0, vb + c * 16u + 8u * kh)), o[dd + c]);
        }
    }
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
#endif  // EAGER
