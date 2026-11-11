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
//     (bf16, staged once), K^T by the transposed 32-bit 2D block read, surface height `end` (KV8: the
//     same read over the int8 rows, 16 keys x 32 dims a message, each byte widened to its exact bf16 -
//     3l; it was one 16-byte load a key and dim atom);
//   QSPLIT = 1: q = hi + lo, hi = rne_bf16(q), lo = rne_bf16(q - hi), two DPAS per atom (hi first) -
//     ~16 significant bits of q, against fp32 q in v2; QSPLIT = 0: hi alone (prefill's rounding);
//   s = S * 1/16 (KV8: (S * k_scale) * 1/16); a row's keys past its bound pos + m, and every key of a
//     row outside the group being walked, are -INF;
//   online softmax per row in fp32 with `exp` (spec 10 §8): mnew = max(m, max over the tile's keys);
//     when mnew = -INF the row has seen no key yet and the tile is skipped whole (corr 1, p 0 - as
//     v2's waves: exp(-INF - -INF) is the only NaN); else corr = exp(m - mnew), p = exp(s - mnew);
//     sm = sm corr + sum p (fp32 p), O = O corr + P V with P = rne_bf16(p) (KV8: rne_bf16(p v_scale))
//     as the DPAS A operand - S's layout, so P goes to PV with no data movement; V by the
//     VNNI-transform 2D read (KV8: the 8-bit transform read, key pairs widened to exact bf16);
//   PSPLIT = 1: P = hi + lo like q's (QSPLIT), two DPAS per key atom pair (hi first) - ~16 significant
//     bits of p; PSPLIT = 0: hi alone. Why 1 ships (2026-11-04, docs/07-open-questions.md, "Recently answered"): P's
//     bf16 rounding (2^-9 a weight) is not averaged away on real heads - their blocks' P V partly
//     cancel in the merge, so a row's error grows to 1 - cos ~2e-5 against fp64 (v2: ~1e-8), and over
//     16 FA layers that moved a DFlash verify row's logits by up to 9 and its argmax off the f32 top
//     (t6_usage-loader, position 103). Random-data tests (cos ~0.99999 either way) cannot see it.
// The KS slices of a row block merge in SLM in slice order 0, 1, ..., KS - 1 (attn_reduce_v2's
// formulas: nmx = max; a = exp(m_a - nmx), b = exp(m_b - nmx); sm = fma(sm_a, a, sm_b b),
// acc = fma(acc_a, a, acc_b b); both -INF: unchanged), the last slice writing the partial.
//
// ---------------------------------------------------------------------------------------------
// PAIR = 1 (3l: who computes what - every value, operand and order above is unchanged)
// ---------------------------------------------------------------------------------------------
// KT 32, DSPLIT 128 (two key atoms a tile, two sub-groups a row block). The profile at 150k, M = 4
// (docs/probes/probe-longctx-decode-2026-11-09.md, 3l) found the walk latency-bound: one 12- or 24-
// sub-group work-group a core, every row block's sub-groups reading (and, KV8, widening) the whole
// V tile, the int8 K by one gather a key and atom, and ~100 registers spilled. Per tile step:
//   S: sub-group ds of row block rb computes key atom ds only (its DPAS chain as before), scales and
//     masks it, and the pair swaps the two atoms through SLM (barrier 1) - both then hold S whole
//     and run the same softmax (bit-identical inputs, so identical P, corr, m, sm);
//   P V: the pair publishes P (hi, lo) and corr in SLM (barrier 2); the slice's SPS sub-groups own
//     dim atoms [a0, a0 + na) for EVERY row block, so each V element is read (and widened) once a
//     slice. Each O accumulator takes corr then hi kh 0, hi kh 1, lo kh 0, lo kh 1 as before;
//   PVX extra sub-groups a slice (M >= 3: 6 + 2 = 8, so a work-group is 32 sub-groups - a whole core
//     at 256 GRF) only own dim atoms; while the pairs compute S they prefetch the step's V rows;
//   the slice merge runs as before, split the same way: the dim owners merge O, the pair's ds 0 its
//     sums and maxima; each slice's own row maxima go to SLM, and the merged maximum before slice k
//     is their fmax chain (= the one-owner merge's nmx, or both -INF).
// The steps a slice takes are ceil(ntile / KS) for every sub-group (two barriers each); a step past
// a slice's last tile only meets them.
//
// ---------------------------------------------------------------------------------------------
// Property R (spec 28c N2a: a row bitwise equal at the same key count in any M >= 2 launch)
// ---------------------------------------------------------------------------------------------
// Row m's tiles, their slices and the merge order depend only on its own key count (its stride and
// block start); a tile past its bound but inside the group's walk is all -INF for it, which leaves
// its state bit for bit (corr = exp(0) = 1, p = 0, 0 x V = 0; or the whole-skip above); a DPAS output
// row depends only on its A row; the row max and sum are per row (one component of the sub-group
// reductions). So nothing in a row's result depends on M, its slot, or the other rows - PROVIDED one
// lever setting (KT, KS, QSPLIT, PSPLIT) serves every M of a KV form (Review Focus 2). PACK changes a row's
// slot only; DSPLIT, PAIR, PVX and GRF change who computes a value, not the value (3l: production's M = 2..4
// lines differ from M = 5..8's there, tests/kernels/attn_verify_bitwise_test.cc holds every M to the frozen
// kernel word for word). Checked bitwise on the card, not assumed.
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
//   PSPLIT  0 | 1              P in bf16 (0) or as hi + lo (1, the production line)
//   KV8     0 | 1
//   PAIR    0 | 1              (default 0) the 3l split above: needs KT 32, DSPLIT 128, KS > 1
//   PVX     0 | 2 | ...        (default 0, PAIR) P V-only sub-groups a slice, keeping it even
//   EAGER   1: the eager build - plain loads, no sub-group
//           function at all: one work-item per (q head, row) runs the same walk (blocks, tiles,
//           slices, masks, the P rounding, the slice merge order, the partial layout) with a scalar
//           tile product (16-dim atoms ascending, hi then lo). Never a production line.
//   GRF     the ocloc option -cl-intel-256-GRF-per-thread on the command line (not a define).
// Work-group = 16 (RB DS + PVX) KS lanes: at most 512 (a 256-GRF work-group's 32 sub-groups); the probe
// builds the 1024-lane shapes at 128 GRF only. SLM: q (RB x 4 KB, x 2 at QSPLIT) + the slice merge
// (RB x 8 KB at KS > 1) + PAIR's exchange (2 KB a sub-group, 3 KB a slice and row block) <= 128 KB.
//
// Heads: Qwen3.8's 24 q / 4 kv (Agnes shares them), or (spec 17d) a tensor-parallel card's half, 12 q / 2 kv - the
// same groups of 6, so a work-group's walk, operands and order are the 24 / 4 build's for those heads (property R:
// row for row bitwise the one-card kernel's heads 12 r .. 12 r + 11); the grid is (2, TGT). Another model's heads
// keep v2 (the rule).
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
#ifndef EAGER
#define EAGER 0
#endif
#ifndef PAIR
#define PAIR 0
#endif
#ifndef PVX
#define PVX 0
#endif
#if defined(FA_Q_HEADS) != defined(FA_KV_HEADS)
#error "attn_verify_dpas: FA_Q_HEADS and FA_KV_HEADS come together"
#endif
#if defined(FA_Q_HEADS) && !(FA_Q_HEADS == 12 && FA_KV_HEADS == 2)
#error "attn_verify_dpas: Qwen3.8's 24 / 4 heads, or a tensor-parallel card's 12 / 2 (spec 17d); other heads keep attn_v2"
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

#ifdef FA_Q_HEADS
#define Q_HEADS FA_Q_HEADS
#define KV_HEADS FA_KV_HEADS
#else
#define Q_HEADS 24
#define KV_HEADS 4
#endif
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
#if PAIR && (KT != 32 || DSPLIT != 128 || KS == 1)
#error "attn_verify_dpas: PAIR needs KT 32, DSPLIT 128 and KS > 1"
#endif
#if PVX && (!PAIR || SPS % 2)
#error "attn_verify_dpas: PVX needs PAIR and keeps a slice's sub-group count even"
#endif
#define NSG (SPS * KS)
#define WG (SG * NSG)
#if WG > 1024
#error "attn_verify_dpas: more than 1024 lanes a work-group (RB x DS x KS sub-groups of 16)"
#endif
#define QS_BYTES (RB * NDA * 128 * 2 * (1 + QSPLIT))
#define MERGE_BYTES (KS > 1 ? RB * 8 * HD * 4 + RB * DS * 8 * 2 * 4 : 0)
#define PAIR_BYTES (PAIR ? NSG * 1024 + KS * RB * (4 * 256 + 512) + (KS + 1) * RB * 32 : 0)
#if QS_BYTES + MERGE_BYTES + PAIR_BYTES > 131072
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
// Four int8 (a dword, low byte first) as two dwords of exact bf16 pairs, low byte first.
inline uint2 kv8_bf4(uint w) {
  return (uint2)(kv8_bf(as_char((uchar)(w & 0xFFu))) | (kv8_bf(as_char((uchar)((w >> 8) & 0xFFu))) << 16),
                 kv8_bf(as_char((uchar)((w >> 16) & 0xFFu))) | (kv8_bf(as_char((uchar)(w >> 24))) << 16));
}
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
// The eager work-group: 64 work-items whatever the levers; work-item i runs
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
        float sc[KT], pr16[KT], prl[KT];   // P's bf16 hi (and lo at PSPLIT)
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
          const float pv = p * sv;
#else
          const float pv = p;
#endif
          pr16[t] = bf16f(rne_bf16(pv));
          prl[t] = PSPLIT ? bf16f(rne_bf16(pv - pr16[t])) : 0.0f;
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
#if PSPLIT
          for (uint t = 0; t < KT; ++t) {
            const uint key = t0 + t;
            const float vv = key < end ? V_AT(key, d) : 0.0f;
            o = fma(prl[t], vv, o);
          }
#endif
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

// S += q K^T for key atom `kbase` (16 keys from it, lane = key) of row block rb: 16 dim atoms ascending, hi then lo.
#if KV8
// int8 K: 16 keys x 32 dims a transposed 2D read (lane = key, dword z = its dims 32 kp + 4 z .. + 3; rows past
// `end` read as 0 - widened, the zeros the old per-key load gave there), each dim atom widened to exact bf16.
#define S_ATOM(sacc_b, kbase)                                                                          \
  _Pragma("unroll") for (uint kp = 0; kp < NDA / 2; ++kp) {                                           \
    uint k8[8];                                                                                       \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kv_k, KV_HEADS * HD, (int)end, \
                                                        KV_HEADS * HD,                                \
                                                        (int2)((int)((j * HD + 32u * kp) / 4u), (int)(kbase)), k8); \
    _Pragma("unroll") for (uint h = 0; h < 2u; ++h) {                                                 \
      const uint kk = 2u * kp + h;                                                                    \
      uint kb[8];                                                                                     \
      _Pragma("unroll") for (uint z = 0; z < 4u; ++z) {                                               \
        const uint2 pr = kv8_bf4(k8[4u * h + z]);                                                     \
        kb[2u * z] = pr.x;                                                                            \
        kb[2u * z + 1u] = pr.y;                                                                       \
      }                                                                                               \
      S_DPAS(sacc_b, kk, kb)                                                                          \
    }                                                                                                 \
  }
#else
#define S_ATOM(sacc_b, kbase)                                                                          \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk) {                                               \
    uint kb[8];                                                                                       \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kv_k, KV_HEADS * HD * 2, (int)end, \
                                                        KV_HEADS * HD * 2,                            \
                                                        (int2)((int)((j * HD + 16u * kk) / 2u), (int)(kbase)), kb); \
    S_DPAS(sacc_b, kk, kb)                                                                            \
  }
#endif
#if QSPLIT
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
// s = S * 1/16 (KV8: (S * k_scale) * 1/16) for the keys kbase + lane, -INF past each row's bound.
#if KV8
#define S_MASK(sacc_b, kbase)                                                                          \
  {                                                                                                   \
    const int key = (int)((kbase) + l);                                                               \
    const float sk = (uint)key < end ? kv8_f16f(k_scale[(size_t)key * KV_HEADS + j]) : 0.0f;          \
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
  __local float sx[NSG * 128];                  // [sub-group][r * 16 + lane]: its key atom of S
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
          svl[b] = key < end ? kv8_f16f(v_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
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
#if KV8
#pragma unroll
        for (uint b = 0; b < NKA; ++b) S_ATOM(sacc[b], t0 + 16u * b)
#else
#pragma unroll
        for (uint kk = 0; kk < NDA; ++kk) {     // main's order: q's atom read once for both key atoms
          const short8 qa = as_short8(intel_sub_group_block_read_us8(&qs[(rb * NDA + kk) * 128]));
#if QSPLIT
          const short8 ql = as_short8(intel_sub_group_block_read_us8(&qsl[(rb * NDA + kk) * 128]));
#endif
#pragma unroll
          for (uint b = 0; b < NKA; ++b) {
            uint kb[8];
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (__global void*)kv_k, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2,
                (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
            sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(vload8(0, kb)), sacc[b]);
#if QSPLIT
            sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(ql, as_int8(vload8(0, kb)), sacc[b]);
#endif
          }
        }
#endif
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
          svl[b] = key < end ? kv8_f16f(v_scale[(size_t)key * KV_HEADS + j]) : 0.0f;
#endif
          tmax = fmax(tmax, sacc[b]);
        }
#endif  // PAIR
        float8 mnew;
#define RED(r) mnew.s##r = fmax(mrow.s##r, sub_group_reduce_max(tmax.s##r));
        FOR8(RED)
#undef RED
        // corr / p with the whole-skip of a row that has seen no key yet (header)
        const int8 seen = isfinite(mnew);       // -1 where mnew is finite
        const float8 corr = select((float8)(1.0f), exp(mrow - mnew), seen);
        float8 psum = (float8)(0.0f);
        short8 pa[NKA];
#if PSPLIT
        short8 pl[NKA];                         // P's lo half (header, PSPLIT)
#endif
#pragma unroll
        for (uint b = 0; b < NKA; ++b) {
          const float8 p = select((float8)(0.0f), exp(sacc[b] - mnew), seen);
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
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
        FOR8(SUM)
#undef SUM
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
          intel_sub_group_2d_block_read_transform_8b_32r16x1c(
              (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD, (int2)((int)(j * HD + dim0), (int)t0), v4);
#pragma unroll
          for (uint i = 0; i < 8u; ++i) {
            const uint2 pw = kv8_bf4(v4[i]);
            vb[2u * i] = pw.x;
            vb[2u * i + 1u] = pw.y;
          }
#else
          intel_sub_group_2d_block_read_transform_16b_32r16x1c(
              (__global void*)kv_v, KV_HEADS * HD * 2, (int)end, KV_HEADS * HD * 2, (int2)((int)(j * HD + dim0), (int)t0),
              vb);
#endif
#pragma unroll
          for (uint r2 = 0; r2 < RB; ++r2) {
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][0], as_int8(vload8(0, vb)), o[r2][a]);
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][1], as_int8(vload8(0, vb + 8u)), o[r2][a]);
#if PSPLIT
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][2], as_int8(vload8(0, vb)), o[r2][a]);
            o[r2][a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pr[r2][3], as_int8(vload8(0, vb + 8u)), o[r2][a]);
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
              intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                  (__global void*)kv_v, KV_HEADS * HD, (int)end, KV_HEADS * HD,
                  (int2)((int)(j * HD + dim0 + 16u * c), (int)(t0 + 16u * b)), v4);
#pragma unroll
              for (uint i = 0; i < 8u; ++i) {
                const uint2 pw = kv8_bf4(v4[i]);
                vb[c * 16u + 2u * i] = pw.x;
                vb[c * 16u + 2u * i + 1u] = pw.y;
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
#if PSPLIT
#pragma unroll
            for (uint c = 0; c < 2u; ++c)
#pragma unroll
              for (uint kh = 0; kh < 2u; ++kh)
                o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    pl[b + kh], as_int8(vload8(0, vb + c * 16u + 8u * kh)), o[dd + c]);
#endif
          }
      }
#endif  // PAIR
    }
  }

  // ---- the slice merge (slice order) and the partial write ----
#if PAIR
  // Split as P V was (header): the dim owners merge O for every row block, the pair's ds 0 its sums and maxima.
  __local float macc[RB * 8 * HD];              // [rb][r][dim]
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
      } else {
#pragma unroll
        for (uint r2 = 0; r2 < RB; ++r2) {
#define OUT(r)                                                                          \
  {                                                                                     \
    uint qq, mm;                                                                        \
    bool oo;                                                                            \
    row_of(r2, r, &qq, &mm, &oo);                                                       \
    if (oo && mm < n_act && (mm < msplit ? live0 : live1)) {                            \
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
#endif  // PAIR
}
#endif  // EAGER
