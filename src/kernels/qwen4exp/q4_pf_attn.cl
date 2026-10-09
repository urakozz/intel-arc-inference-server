// q4_pf_attn.cl - spec 21d: Qwen3.8-Flash-Next's prefill QSA attention for the rows past the dense range - every row
// at a position p >= 2051 of a chunk attends its OWN selection (q4_qsa_select's list: the top 512 blocks expanded and
// the open block's tail, ascending, 2048..2051 positions), so the dense flash (prefill/pf_flash_attn.cl, which the
// walk binds for the rows p <= 2050) does not apply. Spec 6's flash idioms (DPAS bf16 x bf16 -> fp32, the score
// tile with the row in the component and the key in the lane - the A operand of P.V with no data movement - the fp32
// online softmax) carried to a GATHERED key set. One binary per form (kernels::qwen4exp::pf_sparse_attn_variant):
//
//   q4_pf_sparse_attn(Q, Kc, Vc, lists, O, r0, rows)    grid (C - r0, KV_HEADS), WG 32 (2 sub-groups of 16)
//     Q      bf16 [C][24][256]        pf_attn_prep_q16's q (q / k (1 + w) norms, partial RoPE at p)
//     Kc/Vc  bf16 [max_len][2][256]   this layer's cache, rows by position (the chunk's own rows written first)
//     lists  u32 [C][LIST_ROW]        q4_qsa_select's rows at M = C: positions [0, count), count at COUNT_W
//     O      fp32 [24][rows][256]     pf_o's layout (pf_flash_attn's), so pf_attn's gate (pf_attn_gate, the
//                                     _Q24KV2 binary) applies to the dense and the sparse rows alike
//
// **The tile.** Work-group (row r0 + x, kv head j): the row's 12 q heads of kv head j are the DPAS M, padded to 16:
// sub-group 0 owns heads 12 j .. 12 j + 7, sub-group 1 heads 12 j + 8 .. 12 j + 11 and four zero rows (their output is
// never written). A tile is KT list entries, ascending (the reference's masked-row order); lane l of a 16-key atom
// owns list entry t0 + 16 b + l: its position read from the list (entries past the count read entry count - 1 - a
// valid address - and are masked to -INF), its K row 16 dims at a time as the DPAS B operand (8 dwords = the VNNI
// pairs of one key column), S = Q K^T per atom, scaled by 1/16, masked. P.V: the B operand of an atom's 16 keys x 16
// dims is gathered lane by lane (lane = dim, the key pair's two positions broadcast from their lanes), each a
// coalesced 32-byte row segment.
// **The softmax** (default, EAGER 0): spec 6's online form at fp32 - m_new = max(m, the tile's max), corr =
// exp(m - m_safe), p = exp(s - m_safe) rounded to bf16 as the P.V operand, l = l corr + sum p, O = O corr + P V;
// O / l at the end - natural `exp` (EXP2 0: q4_qsa_attn's decode form, never native_exp). m_safe is 0 while a row
// has seen no key (a row's first tile always holds one: count >= 2048).
// **EAGER 1** (B70_Q4_ATTN=eager - one switch for decode and prefill): the reference's ROUNDING POINTS, not its sum
// orders (spec 18 §11's rule): s = rne(rne(q . k) x 1/16) (the bf16 matmul's output and the bf16 multiply), pass 1
// the row's max and sum, pass 2 p = rne(exp(s - m) / l) (the softmax's bf16 output), O = sum p v rounded once
// (written as its bf16 value: pf_attn_gate's own rounding of O is then exact).
//
// tests/kernels/qwen4exp_pf_ref.h sparse_attn_fp64 is the default form's walk in fp64 (the cosine >= 0.99999 bar of
// spec 6 K1, qwen4exp_pf_kernels_test on the card); sparse_row_eager the eager chain. Card only: DPAS and sub-group
// broadcasts / reductions (the Mac checks its OpenCL syntax, not its arithmetic).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !defined(KT) || KT % 16 != 0 || KT < 16 || KT > 64
#error "q4_pf_sparse_attn: KT (list entries per tile) is a multiple of 16 in [16, 64]"
#endif
#if !defined(LIST_ROW) || !defined(COUNT_W)
#error "q4_pf_sparse_attn: LIST_ROW and COUNT_W must be defined (kernels::qwen4exp)"
#endif
#ifndef EAGER
#error "q4_pf_sparse_attn: EAGER (0: flash, 1: the reference's rounding points) must be defined"
#endif
#define Q_HEADS 24
#define KV_HEADS 2
#define GQA 12
#define HD 256
#define SG 16
#define NDA (HD / 16)          /* 16 dim-atoms of O */
#define NKA (KT / 16)          /* key atoms per tile */
#define SSCALE 0.0625f         /* 1 / sqrt(256) */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}
inline float score_of(float dot) {
#if EAGER
  return bf16f(rne_bf16(bf16f(rne_bf16(dot)) * SSCALE));
#else
  return dot * SSCALE;
#endif
}
inline float8 m_safe(float8 m) { return select(m, (float8)(0.0f), isequal(m, (float8)(-INFINITY))); }

__attribute__((reqd_work_group_size(2 * SG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void q4_pf_sparse_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                                __global const ushort* restrict Vc, __global const uint* restrict lists,
                                __global float* restrict O, uint r0, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint m = r0 + get_group_id(0);                 // the chunk's row
  const uint j = get_group_id(1);                      // kv head
  const uint hb = j * GQA + 8u * s;                    // the sub-group's first q head
  const uint nh = s == 0 ? 8u : GQA - 8u;              // its live heads (8, 4)
  __global const uint* restrict list = lists + (size_t)m * LIST_ROW;
  const uint count = list[COUNT_W];

  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 mrow = (float8)(-INFINITY), lsum = (float8)(0.0f);

  // Q as the DPAS A operand for dims 16 kk .. 16 kk + 15: component r = head hb + r (zero past nh), lane = the dim.
#define Q_ATOM(kk, qa)                                                                       \
  {                                                                                          \
    ushort t_[8];                                                                            \
    _Pragma("unroll") for (uint r = 0; r < 8; ++r)                                           \
      t_[r] = r < nh ? Q[((size_t)m * Q_HEADS + hb + r) * HD + 16u * (kk) + l] : (ushort)0;  \
    qa = as_short8(vload8(0, t_));                                                           \
  }
  // The tile's scores: sacc[b] component r = head r, lane = list entry t0 + 16 b + l (its position kp[b]).
#define SCORES(t0)                                                                           \
  uint kp[NKA];                                                                              \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b)                                           \
    kp[b] = list[min((t0) + 16u * b + l, count - 1u)];                                       \
  float8 sacc[NKA];                                                                          \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                 \
  for (uint kk = 0; kk < NDA; ++kk) {                                                        \
    short8 qa;                                                                               \
    Q_ATOM(kk, qa)                                                                           \
    _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                       \
      const uint8 kb = vload8(0, (__global const uint*)(Kc + ((size_t)kp[b] * KV_HEADS + j) * HD + 16u * kk)); \
      sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa, as_int8(kb), sacc[b]);          \
    }                                                                                        \
  }                                                                                          \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                         \
    const bool live = (t0) + 16u * b + l < count;                                            \
    sacc[b].s0 = live ? score_of(sacc[b].s0) : -INFINITY;                                    \
    sacc[b].s1 = live ? score_of(sacc[b].s1) : -INFINITY;                                    \
    sacc[b].s2 = live ? score_of(sacc[b].s2) : -INFINITY;                                    \
    sacc[b].s3 = live ? score_of(sacc[b].s3) : -INFINITY;                                    \
    sacc[b].s4 = live ? score_of(sacc[b].s4) : -INFINITY;                                    \
    sacc[b].s5 = live ? score_of(sacc[b].s5) : -INFINITY;                                    \
    sacc[b].s6 = live ? score_of(sacc[b].s6) : -INFINITY;                                    \
    sacc[b].s7 = live ? score_of(sacc[b].s7) : -INFINITY;                                    \
  }
#define ROWMAX()                                                                             \
  float8 tmax = (float8)(-INFINITY);                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) tmax = fmax(tmax, sacc[b]);               \
  float8 mnew;                                                                               \
  mnew.s0 = fmax(mrow.s0, sub_group_reduce_max(tmax.s0));                                    \
  mnew.s1 = fmax(mrow.s1, sub_group_reduce_max(tmax.s1));                                    \
  mnew.s2 = fmax(mrow.s2, sub_group_reduce_max(tmax.s2));                                    \
  mnew.s3 = fmax(mrow.s3, sub_group_reduce_max(tmax.s3));                                    \
  mnew.s4 = fmax(mrow.s4, sub_group_reduce_max(tmax.s4));                                    \
  mnew.s5 = fmax(mrow.s5, sub_group_reduce_max(tmax.s5));                                    \
  mnew.s6 = fmax(mrow.s6, sub_group_reduce_max(tmax.s6));                                    \
  mnew.s7 = fmax(mrow.s7, sub_group_reduce_max(tmax.s7));                                    \
  const float8 msafe = m_safe(mnew);
#define ROWSUM(psum, rsum)                                                                   \
  float8 rsum;                                                                               \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);          \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);          \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);          \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define TO_BF16(pa, p)                                                                       \
  pa.s0 = as_short(rne_bf16(p.s0)); pa.s1 = as_short(rne_bf16(p.s1));                        \
  pa.s2 = as_short(rne_bf16(p.s2)); pa.s3 = as_short(rne_bf16(p.s3));                        \
  pa.s4 = as_short(rne_bf16(p.s4)); pa.s5 = as_short(rne_bf16(p.s5));                        \
  pa.s6 = as_short(rne_bf16(p.s6)); pa.s7 = as_short(rne_bf16(p.s7));
  // O += P V for one tile: A = pa[b] (heads x 16 keys), B = V (16 keys x 16 dims, VNNI) gathered - lane = the dim,
  // dword i = the values of the atom's keys 2i and 2i + 1 (their positions broadcast from lanes 2i, 2i + 1).
#define PV()                                                                                 \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                         \
    uint kq[SG];                                                                             \
    _Pragma("unroll") for (uint i = 0; i < SG; ++i) kq[i] = sub_group_broadcast(kp[b], i);   \
    for (uint dd = 0; dd < NDA; ++dd) {                                                      \
      uint vb[8];                                                                            \
      _Pragma("unroll") for (uint i = 0; i < 8; ++i) {                                       \
        const uint lo = Vc[((size_t)kq[2 * i] * KV_HEADS + j) * HD + 16u * dd + l];          \
        const uint hi = Vc[((size_t)kq[2 * i + 1] * KV_HEADS + j) * HD + 16u * dd + l];      \
        vb[i] = lo | (hi << 16);                                                             \
      }                                                                                      \
      o[dd] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[b], as_int8(vload8(0, vb)), o[dd]); \
    }                                                                                        \
  }

#if !EAGER
  for (uint t0 = 0; t0 < count; t0 += KT) {
    SCORES(t0)
    ROWMAX()
    const float8 corr = exp(mrow - msafe);       // mrow = -INF before a row's first key: exp(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - msafe);     // masked: exp(-INF) = 0
      psum += p;
      TO_BF16(pa[b], p)
    }
    ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    mrow = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    PV()
  }
#else
  // Pass 1: the row's max and sum of exp(s - max) over its list (fp32, online).
  for (uint t0 = 0; t0 < count; t0 += KT) {
    SCORES(t0)
    ROWMAX()
    const float8 corr = exp(mrow - msafe);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - msafe);
    ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    mrow = mnew;
  }
  // Pass 2: p = rne(exp(s - m) / l), the softmax's bf16 output; O = sum p v in fp32.
  for (uint t0 = 0; t0 < count; t0 += KT) {
    SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - mrow) / lsum;
      TO_BF16(pa[b], p)
    }
    PV()
  }
#endif
#undef Q_ATOM
#undef SCORES
#undef ROWMAX
#undef ROWSUM
#undef TO_BF16
#undef PV

  // The epilogue: O[h][m][16 dd + lane], h = hb + r for the sub-group's live heads.
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
#if EAGER
    const float8 w = o[dd];                      // already normalised; rounded once below
#else
    const float8 w = o[dd] / lsum;
#endif
    float wv[8];
    vstore8(w, 0, wv);
    for (uint r = 0; r < nh; ++r) {
#if EAGER
      const float v = bf16f(rne_bf16(wv[r]));
#else
      const float v = wv[r];
#endif
      O[((size_t)(hb + r) * rows + m) * HD + 16u * dd + l] = v;
    }
  }
}
