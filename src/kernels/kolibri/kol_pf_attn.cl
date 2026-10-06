// kol_pf_attn.cl - spec 20d: Kolibri-1's prefill attention - 18c's k2_pf_attn.cl (spec 6's flash
// attention carried to head_dim 128) at GQA 12, with Kolibri's two layer kinds and no output gate:
//
//   kol_pf_flash_attn(q, kv_k, kv_v, out, pos, C)   grid (ceil(C / RPW), KV_HEADS, GQA / HPW), 256 GRF
//     q      fp32 [kPfC][Q_HEADS][128]   kol_attn_prep's q (bf16 values: q/k head norm, RoPE in a sliding
//                                       layer) - 20c's decode prep kernel at M = kPfC, S = 1
//     kv_k   bf16 [rows][KV_HEADS][128]  this layer's keys: SLIDING the 4096-slot ring (key k at row
//     kv_v                              k & (RING - 1)), else the growing cache (key k at row k); the
//                                       chunk's own rows were written by kol_attn_prep first
//     out    bf16 [C][Q_HEADS x 128]     rne(o / l) - o_proj's A operand (no gate: Kolibri has none)
//
// **The window** (spec 20 §1, 20a pinned: 513 keys INCLUDING the query). Row t of a chunk at `pos` sits
// at p = pos + t and sees keys k with k <= p and, SLIDING, k > p - WINDOW (i.e. k >= p - 512) - some
// from the ring (positions before pos, the previous chunks' writes) and some from this chunk. A sub-
// group's 8 rows start at p0 = pos + r0; its key tiles run from KT x floor(max(0, p0 - (WINDOW - 1)) /
// KT) (SLIDING; 0 for a full layer) up to its last row: ABSOLUTE multiples of KT, and RING % KT == 0, so
// every tile is KT contiguous ring rows and no 2D block read crosses the ring's end. A chunk of kPfC =
// 2048 rows needs keys [pos - 512, pos + 2047]: 2560 < 4096 slots, so this chunk's writes never
// overwrite a key one of its rows still reads (the reason for 20c's 4096-slot ring).
//
// **A wholly masked tile adds exact zeros.** Every tile a row walks lies within its group's span, but a
// row's FIRST tile may hold none of ITS keys (the group's first row reaches further back): the online
// softmax runs with m_safe = (m_new == -INF ? 0 : m_new), so such a tile leaves (m, l, o) exactly as
// they were (corr = exp(-INF) = 0 on zeros, p = exp(-INF) = 0); once a key was seen the update is
// k2_pf_attn's, unchanged (m_safe = m_new). So a row's result depends on the tiles that hold its keys -
// a function of its ABSOLUTE position alone, whatever chunk or row group it rides in: the prefill
// split argument (a split at a multiple of 64 is bitwise).
//
// **What is k2_pf_attn's, unchanged in structure:** a sub-group owns 8 query rows of one head; O is NDA
// float8 (8 rows x 128 dims) in registers; S for one KV tile is KT/16 float8 with the row in the
// component and the key in the lane - the bf16 DPAS A-operand layout, so P goes to P·V with no data
// movement; K^T through the transposed 32-bit 2D read, V through the VNNI-transform read; the fp32
// online softmax; P rounded to bf16 as the P·V operand; q read from the prep's fp32 rows once (a row
// past C reads row C - 1: finite, never written); the scale 1 / sqrt(128).
//
// **EAGER (B70_KOLIBRI_ATTN=eager, the switch 20c's decode reads too - one parser,
// runtime::kolibri::kolibri_attn()).** k2_pf_attn's two passes: pass 1 the running max and sum (fp32,
// online), pass 2 p = rne(exp(s - m) / l) - the softmax's bf16 output - and o += p·V; s = rne(f32(rne(
// q·k)) x 128^-0.5); the epilogue rounds o once (p already normalised). The reference's ROUNDING POINTS
// (tests/kernels/kolibri_pf_ref.h eager_window), not its sum orders (DPAS, the tile walk): spec 18 §11.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef KT
#error "kol_pf_attn: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "kol_pf_attn: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "kol_pf_attn: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HPW)
#error "kol_pf_attn: Q_HEADS, KV_HEADS and HPW must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0 || (Q_HEADS / KV_HEADS) % HPW != 0
#error "kol_pf_attn: q-heads must be a multiple of kv-heads, and the GQA group of HPW"
#endif
#if !defined(EXP2) || !defined(EAGER) || !defined(WINDOW) || !defined(RING)
#error "kol_pf_attn: EXP2, EAGER, WINDOW and RING must be defined (WINDOW 0 / RING 0: a full layer)"
#endif
#if EAGER && EXP2
#error "kol_pf_attn: the eager-mirroring path uses natural exp (EXP2 0), as the reference's softmax"
#endif
#define SLIDING (WINDOW > 0)
#if SLIDING && (RING == 0 || (RING & (RING - 1)) != 0 || RING % KT != 0 || RING < 2048 + WINDOW - 1)
#error "kol_pf_attn: the ring is a power of two of whole KT tiles holding a chunk and its window"
#endif
#if !SLIDING && RING != 0
#error "kol_pf_attn: a full layer has no ring (RING 0)"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define SG 16
#define HD 128
#define NDA (HD / 16)                  /* 8 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */
#define SSCALE 0.08838834764831845f    /* 1 / sqrt(128), as fp32 (the reference's d ** -0.5) */
#if EXP2
#define SCORE_SCALE (SSCALE * M_LOG2E_F)   /* scores in the log2 domain */
#define EXPF(x) exp2(x)
#else
#define SCORE_SCALE SSCALE
#define EXPF(x) exp(x)
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}
// One score as the walk uses it: the default path scales the fp32 dot; the eager one rounds the dot (the
// bf16 matmul's output) and the scaled score (the bf16 multiply).
inline float score_of(float dot) {
#if EAGER
  return bf16f(rne_bf16(bf16f(rne_bf16(dot)) * SSCALE));
#else
  return dot * SCORE_SCALE;
#endif
}
// The online softmax's reference point: m itself, or 0 while a row has seen no key (m = -INF).
inline float8 m_safe(float8 m) { return select(m, (float8)(0.0f), isequal(m, (float8)(-INFINITY))); }

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void kol_pf_flash_attn(__global const float* restrict Q, __global const ushort* restrict Kc,
                                __global const ushort* restrict Vc, __global ushort* restrict out, uint pos,
                                uint C) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + s / (RPW / 8u);      // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));        // first of 8 rows
  if (r0 >= C) return;                                                  // no barriers below
  const uint depth = pos + C;
  const uint p0 = pos + r0;                                             // the group's first position
#if SLIDING
  const int rows_h = RING;                                              // the ring's rows
  const uint tstart = p0 > (uint)(WINDOW - 1) ? (p0 - (uint)(WINDOW - 1)) / KT * KT : 0u;
#define KROW(t) ((t) & (uint)(RING - 1))
#else
  const int rows_h = (int)depth;                                        // rows past depth read as zero
  const uint tstart = 0u;
#define KROW(t) (t)
#endif
  const int kt_w = KV_HEADS * HD * 2, kt_p = KV_HEADS * HD * 2;
  const int v_w = KV_HEADS * HD * 2, v_p = KV_HEADS * HD * 2;

  // Q, 8 rows x 128 dims as NDA A operands: qa[kk].s_r = q[r0 + r][h][16 kk + lane].
  short8 qa[NDA];
#pragma unroll
  for (uint kk = 0; kk < NDA; ++kk) {
    ushort t[8];
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint row = min(r0 + r, C - 1u);
      t[r] = rne_bf16(Q[((size_t)row * Q_HEADS + h) * HD + 16u * kk + l]);
    }
    qa[kk] = as_short8(vload8(0, t));
  }

  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, p0 + 8u);                // keys this SG's rows can see

  // The score tile at absolute key t0: S = Q K^T over KT keys, scored (score_of) and masked. Component
  // r = row r0 + r; lane = key t0 + 16 b + lane; the 2D read at ring row KROW(t0) + 16 b.
#define KOL_SCORES(t0)                                                                      \
  float8 sacc[NKA];                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk)                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    uint kb[8];                                                                             \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c(                                    \
        (__global void*)Kc, kt_w, rows_h, kt_p,                                             \
        (int2)((int)((j * HD + 16u * kk) / 2u), (int)(KROW(t0) + 16u * b)), kb);            \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]); \
  }                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    const uint key = (t0) + 16u * b + l;                                                    \
    KOL_MASK(0) KOL_MASK(1) KOL_MASK(2) KOL_MASK(3) KOL_MASK(4) KOL_MASK(5) KOL_MASK(6) KOL_MASK(7) \
  }
#if SLIDING
#define KOL_VISIBLE(r) (key <= p0 + r && key < depth && key + (uint)WINDOW > p0 + r)
#else
#define KOL_VISIBLE(r) (key <= p0 + r && key < depth)
#endif
#define KOL_MASK(r) sacc[b].s##r = KOL_VISIBLE(r) ? score_of(sacc[b].s##r) : -INFINITY;

  // P·V for one tile: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI).
#define KOL_PV(t0)                                                                          \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                  \
    uint vb[32];                                                                            \
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(                                   \
        (__global void*)Vc, v_w, rows_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(KROW(t0) + 16u * b)), vb); \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks)                                      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(                                 \
          pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);               \
  }

#define KOL_ROWMAX()                                                                        \
  float8 tmax = (float8)(-INFINITY);                                                        \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) tmax = fmax(tmax, sacc[b]);              \
  float8 mnew;                                                                              \
  mnew.s0 = fmax(m.s0, sub_group_reduce_max(tmax.s0));                                      \
  mnew.s1 = fmax(m.s1, sub_group_reduce_max(tmax.s1));                                      \
  mnew.s2 = fmax(m.s2, sub_group_reduce_max(tmax.s2));                                      \
  mnew.s3 = fmax(m.s3, sub_group_reduce_max(tmax.s3));                                      \
  mnew.s4 = fmax(m.s4, sub_group_reduce_max(tmax.s4));                                      \
  mnew.s5 = fmax(m.s5, sub_group_reduce_max(tmax.s5));                                      \
  mnew.s6 = fmax(m.s6, sub_group_reduce_max(tmax.s6));                                      \
  mnew.s7 = fmax(m.s7, sub_group_reduce_max(tmax.s7));                                      \
  const float8 msafe = m_safe(mnew);
#define KOL_ROWSUM(psum, rsum)                                                              \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define KOL_CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));

#if !EAGER
  // The default path: the online softmax, P = EXPF(s - m) rounded to bf16 for P·V, O / l last.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    KOL_ROWMAX()
    const float8 corr = EXPF(m - msafe);         // m = -INF before a row's first key: EXPF(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - msafe);    // masked: EXPF(-INF) = 0
      psum += p;
      KOL_CV(0) KOL_CV(1) KOL_CV(2) KOL_CV(3) KOL_CV(4) KOL_CV(5) KOL_CV(6) KOL_CV(7)
    }
    KOL_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    KOL_PV(t0)
  }
#else
  // Pass 1: the row's max and Σ exp(s - max) over its keys (fp32, online).
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    KOL_ROWMAX()
    const float8 corr = exp(m - msafe);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - msafe);
    KOL_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
  }
  // Pass 2: p = rne(exp(s - m) / l), the softmax's bf16 output; O = Σ p v in fp32. Every row saw its own
  // key, so m is finite here.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - m) / lsum;
      KOL_CV(0) KOL_CV(1) KOL_CV(2) KOL_CV(3) KOL_CV(4) KOL_CV(5) KOL_CV(6) KOL_CV(7)
    }
    KOL_PV(t0)
  }
#endif
#undef KOL_SCORES
#undef KOL_VISIBLE
#undef KOL_MASK
#undef KOL_PV
#undef KOL_ROWMAX
#undef KOL_ROWSUM
#undef KOL_CV
#undef KROW

  // The epilogue: per element (row r0 + r, dim 16 dd + lane), bf16 [C][Q_HEADS x 128].
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
#if EAGER
    const float8 w = o[dd];                      // already normalised
#else
    const float8 w = o[dd] / lsum;
#endif
    float wv[8];
    vstore8(w, 0, wv);
    for (uint r = 0; r < 8; ++r) {
      const uint row = r0 + r;
      if (row >= C) break;
      out[((size_t)row * Q_HEADS + h) * HD + 16u * dd + l] = rne_bf16(wv[r]);
    }
  }
}
