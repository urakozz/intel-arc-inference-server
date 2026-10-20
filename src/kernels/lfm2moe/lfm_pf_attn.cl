// lfm_pf_attn.cl - spec 24d: LFM2.5-24B-A2B's prefill attention - Kolibri's kol_pf_attn.cl full path (spec 20d; 18c's
// k2_pf_attn.cl, spec 6's DPAS flash attention) carried to head_dim 64, GQA 4, and only the growing cache (every LFM2
// attention layer is full: no window, no ring):
//
//   lfm_pf_attn(q, kv_k, kv_v, out, pos, C)   grid (ceil(C / RPW), KV_HEADS, GQA / HPW), 256 GRF
//     q      fp32 [kPfC][Q_HEADS][64]   lfm_attn_prep's q (bf16 values: the q head norm and RoPE) - 24c's decode prep
//                                       kernel at M = kPfC, S = 1
//     kv_k   bf16 [max_len][KV_HEADS][64]  this layer's cache, key k at row k; the chunk's own rows were written by
//     kv_v                              lfm_attn_prep first (an earlier launch of the in-order list)
//     out    bf16 [C][Q_HEADS x 64]     rne(o / l) - out_proj's A operand (no gate)
//
// **Causal over the cache, across chunk boundaries** (plan 24d Review Focus 4). Row t of a chunk at `pos` sits at
// p = pos + t and sees keys k <= p: the earlier chunks' keys in the cache and this chunk's rows up to its own. A
// sub-group's 8 rows start at p0 = pos + r0; its key tiles run from key 0 up to its last row, absolute multiples of KT.
// **A wholly masked tile adds exact zeros** (m_safe = (m_new == -INF ? 0 : m_new)), so a row's result depends on the
// tiles that hold its keys - a function of its ABSOLUTE position alone, whatever chunk or row group it rides in (a
// split anywhere is bitwise: the tiles start at key 0 whatever the chunk, and every tile past a row's key is wholly masked).
//
// **What is kol_pf_attn's, unchanged in structure:** a sub-group owns 8 query rows of one head; O is NDA = 4 float8
// (8 rows x 64 dims) in registers; S for one KV tile is KT/16 float8 with the row in the component and the key in the
// lane - the bf16 DPAS A-operand layout, so P goes to P·V with no data movement; K^T through the transposed 32-bit 2D
// read (the QK^T K-dim 64 = 4 DPAS k-steps), V through the VNNI-transform read; the fp32 online softmax; P rounded to
// bf16 as the P·V operand; q read from the prep's fp32 rows once (a row past C reads row C - 1: finite, never written);
// the scale 64^-0.5 = 0.125 (exact).
//
// **EAGER (B70_LFM_ATTN=eager, the switch 24c's decode reads too - one parser, runtime::lfm2moe::lfm_attn()).**
// kol_pf_attn's two passes: pass 1 the running max and sum (fp32, online), pass 2 p = rne(exp(s - m) / l) - the
// softmax's bf16 output - and o += p·V; s = rne(f32(rne(q·k)) x 0.125); the epilogue rounds o once (p already
// normalised). The reference's ROUNDING POINTS (tests/kernels/lfm2moe_pf_ref.h eager_row), not its sum orders (DPAS,
// the tile walk): spec 18 §11.
//
// 2D block I/O: the cache's surfaces are KV_HEADS x 64 x 2 = 1024 bytes wide and pitched, the layer base 64-byte
// aligned (Lfm2MoeBuffers: [max_len][512] bf16 per layer); rows past `depth` read as zero (the descriptor's height).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef KT
#error "lfm_pf_attn: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "lfm_pf_attn: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "lfm_pf_attn: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HPW)
#error "lfm_pf_attn: Q_HEADS, KV_HEADS and HPW must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0 || (Q_HEADS / KV_HEADS) % HPW != 0
#error "lfm_pf_attn: q-heads must be a multiple of kv-heads, and the GQA group of HPW"
#endif
#if !defined(EXP2) || !defined(EAGER)
#error "lfm_pf_attn: EXP2 and EAGER must be defined"
#endif
#if EAGER && EXP2
#error "lfm_pf_attn: the eager-mirroring path uses natural exp (EXP2 0), as the reference's softmax"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define SG 16
#define HD 64
#define NDA (HD / 16)                  /* 4 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */
#define SSCALE 0.125f                  /* 1 / sqrt(64), exact (the reference's d ** -0.5) */
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
__kernel void lfm_pf_attn(__global const float* restrict Q, __global const ushort* restrict Kc,
                          __global const ushort* restrict Vc, __global ushort* restrict out, uint pos, uint C) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + s / (RPW / 8u);      // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));        // first of 8 rows
  if (r0 >= C) return;                                                  // no barriers below
  const uint depth = pos + C;
  const uint p0 = pos + r0;                                             // the group's first position
  const int rows_h = (int)depth;                                        // rows past depth read as zero
  const uint tstart = 0u;
#define KROW(t) (t)
  const int kt_w = KV_HEADS * HD * 2, kt_p = KV_HEADS * HD * 2;
  const int v_w = KV_HEADS * HD * 2, v_p = KV_HEADS * HD * 2;

  // Q, 8 rows x 64 dims as NDA A operands: qa[kk].s_r = q[r0 + r][h][16 kk + lane].
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
  // r = row r0 + r; lane = key t0 + 16 b + lane; the 2D read at cache row t0 + 16 b.
#define LFM_SCORES(t0)                                                                      \
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
    LFM_MASK(0) LFM_MASK(1) LFM_MASK(2) LFM_MASK(3) LFM_MASK(4) LFM_MASK(5) LFM_MASK(6) LFM_MASK(7) \
  }
#define LFM_VISIBLE(r) (key <= p0 + r && key < depth)
#define LFM_MASK(r) sacc[b].s##r = LFM_VISIBLE(r) ? score_of(sacc[b].s##r) : -INFINITY;

  // P·V for one tile: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI).
#define LFM_PV(t0)                                                                          \
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

#define LFM_ROWMAX()                                                                        \
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
#define LFM_ROWSUM(psum, rsum)                                                              \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define LFM_CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));

#if !EAGER
  // The default path: the online softmax, P = EXPF(s - m) rounded to bf16 for P·V, O / l last.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    LFM_SCORES(t0)
    LFM_ROWMAX()
    const float8 corr = EXPF(m - msafe);         // m = -INF before a row's first key: EXPF(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - msafe);    // masked: EXPF(-INF) = 0
      psum += p;
      LFM_CV(0) LFM_CV(1) LFM_CV(2) LFM_CV(3) LFM_CV(4) LFM_CV(5) LFM_CV(6) LFM_CV(7)
    }
    LFM_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    LFM_PV(t0)
  }
#else
  // Pass 1: the row's max and Σ exp(s - max) over its keys (fp32, online).
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    LFM_SCORES(t0)
    LFM_ROWMAX()
    const float8 corr = exp(m - msafe);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - msafe);
    LFM_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
  }
  // Pass 2: p = rne(exp(s - m) / l), the softmax's bf16 output; O = Σ p v in fp32. Every row saw its own
  // key, so m is finite here.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    LFM_SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - m) / lsum;
      LFM_CV(0) LFM_CV(1) LFM_CV(2) LFM_CV(3) LFM_CV(4) LFM_CV(5) LFM_CV(6) LFM_CV(7)
    }
    LFM_PV(t0)
  }
#endif
#undef LFM_SCORES
#undef LFM_VISIBLE
#undef LFM_MASK
#undef LFM_PV
#undef LFM_ROWMAX
#undef LFM_ROWSUM
#undef LFM_CV
#undef KROW

  // The epilogue: per element (row r0 + r, dim 16 dd + lane), bf16 [C][Q_HEADS x 64].
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
