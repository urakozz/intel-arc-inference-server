// k2_pf_attn.cl - spec 18c: K2-Horizon's prefill attention, spec 6's flash attention
// (src/kernels/prefill/pf_flash_attn.cl, which is untouched - its binaries are Qwen3.8's,
// Agnes's and Ornith's) carried to head_dim 128 / GQA 4 with K2's softplus output gate fused
// into the epilogue:
//
//   k2_pf_flash_attn(q, kv_k, kv_v, gate, out, pos, C)   grid (ceil(C / RPW), KV_HEADS, GQA / HPW)
//     q      fp32 [kC][Q_HEADS][128]   k2_attn_prep's RoPE'd q (bf16 values) - the decode prep
//                                      kernel at M = kC, S = 1 (src/kernels/k2/k2_prep.cl)
//     kv_k   bf16 [max_len][KV_HEADS][128]   this layer's caches; rows [0, pos + C) are read
//     kv_v
//     gate   fp32 [kC][Q_HEADS][128]   f32(rne(gate column)), k2_attn_prep's attn_gate
//     out    GATE_EPI 1: bf16 [C][Q_HEADS x 128], o_proj's A operand
//            GATE_EPI 0: fp32 [Q_HEADS][C][128], the ungated o / l (the K1 test's form)
//
// **What is pf_flash_attn's, unchanged in structure:** a sub-group owns 8 query rows of one
// head; O is NDA float8 (8 rows x 128 dims) in registers; S for one KV tile is KT/16 float8
// with the row in the component and the key in the lane - the bf16 DPAS A-operand layout, so
// P goes to P·V with no data movement; K^T through the transposed 32-bit 2D read, V through
// the VNNI-transform read; the mask key <= pos + row AND key < depth; the fp32 online softmax;
// P rounded to bf16 as the P·V operand. Tiles start at key 0 and step KT, so a row's tile
// sequence depends on its ABSOLUTE position only, never on the chunk it rides in (a tile
// wholly past a row's bound adds exact zeros: corr = 1, p = 0): the prefill split argument.
//
// **What is K2's:**
//   * HD 128 (NDA 8 dim-atoms), GQA 4 (HPW 4 heads of one kv head per work-group: 4
//     sub-groups x 16 = 64 lanes at RPW 8), the scale 1 / sqrt(128) (not a power of two);
//   * q is read from k2_attn_prep's fp32 rows (each a bf16 value, so the bf16 conversion is
//     exact) with plain loads, once (QREG: 4 GRF at HD 128); a row past C reads row C - 1
//     (finite; its output is never written);
//   * the epilogue (GATE_EPI 1) is k2_attn_reduce's last line, per element:
//       attn_out = rne(f32(rne(o / l)) x f32(rne(softplus(gate))))   (o / l a true division,
//     as the decode reduce's acc / sm), softplus with beta ln 2 and torch's threshold 20.
//
// **EAGER (B70_K2_ATTN=eager, spec 18c's opt-in).** The default path keeps scores and
// probabilities in fp32 (the engine's convention since spec 6). The reference
// (tools/oracle/k2_ref.py attention(), bitwise HF eager in bf16) rounds them: s_b = rne(q·k)
// (a bf16 matmul), s = rne(f32(s_b) x 1/sqrt(128)), p = rne(exp(s - max) / Σ exp(s - max))
// (fp32 softmax, bf16 out), o = rne(Σ p v) (a bf16 matmul), then the gate. EAGER 1 mirrors
// those rounding points in two passes over the row's keys: pass 1 the running max and sum
// (fp32, online), pass 2 p = rne(exp(s - m) / l) and o += p·V; the epilogue rounds o itself
// (no division: p is already normalised). Natural `exp` (EXP2 must be 0). Not bitwise to the
// reference (the dot products' and the sum's orders are DPAS's and the tile walk's, torch's
// are its own) - it moves the ROUNDING POINTS to the reference's, which is what the router-tie
// question needs (spec 18 §10.2 / §11). Twice the QK^T work; for diagnosis, not speed.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef KT
#error "k2_pf_attn: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "k2_pf_attn: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "k2_pf_attn: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HPW)
#error "k2_pf_attn: Q_HEADS, KV_HEADS and HPW must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0 || (Q_HEADS / KV_HEADS) % HPW != 0
#error "k2_pf_attn: q-heads must be a multiple of kv-heads, and the GQA group of HPW"
#endif
#if !defined(EXP2) || !defined(EAGER) || !defined(GATE_EPI)
#error "k2_pf_attn: EXP2, EAGER and GATE_EPI must be defined (0 or 1 each)"
#endif
#if EAGER && EXP2
#error "k2_pf_attn: the eager-mirroring path uses natural exp (EXP2 0), as the reference's softmax"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define SG 16
#define HD 128
#define NDA (HD / 16)                  /* 8 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */
#define SSCALE 0.08838834764831845f    /* 1 / sqrt(128), as fp32 (the reference's d ** -0.5) */
#define SP_BETA 0.6931471805599453f    /* F.softplus(beta = math.log(2)), as fp32 */
#define SP_THRESHOLD 20.0f
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
// torch.nn.functional.softplus(x, beta, threshold = 20) - k2_attn.cl's softplus_k2.
inline float softplus_k2(float x) {
  const float bx = x * SP_BETA;
  return bx > SP_THRESHOLD ? x : log1p(exp(bx)) / SP_BETA;
}
// One score as the walk uses it: the default path scales the fp32 dot; the eager one rounds
// the dot (the bf16 matmul's output) and the scaled score (the bf16 multiply).
inline float score_of(float dot) {
#if EAGER
  return bf16f(rne_bf16(bf16f(rne_bf16(dot)) * SSCALE));
#else
  return dot * SCORE_SCALE;
#endif
}

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void k2_pf_flash_attn(__global const float* restrict Q, __global const ushort* restrict Kc,
                               __global const ushort* restrict Vc, __global const float* restrict G,
#if GATE_EPI
                               __global ushort* restrict out,
#else
                               __global float* restrict out,
#endif
                               uint pos, uint C) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + s / (RPW / 8u);      // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));        // first of 8 rows
  if (r0 >= C) return;                                                  // no barriers below
  const uint depth = pos + C;
  const int kt_w = KV_HEADS * HD * 2, kt_h = (int)depth, kt_p = KV_HEADS * HD * 2;
  const int v_w = KV_HEADS * HD * 2, v_h = (int)depth, v_p = KV_HEADS * HD * 2;
#if !GATE_EPI
  (void)G;
#endif

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
  const uint last = min(depth, pos + r0 + 8u);          // keys this SG's rows can see

  // The score tile at t0: S = Q K^T over KT keys, scored (score_of) and masked. Component
  // r = row r0 + r; lane = key t0 + 16 b + lane.
#define K2_SCORES(t0)                                                                       \
  float8 sacc[NKA];                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk)                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    uint kb[8];                                                                             \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c(                                    \
        (__global void*)Kc, kt_w, kt_h, kt_p,                                               \
        (int2)((int)((j * HD + 16u * kk) / 2u), (int)((t0) + 16u * b)), kb);                \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]); \
  }                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    const uint key = (t0) + 16u * b + l;                                                    \
    K2_MASK(0) K2_MASK(1) K2_MASK(2) K2_MASK(3) K2_MASK(4) K2_MASK(5) K2_MASK(6) K2_MASK(7) \
  }
#define K2_MASK(r) \
  sacc[b].s##r = (key <= pos + r0 + r && key < depth) ? score_of(sacc[b].s##r) : -INFINITY;

  // P·V for one tile: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI).
#define K2_PV(t0)                                                                           \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                  \
    uint vb[32];                                                                            \
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(                                   \
        (__global void*)Vc, v_w, v_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)((t0) + 16u * b)), vb); \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks)                                      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(                                 \
          pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);               \
  }

#define K2_ROWMAX()                                                                         \
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
  mnew.s7 = fmax(m.s7, sub_group_reduce_max(tmax.s7));
#define K2_ROWSUM(psum, rsum)                                                               \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define K2_CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));

#if !EAGER
  // The default path: the online softmax, P = EXPF(s - m) rounded to bf16 for P·V, O / l last.
  for (uint t0 = 0; t0 < last; t0 += KT) {
    K2_SCORES(t0)
    K2_ROWMAX()
    const float8 corr = EXPF(m - mnew);          // m = -INF on the first tile: EXPF(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - mnew);     // masked: EXPF(-INF) = 0
      psum += p;
      K2_CV(0) K2_CV(1) K2_CV(2) K2_CV(3) K2_CV(4) K2_CV(5) K2_CV(6) K2_CV(7)
    }
    K2_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    K2_PV(t0)
  }
#else
  // Pass 1: the row's max and Σ exp(s - max) over its keys (fp32, online).
  for (uint t0 = 0; t0 < last; t0 += KT) {
    K2_SCORES(t0)
    K2_ROWMAX()
    const float8 corr = exp(m - mnew);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - mnew);
    K2_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
  }
  // Pass 2: p = rne(exp(s - m) / l), the softmax's bf16 output; O = Σ p v in fp32.
  for (uint t0 = 0; t0 < last; t0 += KT) {
    K2_SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - m) / lsum;
      K2_CV(0) K2_CV(1) K2_CV(2) K2_CV(3) K2_CV(4) K2_CV(5) K2_CV(6) K2_CV(7)
    }
    K2_PV(t0)
  }
#endif
#undef K2_SCORES
#undef K2_MASK
#undef K2_PV
#undef K2_ROWMAX
#undef K2_ROWSUM
#undef K2_CV

  // The epilogue: per element (row r0 + r, dim 16 dd + lane).
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
      const uint d = 16u * dd + l;
#if GATE_EPI
      const size_t idx = ((size_t)row * Q_HEADS + h) * HD + d;
      const ushort gate_b = rne_bf16(softplus_k2(G[idx]));
      out[idx] = rne_bf16(bf16f(rne_bf16(wv[r])) * bf16f(gate_b));
#else
      out[((size_t)h * C + row) * HD + d] = wv[r];
#endif
    }
  }
}
