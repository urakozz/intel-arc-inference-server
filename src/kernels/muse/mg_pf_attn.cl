// mg_pf_attn.cl - spec 23d: Muse-Glimmer-30B's prefill attention over one chunk of C <= 2048 rows at absolute
// position `pos`, GQA 16 (32 q / 2 kv heads x 128), in two builds of one entry point:
//
//   mg_pf_flash_attn(q, kv_k, kv_v, out, pos, C)
//     q      fp32 [kPfC][Q_HEADS][128]   mg_attn_prep's q at M 2048 (bf16 values: the weight-less q norm, x 3.87,
//                                       RoPE on a sliding layer)
//     kv_k   bf16 [rows][KV_HEADS][128]  this layer's keys: SLIDING the 4096-slot ring (key k at row k & 4095), else
//     kv_v                              the growing NoPE cache (key k at row k); the chunk's own rows were written by
//                                       mg_attn_prep first
//     out    bf16 [C][Q_HEADS x 128]     rne(attention) - the gate is the NEXT launch (mg_pf_prep.cl mg_pf_attn_gate:
//                                       rne(rne(o) x bf16(sigmoid(g))), decode's mg_attn_reduce chain)
//
// **The window** (spec 23 §1: 2048 keys INCLUDING the query, MU:93-102). Row t sits at p = pos + t and sees keys k
// with k <= p and, SLIDING, k > p - 2048. A chunk of 2048 rows needs keys [pos - 2047, pos + 2047]: 4095 positions in
// 4096 slots, so no key a row still reads is overwritten by another row's write of this chunk (the 4096-slot ring's
// reason, spec 23 §5.2). Full layers: causal over the cache from 0, no window, no rotary (NoPE).
//
// **EAGER 0, the flash build** (the default; grid (ceil(C / RPW), KV_HEADS, GQA / HPW), 256 GRF): Kolibri's
// kol_pf_attn.cl default path (spec 20d; 18c's k2_pf_attn walk) at Muse's heads and window, unchanged in structure:
// a sub-group owns 8 query rows of one head; its key tiles run from KT x floor(lo(p0) / KT) - an ABSOLUTE multiple of
// KT, the window's first tile for the group's first row p0 (0 for a full layer) - up to its last row; RING % KT == 0,
// so a tile is KT contiguous ring rows and no 2D block read crosses the ring's end. The fp32 online softmax runs with
// m_safe = (m == -INF ? 0 : m): a tile holding none of a row's keys (the group's first row reaches further back)
// leaves (m, l, o) EXACTLY as they were - a row's result is a function of its absolute position alone, whatever chunk
// it rides in (a split at a multiple of 64 is bitwise; plan 23d Review Focus 2). P rounded to bf16 as the P·V DPAS
// operand, O / l rounded once; natural exp; scale 1 / sqrt(128). Its host twin is tests/kernels/muse_pf_ref.h
// flash_window_sim (fp32, the bar 0.99999 against flash_window_fp64).
//
// **EAGER 1, B70_MUSE_ATTN=eager** (grid (Q_HEADS, C), WG 128, plain OpenCL C - no sub-group function, no block read:
// the Mac's OpenCL runs it): the reference's bf16 eager chain row by row, BITWISE decode's eager row
// (tests/kernels/muse_ref.h attention_eager_head = mg_attn_eager.cl at M 1, itself bitwise transformers 5.19.0's
// eager_attention_forward) - so prefill and decode agree exactly under eager. Per (row, head) a work-group walks the
// row's L keys [lo, p] in 128-key tiles three times (a score is recomputed each pass - the same fma chain, the same
// bits - so no [C][heads][L] score scratch exists):
//   1  s_i = rne(f32(rne(Σ_d q_d·k_d)) x SCALE)   one fma chain, d ascending; the row's max (a tree: exact)
//   2  e_i = exp_torch(s_i - max) into local memory; lane c < 8 adds the tile's e_{c + 8n} ascending - torch's
//      reduce_all lanes (i % 8: a tile starts at a multiple of 128 from lo); sum = the 8 lanes' fixed fold; 1 / sum
//   3  p_i = rne(e_i x (1 / sum)); lane d: o_d = Σ over eager_block(L)-key blocks from i = 0 (each an fma chain from
//      0; the blocks added ascending - a block edge may fall inside a tile); out = rne(o_d)
// tests/kernels/muse_pf_ref.h eager_row is its host twin (bitwise museref::attention_eager_head).
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(EAGER) || !defined(WINDOW) || !defined(RING)
#error "mg_pf_attn: Q_HEADS, KV_HEADS, EAGER, WINDOW and RING must be defined (WINDOW 0 / RING 0: a full layer)"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "mg_pf_attn: q-heads must be a multiple of kv-heads"
#endif
#define SLIDING (WINDOW > 0)
#if SLIDING && (RING == 0 || (RING & (RING - 1)) != 0 || RING < 2048 + WINDOW - 1)
#error "mg_pf_attn: the ring is a power of two holding a chunk of 2048 and the window before it"
#endif
#if !SLIDING && RING != 0
#error "mg_pf_attn: a full layer has no ring (RING 0)"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define SSCALE 0.08838834764831845f    /* 1 / sqrt(128), as fp32 (the reference's d ** -0.5) */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  return (ushort)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}

#if EAGER
// =====================================================================================================================
// The eager build: plain OpenCL C.
#pragma OPENCL FP_CONTRACT OFF
#ifndef TGT
#error "mg_pf_attn EAGER: TGT (eager_block's target block count, decode's 64) must be defined"
#endif
#define ETILE 128
#if SLIDING
#define EROW(k) ((k) & (uint)(RING - 1))
#else
#define EROW(k) (k)
#endif

// Sleef_expf8_u10 (sleefsimdsp.c xexpf) - mg_prep.cl / mg_attn_eager.cl's exp_torch, the same lines.
inline float exp_torch(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}
// The score of key k (absolute) for the q row in local memory: one fma chain, d ascending, two roundings.
inline float eager_score(__local const float* qf, __global const ushort* restrict Kc, uint k, uint j) {
  const __global ushort* kp = Kc + ((size_t)EROW(k) * KV_HEADS + j) * HD;
  float a = 0.0f;
  for (uint d = 0; d < HD; ++d) a = fma(qf[d], bf16f(kp[d]), a);
  return bf16f(rne_bf16(bf16f(rne_bf16(a)) * SSCALE));
}

__attribute__((reqd_work_group_size(ETILE, 1, 1)))
__kernel void mg_pf_flash_attn(__global const float* restrict Q, __global const ushort* restrict Kc,
                               __global const ushort* restrict Vc, __global ushort* restrict out, uint pos, uint C) {
  const uint h = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  __local float qf[HD];
  __local float ev[ETILE];
  __local float red[ETILE];
  __local float lanes[8];
  if (t >= C) return;   // uniform across the work-group
  const uint j = h / GQA, p = pos + t;
#if SLIDING
  const uint lo = p + 1u > (uint)WINDOW ? p + 1u - (uint)WINDOW : 0u;
#else
  const uint lo = 0u;
#endif
  const uint L = p - lo + 1u;
  uint blk = (L + TGT - 1u) / TGT;   // museref::eager_block(L): v2's ppw over the row's span
  blk = (blk + 63u) & ~63u;
  blk = blk < 64u ? 64u : blk;
  qf[lid] = Q[((size_t)t * Q_HEADS + h) * HD + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // pass 1: the row's max
  float mx = -INFINITY;
  for (uint i0 = 0; i0 < L; i0 += ETILE)
    if (i0 + lid < L) mx = fmax(mx, eager_score(qf, Kc, lo + i0 + lid, j));
  red[lid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = ETILE / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] = fmax(red[lid], red[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  mx = red[0];

  // pass 2: the 8 lanes' sums, tile after tile
  float lane_acc = 0.0f;
  for (uint i0 = 0; i0 < L; i0 += ETILE) {
    const uint n = min((uint)ETILE, L - i0);
    ev[lid] = lid < n ? exp_torch(eager_score(qf, Kc, lo + i0 + lid, j) - mx) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 8u)
      for (uint i = lid; i < n; i += 8u) lane_acc += ev[i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid < 8u) lanes[lid] = lane_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sum = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
  const float inv = 1.0f / sum;

  // pass 3: p = rne(e x inv), P·V in eager_block(L)-key blocks; lane d owns dim d
  float o = 0.0f, acc = 0.0f;
  for (uint i0 = 0; i0 < L; i0 += ETILE) {
    const uint n = min((uint)ETILE, L - i0);
    ev[lid] = lid < n ? bf16f(rne_bf16(exp_torch(eager_score(qf, Kc, lo + i0 + lid, j) - mx) * inv)) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < n; ++i) {
      const uint ii = i0 + i;
      if (ii != 0u && ii % blk == 0u) {
        o += acc;
        acc = 0.0f;
      }
      acc = fma(ev[i], bf16f(Vc[((size_t)EROW(lo + ii) * KV_HEADS + j) * HD + lid]), acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[((size_t)t * Q_HEADS + h) * HD + lid] = rne_bf16(o + acc);
}
#undef EROW

#else
// =====================================================================================================================
// The flash build: kol_pf_attn.cl's default path at Muse's heads and window.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef KT
#error "mg_pf_attn: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "mg_pf_attn: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "mg_pf_attn: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#ifndef HPW
#error "mg_pf_attn: HPW (q heads per work-group) must be defined"
#endif
#if (Q_HEADS / KV_HEADS) % HPW != 0
#error "mg_pf_attn: the GQA group must be a multiple of HPW"
#endif
#if SLIDING && RING % KT != 0
#error "mg_pf_attn: the ring is whole KT tiles"
#endif
#define SG 16
#define NDA (HD / 16)                  /* 8 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */

inline float8 m_safe(float8 m) { return select(m, (float8)(0.0f), isequal(m, (float8)(-INFINITY))); }

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mg_pf_flash_attn(__global const float* restrict Q, __global const ushort* restrict Kc,
                               __global const ushort* restrict Vc, __global ushort* restrict out, uint pos, uint C) {
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

  // The score tile at absolute key t0: S = Q K^T over KT keys, scaled and masked. Component r = row r0 + r;
  // lane = key t0 + 16 b + lane; the 2D read at ring row KROW(t0) + 16 b.
#define MG_SCORES(t0)                                                                       \
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
    MG_MASK(0) MG_MASK(1) MG_MASK(2) MG_MASK(3) MG_MASK(4) MG_MASK(5) MG_MASK(6) MG_MASK(7) \
  }
#if SLIDING
#define MG_VISIBLE(r) (key <= p0 + r && key < depth && key + (uint)WINDOW > p0 + r)
#else
#define MG_VISIBLE(r) (key <= p0 + r && key < depth)
#endif
#define MG_MASK(r) sacc[b].s##r = MG_VISIBLE(r) ? sacc[b].s##r * SSCALE : -INFINITY;

  // P·V for one tile: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI).
#define MG_PV(t0)                                                                           \
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

#define MG_ROWMAX()                                                                         \
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
#define MG_ROWSUM(psum, rsum)                                                               \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define MG_CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));

  // The online softmax, P = exp(s - m) rounded to bf16 for P·V, O / l last.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    MG_SCORES(t0)
    MG_ROWMAX()
    const float8 corr = exp(m - msafe);          // m = -INF before a row's first key: exp(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - msafe);     // masked: exp(-INF) = 0
      psum += p;
      MG_CV(0) MG_CV(1) MG_CV(2) MG_CV(3) MG_CV(4) MG_CV(5) MG_CV(6) MG_CV(7)
    }
    MG_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    MG_PV(t0)
  }
#undef MG_SCORES
#undef MG_VISIBLE
#undef MG_MASK
#undef MG_PV
#undef MG_ROWMAX
#undef MG_ROWSUM
#undef MG_CV
#undef KROW

  // The epilogue: per element (row r0 + r, dim 16 dd + lane), bf16 [C][Q_HEADS x 128].
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    const float8 w = o[dd] / lsum;
    float wv[8];
    vstore8(w, 0, wv);
    for (uint r = 0; r < 8; ++r) {
      const uint row = r0 + r;
      if (row >= C) break;
      out[((size_t)row * Q_HEADS + h) * HD + 16u * dd + l] = rne_bf16(wv[r]);
    }
  }
}
#endif  // EAGER
