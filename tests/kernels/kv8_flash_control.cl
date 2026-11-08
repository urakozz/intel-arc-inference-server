// Item 3e: frozen pre-optimization flash kernel from ba99cb9f, test-only.
// Compare every output bit when changing operand loads in src/kernels/kv8.cl.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#define HD 256
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float kv8_f16f(ushort h) {
  const uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0u ? (float)m * 5.9604644775390625e-08f
                          : as_float(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}

// ---------------------------------------------------------------------------
// pf_flash_attn_kv8 - pf_flash_attn.cl (its header is the contract: 8 query rows per
// sub-group, S in DPAS A layout, online softmax, O / l into pf_o), reading the int8 cache.
//
//   pf_flash_attn_kv8(Q, Kc, Ks, Vc, Vs, O, pos, C, rows)
//     Q   bf16 [C][24][256]          rotated q (attn_prep_kv8 PF = 1)
//     Kc  int8 [depth][4][256], Ks fp16 bits [depth][4]   one layer's K rows and scales
//     Vc  int8 [depth][4][256], Vs fp16 bits [depth][4]
//     O   fp32 [24][rows][256]       still in the rotated basis: pf_attn_gate_kv8 un-rotates
//
// K^T operand: lane l = key t0 + 16 b + l, its 16 dims 16 kk .. 16 kk + 15 as 8 bf16 pairs
//   (low half = even dim) - the layout the transposed 32-bit 2D read gives the bf16 kernel -
//   from one 16-byte load of the key's int8 row; an int8 is exactly a bf16. The key's scale
//   multiplies the score: s = (dot q8 * sk) * SCORE_SCALE.
// V operand: lane l = dim, 8 pairs of consecutive keys (VNNI), int8 as bf16 exactly; the
//   key's scale goes into P: pa = rne_bf16(p * sv) (lane = key), the row sum stays sum p.
// Keys at or past `depth` load as 0 and score -INF; nothing past the cache's rows is read.
// ---------------------------------------------------------------------------
#ifndef KT
#error "KT (KV positions per tile, 32 or 64) must be defined"
#endif
#if KT % 32
#error "KT must be a multiple of 32 (the PV loop takes key atoms in pairs)"
#endif
#if RPW % 8
#error "RPW must be a multiple of 8"
#endif
#define SG 16
#define NDA (HD / 16)
#define NKA (KT / 16)
#define SGS (HPW * RPW / 8)
#define ATTN_SCALE (1.0f / 16.0f)
#ifndef EXP2
#error "EXP2 must be defined"
#endif
#if EXP2
#define SCORE_SCALE (ATTN_SCALE * M_LOG2E_F)
#define EXPF(x) exp2(x)
#else
#define SCORE_SCALE ATTN_SCALE
#define EXPF(x) exp(x)
#endif

// An int8 as its exact bf16 bits.
inline uint kv8_bf(char c) { return as_uint((float)c) >> 16; }

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_flash_attn_kv8(__global const ushort* restrict Q,
                                __global const char* restrict Kc,
                                __global const ushort* restrict Ks,
                                __global const char* restrict Vc,
                                __global const ushort* restrict Vs,
                                __global float* restrict O, uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);
  const uint h = j * 6u + get_group_id(2) * HPW + s / (RPW / 8u);
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));
  if (r0 >= C) return;
  const uint depth = pos + C;
  const int q_w = 24 * HD * 2, q_h = (int)C, q_p = 24 * HD * 2;

  short8 qa[NDA];
#define LOAD_Q()                                                                      \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; kk += 2) {                            \
    ushort t[16];                                                                     \
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,       \
                                              (int2)((int)(h * HD + 16u * kk), (int)r0), t); \
    qa[kk] = as_short8(vload8(0, t));                                                 \
    qa[kk + 1] = as_short8(vload8(0, t + 8));                                         \
  }
#if QREG
  LOAD_Q();
#endif
  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);
  for (uint t0 = 0; t0 < last; t0 += KT) {
#if !QREG
    LOAD_Q();
#endif
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        const uint key = t0 + 16u * b + l;
        uint kb[8];
        if (key < depth) {
          const char16 c = vload16(0, Kc + ((size_t)key * 4u + j) * HD + 16u * kk);
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
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
      }
    // scale by the key's K scale, mask, running max (component r = row r0 + r, lane = key)
    float8 tmax = (float8)(-INFINITY);
    float svl[NKA];   // this lane's key's V scale, per key atom
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const uint key = t0 + 16u * b + l;
      const bool live = key < depth;
      const float sk = live ? kv8_f16f(Ks[(size_t)key * 4u + j]) : 0.0f;
      svl[b] = live ? kv8_f16f(Vs[(size_t)key * 4u + j]) : 0.0f;
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && live) ? (sacc[b].s##r * sk) * SCORE_SCALE : -INFINITY;
      MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
      tmax = fmax(tmax, sacc[b]);
    }
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
    const float8 corr = EXPF(m - mnew);
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - mnew);
      psum += p;
      const float8 ps = p * svl[b];
#define CV(r) pa[b].s##r = as_short(rne_bf16(ps.s##r));
      CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
    }
    float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
    SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    // O += (P s) V8: A = pa[b] (8 rows x 16 keys), B = V8 (16 keys x 16 dims, VNNI)
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA; dd += 2) {
        uint vb[32];   // [c * 16 + jj]: c = dim half, jj = key pair of the 32 keys
#pragma unroll
        for (uint c = 0; c < 2u; ++c) {
          const size_t col = (size_t)j * HD + 16u * (dd + c) + l;
#pragma unroll
          for (uint jj = 0; jj < 16u; ++jj) {
            const uint k0 = t0 + 16u * b + 2u * jj;
            const uint lo = k0 < depth ? kv8_bf(Vc[(size_t)k0 * 4u * HD + col]) : 0u;
            const uint hi = k0 + 1u < depth ? kv8_bf(Vc[(size_t)(k0 + 1u) * 4u * HD + col]) : 0u;
            vb[c * 16u + jj] = lo | (hi << 16);
          }
        }
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
            o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);
      }
  }
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
}
