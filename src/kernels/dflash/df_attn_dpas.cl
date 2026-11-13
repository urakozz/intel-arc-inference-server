// df_attn_dpas.cl - spec 19 v3d Task 6: df_attn_part on DPAS. The drafter's attention over its context ring (spec
// 19b's df_attn.cl: the window, the ring, the partial layout - read its header first) with the scores and the P V
// products on the card's matrix instruction instead of a scalar fp32 walk. It writes the SAME partials, so the
// reduce is df_attn_rows.cl's df_attn_reduce (the _H1_B4 binary), unchanged:
//
//   df_attn_part(ctrl, dctl, q, ring_k, ring_v, scale, part)   grid (NKV, TGT), WG 16 G (one sub-group a q head)
//     part fp32 [NH][TGT][M][HD + 2] {m, l, acc[HD]} - df_attn.cl's meaning: over the block's PB keys, m = max of
//     the row's live scores, l = sum exp(s - m), acc[d] = sum exp(s - m) v[d]; a row with no live key: -INF, 0, 0
//
// Inside a work-group (kv head kvh, ring block b): sub-group g is q head kvh G + g and its M = 8 rows - one DPAS row
// block. Per key atom (16 keys, lane = key):
//   S (8 rows x 16 keys) = sum over the HD / 16 dim atoms of DPAS(q atom, K atom) with q = hi + lo (hi = rne_bf16(q),
//     lo = rne_bf16(q - hi); hi first, then lo - attn_verify_dpas.cl's QSPLIT); K straight from the ring row (a key's
//     16 dims are its lane's 8 packed words, the B operand's layout); s = S * scale, -INF outside the row's window or
//     the block's live positions, or for a row past K - df_attn.cl's masks.
// The scores go to SLM; each row's max over the block; then per key atom p = df_exp(s - m) (0 for a dead score),
// l += p (per lane, summed over the lanes at the end), and O += P V with P = hi + lo of p (attn_verify_dpas.cl's
// PSPLIT, hi first), P in S's layout as the A operand and V's key pairs packed per lane (lane = output dim).
// Against the scalar form the products are DPAS's (fp32 accumulation of bf16 pairs, ~16 significant bits of q and
// p) and the sums' order differs: a cosine bar, not bitwise (tests/kernels/dflash_kernels_test.cc). A row's result
// depends only on its own q row (a DPAS output row is its A row's): property R.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL FP_CONTRACT OFF

#if !defined(NH) || !defined(NKV) || !defined(HD) || !defined(RING) || !defined(WIN) || !defined(PB)
#error "NH, NKV, HD, RING, WIN and PB must be defined"
#endif
#ifndef M
#define M 8
#endif
#define G (NH / NKV)
#define TGT (RING / PB)
#define PART (HD + 2)
#define SG 16
#define DA (HD / 16)   // dim atoms
#define KA (PB / 16)   // key atoms
#if M != 8 || (HD % 16) != 0 || (PB % 16) != 0 || (RING % PB) != 0 || RING < WIN
#error "df_attn_dpas: M 8 (one DPAS row block), HD and PB multiples of 16, RING a multiple of PB and >= WIN"
#endif

inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)0x7FC0;
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// df_attn.cl's exp (bitwise dfref::attn_exp), not the device's.
inline float df_exp(float x) {
  if (x != x) return x;
  if (x < -86.0f) return 0.0f;
  if (x > 88.0f) return INFINITY;
  const float n = rint(x * 1.44269504088896341f);
  const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  const float y = p * (r * r) + r + 1.0f;
  return y * as_float((uint)((int)n + 127) << 23);
}

inline float f8(float8 v, int i) {
  switch (i) {
    case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
    case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7;
  }
}

__attribute__((reqd_work_group_size(SG * G, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void df_attn_part(__global const uint* restrict ctrl, __global const uint* restrict dctl,
                           __global const float* restrict q, __global const ushort* restrict ring_k,
                           __global const ushort* restrict ring_v, const float scale, __global float* restrict part) {
  __local float s_all[G][M][PB];   // the scores, then read back by row for the max
  const uint kvh = get_group_id(0), b = get_group_id(1);
  const uint g = get_sub_group_id(), lane = get_sub_group_local_id(), h = kvh * G + g;
  const uint p = ctrl[CTRL_POS], K = dctl[DCTL_K];

  // ---- q as the A operand, hi and lo, once: row i of dim atom a at lane = the atom's dim -------------------------
  short8 qh[DA], ql[DA];
  for (uint a = 0; a < DA; ++a)
    for (int i = 0; i < 8; ++i) {
      const float x = q[((size_t)i * NH + h) * HD + a * 16 + lane];
      const ushort hi = f2bf(x);
      const ushort lo = f2bf(x - bf16f(hi));
      switch (i) {
        case 0: qh[a].s0 = hi; ql[a].s0 = lo; break; case 1: qh[a].s1 = hi; ql[a].s1 = lo; break;
        case 2: qh[a].s2 = hi; ql[a].s2 = lo; break; case 3: qh[a].s3 = hi; ql[a].s3 = lo; break;
        case 4: qh[a].s4 = hi; ql[a].s4 = lo; break; case 5: qh[a].s5 = hi; ql[a].s5 = lo; break;
        case 6: qh[a].s6 = hi; ql[a].s6 = lo; break; default: qh[a].s7 = hi; ql[a].s7 = lo; break;
      }
    }

  // ---- S: per key atom, the HD / 16 dim atoms, q hi then lo ----------------------------------------------------
  for (uint ka = 0; ka < KA; ++ka) {
    const uint off = b * PB + ka * 16 + lane;   // this lane's key: ring offset
    const long pos = (long)p - (long)RING + (long)off;
    const uint slot = (uint)(((ulong)p + off) % RING);
    __global const uint* krow = (__global const uint*)(ring_k + ((size_t)slot * NKV + kvh) * HD);
    float8 acc = (float8)0.0f;
    for (uint a = 0; a < DA; ++a) {
      const int8 kb = as_int8(vload8(0, krow + a * 8));
      acc = intel_sub_group_bf16_bf16_matrix_mad_k16(qh[a], kb, acc);
      acc = intel_sub_group_bf16_bf16_matrix_mad_k16(ql[a], kb, acc);
    }
    for (int i = 0; i < 8; ++i) {
      const long lo = (long)p + (long)i - (long)(WIN - 1);
      const bool live = pos >= 0 && pos >= lo && (uint)i <= K;
      s_all[g][i][ka * 16 + lane] = live ? f8(acc, i) * scale : -INFINITY;
    }
  }
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);

  // ---- each row's max over the block ----------------------------------------------------------------------------
  float mrow[8];
  for (int i = 0; i < 8; ++i) {
    float m = -INFINITY;
    for (uint t = lane; t < PB; t += SG) m = fmax(m, s_all[g][i][t]);
    mrow[i] = sub_group_reduce_max(m);
  }

  // ---- P V: per key atom, p in S's layout (lane = key), V's key pairs per lane (lane = dim) ------------------------
  float8 o[DA];
  for (uint a = 0; a < DA; ++a) o[a] = (float8)0.0f;
  float lsum[8];
  for (int i = 0; i < 8; ++i) lsum[i] = 0.0f;
  for (uint ka = 0; ka < KA; ++ka) {
    short8 ph, pl;
    for (int i = 0; i < 8; ++i) {
      const float sc = s_all[g][i][ka * 16 + lane];
      const float e = (mrow[i] == -INFINITY || sc == -INFINITY) ? 0.0f : df_exp(sc - mrow[i]);
      lsum[i] += e;
      const ushort hi = f2bf(e);
      const ushort lo = f2bf(e - bf16f(hi));
      switch (i) {
        case 0: ph.s0 = hi; pl.s0 = lo; break; case 1: ph.s1 = hi; pl.s1 = lo; break;
        case 2: ph.s2 = hi; pl.s2 = lo; break; case 3: ph.s3 = hi; pl.s3 = lo; break;
        case 4: ph.s4 = hi; pl.s4 = lo; break; case 5: ph.s5 = hi; pl.s5 = lo; break;
        case 6: ph.s6 = hi; pl.s6 = lo; break; default: ph.s7 = hi; pl.s7 = lo; break;
      }
    }
    uint vslot[16];
    for (int t = 0; t < 16; ++t) vslot[t] = (uint)(((ulong)p + b * PB + ka * 16 + t) % RING);
    for (uint a = 0; a < DA; ++a) {
      int8 vb;
      for (int j = 0; j < 8; ++j) {
        const ushort v0 = ring_v[((size_t)vslot[2 * j] * NKV + kvh) * HD + a * 16 + lane];
        const ushort v1 = ring_v[((size_t)vslot[2 * j + 1] * NKV + kvh) * HD + a * 16 + lane];
        const int w = (int)((uint)v0 | ((uint)v1 << 16));
        switch (j) {
          case 0: vb.s0 = w; break; case 1: vb.s1 = w; break; case 2: vb.s2 = w; break; case 3: vb.s3 = w; break;
          case 4: vb.s4 = w; break; case 5: vb.s5 = w; break; case 6: vb.s6 = w; break; default: vb.s7 = w; break;
        }
      }
      o[a] = intel_sub_group_bf16_bf16_matrix_mad_k16(ph, vb, o[a]);
      o[a] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl, vb, o[a]);
    }
  }

  // ---- the partials ----------------------------------------------------------------------------------------------
  for (int i = 0; i < 8; ++i) {
    const float l = sub_group_reduce_add(lsum[i]);
    __global float* dst = part + (((size_t)h * TGT + b) * M + i) * PART;
    if (lane == 0) {
      dst[0] = mrow[i];
      dst[1] = l;
    }
    for (uint a = 0; a < DA; ++a) dst[2 + a * 16 + lane] = f8(o[a], i);
  }
}
