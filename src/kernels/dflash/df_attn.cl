// df_attn.cl - spec 19b: the DFlash drafter's attention (spec 19 §2, §5 "Kernels"): the M block rows' queries over the
// context ring (target-derived K / V of committed positions) plus the block's own K / V, non-causal inside the block.
// Twin: tests/kernels/dflash_ref.h attention() (same order; FP_CONTRACT off).
//
// **The window** (tools/oracle/dflash_ref.py, vLLM's non-causal sliding layer): a query at position q sees a key at
// position k iff |q - k| <= WIN - 1. Block row r sits at p + r (p = ctrl.pos, the anchor's position), the context is
// positions < p, so row r sees context positions max(0, p + r - (WIN - 1)) .. p - 1, and every block row 0..K.
//
// **The ring** holds position pos at slot pos % RING (RING >= WIN; the commit side, df_ctx_store, writes it). Context
// block b of TGT = RING / PB covers ring offsets t = b·PB .. b·PB + PB - 1: offset t is position p - RING + t at slot
// (p + t) % RING - the RING positions before p, oldest first. A position below 0 or below the row's bound is masked.
//
//   df_attn_part    grid (NKV, TGT), WG WGA: one kv head's PB context keys against its G = NH / NKV q heads x M rows
//                   (P = G·M pairs): scores (q · k, ascending d) · scale, then per pair m = max, e = exp(s - m),
//                   l = Σ e (ascending t), acc[d] = Σ e · v[d] (ascending t) -> part[q head][b][r] = (m, l, acc[HD])
//   df_attn_reduce  grid (NH, M), WG HD: row r > K writes zeros; else the block keys' scores (rows 0..K, ascending
//                   d), m over the blocks then the keys, l and acc merged blocks-ascending then keys-ascending,
//                   out = bf16(acc / l) at attn_out[r][h·HD + d]
//
// The block's K / V (df_qkv_prep's kb / vb, fp32) are read here and never written to the ring (Review Focus 2).
#pragma OPENCL FP_CONTRACT OFF

#if !defined(NH) || !defined(NKV) || !defined(HD) || !defined(RING) || !defined(WIN) || !defined(PB)
#error "NH, NKV, HD, RING, WIN and PB must be defined"
#endif
#ifndef M
#define M 8
#endif
#ifndef WGA
#define WGA 128
#endif
#ifndef CAUSAL
#define CAUSAL 0
#endif
#define G (NH / NKV)
#define P (G * M)
#define TGT (RING / PB)
#define PART (HD + 2)
#if (RING % PB) != 0 || RING < WIN
#error "RING must be a multiple of PB and at least WIN"
#endif
#if HD < M || (HD & (HD - 1)) != 0
#error "df_attn_reduce's lanes are the head dim: HD must be a power of two and at least M"
#endif

inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)0x7FC0;
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((reqd_work_group_size(WGA, 1, 1)))
__kernel void df_attn_part(__global const uint* restrict ctrl, __global const uint* restrict dctl,
                           __global const float* restrict q, __global const ushort* restrict ring_k,
                           __global const ushort* restrict ring_v, const float scale, __global float* restrict part) {
  __local float qs[P * HD];   // the G heads' M rows of q: pair pr = gh · M + r
  __local float s[P * PB];    // scores, then e = exp(s - m)
  __local float ml[2 * P];
  const uint kvh = get_group_id(0), b = get_group_id(1), lid = get_local_id(0);
  const uint p = ctrl[CTRL_POS], K = dctl[DCTL_K];
  for (uint e = lid; e < P * HD; e += WGA) {
    const uint pr = e / HD, d = e % HD, gh = pr / M, r = pr % M;
    qs[e] = q[((size_t)r * NH + kvh * G + gh) * HD + d];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint t = lid; t < PB; t += WGA) {
    const uint off = b * PB + t;
    const long pos = (long)p - (long)RING + (long)off;
    const uint slot = (uint)(((ulong)p + off) % RING);
    __global const ushort* k = ring_k + ((size_t)slot * NKV + kvh) * HD;
    for (uint pr = 0; pr < P; ++pr) {
      const uint r = pr % M;
      const long lo = (long)p + (long)r - (long)(WIN - 1);
      float sc = -INFINITY;
      if (pos >= 0 && pos >= lo && r <= K) {
        float a = 0.0f;
        for (uint d = 0; d < HD; ++d) a = a + qs[pr * HD + d] * bf16f(k[d]);
        sc = a * scale;
      }
      s[pr * PB + t] = sc;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint pr = lid; pr < P; pr += WGA) {
    float m = -INFINITY;
    for (uint t = 0; t < PB; ++t) m = fmax(m, s[pr * PB + t]);
    float l = 0.0f;
    for (uint t = 0; t < PB; ++t) {
      const float sc = s[pr * PB + t];
      const float e = (m == -INFINITY || sc == -INFINITY) ? 0.0f : exp(sc - m);
      s[pr * PB + t] = e;
      l = l + e;
    }
    ml[2 * pr] = m;
    ml[2 * pr + 1] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint d = lid; d < HD; d += WGA) {
    float acc[P];
    for (uint pr = 0; pr < P; ++pr) acc[pr] = 0.0f;
    for (uint t = 0; t < PB; ++t) {
      const uint slot = (uint)(((ulong)p + b * PB + t) % RING);
      const float v = bf16f(ring_v[((size_t)slot * NKV + kvh) * HD + d]);
      for (uint pr = 0; pr < P; ++pr) {
        const float e = s[pr * PB + t];
        if (e != 0.0f) acc[pr] = acc[pr] + e * v;
      }
    }
    for (uint pr = 0; pr < P; ++pr) {
      const uint gh = pr / M, r = pr % M;
      __global float* o = part + (((size_t)(kvh * G + gh) * TGT + b) * M + r) * PART;
      o[2 + d] = acc[pr];
      if (d == 0) {
        o[0] = ml[2 * pr];
        o[1] = ml[2 * pr + 1];
      }
    }
  }
}

__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void df_attn_reduce(__global const uint* restrict ctrl, __global const uint* restrict dctl,
                             __global const float* restrict part, __global const float* restrict q,
                             __global const float* restrict kb, __global const float* restrict vb, const float scale,
                             __global ushort* restrict out) {
  __local float sb[M];
  __local float wts[TGT + M];
  __local float mlsum[2];
  const uint h = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  const uint K = dctl[DCTL_K], kvh = h / G;
  __global ushort* o = out + (size_t)r * NH * HD + (size_t)h * HD;
  if (r > K) {
    o[d] = 0;
    return;
  }
  __global const float* qr = q + ((size_t)r * NH + h) * HD;
  if (d < M) {
    float sc = -INFINITY;
    if (d <= K && (!CAUSAL || d <= r)) {
      __global const float* k = kb + ((size_t)d * NKV + kvh) * HD;
      float a = 0.0f;
      for (uint i = 0; i < HD; ++i) a = a + qr[i] * k[i];
      sc = a * scale;
    }
    sb[d] = sc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const float* pb = part + ((size_t)h * TGT * M + r) * PART;   // block b at pb + b · M · PART
  if (d == 0) {
    float m = -INFINITY;
    for (uint b = 0; b < TGT; ++b) m = fmax(m, pb[(size_t)b * M * PART]);
    for (uint j = 0; j < M; ++j) m = fmax(m, sb[j]);
    float l = 0.0f;
    for (uint b = 0; b < TGT; ++b) {
      const float mb = pb[(size_t)b * M * PART];
      const float w = mb == -INFINITY ? 0.0f : exp(mb - m);
      wts[b] = w;
      l = l + pb[(size_t)b * M * PART + 1] * w;
    }
    for (uint j = 0; j < M; ++j) {
      const float w = sb[j] == -INFINITY ? 0.0f : exp(sb[j] - m);
      wts[TGT + j] = w;
      l = l + w;
    }
    mlsum[0] = m;
    mlsum[1] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc = 0.0f;
  for (uint b = 0; b < TGT; ++b) acc = acc + pb[(size_t)b * M * PART + 2 + d] * wts[b];
  for (uint j = 0; j < M; ++j) {
    const float w = wts[TGT + j];
    if (w != 0.0f) acc = acc + w * vb[((size_t)j * NKV + kvh) * HD + d];
  }
  o[d] = f2bf(acc / mlsum[1]);
}
