// ll_topk.cl - spec 26b: the LiLiCorr candidates (spec 26 §2.2, V/models/lilicorr.py:478-486) - per draft row the TOPK (8)
// best RAW head logits and their log-probabilities over the WHOLE vocabulary: lp = v - logsumexp(row). df_topk.cl's two
// fixed stages at TOPK 8 (a family copy; df_topk.cl is not edited) plus the row's logsumexp, merged in a fixed order - no
// atomics, so a replayed list gives the same bits. Twin: tests/kernels/lilicorr_ref.h topk_lse().
//
//   ll_topk_stage1  grid (CHUNKS, R), WG 256: chunk c of row j - logits[j][c·CH, +CH) - to its TOPK best (value, index),
//                   best first, AND the chunk's (m_c, s_c): m_c = max of its non-NaN values (a lane max, then the tree),
//                   s_c = Σ exp(v - m_c) over its non-NaN, non -inf values (v == m_c adds exactly 1) - lane partials over
//                   i = c·CH + e·WG + lane, e ascending, then strides WG / 2 .. 1. part[j][c] = [TOPK (v, i)] + (m_c, s_c)
//   ll_topk_stage2  grid (R), WG 256: the CHUNKS·TOPK pairs to the row's TOPK best (df_topk_stage2's rounds); lane 0 merges
//                   the chunks IN CHUNK ORDER: M = max m_c, S = Σ_c s_c · exp(m_c - M) (a chunk of m_c = -inf adds 0;
//                   m_c == M multiplies by exactly 1), lse = M + log(S) (-inf when every chunk is). Writes cand[j][TOPK]
//                   (u32; NO_IDX past a short row), logit[j][TOPK] (the raw v) and lp[j][TOPK] = v - lse
//
// **The order** is df_topk's comparator: a larger value wins, an equal value goes to the lower index, NaN never wins (nor
// enters the sum); indices >= USED are never candidates nor summed. A huge finite value gives lse = it and its lp 0 - no
// NaN (Review Focus 3).
#pragma OPENCL FP_CONTRACT OFF

#if !defined(N) || !defined(USED) || !defined(R)
#error "N (the logits row), USED (indices below it are candidates) and R (rows) must be defined"
#endif
#ifndef TOPK
#define TOPK 8
#endif
#ifndef CH
#define CH 4096
#endif
#define WG 256
#define CHUNKS ((N + CH - 1) / CH)
#define NO_IDX 0x7FFFFFFFu
#define PER1 (CH / WG)
#define PER2 ((CHUNKS * TOPK + WG - 1) / WG)
#define PSTRIDE (2 * TOPK + 2)                     /* floats per (row, chunk) in `part` */
#if (CH % WG) != 0 || CHUNKS * TOPK > (1 << 24)
#error "CH must be a multiple of 256"
#endif

inline int better(float av, uint ai, float bv, uint bi) { return (av > bv) || (av == bv && ai < bi); }

inline void insert(float* lv, uint* li, float v, uint i) {
  if (!better(v, i, lv[TOPK - 1], li[TOPK - 1])) return;
  int k = TOPK - 1;
  while (k > 0 && better(v, i, lv[k - 1], li[k - 1])) {
    lv[k] = lv[k - 1];
    li[k] = li[k - 1];
    --k;
  }
  lv[k] = v;
  li[k] = i;
}

inline void merge_rounds(float* lv, uint* li, __local float* sv, __local uint* si, float* ov, uint* oi) {
  const uint lid = get_local_id(0);
  uint ptr = 0;
  for (uint k = 0; k < TOPK; ++k) {
    sv[lid] = ptr < TOPK ? lv[ptr] : -INFINITY;
    si[lid] = ptr < TOPK ? li[ptr] : NO_IDX;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = WG / 2; s > 0; s >>= 1) {
      if (lid < s && better(sv[lid + s], si[lid + s], sv[lid], si[lid])) {
        sv[lid] = sv[lid + s];
        si[lid] = si[lid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float wv = sv[0];
    const uint wi = si[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ptr < TOPK && wi != NO_IDX && li[ptr] == wi) ++ptr;
    if (lid == 0) {
      ov[k] = wv;
      oi[k] = wi;
    }
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_topk_stage1(__global const float* restrict logits, __global float* restrict part) {
  __local float sv[WG];
  __local uint si[WG];
  const uint c = get_group_id(0), j = get_group_id(1), lid = get_local_id(0);
  float lv[TOPK], ov[TOPK];
  uint li[TOPK], oi[TOPK];
  for (uint k = 0; k < TOPK; ++k) {
    lv[k] = -INFINITY;
    li[k] = NO_IDX;
  }
  __global const float* row = logits + (size_t)j * N;
  float lm = -INFINITY;
  for (uint e = 0; e < PER1; ++e) {
    const uint i = c * CH + e * WG + lid;
    if (i < USED && i < N) {
      const float v = row[i];
      insert(lv, li, v, i);
      if (!isnan(v)) lm = fmax(lm, v);
    }
  }
  merge_rounds(lv, li, sv, si, ov, oi);
  // the chunk's max (a tree of lane maxima) and Σ exp(v - m) (lane partials, then the tree)
  sv[lid] = lm;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (lid < s) sv[lid] = fmax(sv[lid], sv[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float m = sv[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float a = 0.0f;
  if (m != -INFINITY)
    for (uint e = 0; e < PER1; ++e) {
      const uint i = c * CH + e * WG + lid;
      if (i >= USED || i >= N) continue;
      const float v = row[i];
      if (isnan(v) || v == -INFINITY) continue;
      a = a + (v == m ? 1.0f : exp(v - m));
    }
  sv[lid] = a;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (lid < s) sv[lid] = sv[lid] + sv[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    __global float* o = part + ((size_t)j * CHUNKS + c) * PSTRIDE;
    for (uint k = 0; k < TOPK; ++k) {
      o[2 * k] = ov[k];
      o[2 * k + 1] = (float)oi[k];   // exact: < 2^24, NO_IDX -> 2147483648.0f, mapped back below
    }
    o[2 * TOPK] = m;
    o[2 * TOPK + 1] = sv[0];
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_topk_stage2(__global const float* restrict part, __global uint* restrict cand,
                             __global float* restrict logit, __global float* restrict lp) {
  __local float sv[WG];
  __local uint si[WG];
  const uint j = get_group_id(0), lid = get_local_id(0);
  float lv[TOPK], ov[TOPK];
  uint li[TOPK], oi[TOPK];
  for (uint k = 0; k < TOPK; ++k) {
    lv[k] = -INFINITY;
    li[k] = NO_IDX;
  }
  __global const float* pr = part + (size_t)j * CHUNKS * PSTRIDE;
  for (uint e = 0; e < PER2; ++e) {
    const uint x = e * WG + lid;
    if (x < CHUNKS * TOPK) {
      __global const float* q = pr + (size_t)(x / TOPK) * PSTRIDE + 2 * (x % TOPK);
      const float fi = q[1];
      const uint i = fi >= 16777216.0f ? NO_IDX : (uint)fi;
      if (i != NO_IDX) insert(lv, li, q[0], i);
    }
  }
  merge_rounds(lv, li, sv, si, ov, oi);
  if (lid == 0) {
    float M = -INFINITY;
    for (uint c = 0; c < CHUNKS; ++c) M = fmax(M, pr[(size_t)c * PSTRIDE + 2 * TOPK]);
    float S = 0.0f;
    for (uint c = 0; c < CHUNKS; ++c) {
      const float mc = pr[(size_t)c * PSTRIDE + 2 * TOPK], sc = pr[(size_t)c * PSTRIDE + 2 * TOPK + 1];
      if (mc != -INFINITY) S = S + sc * (mc == M ? 1.0f : exp(mc - M));
    }
    const float lse = M == -INFINITY ? -INFINITY : M + log(S);
    for (uint k = 0; k < TOPK; ++k) {
      cand[(size_t)j * TOPK + k] = oi[k];
      logit[(size_t)j * TOPK + k] = oi[k] == NO_IDX ? -INFINITY : ov[k];
      lp[(size_t)j * TOPK + k] = oi[k] == NO_IDX ? -INFINITY : ov[k] - lse;
    }
  }
}
