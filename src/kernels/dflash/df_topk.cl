// df_topk.cl - spec 19b: the TOPK (16) best head logits per draft row (spec 19 §2 "Candidate selector": the head
// keeps the top 16 ids per mask row with their logits), in two fixed stages - no atomics, so a replayed list gives
// the same bits (argmax.cl's argument, generalised from 1 to TOPK). Twin: tests/kernels/dflash_ref.h topk().
//
//   df_topk_stage1  grid (CHUNKS, R), WG 256: chunk c of row j - logits[j][c·CH, +CH) - to its TOPK best
//                   (value, index) pairs, best first, in part[j][c][TOPK][2] (the index as a float: exact below 2^24)
//   df_topk_stage2  grid (R), WG 256: the CHUNKS·TOPK pairs of row j to the row's TOPK best; ids through the
//                   draft vocabulary's table when COMPACT (Review Focus 3: the selector, the walk and the readback see
//                   real ids, never compact indices); values · out_mult, then softcap · tanh(v / softcap) when softcap
//                   > 0 (dflash_ref: the head's logits_processor). Writes cand[j][TOPK] (u32) and unary[j][TOPK].
//
// **The order** is one comparator everywhere: a larger value wins, an equal value goes to the lower index (so the
// lower id - the draft vocabulary's table is ascending), NaN never wins. An index >= USED (the target's padded head
// rows, argmax.cl's VOCAB_USED) is never a candidate. Both stages pick by TOPK rounds of a fixed pairwise tree over
// the lanes' sorted private lists: every comparison is between fixed slots in a fixed sequence.
// A row with fewer than TOPK candidates fills with (-inf, NO_IDX); stage 2 writes NO_IDX as the id.
#pragma OPENCL FP_CONTRACT OFF

#if !defined(N) || !defined(USED) || !defined(R)
#error "N (the logits row), USED (indices below it are candidates) and R (rows) must be defined"
#endif
#ifndef TOPK
#define TOPK 16
#endif
#ifndef CH
#define CH 4096
#endif
#ifndef COMPACT
#define COMPACT 0
#endif
#define WG 256
#define CHUNKS ((N + CH - 1) / CH)
#define NO_IDX 0x7FFFFFFFu
#define PER1 (CH / WG)                              /* stage 1: elements per lane */
#define PER2 ((CHUNKS * TOPK + WG - 1) / WG)        /* stage 2: candidates per lane */
#if (CH % WG) != 0 || CHUNKS * TOPK > (1 << 24)
#error "CH must be a multiple of 256"
#endif

inline int better(float av, uint ai, float bv, uint bi) { return (av > bv) || (av == bv && ai < bi); }

// Insert (v, i) into the lane's sorted list (best first) of TOPK.
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

// TOPK rounds: the lanes' current heads through the fixed tree; the winning lane advances. Lane 0 gets the picks in
// (ov, oi). The heads' indices are unique across lanes, so the winner is the lane whose head index won.
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
__kernel void df_topk_stage1(__global const float* restrict logits, __global float* restrict part) {
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
  for (uint e = 0; e < PER1; ++e) {
    const uint i = c * CH + e * WG + lid;
    if (i < USED && i < N) insert(lv, li, row[i], i);
  }
  merge_rounds(lv, li, sv, si, ov, oi);
  if (lid == 0) {
    __global float* o = part + ((size_t)j * CHUNKS + c) * TOPK * 2;
    for (uint k = 0; k < TOPK; ++k) {
      o[2 * k] = ov[k];
      o[2 * k + 1] = (float)oi[k];   // exact: < 2^24, NO_IDX -> 2147483648.0f, mapped back below
    }
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_topk_stage2(__global const float* restrict part, __global const uint* restrict ids,
                             const float out_mult, const float softcap, __global uint* restrict cand,
                             __global float* restrict unary) {
  __local float sv[WG];
  __local uint si[WG];
  const uint j = get_group_id(0), lid = get_local_id(0);
  float lv[TOPK], ov[TOPK];
  uint li[TOPK], oi[TOPK];
  for (uint k = 0; k < TOPK; ++k) {
    lv[k] = -INFINITY;
    li[k] = NO_IDX;
  }
  __global const float* pr = part + (size_t)j * CHUNKS * TOPK * 2;
  for (uint e = 0; e < PER2; ++e) {
    const uint x = e * WG + lid;
    if (x < CHUNKS * TOPK) {
      const float fi = pr[2 * x + 1];
      const uint i = fi >= 16777216.0f ? NO_IDX : (uint)fi;
      if (i != NO_IDX) insert(lv, li, pr[2 * x], i);
    }
  }
  merge_rounds(lv, li, sv, si, ov, oi);
  if (lid == 0)
    for (uint k = 0; k < TOPK; ++k) {
      float v = ov[k] * out_mult;
      if (softcap > 0.0f) v = tanh(v / softcap) * softcap;
#if COMPACT
      cand[(size_t)j * TOPK + k] = oi[k] == NO_IDX ? NO_IDX : ids[oi[k]];
#else
      cand[(size_t)j * TOPK + k] = oi[k];
#endif
      unary[(size_t)j * TOPK + k] = oi[k] == NO_IDX ? -INFINITY : v;
    }
}
