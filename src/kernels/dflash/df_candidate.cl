// Spec 19v2 C3: full-vocabulary top64. Each lane sorts at most 16 inputs; subgroup reductions merge their heads.
// The final candidate list contains real ids only. NaN never wins; ties choose the lower id.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL FP_CONTRACT OFF
#define WG 256
#define SG 16
#ifndef C
#define C 64   /* the candidates a row (spec 19 v3d Task 7: 80 for the 2-bit head) */
#endif
#define CH 4096
#define CHUNKS ((N + CH - 1) / CH)
#define PER1 (CH / WG)
#define PER2 ((CHUNKS * C + WG - 1) / WG)
#define NO_IDX 0x7FFFFFFFu

inline int better(float a, uint ai, float b, uint bi) { return a > b || (a == b && ai < bi); }
inline void insert(float* v, uint* id, uint n, float x, uint i) {
  if (!better(x, i, v[n - 1], id[n - 1])) return;
  uint p = n - 1;
  while (p && better(x, i, v[p - 1], id[p - 1])) {
    v[p] = v[p - 1];
    id[p] = id[p - 1];
    --p;
  }
  v[p] = x;
  id[p] = i;
}
// Three workgroup barriers per pick, independent of C; no private 64-element output arrays.
inline uint pick(float* v, uint* id, uint n, uint* cursor, __local float* sv, __local uint* si, float* value) {
  const uint lid = get_local_id(0), sg = get_sub_group_id();
  const float x = *cursor < n ? v[*cursor] : -INFINITY;
  const uint i = *cursor < n ? id[*cursor] : NO_IDX;
  const float mx = sub_group_reduce_max(x);
  const uint mi = sub_group_reduce_min(x == mx ? i : NO_IDX);
  if (get_sub_group_local_id() == 0) { sv[sg] = mx; si[sg] = mi; }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float best = sv[0];
    uint bi = si[0];
    for (uint s = 1; s < WG / SG; ++s)
      if (better(sv[s], si[s], best, bi)) { best = sv[s]; bi = si[s]; }
    sv[0] = best;
    si[0] = bi;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float best = sv[0];
  const uint bi = si[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (*cursor < n && bi != NO_IDX && id[*cursor] == bi) ++*cursor;
  *value = best;
  return bi;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_candidate_stage1(__global const float* restrict logits, __global float* restrict part) {
  const uint chunk = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  __local float sv[WG / SG];
  __local uint si[WG / SG];
  float v[PER1];
  uint id[PER1];
  for (uint k = 0; k < PER1; ++k) { v[k] = -INFINITY; id[k] = NO_IDX; }
  for (uint k = 0; k < PER1; ++k) {
    const uint i = chunk * CH + k * WG + lid;
    if (i < USED) insert(v, id, PER1, logits[(size_t)r * N + i], i);
  }
  uint cursor = 0;
  for (uint c = 0; c < C; ++c) {
    float value;
    const uint i = pick(v, id, PER1, &cursor, sv, si, &value);
    if (lid == 0) {
      const size_t at = ((size_t)r * CHUNKS + chunk) * C * 2 + c * 2;
      part[at] = value;
      part[at + 1] = (float)i;
    }
  }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_candidate_stage2(__global const float* restrict part, __global uint* restrict cand) {
  const uint r = get_group_id(0), lid = get_local_id(0);
  __local float sv[WG / SG];
  __local uint si[WG / SG];
  float v[PER2];
  uint id[PER2];
  for (uint k = 0; k < PER2; ++k) { v[k] = -INFINITY; id[k] = NO_IDX; }
  for (uint k = 0; k < PER2; ++k) {
    const uint i = k * WG + lid;
    if (i < CHUNKS * C) {
      const size_t at = ((size_t)r * CHUNKS * C + i) * 2;
      const float fi = part[at + 1];
      if (fi < (float)USED) insert(v, id, PER2, part[at], (uint)fi);
    }
  }
  uint cursor = 0;
  for (uint c = 0; c < C; ++c) {
    float value;
    const uint i = pick(v, id, PER2, &cursor, sv, si, &value);
    if (lid == 0) cand[(size_t)r * C + c] = i;
  }
}

// Spec 19 v3d Task 7: stage 1 by a radix select instead of 64 serial picks - the same 64 (value, id) pairs of the
// chunk (the best by `better`: larger value, then the lower id; NaN never), written in id order rather than rank
// order. Stage 2 re-sorts them (its insert is order-free), so `cand` is bitwise stage 1's. A value's key is its
// float bits made orderable; eight 4-bit passes over the WG's 4096 keys find the 64th best key T (a histogram of
// the keys under the prefix found so far, summed over the sub-groups in SLM); every key > T is taken, and of the keys
// equal to T the lowest ids fill the rest (all of them in one go when they all fit - the case without ties).
inline uint okey(float x) {
  if (x != x) return 0u;   // NaN never wins
  const uint u = as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_candidate_select1(__global const float* restrict logits, __global float* restrict part) {
  const uint chunk = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
  __local uint hist[WG / SG][16];
  __local uint tot[16];
  __local uint offs[WG / SG];
  __local uchar eq_take[CH];
  uint key[PER1];
  float val[PER1];
  for (uint k = 0; k < PER1; ++k) {
    const uint i = chunk * CH + k * WG + lid;
    val[k] = i < USED ? logits[(size_t)r * N + i] : -INFINITY;
    key[k] = i < USED ? okey(val[k]) : 0u;
  }
  uint prefix = 0, need = C, eq_count = 0;
  for (int shift = 28; shift >= 0; shift -= 4) {
    const uint hi_mask = shift == 28 ? 0u : ~((1u << (shift + 4)) - 1u);
    uint cnt[16];
    for (uint b = 0; b < 16; ++b) cnt[b] = 0;
    for (uint k = 0; k < PER1; ++k)
      if (key[k] != 0u && (key[k] & hi_mask) == prefix) ++cnt[(key[k] >> shift) & 15u];
    for (uint b = 0; b < 16; ++b) {
      const uint s = sub_group_reduce_add(cnt[b]);
      if (lane == 0) hist[sg][b] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 16) {
      uint s = 0;
      for (uint g = 0; g < WG / SG; ++g) s += hist[g][lid];
      tot[lid] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint b = 15;
    for (;; --b) {   // the digit of the need-th best key under the prefix
      if (tot[b] >= need || b == 0) break;
      need -= tot[b];
    }
    prefix |= b << shift;
    eq_count = tot[b];
    barrier(CLK_LOCAL_MEM_FENCE);   // tot / hist are rewritten by the next pass
  }
  // prefix = T; `need` keys equal to T are taken, eq_count exist (all of them when need == eq_count).
  const bool all_eq = need >= eq_count;
  if (!all_eq) {
    for (uint k = 0; k < PER1; ++k) eq_take[k * WG + lid] = key[k] == prefix && key[k] != 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
      uint left = need;
      for (uint j = 0; j < CH; ++j)
        if (eq_take[j]) {
          if (left) --left;
          else eq_take[j] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint mine = 0;
  for (uint k = 0; k < PER1; ++k) {
    const bool take = key[k] != 0u && (key[k] > prefix || (key[k] == prefix && (all_eq || eq_take[k * WG + lid])));
    mine += take;
  }
  const uint before = sub_group_scan_exclusive_add(mine);
  if (lane == SG - 1) offs[sg] = before + mine;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint at_c = before;
  for (uint g = 0; g < sg; ++g) at_c += offs[g];
  for (uint k = 0; k < PER1; ++k) {
    const bool take = key[k] != 0u && (key[k] > prefix || (key[k] == prefix && (all_eq || eq_take[k * WG + lid])));
    if (take) {
      const size_t at = ((size_t)r * CHUNKS + chunk) * C * 2 + at_c * 2;
      part[at] = val[k];
      part[at + 1] = (float)(chunk * CH + k * WG + lid);
      ++at_c;
    }
  }
}
