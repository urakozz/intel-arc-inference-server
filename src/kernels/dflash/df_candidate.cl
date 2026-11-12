// Spec 19v2 C3: full-vocabulary top64. Each lane sorts at most 16 inputs; subgroup reductions merge their heads.
// The final candidate list contains real ids only. NaN never wins; ties choose the lower id.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL FP_CONTRACT OFF
#define WG 256
#define SG 16
#define C 64
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
