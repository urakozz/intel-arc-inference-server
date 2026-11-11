// draft_prob.cl - how sure the MTP head is of its draft (open-work 13: the per-step guess count): the softmax
// maximum and normaliser of one draft list's logits row, so the host reads the draft's probability
// p = 1 / sum_k exp(l_k - max) as two floats instead of the 1 MB row.
//
//   stage 1  grid (GROUPS, 1), WG 256 - group g reduces logits[g*1024 .. +1024) to (its max m_g,
//            s_g = sum exp(l - m_g)) in part[g]
//   stage 2  grid (1, 1),      WG 256 - folds the GROUPS pairs: M = max m_g, S = sum s_g exp(m_g - M),
//            and writes M and S (fp32 bits) to ctrl[CTRL_DSTAT], ctrl[CTRL_DSTAT + 1]
//
// One row (the draft list's M = 1). A tensor-parallel card reduces its own vocabulary half; the host merges the
// two (M, S) pairs. Indices >= VOCAB_USED are masked as argmax.cl masks them, and -inf logits (a draft
// vocabulary's row outside V') add nothing. The result steers only how many drafts are made and verified,
// never which id is drafted or kept: the fixed trees keep it deterministic all the same.

#ifndef VOCAB
#error "draft_prob: VOCAB (the row length) is the variant's"
#endif
#ifndef VOCAB_USED
#error "draft_prob: VOCAB_USED (the real ids in the row) is the variant's"
#endif
#ifndef CTRL_DSTAT
#error "draft_prob: CTRL_DSTAT (runtime::Control::draft_stat) is the variant's"
#endif

#define WG_DPROB 256
#define CHUNK 1024
#define GROUPS ((VOCAB + CHUNK - 1) / CHUNK)
#define PER_LANE (CHUNK / WG_DPROB)
#if GROUPS > WG_DPROB
#error "draft_prob: stage 2's one work-group folds at most 256 groups"
#endif

__attribute__((reqd_work_group_size(WG_DPROB, 1, 1)))
__kernel void draft_prob_stage1(__global const float* restrict logits, __global float* restrict part) {
  const uint g = get_group_id(0);
  const uint lid = get_local_id(0);
  __local float s[WG_DPROB];

  float v[PER_LANE];
  float m = -INFINITY;
  for (uint t = 0; t < PER_LANE; ++t) {
    const uint k = g * CHUNK + t * WG_DPROB + lid;
    v[t] = (k < VOCAB && k < VOCAB_USED) ? logits[k] : -INFINITY;
    m = fmax(m, v[t]);
  }
  s[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_DPROB / 2; stride > 0; stride >>= 1) {
    if (lid < stride) s[lid] = fmax(s[lid], s[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float gm = s[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  float acc = 0.0f;
  if (gm != -INFINITY)
    for (uint t = 0; t < PER_LANE; ++t) acc += exp(v[t] - gm);   // exp(-inf) = 0
  s[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_DPROB / 2; stride > 0; stride >>= 1) {
    if (lid < stride) s[lid] += s[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    part[2 * g + 0] = gm;
    part[2 * g + 1] = s[0];
  }
}

__attribute__((reqd_work_group_size(WG_DPROB, 1, 1)))
__kernel void draft_prob_stage2(__global uint* restrict ctrl, __global const float* restrict part) {
  const uint lid = get_local_id(0);
  __local float s[WG_DPROB];
  const float m = lid < GROUPS ? part[2 * lid + 0] : -INFINITY;
  const float t = lid < GROUPS ? part[2 * lid + 1] : 0.0f;

  s[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_DPROB / 2; stride > 0; stride >>= 1) {
    if (lid < stride) s[lid] = fmax(s[lid], s[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float M = s[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  s[lid] = m == -INFINITY ? 0.0f : t * exp(m - M);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_DPROB / 2; stride > 0; stride >>= 1) {
    if (lid < stride) s[lid] += s[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    ctrl[CTRL_DSTAT + 0] = as_uint(M);
    ctrl[CTRL_DSTAT + 1] = as_uint(s[0]);
  }
}
