// mg_argmax.cl - spec 23f: the verify lists' greedy argmax at M <= 16 rows. A COPY of src/kernels/argmax.cl's
// two stages (which is untouched, its binaries too - MG0 / D0): argmax.cl caps M at 8 (Control::out_token holds 8
// ids) and its stage 2 advances Control::pos, while a verify of M = K + 1 rows (K <= 15, spec 23 decision 10) writes
// every row's id to its own buffer and leaves pos to the host's commit (runtime::muse::MuseEngine::commit).
//
//   mg_vargmax_stage1(logits, part)        grid (GROUPS, M), WG 256   argmax.cl's argmax_stage1, line for line
//   mg_vargmax_stage2(ctrl, part, vout)    grid (1, 1),      WG 256   rows m < min(ctrl.n_active, M): vout[m] =
//                                                                     the row's argmax; Control is only read
//
// The comparator and both trees are argmax.cl's (strictly greater wins, an exact tie goes to the LOWER index), so a
// verify row's id is bitwise the plain step's for the same logits row (spec 23f D3). VOCAB_USED = VOCAB at Muse's
// 202048 (every id real).
#ifndef M
#define M 1
#endif
#if M < 1 || M > 16
#error "mg_vargmax: M is 1..16 (the verify's rows)"
#endif
#if !defined(VOCAB) || !defined(VOCAB_USED) || !defined(CTRL_NACT)
#error "mg_vargmax: VOCAB, VOCAB_USED and CTRL_NACT must be defined"
#endif

#define WG_ARGMAX 256
#define CHUNK 1024
#define GROUPS ((VOCAB + CHUNK - 1) / CHUNK)
#define NO_IDX 0x7FFFFFFFu
#if GROUPS > WG_ARGMAX
#error "mg_vargmax: stage 2 folds at most 256 groups"
#endif

inline int argmax_better(float av, uint ai, float bv, uint bi) {
  return (av > bv) || (av == bv && ai < bi);
}

__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void mg_vargmax_stage1(__global const float* restrict logits, __global float* restrict part) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];

  const uint k0 = g * CHUNK;
  uint k1 = k0 + CHUNK;
  if (k1 > VOCAB) k1 = VOCAB;

  __global const float* restrict row = logits + (size_t)m * VOCAB;
  float bv = -INFINITY;
  uint bi = NO_IDX;
  for (uint k = k0 + lid; k < k1; k += WG_ARGMAX) {
    const float v = (k < VOCAB_USED) ? row[k] : -INFINITY;
    if (argmax_better(v, k, bv, bi)) { bv = v; bi = k; }
  }

  sv[lid] = bv;
  si[lid] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_ARGMAX / 2; stride > 0; stride >>= 1) {
    if (lid < stride && argmax_better(sv[lid + stride], si[lid + stride], sv[lid], si[lid])) {
      sv[lid] = sv[lid + stride];
      si[lid] = si[lid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    part[((size_t)m * GROUPS + g) * 2 + 0] = sv[0];
    part[((size_t)m * GROUPS + g) * 2 + 1] = (float)si[0];   // exact: idx < 2^24
  }
}

__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void mg_vargmax_stage2(__global const uint* restrict ctrl, __global const float* restrict part,
                                __global uint* restrict vout) {
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];
  uint n = ctrl[CTRL_NACT];
  if (n > M) n = M;

  for (uint m = 0; m < n; ++m) {
    float bv = -INFINITY;
    uint bi = NO_IDX;
    if (lid < GROUPS) {
      bv = part[((size_t)m * GROUPS + lid) * 2 + 0];
      bi = (uint)part[((size_t)m * GROUPS + lid) * 2 + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sv[lid] = bv;
    si[lid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG_ARGMAX / 2; stride > 0; stride >>= 1) {
      if (lid < stride && argmax_better(sv[lid + stride], si[lid + stride], sv[lid], si[lid])) {
        sv[lid] = sv[lid + stride];
        si[lid] = si[lid + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) vout[m] = si[0];
  }
}
