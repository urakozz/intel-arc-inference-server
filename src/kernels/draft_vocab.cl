// draft_vocab.cl - spec 8 §11: the MTP draft's argmax over a reduced vocabulary V',
// |V'| = NV (32768 / 65536 / 131072), after `gemv_i8w` at N = NV over the compact head
// (loader::DraftVocab: the int8 head's rows ids[0..NV), gathered ascending by id).
//
//   dv_argmax_stage1  grid (NV/1024), WG 256 - group g reduces compact logits
//                     [g*1024, +1024) to one (value, compact index) pair in part[g],
//                     and scatters each compact logit into the draft's full-vocabulary
//                     row at its id: full_row[ids[k]] = logits_c[k]
//   dv_argmax_stage2  grid (1),        WG 256 - folds the NV/1024 pairs, maps the winning
//                     compact index through `ids`, and writes the Control fields the
//                     full head's argmax_stage2 writes for the draft list's one row:
//                     out_token[0], cur_token[0], pos += 1
//
// **argmax.cl's two fixed stages and its comparator, unchanged in meaning.** Strictly
// greater wins and an exact tie goes to the lower index; every comparison is between
// fixed SLM slots in a fixed order, so the result is a pure function of the logits and a
// replayed list gives the same bits (argmax.cl argues both at length). Because `ids` is
// ascending, the lower compact index IS the lower id: the winner is exactly the full
// head's argmax restricted to V', ties included - the property draft_vocab_kernels_test
// checks on the card. No VOCAB_USED mask: loader::select_draft_vocab never puts a masked
// id in V'.
//
// **Why the scatter rides in stage 1 and is not a third kernel.** The host's sampled
// acceptance reads the draft's logits row q_i over the full vocabulary (spec 8 plan 8c,
// src/server/spec_accept.h). With V', q_i must be -inf outside V' (q = 0 there) and the
// compact logits at V''s ids. The -inf entries are written once (MtpBuffers::zero, at
// construction and every reset - nothing else writes those entries); the V' entries change
// every draft and are written here, by the lanes that already hold them in registers: one
// extra store per logit and no extra launch, so the draft list keeps its 23 launches with
// or without V'. A greedy draft never reads the row; the stores cost NV * 4 B (at most
// 512 KiB, against the 671 MB the compact GEMV reads at 128k).
//
// The compact index travels to stage 2 as a float, exactly as argmax.cl's index does:
// every integer below 2^24 is exact in fp32 and NV - 1 <= 131071.

#ifndef NV
#error "NV (the draft vocabulary's size) must be defined"
#endif
#define WG_ARGMAX 256
#define CHUNK 1024                        /* logits per stage-1 group, argmax.cl's */
#define GROUPS (NV / CHUNK)               /* 32 / 64 / 128 */
#define NO_IDX 0x7FFFFFFFu                /* neutral seed: loses every tie */
#if (NV % CHUNK) != 0
#error "NV must be a multiple of the 1024-logit chunk: the stage-1 grid is NV / 1024"
#endif
#if GROUPS > WG_ARGMAX
#error "stage 2 folds one pair per lane: NV / 1024 must not exceed 256"
#endif

inline int argmax_better(float av, uint ai, float bv, uint bi) {
  return (av > bv) || (av == bv && ai < bi);
}

__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void dv_argmax_stage1(__global const float* restrict logits_c,
                               __global const uint* restrict ids,
                               __global float* restrict part,
                               __global float* restrict full_row) {
  const uint g = get_group_id(0);
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];

  const uint k0 = g * CHUNK, k1 = k0 + CHUNK;
  float bv = -INFINITY;
  uint bi = NO_IDX;
  for (uint k = k0 + lid; k < k1; k += WG_ARGMAX) {
    const float v = logits_c[k];
    full_row[ids[k]] = v;
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
    part[(size_t)g * 2 + 0] = sv[0];
    part[(size_t)g * 2 + 1] = (float)si[0];   // exact: idx < 2^24
  }
}

// The draft list is M = 1 and runs with hctl.n_active = 1 (runtime::Engine::draft), so
// this writes one row's fields - what argmax_stage2 writes at n = 1: out_token[0], then
// cur_token[0] = out_token[0] and pos += 1, the chain to the next draft list. Any other
// n_active would be an engine bug; like argmax_stage2 at n = 0, it then writes nothing
// rather than something plausible. A winner of NO_IDX (every logit NaN) has no id to map:
// it is written as NO_IDX, which the host refuses (Engine::verify's vocabulary check), as
// the full head's argmax would have emitted the same value.
__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void dv_argmax_stage2(__global uint* restrict ctrl,
                               __global const float* restrict part,
                               __global const uint* restrict ids) {
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];

  float bv = -INFINITY;
  uint bi = NO_IDX;
  if (lid < GROUPS) {
    bv = part[(size_t)lid * 2 + 0];
    bi = (uint)part[(size_t)lid * 2 + 1];   // exact: idx < 2^24
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

  if (lid == 0 && ctrl[CTRL_NACT] == 1u) {
    const uint i = si[0];
    const uint id = i < (uint)NV ? ids[i] : NO_IDX;
    ctrl[CTRL_OUT] = id;
    ctrl[CTRL_CUR] = id;
    ctrl[CTRL_POS] += 1u;
  }
}
