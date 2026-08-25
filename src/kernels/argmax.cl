// argmax.cl - greedy sampling: the arg-max of one row of `lm_head` logits, in
// two fixed stages, plus the control block's per-token bookkeeping.
//
//   stage 1  grid (243, M), WG 256 - group g reduces logits[m][g·1024 …+1024)
//            to one (value, index) pair in `part[m][g]`
//   stage 2  grid (1, 1),   WG 256 - one work-group folds the 243 pairs of each
//            active token, writes ctrl.out_token[m], then sets
//            ctrl.cur_token[0] and advances ctrl.pos
//
// **Why two fixed stages and not atomics.** The list is captured once and
// replayed per token, and the acceptance for every kernel in this plan is that
// two replays of the same inputs produce the same bits. A global `atomic_max`
// (or a float atomic compare-exchange loop) over 248320 candidates is
// order-dependent by construction: whichever work-group happens to arrive first
// decides which of two equal logits is seen as "already the max", and equal
// logits are not exotic at the top of a peaked softmax - the tie between two
// spellings of the same word is exactly where greedy decoding is fragile. A
// fixed two-stage tree has no data-dependent order at all: every comparison is
// between a fixed pair of SLM slots, in a fixed sequence, so the answer is a
// pure function of the logits. It is also *cheaper*: 243 work-groups doing 4
// KB each, then one work-group doing 2 KB.
//
// **The comparator** - one function, used by both stages and by the host
// reference - is `(a.v > b.v) || (a.v == b.v && a.i < b.i)`: strictly greater
// wins, and an exact tie goes to the LOWER index. Ties therefore resolve the
// same way no matter how the reduction is bracketed, which is what makes the
// tree order irrelevant to the *result* while keeping it deterministic. (Torch's
// `argmax` has the same tie rule, so the golden gate compares like with like.)
//
// **The 248077 mask.** `lm_head` is [5120][248320] because the tiling wants a
// round row count, but the tokenizer only defines 248077 ids
// (model::Qwen35::kVocabUsed - the `vocab_size` of the checkpoint's config,
// docs/03-models.md). Rows 248077..248319 are whatever the checkpoint happened
// to store there, and their logits are ordinary finite numbers that can perfectly
// well be the largest in the row. Emitting one would be an id the tokenizer
// cannot decode, so a candidate at index >= VOCAB_USED contributes
// `(-INFINITY, idx)` and can never win against a real token. Indices >= VOCAB
// are not read at all - `logits` is only VOCAB wide.

#ifndef M
#define M 1
#endif
#if M > 8
// out_token[] holds 8 ids (runtime::Control); a larger M has nowhere to write.
#error "argmax: M > 8 exceeds Control::out_token[]"
#endif
#ifndef VOCAB
#define VOCAB 248320       /* model::Qwen35::kVocab - the logits row length */
#endif
#ifndef VOCAB_USED
#define VOCAB_USED 248077  /* model::Qwen35::kVocabUsed - real tokenizer ids */
#endif

#define WG_ARGMAX 256
#define CHUNK 1024                                   /* logits per stage-1 group */
#define GROUPS ((VOCAB + CHUNK - 1) / CHUNK)         /* 243 */
#define NO_IDX 0x7FFFFFFFu                           /* neutral seed: loses every tie */

// The comparator. Written once so the two stages cannot drift apart: "is
// (av, ai) a better candidate than (bv, bi)?"
inline int argmax_better(float av, uint ai, float bv, uint bi) {
  return (av > bv) || (av == bv && ai < bi);
}

// ---------------------------------------------------------------------------
// argmax_stage1 - one work-group per 1024 logits.
//
// Each lane scans its strided slice of the chunk into a register pair, then a
// fixed pairwise SLM tree (256 -> 128 -> … -> 1, barrier after every step, the
// same shape as prep.cl's variance tree) collapses the 256 lanes.
//
// The index travels to stage 2 **as a float**, because that keeps `part` a
// plain fp32 [M][243][2] buffer (runtime::DecodeBuffers::argmax_part) instead
// of a struct the host would have to lay out by hand. That is lossless here and
// not merely "close enough": every integer below 2^24 = 16777216 is exactly
// representable in fp32, and the largest index this kernel can emit is
// VOCAB-1 = 248319 - 67x under the limit. `(uint)(float)idx` is the identity
// for every id in this vocabulary.
__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void argmax_stage1(__global const float* restrict logits,
                            __global float* restrict part) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];

  const uint k0 = g * CHUNK;
  uint k1 = k0 + CHUNK;
  if (k1 > VOCAB) k1 = VOCAB;   // the last group covers 247808..248320

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

// ---------------------------------------------------------------------------
// argmax_stage2 - one work-group of 256 for the whole step, and the ONLY writer
// of `pos` / `cur_token` inside the captured list.
//
// GROUPS = 243 < 256, so lanes 243..255 seed `(-INFINITY, NO_IDX)`: -INF loses
// to every real logit, and NO_IDX loses every tie against a real index, so the
// idle lanes cannot affect the result. The same tree as stage 1 follows.
//
// The m loop is bounded by `ctrl[CTRL_NACT]`, which every lane reads, so the
// bound is uniform and the barriers inside are reached by all 256 lanes. The
// barrier at the top of the body is the one that makes the SLM reuse safe: the
// previous iteration's lanes must be done reading `sv`/`si` before this one
// overwrites them.
__attribute__((reqd_work_group_size(WG_ARGMAX, 1, 1)))
__kernel void argmax_stage2(__global uint* restrict ctrl,
                            __global const float* restrict part) {
  const uint lid = get_local_id(0);
  __local float sv[WG_ARGMAX];
  __local uint si[WG_ARGMAX];
  const uint n = ctrl[CTRL_NACT];

  for (uint m = 0; m < n; ++m) {
    float bv = -INFINITY;
    uint bi = NO_IDX;
    if (lid < GROUPS) {
      bv = part[((size_t)m * GROUPS + lid) * 2 + 0];
      bi = (uint)part[((size_t)m * GROUPS + lid) * 2 + 1];   // exact: idx < 2^24
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
    if (lid == 0) ctrl[CTRL_OUT + m] = si[0];
  }

  // The token boundary, in one lane: the last token sampled this step is what
  // the next step embeds (embed_gather reads cur_token[0] at execution time),
  // and the KV cursor moves by however many positions this step consumed.
  // Only lane 0 wrote out_token above, so it needs no fence to read it back.
  // n == 0 would be an empty step - an engine bug - and out_token[n-1] would
  // index off the front of the field, so do nothing rather than something
  // plausible.
  if (lid == 0 && n != 0u) {
    ctrl[CTRL_CUR] = ctrl[CTRL_OUT + n - 1u];
    ctrl[CTRL_POS] += n;
  }
}
