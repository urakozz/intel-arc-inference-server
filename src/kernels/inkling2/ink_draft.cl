// DSpark's checkpoint-specific block inputs, four-tap context mix and
// recurrent Markov step. Dense projections use the shared GEMV kernels.
#include "bf16.h"
#define H 2304
#define RANK 256
#define WG 256
#define LIVE_ROWS 200058
#pragma OPENCL FP_CONTRACT OFF

// Row zero uses the target's embedding norm on the anchor. Mask rows use
// the dedicated learned mask vector in mtp.safetensors.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_embed(__global const uint* ctrl, __global const ushort* embedding,
                              __global const float* embed_norm, __global const ushort* mask,
                              __global float* hidden, float eps) {
  const uint row = get_group_id(0), lane = get_local_id(0);
  if (row != 0) {
    for (uint i = lane; i < H; i += WG) hidden[(size_t)row * H + i] = bf16f(mask[i]);
    return;
  }
  const __global ushort* src = embedding + (size_t)ctrl[CTRL_CUR] * H;
  __local float sums[WG];
  float ss = 0.f;
  for (uint i = lane; i < H; i += WG) { float x = bf16f(src[i]); ss += x * x; }
  sums[lane] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) sums[lane] += sums[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(sums[0] / H + eps);
  for (uint i = lane; i < H; i += WG) hidden[i] = (bf16f(src[i]) * inv) * embed_norm[i];
}

inline float tap_mix(__global const ushort* taps, const __local float* a, uint i) {
  float x = 0.f;
  for (uint t = 0; t < 4; ++t) x += bf16f(taps[t * H + i]) * a[t];
  return x;
}

// Each layer uses a different row of fuse_weight and a different norm.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_mix(__global const ushort* tap_rows, __global const float* logits,
                            __global const float* norm, __global ushort* out, float eps) {
  const uint row = get_group_id(0), lane = get_local_id(0);
  const __global ushort* taps = tap_rows + (size_t)row * 4 * H;
  __local float a[4], sum[WG];
  if (lane == 0) {
    const float mx = fmax(fmax(logits[0], logits[1]), fmax(logits[2], logits[3]));
    float z = 0.f;
    for (uint t = 0; t < 4; ++t) { a[t] = exp(logits[t] - mx); z += a[t]; }
    for (uint t = 0; t < 4; ++t) a[t] /= z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float ss = 0.f;
  for (uint i = lane; i < H; i += WG) { const float x = tap_mix(taps, a, i); ss += x * x; }
  sum[lane] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) sum[lane] += sum[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(sum[0] / H + eps);
  for (uint i = lane; i < H; i += WG)
    out[(size_t)row * H + i] = rne_bf16((tap_mix(taps, a, i) * inv) * norm[i]);
}

// The caller forms the dense row [state; w1[previous]; h; context] before
// the shared linear, then applies this recurrence and scores the latent.
__kernel void ink_markov_input(__global const ushort* state, __global const ushort* w1,
                               __global const ushort* hidden, __global const ushort* context,
                               __global const uint* previous, __global ushort* input) {
  const uint i = get_global_id(0);
  ushort x;
  if (i < RANK) x = state[i];
  else if (i < 2 * RANK) x = w1[(size_t)*previous * RANK + i - RANK];
  else if (i < 2 * RANK + H) x = hidden[i - 2 * RANK];
  else x = context[i - 2 * RANK - H];
  input[i] = x;
}

inline float draft_bf16(float x) { return bf16f(rne_bf16(x)); }

__kernel void ink_markov_step(__global const float* raw, __global const float* bias,
                              __global ushort* state, __global ushort* latent) {
  const uint i = get_global_id(0);
  const float g = draft_bf16(1.f / (1.f + exp(-draft_bf16(raw[i] + bias[i]))));
  const float c = draft_bf16(tanh(draft_bf16(raw[RANK + i] + bias[RANK + i])));
  const float keep = draft_bf16(g * bf16f(state[i]));
  const float update = draft_bf16(draft_bf16(1.f - g) * c);
  state[i] = rne_bf16(keep + update);
  latent[i] = rne_bf16(tanh(draft_bf16(raw[2 * RANK + i] + bias[2 * RANK + i])));
}

// Softcap is monotonic, so the greedy pick can compare the joint logits
// before applying it. No padded row participates. The highest score with the lowest id among equals - whatever
// the order the candidates meet in, so the two-stage form (a work-group a chunk of ARGMAX_CHUNK ids into `parts`,
// then one work-group over the parts) picks what one work-group over every id would.
#define ARGMAX_CHUNK 1024
inline void ink_argmax_tree(float best, uint id, uint i, __local float* val, __local uint* ids) {
  val[i] = best;
  ids[i] = id;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (i < s && (val[i + s] > val[i] || (val[i + s] == val[i] && ids[i + s] < ids[i]))) {
      val[i] = val[i + s];
      ids[i] = ids[i + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_argmax1(__global const float* base, __global const float* fold,
                                __global const float* bias, __global float* parts) {
  const uint i = get_local_id(0), chunk = get_group_id(0);
  float best = -INFINITY;
  uint id = LIVE_ROWS;
  for (uint n = chunk * ARGMAX_CHUNK + i; n < min((chunk + 1) * ARGMAX_CHUNK, (uint)LIVE_ROWS); n += WG) {
    const float score = base[n] + fold[n] + bias[n];
    if (score > best) { best = score; id = n; }
  }
  __local float val[WG];
  __local uint ids[WG];
  ink_argmax_tree(best, id, i, val, ids);
  if (i == 0) {
    parts[2 * chunk] = val[0];
    parts[2 * chunk + 1] = as_float(ids[0]);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_argmax2(__global const float* parts, __global uint* previous) {
  const uint i = get_local_id(0), chunks = (LIVE_ROWS + ARGMAX_CHUNK - 1) / ARGMAX_CHUNK;
  float best = -INFINITY;
  uint id = LIVE_ROWS;
  for (uint c = i; c < chunks; c += WG) {
    const float v = parts[2 * c];
    const uint n = as_uint(parts[2 * c + 1]);
    if (v > best || (v == best && n < id)) { best = v; id = n; }
  }
  __local float val[WG];
  __local uint ids[WG];
  ink_argmax_tree(best, id, i, val, ids);
  if (i == 0) *previous = ids[0];
}

__kernel void ink_draft_silu(__global const float* gate, __global const float* up, __global ushort* out) {
  const uint i = get_global_id(0);
  out[i] = rne_bf16((gate[i] / (1.f + exp(-gate[i]))) * up[i]);
}

#ifdef TP_RANK
// `--parallel tp` with the head's rows split (TP_RANK's TP_HALF rows from TP_RANK x TP_HALF, the live ones): each draft's
// greedy pick over the card's rows - ink_draft_argmax1's chunks at the card's row indices, base and fold the card's
// [TP_HALF] rows, bias at its first - then its best into an all-reduce row (zeros but the card's value and id, both
// exact in fp32: x + 0 = x), and after the all-reduce both cards pick the same: the higher score, card 0's (lower ids)
// on a tie, as one card's two-stage pick would.
#define TP_FIRST (TP_RANK * TP_HALF)
#define TP_ROWS (TP_RANK ? LIVE_ROWS - TP_HALF : TP_HALF)
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_argmax1_tp(__global const float* base, __global const float* fold,
                                   __global const float* bias, __global float* parts) {
  const uint i = get_local_id(0), chunk = get_group_id(0);
  float best = -INFINITY;
  uint id = LIVE_ROWS;
  for (uint n = chunk * ARGMAX_CHUNK + i; n < min((chunk + 1) * ARGMAX_CHUNK, (uint)TP_ROWS); n += WG) {
    const float score = base[n] + fold[n] + bias[n];
    if (score > best) { best = score; id = TP_FIRST + n; }
  }
  __local float val[WG];
  __local uint ids[WG];
  ink_argmax_tree(best, id, i, val, ids);
  if (i == 0) {
    parts[2 * chunk] = val[0];
    parts[2 * chunk + 1] = as_float(ids[0]);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_argmax2_tp(__global const float* parts, __global float* row) {
  const uint i = get_local_id(0), chunks = (TP_ROWS + ARGMAX_CHUNK - 1) / ARGMAX_CHUNK;
  float best = -INFINITY;
  uint id = LIVE_ROWS;
  for (uint c = i; c < chunks; c += WG) {
    const float v = parts[2 * c];
    const uint n = as_uint(parts[2 * c + 1]);
    if (v > best || (v == best && n < id)) { best = v; id = n; }
  }
  __local float val[WG];
  __local uint ids[WG];
  ink_argmax_tree(best, id, i, val, ids);
  for (uint x = i; x < H; x += WG) row[x] = 0.f;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (i == 0) {
    row[2 * TP_RANK] = val[0];
    row[2 * TP_RANK + 1] = (float)ids[0];
  }
}

__kernel void ink_draft_pick(__global const float* sum, __global uint* previous) {
  *previous = sum[2] > sum[0] ? (uint)sum[3] : (uint)sum[1];
}

// The DFlash 2 drafter's candidates with the head's rows split the same way: the card's top 16 a draft row (df_topk
// over its rows: local ids, raw values, best first) into an all-reduce row - [rank][row][16] (value, the id at the
// card's first row on, exact in fp32), zeros elsewhere - and after the all-reduce both cards merge the two lists by
// df_topk's order (the larger value, the lower id on a tie): the whole head's top 16, then out_mult as
// df_topk_stage2 applies it (softcap 0: the selector caps).
#define DF_ROWS 7
#define DF_TOPK 16
__kernel void ink_df_topk_tp(__global const uint* cand, __global const float* unary, __global float* row) {
  const uint x = get_global_id(0), mine = TP_RANK * DF_ROWS * DF_TOPK * 2;
  float v = 0.f;
  if (x >= mine && x < mine + DF_ROWS * DF_TOPK * 2) {
    const uint e = (x - mine) / 2;
    v = (x & 1) ? (float)(cand[e] + TP_FIRST) : unary[e];
  }
  row[x] = v;
}

__kernel void ink_df_topk_merge(__global const float* sum, const float out_mult, __global uint* cand,
                                __global float* unary) {
  const uint j = get_global_id(0);
  if (j >= DF_ROWS) return;
  __global const float* a = sum + j * DF_TOPK * 2;
  __global const float* b = sum + (DF_ROWS + j) * DF_TOPK * 2;
  uint ia = 0, ib = 0;
  for (uint k = 0; k < DF_TOPK; ++k) {   // ia + ib = k < 16: neither list runs out
    const float av = a[2 * ia], bv = b[2 * ib];
    const uint ai = (uint)a[2 * ia + 1], bi = (uint)b[2 * ib + 1];
    const int take_a = av > bv || (av == bv && ai < bi);
    cand[j * DF_TOPK + k] = take_a ? ai : bi;
    unary[j * DF_TOPK + k] = (take_a ? av : bv) * out_mult;
    if (take_a) ++ia;
    else ++ib;
  }
}
#endif

// A drafted round as one submission (draft, verify, commit; Engine::step_drafted): the accepted prefix - the anchor
// and each draft the target predicted before the first miss - as the commit's projection rows (proj_n; the host wrote
// proj_pos, the round's position: the target's argmax has advanced ctrl's by then). The target's predictions stay in
// out_token for the host.
__kernel void ink_accept(__global const uint* ctrl, __global uint* dctl) {
  const uint rows = ctrl[CTRL_NACT];
  uint accepted = 1;
  while (accepted < rows && ctrl[CTRL_CUR + accepted] == ctrl[CTRL_OUT + accepted - 1]) ++accepted;
  dctl[DCTL_PN] = accepted;
}

// The last accepted row's four taps into `kept`, the row the next draft's context mix reads.
__kernel void ink_tap_keep(__global const uint* dctl, __global const ushort* taps, __global ushort* kept) {
  const uint i = get_global_id(0);
  kept[i] = taps[(size_t)(dctl[DCTL_PN] - 1) * 4 * H + i];
}
