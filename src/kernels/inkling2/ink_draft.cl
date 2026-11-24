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
// before applying it. No padded row participates.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_draft_argmax(__global const float* base, __global const float* fold,
                               __global const float* bias, __global uint* previous) {
  const uint i = get_local_id(0);
  float best = -INFINITY;
  uint id = LIVE_ROWS;
  for (uint n = i; n < LIVE_ROWS; n += WG) {
    const float score = base[n] + fold[n] + bias[n];
    if (score > best) { best = score; id = n; }
  }
  __local float val[WG];
  __local uint ids[WG];
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
  if (i == 0) *previous = ids[0];
}

__kernel void ink_draft_silu(__global const float* gate, __global const float* up, __global ushort* out) {
  const uint i = get_global_id(0);
  out[i] = rne_bf16((gate[i] / (1.f + exp(-gate[i]))) * up[i]);
}
