// Inkling's FP32 residual, depth-wise mix and loop-controller glue.
#include "bf16.h"

#ifndef H
#define H 2304
#endif
#define WG 128
#define BANK 5
#pragma OPENCL FP_CONTRACT OFF

// Finished states stay in the bank. An optional final state is the working
// residual, so the caller can mix it in place without copying it to the bank.
inline const __global float* ink_depth_row(__global const float* states, __global const float* last,
                                           uint s, uint count, uint rows, uint row) {
  return last && s + 1 == count ? last + (size_t)row * H : states + ((size_t)s * rows + row) * H;
}

inline void ink_depth_scales(__global const float* states, __global const float* last,
                             __global const float* query, uint count, uint rows,
                             uint r, uint q, uint lane, float eps, __local float* dots, __local float* squares,
                             __local float* scale, __local float* score) {
  for (uint s = 0; s < count; ++s) {
    const __global float* h = ink_depth_row(states, last, s, count, rows, r);
    float sum = 0.0f, dot = 0.0f;
    for (uint i = lane; i < H; i += WG) {
      sum += h[i] * h[i];
      dot += query[(size_t)q * H + i] * h[i];
    }
    squares[lane] = sum;
    dots[lane] = dot;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG / 2; stride; stride >>= 1) {
      if (lane < stride) {
        squares[lane] += squares[lane + stride];
        dots[lane] += dots[lane + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      scale[s] = 1.0f / sqrt(squares[0] / H + eps);
      score[s] = dots[0] * scale[s] / sqrt((float)H);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0) {
    float high = score[0], total = 0.0f;
    for (uint s = 1; s < count; ++s) high = fmax(high, score[s]);
    for (uint s = 0; s < count; ++s) {
      score[s] = exp(score[s] - high);
      total += score[s];
    }
    for (uint s = 0; s < count; ++s) scale[s] *= score[s] / total;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

inline float ink_depth_value(__global const float* states, __global const float* last, const __local float* scale,
                             uint count, uint rows, uint row, uint i) {
  float sum = 0.0f;
  for (uint s = 0; s < count; ++s) sum += scale[s] * ink_depth_row(states, last, s, count, rows, row)[i];
  return sum;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_depth_mix(__global const float* states, __global const float* query,
                            __global float* output, uint count, uint rows, float eps, __global const float* last) {
  const uint r = get_group_id(0), q = get_group_id(1), lane = get_local_id(0);
  __local float dots[WG], squares[WG], scale[BANK], score[BANK];
  ink_depth_scales(states, last, query, count, rows, r, q, lane, eps, dots, squares, scale, score);
  for (uint i = lane; i < H; i += WG)
    output[((size_t)q * rows + r) * H + i] = ink_depth_value(states, last, scale, count, rows, r, i);
}

// Fuse the mixed residual with the projection's weighted RMS and BF16 pack.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_depth_mix_norm(__global const float* states, __global const float* query,
                                 __global ushort* output, uint count, uint rows, float eps,
                                 __global const float* gamma0, __global const float* gamma1,
                                 __global const float* gamma2, __global const float* last) {
  const uint r = get_group_id(0), q = get_group_id(1), lane = get_local_id(0);
  __local float dots[WG], squares[WG], scale[BANK], score[BANK], mixed[H];
  ink_depth_scales(states, last, query, count, rows, r, q, lane, 1e-6f, dots, squares, scale, score);
  float sum = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const float value = ink_depth_value(states, last, scale, count, rows, r, i);
    mixed[i] = value;
    sum += value * value;
  }
  squares[lane] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) squares[lane] += squares[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(squares[0] / H + eps);
  const __global float* gamma = q == 0 ? gamma0 : q == 1 ? gamma1 : gamma2;
  for (uint i = lane; i < H; i += WG)
    output[((size_t)q * rows + r) * H + i] = rne_bf16((mixed[i] * inv) * gamma[i]);
}

// One work-group per row. A null message produces an ordinary weighted
// norm. With a message, each arm gets unweighted RMS into one controller A row.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_norm(__global const float* input, __global const float* message,
                       __global const float* gamma, __global ushort* out, float eps) {
  const uint row = get_group_id(0), lane = get_local_id(0);
  const uint arms = message ? 2 : 1;
  __local float sums[WG];
  for (uint a = 0; a < arms; ++a) {
    const __global float* x = (a ? message : input) + (size_t)row * H;
    float local_sum = 0.f;
    for (uint i = lane; i < H; i += WG) local_sum += x[i] * x[i];
    sums[lane] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = WG / 2; s; s >>= 1) {
      if (lane < s) sums[lane] += sums[lane + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.f / sqrt(sums[0] / H + eps);
    for (uint i = lane; i < H; i += WG)
      out[((size_t)row * arms + a) * H + i] = rne_bf16((x[i] * inv) * (gamma ? gamma[i] : 1.f));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// The controller's down, gain and forget projections use the shared bf16 GEMV.
__kernel void ink_controller_blend(__global const float* input, __global const float* message,
                                   __global const float* forget, __global const float* gain,
                                   __global float* out) {
  const size_t i = get_global_id(0);
  out[i] = (1.f - tanh(forget[i])) * input[i] + tanh(gain[i]) * message[i];
}

// A static post-sublayer convolution over current rows and three committed
// rows. Weights are chronological [channel][4], the newest tap last.
__kernel void ink_post_conv(__global const float* input, __global const float* history,
                            __global const float* taps, __global const float* residual,
                            __global float* out, __global ushort* draft_tap) {
  const uint i = get_global_id(0), r = get_global_id(1);
  float sum = 0.f;
  for (uint tap = 0; tap < 4; ++tap) {
    const int t = (int)r + (int)tap - 3;
    const float x = t < 0 ? history[(t + 3) * H + i] : input[t * H + i];
    sum += taps[i * 4 + tap] * x;
  }
  const size_t at = (size_t)r * H + i;
  const float value = (input[at] + sum) / sqrt(5.f) + (residual ? residual[at] : 0.f);
  out[at] = value;
  if (draft_tap) draft_tap[(size_t)r * 4 * H + i] = rne_bf16(value);
}

// One launch shifts all four three-row histories at every visit. Each
// work-item owns a channel, so the accepted prefix can be applied in place.
__kernel void ink_commit_conv(__global const float* pending, __global float* k, __global float* v,
                              __global float* attn, __global float* mlp, uint rows, uint committed) {
  const uint visit = get_group_id(1), i = get_global_id(0);
  const uint kv = 1024;
  const uint width = i < 2 * kv ? kv : H;
  const uint offset = i < kv ? 0 : i < 2 * kv ? kv : i < 2 * kv + H ? 2 * kv : 2 * kv + H;
  const uint d = i - offset;
  __global float* history = i < kv ? k : i < 2 * kv ? v : i < 2 * kv + H ? attn : mlp;
  history += (size_t)visit * 3 * width;
  pending += (size_t)rows * (visit * 2 * (kv + H) + offset);
  float next[3];
  for (uint t = 0; t < 3; ++t) {
    const uint row = t + committed;
    next[t] = row < 3 ? history[(size_t)row * width + d] : pending[(size_t)(row - 3) * width + d];
  }
  for (uint t = 0; t < 3; ++t) history[(size_t)t * width + d] = next[t];
}

// Token embedding followed by the release's weighted RMS norm. The residual
// walk retains the result in FP32 and each projection packs to BF16.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_embed_norm(__global const uint* ctrl, __global const ushort* table,
                              __global const float* gamma, __global float* x, float eps) {
  const uint row = get_group_id(0), lane = get_local_id(0);
  const __global ushort* e = table + (size_t)ctrl[CTRL_CUR + row] * H;
  __local float squares[WG];
  float ss = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const float v = bf16f(e[i]);
    ss += v * v;
  }
  squares[lane] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) squares[lane] += squares[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(squares[0] / H + eps);
  for (uint i = lane; i < H; i += WG)
    x[(size_t)row * H + i] = (bf16f(e[i]) * inv) * gamma[i];
}

__kernel void ink_pack(__global const float* x, __global ushort* a) {
  const uint i = get_global_id(0);
  a[i] = rne_bf16(x[i]);
}

__kernel void ink_tap(__global const float* x, __global const ushort* a, __global ushort* tap, uint slot) {
  const uint i = get_global_id(0), row = get_global_id(1);
  const size_t at = (size_t)row * H + i;
  tap[((size_t)row * 4 + slot) * H + i] = x ? rne_bf16(x[at]) : a[at];
}

#if INK_CONTROLLER_FUSION
inline float ink_rms_inverse(float sum, uint lane, float eps, __local float* squares) {
  squares[lane] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) squares[lane] += squares[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(squares[0] / H + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  return inv;
}

// Each work-group produces its own row and reads only columns of that row
// in the following depth mix. Other rows are read from the raw projection.
inline void ink_attention_residual(__global float* out, __global const float* raw,
                                   __global const float* history, __global const float* taps,
                                   __global const float* residual, uint row, uint lane) {
  for (uint i = lane; i < H; i += WG) {
    float sum = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
      const int t = (int)row + (int)tap - 3;
      const float x = t < 0 ? history[(t + 3) * H + i] : raw[t * H + i];
      sum += taps[i * 4 + tap] * x;
    }
    const size_t at = (size_t)row * H + i;
    out[at] = (raw[at] + sum) / sqrt(5.f) + (residual ? residual[at] : 0.f);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_post_mix_norm(__global const float* bank, __global float* last,
                                 __global const float* query, __global ushort* out,
                                 __global const float* gamma, uint states, uint rows, float eps,
                                 __global const float* raw, __global const float* history,
                                 __global const float* taps, __global const float* residual) {
  const uint r = get_group_id(0), lane = get_local_id(0);
  __local float dots[WG], squares[WG], scale[BANK], score[BANK], temp[H];
  ink_attention_residual(last, raw, history, taps, residual, r, lane);
  ink_depth_scales(bank, last, query, states, rows, r, 0, lane, 1e-6f, dots, squares, scale, score);
  float ss = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const float x = ink_depth_value(bank, last, scale, states, rows, r, i);
    temp[i] = x;
    ss += x * x;
  }
  const float inv = ink_rms_inverse(ss, lane, eps, squares);
  for (uint i = lane; i < H; i += WG)
    out[(size_t)r * H + i] = rne_bf16((temp[i] * inv) * gamma[i]);
}

// All four message streams for the next pass share its saved midpoint and output.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_cross_mix(__global const float* states, __global const float* q,
                            __global const float* kv, __global float* out, uint rows) {
  const uint r = get_group_id(0), i = get_group_id(1), lane = get_local_id(0);
  const uint stream = i % 4, layer = i / 4;
  const __global float* query = stream == 0 || stream == 3 ? q : kv;
  const uint which = 2 * layer + (stream >= 2);
  __local float dots[WG], squares[WG], scale[BANK], score[BANK];
  ink_depth_scales(states, 0, query, 2, rows, r, which, lane, 1e-6f, dots, squares, scale, score);
  for (uint c = lane; c < H; c += WG)
    out[((size_t)i * rows + r) * H + c] = ink_depth_value(states, 0, scale, 2, rows, r, c);
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_mix_cat(__global const float* bank, __global float* last,
                          __global const float* query, __global const float* messages,
                          __global float* mixed, __global ushort* cat,
                          uint states, uint rows, uint shared_mix, float eps,
                          __global const float* raw, __global const float* history,
                          __global const float* taps, __global const float* residual) {
  const uint r = get_group_id(0), stream = get_group_id(1), lane = get_local_id(0);
  const size_t start = ((size_t)stream * rows + r) * H;
  __local float dots[WG], squares[WG], scale[BANK], score[BANK], temp[H];
  if (raw) ink_attention_residual(last, raw, history, taps, residual, r, lane);
  ink_depth_scales(bank, last, query, states, rows, r, shared_mix ? 0 : stream, lane, 1e-6f,
                   dots, squares, scale, score);
  float sx = 0.f, sm = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const float x = ink_depth_value(bank, last, scale, states, rows, r, i);
    const float msg = messages[start + i];
    temp[i] = x;
    mixed[start + i] = x;
    sx += x * x;
    sm += msg * msg;
  }
  const float rx = ink_rms_inverse(sx, lane, eps, squares);
  const float rm = ink_rms_inverse(sm, lane, eps, squares);
  for (uint i = lane; i < H; i += WG) {
    cat[2 * start + i] = rne_bf16(temp[i] * rx);
    cat[2 * start + H + i] = rne_bf16(messages[start + i] * rm);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_controller_norm(__global const float* mixed, __global const float* messages,
                                  __global const float* fg, __global ushort* out,
                                  __global const float* q_norm, __global const float* k_norm,
                                  __global const float* v_norm, uint rows, float eps) {
  const uint r = get_group_id(0), stream = get_group_id(1), lane = get_local_id(0);
  const size_t start = ((size_t)stream * rows + r) * H;
  const size_t gain = (size_t)get_num_groups(1) * rows * H;
  const __global float* gamma = stream == 0 ? q_norm : stream == 1 ? k_norm : v_norm;
  __local float temp[H], squares[WG];
  float sum = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const size_t j = start + i;
    const float x = (1.f - tanh(fg[j])) * mixed[j] + tanh(fg[gain + j]) * messages[j];
    temp[i] = x;
    sum += x * x;
  }
  const float inv = ink_rms_inverse(sum, lane, eps, squares);
  for (uint i = lane; i < H; i += WG)
    out[start + i] = rne_bf16((temp[i] * inv) * gamma[i]);
}
#endif
