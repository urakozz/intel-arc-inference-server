// Inkling's FP32 residual, depth-wise mix and loop-controller glue.
#include "bf16.h"

#ifndef H
#define H 2304
#endif
#define WG 128
#define BANK 5
// INK_TP: a `--parallel tp` card's build - its 512 k / v channels in the convolution histories (INK_KV_N, every kv
// head's 1024 otherwise) and the prompt pass's all-reduce sum (ink_tp_sum).
#if INK_TP
#define INK_KV_N 512
#else
#define INK_KV_N 1024
#endif
#pragma OPENCL FP_CONTRACT OFF
// The (row, stream) work-groups of the multi-stream mixes. A prompt chunk numbers the streams of one row next to
// each other, so that row's depth states come from memory once and from the cache for its other streams.
// The prompt pass's row kernels take the launch's first row (`row0`; the rows' stride stays the chunk's): under
// `--parallel tp` each card runs the glue of its own rows of the chunk (card 1's from its first), one card from 0.
#if INK_PREFILL
#define PF_ROW0_ARG , uint row0
#define PF_ROW0 row0
#define MIX_ROW (get_group_id(1) + row0)
#define MIX_STREAM get_group_id(0)
#else
#define PF_ROW0_ARG
#define PF_ROW0 0u
#define MIX_ROW get_group_id(0)
#define MIX_STREAM get_group_id(1)
#endif
// INK_GLUE_BF16 (prompt pass, lossy): the rows kept between glue kernels - the cross-pass messages and the
// controllers' mixed rows - are stored as bf16 (one rounding at the store); every sum stays fp32. Off, the
// rows are fp32 and the text below is the exact path.
// Decode runs one work-group a row, so each lane's strided loop over the row is unrolled (H / WG steps) and its loads
// are in flight together: the same sums in the same order. The prompt pass has the rows to hide the latency.
#if H % WG != 0
#error "ink_glue: a row is whole work-group steps"
#endif
#if INK_PREFILL
#define ROW_LOOP(i) for (uint i = lane; i < H; i += WG)
#else
#define ROW_LOOP(i) \
  __attribute__((opencl_unroll_hint)) for (uint i##_step = 0, i = lane; i##_step < H / WG; ++i##_step, i += WG)
#endif
#if !INK_PREFILL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define GLUE_SG __attribute__((intel_reqd_sub_group_size(16)))
// The work-group's tree (stride WG / 2 .. 1, lane l + stride into lane l) behind one barrier: each sub-group reads
// the WG / 16 partials of its lanes' column and adds them in the tree's order, then finishes by shuffles - the same
// additions, so bitwise the loop form; every lane gets each total. n sums at once in n rows of WG in `scratch`; the
// caller's next write to `scratch` needs a barrier first.
inline void ink_tree_sums(float* x, uint n, uint lane, __local float* scratch) {
  for (uint j = 0; j < n; ++j) scratch[j * WG + lane] = x[j];
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint l = lane % 16;
  for (uint j = 0; j < n; ++j) {
    const __local float* v = scratch + j * WG + l;
    float w = ((v[0] + v[64]) + (v[32] + v[96])) + ((v[16] + v[80]) + (v[48] + v[112]));
    for (uint s = 8; s; s >>= 1) w += intel_sub_group_shuffle_down(w, w, s);
    x[j] = sub_group_broadcast(w, 0);
  }
}
#if WG != 128
#error "ink_tree_sums: the tree's first three strides read 8 partials a lane"
#endif
#else
#define GLUE_SG
#endif
#if INK_GLUE_BF16
#define GLUE_T ushort
#define GLUE_LOAD(x) bf16f(x)
#define GLUE_STORE(x) rne_bf16(x)
#else
#define GLUE_T float
#define GLUE_LOAD(x) (x)
#define GLUE_STORE(x) (x)
#endif

// Finished states stay in the bank. An optional final state is the working
// residual, so the caller can mix it in place without copying it to the bank.
inline const __global float* ink_depth_row(__global const float* states, __global const float* last,
                                           uint s, uint count, uint rows, uint row) {
  return last && s + 1 == count ? last + (size_t)row * H : states + ((size_t)s * rows + row) * H;
}

#if INK_PREFILL
#define SCALE_ROWS 1
inline void ink_depth_scales(__global const float* states, __global const float* last,
                             __global const float* query, uint count, uint rows,
                             uint r, uint q, uint lane, float eps, __local float* dots, __local float* squares,
                             __local float* scale, __local float* score) {
  for (uint s = 0; s < count; ++s) {
    const __global float* h = ink_depth_row(states, last, s, count, rows, r);
    float sum = 0.0f, dot = 0.0f;
    ROW_LOOP(i) {
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
#else
// Decode: every state's two sums in one pass and one tree - each state's own sums in its own order, so bitwise the
// state-at-a-time form above; dots and squares hold BANK rows of WG. The caller writes them again after a barrier.
#define SCALE_ROWS BANK
inline void ink_depth_scales(__global const float* states, __global const float* last,
                             __global const float* query, uint count, uint rows,
                             uint r, uint q, uint lane, float eps, __local float* dots, __local float* squares,
                             __local float* scale, __local float* score) {
  float sum[BANK], dot[BANK];
  __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s) sum[s] = dot[s] = 0.0f;
  ROW_LOOP(i) {
    const float w = query[(size_t)q * H + i];
    __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s)
      if (s < count) {
        const float h = ink_depth_row(states, last, s, count, rows, r)[i];
        sum[s] += h * h;
        dot[s] += w * h;
      }
  }
  ink_tree_sums(sum, count, lane, squares);
  ink_tree_sums(dot, count, lane, dots);
  if (lane == 0) {
    for (uint s = 0; s < count; ++s) {
      scale[s] = 1.0f / sqrt(sum[s] / H + eps);
      score[s] = dot[s] * scale[s] / sqrt((float)H);
    }
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
#endif

inline float ink_depth_value(__global const float* states, __global const float* last, const __local float* scale,
                             uint count, uint rows, uint row, uint i) {
  float sum = 0.0f;
  for (uint s = 0; s < count; ++s) sum += scale[s] * ink_depth_row(states, last, s, count, rows, row)[i];
  return sum;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_depth_mix(__global const float* states, __global const float* query,
                            __global float* output, uint count, uint rows, float eps, __global const float* last) {
  const uint r = get_group_id(0), q = get_group_id(1), lane = get_local_id(0);
  __local float dots[SCALE_ROWS * WG], squares[SCALE_ROWS * WG], scale[BANK], score[BANK];
  ink_depth_scales(states, last, query, count, rows, r, q, lane, eps, dots, squares, scale, score);
  ROW_LOOP(i)
    output[((size_t)q * rows + r) * H + i] = ink_depth_value(states, last, scale, count, rows, r, i);
}

// Fuse the mixed residual with the projection's weighted RMS and BF16 pack.
__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_depth_mix_norm(__global const float* states, __global const float* query,
                                 __global ushort* output, uint count, uint rows, float eps,
                                 __global const float* gamma0, __global const float* gamma1,
                                 __global const float* gamma2, __global const float* last PF_ROW0_ARG) {
  const uint r = MIX_ROW, q = MIX_STREAM, lane = get_local_id(0);
  __local float dots[SCALE_ROWS * WG], squares[SCALE_ROWS * WG], scale[BANK], score[BANK], mixed[H];
  ink_depth_scales(states, last, query, count, rows, r, q, lane, 1e-6f, dots, squares, scale, score);
  float sum = 0.f;
  ROW_LOOP(i) {
    const float value = ink_depth_value(states, last, scale, count, rows, r, i);
    mixed[i] = value;
    sum += value * value;
  }
#if INK_PREFILL
  squares[lane] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) squares[lane] += squares[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(squares[0] / H + eps);
#else
  ink_tree_sums(&sum, 1, lane, squares);
  const float inv = 1.f / sqrt(sum / H + eps);
#endif
  const __global float* gamma = q == 0 ? gamma0 : q == 1 ? gamma1 : gamma2;
  ROW_LOOP(i)
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
    ROW_LOOP(i) local_sum += x[i] * x[i];
    sums[lane] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = WG / 2; s; s >>= 1) {
      if (lane < s) sums[lane] += sums[lane + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.f / sqrt(sums[0] / H + eps);
    ROW_LOOP(i)
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
                            __global float* out, __global ushort* draft_tap PF_ROW0_ARG) {
  const uint i = get_global_id(0), r = get_global_id(1) + PF_ROW0;
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
// work-item owns a channel, so the accepted prefix can be applied in place. A non-null `accepted` holds the
// accepted rows on the device (a drafted round's ink_accept) in place of `committed`.
__kernel void ink_commit_conv(__global const float* pending, __global float* k, __global float* v,
                              __global float* attn, __global float* mlp, uint rows, uint committed,
                              __global const uint* accepted) {
  if (accepted) committed = *accepted;
  const uint visit = get_group_id(1), i = get_global_id(0);
  const uint kv = INK_KV_N;
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

#if INK_PREFILL
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_prefill_commit_conv(__global const float* pending, __global float* k, __global float* v,
                                     __global float* attn, __global float* mlp, uint rows) {
  const uint i = get_global_id(0), kv = INK_KV_N;
  const uint width = i < 2 * kv ? kv : H;
  const uint offset = i < kv ? 0 : i < 2 * kv ? kv : i < 2 * kv + H ? 2 * kv : 2 * kv + H;
  const uint d = i - offset;
  __global float* history = i < kv ? k : i < 2 * kv ? v : i < 2 * kv + H ? attn : mlp;
  pending += (size_t)rows * offset;
  float next[3];
  for (uint t = 0; t < 3; ++t) {
    const uint row = t + rows;
    next[t] = row < 3 ? history[(size_t)row * width + d] : pending[(size_t)(row - 3) * width + d];
  }
  for (uint t = 0; t < 3; ++t) history[(size_t)t * width + d] = next[t];
}
#endif

#if INK_TP
// The prompt pass's all-reduce: both cards' fp32 rows, card 0's first (both cards add the same bits), into `out`.
__kernel void ink_tp_sum(__global const float* first, __global const float* second, __global float* out) {
  const size_t i = (size_t)get_global_id(1) * H + get_global_id(0);
  out[i] = first[i] + second[i];
}
#endif

// Token embedding followed by the release's weighted RMS norm. The residual
// walk retains the result in FP32 and each projection packs to BF16.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_embed_norm(__global const uint* ctrl, __global const ushort* table,
                              __global const float* gamma, __global float* x, float eps) {
  const uint row = get_group_id(0), lane = get_local_id(0);
#if INK_PREFILL
  const __global ushort* e = table + (size_t)ctrl[row] * H;
#else
  const __global ushort* e = table + (size_t)ctrl[CTRL_CUR + row] * H;
#endif
  __local float squares[WG];
  float ss = 0.f;
  ROW_LOOP(i) {
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
  ROW_LOOP(i)
    x[(size_t)row * H + i] = (bf16f(e[i]) * inv) * gamma[i];
}

__kernel void ink_pack(__global const float* x, __global ushort* a) {
  const uint i = get_global_id(0);
  a[i] = rne_bf16(x[i]);
}

__kernel void ink_tap(__global const float* x, __global const ushort* a, __global ushort* tap, uint slot PF_ROW0_ARG) {
  const uint i = get_global_id(0), row = get_global_id(1) + PF_ROW0;
  const size_t at = (size_t)row * H + i;
  tap[((size_t)row * 4 + slot) * H + i] = x ? rne_bf16(x[at]) : a[at];
}

#if INK_CONTROLLER_FUSION
inline float ink_rms_inverse(float sum, uint lane, float eps, __local float* squares) {
#if INK_PREFILL
  squares[lane] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s; s >>= 1) {
    if (lane < s) squares[lane] += squares[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.f / sqrt(squares[0] / H + eps);
#else
  ink_tree_sums(&sum, 1, lane, squares);
  const float inv = 1.f / sqrt(sum / H + eps);
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  return inv;
}

// Each work-group produces its own row and reads only columns of that row
// in the following depth mix. Other rows are read from the raw projection. In decode each lane reads back only the
// columns it wrote (ROW_LOOP's mapping), so no fence is needed there.
inline void ink_attention_residual(__global float* out, __global const float* raw,
                                   __global const float* history, __global const float* taps,
                                   __global const float* residual, uint row, uint lane) {
  ROW_LOOP(i) {
    float sum = 0.f;
    for (uint tap = 0; tap < 4; ++tap) {
      const int t = (int)row + (int)tap - 3;
      const float x = t < 0 ? history[(t + 3) * H + i] : raw[t * H + i];
      sum += taps[i * 4 + tap] * x;
    }
    const size_t at = (size_t)row * H + i;
    out[at] = (raw[at] + sum) / sqrt(5.f) + (residual ? residual[at] : 0.f);
  }
#if INK_PREFILL
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_post_mix_norm(__global const float* bank, __global float* last,
                                 __global const float* query, __global ushort* out,
                                 __global const float* gamma, uint states, uint rows, float eps,
                                 __global const float* raw, __global const float* history,
                                 __global const float* taps, __global const float* residual PF_ROW0_ARG) {
  const uint r = get_group_id(0) + PF_ROW0, lane = get_local_id(0);
  __local float dots[SCALE_ROWS * WG], squares[SCALE_ROWS * WG], scale[BANK], score[BANK], temp[H];
#if INK_PREFILL
  ink_attention_residual(last, raw, history, taps, residual, r, lane);
#else
  if (raw) ink_attention_residual(last, raw, history, taps, residual, r, lane);   // else o_proj's epilogue did
#endif
  ink_depth_scales(bank, last, query, states, rows, r, 0, lane, 1e-6f, dots, squares, scale, score);
  float ss = 0.f;
  ROW_LOOP(i) {
    const float x = ink_depth_value(bank, last, scale, states, rows, r, i);
    temp[i] = x;
    ss += x * x;
  }
  const float inv = ink_rms_inverse(ss, lane, eps, squares);
  ROW_LOOP(i)
    out[(size_t)r * H + i] = rne_bf16((temp[i] * inv) * gamma[i]);
}

// All four message streams for the next pass share its saved midpoint and output.
__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_cross_mix(__global const float* states, __global const float* q,
                            __global const float* kv, __global float* out, uint rows) {
  const uint r = get_group_id(0), i = get_group_id(1), lane = get_local_id(0);
  const uint stream = i % 4, layer = i / 4;
  const __global float* query = stream == 0 || stream == 3 ? q : kv;
  const uint which = 2 * layer + (stream >= 2);
  __local float dots[SCALE_ROWS * WG], squares[SCALE_ROWS * WG], scale[BANK], score[BANK];
  ink_depth_scales(states, 0, query, 2, rows, r, which, lane, 1e-6f, dots, squares, scale, score);
  ROW_LOOP(c)
    out[((size_t)i * rows + r) * H + c] = ink_depth_value(states, 0, scale, 2, rows, r, c);
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_mix_cat(__global const float* bank, __global float* last,
                          __global const float* query, __global const GLUE_T* messages,
                          __global GLUE_T* mixed, __global ushort* cat,
                          uint states, uint rows, uint shared_mix, float eps,
                          __global const float* raw, __global const float* history,
                          __global const float* taps, __global const float* residual PF_ROW0_ARG) {
  const uint r = MIX_ROW, stream = MIX_STREAM, lane = get_local_id(0);
  const size_t start = ((size_t)stream * rows + r) * H;
  __local float dots[SCALE_ROWS * WG], squares[SCALE_ROWS * WG], scale[BANK], score[BANK], temp[H];
  if (raw) ink_attention_residual(last, raw, history, taps, residual, r, lane);
  ink_depth_scales(bank, last, query, states, rows, r, shared_mix ? 0 : stream, lane, 1e-6f,
                   dots, squares, scale, score);
  float sx = 0.f, sm = 0.f;
  ROW_LOOP(i) {
    const float x = ink_depth_value(bank, last, scale, states, rows, r, i);
    const float msg = GLUE_LOAD(messages[start + i]);
    temp[i] = x;
    mixed[start + i] = GLUE_STORE(x);
    sx += x * x;
    sm += msg * msg;
  }
  const float rx = ink_rms_inverse(sx, lane, eps, squares);
  const float rm = ink_rms_inverse(sm, lane, eps, squares);
  ROW_LOOP(i) {
    cat[2 * start + i] = rne_bf16(temp[i] * rx);
    cat[2 * start + H + i] = rne_bf16(GLUE_LOAD(messages[start + i]) * rm);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
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
  ROW_LOOP(i) {
    const size_t j = start + i;
    const float x = (1.f - tanh(fg[j])) * mixed[j] + tanh(fg[gain + j]) * messages[j];
    temp[i] = x;
    sum += x * x;
  }
  const float inv = ink_rms_inverse(sum, lane, eps, squares);
  ROW_LOOP(i)
    out[start + i] = rne_bf16((temp[i] * inv) * gamma[i]);
}
#endif

#if INK_PREFILL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
// ink_prefill_controller - a prompt chunk's controller gates and norm in one pass, for 8 rows of one stream:
// f = rank . forget^T and g = rank . gain^T (K = 64) by DPAS in pf_gemm's chain (k16 steps ascending from zero,
// the same bf16 operands, so the sums are its fp32 words), x = (1 - tanh f) mixed + tanh g message (ink_controller_norm's
// expression), and each row's RMS in ink_controller_norm's order: sub-group s takes the atoms s, s + 8, ..., so lane
// l of it sums columns 16 s + l + 128 k in ascending k exactly as that kernel's lane 16 s + l, then the same tree.
// The f / g rows never reach memory; x waits in `mixed` (each word read, then written, by one lane) for the scale.
// Weights are the release's bf16 tiles [N / 16][K][16].
#define CR 64
#define CTRL_ATOMS (H / 16)
__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ink_prefill_controller(__global const ushort* rank, __global GLUE_T* mixed,
                                     __global const GLUE_T* messages, __global ushort* out,
                                     __global const ushort* f0, __global const ushort* f1,
                                     __global const ushort* f2, __global const ushort* g0,
                                     __global const ushort* g1, __global const ushort* g2,
                                     __global const float* q_norm, __global const float* k_norm,
                                     __global const float* v_norm, uint rows, float eps, uint row0) {
  const uint r0 = row0 + get_group_id(0) * 8, stream = get_group_id(1);
  const uint sg = get_sub_group_id(), l = get_sub_group_local_id(), lane = get_local_id(0);
  __global const ushort* fw = stream == 0 ? f0 : stream == 1 ? f1 : f2;
  __global const ushort* gw = stream == 0 ? g0 : stream == 1 ? g1 : g2;
  const __global float* gamma = stream == 0 ? q_norm : stream == 1 ? k_norm : v_norm;
  __local float squares[8][WG];
  __local float inv[8];
  short8 a[CR / 16];   // lane = k within the k16 step, component = row
#pragma unroll
  for (uint kk = 0; kk < CR / 16; ++kk) {
    ushort t[8];
#pragma unroll
    for (uint r = 0; r < 8; ++r)
      t[r] = rank[((size_t)stream * rows + min(r0 + r, rows - 1)) * CR + 16 * kk + l];
    a[kk] = as_short8(vload8(0, t));
  }
  float sum[8];
#pragma unroll
  for (uint r = 0; r < 8; ++r) sum[r] = 0.f;
  for (uint atom = sg; atom < CTRL_ATOMS; atom += WG / 16) {
    float8 f = (float8)(0.0f), gv = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < CR / 16; ++kk) {
      const size_t at = (size_t)atom * CR * 16 + kk * 256;
      const ushort8 flo = intel_sub_group_block_read_us8(fw + at), fhi = intel_sub_group_block_read_us8(fw + at + 128);
      const ushort8 glo = intel_sub_group_block_read_us8(gw + at), ghi = intel_sub_group_block_read_us8(gw + at + 128);
      const int8 fb = as_int8((uint8)((uint)flo.s0 | (uint)flo.s1 << 16, (uint)flo.s2 | (uint)flo.s3 << 16,
                                      (uint)flo.s4 | (uint)flo.s5 << 16, (uint)flo.s6 | (uint)flo.s7 << 16,
                                      (uint)fhi.s0 | (uint)fhi.s1 << 16, (uint)fhi.s2 | (uint)fhi.s3 << 16,
                                      (uint)fhi.s4 | (uint)fhi.s5 << 16, (uint)fhi.s6 | (uint)fhi.s7 << 16));
      const int8 gb = as_int8((uint8)((uint)glo.s0 | (uint)glo.s1 << 16, (uint)glo.s2 | (uint)glo.s3 << 16,
                                      (uint)glo.s4 | (uint)glo.s5 << 16, (uint)glo.s6 | (uint)glo.s7 << 16,
                                      (uint)ghi.s0 | (uint)ghi.s1 << 16, (uint)ghi.s2 | (uint)ghi.s3 << 16,
                                      (uint)ghi.s4 | (uint)ghi.s5 << 16, (uint)ghi.s6 | (uint)ghi.s7 << 16));
      f = intel_sub_group_bf16_bf16_matrix_mad_k16(a[kk], fb, f);
      gv = intel_sub_group_bf16_bf16_matrix_mad_k16(a[kk], gb, gv);
    }
    float fr[8], gr[8];
    vstore8(f, 0, fr);
    vstore8(gv, 0, gr);
    const uint col = atom * 16 + l;
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const size_t j = ((size_t)stream * rows + min(r0 + r, rows - 1)) * H + col;
      const float x = (1.f - tanh(fr[r])) * GLUE_LOAD(mixed[j]) + tanh(gr[r]) * GLUE_LOAD(messages[j]);
      sum[r] += x * x;
      if (r0 + r < rows) mixed[j] = GLUE_STORE(x);
    }
  }
#pragma unroll
  for (uint r = 0; r < 8; ++r) squares[r][lane] = sum[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint st = WG / 2; st; st >>= 1) {
    if (lane < st)
#pragma unroll
      for (uint r = 0; r < 8; ++r) squares[r][lane] += squares[r][lane + st];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane < 8) inv[lane] = 1.f / sqrt(squares[lane][0] / H + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint r = 0; r < 8 && r0 + r < rows; ++r)
    for (uint i = lane; i < H; i += WG) {
      const size_t j = ((size_t)stream * rows + r0 + r) * H + i;
      out[j] = rne_bf16((GLUE_LOAD(mixed[j]) * inv[r]) * gamma[i]);
    }
}

// The prompt pass's row kernels for a window controller's shared mix and the cross-pass messages: one work-group a
// row makes every stream of it, so each depth state's squares are summed once for all of the row's queries (a shared
// mix's x once). The arithmetic is ink_depth_scales / ink_depth_value / ink_rms_inverse's term for term: each
// state's squares and each query's dots in the same lane order, every total by the same stride tree (the row's
// totals side by side, one barrier a stride), the scales' softmax in the same steps - so bitwise the
// stream-a-work-group kernels above.
#define QMAX 4
#define TREE_ROWS (BANK * (1 + QMAX))

// The stride tree over rows j < n of scratch (WG partials each); the totals land in scratch[j * WG]. The caller wrote
// the partials; the totals are read before scratch is written again (a barrier between).
inline void ink_pf_tree(__local float* scratch, uint n, uint lane) {
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride; stride >>= 1) {
    if (lane < stride)
      for (uint j = 0; j < TREE_ROWS; ++j)
        if (j < n) scratch[j * WG + lane] += scratch[j * WG + lane + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// The depth mix's scales of nq queries (q[j]: query j's row) over count states of row r, each state read once for
// them all. scratch: count * (1 + nq) rows of WG.
inline void ink_pf_scales(__global const float* states, __global const float* last,
                          __global const float* const* q, uint nq, uint count, uint rows, uint r, uint lane,
                          __local float* scratch, float scale[QMAX][BANK]) {
  float sq[BANK], dot[QMAX][BANK];
  for (uint s = 0; s < BANK; ++s) {
    sq[s] = 0.0f;
    for (uint j = 0; j < QMAX; ++j) dot[j][s] = 0.0f;
  }
  for (uint i = lane; i < H; i += WG) {
    float w[QMAX];
    for (uint j = 0; j < QMAX; ++j)
      if (j < nq) w[j] = q[j][i];
    for (uint s = 0; s < BANK; ++s)
      if (s < count) {
        const float h = ink_depth_row(states, last, s, count, rows, r)[i];
        sq[s] += h * h;
        for (uint j = 0; j < QMAX; ++j)
          if (j < nq) dot[j][s] += w[j] * h;
      }
  }
  const uint stride = 1 + nq;
  for (uint s = 0; s < BANK; ++s)
    if (s < count) {
      scratch[s * stride * WG + lane] = sq[s];
      for (uint j = 0; j < QMAX; ++j)
        if (j < nq) scratch[(s * stride + 1 + j) * WG + lane] = dot[j][s];
    }
  ink_pf_tree(scratch, count * stride, lane);
  for (uint j = 0; j < QMAX; ++j)
    if (j < nq) {
      float score[BANK];
      for (uint s = 0; s < BANK; ++s)
        if (s < count) {
          scale[j][s] = 1.0f / sqrt(scratch[s * stride * WG] / H + 1e-6f);
          score[s] = scratch[(s * stride + 1 + j) * WG] * scale[j][s] / sqrt((float)H);
        }
      float high = score[0], total = 0.0f;
      for (uint s = 1; s < BANK; ++s)
        if (s < count) high = fmax(high, score[s]);
      for (uint s = 0; s < BANK; ++s)
        if (s < count) {
          score[s] = exp(score[s] - high);
          total += score[s];
        }
      for (uint s = 0; s < BANK; ++s)
        if (s < count) scale[j][s] *= score[s] / total;
    }
}

inline float ink_pf_value(__global const float* states, __global const float* last, const float* scale,
                          uint count, uint rows, uint r, uint i) {
  float sum = 0.0f;
  for (uint s = 0; s < BANK; ++s)
    if (s < count) sum += scale[s] * ink_depth_row(states, last, s, count, rows, r)[i];
  return sum;
}

// ink_mix_cat of a window visit's attention controller a row: a shared mix, so one x serves its three streams.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_prefill_mix_cat(__global const float* bank, __global const float* last,
                                  __global const float* query, __global const GLUE_T* messages,
                                  __global GLUE_T* mixed, __global ushort* cat, uint count, uint rows, float eps,
                                  uint row0) {
  const uint r = row0 + get_group_id(0), lane = get_local_id(0);
  __local float scratch[BANK * 2 * WG], temp[H];
  __global const float* q[QMAX] = {query};
  float scale[QMAX][BANK];
  ink_pf_scales(bank, last, q, 1, count, rows, r, lane, scratch, scale);
  float sx = 0.f, sm[3] = {0.f, 0.f, 0.f};
  for (uint i = lane; i < H; i += WG) {
    const float x = ink_pf_value(bank, last, scale[0], count, rows, r, i);
    temp[i] = x;
    sx += x * x;
    for (uint st = 0; st < 3; ++st) {
      const size_t at = ((size_t)st * rows + r) * H + i;
      const float msg = GLUE_LOAD(messages[at]);
      mixed[at] = GLUE_STORE(x);
      sm[st] += msg * msg;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);   // the scales' totals read; scratch again
  scratch[lane] = sx;
  for (uint st = 0; st < 3; ++st) scratch[(1 + st) * WG + lane] = sm[st];
  ink_pf_tree(scratch, 4, lane);
  const float rx = 1.f / sqrt(scratch[0] / H + eps);
  for (uint st = 0; st < 3; ++st) {
    const float rm = 1.f / sqrt(scratch[(1 + st) * WG] / H + eps);
    const size_t start = ((size_t)st * rows + r) * H;
    for (uint i = lane; i < H; i += WG) {
      cat[2 * start + i] = rne_bf16(temp[i] * rx);
      cat[2 * start + H + i] = rne_bf16(GLUE_LOAD(messages[start + i]) * rm);
    }
  }
}

// ink_cross_mix a row: the four message streams over the saved midpoint and the previous pass's output.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_prefill_cross_mix(__global const float* states, __global const float* qs,
                                    __global const float* kv, __global GLUE_T* out, uint rows,
                                    __global const float* last, uint layer, uint row0) {
  const uint r = row0 + get_group_id(0), lane = get_local_id(0);
  __local float scratch[2 * 5 * WG];
  __global const float* q[QMAX] = {qs + (size_t)(2 * layer) * H, kv + (size_t)(2 * layer) * H,
                                   kv + (size_t)(2 * layer + 1) * H, qs + (size_t)(2 * layer + 1) * H};
  float scale[QMAX][BANK];
  ink_pf_scales(states, last, q, 4, 2, rows, r, lane, scratch, scale);
  for (uint i = lane; i < H; i += WG)
    for (uint j = 0; j < 4; ++j)
      out[((size_t)j * rows + r) * H + i] = GLUE_STORE(ink_pf_value(states, last, scale[j], 2, rows, r, i));
}
#endif

#if INK_GLUE_SPLIT
#if INK_PREFILL || !INK_CONTROLLER_FUSION
#error "INK_GLUE_SPLIT: the decode build with the controller kernels"
#endif
// Decode's one-row mixes and norms with their row spread over SPLIT_PARTS work-groups of one sub-group each (one
// work-group a row sat on one Xe core, waiting on its loads). Global lane L = 16 part + l keeps ROW_LOOP's mapping
// (columns L + WG k), so each lane's sums are the one-work-group kernels' lane sums, and every reduction is
// ink_tree_sums' tree over those WG partials read back from `parts`: the same additions in the same order, bitwise
// ink_depth_mix_norm, ink_post_mix_norm, ink_mix_cat and ink_controller_norm. A dependent reduction is a new launch.
// Grid (SPLIT_PARTS, rows, streams); `parts` holds a (stream, row)'s partial rows of WG, `out_parts` the next phase's.
// The loads are unguarded (a state past `count` reads state 0 and its sums are dropped by a select) and the pointers
// restrict, so a lane's reads are in flight together rather than one after another.
#define SPLIT_PARTS (WG / 16)
#define SPLIT_ROW get_group_id(1)
#define SPLIT_STREAM get_group_id(2)
#define SPLIT_LANE (get_group_id(0) * 16 + get_local_id(0))
#define SPLIT_SCALE_SUMS (2 * BANK)

// ink_tree_sums' total of one row of WG partials: sub-group lane l adds column l in the tree's first three strides,
// then the shuffles.
inline float ink_split_tree(__global const float* v) {
  v += get_sub_group_local_id();
  float w = ((v[0] + v[64]) + (v[32] + v[96])) + ((v[16] + v[80]) + (v[48] + v[112]));
  for (uint s = 8; s; s >>= 1) w += intel_sub_group_shuffle_down(w, w, s);
  return sub_group_broadcast(w, 0);
}

// ink_depth_scales' scales from its partial sums (squares first, then dots), in every lane.
inline void ink_split_scales(__global const float* parts, uint count, float eps, float* scale) {
  float score[BANK];
  for (uint s = 0; s < count; ++s) {
    const float sum = ink_split_tree(parts + s * WG), dot = ink_split_tree(parts + (BANK + s) * WG);
    scale[s] = 1.0f / sqrt(sum / H + eps);
    score[s] = dot * scale[s] / sqrt((float)H);
  }
  float high = score[0], total = 0.0f;
  for (uint s = 1; s < count; ++s) high = fmax(high, score[s]);
  for (uint s = 0; s < count; ++s) {
    score[s] = exp(score[s] - high);
    total += score[s];
  }
  for (uint s = 0; s < count; ++s) scale[s] *= score[s] / total;
}

// The states' rows; past `count`, state 0's (read, never used).
#define SPLIT_ROWS(rowp, states, last, count, rows, r)                                              \
  const __global float* restrict rowp[BANK];                                                       \
  __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s)                              \
    rowp[s] = ink_depth_row(states, last, s < count ? s : 0, count, rows, r);

// ink_depth_value's sum: the same terms in the same order, the ones past `count` not added.
inline float ink_split_value(const __global float* restrict* rowp, const float* scale, uint count, uint i) {
  float h[BANK];
  __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s) h[s] = rowp[s][i];
  float sum = 0.0f;
  __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s) {
    const float t = scale[s] * h[s];
    sum = s < count ? sum + t : sum;
  }
  return sum;
}

// Phase 1 of every mix: each state's squares and dots with the stream's query (`shared`: query 0 for all streams).
__attribute__((reqd_work_group_size(16, 1, 1)))
GLUE_SG
__kernel void ink_split_sums(__global const float* restrict states, __global const float* restrict last,
                             __global const float* restrict query, __global float* restrict parts, uint count,
                             uint rows, uint shared) {
  const uint r = SPLIT_ROW, stream = SPLIT_STREAM, lane = SPLIT_LANE, q = shared ? 0 : stream;
  SPLIT_ROWS(rowp, states, last, count, rows, r)
  const __global float* restrict w = query + (size_t)q * H;
  float sum[BANK], dot[BANK];
  __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s) sum[s] = dot[s] = 0.0f;
  ROW_LOOP(i) {
    const float wi = w[i];
    __attribute__((opencl_unroll_hint)) for (uint s = 0; s < BANK; ++s) {
      const float h = rowp[s][i];
      sum[s] += h * h;
      dot[s] += wi * h;
    }
  }
  parts += ((size_t)stream * rows + r) * SPLIT_SCALE_SUMS * WG + lane;
  for (uint s = 0; s < count; ++s) {
    parts[s * WG] = sum[s];
    parts[(BANK + s) * WG] = dot[s];
  }
}

// Phase 2 of ink_depth_mix_norm / ink_post_mix_norm: the mixed row into `mixed` (fp32), a work-item a value over
// the row (grid (H / WG, rows, streams); each sub-group its own copy of the scales); its squares are
// ink_split_squares' launch. On the split lanes each lane's 18 values ran one after another (~5 us a launch at any
// row count). The same values and sums, so bitwise.
__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_split_mix(__global const float* restrict states, __global const float* restrict last,
                            __global const float* restrict parts, __global float* restrict mixed, uint count,
                            uint rows) {
  const uint r = get_group_id(1), stream = get_group_id(2);
  const size_t row = (size_t)stream * rows + r;
  SPLIT_ROWS(rowp, states, last, count, rows, r)
  float scale[BANK];
  ink_split_scales(parts + row * SPLIT_SCALE_SUMS * WG, count, 1e-6f, scale);
  const uint i = get_global_id(0);
  mixed[row * H + i] = ink_split_value(rowp, scale, count, i);
}

// Phase 3: the weighted norm and bf16 pack (gamma by stream, as ink_depth_mix_norm).
__attribute__((reqd_work_group_size(16, 1, 1)))
GLUE_SG
__kernel void ink_split_norm(__global const float* restrict mixed, __global const float* restrict parts,
                             __global ushort* restrict output,
                             uint rows, float eps, __global const float* gamma0, __global const float* gamma1,
                             __global const float* gamma2) {
  const uint r = SPLIT_ROW, q = SPLIT_STREAM, lane = SPLIT_LANE;
  const size_t row = (size_t)q * rows + r;
  const float inv = 1.f / sqrt(ink_split_tree(parts + row * WG) / H + eps);
  const __global float* gamma = q == 0 ? gamma0 : q == 1 ? gamma1 : gamma2;
  ROW_LOOP(i)
    output[row * H + i] = rne_bf16((mixed[row * H + i] * inv) * gamma[i]);
}

// Phase 2 of ink_mix_cat: the mixed row into `mixed`, a work-item a value as ink_split_mix; then each lane's squares
// of it and of the message (ink_split_cat_squares).
__attribute__((reqd_work_group_size(WG, 1, 1)))
GLUE_SG
__kernel void ink_split_mix_cat(__global const float* restrict bank, __global const float* restrict last,
                                __global const float* restrict parts, __global float* restrict mixed, uint states,
                                uint rows) {
  const uint r = get_group_id(1), stream = get_group_id(2);
  const size_t row = (size_t)stream * rows + r;
  SPLIT_ROWS(rowp, bank, last, states, rows, r)
  float scale[BANK];
  ink_split_scales(parts + row * SPLIT_SCALE_SUMS * WG, states, 1e-6f, scale);
  const uint i = get_global_id(0);
  mixed[row * H + i] = ink_split_value(rowp, scale, states, i);
}

__attribute__((reqd_work_group_size(16, 1, 1)))
GLUE_SG
__kernel void ink_split_cat_squares(__global const float* restrict mixed, __global const float* restrict messages,
                                    __global float* restrict out_parts, uint rows) {
  const uint r = SPLIT_ROW, stream = SPLIT_STREAM, lane = SPLIT_LANE;
  const size_t row = (size_t)stream * rows + r, start = row * H;
  float x[H / WG], msg[H / WG];
  ROW_LOOP(i) {
    x[i_step] = mixed[start + i];
    msg[i_step] = messages[start + i];
  }
  float sx = 0.f, sm = 0.f;
  ROW_LOOP(i) {
    sx += x[i_step] * x[i_step];
    sm += msg[i_step] * msg[i_step];
  }
  out_parts[(2 * row) * WG + lane] = sx;
  out_parts[(2 * row + 1) * WG + lane] = sm;
}

// Phase 3 of ink_mix_cat: both unweighted norms into the controller's A row.
__attribute__((reqd_work_group_size(16, 1, 1)))
GLUE_SG
__kernel void ink_split_cat(__global const float* restrict mixed, __global const float* restrict messages,
                            __global const float* restrict parts, __global ushort* restrict cat, uint rows, float eps) {
  const uint r = SPLIT_ROW, stream = SPLIT_STREAM, lane = SPLIT_LANE;
  const size_t row = (size_t)stream * rows + r, start = row * H;
  const float rx = 1.f / sqrt(ink_split_tree(parts + (2 * row) * WG) / H + eps);
  const float rm = 1.f / sqrt(ink_split_tree(parts + (2 * row + 1) * WG) / H + eps);
  ROW_LOOP(i) {
    cat[2 * start + i] = rne_bf16(mixed[start + i] * rx);
    cat[2 * start + H + i] = rne_bf16(messages[start + i] * rm);
  }
}

// ink_controller_norm in three phases: the blend written back over `mixed`, a work-item a value over the whole row
// (on the split lanes each lane's 18 values' tanh pairs ran one after another: ~14 us a launch at any row count);
// then each lane's squares in ROW_LOOP's order; then the weighted norm. The same values and sums, so bitwise.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_split_blend(__global float* restrict mixed, __global const float* restrict messages,
                              __global const float* restrict fg, uint rows, uint streams) {
  const size_t j = ((size_t)get_group_id(2) * rows + get_group_id(1)) * H + get_global_id(0);
  const size_t gain = (size_t)streams * rows * H;
  mixed[j] = (1.f - tanh(fg[j])) * mixed[j] + tanh(fg[gain + j]) * messages[j];
}

__attribute__((reqd_work_group_size(16, 1, 1)))
GLUE_SG
__kernel void ink_split_squares(__global const float* restrict mixed, __global float* restrict out_parts, uint rows) {
  const uint r = SPLIT_ROW, stream = SPLIT_STREAM, lane = SPLIT_LANE;
  const size_t row = (size_t)stream * rows + r, start = row * H;
  float x[H / WG];
  ROW_LOOP(i) x[i_step] = mixed[start + i];
  float sum = 0.f;
  ROW_LOOP(i) sum += x[i_step] * x[i_step];
  out_parts[row * WG + lane] = sum;
}
#endif
