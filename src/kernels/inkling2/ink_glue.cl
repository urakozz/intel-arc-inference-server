// Inkling's FP32 residual, depth-wise mix and loop-controller glue.
#include "bf16.h"

#ifndef H
#define H 2304
#endif
#define WG 128
#define BANK 5
#pragma OPENCL FP_CONTRACT OFF
// The (row, stream) work-groups of the multi-stream mixes. A prompt chunk numbers the streams of one row next to
// each other, so that row's depth states come from memory once and from the cache for its other streams.
#if INK_PREFILL
#define MIX_ROW get_group_id(1)
#define MIX_STREAM get_group_id(0)
#else
#define MIX_ROW get_group_id(0)
#define MIX_STREAM get_group_id(1)
#endif
// INK_GLUE_BF16 (prompt pass, lossy): the rows kept between glue kernels - the cross-pass messages and the
// controllers' mixed rows - are stored as bf16 (one rounding at the store); every sum stays fp32. Off, the
// rows are fp32 and the text below is the exact path.
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
  const uint r = MIX_ROW, q = MIX_STREAM, lane = get_local_id(0);
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

#if INK_PREFILL
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_prefill_commit_conv(__global const float* pending, __global float* k, __global float* v,
                                     __global float* attn, __global float* mlp, uint rows) {
  const uint i = get_global_id(0), kv = 1024;
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
#if INK_PREFILL
                            __global const float* kv, __global GLUE_T* out, uint rows,
                            __global const float* last, uint layer) {
  const uint r = MIX_ROW, i = MIX_STREAM, lane = get_local_id(0);
  const uint stream = i;
#else
                            __global const float* kv, __global float* out, uint rows) {
  const uint r = get_group_id(0), i = get_group_id(1), lane = get_local_id(0);
  const uint stream = i % 4, layer = i / 4;
#endif
  const __global float* query = stream == 0 || stream == 3 ? q : kv;
  const uint which = 2 * layer + (stream >= 2);
  __local float dots[WG], squares[WG], scale[BANK], score[BANK];
#if INK_PREFILL
  ink_depth_scales(states, last, query, 2, rows, r, which, lane, 1e-6f, dots, squares, scale, score);
  for (uint c = lane; c < H; c += WG)
    out[((size_t)i * rows + r) * H + c] = GLUE_STORE(ink_depth_value(states, last, scale, 2, rows, r, c));
#else
  ink_depth_scales(states, 0, query, 2, rows, r, which, lane, 1e-6f, dots, squares, scale, score);
  for (uint c = lane; c < H; c += WG)
    out[((size_t)i * rows + r) * H + c] = ink_depth_value(states, 0, scale, 2, rows, r, c);
#endif
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ink_mix_cat(__global const float* bank, __global float* last,
                          __global const float* query, __global const GLUE_T* messages,
                          __global GLUE_T* mixed, __global ushort* cat,
                          uint states, uint rows, uint shared_mix, float eps,
                          __global const float* raw, __global const float* history,
                          __global const float* taps, __global const float* residual) {
  const uint r = MIX_ROW, stream = MIX_STREAM, lane = get_local_id(0);
  const size_t start = ((size_t)stream * rows + r) * H;
  __local float dots[WG], squares[WG], scale[BANK], score[BANK], temp[H];
  if (raw) ink_attention_residual(last, raw, history, taps, residual, r, lane);
  ink_depth_scales(bank, last, query, states, rows, r, shared_mix ? 0 : stream, lane, 1e-6f,
                   dots, squares, scale, score);
  float sx = 0.f, sm = 0.f;
  for (uint i = lane; i < H; i += WG) {
    const float x = ink_depth_value(bank, last, scale, states, rows, r, i);
    const float msg = GLUE_LOAD(messages[start + i]);
    temp[i] = x;
    mixed[start + i] = GLUE_STORE(x);
    sx += x * x;
    sm += msg * msg;
  }
  const float rx = ink_rms_inverse(sx, lane, eps, squares);
  const float rm = ink_rms_inverse(sm, lane, eps, squares);
  for (uint i = lane; i < H; i += WG) {
    cat[2 * start + i] = rne_bf16(temp[i] * rx);
    cat[2 * start + H + i] = rne_bf16(GLUE_LOAD(messages[start + i]) * rm);
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
                                     __global const float* v_norm, uint rows, float eps) {
  const uint r0 = get_group_id(0) * 8, stream = get_group_id(1);
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
#endif
