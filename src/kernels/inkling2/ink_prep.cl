// Rows from the shared projections into the attention kernel's Q/relative
// vectors and BF16 growing/ring cache. Convolution history is committed
// separately after selecting the accepted verify rows.
#include "bf16.h"

#define HD 128
#define REL 16
// NH / KV: the heads a launch prepares - every head on one card; a `--parallel tp` card's half (12 / 4, INK_TP), whose
// per-head weights the engine passes from the card's first head. STATIC_KW: the full layers' static taps' row pitch
// ([4][kv channels] of the whole model).
#if INK_TP
#define NH 12
#define KV 4
#define STATIC_KW 1024
#else
#define NH 24
#define KV 8
#endif
#define KW (KV * HD)
#ifndef STATIC_KW
#define STATIC_KW KW
#endif
#pragma OPENCL FP_CONTRACT OFF
// INK_KV8 (PERF-102, `--kv-cache int8`; undefined on every other command line): the fused prep of a FULL visit with the
// int8 cache - launched for the full visits only (ring 0), the window visits keep the default build. K and V as the
// bf16 cache would hold them (rne_bf16 of the clipped / normed row), rotated (kv8_scheme.h, head_dim 128) and stored
// as int8 [max_len][8][128] with an fp16 scale a (position, kv head) in ks / vs [max_len][8]; q rotated to match.
#ifndef INK_KV8
#define INK_KV8 0
#endif
#if INK_KV8
#if !INK_FUSED_PREP
#error "ink_prep INK_KV8: the fused prep only"
#endif
#define KV8_N HD
#include "kv8_scheme.h"
#endif

inline float norm_square(float x, uint lane, __local float* scratch) {
  scratch[lane] = x * x;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = HD / 2; s; s >>= 1) {
    if (lane < s) scratch[lane] += scratch[lane + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float sum = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return sum;
}

#if INK_FUSED_PREP
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void ink_qkv_prep(__global float* q, __global float* relative,
                           __global const float* q_scale, __global const float* rel_scale,
                           __global const float* k, __global const float* v,
                           __global const float* kh, __global const float* vh,
                           __global const float* kt, __global const float* vt,
                           __global const float* kd, __global const float* vd,
#if INK_KV8
                           __global char* ck, __global char* cv, __global ushort* ks, __global ushort* vs,
#else
                           __global ushort* ck, __global ushort* cv,
#endif
#if INK_PREFILL
                           uint pos,
#else
                           __global const uint* ctrl,
#endif
                           uint ring, float eps, float clip, float r_clip, uint log_floor, float alpha) {
  const uint head = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
#if !INK_PREFILL
  if (r >= ctrl[CTRL_NACT]) return;
  const uint pos = ctrl[CTRL_POS];
#endif
  __local float square[HD];
  if (head < NH) {
    const size_t i = ((size_t)r * NH + head) * HD + d;
    const size_t j = ((size_t)r * NH + head) * REL + d;
    const float qx = q[i] * q_scale[head * HD + d];
    const float rx = d < REL ? relative[j] * rel_scale[head * REL + d] : 0.f;
    const float qs = norm_square(qx, d, square), rs = norm_square(rx, d, square);
    const float tau = ring ? 1.f : 1.f + alpha * log(fmax((float)(pos + r + 1) / log_floor, 1.f));
#if INK_KV8
    q[i] = kv8n_rotate_q(qx * fmin(1.f, clip / (sqrt(qs) + 1e-6f)) * tau, d, square);
#else
    q[i] = qx * fmin(1.f, clip / (sqrt(qs) + 1e-6f)) * tau;
#endif
    if (d < REL) relative[j] = rx * fmin(1.f, r_clip / (sqrt(rs) + 1e-6f)) * tau;
  } else {
    const uint key = head < NH + KV, h = head - NH - (key ? 0 : KV), i = h * HD + d;
    const __global float* input = key ? k : v;
    const __global float* history = key ? kh : vh;
    const __global float* static_taps = key ? kt : vt;
    const __global float* projected = key ? kd : vd;
#if !INK_KV8
    __global ushort* cache = key ? ck : cv;
#endif
    float sum = 0.f;
    for (uint a = 0; a < 4; ++a) {
      const int t = projected ? (int)r - (int)a : (int)r + (int)a - 3;
      const float x = t < 0 ? history[(t + 3) * KW + i] : input[t * KW + i];
      const float w = projected ? static_taps[a * STATIC_KW + i] + projected[(size_t)r * 64 + h * 4 + a]
                                 : static_taps[i * 4 + a];
      sum += x * w;
    }
    const float y = (input[(size_t)r * KW + i] + sum) / (projected ? 3.f : sqrt(5.f));
    const float ss = norm_square(y, d, square);
    const float scale = key ? fmin(1.f, clip / (sqrt(ss) + 1e-6f)) : 1.f / sqrt(ss / HD + eps);
    const uint slot = ring ? (pos + r) & (ring - 1) : pos + r;
#if INK_KV8
    kv8n_store(kv8n_rotate(bf16f(rne_bf16(y * scale)), d, square), d, square, (key ? ck : cv) + (size_t)slot * KW + h * HD,
               (key ? ks : vs) + (size_t)slot * KV + h);
#else
    cache[(size_t)slot * KW + i] = rne_bf16(y * scale);
#endif
  }
}
#else
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void ink_qr_prep(__global const float* q, __global const float* relative,
                          __global const float* q_scale, __global const float* rel_scale,
                          __global float* q_out, __global float* rel_out,
                          __global const uint* ctrl, uint global_layer, float q_clip, float r_clip,
                          uint log_floor, float alpha) {
  const uint h = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  if (r >= ctrl[CTRL_NACT]) return;
  const uint pos = ctrl[CTRL_POS];
  const size_t i = ((size_t)r * NH + h) * HD + d;
  const size_t j = ((size_t)r * NH + h) * REL + d;
  const float qx = q[i] * q_scale[h * HD + d];
  const float rx = d < REL ? relative[j] * rel_scale[h * REL + d] : 0.f;
  __local float square[HD];
  const float qs = norm_square(qx, d, square), rs = norm_square(rx, d, square);
  const float tau = global_layer ? 1.f + alpha * log(fmax((float)(pos + r + 1) / log_floor, 1.f)) : 1.f;
  q_out[i] = qx * fmin(1.f, q_clip / (sqrt(qs) + 1e-6f)) * tau;
  if (d < REL) rel_out[j] = rx * fmin(1.f, r_clip / (sqrt(rs) + 1e-6f)) * tau;
}

// One head and row per work-group. With projected taps present, the static
// weights are [lag][channel] and the dynamic row is [head][lag]. The static
// local form stores chronological taps as [channel][tap].
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void ink_kv_prep(__global const float* input, __global const float* history,
                          __global const float* static_taps, __global const float* projected,
                          __global ushort* cache, __global const uint* ctrl,
                          uint ring, uint key, float eps, float clip) {
  const uint h = get_group_id(0), r = get_group_id(1), d = get_local_id(0), i = h * HD + d;
  if (r >= ctrl[CTRL_NACT]) return;
  const uint pos = ctrl[CTRL_POS];
  float sum = 0.f;
  for (uint a = 0; a < 4; ++a) {
    const int t = projected ? (int)r - (int)a : (int)r + (int)a - 3;
    const float x = t < 0 ? history[(t + 3) * KW + i] : input[t * KW + i];
    const float w = projected ? static_taps[a * KW + i] + projected[(size_t)r * 64 + h * 4 + a]
                               : static_taps[i * 4 + a];
    sum += x * w;
  }
  const float y = (input[(size_t)r * KW + i] + sum) / (projected ? 3.f : sqrt(5.f));
  __local float square[HD];
  const float ss = norm_square(y, d, square);
  const float scale = key ? fmin(1.f, clip / (sqrt(ss) + 1e-6f)) : 1.f / sqrt(ss / HD + eps);
  const uint slot = ring ? (pos + r) & (ring - 1) : pos + r;
  cache[(size_t)slot * KW + i] = rne_bf16(y * scale);
}
#endif
