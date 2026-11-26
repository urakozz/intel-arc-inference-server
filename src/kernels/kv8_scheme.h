// kv8_scheme.h - the int8 KV cache's scale format and signs (spec 12's `rotkv`; src/common/kv8.h is the host twin,
// edit the three together): kv8.cl's Qwen3.8 kernels at head_dim 256 and PERF-102's KV8 builds at head_dim 128
// (kol_attn.cl, kol_pf_attn.cl, inkling2/ink_prep.cl), which use the first 128 signs.
#ifndef B70_KV8_SCHEME_H
#define B70_KV8_SCHEME_H
// The probe's signs (src/common/kv8.h kSignWords).
__constant uint KV8_SIGNS[8] = {0xA197D809u, 0xA6850592u, 0xB205A73Eu, 0x20AFA769u,
                                0x45C2CF64u, 0xBEAEAACBu, 0x75F39ADDu, 0x2385A70Cu};

// s[j] applied as the sign bit: exact, and the same bits as a multiply by -1.
inline float kv8_flip(float v, uint j) {
  return as_float(as_uint(v) ^ (((KV8_SIGNS[j >> 5] >> (j & 31u)) & 1u) << 31));
}

// binary16 <-> fp32 in integer arithmetic (kv8.h's f32_to_f16_rne / f16f, op for op): no
// cl_khr_fp16 needed, subnormals kept, >= 65504 saturates. A scale is >= 0 and finite.
inline ushort kv8_f16_rne(float f) {
  uint u = as_uint(f);
  const uint sign = (u >> 16) & 0x8000u;
  u &= 0x7FFFFFFFu;
  if (u >= 0x477FE000u) return (ushort)(sign | 0x7BFFu);
  if (u < 0x38800000u) return (ushort)(sign | (uint)rint(as_float(u) * 16777216.0f));
  const uint e = (u >> 23) - 112u;
  const uint m = u & 0x7FFFFFu;
  uint h = (e << 10) | (m >> 13);
  const uint rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
  return (ushort)(sign | h);
}
inline float kv8_f16f(ushort h) {
  const uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0u ? (float)m * 5.9604644775390625e-08f
                          : as_float(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}

#ifdef KV8_N
// PERF-102: the scheme at head_dim KV8_N (128), kv8.h's rotate_n / rotate_q_n / unrotate_n / quantise_n op for op, on
// a work-group of KV8_N work-items, work-item i owning element i of one row. Every work-item calls these (barriers);
// `buf` is KV8_N floats of SLM, free again when a call returns. K and V are rotated by 1/16 as at 256; q, and the
// attention output on its way back, by 16 / KV8_N - so q . k and the output are the bf16 cache's.
inline float kv8n_fwht(float v, uint i, __local float* buf) {
  for (uint h = 1u; h < (uint)KV8_N; h <<= 1) {
    buf[i] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float o = buf[i ^ h];
    v = (i & h) ? (o - v) : (v + o);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return v;
}
inline float kv8n_rotate(float x, uint i, __local float* buf) { return kv8_flip(kv8n_fwht(x, i, buf), i) * 0.0625f; }
inline float kv8n_rotate_q(float x, uint i, __local float* buf) {
  return kv8_flip(kv8n_fwht(x, i, buf), i) * (16.0f / KV8_N);
}
inline float kv8n_unrotate(float y, uint i, __local float* buf) {
  return kv8n_fwht(kv8_flip(y, i), i, buf) * (16.0f / KV8_N);
}
// One rotated row's element i into its int8 code, the row's fp16 scale written by work-item 0.
inline void kv8n_store(float y, uint i, __local float* buf, __global char* restrict row,
                       __global ushort* restrict scale) {
  buf[i] = fabs(y);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KV8_N / 2; stride > 0; stride >>= 1) {
    if (i < stride) buf[i] = fmax(buf[i], buf[i + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const ushort s16 = kv8_f16_rne(buf[0] / 127.0f);
  const float sf = kv8_f16f(s16);
  float r = sf == 0.0f ? 0.0f : rint(y / sf);
  r = fmin(fmax(r, -127.0f), 127.0f);
  row[i] = (char)(int)r;
  if (i == 0) *scale = s16;
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif
#endif
