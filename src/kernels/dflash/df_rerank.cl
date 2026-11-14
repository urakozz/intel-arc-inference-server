// Spec 19v2 C3: gather 64 original int8 head rows and rerank their exact dots to the selector's top16.
// Four contiguous K slices per candidate, merged ascending. The original head stays shared with verify.
#pragma OPENCL FP_CONTRACT OFF
#ifndef C
#define C 64   /* the candidates a row (spec 19 v3d Task 7: 80 for the 2-bit head) */
#endif
#define Q 4
#define WG (C * Q)
#define NO_IDX 0x7FFFFFFFu
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_rerank(__global const char* restrict w, __global const float* restrict scales,
                         __global const ushort* restrict x, __global const uint* restrict ids,
                         const float mult, const float cap, __global uint* restrict cand,
                         __global float* restrict unary) {
  const uint lid = get_local_id(0), r = get_group_id(0), c = lid / Q, q = lid % Q;
  const uint id = ids[(size_t)r * C + c];
  __local float partial[WG], scores[C];
  __local uint tokens[C];
  float a = 0;
  if (id < USED) {
    __global const char* col = w + (size_t)(id / 16) * (K / 16) * 256 + id % 16;
    for (uint k = q * (K / Q); k < (q + 1) * (K / Q); ++k)
      a = a + (float)col[(size_t)(k / 16) * 256 + (k % 16) * 16] * bf16f(x[(size_t)r * K + k]);
  }
  partial[lid] = a;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < C) {
    const uint token = ids[(size_t)r * C + lid];
    float sum = 0;
    for (uint s = 0; s < Q; ++s) sum = sum + partial[lid * Q + s];
    tokens[lid] = token < USED ? token : NO_IDX;
    scores[lid] = token < USED ? sum * scales[token] : -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0)
    for (uint j = 0; j < 16; ++j) {
      uint best = NO_IDX, token = NO_IDX;
      float value = -INFINITY;
      for (uint c = 0; c < C; ++c)
        if (tokens[c] != NO_IDX && (scores[c] > value || (scores[c] == value && tokens[c] < token))) {
          best = c; token = tokens[c]; value = scores[c];
        }
      cand[(size_t)r * 16 + j] = token;
      value = value * mult;
      if (cap > 0) value = tanh(value / cap) * cap;
      unary[(size_t)r * 16 + j] = value;
      if (best != NO_IDX) { tokens[best] = NO_IDX; scores[best] = -INFINITY; }
    }
}
