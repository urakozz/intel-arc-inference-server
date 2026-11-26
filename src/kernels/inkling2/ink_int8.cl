// ink_int8.cl - plan 41e: Inkling's int8 prompt linears (`--prefill-backend l0-int8`), PERF-58's h8 form at a 256-wide
// rotation (hidden 2304 is nine 256 blocks, not whole 1024 ones). R_K = D_K blockdiag(H_256) / 16 over K, orthonormal,
// D_K the signs of ink_sign below; x R_K . W R_K = x . W, so the GEMM (pf_int8.cl's pf_gemm_i8) multiplies int8
// rotated rows by int8 rotated columns with the scales in its epilogue.
//
//   ink_quant_rot(x, xq, xs)          grid (rows), WG 16 K / 256   bf16 x [rows][K] -> int8 (x R_K) [rows][K], per-row
//                                                                 scale xs = max |x R_K| / 127 (pf_quant_had's rule)
//   ink_rot_weight(w, sc, w8, ws, N)   grid (N), WG 16             load time: the int8 weight (tile layout
//                                                                 [N / 16][K][16], its row scale sc[n]) -> int8
//                                                                 (q R_K) per column at max / 127, pf_gemm_i8's
//                                                                 w8 [K / 4][N] words, ws[n] = sc[n] x max / 127
//
// The block walk: lane l holds block element 16 j + l (j = 0 .. 15); index bits 0-3 are lane shuffles, 4-7 register
// butterflies. The weight's FWHT is exact (integers, then a power of two); the activation's rounds in fp32.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#include "bf16.h"

#ifndef K
#error "ink_int8: K (a multiple of 256) must be defined"
#endif
#if K % 256
#error "ink_int8: K is whole 256 blocks"
#endif
#define SG 16
#define NBLK (K / 256)

// D_K: a fixed pseudo-random sign a k (bit 15 of a Knuth multiplicative hash).
inline float ink_sign(uint k) { return ((k * 2654435761u) >> 15) & 1u ? -1.0f : 1.0f; }

inline void ink_fwht256(float* v, uint l) {
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1)
#pragma unroll
    for (uint j = 0; j < 16u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? o - v[j] : v[j] + o;
    }
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1)
#pragma unroll
    for (uint j = 0; j < 16u; ++j)
      if ((j & h) == 0u) {
        const float a = v[j], c = v[j + h];
        v[j] = a + c;
        v[j + h] = a - c;
      }
#pragma unroll
  for (uint j = 0; j < 16u; ++j) v[j] *= (1.0f / 16.0f);
}

__attribute__((reqd_work_group_size(SG * NBLK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ink_quant_rot(__global const ushort* restrict x, __global char* restrict xq, __global float* restrict xs) {
  const uint m = get_group_id(0), b = get_sub_group_id(), l = get_sub_group_local_id();
  __local float red[NBLK];
  const size_t base = (size_t)m * K + b * 256u;
  float v[16];
#pragma unroll
  for (uint j = 0; j < 16u; ++j) v[j] = bf16f(x[base + 16u * j + l]) * ink_sign(b * 256u + 16u * j + l);
  ink_fwht256(v, l);
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 16u; ++j) amax = fmax(amax, fabs(v[j]));
  amax = sub_group_reduce_max(amax);
  if (l == 0u) red[b] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax = 0.0f;
  for (uint i = 0; i < (uint)NBLK; ++i) amax = fmax(amax, red[i]);
  const float scale = amax / 127.0f;
  const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
  if (b == 0u && l == 0u) xs[m] = scale;
#pragma unroll
  for (uint j = 0; j < 16u; ++j) xq[base + 16u * j + l] = convert_char_sat_rte(v[j] * iv);
}

__attribute__((reqd_work_group_size(SG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ink_rot_weight(__global const char* restrict w, __global const float* restrict sc,
                             __global uchar* restrict w8, __global float* restrict ws, uint N) {
  const uint n = get_group_id(0), l = get_sub_group_local_id();
  __global const char* col = w + (size_t)(n / 16u) * K * 16u + n % 16u;
  float amax = 0.0f;
  for (uint b = 0; b < (uint)NBLK; ++b) {
    float v[16];
#pragma unroll
    for (uint j = 0; j < 16u; ++j) {
      const uint k = b * 256u + 16u * j + l;
      v[j] = (float)col[(size_t)k * 16u] * ink_sign(k);
    }
    ink_fwht256(v, l);
#pragma unroll
    for (uint j = 0; j < 16u; ++j) amax = fmax(amax, fabs(v[j]));
  }
  amax = sub_group_reduce_max(amax);
  const float iv = amax == 0.0f ? 0.0f : 127.0f / amax;
  if (l == 0u) ws[n] = sc[n] * (amax == 0.0f ? 1.0f : amax / 127.0f);
  for (uint b = 0; b < (uint)NBLK; ++b) {
    float v[16];
#pragma unroll
    for (uint j = 0; j < 16u; ++j) {
      const uint k = b * 256u + 16u * j + l;
      v[j] = (float)col[(size_t)k * 16u] * ink_sign(k);
    }
    ink_fwht256(v, l);
#pragma unroll
    for (uint j = 0; j < 16u; ++j) {
      const uint k = b * 256u + 16u * j + l;
      w8[((size_t)(k / 4u) * N + n) * 4u + k % 4u] = (uchar)convert_char_sat_rte(v[j] * iv);
    }
  }
}
