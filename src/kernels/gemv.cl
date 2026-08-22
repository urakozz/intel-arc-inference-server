// gemv.cl - int4 (group 64, symmetric) weights x bf16 activations, M in [1,8].
// Spec 1 §9.2. Weight-stationary: a subgroup owns 16 consecutive n and streams
// its K-range once; the M activation rows are re-read per k-group from L1/L2
// (16 lanes hit the same line: a broadcast load, no shuffles). Split-K: grid
// dim 1 is the slice s; slices write separate fp32 partials and the consumer
// (prep) sums them, so no atomics and bitwise-identical replays.
//
// Compile-time: M (tokens), K, N, S (split-K slices, S | K/64), LAYOUT (0 GPTQ
// native: w[K/8][N] u32 + scales[K/64][N] f16; 1 tiled: per (n_tile, g) 512 B
// nibbles [k_octet][n] then 32 B of f16 scales, contiguous along K).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define TILE_U32 136   /* 128 u32 of nibbles + 8 u32 of scales = 544 B */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// 8 nibbles of one u32 word times 8 consecutive activations.
inline float dot8(uint word, ushort8 xv) {
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv(__global const uint* restrict w,
                   __global const half* restrict scales,
                   __global const ushort* restrict x,
                   __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  const uint s = get_group_id(1);
  const uint g0 = s * G_PER_S;
  const uint g1 = g0 + G_PER_S;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint g = g0; g < g1; ++g) {
    uint wv[8];
    float scale;
#if LAYOUT == 0
    __global const uint* wp = w + (size_t)(g * 8) * N + n;
    for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
    scale = (float)scales[(size_t)g * N + n];
#else
    __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
    uint8 blk = intel_sub_group_block_read8(tile);
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
    ushort sh = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
    scale = (float)as_half(sh);
#endif
    float gacc[M];
    for (int m = 0; m < M; ++m) gacc[m] = 0.f;
    for (int j = 0; j < 8; ++j) {
      for (int m = 0; m < M; ++m) {
        ushort8 xv = vload8(0, x + (size_t)m * K + g * GROUP + j * 8);
        gacc[m] += dot8(wv[j], xv);
      }
    }
    for (int m = 0; m < M; ++m) acc[m] += gacc[m] * scale;
  }

  for (int m = 0; m < M; ++m) out[((size_t)s * M + m) * N + n] = acc[m];
}
