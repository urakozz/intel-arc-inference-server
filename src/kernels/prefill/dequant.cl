// pf_dequant_tile: one int4 linear, expanded into a reusable bf16 scratch.
// The int4 contract is gemv.cl's GEMV_DEQ_SHIFT path: symmetric group-64,
// f16 scale widened to fp32, then exactly one RNE bf16 rounding.
//
// Compile-time: K, N, LAYOUT (0 native qweight + scales, 1 tiled inline
// scales), TRANSPOSED (0 [K][N], 1 [N][K]). One 16-lane subgroup owns one
// (N tile, K group), so each lane dequantises 64 values of one column.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define SG 16
#define GROUP 64
#define G (K / GROUP)
#define TILE_U32 136  // 128 u32 nibbles + 8 u32 packed f16 scales

// Literal transcription of common::f32_to_bf16. The host and device therefore
// calculate and round one identical fp32 expression for every weight.
inline ushort rne_bf16(float f) {
  const uint u = as_uint(f);
  const uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1)))
__kernel void pf_dequant_tile(__global const uint* restrict w,
                              __global const half* restrict scales,
                              __global ushort* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0);
  const uint g = get_group_id(1);
  const uint n = n_tile * SG + lane;
  uint wv[8];
  float scale;
#if LAYOUT == 0
  __global const uint* wp = w + (size_t)(g * 8) * N + n;
  for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
  scale = (float)scales[(size_t)g * N + n];
#else
  __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
  const uint8 block = intel_sub_group_block_read8(tile);
  wv[0] = block.s0; wv[1] = block.s1; wv[2] = block.s2; wv[3] = block.s3;
  wv[4] = block.s4; wv[5] = block.s5; wv[6] = block.s6; wv[7] = block.s7;
  const ushort scale_bits =
      intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
  scale = (float)as_half(scale_bits);
#endif
  for (int j = 0; j < 8; ++j) {
    const uint u = wv[j] ^ 0x88888888u;
    for (int i = 0; i < 8; ++i) {
      const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;
      const uint k = g * GROUP + (uint)j * 8u + (uint)i;
      const ushort value = rne_bf16((float)qm8 * scale);
#if TRANSPOSED
      out[(size_t)n * K + k] = value;  // [N][K], lane-contiguous stores
#else
      out[(size_t)k * N + n] = value;  // [K][N], native prefill GEMM input
#endif
    }
  }
}
