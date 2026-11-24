// Shared layout-1 floating-point expert path for the MoE decode kernels.
inline float tile_dot(const __global uint* restrict tile, const __global ushort* restrict xr,
                      uint g0, uint g1, uint lane) {
  float acc = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
#pragma unroll
    for (uint b = 0; b < 64 / FP_BLOCK_K; ++b) {
      const float scale = fp_tile_scale(tile, lane, b * FP_BLOCK_K);
#pragma unroll
      for (uint k = 0; k < FP_BLOCK_K; k += FP_VALUES_PER_WORD) {
        const uint word = FP_TILE_WORD(tile, lane, b * FP_BLOCK_K + k);
#pragma unroll
        for (uint i = 0; i < FP_VALUES_PER_WORD; ++i)
          acc += fp_weight(word, i, scale) * bf16f(xr[g * 64 + b * FP_BLOCK_K + k + i]);
      }
    }
  }
  return acc;
}

#if MOE_UNION
// Read and decode each expert tile once, then update the rows using that
// expert. The accumulator order within a row is the scalar tile_dot order.
inline void tile_dot_rows(const __global uint* tile, const __global ushort* x, uint slots,
                          uint row_stride, uint slot_stride, uint g0, uint g1, uint lane, float* acc) {
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
#pragma unroll
    for (uint b = 0; b < 64 / FP_BLOCK_K; ++b) {
      const float scale = fp_tile_scale(tile, lane, b * FP_BLOCK_K);
#pragma unroll
      for (uint k = 0; k < FP_BLOCK_K; k += FP_VALUES_PER_WORD) {
        const uint word = FP_TILE_WORD(tile, lane, b * FP_BLOCK_K + k);
#pragma unroll
        for (uint i = 0; i < FP_VALUES_PER_WORD; ++i) {
          const float w = fp_weight(word, i, scale);
          const uint column = g * 64 + b * FP_BLOCK_K + k + i;
#pragma unroll
          for (uint row = 0; row < M; ++row) {
            const uint slot = (slots >> (4 * row)) & 15u;
            if (slot < MOE_K) acc[row] += w * bf16f(x[(size_t)row * row_stride + slot * slot_stride + column]);
          }
        }
      }
    }
  }
}
#endif
