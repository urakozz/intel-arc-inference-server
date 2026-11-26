// Shared layout-1 floating-point expert path for the MoE decode kernels. Each word's values are decoded first and
// the activations they meet read as one vector, then multiplied and summed in the scalar order (value by value):
// the same products and sums as reading both a value at a time.
#if FP_VALUES_PER_WORD == 8
#define FP_XV uint4
#define FP_XLOAD(p) as_uint4(vload8(0, p))
#else
#define FP_XV uint2
#define FP_XLOAD(p) as_uint2(vload4(0, p))
#endif
// fp_weight's values two codes at a time: codes i and i + FP_VALUES_PER_WORD / 2 sit 16 bits apart once the word is
// shifted, so one mask-and-shift builds both halves and one conversion widens them; the half's power-of-two factor
// is folded into the scale (code x 2^p x scale and code x (2^p scale) are both the exact product: <= 6 / 7
// significant bits).
inline void fp_word(uint word, float scale, float* w) {
#if WFMT == 1
  const float s = scale * 0x1p14f;
#pragma unroll
  for (uint i = 0; i < 4; ++i) {
    const uint t = word >> (4 * i);
    const float2 v = convert_float2(as_half2(((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9))) * s;
    w[i] = v.x;
    w[i + 4] = v.y;
  }
#else
  const float s = scale * 0x1p8f;
#pragma unroll
  for (uint i = 0; i < 2; ++i) {
    const uint t = word >> (8 * i);
    const float2 v = convert_float2(as_half2(((t & 0x00800080u) << 8) | ((t & 0x007F007Fu) << 7))) * s;
    w[i] = v.x;
    w[i + 2] = v.y;
  }
#endif
}
// The activations as bf16 pairs: the even one widened by a shift, the odd one by a mask (bf16f's values).
inline float fp_dot_word(const float* w, FP_XV xv, float acc) {
  uint xs[FP_VALUES_PER_WORD / 2];
  *(FP_XV*)xs = xv;
#pragma unroll
  for (uint i = 0; i < FP_VALUES_PER_WORD / 2; ++i) {
    acc += w[2 * i] * as_float(xs[i] << 16);
    acc += w[2 * i + 1] * as_float(xs[i] & 0xffff0000u);
  }
  return acc;
}

inline float tile_dot(const __global uint* restrict tile, const __global ushort* restrict xr,
                      uint g0, uint g1, uint lane) {
  float acc = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
#pragma unroll
    for (uint b = 0; b < 64 / FP_BLOCK_K; ++b) {
      const float scale = fp_tile_scale(tile, lane, b * FP_BLOCK_K);
#pragma unroll
      for (uint k = 0; k < FP_BLOCK_K; k += FP_VALUES_PER_WORD) {
        float w[FP_VALUES_PER_WORD];
        fp_word(FP_TILE_WORD(tile, lane, b * FP_BLOCK_K + k), scale, w);
        acc = fp_dot_word(w, FP_XLOAD(xr + g * 64 + b * FP_BLOCK_K + k), acc);
      }
    }
  }
  return acc;
}

#if MOE_UNION
// tile_dot over nr rows at once (q4_moe_union.cl's form): each word decoded once, then per row exactly tile_dot's
// chain against that row's activations at x + xo[r]. nr is a constant at each call (tile_dot_n), so the row loops
// unroll, the accumulators stay in registers and no row is computed masked.
inline void tile_dot_nr(const __global uint* restrict tile, const __global ushort* restrict x, const uint* xo,
                        const uint nr, uint g0, uint g1, uint lane, float* acc) {
#pragma unroll
  for (uint r = 0; r < nr; ++r) acc[r] = 0.f;
  for (uint g = g0; g < g1; ++g, tile += TILE_U32) {
#pragma unroll
    for (uint b = 0; b < 64 / FP_BLOCK_K; ++b) {
      const float scale = fp_tile_scale(tile, lane, b * FP_BLOCK_K);
#pragma unroll
      for (uint k = 0; k < FP_BLOCK_K; k += FP_VALUES_PER_WORD) {
        float w[FP_VALUES_PER_WORD];
        fp_word(FP_TILE_WORD(tile, lane, b * FP_BLOCK_K + k), scale, w);
        const uint column = g * 64 + b * FP_BLOCK_K + k;
#pragma unroll
        for (uint r = 0; r < nr; ++r) acc[r] = fp_dot_word(w, FP_XLOAD(x + xo[r] + column), acc[r]);
      }
    }
  }
}
// n <= M rows; uniform across the work-group.
inline void tile_dot_n(const __global uint* restrict tile, const __global ushort* restrict x, const uint* xo, uint n,
                       uint g0, uint g1, uint lane, float* acc) {
  if (n == 1) tile_dot_nr(tile, x, xo, 1, g0, g1, lane, acc);
  else if (n == 2) tile_dot_nr(tile, x, xo, 2, g0, g1, lane, acc);
#if M > 2
  else if (n == 3) tile_dot_nr(tile, x, xo, 3, g0, g1, lane, acc);
#endif
#if M > 3
  else if (n == 4) tile_dot_nr(tile, x, xo, 4, g0, g1, lane, acc);
#endif
#if M > 4
  else if (n == 5) tile_dot_nr(tile, x, xo, 5, g0, g1, lane, acc);
#endif
#if M > 5
  else if (n == 6) tile_dot_nr(tile, x, xo, 6, g0, g1, lane, acc);
#endif
#if M > 6
  else if (n == 7) tile_dot_nr(tile, x, xo, 7, g0, g1, lane, acc);
#endif
#if M > 7
  else tile_dot_nr(tile, x, xo, 8, g0, g1, lane, acc);
#endif
}
#endif
