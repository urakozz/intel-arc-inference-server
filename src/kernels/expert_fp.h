#ifndef B70_EXPERT_FP_H
#define B70_EXPERT_FP_H

#ifndef WFMT
#define WFMT 0
#endif

#if WFMT == 0
#define TILE_U32 136
#define EXPERT_OUTER_WORDS 0
#elif WFMT == 1
#define FP_CODE_BITS 4
#define FP_BLOCK_K 16
#define TILE_U32 144
#define EXPERT_OUTER_WORDS 1
#elif WFMT == 2
#define FP_CODE_BITS 8
#define FP_BLOCK_K 32
#define TILE_U32 264
#define EXPERT_OUTER_WORDS 0
#else
#error "WFMT must select int4, NVFP4 or MXFP8"
#endif

#if WFMT != 0
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FP_VALUES_PER_WORD (32 / FP_CODE_BITS)

inline float fp_e4m3(uint code) {
  return convert_float(as_half((ushort)(((code & 128u) << 8) | ((code & 127u) << 7)))) * 0x1p8f;
}

inline float fp_tile_scale(const __global uint* tile, uint lane, uint k) {
  const uint s = (tile[32 * FP_CODE_BITS + (k / FP_BLOCK_K) * 4 + lane / 4] >> (8 * (lane % 4))) & 255u;
#if WFMT == 1
  return fp_e4m3(s);
#else
  return as_float(s ? s << 23 : 0x00400000u);  // 2^(s - 127), including s == 0
#endif
}

inline float fp_weight(uint word, uint i, float scale) {
  const uint code = (word >> (i * FP_CODE_BITS)) & ((1u << FP_CODE_BITS) - 1);
#if WFMT == 1
  const ushort h = (ushort)(((code & 8u) << 12) | ((code & 7u) << 9));
  return (convert_float(as_half(h)) * 0x1p14f) * scale;
#else
  return fp_e4m3(code) * scale;
#endif
}

#define FP_TILE_WORD(tile, lane, k) ((tile)[((k) / FP_VALUES_PER_WORD) * 16 + (lane)])
#endif

#endif
