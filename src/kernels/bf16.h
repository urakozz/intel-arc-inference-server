// bf16.h - the bf16 <-> fp32 pair the kernels share (cmake/ocloc.cmake puts src/kernels on every kernel's include
// path). fp32 -> bf16 rounds to nearest even: the same add-and-shift as common::f32_to_bf16, NaN not expected and not
// handled.
#ifndef B70_BF16_H
#define B70_BF16_H

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

#endif
