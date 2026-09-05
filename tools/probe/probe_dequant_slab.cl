// pf_dequant_slab: PROBE-ONLY. One N-slab of an int4 linear, expanded into a
// COMPACT [K][NS] bf16 slab small enough to sit in the 24 MB L2.
//
// This is `src/kernels/prefill/dequant.cl` at LAYOUT 0 / TRANSPOSED 0, with
// exactly two changes and no third:
//   1. the source column is `n0 + n_local` instead of `n_local`, so a slab can
//      be taken from anywhere along N; `n0` is a RUNTIME argument, as `M` is in
//      every other prefill kernel;
//   2. the output pitch is the compile-time `NS`, not `N`, so the destination
//      is a packed [K][NS] slab rather than a column window of a [K][N] matrix.
// The dequant arithmetic below is a literal transcription of the production
// kernel's -- same `^ 0x88888888`, same sign-extension shift, same single RNE
// rounding -- so a bitwise comparison against the production kernel's output is
// a real check and not a re-implementation being compared with itself.
//
// No runtime path binds this kernel and `src/kernels/prefill/dequant.cl` is not
// touched by it. At NS == N and n0 == 0 it IS that kernel, which is what makes
// the probe's equivalence control a control.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define SG 16
#define GROUP 64

inline ushort rne_bf16(float f) {
  const uint u = as_uint(f);
  const uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1)))
__kernel void pf_dequant_slab(__global const uint* restrict w,
                              __global const half* restrict scales,
                              __global ushort* restrict out,
                              const uint n0) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0);
  const uint g = get_group_id(1);
  const uint n_local = n_tile * SG + lane;  // column within the slab, < NS
  const uint n = n0 + n_local;              // column within the [K][N] source
  uint wv[8];
  __global const uint* wp = w + (size_t)(g * 8) * N + n;
  for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
  const float scale = (float)scales[(size_t)g * N + n];
  for (int j = 0; j < 8; ++j) {
    const uint u = wv[j] ^ 0x88888888u;
    for (int i = 0; i < 8; ++i) {
      const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;
      const uint k = g * GROUP + (uint)j * 8u + (uint)i;
      out[(size_t)k * NS + n_local] = rne_bf16((float)qm8 * scale);
    }
  }
}
