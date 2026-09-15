// pf_dequant_slab - one NS = 1024-column bf16 slab of an int4 linear (spec 2.1 §3.3), the
// slab probe's kernel (tools/probe/probe_dequant_slab.cl, measured bitwise against
// pf_dequant_tile) with the layout-1 branch added. It is src/kernels/prefill/dequant.cl at
// TRANSPOSED 0 with two changes: the source column is n0 + n_local (n0 a runtime argument),
// and the output pitch is NS, so the destination is a packed [K][NS] slab. The dequant
// arithmetic is the production kernel's, statement for statement (same xor, same
// sign-extension shift, same single RNE), which is what makes the bitwise test a test.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef K
#error "pf_dequant_slab: K must be defined"
#endif
#ifndef N
#error "pf_dequant_slab: N must be defined"
#endif
#ifndef LAYOUT
#error "pf_dequant_slab: LAYOUT must be defined (0 GPTQ-native, 1 tiled)"
#endif
#define NS 1024
#define SG 16
#define GROUP 64
#define G (K / GROUP)
#define TILE_U32 136
#if N % NS != 0
#error "pf_dequant_slab: N must be a whole number of 1024-column slabs"
#endif

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
  const uint n_tile = get_group_id(0);          // 0 .. NS/16 - 1, within the slab
  const uint g = get_group_id(1);               // k-group
  const uint n_local = n_tile * SG + lane;
  const uint n = n0 + n_local;                  // column within the [K][N] source
  uint wv[8];
  float scale;
#if LAYOUT == 0
  __global const uint* wp = w + (size_t)(g * 8) * N + n;
  for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
  scale = (float)scales[(size_t)g * N + n];
#else
  const uint nt = n0 / SG + n_tile;             // the source's global 16-column tile
  __global const uint* tile = w + ((size_t)nt * G + g) * TILE_U32;
  const uint8 block = intel_sub_group_block_read8(tile);
  wv[0] = block.s0; wv[1] = block.s1; wv[2] = block.s2; wv[3] = block.s3;
  wv[4] = block.s4; wv[5] = block.s5; wv[6] = block.s6; wv[7] = block.s7;
  const ushort scale_bits = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
  scale = (float)as_half(scale_bits);
#endif
  for (int j = 0; j < 8; ++j) {
    const uint u = wv[j] ^ 0x88888888u;
    for (int i = 0; i < 8; ++i) {
      const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;
      const uint k = g * GROUP + (uint)j * 8u + (uint)i;
      out[(size_t)k * NS + n_local] = rne_bf16((float)qm8 * scale);
    }
  }
}
