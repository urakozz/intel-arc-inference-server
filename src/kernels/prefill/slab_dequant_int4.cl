// slab_dequant_int4.cl - one bf16 slab of an int4 GPTQ layout-0 linear for the prefill GEMM
// (pf_gemm.cl, unchanged), for widths that are not whole 1024-column slabs (Flash-Next's and
// Kolibri-1's o_proj at N 2560, Bolt's in_proj 18560, q||k||v 4608, fc1||shared_up 6400), which
// prefill/pf_dequant_slab.cl's NS 1024 refuses (`N % NS`). This is that kernel's layout-0
// branch with two changes and its arithmetic untouched (the same xor, sign-extension shift
// and single RNE of (q - 8) x scale):
//
//   1. the slab width `ns` is a runtime argument (the output pitch; the grid is ns / 16), so
//      a linear is walked in slabs of 1024 and ONE tail slab of pad256(N - n0) columns;
//   2. a column n >= N (the tail's padding up to the GEMM's 256-column tile) is written as
//      exact zeros, so the GEMM's padded output columns are exact zeros too and land in the
//      partials row's own padding (the walk's pitch is pad256(N)).
//
//   pf_int4_slab_dequant(w, scales, out, n0, ns)   grid (ns / 16, K / 64), WG 16
//     out[k][c] = bf16 of column n0 + c of the [K][N] weight, c < ns, k < K
//
// No sub-group function (the layout-0 reads are plain loads, the scale through vload_half).
#ifndef K
#error "pf_int4_slab_dequant: K must be defined"
#endif
#ifndef N
#error "pf_int4_slab_dequant: N must be defined"
#endif
#define SG 16
#define GROUP 64
#if K % GROUP != 0 || N % SG != 0
#error "pf_int4_slab_dequant: K must be whole 64-k groups, N whole 16-column tiles"
#endif

inline ushort rne_bf16(float f) {
  const uint u = as_uint(f);
  const uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((reqd_work_group_size(SG, 1, 1)))
__kernel void pf_int4_slab_dequant(__global const uint* restrict w,
                                 __global const half* restrict scales,
                                 __global ushort* restrict out, const uint n0, const uint ns) {
  const uint lane = get_local_id(0);
  const uint n_tile = get_group_id(0);          // 0 .. ns/16 - 1, within the slab
  const uint g = get_group_id(1);               // k-group
  const uint n_local = n_tile * SG + lane;
  const uint n = n0 + n_local;                  // column within the [K][N] source
  if (n >= N) {                                 // uniform per work-group (N % 16 == 0)
    for (uint k = g * GROUP; k < (g + 1) * GROUP; ++k) out[(size_t)k * ns + n_local] = (ushort)0;
    return;
  }
  uint wv[8];
  __global const uint* wp = w + (size_t)(g * 8) * N + n;
  for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
  const float scale = vload_half((size_t)g * N + n, scales);
  for (int j = 0; j < 8; ++j) {
    const uint u = wv[j] ^ 0x88888888u;
    for (int i = 0; i < 8; ++i) {
      const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;
      const uint k = g * GROUP + (uint)j * 8u + (uint)i;
      out[(size_t)k * ns + n_local] = rne_bf16((float)qm8 * scale);
    }
  }
}
