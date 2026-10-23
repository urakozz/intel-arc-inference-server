// ll_gemv_bf16_g.cl - spec 26b: src/kernels/gemv_bf16.cl's tiles, grid and summation order with an fp32 OUTPUT SCALE per
// column - the LiLiCorr drafter's exact arm (spec 26 §3, Decision 2): an NVFP4 linear is W' = s_b x e (exact in bf16) and
// its global g, and y = g x (W' x) is the factorisation that IS exact - W' as gemv_bf16's bf16 tiles, g one fp32 multiply
// on the accumulator (an NVFP4 GEMM's alpha). `gs[n]` is column n's g (a concatenated gate||up carries each part's g over
// its rows). A family copy (23f's MG0 / D0 rule: gemv_bf16.cl is not edited): every line below the scale is gemv_bf16's,
// so out[m][n] == gemv_bf16's out[m][n] x gs[n] bitwise under one compiler. Twin: tests/kernels/lilicorr_ref.h GLin.
//
// COLS_PER_WG / KSPLIT as gemv_bf16 (kernels::gemv_bf16_tiling picks them by N; the KSPLIT merge is the same fixed
// pairwise tree). M in [1, 8].
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 64
#endif
#ifndef KSPLIT
#define KSPLIT 1
#endif
#define SG 16
#define TILES_PER_WG (COLS_PER_WG / SG)
#define SG_PER_WG (TILES_PER_WG * KSPLIT)
#define WG_N (SG * SG_PER_WG)
#define K8 (K / 8)
#define K8_PER_SG (K8 / KSPLIT)
#if (COLS_PER_WG % SG) != 0 || COLS_PER_WG < SG
#error "COLS_PER_WG must be a positive multiple of the SIMD16 subgroup width"
#endif
#if (N % COLS_PER_WG) != 0
#error "N must be a multiple of COLS_PER_WG: the grid is N / COLS_PER_WG"
#endif
#if (K8 % KSPLIT) != 0 || (KSPLIT & (KSPLIT - 1)) != 0
#error "KSPLIT must be a power of two dividing K/8 (the merge is a pairwise tree)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void ll_gemv_bf16_g(__global const ushort* restrict w,
                             __global const ushort* restrict x,
                             __global const float* restrict gs,
                             __global float* restrict out) {
#if KSPLIT > 1
  __local float red[SG_PER_WG][M][SG];
#endif
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile_idx = sg / KSPLIT;
  const uint q = sg % KSPLIT;
  const uint n_tile = get_group_id(0) * TILES_PER_WG + tile_idx;
  const uint n = n_tile * SG + lane;
  const uint k8_0 = q * K8_PER_SG;
  __global const ushort* tile = w + (size_t)n_tile * K8 * 128 + (size_t)k8_0 * 128;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint k8 = k8_0; k8 < k8_0 + K8_PER_SG; ++k8, tile += 128) {
    ushort8 wv = intel_sub_group_block_read_us8(tile);
    for (int m = 0; m < M; ++m) {
      ushort8 xv = vload8(0, x + (size_t)m * K + k8 * 8);
      float a = acc[m];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[m] = a;
    }
  }
  const float g = gs[n];
#if KSPLIT == 1
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m] * g;
#else
  for (int m = 0; m < M; ++m) red[sg][m][lane] = acc[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KSPLIT / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (int m = 0; m < M; ++m) red[sg][m][lane] += red[sg + stride][m][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = red[sg][m][lane] * g;
#endif
}
