// pf_gemv_bf16.cl -- `pf_ab_proj`, the GDN a||b projection at runtime `M`.
// bf16 weights x bf16 activations, `out` fp32 `[M][N]`. Grid
// (N / COLS_PER_WG, ceil(M / MT)), work-group COLS_PER_WG * KSPLIT lanes.
//
// `src/kernels/gemv_bf16.cl` transcribed with exactly these edits:
//
//   1. `#ifndef M / #define M 1` is gone. `MT` (8) is the **register tile** --
//      `float acc[MT]`, the same register array gemv_bf16's `acc[M]` was
//      designed for -- and the grid's second dimension is `ceil(M / MT)` with
//      the tail tile masking itself on the device (interfaces.md). `m_count`
//      is the last kernel argument.
//   2. Every `m` loop becomes `for (int i = 0; i < MT; ++i)` over
//      `m = m0 + i`, and the activation load reads row `m0` when the slot is
//      dead (`mr = live ? m : m0`) so it is never out of bounds and the dead
//      lanes' arithmetic is simply discarded.
//   3. **The writeback is the only place the mask bites**:
//      `if (m < m_count) out[(size_t)m * N + n] = ...`. That is
//      gemv_bf16.cl's `out[(size_t)m * N + n]` unchanged -- this kernel writes
//      one fp32 rectangle, not decode's `[S][M][N]`, which is legal because the
//      prefill path is S = 1 (plan 6b ruling R1).
//
// **The tree order is untouched**: `stride = KSPLIT/2 ... 1`, `q < stride`,
// barrier after every step, and the per-slot trees are independent, so slot 0
// at `m_count = 1` reduces exactly the terms decode's `M = 1` build reduces in
// exactly that order. That is what buys `tests/prefill/pf_gemv_test.cc`'s
// `memcmp` against `gemv_bf16_M1_K5120_N128_C16_S16`, the binary
// `src/runtime/capture.cc:436-452` actually binds.
//
// The a||b tiling `{COLS_PER_WG 16, KSPLIT 16}` is `kernels::gemv_bf16_tiling`'s
// choice for N = 128 and is measured, not chosen (docs/15 SS L2: ksplit
// 1 / 4 / 16 read 48.774 / 13.115 / 5.340 us per launch at M = 1). Whether it
// is still the right tiling at M = C is a question for the measurement, not for
// this file: at M = 2048 the grid is 8 x 256 work-groups rather than 8, so the
// thread scarcity KSPLIT was bought to fix is gone.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef COLS_PER_WG
#define COLS_PER_WG 64
#endif
#ifndef KSPLIT
#define KSPLIT 1
#endif
#define MT 8              /* rows per work-group: the register tile */
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
__kernel void pf_ab_proj(__global const ushort* restrict w,
                         __global const ushort* restrict x,
                         __global float* restrict out, uint m_count) {
#if KSPLIT > 1
  // [subgroup][row slot][lane]: one fp32 partial per column per K slice. At the
  // a||b tiling that is 16 subgroups x MT 8 x 16 lanes x 4 B = **8192 B** per
  // work-group (the decode build's note says 1024 B at M = 1).
  __local float red[SG_PER_WG][MT][SG];
#endif
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile_idx = sg / KSPLIT;  // which 16-column tile of this work-group
  const uint q = sg % KSPLIT;         // which K slice of that tile
  const uint n_tile = get_group_id(0) * TILES_PER_WG + tile_idx;
  const uint n = n_tile * SG + lane;
  const uint m0 = get_group_id(1) * MT;
  const uint k8_0 = q * K8_PER_SG;
  __global const ushort* tile = w + (size_t)n_tile * K8 * 128 + (size_t)k8_0 * 128;

  float acc[MT];
  for (int i = 0; i < MT; ++i) acc[i] = 0.f;

  for (uint k8 = k8_0; k8 < k8_0 + K8_PER_SG; ++k8, tile += 128) {
    ushort8 wv = intel_sub_group_block_read_us8(tile);
    for (int i = 0; i < MT; ++i) {
      const uint m = m0 + (uint)i;
      const uint mr = m < m_count ? m : m0;   // never out of bounds; discarded below
      ushort8 xv = vload8(0, x + (size_t)mr * K + k8 * 8);
      float a = acc[i];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[i] = a;
    }
  }
#if KSPLIT == 1
  for (int i = 0; i < MT; ++i) {
    const uint m = m0 + (uint)i;
    if (m < m_count) out[(size_t)m * N + n] = acc[i];
  }
#else
  for (int i = 0; i < MT; ++i) red[sg][i][lane] = acc[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  // The fixed tree. `q < stride` keeps `sg + stride` inside this tile's own
  // KSPLIT slices, so the tiles of a wide work-group never touch each other.
  // The dead row slots are reduced too -- their garbage is never read back.
  for (uint stride = KSPLIT / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (int i = 0; i < MT; ++i) red[sg][i][lane] += red[sg + stride][i][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (int i = 0; i < MT; ++i) {
      const uint m = m0 + (uint)i;
      if (m < m_count) out[(size_t)m * N + n] = red[sg][i][lane];
    }
#endif
}
