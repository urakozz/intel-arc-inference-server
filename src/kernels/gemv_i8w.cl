// gemv_i8w.cl - int8 weights x bf16 activations, M in [1,8]: the int8 `lm_head`
// (spec 9 §3, W8A16, N = 248320, 1.272 GB read per token against bf16's 2.543).
// `gemv_bf16.cl`'s lane-per-n structure at its lm_head tiling (`{64, 1}`: four
// 16-column tiles per work-group, no K split), with two changes:
//
//   - **The weight is int8 in `[n_tile][k16][16 k][16 n]`** (loader/lm_head_int8.h),
//     so one `intel_sub_group_block_read_uc16` hands lane `n % 16` the 16 k of its
//     column: 256 contiguous bytes per subgroup per step, as `gemv_bf16`'s us8 read.
//   - **One fp32 scale per row (per output column n)**, read once per lane after the
//     column's dot product and applied to the fp32 sum - `(x . q) * s`, plan 9a's
//     formula, not a per-element dequant.
//
// fp32 accumulation, one ascending chain over k: no split, no atomic, no
// data-dependent branch - a replayed list gives the same bits.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 64
#endif
#define SG 16
#define TILES_PER_WG (COLS_PER_WG / SG)
#define WG_N (SG * TILES_PER_WG)
#define K16 (K / 16)
#if (COLS_PER_WG % SG) != 0 || COLS_PER_WG < SG
#error "COLS_PER_WG must be a positive multiple of the SIMD16 subgroup width"
#endif
#if (N % COLS_PER_WG) != 0
#error "N must be a multiple of COLS_PER_WG: the grid is N / COLS_PER_WG"
#endif
#if (K % 16) != 0
#error "K must be a multiple of 16"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv_i8w(__global const uchar* restrict w,
                       __global const float* restrict scales,
                       __global const ushort* restrict x,
                       __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * TILES_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  __global const uchar* tile = w + (size_t)n_tile * K16 * 256;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint k16 = 0; k16 < K16; ++k16, tile += 256) {
    const float16 wf = convert_float16(as_char16(intel_sub_group_block_read_uc16(tile)));
    for (int m = 0; m < M; ++m) {
      const ushort16 xv = vload16(0, x + (size_t)m * K + k16 * 16);
      float a = acc[m];
      a += wf.s0 * bf16f(xv.s0); a += wf.s1 * bf16f(xv.s1);
      a += wf.s2 * bf16f(xv.s2); a += wf.s3 * bf16f(xv.s3);
      a += wf.s4 * bf16f(xv.s4); a += wf.s5 * bf16f(xv.s5);
      a += wf.s6 * bf16f(xv.s6); a += wf.s7 * bf16f(xv.s7);
      a += wf.s8 * bf16f(xv.s8); a += wf.s9 * bf16f(xv.s9);
      a += wf.sa * bf16f(xv.sa); a += wf.sb * bf16f(xv.sb);
      a += wf.sc * bf16f(xv.sc); a += wf.sd * bf16f(xv.sd);
      a += wf.se * bf16f(xv.se); a += wf.sf * bf16f(xv.sf);
      acc[m] = a;
    }
  }
  const float s = scales[n];
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m] * s;
}
