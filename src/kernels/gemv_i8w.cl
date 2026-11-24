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
#if QKV_NKV && QKV_NKV % WG_N != 0
#error "gemv_i8w: Q/K/V outputs must have whole work-group widths"
#endif
#if (K % 16) != 0
#error "K must be a multiple of 16"
#endif
// PERF-111, both off by default (probe_gemv --tiles' cells). GEMV_KMAJOR 1: the 256 B steps ordered [k16][n_tile]
// instead of [n_tile][k16] (common::tiles_k_major at load), so at each step the launch's sub-groups read
// neighbouring bytes rather than each walking its own 80 KB slab. GEMV_AHEAD A: A steps' block reads issued before
// the first is used. Neither changes the arithmetic or its order: bitwise the default build.
#ifndef GEMV_KMAJOR
#define GEMV_KMAJOR 0
#endif
#ifndef GEMV_AHEAD
#define GEMV_AHEAD 1
#endif
#if (K16 % GEMV_AHEAD) != 0
#error "GEMV_AHEAD must divide K / 16"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv_i8w(__global const uchar* restrict w,
                       __global const float* restrict scales,
                       __global const ushort* restrict x,
                       __global float* restrict out
#if QKV_NKV
                       , __global const uchar* wk, __global const float* sk, __global const ushort* xk, __global float* ok
                       , __global const uchar* wv, __global const float* sv, __global const ushort* xv, __global float* ov
#if INK_ATTN_PROJECTIONS
                       , __global const uchar* wg, __global const float* sgate, __global float* og
                       , __global const uchar* wr, __global const float* sr, __global float* orr
                       , __global const uchar* ws, __global const float* ss, __global float* os
#endif
#endif
                       ) {
#if QKV_NKV
  uint group = get_group_id(0);
  uint columns = N;
#if INK_ATTN_PROJECTIONS
  const uint block = WG_N;
  const uint qkv_groups = (N + 2 * QKV_NKV) / block;
  if (group >= qkv_groups) {
    group -= qkv_groups;
    if (group < N / block) {
      w = wg; scales = sgate; out = og;
    } else {
      group -= N / block;
      if (group < 384 / block) {
        columns = 384;
        w = wr; scales = sr; out = orr;
      } else {
        group -= 384 / block;
        columns = 64;
        w = ws; scales = ss; out = os;
      }
    }
  } else
#endif
  if (group >= N / WG_N) {
    group -= N / WG_N;
    if (group < QKV_NKV / WG_N) {
      w = wk; scales = sk; x = xk; out = ok;
    } else {
      group -= QKV_NKV / WG_N;
      w = wv; scales = sv; x = xv; out = ov;
    }
    columns = QKV_NKV;
  }
#endif
  const uint lane = get_sub_group_local_id();
#if QKV_NKV
  const uint n_tile = group * TILES_PER_WG + get_sub_group_id();
#else
  const uint n_tile = get_group_id(0) * TILES_PER_WG + get_sub_group_id();
#endif
  const uint n = n_tile * SG + lane;
  __global const uchar* tile = w + (size_t)n_tile * K16 * 256;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

#define I8W_STEP(wf, k16)                                         \
  for (int m = 0; m < M; ++m) {                                   \
    const ushort16 xv = vload16(0, x + (size_t)m * K + k16 * 16); \
    float a = acc[m];                                             \
    a += wf.s0 * bf16f(xv.s0); a += wf.s1 * bf16f(xv.s1);         \
    a += wf.s2 * bf16f(xv.s2); a += wf.s3 * bf16f(xv.s3);         \
    a += wf.s4 * bf16f(xv.s4); a += wf.s5 * bf16f(xv.s5);         \
    a += wf.s6 * bf16f(xv.s6); a += wf.s7 * bf16f(xv.s7);         \
    a += wf.s8 * bf16f(xv.s8); a += wf.s9 * bf16f(xv.s9);         \
    a += wf.sa * bf16f(xv.sa); a += wf.sb * bf16f(xv.sb);         \
    a += wf.sc * bf16f(xv.sc); a += wf.sd * bf16f(xv.sd);         \
    a += wf.se * bf16f(xv.se); a += wf.sf * bf16f(xv.sf);         \
    acc[m] = a;                                                   \
  }
#if GEMV_KMAJOR || GEMV_AHEAD > 1
#if GEMV_KMAJOR
  tile = w + (size_t)n_tile * 256;
#if QKV_NKV
  const size_t step = (size_t)(columns / SG) * 256;
#else
  const size_t step = (size_t)(N / SG) * 256;
#endif
#else
  const size_t step = 256;
#endif
  for (uint k0 = 0; k0 < K16; k0 += GEMV_AHEAD) {
    uchar16 wq[GEMV_AHEAD];
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < GEMV_AHEAD; ++j)
      wq[j] = intel_sub_group_block_read_uc16(tile + j * step);
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < GEMV_AHEAD; ++j) {
      const float16 wf = convert_float16(as_char16(wq[j]));
      I8W_STEP(wf, (k0 + j))
    }
    tile += GEMV_AHEAD * step;
  }
#else
  for (uint k16 = 0; k16 < K16; ++k16, tile += 256) {
    const float16 wf = convert_float16(as_char16(intel_sub_group_block_read_uc16(tile)));
    I8W_STEP(wf, k16)
  }
#endif
  const float s = scales[n];
#if QKV_NKV
  for (int m = 0; m < M; ++m) out[(size_t)m * columns + n] = acc[m] * s;
#else
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m] * s;
#endif
}
