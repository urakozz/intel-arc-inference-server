// head_cand.cl - PERF-123: the main model's head at a greedy verify step as a 2-bit candidate pass plus an exact
// rescore. After the 2-bit head (gemm_sm LAYOUT 3) and df_candidate's top C a row (`cand` [M][C], real ids only):
//
//   head_cand_gather  grid (M C K/16 / 256), WG 256 - the int8 head's columns of every row's candidates into a compact
//                     head of NC columns in the same tiled layout ([NC/16][K/16][16 k][16 n], column j = slot r C + c)
//                     and their fp32 scales; slots past M C are left as they are (their outputs are never read)
//   (gemm_sm_i8w at N = NC with the full head's knobs - a column's output does not depend on N, so every slot's score
//    is bitwise the full head's for its id)
//   head_cand_argmax  grid (M), WG 256 - row r's best of its C scores (larger, then the lower id: argmax.cl's rule) into
//                     argmax_part[r][0], the other 242 pairs neutral, so argmax_stage2 picks exactly it
//
// The full head's argmax over the vocabulary equals this one whenever its winner is among the row's candidates
// (measured: a 2-bit rank of at most 30 over 5,816 verify rows of the golden and replay sets, C = 64).
// GATE 2 (head_gate.h): both run on greedy steps only.
#include "head_gate.h"
#ifndef K
#define K 5120
#endif
#ifndef C
#define C 64
#endif
#ifndef NC
#define NC 512             /* the rescore's columns: 8 rows x C */
#endif
#define GROUPS 243         /* argmax.cl's stage-1 groups (248320 / 1024) */
// PERF-111: GEMV_KMAJOR 1 reads the full head (N columns) in the loader's k-major order, blocks of GEMV_KMAJOR_B
// 16-k steps (loader::int8_tiled_index at a head width); the compact head stays n-major (NC is not a head width).
#ifndef GEMV_KMAJOR
#define GEMV_KMAJOR 0
#endif
#ifndef GEMV_KMAJOR_B
#define GEMV_KMAJOR_B 1
#endif
#define NO_IDX 0x7FFFFFFFu
#define WG 256

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void head_cand_gather(__global const uchar* restrict w, __global const float* restrict scales,
                               __global const uint* restrict cand, __global uchar* restrict cw,
                               __global float* restrict cs GATE_ARG) {
  GATE_RETURN
  const uint gid = get_global_id(0), j = gid / (K / 16), kb = gid % (K / 16);
  const uint id = cand[j];
#if GEMV_KMAJOR
  __global const uchar* src =
      w + (((size_t)kb / GEMV_KMAJOR_B * (N / 16) + id / 16) * GEMV_KMAJOR_B + kb % GEMV_KMAJOR_B) * 256 + id % 16;
#else
  __global const uchar* src = w + ((size_t)(id / 16) * (K / 16) + kb) * 256 + id % 16;
#endif
  __global uchar* dst = cw + ((size_t)(j / 16) * (K / 16) + kb) * 256 + j % 16;
  for (uint k = 0; k < 16; ++k) dst[k * 16] = src[k * 16];
  if (kb == 0) cs[j] = scales[id];
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void head_cand_argmax(__global const float* restrict score, __global const uint* restrict cand,
                               __global float* restrict part GATE_ARG) {
  GATE_RETURN
  const uint r = get_group_id(0), lid = get_local_id(0);
  __local float bv;
  __local uint bi;
  if (lid == 0) {
    float v = -INFINITY;
    uint i = NO_IDX;
    for (uint c = 0; c < C; ++c) {
      const float x = score[(size_t)r * NC + r * C + c];
      const uint id = cand[r * C + c];
      if (x > v || (x == v && id < i)) { v = x; i = id; }
    }
    bv = v;
    bi = i;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < GROUPS) {
    part[((size_t)r * GROUPS + lid) * 2 + 0] = lid == 0 ? bv : -INFINITY;
    part[((size_t)r * GROUPS + lid) * 2 + 1] = lid == 0 ? (float)bi : (float)NO_IDX;
  }
}
