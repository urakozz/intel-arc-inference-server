// probe_ink_w4a8 - plan 41e Task 1: can int8 DPAS gain on Inkling's NVFP4 experts at all? The production prompt
// expert GEMM (pf_moe_q4gemm.cl's MS 8 form: each 64-k group's B image decoded once a work-group into local memory,
// three buffers, a split barrier) with the image in int8 and the A rows int8 (one fp32 scale a row):
//
//   ARM 1  exact: 2 x E2M1 is an integer in [-12, 12] (int8-exact); the E4M3 block scale is per 16 k, so each 32-k
//          DPAS runs twice, once with the A lanes of k 16-31 zeroed and once with k 0-15 zeroed (the B image whole),
//          and each 16-k int32 sum is rescaled in fp32: acc = fma(float(t), s_block, acc). Bitwise to the host twin.
//   ARM 2  the refused fold (a timing ceiling only, not numerics we may ship): one 32-k DPAS, one rescale by the
//          first block's scale.
//   ARM 3  the int8 core alone (a ceiling): int32 sums over the whole K, one scale at the end.
//
// All arms end with y = (acc x row scale) x the expert's outer scale, fp32 stores a lane (OUT_F32's form).
// pf_moe_q4gemm(items, A int8 [rows][K], xs [rows], W, out f32 [rows][N]); grid (N / 128, items), 256 work-items.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#if !defined(K) || !defined(N) || !defined(ARM)
#error "probe_ink_w4a8: K, N and ARM"
#endif
#define WFMT 1
#include "expert_fp.h"
#define SG 16
#define NONE 0xFFFFFFFFu
#define GROUP 64
#define NA 4
#define NB 4
#define WGN 2
#define MS 8
#define R 32
#define CN (16 * NB)
#define WG_N (CN * WGN)
#define G (K / GROUP)
#define BLK_U32 ((size_t)(N / 16) * (K / GROUP) * TILE_U32 + EXPERT_OUTER_WORDS)
#define UNITS (WGN * NB * 2)          /* (atom, 32-k half) units of a 64-k group */
#define UPS (UNITS / (MS * WGN))      /* 1 */

__attribute__((reqd_work_group_size(SG * WGN * MS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_moe_q4gemm(__global const uint* restrict items, __global const uchar* restrict A,
                            __global const float* restrict xs, __global const uint* restrict W,
                            __global float* restrict out) {
  const uint it = get_group_id(1);
  const uint e = items[4 * it];
  if (e == NONE) return;
  const uint s = get_sub_group_id(), sm = s / WGN, sn = s % WGN;
  const uint row0 = items[4 * it + 1] + R * sm;
  const uint rows = items[4 * it + 2] > R * sm ? items[4 * it + 2] - R * sm : 0u;
  const uint nb = get_group_id(0) * WG_N + CN * sn;
  const uint lane = get_sub_group_local_id();
  __global const uint* restrict Wb = W + (size_t)e * BLK_U32;
  __global const uchar* restrict Ab = A + (size_t)row0 * K;

#if ARM == 3
  int8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (int8)(0);
#else
  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);
#endif

  // 2 x E2M1 as int8 for a code byte's two values: lo | hi << 8.
  __local ushort e2m1_i8[256];
  for (uint i = get_local_id(0); i < 256; i += get_local_size(0)) {
    const int lo = (int)(fp_weight(i, 0, 2.f)), hi = (int)(fp_weight(i, 1, 2.f));
    e2m1_i8[i] = (ushort)((uchar)(char)lo | ((uint)(uchar)(char)hi << 8));
  }
  __local uint slm[3][UNITS][128];   // a unit: 16 columns x 32 k int8, lane l's word j (k 4j..4j+3) at [j 16 + l]
  const uint n_wg = get_group_id(0) * WG_N / 16u;
  uint uw[4];
  // unit u = (atom u / 2, 32-k half u % 2); sub-group s decodes unit s
#define UNIT_LOAD(gg)                                                                             \
  {                                                                                               \
    const uint u = s, k = 32u * (u % 2u);                                                         \
    const __global uint* tl = Wb + ((size_t)(n_wg + u / 2u) * G + (gg)) * TILE_U32;               \
    _Pragma("unroll") for (uint w = 0; w < 4; ++w) uw[w] = FP_TILE_WORD(tl, lane, k + w * 8u);    \
  }
  UNIT_LOAD(0u)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint g = 0; g < G; ++g) {
    __local uint (*buf)[128] = slm[g % 3u];
    for (uint dg = g == 0 ? 0u : g + 1; dg <= g + 1 && dg < G; ++dg) {
      __local uint (*dst)[128] = slm[dg % 3u];
      uint img[8];
#pragma unroll
      for (uint w = 0; w < 4; ++w) {   // a code word's 8 values -> two int8 words
        const uint lo = (uint)e2m1_i8[uw[w] & 255u] | ((uint)e2m1_i8[(uw[w] >> 8) & 255u] << 16);
        const uint hi = (uint)e2m1_i8[(uw[w] >> 16) & 255u] | ((uint)e2m1_i8[uw[w] >> 24] << 16);
        img[2 * w] = lo;
        img[2 * w + 1] = hi;
      }
      intel_sub_group_block_write8(dst[s], vload8(0, img));
      if (dg + 1 < G) UNIT_LOAD(dg + 1)
      if (dg == 0) barrier(CLK_LOCAL_MEM_FENCE);
    }
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    if (rows == 0) {
      intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
      continue;
    }
    const uint k0 = g * GROUP;
    // 32 rows x 64 int8: afrag[c 32 + r] = the k32 half c of row r (lane l: k 2l, 2l + 1)
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + k0), K - (int)k0, R, K, (int2)(0, 0), afrag);
#if ARM != 3
    float sc[NB][4];   // this lane's column's four E4M3 block scales of the group
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const __global uint* tl = Wb + ((size_t)(nb / 16u + (uint)b) * G + g) * TILE_U32;
#pragma unroll
      for (uint kb = 0; kb < 4; ++kb) sc[b][kb] = fp_e4m3((tl[32 * FP_CODE_BITS + kb * 4 + lane / 4] >> (8 * (lane % 4))) & 255u);
    }
#endif
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      short8 af[NA];
#pragma unroll
      for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(32 * h + 8 * a)));
#if ARM == 1
      short8 alo[NA], ahi[NA];
#pragma unroll
      for (int a = 0; a < NA; ++a) {
        alo[a] = lane < 8 ? af[a] : (short8)(0);
        ahi[a] = lane < 8 ? (short8)(0) : af[a];
      }
#endif
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const int8 bv = as_int8(intel_sub_group_block_read8(buf[(NB * sn + (uint)b) * 2u + (uint)h]));
#pragma unroll
        for (int a = 0; a < NA; ++a) {
#if ARM == 1
          const int8 t0 = intel_sub_group_i8_i8_matrix_mad_k32(alo[a], bv, (int8)(0));
          const int8 t1 = intel_sub_group_i8_i8_matrix_mad_k32(ahi[a], bv, (int8)(0));
          acc[a][b] = fma(convert_float8(t0), (float8)(sc[b][2 * h]), acc[a][b]);
          acc[a][b] = fma(convert_float8(t1), (float8)(sc[b][2 * h + 1]), acc[a][b]);
#elif ARM == 2
          const int8 t = intel_sub_group_i8_i8_matrix_mad_k32(af[a], bv, (int8)(0));
          acc[a][b] = fma(convert_float8(t), (float8)(sc[b][2 * h]), acc[a][b]);
#else
          acc[a][b] = intel_sub_group_i8_i8_matrix_mad_k32(af[a], bv, acc[a][b]);
#endif
        }
      }
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }
#undef UNIT_LOAD
  if (rows == 0) return;
  const float outer = as_float(Wb[BLK_U32 - 1]);
  const int live_rows = min((int)R, (int)rows);
  __global float* restrict Y = out + (size_t)row0 * N;
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    float8 rs;
#pragma unroll
    for (int r = 0; r < 8; ++r) rs[r] = xs[row0 + 8u * (uint)a + (uint)r];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#if ARM == 3
      const float8 v = (convert_float8(acc[a][b]) * rs) * outer;
#else
      const float8 v = (acc[a][b] * rs) * outer;
#endif
#pragma unroll
      for (int r = 0; r < 8; ++r)
        if (8 * a + r < live_rows) Y[(size_t)(8 * a + r) * N + nb + 16u * (uint)b + lane] = v[r];
    }
  }
}
