// gemv.cl - int4 (group 64, symmetric) weights x bf16 activations, M in [1,8].
// Spec 1 §9.2. Weight-stationary: a subgroup owns 16 consecutive n and streams
// its K-range once; the M activation rows are re-read per k-group from L1/L2
// (16 lanes hit the same line: a broadcast load, no shuffles). Split-K: grid
// dim 1 is the slice s; slices write separate fp32 partials and the consumer
// (prep) sums them, so no atomics and bitwise-identical replays.
//
// Compile-time: M (tokens), K, N, S (split-K slices, S | K/64), LAYOUT (0 GPTQ
// native: w[K/8][N] u32 + scales[K/64][N] f16; 1 tiled: per (n_tile, g) 512 B
// nibbles [k_octet][n] then 32 B of f16 scales, contiguous along K).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#ifndef GEMV_BLOCK2D
#define GEMV_BLOCK2D 0
#endif
#ifndef GEMV_DEQ_SHIFT
#define GEMV_DEQ_SHIFT 0
#endif
// GEMV_AHEAD A: a subgroup issues A k-groups' weight and scale loads before it computes the first of them, so A
// round trips are in flight instead of one. The arithmetic and its order are unchanged (bitwise the A = 1 form).
// For the narrow shapes whose few work-groups leave the device latency bound (a tensor parallel card's halves).
#ifndef GEMV_AHEAD
#define GEMV_AHEAD 1
#endif
// GEMV_KMAJOR 1 (layout 1 only, PERF-111): the 544 B tiles ordered [g][n_tile] instead of [n_tile][g]
// (common::tiles_k_major, one repack at load), so at each k-group the launch's sub-groups read neighbouring tiles
// rather than each walking its own slab of K. probe_bw measured the two fronts at ~603 against 495-584 GB/s. Same
// loads, same arithmetic and order: bitwise the [n_tile][g] build.
#ifndef GEMV_KMAJOR
#define GEMV_KMAJOR 0
#endif
#if GEMV_KMAJOR && LAYOUT != 1
#error "gemv: GEMV_KMAJOR is layout 1 only"
#endif
// GEMV_KSL (PERF-108, a tensor parallel card's qkv||z): KSL sub-groups split each 16-column tile's k-groups of the slice
// inside the work-group - sub-group q takes groups q, q + KSL, q + 2 KSL, ... of the slice, so the tile's sub-groups read
// one contiguous run together - and SLM's fixed pairwise tree (gemv_bf16.cl's, prep.cl's: strides KSL/2 .. 1) folds them
// into the slice's one partial. The grid (N / 64, S) and the output are unchanged; the work-group is 64 x KSL work-items
// and the launch KSL x the sub-groups. The sum's order is not the unsplit chain's (judged, not bitwise).
#ifndef GEMV_KSL
#define GEMV_KSL 1
#endif
#if GEMV_BLOCK2D
#if LAYOUT != 0
#error "gemv: GEMV_BLOCK2D is layout 0 only"
#endif
#if GEMV_BLOCK2D != 8
#error "gemv: GEMV_BLOCK2D must be 8 (one 64-K group)"
#endif
#endif
#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define TILE_U32 136   /* 128 u32 of nibbles + 8 u32 of scales = 544 B */
// The host's require on N (capture.cc), here too: a binary built at any other shape would leave columns or k-groups
// unread.
#if N % WG_N != 0
#error "gemv: N must be a multiple of 64 (one work-group's columns)"
#endif
#if K % GROUP != 0 || G % S != 0
#error "gemv: K must be whole 64-row groups and S must divide K / 64"
#endif
#if G_PER_S % GEMV_AHEAD != 0
#error "gemv: GEMV_AHEAD must divide the k-groups of a slice"
#endif
#if (GEMV_KSL & (GEMV_KSL - 1)) != 0 || G_PER_S % (GEMV_KSL * GEMV_AHEAD) != 0
#error "gemv: GEMV_KSL must be a power of two (the tree) and KSL x GEMV_AHEAD must divide the k-groups of a slice"
#endif
#if GEMV_KSL > 1
#define G_FIRST(g0) ((g0) + q)
#define G_STRIDE (GEMV_AHEAD * GEMV_KSL)
#define G_AT(ga, a) ((ga) + (a) * GEMV_KSL)
#else
#define G_FIRST(g0) g0
#define G_STRIDE GEMV_AHEAD
#define G_AT(ga, a) ga + a
#endif
// GEMV_TP_SEND (PERF-44, Qwen3.8-Flash-Next tp decode's out_proj / o_proj): tp.cl's tp_send folded into the linear -
// each work-group is one chunk of the crossing per row (its 64 columns of slice s: chunk s x N / 64 + group x, row m)
// and writes its partial both to `out` and to the peer's landing (tp_wire.h's layout at [S][M][N]), then the stamp
// lines and the flags, as tp_send does. tp.cl's tp_sum under TP_WIRE_COLS 64 waits for these chunks. The arithmetic
// is unchanged; every other binary leaves it unset.
#ifndef GEMV_TP_SEND
#define GEMV_TP_SEND 0
#endif
#if GEMV_TP_SEND
#if GEMV_KSL > 1
#error "gemv GEMV_TP_SEND: one sub-group a tile (GEMV_KSL 1)"
#endif
#define TP_MROWS 8
#if M > TP_MROWS
#error "gemv GEMV_TP_SEND: M <= 8 (the landing's rows)"
#endif
#define DATA_WORDS (S * TP_MROWS * N)
#define J_CHUNKS ((N / WG_N) * S * TP_MROWS)
#include "tp_wire.h"
#endif
// TP_FOLD_SEND (PERF-44, Qwen3.8 tp's out_proj / o_proj / down at the folded crossing): tp.cl's folded tp_send as this
// linear's tail (tp_fold_send.h - the last work-group under a chunk sums its slices and sends it); the arithmetic is
// unchanged; every other binary leaves it unset.
#ifndef TP_FOLD_SEND
#define TP_FOLD_SEND 0
#endif
#if TP_FOLD_SEND
#if GEMV_TP_SEND
#error "gemv: GEMV_TP_SEND or TP_FOLD_SEND"
#endif
#if GEMV_KSL > 1
#error "gemv TP_FOLD_SEND: one sub-group a tile (GEMV_KSL 1)"
#endif
#define WG_COLS WG_N
#include "tp_fold_send.h"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// 8 nibbles of one u32 word times 8 consecutive activations.
inline float dot8(uint word, ushort8 xv) {
#if GEMV_DEQ_SHIFT
  // OpenCL C defines right shift of a signed integer as arithmetic sign-fill.
  // For every q in [0,16), signext4(q ^ 8) == q - 8. One word-wide xor plus
  // the shift pair per nibble therefore preserves the exact integer values and
  // the dot-product order while removing mask/subtract work. gemv_test drives
  // all 16 nibbles through this path and requires bit-identity with the control.
  const uint u = word ^ 0x88888888u;
  float a = 0.f;
  a += (float)(((int)(u << 28)) >> 28) * bf16f(xv.s0);
  a += (float)(((int)(u << 24)) >> 28) * bf16f(xv.s1);
  a += (float)(((int)(u << 20)) >> 28) * bf16f(xv.s2);
  a += (float)(((int)(u << 16)) >> 28) * bf16f(xv.s3);
  a += (float)(((int)(u << 12)) >> 28) * bf16f(xv.s4);
  a += (float)(((int)(u <<  8)) >> 28) * bf16f(xv.s5);
  a += (float)(((int)(u <<  4)) >> 28) * bf16f(xv.s6);
  a += (float)(((int)(u      )) >> 28) * bf16f(xv.s7);
  return a;
#else
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
#if GEMV_KSL > 1
__attribute__((reqd_work_group_size(WG_N * GEMV_KSL, 1, 1)))
#else
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
#endif
__kernel void gemv(__global const uint* restrict w,
                   __global const half* restrict scales,
                   __global const ushort* restrict x,
                   __global float* restrict out
#if GEMV_TP_SEND
                   , __global uint* land, __global uint* ctr
#endif
#if TP_FOLD_SEND
                   , __global uint* land, __global uint* ctr, __global float* own
#endif
                   ) {
  const uint lane = get_sub_group_local_id();
#if GEMV_KSL > 1
  // [sub-group][m][lane]: each K slice's fp32 partial of its column, for the tree.
  __local float red[SG_PER_WG * GEMV_KSL][M][SG];
  const uint sg = get_sub_group_id();
  const uint q = sg / SG_PER_WG;   // the K slice; the column tile is sg % SG_PER_WG
  const uint n_tile = get_group_id(0) * SG_PER_WG + sg % SG_PER_WG;
#else
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
#endif
  const uint n = n_tile * SG + lane;
  const uint s = get_group_id(1);
  const uint g0 = s * G_PER_S;
  const uint g1 = g0 + G_PER_S;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint ga = G_FIRST(g0); ga < g1; ga += G_STRIDE) {
    uint wva[GEMV_AHEAD][8];
    float scalea[GEMV_AHEAD];
    __attribute__((opencl_unroll_hint)) for (uint a = 0; a < GEMV_AHEAD; ++a) {
      const uint g = G_AT(ga, a);
      __private uint* wv = wva[a];
#if LAYOUT == 0
#if GEMV_BLOCK2D
      // One message reads this subgroup's 16 columns across the eight u32 rows
      // of a 64-K group from the GPTQ-native [K/8][N] surface. This is the exact
      // geometry measured by probe_gemv_loads.cl's l0b2ddeq cells.
      intel_sub_group_2d_block_read_32b_8r16x1c(
          (__global void*)(__global uint*)w, (int)(N * 4u), (int)(K / 8u), (int)(N * 4u),
          (int2)((int)(n_tile * SG), (int)(g * 8u)), wv);
#else
      __global const uint* wp = w + (size_t)(g * 8) * N + n;
      for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
#endif
      scalea[a] = (float)scales[(size_t)g * N + n];
#else
#if GEMV_KMAJOR
      __global const uint* tile = w + ((size_t)g * (N / SG) + n_tile) * TILE_U32;
#else
      __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
#endif
      uint8 blk = intel_sub_group_block_read8(tile);
      wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
      wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
      ushort sh = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
      scalea[a] = (float)as_half(sh);
#endif
    }
    __attribute__((opencl_unroll_hint)) for (uint a = 0; a < GEMV_AHEAD; ++a) {
      const uint g = G_AT(ga, a);
      const __private uint* wv = wva[a];
      const float scale = scalea[a];
      float gacc[M];
      for (int m = 0; m < M; ++m) gacc[m] = 0.f;
      for (int j = 0; j < 8; ++j) {
        for (int m = 0; m < M; ++m) {
          ushort8 xv = vload8(0, x + (size_t)m * K + g * GROUP + j * 8);
          gacc[m] += dot8(wv[j], xv);
        }
      }
      for (int m = 0; m < M; ++m) acc[m] += gacc[m] * scale;
    }
  }

#if GEMV_TP_SEND
  // tp_send for the chunks (c, m), m < M: the call numbers, each row's 64 words here and in the peer's landing, the
  // stamp lines, the flags.
  const uint lid = get_local_id(0);
  const uint c = s * (N / WG_N) + get_group_id(0);
  __local uint s_n[M];
  if (lid == 0)
    for (uint m = 0; m < M; ++m) s_n[m] = tp_next(ctr, c, m);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int m = 0; m < M; ++m) {
    const size_t e = ((size_t)s * M + m) * N + n;
    out[e] = acc[m];
    land[(s_n[m] & 1u) * DATA_WORDS + e] = as_uint(acc[m]);
  }
  for (uint i = lid; i < M * TP_TRAILER_STRIDE; i += WG_N) {
    const uint m = i / TP_TRAILER_STRIDE;
    const uint t = s_n[m];
    land[TRAILER(t & 1u, c * TP_MROWS + m) + i % TP_TRAILER_STRIDE] = t;
  }
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    for (uint m = 0; m < M; ++m) {
      const uint t = s_n[m];
      *(volatile __global uint*)(land + FLAG(t & 1u, c * TP_MROWS + m)) = t;
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  }
#elif GEMV_KSL > 1
  for (int m = 0; m < M; ++m) red[sg][m][lane] = acc[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  // The fixed tree: slice q < stride adds slice q + stride of the same column tile (SG_PER_WG sub-groups apart).
  for (uint stride = GEMV_KSL / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (int m = 0; m < M; ++m) red[sg][m][lane] += red[sg + stride * SG_PER_WG][m][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (int m = 0; m < M; ++m) out[((size_t)s * M + m) * N + n] = red[sg][m][lane];
#else
  for (int m = 0; m < M; ++m) out[((size_t)s * M + m) * N + n] = acc[m];
#if TP_FOLD_SEND
  __local uint tpf[TPF_LOCAL];
  tp_fold_send(out, land, ctr, own, tpf);
#endif
#endif
}
