// pf_moe_w4a8 - plan 2026-11-12 (Ornith MoE prefill) Task 5: l0-int8's routed + shared gate||up as W4A8 (operator
// ruling D1, revised 2026-11-12). The chunk's normed rows are quantised to int8 per token AND 64-k group (one fp32
// scale a row and group, no rotation: the int4 weights are not rotated), gathered into sorted order with their
// scales, then multiplied by the int4 expert blocks as the native s4 DPAS operand (the nibble xor 8 is the whole
// weight conversion: exact), accumulated in int32 per 64-k group and rescaled in fp32 per group by the block's f16
// group scale x the row's group scale. Down stays bf16 (pf_moe_q4gemm.cl). Probe and numerics:
// docs/probes/probe-moe-prefill-2026-11-12.md (the MoE block 0.90 % relL2 from f32 against today's l0-int8's
// 1.02 %; the gate is the f32 judge).
//
//   pf_quant_g64(x, xq, xs, C)                     grid (C), WG 32 x 16      x bf16 [C][K] -> int8 [C][K], fp32 [C][K/64]
//   pf_moe_gather_i8g(xq, xs, hdr, row_tok, xg, xsg) grid (rows), WG 64      the sorted rows + their K/64 scales
//   pf_moe_w4a8(items, A, xsg, W, W_sh, h)         grid (N / WG_N, ibound)    gate||up + SiLU -> h bf16 [rows][N/2]
//
// Quantisation (tools/probe/moe_w4a8_accuracy.py's q8 form): per row and group of 64 k, s = max |x| / 127 (1 when
// the group is all zero), q = clamp(rint(x / s), -127, 127) (correctly rounded divide, round half to even). A padding
// row gathers as zeros with scale 0.
//
// The GEMM: pf_moe_q4gemm.cl's work-item walk (pf_moe_sort_mw's items, MI = 8 NA WGM rows, the n-blocks on grid
// dimension 0), sub-groups of 32 rows x 16 NB columns; per group: one 2D read of 32 rows x 64 int8, the block's
// 8 words a column (layout 1) and its scale, two k32 i8 x s4 DPAS into a zeroed int32 tile, then
// acc = fma(float(tile), weight scale x row scale, acc). The epilogue is pf_moe_gemm.cl's SiLU chain on acc.
// Deterministic: a row's value depends on its own row and the block only.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !defined(K) || !defined(N) || !defined(EXPERTS)
#error "pf_moe_w4a8: K (hidden), N (one block's gate||up columns) and EXPERTS must be defined"
#endif
#ifndef NB
#define NB 2
#endif
#ifndef WGM
#define WGM 2           /* MI = 32 WGM rows a work item: pf_moe_sort_mw's MI */
#endif
#ifndef WGN
#define WGN 8
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define GROUP 64
#define TILE_U32 136
#define NA 4
#define R 32
#define CN (16 * NB)
#define WG_N (CN * WGN)
#define G (K / GROUP)
#define BLK_U32 ((size_t)(N / 16) * (K / GROUP) * TILE_U32)
#define H_ROWS 2
#if N % WG_N != 0 || K % GROUP != 0 || NB % 2
#error "pf_moe_w4a8: N whole work-group blocks, K whole 64-k groups, NB even (gate, up atom pairs)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

// One sub-group a 64-k group of one token row: 4 values a lane.
__attribute__((reqd_work_group_size(SG * G, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_quant_g64(__global const ushort* restrict x, __global char* restrict xq, __global float* restrict xs,
                           uint C) {
  const uint t = get_group_id(0);
  if (t >= C) return;
  const uint g = get_sub_group_id(), lane = get_sub_group_local_id();
  const size_t base = (size_t)t * K + g * GROUP + 4u * lane;
  const ushort4 v = vload4(0, x + base);
  const float4 f = (float4)(bf16f(v.s0), bf16f(v.s1), bf16f(v.s2), bf16f(v.s3));
  const float m = sub_group_reduce_max(fmax(fmax(fabs(f.s0), fabs(f.s1)), fmax(fabs(f.s2), fabs(f.s3))));
  const float s = m > 0.0f ? m / 127.0f : 1.0f;
  const char4 q = (char4)((char)clamp(rint(f.s0 / s), -127.0f, 127.0f), (char)clamp(rint(f.s1 / s), -127.0f, 127.0f),
                          (char)clamp(rint(f.s2 / s), -127.0f, 127.0f), (char)clamp(rint(f.s3 / s), -127.0f, 127.0f));
  vstore4(q, 0, xq + base);
  if (lane == 0) xs[(size_t)t * G + g] = s;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void pf_moe_gather_i8g(__global const uint4* restrict xq, __global const float* restrict xs,
                                __global const uint* restrict hdr, __global const uint* restrict row_tok,
                                __global uint4* restrict xg, __global float* restrict xsg) {
  const uint r = get_group_id(0);
  if (r >= hdr[H_ROWS]) return;
  const uint t = row_tok[r];
  const uint W = K / 16;
  for (uint i = get_local_id(0); i < W; i += 64)
    xg[(size_t)r * W + i] = t == NONE ? (uint4)(0u) : xq[(size_t)t * W + i];
  for (uint i = get_local_id(0); i < G; i += 64) xsg[(size_t)r * G + i] = t == NONE ? 0.0f : xs[(size_t)t * G + i];
}

__attribute__((reqd_work_group_size(SG * WGM * WGN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_moe_w4a8(__global const uint* restrict items, __global const uchar* restrict A,
                          __global const float* restrict xsg, __global const uint* restrict W,
                          __global const uint* restrict W_sh, __global ushort* restrict out) {
  const uint it = get_group_id(1);
  const uint e = items[4 * it];
  if (e == NONE) return;                              // uniform; no barrier below
  const uint row0 = items[4 * it + 1];
  const uint rows = items[4 * it + 2];
  const uint s = get_sub_group_id();
  const uint sm = s / WGN, sn = s % WGN;
  if (R * sm >= rows) return;                         // this sub-group's 32 rows are past the item
  const uint mb = row0 + R * sm;
  const uint nb = get_group_id(0) * WG_N + CN * sn;
  __global const uint* restrict Wb = e == EXPERTS ? W_sh : W + (size_t)e * BLK_U32;
  __global const uchar* restrict Ab = A + (size_t)mb * K;
  float8 acc[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (float8)(0.0f);
  uint wq[NB][8];
  float sc[NB];
#define LOAD_GROUP(gg, WQ, SC)                                                                     \
  _Pragma("unroll") for (int b = 0; b < NB; ++b) {                                                 \
    __global const uint* tl = Wb + ((size_t)((nb + 16u * (uint)b) / 16u) * G + (gg)) * TILE_U32;  \
    const uint8 v = intel_sub_group_block_read8(tl);                                               \
    WQ[b][0] = v.s0; WQ[b][1] = v.s1; WQ[b][2] = v.s2; WQ[b][3] = v.s3;                            \
    WQ[b][4] = v.s4; WQ[b][5] = v.s5; WQ[b][6] = v.s6; WQ[b][7] = v.s7;                            \
    SC[b] = (float)as_half(intel_sub_group_block_read_us((__global const ushort*)(tl + 128)));     \
  }
  LOAD_GROUP(0u, wq, sc)
  for (uint g = 0; g < G; ++g) {
    uint wn[NB][8];
    float sn_[NB];
    if (g + 1 < G) { LOAD_GROUP(g + 1, wn, sn_) }
    const uint k0 = g * GROUP;
    // 32 rows x 64 int8 k, one message (the k offset in the base): afrag[c 32 + r] = the k32 half c of row r.
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(Ab + k0), K - (int)k0, R, K, (int2)(0, 0), afrag);
    float8 xg[NA];   // the rows' group scales (uniform across lanes)
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int r = 0; r < 8; ++r) xg[a][r] = xsg[(size_t)(mb + 8u * (uint)a + (uint)r) * G + g];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const int4 b0 = as_int4(vload4(0, wq[b]) ^ (uint4)(0x88888888u));
      const int4 b1 = as_int4(vload4(1, wq[b]) ^ (uint4)(0x88888888u));
#pragma unroll
      for (int a = 0; a < NA; ++a) {
        int8 t = intel_sub_group_i8_i4_matrix_mad_k32(as_short8(vload8(0, afrag + (uint)(8 * a))), b0, (int8)(0));
        t = intel_sub_group_i8_i4_matrix_mad_k32(as_short8(vload8(0, afrag + (uint)(32 + 8 * a))), b1, t);
        acc[a][b] = fma(convert_float8(t), (float8)(sc[b]) * xg[a], acc[a][b]);
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
      for (int j = 0; j < 8; ++j) wq[b][j] = wn[b][j];
      sc[b] = sn_[b];
    }
  }
#undef LOAD_GROUP
  // pf_moe_gemm.cl's SiLU epilogue: atom 2p gate, 2p + 1 its up.
  __global ushort* restrict X = out + (size_t)mb * (N / 2);
  const int x_w = N, x_h = R, x_p = N;
#define PF_SILU_ROW(r)                                             \
  do {                                                             \
    const ushort g_b = rne_bf16(gv.s##r), u_b = rne_bf16(uv.s##r); \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));             \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                     \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int p = 0; p < NB / 2; ++p) {
      const float8 gv = acc[a][2 * p], uv = acc[a][2 * p + 1];
      const uint ng = nb + 32u * (uint)p;
      ushort xv[8];
      PF_SILU_ROW(0); PF_SILU_ROW(1); PF_SILU_ROW(2); PF_SILU_ROW(3);
      PF_SILU_ROW(4); PF_SILU_ROW(5); PF_SILU_ROW(6); PF_SILU_ROW(7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, x_w, x_h, x_p, (int2)((int)(ng >> 1), 8 * a), xv);
    }
#undef PF_SILU_ROW
}
