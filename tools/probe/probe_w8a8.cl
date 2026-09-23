// W8A8 SPEED probe: the i8 x i8 k32 DPAS with PER-CHANNEL int8 weights
// (2026-09-23).
//
// The CPU accuracy study (docs/probe-w4a8-2026-09-23.md, the appended i8 x i8
// section) left one candidate standing: per-channel int8 weights requantised
// from the checkpoint's int4 g64, plus per-token int8 activations after a
// 1024-block Hadamard rotation. This file holds that path's four kernels:
//
//   pw8_requant     int4 g64 (layout 0) -> int8 per-channel, VNNI-4 packed,
//                   one 1024-column slab (or the whole N) per launch. The
//                   column scale is an INPUT: it is a property of the weights
//                   alone and would be computed once at load.
//   pw8_gemm        C = (xq . wq) * xs[m] * ws[n]; int32 accumulation over the
//                   WHOLE K, the only float work is the epilogue.
//   pw8_quant       per-token symmetric int8, no rotation.
//   pw8_quant_had   the same after x * D * blockdiag(H_1024) / 32 (random signs
//                   D, Walsh-Hadamard in SLM). Timed; its output pairs with
//                   ROTATED weights, which this probe does not build, so no
//                   GEMM consumes it.
//
// PROBE-ONLY. No runtime path binds any entry point here.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
#define CHUNK 64          // k per chunk: one A message, two B messages, 2 x k32 dpas per atom

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// ---------------------------------------------------------------------------
// pw8_requant: one work-item = one column n, one qweight row r (8 k).
// Grid (Ns/16, K/8/16) groups of 16 x 16. Writes out rows 2r and 2r+1 of the
// VNNI-4 slab: dword [k/4][n - n0], byte j = k 4d + j.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void pw8_requant(__global const uint* restrict qw,      // [K/8][Nfull]
                          __global const half* restrict sc,      // [K/64][Nfull]
                          __global const float* restrict cs,     // [Nfull] column scale
                          __global uint* restrict out,           // [K/4][ldo]
                          uint n0, uint Nfull, uint ldo) {
  const uint nl = get_global_id(0);            // column within the slab
  const uint r = get_global_id(1);             // qweight row
  const uint n = n0 + nl;
  const uint word = qw[(size_t)r * Nfull + n];
  const float s = vload_half((size_t)(r / 8u) * Nfull + n, sc);
  const float c = cs[n];
  uint lo = 0u, hi = 0u;
  #pragma unroll
  for (uint i = 0; i < 4; ++i) {
    const float v0 = (float)((int)((word >> (4u * i)) & 0xFu) - 8) * s;
    const float v1 = (float)((int)((word >> (4u * (i + 4u))) & 0xFu) - 8) * s;
    lo |= ((uint)(uchar)convert_char_sat_rte(v0 / c)) << (8u * i);
    hi |= ((uint)(uchar)convert_char_sat_rte(v1 / c)) << (8u * i);
  }
  out[(size_t)(2u * r) * ldo + nl] = lo;
  out[(size_t)(2u * r + 1u) * ldo + nl] = hi;
}

// ---------------------------------------------------------------------------
// pw8_gemm: the gsweep tile (WG 256 x 128, 32 sub-groups 8 (m) x 4 (n), per
// sub-group 32 m x 32 n), B operand int8 VNNI-4: a lane's 64-k chunk is 16
// dwords of its own column, one 32b_16r16x1c message per n-atom.
// ---------------------------------------------------------------------------
#define WG_M 256
#define WG_N 128
#define SG_M 32
#define SG_N 32
#define NA 4
#define NB 2

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_gemm(__global const ushort* restrict xq16,   // int8 [M][K] as pairs
                       __global const float* restrict xs,      // [M]
                       __global const uint* restrict w8,       // [K/4][ldb]
                       __global const float* restrict ws,      // [N] (this slab's columns)
                       __global float* restrict C,             // [M][ldc], slab-offset
                       __global uint* restrict sig,
                       uint M, uint K, uint N, uint ldxq, uint ldb, uint ldc) {
  const uint m0 = get_group_id(0) * WG_M;
  const uint n0 = get_group_id(1) * WG_N;
  const uint s = get_sub_group_id();
  const uint mb = m0 + SG_M * (s >> 2);
  const uint nb = n0 + SG_N * (s & 3u);

  if (get_global_linear_id() == 0) sig[0] = 0x57384138u;   // 'W8A8'

  const int x_w = (int)K, x_h = (int)M, x_p = (int)(ldxq * 2u);
  const int b_w = (int)(N * 4u), b_h = (int)(K / 4u), b_p = (int)(ldb * 4u);
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);
  const uint NCH = K / CHUNK;

  int8 acc[NA][NB];
  #pragma unroll
  for (int a = 0; a < NA; ++a)
    #pragma unroll
    for (int b = 0; b < NB; ++b) acc[a][b] = (int8)(0);

  #pragma unroll
  for (int pf = 0; pf < 2; ++pf) {
    const int kpf = pf * CHUNK;
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                                  (int2)(kpf / 2, (int)(m0 + 8u * s)));
    intel_sub_group_2d_block_prefetch_32b_4r16x1c(
        (__global void*)w8, b_w, b_h, b_p,
        (int2)((int)(n0 + 16u * (s >> 2)), kpf / 4 + 4 * (int)(s & 3u)));
  }

  for (uint ch = 0; ch < NCH; ++ch) {
    const uint k0 = ch * CHUNK;
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                               (int2)((int)(k0 / 2u), (int)mb), afrag);
    uint wv[NB][16];
    #pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_read_32b_16r16x1c((__global void*)w8, b_w, b_h, b_p,
                                                 (int2)((int)(nb + 16u * (uint)b),
                                                        (int)(k0 / 4u)),
                                                 wv[b]);
    if (ch + 2u < NCH) {
      const int kpf = (int)((ch + 2u) * CHUNK);
      intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                                    (int2)(kpf / 2, (int)(m0 + 8u * s)));
      intel_sub_group_2d_block_prefetch_32b_4r16x1c(
          (__global void*)w8, b_w, b_h, b_p,
          (int2)((int)(n0 + 16u * (s >> 2)), kpf / 4 + 4 * (int)(s & 3u)));
    }

    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[NA];
      #pragma unroll
      for (int a = 0; a < NA; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
      #pragma unroll
      for (int b = 0; b < NB; ++b) {
        const int8 bv = as_int8(vload8(0, wv[b] + (uint)(8 * ks)));
        #pragma unroll
        for (int a = 0; a < NA; ++a)
          acc[a][b] = intel_sub_group_i8_i8_matrix_mad_k32(af[a], bv, acc[a][b]);
      }
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }

  // epilogue: out = (float)acc * (ws[n] * xs[m]); the column is the lane
  #pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float wsc = as_float(intel_sub_group_block_read((__global const uint*)(ws + nb + 16u * (uint)b)));
    #pragma unroll
    for (int a = 0; a < NA; ++a) {
      float8 o;
      o.s0 = (float)acc[a][b].s0 * (wsc * xs[mb + 8u * (uint)a + 0u]);
      o.s1 = (float)acc[a][b].s1 * (wsc * xs[mb + 8u * (uint)a + 1u]);
      o.s2 = (float)acc[a][b].s2 * (wsc * xs[mb + 8u * (uint)a + 2u]);
      o.s3 = (float)acc[a][b].s3 * (wsc * xs[mb + 8u * (uint)a + 3u]);
      o.s4 = (float)acc[a][b].s4 * (wsc * xs[mb + 8u * (uint)a + 4u]);
      o.s5 = (float)acc[a][b].s5 * (wsc * xs[mb + 8u * (uint)a + 5u]);
      o.s6 = (float)acc[a][b].s6 * (wsc * xs[mb + 8u * (uint)a + 6u]);
      o.s7 = (float)acc[a][b].s7 * (wsc * xs[mb + 8u * (uint)a + 7u]);
      intel_sub_group_2d_block_write_32b_8r16x1c(
          (__global void*)C, c_w, c_h, c_p,
          (int2)((int)(nb + 16u * (uint)b), (int)(mb + 8u * (uint)a)), (__private uint*)&o);
    }
  }
}

// ---------------------------------------------------------------------------
// The activation quantisers: one work-group of 512 per token row, K <= 20480.
// ---------------------------------------------------------------------------
#define QWG 512
#define QPER 40           // K / QWG upper bound (17408 / 512 = 34)

inline float wg_max(float v, __local float* red) {
  v = sub_group_reduce_max(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float r = 0.0f;
  for (uint i = 0; i < QWG / SG; ++i) r = fmax(r, red[i]);
  return r;
}

inline void quant_store(float* v, uint per, float amax, __global char* out, __global float* xs,
                        uint m, uint lid) {
  const float scale = amax / 127.0f;
  if (lid == 0) xs[m] = scale;
  for (uint j = 0; j < per; ++j)
    out[lid + QWG * j] = scale == 0.0f ? (char)0 : convert_char_sat_rte(v[j] / scale);
}

__attribute__((reqd_work_group_size(QWG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_quant(__global const ushort* restrict x, __global char* restrict xq,
                        __global float* restrict xs, uint K, uint ldx) {
  const uint m = get_group_id(0), lid = get_local_id(0), per = K / QWG;
  __global const ushort* row = x + (size_t)m * ldx;
  __local float red[QWG / SG];
  float v[QPER];
  float amax = 0.0f;
  for (uint j = 0; j < per; ++j) {
    v[j] = bf16f(row[lid + QWG * j]);
    amax = fmax(amax, fabs(v[j]));
  }
  amax = wg_max(amax, red);
  quant_store(v, per, amax, xq + (size_t)m * K, xs, m, lid);
}

// x' = (x * D) blockdiag(H_1024) / 32, then per-token int8. Each 1024 block is
// transformed in SLM by 10 butterfly stages, two elements per work-item; the
// work-item keeps elements (lid, lid + 512) of every block, which is exactly
// the `lid + QWG * j` set quant_store writes.
__attribute__((reqd_work_group_size(QWG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_quant_had(__global const ushort* restrict x, __global const float* restrict sgn,
                            __global char* restrict xq, __global float* restrict xs, uint K,
                            uint ldx) {
  const uint m = get_group_id(0), lid = get_local_id(0), per = K / QWG;
  __global const ushort* row = x + (size_t)m * ldx;
  __local float blk[1024];
  __local float red[QWG / SG];
  float v[QPER];
  float amax = 0.0f;
  for (uint b = 0; b < K / 1024u; ++b) {
    const uint base = b * 1024u;
    blk[lid] = bf16f(row[base + lid]) * sgn[base + lid];
    blk[lid + 512u] = bf16f(row[base + lid + 512u]) * sgn[base + lid + 512u];
    for (uint h = 1; h < 1024u; h <<= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      const uint i = (lid / h) * 2u * h + (lid % h);
      const float a = blk[i], c = blk[i + h];
      blk[i] = a + c;
      blk[i + h] = a - c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    v[2u * b] = blk[lid] * (1.0f / 32.0f);
    v[2u * b + 1u] = blk[lid + 512u] * (1.0f / 32.0f);
    amax = fmax(amax, fmax(fabs(v[2u * b]), fabs(v[2u * b + 1u])));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  amax = wg_max(amax, red);
  // v[2b] is element base + lid, v[2b+1] is base + 512 + lid: that is index
  // lid + 512 * (2b) and lid + 512 * (2b + 1), so quant_store's layout holds.
  quant_store(v, per, amax, xq + (size_t)m * K, xs, m, lid);
}

// ===========================================================================
// v2: the same three helpers, rewritten for bandwidth (the GEMM is unchanged).
//
//   pw8_requant2    4 columns x 32 k per work-item, uint4 loads and stores, and
//                   w8 = rte(q * (s * inv_c)) with inv_c = 1 / c precomputed per
//                   column -- no per-value correctly rounded divide.
//   pw8_quant2_N / pw8_quant_had2_N
//                   one SUB-GROUP per 1024-block, N = K / 1024 sub-groups per
//                   token row. The block is loaded with sub-group block reads,
//                   so lane l's value j is element j*16 + l; the Walsh-Hadamard
//                   stages h = 1..8 are lane shuffles and h = 16..512 are
//                   in-register butterflies on v[j], v[j + h/16], ascending, the
//                   same operation order the CPU oracle uses. Quantised with
//                   one reciprocal per row.
// ===========================================================================
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

inline uint pw8_pack4(uint word, uint shift0, float t) {
  uint r = 0u;
  #pragma unroll
  for (uint j = 0; j < 4u; ++j) {
    const float v = (float)((int)((word >> (shift0 + 4u * j)) & 0xFu) - 8) * t;
    r |= ((uint)(uchar)convert_char_sat_rte(v)) << (8u * j);
  }
  return r;
}

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void pw8_requant2(__global const uint* restrict qw,      // [K/8][Nfull]
                           __global const half* restrict sc,      // [K/64][Nfull]
                           __global const float* restrict inv,    // [Nfull] 1 / column scale
                           __global uint* restrict out,           // [K/4][ldo]
                           uint n0, uint Nfull, uint ldo) {
  const uint nl = get_global_id(0) * 4u;       // 4 columns
  const uint r0 = get_global_id(1) * 4u;       // 4 qweight rows = 32 k, inside one group
  const uint n = n0 + nl;
  const float4 t = vload_half4(0, sc + (size_t)(r0 / 8u) * Nfull + n) * vload4(0, inv + n);
  #pragma unroll
  for (uint i = 0; i < 4u; ++i) {
    const uint4 w = vload4(0, qw + (size_t)(r0 + i) * Nfull + n);
    const uint4 lo = (uint4)(pw8_pack4(w.x, 0u, t.x), pw8_pack4(w.y, 0u, t.y),
                             pw8_pack4(w.z, 0u, t.z), pw8_pack4(w.w, 0u, t.w));
    const uint4 hi = (uint4)(pw8_pack4(w.x, 16u, t.x), pw8_pack4(w.y, 16u, t.y),
                             pw8_pack4(w.z, 16u, t.z), pw8_pack4(w.w, 16u, t.w));
    vstore4(lo, 0, out + (size_t)(2u * (r0 + i)) * ldo + nl);
    vstore4(hi, 0, out + (size_t)(2u * (r0 + i) + 1u) * ldo + nl);
  }
}

#define PW8Q2(NAME, NBLK, HAD)                                                        \
__attribute__((reqd_work_group_size(16 * NBLK, 1, 1)))                                \
__attribute__((intel_reqd_sub_group_size(SG)))                                        \
__kernel void NAME(__global const ushort* restrict x, __global const float* restrict sgn, \
                   __global char* restrict xq, __global float* restrict xs, uint K,    \
                   uint ldx) {                                                        \
  const uint m = get_group_id(0);                                                     \
  const uint b = get_sub_group_id();                                                  \
  const uint l = get_sub_group_local_id();                                            \
  __global const ushort* src = x + (size_t)m * ldx + b * 1024u;                       \
  __local float red[NBLK];                                                            \
  float v[64];                                                                        \
  _Pragma("unroll")                                                                   \
  for (uint r = 0; r < 8u; ++r) {                                                     \
    const ushort8 u = intel_sub_group_block_read_us8(src + r * 128u);                 \
    v[8u * r + 0u] = bf16f(u.s0); v[8u * r + 1u] = bf16f(u.s1);                        \
    v[8u * r + 2u] = bf16f(u.s2); v[8u * r + 3u] = bf16f(u.s3);                        \
    v[8u * r + 4u] = bf16f(u.s4); v[8u * r + 5u] = bf16f(u.s5);                        \
    v[8u * r + 6u] = bf16f(u.s6); v[8u * r + 7u] = bf16f(u.s7);                        \
    if (HAD) {                                                                        \
      const float8 sg = as_float8(intel_sub_group_block_read8(                        \
          (__global const uint*)(sgn + b * 1024u + r * 128u)));                       \
      v[8u * r + 0u] *= sg.s0; v[8u * r + 1u] *= sg.s1;                               \
      v[8u * r + 2u] *= sg.s2; v[8u * r + 3u] *= sg.s3;                               \
      v[8u * r + 4u] *= sg.s4; v[8u * r + 5u] *= sg.s5;                               \
      v[8u * r + 6u] *= sg.s6; v[8u * r + 7u] *= sg.s7;                               \
    }                                                                                 \
  }                                                                                   \
  if (HAD) {                                                                          \
    _Pragma("unroll")                                                                 \
    for (uint h = 1u; h < 16u; h <<= 1) {                                             \
      _Pragma("unroll")                                                               \
      for (uint j = 0; j < 64u; ++j) {                                                \
        const float o = sub_group_shuffle_xor(v[j], h);                               \
        v[j] = (l & h) ? (o - v[j]) : (v[j] + o);                                     \
      }                                                                               \
    }                                                                                 \
    _Pragma("unroll")                                                                 \
    for (uint hj = 1u; hj < 64u; hj <<= 1) {                                          \
      _Pragma("unroll")                                                               \
      for (uint j = 0; j < 64u; ++j) {                                                \
        if ((j & hj) == 0u) {                                                         \
          const float a = v[j], c = v[j + hj];                                        \
          v[j] = a + c;                                                               \
          v[j + hj] = a - c;                                                          \
        }                                                                             \
      }                                                                               \
    }                                                                                 \
    _Pragma("unroll")                                                                 \
    for (uint j = 0; j < 64u; ++j) v[j] *= (1.0f / 32.0f);                            \
  }                                                                                   \
  float amax = 0.0f;                                                                  \
  _Pragma("unroll")                                                                   \
  for (uint j = 0; j < 64u; ++j) amax = fmax(amax, fabs(v[j]));                       \
  amax = sub_group_reduce_max(amax);                                                  \
  if (l == 0u) red[b] = amax;                                                         \
  barrier(CLK_LOCAL_MEM_FENCE);                                                       \
  amax = 0.0f;                                                                        \
  for (uint i = 0; i < (uint)(NBLK); ++i) amax = fmax(amax, red[i]);                  \
  const float scale = amax / 127.0f;                                                  \
  const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;                               \
  if (b == 0u && l == 0u) xs[m] = scale;                                              \
  __global uchar* dst = (__global uchar*)(xq + (size_t)m * K + b * 1024u);            \
  _Pragma("unroll")                                                                   \
  for (uint r = 0; r < 8u; ++r) {                                                     \
    uchar8 o;                                                                         \
    o.s0 = (uchar)convert_char_sat_rte(v[8u * r + 0u] * iv);                          \
    o.s1 = (uchar)convert_char_sat_rte(v[8u * r + 1u] * iv);                          \
    o.s2 = (uchar)convert_char_sat_rte(v[8u * r + 2u] * iv);                          \
    o.s3 = (uchar)convert_char_sat_rte(v[8u * r + 3u] * iv);                          \
    o.s4 = (uchar)convert_char_sat_rte(v[8u * r + 4u] * iv);                          \
    o.s5 = (uchar)convert_char_sat_rte(v[8u * r + 5u] * iv);                          \
    o.s6 = (uchar)convert_char_sat_rte(v[8u * r + 6u] * iv);                          \
    o.s7 = (uchar)convert_char_sat_rte(v[8u * r + 7u] * iv);                          \
    intel_sub_group_block_write_uc8(dst + r * 128u, o);                               \
  }                                                                                   \
}

PW8Q2(pw8_quant2_5,       5, 0)    // gate||up input, K = 5120
PW8Q2(pw8_quant_had2_5,   5, 1)
PW8Q2(pw8_quant2_6,       6, 0)    // o_proj / out_proj input, K = 6144
PW8Q2(pw8_quant_had2_6,   6, 1)
PW8Q2(pw8_quant2_17,     17, 0)    // down input, K = 17408
PW8Q2(pw8_quant_had2_17, 17, 1)

// ===========================================================================
// pw8_requant_rot: the v2 requant with the WEIGHT half of the rotation.
//
// Activations are rotated per 1024-block as x' = (x * d) H / 32
// (pw8_quant_had2_*), so each weight column needs W'_blk = H (d * w_blk) / 32:
// x' W' = (x*d) H H (d*W) / 1024 = x W. Per column that is a Walsh-Hadamard
// along K, i.e. ACROSS qweight rows, so:
//   * one work-group = 16 columns x one 1024-k block; 16 sub-groups, sub-group
//     c owns column c, lane l owns k = 64 l .. 64 l + 63 -- exactly one scale
//     group, so a lane's dequant uses ONE scale;
//   * stages h = 1..32 are in-register butterflies on v[j], v[j + h], stages
//     h = 64..512 are lane shuffles (lane xor h / 64): ascending h, the order
//     the CPU oracle uses, so the result is bit-exact;
//   * the qweight tile [128 rows][16 cols] and the VNNI-4 output tile
//     [256 dword rows][16 cols] go through SLM, so every global access is a
//     coalesced 64-byte row.
// The column scale (max |W'[:, n]| / 127) is a load-time property of the
// weights; its reciprocal comes in as `inv`.
// ===========================================================================
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_requant_rot(__global const uint* restrict qw,     // [K/8][Nfull]
                              __global const half* restrict sc,     // [K/64][Nfull]
                              __global const float* restrict sgn,   // [K] the rotation's d
                              __global const float* restrict inv,   // [Nfull] 1 / rotated col scale
                              __global uint* restrict out,          // [K/4][ldo]
                              uint n0, uint Nfull, uint ldo) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;       // first column of this tile, slab-relative
  const uint nbase = n0 + nc;                  // ... and absolute
  __local uint tin[128 * 16];
  __local float tsc[16 * 16];
  __local uint tout[256 * 16];

  #pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    tin[t] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  tsc[lid] = vload_half((size_t)(blk * 16u + lid / 16u) * Nfull + nbase + (lid % 16u), sc);
  barrier(CLK_LOCAL_MEM_FENCE);

  const float s = tsc[l * 16u + c];
  __global const float* dk = sgn + blk * 1024u + 64u * l;
  float v[64];
  #pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint word = tin[(8u * l + i) * 16u + c];
    #pragma unroll
    for (uint b = 0; b < 8u; ++b)
      v[8u * i + b] = ((float)((int)((word >> (4u * b)) & 0xFu) - 8) * s) * dk[8u * i + b];
  }
  #pragma unroll
  for (uint h = 1u; h < 64u; h <<= 1) {
    #pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & h) == 0u) {
        const float a = v[j], e = v[j + h];
        v[j] = a + e;
        v[j + h] = a - e;
      }
    }
  }
  #pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
    #pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
  const float iv = inv[nbase + c];
  #pragma unroll
  for (uint d = 0; d < 16u; ++d) {
    uint w = 0u;
    #pragma unroll
    for (uint b = 0; b < 4u; ++b)
      w |= ((uint)(uchar)convert_char_sat_rte((v[4u * d + b] * (1.0f / 32.0f)) * iv)) << (8u * b);
    tout[(16u * l + d) * 16u + c] = w;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (uint i = 0; i < 16u; ++i) {
    const uint t = lid + 256u * i;
    out[(size_t)(blk * 256u + t / 16u) * ldo + nc + (t % 16u)] = tout[t];
  }
}

// pw8_requant_rot2: pw8_requant_rot with the two access patterns fixed.
//   * SLM tiles are column-major with ODD pitches (129, 257 words): the global
//     row loads write stride-129 (conflict-free), a lane's 8 weight rows are
//     8 consecutive words, and a lane's 16 output dwords are 16 consecutive
//     words. v1 read at stride 128 and wrote at stride 256, a 16-way bank
//     conflict on every access.
//   * the rotation's signs come as a BITMASK (bit j of word w = d[32w + j] < 0),
//     two words per lane, applied as a sign-bit xor -- bit-identical to
//     multiplying by +-1, and no per-lane 256-byte gather.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_requant_rot2(__global const uint* restrict qw,     // [K/8][Nfull]
                               __global const half* restrict sc,     // [K/64][Nfull]
                               __global const uint* restrict sbits,  // [K/32]
                               __global const float* restrict inv,   // [Nfull]
                               __global uint* restrict out,          // [K/4][ldo]
                               uint n0, uint Nfull, uint ldo) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  __local uint tout[16 * 257];

  #pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    tin[(t % 16u) * 129u + t / 16u] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  tsc[(lid % 16u) * 17u + lid / 16u] =
      vload_half((size_t)(blk * 16u + lid / 16u) * Nfull + nbase + (lid % 16u), sc);
  barrier(CLK_LOCAL_MEM_FENCE);

  const float s = tsc[c * 17u + l];
  const uint2 sb = vload2(0, sbits + blk * 32u + 2u * l);
  float v[64];
  #pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint word = tin[c * 129u + 8u * l + i];
    #pragma unroll
    for (uint b = 0; b < 8u; ++b) {
      const uint j = 8u * i + b;
      const uint bit = ((j < 32u ? sb.x : sb.y) >> (j % 32u)) & 1u;
      v[j] = as_float(as_uint((float)((int)((word >> (4u * b)) & 0xFu) - 8) * s) ^ (bit << 31));
    }
  }
  #pragma unroll
  for (uint h = 1u; h < 64u; h <<= 1) {
    #pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & h) == 0u) {
        const float a = v[j], e = v[j + h];
        v[j] = a + e;
        v[j + h] = a - e;
      }
    }
  }
  #pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
    #pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
  const float iv = inv[nbase + c];
  #pragma unroll
  for (uint d = 0; d < 16u; ++d) {
    uint w = 0u;
    #pragma unroll
    for (uint b = 0; b < 4u; ++b)
      w |= ((uint)(uchar)convert_char_sat_rte((v[4u * d + b] * (1.0f / 32.0f)) * iv)) << (8u * b);
    tout[c * 257u + 16u * l + d] = w;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (uint i = 0; i < 16u; ++i) {
    const uint t = lid + 256u * i;
    out[(size_t)(blk * 256u + t / 16u) * ldo + nc + (t % 16u)] = tout[(t % 16u) * 257u + t / 16u];
  }
}

// ===========================================================================
// pw8_requant_rot3: the weight rotation on the DPAS.
//
// H_1024 = H_16 (x) H_64 with k = 64 g + j (Sylvester order, the order the
// butterfly FWHT produces). So per column, per 1024-k block:
//
//   W'[64 g' + j'] = sum_g H16[g', g] * s_g * Z_g[j'],
//   Z_g = A_g q_g,   A_g = H64 diag(d[64 g .. 64 g + 63])   (int8, +-1)
//
// q_g is the checkpoint's int4 group, s_g its ONE scale. Z_g is an exact int32
// product on the i8 x i4 k32 DPAS, with A_g in the A slot and the layout-0
// qweight words in the B slot exactly as pw4a8_gemm uses them. The 16-point
// transform across groups is 4 in-register stages, ascending, on s_g * Z_g --
// which equals the CPU butterfly's value after its first 6 stages exactly
// (sums of s times small integers), so the result is bit-identical to the
// CPU FWHT.
//
// Work-group: 8 sub-groups x 16 columns x one 1024-k block. Sub-group a owns
// rows j' = 8a .. 8a + 7 of every group; lane = column.
// Needs 256 GRF (16 int8 accumulators live): it lives in the GEMM's build.
// ===========================================================================
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_requant_rot3(__global const uint* restrict qw,      // [K/8][Nfull]
                               __global const half* restrict sc,      // [K/64][Nfull]
                               __global const ushort* restrict amat,  // int8 [K/64][64][64]
                               __global const float* restrict inv,    // [Nfull]
                               __global uint* restrict out,           // [K/4][ldo]
                               uint n0, uint Nfull, uint K, uint ldo) {
  const uint a = get_sub_group_id();           // row block j' = 8a .. 8a+7
  const uint l = get_sub_group_local_id();     // column
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;

  const int q_w = (int)(Nfull * 4u), q_h = (int)(K / 8u), q_p = (int)(Nfull * 4u);
  const int a_w = 64, a_h = (int)K, a_p = 64;

  int8 z[16];
  float sg[16];
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const uint gi = blk * 16u + g;             // global group index
    ushort af[16];
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)amat, a_w, a_h, a_p,
                                              (int2)(0, (int)(gi * 64u + 8u * a)), af);
    uint wv[8];
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)qw, q_w, q_h, q_p,
                                              (int2)((int)nbase, (int)(gi * 8u)), wv);
    int8 acc = (int8)(0);
    #pragma unroll
    for (uint ks = 0; ks < 2u; ++ks) {
      const short8 av = as_short8(vload8(0, af + 8u * ks));
      const int4 bv = as_int4(vload4(0, wv + 4u * ks) ^ (uint4)(0x88888888u));
      acc = intel_sub_group_i8_i4_matrix_mad_k32(av, bv, acc);
    }
    z[g] = acc;
    sg[g] = vload_half((size_t)gi * Nfull + nbase + l, sc);
  }

  const float iv = inv[nbase + l];
  #pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    float t[16];
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int8 zz = z[g];
      const int zr = r == 0 ? zz.s0 : r == 1 ? zz.s1 : r == 2 ? zz.s2 : r == 3 ? zz.s3
                   : r == 4 ? zz.s4 : r == 5 ? zz.s5 : r == 6 ? zz.s6 : zz.s7;
      t[g] = (float)zr * sg[g];
    }
    #pragma unroll
    for (uint h = 1u; h < 16u; h <<= 1) {
      #pragma unroll
      for (uint g = 0; g < 16u; ++g) {
        if ((g & h) == 0u) {
          const float p = t[g], e = t[g + h];
          t[g] = p + e;
          t[g + h] = p - e;
        }
      }
    }
    // element k' = 64 g' + 8 a + r  ->  dword row 16 g' + 2 a + r / 4, byte r % 4.
    // The packed dwords reuse z[g].s0 (r = 0..3) and z[g].s4 (r = 4..7): row r
    // of z is read above before any write below touches it, and .s0 / .s4 are
    // only written at or after their own row's iteration.
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int byte = (int)(uint)(uchar)convert_char_sat_rte((t[g] * (1.0f / 32.0f)) * iv);
      if (r == 0u) z[g].s0 = byte;
      else if (r < 4u) z[g].s0 |= byte << (8u * r);
      else if (r == 4u) z[g].s4 = byte;
      else z[g].s4 |= byte << (8u * (r - 4u));
    }
  }
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const size_t row = (size_t)blk * 256u + 16u * g + 2u * a;
    out[row * ldo + nc + l] = (uint)z[g].s0;
    out[(row + 1u) * ldo + nc + l] = (uint)z[g].s4;
  }
}

// pw8_requant_rot4: pw8_requant_rot3 with the int4 tile and scales staged in SLM
// once per work-group -- rot3 had all 8 sub-groups re-read them from global.
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_requant_rot4(__global const uint* restrict qw,      // [K/8][Nfull]
                               __global const half* restrict sc,      // [K/64][Nfull]
                               __global const ushort* restrict amat,  // int8 [K/64][64][64]
                               __global const float* restrict inv,    // [Nfull]
                               __global uint* restrict out,           // [K/4][ldo]
                               uint n0, uint Nfull, uint K, uint ldo) {
  const uint a = get_sub_group_id();           // row block j' = 8a .. 8a+7
  const uint l = get_sub_group_local_id();     // column
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;

  const int a_w = 64, a_h = (int)K, a_p = 64;

  // the block's int4 tile [128 qweight rows][16 cols] and its 16 x 16 scales,
  // loaded ONCE per work-group (coalesced) instead of once per sub-group
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  const uint lid = get_local_id(0);
  #pragma unroll
  for (uint i = 0; i < 16u; ++i) {
    const uint t = lid + 128u * i;
    tin[(t % 16u) * 129u + t / 16u] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  #pragma unroll
  for (uint i = 0; i < 2u; ++i) {
    const uint t = lid + 128u * i;
    tsc[(t / 16u) * 17u + (t % 16u)] =
        vload_half((size_t)(blk * 16u + t / 16u) * Nfull + nbase + (t % 16u), sc);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int8 z[16];
  float sg[16];
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const uint gi = blk * 16u + g;             // global group index
    ushort af[16];
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)amat, a_w, a_h, a_p,
                                              (int2)(0, (int)(gi * 64u + 8u * a)), af);
    uint wv[8];
    #pragma unroll
    for (uint i = 0; i < 8u; ++i) wv[i] = tin[l * 129u + 8u * g + i];
    int8 acc = (int8)(0);
    #pragma unroll
    for (uint ks = 0; ks < 2u; ++ks) {
      const short8 av = as_short8(vload8(0, af + 8u * ks));
      const int4 bv = as_int4(vload4(0, wv + 4u * ks) ^ (uint4)(0x88888888u));
      acc = intel_sub_group_i8_i4_matrix_mad_k32(av, bv, acc);
    }
    z[g] = acc;
    sg[g] = tsc[g * 17u + l];
  }

  const float iv = inv[nbase + l];
  #pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    float t[16];
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int8 zz = z[g];
      const int zr = r == 0 ? zz.s0 : r == 1 ? zz.s1 : r == 2 ? zz.s2 : r == 3 ? zz.s3
                   : r == 4 ? zz.s4 : r == 5 ? zz.s5 : r == 6 ? zz.s6 : zz.s7;
      t[g] = (float)zr * sg[g];
    }
    #pragma unroll
    for (uint h = 1u; h < 16u; h <<= 1) {
      #pragma unroll
      for (uint g = 0; g < 16u; ++g) {
        if ((g & h) == 0u) {
          const float p = t[g], e = t[g + h];
          t[g] = p + e;
          t[g + h] = p - e;
        }
      }
    }
    // element k' = 64 g' + 8 a + r  ->  dword row 16 g' + 2 a + r / 4, byte r % 4.
    // The packed dwords reuse z[g].s0 (r = 0..3) and z[g].s4 (r = 4..7): row r
    // of z is read above before any write below touches it, and .s0 / .s4 are
    // only written at or after their own row's iteration.
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int byte = (int)(uint)(uchar)convert_char_sat_rte((t[g] * (1.0f / 32.0f)) * iv);
      if (r == 0u) z[g].s0 = byte;
      else if (r < 4u) z[g].s0 |= byte << (8u * r);
      else if (r == 4u) z[g].s4 = byte;
      else z[g].s4 |= byte << (8u * (r - 4u));
    }
  }
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const size_t row = (size_t)blk * 256u + 16u * g + 2u * a;
    out[row * ldo + nc + l] = (uint)z[g].s0;
    out[(row + 1u) * ldo + nc + l] = (uint)z[g].s4;
  }
}

// pw8_requant_rot4x: rot4 with runtime switches for attribution only.
// mode bit 0: no global stores; bit 1: no A reads and no DPAS.
__attribute__((reqd_work_group_size(128, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw8_requant_rot4x(__global const uint* restrict qw,      // [K/8][Nfull]
                               __global const half* restrict sc,      // [K/64][Nfull]
                               __global const ushort* restrict amat,  // int8 [K/64][64][64]
                               __global const float* restrict inv,    // [Nfull]
                               __global uint* restrict out,           // [K/4][ldo]
                               uint n0, uint Nfull, uint K, uint ldo, uint mode) {
  const uint a = get_sub_group_id();           // row block j' = 8a .. 8a+7
  const uint l = get_sub_group_local_id();     // column
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;

  const int a_w = 64, a_h = (int)K, a_p = 64;

  // the block's int4 tile [128 qweight rows][16 cols] and its 16 x 16 scales,
  // loaded ONCE per work-group (coalesced) instead of once per sub-group
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  const uint lid = get_local_id(0);
  #pragma unroll
  for (uint i = 0; i < 16u; ++i) {
    const uint t = lid + 128u * i;
    tin[(t % 16u) * 129u + t / 16u] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  #pragma unroll
  for (uint i = 0; i < 2u; ++i) {
    const uint t = lid + 128u * i;
    tsc[(t / 16u) * 17u + (t % 16u)] =
        vload_half((size_t)(blk * 16u + t / 16u) * Nfull + nbase + (t % 16u), sc);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int8 z[16];
  float sg[16];
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const uint gi = blk * 16u + g;             // global group index
    ushort af[16];
    if (!(mode & 2u))
      intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)amat, a_w, a_h, a_p,
                                                (int2)(0, (int)(gi * 64u + 8u * a)), af);
    uint wv[8];
    #pragma unroll
    for (uint i = 0; i < 8u; ++i) wv[i] = tin[l * 129u + 8u * g + i];
    int8 acc = (int8)(0);
    if (mode & 2u) {
      acc = as_int8(vload8(0, wv));
    } else {
      #pragma unroll
      for (uint ks = 0; ks < 2u; ++ks) {
        const short8 av = as_short8(vload8(0, af + 8u * ks));
        const int4 bv = as_int4(vload4(0, wv + 4u * ks) ^ (uint4)(0x88888888u));
        acc = intel_sub_group_i8_i4_matrix_mad_k32(av, bv, acc);
      }
    }
    z[g] = acc;
    sg[g] = tsc[g * 17u + l];
  }

  const float iv = inv[nbase + l];
  #pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    float t[16];
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int8 zz = z[g];
      const int zr = r == 0 ? zz.s0 : r == 1 ? zz.s1 : r == 2 ? zz.s2 : r == 3 ? zz.s3
                   : r == 4 ? zz.s4 : r == 5 ? zz.s5 : r == 6 ? zz.s6 : zz.s7;
      t[g] = (float)zr * sg[g];
    }
    #pragma unroll
    for (uint h = 1u; h < 16u; h <<= 1) {
      #pragma unroll
      for (uint g = 0; g < 16u; ++g) {
        if ((g & h) == 0u) {
          const float p = t[g], e = t[g + h];
          t[g] = p + e;
          t[g + h] = p - e;
        }
      }
    }
    // element k' = 64 g' + 8 a + r  ->  dword row 16 g' + 2 a + r / 4, byte r % 4.
    // The packed dwords reuse z[g].s0 (r = 0..3) and z[g].s4 (r = 4..7): row r
    // of z is read above before any write below touches it, and .s0 / .s4 are
    // only written at or after their own row's iteration.
    #pragma unroll
    for (uint g = 0; g < 16u; ++g) {
      const int byte = (int)(uint)(uchar)convert_char_sat_rte((t[g] * (1.0f / 32.0f)) * iv);
      if (r == 0u) z[g].s0 = byte;
      else if (r < 4u) z[g].s0 |= byte << (8u * r);
      else if (r == 4u) z[g].s4 = byte;
      else z[g].s4 |= byte << (8u * (r - 4u));
    }
  }
  #pragma unroll
  for (uint g = 0; g < 16u; ++g) {
    const size_t row = (size_t)blk * 256u + 16u * g + 2u * a;
    if (!(mode & 1u) || (uint)z[g].s0 == 0x9E3779B9u) {
      out[row * ldo + nc + l] = (uint)z[g].s0;
      out[(row + 1u) * ldo + nc + l] = (uint)z[g].s4;
    }
  }
}
