// pf_int8 - spec 5's int8 prefill linears
// (docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md): the h8 path of
// docs/probe-w4a8-2026-09-23.md sections 14-15, promoted from
// tools/probe/probe_w8a8.cl with the arithmetic unchanged. One source, built
// per family by a define:
//   PF_QUANT_HAD  (NBLK)       pf_quant_had    bf16 x -> int8 (x R_K), per-token scale
//   PF_REQUANT_ROT (LAYOUT)    pf_requant_rot  int4 g64 -> per-channel int8 of (W R_K)
//                              pf_colmax_rot   max |W R_K| per column (load-time scales)
//   PF_GEMM_I8   (SILU_EPI)    pf_gemm_i8      i8 x i8, scales in the epilogue only
// R_K = D_K blockdiag(H_1024) / 32 over K. D_K comes from
// runtime/prefill/int8_signs.h. The quantiser holds block element 16 j + lane
// and the requant holds 64 lane + j; both run the canonical Sylvester FWHT on
// the natural index with ascending stages, so they rotate identically
// (pf_int8_test's cross case pins it).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

#ifdef PF_QUANT_HAD
#ifndef NBLK
#error "pf_quant_had: NBLK (= K / 1024) must be defined"
#endif
// One sub-group per 1024-block, NBLK sub-groups per token row. Lane l's value j
// is element 16 j + l of the block (sub-group block reads). FWHT stages
// h = 1..8 are lane shuffles and 16..512 register butterflies, ascending.
__attribute__((reqd_work_group_size(16 * NBLK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_quant_had(__global const ushort* restrict x, __global const float* restrict sgn,
                           __global char* restrict xq, __global float* restrict xs, uint ldx) {
  const uint K = NBLK * 1024u;
  const uint m = get_group_id(0);
  const uint b = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  __global const ushort* src = x + (size_t)m * ldx + b * 1024u;
  __local float red[NBLK];
  float v[64];
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    const ushort8 u = intel_sub_group_block_read_us8(src + r * 128u);
    const float8 sg = as_float8(intel_sub_group_block_read8(
        (__global const uint*)(sgn + b * 1024u + r * 128u)));
    v[8u * r + 0u] = bf16f(u.s0) * sg.s0; v[8u * r + 1u] = bf16f(u.s1) * sg.s1;
    v[8u * r + 2u] = bf16f(u.s2) * sg.s2; v[8u * r + 3u] = bf16f(u.s3) * sg.s3;
    v[8u * r + 4u] = bf16f(u.s4) * sg.s4; v[8u * r + 5u] = bf16f(u.s5) * sg.s5;
    v[8u * r + 6u] = bf16f(u.s6) * sg.s6; v[8u * r + 7u] = bf16f(u.s7) * sg.s7;
  }
#pragma unroll
  for (uint h = 1u; h < 16u; h <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      const float o = sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
  }
#pragma unroll
  for (uint hj = 1u; hj < 64u; hj <<= 1) {
#pragma unroll
    for (uint j = 0; j < 64u; ++j) {
      if ((j & hj) == 0u) {
        const float a = v[j], c = v[j + hj];
        v[j] = a + c;
        v[j + hj] = a - c;
      }
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 64u; ++j) {
    v[j] *= (1.0f / 32.0f);
    amax = fmax(amax, fabs(v[j]));
  }
  amax = sub_group_reduce_max(amax);
  if (l == 0u) red[b] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax = 0.0f;
  for (uint i = 0; i < (uint)NBLK; ++i) amax = fmax(amax, red[i]);
  const float scale = amax / 127.0f;
  const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
  if (b == 0u && l == 0u) xs[m] = scale;
  __global uchar* dst = (__global uchar*)(xq + (size_t)m * K + b * 1024u);
#pragma unroll
  for (uint r = 0; r < 8u; ++r) {
    uchar8 o;
    o.s0 = (uchar)convert_char_sat_rte(v[8u * r + 0u] * iv);
    o.s1 = (uchar)convert_char_sat_rte(v[8u * r + 1u] * iv);
    o.s2 = (uchar)convert_char_sat_rte(v[8u * r + 2u] * iv);
    o.s3 = (uchar)convert_char_sat_rte(v[8u * r + 3u] * iv);
    o.s4 = (uchar)convert_char_sat_rte(v[8u * r + 4u] * iv);
    o.s5 = (uchar)convert_char_sat_rte(v[8u * r + 5u] * iv);
    o.s6 = (uchar)convert_char_sat_rte(v[8u * r + 6u] * iv);
    o.s7 = (uchar)convert_char_sat_rte(v[8u * r + 7u] * iv);
    intel_sub_group_block_write_uc8(dst + r * 128u, o);
  }
}
#endif  // PF_QUANT_HAD

#ifdef PF_REQUANT_ROT
#ifndef LAYOUT
#error "pf_requant_rot: LAYOUT (0 GPTQ-native, 1 tiled) must be defined"
#endif
// One work-group = 16 columns x one 1024-k block; sub-group c owns column c,
// lane l owns k = 64 l .. 64 l + 63 (one scale group). Stages h = 1..32 are
// register butterflies, h = 64..512 lane shuffles, ascending. The int4 tile
// goes through SLM column-major at pitch 129 (conflict-free); signs arrive as a
// bitmask, applied as a sign-bit xor (bit-identical to multiplying by -1).
inline void pf_rot_load(__global const uint* restrict qw, __global const half* restrict sc,
                        __global const uint* restrict sbits, uint nbase, uint blk, uint Nfull,
                        uint K, __local uint* tin, __local float* tsc, float* v) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
#if LAYOUT == 0
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    tin[(t % 16u) * 129u + t / 16u] = qw[(size_t)(blk * 128u + t / 16u) * Nfull + nbase + (t % 16u)];
  }
  tsc[(lid % 16u) * 17u + lid / 16u] =
      vload_half((size_t)(blk * 16u + lid / 16u) * Nfull + nbase + (lid % 16u), sc);
#else
  // The WG's 16 columns are exactly tile nt = nbase / 16. Tile (nt, group gi)
  // is 136 u32 at (nt G + gi) 136: word j of lane c at j 16 + c, then 16 f16
  // scales (common/repack.h, repack_int4_layout1). Row r of the block is
  // group blk 16 + r / 8, word r % 8.
  const uint G = K / 64u;
  __global const uint* tiles = qw + (size_t)(nbase / 16u) * G * 136u;
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    const uint t = lid + 256u * i;
    const uint r = t / 16u, cc = t % 16u;
    tin[cc * 129u + r] = tiles[(size_t)(blk * 16u + r / 8u) * 136u + (r % 8u) * 16u + cc];
  }
  {
    const uint g = lid / 16u, cc = lid % 16u;
    const __global ushort* sp =
        (const __global ushort*)(tiles + (size_t)(blk * 16u + g) * 136u + 128u);
    tsc[cc * 17u + g] = (float)as_half(sp[cc]);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE);

  const float s = tsc[c * 17u + l];
  const uint2 sb = vload2(0, sbits + blk * 32u + 2u * l);
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
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_requant_rot(__global const uint* restrict qw, __global const half* restrict sc,
                             __global const uint* restrict sbits, __global const float* restrict inv,
                             __global uint* restrict out, uint n0, uint Nfull, uint K, uint ldo) {
  const uint lid = get_local_id(0);
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nc = get_group_id(0) * 16u;
  const uint nbase = n0 + nc;
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  __local uint tout[16 * 257];
  float v[64];
  pf_rot_load(qw, sc, sbits, nbase, blk, Nfull, K, tin, tsc, v);
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

// max |(W R_K)[k, n]| over one block, into colmax[n] as float bits (every value
// is >= 0, so unsigned order is float order). The caller zero-fills colmax and
// runs one launch over all blocks; ws and inv are derived on the host.
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_colmax_rot(__global const uint* restrict qw, __global const half* restrict sc,
                            __global const uint* restrict sbits, __global uint* restrict colmax,
                            uint n0, uint Nfull, uint K) {
  const uint c = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  const uint blk = get_group_id(1);
  const uint nbase = n0 + get_group_id(0) * 16u;
  __local uint tin[16 * 129];
  __local float tsc[16 * 17];
  float v[64];
  pf_rot_load(qw, sc, sbits, nbase, blk, Nfull, K, tin, tsc, v);
  float amax = 0.0f;
#pragma unroll
  for (uint j = 0; j < 64u; ++j) amax = fmax(amax, fabs(v[j] * (1.0f / 32.0f)));
  amax = sub_group_reduce_max(amax);
  if (l == 0u) atomic_max((volatile __global uint*)(colmax + nbase + c), as_uint(amax));
}
#endif  // PF_REQUANT_ROT

#ifdef PF_GEMM_I8
#ifndef SILU_EPI
#define SILU_EPI 0
#endif
// pw8_gemm (tools/probe/probe_w8a8.cl), the gsweep tile: WG 256 x 128, 32
// sub-groups 8 (m) x 4 (n), per sub-group 32 m x 32 n, B int8 VNNI-4, int32
// accumulation over the whole K. Mainloop and plain epilogue copied verbatim
// with only the probe's `sig` argument and its store removed.
#define CHUNK 64
#define WG_M 256
#define WG_N 128
#define SG_M 32
#define SG_N 32
#define NA 4
#define NB 2
#if SILU_EPI
// pf_gemm.cl's chain, verbatim (the numerics contract with pf_prep.cl).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
#endif

__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gemm_i8(__global const ushort* restrict xq16, __global const float* restrict xs,
                         __global const uint* restrict w8, __global const float* restrict ws,
                         __global float* restrict C, uint M, uint K, uint N, uint ldxq, uint ldb,
                         uint ldc) {
  const uint m0 = get_group_id(0) * WG_M;
  const uint n0 = get_group_id(1) * WG_N;
  const uint s = get_sub_group_id();
  const uint mb = m0 + SG_M * (s >> 2);
  const uint nb = n0 + SG_N * (s & 3u);

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

#if SILU_EPI
  // A sub-group's two atoms are nb and nb + 16 with nb a multiple of 32, so
  // atom 0 is a gate block and atom 1 its up block (gate||up's 16-column
  // interleave). The x column of gate column nb + lane is nb / 2 + lane.
  // `C` carries bf16 x and `ldc` carries ldx. The descriptor is o_* because the
  // mainloop above already names the A operand's x_*.
  __global ushort* restrict X = (__global ushort*)C;
  const int o_w = (int)(N * 2u), o_h = (int)M, o_p = (int)(ldc * 2u);
  const float wg = as_float(intel_sub_group_block_read((__global const uint*)(ws + nb)));
  const float wu = as_float(intel_sub_group_block_read((__global const uint*)(ws + nb + 16u)));
#define PF_SILU_ROW_I8(r)                                                     \
  do {                                                                        \
    const float xsr = xs[mb + 8u * (uint)a + r];                              \
    const float g = (float)acc[a][0].s##r * (wg * xsr);                       \
    const float u = (float)acc[a][1].s##r * (wu * xsr);                       \
    const ushort g_b = rne_bf16(g), u_b = rne_bf16(u);                        \
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));                        \
    xv[r] = rne_bf16(bf16f(s_b) * bf16f(u_b));                                \
  } while (0)
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    ushort xv[8];
    PF_SILU_ROW_I8(0); PF_SILU_ROW_I8(1); PF_SILU_ROW_I8(2); PF_SILU_ROW_I8(3);
    PF_SILU_ROW_I8(4); PF_SILU_ROW_I8(5); PF_SILU_ROW_I8(6); PF_SILU_ROW_I8(7);
    intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)X, o_w, o_h, o_p,
                                               (int2)((int)(nb >> 1), (int)(mb + 8u * (uint)a)), xv);
  }
#undef PF_SILU_ROW_I8
#else
  // pw8_gemm's epilogue, unchanged.
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
#endif
}
#endif  // PF_GEMM_I8
