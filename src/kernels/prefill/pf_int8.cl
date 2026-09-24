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
