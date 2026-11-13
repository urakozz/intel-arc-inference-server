// df_proj_wide.cl - the DFlash context projection at prefill width (runtime::dflash::DFlashDrafter::project_wide):
// a prompt's staged tap rows through pf_gemm, hundreds of rows a pass over the weights, where the projection list
// (df_prep.cl's df_norm / df_ctx_store around gemm_sm at M 8) re-reads them once per 8 rows. Its own binary, so the
// shared df_prep binaries keep their bytes; the helpers below are df_prep.cl's, copied (same order, FP_CONTRACT off).
//
//   df_w_slab_i8    grid (NS / 256, K / 16), WG 256: NS columns from n0 of an int8 linear ([N/16][K/16][16 k][16 n],
//                   loader/lm_head_int8.h) as pf_gemm's B, bf16 [K][NS] - the exact integers, unscaled
//   df_w_slab_bf16  grid (NS / 256, K / 8), WG 256: the same of a bf16 linear in gemv_bf16's tiled layout - a copy
//   df_w_norm       grid (1, rows), WG 256: df_norm over the GEMM's rows, the int8 column scales folded in first
//                   (`scaled`; gemm_sm_i8w scales its output, pf_gemm does not); writes the bf16 x only
//   df_w_ctx_store  grid (2 NKV, rows), WG HD: df_ctx_store for ONE layer - row r is position pos0 + r, k (k_norm,
//                   RoPE) and v into ring slot (pos0 + r) % RING; kv is that layer's GEMM output [rows][2 KVN],
//                   scaled as above
// A row's values depend on its own row only, so the ring does not depend on how a prompt was cut into passes.
#pragma OPENCL FP_CONTRACT OFF

#ifndef H
#error "H (the drafter's hidden size) must be defined"
#endif
#ifndef WG
#define WG 256
#endif
#ifndef EPS
#define EPS 1e-06f
#endif
#define KVN (NKV * HD)
#define HALF (HD / 2)

inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)0x7FC0;   // NaN
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// w: the weight from its first column's tile on (a multiple of 16 columns into the tensor); K, NS as launched.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void df_w_slab_i8(__global const char* restrict w, __global ushort* restrict out, const uint n0,
                           const uint K, const uint NS) {
  const uint j = get_global_id(0), k16 = get_group_id(1), n = n0 + j;
  __global const char* restrict blk = w + ((size_t)(n / 16) * (K / 16) + k16) * 256 + n % 16;
  __global ushort* restrict o = out + (size_t)k16 * 16 * NS + j;
  for (uint i = 0; i < 16; ++i) o[(size_t)i * NS] = (ushort)(as_uint((float)blk[i * 16]) >> 16);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void df_w_slab_bf16(__global const ushort* restrict w, __global ushort* restrict out, const uint n0,
                             const uint K, const uint NS) {
  const uint j = get_global_id(0), k8 = get_group_id(1), n = n0 + j;
  __global const ushort* restrict blk = w + ((size_t)(n / 16) * (K / 8) + k8) * 128 + n % 16;
  __global ushort* restrict o = out + (size_t)k8 * 8 * NS + j;
  for (uint i = 0; i < 8; ++i) o[(size_t)i * NS] = blk[i * 16];
}

// The fixed Σ: lane partials then strides WG/2 .. 1 (df_prep.cl's wg_sum).
inline float wg_sum(float v, __local float* red) {
  const uint lid = get_local_id(0);
  red[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (lid < s) red[lid] = red[lid] + red[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_w_norm(__global const float* restrict src, __global const float* restrict scales,
                        __global const float* restrict w, __global ushort* restrict x, const uint scaled) {
  __local float red[WG];
  const uint r = get_group_id(1), lid = get_local_id(0);
  __global const float* s = src + (size_t)r * H;
  float a = 0.0f;
  for (uint i = lid; i < H; i += WG) {
    const float v = scaled ? s[i] * scales[i] : s[i];
    a = a + v * v;
  }
  const float ss = wg_sum(a, red);
  const float rstd = 1.0f / sqrt(ss / (float)H + EPS);
  for (uint i = lid; i < H; i += WG) {
    const float v = scaled ? s[i] * scales[i] : s[i];
    x[(size_t)r * H + i] = f2bf((v * rstd) * w[i]);
  }
}

// One head's RMSNorm over HD lanes and neox RoPE from the table [pos][cos HALF | sin HALF] (df_prep.cl's).
inline float head_norm_rope(float x, __global const float* restrict w, __global const float* restrict rope, uint pos,
                            __local float* y) {
  const uint i = get_local_id(0);
  y[i] = x * x;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = HD / 2; s > 0; s >>= 1) {
    if (i < s) y[i] = y[i] + y[i + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(y[0] / (float)HD + EPS);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float v = (x * rstd) * w[i];
  y[i] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const float* c = rope + (size_t)pos * HD;
  const uint j = i % HALF;
  const float o = i < HALF ? v * c[j] - y[i + HALF] * c[HALF + j] : v * c[j] + y[i - HALF] * c[HALF + j];
  barrier(CLK_LOCAL_MEM_FENCE);
  return o;
}

// kv: this layer's [rows][2 KVN]; scales: its k||v columns' [2 KVN]; k_norm [HD]; ring_k / ring_v: its [RING][NKV][HD].
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void df_w_ctx_store(__global const float* restrict kv, __global const float* restrict scales,
                             __global const float* restrict k_norm, __global const float* restrict rope,
                             __global ushort* restrict ring_k, __global ushort* restrict ring_v, const uint pos0,
                             const uint scaled) {
  __local float y[HD];
  const uint grp = get_group_id(0), r = get_group_id(1), i = get_local_id(0);
  const uint head = grp % NKV;
  const bool is_v = grp >= NKV;
  const uint pos = pos0 + r, slot = pos % RING;
  const uint col = (is_v ? KVN : 0) + head * HD + i;
  float x = kv[(size_t)r * 2 * KVN + col];
  if (scaled) x = x * scales[col];
  const size_t at = ((size_t)slot * NKV + head) * HD + i;
  if (is_v) {
    ring_v[at] = f2bf(x);
  } else {
    ring_k[at] = f2bf(head_norm_rope(x, k_norm, rope, pos, y));
  }
}
