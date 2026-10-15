// mg_pf_prep.cl - spec 23d: the two Muse-Glimmer prefill kernels mg_prep.cl at -DM=2048 cannot serve. Everything
// else between the slab GEMMs is mg_prep.cl at M 2048 (mg_norm, mg_post_add, mg_attn_prep at S 1, mg_silu_mul at S 1)
// and prefill/pf_prep.cl's pf_res_fold _Z at K 6656 (src/kernels/muse_kernels.h's spec 23d block names them):
//
//   mg_pf_embed(ids, embed, resid, sumsq)        EMBED_G   grid (1, C),            WG 256
//   mg_pf_attn_gate(gate, attn)                  GATE_N    grid (GATE_N / 256, C), WG 256
//
// mg_pf_embed is mg_embed's chain (mg_prep.cl, spec 23c) line for line, reading the chunk's ids from a buffer
// (mg_embed reads Control::cur_token, M <= 8): the gathered bf16 row through the weight-less norm (`x * pow(ms + 1e-5,
// -0.5)`, the correctly rounded 1 / sqrt, one rounding) - the residual - and the chunk Σ² layer 0's mg_norm reads
// ([EMBED_G][M], M = the chunk's pitch kPfC). The image / video placeholders embed as row 0 (M:1010-1013). An id >=
// VOCAB writes nothing (the host refuses such ids before the chunk: no debug_flag channel here).
//
// mg_pf_attn_gate is the output gate decode applies in mg_attn_reduce / mg_attn_eager: attn = rne(rne(o) x g) with
// g = bf16(sigmoid(gate_proj(xn))) as mg_attn_prep stored it (M:357-358), in place over the attention's bf16 rows -
// after either prefill attention build (mg_pf_attn.cl), so the two forms share it.
//
// No sub-group function: it builds as OpenCL 1.2 for the Mac's indicative run (tools/mac/clrun/muse_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 2048              /* the chunk's row pitch of sumsq (kPfC) */
#endif
#ifndef K
#define K 6656              /* hidden */
#endif
#define WG_ROW 256

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

#ifdef EMBED_G
#if !defined(VOCAB) || !defined(IMG_TOKEN) || !defined(VID_TOKEN)
#error "mg_pf_embed: VOCAB, IMG_TOKEN and VID_TOKEN must be defined"
#endif
#define EMBED_EPS 1e-5f
#define EMBED_CHUNK ((K + EMBED_G - 1) / EMBED_G)
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_pf_embed(__global const uint* restrict ids, __global const ushort* restrict embed,
                          __global ushort* restrict resid, __global float* restrict sumsq) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_ROW];
  __local float tot[1];
  uint id = ids[m];
  if (id == IMG_TOKEN || id == VID_TOKEN) id = 0u;
  if (id >= VOCAB) return;   // uniform: the host bounds the ids
  const __global ushort* row = embed + (size_t)id * K;
  __global ushort* rp = resid + (size_t)m * K;
  if (lid == 0) tot[0] = 0.0f;
  for (uint g = 0; g < EMBED_G; ++g) {   // pass 1: the raw row's chunk trees, added ascending
    const uint k0 = g * EMBED_CHUNK;
    uint k1 = k0 + EMBED_CHUNK;
    if (k1 > K) k1 = K;
    float acc2 = 0.f;
    for (uint k = k0 + lid; k < k1; k += WG_ROW) {
      const float v = bf16f(row[k]);
      acc2 = fma(v, v, acc2);
    }
    red[lid] = acc2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG_ROW / 2; stride > 0; stride >>= 1) {
      if (lid < stride) red[lid] += red[lid + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) tot[0] += red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(tot[0] / (float)K + EMBED_EPS);
  for (uint g = 0; g < EMBED_G; ++g) {   // pass 2: the normed row (the residual) and its chunk trees
    const uint k0 = g * EMBED_CHUNK;
    uint k1 = k0 + EMBED_CHUNK;
    if (k1 > K) k1 = K;
    float acc2 = 0.f;
    for (uint k = k0 + lid; k < k1; k += WG_ROW) {
      const ushort y = rne_bf16(bf16f(row[k]) * rstd);
      rp[k] = y;
      const float v = bf16f(y);
      acc2 = fma(v, v, acc2);
    }
    red[lid] = acc2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG_ROW / 2; stride > 0; stride >>= 1) {
      if (lid < stride) red[lid] += red[lid + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sumsq[(size_t)g * M + m] = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // EMBED_G

#ifdef GATE_N
#if GATE_N % WG_ROW != 0
#error "mg_pf_attn_gate: GATE_N must be whole 256-column work-groups"
#endif
__attribute__((reqd_work_group_size(WG_ROW, 1, 1)))
__kernel void mg_pf_attn_gate(__global const ushort* restrict gate, __global ushort* restrict attn) {
  const size_t i = (size_t)get_group_id(1) * GATE_N + get_group_id(0) * WG_ROW + get_local_id(0);
  attn[i] = rne_bf16(bf16f(attn[i]) * bf16f(gate[i]));
}
#endif  // GATE_N
