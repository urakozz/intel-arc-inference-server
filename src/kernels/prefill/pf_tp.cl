// pf_tp.cl - spec 17c: tensor-parallel PREFILL's residual fold (`b70-decode --parallel tp --prefill`). A new source,
// so every existing binary keeps its bytes.
//
// After a row-parallel linear (out_proj, o_proj, down) each card holds its half of the output, the product of its K
// rows - card r's fp32 [M][K] part. One card's prefill runs that linear over the whole K in one register chain and
// pf_res_fold (pf_prep.cl, S_PREV 1) folds its one slice. Under TP the two halves meet here: `p0` is card 0's part
// and `p1` card 1's (one of them this card's own, the other the landing buffer the peer's crossing filled), and
//
//   mixer_b = rne_bf16(p0[m][k] + p1[m][k])      (fp32: commutative, so both cards compute the same bits - T1)
//   r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))
//   resid[m][k] = r_b
//   sumsq[g][m] = tree_{lanes}(Sigma_{k in chunk g} f32(r_b)^2)
//
// pf_res_fold's chain at S_PREV 1 with its one slice replaced by the two parts' fp32 sum (acc = 0 + p0, + p1: the
// order its slice loop would add two slices in), then the same RNE steps, `fma` square-accumulate and `red[WG]`
// tree - so pf_norm_finish downstream is unchanged. Not bitwise one card (the K split changes the GEMM's chain; the
// result is judged, plan 17c), but bitwise between the cards.
#ifndef K
#define K 5120
#endif
#ifndef FOLD_G
#define FOLD_G 20
#endif
#define WG_FOLD 256
#define FOLD_CHUNK ((K + FOLD_G - 1) / FOLD_G)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// Grid (FOLD_G, M), WG 256: pf_res_fold's.
__attribute__((reqd_work_group_size(WG_FOLD, 1, 1)))
__kernel void pf_tp_res_fold(__global const float* restrict p0, __global const float* restrict p1,
                             __global ushort* restrict resid, __global float* restrict sumsq, uint m_count) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_FOLD];
  __global ushort* rp = resid + (size_t)m * K;

  const uint k0 = g * FOLD_CHUNK;
  uint k1 = k0 + FOLD_CHUNK;
  if (k1 > K) k1 = K;

  float acc2 = 0.f;
  for (uint k = k0 + lid; k < k1; k += WG_FOLD) {
    float acc = 0.f;
    acc += p0[(size_t)m * K + k];
    acc += p1[(size_t)m * K + k];
    const ushort r_b = rne_bf16(bf16f(rp[k]) + bf16f(rne_bf16(acc)));
    rp[k] = r_b;
    const float v = bf16f(r_b);
    acc2 = fma(v, v, acc2);
  }
  red[lid] = acc2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_FOLD / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sumsq[(size_t)g * m_count + m] = red[0];
}
