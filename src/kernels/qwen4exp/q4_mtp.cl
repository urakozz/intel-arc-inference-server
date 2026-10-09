// q4_mtp.cl - spec 21e: the Qwen3.8-Flash-Next MTP head's fusion (spec 21 §2.5, §4.5): vLLM's form
// (V/nvidia/mtp.py:293-317; tools/oracle/qwen4exp_mtp.py MtpHead.fuse, 21a's port):
//
//   e  = fc_embedding(pre_fc_norm_embedding(embed(t)))                       one 2560-row
//   h  = fc_hidden(pre_fc_norm_hidden(R) viewed [4][2560]) per stream        R: the main model's (or the previous
//                                                                             draft step's) PRE-mixer 4-stream H
//   H  = h + e on every stream (unit injection: the product by 1 is exact)   the head layer's input, materialised
//
// Two entry points, the two GEMVs (gemv_bf16 {2560, 2560}: fc_embedding at M rows, fc_hidden at M = 4 per row - a
// row's 4 streams) between them:
//
//   q4_mtp_norm(ctrl, R, embed, w_e, w_h, xe, xh [, ids])   grid (1 + 4, M), WG 256
//     work-group (0, m): t = cur_token[m] (MTP_PF: ids[m]); xe[m][k] = rne(e_k x rstd_e x w_e[k]), e = embed[t] (bf16)
//     work-group (1 + s, m): stream s of row m of R: xh[m][s x 2560 + k] = rne(R[m][s x 2560 + k] x rstd x w_h[s x
//       2560 + k]); MTP_NORM_SINGLE 1: rstd over the row's 10240 values - vLLM's one GemmaRMSNorm(hc x H)
//       (mtp.py:227-229; decision 4's default) - each of the 4 work-groups sums the whole row in the same order;
//       MTP_NORM_SINGLE 0: over the stream's 2560 (the per-stream reading of the same [10240] weight)
//     w_e [2560], w_h [10240]: the (1 + w) norms baked to fp32 at load (loader/qwen4exp_layout.h Q4MtpFcOffsets)
//   q4_mtp_fuse(ctrl, fe, fh, H)                            grid (10240 / 256, M), WG 256
//     H[m][s x 2560 + k] = rne(rf(fh[4 m + s][k]) + rf(fe[m][k])) - the two linears' bf16 outputs (fp32 GEMV sums
//     rounded once, torch's linear), then their bf16 add (the reference's `h + e.repeat(1, hc)`)
//
// Every norm is q4_hc's: lane i accumulates fma(x, x, acc) over k = i + 256 j, j ascending, the pairwise tree 128 .. 1,
// rstd = 1 / sqrt(sum / N + 1e-6) (correctly rounded, never rsqrt), the product (x x rstd) x w in fp32, rounded once.
// MTP_PF (the prefill's head pass, M = kPfC): the rows' tokens from the chunk's id buffer `ids` (the last argument) -
// Control holds 8. Rows m >= n_active exit (uniform). Plain OpenCL C (no sub-group functions): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_NACT) || !defined(CTRL_CUR)
#error "q4_mtp: CTRL_NACT / CTRL_CUR must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef MTP_NORM_SINGLE
#error "q4_mtp: MTP_NORM_SINGLE (1: one RMS over the row's 10240 values, 0: per stream) must be defined"
#endif
#define HIDDEN 2560
#define HC 4
#define HCN (HC * HIDDEN)
#define WG 256
#define PER_LANE (HIDDEN / WG)   /* 10 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}

// The work-group's sum of x^2 over `n` values of `v` (lane i: k = i + 256 j ascending; the tree), then rstd.
inline float wg_rstd(__global const ushort* restrict v, uint n, __local float* red, uint lid) {
  float acc = 0.0f;
  for (uint k = lid; k < n; k += WG) {
    const float x = bf16f(v[k]);
    acc = fma(x, x, acc);
  }
  red[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = 1.0f / sqrt(red[0] / (float)n + 1e-6f);
  barrier(CLK_LOCAL_MEM_FENCE);   // red is reused
  return r;
}

#ifdef MTP_PF
#define MTP_TOK(m) ids[m]
#else
#define MTP_TOK(m) ctrl[CTRL_CUR + (m)]
#endif

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_mtp_norm(__global const uint* restrict ctrl, __global const ushort* restrict R,
                          __global const ushort* restrict embed, __global const float* restrict w_e,
                          __global const float* restrict w_h, __global ushort* restrict xe, __global ushort* restrict xh
#ifdef MTP_PF
                          , __global const uint* restrict ids
#endif
                          ) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float red[WG];
  if (g == 0) {
    const uint t = MTP_TOK(m);
    __global const ushort* restrict e = embed + (size_t)t * HIDDEN;
    const float r = wg_rstd(e, HIDDEN, red, lid);
    for (uint j = 0; j < PER_LANE; ++j) {
      const uint k = lid + WG * j;
      xe[(size_t)m * HIDDEN + k] = rne_bf16(bf16f(e[k]) * r * w_e[k]);
    }
    return;
  }
  const uint s = g - 1;
  __global const ushort* restrict row = R + (size_t)m * HCN;
#if MTP_NORM_SINGLE
  const float r = wg_rstd(row, HCN, red, lid);
#else
  const float r = wg_rstd(row + (size_t)s * HIDDEN, HIDDEN, red, lid);
#endif
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint c = s * HIDDEN + lid + WG * j;
    xh[(size_t)m * HCN + c] = rne_bf16(bf16f(row[c]) * r * w_h[c]);
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_mtp_fuse(__global const uint* restrict ctrl, __global const float* restrict fe,
                          __global const float* restrict fh, __global ushort* restrict H) {
  const uint c = get_group_id(0) * WG + get_local_id(0);
  const uint m = get_group_id(1);
  if (m >= n_rows(ctrl)) return;   // uniform
  const uint s = c / HIDDEN, k = c % HIDDEN;
  H[(size_t)m * HCN + c] = rne_bf16(rf(fh[((size_t)m * HC + s) * HIDDEN + k]) + rf(fe[(size_t)m * HIDDEN + k]));
}
