// q4_hc.cl - spec 21c: Qwen3.8-Flash-Next's hyper-connections (spec 21 §2.2, §4.1). The residual is H, 4
// streams x 2560 bf16 per row; a gated residual is THREE launches - q4_hc_combine_norm, gemv_bf16 {10240, 336}
// (down's 320 rows and block_inject's 4, padded; gemv_bf16.cl unchanged, fp32 out) and q4_hc_up_mix. Two families,
// each compiled only when its defines are given:
//
//   q4_hc_combine_norm(ctrl, H, src, inj, norm_w, xn)        HC_SRC   grid (4 streams, M), WG 256
//   q4_hc_up_mix(ctrl, down_f32, up, xn, x, inj)              UP_DOWN  grid (2560 / 16, M), WG 64
//
// **The pending combine (vLLM's fusion, V/nvidia/hyperconnection.py:177-203).** A block's output y and its
// inject weights inj are NOT folded into H by the block: the NEXT combine_norm folds them, once, then norms -
// so one launch does the reference's `H = H0 + y (x) inj` (M:1294-1301: the product rounded to bf16, then the
// add rounded) and the next gated residual's grouped norm. What is folded is the variant (the binary's name):
//   HC_SRC 0  _E   layer 0: H[s] = the embedding row (embed_gather's), every stream (M:1472's repeat)
//   HC_SRC 1  _S<S> the mixer's GEMV split-K slices: y = rne(sum_{j < SRC_S} slices[j][m][k]) ascending j
//   HC_SRC 2  _Y   the MoE block's bf16 output
//   HC_SRC 3  _X   nothing (H already materialised: the PLE block's output, the two-card hand-off's landing)
//   HC_NORM 0 _NN  combine only (the PLE layer's prologue, device 0's last launch on two cards): no xn
// then, unless _NN, per stream s:  xn[m][s][k] = rne(f32(H) x rstd_s x (1 + w)[s x 2560 + k])  with rstd_s over
// the stream's 2560 values: lane i accumulates fma(h, h, acc) over k = i + 256 j, j ascending, then the pairwise
// tree 128 .. 1; 1 / sqrt(sum / 2560 + 1e-6) (correctly rounded; never rsqrt). One work-group owns one (stream,
// row): no cross-work-group dependency, so the combine and the norm are one launch.
//
// **q4_hc_up_mix** (M:1013-1023, rounding at every point the reference rounds): down's 320 bf16 rows ->
//   a_j = rne(silu(f32(rne(f32(rne(d_j)) / 4))))                (each work-group recomputes the 320: SLM)
//   g[s][c] = rne(sigmoid(f32(rne(sum_j a_j x up[j][s x 2560 + c]))))   one fma chain, j ascending (the up
//             linear's {K 320, N 10240} gemv_bf16 tiles; bf16 x bf16 products are exact, so the chain is
//             contraction-proof)
//   x[m][c]  = rne(((p0 + p1) + p2 + p3) / 4),  p_s = rne(f32(g[s][c]) x f32(xn[s][c]))     (.mean(dim=-2))
//   UP_INJECT 1 (_I): work-group 0 writes inj[m][s] = rne(2 x rne(sigmoid(rne(rne(d_{320+s}) / 4)))), the 4
//             block_inject rows; the final mixer (_I absent) has none
// Every sigmoid / SiLU is 1 / (1 + exp_torch(-x)) / x / (1 + exp_torch(-x)): exp_torch is Sleef's expf u10 step
// for step (torch's Vectorized<float>::exp; k2_attn_eager.cl's), so this file and tests/kernels/qwen4exp_ref.h
// (hc_combine_norm, hc_up_mix) agree bit for bit.
//
// Plain OpenCL C (no sub-group functions): the Mac's OpenCL runs it (tools/mac/clrun/qwen4exp_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_hc: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#define HIDDEN 2560
#define HC 4
#define HCN (HC * HIDDEN)
#define LOW 320
#define PER_LANE (HIDDEN / 256)   /* 10 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
inline float exp_torch(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}

// ---------------------------------------------------------------------------------------------------------------
#ifdef HC_SRC
#ifndef HC_NORM
#error "q4_hc_combine_norm: HC_NORM must be defined (1, or 0 for _NN)"
#endif
#if HC_SRC == 1 && !defined(SRC_S)
#error "q4_hc_combine_norm _S: SRC_S (the mixer GEMV's split-K) must be defined"
#endif
#define WG_HC 256
//   src   HC_SRC 0: bf16 [M][2560] (the embedding row); 1: fp32 [SRC_S][M][2560]; 2: bf16 [M][2560]; 3: unread
//   inj   fp32 [M][4] (bf16 values, q4_hc_up_mix's)        H bf16 [M][4][2560]        xn bf16 [M][4][2560]
__attribute__((reqd_work_group_size(WG_HC, 1, 1)))
__kernel void q4_hc_combine_norm(__global const uint* restrict ctrl, __global ushort* restrict H,
                                 __global const void* restrict src, __global const float* restrict inj,
                                 __global const float* restrict norm_w, __global ushort* restrict xn) {
  const uint s = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __global ushort* restrict h = H + (size_t)m * HCN + (size_t)s * HIDDEN;
  float hv[PER_LANE];
  float acc = 0.0f;
#if HC_SRC == 1 || HC_SRC == 2
  const float w_inj = inj[m * HC + s];
#endif
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint k = lid + 256u * j;
#if HC_SRC == 0
    const ushort hb = ((__global const ushort*)src)[(size_t)m * HIDDEN + k];
    h[k] = hb;
#elif HC_SRC == 1
    float ys = 0.0f;
    for (uint q = 0; q < SRC_S; ++q) ys += ((__global const float*)src)[((size_t)q * M + m) * HIDDEN + k];
    const ushort hb = rne_bf16(bf16f(h[k]) + rf(rf(ys) * w_inj));
    h[k] = hb;
#elif HC_SRC == 2
    const float y = bf16f(((__global const ushort*)src)[(size_t)m * HIDDEN + k]);
    const ushort hb = rne_bf16(bf16f(h[k]) + rf(y * w_inj));
    h[k] = hb;
#else
    const ushort hb = h[k];
#endif
    hv[j] = bf16f(hb);
    acc = fma(hv[j], hv[j], acc);
  }
#if HC_NORM
  __local float red[WG_HC];
  red[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_HC / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)HIDDEN + 1e-6f);
  __global const float* restrict w = norm_w + (size_t)s * HIDDEN;
  __global ushort* restrict o = xn + (size_t)m * HCN + (size_t)s * HIDDEN;
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint k = lid + 256u * j;
    o[k] = rne_bf16(hv[j] * rstd * w[k]);
  }
#else
  (void)acc;
  (void)norm_w;
  (void)xn;
#endif
}
#endif  // HC_SRC

// ---------------------------------------------------------------------------------------------------------------
#ifdef UP_DOWN
#ifndef UP_INJECT
#error "q4_hc_up_mix: UP_INJECT must be defined (1: _I, 0: the final mixer)"
#endif
#if UP_DOWN < LOW + 4 * UP_INJECT
#error "q4_hc_up_mix: UP_DOWN (the down GEMV's row pitch) must hold the 320 rows and the inject's 4"
#endif
#define WG_UP 64
#define K8 (LOW / 8)   /* 40 k-octets of the up tiles */
//   down_f32  fp32 [M][UP_DOWN]   gemv_bf16's output (320 down rows, then the 4 inject rows under _I)
//   up        bf16 gemv_bf16 tiles {K 320, N 10240}: [n_tile][k_octet][8 k][16 n]
//   xn        bf16 [M][4][2560]   x bf16 [M][2560]   inj fp32 [M][4]
__attribute__((reqd_work_group_size(WG_UP, 1, 1)))
__kernel void q4_hc_up_mix(__global const uint* restrict ctrl, __global const float* restrict down_f32,
                           __global const ushort* restrict up, __global const ushort* restrict xn,
                           __global ushort* restrict x, __global float* restrict inj) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float a[LOW];
  __local float p[WG_UP];
  __global const float* restrict d = down_f32 + (size_t)m * UP_DOWN;
  for (uint jj = lid; jj < LOW; jj += WG_UP) a[jj] = rf(silu_t(rf(rf(d[jj]) / 4.0f)));
#if UP_INJECT
  if (g == 0 && lid < HC) inj[m * HC + lid] = rf(2.0f * rf(sigmoid_t(rf(rf(d[LOW + lid]) / 4.0f))));
#else
  (void)inj;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint s = lid / 16, ci = lid % 16;
  const uint c = g * 16 + ci, n = s * HIDDEN + c;
  __global const ushort* restrict tile = up + (size_t)(n / 16) * K8 * 128 + n % 16;
  float acc = 0.0f;
  for (uint k8 = 0; k8 < K8; ++k8)
    for (uint i = 0; i < 8; ++i) acc = fma(a[k8 * 8 + i], bf16f(tile[(size_t)k8 * 128 + i * 16]), acc);
  const float gv = rf(sigmoid_t(rf(acc)));
  p[lid] = rf(gv * bf16f(xn[(size_t)m * HCN + n]));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16) x[(size_t)m * HIDDEN + c] = rne_bf16((((p[lid] + p[16 + lid]) + p[32 + lid]) + p[48 + lid]) / 4.0f);
}
#endif  // UP_DOWN
