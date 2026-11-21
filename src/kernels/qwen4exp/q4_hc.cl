// q4_hc.cl - spec 21-QFN-v1c: Qwen3.8-Flash-Next's hyper-connections (spec 21-QFN-v1 §2.2, §4.1). The residual is H, 4
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
// for step (torch's Vectorized<float>::exp; tests/kernels/eager_attn_ref.h's), so this file and tests/kernels/qwen4exp_ref.h
// (hc_combine_norm, hc_up_mix) agree bit for bit.
//
// **PERF-66: the int8 form** (spec 21-QFN-v2 §7 lever 5; loader::Q4HcOffsets under Qwen4ExpDesc::hc_int8): each
// linear's weights are int8 tiles [N/16][K/16][16 k][16 n] (element (k, n) at (n / 16) x 16K + 16k + n % 16) and its
// fp16 scales [K/32][N] follow the tiles directly (one per 32 k of a column: loader::quantise_int8_g32_tiled). Judged,
// not bitwise against the bf16 form; each kernel below is bitwise its host twin (tests/kernels/qwen4exp_ref.h). With
// g = a group of 32 k:  t_g = fma chain over the group's k ascending of f32(q) x v (exact products), then
// acc = fma(t_g, f32(s_g), acc), g ascending.
//   q4_hc_down_i8(w, x, out)                 HC_DOWN_I8 = N (336 / 320)   grid N / 16, WG 16 x DN_KS
//       the down||inject GEMV in place of gemv_bf16 (same arguments, fp32 [M][N] out): sub-group q of a tile owns
//       k in [q K / DN_KS, (q + 1) K / DN_KS) and its own acc; the slices merge by gemv_bf16's tree (strides
//       DN_KS / 2 .. 1). Sub-group block reads (cl_intel_subgroups), as gemv_i8w.
//   q4_hc_up_mix ... _Q8                     UP_I8 = 1: the up linear's chain above in place of the bf16 fma chain;
//       everything else (activations, sigmoid, mean, inject) unchanged; the one-row and the _R forms
//   q4_hc_slab_i8(w, out, n0, ns)            HC_SLAB_I8 = N   grid (ns / 16, K / 8), WG 16
//       the prefill's down||inject slab in place of kol_pf_bf16_slab (same arguments and grid):
//       out[k][c] = rne_bf16(f32(q) x f32(s)) for n = n0 + c < N, else 0
//
// **PERF-106: the decode mixers tuned** (each form bitwise the one before it):
//   UP_BLK 1 (_B)    the up-mix reads its weights by sub-group block reads: a sub-group (16 lanes) is one stream's 16
//                    columns, one weight tile, so a lane gets 16 k (int8) / 8 k (bf16) a read instead of one scalar
//                    load a k at stride 16; the fma chain and its order are unchanged
//   _R<M>C16         at M >= 2 (the verify): the _R form below over all M rows in one work-group (grid (160, 1)), each
//                    weight read once for the M rows instead of once a row
//   DN_ACT 1 (_A)    q4_hc_down_i8's merged lanes write a_j (n < 320) and the inject value (320 <= n < 324) in place
//                    of the raw sums; UP_ACT 1 (_A) the up-mix reads them instead of computing the 320 SiLUs in each
//                    of its 160 work-groups (the int8 form only: the bf16 down is gemv_bf16)
//   The lists bind _B (one work-group a row) and the down's raw sums at every M: _R<M>C16 alone lost to _B
//   (0.64-0.84x); _R<M>C16 with _A is the bench's candidate pair only
//
// Plain OpenCL C (no sub-group functions) but q4_hc_down_i8 and the up-mix's _B.
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

#include "bf16.h"
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
// down's row d -> a_j = rne(silu(f32(rne(f32(rne(d_j)) / 4)))), inj_s = rne(2 x rne(sigmoid(rne(rne(d_{320+s}) / 4))))
inline float hc_act(float d) { return rf(silu_t(rf(rf(d) / 4.0f))); }
inline float hc_inj(float d) { return rf(2.0f * rf(sigmoid_t(rf(rf(d) / 4.0f)))); }

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
#ifndef UP_I8
#define UP_I8 0
#endif
#ifndef UP_BLK
#define UP_BLK 0
#endif
#ifndef UP_ACT
#define UP_ACT 0
#endif
#if UP_ACT && !UP_I8
#error "q4_hc_up_mix _A: only the int8 form's down (q4_hc_down_i8 _A) writes the activations"
#endif
#if UP_ACT   // down_f32 holds a_j and the inject values (q4_hc_down_i8 _A)
#define UP_A(v) (v)
#define UP_INJ(v) (v)
#else
#define UP_A(v) hc_act(v)
#define UP_INJ(v) hc_inj(v)
#endif
#if UP_BLK
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define UP_SG __attribute__((intel_reqd_sub_group_size(16)))
// t = the fma chain over 8 k ascending: a[0..7] x w's lanes in order (the scalar loop's arithmetic)
inline float fma8(__local const float* a, float8 w, float t) {
  t = fma(a[0], w.s0, t); t = fma(a[1], w.s1, t); t = fma(a[2], w.s2, t); t = fma(a[3], w.s3, t);
  t = fma(a[4], w.s4, t); t = fma(a[5], w.s5, t); t = fma(a[6], w.s6, t); t = fma(a[7], w.s7, t);
  return t;
}
// The sub-group's weights by block reads: lane l gets column n0 + l of the tile, component i its k (tile + i x 16).
inline float16 up_i8_16k(__global const uchar* tile) {   // 16 k of an int8 tile
  return convert_float16(as_char16(intel_sub_group_block_read_uc16(tile)));
}
inline float8 up_bf16_8k(__global const ushort* tile) {   // one k-octet of a bf16 tile
  return as_float8(convert_uint8(intel_sub_group_block_read_us8(tile)) << 16);
}
#else
#define UP_SG
#endif
#define WG_UP 64
#define K8 (LOW / 8)   /* 40 k-octets of the up tiles */
//   down_f32  fp32 [M][UP_DOWN]   gemv_bf16's output (320 down rows, then the 4 inject rows under _I)
//   up        bf16 gemv_bf16 tiles {K 320, N 10240}: [n_tile][k_octet][8 k][16 n]; _Q8: int8 tiles, then the scales
//   xn        bf16 [M][4][2560]   x bf16 [M][2560]   inj fp32 [M][4]
#ifdef UP_ROWS
// _R<UP_ROWS>[C<UP_COLS>] (the prefill's, M = 2048; decode's at M = UP_ROWS >= 2, PERF-106): one work-group owns
// UP_ROWS rows of UP_COLS columns (4 x UP_COLS lanes, a lane a (stream, column)), so each up weight is read once for
// UP_ROWS rows and each row's 320 activations are computed once for UP_COLS columns; every row's arithmetic is the
// one-row form's, in its order. Grid (2560 / UP_COLS, M / UP_ROWS).
#ifndef UP_COLS
#define UP_COLS 16
#endif
#define WG_UR (HC * UP_COLS)
#if UP_ROWS * HC > WG_UR
#error "q4_hc_up_mix _R: the inject rows (UP_ROWS x 4) must fit one work-group"
#endif
#if UP_BLK && UP_COLS % 16 != 0
#error "q4_hc_up_mix _R _B: a sub-group's 16 lanes must be one weight tile's columns"
#endif
UP_SG __attribute__((reqd_work_group_size(WG_UR, 1, 1)))
__kernel void q4_hc_up_mix(__global const uint* restrict ctrl, __global const float* restrict down_f32,
                           __global const ushort* restrict up, __global const ushort* restrict xn,
                           __global ushort* restrict x, __global float* restrict inj) {
  const uint g = get_group_id(0);
  const uint m0 = get_group_id(1) * UP_ROWS;
  const uint lid = get_local_id(0);
  const uint nact = n_rows(ctrl);
  if (m0 >= nact) return;   // uniform
  const uint nr = min((uint)UP_ROWS, nact - m0);
  __local float a[UP_ROWS * LOW];
  __local float p[UP_ROWS * WG_UR];
  for (uint i = lid; i < UP_ROWS * LOW; i += WG_UR) {
    const uint r = i / LOW, jj = i % LOW;
    a[i] = r < nr ? UP_A(down_f32[(size_t)(m0 + r) * UP_DOWN + jj]) : 0.0f;
  }
#if UP_INJECT
  if (g == 0 && lid < nr * HC) {
    const uint r = lid / HC, s = lid % HC;
    inj[(m0 + r) * HC + s] = UP_INJ(down_f32[(size_t)(m0 + r) * UP_DOWN + LOW + s]);
  }
#else
  (void)inj;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint s = lid / UP_COLS, ci = lid % UP_COLS;
  const uint c = g * UP_COLS + ci, n = s * HIDDEN + c;
#if UP_I8
  float acc[UP_ROWS];
  for (uint r = 0; r < UP_ROWS; ++r) acc[r] = 0.0f;
#if UP_BLK
  __global const uchar* restrict tile = (__global const uchar*)up + (size_t)(n / 16) * LOW * 16;   // the sub-group's
#else
  __global const char* restrict tile = (__global const char*)up + (size_t)(n / 16) * LOW * 16 + n % 16;
#endif
  __global const half* restrict sc = (__global const half*)((__global const uchar*)up + (size_t)LOW * HCN);
  for (uint kg = 0; kg < LOW / 32; ++kg) {   // kg: the 32-k group (g is the work-group)
    float t[UP_ROWS];
    for (uint r = 0; r < UP_ROWS; ++r) t[r] = 0.0f;
#if UP_BLK
    const float16 w0 = up_i8_16k(tile + kg * 512), w1 = up_i8_16k(tile + kg * 512 + 256);
    for (uint r = 0; r < UP_ROWS; ++r) {
      __local const float* ar = a + r * LOW + kg * 32;
      t[r] = fma8(ar + 24, w1.hi, fma8(ar + 16, w1.lo, fma8(ar + 8, w0.hi, fma8(ar, w0.lo, t[r]))));
    }
#else
    for (uint i = 0; i < 32; ++i) {
      const float wv = (float)tile[(size_t)(kg * 32 + i) * 16];
      for (uint r = 0; r < UP_ROWS; ++r) t[r] = fma(a[r * LOW + kg * 32 + i], wv, t[r]);
    }
#endif
    const float sv = vload_half((size_t)kg * HCN + n, sc);
    for (uint r = 0; r < UP_ROWS; ++r) acc[r] = fma(t[r], sv, acc[r]);
  }
#else
  float acc[UP_ROWS];
  for (uint r = 0; r < UP_ROWS; ++r) acc[r] = 0.0f;
#if UP_BLK
  __global const ushort* restrict tile = up + (size_t)(n / 16) * K8 * 128;   // the sub-group's
  for (uint k8 = 0; k8 < K8; ++k8) {
    const float8 w = up_bf16_8k(tile + k8 * 128);
    for (uint r = 0; r < UP_ROWS; ++r) acc[r] = fma8(a + r * LOW + k8 * 8, w, acc[r]);
  }
#else
  __global const ushort* restrict tile = up + (size_t)(n / 16) * K8 * 128 + n % 16;
  for (uint k8 = 0; k8 < K8; ++k8)
    for (uint i = 0; i < 8; ++i) {
      const float wv = bf16f(tile[(size_t)k8 * 128 + i * 16]);
      for (uint r = 0; r < UP_ROWS; ++r) acc[r] = fma(a[r * LOW + k8 * 8 + i], wv, acc[r]);
    }
#endif
#endif
  for (uint r = 0; r < UP_ROWS; ++r)
    if (r < nr) {
      const float gv = rf(sigmoid_t(rf(acc[r])));
      p[r * WG_UR + lid] = rf(gv * bf16f(xn[(size_t)(m0 + r) * HCN + n]));
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < nr * UP_COLS; i += WG_UR) {
    const uint r = i / UP_COLS, l = i % UP_COLS;
    __local const float* restrict pr = p + r * WG_UR;
    x[(size_t)(m0 + r) * HIDDEN + g * UP_COLS + l] =
        rne_bf16((((pr[l] + pr[UP_COLS + l]) + pr[2 * UP_COLS + l]) + pr[3 * UP_COLS + l]) / 4.0f);
  }
}
#else
UP_SG __attribute__((reqd_work_group_size(WG_UP, 1, 1)))
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
  for (uint jj = lid; jj < LOW; jj += WG_UP) a[jj] = UP_A(d[jj]);
#if UP_INJECT
  if (g == 0 && lid < HC) inj[m * HC + lid] = UP_INJ(d[LOW + lid]);
#else
  (void)inj;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint s = lid / 16, ci = lid % 16;   // _B: sub-group s, lane ci
  const uint c = g * 16 + ci, n = s * HIDDEN + c;
#if UP_I8
  float acc = 0.0f;
#if UP_BLK
  __global const uchar* restrict tile = (__global const uchar*)up + (size_t)(n / 16) * LOW * 16;   // the sub-group's
#else
  __global const char* restrict tile = (__global const char*)up + (size_t)(n / 16) * LOW * 16 + n % 16;
#endif
  __global const half* restrict sc = (__global const half*)((__global const uchar*)up + (size_t)LOW * HCN);
  for (uint kg = 0; kg < LOW / 32; ++kg) {   // kg: the 32-k group (g is the work-group)
    float t = 0.0f;
#if UP_BLK
    const float16 w0 = up_i8_16k(tile + kg * 512), w1 = up_i8_16k(tile + kg * 512 + 256);
    __local const float* ak = a + kg * 32;
    t = fma8(ak + 24, w1.hi, fma8(ak + 16, w1.lo, fma8(ak + 8, w0.hi, fma8(ak, w0.lo, t))));
#else
    for (uint i = 0; i < 32; ++i) t = fma(a[kg * 32 + i], (float)tile[(size_t)(kg * 32 + i) * 16], t);
#endif
    acc = fma(t, vload_half((size_t)kg * HCN + n, sc), acc);
  }
#else
  float acc = 0.0f;
#if UP_BLK
  __global const ushort* restrict tile = up + (size_t)(n / 16) * K8 * 128;   // the sub-group's
  for (uint k8 = 0; k8 < K8; ++k8) acc = fma8(a + k8 * 8, up_bf16_8k(tile + k8 * 128), acc);
#else
  __global const ushort* restrict tile = up + (size_t)(n / 16) * K8 * 128 + n % 16;
  for (uint k8 = 0; k8 < K8; ++k8)
    for (uint i = 0; i < 8; ++i) acc = fma(a[k8 * 8 + i], bf16f(tile[(size_t)k8 * 128 + i * 16]), acc);
#endif
#endif
  const float gv = rf(sigmoid_t(rf(acc)));
  p[lid] = rf(gv * bf16f(xn[(size_t)m * HCN + n]));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16) x[(size_t)m * HIDDEN + c] = rne_bf16((((p[lid] + p[16 + lid]) + p[32 + lid]) + p[48 + lid]) / 4.0f);
}
#endif  // UP_ROWS
#endif  // UP_DOWN

// ---------------------------------------------------------------------------------------------------------------
#ifdef HC_DOWN_I8
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef DN_KS
#error "q4_hc_down_i8: DN_KS (the sub-groups splitting a tile's K) must be defined"
#endif
#ifndef DN_ACT
#define DN_ACT 0
#endif
#define DN_N HC_DOWN_I8
#define DN_G (HCN / 32 / DN_KS)   /* groups of 32 k a sub-group */
#if (HCN / 32) % DN_KS != 0 || (DN_KS & (DN_KS - 1)) != 0 || DN_N % 16 != 0
#error "q4_hc_down_i8: DN_KS must be a power of two dividing K / 32, N whole 16-column tiles"
#endif
//   w    int8 tiles {K 10240, N}, then fp16 scales [320][N]      x  bf16 [M][10240] (xn)      out  fp32 [M][N]
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * DN_KS, 1, 1)))
__kernel void q4_hc_down_i8(__global const uchar* restrict w, __global const ushort* restrict x,
                            __global float* restrict out) {
  __local float red[DN_KS][M][16];
  const uint lane = get_sub_group_local_id();
  const uint q = get_sub_group_id();
  const uint n = get_group_id(0) * 16 + lane;
  const uint g0 = q * DN_G;
  __global const uchar* tile = w + (size_t)get_group_id(0) * HCN * 16 + (size_t)g0 * 512;
  __global const half* restrict sc = (__global const half*)(w + (size_t)HCN * DN_N);
  float acc[M];
  for (uint m = 0; m < M; ++m) acc[m] = 0.0f;
  for (uint g = g0; g < g0 + DN_G; ++g, tile += 512) {
    const float16 w0 = convert_float16(as_char16(intel_sub_group_block_read_uc16(tile)));
    const float16 w1 = convert_float16(as_char16(intel_sub_group_block_read_uc16(tile + 256)));
    const float sv = vload_half((size_t)g * DN_N + n, sc);
    for (uint m = 0; m < M; ++m) {
      const ushort16 x0 = vload16(0, x + (size_t)m * HCN + g * 32);
      const ushort16 x1 = vload16(0, x + (size_t)m * HCN + g * 32 + 16);
      float t = 0.0f;
      t = fma(w0.s0, bf16f(x0.s0), t); t = fma(w0.s1, bf16f(x0.s1), t);
      t = fma(w0.s2, bf16f(x0.s2), t); t = fma(w0.s3, bf16f(x0.s3), t);
      t = fma(w0.s4, bf16f(x0.s4), t); t = fma(w0.s5, bf16f(x0.s5), t);
      t = fma(w0.s6, bf16f(x0.s6), t); t = fma(w0.s7, bf16f(x0.s7), t);
      t = fma(w0.s8, bf16f(x0.s8), t); t = fma(w0.s9, bf16f(x0.s9), t);
      t = fma(w0.sa, bf16f(x0.sa), t); t = fma(w0.sb, bf16f(x0.sb), t);
      t = fma(w0.sc, bf16f(x0.sc), t); t = fma(w0.sd, bf16f(x0.sd), t);
      t = fma(w0.se, bf16f(x0.se), t); t = fma(w0.sf, bf16f(x0.sf), t);
      t = fma(w1.s0, bf16f(x1.s0), t); t = fma(w1.s1, bf16f(x1.s1), t);
      t = fma(w1.s2, bf16f(x1.s2), t); t = fma(w1.s3, bf16f(x1.s3), t);
      t = fma(w1.s4, bf16f(x1.s4), t); t = fma(w1.s5, bf16f(x1.s5), t);
      t = fma(w1.s6, bf16f(x1.s6), t); t = fma(w1.s7, bf16f(x1.s7), t);
      t = fma(w1.s8, bf16f(x1.s8), t); t = fma(w1.s9, bf16f(x1.s9), t);
      t = fma(w1.sa, bf16f(x1.sa), t); t = fma(w1.sb, bf16f(x1.sb), t);
      t = fma(w1.sc, bf16f(x1.sc), t); t = fma(w1.sd, bf16f(x1.sd), t);
      t = fma(w1.se, bf16f(x1.se), t); t = fma(w1.sf, bf16f(x1.sf), t);
      acc[m] = fma(t, sv, acc[m]);
    }
  }
  for (uint m = 0; m < M; ++m) red[q][m][lane] = acc[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DN_KS / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (uint m = 0; m < M; ++m) red[q][m][lane] += red[q + stride][m][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (uint m = 0; m < M; ++m) {
#if DN_ACT   // _A: the up-mix's a_j and inject values (columns 324.. are padding, left raw)
      const float v = red[0][m][lane];
      out[(size_t)m * DN_N + n] = n < LOW ? hc_act(v) : n < LOW + HC ? hc_inj(v) : v;
#else
      out[(size_t)m * DN_N + n] = red[0][m][lane];
#endif
    }
}
#endif  // HC_DOWN_I8

// ---------------------------------------------------------------------------------------------------------------
#ifdef HC_SLAB_I8
#define SL_N HC_SLAB_I8
#if SL_N % 16 != 0
#error "q4_hc_slab_i8: N must be whole 16-column tiles"
#endif
//   w    int8 tiles {K 10240, N}, then fp16 scales [320][N]      out  bf16 [K][ns] row-major (pf_gemm's B operand)
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void q4_hc_slab_i8(__global const uchar* restrict w, __global ushort* restrict out, const uint n0,
                            const uint ns) {
  const uint lane = get_local_id(0);
  const uint t = get_group_id(0);
  const uint k8 = get_group_id(1);
  __global ushort* restrict o = out + (size_t)(k8 * 8) * ns + t * 16 + lane;
  if (n0 + t * 16 >= SL_N) {   // uniform per work-group (N % 16 == 0)
    for (uint i = 0; i < 8; ++i) o[(size_t)i * ns] = (ushort)0;
    return;
  }
  const uint n = n0 + t * 16 + lane;
  __global const char* restrict tile = (__global const char*)w + (size_t)(n / 16) * HCN * 16 + n % 16;
  __global const half* restrict sc = (__global const half*)(w + (size_t)HCN * SL_N);
  const float sv = vload_half((size_t)(k8 / 4) * SL_N + n, sc);   // 8 k of one 32-k group
  for (uint i = 0; i < 8; ++i) o[(size_t)i * ns] = rne_bf16((float)tile[(size_t)(k8 * 8 + i) * 16] * sv);
}
#endif  // HC_SLAB_I8
