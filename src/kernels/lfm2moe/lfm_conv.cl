// lfm_conv.cl - spec 24c: LFM2.5-24B-A2B's short conv (Lfm2MoeShortConv, spec 24 §2.5, §4.1) over a ring of Bx.
//
//   lfm_conv(ctrl, slices, taps, ring, y)       CONV_S   grid (HIDDEN / 256, 1), WG 256
//
// One work-item per channel c, every row m = 0 .. n_active - 1 of the launch in order (spec 24 §4.1):
//
//   B, C, x = rne(Σ_s slices[s][m][c]), rne(Σ_s ...[HIDDEN + c]), rne(Σ_s ...[2 HIDDEN + c])   in_proj's bf16 output
//   Bx      = rne(f32(B) · f32(x));  ring[(pos + m) % 16][c] = Bx                                  the state holds Bx
//   acc     = fma(w2, Bx, fma(w1, f1, fma(w0, f2, 0)))      taps ascending (tap 0 multiplies Bx[p - 2]), fp32, ONE
//                                                           rounding: 24a measured this equal to torch's bf16
//                                                           depthwise conv1d on 100,000 rows x 2048 channels
//   y[m][c] = rne(f32(C) · f32(rne(acc)))
//
// with f1 / f2 the channel's Bx at positions p - 1 / p - 2: from the ring for the launch's first rows, from the
// work-item's own registers after (gdn_step.cl's ownership pattern), and 0 below position 0 - the ring is never
// cleared (a fresh sequence never reads it). Rows of one launch write slots pos .. pos + M - 1 and read pos - 2 ..
// pos + M - 3 (mod 16): disjoint for M <= 14, so no work-item reads a slot another one is writing.
// `taps` is the layer's conv weight, fp32 [HIDDEN][3] (loader::lfm::small_block's conv_taps: bf16 widened, exact).
// tests/kernels/lfm2moe_ref.h conv_step() is the host twin.
//
// Plain OpenCL C (the Mac runs it, tools/mac/clrun/lfm2moe_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT) || !defined(HIDDEN)
#error "lfm_conv: CTRL_POS / CTRL_NACT and HIDDEN must be defined"
#endif
#define RING 16
#if M > RING - 2
#error "lfm_conv: one launch's rows must not reach a slot another row reads (M <= 14)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

#ifdef CONV_S
#define WG_CONV 256
#define CONV_N (3 * HIDDEN)
inline float conv_col(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < CONV_S; ++s) v += p[((size_t)s * M + m) * CONV_N + col];
  return v;
}
__attribute__((reqd_work_group_size(WG_CONV, 1, 1)))
__kernel void lfm_conv(__global const uint* restrict ctrl, __global const float* restrict slices,
                       __global const float* restrict taps, __global ushort* restrict ring,
                       __global ushort* restrict y) {
  const uint c = get_group_id(0) * WG_CONV + get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (c >= HIDDEN || n_act == 0) return;
  const float w0 = taps[(size_t)c * 3 + 0], w1 = taps[(size_t)c * 3 + 1], w2 = taps[(size_t)c * 3 + 2];
  const int p2 = (int)pos - 2, p1 = (int)pos - 1;
  float f2 = p2 < 0 ? 0.0f : bf16f(ring[(size_t)((uint)p2 % RING) * HIDDEN + c]);
  float f1 = p1 < 0 ? 0.0f : bf16f(ring[(size_t)((uint)p1 % RING) * HIDDEN + c]);
  for (uint m = 0; m < n_act; ++m) {
    const ushort B = rne_bf16(conv_col(slices, m, c));
    const ushort C = rne_bf16(conv_col(slices, m, HIDDEN + c));
    const ushort X = rne_bf16(conv_col(slices, m, 2 * HIDDEN + c));
    const ushort bx = rne_bf16(bf16f(B) * bf16f(X));
    ring[(size_t)((pos + m) % RING) * HIDDEN + c] = bx;
    const float acc = fma(w2, bf16f(bx), fma(w1, f1, fma(w0, f2, 0.0f)));
    y[(size_t)m * HIDDEN + c] = rne_bf16(bf16f(C) * bf16f(rne_bf16(acc)));
    f2 = f1;
    f1 = bf16f(bx);
  }
}
#endif  // CONV_S
