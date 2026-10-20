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
//
//   lfm_conv_in(ctrl, w, scales, x, taps, ring, y)   CONV_IN_L   the arm `conv` (Task 5): in_proj's GEMV + the conv
//
// Plain OpenCL C (the Mac runs it, tools/mac/clrun/lfm2moe_run.cc). No `FP_CONTRACT OFF` here, on purpose: every
// product in the conv chain is of two bf16 values (exact in fp32) and its sums are explicit fma, so contraction cannot
// move it - and lfm_conv_in's GEMV must compile exactly as gemv.cl does (contraction at OpenCL's default).

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

// ---------------------------------------------------------------------------------------------------------------
// lfm_conv_in - the arm `conv` (spec 24 §4.1 "P0's arm c"): in_proj's GEMV and the conv in ONE launch. in_proj is
// loaded in the conv-tile column order (loader::repack_in_proj(..., conv_tiles = true): 16-channel tile t is n-tiles
// 3t, 3t + 1, 3t + 2 = [B 16 | C 16 | x 16]), so work-group t holds channel tile t's B, C and x columns: sub-group q
// (0 B, 1 C, 2 x) runs gemv.cl's inner loop for its 16 columns over every k-group at split-K 1 - THE SAME expression
// gemv.cl compiles (dot8 into a group sum, `acc += gacc * scale`; this file leaves fp contraction at OpenCL's
// default, as gemv.cl does, so the device fuses or does not fuse it the same way) - and the conv is the epilogue:
// rne(0.0f + acc) is the as-built conv's rne(Σ_s slices) at S = 1, bit for bit. Bitwise the as-built pair
// (gemv S1 + lfm_conv) by construction; lfm2moe_kernels_test and lfm2moe_decode_test --fuse conv hold it there.
//
//   lfm_conv_in(ctrl, w, scales, x, taps, ring, y)     CONV_IN_L (layout)   grid (HIDDEN / 16, 1), WG 48
#ifdef CONV_IN_L
#if HIDDEN % 64 != 0
#error "lfm_conv_in: the hidden row is whole k-groups"
#endif
#if CONV_IN_L != 0 && CONV_IN_L != 1
#error "lfm_conv_in: CONV_IN_L is gemv.cl's layout 0 or 1"
#endif
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define LFM_CI_SG16 __attribute__((intel_reqd_sub_group_size(16)))
#else
#define LFM_CI_SG16
#endif
#define CI_K HIDDEN
#define CI_N (3 * HIDDEN)
#define CI_G (CI_K / 64)
#define CI_TILE_U32 136
// gemv.cl's dot8 (GEMV_DEQ_SHIFT 0: no LFM2 cell sets it), verbatim.
inline float ci_dot8(uint word, ushort8 xv) {
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
}
__attribute__((reqd_work_group_size(48, 1, 1)))
LFM_CI_SG16
__kernel void lfm_conv_in(__global const uint* restrict ctrl, __global const uint* restrict w,
                          __global const half* restrict scales, __global const ushort* restrict x,
                          __global const float* restrict taps, __global ushort* restrict ring,
                          __global ushort* restrict y) {
  const uint t = get_group_id(0);              // the 16-channel tile
  const uint lid = get_local_id(0);
  const uint q = lid / 16, lane = lid % 16;    // q: 0 B, 1 C, 2 x
  const uint n_tile = 3 * t + q;
  const uint n = n_tile * 16 + lane;
  __local float vals[M][3][16];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;                      // uniform

  // gemv.cl at M rows, split-K 1, its loop order (k-groups ascending; j then m inside a group).
  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
  for (uint g = 0; g < CI_G; ++g) {
    uint wv[8];
    float scale;
#if CONV_IN_L == 0
    (void)lane;
    __global const uint* wp = w + (size_t)(g * 8) * CI_N + n;
    for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * CI_N];
    scale = vload_half((size_t)g * CI_N + n, scales);   // gemv.cl's (float)scales[...]: the same exact value
#else
    (void)scales;
    __global const uint* tile = w + ((size_t)n_tile * CI_G + g) * CI_TILE_U32;
#ifdef cl_intel_subgroups
    uint8 blk = intel_sub_group_block_read8(tile);
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
#else
    for (int j = 0; j < 8; ++j) wv[j] = tile[j * 16 + lane];
#endif
    // gemv.cl reads the lane's f16 scale with a sub-group block read and as_half; vload_half is the same value
    // (f16 -> f32 is exact) and needs no cl_khr_fp16 (the Mac's OpenCL lacks it).
    scale = vload_half(lane, (const __global half*)(tile + 128));
#endif
    float gacc[M];
    for (int m = 0; m < M; ++m) gacc[m] = 0.f;
    for (int j = 0; j < 8; ++j) {
      for (int m = 0; m < M; ++m) {
        ushort8 xv = vload8(0, x + (size_t)m * CI_K + g * 64 + j * 8);
        gacc[m] += ci_dot8(wv[j], xv);
      }
    }
    for (int m = 0; m < M; ++m) acc[m] += gacc[m] * scale;
  }
  for (int m = 0; m < M; ++m) vals[m][q][lane] = acc[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (q != 0) return;
  // The epilogue: lfm_conv's chain for channel c (bitwise its, the slices' sum at S = 1 being 0.0f + acc).
  const uint c = t * 16 + lane;
  const float w0 = taps[(size_t)c * 3 + 0], w1 = taps[(size_t)c * 3 + 1], w2 = taps[(size_t)c * 3 + 2];
  const int p2 = (int)pos - 2, p1 = (int)pos - 1;
  float f2 = p2 < 0 ? 0.0f : bf16f(ring[(size_t)((uint)p2 % RING) * HIDDEN + c]);
  float f1 = p1 < 0 ? 0.0f : bf16f(ring[(size_t)((uint)p1 % RING) * HIDDEN + c]);
  for (uint m = 0; m < n_act; ++m) {
    float b = 0.0f, cc = 0.0f, xx = 0.0f;
    b += vals[m][0][lane];
    cc += vals[m][1][lane];
    xx += vals[m][2][lane];
    const ushort B = rne_bf16(b), C = rne_bf16(cc), X = rne_bf16(xx);
    const ushort bx = rne_bf16(bf16f(B) * bf16f(X));
    ring[(size_t)((pos + m) % RING) * HIDDEN + c] = bx;
    const float a = fma(w2, bf16f(bx), fma(w1, f1, fma(w0, f2, 0.0f)));
    y[(size_t)m * HIDDEN + c] = rne_bf16(bf16f(C) * bf16f(rne_bf16(a)));
    f2 = f1;
    f1 = bf16f(bx);
  }
}
#endif  // CONV_IN_L
