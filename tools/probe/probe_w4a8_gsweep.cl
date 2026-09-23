// W4A8 group-size SWEEP kernels (2026-09-23).
//
// The W4A8 probe (docs/probe-w4a8-2026-09-23.md, tools/probe/probe_w4a8.cl) was
// REJECTED at the checkpoint's group size 64: the i8_i4 k32 DPAS delivers its
// 2x instruction rate, but 192 SIMD16 float instructions per group boundary per
// sub-group (64 int-to-float converts, 64 multiplies, 64 mads) against 16
// dpas.8x8 cost 3.0 to 3.4 ms of the 5.440 ms kernel.
//
// THE ONE THING THIS FILE VARIES: how often the kernel applies a scale.
//
// The rescale frequency is a function of the scale STRIDE, not the scale
// VALUES, so the whole speed curve is measurable on the checkpoint that exists:
// same real weights (the same layout-0 nibbles, byte for byte), same real
// activations, same tile, same messages, same dpas -- only the number of 64-k
// chunks that accumulate in int32 before an fp32 rescale changes.
//
// THIS MEASURES SPEED ONLY. At SUBS > 1 the scales the host hands these kernels
// are SYNTHESISED at the coarser stride from the checkpoint's own g64 scales;
// they are not a real quantisation of anything and the outputs at g128, g256
// and per-channel are numerically wrong on purpose. No error figure from those
// arms means anything about model quality. The published 2.79 % relative L2
// came from the activation side with the weights unchanged and is unaffected by
// group size; it is not re-derived here.
//
// PROBE-ONLY. No runtime path binds any entry point here, the production
// `GROUP 64` in the dequant kernels and the loader's group-size check are
// untouched, and production stays g64.
//
// STRUCTURE, relative to probe_w4a8.cl's `pw4a8_gemm`:
//   * the k axis is walked in 64-k CHUNKS, exactly the chunk that kernel's one
//     A message and two B messages cover, with the same messages at the same
//     coordinates and the same two k32 dpas per accumulator per chunk, and the
//     same two-chunk-ahead prefetch. Memory behaviour is therefore identical
//     across every arm.
//   * SUBS chunks accumulate into the int32 accumulator before the fp32
//     rescale runs. SUBS = 1 is `pw4a8_gemm` (g64) and is run beside the
//     original as a calibration; SUBS = 80 puts the rescale outside the
//     mainloop entirely (one scale for the whole K = 5120: per-channel).
//   * the rescale expression is the pre-registered one, unchanged:
//     fma((float)acc_i32, wscale[g][n] * xscale[m], acc_f32) -- NOT with
//     xscale[m] factored out to the epilogue.
//
// SIGNATURE: every entry point writes its OWN group stride into sig[0] and
// sig[1] before the mainloop, so no arm can silently measure another arm's
// configuration. sig[0] = 0x47 << 24 | stride ('G'), sig[1] = SUBS.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
#define CHUNK 64          // k per chunk: one A message, two B messages, 2 x k32 dpas

// Same tile as pw4a8_gemm: WG = 512 work-items = 32 sub-groups, 8 (m) x 4 (n);
// per sub-group 32 m rows (4 dpas repeat-blocks of 8) x 32 n (2 atoms of 16).
#define WG_M 256
#define WG_N 128
#define SG_M 32
#define SG_N 32
#define NA 4
#define NB 2

#define PW_GSWEEP(NAME, SUBS, GSTRIDE)                                              \
__attribute__((reqd_work_group_size(512, 1, 1)))                                    \
__attribute__((intel_reqd_sub_group_size(SG)))                                      \
__kernel void NAME(__global const ushort* restrict xq16,                            \
                   __global const float* restrict xs,                               \
                   __global const uint* restrict qw,                                \
                   __global const half* restrict wscale,                            \
                   __global float* restrict C,                                      \
                   __global uint* restrict sig,                                     \
                   uint M, uint K, uint N, uint ldxq, uint ldc) {                   \
  const uint gx = get_group_id(0);                                                  \
  const uint gy = get_group_id(1);                                                  \
  const uint m0 = gx * WG_M;                                                        \
  const uint n0 = gy * WG_N;                                                        \
  const uint s = get_sub_group_id();                                                \
  const uint sm = s >> 2;                                                           \
  const uint sn = s & 3u;                                                           \
  const uint mb = m0 + SG_M * sm;                                                   \
  const uint nb = n0 + SG_N * sn;                                                   \
                                                                                    \
  if (get_global_linear_id() == 0) {                                                \
    sig[0] = 0x47000000u | (uint)(GSTRIDE);                                         \
    sig[1] = (uint)(SUBS);                                                          \
  }                                                                                 \
                                                                                    \
  const int x_w = (int)(K), x_h = (int)M, x_p = (int)(ldxq * 2u);                   \
  const int q_w = (int)(N * 4u), q_h = (int)(K / 8u), q_p = (int)(N * 4u);          \
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);               \
  const uint NCH = K / CHUNK;               /* 64-k chunks over the whole K */      \
  const uint G = NCH / (uint)(SUBS);        /* rescale groups over the whole K */   \
                                                                                    \
  float xsv[NA][8];                                                                 \
  _Pragma("unroll")                                                                 \
  for (int a = 0; a < NA; ++a)                                                      \
    _Pragma("unroll")                                                               \
    for (int r = 0; r < 8; ++r) xsv[a][r] = xs[mb + 8u * (uint)a + (uint)r];        \
                                                                                    \
  float8 accf[NA][NB];                                                              \
  _Pragma("unroll")                                                                 \
  for (int a = 0; a < NA; ++a)                                                      \
    _Pragma("unroll")                                                               \
    for (int b = 0; b < NB; ++b) accf[a][b] = (float8)(0.0f);                       \
                                                                                    \
  _Pragma("unroll")                                                                 \
  for (int pf = 0; pf < 2; ++pf) {                                                  \
    const int kpf = pf * CHUNK;                                                     \
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w, x_h,   \
                                                  x_p,                              \
                                                  (int2)(kpf / 2,                   \
                                                         (int)(m0 + 8u * s)));      \
    if (s < 16u)                                                                    \
      intel_sub_group_2d_block_prefetch_32b_4r16x1c(                                \
          (__global void*)qw, q_w, q_h, q_p,                                        \
          (int2)((int)(n0 + 16u * (s >> 1)), kpf / 8 + 4 * (int)(s & 1u)));         \
  }                                                                                 \
                                                                                    \
  for (uint g = 0; g < G; ++g) {                                                    \
    /* the group's weight scales: one f16 per column, one message per n-atom, */    \
    /* read once per RESCALE GROUP -- this is the frequency under test.        */   \
    float wsc[NB];                                                                  \
    _Pragma("unroll")                                                               \
    for (int b = 0; b < NB; ++b)                                                    \
      wsc[b] = (float)as_half(intel_sub_group_block_read_us(                        \
          (__global const ushort*)(wscale + (size_t)g * N + nb + 16u * (uint)b)));  \
                                                                                    \
    int8 acci[NA][NB];                                                              \
    _Pragma("unroll")                                                               \
    for (int a = 0; a < NA; ++a)                                                    \
      _Pragma("unroll")                                                             \
      for (int b = 0; b < NB; ++b) acci[a][b] = (int8)(0);                          \
                                                                                    \
    __attribute__((opencl_unroll_hint(1)))                                          \
    for (uint u = 0; u < (uint)(SUBS); ++u) {                                       \
      const uint ch = g * (uint)(SUBS) + u;                                         \
      const uint k0 = ch * CHUNK;                                                   \
      intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);                         \
                                                                                    \
      ushort afrag[64];                                                             \
      intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)xq16, x_w, x_h,    \
                                                 x_p,                               \
                                                 (int2)((int)(k0 / 2u), (int)mb),   \
                                                 afrag);                            \
      uint wv[NB][8];                                                               \
      _Pragma("unroll")                                                             \
      for (int b = 0; b < NB; ++b)                                                  \
        intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)qw, q_w, q_h,     \
                                                  q_p,                              \
                                                  (int2)((int)(nb + 16u * (uint)b), \
                                                         (int)(k0 / 8u)),           \
                                                  wv[b]);                           \
                                                                                    \
      if (ch + 2u < NCH) {                                                          \
        const int kpf = (int)((ch + 2u) * CHUNK);                                   \
        intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w,    \
                                                      x_h, x_p,                     \
                                                      (int2)(kpf / 2,               \
                                                             (int)(m0 + 8u * s)));  \
        if (s < 16u)                                                                \
          intel_sub_group_2d_block_prefetch_32b_4r16x1c(                            \
              (__global void*)qw, q_w, q_h, q_p,                                    \
              (int2)((int)(n0 + 16u * (s >> 1)), kpf / 8 + 4 * (int)(s & 1u)));     \
      }                                                                             \
                                                                                    \
      _Pragma("unroll")                                                             \
      for (int ks = 0; ks < 2; ++ks) {                                              \
        short8 af[NA];                                                              \
        _Pragma("unroll")                                                           \
        for (int a = 0; a < NA; ++a)                                                \
          af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));            \
        _Pragma("unroll")                                                           \
        for (int b = 0; b < NB; ++b) {                                              \
          const int4 bv = as_int4(vload4(0, wv[b] + (uint)(4 * ks)) ^               \
                                  (uint4)(0x88888888u));                            \
          _Pragma("unroll")                                                         \
          for (int a = 0; a < NA; ++a)                                              \
            acci[a][b] = intel_sub_group_i8_i4_matrix_mad_k32(af[a], bv,            \
                                                              acci[a][b]);          \
        }                                                                           \
      }                                                                             \
      intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);                           \
    }                                                                               \
                                                                                    \
    /* --- the rescale, at the stride under test --- */                             \
    _Pragma("unroll")                                                               \
    for (int a = 0; a < NA; ++a)                                                    \
      _Pragma("unroll")                                                             \
      for (int b = 0; b < NB; ++b) {                                                \
        PW_RESCALE(a, b, 0); PW_RESCALE(a, b, 1);                                   \
        PW_RESCALE(a, b, 2); PW_RESCALE(a, b, 3);                                   \
        PW_RESCALE(a, b, 4); PW_RESCALE(a, b, 5);                                   \
        PW_RESCALE(a, b, 6); PW_RESCALE(a, b, 7);                                   \
      }                                                                             \
  }                                                                                 \
                                                                                    \
  _Pragma("unroll")                                                                 \
  for (int a = 0; a < NA; ++a)                                                      \
    _Pragma("unroll")                                                               \
    for (int b = 0; b < NB; ++b)                                                    \
      intel_sub_group_2d_block_write_32b_8r16x1c(                                   \
          (__global void*)C, c_w, c_h, c_p,                                         \
          (int2)((int)(nb + 16u * (uint)b), (int)(mb + 8u * (uint)a)),              \
          (__private uint*)&accf[a][b]);                                            \
}

// The pre-registered rescale expression, verbatim from probe_w4a8.cl.
#define PW_RESCALE(a, b, r)                                                       \
  accf[a][b].s##r = fma((float)acci[a][b].s##r, wsc[b] * xsv[a][r], accf[a][b].s##r)

// K = 5120, so 80 chunks of 64 k. The four arms:
PW_GSWEEP(pw4a8s_g64,  1,   64)     // the checkpoint's stride: 80 rescale groups
PW_GSWEEP(pw4a8s_g128, 2,  128)     // 40 rescale groups
PW_GSWEEP(pw4a8s_g256, 4,  256)     // 20 rescale groups
PW_GSWEEP(pw4a8s_gpc,  80, 5120)    // per-channel: ONE group, rescale out of the mainloop
