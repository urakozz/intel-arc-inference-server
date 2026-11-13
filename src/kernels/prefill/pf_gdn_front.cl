// pf_gdn_front.cl - phase 2 Task 2 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): the GDN
// front's conv1d + SiLU and the q / k l2norm in ONE launch, BITWISE pf_gdn_conv.cl's pf_gdn_conv followed by
// pf_gdn_l2norm (the seed and the gate stay their own launches: the seed's ring read must precede the
// ring writeback by a launch whatever C is - pf_gdn_conv.cl's header - and the gate is 0.5 ms of 60 calls).
//
// Grid (conv rows / 256, ceil(C / FB)), work-group 256 = 256 channels x FB positions. The conv is
// pf_gdn_conv's per-position body, statement for statement (P1-P3: rne of the fp32 partials, four ascending
// fma taps, rne(silu)), and its three-position halo / ring writeback. A work-group's 256 channels are two
// whole heads; on the q / k work-groups (channels < V_OFF) each (position, head)'s 128 SiLU outputs are
// parked in SLM and one sub-group evaluates pf_gdn_l2norm's 128-lane tree on them - stride 64 .. 16 in
// registers, 8 .. 1 by shuffles: the same pairs and the same adds (P4) - then 1 / sqrt(sum + 1e-6f) (P5) and
// rne(t * inv) (P6 / P7). v work-groups store the conv output as pf_gdn_conv does. tests/prefill/
// gdn_front_test.cc compares every byte of xb and the ring with the two-launch walk. FP_CONTRACT is off so the
// squares and the tree's adds stay separate operations, as they are across pf_gdn_l2norm's SLM.
#pragma OPENCL FP_CONTRACT OFF

#ifndef GDN_V_HEADS
#define HEADS 48
#define CONV_ROWS 10240
#define QKVZ_N 16384
#define V_OFF 4096
#else
#ifndef GDN_K_HEADS
#error "pf_gdn_front: GDN_V_HEADS needs GDN_K_HEADS"
#endif
#define HEADS GDN_V_HEADS
#define CONV_ROWS ((2 * GDN_K_HEADS + GDN_V_HEADS) * 128)
#define QKVZ_N (CONV_ROWS + GDN_V_HEADS * 128)
#define V_OFF (2 * GDN_K_HEADS * 128)
#endif
#define DIM 128
#define CONV_TAPS 4
#define RING 16
#define WG_CH 256
#define SG 16
#define FB 32               /* positions a work-group */
#if V_OFF % WG_CH != 0
#error "pf_gdn_front: a work-group must be all q / k or all v channels"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

__attribute__((reqd_work_group_size(WG_CH, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_front(__global const float* restrict qkvz_partials,
                           __global const ushort* restrict seed,
                           __global const float* restrict conv_w,
                           __global ushort* restrict xb,
                           __global ushort* restrict conv_ring, uint pos, uint c_count) {
  const uint m0 = get_group_id(1) * FB;
  if (m0 >= c_count) return;                       // uniform across the work-group
  const uint m1 = min(m0 + FB, c_count);
  const uint lid = get_local_id(0), sg = lid / SG, lane = lid % SG;
  const uint ch = get_group_id(0) * WG_CH + lid;
  const bool qk = get_group_id(0) * WG_CH < V_OFF;  // work-group uniform
  __local ushort xs[FB * WG_CH];
  __local float inv[FB * 2];

  const __global float* restrict w = conv_w + (size_t)ch * CONV_TAPS;
  const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
  float win[3];
  for (uint t = 0; t < 3; ++t) {
    const int p = (int)m0 - 3 + (int)t;
    win[t] = p < 0 ? bf16f(seed[(size_t)(p + 3) * CONV_ROWS + ch])
                   : bf16f(rne_bf16(qkvz_partials[(size_t)p * QKVZ_N + ch]));
  }
  float win0 = win[0], win1 = win[1], win2 = win[2];
  for (uint m = m0; m < m1; ++m) {
    const ushort raw_b = rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);   // P1
    const float win3 = bf16f(raw_b);
    float acc = 0.0f;                                                       // P2
    acc = fma(w0, win0, acc);
    acc = fma(w1, win1, acc);
    acc = fma(w2, win2, acc);
    acc = fma(w3, win3, acc);
    const ushort o = rne_bf16(silu_f32(acc));                               // P3
    if (qk) xs[(m - m0) * WG_CH + lid] = o;
    else xb[(size_t)m * CONV_ROWS + ch] = o;
    win0 = win1;
    win1 = win2;
    win2 = win3;
  }
  for (uint t = 0; t < 3; ++t) {
    const int m = (int)c_count - 3 + (int)t;
    if (m >= (int)m0 && m < (int)m1)
      conv_ring[((size_t)((pos + (uint)m) % RING)) * CONV_ROWS + ch] =
          rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);
  }
  if (!qk) return;
  barrier(CLK_LOCAL_MEM_FENCE);

  // pf_gdn_l2norm's tree per (position, head): red[i] = t_i * t_i, then stride 64, 32, .., 1.
  for (uint pr = sg; pr < (m1 - m0) * 2; pr += WG_CH / SG) {
    const uint p = pr >> 1, hh = pr & 1u;
    __local const ushort* row = xs + p * WG_CH + hh * DIM;
    float s[8];
#pragma unroll
    for (uint u = 0; u < 8; ++u) {
      const float t = bf16f(row[lane + SG * u]);
      s[u] = t * t;                                  // P4: one term per lane, a plain multiply
    }
    float a[4];
#pragma unroll
    for (uint u = 0; u < 4; ++u) a[u] = s[u] + s[u + 4];    // stride 64
    const float b0 = a[0] + a[2], b1 = a[1] + a[3];         // stride 32
    float d = b0 + b1;                                      // stride 16
    d = d + intel_sub_group_shuffle(d, (lane + 8) & 15);    // 8
    d = d + intel_sub_group_shuffle(d, (lane + 4) & 15);    // 4
    d = d + intel_sub_group_shuffle(d, (lane + 2) & 15);    // 2
    d = d + intel_sub_group_shuffle(d, (lane + 1) & 15);    // 1
    if (lane == 0) inv[pr] = 1.0f / sqrt(d + 1e-6f);        // P5
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint hh = lid / DIM;
  for (uint m = m0; m < m1; ++m) {
    const float t = bf16f(xs[(m - m0) * WG_CH + lid]);
    xb[(size_t)m * CONV_ROWS + ch] = rne_bf16(t * inv[(m - m0) * 2 + hh]);  // P6 / P7
  }
}
