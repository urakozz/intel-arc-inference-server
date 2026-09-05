// pf_gdn_conv.cl - `gdn_chunk` part A: the conv ring's explicit seed and
// writeback, the batched depthwise 4-tap causal conv1d + SiLU over a whole
// chunk, the q/k l2norm, and the head scalars with the intra-chunk cumulative
// gate. Four entry points, one file, because they share the same literals and
// the same rounding discipline.
//
// The maths is docs/03-models.md's GDN block, exactly as `gdn_step.cl` runs it
// one position at a time. This file is the SAME op chain over `c_count`
// positions; `tests/prefill/gdn_chunk_ref.h` is the same chain on the host and
// the two must be edited together.
//
// ---------------------------------------------------------------------------
// What replaces decode's `M + 3 <= RING` argument
// ---------------------------------------------------------------------------
// `gdn_step.cl:176-178` compiles a static assertion that the ring is at least
// `M + 3` deep, because a work-group there reads slots `(pos-1..3) % 16` while
// other work-groups write `(pos+m) % 16`; the depth is what makes the two sets
// disjoint. At `c_count` = 2048 no ring depth could do that.
//
// The replacement is **an explicit prior launch**: `pf_gdn_seed` lifts the
// three older slots into a flat `seed` buffer, and only then does
// `pf_gdn_conv` run, reading `seed` and writing at most the chunk's last three
// slots back. No work-group reads a slot any work-group is writing, whatever
// `c_count` is, because the read happened in a previous launch and the list is
// in-order (`src/sycl/context.cc` sets ZE_COMMAND_QUEUE_FLAG_IN_ORDER; the
// 1024-launch chain in tests/prefill/context_test.cc is the empirical check).
// That is the whole proof, and it is why the ring is not made deeper.
//
// **`gdn_step.cl:73-93`'s ring-ownership predicate is also gone.** There, up to
// 12 work-groups recompute the same channel and exactly one is elected to store
// it. Here the grid is over CHANNELS, not over heads, so every channel is owned
// by exactly one work-item and there is nothing to elect.
//
// ---------------------------------------------------------------------------
// Rounding discipline - P1..P8, read out of gdn_step.cl and reproduced here
// ---------------------------------------------------------------------------
//   P1  gdn_step.cl:253 - `raw_b = rne_bf16(qkvz_partials[m*16384 + ch])`, the
//       qkv linear's single bf16 output. **This is what the ring stores**: the
//       reference's conv state holds the *input* sequence, not the convolved
//       one. S = 1 (ruling R1), so there is no slice to sum.
//   P2  gdn_step.cl:256-260 - the conv accumulates fp32 over widened bf16
//       inputs and fp32 weights, taps ASCENDING (t = 0 oldest, t = 3 the
//       current position), with explicit `fma`.
//   P3  gdn_step.cl:261 - `xb = rne_bf16(silu_f32(acc))`;
//       `silu_f32(x) = x / (1 + exp(-x))`, plain `exp`, never `native_exp`.
//   P4  gdn_step.cl:296-308 - the l2norm sums squares of the widened bf16 in
//       fp32, ONE TERM PER LANE so a plain multiply and no `fma`, then the
//       128-wide pairwise tree `stride = 64, 32, ..., 1` with
//       `r[i] += r[i+stride]` and a barrier after every step.
//   P5  gdn_step.cl:309-310 - `inv = 1.0f / sqrt(sum + 1e-6f)`, never `rsqrt`.
//   P6  gdn_step.cl:312 - `qf = bf16f(rne_bf16(bf16f(xb_q) * inv_q)) * Q_SCALE`:
//       the normalised value is rounded to bf16 and the 1/sqrt(128) scale is
//       applied AFTER the round, in fp32. `pf_gdn_l2norm` therefore stores the
//       **rounded, unscaled** word and every consumer multiplies by Q_SCALE in
//       fp32 at read.
//   P7  gdn_step.cl:314 - `kf = bf16f(rne_bf16(bf16f(xb_k) * inv_k))`, unscaled.
//   P8  gdn_step.cl:289-291 - `a_b = rne_bf16(ab_out[m*128 + h])`,
//       `b_b = rne_bf16(ab_out[m*128 + 48 + h])`, then
//       `g = negA * softplus_f32(bf16f(a_b) + dt_bias)` and
//       `beta = 1/(1 + exp(-bf16f(b_b)))` in fp32.
//       `softplus_f32(x) = x > 20.0f ? x : log1p(exp(x))` - torch's threshold.
//   P9  gdn_step.cl:317-356 - the recurrence rounds nothing (pf_gdn_scan.cl).
//   P10 gdn_step.cl:368-370 - `gdn_o` is written fp32 (pf_gdn_scan.cl).
//   P11-P13 prep.cl:365-384 - `prep_gated_head`'s five RNE points, which
//       `pf_gated_head.cl` already mirrors.

#ifndef NEGA_OFF
#error "pf_gdn_conv: NEGA_OFF must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif
#ifndef DTBIAS_OFF
#error "pf_gdn_conv: DTBIAS_OFF must be defined (src/kernels/prefill/CMakeLists.txt)"
#endif

// The qkv‖z GEMV runs UNSPLIT on the prefill path (plan 6b ruling R1), so a
// position's row base is `m * QKVZ_N` and there is no slice loop. The pairing
// with model::Qwen35's table is checked host-side, as gdn_step.cl's is.
#define QKVZ_S 1
#if QKVZ_S != 1
#error "pf_gdn_conv assumes qkv||z runs S=1; the partials index must loop s otherwise"
#endif

// model::Qwen35's dimensions; none of them is a variant.
#define HEADS 48          /* v-heads */
#define DIM 128           /* head dim, k and v alike */
#define QKVZ_N 16384      /* qkv‖z row width; z lives at 10240 */
#define CONV_ROWS 10240   /* the qkv channels the depthwise conv covers */
#define CONV_TAPS 4
#define Q_OFF 0           /* flat qkv channel bases: q 16x128, k 16x128, v 48x128 */
#define K_OFF 2048
#define V_OFF 4096
#define RING 16           /* conv ring depth (runtime::PersistentBuffers::kConvRing) */
#define AB_STRIDE 128     /* ab_out row: a at [0,48), b at [48,96), zero-padded to 128 */
#define B_OFF 48
#define CT 64             /* the FLA intra-chunk size (PrefillScratch::kGdnChunk) */
#define K_HEADS 16        /* q and k heads: CONV_ROWS = (2*16 + 48) * 128 */

#define WG_CH 256         /* channel-parallel kernels: 40 groups x 256 = 10240 */
#define WG_NORM 128       /* one lane per head dim */
#define WG_GATE 64        /* one lane per position of a 64-chunk */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even. The same add-and-shift every .cl in the
// tree uses (common::f32_to_bf16); NaN is not expected and not handled.
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// silu(x) = x / (1 + exp(-x)), plain `exp` (not `native_exp`) - gdn_step.cl's.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

// torch's softplus, threshold and all. Spelled identically in gdn_ref.h and
// gdn_chunk_ref.h.
inline float softplus_f32(float x) { return x > 20.0f ? x : log1p(exp(x)); }

// ---------------------------------------------------------------------------
// (1) Lift the ring's three older slots into a flat seed. Grid (40, 1, 1).
//     seed[t][ch] = pos-3+t < 0 ? 0 : conv_ring[(pos-3+t) % RING][ch], t = 0,1,2
//     A position below zero contributes 0 - the reference's conv state is
//     zero-initialised (gdn_step.cl:243), so a sequence's first positions
//     convolve against zeros.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_CH, 1, 1)))
__kernel void pf_gdn_seed(__global const ushort* restrict conv_ring,
                          __global ushort* restrict seed, uint pos) {
  const uint stride = get_global_size(0);
  for (uint ch = get_global_id(0); ch < CONV_ROWS; ch += stride) {
    for (uint t = 0; t < 3; ++t) {
      const int p = (int)pos - 3 + (int)t;
      seed[(size_t)t * CONV_ROWS + ch] =
          p < 0 ? (ushort)0 : conv_ring[((size_t)((uint)p % RING)) * CONV_ROWS + ch];
    }
  }
}

// ---------------------------------------------------------------------------
// (2) The batched depthwise 4-tap causal conv1d + SiLU over the whole chunk,
//     AND the ring writeback for the chunk's last min(c_count, 3) positions.
//
//     REWRITTEN under ruling A27 (2026-09-05): the position range is BLOCKED.
//     ------------------------------------------------------------------------
//     As delivered by L1-core the grid was (40, 1, 1) and one work-item owned
//     one channel for EVERY position, walking them serially. That is
//     10,240 work-items = **640 threads on a 2048-slot machine (31%)**, and
//     because consecutive iterations of the `m` loop are 64 KB apart, the bytes
//     in flight at any instant were one iteration's worth of the whole grid =
//     10,240 lanes x 4 B = **40 KB**, against the ~295 KB Little's law asks for
//     at 590 GB/s and this part's ~500 ns latency. It measured **1.3182 ms per
//     GDN layer per chunk = 95.4 GB/s on 125.8 MB** (docs/prefill-gdn-scan-
//     2026-09-05.md §5.2), i.e. 16% of the device's bandwidth while being
//     neither bandwidth- nor compute-bound.
//
//     The FIR is three deep and **every predecessor is readable straight out of
//     `qkvz_partials`** -- `raw(m)` below is exactly what the ring stores (P1)
//     and exactly what the serial walk carried in `win0..win2`. So the position
//     range blocks with a three-position halo that costs three loads per block,
//     and no recurrence crosses a block boundary.
//
//     Grid is now **(40, NB, 1)**, work-group 256 unchanged. The block width is
//     derived from `get_num_groups(1)` rather than shared as a literal with the
//     host, so the two cannot disagree about coverage; `gdn.cc` asks for
//     `NB = ceil(C / kConvBlock)` with `kConvBlock = 128`, which at C = 2048 is
//     640 work-groups = 10,240 threads = 5 waves.
//
//     **No rounding point moves and no sum is re-associated.** The per-position
//     body is the same four ascending `fma`s (P2) over the same widened bf16
//     words and the same `rne_bf16(silu_f32(acc))` store (P3); the output is a
//     function of `(m, ch)` alone, so the bar is bitwise identity against
//     `pf_gdn_conv_legacy` below, which `gdn_conv_test` case 7 asserts.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_CH, 1, 1)))
__kernel void pf_gdn_conv(__global const float* restrict qkvz_partials,
                          __global const ushort* restrict seed,
                          __global const float* restrict conv_w,
                          __global ushort* restrict xb,
                          __global ushort* restrict conv_ring, uint pos, uint c_count) {
  // The block this work-group owns. `blk * nb >= c_count` by construction, so
  // the blocks tile [0, c_count) exactly and a trailing empty block returns.
  const uint nb = get_num_groups(1);
  const uint blk = (c_count + nb - 1) / nb;
  const uint m0 = get_group_id(1) * blk;
  if (m0 >= c_count) return;                      // uniform across the work-group
  const uint m1 = min(m0 + blk, c_count);

  const uint stride = get_global_size(0);
  for (uint ch = get_global_id(0); ch < CONV_ROWS; ch += stride) {
    const __global float* restrict w = conv_w + (size_t)ch * CONV_TAPS;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];

    // The three-position halo. `seed[t]` holds chunk-relative position `t - 3`
    // (pf_gdn_seed above), so a negative position indexes `seed` at `p + 3` and
    // a non-negative one is the SAME `rne_bf16(qkvz_partials[...])` the serial
    // walk carried in its registers -- which is why blocking is bit-exact.
    float win[3];
    for (uint t = 0; t < 3; ++t) {
      const int p = (int)m0 - 3 + (int)t;
      win[t] = p < 0 ? bf16f(seed[(size_t)(p + 3) * CONV_ROWS + ch])
                     : bf16f(rne_bf16(qkvz_partials[(size_t)p * QKVZ_N + ch]));
    }
    float win0 = win[0], win1 = win[1], win2 = win[2];

    for (uint m = m0; m < m1; ++m) {
      const ushort raw_b = rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);   // P1
      const float win3 = bf16f(raw_b);            // tap 3 is the current position
      float acc = 0.0f;                           // taps ascending, explicit fma (P2)
      acc = fma(w0, win0, acc);
      acc = fma(w1, win1, acc);
      acc = fma(w2, win2, acc);
      acc = fma(w3, win3, acc);
      xb[(size_t)m * CONV_ROWS + ch] = rne_bf16(silu_f32(acc));                // P3
      win0 = win1;
      win1 = win2;
      win2 = win3;
    }

    // The ring keeps only what the NEXT chunk's seed reads: positions
    // pos+c_count-3 .. pos+c_count-1. The RNE is recomputed rather than cached
    // in three registers - it is three loads out of c_count and the code stays
    // one expression. Slots this chunk did not write still hold whatever an
    // earlier chunk or an earlier decode step left there, which nothing reads,
    // and the three that ARE written are exactly the three `gdn_step` will read
    // if decode resumes at pos+c_count. That is the whole prefill/decode
    // handoff for the conv state.
    //
    // With the position range blocked, each of those three positions belongs to
    // exactly one block, so the `m0 <= m < m1` guard is what keeps every live
    // slot written exactly once and by the same expression as before.
    for (uint t = 0; t < 3; ++t) {
      const int m = (int)c_count - 3 + (int)t;
      if (m >= (int)m0 && m < (int)m1)
        conv_ring[((size_t)((pos + (uint)m) % RING)) * CONV_ROWS + ch] =
            rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);
    }
  }
}

// ---------------------------------------------------------------------------
// (2b) `pf_gdn_conv` EXACTLY as it stood before ruling A27's rewrite, kept as
//      the bitwise reference for `tests/prefill/gdn_conv_test.cc` case 7 and
//      launched by nothing else -- `src/runtime/prefill/gdn.cc` binds only the
//      blocked kernel above.
//
//      It is here rather than in a golden dump because the artefact would be
//      41.9 MB per width, and rather than as a host comparison because
//      `silu`'s `exp` is 3 ulp on the device and correctly rounded on the host
//      (which is why case 1's bar against `gdn_chunk_ref` is <= 2 bf16 ulp and
//      not equality). Device-vs-device is the only comparison that can be
//      bit-exact, so this is the only shape the pre-registered bar could take.
//      It is also a permanent regression bar: any future re-partition of this
//      loop has to reproduce this kernel word for word.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_CH, 1, 1)))
__kernel void pf_gdn_conv_legacy(__global const float* restrict qkvz_partials,
                                 __global const ushort* restrict seed,
                                 __global const float* restrict conv_w,
                                 __global ushort* restrict xb,
                                 __global ushort* restrict conv_ring, uint pos,
                                 uint c_count) {
  const uint stride = get_global_size(0);
  for (uint ch = get_global_id(0); ch < CONV_ROWS; ch += stride) {
    const __global float* restrict w = conv_w + (size_t)ch * CONV_TAPS;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];

    float win0 = bf16f(seed[(size_t)0 * CONV_ROWS + ch]);
    float win1 = bf16f(seed[(size_t)1 * CONV_ROWS + ch]);
    float win2 = bf16f(seed[(size_t)2 * CONV_ROWS + ch]);

    for (uint m = 0; m < c_count; ++m) {
      const ushort raw_b = rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);   // P1
      const float win3 = bf16f(raw_b);
      float acc = 0.0f;
      acc = fma(w0, win0, acc);
      acc = fma(w1, win1, acc);
      acc = fma(w2, win2, acc);
      acc = fma(w3, win3, acc);
      xb[(size_t)m * CONV_ROWS + ch] = rne_bf16(silu_f32(acc));                // P3
      win0 = win1;
      win1 = win2;
      win2 = win3;
    }

    for (uint t = 0; t < 3; ++t) {
      const int m = (int)c_count - 3 + (int)t;
      if (m >= 0)
        conv_ring[((size_t)((pos + (uint)m) % RING)) * CONV_ROWS + ch] =
            rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]);
    }
  }
}

// ---------------------------------------------------------------------------
// (3) l2norm q and k IN PLACE in `xb`, per (position, k-head). Grid (32, C, 1),
//     WG 128: group x in [0,16) is the q of k-head x, [16,32) the k of x-16.
//     P4/P5/P6/P7. The stored q word is the ROUNDED, UNSCALED one; consumers
//     apply Q_SCALE in fp32 at read.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void pf_gdn_l2norm(__global ushort* restrict xb, uint c_count) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  if (m >= c_count) return;                 // uniform across the work-group

  const uint kh = wg < K_HEADS ? wg : wg - K_HEADS;
  const uint base = (wg < K_HEADS ? Q_OFF : K_OFF) + kh * DIM;
  const size_t idx = (size_t)m * CONV_ROWS + base + i;

  __local float red[DIM];
  const float t = bf16f(xb[idx]);
  red[i] = t * t;                           // one term per lane: a plain multiply (P4)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = DIM / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / sqrt(red[0] + 1e-6f);                              // P5
  // No barrier is needed before the store: `t` was read into a register before
  // the tree, every lane owns its own `i`, and the tree's last barrier already
  // separates the reduction from this read of red[0].
  xb[idx] = rne_bf16(t * inv);                                             // P6 / P7
}

// ---------------------------------------------------------------------------
// (4) Head scalars and the intra-chunk INCLUSIVE cumulative gate. Grid
//     (48 heads, ceil(C/64), 1), WG 64 = one lane per position of the chunk.
//       g_out[m][h]    = gc[m] = SUM_{j<=m, same 64-chunk} g[j]   fp32, <= 0
//       beta_out[m][h] = beta[m]                                  fp32
//     The cumulative sum RESTARTS at every multiple of 64: each 64-chunk's
//     algebra is written against its own chunk-start state, so a running sum
//     across the whole call would be wrong by exp(gc[63]) at position 64.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_GATE, 1, 1)))
__kernel void pf_gdn_gate(__global const float* restrict ab_out,
                          __global const float* restrict gdn_small,
                          __global float* restrict g_out,
                          __global float* restrict beta_out, uint c_count) {
  const uint h = get_group_id(0);
  const uint chunk = get_group_id(1);
  const uint i = get_local_id(0);
  const uint m = chunk * CT + i;

  const float negA = gdn_small[NEGA_OFF + h];      // the loader's -exp(A_log), fp32
  const float dt_bias = gdn_small[DTBIAS_OFF + h];

  __local float gs[CT];
  if (m < c_count) {
    const ushort a_b = rne_bf16(ab_out[(size_t)m * AB_STRIDE + h]);           // P8
    const ushort b_b = rne_bf16(ab_out[(size_t)m * AB_STRIDE + B_OFF + h]);
    gs[i] = negA * softplus_f32(bf16f(a_b) + dt_bias);
    beta_out[(size_t)m * HEADS + h] = 1.0f / (1.0f + exp(-bf16f(b_b)));
  } else {
    gs[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Lane 0 walks `gs` ASCENDING accumulating one fp32 running sum. A sequential
  // scan, not a Hillis-Steele tree: 64 elements are nothing, and an ascending
  // fp32 sum is exactly `tl.cumsum`'s order (fla cumsum.py:67) and exactly what
  // the host reference does - which is what lets the bar be bit-exact.
  if (i == 0) {
    const uint L = min((uint)CT, c_count - chunk * CT);
    float acc = 0.0f;
    for (uint j = 0; j < L; ++j) {
      acc += gs[j];
      g_out[(size_t)(chunk * CT + j) * HEADS + h] = acc;
    }
  }
}
