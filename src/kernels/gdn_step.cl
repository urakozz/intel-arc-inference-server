// gdn_step.cl - the gated delta-rule decode step: one GDN layer's depthwise
// conv1d update + SiLU, the q/k l2norm, and the recurrent state update. 48 of
// the model's 64 layers run this, `sycl-tla` has no example for it, and no
// library implements it - it is the engine's original-work kernel.
//
// The maths is docs/03-models.md, "Layer math - verified in the modeling file"
// (the GDN block), a transcription of `torch_recurrent_gated_delta_rule` and
// `causal_conv1d_update`. Doc 03 wins over any other text, this comment
// included. `tests/kernels/gdn_ref.h` is the same chain on the host and repeats
// every ordering statement below verbatim; the two must be edited together.
//
//   gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o)
//     ctrl            uint[]  runtime::Control: pos at CTRL_POS, n_active at CTRL_NACT
//     qkvz_partials   fp32 [1][M][16384]   the qkv‖z GEMV's partials (S = 1)
//     ab_out          fp32 [M][128]        a at [0,48), b at [48,96)
//     gdn_small       fp32                 the layer's GDN block: conv[10240][4]
//                                          at 0, negA[48] at NEGA_OFF, dt_bias[48]
//                                          at DTBIAS_OFF (loader/small_layout.h;
//                                          the gated norm is prep_gated_head's)
//     conv_ring       bf16 [16][10240]     this layer's conv ring, slot-major
//     state           fp32 [48][128][128]  this layer's S, k-major rows, v columns
//     gdn_o           fp32 [M][48][128]    written fp32; prep_gated_head rounds it
//
// ---------------------------------------------------------------------------
// Grid and the tile mapping
// ---------------------------------------------------------------------------
// Grid = (48 heads, 4 chunks), work-group 256 = **16 subgroups of 16 lanes**
// (SIMD16, the project's width). Work-group `(h, c)` owns state columns
// `[32c, 32c+32)` of head `h` - 192 work-groups per layer instead of the 48 a
// head-per-group decomposition would give (docs/12-kernels.md). With
// `lid = get_local_id(0)`, `sgid = lid / 16` and `lane = lid % 16`, work-item
// `(sgid, lane)` owns the 8x2 state tile
//
//     k-rows   8·sgid .. 8·sgid+7                  (subgroup sgid owns k-band sgid)
//     columns  32c + lane   and   32c + lane + 16  (lane owns two of the 32)
//
// = **16 fp32 of state per work-item** in registers, 16 x 16 x 16 = 128 k x 32 v
// per work-group. Two properties pay for this exact shape:
//
//   * **Coalescing.** `state` is k-major, so the 128 v of a row are contiguous.
//     At a fixed (k-row, column-of-the-pair) the 16 lanes of a subgroup read
//     `32c + lane`, i.e. 16 *consecutive* fp32 = one 64 B cache line, and the
//     second column of the pair is the next line. Letting the lane index pick
//     the k-row instead would have made every subgroup access a 16-way gather
//     at a 4 KB stride for 8 bytes each.
//   * **No atomics.** Columns are independent under the rank-1 update
//     `S[k][v] += kf[k]·Δ[v]`, so the update itself needs no communication at
//     all. Only the two contractions cross work-items, and for a fixed column
//     they cross exactly the 16 subgroups - a fixed SLM tree, stated below.
//
// ---------------------------------------------------------------------------
// The reduction trees (stated identically in tests/kernels/gdn_ref.h)
// ---------------------------------------------------------------------------
//   * `kv[v] = Σ_k S[k][v]·kf[k]`: band `sgid` contributes
//     `Σ_{kk=0..7} S[8·sgid+kk][v]·kf[8·sgid+kk]`, accumulated in **ascending
//     kk** with explicit `fma`, into `kv_red[sgid][v−32c]`. The 16 band partials
//     then collapse with a fixed pairwise tree: for `stride = 8, 4, 2, 1`, the
//     work-items with `sgid < stride` do
//     `kv_red[sgid][j] += kv_red[sgid+stride][j]`, with a barrier after every
//     step. `kv_red[0][j]` is the column's `kv`.
//   * `o[v] = Σ_k qf[k]·S[k][v]`: identical shape in `o_red`, with `qf`.
//   * the two l2norm sums are a differently shaped tree: lane `lid < 128`
//     contributes `f32(q_b[lid])²` (one term per lane, so a plain multiply and
//     no `fma`) and lane `128+i` the k term; each 128-wide array then collapses
//     with `stride = 64, 32, …, 1`, `r[i] += r[i+stride]`, barrier per step.
//
// The order of a tree is a property of the *bands*, not of `c` or `lane`, which
// is what lets the host reference walk whole 128-column heads and still land on
// the same bits. Nothing here is data-dependent and no atomic appears anywhere,
// so a replayed command list reproduces its own output exactly.
//
// ---------------------------------------------------------------------------
// Ring ownership - why this is safe without a cross-work-group barrier
// ---------------------------------------------------------------------------
// Every work-group convolves the same 384 channels for its head (128 q and
// 128 k of k-head `h/3`, 128 v of head `h`), so the same raw value is computed
// by up to 12 work-groups. Exactly one of them stores it:
//
//   * `c == 0` writes this head's **v** channels, `4096 + 128h .. +128`;
//   * `c == 0 && h % 3 == 0` also writes the **q** and **k** channels of k-head
//     `h/3`, which the three v-heads of that triple share.
//
// Together those cover all 10240 qkv channels exactly once. The value written
// is identical wherever it is recomputed (it is `rne_bf16` of a buffer nobody
// writes during this step), so the choice of owner is a bandwidth decision, not
// a correctness one. What *is* a correctness argument is the slot disjointness:
// a work-group writes slots `(pos+m) % 16` for `m < n_active` and reads slots
// `(pos−1) % 16`, `(pos−2) % 16`, `(pos−3) % 16`; with ring depth 16 >= M + 3
// those sets cannot intersect (plan 1 §9.4), so **no work-group ever reads a
// slot any work-group is writing this step** and the missing cross-work-group
// barrier is not needed. Within a work-group each work-item reads its own
// channel's history before the loop that writes it, and the newer window slots
// come from its own registers, not from the ring.
//
// ---------------------------------------------------------------------------
// Rounding discipline (plan 3 Task 2's preamble, op by op)
// ---------------------------------------------------------------------------
//   * `raw_b = rne_bf16(qkvz_partials[…])` - the qkv linear's bf16 output (S = 1,
//     so the split-K sum is a single term; it is still rounded, because that
//     rounding is the linear's). The ring stores `raw_b`, matching the
//     reference's conv state, which holds the *input* sequence and not the
//     convolved one;
//   * the conv accumulates fp32 over widened bf16 inputs and fp32 weights, taps
//     **ascending** (t = 0 the oldest, t = 3 the current token), with explicit
//     `fma` so neither compiler's contraction default matters. That tap order is
//     `causal_conv1d_update`'s: it rolls the conv state left by one and drops
//     the new input into the last column, then multiplies column `i` by
//     `weight[i]` - so `conv_state[..., -1]` (the newest value) meets weight
//     index `width − 1`, i.e. tap 3 here. Checked against the SYCL reference's
//     decode path, `vllm-xpu-kernels/csrc/xpu/gdn_attn/causal_conv1d.hpp`, which
//     fills `local_input[Width-1]` from the current token and sums
//     `local_input[i] * local_weights[i]` over ascending `i`;
//   * `x_b = rne_bf16(silu_f32(conv))` - the activation's bf16 output;
//   * l2norm sums squares of the widened bf16 in fp32 (tree above),
//     `inv = 1.0f / sqrt(sum + 1e-6f)` - **never `rsqrt`**, which is a ~2 ulp
//     approximation with no cross-implementation guarantee - and rounds the
//     normalised value to bf16; `q` is then widened and scaled by `1/√128` in
//     fp32, because the reference applies `query * scale` after `.float()`;
//   * `a`/`b` are rounded to bf16 (they are the a‖b linear's outputs) and β and
//     g come from them in fp32 - the reference's `.float()` path;
//   * the recurrence is pure fp32 and rounds nothing;
//   * `gdn_o` is written fp32; `prep_gated_head` (prep.cl) does that rounding.
//
// Every kernel is built with `-cl-fp32-correctly-rounded-divide-sqrt`
// (cmake/ocloc.cmake), so `/` and `sqrt` match the host bit for bit. `exp` (3
// ulp) and `log1p` (2 ulp) do not, and that slack - in the decay, the sigmoid,
// the conv's SiLU and the softplus - is the entire reason gdn_step_test carries
// a tolerance rather than a bit-exact bar on `state` and `gdn_o`.

#ifndef M
#define M 1
#endif
#ifndef NEGA_OFF
#error "gdn_step: NEGA_OFF must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef DTBIAS_OFF
#error "gdn_step: DTBIAS_OFF must be defined (src/kernels/CMakeLists.txt)"
#endif

// The qkv‖z GEMV's split-K slice count (model::Qwen35's table, docs/13-loader.md).
// At S = 1 the fp32 `[S][M][N]` partials collapse to `[M][N]` and a token's row
// base is `m·QKVZ_N`, which is what the `raw_b` load below assumes. If qkv‖z is
// ever re-tuned to a wider split-K, that load must sum the slices the way
// prep_gated_head does for `z` - and the reference and the ring's bit-exact bar
// with it. Fail the build here rather than silently read slice 0.
#define QKVZ_S 1
#if QKVZ_S != 1
#error "gdn_step assumes qkv||z runs S=1; the partials index must loop s otherwise"
#endif

// The model's dimensions (model::Qwen35); none of them is a variant.
#define HEADS 48          /* v-heads = the grid's x extent */
#define DIM 128           /* head dim, k and v alike */
#define QKVZ_N 16384      /* qkv‖z row width; z lives at 10240 and is not read here */
#define CONV_ROWS 10240   /* the qkv channels the depthwise conv covers */
#define CONV_TAPS 4
#define Q_OFF 0           /* flat qkv channel bases: q 16x128, k 16x128, v 48x128 */
#define K_OFF 2048
#define V_OFF 4096
#define RING 16           /* conv ring depth (runtime::DecodeBuffers::kConvRing) */
#define AB_STRIDE 128     /* ab_out row: a at [0,48), b at [48,96), zero-padded to 128 */
#define B_OFF 48

#define WG_GDN 256
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define BAND_K 8          /* k-rows per subgroup band: 128 / 16 */
#define CHUNK_V 32        /* state columns per work-group: 128 / 4 */
#define VPW 2             /* v columns per work-item: 32 / 16 */
#define CH_PER_WG 384     /* channels convolved per work-group: 128 q + 128 k + 128 v */
#define Q_SCALE 0.08838834764831845f   /* 1/sqrt(128), applied to q in fp32 */

#if M + 3 > RING
#error "gdn_step: the conv ring must be at least M + 3 deep (plan 1 §9.4)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even. Same add-and-shift as common::f32_to_bf16
// and prep.cl's; NaN is not expected and not handled.
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// silu(x) = x / (1 + exp(-x)), plain `exp` (not `native_exp`) - prep.cl's.
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

// torch's softplus, threshold and all: above 20 the function is its own
// argument to far more than fp32 can resolve. Spelled identically in gdn_ref.h.
inline float softplus_f32(float x) { return x > 20.0f ? x : log1p(exp(x)); }

__attribute__((reqd_work_group_size(WG_GDN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gdn_step(__global const uint* restrict ctrl,
                       __global const float* restrict qkvz_partials,
                       __global const float* restrict ab_out,
                       __global const float* restrict gdn_small,
                       __global ushort* restrict conv_ring,
                       __global float* restrict state,
                       __global float* restrict gdn_o) {
  const uint h = get_group_id(0);          // v-head
  const uint c = get_group_id(1);          // state-column chunk
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;              // owns k-rows [8·sgid, 8·sgid+8)
  const uint lane = lid % SG;              // owns columns 32c+lane and 32c+lane+16

  __local ushort xs[3][M][DIM];            // conv+silu outputs: [0] q, [1] k, [2] v
  __local float rq[DIM], rk[DIM];          // the two l2norm sums of squares
  __local float qf_s[DIM], kf_s[DIM];      // normalised q (scaled) and k, fp32
  __local float kv_red[SG][CHUNK_V];       // kv band partials: [band][column − 32c]
  __local float o_red[SG][CHUNK_V];        // o  band partials

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;                // the compiled variant is the ceiling
  if (n_act == 0) return;                  // uniform across the work-group

  // -------------------------------------------------------------------------
  // 1. conv1d update + SiLU for this work-group's 384 channels, all tokens.
  // One work-item owns one channel for every token, so the window's newer slots
  // are its own registers and only the pre-step history touches the ring.
  // -------------------------------------------------------------------------
  const uint kh = h / 3;                   // repeat_interleave(·, 3): v-head -> k-head
  for (uint cid = lid; cid < CH_PER_WG; cid += WG_GDN) {
    const uint which = cid >> 7;           // 0 = q, 1 = k, 2 = v
    const uint i = cid & (DIM - 1);
    const uint ch = which == 0 ? (Q_OFF + kh * DIM + i)
                  : which == 1 ? (K_OFF + kh * DIM + i)
                               : (V_OFF + h * DIM + i);
    const __global float* restrict w = gdn_small + (size_t)ch * CONV_TAPS;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    // Ring ownership, argued in the header.
    const bool own_ring = (c == 0) && (which == 2 || (h % 3) == 0);

    // The window's three older slots: absolute positions pos−3, pos−2, pos−1.
    // A position below zero contributes 0 - the reference's conv state is
    // zero-initialised, so a sequence's first tokens convolve against zeros.
    const int p3 = (int)pos - 3, p2 = (int)pos - 2, p1 = (int)pos - 1;
    float win0 = p3 < 0 ? 0.0f : bf16f(conv_ring[((size_t)((uint)p3 % RING)) * CONV_ROWS + ch]);
    float win1 = p2 < 0 ? 0.0f : bf16f(conv_ring[((size_t)((uint)p2 % RING)) * CONV_ROWS + ch]);
    float win2 = p1 < 0 ? 0.0f : bf16f(conv_ring[((size_t)((uint)p1 % RING)) * CONV_ROWS + ch]);

    for (uint m = 0; m < n_act; ++m) {
      // The qkv linear's bf16 output (gdn_ref.h: `raw_b`). The slice index is
      // `QKVZ_S - 1 == 0`: with one slice there is nothing to sum, which the
      // `#if QKVZ_S != 1` above is what keeps true.
      const ushort raw_b =
          rne_bf16(qkvz_partials[((size_t)(QKVZ_S - 1) * M + m) * QKVZ_N + ch]);
      if (own_ring) conv_ring[((size_t)((pos + m) % RING)) * CONV_ROWS + ch] = raw_b;
      const float win3 = bf16f(raw_b);            // tap 3 is the current token
      float acc = 0.0f;                           // taps ascending, explicit fma
      acc = fma(w0, win0, acc);
      acc = fma(w1, win1, acc);
      acc = fma(w2, win2, acc);
      acc = fma(w3, win3, acc);
      xs[which][m][i] = rne_bf16(silu_f32(acc));  // the activation's bf16 output (ref: xb)
      win0 = win1;
      win1 = win2;
      win2 = win3;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // -------------------------------------------------------------------------
  // 2. This work-item's 8x2 state tile into registers. `sbase` is the tile's
  // first cell; +DIM steps one k-row, +SG the column pair's second column.
  // -------------------------------------------------------------------------
  const size_t sbase = (size_t)h * DIM * DIM + (size_t)(sgid * BAND_K) * DIM + (c * CHUNK_V + lane);
  float S[BAND_K][VPW];
  for (uint kk = 0; kk < BAND_K; ++kk) {
    S[kk][0] = state[sbase + (size_t)kk * DIM];
    S[kk][1] = state[sbase + (size_t)kk * DIM + SG];
  }

  const float negA = gdn_small[NEGA_OFF + h];      // the loader's -exp(A_log), fp32
  const float dt_bias = gdn_small[DTBIAS_OFF + h];

  // -------------------------------------------------------------------------
  // 3. One token at a time: head scalars, l2norm, the recurrence.
  // -------------------------------------------------------------------------
  for (uint m = 0; m < n_act; ++m) {
    // Head scalars - fp32 from the a‖b linear's bf16 outputs. Every work-item
    // computes them from the same words, so they agree bit for bit.
    const ushort a_b = rne_bf16(ab_out[(size_t)m * AB_STRIDE + h]);        // ref: a_b
    const ushort b_b = rne_bf16(ab_out[(size_t)m * AB_STRIDE + B_OFF + h]);  // ref: b_b
    const float g = negA * softplus_f32(bf16f(a_b) + dt_bias);
    const float beta = 1.0f / (1.0f + exp(-bf16f(b_b)));
    const float decay = exp(g);

    // l2norm over the 128-wide q and k: one term per lane, then 128 -> 1.
    if (lid < DIM) {
      const float t = bf16f(xs[0][m][lid]);
      rq[lid] = t * t;                     // one term per lane: a plain multiply
    } else {
      const float t = bf16f(xs[1][m][lid - DIM]);
      rk[lid - DIM] = t * t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = DIM / 2; stride > 0; stride >>= 1) {
      if (lid < stride) rq[lid] += rq[lid + stride];
      else if (lid >= DIM && lid < DIM + stride) rk[lid - DIM] += rk[lid - DIM + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_q = 1.0f / sqrt(rq[0] + 1e-6f);   // never rsqrt
    const float inv_k = 1.0f / sqrt(rk[0] + 1e-6f);
    if (lid < DIM)
      qf_s[lid] = bf16f(rne_bf16(bf16f(xs[0][m][lid]) * inv_q)) * Q_SCALE;   // ref: qf, scaled after the round
    else
      kf_s[lid - DIM] = bf16f(rne_bf16(bf16f(xs[1][m][lid - DIM]) * inv_k)); // ref: kf, not scaled
    barrier(CLK_LOCAL_MEM_FENCE);

    // The recurrence (doc 03), pure fp32 - nothing below rounds.
    for (uint kk = 0; kk < BAND_K; ++kk) {          // S *= exp(g)
      S[kk][0] *= decay;
      S[kk][1] *= decay;
    }

    float p0 = 0.0f, p1 = 0.0f;                     // kv[v] = Σ_k S[k][v]·kf[k]
    for (uint kk = 0; kk < BAND_K; ++kk) {          // ascending kk within the band
      const float kfv = kf_s[sgid * BAND_K + kk];
      p0 = fma(S[kk][0], kfv, p0);
      p1 = fma(S[kk][1], kfv, p1);
    }
    kv_red[sgid][lane] = p0;
    kv_red[sgid][lane + SG] = p1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SG / 2; stride > 0; stride >>= 1) {
      if (sgid < stride) {
        kv_red[sgid][lane] += kv_red[sgid + stride][lane];
        kv_red[sgid][lane + SG] += kv_red[sgid + stride][lane + SG];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float kv0 = kv_red[0][lane], kv1 = kv_red[0][lane + SG];

    // Δ[v] = (v − kv)·β, then the rank-1 update, both entirely local to the
    // work-item's own columns.
    const float d0 = (bf16f(xs[2][m][c * CHUNK_V + lane]) - kv0) * beta;
    const float d1 = (bf16f(xs[2][m][c * CHUNK_V + lane + SG]) - kv1) * beta;
    for (uint kk = 0; kk < BAND_K; ++kk) {          // S[k][v] += kf[k]·Δ[v]
      const float kfv = kf_s[sgid * BAND_K + kk];
      S[kk][0] = fma(kfv, d0, S[kk][0]);
      S[kk][1] = fma(kfv, d1, S[kk][1]);
    }

    float r0 = 0.0f, r1 = 0.0f;                     // o[v] = Σ_k qf[k]·S[k][v]
    for (uint kk = 0; kk < BAND_K; ++kk) {
      const float qfv = qf_s[sgid * BAND_K + kk];
      r0 = fma(qfv, S[kk][0], r0);
      r1 = fma(qfv, S[kk][1], r1);
    }
    o_red[sgid][lane] = r0;
    o_red[sgid][lane + SG] = r1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SG / 2; stride > 0; stride >>= 1) {
      if (sgid < stride) {
        o_red[sgid][lane] += o_red[sgid + stride][lane];
        o_red[sgid][lane + SG] += o_red[sgid + stride][lane + SG];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (sgid == 0) {   // 16 lanes x 2 = the work-group's 32 columns, fp32
      __global float* restrict op = gdn_o + ((size_t)m * HEADS + h) * DIM + c * CHUNK_V;
      op[lane] = o_red[0][lane];
      op[lane + SG] = o_red[0][lane + SG];
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next token reuses rq/rk/kv_red/o_red
  }

  // -------------------------------------------------------------------------
  // 4. The tile goes back once, after every token of this step.
  // -------------------------------------------------------------------------
  for (uint kk = 0; kk < BAND_K; ++kk) {
    state[sbase + (size_t)kk * DIM] = S[kk][0];
    state[sbase + (size_t)kk * DIM + SG] = S[kk][1];
  }
}
