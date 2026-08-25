// attn.cl - the decode-attention trio: `attn_prep` (q/k RMSNorm + partial RoPE
// + the KV cache write), `attn_decode` (one flash-decode block of 256 KV
// positions per work-group, with the device-side early-out that makes a
// `max_len`-sized grid affordable under replay) and `attn_reduce` (merge the
// blocks, divide by the softmax denominator, apply the output gate). 16 of the
// model's 64 layers run all three.
//
// The maths is docs/03-models.md, "Layer math - verified in the modeling file"
// (the full-attention block), a transcription of `Qwen3_5Attention.forward`.
// Doc 03 wins over any other text, this comment included.
// `tests/kernels/attn_ref.h` is the same chain on the host and repeats every
// ordering statement below verbatim; the two must be edited together.
//
//   attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v)
//     ctrl          uint[]  runtime::Control: pos at CTRL_POS, n_active at CTRL_NACT
//     qkv_partials  fp32 [1][M][14336]      the fused qkv GEMV's partials (S = 1)
//     fa_small      fp32                    this layer's FA block: q_norm (1+w)[256]
//                                           at QNORM_OFF, k_norm at KNORM_OFF
//                                           (loader/small_layout.h)
//     rope          fp32 [max_len][2][32]   cos at [p][0][i], sin at [p][1][i]
//     attn_q        fp32 [M][24][256]       normed + roped, fp32
//     attn_gate     fp32 [M][24][256]       f32(rne_bf16(the gate columns))
//     kv_k, kv_v    bf16 [max_len][4][256]  this layer's caches
//
//   attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)
//     attn_part     fp32 [24][MAXLEN/256][M][258]   {mx, sm, acc[256]}
//
//   attn_reduce(ctrl, attn_part, attn_gate, attn_out)
//     attn_out      bf16 [M][6144]
//
// ---------------------------------------------------------------------------
// The column map of the fused qkv linear
// ---------------------------------------------------------------------------
// `q_proj ‖ k_proj ‖ v_proj` = 12288 + 1024 + 1024 = 14336 columns (doc 03;
// model::Qwen35's `LinearId::Qkv`, `Fuse::Concat`, S = 1). Inside q_proj the 24
// heads are **interleaved per head, not two halves**: head `h` is
// `[h·512, h·512+256)` and its output gate is the next 256. k-head `j` sits at
// `12288 + j·256` and v-head `j` at `13312 + j·256`. GQA is 6:1, so q-head `h`
// reads kv-head `h/6`.
//
// ---------------------------------------------------------------------------
// Grids, and why
// ---------------------------------------------------------------------------
// **attn_prep: grid (28, M), work-group 256** - one work-group per head, and a
// head is exactly 256 wide, so work-item `i` owns dim `i` and the RMSNorm tree
// is the work-group itself. Work-groups 0..23 are q-heads (each also writing
// its gate), 24..27 are the four kv-heads (each writing both `kv_k` and
// `kv_v`). 28 rather than 24+4+4 because k and v of a kv-head share a
// work-group: v needs no norm and no RoPE, so it is one extra rounding on a
// work-item that is already resident.
//
// **attn_decode: grid (4 kv-heads, MAXLEN/256 blocks), work-group 256 = 16
// subgroups × 16 lanes (SIMD16).** The grid spans the whole cache because the
// captured list cannot be re-sized per token; the work-groups past the context
// exit on their first instruction (below). A work-group owns one 256-position
// block of one kv-head and loops over that head's **six** q-heads (GQA 6:1) and
// the tokens in flight.
//
// The q-head loop is the OUTER one, so this kernel reads its block's K and V
// **six times over** (once per q-head), not once. That is a real cost and it is
// not hidden: at depth D a layer moves 6·M·D·4·1024 B instead of D·4·1024 B
// (docs/12-kernels.md, "Traffic per token"). What the (kv-head, block) grid
// buys against the obvious (q-head, block) alternative - which reads exactly
// the same bytes - is that the six passes happen **inside one work-group, back
// to back**, so passes 2..6 are L1/L2 hits by construction rather than by 24
// independent work-groups happening to land on the same cache at the same time;
// and it launches 6× fewer work-groups. Reading the block once outright needs
// the wave's K and V staged in SLM with the q-head loop moved *inside* the wave
// loop (16 positions × 256 × 2 B × 2 = 16 KB of SLM, six accumulators per
// work-item). That is the main unmeasured lever on this kernel and it is
// deferred to Task 9's measurement, not assumed (docs/12-kernels.md,
// "Rejected, and what was not measured").
//
// **attn_reduce: grid (24, M), work-group 256** - one work-group per (q-head,
// token), work-item `d` owning dim `d` of the 256-wide output. The merge
// scalars are computed redundantly by all 256 work-items from SLM-staged block
// headers: identical inputs in an identical order give identical bits, and no
// work-item diverges from any other.
//
// ---------------------------------------------------------------------------
// The early-out - what makes a max_len-sized grid affordable
// ---------------------------------------------------------------------------
// `attn_decode`'s grid is sized for `MAXLEN` at capture and never changes, so
// at a context of `pos` tokens the great majority of its work-groups have
// nothing to do. Each one tests **one uniform condition before touching
// anything**:
//
//     if (block_start >= pos + n_active) return;
//
// The test is per **block**, not per token in flight: a block that is live for
// any `m` runs for every `m`, and a position outside a given `m`'s causal bound
// is masked to −INF inside the wave. So a block that is partially valid writes
// real partials for each `m`, and the all-masked-for-this-m case falls out of
// the online update as `(−INF, 0, 0)` with no special case. `attn_reduce` reads
// exactly `nb(m) = (pos+m)/256 + 1` blocks - every one of which starts at or
// before `pos+m` and therefore ran - so it never reads a block the early-out
// skipped, and needs no data-dependent skip logic of its own.
// What that costs at short context is doc 07 #12, measured in Task 9.
//
// ---------------------------------------------------------------------------
// The orders (stated identically in tests/kernels/attn_ref.h)
// ---------------------------------------------------------------------------
//  1. **The RMSNorm sum of squares** (`attn_prep`): lane `i` contributes exactly
//     one term, `f32(x_b[i])²` - a plain multiply, no `fma` - and the 256-wide
//     SLM array collapses with a fixed pairwise tree: for
//     `stride = 128, 64, …, 1`, `red[i] += red[i + stride]`, barrier per step.
//  2. **The score dot** (`attn_decode`): subgroup `s` of the 16 owns one
//     position; its lane `l` accumulates the 16 elements `d = l + 16·t`,
//     `t = 0..15` **ascending**, with an explicit `fma`, into `dot_red[16s+l]`.
//     The 16 lane partials collapse with a fixed pairwise tree: for
//     `stride = 8, 4, 2, 1`, `dot_red[16s+l] += dot_red[16s+l+stride]`.
//     `dot_red[16s] · 1/16` is the score (1/16 = 1/√256, doc 03).
//  3. **The online softmax wave** (`attn_decode`): the block's 256 positions are
//     walked in 16 waves of 16. With the wave's scores `sc[0..15]` (−INF where
//     the causal bound masks the position),
//         nmx  = max(mx, sc[0], sc[1], …, sc[15])          (ascending s)
//         resc = exp(mx − nmx)                              (0 when mx = −INF)
//         sc[s] = exp(sc[s] − nmx) ;  ssum = Σ_s sc[s]      (ascending s)
//         sm   = fma(sm, resc, ssum)
//         mx   = nmx
//     and, in the work-item that owns dim `d`,
//         t      = Σ_s fma(sc[s], f32(kv_v[p_s][j][d]), t)  (ascending s)
//         acc    = fma(acc, resc, t)
//     A wave whose every position is masked **and** with no earlier valid
//     position leaves `nmx = −INF`; that case is skipped whole (`resc = 1`,
//     `sc = 0`), because `exp(−INF − (−INF))` is a NaN and nothing else here is.
//  4. **The block merge** (`attn_reduce`): ascending block order, `b = 0 … nb−1`:
//         nmx = max(mx, bmx) ; a = exp(mx − nmx) ; bs = exp(bmx − nmx)
//         sm  = fma(sm, a, bsm·bs) ; acc = fma(acc, a, bacc·bs) ; mx = nmx
//     Every merged block has its own first position inside the causal bound, so
//     `bmx` is finite and `nmx` is never −INF.
//
// None of the four is data-dependent in its *order*, no atomic appears
// anywhere, and every work-item that computes a shared scalar computes it from
// the same words in the same sequence. Two replays of the captured list
// therefore produce identical bytes, which `attn_test` asserts directly.
//
// ---------------------------------------------------------------------------
// Rounding discipline (plan 3 Task 2's preamble, op by op)
// ---------------------------------------------------------------------------
//   * `x_b = rne_bf16(qkv_partials[…])` - the qkv linear's bf16 output, rounded
//     once (S = 1, so the split-K sum is a single term; it is still rounded,
//     because that rounding is the linear's);
//   * the norm widens to fp32, sums squares in fp32 (tree 1 above), uses
//     `1.0f / sqrt(mean + 1e-6f)` - **never `rsqrt`**, a ~2 ulp approximation
//     with no cross-implementation guarantee - multiplies by the fp32 `(1 + w)`
//     weight the loader baked, rounds back to bf16 and widens again:
//     `nrm[i] = bf16f(rne_bf16(bf16f(x_b) · rstd · w[i]))`;
//   * **RoPE runs on that fp32 widened normalised value**, one rounded product
//     plus one `fma` per pair. `attn_q` keeps the fp32 result; the k written to
//     the cache is `rne_bf16` of it. torch reaches the same value through bf16
//     tensor ops and so rounds once more, inside the cos/sin multiply-add - a
//     **≤ 1-op difference**, taken deliberately (controller ruling 2026-08-25)
//     and recorded in docs/12-kernels.md;
//   * `attn_gate[m][h][i] = bf16f(rne_bf16(gate columns))` - the linear's
//     rounding, widened for the sigmoid that comes at the very end;
//   * `kv_v` is just `rne_bf16(partials)`: v is never normed and never roped;
//   * the whole softmax - scores, `mx`, `sm`, `acc`, the block merge - is fp32
//     and rounds nothing; `attn_part` is fp32 for the same reason;
//   * the final chain is the reference's - eager attention output cast to bf16,
//     then `attn_output * sigmoid(gate)` as a bf16 × fp32 → fp32 product cast
//     back: `rne_bf16(bf16f(rne_bf16(acc/sm)) · sigmoid_f32(gate))`.
//
// Every kernel is built with `-cl-fp32-correctly-rounded-divide-sqrt`
// (cmake/ocloc.cmake), so `/` and `sqrt` match the host bit for bit - which is
// what lets `kv_k`, `kv_v`, `attn_gate` and `attn_q`'s pass-through dims be
// held **bit-exact**. `exp` is the one function here that OpenCL does not
// require to be correctly rounded (3 ulp), and its slack - in the softmax, in
// the merge and in the final sigmoid - is the entire reason `attn_out` and
// `attn_part` carry a tolerance.

#ifndef M
#define M 1
#endif
#ifndef MAXLEN
#define MAXLEN 16384
#endif
#ifndef CTRL_POS
#error "attn: CTRL_POS must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef CTRL_NACT
#error "attn: CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef QNORM_OFF
#error "attn: QNORM_OFF must be defined (loader/small_layout.h kFaOffQNorm / 4)"
#endif
#ifndef KNORM_OFF
#error "attn: KNORM_OFF must be defined (loader/small_layout.h kFaOffKNorm / 4)"
#endif

// The model's dimensions (model::Qwen35); none of them is a variant.
#define Q_HEADS 24        /* full-attention q-heads */
#define KV_HEADS 4        /* k/v heads */
#define GQA 6             /* q-heads per kv-head: 24 / 4 */
#define HD 256            /* head dim, q, k and v alike */
#define QKV_N 14336       /* q‖gate (12288) ‖ k (1024) ‖ v (1024) */
#define K_OFF 12288
#define V_OFF 13312
#define ROT_HALF 32       /* partial_rotary_factor 0.25 of 256 -> 64 dims, 32 pairs */
#define ROT_DIM 64
#define OUT_N 6144        /* 24 x 256 */
#define SCALE 0.0625f     /* 1/sqrt(256) */

// The fused qkv GEMV's split-K slice count (model::Qwen35's table). At S = 1
// the fp32 `[S][M][N]` partials collapse to `[M][N]`, which is what every load
// below assumes. If qkv is ever re-tuned to a wider split-K, those loads must
// sum the slices - and attn_ref.h with them. Fail the build here rather than
// silently read slice 0.
// The `#error` below can only check this file against itself; the other half of
// the pairing - that QKV_S still equals `model::Qwen35`'s Qkv.S - is asserted at
// capture time by runtime::Capture::check_sizes (src/runtime/capture.cc), which
// is the only place that can see both numbers.
#define QKV_S 1
#if QKV_S != 1
#error "attn assumes qkv runs S=1; the partials index must loop s otherwise"
#endif

#define ATTN_BLOCK 256    /* KV positions per attn_decode work-group */
#define WAVES 16          /* ATTN_BLOCK / WAVE_P */
#define WAVE_P 16         /* positions per wave = subgroups per work-group */
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define PER_LANE 16       /* elements of the 256-dim dot per lane: 256 / 16 */
#define NBLOCKS (MAXLEN / ATTN_BLOCK)
#define PART 258          /* {mx, sm, acc[256]} per (q-head, block, token) */

#define WG_PREP 256
#define WG_DEC 256
#define WG_RED 256

#if MAXLEN % ATTN_BLOCK != 0
#error "attn: MAXLEN must be a multiple of ATTN_BLOCK (the grid is MAXLEN/256 blocks)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// f32 -> bf16, round-to-nearest-even. Same add-and-shift as common::f32_to_bf16
// and prep.cl's / gdn_step.cl's; NaN is not expected and not handled.
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// sigmoid(x) = 1 / (1 + exp(-x)), plain `exp` (not `native_exp`). It is the
// LAST op of attn_reduce for the same reason prep.cl keeps its SiLU last:
// everything before it stays bit-comparable with the host.
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------
// attn_prep - grid (28, M), work-group 256. Work-groups 0..23 are q-heads,
// 24..27 are kv-heads (kv-head j = group 24+j). Work-item `i` owns dim `i`, so
// the norm's reduction domain is exactly the work-group.
//
//   x_b   = rne_bf16(qkv_partials[0][m][col])          (the linear's bf16 out)
//   rstd  = 1 / sqrt(mean_i(f32(x_b)²) + 1e-6)          (tree 1; never rsqrt)
//   nrm[i]= f32(rne_bf16(f32(x_b[i]) · rstd · w[i]))    (fp32 (1+w) weight)
//   RoPE over dims 0..63, pairs (i, i+32), on nrm; dims 64..255 pass through
//   q-head:  attn_q[m][h][i]    = roped (fp32)
//            attn_gate[m][h][i] = f32(rne_bf16(gate column))
//   kv-head: kv_k[pos+m][j][i]  = rne_bf16(roped)
//            kv_v[pos+m][j][i]  = rne_bf16(v column)     (never normed or roped)
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_PREP, 1, 1)))
__kernel void attn_prep(__global const uint* restrict ctrl,
                        __global const float* restrict qkv_partials,
                        __global const float* restrict fa_small,
                        __global const float* restrict rope,
                        __global float* restrict attn_q, __global float* restrict attn_gate,
                        __global ushort* restrict kv_k, __global ushort* restrict kv_v) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[WG_PREP];   // the norm's sum of squares
  __local float nrm[HD];        // the normalised head, fp32 - RoPE needs dim i+-32

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;     // the compiled variant is the ceiling
  if (m >= n_act) return;       // uniform across the work-group

  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;                  // q-head, or kv-head j
  __global const float* restrict nw = fa_small + (is_q ? QNORM_OFF : KNORM_OFF);
  // Slice QKV_S - 1 == 0: with one slice there is nothing to sum (the `#if`
  // above is what keeps that true).
  const size_t row = ((size_t)(QKV_S - 1) * M + m) * QKV_N;
  const size_t base = row + (is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD);

  const float xf = bf16f(rne_bf16(qkv_partials[base + i]));
  red[i] = xf * xf;                                        // one term per lane: plain multiply
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_PREP / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = red[0] / (float)HD;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);             // never rsqrt
  nrm[i] = bf16f(rne_bf16(xf * rstd * nw[i]));
  barrier(CLK_LOCAL_MEM_FENCE);

  // Partial RoPE, rotate_half over the 64-slice (doc 03: non-interleaved
  // halves, dims 64..255 untouched):
  //   out_i      = x_i·cos_i     − x_{i+32}·sin_i
  //   out_{i+32} = x_{i+32}·cos_i + x_i·sin_i
  // One rounded product plus one `fma`, spelled identically in attn_ref.h so
  // the two land on the same bits. cos/sin come from the loader's table, which
  // computed the angles in double (docs/13-loader.md, "The RoPE table").
  __global const float* restrict cs = rope + (size_t)(pos + m) * 2 * ROT_HALF;
  float outv;
  if (i < ROT_HALF) {
    const float c = cs[i], s = cs[ROT_HALF + i];
    const float t = nrm[i + ROT_HALF] * s;
    outv = fma(nrm[i], c, -t);
  } else if (i < ROT_DIM) {
    const uint ii = i - ROT_HALF;
    const float c = cs[ii], s = cs[ROT_HALF + ii];
    const float t = nrm[ii] * s;
    outv = fma(nrm[i], c, t);
  } else {
    outv = nrm[i];
  }

  if (is_q) {
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = outv;
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_partials[base + HD + i]));
  } else {
    const size_t slot = ((size_t)(pos + m) * KV_HEADS + h) * HD + i;
    kv_k[slot] = rne_bf16(outv);
    kv_v[slot] = rne_bf16(qkv_partials[row + V_OFF + (size_t)h * HD + i]);
  }
}

// ---------------------------------------------------------------------------
// attn_decode - grid (4 kv-heads, MAXLEN/256 blocks), work-group 256 = 16
// subgroups × 16 lanes. Work-group `(j, blk)` owns one block of kv-head `j` and
// loops over that head's six q-heads and the tokens in flight. Work-item `lid`
// owns accumulator dim `lid`; subgroup `lid/16` owns one position of the wave
// and its lane `lid%16` sixteen elements of that position's dot.
//
// SLM is 2 KB: the q-head staged once per (q-head, token) - all 16 subgroups
// read the same 256 floats, so staging turns 16 redundant global reads into one
// - and the 256 lane partials of the score dot.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_DEC, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode(__global const uint* restrict ctrl,
                          __global const float* restrict attn_q,
                          __global const ushort* restrict kv_k,
                          __global const ushort* restrict kv_v,
                          __global float* restrict attn_part) {
  const uint j = get_group_id(0);            // kv-head
  const uint blk = get_group_id(1);          // 256-position block
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;                // owns this wave's position sgid
  const uint lane = lid % SG;                // owns 16 elements of that dot
  __local float qs[HD];                      // the q-head, staged fp32
  __local float dot_red[WG_DEC];             // [16 subgroups][16 lanes]

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

  // **The early-out.** Uniform across the work-group, per BLOCK and not per
  // token: a block live for any `m` runs for every `m`, and a position past a
  // given `m`'s causal bound is masked inside the wave. Argued in the header.
  const uint bstart = blk * ATTN_BLOCK;
  if (bstart >= pos + n_act) return;

  for (uint qhl = 0; qhl < GQA; ++qhl) {
    const uint qh = j * GQA + qhl;           // GQA 6:1 - q-head qh reads kv-head qh/6
    for (uint m = 0; m < n_act; ++m) {
      barrier(CLK_LOCAL_MEM_FENCE);          // the previous (q-head, token) is done with SLM
      qs[lid] = attn_q[((size_t)m * Q_HEADS + qh) * HD + lid];
      barrier(CLK_LOCAL_MEM_FENCE);

      const uint bound = pos + m;            // causal: position p contributes iff p <= bound
      float mx = -INFINITY, sm = 0.0f, acc = 0.0f;

      for (uint w = 0; w < WAVES; ++w) {
        // --- the wave's 16 dots: subgroup sgid owns position `p` -------------
        const uint p = bstart + w * WAVE_P + sgid;
        float a = 0.0f;
        if (p <= bound) {                    // uniform within the subgroup
          __global const ushort* restrict krow = kv_k + ((size_t)p * KV_HEADS + j) * HD;
          for (uint t = 0; t < PER_LANE; ++t) {          // ascending t, explicit fma
            const uint d = lane + SG * t;                // lanes read 16 consecutive dims
            a = fma(qs[d], bf16f(krow[d]), a);
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);        // every read of last wave's dot_red is done
        dot_red[lid] = a;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = SG / 2; stride > 0; stride >>= 1) {
          if (lane < stride) dot_red[sgid * SG + lane] += dot_red[sgid * SG + lane + stride];
          barrier(CLK_LOCAL_MEM_FENCE);
        }

        // --- the online update, computed redundantly by all 256 work-items ---
        // Same words, same ascending order, so every work-item holds the same
        // bits of (mx, sm) and of the 16 weights; nothing is published and no
        // work-item diverges. `sc` becomes the weights in place.
        float sc[WAVE_P];
        for (uint s = 0; s < WAVE_P; ++s) {
          const uint ps = bstart + w * WAVE_P + s;
          sc[s] = ps <= bound ? dot_red[s * SG] * SCALE : -INFINITY;   // causal mask
        }
        float nmx = mx;
        for (uint s = 0; s < WAVE_P; ++s) nmx = fmax(nmx, sc[s]);
        float resc;
        if (nmx > -INFINITY) {
          resc = exp(mx - nmx);              // mx = -INF -> 0; mx = nmx -> 1
          float ssum = 0.0f;
          for (uint s = 0; s < WAVE_P; ++s) {
            sc[s] = exp(sc[s] - nmx);        // masked -> exp(-INF) = 0
            ssum += sc[s];
          }
          sm = fma(sm, resc, ssum);
          mx = nmx;
        } else {
          resc = 1.0f;                       // nothing valid yet: skip the wave whole,
          for (uint s = 0; s < WAVE_P; ++s) sc[s] = 0.0f;   // exp(-INF - -INF) is a NaN
        }

        // --- this work-item's accumulator dim -------------------------------
        float tsum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) {   // ascending s, explicit fma
          const uint ps = bstart + w * WAVE_P + s;
          // The masked slot's weight is already 0; skipping the load as well is
          // what stops an unwritten cache slot from turning 0·x into a NaN.
          const float vf =
              ps <= bound ? bf16f(kv_v[((size_t)ps * KV_HEADS + j) * HD + lid]) : 0.0f;
          tsum = fma(sc[s], vf, tsum);
        }
        acc = fma(acc, resc, tsum);
      }

      __global float* restrict out =
          attn_part + (((size_t)qh * NBLOCKS + blk) * M + m) * PART;
      out[2 + lid] = acc;
      if (lid == 0) {
        out[0] = mx;
        out[1] = sm;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// attn_reduce - grid (24 q-heads, M), work-group 256. Work-item `d` owns dim
// `d`. The `nb` block headers are staged into SLM in one pass so the merge loop
// itself has no barrier; every work-item then runs the identical scalar merge
// (order 4 of the header) alongside its own `acc`.
//
//   nb = (pos + m)/256 + 1
//   merge blocks 0 … nb-1 ascending ;  out = acc / sm
//   attn_out[m][h·256+d] = rne_bf16(f32(rne_bf16(out)) · sigmoid_f32(gate))
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_RED, 1, 1)))
__kernel void attn_reduce(__global const uint* restrict ctrl,
                          __global const float* restrict attn_part,
                          __global const float* restrict attn_gate,
                          __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[NBLOCKS], hsm[NBLOCKS];

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;                    // uniform across the work-group

  // Every block up to and including the one holding position `pos+m` ran, and
  // no other block is read - which is what lets attn_decode's early-out leave
  // the rest untouched (header, "The early-out").
  uint nb = (pos + m) / ATTN_BLOCK + 1;
  // The real precondition is `pos + n_active <= max_len`, which the engine
  // enforces when it advances `Control::pos` - under it `nb <= NBLOCKS` always.
  // This clamp buys nothing when that holds; it exists so that a *violated*
  // precondition is a wrong answer rather than an SLM buffer overrun writing
  // past `hmx`/`hsm` into whatever the compiler put next.
  if (nb > NBLOCKS) nb = NBLOCKS;
  for (uint b = d; b < nb; b += WG_RED) {
    __global const float* restrict p =
        attn_part + (((size_t)h * NBLOCKS + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {            // ascending block order
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[(((size_t)h * NBLOCKS + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);         // finite: block b starts at or before pos+m
    const float a = exp(mx - nmx);           // mx = -INF on the first block -> 0
    const float bs = exp(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }

  const float o = acc / sm;
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * sigmoid_f32(g));
}
