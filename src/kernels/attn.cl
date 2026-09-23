// attn.cl - the decode-attention trio: `attn_prep` (q/k RMSNorm + partial RoPE
// + the KV cache write), `attn_decode` (one flash-decode block of ATTN_BLOCK
// KV positions per work-group, with the device-side early-out that makes a
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
//     qkv_partials  fp32 [2][M][14336]      the fused qkv GEMV's partials (S = 2)
//     fa_small      fp32                    this layer's FA block: q_norm (1+w)[256]
//                                           at QNORM_OFF, k_norm at KNORM_OFF
//                                           (loader/small_layout.h)
//     rope          fp32 [max_len][2][32]   cos at [p][0][i], sin at [p][1][i]
//     attn_q        fp32 [M][24][256]       normed + roped, fp32
//     attn_gate     fp32 [M][24][256]       f32(rne_bf16(the gate columns))
//     kv_k, kv_v    bf16 [max_len][4][256]  this layer's caches
//
//   attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)
//     attn_part     fp32 [24][MAXLEN/ATTN_BLOCK][M][258]   {mx, sm, acc[256]}
//
//   attn_reduce(ctrl, attn_part, attn_gate, attn_out)
//     attn_out      bf16 [M][6144]
//
// ---------------------------------------------------------------------------
// The column map of the fused qkv linear
// ---------------------------------------------------------------------------
// `q_proj ‖ k_proj ‖ v_proj` = 12288 + 1024 + 1024 = 14336 columns (doc 03;
// model::Qwen35's `LinearId::Qkv`, `Fuse::Concat`, S = 2). Inside q_proj the 24
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
// **attn_decode: grid (4 kv-heads, MAXLEN/ATTN_BLOCK blocks), work-group 256 =
// 16 subgroups × 16 lanes (SIMD16).** The grid spans the whole cache because
// the captured list cannot be re-sized per token; the work-groups past the
// context exit on their first instruction (below). A work-group owns one
// ATTN_BLOCK-position block of one kv-head and loops over that head's **six**
// q-heads (GQA 6:1) and the tokens in flight.
//
// **The block is 64 positions, and that is the whole of spec 1.5's lever L5.**
// It was 256. docs/15 §2's four in-situ points fit `F + fill · P` - F ≈ 90.8 µs
// of per-launch fixed cost, P ≈ 247.4 µs for one work-group's serial walk of
// one FULL block - so at depth 4096 roughly 73% of a 361 µs launch was one
// work-group walking one block, and the same measurement found work-group
// COUNT nearly free (`nb` 5 → 17 is 3.40× the live work-groups for +6.9% of
// time). That fit did NOT survive the retile (see ATTN_BLOCK below: it
// over-predicted the B128 gain by 38%), but its *direction* did, and the block
// size that pays was then measured at four values rather than extrapolated.
// Quartering the block shortens that walk and multiplies the grid: at
// depth 4096, 65 blocks × 4 kv-heads = **260 work-groups**, against 68 before.
// docs/15 called 264 work-groups "past the point any measurement here reaches";
// it is measured now, and it is fine. Nothing else about the kernel changed -
// the wave is still 16 positions, the online-softmax arithmetic inside a wave
// is untouched, and the order the waves compose in is untouched. What moves is
// only WHERE the partial-accumulation boundary falls: **4** waves per block
// instead of 16, with `attn_reduce` merging four times as many blocks in the
// same fixed ascending order.
// That reassociation is a real (last-ulp) change to `attn_part` and `attn_out`,
// which is why the golden gate is run before and after (docs/14).
//
// Register-packed GQA (Task 5) keeps the (kv-head, block) grid but inverts the
// six q-head walks inside each work-group. For one token, all six q heads occupy
// qpack (6 KB SLM); then each work-item loads its 16 K values and 16 V values for
// one wave into private registers once and reuses them for six independent
// (mx, sm, acc) states. The per-head score fma order, 16-lane tree, online
// softmax order and V fma order remain exactly as stated below. qpack plus dot_red
// is 7 KB SLM; K/V are deliberately private rather than staged through SLM. The
// grid, launch count and module count do not move. docs/12 records the probe and
// in-situ attribution.
//
// **Two separate claims, with separate evidence - do not merge them.**
//   * *Replay determinism*: `replay_determinism_test` proves the captured list
//     replays byte-identically. That is necessary but it is NOT evidence of
//     order preservation - it passed before this change too, and would pass for
//     any deterministic reassociation.
//   * *Order preservation*: the evidence is the probe's byte-identity **to the
//     base kernel's output** (the register-GQA probe's `greg6` row reported
//     "bytes: identical"), which is a comparison ACROSS the change and is the
//     claim that matters.
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
// exactly `nb(m) = (pos+m)/ATTN_BLOCK + 1` blocks - every one of which starts at or
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
//  3. **The online softmax wave** (`attn_decode`): the block's ATTN_BLOCK
//     positions are walked in ATTN_BLOCK/16 waves of 16 - 4 waves since lever
//     L5. With the wave's scores `sc[0..15]` (−INF where
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
//   * `x_b = rne_bf16(Σ_s qkv_partials[s][…])` - the qkv linear's two slices,
//     summed in ascending order and rounded once;
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

// The fused qkv GEMV's split-K slice count (model::Qwen35's table). Task 4
// retunes it to S=2; attn_prep folds those slices in ascending order before the
// linear's bf16 rounding. attn_ref.h mirrors that exact order. The other half
// of the pairing is asserted by runtime::Capture::check_sizes.
#define QKV_S 2

// ATTN_BLOCK - KV positions per attn_decode work-group, and the ONE number
// this kernel's blocking depends on. It is a `-D` rather than a literal
// because it must agree with `runtime::DecodeBuffers::kAttnBlock` on the host,
// which sizes `attn_part` and sets attn_decode's grid: it is in the compiled
// binary's NAME (`attn_decode_M1_L16384_B64`), so a host that disagrees names
// a file that does not exist and throws at capture rather than striding
// `attn_part` wrongly - the arrangement `prep_res_fold`'s `G` uses for exactly
// the same reason (src/kernels/CMakeLists.txt).
//
// It was 256 until spec 1.5's lever L5, and it is now **64** - a number that
// was measured, not derived. docs/15 §2's `F + fill·P` fit (F ≈ 90.8 µs fixed,
// P ≈ 247.4 µs per full block, from the `nb` = 1 and 5 points) predicted
// 214.5 µs/launch at a 128-position block; the retile measured **296.684**,
// +38%, which is the third model this kernel has falsified - and the refit on
// the 256/128 pair (F ≈ 223.4 µs, walk ≈ 146.6 µs per 256 positions) then
// predicted 260.0 µs at a 64-position block and measured **224.046**, −13.8%,
// which is the fourth. Four block sizes,
// measured in situ at depth 4096 - `attn_decode` µs/launch, then the attn
// family's ms/token:
//
//     B256  369.988  6.058 | B128  296.684  4.931
//     B64   224.046  3.839 | B32   203.837  3.769
//
// **64 is NOT a knee** - B32's family total is 0.070 ms/token *better*, and the
// step's Σ difference that once looked like a crossover (+19.5 µs) is inside
// the +0.283% drift the untouched launches showed between those two runs. 64 is
// chosen because the marginal gain has collapsed (the three halvings bought
// −1.127, −1.092 and −0.070 ms/token, the last about a third of one run's
// drift), because `attn_reduce`'s merge is on a steep ramp (80 → 127 → 196 →
// 450 µs/step, taking back 254 of `attn_decode`'s 323 at B32), and because
// `attn_part` would double again to 101.4 MB of device scratch for that
// 0.070 ms. What is *measured* and not fitted: the three points 256 → 128 → 64 each cost
// ~73 µs less per launch than the last, which is linear in log2(block) and not
// in the block; nothing here explains that, and this file does not invent a
// fifth model to fit it (docs/12 `attn` → Measured).
#ifndef ATTN_BLOCK
#error "attn: ATTN_BLOCK must be defined (src/kernels/CMakeLists.txt); it must equal runtime::DecodeBuffers::kAttnBlock"
#endif
#define WAVE_P 16         /* positions per wave = subgroups per work-group */
#define WAVES (ATTN_BLOCK / WAVE_P)   /* waves per block: 4 at ATTN_BLOCK 64 */
#if ATTN_BLOCK % WAVE_P != 0
#error "attn: ATTN_BLOCK must be a multiple of WAVE_P (16) - a block is walked in whole waves"
#endif
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define PER_LANE 16       /* elements of the 256-dim dot per lane: 256 / 16 */
#define NBLOCKS (MAXLEN / ATTN_BLOCK)
#define PART 258          /* {mx, sm, acc[256]} per (q-head, block, token) */

#define WG_PREP 256
#define WG_DEC 256
#define WG_RED 256

#if MAXLEN % ATTN_BLOCK != 0
#error "attn: MAXLEN must be a multiple of ATTN_BLOCK (the grid is MAXLEN/ATTN_BLOCK blocks)"
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

inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
}

// ---------------------------------------------------------------------------
// attn_prep - grid (28, M), work-group 256. Work-groups 0..23 are q-heads,
// 24..27 are kv-heads (kv-head j = group 24+j). Work-item `i` owns dim `i`, so
// the norm's reduction domain is exactly the work-group.
//
//   x_b   = rne_bf16(Σ_s qkv_partials[s][m][col])      (the linear's bf16 out)
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
  const size_t base = is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD;

  const float xf = bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + i)));
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
        bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + HD + i)));
  } else {
    const size_t slot = ((size_t)(pos + m) * KV_HEADS + h) * HD + i;
    kv_k[slot] = rne_bf16(outv);
    kv_v[slot] = rne_bf16(qkv_sum(qkv_partials, m, V_OFF + (size_t)h * HD + i));
  }
}

// ---------------------------------------------------------------------------
// attn_decode - grid (4 kv-heads, MAXLEN/ATTN_BLOCK blocks), work-group 256 = 16
// subgroups × 16 lanes. Work-group `(j, blk)` owns one block of kv-head `j` and
// loops over that head's six q-heads and the tokens in flight. Work-item `lid`
// owns accumulator dim `lid`; subgroup `lid/16` owns one position of the wave
// and its lane `lid%16` sixteen elements of that position's dot.
//
// SLM is 7 KB: all six q-heads are staged once per token (6 × 1 KB), then the
// 256 lane partials of the score dot. K/V never enter SLM: a work-item loads its
// own 16 K elements and 16 V elements into private registers once per wave, then
// runs the unchanged per-head dot, tree, online-softmax, and V-FMA sequences.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_DEC, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode(__global const uint* restrict ctrl,
                          __global const float* restrict attn_q,
                          __global const ushort* restrict kv_k,
                          __global const ushort* restrict kv_v,
                          __global float* restrict attn_part) {
  const uint j = get_group_id(0);            // kv-head
  const uint blk = get_group_id(1);          // ATTN_BLOCK-position block
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;                // owns this wave's position sgid
  const uint lane = lid % SG;                // owns 16 elements of that dot
  __local float qpack[GQA * HD];             // all six q-heads, staged fp32
  __local float dot_red[WG_DEC];             // [16 subgroups][16 lanes]

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

  // The early-out is uniform across the work-group, per block and not per token.
  const uint bstart = blk * ATTN_BLOCK;
  if (bstart >= pos + n_act) return;

  // Invert the former q-head-outer walk: one private K/V wave feeds all six
  // heads of kv-head j. Each head retains the base kernel's arithmetic order.
  for (uint m = 0; m < n_act; ++m) {
    for (uint qhl = 0; qhl < GQA; ++qhl) {
      const uint qh = j * GQA + qhl;
      qpack[qhl * HD + lid] = attn_q[((size_t)m * Q_HEADS + qh) * HD + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float mx[GQA];
    float sm[GQA];
    float acc[GQA];
    for (uint qhl = 0; qhl < GQA; ++qhl) {
      mx[qhl] = -INFINITY;
      sm[qhl] = 0.0f;
      acc[qhl] = 0.0f;
    }

    const uint bound = pos + m;              // causal: position p contributes iff p <= bound
    for (uint w = 0; w < WAVES; ++w) {
      // Every work-item owns one slice of K and one accumulator dimension of
      // all 16 V rows. These are exactly the base addresses, loaded once.
      const uint p = bstart + w * WAVE_P + sgid;
      ushort kreg[PER_LANE];
      ushort vreg[WAVE_P];
      for (uint t = 0; t < PER_LANE; ++t) {
        const uint d = lane + SG * t;
        const size_t kidx = ((size_t)p * KV_HEADS + j) * HD + d;
        kreg[t] = p <= bound ? kv_k[kidx] : (ushort)0;
      }
      // The masked slot's weight is already 0; skipping the load as well is what
      // stops an unwritten cache slot from turning 0·x into a NaN. The register
      // form does not weaken that: the `(ushort)0` substitute widens through
      // `bf16f` to exactly `+0.0f`, bit-identical to the base path's literal
      // `0.0f`, so the V FMA below sees the same operand it always saw.
      // **`vreg[s]` is consumed UNCONDITIONALLY** by that FMA (the base path's
      // ternary was at the point of use; here it is at the point of load), so an
      // editor who makes this load unconditional re-introduces the NaN.
      // `tests/kernels/attn_ref.h:283-284` states the same invariant and
      // `attn.cl:11-12` requires the two to be edited together.
      for (uint s = 0; s < WAVE_P; ++s) {
        const uint ps = bstart + w * WAVE_P + s;
        const size_t vidx = ((size_t)ps * KV_HEADS + j) * HD + lid;
        vreg[s] = ps <= bound ? kv_v[vidx] : (ushort)0;
      }

      for (uint qhl = 0; qhl < GQA; ++qhl) {
        float a = 0.0f;
        if (p <= bound) {                    // uniform within the subgroup
          for (uint t = 0; t < PER_LANE; ++t) {          // ascending t, explicit fma
            const uint d = lane + SG * t;                // lanes read 16 consecutive dims
            a = fma(qpack[qhl * HD + d], bf16f(kreg[t]), a);
          }
        }
        dot_red[lid] = a;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = SG / 2; stride > 0; stride >>= 1) {
          if (lane < stride) dot_red[sgid * SG + lane] += dot_red[sgid * SG + lane + stride];
          barrier(CLK_LOCAL_MEM_FENCE);
        }

        // --- the online update, computed redundantly by all 256 work-items ---
        // Ordered exactly as the base path, and redundant for the same reason:
        // same words, same ascending order, so every work-item holds the same
        // bits of (mx[qhl], sm[qhl]) and of the 16 weights; nothing is published
        // and no work-item diverges. `sc` becomes the weights in place. Register
        // packing makes the six heads' states six private slots instead of one -
        // it does not make any of them work-item-dependent.
        float sc[WAVE_P];
        for (uint s = 0; s < WAVE_P; ++s) {
          const uint ps = bstart + w * WAVE_P + s;
          sc[s] = ps <= bound ? dot_red[s * SG] * SCALE : -INFINITY;   // causal mask
        }
        float nmx = mx[qhl];
        for (uint s = 0; s < WAVE_P; ++s) nmx = fmax(nmx, sc[s]);
        float resc;
        if (nmx > -INFINITY) {
          resc = exp(mx[qhl] - nmx);          // mx = -INF -> 0; mx = nmx -> 1
          float ssum = 0.0f;
          for (uint s = 0; s < WAVE_P; ++s) {
            sc[s] = exp(sc[s] - nmx);         // masked -> exp(-INF) = 0
            ssum += sc[s];
          }
          sm[qhl] = fma(sm[qhl], resc, ssum);
          mx[qhl] = nmx;
        } else {
          resc = 1.0f;                        // nothing valid yet: skip the wave whole,
          for (uint s = 0; s < WAVE_P; ++s) sc[s] = 0.0f;   // exp(-INF - -INF) is a NaN
        }

        float tsum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) {   // ascending s, explicit fma
          tsum = fma(sc[s], bf16f(vreg[s]), tsum);
        }
        acc[qhl] = fma(acc[qhl], resc, tsum);

        // **The only fence closing the head loop's two write-after-reads, and
        // it closes both.** (a) `dot_red`: the next head overwrites it, and all
        // 256 work-items must finish reading this head's tree - including the
        // cross-subgroup `dot_red[s·16]` reads above - before that happens.
        // (b) `qpack` across the `m` loop: the last `qpack` read of token `m` is
        // head 5's score dot, and the first `qpack` write of token `m+1` is the
        // staging store at the top; this fence lies between them.
        // A leading fence before `dot_red[lid] = a` closes exactly the same two
        // and used to be here as well. **At least one of the two must survive.**
        // The leading one was the one deleted, because it was the redundant
        // copy: 24 instances per launch. Executed work-group barriers per launch
        // at M = 1 are now `1 + WAVES·GQA·6` = **145**, against 169 with both and
        // 156 in the pre-Task-5 kernel.
        //   It bought **−2.295 µs/launch, −0.037 ms/token** (measured in situ,
        // iterate/relative, byte-identical output). That is **39% of what
        // `probe_attn`'s `nobar` row predicted** - 0.242 µs per barrier × 24 =
        // 5.82 µs/launch - so the fifth model to die on this kernel is
        // "barriers cost their average": the marginal price of the *redundant*
        // fence is **0.096 µs**, because the deleted one sat two instructions
        // before a surviving one and most of the convergence is paid there
        // anyway. docs/12 records the pair.
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    for (uint qhl = 0; qhl < GQA; ++qhl) {
      const uint qh = j * GQA + qhl;
      __global float* restrict out =
          attn_part + (((size_t)qh * NBLOCKS + blk) * M + m) * PART;
      out[2 + lid] = acc[qhl];
      if (lid == 0) {
        out[0] = mx[qhl];
        out[1] = sm[qhl];
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
//   nb = (pos + m)/ATTN_BLOCK + 1
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
  // NBLOCKS went 4x with lever L5 (ATTN_BLOCK 256 -> 64), so these two arrays
  // did too: at MAXLEN 16384 they are 256 floats each, **2048 B of SLM
  // together** against the 64 KB a work-group may have. The merge loop below is
  // bounded by `nb`, not by NBLOCKS, so the extra length costs nothing at short
  // context. What it DOES cost at depth 4096 is four times as many merge steps
  // (nb 17 -> 65), and that is measured: this kernel goes 80.241 -> 196.195
  // µs/step, +116 µs, against `attn_decode`'s -2336 µs. It is why the lever is
  // judged on the attn FAMILY's net and not on `attn_decode` alone, and it is
  // also what stops the retile one step earlier than `attn_decode` would like:
  // at ATTN_BLOCK 32 this row reaches 450.208 µs/step and eats the whole
  // remaining gain (docs/12 `attn` -> Measured).
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
