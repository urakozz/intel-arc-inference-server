// kv8.cl - spec 12b: the int8 KV cache, the `rotkv` scheme (spec 12 §8), every
// kernel that writes or reads it. ONE source, built per family by a define (pf_int8.cl's
// arrangement), so the rotation, the quantiser and the scale format are one piece of text:
//
//   KV8_PREP   attn_prep_kv8         the writer: attn.cl's attn_prep (decode, M rows, QKV_S
//              (M, QKV_S | PF=1)      slices) or pf_attn_prep.cl's q16 build (PF = 1:
//                                     prefill, one slice, bf16 q, no gate), then q, K and V
//                                     rotated and K, V quantised into the int8 cache
//   KV8_DEC    attn_decode_v2_kv8    attn_v2.cl's pair reading int8 K / V + fp16 scales,
//              attn_reduce_v2_kv8     the output un-rotated per head before the gate
//              (M, TGT)
//   KV8_FLASH  pf_flash_attn_kv8     pf_flash_attn.cl reading int8 K / V + scales
//              (KT, RPW, HPW, QREG, EXP2)
//   KV8_GATE   pf_attn_gate_kv8      pf_attn.cl's pf_attn_gate with the un-rotation first
//
// Every existing kernel source is untouched: these are new binaries (`*_KV8`, bound only
// under --kv-cache int8), so `--kv-cache bf16` runs today's binaries bit for bit.
//
// ---------------------------------------------------------------------------
// The scheme (src/common/kv8.h is the host twin, kv8_scheme.h the signs and scale format; edit them together)
// ---------------------------------------------------------------------------
//   R = H_256 diag(s) / 16, H Sylvester in natural order, s = KV8_SIGNS (the 12a probe's
//   hadamard(256, 0): bit j of word j / 32 set when s[j] = -1).
//   rotate(x)[j]   = s[j] FWHT(x)[j] / 16          (q, and the K and V rows)
//   unrotate(y)[i] = FWHT(s (.) y)[i] / 16         (the attention output, per head)
//   FWHT: stages h = 1, 2, ..., 128 ascending; (i, i + h), i & h == 0 -> (a + c, a - c).
//   q.k = (qR).(kR), so the scores are unchanged; V's rotation comes back out of
//   sum_k p_k (v_k R) = o R by the un-rotation, which must come BEFORE the sigmoid gate
//   (the gate is channel-wise in the un-rotated basis, so R^T cannot fold into o_proj).
//
//   What is rotated is what the bf16 cache would have stored: K = f32(rne_bf16(roped k)),
//   V = f32(rne_bf16(v)). q is rotated from attn_prep's fp32 value (decode keeps it fp32,
//   prefill rounds it to bf16 as pf_attn_prep_q16 does).
//
//   Per (position, kv head):  amax = max |y|;  s16 = f16_rne(amax / 127);
//     q8 = s16 == 0 ? 0 : clamp(rint(y / f16f(s16)), -127, 127);  deq = float(q8) f16f(s16)
//   (deq is exact in fp32: 7 + 11 significant bits).
//
// Layout (runtime/buffer_sizes.h KvLayout): one layer's K (or V) is int8 [max_len][4][256];
// its scales are fp16 bits [max_len][4], in the same allocation after every layer's rows.
// The kernels take the two pointers.
//
// The readers dequantise while loading:
//   decode  K and V exactly, deq in fp32 (the attention arithmetic is attn_v2.cl's, fp32,
//           on those values);
//   flash   K's int8 is the bf16 DPAS operand as is (exact) and the score is multiplied by
//           the key's scale; V's int8 likewise, its scale folded into P (P s_k rounded once
//           to bf16, where the bf16 engine rounds P alone). The row sum is P's, unscaled.
//
// Rounding: the FWHT, the flips, the 1/16 and the quantiser are bit-defined (the build has
// -cl-fp32-correctly-rounded-divide-sqrt, `rint` is round-half-even, max is exact), so the
// writer is bitwise src/common/kv8.h; the readers inherit their bf16 twins' `exp` slack.
#ifndef KV8_PREP
#define KV8_PREP 0
#endif
#ifndef KV8_DEC
#define KV8_DEC 0
#endif
#ifndef KV8_FLASH
#define KV8_FLASH 0
#endif
#ifndef KV8_GATE
#define KV8_GATE 0
#endif
#if KV8_PREP + KV8_DEC + KV8_FLASH + KV8_GATE != 1
#error "kv8: define exactly one of KV8_PREP, KV8_DEC, KV8_FLASH, KV8_GATE (src/kernels/CMakeLists.txt)"
#endif

// BATCH_ROWS (spec 13 §9, plan 13a; undefined on every existing command line): the decode writer and reader for rows
// of DIFFERENT sequences - attn.cl's / attn_v2.cl's BATCH_ROWS builds, whose headers are the contract, over the int8
// cache: the row table `rows` (runtime::BatchRowTable, words BROW_*) as the last argument, row m's position and KV
// base (the K / V rows AND their scales at kv_base + position) from it. M = 1..8 (the 8-row budget) in this build.
#ifdef BATCH_ROWS
#if !(KV8_PREP || KV8_DEC) || (KV8_PREP && defined(PF) && PF)
#error "kv8 BATCH_ROWS: the decode writer (KV8_PREP, PF 0) and reader (KV8_DEC) only"
#endif
#if !defined(BROW_WORDS) || !defined(BROW_POS) || !defined(BROW_KVB) || !defined(BROW_SLOT)
#error "kv8 BATCH_ROWS: the row table's BROW_* words must be defined (src/kernels/CMakeLists.txt)"
#endif
#endif

// KV_PAGED (plan 19v3b Task 0: the KV page in positions, runtime::kKvPage; undefined on every other command line): the
// writer, the decode reader, kv8_widen and the flash readers take the layer's whole cache (rows and scales) and a last
// argument `pages` mapping the KV pages - position p (with BATCH_ROWS kv_base + p, the slot's first position in the
// table's positions) is cache row pages[p / KV_PAGED] x KV_PAGED + p % KV_PAGED, its scale the same row of the scales.
// Readers look a page up once per page (attn_v2.cl's KV_PAGED note; a flash tile of KT keys never crosses one) and read
// through its base; every value and order is the contiguous kernel's. The widened scratch stays the sequence's own
// positions, contiguous.
#if defined(KV_PAGED) && KV8_GATE
#error "kv8 KV_PAGED: the gate reads no KV"
#endif

// KV_ONLY (plan 21-QFN-v3a, PERF-85's form: the Flash-Next MTP head's prompt pass reads the cached K / V and never
// attn_q; undefined on every other command line): the writer's grid is (KV_HEADS, rows), work-group w the full grid's
// Q_HEADS + w, the same text; no q-head work-group runs.
#if defined(KV_ONLY) && !KV8_PREP
#error "kv8 KV_ONLY: the writer only"
#endif

// KV8_I8S (PERF-114, the int8 prompt pass; undefined on every other command line): int8 attention scores in the prompt pass. The
// writer (KV8_PREP, PF = 1) quantises the rotated bf16 q per row and head over its 256 dims - s = amax / 127 (fp32),
// q8 = s == 0 ? 0 : clamp(rint(q / s), -127, 127) - into attn_q as int8 [C][q-heads][256] and s into q_scale
// [C][q-heads]. The pair flash (KV8_PAIR) reads that q8 and K's int8 rows (kv8_widen copies them, V still widened to
// bf16) and scores by int8 DPAS: s = ((float(int32 dot) * s_q) * s_k) * SCORE_SCALE - the dot exact (|dot| <= 256 x
// 127 x 127 < 2^24), the three products in that order. Softmax, P and P V are the bf16 pair's, unchanged.
#ifndef KV8_I8S
#define KV8_I8S 0
#endif
#if KV8_I8S && !((KV8_PREP && defined(PF) && PF) || (KV8_FLASH && defined(KV8_PAIR) && KV8_PAIR))
#error "kv8 KV8_I8S: the prefill writer (KV8_PREP, PF 1) and the pair flash (KV8_PAIR) only"
#endif
// KV8_OWN (PERF-107; undefined on every other command line): the l0-int8 prompt pass's attention reads the chunk's
// own rows as bf16, older rows from the int8 cache, where it scores in bf16 - a split chunk (a whole chunk scores by
// int8 DPAS, KV8_I8S, its K int8 by design). The writer (PF = 1, bf16 q) also stores rne_bf16 of each rotated K / V
// row into the widened scratch (kr, vr) at the row's position; the split readers take `own` (the chunk's first
// position) and use scale 1 for keys at or past it, whose scratch rows are those bf16 values. The raw split widens
// only the rows before `own`; the in-place split reads its tiles from int8 and, where a tile reaches `own`, takes
// those keys from the scratch instead (Kr, Vr). Every other value and order is the build without it.
#ifndef KV8_OWN
#define KV8_OWN 0
#endif
#if KV8_OWN && (KV8_I8S || !((KV8_PREP && defined(PF) && PF && !defined(BATCH_ROWS)) || KV8_FLASH))
#error "kv8 KV8_OWN: the prefill writer (PF = 1, bf16 q) and the split flash readers only"
#endif
#if KV8_OWN
#define KV8_SC(key, x) ((key) >= own ? 1.0f : (x))
#else
#define KV8_SC(key, x) x
#endif

#if KV8_DEC || KV8_FLASH
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif
#if KV8_FLASH
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#endif

// The model's dimensions (model::Qwen35; FA heads as attn.cl bakes them).
// Spec 17b: another head count comes in as -DFA_Q_HEADS / -DFA_KV_HEADS (attn.cl's override; the binary is
// `_Q<q>KV<kv>_KV8`) - tensor parallel's per-card 12 / 2. Unset, the eight lines below are token for token as before.
#ifndef FA_Q_HEADS
#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define OUT_N 6144        /* 24 x 256 */
#define QKV_N 14336       /* q||gate (12288) || k (1024) || v (1024) */
#define K_OFF 12288
#define V_OFF 13312
/* KV8_FLASH's literals by name (spec 17c), token for token the numbers they replaced */
#define KV8F_Q 24
#define KV8F_KV 4
#define KV8F_KVU 4u
#define KV8F_GQAU 6u
#else
#ifndef FA_KV_HEADS
#error "kv8: FA_Q_HEADS needs FA_KV_HEADS"
#endif
#define Q_HEADS FA_Q_HEADS
#define KV_HEADS FA_KV_HEADS
#define GQA (FA_Q_HEADS / FA_KV_HEADS)
#define HD 256
#define OUT_N (FA_Q_HEADS * HD)
#define K_OFF (2 * FA_Q_HEADS * HD)
#define V_OFF (K_OFF + FA_KV_HEADS * HD)
#define QKV_N (V_OFF + FA_KV_HEADS * HD)
#define KV8F_Q FA_Q_HEADS
#define KV8F_KV FA_KV_HEADS
#define KV8F_KVU ((uint)FA_KV_HEADS)
#define KV8F_GQAU ((uint)GQA)
#if FA_Q_HEADS % FA_KV_HEADS != 0
#error "kv8: q-heads must be a multiple of kv-heads"
#endif
#endif
#define KV8_WG 256        /* every KV8_PREP / KV8_DEC reduce / KV8_GATE work-group: one head */

#include "bf16.h"
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

#include "kv8_scheme.h"

#if KV8_PREP || KV8_DEC || KV8_GATE
// The 256-point FWHT across a 256-work-item work-group, work-item i owning element i,
// staged through `buf` (256 floats of SLM). Every work-item must call it (barriers). It
// starts and ends on a barrier-safe footing: the first write follows the caller's last
// read only through the trailing barrier of a previous call or the caller's own barrier.
inline float kv8_fwht(float v, uint i, __local float* buf) {
  for (uint h = 1u; h < (uint)HD; h <<= 1) {
    buf[i] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float o = buf[i ^ h];
    v = (i & h) ? (o - v) : (v + o);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return v;
}
inline float kv8_rotate(float x, uint i, __local float* buf) {
  return kv8_flip(kv8_fwht(x, i, buf), i) * 0.0625f;
}
inline float kv8_unrotate(float y, uint i, __local float* buf) {
  return kv8_fwht(kv8_flip(y, i), i, buf) * 0.0625f;
}
#endif

#if KV8_PREP
// ---------------------------------------------------------------------------
// attn_prep_kv8 - grid (28, rows), work-group 256 (attn.cl's attn_prep / pf_attn_prep.cl,
// whose header states the norm / RoPE chain this transcribes unchanged).
//
//   attn_prep_kv8(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate,
//                 kv_k, kv_v, k_scale, v_scale)
//     kv_k, kv_v        int8 [max_len][4][256]   this layer's rows
//     k_scale, v_scale  fp16 bits [max_len][4]   this layer's scales
//   decode (PF = 0): attn_q fp32 [M][24][256] = rotate(roped q); attn_gate as attn_prep.
//   prefill (PF = 1): attn_q is bf16 [C][24][256] = rne_bf16(rotate(roped q)); no gate
//     (pf_attn_gate_kv8 reads it out of qkv_partials); one slice at row base m * QKV_N.
// ---------------------------------------------------------------------------
#ifndef CTRL_POS
#error "kv8 prep: CTRL_POS must be defined"
#endif
#ifndef CTRL_NACT
#error "kv8 prep: CTRL_NACT must be defined"
#endif
#ifndef QNORM_OFF
#error "kv8 prep: QNORM_OFF must be defined (loader/small_layout.h kFaOffQNorm / 4)"
#endif
#ifndef KNORM_OFF
#error "kv8 prep: KNORM_OFF must be defined (loader/small_layout.h kFaOffKNorm / 4)"
#endif
#ifndef PF
#define PF 0
#endif
#if PF
#define QKV_S 1
#else
#ifndef M
#error "kv8 prep: M must be defined for the decode build (PF = 0)"
#endif
#ifndef QKV_S
#error "kv8 prep: QKV_S must be defined (2: the main FA layers; 1: spec 8's bf16 MTP head)"
#endif
#endif
#define ROT_HALF 32
#define ROT_DIM 64

inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
#if PF
  return p[(size_t)m * QKV_N + col];
#else
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
#endif
}

// Quantise this work-item's element of one rotated row: the row's amax by the pairwise
// fmax tree over `red`, the fp16 scale, the int8. Ends on a barrier, so `red` and the
// caller's next FWHT may be reused at once.
inline void kv8_store(float y, uint i, __local float* red, __global char* restrict row,
                      __global ushort* restrict scale) {
  red[i] = fabs(y);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KV8_WG / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] = fmax(red[i], red[i + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const ushort s16 = kv8_f16_rne(red[0] / 127.0f);
  const float sf = kv8_f16f(s16);
  float r = sf == 0.0f ? 0.0f : rint(y / sf);
  r = fmin(fmax(r, -127.0f), 127.0f);
  row[i] = (char)(int)r;
  if (i == 0) *scale = s16;
  barrier(CLK_LOCAL_MEM_FENCE);   // everyone has read red[0]
}

#if KV8_I8S
// KV8_I8S's q: kv8_store's tree over the bf16 rotated q, an fp32 scale (no fp16 rounding).
inline void kv8_store_q(float y, uint i, __local float* red, __global char* restrict row,
                        __global float* restrict scale) {
  red[i] = fabs(y);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KV8_WG / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] = fmax(red[i], red[i + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float sf = red[0] / 127.0f;
  float r = sf == 0.0f ? 0.0f : rint(y / sf);
  r = fmin(fmax(r, -127.0f), 127.0f);
  row[i] = (char)(int)r;
  if (i == 0) *scale = sf;
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void attn_prep_kv8(__global const uint* restrict ctrl,
                            __global const float* restrict qkv_partials,
                            __global const float* restrict fa_small,
                            __global const float* restrict rope,
                            __global float* restrict attn_q, __global float* restrict attn_gate,
                            __global char* restrict kv_k, __global char* restrict kv_v,
                            __global ushort* restrict k_scale, __global ushort* restrict v_scale
#if KV8_I8S
                            , __global float* restrict q_scale
#endif
#if KV8_OWN
                            , __global ushort* restrict kr, __global ushort* restrict vr
#endif
#ifdef BATCH_ROWS
                            , __global const uint* restrict rows
#endif
#ifdef KV_PAGED
                            , __global const uint* restrict pages
#endif
                            ) {
#ifdef KV_ONLY
  const uint wg = Q_HEADS + get_group_id(0);
#else
  const uint wg = get_group_id(0);
#endif
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[KV8_WG];   // the norm's sum of squares, then the amax tree
  __local float nrm[HD];       // the normalised head, fp32 - RoPE needs dim i+-32
  __local float xch[HD];       // the FWHT's exchange

#ifndef BATCH_ROWS
  const uint pos = ctrl[CTRL_POS];
#endif
  uint n_act = ctrl[CTRL_NACT];
#if !PF
  if (n_act > M) n_act = M;    // the compiled variant is the ceiling
#endif
  if (m >= n_act) return;      // uniform across the work-group
#ifdef BATCH_ROWS
  const uint pm = rows[m * BROW_WORDS + BROW_POS];    // the row's position (BATCH_ROWS, above)
  const uint kvb = rows[m * BROW_WORDS + BROW_KVB];   // its slot's first KV position
#endif

  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  __global const float* restrict nw = fa_small + (is_q ? QNORM_OFF : KNORM_OFF);
  const size_t base = is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD;

  // attn_prep's chain, unchanged: the linear's bf16, the norm (tree, 1/sqrt, (1+w)), RoPE.
  const float xf = bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + i)));
  red[i] = xf * xf;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KV8_WG / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = red[0] / (float)HD;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);
  nrm[i] = bf16f(rne_bf16(xf * rstd * nw[i]));
  barrier(CLK_LOCAL_MEM_FENCE);

#ifndef BATCH_ROWS
  __global const float* restrict cs = rope + (size_t)(pos + m) * 2 * ROT_HALF;
#else
  __global const float* restrict cs = rope + (size_t)pm * 2 * ROT_HALF;
#endif
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

  if (is_q) {                  // uniform: the barriers inside are reached by all or none
    const float y = kv8_rotate(outv, i, xch);
#if KV8_I8S
    kv8_store_q(bf16f(rne_bf16(y)), i, red, (__global char*)attn_q + ((size_t)m * Q_HEADS + h) * HD,
                q_scale + (size_t)m * Q_HEADS + h);
#elif PF
    ((__global ushort*)attn_q)[((size_t)m * Q_HEADS + h) * HD + i] = rne_bf16(y);
#else
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = y;
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + HD + i)));
#endif
  } else {
#if defined(KV_PAGED)
#ifndef BATCH_ROWS
    const uint vp = pos + m;
#else
    const uint vp = kvb + pm;
#endif
    const size_t row = ((size_t)pages[vp / KV_PAGED] * KV_PAGED + vp % KV_PAGED) * KV_HEADS + h;
#elif !defined(BATCH_ROWS)
    const size_t row = (size_t)(pos + m) * KV_HEADS + h;
#else
    const size_t row = (size_t)(kvb + pm) * KV_HEADS + h;
#endif
    const float yk = kv8_rotate(bf16f(rne_bf16(outv)), i, xch);
    kv8_store(yk, i, red, kv_k + row * HD, k_scale + row);
    const float vb = bf16f(rne_bf16(qkv_sum(qkv_partials, m, V_OFF + (size_t)h * HD + i)));
    const float yv = kv8_rotate(vb, i, xch);
    kv8_store(yv, i, red, kv_v + row * HD, v_scale + row);
#if KV8_OWN
    const size_t own_row = ((size_t)(pos + m) * KV_HEADS + h) * HD + i;   // the scratch's sequence position
    kr[own_row] = rne_bf16(yk);
    vr[own_row] = rne_bf16(yv);
#endif
  }
}
#endif  // KV8_PREP

#if KV8_DEC
// ---------------------------------------------------------------------------
// attn_decode_v2_kv8 / attn_reduce_v2_kv8 - attn_v2.cl's pair (its header is the
// contract: per-row stride ppw(L), grid (4, TGT), partials [24][TGT][M][258], verify rows
// bitwise the M = 1 steps), with the loads replaced and the reduce's un-rotation added.
//
//   attn_decode_v2_kv8(ctrl, attn_q, kv_k, k_scale, kv_v, v_scale, attn_part)
//   attn_reduce_v2_kv8(ctrl, attn_part, attn_gate, attn_out)
//
// K: sub-group s owns position p; lane l's 16 elements are dims l + 16 t (two uchar8
//    sub-group block reads, attn_v2.cl's order), dequantised as float(q8) * f16f(scale[p]).
// V: packed 16-bit 2D reads; sub-group pairs share a block and take its low / high bytes, so
//    a work-item's dim is v8_dim(sgid, lane), a fixed permutation of the work-group's 256
//    (3h; the 3f reader shuffled each lane's byte in). One scale gather per lane supplies
//    the wave through subgroup broadcasts; rows past `last` remain zero.
// The online softmax and its orders are attn_v2.cl's, unchanged. 3h: each (q head, row)
// pair's max, weights and sum are computed by one owner sub-group per wave and passed
// through SLM (one more barrier a wave), where every sub-group used to repeat them; each
// value is the same operation on the same operands, so the partials are bitwise the 3f
// reader's (tests/kernels/kv8_decode_bitwise_test.cc). Also 3h: at M = 1 the score dot's lane
// partials reach the owner through SLM (DOT_SLM), x s dequantises in one fma (kv8_deq), and
// the K / V rows and scales PFD waves ahead are prefetched into the cache. At 150k this
// reader takes ~0.53 of bf16's time where the 3f one took ~1.27 (probe_kv8_decode, 3h record).
// ---------------------------------------------------------------------------
#ifndef M
#define M 1
#endif
#ifndef CTRL_POS
#error "kv8 dec: CTRL_POS must be defined"
#endif
#ifndef CTRL_NACT
#error "kv8 dec: CTRL_NACT must be defined"
#endif
#ifndef TGT
#error "kv8 dec: TGT must be defined (runtime::DecodeScratch::kAttnV2Blocks)"
#endif
#ifndef BATCH_ROWS
#if M < 1 || M > 8
#error "kv8 dec: M is 1..8"
#endif
#elif M < 1 || M > 8
#error "kv8 dec BATCH_ROWS: M is 1..8"
#endif
#define WAVE_P 16
#define SG 16
#define PER_LANE 16
#define PART 258
#define NPR (M * GQA)
#define NOWN ((NPR + SG - 1) / SG)   /* the (q head, row) pairs one sub-group owns (3h) */
// 3h: the score dot's lane partials go through SLM and the owner sums them, the shuffle
// tree's adds in its order, in its own lanes (M = 1: measured slower at M = 4).
#define DOT_SLM (NPR <= 6)
#define PFD 2u   /* 3h: waves ahead the cache prefetch runs (2 measured best of 2, 3, 4, 6) */
#define SSCALE 0.0625f       /* attn_v2's shipped EXP2 = 0 form: exp and 1/16 */
#define EXPF(x) exp(x)

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

// Writer scales are finite fp16; native widening preserves every stored bit.
inline float kv8_decode_scale(__global const ushort* p, size_t index) {
  return vload_half(index, (__global const half*)p);
}

// 3h: an int8 times its scale, x s, in one fma: as_float(0x4B400000 + x) is 12582912 + x exactly
// (the float's unit in [2^23, 2^24)) and ns = -12582912 s is exact (12582912 = 3 x 2^22), so the
// fma's single rounding is of x s itself, which is exact (8 x 11 bits). Only a zero product's sign
// can differ from (float)x * s (+0 here); every reader adds it to a sum that is +0 or nonzero.
inline float kv8_deq(int x, float s, float ns) { return fma(as_float(0x4B400000 + x), s, ns); }

// K stays int8 in registers with its row's scale beside it (16 bytes + one float per lane,
// where 16 dequantised floats would be 16 GRF); the score dot dequantises at use, and since
// float(q8) * scale is exact the operand is the same value either way.
inline void load_k8(__global const char* restrict kv_k, __global const ushort* restrict k_scale,
                    uint p, uint j, bool ok, char* kq, float* ksc) {
  if (ok) {   // uniform within the sub-group
    const size_t r = (size_t)p * KV_HEADS + j;
    __global const uchar* row = (__global const uchar*)(kv_k + r * HD);
    *ksc = kv8_decode_scale(k_scale, r);
    const uchar8 a = intel_sub_group_block_read_uc8(row);
    const uchar8 b = intel_sub_group_block_read_uc8(row + 128);
    kq[0] = as_char(a.s0);  kq[1] = as_char(a.s1);  kq[2] = as_char(a.s2);  kq[3] = as_char(a.s3);
    kq[4] = as_char(a.s4);  kq[5] = as_char(a.s5);  kq[6] = as_char(a.s6);  kq[7] = as_char(a.s7);
    kq[8] = as_char(b.s0);  kq[9] = as_char(b.s1);  kq[10] = as_char(b.s2); kq[11] = as_char(b.s3);
    kq[12] = as_char(b.s4); kq[13] = as_char(b.s5); kq[14] = as_char(b.s6); kq[15] = as_char(b.s7);
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kq[t] = 0;
    *ksc = 0.0f;
  }
}

// The wave's 16 V values of dim v8_dim(sgid, lane): sub-groups 2h and 2h + 1 read the same
// packed 16-bit block (lane l: the byte pair 32 h + 2 l, + 1) and take its low / high byte,
// so no lane needs another's pair. One scale gather per lane supplies the wave through
// subgroup broadcasts; rows past `last` remain zero.
inline uint v8_dim(uint sgid, uint lane) { return 32u * (sgid >> 1) + 2u * lane + (sgid & 1u); }

inline void load_v8(__global const char* restrict kv_v, __global const ushort* restrict v_scale,
                    uint p0, uint j, uint sgid, uint lane, uint last, float* vreg) {
  ushort packed[WAVE_P];
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD),
      HD, last + 1, KV_HEADS * HD, (int2)((int)((sgid >> 1) * 16u), (int)p0), packed);
  const uint p = p0 + lane;
  const float scale = p <= last ? kv8_decode_scale(v_scale, (size_t)p * KV_HEADS + j) : 0.0f;
  const float nscale = -12582912.0f * scale;
  // A row past `last` reads 0 (the 2D read's bound) with scale 0, so its value is +0.0
  // without a select, as the 3f reader's 0.0f.
  if (sgid & 1u) {
#pragma unroll
    for (uint s = 0; s < WAVE_P; ++s)
      vreg[s] = kv8_deq((int)as_short(packed[s]) >> 8, intel_sub_group_shuffle(scale, s),
                        intel_sub_group_shuffle(nscale, s));
  } else {
#pragma unroll
    for (uint s = 0; s < WAVE_P; ++s)
      vreg[s] = kv8_deq((int)as_char((uchar)packed[s]), intel_sub_group_shuffle(scale, s),
                        intel_sub_group_shuffle(nscale, s));
  }
}

__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_q,
#ifndef BATCH_ROWS
                                 __global const char* restrict kv_k,
                                 __global const ushort* restrict k_scale,
                                 __global const char* restrict kv_v,
                                 __global const ushort* restrict v_scale,
#else
                                 __global const char* restrict kv_k_all,
                                 __global const ushort* restrict k_scale_all,
                                 __global const char* restrict kv_v_all,
                                 __global const ushort* restrict v_scale_all,
#endif
                                 __global float* restrict attn_part
#ifdef BATCH_ROWS
                                 , __global const uint* restrict rows
#endif
#ifdef KV_PAGED
                                 , __global const uint* restrict pages
#endif
                                 ) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;

#ifndef BATCH_ROWS
  const uint pos = ctrl[CTRL_POS];
#endif
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

#ifndef BATCH_ROWS
  const uint s0 = v2_ppw(pos + 1);
  const uint s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
  if (!live0 && !live1) return;
#ifdef KV_PAGED
  // The first group's first page, read before q is staged so its latency hides behind that (the walk's first loads
  // need it); a block past the sequence reads no entry.
  const uint pg0v = blk * s0 / KV_PAGED;
  const uint pg0p = live0 ? pages[pg0v] : 0u;
#endif
#else
  uint rp[M], rs[M], rb[M];   // attn_v2.cl's BATCH_ROWS build, line for line
  bool live = false;
  for (uint m = 0; m < M; ++m) {
    rp[m] = m < n_act ? rows[m * BROW_WORDS + BROW_POS] : 0u;
    rs[m] = m < n_act ? rows[m * BROW_WORDS + BROW_SLOT] : 0u;
    rb[m] = m < n_act ? rows[m * BROW_WORDS + BROW_KVB] : 0u;
    if (m < n_act && blk * v2_ppw(rp[m] + 1) <= rp[m]) live = true;
  }
  if (!live) return;
#endif

  __local float qpack[NPR * HD];
#if DOT_SLM
  __local float dot_x[NPR][WAVE_P * (SG + 1)];   // [pr][position * 17 + lane]: lane l's partial
#else
  __local float sc_x[2][NPR][WAVE_P];
#endif
  __local float wt_x[2][NPR][WAVE_P];   // 3h: the owner's weights (lane = position) and rescale
  __local float rs_x[2][NPR];
  for (uint i = lid; i < NPR * HD; i += KV8_WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 3h: (q head, row) pair pr's running max and sum live in ONE sub-group, its owner
  // (pr % 16 == sgid), which computes the wave's weights once for the work-group; acc
  // (lane = dim v8_dim) is every sub-group's. The values and their orders are unchanged.
  float mx[NOWN], sm[NOWN], acc[NPR];
  for (uint k = 0; k < NOWN; ++k) { mx[k] = -INFINITY; sm[k] = 0.0f; }
  for (uint q = 0; q < NPR; ++q) acc[q] = 0.0f;
  const uint dim = v8_dim(sgid, lane);
  uint wc = 0;

#ifndef BATCH_ROWS
  for (uint g = 0; g < 2; ++g) {
    const uint mlo = g ? msplit : 0u, mhi = g ? n_act : msplit;
    if (mlo >= mhi) continue;
    const uint s = g ? s1 : s0;
    const uint end = pos + mhi;
#else
  for (uint mlo = 0, mhi = 0; mlo < n_act; mlo = mhi) {
    const uint s = v2_ppw(rp[mlo] + 1);
    mhi = mlo + 1;
    while (mhi < n_act && rs[mhi] == rs[mlo] && v2_ppw(rp[mhi] + 1) == s) ++mhi;
    const uint end = rp[mhi - 1] + 1;
#ifndef KV_PAGED
    const size_t r0 = (size_t)rb[mlo] * KV_HEADS;   // the slot's first (position, kv head) row
    __global const char* restrict kv_k = kv_k_all + r0 * HD;
    __global const ushort* restrict k_scale = k_scale_all + r0;
    __global const char* restrict kv_v = kv_v_all + r0 * HD;
    __global const ushort* restrict v_scale = v_scale_all + r0;
#else
    __global const char* restrict kv_k = kv_k_all;
    __global const ushort* restrict k_scale = k_scale_all;
    __global const char* restrict kv_v = kv_v_all;
    __global const ushort* restrict v_scale = v_scale_all;
    const uint pgo = rb[mlo] / KV_PAGED;   // the slot's first page in the table
#endif
#endif
    const uint bstart = blk * s;
    if (bstart >= end) continue;
    const uint bend = min(bstart + s, end);
    const uint last = end - 1;
    const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

    char kq[PER_LANE];
    float ksc, vreg[WAVE_P];
#ifndef KV_PAGED
    load_k8(kv_k, k_scale, bstart + sgid, j, bstart + sgid <= last, kq, &ksc);
    load_v8(kv_v, v_scale, bstart, j, sgid, lane, last, vreg);
#else
#ifndef BATCH_ROWS
    const uint pgo = 0;
#endif
    // Paged (attn_v2.cl's KV_PAGED walk): the loads take cache rows - a wave's positions plus its page's delta (the
    // page's first cache row less its first position; uint arithmetic, wrapping) - through the unmoved bases, lastr the
    // page's last readable cache row. The prefetch keeps the page of the wave PFD ahead (pdelta, plastr), looked up as
    // it enters one, so a page's first waves are prefetched like any other.
    uint pgv = bstart / KV_PAGED;
#ifndef BATCH_ROWS
    uint delta = (live0 && pgv == pg0v ? pg0p : pages[pgv]) * KV_PAGED - pgv * KV_PAGED;
#else
    uint delta = pages[pgo + pgv] * KV_PAGED - pgv * KV_PAGED;
#endif
    uint lastr = min(last, pgv * KV_PAGED + KV_PAGED - 1u) + delta;
    uint ppv = pgv, pdelta = delta, plastr = lastr;
    load_k8(kv_k, k_scale, bstart + sgid + delta, j, bstart + sgid <= last, kq, &ksc);
    load_v8(kv_v, v_scale, bstart + delta, j, sgid, lane, lastr, vreg);
#endif
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
      // 3h: the wave PFD ahead into the cache - 64-byte quarters of its K rows (sub-groups
      // 0..3) and V rows (4..7), its K and V scales (8, 9). Hints only: no value depends on them.
#ifndef KV_PAGED
      if (w + PFD < nwaves && sgid < 8u)
        intel_sub_group_2d_block_prefetch_16b_16r16x2c(
            (__global void*)((sgid < 4u ? (__global const char*)kv_k : kv_v) + (size_t)j * HD), HD,
            last + 1, KV_HEADS * HD, (int2)((int)(32u * (sgid & 3u)), (int)(p0 + PFD * WAVE_P)));
      if (w + PFD < nwaves && (sgid == 8u || sgid == 9u))
        prefetch((sgid == 8u ? k_scale : v_scale) + (size_t)(p0 + PFD * WAVE_P) * KV_HEADS, WAVE_P * KV_HEADS);
#else
      // Paged: the same hints at the cache rows of the wave PFD ahead (its page looked up as the prefetch enters it).
      if (w + PFD < nwaves) {   // uniform
        const uint pf0 = p0 + PFD * WAVE_P;
        if (pf0 / KV_PAGED != ppv) {
          ppv = pf0 / KV_PAGED;
          pdelta = pages[pgo + ppv] * KV_PAGED - ppv * KV_PAGED;
          plastr = min(last, ppv * KV_PAGED + KV_PAGED - 1u) + pdelta;
        }
        if (sgid < 8u)
          intel_sub_group_2d_block_prefetch_16b_16r16x2c(
              (__global void*)((sgid < 4u ? (__global const char*)kv_k : kv_v) + (size_t)j * HD), HD,
              plastr + 1, KV_HEADS * HD, (int2)((int)(32u * (sgid & 3u)), (int)(pf0 + pdelta)));
        if (sgid == 8u || sgid == 9u)
          prefetch((sgid == 8u ? k_scale : v_scale) + (size_t)(pf0 + pdelta) * KV_HEADS, WAVE_P * KV_HEADS);
      }
#endif
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;
        float qv[PER_LANE];
        const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD]);
        const uint8 qb = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD + 128]);
        qv[0] = as_float(qa.s0); qv[1] = as_float(qa.s1); qv[2] = as_float(qa.s2);
        qv[3] = as_float(qa.s3); qv[4] = as_float(qa.s4); qv[5] = as_float(qa.s5);
        qv[6] = as_float(qa.s6); qv[7] = as_float(qa.s7);
        qv[8] = as_float(qb.s0); qv[9] = as_float(qb.s1); qv[10] = as_float(qb.s2);
        qv[11] = as_float(qb.s3); qv[12] = as_float(qb.s4); qv[13] = as_float(qb.s5);
        qv[14] = as_float(qb.s6); qv[15] = as_float(qb.s7);
        float a = 0.0f;
        const float nks = -12582912.0f * ksc;
        for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], kv8_deq(kq[t], ksc, nks), a);   // exact operand
#if DOT_SLM
        dot_x[pr][sgid * (SG + 1) + lane] = a;
#else
        for (uint stride = SG / 2; stride > 0; stride >>= 1)
          a += intel_sub_group_shuffle_down(a, a, stride);
        if (lane == 0) sc_x[buf][pr][sgid] = a;
#endif
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // The owners: the wave's max, weights and sum, once per work-group.
#pragma unroll
      for (uint k = 0; k < NOWN; ++k) {
        const uint pr = sgid + SG * k;
        const uint m = pr / GQA;
        if (pr >= NPR || m < mlo || m >= mhi) continue;
#ifndef BATCH_ROWS
        const uint bound = pos + m;
#else
        const uint bound = rp[m];
#endif
#if DOT_SLM
        // The shuffle tree's sums (stride 8, 4, 2, 1 into lane 0) in this lane, for position lane.
        float x[SG];
        for (uint l = 0; l < SG; ++l) x[l] = dot_x[pr][lane * (SG + 1) + l];
        for (uint stride = SG / 2; stride > 0; stride >>= 1)
          for (uint l = 0; l < stride; ++l) x[l] += x[l + stride];
        const float mine = p0 + lane <= bound ? x[0] * SSCALE : -INFINITY;
#else
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
#endif
        const float nmx = fmax(mx[k], sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = EXPF(mx[k] - nmx);
          wl = EXPF(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) ssum += intel_sub_group_shuffle(wl, s2);
        if (nmx > -INFINITY) {
          sm[k] = fma(sm[k], resc, ssum);
          mx[k] = nmx;
        }
        wt_x[buf][pr][lane] = wl;
        if (lane == 0) rs_x[buf][pr] = resc;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;
        const float wl = wt_x[buf][pr][lane];
        float tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) tsum = fma(intel_sub_group_shuffle(wl, s2), vreg[s2], tsum);
        acc[pr] = fma(acc[pr], rs_x[buf][pr], tsum);
      }
      if (w + 1 < nwaves) {
#ifndef KV_PAGED
        load_k8(kv_k, k_scale, p0 + WAVE_P + sgid, j, p0 + WAVE_P + sgid <= last, kq, &ksc);
        load_v8(kv_v, v_scale, p0 + WAVE_P, j, sgid, lane, last, vreg);
#else
        const uint q0 = p0 + WAVE_P;
        if (q0 / KV_PAGED != pgv) {   // uniform: the next wave starts a page (the prefetch's, unless it stopped short)
          pgv = q0 / KV_PAGED;
          if (pgv == ppv) {
            delta = pdelta;
            lastr = plastr;
          } else {
            delta = pages[pgo + pgv] * KV_PAGED - pgv * KV_PAGED;
            lastr = min(last, pgv * KV_PAGED + KV_PAGED - 1u) + delta;
          }
        }
        load_k8(kv_k, k_scale, q0 + sgid + delta, j, q0 + sgid <= last, kq, &ksc);
        load_v8(kv_v, v_scale, q0 + delta, j, sgid, lane, lastr, vreg);
#endif
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out =
          attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + dim] = acc[pr];
      if (pr % SG == sgid && lane == 0) { out[0] = mx[pr / SG]; out[1] = sm[pr / SG]; }
    }
  }
}

// attn_reduce_v2's merge, then o = acc / sm un-rotated across the work-group (work-item
// d = dim d), then attn.cl's gated chain on the un-rotated value.
__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void attn_reduce_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_part,
                                 __global const float* restrict attn_gate,
                                 __global ushort* restrict attn_out
#ifdef BATCH_ROWS
                                 , __global const uint* restrict rows
#endif
                                 ) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  __local float xch[HD];
#ifndef BATCH_ROWS
  const uint pos = ctrl[CTRL_POS];
#endif
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
#ifndef BATCH_ROWS
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
#else
  const uint pm = rows[m * BROW_WORDS + BROW_POS];
  uint nb = pm / v2_ppw(pm + 1) + 1;
#endif
  if (nb > TGT) nb = TGT;
  for (uint b = d; b < nb; b += KV8_WG) {
    __global const float* restrict p = attn_part + (((size_t)h * TGT + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = EXPF(mx - nmx);
    const float bs = EXPF(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const float o = kv8_unrotate(acc / sm, d, xch);
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * sigmoid_f32(g));
}
#endif  // KV8_DEC

#if KV8_GATE
// ---------------------------------------------------------------------------
// pf_attn_gate_kv8 - grid (24 q-heads, C), work-group 256: pf_attn.cl's pf_attn_gate on
// the un-rotated row. o fp32 [24][rows][256] (pf_o, head stride `stride_h`), rotated.
//   x   = unrotate(o[h][m][.])[d]
//   out[m][h 256 + d] = rne_bf16(f32(rne_bf16(x)) sigmoid(f32(rne_bf16(gate))))
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void pf_attn_gate_kv8(__global const float* restrict o,
                               __global const float* restrict qkv_partials,
                               __global ushort* restrict out, uint stride_h) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float xch[HD];
  const float x = kv8_unrotate(o[(size_t)h * stride_h + (size_t)m * HD + d], d, xch);
  const float g =
      bf16f(rne_bf16(qkv_partials[(size_t)m * QKV_N + (size_t)h * 2 * HD + HD + d]));
  out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(x)) * sigmoid_f32(g));
}
#endif  // KV8_GATE

#if KV8_FLASH
// ---------------------------------------------------------------------------
// pf_flash_attn_kv8 - pf_flash_attn.cl (its header is the contract: 8 query rows per
// sub-group, S in DPAS A layout, online softmax, O / l into pf_o), reading the int8 cache.
//
//   pf_flash_attn_kv8(Q, Kc, Ks, Vc, Vs, O, pos, C, rows)
//     Q   bf16 [C][24][256]          rotated q (attn_prep_kv8 PF = 1)
//     Kc  int8 [depth][4][256], Ks fp16 bits [depth][4]   one layer's K rows and scales
//     Vc  int8 [depth][4][256], Vs fp16 bits [depth][4]
//     O   fp32 [24][rows][256]       still in the rotated basis: pf_attn_gate_kv8 un-rotates
//
// K^T operand: lane l = key t0 + 16 b + l, its 16 dims 16 kk .. 16 kk + 15 as 8 bf16 pairs
//   (low half = even dim) - the layout the transposed 32-bit 2D read gives the bf16 kernel -
//   from a transposed 2D read of two dim atoms; an int8 is exactly a bf16. The key's scale
//   multiplies the score: s = (dot q8 * sk) * SCORE_SCALE.
// V operand: lane l = dim, 8 pairs of consecutive keys (VNNI), int8 as bf16 exactly; the
//   key's scale goes into P: pa = rne_bf16(p * sv) (lane = key), the row sum stays sum p.
// KV8_RAW reads those same unscaled integer values from a bf16 scratch, widened once
// by kv8_widen. Scale multiplications and all rounding points stay in this kernel.
// Keys at or past `depth` load as 0 and score -INF; nothing past the cache's rows is read.
// ---------------------------------------------------------------------------
#ifndef KV8_RAW
#define KV8_RAW 0
#endif
#if KV8_RAW
#define KV8_VALUE ushort
#else
#define KV8_VALUE char
#endif
#ifndef KT
#error "KT (KV positions per tile, 32 or 64) must be defined"
#endif
#if KT % 32
#error "KT must be a multiple of 32 (the PV loop takes key atoms in pairs)"
#endif
#if RPW % 8
#error "RPW must be a multiple of 8"
#endif
#define SG 16
#define NDA (HD / 16)
#define NKA (KT / 16)
#define SGS (HPW * RPW / 8)
#define ATTN_SCALE (1.0f / 16.0f)
#ifndef EXP2
#error "EXP2 must be defined"
#endif
#if EXP2
#define SCORE_SCALE (ATTN_SCALE * M_LOG2E_F)
#define EXPF(x) exp2(x)
#else
#define SCORE_SCALE ATTN_SCALE
#define EXPF(x) exp(x)
#endif

// An int8 as its exact bf16 bits.
inline uint kv8_bf(char c) { return as_uint((float)c) >> 16; }

inline uint2 kv8_bf4(uint w) {
  const char4 c = as_char4(w);
  return (uint2)(kv8_bf(c.s0) | (kv8_bf(c.s1) << 16),
                 kv8_bf(c.s2) | (kv8_bf(c.s3) << 16));
}

inline float kv8_flash_scale(__global const ushort* p, size_t i) {
#if KV8_RAW
  // The writer saturates scales at 65504. Native widening is exact for every finite
  // fp16 code, including subnormals and signed zero (kv8_scale_test).
  return vload_half(i, (__global const half*)p);
#else
  return kv8_f16f(p[i]);
#endif
}

#if KV8_RAW
// Every input row has 4 * HD values, a multiple of the 16-value vector below.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kv8_widen(__global const char* restrict K,
                       __global const char* restrict V,
                       __global ushort* restrict Kr,
                       __global ushort* restrict Vr, uint count
#ifdef KV_PAGED
                       , __global const uint* restrict pages
#endif
                       ) {
  const size_t i = get_global_id(0) * 16u;
  if (i >= count) return;
#ifndef KV_PAGED
  const char16 k = vload16(0, K + i), v = vload16(0, V + i);
#else
  // Position i / row's cache row (16 values never cross a row of KV8F_KV x HD).
  const uint p = (uint)(i / (KV8F_KVU * HD));
  const size_t src = ((size_t)pages[p / KV_PAGED] * KV_PAGED + p % KV_PAGED) * KV8F_KVU * HD + i % (KV8F_KVU * HD);
  const char16 k = vload16(0, K + src), v = vload16(0, V + src);
#endif
#if KV8_I8S
  vstore16(k, 0, (__global char*)Kr + i);   // K stays int8: the scores' DPAS operand
#else
  vstore16(convert_ushort16(as_uint16(convert_float16(k)) >> 16), 0, Kr + i);
#endif
  vstore16(convert_ushort16(as_uint16(convert_float16(v)) >> 16), 0, Vr + i);
}
#endif

#ifndef KV8_PAIR
#define KV8_PAIR 0
#endif
#ifndef KV8_SPLIT
#define KV8_SPLIT 0
#endif
#if KV8_OWN && !KV8_SPLIT
#error "kv8 KV8_OWN: the split forms only (a whole chunk on the int8 pass scores by int8, KV8_I8S)"
#endif
// The reader's KV8_OWN arguments, after its own: `own`, and for the in-place split the widened scratch.
#if KV8_OWN && KV8_RAW
#define KV8_OWN_ARGS , uint own
#elif KV8_OWN
#define KV8_OWN_ARGS , uint own, __global const ushort* restrict Kr, __global const ushort* restrict Vr
#else
#define KV8_OWN_ARGS
#endif
#if KV8_PAIR
// KV8_PAIR=1 (long-context prefill over the widened rows): pf_flash_attn.cl's PAIR form (its comment is the contract:
// two sub-groups share 16 query rows of one head, scores by key-atom pair, row maxima through SLM, sub-group 0 hands
// p0 + p1 to sub-group 1, which finishes l in the order ((p0 + p1) + p2) + p3, both publish P, each owns half of O's
// dims over all four key atoms in order). The scales stay with the key atoms that own them: a sub-group multiplies
// its own keys' scores by their K scale and its own P atoms by their V scale before publishing them; the row sum
// stays sum p. Bitwise the RPW 8 raw form above. Grid (ceil(C / 16), kv-heads, GQA / HPW).
// KV8_I8S (above): Q is int8 with QS [C][q-heads] fp32 beside it, Kc kv8_widen's int8 rows; each K tile is 8 int8
// DPAS (32 dims) a row block and key atom into an int32 sum, then times the row's q scale; the rest is the form above.
// With KV_PAGED the rows are the widened sequence's own and only the scales are read by page (PERF-64: the bf16-score
// pair too, Ornith's slots; bitwise the paged RPW 8 raw form as the unpaged pair is the unpaged one).
#if !KV8_RAW || KV8_SPLIT || KT != 64 || RPW != 16 || QREG
#error "pf_flash_attn_kv8 KV8_PAIR: KV8_RAW, KT 64, RPW 16, no KV8_SPLIT / QREG"
#endif
__attribute__((reqd_work_group_size(SG * 2 * HPW, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#if KV8_I8S
__kernel void pf_flash_attn_kv8(__global const char* restrict Q,
                                __global const float* restrict QS,
                                __global const char* restrict Kc,
#else
__kernel void pf_flash_attn_kv8(__global const ushort* restrict Q,
                                __global const KV8_VALUE* restrict Kc,
#endif
                                __global const ushort* restrict Ks,
                                __global const KV8_VALUE* restrict Vc,
                                __global const ushort* restrict Vs,
                                __global float* restrict O, uint pos, uint C, uint rows
#ifdef KV_PAGED
                                , __global const uint* restrict pages
#endif
                                ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint k = s & 1u, hp = s >> 1;                              // key half / dim half; the pair
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * KV8F_GQAU + get_group_id(2) * HPW + hp;       // q head
  const uint r0 = get_group_id(0) * 16u;                           // first of 16 rows
  if (r0 >= C) return;                                             // uniform over the work-group
  const uint depth = pos + C;
#if KV8_I8S
  const int q_w = KV8F_Q * HD, q_h = (int)C, q_p = KV8F_Q * HD;   // int8 q
  const int k_w = KV8F_KV * HD;                                    // int8 K rows (width = pitch)
  float8 qsr[2];   // the q scale of row r0 + 8 rb + r in component r; 0 past C (its scores are never written)
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    float t[8];
#pragma unroll
    for (uint r = 0; r < 8u; ++r) {
      const uint row = r0 + 8u * rb + r;
      t[r] = row < C ? QS[(size_t)row * KV8F_Q + h] : 0.0f;
    }
    qsr[rb] = vload8(0, t);
  }
#else
  const int q_w = KV8F_Q * HD * 2, q_h = (int)C, q_p = KV8F_Q * HD * 2;
#endif
  const int kv_w = KV8F_KV * HD * 2, kv_h = (int)depth, kv_p = KV8F_KV * HD * 2;
  __local float smax[HPW * 32];      // [pair][k][16 rows]: the tile's row maxima of key half k
  __local float sa[HPW * 256];       // [pair][rb][r * 16 + lane]: sub-group 0's p0 + p1
  __local ushort sp[HPW * 1024];     // [pair][rb * 4 + b][r * 16 + lane]: P times the key's V scale, bf16
  __local float sl[HPW * 16];        // [pair][16 rows]: l after the walk

  float8 o[2][NDA / 2];
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] = (float8)(0.0f);
  float8 m[2] = {(float8)(-INFINITY), (float8)(-INFINITY)}, lsum[2] = {(float8)(0.0f), (float8)(0.0f)};
  const uint last = min(depth, pos + r0 + 16u);
  for (uint t0 = 0; t0 < last; t0 += KT) {
    float8 sacc[2][2];
#if KV8_I8S
    int8 iacc[2][2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) iacc[rb][0] = iacc[rb][1] = (int8)(0);
#pragma unroll
    for (uint kk = 0; kk < HD / 32u; kk += 2) {   // int8 atoms of 32 dims, two a read
      short8 qa[2][2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        ushort t[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                  (int2)((int)((h * HD + 32u * kk) / 2u), (int)(r0 + 8u * rb)), t);
        qa[rb][0] = as_short8(vload8(0, t));
        qa[rb][1] = as_short8(vload8(0, t + 8));
      }
#pragma unroll
      for (uint kq = 0; kq < 2u; ++kq)
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          uint kb[8];   // lane = key, its 32 dims as 8 words of 4 (low byte = lowest dim)
          intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
              (__global void*)Kc, k_w, kv_h, k_w,
              (int2)((int)((j * HD + 32u * (kk + kq)) / 4u), (int)(t0 + 16u * (2u * k + bb))), kb);
#pragma unroll
          for (uint rb = 0; rb < 2u; ++rb)
            iacc[rb][bb] = intel_sub_group_i8_i8_matrix_mad_k32(qa[rb][kq], as_int8(vload8(0, kb)), iacc[rb][bb]);
        }
    }
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) sacc[rb][bb] = convert_float8(iacc[rb][bb]) * qsr[rb];
#else
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) sacc[rb][0] = sacc[rb][1] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2) {
      short8 qa[2][2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        ushort t[16];
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                  (int2)((int)(h * HD + 16u * kk), (int)(r0 + 8u * rb)), t);
        qa[rb][0] = as_short8(vload8(0, t));
        qa[rb][1] = as_short8(vload8(0, t + 8));
      }
#pragma unroll
      for (uint kq = 0; kq < 2u; ++kq)
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          uint kb[8];
          intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
              (__global void*)Kc, kv_w, kv_h, kv_p,
              (int2)((int)((j * HD + 16u * (kk + kq)) / 2u), (int)(t0 + 16u * (2u * k + bb))), kb);
#pragma unroll
          for (uint rb = 0; rb < 2u; ++rb)
            sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kq], as_int8(vload8(0, kb)), sacc[rb][bb]);
        }
    }
#endif
    // this lane's key's K and V scales, per key atom of this half (lane = key t0 + 16 (2k + bb) + l)
    float skl[2], svl[2];
#ifdef KV_PAGED
    // the tile's page (KT divides KV_PAGED): key's scale row is ppr + key - pgs
    const uint pgs = t0 / KV_PAGED * KV_PAGED;
    const size_t ppr = (size_t)pages[t0 / KV_PAGED] * KV_PAGED;
#endif
#pragma unroll
    for (uint bb = 0; bb < 2u; ++bb) {
      const uint key = t0 + 16u * (2u * k + bb) + l;
      const bool live = key < depth;
#ifndef KV_PAGED
      skl[bb] = live ? kv8_flash_scale(Ks, (size_t)key * KV8F_KVU + j) : 0.0f;
      svl[bb] = live ? kv8_flash_scale(Vs, (size_t)key * KV8F_KVU + j) : 0.0f;
#else
      const size_t krow = ppr + (key - pgs);
      skl[bb] = live ? kv8_flash_scale(Ks, krow * KV8F_KVU + j) : 0.0f;
      svl[bb] = live ? kv8_flash_scale(Vs, krow * KV8F_KVU + j) : 0.0f;
#endif
    }
    // scale by the key's K scale, mask, this half's row maxima (component r = row r0 + 8 rb + r)
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      const uint rr = r0 + 8u * rb;
      float8 tmax = (float8)(-INFINITY), red;
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint key = t0 + 16u * (2u * k + bb) + l;
        const float sk = skl[bb];
#define MASK(r) sacc[rb][bb].s##r = (key <= pos + rr + r && key < depth) ? (sacc[rb][bb].s##r * sk) * SCORE_SCALE : -INFINITY;
        MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
        tmax = fmax(tmax, sacc[rb][bb]);
      }
#define RED(r) red.s##r = sub_group_reduce_max(tmax.s##r);
      RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
      if (l == 0) vstore8(red, 0, smax + hp * 32u + k * 16u + 8u * rb);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 corr[2], mnew[2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      mnew[rb] = fmax(m[rb], fmax(vload8(0, smax + hp * 32u + 8u * rb), vload8(0, smax + hp * 32u + 16u + 8u * rb)));
      corr[rb] = EXPF(m[rb] - mnew[rb]);
      float8 part = (float8)(0.0f);
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        sacc[rb][bb] = EXPF(sacc[rb][bb] - mnew[rb]);   // p, kept for sub-group 1's sum
        part += sacc[rb][bb];
        const float8 ps = sacc[rb][bb] * svl[bb];
        short8 pa;
#define CV(r) pa.s##r = as_short(rne_bf16(ps.s##r));
        CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
        intel_sub_group_block_write_us8(sp + hp * 1024u + (rb * 4u + 2u * k + bb) * 128u, as_ushort8(pa));
      }
      if (k == 0) intel_sub_group_block_write8((__local uint*)(sa + hp * 256u + rb * 128u), as_uint8(part));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 1) {
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        float8 psum = as_float8(intel_sub_group_block_read8((__local const uint*)(sa + hp * 256u + rb * 128u)));
        psum += sacc[rb][0];
        psum += sacc[rb][1];
        float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
        SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
        lsum[rb] = lsum[rb] * corr[rb] + rsum;
      }
    }
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      m[rb] = mnew[rb];
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] *= corr[rb];
    }
    short8 pa[2][NKA];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
      for (uint b = 0; b < NKA; ++b)
        pa[rb][b] = as_short8(intel_sub_group_block_read_us8(sp + hp * 1024u + (rb * 4u + b) * 128u));
    // O += (P s) V over this sub-group's dim atoms 8k + dd, key atoms in order
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; dd += 2) {
        uint vb[32];
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, kv_w, kv_h, kv_p, (int2)((int)(j * HD + 16u * (8u * k + dd)), (int)(t0 + 16u * b)), vb);
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
#pragma unroll
            for (uint rb = 0; rb < 2u; ++rb)
              o[rb][dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                  pa[rb][b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[rb][dd + c]);
      }
  }
  if (k == 1 && l == 0) {
    vstore8(lsum[0], 0, sl + hp * 16u);
    vstore8(lsum[1], 0, sl + hp * 16u + 8u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    const uint rr = r0 + 8u * rb;
    if (rr >= C) break;                                            // a block wholly past C is not written
    const float8 inv = 1.0f / vload8(0, sl + hp * 16u + 8u * rb);
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      float8 w = o[rb][dd] * inv;
      intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                                 (int2)((int)(16u * (8u * k + dd)), (int)rr), (__private uint*)&w);
    }
  }
}
#else
// 3l (2): KV8_SPLIT=1 is pf_flash_attn.cl's SPLIT form (its header: spans of the key range,
// unnormalised O and m, l into the partials area; pf_flash_merge folds them). Unset, the token
// stream is unchanged.
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#if KV8_SPLIT
__kernel void pf_flash_attn_kv8(__global const ushort* restrict Q,
                                __global const KV8_VALUE* restrict Kc,
                                __global const ushort* restrict Ks,
                                __global const KV8_VALUE* restrict Vc,
                                __global const ushort* restrict Vs,
                                __global float* restrict O, uint pos, uint C, uint rows,
                                __global float* restrict ML, uint span KV8_OWN_ARGS
#ifdef KV_PAGED
                                , __global const uint* restrict pages
#endif
                                ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);
  const uint hgs = KV8F_GQAU / HPW;
  const uint sp = get_group_id(2) / hgs, nsp = get_num_groups(2) / hgs;
  const uint h = j * KV8F_GQAU + (get_group_id(2) % hgs) * HPW + s / (RPW / 8u);
#else
__kernel void pf_flash_attn_kv8(__global const ushort* restrict Q,
                                __global const KV8_VALUE* restrict Kc,
                                __global const ushort* restrict Ks,
                                __global const KV8_VALUE* restrict Vc,
                                __global const ushort* restrict Vs,
                                __global float* restrict O, uint pos, uint C, uint rows
#ifdef KV_PAGED
                                , __global const uint* restrict pages
#endif
                                ) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);
  const uint h = j * KV8F_GQAU + get_group_id(2) * HPW + s / (RPW / 8u);
#endif
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));
  if (r0 >= C) return;
  const uint depth = pos + C;
  const int q_w = KV8F_Q * HD * 2, q_h = (int)C, q_p = KV8F_Q * HD * 2;

  short8 qa[NDA];
#define LOAD_Q()                                                                      \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; kk += 2) {                            \
    ushort t[16];                                                                     \
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,       \
                                              (int2)((int)(h * HD + 16u * kk), (int)r0), t); \
    qa[kk] = as_short8(vload8(0, t));                                                 \
    qa[kk + 1] = as_short8(vload8(0, t + 8));                                         \
  }
#if QREG
  LOAD_Q();
#endif
  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);
#if KV8_SPLIT
  const uint kend = sp + 1u == nsp ? last : sp * span + span;
  for (uint t0 = sp * span; t0 < kend; t0 += KT) {
#else
  for (uint t0 = 0; t0 < last; t0 += KT) {
#endif
#if !QREG
    LOAD_Q();
#endif
#ifdef KV_PAGED
    // The tile's page (KT divides KV_PAGED and every tile start is a multiple of KT): its first cache row ppr, the
    // tile's keys key - pgs + ppr there. The raw form reads the widened, contiguous rows and only the scales by page.
    const uint pgs = t0 / KV_PAGED * KV_PAGED;
    const size_t ppr = (size_t)pages[t0 / KV_PAGED] * KV_PAGED;
#if !KV8_RAW
    __global const KV8_VALUE* const Kp = Kc + ppr * KV8F_KVU * HD;
    __global const KV8_VALUE* const Vp = Vc + ppr * KV8F_KVU * HD;
    const int ph = (int)min(depth - pgs, (uint)KV_PAGED);   // the page's rows the surfaces may read
#endif
#endif
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#if KV8_RAW
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        uint kb[8];
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, KV8F_KV * HD * 2, (int)depth, KV8F_KV * HD * 2,
            (int2)((int)((j * HD + 16u * kk) / 2u), (int)(t0 + 16u * b)), kb);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
      }
#else
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        uint k4[8], kb[16];
#ifndef KV_PAGED
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kc, KV8F_KV * HD, (int)depth, KV8F_KV * HD,
            (int2)((int)((j * HD + 16u * kk) / 4u), (int)(t0 + 16u * b)), k4);
#else
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (__global void*)Kp, KV8F_KV * HD, ph, KV8F_KV * HD,
            (int2)((int)((j * HD + 16u * kk) / 4u), (int)(t0 - pgs + 16u * b)), k4);
#endif
#pragma unroll
        for (uint z = 0; z < 8u; ++z) {
          const uint2 pair = kv8_bf4(k4[z]);
          kb[2u * z] = pair.s0;
          kb[2u * z + 1u] = pair.s1;
        }
#if KV8_OWN
        if (t0 + KT > own) {   // the tile reaches the chunk's own rows: those keys (lane = key) from the scratch
          uint rk[16];
#pragma unroll
          for (uint a = 0; a < 2u; ++a)
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (__global void*)Kr, KV8F_KV * HD * 2, (int)depth, KV8F_KV * HD * 2,
                (int2)((int)((j * HD + 16u * (kk + a)) / 2u), (int)(t0 + 16u * b)), rk + 8u * a);
          if (t0 + 16u * b + l >= own)
#pragma unroll
            for (uint z = 0; z < 16u; ++z) kb[z] = rk[z];
        }
#endif
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk + 1u], as_int8(vload8(1, kb)), sacc[b]);
      }
#endif
    // scale by the key's K scale, mask, running max (component r = row r0 + r, lane = key)
    float8 tmax = (float8)(-INFINITY);
    float svl[NKA];   // this lane's key's V scale, per key atom
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const uint key = t0 + 16u * b + l;
      const bool live = key < depth;
#ifndef KV_PAGED
      const float sk = live ? KV8_SC(key, kv8_flash_scale(Ks, (size_t)key * KV8F_KVU + j)) : 0.0f;
      svl[b] = live ? KV8_SC(key, kv8_flash_scale(Vs, (size_t)key * KV8F_KVU + j)) : 0.0f;
#else
      const size_t krow = ppr + (key - pgs);
      const float sk = live ? KV8_SC(key, kv8_flash_scale(Ks, krow * KV8F_KVU + j)) : 0.0f;
      svl[b] = live ? KV8_SC(key, kv8_flash_scale(Vs, krow * KV8F_KVU + j)) : 0.0f;
#endif
#define MASK(r) sacc[b].s##r = (key <= pos + r0 + r && live) ? (sacc[b].s##r * sk) * SCORE_SCALE : -INFINITY;
      MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
      tmax = fmax(tmax, sacc[b]);
    }
    float8 mnew;
#define RED(r) mnew.s##r = fmax(m.s##r, sub_group_reduce_max(tmax.s##r));
    RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
    const float8 corr = EXPF(m - mnew);
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - mnew);
      psum += p;
      const float8 ps = p * svl[b];
#define CV(r) pa[b].s##r = as_short(rne_bf16(ps.s##r));
      CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
    }
    float8 rsum;
#define SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
    SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    // O += (P s) V8: A = pa[b] (8 rows x 16 keys), B = V8 (16 keys x 16 dims, VNNI)
#pragma unroll
    for (uint b = 0; b < NKA; b += 2)
#pragma unroll
      for (uint dd = 0; dd < NDA; dd += 2) {
#if KV8_RAW
        uint vb[32];
        intel_sub_group_2d_block_read_transform_16b_32r16x2c(
            (__global void*)Vc, KV8F_KV * HD * 2, (int)depth, KV8F_KV * HD * 2,
            (int2)((int)(j * HD + 16u * dd), (int)(t0 + 16u * b)), vb);
#else
        uint vb[32];   // [c * 16 + jj]: c = dim half, jj = key pair of the 32 keys
#pragma unroll
        for (uint c = 0; c < 2u; ++c) {
          uint v4[8];
#ifndef KV_PAGED
          intel_sub_group_2d_block_read_transform_8b_32r16x1c(
              (__global void*)Vc, KV8F_KV * HD, (int)depth, KV8F_KV * HD,
              (int2)((int)(j * HD + 16u * (dd + c)), (int)(t0 + 16u * b)), v4);
#else
          intel_sub_group_2d_block_read_transform_8b_32r16x1c(
              (__global void*)Vp, KV8F_KV * HD, ph, KV8F_KV * HD,
              (int2)((int)(j * HD + 16u * (dd + c)), (int)(t0 - pgs + 16u * b)), v4);
#endif
#pragma unroll
          for (uint z = 0; z < 8u; ++z) {
            const uint2 pair = kv8_bf4(v4[z]);
            vb[c * 16u + 2u * z] = pair.s0;
            vb[c * 16u + 2u * z + 1u] = pair.s1;
          }
        }
#if KV8_OWN
        if (t0 + KT > own) {   // the own keys' halves from the scratch: vb[c * 16 + jj] holds keys 2 jj, 2 jj + 1
          uint rv[32];
          intel_sub_group_2d_block_read_transform_16b_32r16x2c(
              (__global void*)Vr, KV8F_KV * HD * 2, (int)depth, KV8F_KV * HD * 2,
              (int2)((int)(j * HD + 16u * dd), (int)(t0 + 16u * b)), rv);
#pragma unroll
          for (uint c = 0; c < 2u; ++c)
#pragma unroll
            for (uint jj = 0; jj < 16u; ++jj) {
              const uint key = t0 + 16u * b + 2u * jj;   // the low half's key (even)
              const uint msk = (key >= own ? 0xFFFFu : 0u) | (key + 1u >= own ? 0xFFFF0000u : 0u);
              vb[c * 16u + jj] = (rv[c * 16u + jj] & msk) | (vb[c * 16u + jj] & ~msk);
            }
        }
#endif
#endif
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
            o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);
      }
  }
#if KV8_SPLIT
  const uint c8 = (C + 7u) & ~7u;
  const size_t slot = (size_t)(sp * KV8F_Q + h) * c8;
  __global float* Op = O + slot * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd)
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Op, HD * 4, (int)c8, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&o[dd]);
  if (l == 0) {
    vstore8(m, 0, ML + slot + r0);
    vstore8(lsum, 0, ML + (size_t)nsp * KV8F_Q * c8 + slot + r0);
  }
#else
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
#endif
}
#endif  // KV8_PAIR
#endif  // KV8_FLASH
