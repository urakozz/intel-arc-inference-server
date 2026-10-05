// kv8.cl - spec 12b: the int8 KV cache, the operator's `rotkv` scheme (spec 12 §8), every
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
// The scheme (src/common/kv8.h is the host twin; edit the two together)
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

#if KV8_DEC || KV8_FLASH
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif
#if KV8_FLASH
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#endif

// The model's dimensions (model::Qwen35; FA heads as attn.cl bakes them).
#define Q_HEADS 24
#define KV_HEADS 4
#define GQA 6
#define HD 256
#define OUT_N 6144        /* 24 x 256 */
#define QKV_N 14336       /* q||gate (12288) || k (1024) || v (1024) */
#define K_OFF 12288
#define V_OFF 13312
#define KV8_WG 256        /* every KV8_PREP / KV8_DEC reduce / KV8_GATE work-group: one head */

// The probe's signs (src/common/kv8.h kSignWords).
__constant uint KV8_SIGNS[8] = {0xA197D809u, 0xA6850592u, 0xB205A73Eu, 0x20AFA769u,
                                0x45C2CF64u, 0xBEAEAACBu, 0x75F39ADDu, 0x2385A70Cu};

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + exp(-x)); }

// s[j] applied as the sign bit: exact, and the same bits as a multiply by -1.
inline float kv8_flip(float v, uint j) {
  return as_float(as_uint(v) ^ (((KV8_SIGNS[j >> 5] >> (j & 31u)) & 1u) << 31));
}

// binary16 <-> fp32 in integer arithmetic (kv8.h's f32_to_f16_rne / f16f, op for op): no
// cl_khr_fp16 needed, subnormals kept, >= 65504 saturates. A scale is >= 0 and finite.
inline ushort kv8_f16_rne(float f) {
  uint u = as_uint(f);
  const uint sign = (u >> 16) & 0x8000u;
  u &= 0x7FFFFFFFu;
  if (u >= 0x477FE000u) return (ushort)(sign | 0x7BFFu);
  if (u < 0x38800000u) return (ushort)(sign | (uint)rint(as_float(u) * 16777216.0f));
  const uint e = (u >> 23) - 112u;
  const uint m = u & 0x7FFFFFu;
  uint h = (e << 10) | (m >> 13);
  const uint rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
  return (ushort)(sign | h);
}
inline float kv8_f16f(ushort h) {
  const uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  const float v = e == 0u ? (float)m * 5.9604644775390625e-08f
                          : as_float(((e + 112u) << 23) | (m << 13));
  return (h & 0x8000u) ? -v : v;
}

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

__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void attn_prep_kv8(__global const uint* restrict ctrl,
                            __global const float* restrict qkv_partials,
                            __global const float* restrict fa_small,
                            __global const float* restrict rope,
                            __global float* restrict attn_q, __global float* restrict attn_gate,
                            __global char* restrict kv_k, __global char* restrict kv_v,
                            __global ushort* restrict k_scale, __global ushort* restrict v_scale) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float red[KV8_WG];   // the norm's sum of squares, then the amax tree
  __local float nrm[HD];       // the normalised head, fp32 - RoPE needs dim i+-32
  __local float xch[HD];       // the FWHT's exchange

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
#if !PF
  if (n_act > M) n_act = M;    // the compiled variant is the ceiling
#endif
  if (m >= n_act) return;      // uniform across the work-group

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

  if (is_q) {                  // uniform: the barriers inside are reached by all or none
    const float y = kv8_rotate(outv, i, xch);
#if PF
    ((__global ushort*)attn_q)[((size_t)m * Q_HEADS + h) * HD + i] = rne_bf16(y);
#else
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = y;
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_sum(qkv_partials, m, base + HD + i)));
#endif
  } else {
    const size_t row = (size_t)(pos + m) * KV_HEADS + h;
    const float yk = kv8_rotate(bf16f(rne_bf16(outv)), i, xch);
    kv8_store(yk, i, red, kv_k + row * HD, k_scale + row);
    const float vb = bf16f(rne_bf16(qkv_sum(qkv_partials, m, V_OFF + (size_t)h * HD + i)));
    const float yv = kv8_rotate(vb, i, xch);
    kv8_store(yv, i, red, kv_v + row * HD, v_scale + row);
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
// V: work-item d's 16 values of the wave, dim d of positions p0 .. p0 + 15, plain loads
//    (lanes read 16 consecutive bytes), zero past `last` (the 2D read's surface height in
//    the bf16 kernel), dequantised the same way.
// The online softmax and its orders are attn_v2.cl's, unchanged.
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
#if M < 1 || M > 4
#error "kv8 dec: M is 1..4"
#endif
#define WAVE_P 16
#define SG 16
#define PER_LANE 16
#define PART 258
#define NPR (M * GQA)
#define SSCALE 0.0625f       /* attn_v2's shipped EXP2 = 0 form: exp and 1/16 */
#define EXPF(x) exp(x)

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

inline void load_k8(__global const char* restrict kv_k, __global const ushort* restrict k_scale,
                    uint p, uint j, bool ok, float* kreg) {
  if (ok) {   // uniform within the sub-group
    const size_t r = (size_t)p * KV_HEADS + j;
    __global const uchar* row = (__global const uchar*)(kv_k + r * HD);
    const float sk = kv8_f16f(k_scale[r]);
    const uchar8 a = intel_sub_group_block_read_uc8(row);
    const uchar8 b = intel_sub_group_block_read_uc8(row + 128);
    kreg[0] = (float)as_char(a.s0) * sk;  kreg[1] = (float)as_char(a.s1) * sk;
    kreg[2] = (float)as_char(a.s2) * sk;  kreg[3] = (float)as_char(a.s3) * sk;
    kreg[4] = (float)as_char(a.s4) * sk;  kreg[5] = (float)as_char(a.s5) * sk;
    kreg[6] = (float)as_char(a.s6) * sk;  kreg[7] = (float)as_char(a.s7) * sk;
    kreg[8] = (float)as_char(b.s0) * sk;  kreg[9] = (float)as_char(b.s1) * sk;
    kreg[10] = (float)as_char(b.s2) * sk; kreg[11] = (float)as_char(b.s3) * sk;
    kreg[12] = (float)as_char(b.s4) * sk; kreg[13] = (float)as_char(b.s5) * sk;
    kreg[14] = (float)as_char(b.s6) * sk; kreg[15] = (float)as_char(b.s7) * sk;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = 0.0f;
  }
}

inline void load_v8(__global const char* restrict kv_v, __global const ushort* restrict v_scale,
                    uint p0, uint j, uint lid, uint last, float* vreg) {
  for (uint s = 0; s < WAVE_P; ++s) {
    const uint p = p0 + s;
    if (p <= last) {
      const size_t r = (size_t)p * KV_HEADS + j;
      vreg[s] = (float)kv_v[r * HD + lid] * kv8_f16f(v_scale[r]);
    } else {
      vreg[s] = 0.0f;   // never loaded: an unwritten slot cannot turn 0 x into a NaN
    }
  }
}

__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void attn_decode_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_q,
                                 __global const char* restrict kv_k,
                                 __global const ushort* restrict k_scale,
                                 __global const char* restrict kv_v,
                                 __global const ushort* restrict v_scale,
                                 __global float* restrict attn_part) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;

  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;

  const uint s0 = v2_ppw(pos + 1);
  const uint s1 = v2_ppw(pos + n_act);
  uint msplit = 1;
  while (msplit < n_act && v2_ppw(pos + msplit + 1) == s0) ++msplit;
  const bool live0 = blk * s0 < pos + msplit;
  const bool live1 = msplit < n_act && blk * s1 < pos + n_act;
  if (!live0 && !live1) return;

  __local float qpack[NPR * HD];
  __local float sc_x[2][NPR][WAVE_P];
  for (uint i = lid; i < NPR * HD; i += KV8_WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx[NPR], sm[NPR], acc[NPR];
  for (uint q = 0; q < NPR; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }
  uint wc = 0;

  for (uint g = 0; g < 2; ++g) {
    const uint mlo = g ? msplit : 0u, mhi = g ? n_act : msplit;
    if (mlo >= mhi) continue;
    const uint s = g ? s1 : s0;
    const uint end = pos + mhi;
    const uint bstart = blk * s;
    if (bstart >= end) continue;
    const uint bend = min(bstart + s, end);
    const uint last = end - 1;
    const uint nwaves = (bend - bstart + WAVE_P - 1) / WAVE_P;

    float kreg[PER_LANE], vreg[WAVE_P];
    load_k8(kv_k, k_scale, bstart + sgid, j, bstart + sgid <= last, kreg);
    load_v8(kv_v, v_scale, bstart, j, lid, last, vreg);
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
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
        for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], kreg[t], a);
        for (uint stride = SG / 2; stride > 0; stride >>= 1)
          a += intel_sub_group_shuffle_down(a, a, stride);
        if (lane == 0) sc_x[buf][pr][sgid] = a;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;
        const uint bound = pos + m;
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
        const float nmx = fmax(mx[pr], sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = EXPF(mx[pr] - nmx);
          wl = EXPF(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f, tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) {
          const float ws = intel_sub_group_shuffle(wl, s2);
          ssum += ws;
          tsum = fma(ws, vreg[s2], tsum);
        }
        if (nmx > -INFINITY) {
          sm[pr] = fma(sm[pr], resc, ssum);
          mx[pr] = nmx;
        }
        acc[pr] = fma(acc[pr], resc, tsum);
      }
      if (w + 1 < nwaves) {
        load_k8(kv_k, k_scale, p0 + WAVE_P + sgid, j, p0 + WAVE_P + sgid <= last, kreg);
        load_v8(kv_v, v_scale, p0 + WAVE_P, j, lid, last, vreg);
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out =
          attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + lid] = acc[pr];
      if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
    }
  }
}

// attn_reduce_v2's merge, then o = acc / sm un-rotated across the work-group (work-item
// d = dim d), then attn.cl's gated chain on the un-rotated value.
__attribute__((reqd_work_group_size(KV8_WG, 1, 1)))
__kernel void attn_reduce_v2_kv8(__global const uint* restrict ctrl,
                                 __global const float* restrict attn_part,
                                 __global const float* restrict attn_gate,
                                 __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  __local float xch[HD];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
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
//   from one 16-byte load of the key's int8 row; an int8 is exactly a bf16. The key's scale
//   multiplies the score: s = (dot q8 * sk) * SCORE_SCALE.
// V operand: lane l = dim, 8 pairs of consecutive keys (VNNI), int8 as bf16 exactly; the
//   key's scale goes into P: pa = rne_bf16(p * sv) (lane = key), the row sum stays sum p.
// Keys at or past `depth` load as 0 and score -INF; nothing past the cache's rows is read.
// ---------------------------------------------------------------------------
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

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_flash_attn_kv8(__global const ushort* restrict Q,
                                __global const char* restrict Kc,
                                __global const ushort* restrict Ks,
                                __global const char* restrict Vc,
                                __global const ushort* restrict Vs,
                                __global float* restrict O, uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);
  const uint h = j * 6u + get_group_id(2) * HPW + s / (RPW / 8u);
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));
  if (r0 >= C) return;
  const uint depth = pos + C;
  const int q_w = 24 * HD * 2, q_h = (int)C, q_p = 24 * HD * 2;

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
  for (uint t0 = 0; t0 < last; t0 += KT) {
#if !QREG
    LOAD_Q();
#endif
    float8 sacc[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint b = 0; b < NKA; ++b) {
        const uint key = t0 + 16u * b + l;
        uint kb[8];
        if (key < depth) {
          const char16 c = vload16(0, Kc + ((size_t)key * 4u + j) * HD + 16u * kk);
          kb[0] = kv8_bf(c.s0) | (kv8_bf(c.s1) << 16);
          kb[1] = kv8_bf(c.s2) | (kv8_bf(c.s3) << 16);
          kb[2] = kv8_bf(c.s4) | (kv8_bf(c.s5) << 16);
          kb[3] = kv8_bf(c.s6) | (kv8_bf(c.s7) << 16);
          kb[4] = kv8_bf(c.s8) | (kv8_bf(c.s9) << 16);
          kb[5] = kv8_bf(c.sa) | (kv8_bf(c.sb) << 16);
          kb[6] = kv8_bf(c.sc) | (kv8_bf(c.sd) << 16);
          kb[7] = kv8_bf(c.se) | (kv8_bf(c.sf) << 16);
        } else {
          for (uint z = 0; z < 8u; ++z) kb[z] = 0u;
        }
        sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);
      }
    // scale by the key's K scale, mask, running max (component r = row r0 + r, lane = key)
    float8 tmax = (float8)(-INFINITY);
    float svl[NKA];   // this lane's key's V scale, per key atom
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const uint key = t0 + 16u * b + l;
      const bool live = key < depth;
      const float sk = live ? kv8_f16f(Ks[(size_t)key * 4u + j]) : 0.0f;
      svl[b] = live ? kv8_f16f(Vs[(size_t)key * 4u + j]) : 0.0f;
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
        uint vb[32];   // [c * 16 + jj]: c = dim half, jj = key pair of the 32 keys
#pragma unroll
        for (uint c = 0; c < 2u; ++c) {
          const size_t col = (size_t)j * HD + 16u * (dd + c) + l;
#pragma unroll
          for (uint jj = 0; jj < 16u; ++jj) {
            const uint k0 = t0 + 16u * b + 2u * jj;
            const uint lo = k0 < depth ? kv8_bf(Vc[(size_t)k0 * 4u * HD + col]) : 0u;
            const uint hi = k0 + 1u < depth ? kv8_bf(Vc[(size_t)(k0 + 1u) * 4u * HD + col]) : 0u;
            vb[c * 16u + jj] = lo | (hi << 16);
          }
        }
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks)
            o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);
      }
  }
  const float8 inv = 1.0f / lsum;
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float8 w = o[dd] * inv;
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                               (int2)((int)(16u * dd), (int)r0), (__private uint*)&w);
  }
}
#endif  // KV8_FLASH
