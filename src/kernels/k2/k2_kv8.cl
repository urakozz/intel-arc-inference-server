// k2_kv8.cl - spec 18e Task 1: K2-Horizon's int8 KV cache, the operator's `rotkv` scheme (spec 12
// §8) at head_dim 128 - every kernel that writes or reads it. ONE source, built per family by a
// define (kv8.cl's arrangement), so the rotation, the quantiser and the scale format are one piece
// of text:
//
//   K2KV8_PREP   k2_attn_prep_kv8            the writer: k2_prep.cl's k2_attn_prep chain (q / k RoPE,
//                (M, QKV_N, QKV_S,             the gate), then q rotated and K and V rotated and
//                 V_FROM_PARTIALS)             quantised into the int8 cache; V from the fused row
//                                              (dense layers 0-2) or from MoVA's routed mix (staged)
//   K2KV8_DEC    k2_attn_decode_kv8          k2_attn.cl's decode pair over int8 K / V + fp16
//                k2_attn_reduce_kv8            scales, the output un-rotated per head BEFORE the gate
//                (M, TGT)
//   K2KV8_EAGER  k2_attn_eager_score_kv8     k2_attn_eager.cl's score / P·V / reduce over the int8
//                k2_attn_eager_pv_kv8          cache (the softmax between them is k2_attn_eager.cl's
//                k2_attn_eager_reduce_kv8      own kernel, unchanged: it never sees the cache)
//                (M, TGT)
//   K2KV8_FLASH  k2_pf_flash_attn_kv8        k2_pf_attn.cl over the int8 cache (EAGER 0 / 1), the
//                (KT, RPW, HPW, EAGER,         un-rotation fused into the epilogue (one sub-group
//                 GATE_EPI)                    holds a row's 128 dims), then the softplus gate
//
// Every existing kernel source keeps its bytes (k2_moe.cl gains one define, MOVA_STAGE, whose
// default is the old line): these are new binaries (`*_KV8`, bound only under --kv-cache int8),
// so `--kv-cache bf16` runs today's binaries bit for bit.
//
// ---------------------------------------------------------------------------
// The scheme at head_dim 128 (src/common/kv8.h namespace hd128 is the host twin; edit together)
// ---------------------------------------------------------------------------
//   R = H_128 diag(s) / sqrt(128), H Sylvester in natural order, s = K2KV8_SIGNS: torch's
//   hadamard(128, 0) = the first 128 of 12a's hadamard(256, 0) (kv8.cl's first four words).
//   1 / sqrt(128) is not a power of two, so R is carried with exact scales:
//     rotate_kv(x)[j] = s[j] FWHT(x)[j] / 8      K and V rows        (= sqrt(2) x R)
//     rotate_q(x)[j]  = s[j] FWHT(x)[j] / 16     q                   (= x R / sqrt(2))
//     unrotate(y)[i]  = FWHT(s (.) y)[i] / 16    the attention output (= y R^T / sqrt(2))
//   so rotate_q(q).rotate_kv(k) = q.k and unrotate(sum p rotate_kv(v)) = sum p v: the scores,
//   the softmax and the output are bf16 KV's up to the int8 error. The un-rotation is per head
//   and comes BEFORE the softplus gate (the gate is channel-wise in the un-rotated basis).
//   FWHT: stages h = 1, 2, ..., 64 ascending; (i, i + h), i & h == 0 -> (a + c, a - c).
//
//   What is rotated is what the bf16 cache would have stored: K = f32(the RoPE chain's bf16 out),
//   V = f32(rne(v column)) on the dense layers and MoVA's routed mix - k2_mova_value's /
//   k2_pf_mova_combine's bf16 result, AFTER the value experts' combine (spec 18e Review Focus 2).
//   q is rotated from k2_attn_prep's fp32 value (the RoPE chain's bf16 out): decode keeps it
//   fp32, the prefill flash rounds it to bf16 at load (as k2_pf_attn.cl does with q).
//
//   Per (position, kv head):  amax = max |y|;  s16 = f16_rne(amax / 127);
//     q8 = s16 == 0 ? 0 : clamp(rint(y / f16f(s16)), -127, 127);  deq = float(q8) f16f(s16)
//   (deq is exact in fp32: 7 + 11 significant bits).
//
// Layout (runtime::KvLayout at K2's 8 kv heads x 128): one layer's K (or V) is int8
// [max_len][8][128]; its scales fp16 bits [max_len][8], in the same allocation after every
// layer's rows (runtime/k2/k2_buffers.h kv_layer). The kernels take the two pointers.
//
// The readers dequantise while loading:
//   decode flash  K and V exactly, deq in fp32 (k2_attn.cl's arithmetic and orders on those values);
//   decode eager  the same exact deq; the reference's rounding points (score rounded twice, p to
//                 bf16) on the rotated q and the dequantised K, P·V in fp32, then the un-rotation;
//   prefill       K's int8 is the bf16 DPAS operand as is (exact) and the score is multiplied by the
//                 key's scale; V's int8 likewise, its scale folded into P (rne(p s_v) - one rounding
//                 more than bf16 KV's rne(p)); the row sum is P's, unscaled.
//
// Rounding: the FWHT, the flips, the power-of-two scales and the quantiser are bit-defined (the
// build has -cl-fp32-correctly-rounded-divide-sqrt, `rint` is round-half-even, max is exact, and
// FP_CONTRACT is off where a host twin is bitwise), so the writer is bitwise src/common/kv8.h
// and the un-rotations are bitwise its unrotate on the same fp32 input.
#ifndef K2KV8_PREP
#define K2KV8_PREP 0
#endif
#ifndef K2KV8_DEC
#define K2KV8_DEC 0
#endif
#ifndef K2KV8_EAGER
#define K2KV8_EAGER 0
#endif
#ifndef K2KV8_FLASH
#define K2KV8_FLASH 0
#endif
#if K2KV8_PREP + K2KV8_DEC + K2KV8_EAGER + K2KV8_FLASH != 1
#error "k2_kv8: define exactly one of K2KV8_PREP, K2KV8_DEC, K2KV8_EAGER, K2KV8_FLASH (src/kernels/CMakeLists.txt)"
#endif

#if K2KV8_DEC || K2KV8_FLASH
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif
#if K2KV8_PREP || K2KV8_EAGER
#pragma OPENCL FP_CONTRACT OFF
#endif

#if !defined(Q_HEADS) || !defined(KV_HEADS)
#error "k2_kv8: Q_HEADS and KV_HEADS must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "k2_kv8: q-heads must be a multiple of kv-heads"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define KV_N (KV_HEADS * HD)
#define SSCALE 0.08838834764831845f   /* float(128 ** -0.5): the reference's scaling, as fp32 */
#define SP_BETA 0.6931471805599453f   /* F.softplus(beta = math.log(2)), as fp32 */
#define SP_THRESHOLD 20.0f
#define KV8_ROT_KV 0.125f             /* rotate_kv's 1/8 */
#define KV8_ROT_Q 0.0625f             /* rotate_q's and unrotate's 1/16 */

// hadamard(128, 0)'s signs (src/common/kv8.h kSignWords[0..3]): bit j of word j / 32 set when
// s[j] = -1.
__constant uint K2KV8_SIGNS[4] = {0xA197D809u, 0xA6850592u, 0xB205A73Eu, 0x20AFA769u};

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
// torch.nn.functional.softplus(x, beta, threshold = 20) - k2_attn.cl's softplus_k2.
inline float softplus_k2(float x) {
  const float bx = x * SP_BETA;
  return bx > SP_THRESHOLD ? x : log1p(exp(bx)) / SP_BETA;
}
// k2_attn_reduce's last line: rne(f32(rne(o)) x f32(rne(softplus(gate)))).
inline ushort k2_gated(float o, float g) {
  return rne_bf16(bf16f(rne_bf16(o)) * bf16f(rne_bf16(softplus_k2(g))));
}

// s[j] applied as the sign bit: exact, the same bits as a multiply by -1.
inline uint kv8_negbit(uint j) { return ((K2KV8_SIGNS[j >> 5] >> (j & 31u)) & 1u) << 31; }
inline float kv8_flip(float v, uint j) { return as_float(as_uint(v) ^ kv8_negbit(j)); }

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

#if K2KV8_PREP || K2KV8_DEC || K2KV8_EAGER
// The 128-point FWHT across a 128-work-item work-group, work-item i owning element i, staged
// through `buf` (128 floats of SLM). Every work-item must call it (barriers). It starts and ends
// on a barrier-safe footing: the first write follows the caller's last read of `buf` only through
// the trailing barrier of a previous call.
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
inline float kv8_unrotate(float y, uint i, __local float* buf) {
  return kv8_fwht(kv8_flip(y, i), i, buf) * KV8_ROT_Q;
}
#endif

#if K2KV8_PREP
// ---------------------------------------------------------------------------
// k2_attn_prep_kv8 - grid (Q_HEADS + KV_HEADS, M), work-group 128 (k2_prep.cl's k2_attn_prep,
// whose header states the RoPE chain transcribed here unchanged).
//
//   k2_attn_prep_kv8(ctrl, partials, rope, attn_q, attn_gate, kv_k, kv_v, k_scale, v_scale, v_rows)
//     attn_q            fp32 [M][Q_HEADS][128]   rotate_q(the RoPE chain's bf16 out)
//     attn_gate         fp32 [M][Q_HEADS][128]   f32(rne(gate column)) - k2_attn_prep's, unchanged
//     kv_k, kv_v        int8 [max_len][KV_HEADS][128]   this layer's rows
//     k_scale, v_scale  fp16 bits [max_len][KV_HEADS]   this layer's scales
//     v_rows            bf16 [M][KV_HEADS x 128]  V_FROM_PARTIALS 0 (MoVA): the routed mix, row m -
//                       k2_mova_value's (MOVA_STAGE) / k2_pf_mova_combine's output, written there
//                       instead of the cache; unread on the dense build
//   kv-head j: K = rotate_kv(f32(RoPE out)), V = rotate_kv(f32(v)), each quantised at row pos + m.
// ---------------------------------------------------------------------------
#ifndef QKV_N
#error "k2_kv8 prep: QKV_N must be defined"
#endif
#ifndef QKV_S
#error "k2_kv8 prep: QKV_S (the fused GEMV's split-K) must be defined"
#endif
#ifndef M
#error "k2_kv8 prep: M must be defined"
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_kv8 prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef V_FROM_PARTIALS
#error "k2_kv8 prep: V_FROM_PARTIALS must be defined (1: dense layers, 0: MoVA's staged mix)"
#endif
#define K_OFF (Q_HEADS * HD)
#define GATE_OFF (K_OFF + KV_HEADS * HD)
#define V_OFF (GATE_OFF + Q_HEADS * HD)
#define HALF (HD / 2)
#if V_FROM_PARTIALS && QKV_N != V_OFF + KV_HEADS * HD
#error "k2_kv8 prep: a dense layer's q||k||gate||v is V_OFF + KV_HEADS x HD wide"
#endif
#if !V_FROM_PARTIALS && QKV_N < V_OFF
#error "k2_kv8 prep: the fused row ends before its v_router columns"
#endif

inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
}

// Quantise this work-item's element of one rotated row: the row's amax by the pairwise fmax tree
// over `red`, the fp16 scale, the int8. Ends on a barrier, so `red` may be reused at once.
inline void kv8_store(float y, uint i, __local float* red, __global char* restrict row,
                      __global ushort* restrict scale) {
  red[i] = fabs(y);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = HD / 2; stride > 0; stride >>= 1) {
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

__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void k2_attn_prep_kv8(__global const uint* restrict ctrl,
                               __global const float* restrict partials,
                               __global const float* restrict rope,
                               __global float* restrict attn_q, __global float* restrict attn_gate,
                               __global char* restrict kv_k, __global char* restrict kv_v,
                               __global ushort* restrict k_scale, __global ushort* restrict v_scale,
                               __global const ushort* restrict v_rows) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float xs[HD];    // the head's linear output (bf16 values) - RoPE needs dim i +- 64
  __local float red[HD];   // the amax tree
  __local float xch[HD];   // the FWHT's exchange
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform across the work-group

  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  const size_t col = is_q ? (size_t)h * HD + i : (size_t)K_OFF + (size_t)h * HD + i;
  xs[i] = bf16f(rne_bf16(qkv_sum(partials, m, col)));
  barrier(CLK_LOCAL_MEM_FENCE);

  // k2_attn_prep's RoPE chain, unchanged (three roundings; exact products).
  const __global float* restrict cs = rope + (size_t)(pos + m) * 2 * HALF;
  const uint ii = i % HALF;
  const float c = cs[ii], sn = cs[HALF + ii];
  const float rot = i < HALF ? -xs[i + HALF] : xs[i - HALF];
  const ushort t1 = rne_bf16(xs[i] * c);
  const ushort t2 = rne_bf16(rot * sn);
  const ushort out = rne_bf16(bf16f(t1) + bf16f(t2));

  if (is_q) {   // uniform: the barriers inside are reached by all or none
    const float y = kv8_flip(kv8_fwht(bf16f(out), i, xch), i) * KV8_ROT_Q;
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = y;
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_sum(partials, m, (size_t)GATE_OFF + (size_t)h * HD + i)));
  } else {
    const size_t row = (size_t)(pos + m) * KV_HEADS + h;
    const float yk = kv8_flip(kv8_fwht(bf16f(out), i, xch), i) * KV8_ROT_KV;
    kv8_store(yk, i, red, kv_k + row * HD, k_scale + row);
#if V_FROM_PARTIALS
    const float vb = bf16f(rne_bf16(qkv_sum(partials, m, (size_t)V_OFF + (size_t)h * HD + i)));
#else
    const float vb = bf16f(v_rows[(size_t)m * KV_N + (size_t)h * HD + i]);
#endif
    const float yv = kv8_flip(kv8_fwht(vb, i, xch), i) * KV8_ROT_KV;
    kv8_store(yv, i, red, kv_v + row * HD, v_scale + row);
  }
#if V_FROM_PARTIALS
  (void)v_rows;
#endif
}
#endif  // K2KV8_PREP

#if K2KV8_DEC || K2KV8_EAGER
#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_kv8: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef TGT
#error "k2_kv8: TGT must be defined (kernels::k2::kAttnTgt)"
#endif
#if M < 1 || M > 4
#error "k2_kv8: M is 1..4"
#endif
#define PART (HD + 2)
#define WG HD
#define NPR (M * GQA)
inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}
#endif

#if K2KV8_DEC
// ---------------------------------------------------------------------------
// k2_attn_decode_kv8 / k2_attn_reduce_kv8 - k2_attn.cl's pair (its header is the contract:
// ppw(L) per row, grid (KV_HEADS, TGT), partials [Q_HEADS][TGT][M][130], waves of 16 positions,
// two per sub-group) with the loads replaced and the reduce's un-rotation added.
//
//   k2_attn_decode_kv8(ctrl, attn_q, kv_k, k_scale, kv_v, v_scale, attn_part)
//   k2_attn_reduce_kv8(ctrl, attn_part, attn_gate, attn_out)
//
// K: a row is ONE sub-group block read of 8 bytes per lane (lane l element t = dim l + 16 t,
//    k2_attn.cl's order), dequantised at use as float(q8) x f16f(scale) - exact, so the dot's
//    operands are the dequantised values whichever way they are formed.
// V: work-item d's 16 values of the wave, dim d of positions p0 .. p0 + 15, plain loads, zero
//    past `last` (the bf16 kernel's 2D read surface height), dequantised the same way.
// The online softmax and every order are k2_attn.cl's.
// ---------------------------------------------------------------------------
#define WAVE_P 16
#define SG 16
#define NSG (HD / SG)          /* 8 sub-groups */
#define PPS (WAVE_P / NSG)     /* 2 positions of a wave per sub-group */
#define PER_LANE (HD / SG)     /* 8 elements of a 128-dim dot per lane */
#define DSCALE 0.08838834764831845f   /* k2_attn.cl's SSCALE */

inline void load_k8(__global const char* restrict kv_k, __global const ushort* restrict k_scale,
                    uint p, uint j, bool ok, char* kq, float* ksc) {
  if (ok) {   // uniform within the sub-group: p is the sub-group's position
    const size_t r = (size_t)p * KV_HEADS + j;
    const uchar8 a = intel_sub_group_block_read_uc8((__global const uchar*)(kv_k + r * HD));
    *ksc = kv8_f16f(k_scale[r]);
    kq[0] = as_char(a.s0); kq[1] = as_char(a.s1); kq[2] = as_char(a.s2); kq[3] = as_char(a.s3);
    kq[4] = as_char(a.s4); kq[5] = as_char(a.s5); kq[6] = as_char(a.s6); kq[7] = as_char(a.s7);
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kq[t] = 0;
    *ksc = 0.0f;
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

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void k2_attn_decode_kv8(__global const uint* restrict ctrl,
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

  __local float qpack[NPR * HD];              // [m][qhl][128], rotated q
  __local float sc_x[2][NPR][WAVE_P];         // the wave's raw dots, double buffered
  for (uint i = lid; i < NPR * HD; i += WG) {
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

    char kq[PPS][PER_LANE];
    float ksc[PPS], vreg[WAVE_P];
    for (uint pp = 0; pp < PPS; ++pp) {
      const uint p = bstart + sgid + NSG * pp;
      load_k8(kv_k, k_scale, p, j, p <= last, kq[pp], &ksc[pp]);
    }
    load_v8(kv_v, v_scale, bstart, j, lid, last, vreg);
    for (uint w = 0; w < nwaves; ++w, ++wc) {
      const uint p0 = bstart + w * WAVE_P;
      const uint buf = wc & 1u;
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[pr * HD]);
        float qv[PER_LANE];
        qv[0] = as_float(qa.s0); qv[1] = as_float(qa.s1); qv[2] = as_float(qa.s2);
        qv[3] = as_float(qa.s3); qv[4] = as_float(qa.s4); qv[5] = as_float(qa.s5);
        qv[6] = as_float(qa.s6); qv[7] = as_float(qa.s7);
        for (uint pp = 0; pp < PPS; ++pp) {
          float a = 0.0f;
          for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], (float)kq[pp][t] * ksc[pp], a);   // exact operand
          for (uint stride = SG / 2; stride > 0; stride >>= 1)
            a += intel_sub_group_shuffle_down(a, a, stride);
          if (lane == 0) sc_x[buf][pr][sgid + NSG * pp] = a;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
      for (uint pr = 0; pr < NPR; ++pr) {
        const uint m = pr / GQA;
        if (m < mlo || m >= mhi) continue;    // uniform
        const uint bound = pos + m;
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * DSCALE : -INFINITY;
        const float nmx = fmax(mx[pr], sub_group_reduce_max(mine));
        float resc, wl;
        if (nmx > -INFINITY) {
          resc = exp(mx[pr] - nmx);
          wl = exp(mine - nmx);
        } else {
          resc = 1.0f;
          wl = 0.0f;
        }
        float ssum = 0.0f, tsum = 0.0f;
        for (uint s2 = 0; s2 < WAVE_P; ++s2) {   // ascending s
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
        for (uint pp = 0; pp < PPS; ++pp) {
          const uint p = p0 + WAVE_P + sgid + NSG * pp;
          load_k8(kv_k, k_scale, p, j, p <= last, kq[pp], &ksc[pp]);
        }
        load_v8(kv_v, v_scale, p0 + WAVE_P, j, lid, last, vreg);
      }
    }
#pragma unroll
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m < mlo || m >= mhi) continue;
      __global float* restrict out = attn_part + (((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART;
      out[2 + lid] = acc[pr];
      if (lid == 0) { out[0] = mx[pr]; out[1] = sm[pr]; }
    }
  }
}

// k2_attn_reduce's merge, then o = acc / sm un-rotated across the work-group (work-item d = dim
// d), then the softplus gate on the un-rotated value.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_reduce_kv8(__global const uint* restrict ctrl,
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
  if (m >= n_act) return;   // uniform
  uint nb = (pos + m) / v2_ppw(pos + m + 1) + 1;
  if (nb > TGT) nb = TGT;
  for (uint b = d; b < nb; b += WG) {
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
    const float a = exp(mx - nmx);
    const float bs = exp(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const float o = kv8_unrotate(acc / sm, d, xch);
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = k2_gated(o, g);
}
#endif  // K2KV8_DEC

#if K2KV8_EAGER
// ---------------------------------------------------------------------------
// The eager decode attention over the int8 cache - k2_attn_eager.cl's chain (its header is the
// contract: the score row attn_s [M][Q_HEADS][stride], the blocks of eager_ppw over the step's
// longest row, attn_part words 2..) with the cache's loads replaced and the un-rotation added:
//
//   k2_attn_eager_score_kv8 (ctrl, attn_q, kv_k, k_scale, attn_s, stride)   grid (KV_HEADS, TGT)
//     s_p = rne(f32(rne(sum_d q_rot,d x deq(K)_pd)) x SCALE)   one fma chain, d ascending; the
//     operand deq = float(q8) x f16f(scale) is exact, q_rot fp32 (so the products are not: the
//     fma chain is the one order, contraction off - kv8_ref::eager_score is the twin)
//   k2_attn_eager_softmax   k2_attn_eager.cl's own kernel, unchanged (torch's softmax bit for bit)
//   k2_attn_eager_pv_kv8    (ctrl, attn_s, kv_v, v_scale, attn_part, stride) grid (KV_HEADS, TGT)
//     per key block sum_p p_p x deq(V)_pd, an fma chain ascending
//   k2_attn_eager_reduce_kv8 (ctrl, attn_part, attn_gate, attn_out)          grid (Q_HEADS, M)
//     o_rot = the blocks added ascending (fp32); o = unrotate(o_rot); out = k2_gated(o, gate)
// Where bf16 KV's eager rounds P·V to bf16 and gates it, the int8 form un-rotates the fp32 sum
// first and rounds the un-rotated value (the reference's one rounding of o, in the right basis).
// Plain OpenCL C (the Mac's OpenCL runs it: tools/mac/clrun/k2_run.cc).
// ---------------------------------------------------------------------------
inline uint eager_ppw(uint len) { return v2_ppw(len); }
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_score_kv8(__global const uint* restrict ctrl,
                                      __global const float* restrict attn_q,
                                      __global const char* restrict kv_k,
                                      __global const ushort* restrict k_scale,
                                      __global float* restrict attn_s,
                                      const uint stride) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (n_act == 0) return;
  const uint end = pos + n_act;
  const uint s = eager_ppw(end);
  const uint bstart = blk * s;
  if (bstart >= end) return;   // uniform
  const uint bend = min(bstart + s, end);

  __local float qpack[NPR * HD];   // [m][qhl][128]
  for (uint i = lid; i < NPR * HD; i += WG) {
    const uint pr = i / HD, d = i % HD;
    const uint m = pr / GQA, qhl = pr % GQA;
    qpack[i] = m < n_act ? attn_q[((size_t)m * Q_HEADS + j * GQA + qhl) * HD + d] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint p = bstart + lid; p < bend; p += WG) {
    const size_t r = (size_t)p * KV_HEADS + j;
    __global const char* restrict kr = kv_k + r * HD;
    const float ksc = kv8_f16f(k_scale[r]);
    float a[NPR];
    for (uint pr = 0; pr < NPR; ++pr) a[pr] = 0.0f;
    for (uint d0 = 0; d0 < HD; d0 += 8) {
      const char8 k8 = vload8(0, kr + d0);
      const float kf[8] = {(float)k8.s0 * ksc, (float)k8.s1 * ksc, (float)k8.s2 * ksc, (float)k8.s3 * ksc,
                           (float)k8.s4 * ksc, (float)k8.s5 * ksc, (float)k8.s6 * ksc, (float)k8.s7 * ksc};
      for (uint pr = 0; pr < NPR; ++pr)
        for (uint t = 0; t < 8; ++t) a[pr] = fma(qpack[pr * HD + d0 + t], kf[t], a[pr]);   // d ascending
    }
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m >= n_act || p > pos + m) continue;   // row m's keys end at pos + m
      attn_s[((size_t)m * Q_HEADS + j * GQA + qhl) * stride + p] = rf(rf(a[pr]) * SSCALE);
    }
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_pv_kv8(__global const uint* restrict ctrl,
                                   __global const float* restrict attn_s,
                                   __global const char* restrict kv_v,
                                   __global const ushort* restrict v_scale,
                                   __global float* restrict attn_part,
                                   const uint stride) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint d = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (n_act == 0) return;
  const uint end = pos + n_act;
  const uint s = eager_ppw(end);
  const uint bstart = blk * s;
  if (bstart >= end) return;   // uniform
  const uint bend = min(bstart + s, end);

  float acc[NPR];
  for (uint pr = 0; pr < NPR; ++pr) acc[pr] = 0.0f;
  for (uint p = bstart; p < bend; ++p) {
    const size_t r = (size_t)p * KV_HEADS + j;
    const float v = (float)kv_v[r * HD + d] * kv8_f16f(v_scale[r]);   // exact
    for (uint pr = 0; pr < NPR; ++pr) {
      const uint m = pr / GQA, qhl = pr % GQA;
      if (m >= n_act || p > pos + m) continue;   // uniform
      acc[pr] = fma(attn_s[((size_t)m * Q_HEADS + j * GQA + qhl) * stride + p], v, acc[pr]);
    }
  }
  for (uint pr = 0; pr < NPR; ++pr) {
    const uint m = pr / GQA, qhl = pr % GQA;
    if (m >= n_act || bstart > pos + m) continue;   // the reduce reads row m's blocks only
    attn_part[(((size_t)(j * GQA + qhl) * TGT + blk) * M + m) * PART + 2 + d] = acc[pr];
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_reduce_kv8(__global const uint* restrict ctrl,
                                       __global const float* restrict attn_part,
                                       __global const float* restrict attn_gate,
                                       __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float xch[HD];
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (m >= n_act) return;   // uniform
  const uint s = eager_ppw(pos + n_act);
  const uint nb = (pos + m + 1 + s - 1) / s;
  float o_rot = 0.0f;
  for (uint b = 0; b < nb; ++b) o_rot += attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
  const float o = kv8_unrotate(o_rot, d, xch);
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = k2_gated(o, g);
}
#endif  // K2KV8_EAGER

#if K2KV8_FLASH
// ---------------------------------------------------------------------------
// k2_pf_flash_attn_kv8 - k2_pf_attn.cl (its header is the contract: 8 query rows per sub-group,
// S in the DPAS A layout, the online softmax, the tiles from key 0, EAGER's two passes) over the
// int8 cache, the un-rotation and the gate in the epilogue:
//
//   k2_pf_flash_attn_kv8(Q, Kc, Ks, Vc, Vs, G, out, pos, C)
//     Q      fp32 [kC][Q_HEADS][128]   rotate_q'd q (k2_attn_prep_kv8 at M = kC), rounded to bf16
//                                      at load as k2_pf_attn.cl rounds q
//     Kc/Vc  int8 [depth][KV_HEADS][128], Ks/Vs fp16 bits [depth][KV_HEADS]   this layer's cache
//     G      fp32 [kC][Q_HEADS][128]   the gate (k2_attn_prep_kv8's attn_gate)
//     out    GATE_EPI 1: bf16 [C][Q_HEADS x 128]  o_proj's A operand
//            GATE_EPI 0: fp32 [Q_HEADS][C][128]   the UN-ROTATED, ungated o (the K1 test's form)
//
// K^T operand: lane l = key t0 + 16 b + l, its 16 dims 16 kk .. 16 kk + 15 as 8 bf16 pairs (low
//   half = even dim) - the transposed 2D read's layout in k2_pf_attn.cl - from one 16-byte load of
//   the key's int8 row; an int8 is exactly a bf16. The key's K scale multiplies the dot: the score
//   is score_of(dot x sk) - k2_pf_attn.cl's score_of (fp32 scale, or EAGER's two roundings).
// V operand: lane l = dim, 8 pairs of consecutive keys (VNNI), int8 as bf16 exactly; the key's V
//   scale goes into P: pa = rne(p x sv) (default: p = exp(s - m); EAGER: p = rf(exp(s - m) / l),
//   the reference's bf16 probability, scaled and rounded once more). The row sum stays sum p.
// Keys at or past `depth` load as 0 and score -INF; nothing past the cache's rows is read.
// Epilogue: per row r0 + r, w = o / l (EAGER: o, p was normalised); the 128 dims of the row are
//   the sub-group's (dim 16 dd + lane): flipped by s, the FWHT's stages 1, 2, 4, 8 across lanes
//   (sub-group xor shuffles) and 16, 32, 64 across dd (registers), in ascending order - the host's
//   unrotate op for op, so bitwise kv8::hd128::unrotate on the same w - times 1/16, then the gate.
// ---------------------------------------------------------------------------
#ifndef KT
#error "k2_kv8 flash: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "k2_kv8 flash: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "k2_kv8 flash: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#ifndef HPW
#error "k2_kv8 flash: HPW must be defined"
#endif
#if GQA % HPW != 0
#error "k2_kv8 flash: the GQA group must be a multiple of HPW"
#endif
#if !defined(EAGER) || !defined(GATE_EPI)
#error "k2_kv8 flash: EAGER and GATE_EPI must be defined (0 or 1 each)"
#endif
#define SG 16
#define NDA (HD / 16)                  /* 8 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */

// An int8 as its exact bf16 bits.
inline uint kv8_bf(char c) { return as_uint((float)c) >> 16; }
// k2_pf_attn.cl's score_of on the scaled dot.
inline float score_of(float dot) {
#if EAGER
  return rf(rf(dot) * SSCALE);
#else
  return dot * SSCALE;
#endif
}
// One xor-shuffle FWHT stage across the sub-group's lanes on a float8 (8 rows), per component.
inline float8 fwht_lanes(float8 v, uint l, uint h) {
  float8 o;
  o.s0 = intel_sub_group_shuffle_xor(v.s0, h); o.s1 = intel_sub_group_shuffle_xor(v.s1, h);
  o.s2 = intel_sub_group_shuffle_xor(v.s2, h); o.s3 = intel_sub_group_shuffle_xor(v.s3, h);
  o.s4 = intel_sub_group_shuffle_xor(v.s4, h); o.s5 = intel_sub_group_shuffle_xor(v.s5, h);
  o.s6 = intel_sub_group_shuffle_xor(v.s6, h); o.s7 = intel_sub_group_shuffle_xor(v.s7, h);
  return (l & h) ? (o - v) : (v + o);
}

__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void k2_pf_flash_attn_kv8(__global const float* restrict Q,
                                   __global const char* restrict Kc,
                                   __global const ushort* restrict Ks,
                                   __global const char* restrict Vc,
                                   __global const ushort* restrict Vs,
                                   __global const float* restrict G,
#if GATE_EPI
                                   __global ushort* restrict out,
#else
                                   __global float* restrict out,
#endif
                                   uint pos, uint C) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + s / (RPW / 8u);      // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));        // first of 8 rows
  if (r0 >= C) return;                                                  // no barriers below
  const uint depth = pos + C;
#if !GATE_EPI
  (void)G;
#endif

  // Q, 8 rows x 128 dims as NDA A operands: qa[kk].s_r = q[r0 + r][h][16 kk + lane].
  short8 qa[NDA];
#pragma unroll
  for (uint kk = 0; kk < NDA; ++kk) {
    ushort t[8];
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint row = min(r0 + r, C - 1u);
      t[r] = rne_bf16(Q[((size_t)row * Q_HEADS + h) * HD + 16u * kk + l]);
    }
    qa[kk] = as_short8(vload8(0, t));
  }

  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 8u);          // keys this SG's rows can see

  // The score tile at t0 from the int8 K rows, scaled by each key's K scale, scored and masked;
  // svl[b]: this lane's key's V scale. Component r = row r0 + r; lane = key t0 + 16 b + lane.
#define KV8_SCORES(t0)                                                                      \
  float8 sacc[NKA];                                                                         \
  float svl[NKA];                                                                           \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk)                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    const uint key = (t0) + 16u * b + l;                                                    \
    uint kb[8];                                                                             \
    if (key < depth) {                                                                      \
      const char16 c = vload16(0, Kc + ((size_t)key * KV_HEADS + j) * HD + 16u * kk);       \
      kb[0] = kv8_bf(c.s0) | (kv8_bf(c.s1) << 16);                                          \
      kb[1] = kv8_bf(c.s2) | (kv8_bf(c.s3) << 16);                                          \
      kb[2] = kv8_bf(c.s4) | (kv8_bf(c.s5) << 16);                                          \
      kb[3] = kv8_bf(c.s6) | (kv8_bf(c.s7) << 16);                                          \
      kb[4] = kv8_bf(c.s8) | (kv8_bf(c.s9) << 16);                                          \
      kb[5] = kv8_bf(c.sa) | (kv8_bf(c.sb) << 16);                                          \
      kb[6] = kv8_bf(c.sc) | (kv8_bf(c.sd) << 16);                                          \
      kb[7] = kv8_bf(c.se) | (kv8_bf(c.sf) << 16);                                          \
    } else {                                                                                \
      for (uint z = 0; z < 8u; ++z) kb[z] = 0u;                                             \
    }                                                                                       \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]); \
  }                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    const uint key = (t0) + 16u * b + l;                                                    \
    const bool live = key < depth;                                                          \
    const float sk = live ? kv8_f16f(Ks[(size_t)key * KV_HEADS + j]) : 0.0f;                \
    svl[b] = live ? kv8_f16f(Vs[(size_t)key * KV_HEADS + j]) : 0.0f;                        \
    KV8_MASK(0) KV8_MASK(1) KV8_MASK(2) KV8_MASK(3) KV8_MASK(4) KV8_MASK(5) KV8_MASK(6) KV8_MASK(7) \
  }
#define KV8_MASK(r) \
  sacc[b].s##r = (key <= pos + r0 + r && live) ? score_of(sacc[b].s##r * sk) : -INFINITY;

  // P·V for one tile: A = pa[b] (8 rows x 16 keys, the V scale folded in), B = V int8 as bf16
  // (16 keys x 16 dims, VNNI: key pairs packed low / high).
#define KV8_PV(t0)                                                                          \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                  \
    uint vb[32];                                                                            \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c) {                                       \
      const size_t col = (size_t)j * HD + 16u * (dd + c) + l;                               \
      _Pragma("unroll") for (uint jj = 0; jj < 16u; ++jj) {                                 \
        const uint k0 = (t0) + 16u * b + 2u * jj;                                           \
        const uint lo = k0 < depth ? kv8_bf(Vc[(size_t)k0 * KV_N + col]) : 0u;             \
        const uint hi = k0 + 1u < depth ? kv8_bf(Vc[(size_t)(k0 + 1u) * KV_N + col]) : 0u;  \
        vb[c * 16u + jj] = lo | (hi << 16);                                                 \
      }                                                                                     \
    }                                                                                       \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks)                                      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(                                 \
          pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);               \
  }

#define KV8_ROWMAX()                                                                        \
  float8 tmax = (float8)(-INFINITY);                                                        \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) tmax = fmax(tmax, sacc[b]);              \
  float8 mnew;                                                                              \
  mnew.s0 = fmax(m.s0, sub_group_reduce_max(tmax.s0));                                      \
  mnew.s1 = fmax(m.s1, sub_group_reduce_max(tmax.s1));                                      \
  mnew.s2 = fmax(m.s2, sub_group_reduce_max(tmax.s2));                                      \
  mnew.s3 = fmax(m.s3, sub_group_reduce_max(tmax.s3));                                      \
  mnew.s4 = fmax(m.s4, sub_group_reduce_max(tmax.s4));                                      \
  mnew.s5 = fmax(m.s5, sub_group_reduce_max(tmax.s5));                                      \
  mnew.s6 = fmax(m.s6, sub_group_reduce_max(tmax.s6));                                      \
  mnew.s7 = fmax(m.s7, sub_group_reduce_max(tmax.s7));
#define KV8_ROWSUM(psum, rsum)                                                              \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define KV8_CV(r) pa[b].s##r = as_short(rne_bf16(ps.s##r));

#if !EAGER
  // The default path: the online softmax, P = exp(s - m) x s_v rounded to bf16 for P·V, O / l last.
  for (uint t0 = 0; t0 < last; t0 += KT) {
    KV8_SCORES(t0)
    KV8_ROWMAX()
    const float8 corr = exp(m - mnew);           // m = -INF on the first tile: exp(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - mnew);      // masked: exp(-INF) = 0
      psum += p;
      const float8 ps = p * svl[b];
      KV8_CV(0) KV8_CV(1) KV8_CV(2) KV8_CV(3) KV8_CV(4) KV8_CV(5) KV8_CV(6) KV8_CV(7)
    }
    KV8_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    KV8_PV(t0)
  }
#else
  // Pass 1: the row's max and sum exp(s - max) over its keys (fp32, online).
  for (uint t0 = 0; t0 < last; t0 += KT) {
    KV8_SCORES(t0)
    KV8_ROWMAX()
    const float8 corr = exp(m - mnew);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - mnew);
    KV8_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
  }
  // Pass 2: p = rf(exp(s - m) / l), the softmax's bf16 output, times s_v rounded once more;
  // O = sum p v in fp32.
  for (uint t0 = 0; t0 < last; t0 += KT) {
    KV8_SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 pe = exp(sacc[b] - m) / lsum;
      float8 ps;
      ps.s0 = rf(pe.s0) * svl[b]; ps.s1 = rf(pe.s1) * svl[b]; ps.s2 = rf(pe.s2) * svl[b];
      ps.s3 = rf(pe.s3) * svl[b]; ps.s4 = rf(pe.s4) * svl[b]; ps.s5 = rf(pe.s5) * svl[b];
      ps.s6 = rf(pe.s6) * svl[b]; ps.s7 = rf(pe.s7) * svl[b];
      KV8_CV(0) KV8_CV(1) KV8_CV(2) KV8_CV(3) KV8_CV(4) KV8_CV(5) KV8_CV(6) KV8_CV(7)
    }
    KV8_PV(t0)
  }
#endif
#undef KV8_SCORES
#undef KV8_MASK
#undef KV8_PV
#undef KV8_ROWMAX
#undef KV8_ROWSUM
#undef KV8_CV

  // The epilogue: w[dd].s_r = o of row r0 + r at dim 16 dd + l (rotated), then unrotate.
  float8 w[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
#if EAGER
    w[dd] = o[dd];                               // already normalised
#else
    w[dd] = o[dd] / lsum;
#endif
    w[dd] = as_float8(as_uint8(w[dd]) ^ (uint8)(kv8_negbit(16u * dd + l)));   // s (.) y
  }
  for (uint hl = 1u; hl < SG; hl <<= 1) {        // stages 1, 2, 4, 8: lane bits
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) w[dd] = fwht_lanes(w[dd], l, hl);
  }
#pragma unroll
  for (uint hb = 1u; hb < NDA; hb <<= 1) {       // stages 16, 32, 64: dd bits
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd)
      if ((dd & hb) == 0u) {
        const float8 a = w[dd], c = w[dd | hb];
        w[dd] = a + c;
        w[dd | hb] = a - c;
      }
  }
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    const float8 x = w[dd] * KV8_ROT_Q;
    float wv[8];
    vstore8(x, 0, wv);
    for (uint r = 0; r < 8; ++r) {
      const uint row = r0 + r;
      if (row >= C) break;
      const uint d = 16u * dd + l;
#if GATE_EPI
      const size_t idx = ((size_t)row * Q_HEADS + h) * HD + d;
      out[idx] = k2_gated(wv[r], G[idx]);
#else
      out[((size_t)h * C + row) * HD + d] = wv[r];
#endif
    }
  }
}
#endif  // K2KV8_FLASH
