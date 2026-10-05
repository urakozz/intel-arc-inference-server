// k2_attn_eager.cl - K2-Horizon's EAGER decode attention (B70_K2_ATTN=eager; spec 18 §10): the
// reference's bf16 eager op chain (tools/oracle/k2_ref.py attention() = transformers'
// eager_attention_forward in bf16, modeling_k2_horizon.py:110-133) on the device, beside
// k2_attn.cl's flash form, which it replaces only when the capture selects it. k2_attn.cl and
// its binary are untouched.
//
//   k2_attn_eager_score  (ctrl, attn_q, kv_k, attn_s, stride)              grid (KV_HEADS, TGT)
//   k2_attn_eager_softmax(ctrl, attn_s, stride)                            grid (Q_HEADS, M)
//   k2_attn_eager_pv     (ctrl, attn_s, kv_v, attn_part, stride)           grid (KV_HEADS, TGT)
//   k2_attn_eager_reduce (ctrl, attn_part, attn_gate, attn_out)            grid (Q_HEADS, M)
//     every work-group 128 work-items
//     attn_q     fp32 [M][Q_HEADS][128]           k2_attn_prep's RoPE'd q (bf16 values)
//     kv_k/kv_v  bf16 [max_len][KV_HEADS][128]    this layer's caches
//     attn_s     fp32 [M][Q_HEADS][stride]        a row of scores, then of probabilities
//                                                 (bf16 values); stride = max_len
//     attn_part  fp32 [Q_HEADS][TGT][M][130]      words 2.. hold a block's P·V sums (flash's
//                                                 buffer, its {mx, sm} words unused here)
//     attn_gate  fp32 [M][Q_HEADS][128]; attn_out bf16 [M][Q_HEADS x 128]   as k2_attn.cl
//
// **The op chain, rounding point for rounding point the reference's** (row m of q head h has
// the L = pos + m + 1 keys 0..pos+m; tests/kernels/k2_ref.h attention_eager is the host twin,
// line for line):
//   score    s_p = rne(f32(rne(Σ_d q_d·k_pd)) · SCALE)    the bf16 GEMM rounds the dot once,
//            `* scaling` on the bf16 tensor rounds again (fp32 product inside). The dot is ONE
//            fma chain, d ascending: q and k are bf16 values, so every product is exact in fp32
//            and fma == mul + add - the chain cannot be contracted into another result.
//   softmax  torch's fp32 softmax over the FULL row (SoftMaxKernel.cpp _vec_softmax_lastdim at
//            AVX2), then .to(bf16): mx = max s (exact in any order); e_p = exp_torch(s_p - mx)
//            (Sleef_expf8_u10 step for step: torch's Vectorized<float>::exp); Σ e in reduce_all's
//            order - 8 lane accumulators, lane j summing e_j, e_{j+8}, ... ascending, then
//            ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)); inv = 1 / Σ (correctly rounded:
//            -cl-fp32-correctly-rounded-divide-sqrt); p_p = rne(e_p · inv).
//   P·V      o_d = Σ_p p_p·v_pd in fp32: per key block an fma chain ascending (exact products
//            again), the blocks added ascending; rounded once to bf16.
//   gate     out = rne(f32(rne(o)) · f32(rne(softplus(gate))))   k2_attn_reduce's last line.
//
// **Why two passes and not flash's online softmax:** the reference rounds every probability
// to bf16 against the WHOLE row's max and sum before P·V. An online merge rescales partial sums
// by exp(m_old - m_new) - a different fp32 value than the full row's exp(s - mx) - and it can
// only round p after the row's sum exists, which is after every block has run. So: the scores
// go to a row in memory (k2_attn_eager_score), one work-group per row takes the exact max, the
// exps, the sum in torch's order and the rounded p (k2_attn_eager_softmax), and P·V reads the
// rounded p back (k2_attn_eager_pv + k2_attn_eager_reduce). Four launches per layer for flash's
// two; the row costs 4 B x max_len x Q_HEADS x M of scratch (runtime/k2/k2_sizes.h).
//
// **What is bitwise against torch:** the softmax entirely (the host twin reproduces torch's
// probabilities bit for bit on the fixtures; this kernel is the host twin); the rounding points
// of the score and of P·V. NOT bitwise: the two GEMMs' fp32 ACCUMULATION ORDERS (torch's are
// its own, and differ between its prompt pass and its decode steps) - they part only where an
// fp32 dot lies within its order noise of a bf16 boundary. And exp_torch below p's subnormal
// range (s - mx < -87.3: an exp below 2^-126) assumes the device keeps fp32 denormals.
//
// Plain OpenCL C on purpose (no sub-group operations, no block reads): the Mac's OpenCL runs it
// (tools/mac/clrun/k2_run.cc), and nothing about the device's lanes can move a result.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_attn_eager: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef TGT
#error "k2_attn_eager: TGT must be defined (kernels::k2::kAttnTgt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS)
#error "k2_attn_eager: Q_HEADS and KV_HEADS must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "k2_attn_eager: q-heads must be a multiple of kv-heads"
#endif
#if M < 1 || M > 4
#error "k2_attn_eager: M is 1..4"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define PART (HD + 2)
#define WG HD
#define NPR (M * GQA)
#define SSCALE 0.08838834764831845f   /* float(128 ** -0.5): torch casts the scalar to fp32 */
#define SP_BETA 0.6931471805599453f
#define SP_THRESHOLD 20.0f
#define LANES 8                       /* reduce_all's Vectorized<float> width at AVX2 */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
inline float softplus_k2(float x) {
  const float bx = x * SP_BETA;
  return bx > SP_THRESHOLD ? x : log1p(exp(bx)) / SP_BETA;
}

// Sleef_expf8_u10 (sleefsimdsp.c xexpf, AVX2 + FMA) - k2_ref::exp_torch, the same lines.
inline float exp_torch(float d) {
  const int q = convert_int_rte(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
  const float qf = (float)q;
  float s = fma(qf, -0.693145751953125f, d);
  s = fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = fma(u, s, 0.00139304355252534151077271f);
  u = fma(u, s, 0.00833336077630519866943359f);
  u = fma(u, s, 0.0416664853692054748535156f);
  u = fma(u, s, 0.166666671633720397949219f);
  u = fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  u = u * as_float((uint)((q >> 1) + 127) << 23);
  return u * as_float((uint)((q - (q >> 1)) + 127) << 23);
}

// The key block: v2's ppw over the step's LONGEST row (pos + n_act keys), the same for every
// row, so no row has more than TGT blocks (k2_ref::eager_block).
inline uint eager_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}

// Scores: work-item = key (strided over the block), all NPR (row, q head) dots of the kv head
// at once, so a K row is read once.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_score(__global const uint* restrict ctrl,
                                  __global const float* restrict attn_q,
                                  __global const ushort* restrict kv_k,
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
    __global const ushort* restrict kr = kv_k + ((size_t)p * KV_HEADS + j) * HD;
    float a[NPR];
    for (uint pr = 0; pr < NPR; ++pr) a[pr] = 0.0f;
    for (uint d0 = 0; d0 < HD; d0 += 8) {
      const ushort8 k8 = vload8(0, kr + d0);
      const float kf[8] = {bf16f(k8.s0), bf16f(k8.s1), bf16f(k8.s2), bf16f(k8.s3),
                           bf16f(k8.s4), bf16f(k8.s5), bf16f(k8.s6), bf16f(k8.s7)};
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

// The softmax of one row, in place: scores in, bf16-valued probabilities out.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_softmax(__global const uint* restrict ctrl,
                                    __global float* restrict attn_s,
                                    const uint stride) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  if (m >= n_rows(ctrl)) return;
  const uint len = pos + m + 1;
  __global float* restrict row = attn_s + ((size_t)m * Q_HEADS + h) * stride;
  __local float red[WG];

  float mx = -INFINITY;
  for (uint i = lid; i < len; i += WG) mx = fmax(mx, row[i]);
  red[lid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint w = WG / 2; w > 0; w >>= 1) {   // max: exact in any order
    if (lid < w) red[lid] = fmax(red[lid], red[lid + w]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float gmx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = lid; i < len; i += WG) row[i] = exp_torch(row[i] - gmx);
  barrier(CLK_GLOBAL_MEM_FENCE);   // this work-group reads its own e back

  if (lid < LANES) {   // reduce_all's lane chains: lane j sums e_j, e_{j+8}, ... ascending
    float acc = 0.0f;
    for (uint i = lid; i < len; i += LANES) acc += row[i];
    red[lid] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sum = ((red[0] + red[4]) + (red[2] + red[6])) + ((red[1] + red[5]) + (red[3] + red[7]));
  const float inv = 1.0f / sum;
  for (uint i = lid; i < len; i += WG) row[i] = rf(row[i] * inv);
}

// P·V over one key block: work-item d = dim d, every (row, q head) of the kv head; the
// block's sums to attn_part words 2.. (fma chains, keys ascending).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_pv(__global const uint* restrict ctrl,
                               __global const float* restrict attn_s,
                               __global const ushort* restrict kv_v,
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
    const float v = bf16f(kv_v[((size_t)p * KV_HEADS + j) * HD + d]);
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

// Row m's ceil(L / ppw) block sums added ascending, rounded once, then the softplus gate.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_eager_reduce(__global const uint* restrict ctrl,
                                   __global const float* restrict attn_part,
                                   __global const float* restrict attn_gate,
                                   __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  const uint n_act = n_rows(ctrl);
  if (m >= n_act) return;
  const uint s = eager_ppw(pos + n_act);
  const uint nb = (pos + m + 1 + s - 1) / s;
  float o = 0.0f;
  for (uint b = 0; b < nb; ++b) o += attn_part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  const ushort gate_b = rne_bf16(softplus_k2(g));
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * bf16f(gate_b));
}
