// q4_qsa_attn_eager.cl - spec 21c: Qwen3.8-Flash-Next's QSA decode attention as the reference computes it
// (eager_attention_forward + the gate, M:786-808 / M:889-892), over q4_qsa_select's list - the engine binds it
// for q4_qsa_attn + q4_qsa_reduce under B70_Q4_ATTN=eager (spec 18 §10.1's switch). One launch:
//
//   q4_qsa_attn_eager(ctrl, attn_q, attn_gate, kv_k, kv_v, list, attn_out)    grid (Q_HEADS, M), WG 256
//
// One work-group per (q head h, row m), kv head h / 12, over the row's c listed positions (ascending):
//   q_b   = rne(attn_q)                     (attn_prep keeps the RoPE'd q in fp32; the reference's q is bf16)
//   s_i   = rne(rne(sum_d q_b[d] k_i[d]) x 1/16)    one fma chain, d ascending (bf16 x bf16 products exact)
//   p     = torch's fp32 softmax over the list, rounded to bf16: mx exact; e_i = exp_torch(s_i - mx); the sum in
//           reduce_all's order (8 lane sums, lane l: e_l, e_{l+8}, ... ascending, then ((l0 + l4) + (l2 + l6)) +
//           ((l1 + l5) + (l3 + l7))); p_i = rne(e_i x (1 / sum))   (k2_ref::eager_softmax, k2_attn_eager.cl's)
//   o_d   = sum_i p_i v_i[d]                one fma chain, i ascending, rounded once
//   out   = rne(f32(rne(o)) x rne(sigmoid(gate)))
// tests/kernels/qwen4exp_ref.h qsa_attn_eager is this chain; kernel against it is bitwise. Against torch it parts
// only on the two GEMMs' fp32 accumulation orders (its own) and the softmax's lane assignment (torch's runs over the
// whole masked row, this over the list).
//
// Plain OpenCL C (no sub-group functions): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_qsa_attn_eager: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(LIST_ROW) || !defined(COUNT_W) || !defined(LISTMAX)
#error "q4_qsa_attn_eager: LIST_ROW, COUNT_W and LISTMAX must be defined (kernels::qwen4exp)"
#endif
#define Q_HEADS 24
#define KV_HEADS 2
#define GQA 12
#define HD 256
#define OUT_N (Q_HEADS * HD)
#define WG 256

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float rf(float f) { return bf16f(rne_bf16(f)); }
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

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_qsa_attn_eager(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                                __global const float* restrict attn_gate, __global const ushort* restrict kv_k,
                                __global const ushort* restrict kv_v, __global const uint* restrict list,
                                __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform
  const uint j = h / GQA;
  __global const uint* restrict row = list + (size_t)m * LIST_ROW;
  const uint c = row[COUNT_W];
  __local float qb[HD];
  __local float sp[LISTMAX];   // the scores, then exp(s - mx), then the bf16 probabilities
  __local float red[WG];
  __local float lanes[8];
  qb[lid] = rf(attn_q[((size_t)m * Q_HEADS + h) * HD + lid]);
  barrier(CLK_LOCAL_MEM_FENCE);
  // 1. the scores; each lane's max
  float lmx = -INFINITY;
  for (uint i = lid; i < c; i += WG) {
    __global const ushort* restrict k = kv_k + ((size_t)row[i] * KV_HEADS + j) * HD;
    float a = 0.0f;
    for (uint d = 0; d < HD; ++d) a = fma(qb[d], bf16f(k[d]), a);
    const float s = rf(rf(a) * 0.0625f);
    sp[i] = s;
    lmx = fmax(lmx, s);
  }
  red[lid] = lmx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {   // a max: exact in any order
    if (lid < stride) red[lid] = fmax(red[lid], red[lid + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mx = red[0];
  // 2. torch's softmax: the exps, the 8 lane sums (lane l: entries l, l + 8, ... ascending), the tree, p = rne(e / sum)
  for (uint i = lid; i < c; i += WG) sp[i] = exp_torch(sp[i] - mx);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8) {
    float a = 0.0f;
    for (uint i = lid; i < c; i += 8) a += sp[i];
    lanes[lid] = a;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sum = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
  const float inv = 1.0f / sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < c; i += WG) sp[i] = rf(sp[i] * inv);
  barrier(CLK_LOCAL_MEM_FENCE);
  // 3. P.V for dim lid, the gate
  float o = 0.0f;
  for (uint i = 0; i < c; ++i) o = fma(sp[i], bf16f(kv_v[((size_t)row[i] * KV_HEADS + j) * HD + lid]), o);
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + lid];
  const float sg = rf(1.0f / (1.0f + exp_torch(-g)));
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + lid] = rne_bf16(rf(o) * sg);
}
