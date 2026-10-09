// q4_qsa_attn.cl - spec 21c: Qwen3.8-Flash-Next's QSA decode attention over a SELECTED position list (spec 21
// §4.2, decision 10: one kernel at every depth - below 2052 visible positions the list is the identity), spec 10
// v2's structure (attn_v2.cl / kol_attn.cl: waves of 16 positions, sub-group block reads of K, a fixed-order
// online softmax in fp32, the slices merged ascending) carried to a gathered list. attn_v2.cl and kol_attn.cl are
// untouched.
//
//   q4_qsa_attn(ctrl, attn_q, kv_k, kv_v, list, part)          grid (KV_HEADS, TGT, M), WG 256
//   q4_qsa_reduce(ctrl, part, attn_gate, list, attn_out)       grid (Q_HEADS, M),       WG 256
//     attn_q     fp32 [M][24][256]       attn_prep's q (normed, partial RoPE at p; fp32)
//     attn_gate  fp32 [M][24][256]       attn_prep's gate columns (bf16 values)
//     kv_k/kv_v  bf16 [max_len][2][256]  this layer's cache, rows by position
//     list       u32 [M][LIST_ROW]       q4_qsa_select's row: positions ascending [0, count), count at COUNT_W
//     part       fp32 [24][TGT][M][258]  {mx, sm, acc[256]} per (q head, slice, row)
//     attn_out   bf16 [M][6144]          o_proj's input
//
// **The slices:** row m's list of c positions is cut into contiguous slices of ppw = max(16, roundup16(ceil(c /
// TGT))) list entries (whole 16-position waves), slice blk = entries [blk ppw, min(c, (blk + 1) ppw)); a slice past
// c exits whole. A work-group owns one (kv head j, slice, row): work-item d = dim d (16 sub-groups x 16 lanes), the
// 12 q heads of kv head j (GQA 12) in SLM. A wave: sub-group sg owns list entry w0 + sg - its position read from
// the list, its K row two sub-group block reads (dims lane + 16 t, t = 0..15) - and for each q head the 16-fma dot
// (t ascending) then the 16-lane shuffle tree; the 12 x 16 raw dots go through SLM; then EVERY sub-group runs the
// same online softmax over the wave (identical inputs, identical bits): scores x 1/16, nmx = max(mx, wave max),
// resc = exp(mx - nmx), weights exp(s - nmx), sm = fma(sm, resc, sum ascending), acc = fma(acc, resc, sum_s w_s v_s
// ascending) - with the V value of dim d read per position (a gathered row: one coalesced 512 B row per position
// across the work-group). Positions are consumed in list order = ascending (the reference's masked-row order).
// **q4_qsa_reduce:** the row's live slices merged ascending (kol_attn_reduce's merge), o = acc / sm, then the
// reference's tail: out = rne(f32(rne(o)) x rne(sigmoid(gate))) (`attn_output * torch.sigmoid(gate)` in bf16,
// M:889-892; the sigmoid through exp_torch, Sleef's expf u10).
//
// Flash keeps the probabilities in fp32 (one rounding at the end); the reference rounds scores and
// probabilities to bf16 (q4_qsa_attn_eager.cl behind B70_Q4_ATTN=eager). Card only: sub-group shuffles,
// reductions and block reads.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_qsa_attn: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TGT) || !defined(LIST_ROW) || !defined(COUNT_W)
#error "q4_qsa_attn: TGT, LIST_ROW and COUNT_W must be defined (kernels::qwen4exp)"
#endif
#if M < 1 || M > 4
#error "q4_qsa_attn: M is 1..4"
#endif
#define Q_HEADS 24
#define KV_HEADS 2
#define GQA 12
#define HD 256
#define OUT_N (Q_HEADS * HD)
#define WAVE_P 16
#define SG 16
#define NSG (HD / SG)          /* 16 sub-groups: one position of the wave each */
#define PER_LANE (HD / SG)     /* 16 elements of a 256-dim dot per lane */
#define PART (HD + 2)
#define WG HD
#define SSCALE 0.0625f         /* 1 / sqrt(256) */

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
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}
// List entries per slice: whole waves, at most TGT slices for any count <= LIST_ROW.
inline uint slice_len(uint c) {
  uint p = (c + TGT - 1) / TGT;
  p = (p + WAVE_P - 1) / WAVE_P * WAVE_P;
  return max((uint)WAVE_P, p);
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void q4_qsa_attn(__global const uint* restrict ctrl, __global const float* restrict attn_q,
                          __global const ushort* restrict kv_k, __global const ushort* restrict kv_v,
                          __global const uint* restrict list, __global float* restrict part) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint m = get_group_id(2);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;
  if (m >= n_rows(ctrl)) return;   // uniform
  __global const uint* restrict row = list + (size_t)m * LIST_ROW;
  const uint c = row[COUNT_W];
  const uint ppw = slice_len(c);
  const uint start = blk * ppw;
  if (start >= c) return;          // uniform: a slice past the row's list
  const uint end = min(start + ppw, c);
  const uint nwaves = (end - start + WAVE_P - 1) / WAVE_P;

  __local float qpack[GQA * HD];          // the 12 q heads of kv head j
  __local float sc[GQA][WAVE_P];          // the wave's raw dots
  for (uint i = lid; i < GQA * HD; i += WG) qpack[i] = attn_q[((size_t)m * Q_HEADS + j * GQA) * HD + i];
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx[GQA], sm[GQA], acc[GQA];
  for (uint q = 0; q < GQA; ++q) { mx[q] = -INFINITY; sm[q] = 0.0f; acc[q] = 0.0f; }

  for (uint w = 0; w < nwaves; ++w) {
    const uint w0 = start + w * WAVE_P;
    // sub-group sgid: list entry w0 + sgid (uniform within the sub-group)
    const uint e = w0 + sgid;
    const bool ok = e < end;
    const uint pk = ok ? row[e] : 0u;
    ushort kreg[PER_LANE];
    if (ok) {
      __global const ushort* kr = kv_k + ((size_t)pk * KV_HEADS + j) * HD;
      const ushort8 a0 = intel_sub_group_block_read_us8(kr);
      const ushort8 a1 = intel_sub_group_block_read_us8(kr + 128);
      kreg[0] = a0.s0; kreg[1] = a0.s1; kreg[2] = a0.s2; kreg[3] = a0.s3;
      kreg[4] = a0.s4; kreg[5] = a0.s5; kreg[6] = a0.s6; kreg[7] = a0.s7;
      kreg[8] = a1.s0; kreg[9] = a1.s1; kreg[10] = a1.s2; kreg[11] = a1.s3;
      kreg[12] = a1.s4; kreg[13] = a1.s5; kreg[14] = a1.s6; kreg[15] = a1.s7;
    } else {
      for (uint t = 0; t < PER_LANE; ++t) kreg[t] = (ushort)0;
    }
    for (uint qh = 0; qh < GQA; ++qh) {
      const uint8 qa = intel_sub_group_block_read8((__local const uint*)&qpack[qh * HD]);
      const uint8 qb = intel_sub_group_block_read8((__local const uint*)&qpack[qh * HD + 128]);
      float a = 0.0f;
      a = fma(as_float(qa.s0), bf16f(kreg[0]), a);  a = fma(as_float(qa.s1), bf16f(kreg[1]), a);
      a = fma(as_float(qa.s2), bf16f(kreg[2]), a);  a = fma(as_float(qa.s3), bf16f(kreg[3]), a);
      a = fma(as_float(qa.s4), bf16f(kreg[4]), a);  a = fma(as_float(qa.s5), bf16f(kreg[5]), a);
      a = fma(as_float(qa.s6), bf16f(kreg[6]), a);  a = fma(as_float(qa.s7), bf16f(kreg[7]), a);
      a = fma(as_float(qb.s0), bf16f(kreg[8]), a);  a = fma(as_float(qb.s1), bf16f(kreg[9]), a);
      a = fma(as_float(qb.s2), bf16f(kreg[10]), a); a = fma(as_float(qb.s3), bf16f(kreg[11]), a);
      a = fma(as_float(qb.s4), bf16f(kreg[12]), a); a = fma(as_float(qb.s5), bf16f(kreg[13]), a);
      a = fma(as_float(qb.s6), bf16f(kreg[14]), a); a = fma(as_float(qb.s7), bf16f(kreg[15]), a);
      for (uint stride = SG / 2; stride > 0; stride >>= 1) a += intel_sub_group_shuffle_down(a, a, stride);
      if (lane == 0) sc[qh][sgid] = a;
    }
    // the wave's V values of dim lid at its 16 positions (zero past the slice)
    float vreg[WAVE_P];
    for (uint s = 0; s < WAVE_P; ++s) {
      const uint es = w0 + s;
      vreg[s] = es < end ? bf16f(kv_v[((size_t)row[es] * KV_HEADS + j) * HD + lid]) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint qh = 0; qh < GQA; ++qh) {
      const float mine = (w0 + lane < end) ? sc[qh][lane] * SSCALE : -INFINITY;
      const float nmx = fmax(mx[qh], sub_group_reduce_max(mine));
      float resc, wl;
      if (nmx > -INFINITY) {
        resc = exp(mx[qh] - nmx);
        wl = exp(mine - nmx);
      } else {
        resc = 1.0f;
        wl = 0.0f;
      }
      float ssum = 0.0f, tsum = 0.0f;
      for (uint s = 0; s < WAVE_P; ++s) {   // ascending s
        const float ws = intel_sub_group_shuffle(wl, s);
        ssum += ws;
        tsum = fma(ws, vreg[s], tsum);
      }
      if (nmx > -INFINITY) {
        sm[qh] = fma(sm[qh], resc, ssum);
        mx[qh] = nmx;
      }
      acc[qh] = fma(acc[qh], resc, tsum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // sc is rewritten by the next wave
  }
  for (uint qh = 0; qh < GQA; ++qh) {
    __global float* restrict out = part + (((size_t)(j * GQA + qh) * TGT + blk) * M + m) * PART;
    out[2 + lid] = acc[qh];
    if (lid == 0) { out[0] = mx[qh]; out[1] = sm[qh]; }
  }
}

// q4_qsa_reduce - grid (Q_HEADS, M), work-group 256, work-item d = dim d: the row's live slices merged ascending,
// o = acc / sm, out = rne(f32(rne(o)) x rne(sigmoid(gate))).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_qsa_reduce(__global const uint* restrict ctrl, __global const float* restrict part,
                            __global const float* restrict attn_gate, __global const uint* restrict list,
                            __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float hmx[TGT], hsm[TGT];
  const uint c = list[(size_t)m * LIST_ROW + COUNT_W];
  const uint ppw = slice_len(c);
  uint nb = (c + ppw - 1) / ppw;
  if (nb > TGT) nb = TGT;
  for (uint b = d; b < nb; b += WG) {
    __global const float* restrict p = part + (((size_t)h * TGT + b) * M + m) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    if (!(bmx > -INFINITY)) continue;
    const float bacc = part[(((size_t)h * TGT + b) * M + m) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = exp(mx - nmx);
    const float bs = exp(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  const float sg = rf(1.0f / (1.0f + exp_torch(-g)));
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(rf(acc / sm) * sg);
}
