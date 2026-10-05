// k2_attn.cl - spec 18b: K2-Horizon's decode attention, spec 10's v2 (src/kernels/attn_v2.cl)
// at head_dim 128 / GQA 4 with K2's softplus output gate. attn_v2.cl is untouched (its
// binaries are Qwen3.8's, Agnes's and Ornith's); this is its structure carried to K2:
//
//   k2_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)    grid (KV_HEADS, TGT), WG 128
//   k2_attn_reduce(ctrl, attn_part, attn_gate, attn_out)   grid (Q_HEADS, M),    WG 128
//     attn_q     fp32 [M][Q_HEADS][128]     k2_attn_prep's RoPE'd q (bf16 values)
//     kv_k/kv_v  bf16 [max_len][KV_HEADS][128]   this layer's caches
//     attn_part  fp32 [Q_HEADS][TGT][M][130]   {mx, sm, acc[128]}
//     attn_gate  fp32 [M][Q_HEADS][128]     f32(rne(gate column)), k2_attn_prep's
//     attn_out   bf16 [M][Q_HEADS x 128]    o_proj's input
//
// **What is v2's, unchanged:** the per-row stride ppw(L) = max(64, roundup64(ceil(L / TGT)))
// over L = pos + m + 1 keys (so a row's blocks depend on its own key count only), the grid
// (KV_HEADS, TGT) with the uniform early-out, waves of 16 positions with one barrier per
// wave publishing the raw dots through double-buffered SLM, the online softmax in fp32 with
// `exp` (EXP2 0: spec 10 §8), the ascending orders of the sums and of the block merge, the
// V 2D block read with the surface height from `last + 1`, and the masking (a position past
// row m's bound scores -INF, weight exactly 0).
//
// **What is K2's:**
//   * HD 128: a work-group is 128 = 8 sub-groups x 16 lanes, work-item d owns dim d of
//     every q head of the kv head; a wave is still 16 positions, so sub-group s owns TWO of
//     them (p0 + s and p0 + s + 8); a K row is ONE sub-group block read of 8 (lane l element
//     t = dim l + 16 t), the score dot 8 fmas per lane then the 16-lane shuffle tree.
//   * GQA 4: NPR = M x 4 (q, row) pairs per wave; qpack 2 KB x M.
//   * the scale 1 / sqrt(128) (not a power of two: the dot's fp32 product is rounded).
//   * the softplus gate (modeling_k2_horizon.py:500-508): the reference's attention output
//     is bf16, its gate is gate_proj's bf16 output through F.softplus(beta = ln 2), a bf16
//     tensor, and their product is a bf16 multiply. torch's softplus (threshold 20):
//         sp = (x·beta > 20) ? x : log1p(exp(x·beta)) / beta          (fp32 inside)
//     so  attn_out = rne(f32(rne(acc / sm)) · f32(rne(sp(f32(gate_b)))))
//     The `x·beta > 20 -> x` branch is load-bearing for large gates (the softplus of 30 is 30
//     to fp32 precision either way, but the branch is what torch evaluates) and
//     tests/kernels/k2_kernels_test.cc drives values on both sides of it.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_attn: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef TGT
#error "k2_attn: TGT must be defined (kernels::k2::kAttnTgt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS)
#error "k2_attn: Q_HEADS and KV_HEADS must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0
#error "k2_attn: q-heads must be a multiple of kv-heads"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define HD 128
#define OUT_N (Q_HEADS * HD)
#define WAVE_P 16
#define SG 16
#define NSG (HD / SG)          /* 8 sub-groups */
#define PPS (WAVE_P / NSG)     /* 2 positions of a wave per sub-group */
#define PER_LANE (HD / SG)     /* 8 elements of a 128-dim dot per lane */
#define PART (HD + 2)
#define WG HD
#define NPR (M * GQA)
#define SSCALE 0.08838834764831845f   /* 1 / sqrt(128) */
#define SP_BETA 0.6931471805599453f   /* F.softplus(beta = math.log(2)), as fp32 */
#define SP_THRESHOLD 20.0f
#if M < 1 || M > 4
#error "k2_attn: M is 1..4"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
// torch.nn.functional.softplus(x, beta, threshold = 20), the CPU kernel's expression.
inline float softplus_k2(float x) {
  const float bx = x * SP_BETA;
  return bx > SP_THRESHOLD ? x : log1p(exp(bx)) / SP_BETA;
}

inline uint v2_ppw(uint len) {
  uint p = (len + TGT - 1) / TGT;
  p = (p + 63u) & ~63u;
  return max(64u, p);
}

// One K row's 8 elements for this lane: dim lane + 16 t, t ascending.
inline void load_k(__global const ushort* restrict kv_k, uint p, uint j, bool ok, ushort* kreg) {
  if (ok) {   // uniform within the sub-group: p is the sub-group's position
    const ushort8 a = intel_sub_group_block_read_us8(kv_k + ((size_t)p * KV_HEADS + j) * HD);
    kreg[0] = a.s0; kreg[1] = a.s1; kreg[2] = a.s2; kreg[3] = a.s3;
    kreg[4] = a.s4; kreg[5] = a.s5; kreg[6] = a.s6; kreg[7] = a.s7;
  } else {
    for (uint t = 0; t < PER_LANE; ++t) kreg[t] = (ushort)0;
  }
}

// The wave's 16 V values of dim `lid` (sub-group sgid holds dims 16 sgid .. +15), position
// p0 + s at register s; past `last` 0. The surface is [last + 1 rows][HD dims] of kv-head j.
inline void load_v(__global const ushort* restrict kv_v, uint p0, uint j, uint sgid, uint last,
                   ushort* vreg) {
  ushort vb[WAVE_P];
  intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)(kv_v + (size_t)j * HD), HD * 2,
                                             last + 1, KV_HEADS * HD * 2,
                                             (int2)((int)(SG * sgid), (int)p0), vb);
  for (uint s = 0; s < WAVE_P; ++s) vreg[s] = p0 + s <= last ? vb[s] : (ushort)0;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void k2_attn_decode(__global const uint* restrict ctrl,
                             __global const float* restrict attn_q,
                             __global const ushort* restrict kv_k,
                             __global const ushort* restrict kv_v,
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

  __local float qpack[NPR * HD];              // [m][qhl][128]
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

    ushort kreg[PPS][PER_LANE], vreg[WAVE_P];
    for (uint pp = 0; pp < PPS; ++pp) {
      const uint p = bstart + sgid + NSG * pp;
      load_k(kv_k, p, j, p <= last, kreg[pp]);
    }
    load_v(kv_v, bstart, j, sgid, last, vreg);
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
          for (uint t = 0; t < PER_LANE; ++t) a = fma(qv[t], bf16f(kreg[pp][t]), a);
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
        const float mine = p0 + lane <= bound ? sc_x[buf][pr][lane] * SSCALE : -INFINITY;
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
          tsum = fma(ws, bf16f(vreg[s2]), tsum);
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
          load_k(kv_k, p, j, p <= last, kreg[pp]);
        }
        load_v(kv_v, p0 + WAVE_P, j, sgid, last, vreg);
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

// k2_attn_reduce - grid (Q_HEADS, M), work-group 128, work-item d = dim d: row m's nb =
// (pos + m) / ppw(pos + m + 1) + 1 <= TGT partials merged ascending, then the softplus gate.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_attn_reduce(__global const uint* restrict ctrl,
                             __global const float* restrict attn_part,
                             __global const float* restrict attn_gate,
                             __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint d = get_local_id(0);
  __local float hmx[TGT], hsm[TGT];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;
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
  const float o = acc / sm;
  const float g = attn_gate[((size_t)m * Q_HEADS + h) * HD + d];
  const ushort gate_b = rne_bf16(softplus_k2(g));
  attn_out[(size_t)m * OUT_N + (size_t)h * HD + d] = rne_bf16(bf16f(rne_bf16(o)) * bf16f(gate_b));
}
