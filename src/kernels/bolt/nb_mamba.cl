// nb_mamba.cl - spec 25c: Nemotron 3.5 Bolt's Mamba-2 mixer between its two GEMVs (spec 25 §2.2, reading 1 - the
// training kernels' semantics, Decision 1). Three families, each compiled only when its defines are given:
//
//   nb_conv(ctrl, slices, cw, cb, ring, xc)                              CONV_S  grid (10240 / 256, 1), WG 256
//   nb_ssm(ctrl, slices, xc, A, D, dt_bias, state, y)                    SSM_S   grid (128 x 4, 1), WG 256
//   nb_ssm (SSM_CONV: the arm `conv`) (ctrl, slices, cw, cb, ring, A, D, dt_bias, state, y)
//   nb_gnorm(slices, y, w, out)                                          GN_S    grid (8, M), WG 256
//
// in_proj's ONE GEMV (N 18560) leaves its S split-K slices [S][M][18560]: z [0, 8192), xBC [8192, 18432), dt [18432,
// 18560); every consumer below sums its columns from 0.0f in ascending slice order and rounds once (the linear's bf16
// output). tests/kernels/bolt_ref.h (conv_step, ssm_step, gnorm) repeats every chain - EDIT THE TWO TOGETHER.
//
// **nb_conv** (causal_conv1d_update): one work-item a channel c of xBC, the rows m = 0 .. n_active - 1 in order:
//   raw = rne(Σ_s slices[s][m][8192 + c]);  ring[(pos + m) % 16][c] = raw         the ring holds RAW xBC (R:487-490)
//   acc = cb[c]; acc += cw[c][0] x[p-3]; ... acc += cw[c][3] raw                   the bias seeds the accumulator, taps
//                                                                                 OLDEST first (25a), fp32
//   xc[m][c] = rne(silu(acc)), silu(v) = v / (1 + exp(-v))                        ONE rounding
// with x[p-3..p-1] from the ring for the launch's first rows, from the work-item's registers after (gdn_step.cl's
// ownership pattern), and 0 below position 0 - the ring is never cleared. Rows of one launch write slots pos .. pos + M - 1
// and read pos - 3 .. pos + M - 2 (mod 16): disjoint for M <= 13, so no work-item reads a slot another one writes.
//
// **nb_ssm** (selective_state_update, mamba_ssm's order - 25a): work-group (h, q) owns head h's dims p = 16 q .. 16 q + 15;
// work-item (p_local = lid / 16, lane = lid % 16) holds S[h][p][n] for n = lane + 16 j (j < 8) in registers (coalesced:
// the state is [128][64][128], n fastest), read once and written back once after the launch's rows. Per row m in order:
//   dt  = softplus(f32(rne(Σ_s slices[s][m][18432 + h])) + dt_bias[h])          threshold 20, NO clamp (Decision 1)
//   dA  = exp(A[h] · dt)                                                         A = -exp(A_log): the loader's (fp32)
//   S   = S · dA + (B[g][n] · dt) · x[h][p]                                      products and adds rounded apart
//   y   = rne(tree_n(S · C[g][n]) + x · D[h])                                    g = h / 16; the HALVING tree over 128:
//         in-lane j vs j + 4, + 2, + 1 (n offsets 64, 32, 16), then 8 .. 1 across the 16 lanes (SLM)
// The bf16-state arm (SSM_B16, Decision 2): S is stored bf16, read widened, and rounded back to bf16 after EVERY row's
// update (y from the fp32 update) - a step at a time as one M-row launch.
// SPEC_SLOTS > 0 (spec 26c's verify lists; 0 = undefined here: one live slot, in place): gdn_step.cl's slot form (spec
// 19c) - the state lives in SPEC_SLOTS slots (slot 0 `state`, slot s > 0 `state_spec + (s - 1) x SPEC_SLOT_STRIDE`
// floats), read from the live slot ctrl[CTRL_LIVE] and, after row m, written to slot (live + m) % SPEC_SLOTS; committing j
// accepted rows is the host setting live. Safe without a cross-work-group barrier: every work-item reads its own cells of
// the live slot before any write and writes exactly those offsets in other slots.
// **The arm `conv`** (SSM_CONV, spec 25 §6.1): the conv computed inside nb_ssm - work-group (h, q) convolves its 16 x
// channels and its group's 256 B / C channels (recomputed by the 64 work-groups of the group) with nb_conv's arithmetic
// into SLM; the ring's x channels are written by their own work-group, the group's B / C channels by work-group (16 g, 0)
// only. The same values as nb_conv + nb_ssm, bitwise, one launch fewer.
//
// **nb_gnorm** (rmsnorm_fn, norm_before_gate false, groups of 1024): work-group (g, m), work-item l holds i = l + 256 j
// (j < 4) of group g: v = f32(y) · silu_gate(f32(rne(Σ_s z slices))), silu_gate(z) = z · (1 / (1 + exp(-z))); ms =
// tree(v²) / 1024 (the halving tree: in-lane j vs j + 2, + 1, then 128 .. 1 in SLM), rstd = 1 / sqrt(ms + 1e-5), out =
// rne((v · rstd) · w) - one work-group a group, no recompute.
//
// Plain OpenCL C (every cross-lane step through SLM), so the file also runs on the Mac's GPU (bolt_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#include "nb_common.h"

#define NB_INPROJ_N 18560
#define NB_XBC_OFF 8192
#define NB_DT_OFF 18432
#define NB_XBC 10240
#define NB_INNER 8192
#define NB_HEADS 128
#define NB_P 64
#define NB_N 128
#define NB_G 8
#define NB_B_OFF 8192            /* within xc: x [0, 8192), B [8192, 9216), C [9216, 10240) */
#define NB_C_OFF 9216
#define NB_RING 16
#define NB_TAPS 4

inline float nb_col(__global const float* restrict slices, uint S, uint m, uint col) {
  float v = 0.0f;
  for (uint s = 0; s < S; ++s) v += slices[((size_t)s * M + m) * NB_INPROJ_N + col];
  return v;
}

// One channel's conv over the launch's rows (nb_conv's and the arm's arithmetic): hist[0..2] the raw x at p-3 .. p-1.
inline float nb_conv_acc(__global const float* restrict cw, __global const float* restrict cb, uint c, const float* hist,
                         float raw) {
  float acc = cb[c];
  acc = acc + cw[(size_t)c * NB_TAPS + 0] * hist[0];
  acc = acc + cw[(size_t)c * NB_TAPS + 1] * hist[1];
  acc = acc + cw[(size_t)c * NB_TAPS + 2] * hist[2];
  acc = acc + cw[(size_t)c * NB_TAPS + 3] * raw;
  return acc;
}
inline void nb_conv_hist(__global const ushort* restrict ring, uint c, uint pos, float* hist) {
  for (uint j = 0; j < NB_TAPS - 1; ++j) {
    const int p = (int)pos - (int)(NB_TAPS - 1) + (int)j;
    hist[j] = p < 0 ? 0.0f : bf16f(ring[(size_t)((uint)p % NB_RING) * NB_XBC + c]);
  }
}

// ---------------------------------------------------------------------------------------------------------------
#ifdef CONV_S
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "nb_conv: CTRL_POS / CTRL_NACT must be defined"
#endif
#if M > NB_RING - NB_TAPS + 1
#error "nb_conv: one launch's rows must not reach a slot another row reads (M <= 13)"
#endif
#define WG_CONV 256
__attribute__((reqd_work_group_size(WG_CONV, 1, 1)))
__kernel void nb_conv(__global const uint* restrict ctrl, __global const float* restrict slices,
                      __global const float* restrict cw, __global const float* restrict cb,
                      __global ushort* restrict ring, __global ushort* restrict xc) {
  const uint c = get_group_id(0) * WG_CONV + get_local_id(0);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (c >= NB_XBC || n_act == 0) return;
  float hist[NB_TAPS - 1];
  nb_conv_hist(ring, c, pos, hist);
  for (uint m = 0; m < n_act; ++m) {
    const ushort raw = rne_bf16(nb_col(slices, CONV_S, m, NB_XBC_OFF + c));
    ring[(size_t)((pos + m) % NB_RING) * NB_XBC + c] = raw;
    const float acc = nb_conv_acc(cw, cb, c, hist, bf16f(raw));
    xc[(size_t)m * NB_XBC + c] = rne_bf16(nb_silu_conv(acc));
    hist[0] = hist[1];
    hist[1] = hist[2];
    hist[2] = bf16f(raw);
  }
}
#endif  // CONV_S

// ---------------------------------------------------------------------------------------------------------------
#ifdef SSM_S
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "nb_ssm: CTRL_POS / CTRL_NACT must be defined"
#endif
#if !defined(SSM_B16) || !defined(SSM_CONV)
#error "nb_ssm: SSM_B16 (0 fp32 state, 1 bf16) and SSM_CONV (0, 1 the arm conv) must be defined"
#endif
#ifndef SPEC_SLOTS
#define SPEC_SLOTS 0
#endif
#if SPEC_SLOTS > 0
#if !defined(CTRL_LIVE) || !defined(SPEC_SLOT_STRIDE)
#error "nb_ssm SPEC_SLOTS: CTRL_LIVE and SPEC_SLOT_STRIDE (elements per slot) must be defined"
#endif
#if M > SPEC_SLOTS
#error "nb_ssm SPEC_SLOTS: M rows need M slots"
#endif
#endif
#if SSM_CONV && M > NB_RING - NB_TAPS + 1
#error "nb_ssm _C: one launch's rows must not reach a slot another row reads (M <= 13)"
#endif
#define WG_SSM 256
#define SSM_PW 16                /* dims p a work-group */
#define SSM_PER 8                /* state entries a work-item: n = lane + 16 j */
#if SSM_B16
#define STATE_T ushort
#define STATE_LD(v) bf16f(v)
#define STATE_ST(f) rne_bf16(f)
#else
#define STATE_T float
#define STATE_LD(v) (v)
#define STATE_ST(f) (f)
#endif

#if SPEC_SLOTS > 0
inline __global STATE_T* nb_slot(__global STATE_T* s0, __global STATE_T* spec, uint s) {
  return s == 0 ? s0 : spec + (size_t)(s - 1) * SPEC_SLOT_STRIDE;
}
#endif

__attribute__((reqd_work_group_size(WG_SSM, 1, 1)))
__kernel void nb_ssm(__global const uint* restrict ctrl, __global const float* restrict slices,
#if SSM_CONV
                     __global const float* restrict cw, __global const float* restrict cb, __global ushort* restrict ring,
#else
                     __global const ushort* restrict xc,
#endif
                     __global const float* restrict A, __global const float* restrict D,
                     __global const float* restrict dt_bias, __global STATE_T* restrict state,
                     __global ushort* restrict y
#if SPEC_SLOTS > 0
                     , __global STATE_T* restrict state_spec
#endif
                     ) {
  const uint h = get_group_id(0) / (NB_P / SSM_PW);
  const uint q = get_group_id(0) % (NB_P / SSM_PW);
  const uint lid = get_local_id(0);
  const uint pl = lid / 16, lane = lid % 16;
  const uint p = q * SSM_PW + pl;
  const uint g = h / (NB_HEADS / NB_G);
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (n_act == 0) return;   // uniform
  __local float red[SSM_PW][16];
#if SSM_CONV
  __local float xs[SSM_PW], bs[NB_N], cs[NB_N];   // this row's conv outputs (bf16 values): x of my dims, B / C of g
  // Work-item lid convolves B / C channel lid of group g (lid < 128: B, else C) and, lid < 16, x channel h x 64 + p'.
  const uint cbc = lid < NB_N ? NB_INNER + g * NB_N + lid : NB_INNER + NB_G * NB_N + g * NB_N + (lid - NB_N);
  const uint cx = h * NB_P + q * SSM_PW + lid;
  float hbc[NB_TAPS - 1], hx[NB_TAPS - 1];
  nb_conv_hist(ring, cbc, pos, hbc);
  if (lid < SSM_PW) nb_conv_hist(ring, cx, pos, hx);
  const bool own_bc = (h % (NB_HEADS / NB_G)) == 0 && q == 0;
#endif
  const size_t row = ((size_t)h * NB_P + p) * NB_N;
#if SPEC_SLOTS > 0
  const uint live = ctrl[CTRL_LIVE] % SPEC_SLOTS;
  const __global STATE_T* src = nb_slot(state, state_spec, live);   // not restrict: the slots alias `state`
#else
  const __global STATE_T* src = state;   // not restrict: read here, `state` written at the end
#endif
  float st[SSM_PER];
  for (uint j = 0; j < SSM_PER; ++j) st[j] = STATE_LD(src[row + lane + 16 * j]);
  const float Ah = A[h], Dh = D[h], bh = dt_bias[h];

  for (uint m = 0; m < n_act; ++m) {
#if SSM_CONV
    {   // the conv of this row's channels (nb_conv's arithmetic) into SLM; the ring written by each channel's owner
      const ushort rbc = rne_bf16(nb_col(slices, SSM_S, m, NB_XBC_OFF + cbc));
      const float abc = nb_conv_acc(cw, cb, cbc, hbc, bf16f(rbc));
      if (own_bc) ring[(size_t)((pos + m) % NB_RING) * NB_XBC + cbc] = rbc;
      if (lid < NB_N) bs[lid] = rf(nb_silu_conv(abc));
      else cs[lid - NB_N] = rf(nb_silu_conv(abc));
      hbc[0] = hbc[1];
      hbc[1] = hbc[2];
      hbc[2] = bf16f(rbc);
      if (lid < SSM_PW) {
        const ushort rx = rne_bf16(nb_col(slices, SSM_S, m, NB_XBC_OFF + cx));
        const float ax = nb_conv_acc(cw, cb, cx, hx, bf16f(rx));
        ring[(size_t)((pos + m) % NB_RING) * NB_XBC + cx] = rx;
        xs[lid] = rf(nb_silu_conv(ax));
        hx[0] = hx[1];
        hx[1] = hx[2];
        hx[2] = bf16f(rx);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float x = xs[pl];
#else
    const __global ushort* restrict xr = xc + (size_t)m * NB_XBC;
    const float x = bf16f(xr[(size_t)h * NB_P + p]);
#endif
    const float dt = nb_softplus(rf(nb_col(slices, SSM_S, m, NB_DT_OFF + h)) + bh);
    const float dA = nb_exp(Ah * dt);
    float prod[SSM_PER];
    for (uint j = 0; j < SSM_PER; ++j) {
      const uint n = lane + 16 * j;
#if SSM_CONV
      const float Bn = bs[n], Cn = cs[n];
#else
      const float Bn = bf16f(xr[NB_B_OFF + g * NB_N + n]), Cn = bf16f(xr[NB_C_OFF + g * NB_N + n]);
#endif
      const float dB = Bn * dt;
      st[j] = st[j] * dA + dB * x;
      prod[j] = st[j] * Cn;
    }
    for (uint hv = SSM_PER / 2; hv > 0; hv >>= 1)
      for (uint j = 0; j < hv; ++j) prod[j] = prod[j] + prod[j + hv];
    red[pl][lane] = prod[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 8; stride > 0; stride >>= 1) {
      if (lane < stride) red[pl][lane] = red[pl][lane] + red[pl][lane + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) y[(size_t)m * NB_INNER + (size_t)h * NB_P + p] = rne_bf16(red[pl][0] + x * Dh);
#if SSM_B16
    for (uint j = 0; j < SSM_PER; ++j) st[j] = rf(st[j]);
#endif
#if SPEC_SLOTS > 0
    {   // the state after row m, into its slot (header, SPEC_SLOTS)
      __global STATE_T* restrict dst = nb_slot(state, state_spec, (live + m) % SPEC_SLOTS);
      for (uint j = 0; j < SSM_PER; ++j) dst[row + lane + 16 * j] = STATE_ST(st[j]);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);   // the next row reuses red (and the arm's xs / bs / cs)
  }
#if SPEC_SLOTS == 0
  for (uint j = 0; j < SSM_PER; ++j) state[row + lane + 16 * j] = STATE_ST(st[j]);
#endif
}
#endif  // SSM_S

// ---------------------------------------------------------------------------------------------------------------
#ifdef GN_S
#if !defined(GN_EPS)
#error "nb_gnorm: GN_EPS must be defined"
#endif
#define GN_GROUP 1024
#define WG_GN 256
#define GN_PER (GN_GROUP / WG_GN)   /* 4 */
__attribute__((reqd_work_group_size(WG_GN, 1, 1)))
__kernel void nb_gnorm(__global const float* restrict slices, __global const ushort* restrict y,
                       __global const float* restrict w, __global ushort* restrict out) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float red[WG_GN];
  float v[GN_PER], sq[GN_PER];
  for (uint j = 0; j < GN_PER; ++j) {
    const uint c = g * GN_GROUP + lid + j * WG_GN;
    const float z = rf(nb_col(slices, GN_S, m, c));
    v[j] = bf16f(y[(size_t)m * NB_INNER + c]) * nb_silu_gate(z);
    sq[j] = v[j] * v[j];
  }
  for (uint hv = GN_PER / 2; hv > 0; hv >>= 1)
    for (uint j = 0; j < hv; ++j) sq[j] = sq[j] + sq[j + hv];
  red[lid] = sq[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_GN / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] = red[lid] + red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)GN_GROUP + GN_EPS);
  for (uint j = 0; j < GN_PER; ++j) {
    const uint c = g * GN_GROUP + lid + j * WG_GN;
    out[(size_t)m * NB_INNER + c] = rne_bf16((v[j] * rstd) * w[c]);
  }
}
#endif  // GN_S
