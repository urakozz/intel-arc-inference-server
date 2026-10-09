// q4_pf_ple.cl - spec 21d: Qwen3.8-Flash-Next's per-layer n-gram embedding block (spec 21 §2.4) over a PREFILL chunk
// - 21c's q4_ple_block (q4_ple.cl, M = 1 only: a row's conv reads the ring rows its earlier launches wrote) cut into
// three launches, so a row's conv never reads a ring slot this chunk overwrites (plan 21d Review Focus 4). One binary
// (kernels::qwen4exp::pf_ple_variant), three entry points:
//
//   q4_pf_ple_gate(ctrl, kv, H, pw, gated, gn)            grid (4 streams, C),       WG 256
//   q4_pf_ple_conv(ctrl, gated, gn, ring, pw, H)          grid (10240 / 256, C),     WG 256
//   q4_pf_ple_ring(ctrl, gn, ids, ring, id_ring)          grid (10240 / 256, 16),    WG 256
//
// **q4_pf_ple_gate**, one work-group per (stream s, row m): q4_ple_block's chain, statement for statement, up to the
// gated value and its conv norm - key = rne(kv[s x 2560 + k]) -> its stream's (1 + w) norm -> kn; qn = norm_query(H)
// (the stream); sb = rne(sum_k rne(kn x qn)) (lane i: k = i + 256 j ascending, a plain add; the tree); s2 = rne(sb /
// sqrt(2560)); gate = sign(s2) x rne(sqrt(max(|s2|, bf16(1e-6)))); sg = rne(sigmoid(gate)); gated = rne(sg x
// rne(kv[10240 + k])); gn = norm_conv(gated) (the stream) -> gated, gn bf16 [C][10240]. H is read, not written.
// **q4_pf_ple_conv**, one work-item per (channel c, row m), p = pos + m: x_0 .. x_2 = gn of positions p - 9, p - 6,
// p - 3 - THIS chunk's row when the position is >= pos, else the conv ring's slot (position % 16) written by an
// earlier chunk or decode step, zero before position 0 - and x_3 = gn[m]; conv = rne(silu(rne(sum_t w[c][t] x_t)))
// (one fma chain, t ascending: dilation 3, kernel 4, M:1208-1225); H[m][c] = rne(H + rne(gated + conv)).
// **q4_pf_ple_ring**: the last min(16, C) rows' gn into conv ring slot (pos + m) % 16 and their ids into the id ring
// - what decode's rings hold after the same ids (q4_ple_block / q4_ple_gather write slot p % 16 every step). The PLE
// gather at M = C (q4_ple.cl -DPLE_PF) writes no ring: its rows read p - 1, p - 2 from the chunk or the ring.
//
// tests/kernels/qwen4exp_pf_ref.h ple_gate_row / ple_conv_row / ple_chunk are this file; qwen4exp_pf_ref_test holds
// them to 21c's q4ref::ple_block applied row by row, bitwise. Every norm is q4_hc's (256 lanes, fma, the tree,
// 1 / sqrt); every sigmoid / SiLU is 1 / (1 + exp_torch(-x)) / x / (1 + exp_torch(-x)).
//
// Plain OpenCL C (no sub-group functions): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_pf_ple: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#ifndef PF_C
#error "q4_pf_ple: PF_C (the chunk's rows, kernels::qwen4exp::kPfC) must be defined"
#endif
#define HIDDEN 2560
#define HC 4
#define HCN (HC * HIDDEN)
#define KVN (HCN + HIDDEN)   /* key 10240 | value 2560: the key||value GEMM's row pitch (pf_ld(12800) = 12800) */
#define RING 16
#define PER_LANE (HIDDEN / 256)
#define WG 256
#define NK_OFF 0
#define NQ_OFF HCN
#define NC_OFF (2 * HCN)
#define TAP_OFF (3 * HCN)

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
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > PF_C ? PF_C : n;
}

// rstd of the work-group's 2560 values (each lane PER_LANE of them, k = lid + 256 j): q4_ple_block's wg_rstd.
inline float wg_rstd(const float* v, __local float* red, uint lid) {
  float acc = 0.0f;
  for (uint j = 0; j < PER_LANE; ++j) acc = fma(v[j], v[j], acc);
  red[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = 1.0f / sqrt(red[0] / (float)HIDDEN + 1e-6f);
  barrier(CLK_LOCAL_MEM_FENCE);   // red is reused
  return r;
}

//   kv fp32 [C][12800] (the key||value GEMM)   H bf16 [C][10240]   pw the PLE block at norm_key (norm_query,
//   norm_conv, the taps follow: loader/qwen4exp_layout.h Q4PleOffsets)   gated, gn bf16 [C][10240]
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_pf_ple_gate(__global const uint* restrict ctrl, __global const float* restrict kv,
                             __global const ushort* restrict H, __global const float* restrict pw,
                             __global ushort* restrict gated_out, __global ushort* restrict gn_out) {
  const uint s = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float red[WG];
  __global const ushort* restrict h = H + (size_t)m * HCN + (size_t)s * HIDDEN;
  __global const float* restrict kvr = kv + (size_t)m * KVN;
  const uint c0 = s * HIDDEN;
  float key[PER_LANE], hv[PER_LANE], kn[PER_LANE], qn[PER_LANE], gated[PER_LANE];
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint k = lid + 256u * j;
    key[j] = rf(kvr[c0 + k]);
    hv[j] = bf16f(h[k]);
  }
  const float rk = wg_rstd(key, red, lid);
  const float rq = wg_rstd(hv, red, lid);
  float dot = 0.0f;
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint c = c0 + lid + 256u * j;
    kn[j] = rf(key[j] * rk * pw[NK_OFF + c]);
    qn[j] = rf(hv[j] * rq * pw[NQ_OFF + c]);
    dot += rf(kn[j] * qn[j]);
  }
  red[lid] = dot;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float s2 = rf(rf(red[0]) / 50.596442562694070f);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float eps_b = rf(1e-6f);
  const float a = fabs(s2);
  const float r = rf(sqrt(a < eps_b ? eps_b : a));
  const float gate = s2 > 0.0f ? r : (s2 < 0.0f ? -r : 0.0f);
  const float sg = rf(sigmoid_t(gate));
  for (uint j = 0; j < PER_LANE; ++j) gated[j] = rf(sg * rf(kvr[HCN + lid + 256u * j]));
  const float rc = wg_rstd(gated, red, lid);
  for (uint j = 0; j < PER_LANE; ++j) {
    const uint c = c0 + lid + 256u * j;
    gated_out[(size_t)m * HCN + c] = rne_bf16(gated[j]);
    gn_out[(size_t)m * HCN + c] = rne_bf16(gated[j] * rc * pw[NC_OFF + c]);
  }
}

// One of the conv's three history rows for row m at position p: back = 9, 6 or 3.
inline float hist_at(__global const ushort* restrict gn, __global const ushort* restrict ring, uint pos, uint m, uint back,
                     uint c) {
  const uint p = pos + m;
  if (p < back) return 0.0f;
  const uint q = p - back;
  return q >= pos ? bf16f(gn[(size_t)(q - pos) * HCN + c]) : bf16f(ring[(size_t)(q % RING) * HCN + c]);
}

//   gated, gn bf16 [C][10240] (q4_pf_ple_gate's)   ring bf16 [16][10240] (the PLE conv ring, read only)   H in / out
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_pf_ple_conv(__global const uint* restrict ctrl, __global const ushort* restrict gated,
                             __global const ushort* restrict gn, __global const ushort* restrict ring,
                             __global const float* restrict pw, __global ushort* restrict H) {
  const uint c = get_group_id(0) * WG + get_local_id(0);
  const uint m = get_group_id(1);
  if (m >= n_rows(ctrl)) return;
  const uint pos = ctrl[CTRL_POS];
  const float x0 = hist_at(gn, ring, pos, m, 9, c);
  const float x1 = hist_at(gn, ring, pos, m, 6, c);
  const float x2 = hist_at(gn, ring, pos, m, 3, c);
  const float x3 = bf16f(gn[(size_t)m * HCN + c]);
  __global const float* restrict t = pw + TAP_OFF + (size_t)c * 4;
  float acc = 0.0f;
  acc = fma(x0, t[0], acc);
  acc = fma(x1, t[1], acc);
  acc = fma(x2, t[2], acc);
  acc = fma(x3, t[3], acc);
  const float conv = rf(silu_t(rf(acc)));
  const float out = rf(bf16f(gated[(size_t)m * HCN + c]) + conv);
  __global ushort* restrict h = H + (size_t)m * HCN + c;
  *h = rne_bf16(bf16f(*h) + out);
}

//   gn bf16 [C][10240]   ids u32 [C] (the chunk's tokens)   ring bf16 [16][10240]   id_ring u32 [16]
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void q4_pf_ple_ring(__global const uint* restrict ctrl, __global const ushort* restrict gn,
                             __global const uint* restrict ids, __global ushort* restrict ring,
                             __global uint* restrict id_ring) {
  const uint c = get_group_id(0) * WG + get_local_id(0);
  const uint C = n_rows(ctrl);
  const uint y = get_group_id(1);   // the y-th of the last 16 rows
  if (C + y < RING) return;         // fewer than 16 rows: only the last C
  const uint m = C + y - RING;
  const uint slot = (ctrl[CTRL_POS] + m) % RING;
  ring[(size_t)slot * HCN + c] = gn[(size_t)m * HCN + c];
  if (c == 0) id_ring[slot] = ids[m];
}
