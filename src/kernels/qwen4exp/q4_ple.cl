// q4_ple.cl - spec 21c: Qwen3.8-Flash-Next's per-layer n-gram embedding (PLE, spec 21 §2.4, §4.3) on its one
// layer (layer_idx 1). The table lives in HOST memory (21b's 16 + 16 host-USM ranges, int8 rows + one scale a
// row); the device reads it zero-copy by an index it computes itself - no host doorbell, no per-token copy: the
// list stays fixed under replay. Two families, each compiled only when its define is given:
//
//   PLE_GATHER  q4_ple_gather(ctrl, ids_ring, ptrs, consts, e, ids_out)   grid (16 heads, M), WG 160
//               q4_ple_check(ptrs, first_page, out, pages)                 grid (ceil(pages / 64)), WG 64
//   PLE_BLOCK   q4_ple_block(ctrl, H, kv, pw, conv_ring)                   grid (4 streams, M), WG 256
//
// **q4_ple_gather** (M:1107-1166; tests/kernels/qwen4exp_ref.h ple_history + loader::q4_ple_ids): row m's position
// p = pos + m and token t0 = Control's cur_token[m]; its predecessors by the id-ring rule (21a's reading of
// M:1107-1121): t1 = id(p - 1) (EOS if p < 1); t2 = t1 == EOS ? EOS : id(p - 2) (EOS if p < 2), where id(q) is
// cur_token[q - pos] for a row of this launch and ids_ring[q % 16] for an earlier one. Then in exact uint64:
// mixed2 = (t0 m0) ^ (t1 m1), mixed3 = mixed2 ^ (t2 m2); head h < 8 reads mixed2, h >= 8 mixed3; its local row
// r = mixed mod size_h (the global id r + offset_h goes to ids_out, the golden gate's `ple.ids`). Lane i
// dequantises element i: e[m][h x 160 + i] = rne(float(q[r][i]) x float(scale[r])) (scale fp32, or bf16 under
// _BF16) - qwen4exp_ref.PleTable's int8 reader. Work-group (0, m)'s lane 0 then writes ids_ring[p % 16] = t0:
// rows of one launch write slots pos .. pos + M - 1 and read pos - 2 .. pos - 1 - disjoint at 16 slots.
// consts u64 [35]: the multipliers [3], the 16 head sizes, the 16 offsets (the engine copies them from the
// checkpoint's I64 tensors, checked by 21b against the formula). ptrs u64 [32]: 21b's Q4PleTable::ptrs (q[0..15],
// s[0..15]): a device-side table of host-USM addresses (`PLE_DIRECT`: the Mac's indicative build passes one
// buffer of all heads' rows / scales and byte offsets instead - OpenCL 1.2 has no pointer to pass).
//
// **q4_ple_block** (M:1227-1247, the decoder layer's `H = H + ple(H)`; qwen4exp_ref.h ple_block), one work-group
// per (stream s, row m); `kv` the key||value gemv_bf16 row fp32 [M][12800] over e; pw the layer's PLE block at
// its norms (loader/qwen4exp_layout.h Q4PleOffsets: norm_key, norm_query, norm_conv fp32 (1 + w) [10240] each,
// then the conv taps fp32 [10240][4]):
//   key = rne(kv[s x 2560 + k]) -> its stream's (1 + w) norm -> kn;  qn = norm_query(H) (the stream)
//   sb = rne(sum_k rne(kn x qn)) (lane i: k = i + 256 j ascending, a plain add; the tree); s2 = rne(sb / sqrt(2560))
//   gate = sign(s2) x rne(sqrt(max(|s2|, bf16(1e-6))));  sg = rne(sigmoid(gate))
//   gated = rne(sg x rne(kv[10240 + k]));  gn = norm_conv(gated) (the stream)
//   conv = rne(silu(rne(sum_t w[c][t] x x_t))) over x_0 .. x_3 = the conv ring's rows p - 9, p - 6, p - 3 (zero
//          before position 0) and gn - one fma chain, t ascending (dilation 3, kernel 4: M:1208-1225)
//   H[c] = rne(H[c] + rne(gated + conv));  conv_ring[p % 16][c] = gn
// Every norm is q4_hc's (256 lanes, fma, the tree, 1 / sqrt). M = 1 only: a row's p - 3 must be a ring row of an
// earlier launch (a 4-row build needs the rows of its own launch - 21e's).
//
// **q4_ple_check** (21b's alias check, the device half; spec 22 §1's xe hazard): one lane per 2 MiB page of every
// range: out[g] = the first u64 of page g - first_page[r] of range r (first_page [33], the ranges' page prefix
// sums); the host compares with Q4PleTable::page_words. Runs once, at engine construction.
//
// Plain OpenCL C (no sub-group functions; the Mac's OpenCL runs gather and block under PLE_DIRECT).
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT) || !defined(CTRL_CUR)
#error "q4_ple: CTRL_POS / CTRL_NACT / CTRL_CUR must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(PLE_GATHER) && !defined(PLE_BLOCK)
#error "q4_ple: PLE_GATHER (q4_ple_gather + q4_ple_check) or PLE_BLOCK (q4_ple_block) must be defined"
#endif
#define HIDDEN 2560
#define HC 4
#define HCN (HC * HIDDEN)
#define KVN (HCN + HIDDEN)   /* key 10240 | value 2560 */
#define HEADS 16
#define DIM 160
#define RING 16
#define PAGE (2u * 1024u * 1024u)
#define PER_LANE (HIDDEN / 256)

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
  return n > M ? M : n;
}

// ---------------------------------------------------------------------------------------------------------------
#ifdef PLE_GATHER
#ifndef PLE_EOS
#error "q4_ple_gather: PLE_EOS (the text config's eos_token_id, 248044) must be defined"
#endif
#if !defined(PLE_SCALE_BF16)
#error "q4_ple_gather: PLE_SCALE_BF16 (0: fp32 row scales, 1: bf16) must be defined"
#endif
#define WG_GATHER DIM
__attribute__((reqd_work_group_size(WG_GATHER, 1, 1)))
__kernel void q4_ple_gather(__global const uint* restrict ctrl, __global uint* restrict ids_ring,
#ifdef PLE_DIRECT
                            __global const char* restrict q_all, __global const uchar* restrict s_all,
                            __global const ulong* restrict head_off,   // [32]: byte offsets of q[h], s[h]
#else
                            __global const ulong* restrict ptrs,
#endif
                            __global const ulong* restrict consts, __global ushort* restrict e,
                            __global ulong* restrict ids_out) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  const uint pos = ctrl[CTRL_POS];
  const uint p = pos + m;
  const uint t0 = ctrl[CTRL_CUR + m];
  // id(q): this launch's rows from Control, earlier ones from the ring
  uint t1 = PLE_EOS, t2 = PLE_EOS;
  if (p >= 1) t1 = p - 1 >= pos ? ctrl[CTRL_CUR + (p - 1 - pos)] : ids_ring[(p - 1) % RING];
  if (t1 != PLE_EOS && p >= 2) t2 = p - 2 >= pos ? ctrl[CTRL_CUR + (p - 2 - pos)] : ids_ring[(p - 2) % RING];
  const ulong mixed2 = ((ulong)t0 * consts[0]) ^ ((ulong)t1 * consts[1]);
  const ulong mixed = h < 8 ? mixed2 : mixed2 ^ ((ulong)t2 * consts[2]);
  const ulong r = mixed % consts[3 + h];
  if (i == 0) ids_out[(size_t)m * HEADS + h] = r + consts[3 + HEADS + h];
#ifdef PLE_DIRECT
  const char qv = q_all[head_off[h] + r * DIM + i];
#if PLE_SCALE_BF16
  const float sc = bf16f(((__global const ushort*)(s_all + head_off[HEADS + h]))[r]);
#else
  const float sc = ((__global const float*)(s_all + head_off[HEADS + h]))[r];
#endif
#else
  const char qv = ((__global const char*)(intptr_t)ptrs[h])[r * DIM + i];
#if PLE_SCALE_BF16
  const float sc = bf16f(((__global const ushort*)(intptr_t)ptrs[HEADS + h])[r]);
#else
  const float sc = ((__global const float*)(intptr_t)ptrs[HEADS + h])[r];
#endif
#endif
  e[(size_t)m * HIDDEN + h * DIM + i] = rne_bf16((float)qv * sc);
  if (h == 0 && i == 0) ids_ring[p % RING] = t0;
}

#ifndef PLE_DIRECT
#define WG_CHECK 64
__attribute__((reqd_work_group_size(WG_CHECK, 1, 1)))
__kernel void q4_ple_check(__global const ulong* restrict ptrs, __global const uint* restrict first_page,
                           __global ulong* restrict out, uint pages) {
  const uint g = get_global_id(0);
  if (g >= pages) return;
  uint r = 0;
  while (r + 1 < 2 * HEADS && first_page[r + 1] <= g) ++r;
  const ulong addr = ptrs[r] + (ulong)(g - first_page[r]) * PAGE;
  out[g] = *(__global const ulong*)(intptr_t)addr;
}
#endif
#endif  // PLE_GATHER

// ---------------------------------------------------------------------------------------------------------------
#ifdef PLE_BLOCK
#if M != 1
#error "q4_ple_block: M = 1 only (a row's p - 3 is a ring row of an earlier launch)"
#endif
#define WG_BLOCK 256
#define NK_OFF 0
#define NQ_OFF HCN
#define NC_OFF (2 * HCN)
#define TAP_OFF (3 * HCN)
// rstd of the work-group's 2560 values (each lane PER_LANE of them, k = lid + 256 j): q4_hc's tree.
inline float wg_rstd(const float* v, __local float* red, uint lid) {
  float acc = 0.0f;
  for (uint j = 0; j < PER_LANE; ++j) acc = fma(v[j], v[j], acc);
  red[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_BLOCK / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = 1.0f / sqrt(red[0] / (float)HIDDEN + 1e-6f);
  barrier(CLK_LOCAL_MEM_FENCE);   // red is reused
  return r;
}
__attribute__((reqd_work_group_size(WG_BLOCK, 1, 1)))
__kernel void q4_ple_block(__global const uint* restrict ctrl, __global ushort* restrict H,
                           __global const float* restrict kv, __global const float* restrict pw,
                           __global ushort* restrict conv_ring) {
  const uint s = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float red[WG_BLOCK];
  const uint p = ctrl[CTRL_POS] + m;
  __global ushort* restrict h = H + (size_t)m * HCN + (size_t)s * HIDDEN;
  __global const float* restrict kvr = kv + (size_t)m * KVN;
  const uint c0 = s * HIDDEN;
  float key[PER_LANE], hv[PER_LANE], kn[PER_LANE], qn[PER_LANE], gated[PER_LANE], gn[PER_LANE];
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
  for (uint stride = WG_BLOCK / 2; stride > 0; stride >>= 1) {
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
    gn[j] = rf(gated[j] * rc * pw[NC_OFF + c]);
    const float x0 = p >= 9 ? bf16f(conv_ring[(size_t)((p - 9) % RING) * HCN + c]) : 0.0f;
    const float x1 = p >= 6 ? bf16f(conv_ring[(size_t)((p - 6) % RING) * HCN + c]) : 0.0f;
    const float x2 = p >= 3 ? bf16f(conv_ring[(size_t)((p - 3) % RING) * HCN + c]) : 0.0f;
    __global const float* restrict t = pw + TAP_OFF + (size_t)c * 4;
    float acc = 0.0f;
    acc = fma(x0, t[0], acc);
    acc = fma(x1, t[1], acc);
    acc = fma(x2, t[2], acc);
    acc = fma(gn[j], t[3], acc);
    const float conv = rf(silu_t(rf(acc)));
    const float out = rf(gated[j] + conv);
    h[lid + 256u * j] = rne_bf16(hv[j] + out);
    conv_ring[(size_t)(p % RING) * HCN + c] = rne_bf16(gn[j]);
  }
}

#endif  // PLE_BLOCK
