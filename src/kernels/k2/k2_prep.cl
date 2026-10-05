// k2_prep.cl - spec 18b: K2-Horizon's between-GEMV kernels. Three entry points, each
// compiled only when its family's defines are given (one binary per family and shape):
//
//   k2_norm_finish(sumsq, resid, norm_w, x)          NORM_G      grid (NORM_WGS, M), WG 256
//   k2_silu_mul(partials, x)                         SILU_I      grid (ceil(I/4096), M), WG 256
//   k2_attn_prep(ctrl, partials, rope, attn_q,       QKV_N       grid (Q + KV heads, M), WG HD
//                attn_gate, kv_k, kv_v)
//
// The reference is modeling_k2_horizon.py (the operator's checkpoint, spec 18 §3), and the
// rounding discipline is prep.cl's: torch rounds per op, so a linear's split-K partials are
// summed in fp32 and rounded to bf16 once, every elementwise op widens to fp32 and rounds
// its result. tests/kernels/k2_ref.h repeats every chain below; edit the two together.
//
// **No sub-group function anywhere in this file**, so it also builds as OpenCL 1.2 for an
// indicative run on the Mac's GPU (tools/mac/clrun/k2_run.cc).
#ifndef M
#define M 1
#endif
#ifndef K
#define K 2560              /* hidden */
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
// f32 -> bf16, round-to-nearest-even (prep.cl's; NaN is not expected and not handled).
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// ---------------------------------------------------------------------------------------
// k2_norm_finish - K2HorizonRMSNorm (modeling_k2_horizon.py:625-636), stage B. Stage A is
// prep.cl's prep_res_fold UNCHANGED (its fold chain and its per-chunk Σx² tree), compiled
// at K 2560, FOLD_G = NORM_G. The two traps of spec 18 §3:
//
//   1. GROUPED: the row is NORM_GROUPS groups of K / NORM_GROUPS (2 x 1280), and each group
//      has its OWN mean of squares - `reshape(..., n_groups, -1).pow(2).mean(-1)`. Stage A's
//      chunks are K / NORM_G wide and a group is a whole number of them, so group gi's Σ is
//      the ascending sum of chunks [gi x CPG, (gi + 1) x CPG).
//   2. PLAIN w: `self.weight * hidden_states` - the loader stores the checkpoint's w
//      widened verbatim (no `1 + w`, which is Qwen's).
//
//   rstd_gi     = 1 / sqrt(Σ_{chunks of gi} sumsq / GS + 1e-6)      never rsqrt
//   x_out[m][k] = rne(f32(resid[m][k]) · rstd_{k / GS} · norm_w[k])    (x · rstd) · w, as torch
#ifdef NORM_G
#ifndef NORM_WGS
#error "k2_norm_finish: NORM_WGS must be defined"
#endif
#ifndef NORM_GROUPS
#error "k2_norm_finish: NORM_GROUPS must be defined (2 on K2-Horizon)"
#endif
#define WG_NORM 256
#define GS (K / NORM_GROUPS)
#define CPG (NORM_G / NORM_GROUPS)
#define NORM_CHUNK ((K + NORM_WGS - 1) / NORM_WGS)
#if K % NORM_GROUPS != 0 || NORM_G % NORM_GROUPS != 0 || K % NORM_G != 0 || (K / NORM_G) * CPG != GS
#error "k2_norm_finish: the fold's chunks (K / NORM_G) must tile each norm group exactly"
#endif
#if NORM_GROUPS > 8
#error "k2_norm_finish: at most 8 norm groups"
#endif
__attribute__((reqd_work_group_size(WG_NORM, 1, 1)))
__kernel void k2_norm_finish(__global const float* restrict sumsq,
                             __global const ushort* restrict resid,
                             __global const float* restrict norm_w,
                             __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  float rstd[NORM_GROUPS];
  for (uint gi = 0; gi < NORM_GROUPS; ++gi) {
    float total = 0.f;
    for (uint g = gi * CPG; g < (gi + 1) * CPG; ++g) total += sumsq[(size_t)g * M + m];
    rstd[gi] = 1.0f / sqrt(total / (float)GS + 1e-6f);
  }
  const __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = w * NORM_CHUNK;
  uint k1 = k0 + NORM_CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG_NORM)
    x_out[(size_t)m * K + k] = rne_bf16(bf16f(rp[k]) * rstd[k / GS] * norm_w[k]);
}
#endif  // NORM_G

// ---------------------------------------------------------------------------------------
// k2_silu_mul - the dense layers' MLP activation (K2HorizonMLP: down(act(gate(x)) * up(x))),
// prep.cl's prep_silu_mul at K2's intermediate 6144 and gate||up's split-K SILU_S:
//
//   gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16      (cols_interleave16)
//   g_b = rne(Σ_s partials[s][m][gflat]) ;  u_b likewise
//   x_out[m][k] = rne(f32(rne(silu(f32(g_b)))) · f32(u_b))
#ifdef SILU_I
#ifndef SILU_S
#error "k2_silu_mul: SILU_S (gate||up's split-K) must be defined"
#endif
#define WG_SILU 256
#define SILU_CHUNK 4096
#define SILU_FUSED_N (2 * SILU_I)
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }
__attribute__((reqd_work_group_size(WG_SILU, 1, 1)))
__kernel void k2_silu_mul(__global const float* restrict partials, __global ushort* restrict x_out) {
  const uint m = get_group_id(1);
  const uint k0 = get_group_id(0) * SILU_CHUNK;
  uint k1 = k0 + SILU_CHUNK;
  if (k1 > SILU_I) k1 = SILU_I;
  for (uint k = k0 + get_local_id(0); k < k1; k += WG_SILU) {
    const size_t gflat = (size_t)(k / 16) * 32 + (k % 16);
    const size_t uflat = gflat + 16;
    float ga = 0.f, ua = 0.f;
    for (uint s = 0; s < SILU_S; ++s) {
      const size_t base = ((size_t)s * M + m) * SILU_FUSED_N;
      ga += partials[base + gflat];
      ua += partials[base + uflat];
    }
    const ushort g_b = rne_bf16(ga), u_b = rne_bf16(ua);
    const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
    x_out[(size_t)m * SILU_I + k] = rne_bf16(bf16f(s_b) * bf16f(u_b));
  }
}
#endif  // SILU_I

// ---------------------------------------------------------------------------------------
// k2_attn_prep - the attention's q / k / gate (and on the dense layers v) out of the fused
// q || k || gate || (v | v_router) GEMV, RoPE, the KV cache write. Grid (Q_HEADS + KV_HEADS,
// M), work-group HD (128): work-item i owns dim i of one head. Work-groups 0..Q-1 are
// q-heads (each also writing its gate), Q.. the kv-heads.
//
// K2 has NO q / k norm (`query_key_norm` false) and rotates ALL 128 dims (rope_head_dim =
// head_dim: apply_rotary_pos_emb with rotate_half, modeling_k2_horizon.py:52-83, 454-455).
// The reference's ops on bf16 tensors with bf16 cos / sin, op for op:
//
//   x_b   = rne(Σ_s partials[s][m][col])                       the linear's bf16 output
//   c, s  = rope[pos + m][0|1][i % 64]                         bf16 values (k2_rope_table)
//   t1    = rne(x_b[i] · c)                                    q * cos
//   t2    = rne(-x_b[i + 64] · s)   (i < 64)                   rotate_half(q) * sin
//           rne( x_b[i - 64] · s)   (i >= 64)
//   out   = rne(f32(t1) + f32(t2))
//   q-head: attn_q[m][h][i] = f32(out); attn_gate[m][h][i] = f32(rne(Σ_s gate column))
//   kv-head: kv_k[pos + m][j][i] = out; V_FROM_PARTIALS (the dense layers):
//            kv_v[pos + m][j][i] = rne(Σ_s v column)           (MoVA's v: k2_mova_value)
//
// A product of two bf16 values is exact in fp32, so t1 / t2 are each ONE rounding, as the
// reference's bf16 multiply is: this chain is the reference's bit for bit (unlike attn.cl's
// fused fp32 RoPE, which takes a one-op difference on Qwen3.8).
#ifdef QKV_N
#ifndef QKV_S
#error "k2_attn_prep: QKV_S (the fused GEMV's split-K) must be defined"
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "k2_attn_prep: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HD)
#error "k2_attn_prep: Q_HEADS, KV_HEADS and HD must be defined"
#endif
#ifndef V_FROM_PARTIALS
#define V_FROM_PARTIALS 0
#endif
#define K_OFF (Q_HEADS * HD)
#define GATE_OFF (K_OFF + KV_HEADS * HD)
#define V_OFF (GATE_OFF + Q_HEADS * HD)
#define HALF (HD / 2)
#if V_FROM_PARTIALS && QKV_N != V_OFF + KV_HEADS * HD
#error "k2_attn_prep: a dense layer's q||k||gate||v is V_OFF + KV_HEADS x HD wide"
#endif
#if !V_FROM_PARTIALS && QKV_N < V_OFF
#error "k2_attn_prep: the fused row ends before its v_router columns"
#endif
inline float qkv_sum(__global const float* restrict p, uint m, size_t col) {
  float v = 0.0f;
  for (uint s = 0; s < QKV_S; ++s) v += p[((size_t)s * M + m) * QKV_N + col];
  return v;
}
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void k2_attn_prep(__global const uint* restrict ctrl,
                           __global const float* restrict partials,
                           __global const float* restrict rope,
                           __global float* restrict attn_q, __global float* restrict attn_gate,
                           __global ushort* restrict kv_k, __global ushort* restrict kv_v) {
  const uint wg = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  __local float xs[HD];
  const uint pos = ctrl[CTRL_POS];
  uint n_act = ctrl[CTRL_NACT];
  if (n_act > M) n_act = M;
  if (m >= n_act) return;   // uniform across the work-group

  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  const size_t col = is_q ? (size_t)h * HD + i : (size_t)K_OFF + (size_t)h * HD + i;
  xs[i] = bf16f(rne_bf16(qkv_sum(partials, m, col)));
  barrier(CLK_LOCAL_MEM_FENCE);

  const __global float* restrict cs = rope + (size_t)(pos + m) * 2 * HALF;
  const uint ii = i % HALF;
  const float c = cs[ii], sn = cs[HALF + ii];
  const float rot = i < HALF ? -xs[i + HALF] : xs[i - HALF];
  const ushort t1 = rne_bf16(xs[i] * c);
  const ushort t2 = rne_bf16(rot * sn);
  const ushort out = rne_bf16(bf16f(t1) + bf16f(t2));

  if (is_q) {
    attn_q[((size_t)m * Q_HEADS + h) * HD + i] = bf16f(out);
    attn_gate[((size_t)m * Q_HEADS + h) * HD + i] =
        bf16f(rne_bf16(qkv_sum(partials, m, (size_t)GATE_OFF + (size_t)h * HD + i)));
  } else {
    const size_t slot = ((size_t)(pos + m) * KV_HEADS + h) * HD + i;
    kv_k[slot] = out;
#if V_FROM_PARTIALS
    kv_v[slot] = rne_bf16(qkv_sum(partials, m, (size_t)V_OFF + (size_t)h * HD + i));
#else
    (void)kv_v;
#endif
  }
}
#endif  // QKV_N
