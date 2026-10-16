// df_prep.cl - spec 19b: the DFlash drafter's between-GEMV kernels (spec 19 §2, §5 "Kernels"). Every op chain has a
// host twin in tests/kernels/dflash_ref.h (same order, FP_CONTRACT off); the semantics are tools/oracle/dflash_ref.py's.
//
// The block is M rows (the drafter's block, 8 for Qwen3.8's): row 0 the anchor, rows 1..K mask rows. Rows past K are
// computed like the others and never read by a valid row (the convolutions look back, attention masks them), so one
// captured list serves every K <= M - 1 (DFlashCtl.k).
//
//   df_embed       grid (1, M), WG 256: resid[r] = (r == 0 ? embed[ctrl.cur_token[0]] : mask_row) * in_scale (fp32)
//   df_norm        grid (1, rows), WG 256: h = (x · rstd) · w, rstd = 1 / sqrt(Σx² / H + eps); writes fp32 h and its
//                  bf16 copy x (the next GEMV's input). Σx²: lane l over i = l, l + 256, ... then a pairwise tree.
//   df_conv_prep   grid (ceil(H / 256), M): the grouped two-tap convolution's prepare side (side 0) - out =
//                  bf16(conv(h))
//   df_conv_add    grid (ceil(H / 256), M): its finish side (side 1) folded into the residual - resid += conv(src)
//   df_add         grid (ceil(H / 256), M): resid += src (DFlash v1: no convolution)
//   df_silu_mul    grid (ceil(I / 256), M): a = bf16(g / (1 + exp(-g)) · u), gate || up concatenated
//   df_qkv_prep    grid (NH + 2 NKV, M), WG HD: q heads (q_norm, RoPE at pos + r), the block's k heads (k_norm,
//                  RoPE) and v heads - fp32, the block's own K / V never touch the ring (Review Focus 2)
//   df_ctx_store   grid (L · 2 NKV, M), WG HD: the context projection's last step - per layer k (k_norm, RoPE at
//                  proj_pos + r) and v of rows r < proj_n into ring slot (proj_pos + r) % RING, bf16 (spec 19 §5:
//                  the commit list projects the committed rows, 19c)
//
// The conv (dflash_ref.grouped_conv, vLLM qwen3_dflash2.py:144): kernel_projection's row r is [side][tap][group]
// (KPN = 2 · CTAPS · G floats); side 0 is the prepare, side 1 the finish; base is [side][tap][H]. Tap t reaches row
// r - t only when r >= t - one row of look-back for the published drafters (Review Focus 3 of plan 19a).
#pragma OPENCL FP_CONTRACT OFF

#ifndef H
#error "H (the drafter's hidden size) must be defined"
#endif
#ifndef M
#define M 8
#endif
#ifndef WG
#define WG 256
#endif
#ifndef EPS
#define EPS 1e-06f
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)0x7FC0;   // NaN
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// The fixed Σ: lane partials then strides WG/2 .. 1 (prep.cl's tree). `red` holds WG floats.
inline float wg_sum(float v, __local float* red) {
  const uint lid = get_local_id(0);
  red[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (lid < s) red[lid] = red[lid] + red[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}

#ifdef VOCAB
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_embed(__global const uint* restrict ctrl, __global const ushort* restrict embed,
                       __global const float* restrict mask_row, const float in_scale, __global float* restrict resid) {
  const uint r = get_group_id(1), lid = get_local_id(0);
  uint id = ctrl[CTRL_CUR];
  if (id >= (uint)VOCAB) id = 0;   // an engine bug, not a token: never read past the table
  __global const ushort* e = embed + (size_t)id * H;
  for (uint i = lid; i < H; i += WG) resid[(size_t)r * H + i] = (r == 0 ? bf16f(e[i]) : mask_row[i]) * in_scale;
}
#endif

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_norm(__global const float* restrict src, __global const float* restrict w, __global float* restrict h,
                      __global ushort* restrict x) {
  __local float red[WG];
  const uint r = get_group_id(1), lid = get_local_id(0);
  __global const float* s = src + (size_t)r * H;
  float a = 0.0f;
  for (uint i = lid; i < H; i += WG) a = a + s[i] * s[i];
  const float ss = wg_sum(a, red);
  const float rstd = 1.0f / sqrt(ss / (float)H + EPS);
  for (uint i = lid; i < H; i += WG) {
    const float v = (s[i] * rstd) * w[i];
    h[(size_t)r * H + i] = v;
    x[(size_t)r * H + i] = f2bf(v);
  }
}

#ifdef CTAPS
#define G (H / GROUP)
#define KPN (2 * CTAPS * G)
// co: kernel_projection's output [M][KPN]; base: this convolution's [side][tap][H]; side 0 prepare, 1 finish.
inline float conv_at(__global const float* src, __global const float* co, __global const float* base, uint side,
                     uint r, uint i) {
  const uint g = i / GROUP;
  __global const float* cs = co + (size_t)r * KPN + side * CTAPS * G;
  __global const float* bs = base + (size_t)side * CTAPS * H;
  float o = (bs[i] + cs[g]) * src[(size_t)r * H + i];
  for (uint t = 1; t < CTAPS; ++t)
    if (r >= t) o = o + (bs[(size_t)t * H + i] + cs[t * G + g]) * src[(size_t)(r - t) * H + i];
  return o;
}
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_conv_prep(__global const float* restrict src, __global const float* restrict co,
                           __global const float* restrict base, __global ushort* restrict out) {
  const uint r = get_group_id(1), i = get_group_id(0) * WG + get_local_id(0);
  if (i < H) out[(size_t)r * H + i] = f2bf(conv_at(src, co, base, 0, r, i));
}
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_conv_add(__global const float* restrict src, __global const float* restrict co,
                          __global const float* restrict base, __global float* restrict resid) {
  const uint r = get_group_id(1), i = get_group_id(0) * WG + get_local_id(0);
  if (i < H) resid[(size_t)r * H + i] = resid[(size_t)r * H + i] + conv_at(src, co, base, 1, r, i);
}
#endif

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_add(__global const float* restrict src, __global float* restrict resid) {
  const uint r = get_group_id(1), i = get_group_id(0) * WG + get_local_id(0);
  if (i < H) resid[(size_t)r * H + i] = resid[(size_t)r * H + i] + src[(size_t)r * H + i];
}

#ifdef INTER
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_silu_mul(__global const float* restrict gu, __global ushort* restrict a) {
  const uint r = get_group_id(1), i = get_group_id(0) * WG + get_local_id(0);
  if (i >= INTER) return;
  const float g = gu[(size_t)r * 2 * INTER + i], u = gu[(size_t)r * 2 * INTER + INTER + i];
  a[(size_t)r * INTER + i] = f2bf(g / (1.0f + exp(-g)) * u);
}
#endif

#ifdef HD
#define HALF (HD / 2)
#define QN (NH * HD)
#define KVN (NKV * HD)
#define QKVN (QN + 2 * KVN)
// One head's RMSNorm over HD lanes (Σ² by the pairwise tree) and neox RoPE from the table [pos][cos HALF | sin HALF].
// `y` is HD floats of local scratch; returns this lane's output.
inline float head_norm_rope(float x, __global const float* restrict w, __global const float* restrict rope, uint pos,
                            __local float* y) {
  const uint i = get_local_id(0);
  y[i] = x * x;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = HD / 2; s > 0; s >>= 1) {
    if (i < s) y[i] = y[i] + y[i + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(y[0] / (float)HD + EPS);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float v = (x * rstd) * w[i];
  y[i] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const float* c = rope + (size_t)pos * HD;
  const uint j = i % HALF;
  const float o = i < HALF ? v * c[j] - y[i + HALF] * c[HALF + j] : v * c[j] + y[i - HALF] * c[HALF + j];
  barrier(CLK_LOCAL_MEM_FENCE);
  return o;
}

// qn_kn: this layer's [q_norm | k_norm][HD] (loader/dflash_layout.h).
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void df_qkv_prep(__global const uint* restrict ctrl, __global const float* restrict qkv,
                          __global const float* restrict qn_kn, __global const float* restrict rope,
                          __global float* restrict q, __global float* restrict kb, __global float* restrict vb) {
  __local float y[HD];
  const uint s = get_group_id(0), r = get_group_id(1), i = get_local_id(0);
  const float x = qkv[(size_t)r * QKVN + (size_t)s * HD + i];
  const uint pos = ctrl[CTRL_POS] + r;
  if (s < NH) {
    q[((size_t)r * NH + s) * HD + i] = head_norm_rope(x, qn_kn, rope, pos, y);
  } else if (s < NH + NKV) {
    kb[((size_t)r * NKV + (s - NH)) * HD + i] = head_norm_rope(x, qn_kn + HD, rope, pos, y);
  } else {
    vb[((size_t)r * NKV + (s - NH - NKV)) * HD + i] = x;
  }
}

#ifdef RING
// kv: the projection GEMVs' outputs [L][M][2 KVN] (layer l's k || v rows of q||k||v); qk_norm [L][q | k][HD];
// ring_k / ring_v [L][RING][NKV][HD] bf16.
__attribute__((reqd_work_group_size(HD, 1, 1)))
__kernel void df_ctx_store(__global const uint* restrict dctl, __global const float* restrict kv,
                           __global const float* restrict qk_norm, __global const float* restrict rope,
                           __global ushort* restrict ring_k, __global ushort* restrict ring_v) {
  __local float y[HD];
  const uint grp = get_group_id(0), r = get_group_id(1), i = get_local_id(0);
  if (r >= dctl[DCTL_PN]) return;   // uniform over the work-group: the whole group returns
  const uint l = grp / (2 * NKV), rem = grp % (2 * NKV), head = rem % NKV;
  const bool is_v = rem >= NKV;
  const uint pos = dctl[DCTL_PPOS] + r, slot = pos % RING;
  const float x = kv[((size_t)l * M + r) * 2 * KVN + (is_v ? KVN : 0) + (size_t)head * HD + i];
  const size_t at = (((size_t)l * RING + slot) * NKV + head) * HD + i;
  if (is_v) {
    ring_v[at] = f2bf(x);
  } else {
    ring_k[at] = f2bf(head_norm_rope(x, qk_norm + ((size_t)l * 2 + 1) * HD, rope, pos, y));
  }
}
#endif
#endif
