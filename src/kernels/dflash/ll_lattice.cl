// ll_lattice.cl - spec 26b: the LiLiCorr head (spec 26 §2.3; vLLM's LiLiCorrHead, models/lilicorr.py:125-425, as
// tools/oracle/dflash_ref.py ports it) in plain OpenCL C - no subgroup or 2D-block extension, so it runs on the Mac's GPU as
// it runs on the card. One draft block: S = 7 slots x KC = 8 candidates = R = 56 lattice tokens, fp32 throughout over the
// head's EXACT weights (Decision 3: W'^T bf16 k-major + its g + its bias - loader/lilicorr_layout.h). Twin:
// tests/kernels/lilicorr_ref.h head_lattice() (same ops, same orders; FP_CONTRACT off). Every reduction a fixed order.
//
//   ll_gemm         grid (N / GW), WG GW: y[r][n] = act((Σ_k x[r][k] w[k][n], k ascending) x gs[n] + bias[n]) for rows
//                   r < rows (<= RMAX), written (mode 0) or added to y (mode 1: the residual adds) - one lane a column,
//                   every row's accumulator in the lane, x staged KT columns at a time in local memory (the weights are
//                   read once: 74 MB a block is the head's cost). act 1 = SiLU (x / (1 + exp(-x))). x rows at
//                   x + xoff + r · ldx (fp32): the 7 mask rows of the final-normed block, the anchor row, the head's own.
//   ll_gemm_gather  the same over the candidates' embedding rows (bf16 [V][K] gathered by cand[r]) - token_proj
//   ll_feat         grid (1), WG RMAX: per lattice token the 5 features [lp, exp(lp), lp - max_slot lp, i / (KC - 1),
//                   i == 0] and LayerNorm(5) (mean ((((f0 + f1) + f2) + f3) + f4) / 5, var ascending, eps FEPS)
//   ll_sum          grid (ceil(HL / 256), R): h = (((tok + pass[slot]) + feat) + slot_emb[slot]) + rank_emb[rank]
//   ll_valid        grid (ceil(HL / 256)): a x (dctl.anchor_valid ? 1 : 0) - context_proj's output (a draft with no
//                   committed row reads a zero anchor, vLLM's anchor_valid)
//   ll_rmsnorm      grid (1, rows), WG 256: df_norm's Σx² (lane partials, the pairwise tree), y = (x · rstd) · w, fp32
//   ll_lattn        grid (LHEADS, R), WG RMAX: one (head, query token) - scores over the R keys (q · k ascending d) x
//                   1/sqrt(LHD) + the slot bias (relative_slot_bias[head][slot(i) - slot(j) + BLK - 1] + same_slot_bias
//                   [head] where the slots match: the TRAINED offset BLK - 1 = 7 - Review Focus 4), m, e = exp(s - m),
//                   l ascending j, out[d] = (Σ_j e_j v_j[d], j ascending) / l
//   ll_factor       grid (HL / GW), WG GW: factor_input_proj split in three (k rows [0, HL) self, [HL, 2 HL) anchor,
//                   [2 HL, 3 HL) cross): fh = SiLU(((Σ h w_self) gs + b + (Σ a w_anchor) gs) + (Σ (h ⊙ a) w_cross) gs)
//   ll_l2norm       grid (halves, rows), WG 256: x / max(sqrt(Σx²), eps) over n (F.normalize; Σx² by the tree)
//   ll_scores       grid (S, KC), WG KC: slot 0 = logit_scale · (a_out · in_v[0][j]) for every previous i (start,
//                   broadcast); slot s >= 1 = logit_scale · (out_v[s - 1][i] · in_v[s][j]) - Review Focus 4 - into the
//                   lattice [S][KC (previous)][KC (next)]
//   ll_anchor       grid (ceil(n / 256)): the projection list's last row (dctl.proj_n - 1) of its fp32 hidden_norm output
//                   into the anchor buffer (Review Focus 5) - the next block's anchor is the newest committed row
#pragma OPENCL FP_CONTRACT OFF

#ifndef RMAX
#define RMAX 64
#endif
#ifndef GW
#define GW 64
#endif
#ifndef KT
#define KT 16
#endif
#define WG 256
#ifndef FEPS
#define FEPS 1e-05f
#endif
#ifndef EPS
#define EPS 1e-06f
#endif
#if !defined(KC) || !defined(S) || !defined(HL) || !defined(LHEADS) || !defined(BLK)
#error "KC, S, HL, LHEADS and BLK must be defined"
#endif
#define R (S * KC)
#define LHD (HL / LHEADS)
#if R > RMAX || RMAX > 64
#error "the lattice's tokens (S x KC) must fit RMAX <= 64 (one lane each)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline float silu(float x) { return x / (1.0f + exp(-x)); }

// The fixed Σ over WG lanes (df_prep.cl's tree).
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

// gb: [gs N | bias N] (one allocation a linear).
__attribute__((reqd_work_group_size(GW, 1, 1)))
__kernel void ll_gemm(__global const float* restrict x, const uint xoff, const uint ldx, __global const ushort* restrict wt,
                      __global const float* restrict gb, __global float* restrict y, const uint rows, const uint K,
                      const uint N, const uint mode, const uint act) {
  __local float xs[RMAX * KT];
  const uint lid = get_local_id(0), n = get_group_id(0) * GW + lid;
  float acc[RMAX];
  for (uint r = 0; r < RMAX; ++r) acc[r] = 0.0f;
  for (uint k0 = 0; k0 < K; k0 += KT) {
    const uint kt = min((uint)KT, K - k0);
    for (uint e = lid; e < rows * KT; e += GW) {
      const uint r = e / KT, kk = e % KT;
      xs[e] = kk < kt ? x[xoff + (size_t)r * ldx + k0 + kk] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint kk = 0; kk < kt; ++kk) {
      const float w = bf16f(wt[(size_t)(k0 + kk) * N + n]);
      for (uint r = 0; r < rows; ++r) acc[r] = acc[r] + xs[r * KT + kk] * w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float g = gb[n], b = gb[N + n];
  for (uint r = 0; r < rows; ++r) {
    float v = acc[r] * g + b;
    if (act) v = silu(v);
    y[(size_t)r * N + n] = mode ? y[(size_t)r * N + n] + v : v;
  }
}

__attribute__((reqd_work_group_size(GW, 1, 1)))
__kernel void ll_gemm_gather(__global const ushort* restrict embed, __global const uint* restrict cand,
                             __global const ushort* restrict wt, __global const float* restrict gb,
                             __global float* restrict y, const uint rows, const uint K, const uint N, const uint vrows) {
  __local float xs[RMAX * KT];
  const uint lid = get_local_id(0), n = get_group_id(0) * GW + lid;
  float acc[RMAX];
  for (uint r = 0; r < RMAX; ++r) acc[r] = 0.0f;
  for (uint k0 = 0; k0 < K; k0 += KT) {
    const uint kt = min((uint)KT, K - k0);
    for (uint e = lid; e < rows * KT; e += GW) {
      const uint r = e / KT, kk = e % KT;
      uint id = cand[r];
      if (id >= vrows) id = 0;   // a short row's NO_IDX (an engine bug, not a token): never read past the table
      xs[e] = kk < kt ? bf16f(embed[(size_t)id * K + k0 + kk]) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint kk = 0; kk < kt; ++kk) {
      const float w = bf16f(wt[(size_t)(k0 + kk) * N + n]);
      for (uint r = 0; r < rows; ++r) acc[r] = acc[r] + xs[r * KT + kk] * w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float g = gb[n], b = gb[N + n];
  for (uint r = 0; r < rows; ++r) y[(size_t)r * N + n] = acc[r] * g + b;
}

// small: the head's fp32 block (loader::lili_head_small); fn_off: feature_norm.weight (its bias at fn_off + 5).
__attribute__((reqd_work_group_size(RMAX, 1, 1)))
__kernel void ll_feat(__global const float* restrict lp, __global const float* restrict small, const uint fn_off,
                      __global float* restrict f) {
  const uint r = get_local_id(0);
  if (r >= R) return;
  const uint s = r / KC, i = r % KC;
  float mx = -INFINITY;
  for (uint c = 0; c < KC; ++c) mx = fmax(mx, lp[s * KC + c]);
  const float x = lp[r];
  float o[5];
  o[0] = x;
  o[1] = exp(x);
  o[2] = x - mx;
  o[3] = KC > 1 ? (float)i / (float)(KC - 1) : 0.0f;
  o[4] = i == 0 ? 1.0f : 0.0f;
  const float mean = ((((o[0] + o[1]) + o[2]) + o[3]) + o[4]) / 5.0f;
  float var = 0.0f;
  for (uint j = 0; j < 5; ++j) var = var + (o[j] - mean) * (o[j] - mean);
  var = var / 5.0f;
  const float rstd = 1.0f / sqrt(var + FEPS);
  for (uint j = 0; j < 5; ++j) f[r * 5 + j] = ((o[j] - mean) * rstd) * small[fn_off + j] + small[fn_off + 5 + j];
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_sum(__global const float* restrict tok, __global const float* restrict pas,
                     __global const float* restrict fm, __global const float* restrict small, const uint slot_off,
                     const uint rank_off, __global float* restrict h) {
  const uint r = get_group_id(1), j = get_group_id(0) * WG + get_local_id(0);
  if (j >= HL) return;
  const uint s = r / KC, i = r % KC;
  const size_t at = (size_t)r * HL + j;
  h[at] = (((tok[at] + pas[(size_t)s * HL + j]) + fm[at]) + small[slot_off + s * HL + j]) + small[rank_off + i * HL + j];
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_valid(__global const uint* restrict dctl, __global float* restrict a) {
  const uint j = get_group_id(0) * WG + get_local_id(0);
  if (j < HL) a[j] = a[j] * (dctl[DCTL_AVALID] ? 1.0f : 0.0f);
}

// x and y rows of n floats; w at small + w_off.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_rmsnorm(__global const float* restrict x, __global const float* restrict small, const uint w_off,
                         __global float* restrict y, const uint n) {
  __local float red[WG];
  const uint r = get_group_id(1), lid = get_local_id(0);
  __global const float* xr = x + (size_t)r * n;
  float a = 0.0f;
  for (uint i = lid; i < n; i += WG) a = a + xr[i] * xr[i];
  const float ss = wg_sum(a, red);
  const float rstd = 1.0f / sqrt(ss / (float)n + EPS);
  for (uint i = lid; i < n; i += WG) y[(size_t)r * n + i] = (xr[i] * rstd) * small[w_off + i];
}

// qkv [R][3 HL] (q | k | v, head h at h · LHD); o [R][HL].
__attribute__((reqd_work_group_size(RMAX, 1, 1)))
__kernel void ll_lattn(__global const float* restrict qkv, __global const float* restrict small, const uint rel_off,
                       const uint same_off, const float scale, __global float* restrict o) {
  __local float sc[RMAX];
  __local float ml[2];
  const uint h = get_group_id(0), i = get_group_id(1), lid = get_local_id(0);
  __global const float* qi = qkv + (size_t)i * 3 * HL + h * LHD;
  if (lid < R) {
    const uint j = lid;
    __global const float* kj = qkv + (size_t)j * 3 * HL + HL + h * LHD;
    float d = 0.0f;
    for (uint t = 0; t < LHD; ++t) d = d + qi[t] * kj[t];
    const uint si = i / KC, sj = j / KC;
    const float bias = small[rel_off + h * (2 * BLK - 1) + (si + BLK - 1 - sj)] + (si == sj ? small[same_off + h] : 0.0f);
    sc[j] = d * scale + bias;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    float m = -INFINITY;
    for (uint j = 0; j < R; ++j) m = fmax(m, sc[j]);
    float l = 0.0f;
    for (uint j = 0; j < R; ++j) {
      sc[j] = exp(sc[j] - m);
      l = l + sc[j];
    }
    ml[0] = m;
    ml[1] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint t = lid; t < LHD; t += RMAX) {
    float acc = 0.0f;
    for (uint j = 0; j < R; ++j) acc = acc + sc[j] * qkv[(size_t)j * 3 * HL + 2 * HL + h * LHD + t];
    o[(size_t)i * HL + h * LHD + t] = acc / ml[1];
  }
}

// xn [R][HL] the output-normed tokens, an [HL] the normed anchor; wt k-major [3 HL][HL]; gb [gs | bias].
__attribute__((reqd_work_group_size(GW, 1, 1)))
__kernel void ll_factor(__global const float* restrict xn, __global const float* restrict an,
                        __global const ushort* restrict wt, __global const float* restrict gb,
                        __global float* restrict fh) {
  __local float xs[RMAX * KT];
  const uint lid = get_local_id(0), n = get_group_id(0) * GW + lid;
  const float g = gb[n], b = gb[HL + n];
  float acc[RMAX];
  // pass 1: Σ h w_self
  for (uint r = 0; r < RMAX; ++r) acc[r] = 0.0f;
  for (uint k0 = 0; k0 < HL; k0 += KT) {
    for (uint e = lid; e < R * KT; e += GW) xs[e] = xn[(size_t)(e / KT) * HL + k0 + e % KT];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint kk = 0; kk < KT; ++kk) {
      const float w = bf16f(wt[(size_t)(k0 + kk) * HL + n]);
      for (uint r = 0; r < R; ++r) acc[r] = acc[r] + xs[r * KT + kk] * w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (uint r = 0; r < R; ++r) fh[(size_t)r * HL + n] = acc[r] * g + b;
  // pass 2: Σ a w_anchor (one row, every token's)
  float s2 = 0.0f;
  for (uint k = 0; k < HL; ++k) s2 = s2 + an[k] * bf16f(wt[(size_t)(HL + k) * HL + n]);
  // pass 3: Σ (h ⊙ a) w_cross
  for (uint r = 0; r < RMAX; ++r) acc[r] = 0.0f;
  for (uint k0 = 0; k0 < HL; k0 += KT) {
    for (uint e = lid; e < R * KT; e += GW) xs[e] = xn[(size_t)(e / KT) * HL + k0 + e % KT] * an[k0 + e % KT];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint kk = 0; kk < KT; ++kk) {
      const float w = bf16f(wt[(size_t)(2 * HL + k0 + kk) * HL + n]);
      for (uint r = 0; r < R; ++r) acc[r] = acc[r] + xs[r * KT + kk] * w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (uint r = 0; r < R; ++r) fh[(size_t)r * HL + n] = silu((fh[(size_t)r * HL + n] + s2 * g) + acc[r] * g);
}

// Row r, half c (group 0 / 1): x[r · ld + c · n .. + n) / max(||.||, eps).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_l2norm(__global float* restrict x, const uint n, const uint ld, const float eps) {
  __local float red[WG];
  const uint c = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  __global float* xr = x + (size_t)r * ld + (size_t)c * n;
  float a = 0.0f;
  for (uint i = lid; i < n; i += WG) a = a + xr[i] * xr[i];
  const float den = fmax(sqrt(wg_sum(a, red)), eps);
  for (uint i = lid; i < n; i += WG) xr[i] = xr[i] / den;
}

// ov [R][2 F] (out_v | in_v, normalised), ao [F] (a_out, normalised) -> lat [S][KC][KC].
__attribute__((reqd_work_group_size(KC, 1, 1)))
__kernel void ll_scores(__global const float* restrict ov, __global const float* restrict ao, const uint F,
                        const float lscale, __global float* restrict lat) {
  const uint s = get_group_id(0), i = get_group_id(1), j = get_local_id(0);
  float acc = 0.0f;
  if (s == 0) {
    __global const float* in0 = ov + (size_t)j * 2 * F + F;
    for (uint t = 0; t < F; ++t) acc = acc + ao[t] * in0[t];
  } else {
    __global const float* out_i = ov + ((size_t)(s - 1) * KC + i) * 2 * F;
    __global const float* in_j = ov + ((size_t)s * KC + j) * 2 * F + F;
    for (uint t = 0; t < F; ++t) acc = acc + out_i[t] * in_j[t];
  }
  lat[((size_t)s * KC + i) * KC + j] = lscale * acc;
}

// h: the projection list's hidden_norm output [M][n] (fp32); the anchor [n] = its row dctl.proj_n - 1.
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void ll_anchor(__global const uint* restrict dctl, __global const float* restrict h, const uint n,
                        __global float* restrict anchor) {
  const uint i = get_group_id(0) * WG + get_local_id(0), pn = dctl[DCTL_PN];
  if (i < n && pn > 0) anchor[i] = h[(size_t)(pn - 1) * n + i];
}
