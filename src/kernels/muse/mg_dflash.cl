// mg_dflash.cl - spec 23f: the two kernels Muse-Glimmer's DFlash drafter (meta-models/Muse-Glimmer-30B-assistant)
// needs beside spec 19b's df_* kernels, which run at its shapes from new CMake lines (df_prep.cl: the raw embedding,
// SiLU x up, the residual add, the q / k prep and the context store; df_attn.cl: the ring + block attention with the
// window |q - k| <= 2048 as WIN 2049; df_topk.cl: the head's top 16 per draft row). Both are written here because the
// existing sources cannot carry Muse's semantics without an edit (MG0 / D0: no existing .cl changes):
//
//   mg_df_norm(src, w, h, x)                        grid (1, rows), WG 256
//     MuseGlimmerAssistantRMSNorm (transformers 5.19.0 A:50-55): x-hat = src · rstd in fp32, ROUNDED to bf16, then
//     w x it, rounded again - x = bf16(w · bf16(x-hat)), h = f32(x). rstd = 1 / sqrt(Σsrc² / H + EPS) with df_norm's
//     sum (lane l over i = l, l + 256, ..., then the pairwise tree). df_norm rounds once (bf16((src · rstd) · w)).
//   mg_df_select(ctrl, dctl, cand, unary, out, vids)  grid (1), WG 16
//     the per-row pick of a DFlash (v1) drafter over df_topk's 16 candidates - greedy (the best unary, ties to the
//     lower candidate; NaN never wins) for rows j < K = min(dctl.k, MAXR - 1) - writing the readback (`out`: the
//     words of runtime::muse::MuseDraftOut, df_select.cl's OUT_* layout at MAXR rows) and the verify's input ids
//     `vids` [MAXR]: vids[0] = the anchor ctrl.cur_token[0], vids[1 + j] = draft j. df_select writes the drafts into
//     Control::cur_token[1 + j], which has 7 slots - Muse drafts up to 15. A sampled request's drafts are drawn on
//     the host from the readback (runtime::muse::MuseDrafter::draft), which then rewrites vids / ids.
#pragma OPENCL FP_CONTRACT OFF

#ifndef WG
#define WG 256
#endif
#ifndef TOPK
#define TOPK 16
#endif
#ifndef MAXR
#define MAXR 16
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort f2bf(float f) {
  const uint u = as_uint(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (ushort)0x7FC0;   // NaN
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

#ifdef H
#ifndef EPS
#error "mg_df_norm: EPS (the drafter's rms_norm_eps) must be defined"
#endif
inline float wg_sum(float v, __local float* red) {   // df_prep.cl's tree
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

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void mg_df_norm(__global const float* restrict src, __global const float* restrict w,
                         __global float* restrict h, __global ushort* restrict x) {
  __local float red[WG];
  const uint r = get_group_id(1), lid = get_local_id(0);
  __global const float* s = src + (size_t)r * H;
  float a = 0.0f;
  for (uint i = lid; i < H; i += WG) a = a + s[i] * s[i];
  const float ss = wg_sum(a, red);
  const float rstd = 1.0f / sqrt(ss / (float)H + EPS);
  for (uint i = lid; i < H; i += WG) {
    const float xh = bf16f(f2bf(s[i] * rstd));   // hidden_states.to(input_dtype)
    const ushort y = f2bf(w[i] * xh);            // self.weight * ... in bf16
    h[(size_t)r * H + i] = bf16f(y);
    x[(size_t)r * H + i] = y;
  }
}
#endif  // H

#ifdef SELECT
#if !defined(CTRL_POS) || !defined(CTRL_CUR) || !defined(DCTL_K)
#error "mg_df_select: CTRL_POS, CTRL_CUR and DCTL_K must be defined"
#endif
// runtime::muse::MuseDraftOut's words (df_select.cl's layout at MAXR rows; runtime/muse/muse_drafter.h asserts them)
#define OUT_K 0
#define OUT_ANCHOR 1
#define OUT_POS 2
#define OUT_IDS 4
#define OUT_CHOSEN (OUT_IDS + MAXR)
#define OUT_CAND (OUT_CHOSEN + MAXR)
#define OUT_UNARY (OUT_CAND + MAXR * TOPK)
#define OUT_SCORES (OUT_UNARY + MAXR * TOPK)

__attribute__((reqd_work_group_size(TOPK, 1, 1)))
__kernel void mg_df_select(__global const uint* restrict ctrl, __global const uint* restrict dctl,
                           __global const uint* restrict cand, __global const float* restrict unary,
                           __global uint* restrict out, __global uint* restrict vids) {
  const uint lid = get_local_id(0);
  uint K = dctl[DCTL_K];
  if (K > MAXR - 1) K = MAXR - 1;
  if (lid != 0) return;
  const uint anchor = ctrl[CTRL_CUR];
  out[OUT_K] = K;
  out[OUT_ANCHOR] = anchor;
  out[OUT_POS] = ctrl[CTRL_POS];
  vids[0] = anchor;
  for (uint j = 0; j < K; ++j) {
    __global const uint* cj = cand + (size_t)j * TOPK;
    __global const float* uj = unary + (size_t)j * TOPK;
    uint best = 0;
    float bv = -INFINITY;
    int any = 0;
    for (uint c = 0; c < TOPK; ++c) {
      const float v = uj[c];
      if (!any ? !isnan(v) : v > bv) {
        bv = v;
        best = c;
        any = 1;
      }
    }
    const uint id = cj[best];
    out[OUT_IDS + j] = id;
    out[OUT_CHOSEN + j] = best;
    for (uint c = 0; c < TOPK; ++c) {
      out[OUT_CAND + j * TOPK + c] = cj[c];
      out[OUT_UNARY + j * TOPK + c] = as_uint(uj[c]);
      out[OUT_SCORES + j * TOPK + c] = as_uint(uj[c]);
    }
    vids[1 + j] = id;
  }
}
#endif  // SELECT
