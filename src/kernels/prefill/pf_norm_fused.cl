// pf_norm_fused.cl - phase 2 Task 5: the
// residual fold and the RMSNorm of one row in ONE launch, BITWISE pf_prep.cl's pf_res_fold + pf_norm_finish.
//
// One work-group of 256 lanes owns a whole row (grid (1, M)), so the norm's sum never leaves the work-group:
//   1. the fold, element for element pf_res_fold's (S_PREV 0: the residual as it is; 1: rne(f32(resid) +
//      f32(rne(0.0f + partials)))), the folded row written back, each square fma(v, v, 0.0f) parked in SLM;
//   2. the FOLD_G group sums in pf_res_fold's own tree: group g covers [g FOLD_CHUNK, +FOLD_CHUNK) and its
//      256-lane pairwise tree (stride 128 .. 1, lanes past the chunk an exact 0.0f) is evaluated by one
//      sub-group - strides 128 .. 16 in registers, 8 .. 1 by shuffles: the same pairs, the same adds;
//   3. pf_norm_finish's arithmetic: the group sums added in ascending g from 0.0f, mean = total / K,
//      rstd = 1 / sqrt(mean + 1e-6f), x = rne(f32(r) * rstd * w[k]).
// No sum is re-associated and no rounding point moves, so x and the residual are the two-launch walk's
// bytes; tests/prefill/pf_prep_test.cc compares every byte. The group sums are not written: the
// pipeline-parallel hand-off (spec 16c), which ships them, keeps the two-launch walk.
//
// PERF-85, the MTP head's prompt pass (runtime/prefill/step.cc step_mtp_kv), two more forms of the same text:
//   ZERO_RESID (S_PREV 1): pf_res_fold's _Z fold - the residual starts at the partials, rne(0.0f + partials) -
//      then the norm: the head's input_layernorm over fc's output, bitwise pf_res_fold _Z + pf_norm_finish.
//   MTP_FRONT (S_PREV 0): kernel pf_mtp_front, grid (1, M, 2) - pf_embed_gather of ids[m] and the two pre-fc norms
//      in one launch: z 0 the embedding row normed by w_e into x[m][0, K), z 1 the hidden row hid[m] normed by w_h
//      into x[m][K, 2K) (pf_norm_finish's X_STRIDE 2K). The S_PREV 0 fold writes its row back unchanged, so only x
//      is written: bitwise the five launches (gather, fold + finish twice).

#ifndef K
#error "pf_norm_fused: K (the hidden size) must be defined"
#endif
#ifndef S_PREV
#define S_PREV 0
#endif
#if S_PREV > 1
#error "pf_norm_fused: S_PREV is 0 or 1 (plan 6b ruling R1)"
#endif
#ifndef FOLD_G
#define FOLD_G 20
#endif
#ifndef ZERO_RESID
#define ZERO_RESID 0
#endif
#if ZERO_RESID && S_PREV != 1
#error "pf_norm_fused: ZERO_RESID starts the residual at the partials (S_PREV 1)"
#endif
#define WG 256
#define SG 16
#define FOLD_CHUNK ((K + FOLD_G - 1) / FOLD_G)
#define PER ((K + WG - 1) / WG)   /* elements a lane folds */
#if FOLD_CHUNK > WG
#error "pf_norm_fused: a fold group wider than the work-group (pf_res_fold's lanes would add twice)"
#endif
#if defined(MTP_FRONT) && S_PREV != 0
#error "pf_norm_fused: MTP_FRONT norms rows as they are (S_PREV 0)"
#endif

#include "bf16.h"

__attribute__((reqd_work_group_size(WG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#ifdef MTP_FRONT
__kernel void pf_mtp_front(__global const uint* restrict ids,
                           __global const ushort* restrict embed,
                           __global const ushort* restrict hid,
                           __global const float* restrict w_e,
                           __global const float* restrict w_h,
                           __global ushort* restrict x_out, uint m_count) {
#else
__kernel void pf_res_norm(__global const float* restrict partials,
                          __global ushort* restrict resid,
                          __global const float* restrict norm_w,
                          __global ushort* restrict x_out, uint m_count) {
#endif
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sg = lid / SG, lane = lid % SG;
  __local float sq[K];
  __local float gs[FOLD_G];
#ifdef MTP_FRONT
  const uint z = get_group_id(2);
  __global const ushort* rp = z == 0 ? embed + (size_t)ids[m] * K : hid + (size_t)m * K;
  __global const float* restrict norm_w = z == 0 ? w_e : w_h;
#else
  __global ushort* rp = resid + (size_t)m * K;
#endif

  ushort rb[PER];
#pragma unroll
  for (uint i = 0; i < PER; ++i) {
    const uint k = lid + i * WG;
    if (k < K) {
#if S_PREV == 0
      const ushort r_b = rp[k];
#else
      float acc = 0.f;
      acc += partials[(size_t)m * K + k];
#if ZERO_RESID
      const ushort r_b = rne_bf16(acc);
#else
      const ushort r_b = rne_bf16(bf16f(rp[k]) + bf16f(rne_bf16(acc)));
#endif
      rp[k] = r_b;
#endif
      rb[i] = r_b;
      const float v = bf16f(r_b);
      sq[k] = fma(v, v, 0.f);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint g = sg; g < FOLD_G; g += WG / SG) {
    const uint k0 = g * FOLD_CHUNK;
    const uint n = min((uint)FOLD_CHUNK, (uint)K - k0);
    float a[8];
#pragma unroll
    for (uint t = 0; t < 8; ++t) {   // stride 128
      const uint i = lane + SG * t;
      const float lo = i < n ? sq[k0 + i] : 0.0f;
      const float hi = i + 128 < n ? sq[k0 + i + 128] : 0.0f;
      a[t] = lo + hi;
    }
    float b[4];
#pragma unroll
    for (uint t = 0; t < 4; ++t) b[t] = a[t] + a[t + 4];   // stride 64
    const float c0 = b[0] + b[2], c1 = b[1] + b[3];        // stride 32
    float d = c0 + c1;                                     // stride 16
    d = d + intel_sub_group_shuffle(d, (lane + 8) & 15);   // 8 (lanes < 8 matter)
    d = d + intel_sub_group_shuffle(d, (lane + 4) & 15);   // 4
    d = d + intel_sub_group_shuffle(d, (lane + 2) & 15);   // 2
    d = d + intel_sub_group_shuffle(d, (lane + 1) & 15);   // 1
    if (lane == 0) gs[g] = d;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float total = 0.f;
  for (uint g = 0; g < FOLD_G; ++g) total += gs[g];
  const float mean = total / (float)K;
  const float rstd = 1.0f / sqrt(mean + 1e-6f);
#pragma unroll
  for (uint i = 0; i < PER; ++i) {
    const uint k = lid + i * WG;
#ifdef MTP_FRONT
    if (k < K) x_out[(size_t)m * 2 * K + z * K + k] = rne_bf16(bf16f(rb[i]) * rstd * norm_w[k]);
#else
    if (k < K) x_out[(size_t)m * K + k] = rne_bf16(bf16f(rb[i]) * rstd * norm_w[k]);
#endif
  }
}
