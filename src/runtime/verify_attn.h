#pragma once
// Spec 28c (plan 28c Task 5): which attention an M-row verify list binds - today's v2 pair
// (attn_v2_M<M>_T32[_KV8]: attn_decode_v2 + attn_reduce_v2) or attn_verify_dpas (src/kernels/attn_verify_dpas.cl,
// the DPAS kernel writing v2's partials) + the SAME v2 binary's attn_reduce_v2. Header-only, no Level Zero: the
// capture (runtime/capture.cc's attn_pair) asks it, tests/runtime/verify_attn_test.cc checks it on the host.
//
// The rule (spec 28c §3): V2 at M = 1 (plain decode never switches: spec 28 §8 decision 1), at any heads but
// Qwen3.8's 24 / 4 (Agnes shares them), for an (M, KV form) the table lacks (no binary is built there); else the
// measured table: Dpas from `dpas_from` keys of row 0 (pos + 1) on. Ties and noise keep v2 (plan 28c Task 4's
// verdict rule: paired v2 / dpas >= 1.03 with the range's low end > 1.00).
//
// Not a user flag and not an environment variable (the operator, 2026-11-03: no new env vars in the engine): the
// only override is `CaptureOptions::force_attn` (runtime/gemv_kind.h, beside 28b's force_linear), for tests and probes that compare the two kernels inside one list
// (V2 / Dpas at every M >= 2 the table has a row for; M = 1 and other heads stay V2 whatever it says).
//
// Depth is not known at capture: a captured list replays at every position. Where every row's dpas_from is 0 or
// never (verify_attn_depth_free), the capture decides by M alone (keys = 1 decides like any other depth there).
//
// This sits beside spec 28b's gemv_kind(M) (the capture's GEMM choice by M); built before 28b merged (operator
// ruling 2026-11-01), so it does not depend on it - one more capture-time rule by M, rebased onto 28b's capture.
#include <cstdint>

namespace runtime {

enum class VerifyAttn : uint8_t { V2, Dpas };
// Test / probe only: force one verify attention at every M >= 2 of a capture (`Auto` = the table).
enum class VerifyAttnForce : uint8_t { Auto, V2, Dpas };
inline constexpr uint32_t kDpasNever = UINT32_MAX;

struct VerifyAttnRow {
  uint32_t m;
  bool kv8;
  uint32_t dpas_from;   // row 0's key count (pos + 1) from which DPAS is bound; kDpasNever: never
};

// docs/probes/probe-verify-attn-2026-11-03.md §2 (busy-box timing, card 1; to be re-checked on a quiet box): v2 / dpas
// paired at 512 | 1448 | 4096 | 11585 | 32768 | 130816 keys (the verdict: >= 1.03 with the range's low end > 1.00, and
// N1 >= 0.99999); dpas_from = the smallest measured depth from which every verdict is dpas.
// One row per built (M, KV form): bf16 at M = 2..8, int8 at M = 2..4 (production's int8 decode lists).
// 2026-11-04 (operator: "make it work"): EVERY row on v2 until attn_verify_dpas is fixed - its error at 8 rows
// (below) has no known cause, and the kernel is the same code at every M; the measured thresholds stay as comments.
inline constexpr VerifyAttnRow kVerifyAttnTable[] = {
    {2, false, kDpasNever}, // was 11585; 0.51 | 0.56 | 0.98 | 1.26 | 1.26 | 1.39
    {3, false, kDpasNever}, // was 4096; 0.69 | 0.69 | 1.19 | 1.08 | 1.39 | 1.65
    {4, false, kDpasNever}, // was 4096; 0.90 | 0.76 | 1.67 | 1.43 | 1.79 | 2.06
    {5, false, kDpasNever}, // was 4096; 1.10 | 0.80 | 1.51 | 2.50 | 3.48 | 4.58  (512 dpas, 1448 v2: not monotone)
    // M = 6..8 (DFlash only) back on v2 (operator 2026-11-04): the f32 check of t6_usage-loader (DFlash K = 7,
    // first difference at generated position 103) puts the DPAS path's id at f32 rank 3, 4.0 below the top (the
    // top's own margin 3.44; plain = the f32 top over rows 103..111) - not a rounding near-tie; the kernel fix is
    // queued. The timing verdicts stay recorded: 6 dpas from 0 (1.94 | 1.22 | 2.50 | 3.95 | 5.61 | 6.72),
    // 7 (2.29 | 1.68 | 3.19 | 4.85 | 6.74 | 7.47), 8 (2.96 | 3.49 | 6.45 | 6.11 | 8.41 | 9.46).
    {6, false, kDpasNever},
    {7, false, kDpasNever},
    {8, false, kDpasNever},
    // int8: at 4096 every M's worst row (q head 19, row 0) is 0.99998797 against the fp64 reference - below 0.99999,
    // and no further than pf_flash_attn_kv8 (the prefill's kernel, 0.99998792 on the same row). Operator ruling
    // 2026-11-03: the int8 N1 bar is "no further than pf_flash_attn_kv8 on the same row", so 4096 is a timing verdict.
    {2, true, kDpasNever}, // was 11585; 0.57 | 0.59 | 0.99 | 1.78 | 2.07 | 2.33
    {3, true, kDpasNever}, // was 4096; 0.72 | 0.66 | 1.16 | 1.48 | 1.75 | 1.96
    {4, true, kDpasNever}, // was 4096; 0.92 | 0.79 | 1.58 | 1.84 | 2.17 | 2.43
};

inline const VerifyAttnRow* verify_attn_row(uint32_t M, bool kv8) {
  for (const VerifyAttnRow& r : kVerifyAttnTable)
    if (r.m == M && r.kv8 == kv8) return &r;
  return nullptr;
}

inline const char* verify_attn_force_name(VerifyAttnForce f) {
  return f == VerifyAttnForce::V2 ? "v2" : f == VerifyAttnForce::Dpas ? "dpas" : "rule";
}

inline VerifyAttn verify_attn_kind(uint32_t M, uint32_t keys, bool kv8, uint32_t q_heads, uint32_t kv_heads,
                                   VerifyAttnForce f = VerifyAttnForce::Auto) {
  if (M < 2 || q_heads != 24 || kv_heads != 4) return VerifyAttn::V2;   // before the table or the override
  if (f == VerifyAttnForce::V2) return VerifyAttn::V2;
  const VerifyAttnRow* r = verify_attn_row(M, kv8);
  if (r == nullptr) return VerifyAttn::V2;
  if (f == VerifyAttnForce::Dpas) return VerifyAttn::Dpas;
  return r->dpas_from != kDpasNever && keys >= r->dpas_from ? VerifyAttn::Dpas : VerifyAttn::V2;
}

// dpas_from is 0 / 1 or never (or the row is absent): one list serves every depth.
inline bool verify_attn_depth_free(uint32_t M, bool kv8) {
  const VerifyAttnRow* r = verify_attn_row(M, kv8);
  return r == nullptr || r->dpas_from <= 1 || r->dpas_from == kDpasNever;   // keys >= 1 always
}

// The engines' second list (spec 28c §3; the operator's decision 3 of 2026-10-31): the key count from which the
// list at M must bind DPAS while the list captured by the rule (at keys 1) binds v2 - a finite dpas_from > 1 at
// Qwen3.8's heads - or kDpasNever when one list serves every depth.
inline uint32_t verify_attn_deep_from(uint32_t M, bool kv8, uint32_t q_heads, uint32_t kv_heads) {
  const VerifyAttnRow* r = verify_attn_row(M, kv8);
  if (r == nullptr || verify_attn_depth_free(M, kv8)) return kDpasNever;
  return verify_attn_kind(M, kDpasNever - 1, kv8, q_heads, kv_heads) == VerifyAttn::Dpas ? r->dpas_from : kDpasNever;
}

}  // namespace runtime
