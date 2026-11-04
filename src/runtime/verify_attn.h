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

// docs/probes/probe-verify-attn-2026-11-03.md §2 / §5 (busy-box timing, card 1; to be re-checked on a quiet box): v2 /
// dpas paired at 512 | 1448 | 4096 | 11585 | 32768 | 130816 keys (the verdict: >= 1.03 with the range's low end > 1.00,
// and N1); dpas_from = the smallest measured depth from which every verdict is dpas.
// One row per built (M, KV form): bf16 at M = 2..8, int8 at M = 2..4 (production's int8 decode lists).
// 2026-11-04: the error that put every row on v2 for a day (DFlash K = 7, t6_usage-loader, position 103: f32 rank 3,
// 4.0 below the top) was P's bf16 rounding in attn_verify_dpas, not a compiler (docs/07-open-questions.md §11); the
// kernel now carries P as hi + lo (PSPLIT=1). Re-timed at that kernel (§5), every row's dpas_from is what it was.
inline constexpr VerifyAttnRow kVerifyAttnTable[] = {
    {2, false, 11585},  // 0.49 | 0.53 | 0.92 | 1.20 | 1.22 | 1.36
    {3, false, 4096},   // 0.68 | 0.71 | 1.17 | 1.08 | 1.37 | 1.69
    {4, false, 4096},   // 0.88 | 0.74 | 1.59 | 1.37 | 1.71 | 2.08
    {5, false, 4096},   // 1.10 | 0.79 | 1.47 | 2.44 | 3.34 | 4.41  (512 dpas, 1448 v2: not monotone)
    {6, false, 0},      // 1.91 | 1.20 | 2.28 | 3.90 | 5.36 | 6.21  (DFlash only, M = 6..8)
    {7, false, 0},      // 2.22 | 1.70 | 2.90 | 4.71 | 6.34 | 7.06
    {8, false, 0},      // 2.89 | 3.43 | 6.10 | 5.77 | 8.22 | 8.91
    // int8: N1 now holds at every depth (worst 0.9999984 at 130816; the 4096 row that needed the operator ruling of
    // 2026-11-03 - P in bf16 - is 0.9999999). M = 2 at 4096 is 1.01 with its low end 0.999: v2 there.
    {2, true, 11585},   // 0.57 | 0.59 | 1.01 | 1.75 | 2.05 | 2.36
    {3, true, 4096},    // 0.74 | 0.69 | 1.32 | 1.50 | 1.83 | 2.02
    {4, true, 4096},    // 0.91 | 0.82 | 1.66 | 1.85 | 2.19 | 2.42
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
