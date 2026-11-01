#pragma once
// gemv_kind.h - which linear kernel a captured list binds, by its row count (spec 28 §2, plan 28b).
//
//   runtime::gemv_kind(M)   LinearKind::Gemv at M = 1, LinearKind::SmallM at M >= 2.
//
// The ONE rule, applied where a list binds a linear: runtime/capture.cc's gemv() / gemv_i8w() helpers (every
// decode, verify, tap-verify, stage and row list) and runtime/dflash/dflash_capture.cc's drafter linears and
// draft head. `Gemv` binds today's gemv / gemv_i8w binaries - every M = 1 list (plain decode, ingest, generate,
// the MTP draft list, the prefill tail's head) is bit for bit what it was (the enum and the rule itself live in
// kernels/linear_kind.h so the kernels-side name lists apply the same rule). `SmallM` binds gemm_sm / gemm_sm_i8w
// for the shapes of kernels::kSmallMTuning and gemv / gemv_i8w for the rest (kernels::int4_linear_variant /
// int8_linear_variant, src/kernels/kernels.h). Not a user flag and not an environment variable (the operator,
// 2026-10-27): the only override is `CaptureOptions::force_linear`, for tests and probes that compare the two
// kernels inside one list.
//
// Interface kept small on purpose: spec 28c's verify_attn_kind(M, keys, kv) sits beside this rule and reads
// nothing from it.
#include <cstdint>

#include "kernels/linear_kind.h"

namespace runtime {

using kernels::LinearKind;
using kernels::gemv_kind;   // constexpr LinearKind gemv_kind(unsigned M): Gemv at M = 1, SmallM at M >= 2

// Test / probe only: force one kernel at every M of a capture (`Auto` = gemv_kind).
enum class LinearForce : uint8_t { Auto, Gemv, SmallM };
struct CaptureOptions {
  LinearForce force_linear = LinearForce::Auto;
};
constexpr LinearKind linear_kind(uint32_t M, const CaptureOptions& o) {
  return o.force_linear == LinearForce::Gemv     ? LinearKind::Gemv
         : o.force_linear == LinearForce::SmallM ? LinearKind::SmallM
                                                 : gemv_kind(M);
}

}  // namespace runtime
