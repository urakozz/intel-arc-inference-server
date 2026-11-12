#pragma once
#include <cstdlib>
#include <cstring>

// MoE-model prefill phase 2 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): the MoE models'
// walks (Ornith's runtime/prefill, Flash-Next's runtime/qwen4exp) bind the phase 2 kernels unless
// B70_PF_PHASE2=old is in the environment (tests and probes only; Task 9 deletes it). Read once.
namespace runtime::prefill {
inline bool pf_phase2_on() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PF_PHASE2");
    return !(v != nullptr && std::strcmp(v, "old") == 0);
  }();
  return on;
}
// Task 3's one-launch intra-chunk kernel in its D2 form (pf_gdn_intra_hl + pf_gdn_scan_hl) is NOT bound by
// default: it missed the f32-judged token gates (2026-11-12 record) and waits for the operator. B70_PF_GDN_INTRA=hl
// binds it on a phase 2 walk (experiments only).
inline bool pf_gdn_intra_on() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PF_GDN_INTRA");
    return v != nullptr && std::strcmp(v, "hl") == 0;
  }();
  return on && pf_phase2_on();
}
// Task 6 under D3: the router || shared gate as bf16 DPAS (pf_route.cl ROUTE_DPAS), judged - opt-in until its
// gates pass: B70_PF_ROUTE=dpas.
inline bool pf_route_dpas_on() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PF_ROUTE");
    return v != nullptr && std::strcmp(v, "dpas") == 0;
  }();
  return on;
}
}  // namespace runtime::prefill
