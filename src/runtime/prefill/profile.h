#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/prefill/context.h"

// Per-phase diagnostic timing of the prefill walk, off unless
// `B70_PREFILL_PROFILE=1` is in the environment.
//
// Context timestamps L0 kernels and records host launch submission separately.
// A wait measures only residual synchronization after submission; the GPU can
// already have executed while the host was submitting. SYCL kernels are not
// covered by L0 events. None of these columns is interchangeable or additive.
//
// Events and extra phase waits perturb the walk. The CLI supplies this SAME
// instrumented call's wall time, not an independent plain run. The report must
// not describe the sum of waits as a wall-time upper bound or its difference
// from call wall as instrumentation overhead.
namespace runtime::prefill {

enum class Phase : uint32_t {
  kNorm,        // pf_res_fold + pf_norm_finish
  kDequant,     // pf_dequant_tile
  kGemm,        // the four int4 linears' gemm_bf16
  kLinearL0,    // spec 2.1: pf_dequant_slab + pf_gemm, one linear
  kAbGdn,       // pf_ab_proj (gdn_chunk's ten launches are itemised below)
  // `gdn_chunk`'s ten launches, one bucket each. Itemised rather than lumped
  // because the ledger's GDN term (15.4 ms/chunk) predates `gdn_chunk` existing
  // -- it was a Stage-0 projection of decode's widened `gdn_step` -- and a
  // 66x miss against it has to be attributed to a kernel, not to a stage.
  kGdnSeed, kGdnConv, kGdnL2, kGdnGate, kGdnA, kGdnSolve, kGdnWu, kGdnA2,
  kGdnScan, kGdnHead,
  kSilu,        // pf_silu_mul
  kAttnPrep,    // pf_attn_prep_q16
  kAttnQk,      // the QK^T batched GEMM
  kAttnSm,      // pf_softmax_causal
  kAttnPv,      // the PV batched GEMM
  kAttnGate,    // pf_attn_gate
  kHead,        // step_head: final norm, lm_head, both argmax stages
  kCount
};

// True iff B70_PREFILL_PROFILE=1. Read once, at first use.
bool profile_enabled();

// `cx.wait()`, timed into `p` when profiling is on and a bare wait otherwise.
void timed_wait(Context& cx, Phase p);
// A wait the walk pays ONLY to close a phase for the instrument. A no-op when
// profiling is off, so the unprofiled walk is exactly the walk A24 describes.
void profile_wait(Context& cx, Phase p);

// Zero the accumulators (the CLI does this before the run it reports).
void profile_reset();
// Print residual waits, L0 GPU timestamps and L0 host submission separately.
// `wall_ms` is the actual instrumented call wall time, or 0 if unavailable.
void profile_report(const char* label, double wall_ms);

}  // namespace runtime::prefill
