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
  // Spec 2.1's L0 linear, itemised into its two kernels. Their sum is the row
  // earlier records call `linear_l0`. Split because the GEMM's DPAS floor is
  // derivable (2.M.SUM(K.N) / the measured rate) but the dequant round-trip's
  // cost was not measured, and that residual is the largest unattributed block
  // in the walk. Closing a phase per SLAB costs one wait per launch pair, so
  // the wait column and the instrumented wall inflate heavily here; only the
  // L0_gpu_ms column of these two rows is meant to be read.
  kSlabDequant, // pf_dequant_slab, one 1024-column slab
  kSlabGemm,    // pf_gemm over that slab
  // Spec 5's h8 linear (runtime/prefill/int8.cc), itemised the same way: the
  // rotating activation quantiser once per linear, then per slab the rotating
  // requant and the i8 x i8 GEMM.
  kI8Quant,     // pf_quant_had
  kI8Requant,   // pf_requant_rot, one 1024-column slab
  kI8Gemm,      // pf_gemm_i8 (plain or SiLU) over that slab
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
  kAttnFlash,   // pf_flash_attn (spec 6), in place of QK^T + softmax + PV
  kHead,        // step_head: final norm, lm_head, both argmax stages
  // Spec 15d: a MoE layer's FFN on the prefill path (runtime/prefill/moe.cc).
  kMoeRoute,    // the router || shared-gate GEMV over the chunk + moe_route
  kMoeSort,     // pf_moe_sort: histogram, prefix, scatter, tile table
  kMoeGather,   // pf_moe_gather (bf16) / pf_moe_gather_i8 (+ the h8 quantiser)
  kMoeWeights,  // the expert weights into the grouped GEMMs' B form (requant / dequant)
  kMoeGemm,     // the grouped GEMMs, gate||up and down
  kMoeCombine,  // pf_moe_combine (the weighted sum and the residual fold)
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
