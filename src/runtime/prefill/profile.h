#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/prefill/context.h"

// A per-phase wall-clock attribution of the prefill walk, off unless
// `B70_PREFILL_PROFILE=1` is in the environment.
//
// **Why a wall clock and not device timestamps.** The walk is already a strict
// alternation of "append launches" and "drain both queues" - rulings A23/A24
// put a host `Context::wait()` at every L0<->SYCL boundary because this device
// has one compute queue and no cross-runtime dependency. So the wall time of a
// `wait()` IS the device time of everything queued since the previous drain,
// plus that boundary's own cost, and no event pool is needed to attribute it.
// `--profile`'s `ProfileEvents` machinery cannot help here anyway: it belongs
// to a captured list and the prefill walk is not one (docs/07 open question).
//
// **The instrument perturbs, and by how much is reported rather than argued.**
// In profile mode the walk closes each L0-only section with an EXTRA wait it
// would not otherwise pay, so a profiled chunk is slower than a plain one. The
// report prints the profiled total beside the plain one from the same binary's
// unprofiled run; a term is only worth reading if the two agree to a few
// percent.
namespace runtime::prefill {

enum class Phase : uint32_t {
  kNorm,        // pf_res_fold + pf_norm_finish
  kDequant,     // pf_dequant_tile
  kGemm,        // the four int4 linears' gemm_bf16
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
// Print the table to stderr: per phase, total ms, share, and the wait count.
// `label` names the run; `plain_ms` is the same run's unprofiled wall time, or
// 0 if there is none to compare against.
void profile_report(const char* label, double plain_ms);

}  // namespace runtime::prefill
