#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "loader/loader.h"
#include "runtime/buffers.h"

namespace runtime {

// The decode step, captured once. Owns every Module/Kernel the list references
// (Level Zero resolves a launch's arguments at append time - proven by
// tests/l0/arg_capture_test.cc - but the *handles* must outlive the list) and
// the closed list itself. Replay is `Queue::execute(step.list, &fence)`; the
// host mutates nothing in it, ever.
struct CapturedStep {
  l0::CmdList list;                                            // closed, in-order
  size_t kernel_count = 0;                                     // 645 at M = 1 (spec §9.1)
  std::map<std::string, std::unique_ptr<l0::Module>> modules;  // by variant name
  std::vector<std::unique_ptr<l0::Kernel>> kernels;            // append order
};

// Builds the whole per-token list against `m`'s weights and `b`'s allocations
// and closes it. Throws (std::runtime_error / l0::Error / std::out_of_range) on
// any mismatch it can see at capture: a missing device binary, a buffer whose
// size disagrees with the per-layer slice arithmetic, a `max_len` that is not
// the one the attention variants were compiled for.
//
// debug_resid: when non-null (bf16 [64][M][5120]), append a device-to-device
// copy of `resid` after every layer - the golden gate's per-layer tap. Copies
// are commands, not kernels, so `kernel_count` does not count them and
// determinism is unaffected. The tap runs after the layer's LAST kernel, and
// the residual stream is only advanced by `prep_res_norm`, so tap[L] holds the
// hidden state with layer L's *mixer* contribution folded in and layer L's MLP
// contribution still sitting un-folded in `partials` (layer L+1's leading
// `prep_res_norm` folds it). No tap holds the final hidden state - the final
// `prep_res_norm` folds layer 63's MLP straight into `x` for lm_head.
CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid = nullptr);

}  // namespace runtime
