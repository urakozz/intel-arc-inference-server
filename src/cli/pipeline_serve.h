#pragma once
// Spec 16d: b70-serve under --parallel pp - what the server refuses before any device, as device-free
// checks (tests/cli/pipeline_args_test.cc runs them on any host; b70-serve's own refusals are
// card tests, cli_reject_serve_pp_*). The device flow is b70_serve.cc's serve_pipeline().
//
// b70-serve takes b70-decode's flags (cli/pipeline_args.h): --parallel auto|1|pp, and the overrides
// --pipeline-split auto|N (pp's). Under pp:
//
//   served   Qwen3.8 and Ornith (runtime::PipelineEngine: the descriptor gives the
//            split), with --max-len auto|N (both cards, each with the reserve), --prefill-backend
//            l0|l0-int8, --kv-cache bf16|int8, the prefix cache (--prefix-cache-gb auto|N,
//            --prefix-split-last: snapshots in the one-card layouts, the block hook from the
//            pipeline's shadows), --spec mtp (the head on device 1);
//   refused  B70_PREFILL_PROFILE=1 (its phase waits would serialise the pipeline).
//   Kolibri-1 (spec 20e) is not this path: b70-serve dispatches it to its own engine before these
//            checks (cli/kolibri_serve.h, runtime::kolibri::KolibriEngine on two cards by default), and
//            cli::kolibri::check_args holds its parallel rules; for model_type kolibri1 this refuses nothing.
#include <stdexcept>
#include <string>

#include "cli/pipeline_args.h"
#include "runtime/prefill_backend.h"

namespace cli {

struct ServePipelineContext {
  std::string model_type;   // config.json's (empty when there is none)
  bool mtp = false;         // --spec mtp|lookup (the head is loaded)
  bool prefill_profile = false; // B70_PREFILL_PROFILE=1
};

inline void check_serve_pipeline(const PipelineArgs& a, const ServePipelineContext& c) {
  // One card: the override flag alone is refused, as b70-decode refuses it.
  PipelineContext pc;
  pc.prefill = true;   // the server prefills every request
  pc.mtp = c.mtp;
  if (c.model_type == "kolibri1") return;   // its own engine and rules (cli::kolibri::check_args)
  if (!a.on()) {
    check_pipeline(a, {});
    return;
  }
  check_pipeline(a, pc);
  if (c.prefill_profile)
    throw std::runtime_error("--parallel pp prefill is not profiled (B70_PREFILL_PROFILE=1): the phase "
                             "waits would serialise the pipeline - unset it");
}

}  // namespace cli
