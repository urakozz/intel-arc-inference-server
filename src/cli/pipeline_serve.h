#pragma once
// Spec 16d: b70-serve --pp 2 - what the server refuses before any device, as device-free
// checks (tests/cli/pipeline_args_test.cc runs them on any host; b70-serve's own refusals are
// box tests, cli_reject_serve_pp_*). The device flow is b70_serve.cc's serve_pipeline().
//
// b70-serve takes b70-decode's three flags (cli/pipeline_args.h): --pp N /
// --pipeline-parallel-size N (1, the default, is today's one-card server, untouched; 2 splits
// the model over GPUs 0 and 1 of what Level Zero shows), --pipeline-split auto|N and
// --pipeline-handoff copy|peer (both belong to --pp 2). Under --pp 2:
//
//   served   Qwen3.8, Agnes and Ornith (runtime::PipelineEngine: the descriptor gives the
//            split), with --max-len auto|N and --mem-reserve-gb (both cards, each with the
//            reserve), --prefill-backend l0|l0-int8, --kv-cache bf16|int8, --lm-head, the
//            prefix cache (--prefix-cache-gb auto|N, --prefix-split-last: snapshots in the
//            one-card layouts, the block hook from the pipeline's shadows), --mtp K|auto /
//            --spec mtp (the head on device 1), --draft-vocab, --spec lookup;
//   refused  K2-Horizon (its own engine, runtime/k2: two-card K2 is future work - serve it on
//            one card), Kolibri-1 (not served at all yet; its serving, two cards included, is
//            spec 20e), --device N (names one card: ZE_AFFINITY_MASK picks the two), a
//            sycl-tla prefill and B70_PREFILL_ATTN=composed (no two-card walk, spec 16c),
//            B70_PREFILL_REPLAY=1 (the recordings are one card's walk) and
//            B70_PREFILL_PROFILE=1 (its phase waits would serialise the pipeline).
#include <stdexcept>
#include <string>

#include "cli/pipeline_args.h"
#include "runtime/prefill_backend.h"

namespace cli {

struct ServePipelineContext {
  std::string model_type;   // config.json's (empty when there is none)
  bool device = false;      // --device N given
  bool mtp = false;         // --mtp K|auto, --spec mtp|lookup (the head is loaded)
  runtime::PrefillBackend prefill_backend = runtime::PrefillBackend::L0Int8;   // what it prefills on
  bool composed_attn = false;   // B70_PREFILL_ATTN=composed
  bool prefill_replay = false;  // B70_PREFILL_REPLAY=1
  bool prefill_profile = false; // B70_PREFILL_PROFILE=1
};

inline void check_serve_pipeline(const PipelineArgs& a, const ServePipelineContext& c) {
  // --pp 1: the sub-flags alone are refused, as b70-decode refuses them.
  PipelineContext pc;
  pc.prefill = true;   // the server prefills every request
  pc.mtp = c.mtp;
  pc.device = c.device;
  pc.prefill_backend = c.prefill_backend;
  pc.composed_attn = c.composed_attn;
  if (!a.on()) {
    check_pipeline(a, {});
    return;
  }
  if (c.model_type == "k2_horizon")
    throw std::runtime_error("--pp 2: K2-Horizon runs its own engine (runtime/k2), which has no "
                             "two-card path yet (spec 16 §8: K2 across two cards is future work) - "
                             "serve it on one card (drop --pp)");
  if (c.model_type == "kolibri1")
    throw std::runtime_error("--pp 2: Kolibri-1 (model_type kolibri1) is not served yet - its serving, "
                             "on two cards included, is spec 20e; b70-decode --pp 2 runs it (spec 20c)");
  check_pipeline(a, pc);   // sycl-tla, composed attention, --device - by name
  if (c.prefill_replay)
    throw std::runtime_error("--pp 2 has no prefill replay (B70_PREFILL_REPLAY=1): the recordings "
                             "are one card's walk (spec 16c) - unset it");
  if (c.prefill_profile)
    throw std::runtime_error("--pp 2 prefill is not profiled (B70_PREFILL_PROFILE=1): the phase "
                             "waits would serialise the pipeline (spec 16c) - unset it");
}

}  // namespace cli
