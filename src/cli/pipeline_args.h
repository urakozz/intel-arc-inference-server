#pragma once
// Spec 16b: b70-decode's pipeline-parallel flags, as device-free parsing and refusals
// (tests/cli/pipeline_args_test.cc runs them on any host; the device flow is
// cli/pipeline_decode.h).
//
//   --pipeline 1|2              devices the model's layers are split over (default 1:
//                               today's engine, untouched)
//   --pipeline-split auto|N     device 0 runs layers [0, N), device 1 [N, layers); auto
//                               (default) balances the bytes each card holds
//                               (runtime::pp_auto_split)
//   --pipeline-handoff copy|peer  how the residual crosses (runtime::PpHandoff); copy
//                               (default) or peer
//
// **Why not `--pp 2`, the spelling spec 16 uses.** b70-decode has had `--pp N` since spec 2
// - the bench's prefill length (llama-bench's "pp", prompt processing) - with `--pp-chunk`
// and `--pp-backend` beside it, and every bench script and queue row uses it. `--pp 2` for
// pipeline parallel would silently mean "prefill 2 ids" to them, so the flags say the whole
// word. Spec 16's text keeps "PP" for the concept.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime/pipeline_plan.h"

namespace cli {

struct PipelineArgs {
  uint32_t devices = 1;
  bool split_auto = true;
  uint32_t split = 0;   // the explicit N; 0 with split_auto
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool have_split = false, have_handoff = false;
  bool on() const { return devices > 1; }
};

inline uint32_t parse_pipeline_devices(const std::string& v) {
  if (v == "1") return 1;
  if (v == "2") return 2;
  throw std::runtime_error("--pipeline expects 1 or 2 (spec 16 builds and gates two devices), got '" +
                           v + "'");
}

inline void parse_pipeline_split(const std::string& v, PipelineArgs& a) {
  a.have_split = true;
  if (v == "auto") {
    a.split_auto = true;
    a.split = 0;
    return;
  }
  if (v.empty() || v.size() > 5 || v.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("--pipeline-split expects auto or a layer index, got '" + v + "'");
  const unsigned long n = std::stoul(v);
  if (n == 0)
    throw std::runtime_error("--pipeline-split 0 would leave device 0 no layer; the split is the"
                             " first layer of device 1");
  a.split_auto = false;
  a.split = static_cast<uint32_t>(n);
}

inline runtime::PpHandoff parse_pipeline_handoff(const std::string& v, PipelineArgs& a) {
  a.have_handoff = true;
  if (!runtime::parse_pp_handoff(v, a.handoff))
    throw std::runtime_error("--pipeline-handoff expects copy or peer, got '" + v + "'");
  return a.handoff;
}

// What else b70-decode was asked for, for the refusals below.
struct PipelineContext {
  bool prefill = false;      // --prefill
  bool bench_prefill = false;   // --bench --pp N (the prefill length)
  bool mtp = false;          // --mtp K|auto
  bool profile = false;      // --profile
  bool device = false;       // --device N
};

// Every refusal that needs no device. --pipeline 1 refuses nothing (it is today's engine);
// the two sub-flags belong to --pipeline 2.
inline void check_pipeline(const PipelineArgs& a, const PipelineContext& c) {
  if (!a.on()) {
    if (a.have_split)
      throw std::runtime_error("--pipeline-split belongs to --pipeline 2 (the layer device 1 starts at)");
    if (a.have_handoff)
      throw std::runtime_error("--pipeline-handoff belongs to --pipeline 2 (how the residual crosses)");
    return;
  }
  if (c.prefill || c.bench_prefill)
    throw std::runtime_error(
        "--pipeline 2 decodes only (spec 16b): a prefill across two cards is spec 16c's chunk "
        "pipeline. --ids ingests the prompt through the two decode lists, one replay per id; "
        "--bench takes --depth, not --pp");
  if (c.mtp)
    throw std::runtime_error("--pipeline 2 with --mtp is spec 16d (the MTP head and its embedding "
                             "on device 1); drop --mtp");
  if (c.profile)
    throw std::runtime_error("--profile instruments one device's list; it has no --pipeline 2");
  if (c.device)
    throw std::runtime_error("--pipeline 2 runs on GPUs 0 and 1 of what Level Zero shows "
                             "(ZE_AFFINITY_MASK picks which two); --device names one card - drop it");
}

// The device count the driver shows (l0::Context::gpu_count), held to two before anything
// is opened (P4: a missing card is a clear error, never a hang or a single-card run).
inline void require_two_devices(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--pipeline 2 needs two GPUs and Level Zero shows " +
                             std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

}  // namespace cli
