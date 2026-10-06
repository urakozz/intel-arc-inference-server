#pragma once
// Spec 16b: b70-decode's pipeline-parallel flags, as device-free parsing and refusals
// (tests/cli/pipeline_args_test.cc runs them on any host; the device flow is
// cli/pipeline_decode.h).
//
//   --pp N, --pipeline-parallel-size N
//                               devices the model's layers are split over, 1 or 2 (default 1:
//                               today's engine, untouched). vLLM's spelling and meaning
//   --pipeline-split auto|N     device 0 runs layers [0, N), device 1 [N, layers); auto
//                               (default) balances the bytes each card holds
//                               (runtime::pp_auto_split)
//   --pipeline-handoff copy|peer  how the residual crosses (runtime::PpHandoff); copy
//                               (default) or peer
//
// **`--pp` used to be the bench's prefill length.** Until 2026-10-06 b70-decode's `--pp N`
// was llama-bench's "pp" (prompt processing), with `--pp-chunk` / `--pp-backend` beside it,
// and this flag was spelled `--pipeline 1|2`. The operator moved b70-decode to vLLM's
// convention: `--pp` / `--pipeline-parallel-size` is pipeline parallel, and the prefill flags
// are `--prefill-length`, `--prefill-chunk`, `--prefill-backend`. An old command line's
// `--pp 4096` must not become a 4096-way request, so any value but 1 or 2 is refused naming
// `--prefill-length` (parse_pipeline_devices), and the old names are refused as unknown
// options naming their new spelling (cli/renamed_flags.h).
#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime/pipeline_plan.h"
#include "runtime/prefill_backend.h"

namespace cli {

struct PipelineArgs {
  uint32_t devices = 1;
  bool split_auto = true;
  uint32_t split = 0;   // the explicit N; 0 with split_auto
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool have_split = false, have_handoff = false;
  bool on() const { return devices > 1; }
};

// `flag` is the spelling given (--pp or --pipeline-parallel-size), for the message.
inline uint32_t parse_pipeline_devices(const std::string& flag, const std::string& v) {
  if (v == "1") return 1;
  if (v == "2") return 2;
  throw std::runtime_error(flag + " is the pipeline-parallel size, 1 or 2 (spec 16 builds and gates "
                           "two devices), got '" + v + "'; the bench's prefill length is "
                           "--prefill-length N");
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
  bool bench_prefill = false;   // --bench --prefill-length N
  bool mtp = false;          // --mtp K|auto
  bool profile = false;      // --profile
  bool device = false;       // --device N
  // Spec 16c: what a prefill would run on (read with prefill / bench_prefill only):
  // --prefill-backend (or the build's default) and B70_PREFILL_ATTN=composed.
  runtime::PrefillBackend prefill_backend = runtime::PrefillBackend::L0Int8;
  bool composed_attn = false;
};

// Every refusal that needs no device. --pp 1 refuses nothing (it is today's engine); the two
// sub-flags belong to --pp 2.
inline void check_pipeline(const PipelineArgs& a, const PipelineContext& c) {
  if (!a.on()) {
    if (a.have_split)
      throw std::runtime_error("--pipeline-split belongs to --pp 2 (the layer device 1 starts at)");
    if (a.have_handoff)
      throw std::runtime_error("--pipeline-handoff belongs to --pp 2 (how the residual crosses)");
    return;
  }
  // Spec 16c: --prefill and --bench --prefill-length run the two-card chunk pipeline, on the
  // L0 backends' flash attention only.
  if ((c.prefill || c.bench_prefill) && !runtime::is_l0(c.prefill_backend))
    throw std::runtime_error(std::string("--pp 2 prefills on the L0 backends (l0, l0-int8), not ") +
                             runtime::prefill_backend_name(c.prefill_backend) +
                             ": sycl-tla's walk waits on the host between runtimes and has no "
                             "two-card pipeline (spec 16c); --prefill-backend l0 or l0-int8");
  if ((c.prefill || c.bench_prefill) && c.composed_attn)
    throw std::runtime_error("--pp 2 prefills with the flash attention only: unset "
                             "B70_PREFILL_ATTN=composed (spec 16c)");
  if (c.mtp)
    throw std::runtime_error("--pp 2 with --mtp is spec 16d (the MTP head and its embedding "
                             "on device 1); drop --mtp");
  if (c.profile)
    throw std::runtime_error("--profile instruments one device's list; it has no --pp 2");
  if (c.device)
    throw std::runtime_error("--pp 2 runs on GPUs 0 and 1 of what Level Zero shows "
                             "(ZE_AFFINITY_MASK picks which two); --device names one card - drop it");
}

// The device count the driver shows (l0::Context::gpu_count), held to two before anything
// is opened (P4: a missing card is a clear error, never a hang or a single-card run).
inline void require_two_devices(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--pp 2 needs two GPUs and Level Zero shows " +
                             std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

}  // namespace cli
