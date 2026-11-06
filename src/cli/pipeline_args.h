#pragma once
// The parallel mode and the pipeline's override flags, shared by b70-serve and b70-decode, as
// device-free parsing and refusals (tests/cli/pipeline_args_test.cc runs them on any host; the
// device flows are cli/pipeline_decode.h and b70_serve.cc's serve_pipeline):
//
//   --parallel auto|1|pp        auto (default): the faster choice by fit - one card when the model
//                               and the asked context fit it, else pipeline parallel (each model
//                               family's rule: cli/pipeline_settle.h parallel_auto, Kolibri-1's and
//                               Nemotron 3.5 Bolt's two cards for the whole model); 1: one card;
//                               pp: the model's layers split over GPUs 0 and 1 of what Level Zero
//                               shows (spec 16). `tp` is the name kept for tensor parallel (spec 17),
//                               refused until it is built.
//   --pipeline-split auto|N     the override, for experiments: device 0 runs layers [0, N), device 1
//                               [N, layers); auto (default) balances the bytes each card holds
//                               (runtime::pp_auto_split). Given under --parallel auto, it means pp.
//   --pipeline-handoff copy|peer  how the residual crosses (runtime::PpHandoff); copy (default) or
//                               peer. Given under --parallel auto, it means pp.
//
// (The zero-config cleanup, operator 2026-11-06, replaced `--pp N` / `--pipeline-parallel-size N`
// with --parallel and dropped the refusals of the 2026-10-06 rename's old spellings.)
#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime/pipeline_plan.h"
#include "runtime/prefill_backend.h"

namespace cli {

// The parallel mode as given; `devices` is the resolved device count (1 or 2).
enum class Parallel { Auto, One, Pp };

struct PipelineArgs {
  Parallel mode = Parallel::Auto;
  uint32_t devices = 1;
  bool split_auto = true;
  uint32_t split = 0;   // the explicit N; 0 with split_auto
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool have_split = false, have_handoff = false;
  bool on() const { return devices > 1; }
  // The mode was named (1 or pp) - each family's default applies otherwise.
  bool given() const { return mode != Parallel::Auto; }
};

inline void parse_parallel(const std::string& v, PipelineArgs& a) {
  if (v == "auto") {
    a.mode = Parallel::Auto;
    a.devices = 1;
  } else if (v == "1") {
    a.mode = Parallel::One;
    a.devices = 1;
  } else if (v == "pp") {
    a.mode = Parallel::Pp;
    a.devices = 2;
  } else if (v == "tp") {
    throw std::runtime_error("--parallel tp: tensor parallel is not built yet (spec 17) - --parallel auto, 1 or pp");
  } else {
    throw std::runtime_error("--parallel expects auto, 1 or pp, got '" + v + "'");
  }
}

// The override flags under --parallel auto mean pipeline parallel (they have no meaning on one card).
inline void resolve_overrides(PipelineArgs& a) {
  if (a.mode == Parallel::Auto && (a.have_split || a.have_handoff)) {
    a.mode = Parallel::Pp;
    a.devices = 2;
  }
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
  bool mtp = false;          // --spec mtp (spec 16d: allowed with pipeline parallel)
  bool profile = false;      // --profile
};

// Every refusal that needs no device. One card refuses nothing; the two override flags belong
// to --parallel pp.
inline void check_pipeline(const PipelineArgs& a, const PipelineContext& c) {
  if (!a.on()) {
    if (a.have_split)
      throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at)");
    if (a.have_handoff)
      throw std::runtime_error("--pipeline-handoff belongs to --parallel pp (how the residual crosses)");
    return;
  }
  // Spec 16c: --prefill and --bench --prefill-length run the two-card chunk pipeline (the flash
  // attention, the only prefill attention).
  // (Spec 16d lifted 16b's --spec mtp refusal: the head, its embedding replica and the draft lists
  // run on device 1, the verify lists on both.)
  if (c.profile)
    throw std::runtime_error("--profile instruments one device's list; it has no --parallel pp");
}

// The device count the driver shows (l0::Context::gpu_count), held to two before anything
// is opened (P4: a missing card is a clear error, never a hang or a single-card run).
inline void require_two_devices(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--parallel pp needs two GPUs and Level Zero shows " +
                             std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

}  // namespace cli
