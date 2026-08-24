#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"

namespace loader {

struct DeviceWeight {
  l0::Mem mem;                 // the canonical bytes on device
  model::GemvShape shape;      // K, N, S, layout as the kernel variant needs
  model::WeightKind kind;
};

struct SmallTensors {          // per layer: everything that is not a GEMV weight
  l0::Mem norms;               // input_ln(1+w) ‖ post_ln(1+w) bf16 [2][5120]
  l0::Mem gdn;                 // GDN only: conv fp32 [10240][4] ‖ negA fp32 [48]
                               //   ‖ dt_bias fp32 [48] ‖ gated-norm w bf16 [128]
                               //   ‖ q_norm/k_norm (FA: (1+w) bf16 [2][256]) - one
                               //   packed block per layer; offsets in loader.cc
};

struct LoadReport {            // printed by load(); asserted by the test
  size_t int4_bytes = 0, scale_bytes = 0, bf16_linear_bytes = 0;
  size_t embed_bytes = 0, lm_head_bytes = 0, small_bytes = 0, pad_bytes = 0;
  size_t total() const;             // the seven byte fields above
  size_t unconsumed = 0;            // checkpoint tensors nothing loaded (must be 0)
  double seconds = 0;
};

struct LoadedModel {
  std::map<std::pair<uint32_t, model::LinearId>, DeviceWeight> linears;  // layer 65535 = top level (lm_head)
  std::vector<SmallTensors> layer_small;   // [64]
  l0::Mem embed;                            // bf16 [248320][5120] row-major (gathered)
  l0::Mem final_norm;                       // pre-lm_head RMSNorm, (1+w) bf16 [5120]
  l0::Mem rope;                             // fp32 [max_len][2][32] cos/sin pairs
  LoadReport report;
};

// Loads the qwen3_5 checkpoint at `snapshot_or_repo` (resolve_snapshot rules)
// into ctx's device. Skips model.visual.* and (v1) mtp.*. Strips the
// "model.language_model." prefix so model::Qwen35's layer-relative names bind.
// Asserts quant invariants and the doc-03 shape table; throws by name on any
// mismatch. max_len sizes the RoPE table only (default 16384).
LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len = 16384);

}  // namespace loader
