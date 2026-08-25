#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "loader/small_layout.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"

namespace loader {

// `LoadedModel::linears` is keyed by (layer, id); tensors that belong to no
// layer (lm_head) use this sentinel instead of a second map.
constexpr uint32_t kTopLevel = 65535;

struct DeviceWeight {
  l0::Mem mem;                 // the canonical bytes on device
  model::GemvShape shape;      // K, N, S, layout as the kernel variant needs
  model::WeightKind kind;
};

struct SmallTensors {          // per layer: everything that is not a GEMV weight
  l0::Mem norms;               // input_ln(1+w) ‖ post_ln(1+w) fp32 [2][5120]
  l0::Mem gdn;                 // the layer-kind block: GDN = conv fp32
                               //   [10240][4] ‖ negA fp32 [48] ‖ dt_bias fp32
                               //   [48] ‖ gated-norm plain w bf16 [128];
                               //   FA = q_norm ‖ k_norm (1+w) fp32 [2][256].
                               // Offsets: loader/small_layout.h (one place).
};

struct LoadReport {            // printed by load(); asserted by the checkpoint test
  size_t int4_bytes = 0, scale_bytes = 0, bf16_linear_bytes = 0;
  size_t embed_bytes = 0, lm_head_bytes = 0, pad_bytes = 0;
  // Every non-GEMV byte allocated, the RoPE table included. The printed
  // `small` line nets rope out (it is resident but not per-token traffic) and
  // reports it on its own line; total() and this field do not.
  size_t small_bytes = 0;
  size_t total() const;             // the seven byte fields above
  // **`W`, as this load actually measured it** - the bytes a decode step
  // streams: everything in total() except `embed_tokens` (gathered one row per
  // token) and the RoPE table (~256 B per token). It is a FIELD rather than a
  // constant because it is no longer one number: the published checkpoint's
  // bf16 `lm_head` makes it 15.540 GB and a packed one makes it 13.673 GB
  // (both measured 2026-08-26). Anything that divides by W - the MBU line the
  // bench prints, the roofline - has to read it from here, or it reports one
  // checkpoint's efficiency against another's denominator.
  size_t read_per_token = 0;
  size_t unconsumed = 0;            // checkpoint tensors nothing loaded (must be 0)
  double seconds = 0;
};

struct LoadedModel {
  std::map<std::pair<uint32_t, model::LinearId>, DeviceWeight> linears;  // layer kTopLevel = lm_head
  std::vector<SmallTensors> layer_small;   // [64]
  l0::Mem embed;                            // bf16 [248320][5120] row-major (gathered)
  l0::Mem final_norm;                       // pre-lm_head RMSNorm, (1+w) fp32 [5120]
  l0::Mem rope;                             // fp32 [max_len][2][32] cos/sin pairs
  LoadReport report;
  uint32_t max_len = 0;                     // the max_len this model was loaded with
};

// Loads the qwen3_5 checkpoint at `snapshot_or_repo` (resolve_snapshot rules)
// into ctx's device. Skips model.visual.* and (v1) mtp.*. Strips the
// "model.language_model." prefix so model::Qwen35's layer-relative names bind.
// Asserts quant invariants and the doc-03 shape table; throws by name on any
// mismatch. max_len sizes the RoPE table only (default 16384).
LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len = 16384);

}  // namespace loader
