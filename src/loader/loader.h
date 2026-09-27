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
  // Layout 0's real, independent f16 [K/64][N] allocation. Null for layout 1
  // (scales are inline in mem) and for bf16 weights.
  std::unique_ptr<l0::Mem> scales;
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

// Spec 8 §3.1: the checkpoint's multi-token-prediction head, loaded only when asked
// (`load(..., mtp = true)`). All bf16, in gemv_bf16's tiled layout
// (common::repack_bf16_tiled), [N][K] logically, S = 1, layout 0 (a filler for bf16).
// It shares `embed` and the `lm_head` linear with the main model.
struct MtpHead {
  DeviceWeight fc;        // K 10240 -> N 5120, over cat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h))
  DeviceWeight qkv;       // K 5120 -> N 14336: q_proj (q||gate per head) || k_proj || v_proj
  DeviceWeight o;         // K 6144 -> N 5120
  DeviceWeight gate_up;   // K 5120 -> N 34816, gate/up in alternating 16-column blocks
  DeviceWeight down;      // K 17408 -> N 5120
  l0::Mem norms;          // fp32 (1 + w) [5][5120], kMtpNorm* below
  l0::Mem fa;             // q_norm || k_norm fp32 (1 + w), the FA block (small_layout.h)
};
// Row offsets (bytes) of the five RMSNorms inside MtpHead::norms.
constexpr size_t kMtpNormRow = size_t(model::Qwen35::kHidden) * 4;
constexpr size_t kMtpNormPreE = 0 * kMtpNormRow;    // mtp.pre_fc_norm_embedding
constexpr size_t kMtpNormPreH = 1 * kMtpNormRow;    // mtp.pre_fc_norm_hidden
constexpr size_t kMtpNormInput = 2 * kMtpNormRow;   // mtp.layers.0.input_layernorm
constexpr size_t kMtpNormPost = 3 * kMtpNormRow;    // mtp.layers.0.post_attention_layernorm
constexpr size_t kMtpNormFinal = 4 * kMtpNormRow;   // mtp.norm
constexpr size_t kMtpNormsBytes = 5 * kMtpNormRow;
constexpr size_t kMtpTensors = 15;
constexpr size_t kMtpCheckpointBytes = 849398784;   // the 15 tensors' bf16 bytes (docs/03: 0.849 GB)

struct LoadReport {            // printed by load(); asserted by the checkpoint test
  size_t int4_bytes = 0, scale_bytes = 0, bf16_linear_bytes = 0;
  size_t embed_bytes = 0, lm_head_bytes = 0, pad_bytes = 0;
  // Every non-GEMV byte allocated, the RoPE table included. The printed
  // `small` line nets rope out (it is resident but not per-token traffic) and
  // reports it on its own line; total() and this field do not.
  size_t small_bytes = 0;
  // Spec 8: the MTP head's device bytes (0 unless loaded) - in total(), not in
  // read_per_token (the main step never reads it).
  size_t mtp_bytes = 0;
  size_t mtp_checkpoint_bytes = 0;  // what those came from: kMtpCheckpointBytes when loaded
  size_t mtp_tensors = 0;           // mtp.* tensors consumed (kMtpTensors when loaded)
  size_t total() const;             // the eight byte fields above
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
  std::unique_ptr<MtpHead> mtp;             // null unless load(..., mtp = true) (spec 8)
};

// Loads the qwen3_5 checkpoint at `snapshot_or_repo` (resolve_snapshot rules)
// into ctx's device. Skips model.visual.* and (v1) mtp.*. Strips the
// "model.language_model." prefix so model::Qwen35's layer-relative names bind.
// Asserts quant invariants and the doc-03 shape table; throws by name on any
// mismatch. max_len sizes the RoPE table only (default 16384).
// `mtp` (spec 8 §3.1): also load the MTP head into LoadedModel::mtp. Without it the
// head's tensors are skipped by name, counted, and never read - exactly as before.
LoadedModel load(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len = 16384,
                 bool mtp = false);

}  // namespace loader
