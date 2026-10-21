#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/bolt.h"

// Spec 25b Task 4: the host half of the Nemotron 3.5 Bolt loader - a checkpoint's tensors repacked into the device layouts
// loader/bolt_layout.h sizes, with NO Level Zero, so tests/loader/bolt_repack_test.cc and bolt_synth_host_test.cc run it
// on any host over crafted / synthetic checkpoints. loader::load_bolt (bolt_loader.h) drives these per block.
//
// **One layout from two sources** (spec 25 §0, §5.1): a linear read from SergiioB's g128 GPTQ checkpoint (through
// LinearSrc::classify's exact g128 -> g64 scale expansion) and the same weights packed g64 with each g128 scale written
// to both g64 rows give byte-identical device buffers - the repack only ever sees the g64 words and scales.
namespace loader {

// The language model's tensor prefix: `language_model.model.` (SergiioB's, ours) or NVIDIA's own
// `language_model.backbone.` - the alias is accepted only when the checkpoint ships it (spec 25 §1.3); both or neither is
// an error. Reads only the embedding's name.
std::string bolt_trunk_prefix(const SafetensorsSet& set);
inline constexpr const char* kBoltLmHead = "language_model.lm_head.weight";   // untied, outside the trunk
// Tensors under these prefixes are the vision tower / projector: skipped by name and counted, never read.
inline const std::vector<std::string>& bolt_vision_prefixes() {
  static const std::vector<std::string> p{"vision_model.", "vision_projector.", "mlp1."};
  return p;
}
bool bolt_is_vision(const std::string& name);

// Every language-model tensor of a Bolt checkpoint of `d` in an int4 export, by name (the trunk spelling `trunk`): int4
// linears as {base}.qweight / .qzeros / .scales (a shipped .g_idx is the loader's own check, never listed), everything
// else as shipped. Equals tools/oracle/bolt_ref.py's expected_names(cfg, "ar-g64") (25a's, built from the real indexes) -
// bolt_synth_host_test checks the two on the synthetics' own names.
std::vector<std::string> bolt_expected_names(const model::BoltDesc& d, const std::string& trunk);

// A fused int4 linear's device words, and - at layout 0 - its separate f16 scales (empty at layout 1, where they are
// inline in `words`).
struct BoltHostLinear {
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales;   // layout 0 only
};

// The generic fused linear every per-block linear goes through but the routed experts: `ln.fuse` (Concat over the parts, or
// Single) builds the column order, then `ln.shape.layout` picks the packing. `part_srcs` is already classified and
// shape-checked by the caller, parallel to `ln.parts` / `ln.part_n`.
BoltHostLinear repack_bolt_linear(const model::BoltLinear& ln, const std::vector<LinearSrc>& part_srcs);

// One routed expert's device block (loader/bolt_layout.h's block(I)): up (K latent, N I) then down (K I, N latent), both
// layout 1. `up.K == down.N == latent`, `up.N == down.K == I`. Written at `out` (block(I) / 4 words).
void repack_bolt_expert(const LinearSrc& up, const LinearSrc& down, uint32_t* out);

// The router [E][hidden] bf16 row-major -> gemv_bf16's tiles ({hidden, E}).
std::vector<uint16_t> repack_bolt_router(const uint16_t* w, uint32_t hidden, uint32_t E);

// A bf16 tensor widened to fp32, exact.
std::vector<float> widen_bolt_bf16(const uint16_t* w, size_t n);

// A = -exp(A_log), fp32, one head each (the load-time constant of the decode recurrence, spec 25 §2.2). The exp is
// common::exp_torch (Sleef's expf u10, deterministic on every host); docs/13-loader.md records its distance from
// torch.exp(A_log.float()).
std::vector<float> bolt_neg_exp(const std::vector<float>& a_log);

}  // namespace loader
