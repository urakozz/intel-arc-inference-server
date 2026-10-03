#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "model/qwen35.h"

// Spec 14 §3.1: the per-model shape as data. Everything the two supported
// checkpoints share - hidden, heads, head dims, vocab, rotary, the small-tensor
// tables, the per-layer linear order - stays `constexpr` in `model::Qwen35`.
// What differs is here, chosen by the loader from `config.json`'s
// `architectures[0]`:
//
//                       Qwen3.8-27B                      Agnes 3.0 Flash
//   architecture        Qwen3_5ForConditionalGeneration  AgnesForConditionalGeneration
//   layers              64 = 48 GDN + 16 FA              72 = 54 GDN + 18 FA
//   FA layers           l % 4 == 3                       l % 4 == 3
//   MLP intermediate    17408                            19456 = 17408 + 2048, the
//                                                        parallel FFN folded in (§2)
//   tensor names        linear_attn. / self_attn.        delta_attn. / global_attn.
//   max_len ceiling     none                             65536 (§3.3, until spec 12)
//
// Descriptors are process-lifetime singletons: hold them by reference/pointer.
namespace model {

struct ModelDesc {
  std::string name;           // a short label for logs: "qwen3.8", "agnes-3.0-flash"
  std::string architecture;   // config.json's architectures[0]
  uint32_t layers = 0, gdn_layers = 0, fa_layers = 0;
  // The MLP width the ENGINE runs: for Agnes the folded 17408 + 2048 (spec 14 §2).
  uint32_t intermediate = 0;
  // The checkpoint's `mlp.parallel_ffn` width, folded at load; 0 = none.
  uint32_t parallel_ffn = 0;
  // The largest max_len this model may be loaded at; 0 = no model-specific
  // ceiling. Agnes: 65536 (spec 14 §3.3: 128k bf16 KV does not fit beside the
  // weights until spec 12's int8 KV).
  uint32_t max_len_ceiling = 0;
  // Checkpoint-name infix -> engine-name infix, applied to every
  // `model.language_model.*` and `mtp.*` name (vLLM PR #57003's WeightsMapper).
  std::vector<std::pair<std::string, std::string>> name_map;
  std::array<FusedLinear, kLinearCount> table{};   // indexed by LinearId ordinal

  bool has_parallel_ffn() const { return parallel_ffn != 0; }
  // [GDN, GDN, GDN, FA] repeating - both models (`global_attention_interval` 4).
  static bool is_fa(uint32_t layer) { return layer % 4 == 3; }

  // The table row. Throws std::out_of_range on kCount or a bad cast.
  // `linear(LmHead)` is the bf16 row (see Qwen35's history note); callers that
  // must honour the checkpoint use `lm_head(kind)`.
  const FusedLinear& linear(LinearId id) const;
  const GemvShape& shape(LinearId id) const { return linear(id).shape; }
  const FusedLinear& lm_head(WeightKind kind) const;
  // All `layers` layer descriptors, fully populated, in index order.
  std::vector<LayerDesc> layer_descs() const;

  // Name mapping. `to_engine` rewrites a checkpoint name's first matching infix
  // ("delta_attn." -> "linear_attn."); `to_checkpoint` is its inverse. Both are
  // the identity on Qwen3.8.
  std::string to_engine(const std::string& checkpoint_name) const;
  std::string to_checkpoint(const std::string& engine_name) const;

  // The suffix that kernels baking the intermediate size carry in their variant
  // names: "" at Qwen3.8's 17408 (so every Qwen3.8 binary keeps its name and its
  // command line), "_I<intermediate>" otherwise.
  std::string intermediate_suffix() const;
};

const ModelDesc& qwen38();
// The descriptor for config.json's `architectures[0]`; throws std::runtime_error
// naming the architecture and the supported ones.
const ModelDesc& desc_for_architecture(const std::string& architecture);

}  // namespace model
