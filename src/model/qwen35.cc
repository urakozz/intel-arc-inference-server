#include "model/qwen35.h"

// The per-layer small-tensor tables moved to model/model_desc.cc (spec 15b):
// their element counts and block offsets follow each model's hidden size and
// GDN heads, so they are built per descriptor (`ModelDesc::small_tensors`).

namespace model {

// Per-token execution order within a layer.
const std::vector<LinearId>& Qwen35::linear_order(LayerKind kind) {
  static const std::vector<LinearId> gdn = {LinearId::QkvZ, LinearId::AB, LinearId::OutProj,
                                            LinearId::GateUp, LinearId::Down};
  static const std::vector<LinearId> fa = {LinearId::Qkv, LinearId::OProj, LinearId::GateUp,
                                           LinearId::Down};
  return kind == LayerKind::FA ? fa : gdn;
}

std::string Qwen35::layer_prefix(uint32_t layer) {
  return "layers." + std::to_string(layer) + ".";
}

}  // namespace model
