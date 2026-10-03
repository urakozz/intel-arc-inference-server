#include "model/qwen35.h"


// The one header this translation unit borrows from outside src/model: a
// code-free set of block offsets (fix I2/I3, 2026-08-25). The small-tensor
// table below carries those constants as its destination offsets so the model
// description and the loader's packing cannot drift apart; nothing else about
// the loader is visible here, and no library dependency is created.
#include "loader/small_layout.h"

namespace model {
namespace {

// Everything a layer needs that is not a GEMV weight (docs/03-models.md), with
// the shape, source dtype, device placement and bake of each - the loader
// walks exactly this and hardcodes nothing (fix I3). Offsets are the
// loader/small_layout.h constants; the entries fill their block exactly and
// the loader throws if they do not.
//
// Ruling 2026-08-25: the RMSNorm family (input/post layernorms, q_norm,
// k_norm, and the top-level final norm) is stored fp32 `1 + w` - the HF
// reference multiplies in fp32, so a bf16 multiplier would add a rounding it
// never had. `linear_attn.norm` is RMSNormGated: plain `w`, and it stays bf16
// because the reference's own parameter dtype is bf16 and it multiplies in the
// bf16 domain.
const std::vector<SmallTensor>& gdn_small() {
  static const std::vector<SmallTensor> v = {
      {"input_layernorm.weight", Qwen35::kHidden, "BF16", SmallBlock::Norms,
       loader::kNormsOffInput, SmallBake::OnePlusWFp32},
      {"post_attention_layernorm.weight", Qwen35::kHidden, "BF16", SmallBlock::Norms,
       loader::kNormsOffPost, SmallBake::OnePlusWFp32},
      // bf16 [10240][1][4] in the checkpoint; the 4-tap depthwise state is
      // accumulated in fp32, so the taps are widened once at load.
      {"linear_attn.conv1d.weight", uint32_t(loader::kConvRows * loader::kConvTaps), "BF16",
       SmallBlock::Kind, loader::kGdnOffConv, SmallBake::RawFp32Widen},
      // Only ever used as exp(g) with g = -exp(A_log)*softplus(...): hoisted.
      {"linear_attn.A_log", Qwen35::kGdnVHeads, "BF16", SmallBlock::Kind, loader::kGdnOffNegA,
       SmallBake::NegExpFp32},
      {"linear_attn.dt_bias", Qwen35::kGdnVHeads, "BF16", SmallBlock::Kind,
       loader::kGdnOffDtBias, SmallBake::RawFp32Widen},
      {"linear_attn.norm.weight", Qwen35::kGdnHeadDim, "BF16", SmallBlock::Kind,
       loader::kGdnOffGatedNorm, SmallBake::PlainBf16},
  };
  return v;
}
const std::vector<SmallTensor>& fa_small() {
  static const std::vector<SmallTensor> v = {
      {"input_layernorm.weight", Qwen35::kHidden, "BF16", SmallBlock::Norms,
       loader::kNormsOffInput, SmallBake::OnePlusWFp32},
      {"post_attention_layernorm.weight", Qwen35::kHidden, "BF16", SmallBlock::Norms,
       loader::kNormsOffPost, SmallBake::OnePlusWFp32},
      {"self_attn.q_norm.weight", Qwen35::kFaHeadDim, "BF16", SmallBlock::Kind,
       loader::kFaOffQNorm, SmallBake::OnePlusWFp32},
      {"self_attn.k_norm.weight", Qwen35::kFaHeadDim, "BF16", SmallBlock::Kind,
       loader::kFaOffKNorm, SmallBake::OnePlusWFp32},
  };
  return v;
}

}  // namespace

const std::vector<SmallTensor>& Qwen35::small_tensors(LayerKind kind) {
  return kind == LayerKind::FA ? fa_small() : gdn_small();
}

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
