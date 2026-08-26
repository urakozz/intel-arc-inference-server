#include "model/qwen35.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

// The one header this translation unit borrows from outside src/model: a
// code-free set of block offsets (fix I2/I3, 2026-08-25). The small-tensor
// table below carries those constants as its destination offsets so the model
// description and the loader's packing cannot drift apart; nothing else about
// the loader is visible here, and no library dependency is created.
#include "loader/small_layout.h"

namespace model {
namespace {

// The model's whole linear table, indexed by LinearId ordinal. Shapes are
// read from the checkpoint headers (docs/03-models.md); (layout, S) are the
// decision block of docs/probe-gemv-2026-08-24.md, measured on the box:
//
//   Task 4 production map: out/o_proj L0 S4, q||k||v L0 S2,
//   qkv||z L1 S1, gate||up L0 S8, down L0 S4.
//
// The two bf16 rows (AB, LmHead) have no layout choice - one tiled layout
// (common::repack_bf16_tiled) - so they carry layout 0 as a filler.
// Fuse maps onto the loader's column maps: Concat -> common::cols_concat,
// Interleave16 -> common::cols_interleave16 (gate||up in 16-column blocks).
// Part names are layer-relative; a consumer joins them with layer_prefix().
// LmHead is a top-level tensor and appears in no layer's list; plan 3 binds
// it from this row directly.
constexpr size_t kLinearCount = static_cast<size_t>(LinearId::kCount);
static_assert(kLinearCount == 8, "LinearId grew: add the row below and re-check the shape table");

const std::array<FusedLinear, kLinearCount>& table() {
  static const std::array<FusedLinear, kLinearCount> t = {{
      // GDN: in_proj_qkv (10240 = q 16x128 | k 16x128 | v 48x128) || in_proj_z (6144).
      {LinearId::QkvZ, {5120, 16384, 1, 1}, WeightKind::Int4, Fuse::Concat,
       {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}, 0},
      // GDN decay/beta, bf16 (kept out of the quantisation by the checkpoint's
      // dynamic rules). Source N is 48+48 = 96, zero-padded to the kernel's 128.
      {LinearId::AB, {5120, 128, 1, 0}, WeightKind::Bf16, Fuse::Concat,
       {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, 96},
      // K = 6144 = 48 value heads x 128.
      {LinearId::OutProj, {6144, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"linear_attn.out_proj"}, 0},
      // gate || up interleaved so the lane holding gate[n] finds up[n] at +16.
      {LinearId::GateUp, {5120, 34816, 8, 0}, WeightKind::Int4, Fuse::Interleave16,
       {"mlp.gate_proj", "mlp.up_proj"}, 0},
      {LinearId::Down, {17408, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"mlp.down_proj"}, 0},
      // FA: q_proj is 12288 (24 heads x 256 x (q || gate)), k/v 1024 each.
      {LinearId::Qkv, {5120, 14336, 2, 0}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, 0},
      // K = 6144 = 24 heads x 256.
      {LinearId::OProj, {6144, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"self_attn.o_proj"}, 0},
      // Top level. The bf16 row: 2.543 GB read in full every token. This is
      // the row `linear(LinearId::LmHead)` returns and the one the shipped
      // bf16 checkpoint needs; `lm_head_int4()` below is its counterpart
      // for a checkpoint that packed this tensor, and the loader picks between
      // them by content (Qwen35::lm_head, qwen35.h).
      {LinearId::LmHead, {5120, 248320, 1, 0}, WeightKind::Bf16, Fuse::Single,
       {"lm_head"}, 0},
  }};
  return t;
}

// The int4 `lm_head` - same tensor, same K and N, 0.675 GB instead of 2.543.
// Deliberately NOT a ninth entry in `table()`: `LinearId` indexes that array
// and `LinearId::LmHead` must stay one ordinal, so a second row for the same id
// lives beside it. The `{S, layout}` choice is argued in qwen35.h.
const FusedLinear& lm_head_int4() {
  static const FusedLinear r = {LinearId::LmHead, {5120, 248320, 1, 1}, WeightKind::Int4,
                                Fuse::Single,     {"lm_head"},          0};
  return r;
}

std::vector<FusedLinear> pick(std::initializer_list<LinearId> ids) {
  std::vector<FusedLinear> v;
  v.reserve(ids.size());
  for (LinearId id : ids) v.push_back(table()[static_cast<size_t>(id)]);
  return v;
}

// Per-token execution order within a layer.
const std::vector<FusedLinear>& gdn_linears() {
  static const std::vector<FusedLinear> v = pick(
      {LinearId::QkvZ, LinearId::AB, LinearId::OutProj, LinearId::GateUp, LinearId::Down});
  return v;
}
const std::vector<FusedLinear>& fa_linears() {
  static const std::vector<FusedLinear> v =
      pick({LinearId::Qkv, LinearId::OProj, LinearId::GateUp, LinearId::Down});
  return v;
}

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

const FusedLinear& Qwen35::linear(LinearId id) {
  const size_t i = static_cast<size_t>(id);
  if (i >= kLinearCount)   // kCount, or an out-of-range cast from plan 3's dispatch
    throw std::out_of_range("Qwen35::linear: LinearId ordinal " + std::to_string(i) +
                            " is out of range (table has " + std::to_string(kLinearCount) + ")");
  return table()[i];
}

const FusedLinear& Qwen35::lm_head(WeightKind kind) {
  return kind == WeightKind::Int4 ? lm_head_int4() : linear(LinearId::LmHead);
}

const GemvShape& Qwen35::shape(LinearId id) { return linear(id).shape; }

// Built per call from the two per-kind templates rather than cached: layers()
// returns by value, so a static of all 64 would be copied on every call anyway
// and the only per-layer state is index/kind.
std::vector<LayerDesc> Qwen35::layers() {
  std::vector<LayerDesc> out;
  out.reserve(kLayers);
  for (uint32_t i = 0; i < kLayers; ++i) {
    const bool fa = is_fa(i);  // [LA, LA, LA, FA] x 16 -> 48 GDN, 16 FA
    out.push_back({i, fa ? LayerKind::FA : LayerKind::GDN,
                   fa ? fa_linears() : gdn_linears(),
                   fa ? fa_small() : gdn_small()});
  }
  return out;
}

std::string Qwen35::layer_prefix(uint32_t layer) {
  return "layers." + std::to_string(layer) + ".";
}

}  // namespace model
