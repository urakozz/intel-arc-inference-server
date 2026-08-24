#include "model/qwen35.h"

#include <array>
#include <cstddef>
#include <initializer_list>

namespace model {
namespace {

// The model's whole linear table, indexed by LinearId ordinal. Shapes are
// read from the checkpoint headers (docs/03-models.md); (layout, S) are the
// decision block of docs/probe-gemv-2026-08-24.md, measured on the box:
//
//   canonical layout 1; out/o_proj S=16, q||k||v S=1, qkv||z S=1,
//   gate||up S=4, down S=16.
//
// The two bf16 rows (AB, LmHead) have no layout choice - one tiled layout
// (common::repack_bf16_tiled) - so they carry layout 0 as a filler.
// Fuse maps onto the loader's column maps: Concat -> common::cols_concat,
// Interleave16 -> common::cols_interleave16 (gate||up in 16-column blocks).
// Part names are layer-relative; a consumer joins them with layer_prefix().
// LmHead is a top-level tensor and appears in no layer's list; plan 3 binds
// it from this row directly.
const std::array<FusedLinear, 8>& table() {
  static const std::array<FusedLinear, 8> t = {{
      // GDN: in_proj_qkv (10240 = q 16x128 | k 16x128 | v 48x128) || in_proj_z (6144).
      {LinearId::QkvZ, {5120, 16384, 1, 1}, WeightKind::Int4, Fuse::Concat,
       {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}, 0},
      // GDN decay/beta, bf16 (kept out of the quantisation by the checkpoint's
      // dynamic rules). Source N is 48+48 = 96, zero-padded to the kernel's 128.
      {LinearId::AB, {5120, 128, 1, 0}, WeightKind::Bf16, Fuse::Concat,
       {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, 96},
      // K = 6144 = 48 value heads x 128.
      {LinearId::OutProj, {6144, 5120, 16, 1}, WeightKind::Int4, Fuse::Single,
       {"linear_attn.out_proj"}, 0},
      // gate || up interleaved so the lane holding gate[n] finds up[n] at +16.
      {LinearId::GateUp, {5120, 34816, 4, 1}, WeightKind::Int4, Fuse::Interleave16,
       {"mlp.gate_proj", "mlp.up_proj"}, 0},
      {LinearId::Down, {17408, 5120, 16, 1}, WeightKind::Int4, Fuse::Single,
       {"mlp.down_proj"}, 0},
      // FA: q_proj is 12288 (24 heads x 256 x (q || gate)), k/v 1024 each.
      {LinearId::Qkv, {5120, 14336, 1, 1}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, 0},
      // K = 6144 = 24 heads x 256.
      {LinearId::OProj, {6144, 5120, 16, 1}, WeightKind::Int4, Fuse::Single,
       {"self_attn.o_proj"}, 0},
      // Top level, not quantised, read in full every token (2.54 GB).
      {LinearId::LmHead, {5120, 248320, 1, 0}, WeightKind::Bf16, Fuse::Single,
       {"lm_head"}, 0},
  }};
  return t;
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

// Everything a layer needs that is not a GEMV weight (docs/03-models.md).
// The loader bakes the RMSNorm (1 + w) into the layernorm weights; the GDN
// gated norm is plain w.
const std::vector<std::string>& gdn_small() {
  static const std::vector<std::string> v = {
      "input_layernorm.weight", "post_attention_layernorm.weight",
      "linear_attn.conv1d.weight", "linear_attn.A_log",
      "linear_attn.dt_bias", "linear_attn.norm.weight"};
  return v;
}
const std::vector<std::string>& fa_small() {
  static const std::vector<std::string> v = {
      "input_layernorm.weight", "post_attention_layernorm.weight",
      "self_attn.q_norm.weight", "self_attn.k_norm.weight"};
  return v;
}

}  // namespace

const GemvShape& Qwen35::shape(LinearId id) {
  return table()[static_cast<size_t>(id)].shape;
}

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
