#include "model/model_desc.h"

#include <stdexcept>

namespace model {
namespace {

static_assert(kLinearCount == 8, "LinearId grew: add the row below and re-check the shape table");

// Qwen3.8's linear table, indexed by LinearId ordinal. Shapes are read from the
// checkpoint headers (docs/03-models.md); (layout, S) are the decision block of
// docs/probe-gemv-2026-08-24.md, measured on the box:
//
//   Task 4 production map: out/o_proj L0 S4, q||k||v L0 S2,
//   qkv||z L1 S1, gate||up L0 S8, down L0 S4.
//
// The two bf16 rows (AB, LmHead) have no layout choice - one tiled layout
// (common::repack_bf16_tiled) - so they carry layout 0 as a filler.
// Fuse maps onto the loader's column maps: Concat -> common::cols_concat,
// Interleave16 -> common::cols_interleave16 (gate||up in 16-column blocks).
// Part names are layer-relative engine names; a consumer joins them with
// Qwen35::layer_prefix(). LmHead is a top-level tensor and appears in no
// layer's list; the capture binds it from this row directly.
std::array<FusedLinear, kLinearCount> qwen38_table() {
  return {{
      // GDN: in_proj_qkv (10240 = q 16x128 | k 16x128 | v 48x128) || in_proj_z (6144).
      {LinearId::QkvZ, {5120, 16384, 1, 1}, WeightKind::Int4, Fuse::Concat,
       {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}, 0, Fold::None, {}},
      // GDN decay/beta, bf16 (kept out of the quantisation by the checkpoint's
      // dynamic rules). Source N is 48+48 = 96, zero-padded to the kernel's 128.
      {LinearId::AB, {5120, 128, 1, 0}, WeightKind::Bf16, Fuse::Concat,
       {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}, 96, Fold::None, {}},
      // K = 6144 = 48 value heads x 128.
      {LinearId::OutProj, {6144, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"linear_attn.out_proj"}, 0, Fold::None, {}},
      // gate || up interleaved so the lane holding gate[n] finds up[n] at +16.
      {LinearId::GateUp, {5120, 34816, 8, 0}, WeightKind::Int4, Fuse::Interleave16,
       {"mlp.gate_proj", "mlp.up_proj"}, 0, Fold::None, {}},
      {LinearId::Down, {17408, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"mlp.down_proj"}, 0, Fold::None, {}},
      // FA: q_proj is 12288 (24 heads x 256 x (q || gate)), k/v 1024 each.
      {LinearId::Qkv, {5120, 14336, 2, 0}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, 0, Fold::None, {}},
      // K = 6144 = 24 heads x 256.
      {LinearId::OProj, {6144, 5120, 4, 0}, WeightKind::Int4, Fuse::Single,
       {"self_attn.o_proj"}, 0, Fold::None, {}},
      // Top level. The bf16 row: 2.543 GB read in full every token. This is
      // the row `linear(LinearId::LmHead)` returns and the one the shipped
      // bf16 checkpoint needs; `lm_head_int4()` below is its counterpart
      // for a checkpoint that packed this tensor.
      {LinearId::LmHead, {5120, 248320, 1, 0}, WeightKind::Bf16, Fuse::Single,
       {"lm_head"}, 0, Fold::None, {}},
  }};
}

// **`lm_head` is the ONE linear whose kind is a property of the checkpoint,
// not of the model.** The shipped `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`
// leaves it bf16 (2.543 GB read per token); the self-quantised
// `qwen38-27b-w4g64-rtn` packs it int4 g64 sym (0.675 GB). The loader picks by
// which tensors the checkpoint ships (`loader::LinearSrc::classify`), never by
// a config label (docs/02, docs/13's "Classification" table).
//
// The int4 row is `{K 5120, N 248320, S 1, layout 1}`: layout 1 because it is
// the only int4 repack the loader implements and the measured winner at large
// N (docs/probe-gemv-2026-08-24.md); S = 1 because N/64 = 3880 work-groups
// already saturate the device, and because S = 1 lets the capture bind the
// output straight at `logits` (gemv.cl writes `out[(s*M + m)*N + n]`).
// Deliberately NOT a ninth table entry: `LinearId::LmHead` must stay one ordinal.
const FusedLinear& lm_head_int4() {
  static const FusedLinear r = {LinearId::LmHead, {5120, 248320, 1, 1}, WeightKind::Int4,
                                Fuse::Single,     {"lm_head"},          0, Fold::None, {}};
  return r;
}

// Spec 9: the int8 `lm_head` - quantised from the bf16 tensor at load, 1.272 GB.
// S = 1 for the same reason (gemv_i8w writes [M][N] straight at `logits`);
// `layout` is a filler, the int8 tiling is fixed (loader/lm_head_int8.h).
const FusedLinear& lm_head_int8() {
  static const FusedLinear r = {LinearId::LmHead, {5120, 248320, 1, 0}, WeightKind::Int8,
                                Fuse::Single,     {"lm_head"},          0, Fold::None, {}};
  return r;
}

ModelDesc make_qwen38() {
  ModelDesc d;
  d.name = "qwen3.8";
  d.architecture = "Qwen3_5ForConditionalGeneration";
  d.layers = 64;
  d.fa_layers = 16;   // l % 4 == 3
  d.gdn_layers = 48;
  d.intermediate = 17408;
  d.parallel_ffn = 0;
  d.max_len_ceiling = 0;
  d.table = qwen38_table();
  return d;
}

}  // namespace

const FusedLinear& ModelDesc::linear(LinearId id) const {
  const size_t i = static_cast<size_t>(id);
  if (i >= kLinearCount)   // kCount, or an out-of-range cast from a dispatch
    throw std::out_of_range("ModelDesc::linear: LinearId ordinal " + std::to_string(i) +
                            " is out of range (table has " + std::to_string(kLinearCount) + ")");
  return table[i];
}

const FusedLinear& ModelDesc::lm_head(WeightKind kind) const {
  if (kind == WeightKind::Int8) return lm_head_int8();
  return kind == WeightKind::Int4 ? lm_head_int4() : linear(LinearId::LmHead);
}

// Built per call rather than cached: layer_descs() returns by value, and the
// only per-layer state is index/kind.
std::vector<LayerDesc> ModelDesc::layer_descs() const {
  std::vector<LayerDesc> out;
  out.reserve(layers);
  for (uint32_t i = 0; i < layers; ++i) {
    const LayerKind kind = is_fa(i) ? LayerKind::FA : LayerKind::GDN;
    std::vector<FusedLinear> lin;
    for (LinearId id : Qwen35::linear_order(kind)) lin.push_back(linear(id));
    out.push_back({i, kind, std::move(lin), Qwen35::small_tensors(kind)});
  }
  return out;
}

std::string ModelDesc::to_engine(const std::string& n) const {
  for (const auto& [ckpt, eng] : name_map) {
    const size_t at = n.find(ckpt);
    if (at != std::string::npos) return n.substr(0, at) + eng + n.substr(at + ckpt.size());
  }
  return n;
}

std::string ModelDesc::to_checkpoint(const std::string& n) const {
  for (const auto& [ckpt, eng] : name_map) {
    const size_t at = n.find(eng);
    if (at != std::string::npos) return n.substr(0, at) + ckpt + n.substr(at + eng.size());
  }
  return n;
}

std::string ModelDesc::intermediate_suffix() const {
  return intermediate == 17408 ? std::string() : "_I" + std::to_string(intermediate);
}

const ModelDesc& qwen38() {
  static const ModelDesc d = make_qwen38();
  return d;
}

const ModelDesc& desc_for_architecture(const std::string& architecture) {
  for (const ModelDesc* d : {&qwen38()})
    if (d->architecture == architecture) return *d;
  throw std::runtime_error("config.json architectures[0] is '" + architecture +
                           "'; this engine runs Qwen3_5ForConditionalGeneration (Qwen3.8)");
}

}  // namespace model
