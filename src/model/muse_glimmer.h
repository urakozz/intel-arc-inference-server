#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 23b: meta-models' Muse-Glimmer-30B text model (`MuseGlimmerForConditionalGeneration`, config.json
// model_type `muse_glimmer`, text config `muse_glimmer_text`) as data - a model BESIDE qwen3_5, K2, Kolibri and
// qwen4_exp (spec 23 §5.1: its own table, loader, kernels and engine, Kolibri's pattern). Every number of
// `muse_glimmer()` is the checkpoint's (meta-models/Muse-Glimmer-30B @ a4e59da5 config.json and
// generation_config.json - vendored byte for byte in tests/model/muse_glimmer/ - and the shard headers;
// docs/probe-muse-glimmer-2026-10-15.md):
//
//   layers 52, dense; `layer_types` full attention at l % 4 == 3 (13 full, NoPE), the other 39 sliding with a
//          2048-key window INCLUDING the query (key j visible to query i iff i - 2048 < j <= i), RoPE theta 5e5
//          (NEOX halves) on the sliding layers only - `layer_rope_theta` is a switch (0 = NoPE), never a theta
//   hidden 6656; four CENTERED RMSNorms a layer (`x * rsqrt(ms + eps) * (1 + w)`, eps 1e-5 for input / pre-ff,
//          post_norm_eps 1e-8 for the two post norms), sandwich order; the embedding row then a weight-less
//          RMSNorm; the final norm plain `w`
//   attention 32 q / 2 kv heads x 128 (GQA 16), weight-less q / k norms, q x qk_scale 3.87, an output gate
//          sigmoid(gate_proj(x)) before o_proj; no bias
//   FFN    dense SwiGLU, intermediate 19968
//   head   untied bf16 lm_head [202048][6656]; logits x output_multiplier (1/sqrt 26), softcap 20
//   tokens BOS 200000, EOS {200001, 200008} (generation_config.json), pad 200018; context 131072
//
// The layer's EIGHT int4 linears are fused into FOUR GEMVs (spec 23 §5.2): q || gate || k || v (6656 -> 8704)
// and o_proj; gate || up (6656 -> 39936) and down. The vision tower, adapter and projection (809 tensors) are
// skipped by name by the loader. The descriptor is a struct of plain fields so a test can build a small one (the
// tiny model's widths, tests/loader/muse_repack_test.cc); every width the loader, the planner and the kernels'
// host half use is DERIVED here.
namespace model {

// The fused device weights. kCount is the table size, not a linear.
enum class MuseLinearId { QGKV, OProj, GateUp, Down, LmHead, kCount };
struct MuseLinear {
  MuseLinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative prefixes ("self_attn.q_proj"), in column order
  std::vector<uint32_t> part_n;     // each part's output columns, parallel to `parts` (sum = shape.N)
};

struct MuseGlimmerDesc;

// Layers [0, split) on device 0, [split, layers) on device 1 (spec 16b's cut, KolPlacement's form). One device:
// devices == 1 and split == layers.
struct MusePlacement {
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const { return dev == 0 ? 0 : split; }
  uint32_t end(uint32_t dev) const { return dev == 0 ? split : layers; }
  uint32_t device_of(uint32_t layer) const { return devices > 1 && layer >= split ? 1 : 0; }
  uint32_t count(uint32_t dev) const { return end(dev) - first(dev); }
  static MusePlacement one(const MuseGlimmerDesc& d);
  static MusePlacement two(const MuseGlimmerDesc& d, uint32_t split);
};
// Throws std::invalid_argument unless devices is 1 (split == layers) or 2 (1 <= split <= layers - 1) and the
// placement covers d.layers. Every user of a placement calls it.
void validate(const MusePlacement& p, const MuseGlimmerDesc& d);

struct MuseGlimmerDesc {
  uint32_t layers = 52, hidden = 6656, q_heads = 32, kv_heads = 2, head_dim = 128, inter = 19968, vocab = 202048,
           window = 2048, max_pos = 131072;
  float rms_eps = 1e-5f, post_eps = 1e-8f, qk_scale = 3.87f, rope_theta = 5e5f, out_mult = 0.19611613513818404f,
        softcap = 20.f;
  uint32_t bos = 200000, pad = 200018;
  std::vector<uint32_t> eos = {200001, 200008};
  std::vector<bool> full;   // per layer: full attention (NoPE); the rest sliding (RoPE, the window)

  // The int4 GEMVs' split-K (gemv.cl S, layout 0). PROVISIONAL - Qwen3.8's measured cells at the nearest shapes
  // (docs/probe-gemv-2026-08-24.md: q||k||v L0 S2, o_proj L0 S4, gate||up L0 S8, down L0 S4) until 23c's sweep.
  uint32_t qgkv_s = 2, oproj_s = 4, gate_up_s = 8, down_s = 4;

  bool is_full(uint32_t l) const { return l < full.size() && full[l]; }
  uint32_t full_layers() const;               // 13
  uint32_t sliding_layers() const;            // 39
  uint32_t full_before(uint32_t l) const;     // full layers in [0, l)
  uint32_t sliding_before(uint32_t l) const;  // sliding layers in [0, l)
  uint32_t q_n() const { return q_heads * head_dim; }     // 4096 (q, and the gate)
  uint32_t kv_n() const { return kv_heads * head_dim; }   // 256
  uint32_t gqa() const { return q_heads / kv_heads; }     // 16
  uint32_t qgkv_rows() const { return 2 * q_heads * head_dim + 2 * kv_heads * head_dim; }   // 8704
  // Column offsets inside q || gate || k || v.
  uint32_t qgkv_off_gate() const { return q_n(); }               // 4096
  uint32_t qgkv_off_k() const { return 2 * q_n(); }              // 8192
  uint32_t qgkv_off_v() const { return 2 * q_n() + kv_n(); }     // 8448
  // Sliding-ring slots a layer (spec 23 §5.2): a power of two >= a prefill chunk (2048) + window - 1, so a
  // chunk's own keys and the window before it never collide; 64-key blocks never cross the ring's end.
  static constexpr uint32_t kRing = 4096;
  // The per-layer fp32 norm block on the device: input | post_attention | pre_feedforward | post_feedforward
  // [hidden] each, the checkpoint's bf16 w widened (the kernels compute 1 + w in fp32, as the reference does).
  uint32_t norm_off_input() const { return 0; }
  uint32_t norm_off_post_attn() const { return hidden; }
  uint32_t norm_off_pre_ff() const { return 2 * hidden; }
  uint32_t norm_off_post_ff() const { return 3 * hidden; }
  uint32_t norm_floats() const { return 4 * hidden; }

  MuseLinear linear(MuseLinearId id) const;          // throws on kCount
  std::vector<MuseLinearId> layer_linears() const;   // {QGKV, OProj, GateUp, Down}
  static std::string layer_prefix(uint32_t l);       // "model.language_model.layers.<l>."
  static const char* text_prefix() { return "model.language_model."; }
};

// The published model (52 layers, full at l % 4 == 3).
MuseGlimmerDesc muse_glimmer();
// config.json (the whole file - its `text_config` is read - or a text config alone) -> the descriptor, every
// structural key held to muse_glimmer(): the widths, heads, vocab, window, context, eps, qk scale, output
// multiplier, softcap, silu, untied, no attention bias, rope default at theta 5e5, layer_types the real
// pattern's prefix, layer_rope_theta's switch rule (non-zero entries = rope_parameters.rope_theta, zeros exactly
// at full_attention - spec 23 §1's trap), num_hidden_layers 1..52 (a synthetic prefix). Throws
// std::runtime_error naming the key (and the layer) and its value. `layers_override` > 0 truncates to the first
// layers_override layers (development mode; <= num_hidden_layers).
MuseGlimmerDesc check_muse_config(const common::json::Value& config_json, uint32_t layers_override = 0);
bool is_muse_model_type(const std::string& model_type);   // "muse_glimmer" | "muse_glimmer_text"

}  // namespace model
