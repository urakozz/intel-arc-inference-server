#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 24b Task 2: LiquidAI LFM2.5-24B-A2B (`Lfm2MoeForCausalLM`, config.json model_type `lfm2_moe`) as data -
// a model BESIDE qwen3_5, K2-Horizon, Kolibri-1, qwen4_exp and Muse-Glimmer: its own descriptor, loader,
// kernels and engine (spec 24 §6), Kolibri's / Muse's family pattern. Every number of `lfm2moe()` is the
// checkpoint's own (LiquidAI/LFM2.5-24B-A2B@1769516a's config.json; spec 24 §1, §12 "24a as built"):
//
//   layers 40; `layer_types` (exactly): `full_attention` at 2, 6, 10, 14, 18, 22, 26, 30, 34, 38 (l % 4 == 2:
//          10 attention layers), `conv` everywhere else (30 layers, including layer 39)
//   hidden 2048, vocab 128000 (vocab_used 125017 - the tokenizer's distinct ids; argmax masks past it),
//          RMSNorm eps 1e-5, PLAIN w, no sandwich: h += (attention ? Attn : ShortConv)(operator_norm(h));
//          h += FFN(ffn_norm(h)); the FINAL norm (embedding_norm) runs after layer 39, not after the embedding
//   attention 32 q / 8 kv heads x 64 (GQA 4), q/k RMSNorm per head (self_attn.q_layernorm /
//          k_layernorm), full RoPE (rotate-half, theta 1e7, every dim) on every attention layer (a growing
//          cache, never sliding); no bias, no gate
//   conv   short conv (Lfm2MoeShortConv): in_proj 2048 -> 6144 (B, C, x in that row order), depthwise causal
//          conv1d kernel conv_L_cache 3 (no bias, no activation), out_proj 2048 -> 2048
//   FFN    layers 0-1 (num_dense_layers, by layer INDEX, independent of conv/attention): dense SwiGLU,
//          intermediate_size 11776; layers 2-39: MoE, 64 experts top-4, moe_intermediate_size 1536, no
//          shared expert, use_expert_bias / norm_topk_prob true only, routed_scaling_factor 1.0 only
//   vocab  tied embedding / head (tie_word_embeddings true): no lm_head.* tensor; BOS 124894, EOS 124900,
//          pad 124893; trained context (max_position_embeddings) 131072
//
// The descriptor is a struct of plain fields so a test can build a small one and so a synthetic checkpoint
// (tools/quantize/lfm2moe/make_synth.py, `--layers N`) or `--layers N` on the real checkpoint holds the first
// N layers of the real pattern - every width the loader, the layouts, the planner and the engine use is
// DERIVED here.
namespace model {

// The mixer at a layer (spec 24 §1's layer_types): Conv (Lfm2MoeShortConv) or Attention (full RoPE GQA).
enum class LfmKind { Conv, Attention };

// 24q's export may carry either spelling (spec 24 §5: transformers fuses the checkpoint's own
// `experts.N.{w1,w3,w2}` at load; AutoRound's generic unfusing names them `gate_proj` / `up_proj` /
// `down_proj` instead - which one 24q's export ships is not certain until it is built). The loader reads
// either, all-or-nothing per checkpoint (one format, two name sets).
enum class LfmExpertNames { W, Proj };
const char* lfm_expert_names_name(LfmExpertNames n);   // "w" | "proj"

// Every GEMV a layer runs, beside the routed experts (the loader's per-expert blocks, not a row here).
// `Head` is not a per-layer linear (model::Lfm2MoeDesc::linears never returns it): it names the tied head's
// shape for the loader / planner, which read it off the descriptor's own hidden / vocab fields instead.
enum class LfmLinearId { InProj, OutProj, Qkv, OProj, DenseGateUp, DenseDown, Router, Head, kCount };
struct LfmLinear {
  LfmLinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative checkpoint prefixes, in column order
};

struct Lfm2MoeDesc;

// Layers [0, split) on device 0, [split, layers) on device 1 (spec 16b's cut, spec 24 §4.5's two-card note).
// One device: devices == 1 and split == layers.
struct LfmPlacement {
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const { return dev == 0 ? 0 : split; }
  uint32_t end(uint32_t dev) const { return dev == 0 ? split : layers; }
  uint32_t device_of(uint32_t layer) const { return devices > 1 && layer >= split ? 1 : 0; }
  uint32_t count(uint32_t dev) const { return end(dev) - first(dev); }
  static LfmPlacement one(const Lfm2MoeDesc& d);
  static LfmPlacement two(const Lfm2MoeDesc& d, uint32_t split);
};
// Throws std::invalid_argument unless devices is 1 (split == layers) or 2 (1 <= split <= layers - 1).
void validate(const LfmPlacement& p, const Lfm2MoeDesc& d);

struct Lfm2MoeDesc {
  std::string name, architecture, model_type, prefix;   // "lfm2.5-24b-a2b", "Lfm2MoeForCausalLM", "lfm2_moe", "model."
  uint32_t layers = 0;                                    // 40; a synthetic / --layers checkpoint: 1..40
  uint32_t hidden = 0, vocab = 0, vocab_used = 0, trained_max_len = 0;   // 2048, 128000, 125017, 131072
  uint32_t q_heads = 0, kv_heads = 0, head_dim = 0;       // 32, 8, 64
  double rope_theta = 0;                                   // 1e7 (rope_type "default" only)
  uint32_t conv_taps = 0;                                  // 3 (conv_L_cache; conv_bias false only)
  uint32_t dense_layers = 0, dense_inter = 0;              // 2, 11776
  uint32_t experts = 0, top_k = 0, moe_inter = 0;          // 64, 4, 1536
  float norm_eps = 0;                                      // 1e-5
  uint32_t bos = 0, eos = 0, pad = 0;                      // 124894, 124900, 124893
  std::vector<LfmKind> kinds;                              // layer_types[0, layers) - the published prefix

  bool is_attention(uint32_t l) const;   // kinds[l] == Attention
  bool is_dense(uint32_t l) const;       // l < dense_layers (the FFN, independent of the mixer kind)
  uint32_t attn_index(uint32_t l) const;   // attention layers strictly before l
  uint32_t conv_index(uint32_t l) const;   // conv layers strictly before l
  uint32_t attn_layers() const;   // count of Attention in kinds
  uint32_t conv_layers() const;   // count of Conv in kinds
  uint32_t moe_layers() const { return layers > dense_layers ? layers - dense_layers : 0; }

  uint32_t q_n() const { return q_heads * head_dim; }       // 2048
  uint32_t kv_n() const { return kv_heads * head_dim; }     // 512
  uint32_t qkv_n() const { return q_n() + 2 * kv_n(); }     // 3072: q | k | v
  uint32_t gqa() const { return q_heads / kv_heads; }       // 4

  static std::string layer_prefix(const std::string& model_prefix, uint32_t l);   // "model.layers.<l>."
  std::string layer_prefix(uint32_t l) const { return layer_prefix(prefix, l); }
  // The GEMVs of layer l, in per-token execution order, shapes from the fields above: conv layers InProj +
  // OutProj, attention layers Qkv + OProj, dense layers (is_dense) + DenseGateUp + DenseDown, MoE layers +
  // Router. The routed experts are the loader's per-expert blocks, not a row here.
  std::vector<LfmLinear> linears(uint32_t l) const;
};

// The published model (the real 40-layer checkpoint). A process-lifetime singleton.
const Lfm2MoeDesc& lfm2moe();
// Throws std::invalid_argument naming the key when config.json describes an LFM2.5-24B-A2B the engine is not
// written for: every structural key against lfm2moe()'s (model_type, architectures[0], hidden_size,
// num_attention_heads, num_key_value_heads, head_dim (derived: hidden_size / num_attention_heads),
// num_dense_layers, moe_intermediate_size, num_experts, num_experts_per_tok, intermediate_size, vocab_size,
// conv_L_cache, conv_bias false, norm_eps, use_expert_bias true, norm_topk_prob true,
// routed_scaling_factor 1.0, rope_parameters.rope_type "default", rope_parameters.rope_theta 1e7,
// tie_word_embeddings true), num_hidden_layers in [1, 40] and layer_types[0, N) equal to the PUBLISHED
// pattern's prefix of that length (the real pattern is fixed: full_attention at l % 4 == 2).
void check_lfm2moe_config(const common::json::Value& config);
// config.json -> the descriptor at the checkpoint's own layer count (check_lfm2moe_config first).
// `vocab_used` is not in config.json (it is the tokenizer's distinct-id count, spec 24 §1).
Lfm2MoeDesc lfm2moe_desc(const common::json::Value& config, uint32_t vocab_used);
bool is_lfm2moe(const common::json::Value& config);   // model_type == "lfm2_moe"

}  // namespace model
