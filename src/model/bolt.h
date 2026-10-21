#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 25b Task 2: NVIDIA Nemotron 3.5 Bolt 74B-A9B (text only; `NemotronH_Omni_Reasoning_V3`, config.json model_type
// `nemotron_h_omni` wrapping `llm_config.model_type` `nemotron_h_puzzle`) as data - a model BESIDE qwen3_5, K2-Horizon,
// Kolibri-1, qwen4_exp, Muse-Glimmer and LFM2.5: its own descriptor, loader, kernels and engine (spec 25 §8), the 24b
// family pattern. Every number of `bolt()` is the checkpoint's own (the published config, tests/model/bolt/config.json;
// spec 25 §1.1, §13):
//
//   layers 88; `block_configs` / `layers_block_type`: 40 mamba, 40 moe, 8 attention (at 7 16 25 36 47 58 69 78)
//   hidden 4096, vocab 131072 (every row a token: vocab_used = vocab), max_position_embeddings 262144, untied head
//   norm   NemotronHRMSNorm eps 1e-5, plain w, ONE rounding; block = norm -> mixer -> + residual
//   mamba  128 heads x 64 = d_inner 8192, n_groups 8, state 128, conv 4 (with bias), in_proj 4096 -> 18560 = z 8192 |
//          xBC 10240 | dt 128, out_proj 8192 -> 4096
//   attn   32 q / 2 kv heads x 128, no bias, no q/k norm, no RoPE (NoPE)
//   moe    LatentMoE: router on the hidden (E per LAYER), fc1_latent 4096 -> 1024, routed experts IN the latent
//          (up 1024 -> I, relu2, down I -> 1024, E and I per LAYER, top-9), fc2_latent 1024 -> 4096, one shared expert
//          on the hidden 4096 (up 4096 -> 5376, relu2, down 5376 -> 4096). The top-level `num_experts` (512) is a
//          maximum and `intermediate_size` (2688) is unused by any block: neither is read for a layer.
//
// E and I come from each block's own `block_configs` entry, never from the top level. The descriptor is a struct of
// plain fields so a test can build a small one and so a synthetic checkpoint (tools/quantize/bolt/make_synth.py
// `--layers N`) or `--layers N` on the real one holds the first N blocks of the real pattern.
namespace model {

enum class BoltBlock : uint8_t { Mamba, Moe, Attention };

// Every GEMV of a block, beside the routed experts' per-expert shapes (ExpertUp / ExpertDown carry ONE expert's shape;
// the loader's blocks hold E of them). `Router` is bf16 (fp32 logits, never rounded).
enum class BoltLinearId { InProj, OutProj, Qkv, OProj, Fc1SharedUp, Fc2, SharedDown, ExpertUp, ExpertDown, Router };

// The per-linear {S, layout} is PROVISIONAL until 25c's P0 sweep (spec 24 §13's correction): layout 1 for every routed
// expert and every N = 4096 linear (out_proj, o_proj, fc2, shared_down), layout 0 for in_proj and the fused q||k||v /
// fc1||shared_up; S 1 everywhere. `BoltDesc::linears(l)` is the ONE source of shapes and layouts - the loader, the
// layouts and the planner all read it.
struct BoltLinear {
  BoltLinearId id;
  GemvShape shape;                  // {K, N, S, layout}
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // mixer-relative checkpoint prefixes, in column order
  std::vector<uint32_t> part_n;     // each part's output columns (sum = shape.N)
};

struct BoltDesc {
  std::string name = "nemotron-3.5-bolt-74b-a9b", architecture = "NemotronH_Omni_Reasoning_V3",
              model_type = "nemotron_h_omni", llm_model_type = "nemotron_h_puzzle";
  uint32_t layers = 88, hidden = 4096, vocab = 131072, vocab_used = 131072, max_pos = 262144;
  uint32_t m_heads = 128, m_head_dim = 64, m_groups = 8, m_state = 128, m_conv = 4;
  uint32_t m_inner = 8192, m_conv_dim = 10240, m_inproj = 18560;   // derived: heads x head_dim; inner + 2 x groups x state;
                                                                    // inner + conv_dim + heads
  uint32_t q_heads = 32, kv_heads = 2, head_dim = 128;
  uint32_t latent = 1024, shared_inter = 5376, top_k = 9;
  float eps = 1e-5f, routed_scale = 5.0f;
  std::vector<BoltBlock> block;                  // [layers]
  std::vector<uint32_t> experts, inter;          // [layers]: E and I of a MoE block, 0 otherwise

  bool is_mamba(uint32_t l) const { return block.at(l) == BoltBlock::Mamba; }
  bool is_moe(uint32_t l) const { return block.at(l) == BoltBlock::Moe; }
  bool is_attention(uint32_t l) const { return block.at(l) == BoltBlock::Attention; }
  uint32_t mamba_index(uint32_t l) const;       // mamba blocks strictly before l
  uint32_t attn_index(uint32_t l) const;        // attention blocks strictly before l
  uint32_t moe_index(uint32_t l) const;         // moe blocks strictly before l
  uint32_t mamba_layers() const { return mamba_index(layers); }
  uint32_t attn_layers() const { return attn_index(layers); }
  uint32_t moe_layers() const { return moe_index(layers); }
  uint32_t max_experts() const;                 // the largest E of any block (0 with no MoE block)
  uint64_t routed_experts() const;              // sum of E over the MoE blocks (13,824 on the real model)

  uint32_t q_n() const { return q_heads * head_dim; }       // 4096
  uint32_t kv_n() const { return kv_heads * head_dim; }     // 256
  uint32_t qkv_n() const { return q_n() + 2 * kv_n(); }     // 4608: q | k | v

  // "language_model.model." - the trunk's checkpoint prefix; `backbone` is NVIDIA's own spelling (the loader accepts it
  // only when the checkpoint ships it: loader::bolt_trunk_prefix).
  static std::string layer_prefix(const std::string& trunk, uint32_t l);   // "<trunk>layers.<l>."
  // The linears of block l in per-token execution order, shapes from the fields above: Mamba InProj + OutProj,
  // Attention Qkv + OProj, Moe Fc1SharedUp, Fc2, SharedDown, ExpertUp, ExpertDown (one expert's shapes) and the Router.
  std::vector<BoltLinear> linears(uint32_t l) const;
};

// Layers [0, split) on device 0, [split, layers) on device 1 (spec 16b's cut, spec 25 §4.1). One device: devices == 1
// and split == layers. Device 0 holds the embedding, the last device norm_f and the head.
struct BoltPlacement {
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const { return dev == 0 ? 0 : split; }
  uint32_t end(uint32_t dev) const { return dev == 0 ? split : layers; }
  uint32_t device_of(uint32_t layer) const { return devices > 1 && layer >= split ? 1 : 0; }
  uint32_t count(uint32_t dev) const { return end(dev) - first(dev); }
  static BoltPlacement one(const BoltDesc& d) { return {1, d.layers, d.layers}; }
  static BoltPlacement two(const BoltDesc& d, uint32_t split) { return {2, split, d.layers}; }
};
// Throws std::invalid_argument unless devices is 1 (split == layers) or 2 (1 <= split <= layers - 1), and the placement
// covers d's layers.
void validate(const BoltPlacement& p, const BoltDesc& d);

// The published model (the real 88-block checkpoint). A process-lifetime singleton.
const BoltDesc& bolt();
bool is_bolt(const common::json::Value& config);   // model_type nemotron_h_omni over llm_config.model_type nemotron_h_puzzle

// Throws std::invalid_argument naming the key and its value when config.json describes a Bolt the engine is not written
// for: another model_type (wrapper or llm_config), another hidden / vocab / head count / Mamba or MoE width,
// num_nextn_predict_layers > 0 or a non-empty mtp_block_configs, residual_in_fp32 true, use_qk_norm true, n_group or
// topk_group != 1, scoring_func != sigmoid, norm_topk_prob false, routed_scaling_factor != 5, a block's
// num_experts_per_tok != 9, an n_routed_experts not a multiple of 16 or above 512, a moe_intermediate_size not a multiple
// of 64, a block_type of another kind, layers_block_type disagreeing with block_configs, a non-null sliding_window,
// mamba_proj_bias / attention_bias / mlp_bias / use_bias true, use_conv_bias false, hidden_act / mlp_hidden_act !=
// relu2, mamba_hidden_act != silu, time_step_limit != [0, inf], tie_word_embeddings true at either level, an eps other
// than 1e-5. The block list may be a PREFIX of the real model's (a synthetic / --layers checkpoint holds 1..88 blocks);
// any pattern of the three kinds is accepted (E and I are per block, read from it).
void check_bolt_config(const common::json::Value& config);
// config.json -> the descriptor at the checkpoint's own block count (check_bolt_config first). vocab_used = vocab (every
// row of the 131072 vocabulary is a token - spec 25 §1.1).
BoltDesc bolt_desc(const common::json::Value& config);
// The descriptor truncated to its first `layers` blocks (--layers N). Throws when layers is 0 or above desc.layers.
BoltDesc bolt_truncated(const BoltDesc& d, uint32_t layers);

}  // namespace model
