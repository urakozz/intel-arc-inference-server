#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 18 (18b): K2-Horizon MoVA 36B-A4B (`K2HorizonForCausalLM`, config.json
// model_type `k2_horizon`) as data - a second model BESIDE qwen3_5 (spec 18 §2 / spec 4
// ruling 4: no GDN, a grouped plain-weight norm, MoVA attention, a sigmoid router; it
// shares GemvShape / WeightKind / Fuse and nothing of the Qwen layer plan).
//
// Every number of `k2()` is read from the operator's int4 checkpoint
// `urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` (config.json and the five
// safetensors headers, fetched 2026-10-05 - no weight was downloaded):
//
//   layers 48: 0-2 dense (`mlp_only_layers` [0, 1, 2]: plain attention + SwiGLU 6144),
//              3-47 sparse (MoVA attention + MoE; `decoder_sparse_step` 1)
//   hidden 2560, `layernorm_num_groups` 2 (two groups of 1280), rms_norm_eps 1e-6
//   attention  32 q / 8 kv heads x 128 (GQA 4), every layer full attention, RoPE over all
//              128 dims (`rope_head_dim` 128, rope_theta 1e7, rotate_half), no q/k norm
//              (`query_key_norm` false), `attention_gate_func` softplus (gate_proj 2560 ->
//              4096, softplus with beta = ln 2, threshold 20)
//   MoVA       `mova_num_experts` 64 value experts (self_attn.v_experts.E: 2560 -> 1024,
//              SiLU), top 4 (`mova_num_experts_per_tok`), router self_attn.v_router int4
//              2560 -> 64 + an F16 bias used for selection only
//   MoE        `num_experts` 100 (mlp.experts.E.{gate,up,down}_proj, 768), top 8, router
//              mlp.gate bf16 [100][2560] + a bf16 selection-only bias, sigmoid scores,
//              `norm_topk_prob` true, `router_scaling_factor` 2.5, one shared expert
//              (mlp.shared_experts.*, 768, ungated)
//   vocab 250624 (fully used), bos 0, eos [1, 250019] (generation_config.json)
//   max_position_embeddings 524288; tie_word_embeddings false; no mtp.* tensors
//   quantisation GPTQ int4 g64 sym, desc_act false, 45 `dynamic` exclusions (the 45
//              mlp.gate routers); embed_tokens, lm_head, the norms and mlp.gate bf16
//
// The tensor names, as model.safetensors.index.json spells them (67,290 entries, 5 shards),
// layer-relative after `model.layers.<l>.`:
//
//   every layer   input_layernorm.weight, post_attention_layernorm.weight      BF16 [2560]
//                 self_attn.{q,k,gate,o}_proj.{qweight,scales,qzeros,g_idx}     int4 g64
//   dense         self_attn.v_proj.*, mlp.{gate,up,down}_proj.*                 int4 g64
//   sparse        self_attn.v_router.{qweight,scales,qzeros,g_idx}              int4 g64
//                 self_attn.v_router.bias                                       F16  [64]
//                 self_attn.v_experts.<E>.{qweight,scales,qzeros,g_idx}         E < 64
//                 mlp.gate.weight                                               BF16 [100][2560]
//                 mlp.gate.bias                                                 BF16 [100]
//                 mlp.experts.<E>.{gate,up,down}_proj.{qweight,...}             E < 100
//                 mlp.shared_experts.{gate,up,down}_proj.{qweight,...}
//   top level     model.embed_tokens.weight, lm_head.weight  BF16 [250624][2560];
//                 model.norm.weight BF16 [2560]
//
// The descriptor is a struct of plain fields so a test can build a small one (the
// synthetic checkpoint fixture, tests/loader/k2_repack_test.cc); every width the loader,
// the buffers, the planner and the capture use is DERIVED here from those fields.
namespace model {

// K2's linears that are not routed experts. Every int4 row is GPTQ layout 0.
enum class K2LinearId {
  AttnDense,    // q || k || gate || v           (dense layers)   2560 -> 10240
  AttnSparse,   // q || k || gate || v_router    (MoVA layers)    2560 -> 9280
  OProj,        // q-heads x 128 -> hidden                        4096 -> 2560
  DenseGateUp,  // gate || up, interleaved in 16-column blocks    2560 -> 12288
  DenseDown,    // 6144 -> 2560
  LmHead,       // bf16 (the checkpoint's) or int8 at load (spec 9)   2560 -> 250624
  kCount
};

struct K2Linear {
  K2LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative checkpoint prefixes, in column order
};

// Routed experts: one flat device buffer per (layer, group), int4 g64 layout-1 blocks
// (common/repack.h) addressed by block index inside the kernels (spec 18 decision 3).
// The MoE groups carry the shared expert as their LAST block (index `experts`), so one
// launch serves the top-8 and the shared expert (spec 15 §9's "ninth slot").
enum class K2ExpertId { Value, MoeGateUp, MoeDown, kCount };

struct K2ExpertGroup {
  K2ExpertId id;
  uint32_t K, N;                        // ONE expert's fused shape
  uint32_t blocks;                      // routed experts (+ 1 shared for the MoE groups)
  std::vector<std::string> parts;       // per-expert templates, "{e}" = the expert id
  std::vector<std::string> shared;      // the shared expert's parts (MoE groups only)
};

// The small tensors, widened to fp32 VERBATIM - K2's RMSNorm multiplies by the plain
// weight (`self.weight * hidden_states`, modeling_k2_horizon.py:634): no `1 + w`, the
// first of spec 18 §3's two norm traps. The source dtype is the whole bake.
enum class K2SmallBlock { Norms, Route };
struct K2SmallTensor {
  std::string name;   // layer-relative
  uint32_t elems;
  const char* dtype;  // "BF16" | "F16", as safetensors spells it
  K2SmallBlock block;
  uint32_t offset;    // bytes within the block
};

struct K2Desc {
  std::string name;           // "k2-horizon-mova-36b-a4b"
  std::string architecture;   // config.json architectures[0]
  std::string model_type;     // config.json model_type: the CLIs dispatch on it
  uint32_t layers = 0, dense_layers = 0;   // dense = the first `dense_layers` (mlp_only_layers)
  uint32_t hidden = 0, norm_groups = 0;
  uint32_t q_heads = 0, kv_heads = 0, head_dim = 0;
  uint32_t dense_inter = 0, moe_inter = 0;
  uint32_t experts = 0, top_k = 0;              // MoE
  uint32_t value_experts = 0, value_top_k = 0;  // MoVA
  uint32_t vocab = 0, vocab_used = 0;
  uint32_t bos = 0;
  std::vector<uint32_t> eos;
  double rope_theta = 0;
  float router_scale = 0;   // × after normalising the top-k scores (2.5)
  float rms_eps = 0;
  // The int4 rows' split-K (gemv.cl S; layout 0 throughout). PROVISIONAL - copied from
  // the nearest measured Qwen3.8 / Ornith cells (q||k||v S2, o_proj / down S4, gate||up
  // S4 because k2_silu_mul folds S slices), for 18b Task 4's sweep to replace.
  uint32_t attn_s = 0, oproj_s = 0, gate_up_s = 0, down_s = 0;

  bool is_dense(uint32_t layer) const { return layer < dense_layers; }
  uint32_t sparse_layers() const { return layers - dense_layers; }

  // --- derived widths -------------------------------------------------------------
  uint32_t group_size() const { return hidden / norm_groups; }       // 1280
  uint32_t q_n() const { return q_heads * head_dim; }                // 4096
  uint32_t kv_n() const { return kv_heads * head_dim; }              // 1024
  uint32_t gqa() const { return q_heads / kv_heads; }                // 4
  // The fused attention GEMV's column map, the SAME in both rows so one kernel serves
  // both: q [0, q_n) · k [k_off, +kv_n) · gate [gate_off, +q_n) · v or v_router from v_off.
  uint32_t k_off() const { return q_n(); }                           // 4096
  uint32_t gate_off() const { return q_n() + kv_n(); }               // 5120
  uint32_t v_off() const { return 2 * q_n() + kv_n(); }              // 9216
  uint32_t attn_dense_n() const { return v_off() + kv_n(); }         // 10240
  uint32_t attn_sparse_n() const { return v_off() + value_experts; } // 9280
  // The MoE router's rows on the device: the experts' rows, zero rows up to the route
  // kernel's lane count - the next power of two, at least 16 (gemv_bf16's tile). 128 for
  // K2's 100: the top-k reads rows 0..experts-1 only (spec 18b Review Focus 4).
  uint32_t router_n() const;
  uint32_t moe_blocks() const { return experts + 1; }   // routed + the shared expert, last
  uint32_t shared_block() const { return experts; }
  // The route kernels' work-group: one lane per expert, a power of two (64 / 128).
  static uint32_t route_wg(uint32_t experts);

  // --- the small blocks (bytes) ------------------------------------------------------
  // Norms, every layer: input w fp32 [hidden] || post w fp32 [hidden].
  uint32_t norms_off_input() const { return 0; }
  uint32_t norms_off_post() const { return hidden * 4; }
  uint32_t norms_bytes() const { return 2 * hidden * 4; }
  // Route, sparse layers: the MoE bias fp32 [router_n] (zero past `experts`) || the MoVA
  // bias fp32 [value_experts]. Both are used ONLY for selection (spec 18 §3).
  uint32_t route_off_moe() const { return 0; }
  uint32_t route_off_mova() const { return router_n() * 4; }
  uint32_t route_bytes() const { return (router_n() + value_experts) * 4; }

  // --- the tables ---------------------------------------------------------------------
  K2Linear linear(K2LinearId id) const;   // throws on kCount
  // The per-token execution order of one layer's linears (the lm_head is not in it).
  std::vector<K2LinearId> layer_linears(uint32_t layer) const;
  std::vector<K2ExpertGroup> expert_groups() const;   // a sparse layer's three
  std::vector<K2SmallTensor> small_tensors(uint32_t layer) const;
  static std::string layer_prefix(uint32_t layer);    // "model.layers.<l>."
  // The template with "{e}" replaced by `e`; throws if the template has none.
  static std::string expert_part(const std::string& tmpl, uint32_t e);
  // RoPE table bytes: fp32 [max_len][2][head_dim / 2] (k2_rope_table).
  size_t rope_table_bytes(uint32_t max_len) const {
    return size_t(max_len) * 2 * (head_dim / 2) * sizeof(float);
  }
};

// The descriptor of the operator's checkpoint (values above). A process-lifetime singleton.
const K2Desc& k2();

// config.json's model_type, read without parsing anything else ("" when absent). The
// CLIs dispatch on it (spec 18b): "k2_horizon" goes to the K2 path.
std::string model_type_of(const common::json::Value& config);
bool is_k2_model_type(const std::string& model_type);

// Throws std::runtime_error naming the key when config.json describes a K2-Horizon the
// engine is not written for: every structural key of the descriptor (layers, heads,
// widths, experts, the dense-layer list, the norm groups, the router function and its
// bias, the attention gate, RoPE over all head dims, no q/k norm, no sliding window, one
// shared expert, untied embeddings) is held to `d`. Host only: tests/model/k2_horizon_test.cc
// runs it on the checkpoint's real config.json.
void check_k2_config(const K2Desc& d, const common::json::Value& config);

}  // namespace model
