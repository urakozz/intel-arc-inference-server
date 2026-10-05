#include "model/k2_horizon.h"

#include <cmath>
#include <stdexcept>

namespace model {
namespace {

K2Desc make_k2() {
  K2Desc d;
  d.name = "k2-horizon-mova-36b-a4b";
  d.architecture = "K2HorizonForCausalLM";
  d.model_type = "k2_horizon";
  d.layers = 48;
  d.dense_layers = 3;
  d.hidden = 2560;
  d.norm_groups = 2;
  d.q_heads = 32;
  d.kv_heads = 8;
  d.head_dim = 128;
  d.dense_inter = 6144;
  d.moe_inter = 768;
  d.experts = 100;
  d.top_k = 8;
  d.value_experts = 64;
  d.value_top_k = 4;
  d.vocab = 250624;
  d.vocab_used = 250624;   // fully used (spec 18 §1): the argmax masks nothing
  d.bos = 0;
  d.eos = {1, 250019};
  d.rope_theta = 1e7;
  d.router_scale = 2.5f;
  d.rms_eps = 1e-6f;
  d.attn_s = 2;      // PROVISIONAL (header)
  d.oproj_s = 4;
  d.gate_up_s = 4;
  d.down_s = 4;
  return d;
}

}  // namespace

uint32_t K2Desc::route_wg(uint32_t e) {
  uint32_t w = 16;
  while (w < e) w *= 2;
  return w;
}

uint32_t K2Desc::router_n() const { return route_wg(experts); }

K2Linear K2Desc::linear(K2LinearId id) const {
  const uint32_t H = hidden;
  switch (id) {
    case K2LinearId::AttnDense:
      return {id, {H, attn_dense_n(), attn_s, 0}, WeightKind::Int4, Fuse::Concat,
              {"self_attn.q_proj", "self_attn.k_proj", "self_attn.gate_proj", "self_attn.v_proj"}};
    case K2LinearId::AttnSparse:
      // MoVA: the int4 router's logits ride in v's place (q_n + kv_n + q_n + 64 columns).
      return {id, {H, attn_sparse_n(), attn_s, 0}, WeightKind::Int4, Fuse::Concat,
              {"self_attn.q_proj", "self_attn.k_proj", "self_attn.gate_proj", "self_attn.v_router"}};
    case K2LinearId::OProj:
      return {id, {q_n(), H, oproj_s, 0}, WeightKind::Int4, Fuse::Single, {"self_attn.o_proj"}};
    case K2LinearId::DenseGateUp:
      return {id, {H, 2 * dense_inter, gate_up_s, 0}, WeightKind::Int4, Fuse::Interleave16,
              {"mlp.gate_proj", "mlp.up_proj"}};
    case K2LinearId::DenseDown:
      return {id, {dense_inter, H, down_s, 0}, WeightKind::Int4, Fuse::Single, {"mlp.down_proj"}};
    case K2LinearId::LmHead:
      return {id, {H, vocab, 1, 0}, WeightKind::Bf16, Fuse::Single, {"lm_head"}};
    case K2LinearId::kCount:
      break;
  }
  throw std::out_of_range("K2Desc::linear: no row for ordinal " +
                          std::to_string(static_cast<int>(id)));
}

std::vector<K2LinearId> K2Desc::layer_linears(uint32_t layer) const {
  if (is_dense(layer))
    return {K2LinearId::AttnDense, K2LinearId::OProj, K2LinearId::DenseGateUp,
            K2LinearId::DenseDown};
  return {K2LinearId::AttnSparse, K2LinearId::OProj};
}

std::vector<K2ExpertGroup> K2Desc::expert_groups() const {
  return {
      {K2ExpertId::Value, hidden, kv_n(), value_experts, {"self_attn.v_experts.{e}"}, {}},
      {K2ExpertId::MoeGateUp, hidden, 2 * moe_inter, moe_blocks(),
       {"mlp.experts.{e}.gate_proj", "mlp.experts.{e}.up_proj"},
       {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"}},
      {K2ExpertId::MoeDown, moe_inter, hidden, moe_blocks(), {"mlp.experts.{e}.down_proj"},
       {"mlp.shared_experts.down_proj"}},
  };
}

std::vector<K2SmallTensor> K2Desc::small_tensors(uint32_t layer) const {
  std::vector<K2SmallTensor> v = {
      {"input_layernorm.weight", hidden, "BF16", K2SmallBlock::Norms, norms_off_input()},
      {"post_attention_layernorm.weight", hidden, "BF16", K2SmallBlock::Norms, norms_off_post()},
  };
  if (!is_dense(layer)) {
    v.push_back({"mlp.gate.bias", experts, "BF16", K2SmallBlock::Route, route_off_moe()});
    v.push_back({"self_attn.v_router.bias", value_experts, "F16", K2SmallBlock::Route,
                 route_off_mova()});
  }
  return v;
}

std::string K2Desc::layer_prefix(uint32_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

std::string K2Desc::expert_part(const std::string& tmpl, uint32_t e) {
  const size_t at = tmpl.find("{e}");
  if (at == std::string::npos)
    throw std::invalid_argument("K2Desc::expert_part: template '" + tmpl + "' has no {e}");
  return tmpl.substr(0, at) + std::to_string(e) + tmpl.substr(at + 3);
}

const K2Desc& k2() {
  static const K2Desc d = make_k2();
  return d;
}

std::string model_type_of(const common::json::Value& config) {
  if (!config.is_object()) return "";
  const common::json::Value* v = config.find("model_type");
  return v && v->is_string() ? v->str() : "";
}

bool is_k2_model_type(const std::string& t) { return t == "k2_horizon"; }

namespace {

[[noreturn]] void bad(const K2Desc& d, const std::string& what) {
  throw std::runtime_error("config.json: " + what + " - " + d.name +
                           " (model/k2_horizon.h) is not written for it");
}

uint32_t need_u32(const K2Desc& d, const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()))
    bad(d, std::string("'") + key + "' is missing or not a non-negative integer");
  return uint32_t(v->num());
}

void want_u32(const K2Desc& d, const common::json::Value& c, const char* key, uint32_t want) {
  const uint32_t got = need_u32(d, c, key);
  if (got != want)
    bad(d, std::string("'") + key + "' is " + std::to_string(got) + ", the descriptor says " +
               std::to_string(want));
}

void want_bool(const K2Desc& d, const common::json::Value& c, const char* key, bool want,
               bool absent_ok_as) {
  const common::json::Value* v = c.find(key);
  const bool got = v ? (v->is_bool() ? v->boolean() : !want) : absent_ok_as;
  if (v && !v->is_bool()) bad(d, std::string("'") + key + "' is not a boolean");
  if (got != want)
    bad(d, std::string("'") + key + "' is " + (got ? "true" : "false") + ", the descriptor needs " +
               (want ? "true" : "false"));
}

void want_str(const K2Desc& d, const common::json::Value& c, const char* key, const char* want) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_string() || v->str() != want)
    bad(d, std::string("'") + key + "' is " + (v && v->is_string() ? "'" + v->str() + "'" : "absent") +
               ", the descriptor needs '" + want + "'");
}

}  // namespace

void check_k2_config(const K2Desc& d, const common::json::Value& c) {
  if (!c.is_object()) bad(d, "not a JSON object");
  want_str(d, c, "model_type", d.model_type.c_str());
  const common::json::Value* archs = c.find("architectures");
  if (!archs || !archs->is_array() || archs->arr().empty() || !archs->arr()[0].is_string() ||
      archs->arr()[0].str() != d.architecture)
    bad(d, "architectures[0] is not '" + d.architecture + "'");
  want_u32(d, c, "num_hidden_layers", d.layers);
  want_u32(d, c, "hidden_size", d.hidden);
  want_u32(d, c, "layernorm_num_groups", d.norm_groups);
  want_u32(d, c, "num_attention_heads", d.q_heads);
  want_u32(d, c, "num_key_value_heads", d.kv_heads);
  want_u32(d, c, "head_dim", d.head_dim);
  // RoPE over the whole head (rope_head_dim == head_dim: apply_rotary_pos_emb, rotate_half),
  // not the split_to_interleaved partial path of modeling_k2_horizon.py:270.
  want_u32(d, c, "rope_head_dim", d.head_dim);
  want_u32(d, c, "intermediate_size", d.dense_inter);
  want_u32(d, c, "moe_intermediate_size", d.moe_inter);
  want_u32(d, c, "num_experts", d.experts);
  want_u32(d, c, "num_experts_per_tok", d.top_k);
  want_u32(d, c, "mova_num_experts", d.value_experts);
  want_u32(d, c, "mova_num_experts_per_tok", d.value_top_k);
  want_u32(d, c, "num_shared_experts", 1);
  want_u32(d, c, "decoder_sparse_step", 1);
  want_u32(d, c, "vocab_size", d.vocab);
  // mlp_only_layers = [0, dense_layers): the dense layers lead, every other layer is sparse.
  const common::json::Value* mol = c.find("mlp_only_layers");
  if (!mol || !mol->is_array() || mol->arr().size() != d.dense_layers)
    bad(d, "'mlp_only_layers' is not a list of " + std::to_string(d.dense_layers) + " layers");
  for (uint32_t i = 0; i < d.dense_layers; ++i)
    if (!mol->arr()[i].is_number() || mol->arr()[i].num() != double(i))
      bad(d, "'mlp_only_layers' is not [0 .. " + std::to_string(d.dense_layers - 1) + "]");
  want_str(d, c, "router_score_func", "sigmoid");
  want_str(d, c, "attention_gate_func", "softplus");
  want_str(d, c, "hidden_act", "silu");
  want_bool(d, c, "moe_gate_bias", true, false);
  want_bool(d, c, "norm_topk_prob", true, false);
  want_bool(d, c, "query_key_norm", false, false);
  want_bool(d, c, "attention_bias", false, false);
  want_bool(d, c, "tie_word_embeddings", false, false);
  want_bool(d, c, "use_sliding_window", false, false);
  if (const common::json::Value* sw = c.find("sliding_window"); sw && !sw->is_null())
    bad(d, "'sliding_window' is set; every K2 layer is full attention");
  const common::json::Value* rsf = c.find("router_scaling_factor");
  if (!rsf || !rsf->is_number() || float(rsf->num()) != d.router_scale)
    bad(d, "'router_scaling_factor' is not " + std::to_string(d.router_scale));
  const common::json::Value* eps = c.find("rms_norm_eps");
  if (!eps || !eps->is_number() || float(eps->num()) != d.rms_eps)
    bad(d, "'rms_norm_eps' is not 1e-6");
  // rope_parameters (transformers 5) or the older top-level rope_theta.
  double theta = 0;
  if (const common::json::Value* rp = c.find("rope_parameters"); rp && rp->is_object()) {
    if (const common::json::Value* t = rp->find("rope_type"); t && (!t->is_string() || t->str() != "default"))
      bad(d, "rope_parameters.rope_type is not 'default'");
    if (const common::json::Value* t = rp->find("rope_theta"); t && t->is_number()) theta = t->num();
  } else if (const common::json::Value* t = c.find("rope_theta"); t && t->is_number()) {
    theta = t->num();
  }
  if (theta != d.rope_theta) bad(d, "rope_theta is not " + std::to_string(d.rope_theta));
}

}  // namespace model
