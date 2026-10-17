#include "model/lfm2moe.h"

#include <cmath>
#include <stdexcept>

namespace model {
namespace {

std::vector<LfmKind> real_kinds(uint32_t layers) {
  std::vector<LfmKind> k(layers);
  for (uint32_t l = 0; l < layers; ++l) k[l] = (l % 4 == 2) ? LfmKind::Attention : LfmKind::Conv;
  return k;
}

Lfm2MoeDesc make_lfm2moe() {
  Lfm2MoeDesc d;
  d.name = "lfm2.5-24b-a2b";
  d.architecture = "Lfm2MoeForCausalLM";
  d.model_type = "lfm2_moe";
  d.prefix = "model.";
  d.layers = 40;
  d.hidden = 2048;
  d.vocab = 128000;
  d.vocab_used = 125017;
  d.trained_max_len = 131072;
  d.q_heads = 32;
  d.kv_heads = 8;
  d.head_dim = 64;
  d.rope_theta = 1e7;
  d.conv_taps = 3;
  d.dense_layers = 2;
  d.dense_inter = 11776;
  d.experts = 64;
  d.top_k = 4;
  d.moe_inter = 1536;
  d.norm_eps = 1e-5f;
  d.bos = 124894;
  d.eos = 124900;
  d.pad = 124893;
  d.kinds = real_kinds(d.layers);
  return d;
}

[[noreturn]] void bad(const std::string& what) {
  throw std::invalid_argument("config.json: " + what + " - lfm2.5-24b-a2b (model/lfm2moe.h) is not written for it");
}

uint32_t need_u32(const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()))
    bad(std::string("'") + key + "' is missing or not a non-negative integer");
  return uint32_t(v->num());
}

void want_u32(const common::json::Value& c, const char* key, uint32_t want) {
  const uint32_t got = need_u32(c, key);
  if (got != want)
    bad(std::string("'") + key + "' is " + std::to_string(got) + ", the descriptor says " + std::to_string(want));
}

void want_bool(const common::json::Value& c, const char* key, bool want, bool absent_as) {
  const common::json::Value* v = c.find(key);
  if (v && !v->is_bool()) bad(std::string("'") + key + "' is not a boolean");
  const bool got = v ? v->boolean() : absent_as;
  if (got != want)
    bad(std::string("'") + key + "' is " + (got ? "true" : "false") + ", the descriptor needs " +
        (want ? "true" : "false"));
}

void want_str(const common::json::Value& c, const char* key, const char* want) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_string() || v->str() != want)
    bad(std::string("'") + key + "' is " + (v && v->is_string() ? "'" + v->str() + "'" : std::string("absent")) +
        ", the descriptor needs '" + want + "'");
}

void check_layer_types(const common::json::Value& c, uint32_t layers) {
  const common::json::Value* lt = c.find("layer_types");
  if (!lt || !lt->is_array() || lt->arr().size() != layers)
    bad("'layer_types' is not a list of num_hidden_layers (" + std::to_string(layers) + ") entries");
  const std::vector<LfmKind> real = real_kinds(layers);
  for (uint32_t l = 0; l < layers; ++l) {
    const common::json::Value& v = lt->arr()[l];
    const char* want = real[l] == LfmKind::Attention ? "full_attention" : "conv";
    if (!v.is_string() || v.str() != want)
      bad("'layer_types'[" + std::to_string(l) + "] is " +
          (v.is_string() ? "'" + v.str() + "'" : std::string("not a string")) +
          ", the published pattern (full_attention at l % 4 == 2) has '" + want + "'");
  }
}

}  // namespace

const char* lfm_expert_names_name(LfmExpertNames n) { return n == LfmExpertNames::Proj ? "proj" : "w"; }

LfmPlacement LfmPlacement::one(const Lfm2MoeDesc& d) { return {1, d.layers, d.layers}; }
LfmPlacement LfmPlacement::two(const Lfm2MoeDesc& d, uint32_t split) { return {2, split, d.layers}; }

void validate(const LfmPlacement& p, const Lfm2MoeDesc& d) {
  if (p.layers != d.layers)
    throw std::invalid_argument("LfmPlacement: placed " + std::to_string(p.layers) + " layers, " + d.name +
                                " has " + std::to_string(d.layers));
  if (p.devices == 1) {
    if (p.split != d.layers)
      throw std::invalid_argument("LfmPlacement: one device holds every layer (split " +
                                  std::to_string(p.split) + " != " + std::to_string(d.layers) + ")");
    return;
  }
  if (p.devices != 2)
    throw std::invalid_argument("LfmPlacement: " + std::to_string(p.devices) +
                                " devices (spec 24 §4.5 builds and gates two)");
  if (p.split < 1 || p.split + 1 > d.layers)
    throw std::invalid_argument("LfmPlacement: split " + std::to_string(p.split) + " is outside [1, " +
                                std::to_string(d.layers > 0 ? d.layers - 1 : 0) +
                                "] - each device must hold at least one layer");
}

bool Lfm2MoeDesc::is_attention(uint32_t l) const { return kinds.at(l) == LfmKind::Attention; }
bool Lfm2MoeDesc::is_dense(uint32_t l) const { return l < dense_layers; }

uint32_t Lfm2MoeDesc::attn_index(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l; ++i) n += kinds.at(i) == LfmKind::Attention ? 1 : 0;
  return n;
}
uint32_t Lfm2MoeDesc::conv_index(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l; ++i) n += kinds.at(i) == LfmKind::Conv ? 1 : 0;
  return n;
}
uint32_t Lfm2MoeDesc::attn_layers() const { return attn_index(layers); }
uint32_t Lfm2MoeDesc::conv_layers() const { return conv_index(layers); }

std::string Lfm2MoeDesc::layer_prefix(const std::string& model_prefix, uint32_t l) {
  return model_prefix + "layers." + std::to_string(l) + ".";
}

std::vector<LfmLinear> Lfm2MoeDesc::linears(uint32_t l) const {
  std::vector<LfmLinear> v;
  if (is_attention(l)) {
    v.push_back({LfmLinearId::Qkv, {hidden, qkv_n(), 1, 0}, WeightKind::Int4, Fuse::Concat,
                {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}});
    v.push_back({LfmLinearId::OProj, {q_n(), hidden, 1, 0}, WeightKind::Int4, Fuse::Single, {"self_attn.out_proj"}});
  } else {
    v.push_back({LfmLinearId::InProj, {hidden, 3 * hidden, 1, 0}, WeightKind::Int4, Fuse::Single, {"conv.in_proj"}});
    v.push_back({LfmLinearId::OutProj, {hidden, hidden, 1, 0}, WeightKind::Int4, Fuse::Single, {"conv.out_proj"}});
  }
  if (is_dense(l)) {
    v.push_back({LfmLinearId::DenseGateUp, {hidden, 2 * dense_inter, 1, 0}, WeightKind::Int4, Fuse::Interleave16,
                {"feed_forward.w1", "feed_forward.w3"}});
    v.push_back({LfmLinearId::DenseDown, {dense_inter, hidden, 1, 0}, WeightKind::Int4, Fuse::Single,
                {"feed_forward.w2"}});
  } else {
    v.push_back({LfmLinearId::Router, {hidden, experts, 1, 0}, WeightKind::Bf16, Fuse::Single,
                {"feed_forward.gate"}});
  }
  return v;
}

const Lfm2MoeDesc& lfm2moe() {
  static const Lfm2MoeDesc d = make_lfm2moe();
  return d;
}

bool is_lfm2moe(const common::json::Value& config) {
  if (!config.is_object()) return false;
  const common::json::Value* v = config.find("model_type");
  return v && v->is_string() && v->str() == "lfm2_moe";
}

void check_lfm2moe_config(const common::json::Value& c) {
  const Lfm2MoeDesc& d = lfm2moe();
  if (!c.is_object()) bad("not a JSON object");
  want_str(c, "model_type", d.model_type.c_str());
  const common::json::Value* archs = c.find("architectures");
  if (!archs || !archs->is_array() || archs->arr().empty() || !archs->arr()[0].is_string() ||
      archs->arr()[0].str() != d.architecture)
    bad("architectures[0] is not '" + d.architecture + "'");
  const uint32_t n = need_u32(c, "num_hidden_layers");
  if (n < 1 || n > d.layers)
    bad("'num_hidden_layers' is " + std::to_string(n) + ": an LFM2.5-24B-A2B checkpoint has 1.." +
        std::to_string(d.layers) + " layers (the real 40, or a synthetic / --layers prefix)");
  check_layer_types(c, n);
  want_u32(c, "hidden_size", d.hidden);
  want_u32(c, "num_attention_heads", d.q_heads);
  want_u32(c, "num_key_value_heads", d.kv_heads);
  if (d.hidden % d.q_heads != 0 || d.hidden / d.q_heads != d.head_dim)
    bad("'hidden_size' / 'num_attention_heads' does not give head_dim " + std::to_string(d.head_dim));
  want_u32(c, "num_dense_layers", d.dense_layers);
  want_u32(c, "intermediate_size", d.dense_inter);
  want_u32(c, "moe_intermediate_size", d.moe_inter);
  want_u32(c, "num_experts", d.experts);
  want_u32(c, "num_experts_per_tok", d.top_k);
  want_u32(c, "vocab_size", d.vocab);
  want_u32(c, "conv_L_cache", d.conv_taps);
  want_bool(c, "conv_bias", false, false);
  want_bool(c, "use_expert_bias", true, false);
  want_bool(c, "norm_topk_prob", true, false);
  want_bool(c, "tie_word_embeddings", true, false);
  const common::json::Value* eps = c.find("norm_eps");
  if (!eps || !eps->is_number() || float(eps->num()) != d.norm_eps) bad("'norm_eps' is not 1e-5");
  const common::json::Value* rsf = c.find("routed_scaling_factor");
  if (!rsf || !rsf->is_number() || rsf->num() != 1.0) bad("'routed_scaling_factor' is not 1.0");
  const common::json::Value* rp = c.find("rope_parameters");
  if (!rp || !rp->is_object()) bad("'rope_parameters' is missing or not an object");
  const common::json::Value* rtype = rp->find("rope_type");
  if (!rtype || !rtype->is_string() || rtype->str() != "default")
    bad("'rope_parameters.rope_type' is " +
        (rtype && rtype->is_string() ? "'" + rtype->str() + "'" : std::string("not a string")) +
        ", only 'default' is implemented");
  const common::json::Value* theta = rp->find("rope_theta");
  if (!theta || !theta->is_number() || theta->num() != d.rope_theta) bad("'rope_parameters.rope_theta' is not 1e7");
  want_u32(c, "max_position_embeddings", d.trained_max_len);
  want_u32(c, "bos_token_id", d.bos);
  want_u32(c, "eos_token_id", d.eos);
  want_u32(c, "pad_token_id", d.pad);
}

Lfm2MoeDesc lfm2moe_desc(const common::json::Value& config, uint32_t vocab_used) {
  check_lfm2moe_config(config);
  Lfm2MoeDesc d = lfm2moe();
  d.layers = need_u32(config, "num_hidden_layers");
  d.kinds = real_kinds(d.layers);
  d.vocab_used = vocab_used;
  return d;
}

}  // namespace model
