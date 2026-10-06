#include "model/kolibri1.h"

#include <cmath>
#include <stdexcept>

namespace model {
namespace {

Kolibri1Desc make_kolibri1() {
  Kolibri1Desc d;
  d.name = "kolibri-1";
  d.architecture = "Kolibri1ForCausalLM";
  d.model_type = "kolibri1";
  d.layers = 50;
  d.hidden = 2560;
  d.q_heads = 48;
  d.kv_heads = 4;
  d.head_dim = 128;
  d.experts = 384;
  d.top_k = 6;
  d.moe_inter = 512;
  d.shared_inter = 512;
  d.vocab = 128000;
  d.vocab_used = 128000;   // the head's 128000 rows, argmax over all (the reference's)
  d.window = 513;
  d.full_every = 5;
  d.trained_max_len = 262144;
  d.eos = {127906, 127901};
  d.rope_theta = 1e4;
  d.rms_eps = 1e-6f;
  d.attn = KolAttnForm::Int4;
  d.qkv_s = 2;    // PROVISIONAL (header)
  d.oproj_s = 4;
  return d;
}

[[noreturn]] void bad(const Kolibri1Desc& d, const std::string& what) {
  throw std::runtime_error("config.json: " + what + " - " + d.name +
                           " (model/kolibri1.h) is not written for it");
}

uint32_t need_u32(const Kolibri1Desc& d, const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()))
    bad(d, std::string("'") + key + "' is missing or not a non-negative integer");
  return uint32_t(v->num());
}

void want_u32(const Kolibri1Desc& d, const common::json::Value& c, const char* key, uint32_t want) {
  const uint32_t got = need_u32(d, c, key);
  if (got != want)
    bad(d, std::string("'") + key + "' is " + std::to_string(got) + ", the descriptor says " +
               std::to_string(want));
}

void want_bool(const Kolibri1Desc& d, const common::json::Value& c, const char* key, bool want,
               bool absent_as) {
  const common::json::Value* v = c.find(key);
  if (v && !v->is_bool()) bad(d, std::string("'") + key + "' is not a boolean");
  const bool got = v ? v->boolean() : absent_as;
  if (got != want)
    bad(d, std::string("'") + key + "' is " + (got ? "true" : "false") + ", the descriptor needs " +
               (want ? "true" : "false"));
}

void want_str(const Kolibri1Desc& d, const common::json::Value& c, const char* key, const char* want,
              bool absent_ok) {
  const common::json::Value* v = c.find(key);
  if (!v && absent_ok) return;
  if (!v || !v->is_string() || v->str() != want)
    bad(d, std::string("'") + key + "' is " + (v && v->is_string() ? "'" + v->str() + "'" : "absent") +
               ", the descriptor needs '" + want + "'");
}

// layer_types: `layers` entries, the real pattern's prefix (full exactly at l % 5 == 4).
void check_layer_types(const Kolibri1Desc& d, const common::json::Value& c, uint32_t layers) {
  const common::json::Value* lt = c.find("layer_types");
  if (!lt || !lt->is_array() || lt->arr().size() != layers)
    bad(d, "'layer_types' is not a list of num_hidden_layers (" + std::to_string(layers) + ") entries");
  for (uint32_t l = 0; l < layers; ++l) {
    const common::json::Value& v = lt->arr()[l];
    const char* want = d.is_sliding(l) ? "sliding_attention" : "full_attention";
    if (!v.is_string() || v.str() != want)
      bad(d, "'layer_types'[" + std::to_string(l) + "] is " + (v.is_string() ? "'" + v.str() + "'" : "not a string") +
                 ", the real pattern (full attention at 4, 9, ..., 49) has '" + want + "'");
  }
}

}  // namespace

const char* kol_attn_form_name(KolAttnForm a) { return a == KolAttnForm::Bf16 ? "bf16" : "int4"; }

KolPlacement KolPlacement::one(const Kolibri1Desc& d) { return {1, d.layers, d.layers}; }
KolPlacement KolPlacement::two(const Kolibri1Desc& d, uint32_t split) { return {2, split, d.layers}; }

void validate(const KolPlacement& p, const Kolibri1Desc& d) {
  if (p.layers != d.layers)
    throw std::invalid_argument("KolPlacement: placed " + std::to_string(p.layers) + " layers, " + d.name +
                                " has " + std::to_string(d.layers));
  if (p.devices == 1) {
    if (p.split != d.layers)
      throw std::invalid_argument("KolPlacement: one device holds every layer (split " +
                                  std::to_string(p.split) + " != " + std::to_string(d.layers) + ")");
    return;
  }
  if (p.devices != 2)
    throw std::invalid_argument("KolPlacement: " + std::to_string(p.devices) +
                                " devices (spec 16 builds and gates two)");
  if (p.split < 1 || p.split + 1 > d.layers)
    throw std::invalid_argument("KolPlacement: split " + std::to_string(p.split) + " is outside [1, " +
                                std::to_string(d.layers > 0 ? d.layers - 1 : 0) +
                                "] - each device must hold at least one layer");
}

uint32_t Kolibri1Desc::full_before(uint32_t l) const { return l / full_every; }
uint32_t Kolibri1Desc::sliding_before(uint32_t l) const { return l - full_before(l); }

uint32_t Kolibri1Desc::router_n() const {
  uint32_t w = 16;
  while (w < experts) w *= 2;
  return w;
}

KolLinear Kolibri1Desc::linear(KolLinearId id) const {
  const bool i4 = attn == KolAttnForm::Int4;
  const WeightKind k = i4 ? WeightKind::Int4 : WeightKind::Bf16;
  switch (id) {
    case KolLinearId::Qkv:
      return {id, {hidden, qkv_n(), i4 ? qkv_s : 1u, 0}, k, Fuse::Concat,
              {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}};
    case KolLinearId::OProj:
      return {id, {q_n(), hidden, i4 ? oproj_s : 1u, 0}, k, Fuse::Single, {"self_attn.o_proj"}};
    case KolLinearId::LmHead:
      return {id, {hidden, vocab, 1, 0}, WeightKind::Bf16, Fuse::Single, {"lm_head"}};
    case KolLinearId::kCount:
      break;
  }
  throw std::out_of_range("Kolibri1Desc::linear: no row for ordinal " + std::to_string(static_cast<int>(id)));
}

std::vector<KolLinearId> Kolibri1Desc::layer_linears() const { return {KolLinearId::Qkv, KolLinearId::OProj}; }

std::string Kolibri1Desc::layer_prefix(uint32_t l) { return "model.layers." + std::to_string(l) + "."; }

const Kolibri1Desc& kolibri1() {
  static const Kolibri1Desc d = make_kolibri1();
  return d;
}

bool is_kolibri1_model_type(const std::string& t) { return t == "kolibri1"; }

void check_kolibri1_config(const Kolibri1Desc& d, const common::json::Value& c) {
  if (!c.is_object()) bad(d, "not a JSON object");
  want_str(d, c, "model_type", d.model_type.c_str(), false);
  const common::json::Value* archs = c.find("architectures");
  if (!archs || !archs->is_array() || archs->arr().empty() || !archs->arr()[0].is_string() ||
      archs->arr()[0].str() != d.architecture)
    bad(d, "architectures[0] is not '" + d.architecture + "'");
  want_u32(d, c, "num_hidden_layers", d.layers);
  check_layer_types(d, c, d.layers);
  want_u32(d, c, "hidden_size", d.hidden);
  want_u32(d, c, "num_attention_heads", d.q_heads);
  want_u32(d, c, "num_key_value_heads", d.kv_heads);
  want_u32(d, c, "head_dim", d.head_dim);
  want_u32(d, c, "num_experts", d.experts);
  want_u32(d, c, "num_experts_per_tok", d.top_k);
  want_u32(d, c, "moe_intermediate_size", d.moe_inter);
  want_u32(d, c, "shared_expert_intermediate_size", d.shared_inter);
  want_u32(d, c, "vocab_size", d.vocab);
  want_u32(d, c, "sliding_window", d.window);
  want_u32(d, c, "max_position_embeddings", d.trained_max_len);
  want_bool(d, c, "use_sliding_window", true, true);
  want_bool(d, c, "norm_topk_prob", false, false);
  want_bool(d, c, "tie_word_embeddings", false, false);
  want_bool(d, c, "attention_bias", false, false);
  want_str(d, c, "hidden_act", "silu", true);
  const common::json::Value* eps = c.find("rms_norm_eps");
  if (!eps || !eps->is_number() || float(eps->num()) != d.rms_eps) bad(d, "'rms_norm_eps' is not 1e-6");
  // rope_parameters (transformers 5) or the top-level rope_theta (the checkpoint's).
  double theta = 0;
  if (const common::json::Value* rp = c.find("rope_parameters"); rp && rp->is_object()) {
    if (const common::json::Value* t = rp->find("rope_type"); t && (!t->is_string() || t->str() != "default"))
      bad(d, "'rope_parameters.rope_type' is " + (t->is_string() ? "'" + t->str() + "'" : std::string("not a string")) +
                 ", only 'default' is implemented");
    if (const common::json::Value* t = rp->find("rope_theta"); t && t->is_number()) theta = t->num();
  }
  if (theta == 0)
    if (const common::json::Value* t = c.find("rope_theta"); t && t->is_number()) theta = t->num();
  if (theta != d.rope_theta) bad(d, "'rope_theta' is not " + std::to_string(d.rope_theta));
  if (const common::json::Value* hd = c.find("head_dtype"); hd && (!hd->is_string() || hd->str() != "float32"))
    bad(d, "'head_dtype' is not 'float32' (the fp32 head the engine computes)");
}

Kolibri1Desc kolibri1_desc(const common::json::Value& config, KolAttnForm attn) {
  Kolibri1Desc d = kolibri1();
  if (!config.is_object()) bad(d, "not a JSON object");
  const uint32_t n = need_u32(d, config, "num_hidden_layers");
  if (n < 1 || n > d.layers)
    bad(d, "'num_hidden_layers' is " + std::to_string(n) + ": a Kolibri-1 checkpoint has 1.." +
               std::to_string(d.layers) + " layers (the real 50, or a synthetic prefix)");
  d.layers = n;
  d.attn = attn;
  check_kolibri1_config(d, config);
  return d;
}

}  // namespace model
