#include "model/qwen4exp.h"

#include <cmath>
#include <stdexcept>

#include "common/json_reflect.h"

namespace model {
namespace {

Qwen4ExpDesc make_qwen4exp() {
  Qwen4ExpDesc d;
  d.name = "qwen3.8-flash-next";
  d.architecture = "Qwen4ExpForConditionalGeneration";
  d.model_type = "qwen4_exp";
  d.prefix = "model.language_model.";
  d.layers = 48;
  d.hidden = 2560;
  d.hc = 4;
  d.hc_low = 320;
  d.vocab = 248320;
  d.vocab_used = 248077;   // the original's tokenizer.json: 248044 + 33 added ids (21-QFNa's facts sheet)
  d.trained_max_len = 262144;
  d.gdn_k_heads = 16;
  d.gdn_v_heads = 48;
  d.gdn_head = 128;
  d.conv_taps = 4;
  d.q_heads = 24;
  d.kv_heads = 2;
  d.head_dim = 256;
  d.rope_dims = 64;        // partial_rotary_factor 0.25 x 256
  d.rope_theta = 1e7;
  d.idx_heads = 4;
  d.idx_dim = 128;
  d.idx_budget = 2048;
  d.idx_compress = 4;
  d.experts = 512;
  d.top_k = 10;
  d.moe_inter = 640;
  d.shared_inter = 640;
  d.ple_layer = 1;         // ple_layer_ids [2], one-indexed (M:1260)
  d.ngram = 3;
  d.ple_heads = 16;        // (ngram_size - 1) x heads_per_ngram 8
  d.ple_dim = 160;         // ple_embed_dim 2560 / 16
  d.ple_conv_taps = 4;
  d.ple_dilation = 3;
  d.ple_base = 20000000;
  d.ple_seed = 1234;       // `seed` absent from the config: the class default (C:155)
  d.ple_eos = 248044;
  d.ple_pad = 128;
  d.mtp_layers = 1;
  d.eos = {248046, 248044};
  d.rms_eps = 1e-6f;
  d.forms = {};
  d.gdn_out_s = 4;         // PROVISIONAL (header)
  d.qkvg_s = 2;
  d.o_s = 4;
  return d;
}

[[noreturn]] void bad(const Qwen4ExpDesc& d, const std::string& what) {
  throw std::runtime_error("config.json: " + what + " - " + d.name + " (model/qwen4exp.h) is not written for it");
}

const common::json::Value* num_of(const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  return v && v->is_number() ? v : nullptr;
}

uint32_t need_u32(const Qwen4ExpDesc& d, const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()) || v->num() > 4294967295.0)
    bad(d, std::string("'") + key + "' is missing or not a non-negative integer");
  return uint32_t(v->num());
}

// config.json's keys held to the descriptor by common::json::expect, in the order the checks run: a member without an
// initializer is required, the rest may be absent (their initializer is the class default the engine is written for).
struct WrapperKeys {   // the ConditionalGeneration wrapper around the text model
  std::string model_type;
  bool tie_word_embeddings = false;
};
struct TextType {
  std::string model_type;
};
struct TextKeys {
  uint32_t num_hidden_layers, full_attention_interval = 4, hidden_size, hc_count, hc_lowrank, vocab_size,
      max_position_embeddings, linear_num_key_heads, linear_num_value_heads, linear_key_head_dim,
      linear_value_head_dim, linear_conv_kernel_dim;
  std::string output_gate_type;
  uint32_t num_attention_heads, num_key_value_heads, head_dim, indexer_n_heads, indexer_head_dim, indexer_kv_heads,
      indexer_budget, indexer_compress_ratio, num_experts, num_experts_per_tok, moe_intermediate_size,
      shared_expert_intermediate_size;
  bool norm_topk_prob = true, tie_word_embeddings = false, attention_bias = false;   // norm_topk_prob: C:162
  std::string hidden_act = "silu";
  uint32_t ngram_size, heads_per_ngram, ple_embed_dim, ple_conv_kernel_size, make_ngram_vocab_size_divisible_by,
      ngram_vocab_size_base, mtp_num_hidden_layers = 1;
};

// The text config: `text_config` (the checkpoints) or the top level (a text-only export).
const common::json::Value& text_of(const Qwen4ExpDesc& d, const common::json::Value& c) {
  if (!c.is_object()) bad(d, "not a JSON object");
  const common::json::Value* t = c.find("text_config");
  if (t) {
    if (!t->is_object()) bad(d, "'text_config' is not an object");
    return *t;
  }
  return c;
}

bool is_qsa_type(const std::string& s) {
  return s == "full_attention" || s == "qwen_sparse_attention" || s == "indexed_attention";
}

// layer_types: `layers` entries, the real pattern's prefix (QSA exactly at l % 4 == 3).
void check_layer_types(const Qwen4ExpDesc& d, const common::json::Value& t, uint32_t layers) {
  const common::json::Value* lt = t.find("layer_types");
  if (!lt || !lt->is_array() || lt->arr().size() != layers)
    bad(d, "'layer_types' is not a list of num_hidden_layers (" + std::to_string(layers) + ") entries");
  for (uint32_t l = 0; l < layers; ++l) {
    const common::json::Value& v = lt->arr()[l];
    const bool ok = v.is_string() && (d.is_qsa(l) ? is_qsa_type(v.str()) : v.str() == "linear_attention");
    if (!ok)
      bad(d, "'layer_types'[" + std::to_string(l) + "] is " + (v.is_string() ? "'" + v.str() + "'" : "not a string") +
                 ", the real pattern (QSA at 3, 7, ..., 47) has '" +
                 (d.is_qsa(l) ? "full_attention" : "linear_attention") + "'");
  }
}

void check_ple_ids(const Qwen4ExpDesc& d, const common::json::Value& t) {
  const common::json::Value* v = t.find("ple_layer_ids");
  if (!v || !v->is_array() || v->arr().size() != 1 || !v->arr()[0].is_number() ||
      v->arr()[0].num() != double(d.ple_layer + 1))
    bad(d, "'ple_layer_ids' is not [" + std::to_string(d.ple_layer + 1) +
               "] (one PLE layer, one-indexed: layer_idx " + std::to_string(d.ple_layer) + ")");
}

double rope_theta_of(const common::json::Value& t, double* partial) {
  double theta = 0;
  *partial = -1;
  if (const common::json::Value* rp = t.find("rope_parameters"); rp && rp->is_object()) {
    if (const common::json::Value* v = num_of(*rp, "rope_theta")) theta = v->num();
    if (const common::json::Value* v = num_of(*rp, "partial_rotary_factor")) *partial = v->num();
  }
  if (theta == 0)
    if (const common::json::Value* v = num_of(t, "rope_theta")) theta = v->num();
  if (const common::json::Value* v = num_of(t, "partial_rotary_factor")) *partial = v->num();
  return theta;
}

uint32_t eos_of(const Qwen4ExpDesc& d, const common::json::Value& t) {
  const common::json::Value* v = t.find("eos_token_id");
  if (v && v->is_array() && !v->arr().empty() && v->arr()[0].is_number()) return uint32_t(v->arr()[0].num());
  if (v && v->is_number()) return uint32_t(v->num());
  bad(d, "'eos_token_id' is missing (the PLE's EOS, M:1086)");
}

}  // namespace

const char* q4_form_name(Q4Form f) { return f == Q4Form::Bf16 ? "bf16" : "int4"; }

Q4Placement Q4Placement::one(const Qwen4ExpDesc& d) { return {1, d.layers, d.layers}; }
Q4Placement Q4Placement::two(const Qwen4ExpDesc& d, uint32_t split) { return {2, split, d.layers}; }
Q4Placement Q4Placement::tp2(const Qwen4ExpDesc& d, Q4TpEnds ends) { return {2, d.layers, d.layers, true, ends}; }

const char* q4_tp_ends_name(Q4TpEnds e) { return e == Q4TpEnds::Both ? "both" : "split"; }

Qwen4ExpDesc q4_tp_card(const Qwen4ExpDesc& one, uint32_t rank) {
  if (rank >= kQ4TpCards) throw std::invalid_argument("model::q4_tp_card: rank " + std::to_string(rank));
  const auto half = [&](uint32_t v, const char* what) {
    if (v % kQ4TpCards != 0)
      throw std::invalid_argument("model::q4_tp_card: " + one.name + "'s " + what + " " + std::to_string(v) +
                                  " does not halve");
    return v / kQ4TpCards;
  };
  Qwen4ExpDesc c = one;
  c.q_heads = half(one.q_heads, "q heads");
  c.kv_heads = half(one.kv_heads, "kv heads");
  c.gdn_k_heads = half(one.gdn_k_heads, "GDN k heads");
  c.gdn_v_heads = half(one.gdn_v_heads, "GDN v heads");
  c.moe_inter = half(one.moe_inter, "expert width");
  c.shared_inter = half(one.shared_inter, "shared expert width");
  if (c.moe_inter % 64 != 0 || c.shared_inter % 64 != 0)
    throw std::invalid_argument("model::q4_tp_card: a half expert is not whole groups of 64");
  // out_proj / o_proj at half the split-K over half the K: each slice is one card's slice (2r, 2r + 1)
  c.gdn_out_s = half(one.gdn_out_s, "out_proj split-K");
  c.o_s = half(one.o_s, "o_proj split-K");
  c.vocab = half(one.vocab, "vocabulary");
  const uint32_t lo = rank * c.vocab;
  c.vocab_used = one.vocab_used > lo ? std::min(c.vocab, one.vocab_used - lo) : 0;
  return c;
}

void validate(const Q4Placement& p, const Qwen4ExpDesc& d) {
  if (p.layers != d.layers)
    throw std::invalid_argument("Q4Placement: placed " + std::to_string(p.layers) + " layers, " + d.name + " has " +
                                std::to_string(d.layers));
  if (p.tp) {
    if (p.devices != kQ4TpCards || p.split != d.layers)
      throw std::invalid_argument("Q4Placement: tensor parallel is two devices, each holding every layer");
    return;
  }
  if (p.devices == 1) {
    if (p.split != d.layers)
      throw std::invalid_argument("Q4Placement: one device holds every layer (split " + std::to_string(p.split) +
                                  " != " + std::to_string(d.layers) + ")");
    return;
  }
  if (p.devices != 2)
    throw std::invalid_argument("Q4Placement: " + std::to_string(p.devices) + " devices (two are supported)");
  if (p.split < 1 || p.split + 1 > d.layers)
    throw std::invalid_argument("Q4Placement: split " + std::to_string(p.split) + " is outside [1, " +
                                std::to_string(d.layers > 0 ? d.layers - 1 : 0) +
                                "] - each device must hold at least one layer");
}

uint32_t Qwen4ExpDesc::qsa_before(uint32_t l) const { return l / 4; }
uint32_t Qwen4ExpDesc::gdn_before(uint32_t l) const { return l - qsa_before(l); }
uint32_t Qwen4ExpDesc::conv_rows() const { return (2 * gdn_k_heads + gdn_v_heads) * gdn_head; }
uint32_t Qwen4ExpDesc::qkvz_n() const { return conv_rows() + gdn_z_n(); }
uint32_t Qwen4ExpDesc::router_n() const { return (experts + 1 + 15) / 16 * 16; }

Q4Linear Qwen4ExpDesc::linear(Q4LinearId id) const {
  const bool di4 = forms.dense == Q4Form::Int4;
  const WeightKind dk = di4 ? WeightKind::Int4 : WeightKind::Bf16;
  switch (id) {
    case Q4LinearId::GdnQkvz:
      return {id, {hidden, qkvz_n(), 1, 0}, dk, Fuse::Concat, {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"}};
    case Q4LinearId::GdnAb:
      return {id, {hidden, kAbPaddedN, 1, 0}, WeightKind::Bf16, Fuse::Concat,
              {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}};
    case Q4LinearId::GdnOut:
      return {id, {gdn_z_n(), hidden, di4 ? gdn_out_s : 1u, 0}, dk, Fuse::Single, {"linear_attn.out_proj"}};
    case Q4LinearId::QsaQkvg:
      return {id, {hidden, qkvg_n(), di4 ? qkvg_s : 1u, 0}, dk, Fuse::Concat,
              {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}};
    case Q4LinearId::QsaIdx:
      return {id, {hidden, idx_n(), 1, 0}, WeightKind::Bf16, Fuse::Single, {"self_attn.indexer.index_qk_proj"}};
    case Q4LinearId::QsaO:
      return {id, {q_n(), hidden, di4 ? o_s : 1u, 0}, dk, Fuse::Single, {"self_attn.o_proj"}};
    case Q4LinearId::Router:
      return {id, {hidden, router_n(), 1, 0}, WeightKind::Bf16, Fuse::Concat, {"mlp.gate", "mlp.shared_expert_gate"}};
    case Q4LinearId::PleKv:
      return {id, {ple_e(), ple_kv_n(), 1, 0}, WeightKind::Bf16, Fuse::Concat, {"ple.key_proj", "ple.value_proj"}};
    case Q4LinearId::LmHead:
      return {id, {hidden, vocab, 1, 0}, WeightKind::Bf16, Fuse::Single, {"lm_head"}};
    case Q4LinearId::kCount:
      break;
  }
  throw std::out_of_range("Qwen4ExpDesc::linear: no row for ordinal " + std::to_string(static_cast<int>(id)));
}

std::string Qwen4ExpDesc::layer_prefix(uint32_t l) { return "model.language_model.layers." + std::to_string(l) + "."; }

const Qwen4ExpDesc& qwen4exp() {
  static const Qwen4ExpDesc d = make_qwen4exp();
  return d;
}

bool is_qwen4exp_model_type(const std::string& t) { return t == "qwen4_exp" || t == "qwen4_exp_text"; }

void check_qwen4exp_config(const Qwen4ExpDesc& d, const common::json::Value& c) {
  const common::json::Value& t = text_of(d, c);
  const auto fail = [&](const std::string& what) { bad(d, what); };
  if (&t != &c) {   // the checkpoints: a ConditionalGeneration wrapper around the text model
    common::json::expect(c, WrapperKeys{.model_type = d.model_type}, fail);
    const common::json::Value* archs = c.find("architectures");
    if (!archs || !archs->is_array() || archs->arr().empty() || !archs->arr()[0].is_string() ||
        archs->arr()[0].str() != d.architecture)
      bad(d, "architectures[0] is not '" + d.architecture + "'");
  }
  if (&t == &c || t.find("model_type")) common::json::expect(t, TextType{"qwen4_exp_text"}, fail);
  common::json::expect(
      t,
      TextKeys{.num_hidden_layers = d.layers, .hidden_size = d.hidden, .hc_count = d.hc, .hc_lowrank = d.hc_low,
               .vocab_size = d.vocab, .max_position_embeddings = d.trained_max_len,
               .linear_num_key_heads = d.gdn_k_heads, .linear_num_value_heads = d.gdn_v_heads,
               .linear_key_head_dim = d.gdn_head, .linear_value_head_dim = d.gdn_head,
               .linear_conv_kernel_dim = d.conv_taps, .output_gate_type = "sigmoid", .num_attention_heads = d.q_heads,
               .num_key_value_heads = d.kv_heads, .head_dim = d.head_dim, .indexer_n_heads = d.idx_heads,
               .indexer_head_dim = d.idx_dim, .indexer_kv_heads = 1, .indexer_budget = d.idx_budget,
               .indexer_compress_ratio = d.idx_compress, .num_experts = d.experts, .num_experts_per_tok = d.top_k,
               .moe_intermediate_size = d.moe_inter, .shared_expert_intermediate_size = d.shared_inter,
               .ngram_size = d.ngram, .heads_per_ngram = d.ple_heads / (d.ngram - 1), .ple_embed_dim = d.ple_e(),
               .ple_conv_kernel_size = d.ple_conv_taps, .make_ngram_vocab_size_divisible_by = d.ple_pad,
               .ngram_vocab_size_base = uint32_t(d.ple_base), .mtp_num_hidden_layers = d.mtp_layers},
      fail);
  check_layer_types(d, t, d.layers);
  check_ple_ids(d, t);
  if (const common::json::Value* s = t.find("seed"); s && (!s->is_number() || uint64_t(s->num()) != d.ple_seed))
    bad(d, "'seed' is not " + std::to_string(d.ple_seed) + " (the PLE hash's)");
  else if (!s && d.ple_seed != 1234)
    bad(d, "'seed' is absent (the class default 1234), the descriptor says " + std::to_string(d.ple_seed));
  if (eos_of(d, t) != d.ple_eos) bad(d, "'eos_token_id' is not " + std::to_string(d.ple_eos) + " (the PLE's EOS)");
  const common::json::Value* eps = t.find("rms_norm_eps");
  if (!eps || !eps->is_number() || float(eps->num()) != d.rms_eps) bad(d, "'rms_norm_eps' is not 1e-6");
  if (const common::json::Value* rp = t.find("rope_parameters"); rp && rp->is_object())
    if (const common::json::Value* ty = rp->find("rope_type"); ty && (!ty->is_string() || ty->str() != "default"))
      bad(d, "'rope_parameters.rope_type' is " + (ty->is_string() ? "'" + ty->str() + "'" : std::string("not a string")) +
                 ", only 'default' is implemented");
  double partial = -1;
  const double theta = rope_theta_of(t, &partial);
  if (theta != d.rope_theta) bad(d, "'rope_theta' is " + std::to_string(theta) + ", not " + std::to_string(d.rope_theta));
  if (partial != double(d.rope_dims) / double(d.head_dim))
    bad(d, "'partial_rotary_factor' is " + std::to_string(partial) + ", the descriptor's RoPE covers " +
               std::to_string(d.rope_dims) + " of " + std::to_string(d.head_dim) + " dims (0.25)");
}

Qwen4ExpDesc qwen4exp_desc(const common::json::Value& config, const Q4Forms& forms) {
  Qwen4ExpDesc d = qwen4exp();
  const common::json::Value& t = text_of(d, config);
  const uint32_t n = need_u32(d, t, "num_hidden_layers");
  if (n < d.ple_layer + 1 || n > d.layers)
    bad(d, "'num_hidden_layers' is " + std::to_string(n) + ": a Qwen3.8-Flash-Next checkpoint has " +
               std::to_string(d.ple_layer + 1) + ".." + std::to_string(d.layers) +
               " layers (the real 48, or a synthetic prefix that keeps the PLE layer " + std::to_string(d.ple_layer) + ")");
  d.layers = n;
  d.ple_base = need_u32(d, t, "ngram_vocab_size_base");
  if (const common::json::Value* s = t.find("seed"); s && s->is_number()) d.ple_seed = uint64_t(s->num());
  if (forms.expert_group != 64 && forms.expert_group != 128)
    throw std::invalid_argument("qwen4exp_desc: expert group " + std::to_string(forms.expert_group) + " (64 or 128)");
  d.forms = forms;
  check_qwen4exp_config(d, config);
  return d;
}

}  // namespace model
