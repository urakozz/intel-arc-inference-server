#include "model/muse_glimmer.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace model {
namespace {

[[noreturn]] void bad(const std::string& what) {
  throw std::runtime_error("config.json: " + what +
                           " - Muse-Glimmer-30B (model/muse_glimmer.h) is not written for it");
}

// A JSON value as the message names it: integers without a fraction, other numbers in %.9g, strings quoted.
std::string shown(const common::json::Value* v) {
  if (!v) return "absent";
  if (v->is_null()) return "null";
  if (v->is_bool()) return v->boolean() ? "true" : "false";
  if (v->is_string()) return "'" + v->str() + "'";
  if (v->is_number()) {
    const double x = v->num();
    char buf[64];
    if (x == std::floor(x) && std::fabs(x) < 1e15)
      std::snprintf(buf, sizeof buf, "%.0f", x);
    else
      std::snprintf(buf, sizeof buf, "%.9g", x);
    return buf;
  }
  return v->is_array() ? "a list" : "an object";
}

uint32_t need_u32(const common::json::Value& c, const char* key) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()) || v->num() > 4294967295.0)
    bad(std::string("'") + key + "' is " + shown(v) + ", not a non-negative integer");
  return uint32_t(v->num());
}

void want_u32(const common::json::Value& c, const char* key, uint32_t want) {
  const uint32_t got = need_u32(c, key);
  if (got != want) bad(std::string("'") + key + "' is " + std::to_string(got) + ", the descriptor says " + std::to_string(want));
}

// A float key held to the descriptor's float (both read as fp32: the kernels' constant).
void want_f32(const common::json::Value& c, const char* key, float want) {
  const common::json::Value* v = c.find(key);
  if (!v || !v->is_number() || float(v->num()) != want) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.7g", double(want));
    bad(std::string("'") + key + "' is " + shown(v) + ", the descriptor says " + buf);
  }
}

void want_bool(const common::json::Value& c, const char* key, bool want, bool absent_as) {
  const common::json::Value* v = c.find(key);
  if (v && !v->is_bool()) bad(std::string("'") + key + "' is " + shown(v) + ", not a boolean");
  const bool got = v ? v->boolean() : absent_as;
  if (got != want)
    bad(std::string("'") + key + "' is " + (got ? "true" : "false") + ", the descriptor needs " + (want ? "true" : "false"));
}

void want_str(const common::json::Value& c, const char* key, const char* want, bool absent_ok) {
  const common::json::Value* v = c.find(key);
  if (!v && absent_ok) return;
  if (!v || !v->is_string() || v->str() != want)
    bad(std::string("'") + key + "' is " + shown(v) + ", the descriptor needs '" + want + "'");
}

// The real pattern: full attention exactly at l % 4 == 3.
bool real_full(uint32_t l) { return l % 4 == 3; }

}  // namespace

MusePlacement MusePlacement::one(const MuseGlimmerDesc& d) { return {1, d.layers, d.layers}; }
MusePlacement MusePlacement::two(const MuseGlimmerDesc& d, uint32_t split) { return {2, split, d.layers}; }

void validate(const MusePlacement& p, const MuseGlimmerDesc& d) {
  if (p.layers != d.layers)
    throw std::invalid_argument("MusePlacement: placed " + std::to_string(p.layers) + " layers, the descriptor has " +
                                std::to_string(d.layers));
  if (p.devices == 1) {
    if (p.split != d.layers)
      throw std::invalid_argument("MusePlacement: one device holds every layer (split " + std::to_string(p.split) +
                                  " != " + std::to_string(d.layers) + ")");
    return;
  }
  if (p.devices != 2)
    throw std::invalid_argument("MusePlacement: " + std::to_string(p.devices) + " devices (spec 16 builds and gates two)");
  if (p.split < 1 || p.split + 1 > d.layers)
    throw std::invalid_argument("MusePlacement: split " + std::to_string(p.split) + " is outside [1, " +
                                std::to_string(d.layers > 0 ? d.layers - 1 : 0) +
                                "] - each device must hold at least one layer");
}

uint32_t MuseGlimmerDesc::full_layers() const { return full_before(layers); }
uint32_t MuseGlimmerDesc::sliding_layers() const { return layers - full_layers(); }
uint32_t MuseGlimmerDesc::full_before(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l && i < full.size(); ++i) n += full[i] ? 1 : 0;
  return n;
}
uint32_t MuseGlimmerDesc::sliding_before(uint32_t l) const { return l - full_before(l); }

MuseLinear MuseGlimmerDesc::linear(MuseLinearId id) const {
  switch (id) {
    case MuseLinearId::QGKV:
      return {id, {hidden, qgkv_rows(), qgkv_s, 0}, WeightKind::Int4, Fuse::Concat,
              {"self_attn.q_proj", "self_attn.gate_proj", "self_attn.k_proj", "self_attn.v_proj"},
              {q_n(), q_n(), kv_n(), kv_n()}};
    case MuseLinearId::OProj:
      return {id, {q_n(), hidden, oproj_s, 0}, WeightKind::Int4, Fuse::Single, {"self_attn.o_proj"}, {hidden}};
    case MuseLinearId::GateUp:
      return {id, {hidden, 2 * inter, gate_up_s, 0}, WeightKind::Int4, Fuse::Concat,
              {"mlp.gate_proj", "mlp.up_proj"}, {inter, inter}};
    case MuseLinearId::Down:
      return {id, {inter, hidden, down_s, 0}, WeightKind::Int4, Fuse::Single, {"mlp.down_proj"}, {hidden}};
    case MuseLinearId::LmHead:
      return {id, {hidden, vocab, 1, 0}, WeightKind::Bf16, Fuse::Single, {"lm_head"}, {vocab}};
    case MuseLinearId::kCount:
      break;
  }
  throw std::out_of_range("MuseGlimmerDesc::linear: no row for ordinal " + std::to_string(static_cast<int>(id)));
}

std::vector<MuseLinearId> MuseGlimmerDesc::layer_linears() const {
  return {MuseLinearId::QGKV, MuseLinearId::OProj, MuseLinearId::GateUp, MuseLinearId::Down};
}

std::string MuseGlimmerDesc::layer_prefix(uint32_t l) {
  return std::string(text_prefix()) + "layers." + std::to_string(l) + ".";
}

MuseGlimmerDesc muse_glimmer() {
  MuseGlimmerDesc d;
  d.full.resize(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) d.full[l] = real_full(l);
  return d;
}

bool is_muse_model_type(const std::string& t) { return t == "muse_glimmer" || t == "muse_glimmer_text"; }

MuseGlimmerDesc check_muse_config(const common::json::Value& cfg, uint32_t layers_override) {
  MuseGlimmerDesc d = muse_glimmer();
  if (!cfg.is_object()) bad("not a JSON object");
  // The whole config.json (its text_config) or a text config alone.
  const common::json::Value* tcp = cfg.find("text_config");
  if (tcp) {
    want_str(cfg, "model_type", "muse_glimmer", false);
    if (const common::json::Value* a = cfg.find("architectures"); a) {
      if (!a->is_array() || a->arr().empty() || !a->arr()[0].is_string() ||
          a->arr()[0].str() != "MuseGlimmerForConditionalGeneration")
        bad("'architectures'[0] is not 'MuseGlimmerForConditionalGeneration'");
    }
    want_bool(cfg, "tie_word_embeddings", false, false);
    if (!tcp->is_object()) bad("'text_config' is not an object");
  }
  const common::json::Value& t = tcp ? *tcp : cfg;
  want_str(t, "model_type", "muse_glimmer_text", false);

  const uint32_t n = need_u32(t, "num_hidden_layers");
  if (n < 1 || n > d.layers)
    bad("'num_hidden_layers' is " + std::to_string(n) + ": a Muse-Glimmer checkpoint has 1.." + std::to_string(d.layers) +
        " layers (the real 52, or a synthetic prefix)");
  want_u32(t, "hidden_size", d.hidden);
  want_u32(t, "intermediate_size", d.inter);
  want_u32(t, "num_attention_heads", d.q_heads);
  // GQA 16 (2 kv heads) is a compile-time define of the attention kernels: refused until another is built.
  want_u32(t, "num_key_value_heads", d.kv_heads);
  want_u32(t, "head_dim", d.head_dim);
  want_u32(t, "vocab_size", d.vocab);
  want_u32(t, "sliding_window", d.window);
  want_u32(t, "max_position_embeddings", d.max_pos);
  want_u32(t, "bos_token_id", d.bos);
  if (const common::json::Value* e = t.find("eos_token_id"); e) {
    bool ok = e->is_number() && uint32_t(e->num()) == d.eos[0];
    if (e->is_array()) {
      ok = true;
      for (const common::json::Value& x : e->arr())
        if (!x.is_number() || (uint32_t(x.num()) != d.eos[0] && uint32_t(x.num()) != d.eos[1])) ok = false;
    }
    if (!ok) bad("'eos_token_id' is " + shown(e) + ", the descriptor's EOS are 200001 / 200008");
  }
  want_str(t, "hidden_activation", "silu", false);
  want_bool(t, "tie_word_embeddings", false, false);
  want_bool(t, "attention_bias", false, false);
  want_f32(t, "rms_norm_eps", d.rms_eps);
  want_f32(t, "post_norm_eps", d.post_eps);
  want_f32(t, "qk_scale_factor", d.qk_scale);
  want_f32(t, "output_multiplier", d.out_mult);
  // The host's softcap needs the value (spec 23 §6.4): null is refused, not read as "no cap".
  want_f32(t, "final_logit_softcapping", d.softcap);

  // rope_parameters: default rope at theta 5e5 - the ONE table (M:186-191).
  const common::json::Value* rp = t.find("rope_parameters");
  if (!rp || !rp->is_object()) bad("'rope_parameters' is " + shown(rp) + ", not an object");
  want_str(*rp, "rope_type", "default", false);
  const common::json::Value* th = rp->find("rope_theta");
  if (!th || !th->is_number() || float(th->num()) != d.rope_theta)
    bad("'rope_parameters.rope_theta' is " + shown(th) + ", the descriptor says 500000");
  const float theta = float(th->num());

  // layer_types: n entries, the real pattern's prefix (full exactly at l % 4 == 3).
  const common::json::Value* lt = t.find("layer_types");
  if (!lt || !lt->is_array() || lt->arr().size() != n)
    bad("'layer_types' is not a list of num_hidden_layers (" + std::to_string(n) + ") entries");
  for (uint32_t l = 0; l < n; ++l) {
    const common::json::Value& v = lt->arr()[l];
    const char* want = real_full(l) ? "full_attention" : "sliding_attention";
    if (!v.is_string() || v.str() != want)
      bad("'layer_types[" + std::to_string(l) + "]' is " + shown(&v) +
          ", the real pattern (full attention at 3, 7, ..., 51) has '" + want + "'");
  }
  // layer_rope_theta is only a switch in transformers 5.19.0 (M:512; the table is the global theta's): every
  // non-zero entry must BE the global theta, and the zeros (NoPE) must sit exactly at the full layers.
  const common::json::Value* lr = t.find("layer_rope_theta");
  if (!lr || !lr->is_array() || lr->arr().size() != n)
    bad("'layer_rope_theta' is not a list of num_hidden_layers (" + std::to_string(n) + ") entries");
  for (uint32_t l = 0; l < n; ++l) {
    const common::json::Value& v = lr->arr()[l];
    if (!v.is_number()) bad("'layer_rope_theta[" + std::to_string(l) + "]' is " + shown(&v) + ", not a number");
    const bool zero = v.num() == 0;
    if (real_full(l) && !zero)
      bad("'layer_rope_theta[" + std::to_string(l) + "]' is " + shown(&v) +
          " at a full_attention layer - the full layers are NoPE (0)");
    if (!real_full(l) && zero)
      bad("'layer_rope_theta[" + std::to_string(l) + "]' is 0 at a sliding_attention layer - NoPE where the model "
          "applies RoPE");
    if (!zero && float(v.num()) != theta)
      bad("'layer_rope_theta[" + std::to_string(l) + "]' is " + shown(&v) +
          ", not rope_parameters.rope_theta 500000 - transformers 5.19.0 builds ONE rotary table from the global "
          "theta (M:186-191), so a per-layer theta would be silently ignored");
  }

  d.layers = n;
  if (layers_override != 0) {
    if (layers_override > n)
      bad("--layers " + std::to_string(layers_override) + ": the checkpoint has " + std::to_string(n) + " layers");
    d.layers = layers_override;
  }
  d.full.resize(d.layers);
  return d;
}

}  // namespace model
