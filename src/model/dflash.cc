#include "model/dflash.h"

#include <stdexcept>
#include <string>

// dflash.cc - the drafter's config.json read as tools/oracle/dflash_ref.py's `config_from_dict` reads it (the port's
// line references are to vLLM's qwen3_dflash.py / qwen3_dflash2.py at 96b0efe6b0), with the refusals model/dflash.h
// lists. Every throw names the key.
namespace model {
namespace {

using common::json::Value;

[[noreturn]] void refuse(const std::string& what) { throw std::runtime_error("DFlash drafter config: " + what); }

const Value* opt(const Value& o, const char* key) {
  if (!o.is_object()) return nullptr;
  const Value* v = o.find(key);
  return v && !v->is_null() ? v : nullptr;
}
uint32_t u32(const Value& o, const char* key) {
  const Value* v = opt(o, key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != double(uint32_t(v->num())))
    refuse(std::string(key) + " missing or not a non-negative integer");
  return uint32_t(v->num());
}
double num_or(const Value& o, const char* key, double dflt) {
  const Value* v = opt(o, key);
  if (!v) return dflt;
  if (!v->is_number()) refuse(std::string(key) + " is not a number");
  return v->num();
}
bool bool_or(const Value& o, const char* key, bool dflt) {
  const Value* v = opt(o, key);
  if (!v) return dflt;
  if (!v->is_bool()) refuse(std::string(key) + " is not a boolean");
  return v->boolean();
}

constexpr const char* kSliding = "sliding_attention";

}  // namespace

bool is_dflash_architecture(const std::string& a) { return a == "DFlashDraftModel" || a == "DFlash2DraftModel"; }

bool DFlashDesc::all_sliding() const {
  for (uint32_t w : window)
    if (w == 0) return false;
  return !window.empty();
}
uint32_t DFlashDesc::max_window() const {
  uint32_t m = 0;
  for (uint32_t w : window) m = w > m ? w : m;
  return m;
}

DFlashDesc dflash_desc_from_config(const Value& c) {
  if (!c.is_object()) refuse("config.json is not an object");
  DFlashDesc d;
  const Value* arch = opt(c, "architectures");
  if (!arch || !arch->is_array() || arch->arr().empty() || !arch->arr()[0].is_string() ||
      !is_dflash_architecture(arch->arr()[0].str()))
    refuse("architectures[0] must be DFlashDraftModel or DFlash2DraftModel");
  d.architecture = arch->arr()[0].str();
  if (opt(c, "quantization_config"))
    refuse("quantization_config: a quantised drafter is not one of our formats - load the bf16 drafter "
           "(it is quantised at load: --draft-format int8)");
  // dc = eagle_config, overlaid by dflash_config (config_from_dict).
  Value::Object dco;
  for (const char* k : {"eagle_config", "dflash_config"})
    if (const Value* v = opt(c, k)) {
      if (!v->is_object()) refuse(std::string(k) + " is not an object");
      for (const auto& [kk, vv] : v->obj()) dco[kk] = vv;
    }
  const Value dc = Value::make_obj(dco);
  d.version = (d.architecture.find("DFlash2") != std::string::npos || dc.find("selector_rank") ||
               dc.find("conv_kernel_size"))
                  ? 2
                  : 1;
  if (const Value* a = opt(c, "hidden_act"); a && (!a->is_string() || a->str() != "silu"))
    refuse("hidden_act must be silu");
  if (const Value* rp = opt(c, "rope_parameters")) {
    if (const Value* t = opt(*rp, "rope_type"); t && (!t->is_string() || t->str() != "default"))
      refuse("rope_type must be default");
    if (num_or(*rp, "partial_rotary_factor", 1.0) != 1.0) refuse("partial_rotary_factor must be 1.0");
    d.theta = num_or(*rp, "rope_theta", num_or(c, "rope_theta", 1e6));
  } else {
    d.theta = num_or(c, "rope_theta", 1e6);
  }
  if (bool_or(dc, "sample_from_anchor", false)) refuse("sample_from_anchor=true is not DFlash");
  if (!bool_or(dc, "use_aux_hidden_state", bool_or(c, "use_aux_hidden_state", true)))
    refuse("use_aux_hidden_state=false (no fc) is not supported");
  if (bool_or(c, "attention_bias", false)) refuse("attention_bias=true is not supported");
  if (opt(dc, "attention_value_scale")) refuse("attention_value_scale is not supported");
  if (bool_or(dc, "attention_sink_bias", bool_or(c, "add_swa_attention_sink_bias", false)))
    refuse("attention_sink_bias is not supported");

  d.layers = u32(c, "num_hidden_layers");
  d.hidden = u32(c, "hidden_size");
  d.q_heads = u32(c, "num_attention_heads");
  d.kv_heads = u32(c, "num_key_value_heads");
  d.head_dim = opt(c, "head_dim") ? u32(c, "head_dim") : (d.q_heads ? d.hidden / d.q_heads : 0);
  d.inter = u32(c, "intermediate_size");
  d.vocab = u32(c, "vocab_size");
  if (const Value* dv = opt(c, "draft_vocab_size"); dv && u32(c, "draft_vocab_size") != d.vocab)
    refuse("draft_vocab_size != vocab_size: a drafter vocabulary of its own (d2t) is not supported");
  if (d.layers == 0 || d.hidden == 0 || d.q_heads == 0 || d.kv_heads == 0 || d.head_dim == 0 || d.inter == 0 ||
      d.vocab == 0)
    refuse("a zero width (num_hidden_layers / hidden_size / heads / head_dim / intermediate_size / vocab_size)");
  if (d.q_heads % d.kv_heads) refuse("num_attention_heads must be a multiple of num_key_value_heads");
  if (d.head_dim % 2) refuse("head_dim must be even (RoPE pairs)");
  d.eps = float(num_or(c, "rms_norm_eps", 1e-6));
  d.neox = bool_or(c, "is_neox_style", true);
  if (opt(c, "num_target_layers")) d.num_target_layers = u32(c, "num_target_layers");

  d.block = u32(dc, "block_size");
  if (d.block < 2) refuse("block_size must be at least 2 (an anchor and a mask row)");
  if (opt(dc, "mask_token_id"))
    d.mask_id = u32(dc, "mask_token_id");
  else
    d.mask_id = u32(c, "mask_token_id");
  const Value* ids = opt(dc, "target_layer_ids");
  if (!ids || !ids->is_array() || ids->arr().empty()) refuse("dflash_config.target_layer_ids missing");
  for (const Value& v : ids->arr()) {
    if (!v.is_number() || v.num() < 0) refuse("target_layer_ids holds a non-integer");
    const uint32_t i = uint32_t(v.num());
    if (!d.taps.empty() && i <= d.taps.back())
      refuse("target_layer_ids must be strictly increasing (the aux capture order)");
    d.taps.push_back(i);
  }

  // resolve_layer_attention (qwen3_dflash.py:119) and _dflash_layer_causal (:85), per layer.
  const Value* lt = opt(c, "layer_types");
  if (lt && (!lt->is_array() || lt->arr().size() != d.layers)) refuse("layer_types must list every layer");
  const bool use_swa = bool_or(dc, "use_swa", false);
  bool any_sliding = false;
  if (lt)
    for (const Value& t : lt->arr()) any_sliding |= t.is_string() && t.str() == kSliding;
  for (uint32_t i = 0; i < d.layers; ++i) {
    const bool is_sl = lt && lt->arr()[i].is_string() && lt->arr()[i].str() == kSliding;
    const bool sliding = (!lt || (use_swa && !any_sliding)) ? use_swa : is_sl;
    uint32_t w = 0;
    if (sliding) {
      if (opt(dc, "swa_window_size"))
        w = u32(dc, "swa_window_size");
      else if (opt(c, "sliding_window"))
        w = u32(c, "sliding_window");
      else
        refuse("a sliding layer needs dflash_config.swa_window_size or sliding_window");
      if (w == 0) refuse("sliding_window must be positive");
    }
    d.window.push_back(w);
    bool causal;
    if (const Value* ic = opt(c, "is_causal")) {
      if (!ic->is_bool()) refuse("is_causal is not a boolean");
      causal = ic->boolean();
    } else if (const Value* dcc = opt(dc, "causal")) {
      if (!dcc->is_bool()) refuse("dflash_config.causal is not a boolean");
      causal = dcc->boolean();
    } else {
      causal = lt ? is_sl : false;
    }
    d.causal.push_back(causal);
  }

  if (d.v2()) {
    d.conv_group = u32(dc, "conv_group_size");
    d.conv_taps = u32(dc, "conv_kernel_size");
    d.sel_rank = u32(dc, "selector_rank");
    d.sel_top_k = u32(dc, "selector_top_k");
    if (d.conv_group == 0 || d.hidden % d.conv_group) refuse("conv_group_size must divide hidden_size");
    if (d.conv_taps == 0) refuse("conv_kernel_size must be positive");
    if (d.sel_rank == 0 || d.sel_top_k == 0) refuse("selector_rank / selector_top_k must be positive");
    d.in_scale = float(num_or(dc, "input_embedding_scale", 1.0));
    d.out_mult = float(num_or(dc, "output_multiplier", 1.0));
    d.softcap = float(num_or(dc, "final_logit_softcapping", 0.0));
  } else {
    d.out_mult = float(num_or(c, "logit_scale", 1.0));
  }
  if (!(d.out_mult > 0.0f)) refuse("output_multiplier / logit_scale must be positive (the top-k order)");
  if (d.softcap < 0.0f) refuse("final_logit_softcapping must not be negative");
  return d;
}

void check_dflash_target(const DFlashDesc& d, const ModelDesc& t, uint32_t target_vocab_rows) {
  auto bad = [&](const std::string& what) {
    throw std::runtime_error("DFlash drafter does not fit the target " + t.name + ": " + what);
  };
  if (d.hidden != t.hidden)
    bad("hidden_size " + std::to_string(d.hidden) + " != the target's " + std::to_string(t.hidden) +
        " (the drafter shares its embedding and lm_head)");
  if (d.vocab != target_vocab_rows)
    bad("vocab_size " + std::to_string(d.vocab) + " != the target's lm_head rows " +
        std::to_string(target_vocab_rows));
  for (uint32_t i : d.taps)
    if (i >= t.layers)
      bad("target_layer_ids holds " + std::to_string(i) + ", the target has " + std::to_string(t.layers) +
          " layers");
  if (d.num_target_layers && d.num_target_layers != t.layers)
    bad("num_target_layers " + std::to_string(d.num_target_layers) + " != the target's " +
        std::to_string(t.layers));
  if (t.vocab_used && d.mask_id >= t.vocab_used)
    bad("mask_token_id " + std::to_string(d.mask_id) + " is not a token the target defines (< " +
        std::to_string(t.vocab_used) + ")");
}

}  // namespace model
