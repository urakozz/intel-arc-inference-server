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

bool is_dflash_architecture(const std::string& a) {
  return a == "DFlashDraftModel" || a == "DFlash2DraftModel" || a == kLiLiCorrArchitecture || a == kDSparkArchitecture;
}

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

namespace {
// The vLLM forms' body (config_from_dict): DFlash / DFlash 2 - or, `lili` (spec 26b), the LiLiCorr form's backbone, which
// lilicorr_desc_from_config below has already held to its own refusals (dflash_ref.py lilicorr_config_from_dict reads it
// through the same rules: block_size from the top level when dflash_config has none, no selector, sinks allowed).
DFlashDesc vllm_desc_from_config(const Value& c, const std::string& architecture, bool lili);
DFlashDesc lilicorr_desc_from_config(const Value& c);
DFlashDesc dspark_desc_from_config(const Value& c);
}  // namespace

DFlashDesc dflash_desc_from_config(const Value& c) {
  if (!c.is_object()) refuse("config.json is not an object");
  const Value* arch = opt(c, "architectures");
  if (!arch || !arch->is_array() || arch->arr().empty() || !arch->arr()[0].is_string() ||
      !is_dflash_architecture(arch->arr()[0].str()))
    refuse("architectures[0] must be DFlashDraftModel or DFlash2DraftModel");
  if (arch->arr()[0].str() == kLiLiCorrArchitecture) return lilicorr_desc_from_config(c);
  if (arch->arr()[0].str() == kDSparkArchitecture) return dspark_desc_from_config(c);
  return vllm_desc_from_config(c, arch->arr()[0].str(), false);
}

namespace {
DFlashDesc vllm_desc_from_config(const Value& c, const std::string& architecture, bool lili) {
  DFlashDesc d;
  d.architecture = architecture;
  // The AutoRound int4 export (auto-round / gptq) is the loader's to hold (loader::DFlashCheckpoint, QuantConfig).
  if (const Value* qc = lili ? nullptr : opt(c, "quantization_config")) {
    const Value* m = qc->is_object() ? opt(*qc, "quant_method") : nullptr;
    if (!m || !m->is_string() || (m->str() != "auto-round" && m->str() != "gptq"))
      refuse("quantization_config: a quantised drafter is one of our formats only as an AutoRound int4 export "
             "(quant_method auto-round or gptq) - else load the bf16 drafter (it is quantised to int8 at load)");
  }
  // dc = eagle_config, overlaid by dflash_config (config_from_dict).
  Value::Object dco;
  for (const char* k : {"eagle_config", "dflash_config"})
    if (const Value* v = opt(c, k)) {
      if (!v->is_object()) refuse(std::string(k) + " is not an object");
      for (const auto& [kk, vv] : v->obj()) dco[kk] = vv;
    }
  const Value dc = Value::make_obj(dco);
  d.version = (lili || d.architecture.find("DFlash2") != std::string::npos || dc.find("selector_rank") ||
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
  if (lili)   // dflash_ref.py: dflash_config.attention_sink_bias, else add_swa_attention_sink_bias
    d.sinks = bool_or(dc, "attention_sink_bias", bool_or(c, "add_swa_attention_sink_bias", false));
  else if (bool_or(dc, "attention_sink_bias", bool_or(c, "add_swa_attention_sink_bias", false)))
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

  d.block = (lili && !opt(dc, "block_size")) ? u32(c, "block_size") : u32(dc, "block_size");   // LiLiCorr: L:438
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
    if (!lili) {   // LiLiCorr has no selector (its head scores the candidates): rank 0
      d.sel_rank = u32(dc, "selector_rank");
      d.sel_top_k = u32(dc, "selector_top_k");
    }
    if (d.conv_group == 0 || d.hidden % d.conv_group) refuse("conv_group_size must divide hidden_size");
    if (d.conv_taps == 0) refuse("conv_kernel_size must be positive");
    if (!lili && (d.sel_rank == 0 || d.sel_top_k == 0)) refuse("selector_rank / selector_top_k must be positive");
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

// Spec 26b: LiLiCorrDraftModel (V/models/lilicorr.py:428-476; dflash_ref.py lilicorr_config_from_dict).
DFlashDesc lilicorr_desc_from_config(const Value& c) {
  Value::Object dco;
  if (const Value* v = opt(c, "dflash_config")) {
    if (!v->is_object()) refuse("dflash_config is not an object");
    dco = v->obj();
  }
  const Value dc = Value::make_obj(dco);
  if (!bool_or(dc, "lilicorr_enabled", true))
    refuse("dflash_config.lilicorr_enabled=false: LiLiCorrDraftModel requires the LiLiCorr head (V/models/lilicorr.py:436)");
  if (bool_or(c, "has_own_lm_head", false) || bool_or(dc, "has_own_lm_head", false))
    refuse("has_own_lm_head: LiLiCorr drafts with the target's lm_head");
  if (opt(c, "draft_vocab_size") && u32(c, "draft_vocab_size") != u32(c, "vocab_size"))
    refuse("draft_vocab_size != vocab_size: the LiLiCorr candidates need the target's whole vocabulary "
           "(V/models/lilicorr.py:472)");
  const uint32_t topk = u32(dc, "lilicorr_candidate_topk");
  if (topk == 0 || topk > 16 || (topk & (topk - 1)))
    refuse("lilicorr_candidate_topk " + std::to_string(topk) + " is not a power of two <= 16 (V/models/lilicorr.py:143)");
  if (!opt(dc, "conv_kernel_size") || !opt(dc, "conv_group_size") || u32(dc, "conv_kernel_size") == 0 ||
      u32(dc, "conv_group_size") == 0)
    refuse("conv_kernel_size / conv_group_size: LiLiCorr's backbone is DFlash 2's (its two grouped convolutions)");
  if (opt(dc, "output_multiplier") && num_or(dc, "output_multiplier", 1.0) != 1.0)
    refuse("dflash_config.output_multiplier: the LiLiCorr candidates are the target head's RAW logits");
  if (opt(dc, "final_logit_softcapping") && num_or(dc, "final_logit_softcapping", 0.0) != 0.0)
    refuse("dflash_config.final_logit_softcapping: the LiLiCorr candidates are the target head's RAW logits");
  if (const Value* qc = opt(c, "quantization_config")) {
    if (!qc->is_object()) refuse("quantization_config is not an object");
    const Value* m = opt(*qc, "quant_method");
    const Value* a = opt(*qc, "quant_algo");
    if (!m || !m->is_string() || m->str() != "modelopt_fp4")
      refuse("quantization_config.quant_method " + (m && m->is_string() ? m->str() : std::string("(none)")) +
             ": only ModelOpt's NVFP4 (modelopt_fp4) is read - decoded exactly at load into our formats");
    if (!a || !a->is_string() || a->str() != "W4A16_NVFP4")
      refuse("quantization_config.quant_algo " + (a && a->is_string() ? a->str() : std::string("(none)")) +
             ": only W4A16_NVFP4 is read");
    if (u32(*qc, "group_size") != 16)
      refuse("quantization_config.group_size " + std::to_string(u32(*qc, "group_size")) + ": NVFP4's group is 16");
  }
  DFlashDesc d = vllm_desc_from_config(c, kLiLiCorrArchitecture, true);
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.causal[l])
      refuse("layer " + std::to_string(l) + " is causal: the trained LiLiCorr backbone is non-causal (causal: false)");
  d.form = DFlashForm::LiLiCorr;
  d.out_mult = 1.0f;
  d.softcap = 0.0f;
  d.cand_topk = topk;
  d.sel_top_k = topk;
  d.lil_hidden = opt(dc, "lilicorr_hidden_size") ? u32(dc, "lilicorr_hidden_size") : d.hidden;
  d.lil_layers = u32(dc, "lilicorr_num_layers");
  d.lil_heads = u32(dc, "lilicorr_num_heads");
  d.lil_mlp = uint32_t(double(d.lil_hidden) * num_or(dc, "lilicorr_mlp_ratio", 0.0));   // int(hidden x ratio)
  d.factor_dim = u32(dc, "lilicorr_factor_dim");
  d.vector_eps = float(num_or(dc, "lilicorr_vector_eps", 0.0));
  d.lil_logit_scale = float(num_or(dc, "lilicorr_logit_scale", 1.0));
  if (d.lil_hidden == 0 || d.lil_layers == 0 || d.lil_heads == 0 || d.lil_mlp == 0 || d.factor_dim == 0)
    refuse("a zero LiLiCorr width (lilicorr_hidden_size / num_layers / num_heads / mlp_ratio / factor_dim)");
  if (d.lil_hidden % d.lil_heads)
    refuse("lilicorr_hidden_size " + std::to_string(d.lil_hidden) + " is not a multiple of lilicorr_num_heads " +
           std::to_string(d.lil_heads));
  if (!(d.vector_eps > 0.0f)) refuse("lilicorr_vector_eps must be positive");
  return d;
}

// Kolibri DSpark: the speculators form (algos.py update_dspark) - the layer config flattened to the top level and the
// DSpark keys mapped into the dflash_config vllm_desc_from_config reads (taps = aux ids - 1), then the DSpark fields.
DFlashDesc dspark_desc_from_config(const Value& c) {
  const Value* tl = opt(c, "transformer_layer_config");
  if (!tl || !tl->is_object()) refuse("transformer_layer_config missing: a DSpark drafter is a speculators checkpoint");
  if (opt(c, "quantization_config") || opt(*tl, "quantization_config"))
    refuse("quantization_config: the DSpark drafter runs in bf16 (an int8 / int4 drafter is a later step)");
  const Value* sfa = opt(c, "sample_from_anchor");
  if (!sfa || !sfa->is_bool() || !sfa->boolean())
    refuse("sample_from_anchor must be true (anchor-as-first-prediction; the fill-in form is not built)");
  if (const Value* t = opt(c, "markov_head_type"); t && (!t->is_string() || t->str() != "vanilla"))
    refuse("markov_head_type must be vanilla (the first-order Markov bias)");
  if (bool_or(c, "sliding_window_non_causal", false))
    refuse("sliding_window_non_causal=true: the DSpark kernels attend causally inside the block");
  if (bool_or(c, "tie_word_embeddings", false) || bool_or(*tl, "tie_word_embeddings", false))
    refuse("tie_word_embeddings: the drafter shares the target's untied embedding and lm_head");
  const uint32_t rank = u32(c, "markov_rank");
  if (rank == 0) refuse("markov_rank 0 (no Markov head) is DFlash, not DSpark");
  Value::Object o = tl->obj();
  if (const Value* dv = opt(c, "draft_vocab_size")) o["draft_vocab_size"] = *dv;
  Value::Object dc;
  dc["block_size"] = Value::make_num(u32(c, "block_size"));
  dc["mask_token_id"] = Value::make_num(u32(c, "mask_token_id"));
  const Value* aux = opt(c, "aux_hidden_state_layer_ids");
  if (!aux || !aux->is_array() || aux->arr().empty()) refuse("aux_hidden_state_layer_ids missing");
  // TODO(card): update_dspark's `aux - 1` (the input of target layer `aux` = the output of layer aux - 1) assumes the
  // aleph-alpha plugin captures Kolibri's aux states as vLLM's own models do; a low acceptance on the first run (well under
  // the card's ~3 tokens a verify) points here first.
  Value::Array taps;
  for (const Value& v : aux->arr()) {
    if (!v.is_number() || v.num() < 1 || v.num() != double(uint32_t(v.num())))
      refuse("aux_hidden_state_layer_ids holds a value below 1 (each is the input of a target layer: tap = id - 1)");
    taps.push_back(Value::make_num(v.num() - 1));
  }
  dc["target_layer_ids"] = Value::make_arr(std::move(taps));
  o["dflash_config"] = Value::make_obj(std::move(dc));
  DFlashDesc d = vllm_desc_from_config(Value::make_obj(std::move(o)), kDSparkArchitecture, false);
  if (const Value* th = opt(c, "target_hidden_size"); th && u32(c, "target_hidden_size") != d.hidden)
    refuse("target_hidden_size is not hidden_size: the drafter shares the target's embedding and head");
  d.form = DFlashForm::DSpark;
  d.sel_rank = rank;
  d.sel_top_k = 16;   // the Markov walk's candidates: the head's top 16 a row (vLLM's dspark_draft_topk at 16)
  if (bool_or(c, "enable_confidence_head", false))
    d.conf_in = d.hidden + (bool_or(c, "confidence_head_with_markov", false) ? rank : 0);
  return d;
}
}  // namespace

void check_dflash_target(const DFlashDesc& d, const ModelDesc& t, uint32_t target_vocab_rows) {
  check_dflash_target(d, DFlashTargetShape{t.name, t.layers, t.hidden, target_vocab_rows});
}

void check_dflash_target(const DFlashDesc& d, const DFlashTargetShape& t) {
  const uint32_t target_vocab_rows = t.vocab_rows;
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
  // A row of the shared embedding - not necessarily a token the tokenizer defines: Ornith's drafter masks with
  // 248077, the first padded row past its tokenizer's 248077 ids.
  if (d.mask_id >= target_vocab_rows)
    bad("mask_token_id " + std::to_string(d.mask_id) + " is not a row of the target's embedding (< " +
        std::to_string(target_vocab_rows) + ")");
}

void check_lilicorr(const DFlashDesc& d) {
  auto bad = [](const std::string& what) {
    throw std::runtime_error("LiLiCorr drafter: " + what + " - the drafter's kernels are built for "
                             "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr's shapes (kernels::dflash::kLiLiCorr)");
  };
  if (!d.lilicorr()) bad("architectures[0] is " + d.architecture + ", not LiLiCorrDraftModel");
  if (d.layers != 5) bad("num_hidden_layers " + std::to_string(d.layers) + " is not 5");
  if (d.hidden != 4096) bad("hidden_size " + std::to_string(d.hidden) + " is not 4096");
  if (d.q_heads != 32 || d.kv_heads != 2 || d.head_dim != 128)
    bad("heads " + std::to_string(d.q_heads) + " / " + std::to_string(d.kv_heads) + " x " + std::to_string(d.head_dim) +
        " are not 32 / 2 x 128");
  if (d.inter != 12288) bad("intermediate_size " + std::to_string(d.inter) + " is not 12288");
  if (d.taps.size() != 6) bad("target_layer_ids holds " + std::to_string(d.taps.size()) + " layers, not 6");
  if (d.block != 8) bad("block_size " + std::to_string(d.block) + " is not 8");
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.window[l] != 1024)
      bad("layer " + std::to_string(l) + "'s sliding_window " + std::to_string(d.window[l]) +
          " is not 1024 (|q - k| <= 1023)");
    if (d.causal[l]) bad("layer " + std::to_string(l) + " is causal");
  }
  if (!d.sinks) bad("attention_sink_bias is off (the kernels' reduce adds the sink)");
  if (d.conv_taps != 2 || d.conv_group != 16)
    bad("the convolutions are " + std::to_string(d.conv_taps) + " taps x groups of " + std::to_string(d.conv_group) +
        ", not 2 x 16");
  if (d.eps != 1e-6f) bad("rms_norm_eps is not 1e-6 (df_prep bakes it)");
  if (d.cand_topk != 8) bad("lilicorr_candidate_topk " + std::to_string(d.cand_topk) + " is not 8");
  if (d.lil_hidden != 1024 || d.lil_layers != 2 || d.lil_heads != 8 || d.lil_mlp != 2048 || d.factor_dim != 1024)
    bad("the LiLiCorr head is " + std::to_string(d.lil_hidden) + " hidden / " + std::to_string(d.lil_layers) +
        " layers / " + std::to_string(d.lil_heads) + " heads / MLP " + std::to_string(d.lil_mlp) + " / factor " +
        std::to_string(d.factor_dim) + ", not 1024 / 2 / 8 / 2048 / 1024");
}

void check_dspark(const DFlashDesc& d) {
  auto bad = [](const std::string& what) {
    throw std::runtime_error("DSpark drafter: " + what + " - the drafter's kernels are built for intuit/Kolibri-1-dspark's "
                             "shapes (kernels::dflash::kDSpark)");
  };
  if (!d.dspark()) bad("architectures[0] is " + d.architecture + ", not DSparkDraftModel");
  if (d.hidden != 2560) bad("hidden_size " + std::to_string(d.hidden) + " is not 2560");
  if (d.q_heads != 20 || d.kv_heads != 4 || d.head_dim != 128)
    bad("heads " + std::to_string(d.q_heads) + " / " + std::to_string(d.kv_heads) + " x " + std::to_string(d.head_dim) +
        " are not 20 / 4 x 128");
  if (d.inter != 6144) bad("intermediate_size " + std::to_string(d.inter) + " is not 6144");
  if (d.taps.size() != 5) bad("aux_hidden_state_layer_ids holds " + std::to_string(d.taps.size()) + " layers, not 5");
  if (d.block != 8) bad("block_size " + std::to_string(d.block) + " is not 8");
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.window[l] != 2048)
      bad("layer " + std::to_string(l) + "'s sliding_window " + std::to_string(d.window[l]) + " is not 2048");
    if (!d.causal[l]) bad("layer " + std::to_string(l) + " is not causal");
  }
  if (d.sel_rank != 256) bad("markov_rank " + std::to_string(d.sel_rank) + " is not 256");
  if (d.eps != 1e-6f) bad("rms_norm_eps is not 1e-6 (df_prep bakes it)");
  if (d.vocab != 128000) bad("vocab_size " + std::to_string(d.vocab) + " is not 128000");
}

DFlashDesc dflash_tp_card_desc(const DFlashDesc& d) {
  if (d.q_heads % 2 || d.kv_heads % 2 || d.inter % 32 || (d.q_heads / 2 * d.head_dim) % 64 ||
      (d.kv_heads / 2 * d.head_dim) % 16)
    throw std::invalid_argument("model::dflash_tp_card_desc: " + std::to_string(d.q_heads) + " q / " +
                                std::to_string(d.kv_heads) + " kv heads and intermediate " + std::to_string(d.inter) +
                                " do not halve into whole tiles");
  DFlashDesc c = d;
  c.q_heads /= 2;
  c.kv_heads /= 2;
  c.inter /= 2;
  return c;
}

}  // namespace model
