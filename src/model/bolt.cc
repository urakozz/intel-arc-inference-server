#include "model/bolt.h"

#include <cmath>
#include <stdexcept>

#include "common/json_reflect.h"

namespace model {
namespace {

// The published pattern and per-MoE-layer (E, I): docs/probes/probe-bolt-2026-10-21.md §2 (25a read them from the published
// config's block_configs); tests/model/bolt_test.cc holds this table to the vendored config.json block by block.
const char* const kPattern = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";
struct EI {
  uint32_t layer, e, i;
};
const EI kMoe[] = {
    {1, 240, 2048},  {3, 240, 1536},  {5, 240, 2048},  {8, 304, 2304},  {10, 240, 2304}, {12, 240, 2048},
    {14, 240, 2048}, {17, 304, 2304}, {19, 304, 2048}, {21, 432, 2560}, {23, 432, 2304}, {26, 240, 2048},
    {28, 432, 2304}, {30, 304, 2560}, {32, 304, 2304}, {34, 432, 2304}, {37, 368, 2304}, {39, 368, 2048},
    {41, 368, 2304}, {43, 432, 2560}, {45, 368, 2560}, {48, 496, 2560}, {50, 432, 2560}, {52, 240, 2304},
    {54, 368, 2304}, {56, 368, 2560}, {59, 304, 2048}, {61, 240, 2304}, {63, 240, 2304}, {65, 240, 2048},
    {67, 240, 2048}, {70, 368, 2048}, {72, 368, 2304}, {74, 432, 2304}, {76, 432, 2560}, {79, 496, 2560},
    {81, 432, 2560}, {83, 432, 2560}, {85, 496, 2560}, {87, 368, 2560}};

BoltDesc make_bolt() {
  BoltDesc d;
  d.block.assign(d.layers, BoltBlock::Mamba);
  d.experts.assign(d.layers, 0);
  d.inter.assign(d.layers, 0);
  for (uint32_t l = 0; l < d.layers; ++l)
    d.block[l] = kPattern[l] == 'E' ? BoltBlock::Moe : kPattern[l] == '*' ? BoltBlock::Attention : BoltBlock::Mamba;
  for (const EI& m : kMoe) {
    d.experts.at(m.layer) = m.e;
    d.inter.at(m.layer) = m.i;
  }
  return d;
}

[[noreturn]] void bad(const std::string& what) {
  throw std::invalid_argument("config.json: " + what + " - nemotron-3.5-bolt-74b-a9b (model/bolt.h) is not written for it");
}

using common::json::show;

const common::json::Value* get(const common::json::Value& c, const char* key) { return c.is_object() ? c.find(key) : nullptr; }

uint32_t need_u32(const common::json::Value& c, const char* key) {
  const common::json::Value* v = get(c, key);
  if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()))
    bad(std::string("'") + key + "' is " + (v ? show(*v) : std::string("absent")) + ", not a non-negative integer");
  return uint32_t(v->num());
}

// config.json's keys held to the descriptor by common::json::expect, in the order the checks run: a member without an
// initializer is required, the rest may be absent (their initializer is the class default the engine is written for).
struct BoltLlmKeys {
  std::string model_type;
  bool tie_word_embeddings = false;
  uint32_t hidden_size, vocab_size, num_attention_heads, num_key_value_heads, head_dim, mamba_num_heads, mamba_head_dim,
      n_groups, ssm_state_size, conv_kernel, moe_latent_size, moe_shared_expert_intermediate_size, n_shared_experts = 1,
      chunk_size = 128, expand = 2;
  uint32_t num_nextn_predict_layers = 0;   // MTP: Bolt has none (the drafter is spec 26's, separate)
  bool residual_in_fp32 = false, use_qk_norm = false;
  uint32_t n_group = 1, topk_group = 1;
  std::string scoring_func = "sigmoid";
  bool norm_topk_prob = true;
  double routed_scaling_factor;
  bool mamba_proj_bias = false, attention_bias = false, mlp_bias = false, use_bias = false, use_conv_bias = true;
  std::string hidden_act = "relu2", mlp_hidden_act = "relu2", mamba_hidden_act = "silu";
};
struct BoltKeys {
  std::string model_type;
  bool tie_word_embeddings = false;
  BoltLlmKeys llm_config;
};

// transformers' special-float encoding: {"__float__": "Infinity"}.
bool is_inf(const common::json::Value& v) {
  if (v.is_object()) {
    const common::json::Value* f = v.find("__float__");
    return f && f->is_string() && f->str() == "Infinity";
  }
  return v.is_number() && v.num() >= 1e30;
}

BoltBlock block_of(const std::string& t, size_t i) {
  if (t == "mamba") return BoltBlock::Mamba;
  if (t == "moe") return BoltBlock::Moe;
  if (t == "attention") return BoltBlock::Attention;
  bad("block_configs[" + std::to_string(i) + "].block_type is '" + t + "', one of mamba / moe / attention");
}

}  // namespace

uint32_t BoltDesc::mamba_index(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l; ++i) n += block.at(i) == BoltBlock::Mamba ? 1 : 0;
  return n;
}
uint32_t BoltDesc::attn_index(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l; ++i) n += block.at(i) == BoltBlock::Attention ? 1 : 0;
  return n;
}
uint32_t BoltDesc::moe_index(uint32_t l) const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < l; ++i) n += block.at(i) == BoltBlock::Moe ? 1 : 0;
  return n;
}
uint32_t BoltDesc::max_experts() const {
  uint32_t m = 0;
  for (uint32_t e : experts) m = e > m ? e : m;
  return m;
}
uint64_t BoltDesc::routed_experts() const {
  uint64_t n = 0;
  for (uint32_t l = 0; l < layers; ++l) n += is_moe(l) ? experts.at(l) : 0;
  return n;
}

std::string BoltDesc::layer_prefix(const std::string& trunk, uint32_t l) {
  return trunk + "layers." + std::to_string(l) + ".";
}

std::vector<BoltLinear> BoltDesc::linears(uint32_t l) const {
  std::vector<BoltLinear> v;
  const WeightKind i4 = WeightKind::Int4;
  switch (block.at(l)) {
    case BoltBlock::Mamba:
      v.push_back({BoltLinearId::InProj, {hidden, m_inproj, 1, 0}, i4, Fuse::Single, {"in_proj"}, {m_inproj}});
      v.push_back({BoltLinearId::OutProj, {m_inner, hidden, 1, 1}, i4, Fuse::Single, {"out_proj"}, {hidden}});
      break;
    case BoltBlock::Attention:
      v.push_back({BoltLinearId::Qkv, {hidden, qkv_n(), 1, 0}, i4, Fuse::Concat,
                   {"q_proj", "k_proj", "v_proj"}, {q_n(), kv_n(), kv_n()}});
      v.push_back({BoltLinearId::OProj, {q_n(), hidden, 1, 1}, i4, Fuse::Single, {"o_proj"}, {hidden}});
      break;
    case BoltBlock::Moe:
      v.push_back({BoltLinearId::Fc1SharedUp, {hidden, latent + shared_inter, 1, 0}, i4, Fuse::Concat,
                   {"fc1_latent_proj", "shared_experts.up_proj"}, {latent, shared_inter}});
      v.push_back({BoltLinearId::Fc2, {latent, hidden, 1, 1}, i4, Fuse::Single, {"fc2_latent_proj"}, {hidden}});
      v.push_back({BoltLinearId::SharedDown, {shared_inter, hidden, 1, 1}, i4, Fuse::Single,
                   {"shared_experts.down_proj"}, {hidden}});
      v.push_back({BoltLinearId::ExpertUp, {latent, inter.at(l), 1, 1}, i4, Fuse::Single, {"experts.<e>.up_proj"},
                   {inter.at(l)}});
      v.push_back({BoltLinearId::ExpertDown, {inter.at(l), latent, 1, 1}, i4, Fuse::Single, {"experts.<e>.down_proj"},
                   {latent}});
      v.push_back({BoltLinearId::Router, {hidden, experts.at(l), 1, 0}, WeightKind::Bf16, Fuse::Single, {"gate"},
                   {experts.at(l)}});
      break;
  }
  return v;
}

void validate(const BoltPlacement& p, const BoltDesc& d) {
  if (p.layers != d.layers)
    throw std::invalid_argument("BoltPlacement: placed " + std::to_string(p.layers) + " layers, " + d.name + " has " +
                                std::to_string(d.layers));
  if (p.devices == 1) {
    if (p.split != d.layers)
      throw std::invalid_argument("BoltPlacement: one device holds every layer (split " + std::to_string(p.split) +
                                  " != " + std::to_string(d.layers) + ")");
    return;
  }
  if (p.devices != 2)
    throw std::invalid_argument("BoltPlacement: " + std::to_string(p.devices) + " devices (two are supported)");
  if (p.split < 1 || p.split + 1 > d.layers)
    throw std::invalid_argument("BoltPlacement: split " + std::to_string(p.split) + " is outside [1, " +
                                std::to_string(d.layers > 0 ? d.layers - 1 : 0) + "] - each device must hold at least one layer");
}

const BoltDesc& bolt() {
  static const BoltDesc d = make_bolt();
  return d;
}

bool is_bolt(const common::json::Value& config) {
  if (!config.is_object()) return false;
  const common::json::Value* mt = config.find("model_type");
  const common::json::Value* llm = config.find("llm_config");
  if (!mt || !mt->is_string() || mt->str() != "nemotron_h_omni" || !llm || !llm->is_object()) return false;
  const common::json::Value* lt = llm->find("model_type");
  return lt && lt->is_string() && lt->str() == "nemotron_h_puzzle";
}

void check_bolt_config(const common::json::Value& c) {
  const BoltDesc& d = bolt();
  if (!c.is_object()) bad("not a JSON object");
  common::json::expect<true>(
      c,
      BoltKeys{.model_type = d.model_type,
               .llm_config = {.model_type = d.llm_model_type, .hidden_size = d.hidden, .vocab_size = d.vocab,
                              .num_attention_heads = d.q_heads, .num_key_value_heads = d.kv_heads,
                              .head_dim = d.head_dim, .mamba_num_heads = d.m_heads, .mamba_head_dim = d.m_head_dim,
                              .n_groups = d.m_groups, .ssm_state_size = d.m_state, .conv_kernel = d.m_conv,
                              .moe_latent_size = d.latent, .moe_shared_expert_intermediate_size = d.shared_inter,
                              .routed_scaling_factor = d.routed_scale}},
      bad);
  const common::json::Value& m = c.at("llm_config");
  need_u32(m, "max_position_embeddings");
  if (const common::json::Value* v = get(m, "mtp_block_configs"); v && !(v->is_array() && v->arr().empty()))
    bad("'mtp_block_configs' is " + show(*v) + ", the descriptor needs []");
  if (const common::json::Value* v = get(m, "mtp_layers_block_type"); v && !(v->is_array() && v->arr().empty()))
    bad("'mtp_layers_block_type' is " + show(*v) + ", the descriptor needs []");
  if (const common::json::Value* v = get(m, "sliding_window"); v && !v->is_null())
    bad("'sliding_window' is " + show(*v) + ", the descriptor needs null (no sliding window)");

  // The RMSNorm eps (R's `layer_norm_epsilon`); the two other spellings, when present, agree.
  const common::json::Value* eps = get(m, "layer_norm_epsilon");
  if (!eps || !eps->is_number() || float(eps->num()) != d.eps)
    bad("'layer_norm_epsilon' is " + (eps ? show(*eps) : std::string("absent")) + ", the descriptor says 1e-05");
  for (const char* k : {"norm_eps", "rms_norm_eps"})
    if (const common::json::Value* v = get(m, k); v && (!v->is_number() || float(v->num()) != d.eps))
      bad(std::string("'") + k + "' is " + show(*v) + ", the descriptor says 1e-05");

  // dt_limit (0, inf): the decode chain has no clamp (spec 25 §2.0).
  if (const common::json::Value* v = get(m, "time_step_limit")) {
    if (!v->is_array() || v->arr().size() != 2 || !v->arr()[0].is_number() || v->arr()[0].num() != 0.0 ||
        !is_inf(v->arr()[1]))
      bad("'time_step_limit' is " + show(*v) + ", the descriptor needs [0.0, Infinity] (no dt clamp)");
  }

  // The blocks. E and I come from here, never from the top-level num_experts / intermediate_size.
  const common::json::Value* bc = get(m, "block_configs");
  if (!bc || !bc->is_array() || bc->arr().empty() || bc->arr().size() > d.layers)
    bad("'block_configs' is not a list of 1.." + std::to_string(d.layers) + " blocks");
  const common::json::Value* lt = get(m, "layers_block_type");
  if (lt && (!lt->is_array() || lt->arr().size() != bc->arr().size()))
    bad("'layers_block_type' is not a list of block_configs' " + std::to_string(bc->arr().size()) + " entries");
  for (size_t i = 0; i < bc->arr().size(); ++i) {
    const common::json::Value& b = bc->arr()[i];
    const std::string at = "block_configs[" + std::to_string(i) + "]";
    if (!b.is_object()) bad("'" + at + "' is not an object");
    const common::json::Value* t = b.find("block_type");
    if (!t || !t->is_string()) bad("'" + at + ".block_type' is " + (t ? show(*t) : std::string("absent")));
    const BoltBlock kind = block_of(t->str(), i);
    // the four per-block members the config declares (`blockwise_members`): a MoE block carries all four, the others only
    // block_type - any other key is a block the descriptor does not describe
    for (const auto& [key, unused] : b.obj()) {
      (void)unused;
      const bool ok = key == "block_type" || (kind == BoltBlock::Moe && (key == "moe_intermediate_size" ||
                                                                          key == "n_routed_experts" || key == "num_experts_per_tok"));
      if (!ok) bad("'" + at + "' has the key '" + key + "' - a " + t->str() + " block carries " +
                   (kind == BoltBlock::Moe ? "block_type, n_routed_experts, moe_intermediate_size and num_experts_per_tok" : "block_type only"));
    }
    if (lt) {
      const common::json::Value& lv = lt->arr()[i];
      if (!lv.is_string() || lv.str() != t->str())
        bad("'layers_block_type'[" + std::to_string(i) + "] is " + show(lv) + ", block_configs says '" + t->str() + "'");
    }
    if (kind != BoltBlock::Moe) continue;
    const uint32_t e = need_u32(b, "n_routed_experts"), inter = need_u32(b, "moe_intermediate_size");
    if (e == 0 || e % 16 != 0 || e > 512)
      bad("'" + at + ".n_routed_experts' is " + std::to_string(e) + ", a multiple of 16 in 16..512 (the router tiles)");
    if (inter == 0 || inter % 64 != 0)
      bad("'" + at + ".moe_intermediate_size' is " + std::to_string(inter) + ", not a multiple of 64 (the g64 groups)");
    const uint32_t k = need_u32(b, "num_experts_per_tok");
    if (k != d.top_k) bad("'" + at + ".num_experts_per_tok' is " + std::to_string(k) + ", the descriptor says 9");
  }
}

BoltDesc bolt_desc(const common::json::Value& config) {
  check_bolt_config(config);
  const common::json::Value& llm = config.at("llm_config");
  BoltDesc d = bolt();
  d.max_pos = need_u32(llm, "max_position_embeddings");
  const auto& bc = llm.at("block_configs").arr();
  d.layers = uint32_t(bc.size());
  d.block.assign(d.layers, BoltBlock::Mamba);
  d.experts.assign(d.layers, 0);
  d.inter.assign(d.layers, 0);
  for (uint32_t l = 0; l < d.layers; ++l) {
    d.block[l] = block_of(bc[l].at("block_type").str(), l);
    if (d.block[l] == BoltBlock::Moe) {
      d.experts[l] = need_u32(bc[l], "n_routed_experts");
      d.inter[l] = need_u32(bc[l], "moe_intermediate_size");
    }
  }
  d.vocab_used = d.vocab;
  return d;
}

BoltDesc bolt_truncated(const BoltDesc& d, uint32_t layers) {
  if (layers == 0 || layers > d.layers)
    throw std::invalid_argument("--layers " + std::to_string(layers) + ": the checkpoint has " + std::to_string(d.layers) +
                                " blocks");
  BoltDesc t = d;
  t.layers = layers;
  t.block.resize(layers);
  t.experts.resize(layers);
  t.inter.resize(layers);
  return t;
}

}  // namespace model
