// Spec 25b Task 2: model::BoltDesc against the checkpoint's own facts (spec 25 §1.1, §13; docs/probe-bolt-2026-10-21.md).
// Host only. argv[1] = tests/model/bolt/config.json (SergiioB's published config, copied unchanged - Decision 20).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/bolt.h"

namespace {

using common::json::Value;
using model::BoltBlock;
using model::BoltDesc;
using model::BoltLinearId;

Value slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

// A copy of `v` with the value at `path` replaced (a path element "[3]" indexes an array; any other is an object key;
// `nv` absent = erase the key). Rebuilds the objects on the way - json::Value has no mutators.
Value with(const Value& v, const std::vector<std::string>& path, size_t at, const Value* nv) {
  if (at == path.size()) return *nv;
  const std::string& k = path[at];
  if (k[0] == '[') {
    Value::Array a = v.arr();
    const size_t i = std::stoul(k.substr(1));
    CHECK(i < a.size());
    a[i] = with(a[i], path, at + 1, nv);
    return Value::make_arr(std::move(a));
  }
  Value::Object o = v.obj();
  if (at + 1 == path.size() && !nv) {
    CHECK(o.erase(k) == 1);
  } else {
    auto it = o.find(k);
    CHECK(it != o.end() || (at + 1 == path.size()));
    o[k] = with(it != o.end() ? it->second : Value(), path, at + 1, nv);
  }
  return Value::make_obj(std::move(o));
}
Value set(const Value& v, std::vector<std::string> path, const Value& nv) { return with(v, path, 0, &nv); }
Value erase(const Value& v, std::vector<std::string> path) { return with(v, path, 0, nullptr); }

// Every refusal throws std::invalid_argument naming `key` (the key's own name).
void refused(const Value& cfg, const char* key) {
  bool threw = false;
  try {
    model::check_bolt_config(cfg);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  if (!threw) std::fprintf(stderr, "not refused (or not by name): %s\n", key);
  CHECK(threw);
}

const Value B(bool b) { return Value::make_bool(b); }
const Value N(double d) { return Value::make_num(d); }
const Value S(const char* s) { return Value::make_str(s); }

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const Value cfg = slurp(argv[1]);
  const BoltDesc& d = model::bolt();

  // --- the published config parses, and equals the singleton block by block (spec 25 §1.1) ---------------------
  CHECK(model::is_bolt(cfg));
  model::check_bolt_config(cfg);
  const BoltDesc p = model::bolt_desc(cfg);
  CHECK_EQ(p.layers, 88u);
  CHECK(p.block == d.block && p.experts == d.experts && p.inter == d.inter);
  CHECK(p.hidden == 4096 && p.vocab == 131072 && p.vocab_used == 131072 && p.max_pos == 262144);
  CHECK(p.m_heads == 128 && p.m_head_dim == 64 && p.m_groups == 8 && p.m_state == 128 && p.m_conv == 4);
  CHECK(p.m_inner == 8192 && p.m_conv_dim == 10240 && p.m_inproj == 18560);
  CHECK_EQ(p.m_inner, p.m_heads * p.m_head_dim);
  CHECK_EQ(p.m_conv_dim, p.m_inner + 2 * p.m_groups * p.m_state);
  CHECK_EQ(p.m_inproj, p.m_inner + p.m_conv_dim + p.m_heads);
  CHECK(p.q_heads == 32 && p.kv_heads == 2 && p.head_dim == 128);
  CHECK(p.latent == 1024 && p.shared_inter == 5376 && p.top_k == 9);
  CHECK(p.eps == 1e-5f && p.routed_scale == 5.0f);
  CHECK_EQ(p.q_n(), 4096u);
  CHECK_EQ(p.kv_n(), 256u);
  CHECK_EQ(p.qkv_n(), 4608u);

  // The block pattern against the config's own list (the typed table is held to it here, not trusted).
  const auto& bc = cfg.at("llm_config").at("block_configs").arr();
  CHECK_EQ(bc.size(), size_t(88));
  for (uint32_t l = 0; l < 88; ++l) {
    const std::string t = bc[l].at("block_type").str();
    CHECK(d.block[l] == (t == "mamba" ? BoltBlock::Mamba : t == "moe" ? BoltBlock::Moe : BoltBlock::Attention));
    if (t == "moe") {
      CHECK_EQ(d.experts[l], uint32_t(bc[l].at("n_routed_experts").num()));
      CHECK_EQ(d.inter[l], uint32_t(bc[l].at("moe_intermediate_size").num()));
    } else {
      CHECK(d.experts[l] == 0 && d.inter[l] == 0);
    }
  }
  CHECK_EQ(d.mamba_layers(), 40u);
  CHECK_EQ(d.moe_layers(), 40u);
  CHECK_EQ(d.attn_layers(), 8u);
  const uint32_t att[] = {7, 16, 25, 36, 47, 58, 69, 78};
  for (uint32_t i = 0; i < 8; ++i) {
    CHECK(d.is_attention(att[i]));
    CHECK_EQ(d.attn_index(att[i]), i);
  }
  CHECK(d.is_moe(87));   // the last block is MoE
  CHECK(d.is_mamba(0) && d.is_moe(1));
  CHECK_EQ(d.mamba_index(2), 1u);
  CHECK_EQ(d.moe_index(2), 1u);

  // E and I per layer (the facts sheet's distributions, spec 25 §1.1): 12 distinct (E, I) pairs, E in {240 (12), 304 (6),
  // 368 (9), 432 (10), 496 (3)}, I in {1536 (1), 2048 (11), 2304 (15), 2560 (13)}, 13,824 routed experts.
  std::map<uint32_t, int> ce, ci;
  std::set<std::pair<uint32_t, uint32_t>> pairs;
  for (uint32_t l = 0; l < 88; ++l)
    if (d.is_moe(l)) {
      ++ce[d.experts[l]];
      ++ci[d.inter[l]];
      pairs.insert({d.experts[l], d.inter[l]});
    }
  CHECK((ce == std::map<uint32_t, int>{{240, 12}, {304, 6}, {368, 9}, {432, 10}, {496, 3}}));
  CHECK((ci == std::map<uint32_t, int>{{1536, 1}, {2048, 11}, {2304, 15}, {2560, 13}}));
  CHECK_EQ(pairs.size(), size_t(12));
  CHECK_EQ(d.routed_experts(), uint64_t(13824));
  CHECK_EQ(d.max_experts(), 496u);
  // E / I are never the top level's: num_experts 512 is a maximum no layer has; intermediate_size 2688 no block has.
  const Value& llm = cfg.at("llm_config");
  CHECK_EQ(uint32_t(llm.at("num_experts").num()), 512u);
  CHECK_EQ(uint32_t(llm.at("intermediate_size").num()), 2688u);
  for (uint32_t l = 0; l < 88; ++l) CHECK(d.experts[l] != 512 && d.inter[l] != 2688);
  {
    const Value a = set(cfg, {"llm_config", "num_experts"}, N(64));
    const Value b = set(cfg, {"llm_config", "intermediate_size"}, N(64));
    CHECK(model::bolt_desc(a).experts == d.experts && model::bolt_desc(b).inter == d.inter);
  }

  // --- the per-block linears (spec 25 §3's table; the provisional {S, layout}) --------------------------------
  {
    const auto m = d.linears(0);
    CHECK_EQ(m.size(), size_t(2));
    CHECK(m[0].id == BoltLinearId::InProj && m[0].shape.K == 4096 && m[0].shape.N == 18560 && m[0].shape.S == 1 &&
          m[0].shape.layout == 0);
    CHECK(m[1].id == BoltLinearId::OutProj && m[1].shape.K == 8192 && m[1].shape.N == 4096 && m[1].shape.layout == 1);
    const auto a = d.linears(7);
    CHECK_EQ(a.size(), size_t(2));
    CHECK(a[0].id == BoltLinearId::Qkv && a[0].shape.K == 4096 && a[0].shape.N == 4608 && a[0].shape.layout == 0);
    CHECK(a[0].parts == (std::vector<std::string>{"q_proj", "k_proj", "v_proj"}));
    CHECK(a[0].part_n == (std::vector<uint32_t>{4096, 256, 256}));
    CHECK(a[1].id == BoltLinearId::OProj && a[1].shape.K == 4096 && a[1].shape.N == 4096 && a[1].shape.layout == 1);
    const auto e = d.linears(3);   // (240, 1536)
    CHECK_EQ(e.size(), size_t(6));
    CHECK(e[0].id == BoltLinearId::Fc1SharedUp && e[0].shape.K == 4096 && e[0].shape.N == 6400 && e[0].shape.layout == 0);
    CHECK(e[0].parts == (std::vector<std::string>{"fc1_latent_proj", "shared_experts.up_proj"}));
    CHECK(e[0].part_n == (std::vector<uint32_t>{1024, 5376}));
    CHECK(e[1].id == BoltLinearId::Fc2 && e[1].shape.K == 1024 && e[1].shape.N == 4096 && e[1].shape.layout == 1);
    CHECK(e[2].id == BoltLinearId::SharedDown && e[2].shape.K == 5376 && e[2].shape.N == 4096 && e[2].shape.layout == 1);
    CHECK(e[3].id == BoltLinearId::ExpertUp && e[3].shape.K == 1024 && e[3].shape.N == 1536 && e[3].shape.layout == 1);
    CHECK(e[4].id == BoltLinearId::ExpertDown && e[4].shape.K == 1536 && e[4].shape.N == 1024 && e[4].shape.layout == 1);
    CHECK(e[5].id == BoltLinearId::Router && e[5].shape.K == 4096 && e[5].shape.N == 240 &&
          e[5].kind == model::WeightKind::Bf16);
    // the routed experts of each layer follow THEIR layer's I and E, and every int4 shape is g64- and tile-clean
    for (uint32_t l = 0; l < 88; ++l)
      for (const auto& ln : d.linears(l)) {
        CHECK_EQ(ln.shape.S, 1u);
        uint32_t sum = 0;
        for (uint32_t n : ln.part_n) sum += n;
        CHECK(sum == ln.shape.N || ln.id == BoltLinearId::ExpertUp || ln.id == BoltLinearId::ExpertDown);
        if (ln.kind == model::WeightKind::Int4) {
          CHECK_EQ(ln.shape.K % 64, 0u);
          CHECK_EQ(ln.shape.N % 16, 0u);
          // layout 1 exactly for the routed experts and every N == 4096 linear, layout 0 for the rest
          const bool l1 = ln.id == BoltLinearId::ExpertUp || ln.id == BoltLinearId::ExpertDown || ln.shape.N == 4096;
          CHECK_EQ(ln.shape.layout, l1 ? 1u : 0u);
        }
        if (ln.id == BoltLinearId::ExpertUp) CHECK_EQ(ln.shape.N, d.inter[l]);
        if (ln.id == BoltLinearId::Router) {
          CHECK_EQ(ln.shape.N, d.experts[l]);
          CHECK_EQ(ln.shape.N % 16, 0u);   // no padding rows
        }
      }
  }
  CHECK(model::BoltDesc::layer_prefix("language_model.model.", 12) == "language_model.model.layers.12.");

  // --- the truncation keeps blocks [0, N) ---------------------------------------------------------------------
  {
    const BoltDesc t = model::bolt_truncated(d, 12);
    CHECK_EQ(t.layers, 12u);
    CHECK(std::equal(t.block.begin(), t.block.end(), d.block.begin()));
    CHECK(std::equal(t.experts.begin(), t.experts.end(), d.experts.begin()));
    CHECK_EQ(t.attn_layers(), 1u);   // the first attention block is 7
    CHECK_EQ(t.moe_layers(), 5u);
    bool threw = false;
    try { model::bolt_truncated(d, 0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { model::bolt_truncated(d, 89); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    // a config that lists the first 8 blocks (what make_synth.py writes) is the same descriptor truncated to 8
    Value::Array first8(bc.begin(), bc.begin() + 8), lt8;
    for (const Value& b : first8) lt8.push_back(b.at("block_type"));
    Value c8 = set(cfg, {"llm_config", "block_configs"}, Value::make_arr(first8));
    c8 = set(c8, {"llm_config", "layers_block_type"}, Value::make_arr(lt8));
    const BoltDesc d8 = model::bolt_desc(c8);
    CHECK_EQ(d8.layers, 8u);
    CHECK(d8.block == model::bolt_truncated(d, 8).block && d8.experts == model::bolt_truncated(d, 8).experts &&
          d8.inter == model::bolt_truncated(d, 8).inter);
  }

  // --- every refusal names its key (spec 25b plan Task 2) ------------------------------------------------------
  refused(set(cfg, {"model_type"}, S("nemotron_h")), "model_type");
  refused(set(cfg, {"llm_config", "model_type"}, S("nemotron_h")), "model_type");
  refused(erase(cfg, {"llm_config"}), "llm_config");
  refused(set(cfg, {"tie_word_embeddings"}, B(true)), "tie_word_embeddings");
  refused(set(cfg, {"llm_config", "tie_word_embeddings"}, B(true)), "tie_word_embeddings");
  refused(set(cfg, {"llm_config", "hidden_size"}, N(2048)), "hidden_size");
  refused(set(cfg, {"llm_config", "vocab_size"}, N(131073)), "vocab_size");
  refused(set(cfg, {"llm_config", "num_attention_heads"}, N(16)), "num_attention_heads");
  refused(set(cfg, {"llm_config", "num_key_value_heads"}, N(4)), "num_key_value_heads");
  refused(set(cfg, {"llm_config", "head_dim"}, N(64)), "head_dim");
  refused(set(cfg, {"llm_config", "mamba_num_heads"}, N(64)), "mamba_num_heads");
  refused(set(cfg, {"llm_config", "mamba_head_dim"}, N(128)), "mamba_head_dim");
  refused(set(cfg, {"llm_config", "n_groups"}, N(4)), "n_groups");
  refused(set(cfg, {"llm_config", "ssm_state_size"}, N(64)), "ssm_state_size");
  refused(set(cfg, {"llm_config", "conv_kernel"}, N(5)), "conv_kernel");
  refused(set(cfg, {"llm_config", "moe_latent_size"}, N(512)), "moe_latent_size");
  refused(set(cfg, {"llm_config", "moe_shared_expert_intermediate_size"}, N(4096)), "moe_shared_expert_intermediate_size");
  refused(set(cfg, {"llm_config", "num_nextn_predict_layers"}, N(1)), "num_nextn_predict_layers");
  refused(set(cfg, {"llm_config", "mtp_block_configs"}, Value::make_arr({Value::make_obj({})})), "mtp_block_configs");
  refused(set(cfg, {"llm_config", "residual_in_fp32"}, B(true)), "residual_in_fp32");
  refused(set(cfg, {"llm_config", "use_qk_norm"}, B(true)), "use_qk_norm");
  refused(set(cfg, {"llm_config", "n_group"}, N(8)), "n_group");
  refused(set(cfg, {"llm_config", "topk_group"}, N(4)), "topk_group");
  refused(set(cfg, {"llm_config", "scoring_func"}, S("softmax")), "scoring_func");
  refused(set(cfg, {"llm_config", "norm_topk_prob"}, B(false)), "norm_topk_prob");
  refused(set(cfg, {"llm_config", "routed_scaling_factor"}, N(2.5)), "routed_scaling_factor");
  refused(set(cfg, {"llm_config", "block_configs", "[1]", "num_experts_per_tok"}, N(8)), "num_experts_per_tok");
  refused(set(cfg, {"llm_config", "block_configs", "[1]", "n_routed_experts"}, N(250)), "n_routed_experts");
  refused(set(cfg, {"llm_config", "block_configs", "[1]", "n_routed_experts"}, N(528)), "n_routed_experts");
  refused(set(cfg, {"llm_config", "block_configs", "[1]", "moe_intermediate_size"}, N(2080)), "moe_intermediate_size");
  refused(set(cfg, {"llm_config", "block_configs", "[2]", "block_type"}, S("ssm")), "block_type");
  refused(set(cfg, {"llm_config", "layers_block_type", "[2]"}, S("attention")), "layers_block_type");
  refused(set(cfg, {"llm_config", "sliding_window"}, N(4096)), "sliding_window");
  refused(set(cfg, {"llm_config", "mamba_proj_bias"}, B(true)), "mamba_proj_bias");
  refused(set(cfg, {"llm_config", "attention_bias"}, B(true)), "attention_bias");
  refused(set(cfg, {"llm_config", "mlp_bias"}, B(true)), "mlp_bias");
  refused(set(cfg, {"llm_config", "use_conv_bias"}, B(false)), "use_conv_bias");
  refused(set(cfg, {"llm_config", "hidden_act"}, S("silu")), "hidden_act");
  refused(set(cfg, {"llm_config", "mlp_hidden_act"}, S("silu")), "mlp_hidden_act");
  refused(set(cfg, {"llm_config", "mamba_hidden_act"}, S("relu2")), "mamba_hidden_act");
  refused(set(cfg, {"llm_config", "layer_norm_epsilon"}, N(1e-6)), "layer_norm_epsilon");
  refused(set(cfg, {"llm_config", "time_step_limit"}, Value::make_arr({N(0.001), N(1e38)})), "time_step_limit");
  refused(set(cfg, {"llm_config", "block_configs"}, Value::make_arr({})), "block_configs");
  {
    Value::Array too_many = bc;
    too_many.push_back(bc[0]);
    refused(set(cfg, {"llm_config", "block_configs"}, Value::make_arr(too_many)), "block_configs");
  }
  // the keys the real config carries and the descriptor holds: dropping the required ones is refused too
  refused(erase(cfg, {"llm_config", "hidden_size"}), "hidden_size");
  refused(erase(cfg, {"llm_config", "block_configs", "[1]", "n_routed_experts"}), "n_routed_experts");

  std::puts("bolt_test OK");
  return 0;
}
