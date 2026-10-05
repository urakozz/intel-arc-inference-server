// Spec 18b Task 1: model::K2Desc against the checkpoint's own facts. Host only.
//
// Every shape literal below is a safetensors HEADER shape of
// urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ (fetched 2026-10-05 by range
// request; spec 18 §1), not a derivation from the table under test: qweight [K/8][N] gives
// the K and N each row must have. argv[1] is tests/model/k2/config.json - the checkpoint's
// config.json with `auto_map` dropped and quantization_config's 45 dynamic rules cut to two.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "common/json.h"
#include "loader/k2_layout.h"
#include "model/k2_horizon.h"

namespace {

using model::K2Desc;
using model::K2ExpertId;
using model::K2LinearId;

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// `cfg` with the first `from` replaced by `to` must be refused, and the message must name
// `key`.
void refused(const std::string& cfg, const std::string& from, const std::string& to,
             const char* key) {
  const size_t at = cfg.find(from);
  if (at == std::string::npos) {
    std::fprintf(stderr, "fixture has no '%s'\n", from.c_str());
    std::exit(1);
  }
  std::string bad = cfg;
  bad.replace(at, from.size(), to);
  bool threw = false;
  try {
    model::check_k2_config(model::k2(), common::json::parse(bad));
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  CHECK(threw);
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const K2Desc& d = model::k2();

  // --- the model, spec 18 §1 ------------------------------------------------------------
  CHECK_EQ(d.layers, 48u);
  CHECK_EQ(d.dense_layers, 3u);
  CHECK_EQ(d.sparse_layers(), 45u);
  for (uint32_t l = 0; l < d.layers; ++l) CHECK_EQ(d.is_dense(l), l < 3);
  CHECK_EQ(d.hidden, 2560u);
  CHECK_EQ(d.group_size(), 1280u);   // the second norm trap: two groups, each its own mean
  CHECK_EQ(d.gqa(), 4u);
  CHECK_EQ(d.q_n(), 4096u);
  CHECK_EQ(d.kv_n(), 1024u);
  CHECK_EQ(d.vocab, 250624u);
  CHECK_EQ(d.vocab_used, d.vocab);   // fully used
  CHECK_EQ(d.eos.size(), size_t(2));
  CHECK_EQ(d.eos[1], 250019u);
  CHECK(d.router_scale == 2.5f);
  // The fused attention column map: q 4096 | k 1024 | gate 4096 | v (1024) or v_router (64).
  CHECK_EQ(d.k_off(), 4096u);
  CHECK_EQ(d.gate_off(), 5120u);
  CHECK_EQ(d.v_off(), 9216u);
  CHECK_EQ(d.attn_dense_n(), 10240u);
  CHECK_EQ(d.attn_sparse_n(), 9280u);
  // Review Focus 4: the router's 100 rows padded to 128, the route kernel's lane count.
  CHECK_EQ(d.router_n(), 128u);
  CHECK_EQ(K2Desc::route_wg(64), 64u);
  CHECK_EQ(K2Desc::route_wg(100), 128u);
  CHECK_EQ(K2Desc::route_wg(8), 16u);
  CHECK_EQ(d.moe_blocks(), 101u);
  CHECK_EQ(d.shared_block(), 100u);

  // --- the linears against the header shapes -------------------------------------------
  struct Want {
    K2LinearId id;
    uint32_t K, N;
    model::Fuse fuse;
    size_t parts;
  };
  const Want wants[] = {
      // q_proj [320][4096], k_proj [320][1024], gate_proj [320][4096], v_proj [320][1024]
      {K2LinearId::AttnDense, 2560, 10240, model::Fuse::Concat, 4},
      // ... v_router [320][64] in v's place
      {K2LinearId::AttnSparse, 2560, 9280, model::Fuse::Concat, 4},
      {K2LinearId::OProj, 4096, 2560, model::Fuse::Single, 1},          // o_proj [512][2560]
      {K2LinearId::DenseGateUp, 2560, 12288, model::Fuse::Interleave16, 2},   // [320][6144] x 2
      {K2LinearId::DenseDown, 6144, 2560, model::Fuse::Single, 1},      // down_proj [768][2560]
  };
  for (const Want& w : wants) {
    const model::K2Linear r = d.linear(w.id);
    CHECK_EQ(r.shape.K, w.K);
    CHECK_EQ(r.shape.N, w.N);
    CHECK(r.kind == model::WeightKind::Int4 && r.fuse == w.fuse);
    CHECK_EQ(r.parts.size(), w.parts);
    CHECK_EQ(r.shape.layout, 0u);
    CHECK_EQ(r.shape.K % 64, 0u);   // gemv.cl: K whole groups, N whole 64-column work-groups
    CHECK_EQ(r.shape.N % 64, 0u);
    CHECK_EQ((r.shape.K / 64) % r.shape.S, 0u);   // S divides K/64
  }
  CHECK(d.linear(K2LinearId::AttnSparse).parts[3] == "self_attn.v_router");
  CHECK(d.linear(K2LinearId::AttnDense).parts[3] == "self_attn.v_proj");
  CHECK(d.linear(K2LinearId::AttnDense).parts[2] == "self_attn.gate_proj");
  const model::K2Linear lm = d.linear(K2LinearId::LmHead);
  CHECK(lm.kind == model::WeightKind::Bf16 && lm.shape.N == 250624u && lm.shape.K == 2560u);
  bool threw = false;
  try {
    (void)d.linear(K2LinearId::kCount);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
  CHECK_EQ(d.layer_linears(0).size(), size_t(4));
  CHECK(d.layer_linears(2).back() == K2LinearId::DenseDown);
  CHECK_EQ(d.layer_linears(3).size(), size_t(2));
  CHECK(d.layer_linears(47)[0] == K2LinearId::AttnSparse);

  // --- the expert groups ------------------------------------------------------------------
  const auto groups = d.expert_groups();
  CHECK_EQ(groups.size(), size_t(3));
  for (const model::K2ExpertGroup& g : groups) {
    if (g.id == K2ExpertId::Value) {   // v_experts.E qweight [320][1024]
      CHECK(g.K == 2560 && g.N == 1024 && g.blocks == 64 && g.shared.empty());
    } else if (g.id == K2ExpertId::MoeGateUp) {   // experts.E.{gate,up}_proj [320][768]
      CHECK(g.K == 2560 && g.N == 1536 && g.blocks == 101 && g.parts.size() == 2);
      CHECK(g.shared[0] == "mlp.shared_experts.gate_proj");
    } else {   // experts.E.down_proj [96][2560]
      CHECK(g.K == 768 && g.N == 2560 && g.blocks == 101);
      CHECK(g.shared[0] == "mlp.shared_experts.down_proj");
    }
  }
  CHECK(K2Desc::expert_part("mlp.experts.{e}.up_proj", 99) == "mlp.experts.99.up_proj");
  CHECK(K2Desc::expert_part("self_attn.v_experts.{e}", 0) == "self_attn.v_experts.0");
  CHECK(K2Desc::layer_prefix(7) == "model.layers.7.");

  // --- the small tensors: dense 2, sparse 4, every byte of each block owned once ----------
  CHECK_EQ(d.small_tensors(0).size(), size_t(2));
  const auto ss = d.small_tensors(10);
  CHECK_EQ(ss.size(), size_t(4));
  uint32_t norms = 0, route = 0;
  for (const model::K2SmallTensor& t : ss)
    (t.block == model::K2SmallBlock::Norms ? norms : route) += t.elems * 4;
  CHECK_EQ(norms, d.norms_bytes());
  // The MoE bias fills 100 of its 128 slots; the MoVA bias (F16 in the checkpoint) follows.
  CHECK_EQ(route, 100u * 4 + 64u * 4);
  CHECK_EQ(d.route_bytes(), (128u + 64u) * 4);
  CHECK_EQ(d.route_off_mova(), 512u);
  CHECK(std::string(ss[3].dtype) == "F16" && ss[3].name == "self_attn.v_router.bias");
  CHECK(std::string(ss[2].dtype) == "BF16" && ss[2].name == "mlp.gate.bias");

  // --- the bytes (loader/k2_layout.h), derived from the header shapes --------------------
  const loader::K2ExpertBytes eb = loader::k2_expert_bytes(d);
  CHECK_EQ(eb.value_block, size_t(1392640));
  CHECK_EQ(eb.gate_up_block, size_t(2088960));
  CHECK_EQ(eb.down_block, size_t(1044480));
  CHECK_EQ(eb.experts(), size_t(405606400));
  CHECK_EQ(eb.router, size_t(655360));
  // The layout-1 tiles hold exactly the checkpoint's words + scales (136 = 128 + 8 u32).
  CHECK_EQ(eb.value_block, size_t(320) * 1024 * 4 + size_t(40) * 1024 * 2);
  const loader::K2WeightBytes wb = loader::k2_weight_bytes(d, false);
  CHECK_EQ(wb.linears, size_t(952304640));
  CHECK_EQ(wb.routers, size_t(29491200));
  CHECK_EQ(wb.experts, size_t(18252288000ull));
  CHECK_EQ(wb.small, size_t(1027840));
  CHECK_EQ(wb.embed, size_t(1283194880));
  CHECK_EQ(wb.lm_head, size_t(1283194880));
  CHECK_EQ(wb.total(), size_t(21801501440ull));
  CHECK_EQ(loader::k2_weight_bytes(d, true).total(), size_t(21160906496ull));
  // Spec 18 §1: 3.78 GB read per token with the bf16 head, ~3.14 GB with the int8 one.
  const size_t per_tok = wb.linears + wb.small + wb.lm_head + 45 * eb.per_token(d);
  CHECK_EQ(per_tok, size_t(3785736960ull));
  CHECK_EQ(d.rope_table_bytes(32768), size_t(32768) * 2 * 64 * 4);

  // --- config.json: the checkpoint's accepted, every structural deviation refused --------
  const std::string cfg = slurp(argv[1]);
  const common::json::Value cv = common::json::parse(cfg);
  CHECK(model::model_type_of(cv) == "k2_horizon");
  CHECK(model::is_k2_model_type(model::model_type_of(cv)));
  CHECK(!model::is_k2_model_type("qwen3_5"));
  model::check_k2_config(d, cv);
  refused(cfg, "\"query_key_norm\": false", "\"query_key_norm\": true", "query_key_norm");
  refused(cfg, "\"num_experts\": 100", "\"num_experts\": 128", "num_experts");
  refused(cfg, "\"layernorm_num_groups\": 2", "\"layernorm_num_groups\": 1", "layernorm_num_groups");
  refused(cfg, "\"rope_head_dim\": 128", "\"rope_head_dim\": 64", "rope_head_dim");
  refused(cfg, "\"router_score_func\": \"sigmoid\"", "\"router_score_func\": \"softmax\"",
          "router_score_func");
  refused(cfg, "\"attention_gate_func\": \"softplus\"", "\"attention_gate_func\": \"silu\"",
          "attention_gate_func");
  refused(cfg, "\"router_scaling_factor\": 2.5", "\"router_scaling_factor\": 1.0",
          "router_scaling_factor");
  refused(cfg, "\"num_shared_experts\": 1", "\"num_shared_experts\": 2", "num_shared_experts");
  refused(cfg, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true",
          "tie_word_embeddings");
  refused(cfg, "\"sliding_window\": null", "\"sliding_window\": 4096", "sliding_window");
  refused(cfg, "\"model_type\": \"k2_horizon\"", "\"model_type\": \"k3\"", "model_type");
  std::puts("k2_horizon_test OK");
  return 0;
}
