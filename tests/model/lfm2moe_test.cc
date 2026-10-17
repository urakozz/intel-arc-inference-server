// Spec 24b Task 2: model::Lfm2MoeDesc against the checkpoint's own facts (spec 24 §1, §12 "24a as built").
// Host only. argv[1] = tests/model/lfm2moe/config.json (LiquidAI/LFM2.5-24B-A2B@1769516a's, copied unchanged -
// decision 9), argv[2] = tests/model/lfm2moe/synth4.json (make_synth.py's config at 4 layers).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "common/json.h"
#include "model/lfm2moe.h"

namespace {

using model::Lfm2MoeDesc;
using model::LfmKind;
using model::LfmLinearId;
using model::LfmPlacement;

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

std::string replaced(const std::string& cfg, const std::string& from, const std::string& to) {
  const size_t at = cfg.find(from);
  if (at == std::string::npos) {
    std::fprintf(stderr, "fixture has no '%s'\n", from.c_str());
    std::exit(1);
  }
  std::string bad = cfg;
  bad.replace(at, from.size(), to);
  return bad;
}

template <class F>
void refused_by(const std::string& cfg, const char* key, F fn) {
  bool threw = false;
  try {
    fn(common::json::parse(cfg));
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  if (!threw) std::fprintf(stderr, "not refused: %s\n", key);
  CHECK(threw);
}
void refused(const std::string& cfg, const std::string& from, const std::string& to, const char* key) {
  refused_by(replaced(cfg, from, to), key, [](const common::json::Value& c) { model::check_lfm2moe_config(c); });
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 2);
  const Lfm2MoeDesc& d = model::lfm2moe();

  // --- the model (spec 24 §1) -----------------------------------------------------------------
  CHECK(d.model_type == "lfm2_moe" && d.architecture == "Lfm2MoeForCausalLM" && d.prefix == "model.");
  CHECK_EQ(d.layers, 40u);
  CHECK_EQ(d.hidden, 2048u);
  CHECK(d.q_heads == 32 && d.kv_heads == 8 && d.head_dim == 64);
  CHECK_EQ(d.gqa(), 4u);
  CHECK_EQ(d.q_n(), 2048u);
  CHECK_EQ(d.kv_n(), 512u);
  CHECK_EQ(d.qkv_n(), 3072u);
  CHECK(d.experts == 64 && d.top_k == 4 && d.moe_inter == 1536);
  CHECK(d.dense_layers == 2 && d.dense_inter == 11776);
  CHECK(d.vocab == 128000 && d.vocab_used == 125017);
  CHECK_EQ(d.trained_max_len, 131072u);
  CHECK(d.rope_theta == 1e7 && d.norm_eps == 1e-5f);
  CHECK(d.bos == 124894 && d.eos == 124900 && d.pad == 124893);
  CHECK_EQ(d.conv_taps, 3u);

  // --- the pattern: full_attention exactly at l % 4 == 2; layer 39 conv -----------------------
  for (uint32_t l = 0; l < 40; ++l) CHECK(d.is_attention(l) == (l % 4 == 2));
  CHECK(!d.is_attention(39));
  CHECK_EQ(d.attn_layers(), 10u);
  CHECK_EQ(d.conv_layers(), 30u);
  CHECK_EQ(d.moe_layers(), 38u);
  CHECK(d.is_dense(0) && d.is_dense(1) && !d.is_dense(2) && !d.is_dense(39));
  CHECK_EQ(d.attn_index(2), 0u);
  CHECK_EQ(d.attn_index(6), 1u);
  CHECK_EQ(d.conv_index(2), 2u);
  CHECK(d.layer_prefix(5) == "model.layers.5.");

  // --- the linears (real widths) ---------------------------------------------------------------
  {
    // layer 0: conv + dense (is_dense true, is_attention false)
    const std::vector<model::LfmLinear> l0 = d.linears(0);
    CHECK_EQ(l0.size(), 4u);
    CHECK(l0[0].id == LfmLinearId::InProj && l0[0].shape.K == 2048 && l0[0].shape.N == 6144);
    CHECK(l0[1].id == LfmLinearId::OutProj && l0[1].shape.K == 2048 && l0[1].shape.N == 2048);
    CHECK(l0[2].id == LfmLinearId::DenseGateUp && l0[2].shape.K == 2048 && l0[2].shape.N == 23552);
    CHECK(l0[2].fuse == model::Fuse::Interleave16 && l0[2].parts.size() == 2);
    CHECK(l0[3].id == LfmLinearId::DenseDown && l0[3].shape.K == 11776 && l0[3].shape.N == 2048);
    // layer 2: the first attention + MoE layer
    const std::vector<model::LfmLinear> l2 = d.linears(2);
    CHECK_EQ(l2.size(), 3u);
    CHECK(l2[0].id == LfmLinearId::Qkv && l2[0].shape.K == 2048 && l2[0].shape.N == 3072);
    CHECK(l2[0].fuse == model::Fuse::Concat && l2[0].parts.size() == 3);
    CHECK(l2[1].id == LfmLinearId::OProj && l2[1].shape.K == 2048 && l2[1].shape.N == 2048);
    CHECK(l2[2].id == LfmLinearId::Router && l2[2].shape.K == 2048 && l2[2].shape.N == 64);
    CHECK(l2[2].kind == model::WeightKind::Bf16);
    // layer 3: conv + MoE
    const std::vector<model::LfmLinear> l3 = d.linears(3);
    CHECK_EQ(l3.size(), 3u);
    CHECK(l3[0].id == LfmLinearId::InProj && l3[2].id == LfmLinearId::Router);
  }

  // --- config.json: the real one passes, every structural change is refused by name ------------
  const std::string cfg = slurp(argv[1]);
  model::check_lfm2moe_config(common::json::parse(cfg));
  CHECK(model::is_lfm2moe(common::json::parse(cfg)));
  refused(cfg, "\"model_type\": \"lfm2_moe\"", "\"model_type\": \"lfm2\"", "model_type");
  refused(cfg, "\"conv_bias\": false", "\"conv_bias\": true", "conv_bias");
  refused(cfg, "\"use_expert_bias\": true", "\"use_expert_bias\": false", "use_expert_bias");
  refused(cfg, "\"norm_topk_prob\": true", "\"norm_topk_prob\": false", "norm_topk_prob");
  refused(cfg, "\"routed_scaling_factor\": 1.0", "\"routed_scaling_factor\": 2.0", "routed_scaling_factor");
  refused(cfg, "\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", "rope_type");
  refused(cfg, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false", "tie_word_embeddings");
  refused(cfg, "\"hidden_size\": 2048", "\"hidden_size\": 4096", "hidden_size");
  refused(cfg, "\"num_attention_heads\": 32", "\"num_attention_heads\": 16", "num_attention_heads");
  refused(cfg, "\"num_experts\": 64", "\"num_experts\": 32", "num_experts");
  refused(cfg, "\"num_experts_per_tok\": 4", "\"num_experts_per_tok\": 8", "num_experts_per_tok");
  refused(cfg, "\"intermediate_size\": 11776", "\"intermediate_size\": 8192", "intermediate_size");
  refused(cfg, "\"moe_intermediate_size\": 1536", "\"moe_intermediate_size\": 2048", "moe_intermediate_size");
  refused(cfg, "\"vocab_size\": 128000", "\"vocab_size\": 64000", "vocab_size");
  refused(cfg, "\"num_dense_layers\": 2", "\"num_dense_layers\": 1", "num_dense_layers");
  refused(cfg, "\"num_hidden_layers\": 40", "\"num_hidden_layers\": 41", "num_hidden_layers");
  refused(cfg, "\"full_attention\",\n    \"conv\"\n  ],", "\"conv\",\n    \"conv\"\n  ],", "layer_types");

  // --- a synthetic prefix (make_synth.py --layers 4) --------------------------------------------
  const std::string s4 = slurp(argv[2]);
  const Lfm2MoeDesc d4 = model::lfm2moe_desc(common::json::parse(s4), 1234);
  CHECK(d4.layers == 4 && d4.hidden == 2048 && d4.experts == 64);
  CHECK(d4.vocab_used == 1234);   // from the tokenizer, not config.json
  CHECK(d4.is_attention(2) && !d4.is_attention(3) && d4.is_dense(0) && d4.is_dense(1) && !d4.is_dense(2));
  CHECK_EQ(d4.attn_layers(), 1u);
  CHECK_EQ(model::lfm2moe_desc(common::json::parse(cfg), 125017).layers, 40u);
  refused_by(replaced(s4, "\"full_attention\"", "\"conv\""), "layer_types",
             [](const common::json::Value& c) { (void)model::lfm2moe_desc(c, 0); });
  refused_by(replaced(s4, "\"num_hidden_layers\": 4", "\"num_hidden_layers\": 0"), "num_hidden_layers",
             [](const common::json::Value& c) { (void)model::lfm2moe_desc(c, 0); });

  // --- placement -----------------------------------------------------------------------------
  {
    const LfmPlacement p{2, 20, 40};
    model::validate(p, d);
    CHECK(p.device_of(19) == 0 && p.device_of(20) == 1);
    CHECK(p.first(1) == 20 && p.end(1) == 40 && p.end(0) == 20 && p.count(1) == 20);
    const LfmPlacement one{1, 40, 40};
    model::validate(one, d);
    CHECK(one.end(0) == 40 && one.device_of(39) == 0);
    CHECK(LfmPlacement::one(d4).end(0) == 4 && LfmPlacement::two(d4, 2).first(1) == 2);
    for (uint32_t s : {0u, 40u}) {
      bool threw = false;
      try {
        model::validate(LfmPlacement{2, s, 40}, d);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    }
  }

  std::printf("lfm2moe_test: ok\n");
  return 0;
}
