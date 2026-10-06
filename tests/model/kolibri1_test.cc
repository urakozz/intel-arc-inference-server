// Spec 20c Task 2: model::Kolibri1Desc against the checkpoint's own facts (spec 20 §1,
// docs/probe-kolibri-2026-10-05.md). Host only. argv[1] = tests/model/kolibri1/config.json (the
// checkpoint's, = tools/oracle/third_party/kolibri1/config.json), argv[2] = tests/model/kolibri1/
// synth5_bf16attn.json (make_synth.py's config at 5 layers, bf16 attention).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "common/json.h"
#include "model/kolibri1.h"

namespace {

using model::KolAttnForm;
using model::Kolibri1Desc;
using model::KolLinearId;
using model::KolPlacement;

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

// `fn(config)` must throw a runtime_error whose message names `key`.
template <class F>
void refused_by(const std::string& cfg, const char* key, F fn) {
  bool threw = false;
  try {
    fn(common::json::parse(cfg));
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  if (!threw) std::fprintf(stderr, "not refused: %s\n", key);
  CHECK(threw);
}
void refused(const std::string& cfg, const std::string& from, const std::string& to, const char* key) {
  refused_by(replaced(cfg, from, to), key,
             [](const common::json::Value& c) { model::check_kolibri1_config(model::kolibri1(), c); });
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 2);
  const Kolibri1Desc& d = model::kolibri1();

  // --- the model (spec 20 §1) ---------------------------------------------------------------
  CHECK(d.model_type == "kolibri1" && d.architecture == "Kolibri1ForCausalLM");
  CHECK_EQ(d.layers, 50u);
  CHECK_EQ(d.hidden, 2560u);
  CHECK(d.q_heads == 48 && d.kv_heads == 4 && d.head_dim == 128);
  CHECK_EQ(d.gqa(), 12u);
  CHECK_EQ(d.q_n(), 6144u);
  CHECK_EQ(d.kv_n(), 512u);
  CHECK_EQ(d.qkv_n(), 7168u);
  CHECK(d.experts == 384 && d.top_k == 6 && d.moe_inter == 512 && d.shared_inter == 512);
  CHECK(d.vocab == 128000 && d.vocab_used == 128000);
  CHECK_EQ(d.window, 513u);
  CHECK(d.eos.size() == 2 && d.eos[0] == 127906 && d.eos[1] == 127901);
  CHECK(d.rope_theta == 1e4 && d.rms_eps == 1e-6f);
  CHECK_EQ(d.router_n(), 512u);
  CHECK_EQ(d.route_epl(), 2u);
  CHECK_EQ(d.trained_max_len, 262144u);
  CHECK(d.attn == KolAttnForm::Int4);
  CHECK_EQ(d.rope_table_bytes(262144), size_t(262144) * 512);

  // --- the pattern: full (NoPE) exactly at 4, 9, ..., 49 -------------------------------------
  for (uint32_t l = 0; l < 50; ++l) CHECK(d.is_sliding(l) == (l % 5 != 4));
  CHECK_EQ(d.full_before(50), 10u);
  CHECK_EQ(d.sliding_before(50), 40u);
  CHECK(d.full_before(4) == 0 && d.full_before(5) == 1 && d.sliding_before(5) == 4);

  // --- the linears ----------------------------------------------------------------------------
  {
    const model::KolLinear q = d.linear(KolLinearId::Qkv), o = d.linear(KolLinearId::OProj);
    CHECK(q.shape.K == 2560 && q.shape.N == 7168 && q.shape.S == 2 && q.shape.layout == 0);
    CHECK(q.kind == model::WeightKind::Int4 && q.fuse == model::Fuse::Concat && q.parts.size() == 3);
    CHECK(o.shape.K == 6144 && o.shape.N == 2560 && o.shape.S == 4 && o.kind == model::WeightKind::Int4);
    Kolibri1Desc b = d;
    b.attn = KolAttnForm::Bf16;
    CHECK(b.linear(KolLinearId::Qkv).kind == model::WeightKind::Bf16 && b.linear(KolLinearId::Qkv).shape.S == 1);
    CHECK(b.linear(KolLinearId::OProj).shape.S == 1);
    CHECK(d.linear(KolLinearId::LmHead).shape.N == 128000);
    bool threw = false;
    try {
      (void)d.linear(KolLinearId::kCount);
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw);
  }

  // --- config.json: the real one passes, every structural change is refused by name -----------
  const std::string cfg = slurp(argv[1]);
  model::check_kolibri1_config(d, common::json::parse(cfg));
  refused(cfg, "\"norm_topk_prob\": false", "\"norm_topk_prob\": true", "norm_topk_prob");
  refused(cfg, "\"num_experts\": 384", "\"num_experts\": 256", "num_experts");
  refused(cfg, "\"sliding_window\": 513", "\"sliding_window\": 512", "sliding_window");
  refused(cfg, "\"head_dim\": 128", "\"head_dim\": 64", "head_dim");
  refused(cfg, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true", "tie_word_embeddings");
  refused(cfg, "\"sliding_attention\",\n    \"sliding_attention\",\n    \"sliding_attention\",\n    \"sliding_attention\"",
          "\"sliding_attention\",\n    \"sliding_attention\",\n    \"sliding_attention\",\n    \"full_attention\"",
          "layer_types");
  refused(cfg, "\"rope_theta\": 10000.0", "\"rope_parameters\": {\"rope_type\": \"yarn\", \"rope_theta\": 10000.0}",
          "rope_type");
  refused(cfg, "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", "hidden_act");
  refused(cfg, "\"num_hidden_layers\": 50", "\"num_hidden_layers\": 51", "num_hidden_layers");

  // --- a synthetic prefix ------------------------------------------------------------------------
  const std::string s5 = slurp(argv[2]);
  const Kolibri1Desc d5 = model::kolibri1_desc(common::json::parse(s5), KolAttnForm::Bf16);
  CHECK(d5.layers == 5 && d5.attn == KolAttnForm::Bf16 && d5.hidden == 2560 && d5.experts == 384);
  CHECK(d5.full_before(5) == 1 && !d5.is_sliding(4));
  CHECK(model::kolibri1_desc(common::json::parse(cfg), KolAttnForm::Int4).layers == 50);
  refused_by(replaced(s5, "\"full_attention\"", "\"sliding_attention\""), "layer_types",
             [](const common::json::Value& c) { (void)model::kolibri1_desc(c, KolAttnForm::Bf16); });
  refused_by(replaced(s5, "\"num_hidden_layers\": 5", "\"num_hidden_layers\": 0"), "num_hidden_layers",
             [](const common::json::Value& c) { (void)model::kolibri1_desc(c, KolAttnForm::Bf16); });

  // --- placement ---------------------------------------------------------------------------------
  {
    const KolPlacement p{2, 25, 50};
    model::validate(p, d);
    CHECK(p.device_of(24) == 0 && p.device_of(25) == 1);
    CHECK(p.first(1) == 25 && p.end(1) == 50 && p.end(0) == 25 && p.count(1) == 25);
    const KolPlacement one{1, 50, 50};
    model::validate(one, d);
    CHECK(one.end(0) == 50 && one.device_of(49) == 0);
    CHECK(KolPlacement::one(d5).end(0) == 5 && KolPlacement::two(d5, 3).first(1) == 3);
    for (uint32_t s : {0u, 50u}) {
      bool threw = false;
      try {
        model::validate(KolPlacement{2, s, 50}, d);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    }
    bool threw = false;
    try {
      model::validate(KolPlacement{1, 49, 50}, d);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  std::puts("kolibri1_test OK");
  return 0;
}
