// Spec 23b Task 1: model::MuseGlimmerDesc against the checkpoint's own config (spec 23 §1,
// docs/probe-muse-glimmer-2026-10-15.md). Host only.
//   argv[1] tests/model/muse_glimmer/config.json             (meta-models/Muse-Glimmer-30B @ a4e59da5, byte for byte)
//   argv[2] tests/model/muse_glimmer/generation_config.json
//   argv[3] (optional) the int4 repo's config.json (Vishva007's: dtype float16 + quantization_config) - read when it
//           exists, held to the same descriptor
// Every refusal is an in-memory edit of the vendored config and must name the key (and the layer) and the value.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/muse_glimmer.h"

namespace {

using common::json::Value;
using model::MuseGlimmerDesc;
using model::MuseLinearId;
using model::MusePlacement;

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  if (!f.good()) {
    std::fprintf(stderr, "cannot read %s\n", p.c_str());
    std::exit(1);
  }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// cfg with text_config[key] = v (v null-kind Value() deletes nothing: it sets JSON null).
Value with_text(const Value& cfg, const std::string& key, const Value& v) {
  Value::Object top = cfg.obj();
  Value::Object t = top.at("text_config").obj();
  t[key] = v;
  top["text_config"] = Value::make_obj(t);
  return Value::make_obj(top);
}
Value with_text_elem(const Value& cfg, const std::string& key, size_t i, const Value& v) {
  Value::Array a = cfg.at("text_config").at(key).arr();
  a.at(i) = v;
  return with_text(cfg, key, Value::make_arr(a));
}
Value with_top(const Value& cfg, const std::string& key, const Value& v) {
  Value::Object top = cfg.obj();
  top[key] = v;
  return Value::make_obj(top);
}

// check_muse_config(c) must throw a runtime_error naming every string of `names`.
void refused(const Value& c, const std::vector<std::string>& names, uint32_t override_layers = 0) {
  bool ok = false;
  std::string what;
  try {
    (void)model::check_muse_config(c, override_layers);
  } catch (const std::runtime_error& e) {
    what = e.what();
    ok = true;
    for (const std::string& n : names)
      if (what.find(n) == std::string::npos) ok = false;
  }
  if (!ok) {
    std::fprintf(stderr, "expected a refusal naming");
    for (const std::string& n : names) std::fprintf(stderr, " '%s'", n.c_str());
    std::fprintf(stderr, "; got: %s\n", what.empty() ? "(no throw)" : what.c_str());
  }
  CHECK(ok);
  std::printf("  refused: %s\n", what.c_str());
}

void same(const MuseGlimmerDesc& a, const MuseGlimmerDesc& b) {
  CHECK_EQ(a.layers, b.layers);
  CHECK_EQ(a.hidden, b.hidden);
  CHECK_EQ(a.q_heads, b.q_heads);
  CHECK_EQ(a.kv_heads, b.kv_heads);
  CHECK_EQ(a.head_dim, b.head_dim);
  CHECK_EQ(a.inter, b.inter);
  CHECK_EQ(a.vocab, b.vocab);
  CHECK_EQ(a.window, b.window);
  CHECK_EQ(a.max_pos, b.max_pos);
  CHECK_EQ(a.rms_eps, b.rms_eps);
  CHECK_EQ(a.post_eps, b.post_eps);
  CHECK_EQ(a.qk_scale, b.qk_scale);
  CHECK_EQ(a.rope_theta, b.rope_theta);
  CHECK_EQ(a.out_mult, b.out_mult);
  CHECK_EQ(a.softcap, b.softcap);
  CHECK_EQ(a.bos, b.bos);
  CHECK_EQ(a.pad, b.pad);
  CHECK(a.eos == b.eos);
  CHECK(a.full == b.full);
  CHECK_EQ(a.qgkv_s, b.qgkv_s);
  CHECK_EQ(a.oproj_s, b.oproj_s);
  CHECK_EQ(a.gate_up_s, b.gate_up_s);
  CHECK_EQ(a.down_s, b.down_s);
}

template <class F>
bool throws_invalid(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: muse_glimmer_test <config.json> <generation_config.json> [<int4 config.json>]\n");
    return 2;
  }
  const Value cfg = common::json::parse(slurp(argv[1]));
  const Value gen = common::json::parse(slurp(argv[2]));
  const MuseGlimmerDesc d = model::muse_glimmer();

  // --- the published model = the vendored config, field by field ----------------------------------------
  same(d, model::check_muse_config(cfg));
  same(d, model::check_muse_config(cfg.at("text_config")));   // a text config alone reads the same
  CHECK_EQ(d.layers, 52u);
  CHECK_EQ(d.full.size(), size_t(52));
  for (uint32_t l = 0; l < 52; ++l) CHECK_EQ(bool(d.full[l]), l % 4 == 3);
  CHECK_EQ(d.full_layers(), 13u);
  CHECK_EQ(d.sliding_layers(), 39u);
  CHECK_EQ(d.full_before(4), 1u);
  CHECK_EQ(d.sliding_before(4), 3u);
  CHECK_EQ(d.full_before(52), 13u);
  CHECK_EQ(d.gqa(), 16u);
  // generation_config.json's tokens are the descriptor's
  CHECK_EQ(uint32_t(gen.at("bos_token_id").num()), d.bos);
  CHECK_EQ(uint32_t(gen.at("pad_token_id").num()), d.pad);
  CHECK_EQ(gen.at("eos_token_id").arr().size(), d.eos.size());
  for (size_t i = 0; i < d.eos.size(); ++i) CHECK_EQ(uint32_t(gen.at("eos_token_id").arr()[i].num()), d.eos[i]);
  CHECK_EQ(uint32_t(gen.at("max_length").num()), d.max_pos);

  // --- the fused GEMVs: shapes and column order (Review Focus 1's names) ---------------------------------
  CHECK_EQ(d.qgkv_rows(), 8704u);
  CHECK_EQ(d.qgkv_off_gate(), 4096u);
  CHECK_EQ(d.qgkv_off_k(), 8192u);
  CHECK_EQ(d.qgkv_off_v(), 8448u);
  {
    const model::MuseLinear q = d.linear(MuseLinearId::QGKV);
    CHECK((q.parts == std::vector<std::string>{"self_attn.q_proj", "self_attn.gate_proj", "self_attn.k_proj",
                                               "self_attn.v_proj"}));
    CHECK((q.part_n == std::vector<uint32_t>{4096, 4096, 256, 256}));
    CHECK(q.shape.K == 6656 && q.shape.N == 8704 && q.shape.layout == 0);
    CHECK(q.kind == model::WeightKind::Int4 && q.fuse == model::Fuse::Concat);
    const model::MuseLinear gu = d.linear(MuseLinearId::GateUp);
    CHECK((gu.parts == std::vector<std::string>{"mlp.gate_proj", "mlp.up_proj"}));
    CHECK((gu.part_n == std::vector<uint32_t>{19968, 19968}));
    CHECK(gu.shape.K == 6656 && gu.shape.N == 39936 && gu.fuse == model::Fuse::Concat);
    const model::MuseLinear o = d.linear(MuseLinearId::OProj);
    CHECK(o.shape.K == 4096 && o.shape.N == 6656 && o.parts.size() == 1 && o.parts[0] == "self_attn.o_proj");
    const model::MuseLinear dn = d.linear(MuseLinearId::Down);
    CHECK(dn.shape.K == 19968 && dn.shape.N == 6656 && dn.parts[0] == "mlp.down_proj");
    const model::MuseLinear h = d.linear(MuseLinearId::LmHead);
    CHECK(h.shape.K == 6656 && h.shape.N == 202048 && h.kind == model::WeightKind::Bf16 && h.parts[0] == "lm_head");
    size_t params = 0;
    for (MuseLinearId id : d.layer_linears()) {
      const model::MuseLinear l = d.linear(id);
      uint32_t n = 0;
      for (uint32_t x : l.part_n) n += x;
      CHECK_EQ(n, l.shape.N);
      params += size_t(l.shape.K) * l.shape.N;
    }
    CHECK_EQ(params, size_t(483917824));   // spec 23 §2: the eight linears of a layer
    bool threw = false;
    try {
      (void)d.linear(MuseLinearId::kCount);
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw);
  }
  CHECK_EQ(MuseGlimmerDesc::layer_prefix(7), std::string("model.language_model.layers.7."));

  // --- refusals, each naming the key and its value -------------------------------------------------------
  refused(with_text(cfg, "head_dim", Value::make_num(64)), {"head_dim", "64"});
  refused(with_text(cfg, "num_key_value_heads", Value::make_num(4)), {"num_key_value_heads", "4"});
  refused(with_text(cfg, "num_attention_heads", Value::make_num(16)), {"num_attention_heads", "16"});
  refused(with_text(cfg, "hidden_size", Value::make_num(4096)), {"hidden_size", "4096"});
  refused(with_text(cfg, "intermediate_size", Value::make_num(16384)), {"intermediate_size", "16384"});
  refused(with_text(cfg, "hidden_activation", Value::make_str("gelu")), {"hidden_activation", "gelu"});
  refused(with_text(cfg, "tie_word_embeddings", Value::make_bool(true)), {"tie_word_embeddings", "true"});
  refused(with_top(cfg, "tie_word_embeddings", Value::make_bool(true)), {"tie_word_embeddings", "true"});
  refused(with_text_elem(cfg, "layer_rope_theta", 1, Value::make_num(1e6)), {"layer_rope_theta[1]", "1000000"});
  refused(with_text_elem(cfg, "layer_rope_theta", 0, Value::make_num(0)), {"layer_rope_theta[0]", "sliding"});
  refused(with_text_elem(cfg, "layer_rope_theta", 3, Value::make_num(5e5)), {"layer_rope_theta[3]", "full"});
  refused(with_text_elem(cfg, "layer_types", 2, Value::make_str("full_attention")), {"layer_types[2]", "full_attention"});
  {
    Value::Object rp = cfg.at("text_config").at("rope_parameters").obj();
    rp["rope_type"] = Value::make_str("yarn");
    refused(with_text(cfg, "rope_parameters", Value::make_obj(rp)), {"rope_type", "yarn"});
    rp = cfg.at("text_config").at("rope_parameters").obj();
    rp["rope_theta"] = Value::make_num(1e6);
    refused(with_text(cfg, "rope_parameters", Value::make_obj(rp)), {"rope_theta", "1000000"});
  }
  refused(with_text(cfg, "attention_bias", Value::make_bool(true)), {"attention_bias", "true"});
  refused(with_text(cfg, "final_logit_softcapping", Value()), {"final_logit_softcapping", "null"});
  refused(with_text(cfg, "final_logit_softcapping", Value::make_num(30)), {"final_logit_softcapping", "30"});
  refused(with_text(cfg, "vocab_size", Value::make_num(202049)), {"vocab_size", "202049"});
  refused(with_text(cfg, "sliding_window", Value::make_num(4096)), {"sliding_window", "4096"});
  refused(with_text(cfg, "max_position_embeddings", Value::make_num(262144)), {"max_position_embeddings", "262144"});
  refused(with_text(cfg, "qk_scale_factor", Value::make_num(1.0)), {"qk_scale_factor", "1"});
  refused(with_text(cfg, "post_norm_eps", Value::make_num(1e-6)), {"post_norm_eps"});
  refused(with_text(cfg, "rms_norm_eps", Value::make_num(1e-6)), {"rms_norm_eps"});
  refused(with_text(cfg, "output_multiplier", Value::make_num(1.0)), {"output_multiplier"});
  refused(with_text(cfg, "model_type", Value::make_str("gemma3_text")), {"model_type", "gemma3_text"});
  refused(with_top(cfg, "model_type", Value::make_str("gemma3")), {"model_type", "gemma3"});
  refused(with_text(cfg, "num_hidden_layers", Value::make_num(53)), {"num_hidden_layers", "53"});
  refused(cfg, {"53", "52"}, 53);   // layers_override beyond the checkpoint

  // --- a synthetic prefix and the development override -------------------------------------------------
  {
    const MuseGlimmerDesc d4 = model::check_muse_config(cfg, 4);
    CHECK_EQ(d4.layers, 4u);
    CHECK((d4.full == std::vector<bool>{false, false, false, true}));
    CHECK_EQ(d4.full_layers(), 1u);
    Value::Array lt = cfg.at("text_config").at("layer_types").arr(), th = cfg.at("text_config").at("layer_rope_theta").arr();
    lt.resize(2);
    th.resize(2);
    Value c2 = with_text(with_text(with_text(cfg, "num_hidden_layers", Value::make_num(2)), "layer_types",
                                   Value::make_arr(lt)),
                         "layer_rope_theta", Value::make_arr(th));
    const MuseGlimmerDesc d2 = model::check_muse_config(c2);
    CHECK_EQ(d2.layers, 2u);
    CHECK((d2.full == std::vector<bool>{false, false}));
    refused(with_text(c2, "num_hidden_layers", Value::make_num(3)), {"layer_types", "3"});
  }

  // --- the placement ------------------------------------------------------------------------------------
  {
    const MusePlacement one = MusePlacement::one(d), two = MusePlacement::two(d, 26);
    model::validate(one, d);
    model::validate(two, d);
    CHECK(one.count(0) == 52 && two.count(0) == 26 && two.count(1) == 26);
    CHECK(two.device_of(25) == 0 && two.device_of(26) == 1 && one.device_of(51) == 0);
    CHECK(throws_invalid([&] { model::validate(MusePlacement::two(d, 0), d); }));
    CHECK(throws_invalid([&] { model::validate(MusePlacement::two(d, 52), d); }));
    CHECK(throws_invalid([&] { model::validate(MusePlacement{3, 10, 52}, d); }));
    CHECK(throws_invalid([&] { model::validate(MusePlacement{1, 40, 52}, d); }));
    CHECK(throws_invalid([&] { model::validate(MusePlacement::one(model::check_muse_config(cfg, 4)), d); }));
  }

  CHECK(model::is_muse_model_type("muse_glimmer"));
  CHECK(model::is_muse_model_type("muse_glimmer_text"));
  CHECK(!model::is_muse_model_type("muse_glimmer_assistant"));
  CHECK(!model::is_muse_model_type("kolibri1"));

  // --- the int4 repo's config (either Vishva007 spelling), when given --------------------------------------
  if (argc > 3 && std::filesystem::exists(argv[3])) {
    const Value q = common::json::parse(slurp(argv[3]));
    same(d, model::check_muse_config(q));
    std::printf("  the int4 repo's config.json (%s) reads as the same descriptor\n", argv[3]);
  } else if (argc > 3) {
    std::printf("  (no int4 config at %s - that check skipped)\n", argv[3]);
  }

  std::printf("muse_glimmer_test OK: 52 layers (13 full at l %% 4 == 3, NoPE), q||gate||k||v 8704, gate||up 39936, "
              "483,917,824 linear parameters a layer\n");
  return 0;
}
