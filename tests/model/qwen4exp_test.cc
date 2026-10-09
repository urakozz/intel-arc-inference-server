// Spec 21b Task 2: model::Qwen4ExpDesc against the checkpoint's own facts (spec 21 §1,
// docs/probe-qwen4exp-2026-10-09.md). Host only, no argument.
//
// The original's config.json is NOT vendored (plan 21b Task 2's fallback): the checkpoint's licence is
// `qwen-community-1.0` (its card's front matter), whose text this repository has not read, so the test
// builds the same JSON in code - every structural key of Qwen/Qwen3.8-Flash-Next's config.json with its
// value, and make_synth.py's 4-layer synthetic (layers 4, ngram_vocab_size_base 1000, a
// quantization_config) the same way.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "check.h"
#include "common/json.h"
#include "model/qwen4exp.h"

namespace {

using model::Q4Form;
using model::Q4Forms;
using model::Q4LinearId;
using model::Q4Placement;
using model::Qwen4ExpDesc;

// The original's config.json (its keys and values, read 2026-10-08; vision_config elided - the engine
// skips the tower and holds none of it), at `layers` layers of the real pattern.
std::string real_config(uint32_t layers = 48, uint32_t base = 20000000, const std::string& quant = "") {
  std::string lt;
  for (uint32_t l = 0; l < layers; ++l)
    lt += std::string(l ? ", " : "") + (l % 4 == 3 ? "\"full_attention\"" : "\"linear_attention\"");
  return std::string(R"({"architectures": ["Qwen4ExpForConditionalGeneration"], "image_token_id": 248056,)") +
         R"( "language_model_only": false, "model_type": "qwen4_exp", "text_config": {"attention_bias": false,)" +
         R"( "attention_dropout": 0.0, "bos_token_id": 248044, "dtype": "bfloat16", "eos_token_id": 248044,)" +
         R"( "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320, "head_dim": 256, "heads_per_ngram": 8,)" +
         R"( "hidden_act": "silu", "hidden_size": 2560, "indexer_budget": 2048, "indexer_compress_ratio": 4,)" +
         R"( "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4, "initializer_range": 0.02,)" +
         " \"layer_types\": [" + lt + "]," +
         R"( "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,)" +
         R"( "linear_num_value_heads": 48, "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,)" +
         R"( "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",)" +
         R"( "moe_intermediate_size": 640, "mtp": {"hybrid": true, "layer_types": ["full_attention"],)" +
         R"( "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1, "rope_theta": 10000000},)" +
         R"( "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false, "ngram_size": 3,)" +
         " \"ngram_vocab_size_base\": " + std::to_string(base) + "," +
         R"( "num_attention_heads": 24, "num_experts": 512, "num_experts_per_tok": 10,)" +
         " \"num_hidden_layers\": " + std::to_string(layers) + "," +
         R"( "num_key_value_heads": 2, "output_gate_type": "sigmoid", "output_router_logits": false,)" +
         R"( "pad_token_id": null, "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,)" +
         R"( "ple_layer_ids": [2], "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": true,)" +
         R"( "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000,)" +
         R"( "rope_type": "default"}, "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 640,)" +
         R"( "split_ngram_parts": 128, "tie_word_embeddings": false, "use_cache": true, "vocab_size": 248320},)" +
         R"( "tie_word_embeddings": false, "transformers_version": "5.8.0.dev0", "video_token_id": 248057,)" +
         R"( "vision_end_token_id": 248054, "vision_start_token_id": 248053)" + quant + "}";
}
// make_synth.py --layers 4 --form ours's config: the same keys at 4 layers and the reduced PLE base.
std::string synth4_config() {
  return real_config(4, 1000,
                     R"(, "quantization_config": {"quant_method": "auto-round", "packing_format": "auto_round:auto_gptq",)"
                     R"( "bits": 4, "group_size": 64, "sym": true, "data_type": "int", "iters": 0,)"
                     R"( "block_name_to_quantize": "model.language_model.layers", "autoround_version": "0.17.0",)"
                     R"( "extra_config": {".*mlp\\.gate.*": {"bits": 16, "data_type": "float"}}})");
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
             [](const common::json::Value& c) { model::check_qwen4exp_config(model::qwen4exp(), c); });
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

int main() {
  const Qwen4ExpDesc& d = model::qwen4exp();
  // --- the published model's numbers and the derived widths -------------------------------------------
  CHECK_EQ(d.layers, 48u);
  CHECK_EQ(d.hidden, 2560u);
  CHECK_EQ(d.hc_n(), 10240u);
  CHECK_EQ(d.hc_down_n(), 324u);
  CHECK_EQ(d.hc_down_rows(), 336u);
  CHECK_EQ(d.conv_rows(), 10240u);
  CHECK_EQ(d.qkvz_n(), 16384u);
  CHECK_EQ(d.ab_n(), 96u);
  CHECK_EQ(d.q_n(), 6144u);
  CHECK_EQ(d.kv_n(), 512u);
  CHECK_EQ(d.qkvg_n(), 13312u);
  CHECK_EQ(d.idx_n(), 640u);
  CHECK_EQ(d.gqa(), 12u);
  CHECK_EQ(d.block_topk(), 512u);
  CHECK_EQ(d.max_visible(), 2051u);
  CHECK_EQ(d.router_n(), 528u);
  CHECK_EQ(d.ple_e(), 2560u);
  CHECK_EQ(d.ple_kv_n(), 12800u);
  CHECK_EQ(d.ple_ring(), 9u);
  CHECK_EQ(d.vocab, 248320u);
  CHECK_EQ(d.vocab_used, 248077u);
  CHECK_EQ(d.ple_layer, 1u);
  CHECK(!d.is_qsa(d.ple_layer));
  CHECK_EQ(d.ple_base, uint64_t(20000000));
  CHECK_EQ(d.ple_seed, uint64_t(1234));
  CHECK(d.eos.size() == 2 && d.eos[0] == 248046u && d.eos[1] == 248044u);
  CHECK_EQ(d.rope_table_bytes(1), size_t(256));
  CHECK_EQ(Qwen4ExpDesc::layer_prefix(7), std::string("model.language_model.layers.7."));
  for (uint32_t l = 0; l < 48; ++l) CHECK(d.is_qsa(l) == (l == 3 || (l > 3 && (l - 3) % 4 == 0)));
  CHECK_EQ(d.qsa_before(48), 12u);
  CHECK_EQ(d.gdn_before(48), 36u);
  CHECK_EQ(d.qsa_before(4), 1u);
  CHECK_EQ(d.gdn_before(3), 3u);

  // --- the linear table: ours (int4 dense) and Intel's (bf16 dense) --------------------------------------
  {
    const model::Q4Linear qkvz = d.linear(Q4LinearId::GdnQkvz);
    CHECK(qkvz.shape.K == 2560 && qkvz.shape.N == 16384 && qkvz.shape.S == 1 && qkvz.kind == model::WeightKind::Int4);
    CHECK(qkvz.parts.size() == 2 && qkvz.parts[0] == "linear_attn.in_proj_qkv" && qkvz.parts[1] == "linear_attn.in_proj_z");
    const model::Q4Linear qkvg = d.linear(Q4LinearId::QsaQkvg);
    CHECK(qkvg.shape.N == 13312 && qkvg.parts.size() == 3 && qkvg.parts[0] == "self_attn.q_proj");
    CHECK(d.linear(Q4LinearId::GdnOut).shape.K == 6144 && d.linear(Q4LinearId::QsaO).shape.K == 6144);
    CHECK(d.linear(Q4LinearId::GdnAb).shape.N == 128 && d.linear(Q4LinearId::GdnAb).kind == model::WeightKind::Bf16);
    CHECK(d.linear(Q4LinearId::QsaIdx).shape.N == 640 && d.linear(Q4LinearId::QsaIdx).kind == model::WeightKind::Bf16);
    CHECK(d.linear(Q4LinearId::Router).shape.N == 528 && d.linear(Q4LinearId::Router).parts[1] == "mlp.shared_expert_gate");
    CHECK(d.linear(Q4LinearId::PleKv).shape.K == 2560 && d.linear(Q4LinearId::PleKv).shape.N == 12800);
    CHECK(d.linear(Q4LinearId::LmHead).shape.N == 248320);
    Qwen4ExpDesc intel = d;
    intel.forms = {Q4Form::Bf16, Q4Form::Bf16, Q4Form::Bf16, 128};
    CHECK(intel.linear(Q4LinearId::GdnQkvz).kind == model::WeightKind::Bf16);
    CHECK(intel.linear(Q4LinearId::QsaO).kind == model::WeightKind::Bf16 && intel.linear(Q4LinearId::QsaO).shape.S == 1);
    bool threw = false;
    try {
      d.linear(Q4LinearId::kCount);
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw);
  }

  // --- the real config passes, and every structural key is held, the refusal naming it ---------------
  const std::string real = real_config();
  model::check_qwen4exp_config(d, common::json::parse(real));
  refused(real, "\"num_experts\": 512", "\"num_experts\": 256", "num_experts");
  refused(real, "\"num_experts_per_tok\": 10", "\"num_experts_per_tok\": 8", "num_experts_per_tok");
  refused(real, "\"moe_intermediate_size\": 640", "\"norm_topk_prob\": false, \"moe_intermediate_size\": 640",
          "norm_topk_prob");
  refused(real, "\"output_gate_type\": \"sigmoid\"", "\"output_gate_type\": \"silu\"", "output_gate_type");
  refused(real, "\"hc_count\": 4", "\"hc_count\": 2", "hc_count");
  refused(real, "\"indexer_budget\": 2048", "\"indexer_budget\": 1024", "indexer_budget");
  refused(real, "\"indexer_kv_heads\": 1", "\"indexer_kv_heads\": 2", "indexer_kv_heads");
  refused(real, "\"ple_layer_ids\": [2]", "\"ple_layer_ids\": [3]", "ple_layer_ids");
  refused(real, "\"linear_attention\", \"full_attention\"", "\"linear_attention\", \"linear_attention\"", "layer_types");
  refused(real, "\"partial_rotary_factor\": 0.25, \"ple_conv", "\"partial_rotary_factor\": 0.5, \"ple_conv",
          "partial_rotary_factor");
  refused(real, "\"rope_theta\": 10000000, \"rope_type\"", "\"rope_theta\": 1000000, \"rope_type\"", "rope_theta");
  refused(real, "\"split_ngram_parts\": 128, \"tie_word_embeddings\": false",
          "\"split_ngram_parts\": 128, \"tie_word_embeddings\": true", "tie_word_embeddings");
  refused(real, "\"transformers_version\"", "\"tie_word_embeddings\": true, \"transformers_version\"",
          "tie_word_embeddings");
  refused(real, "\"head_dim\": 256", "\"head_dim\": 128", "head_dim");
  refused(real, "\"num_hidden_layers\": 48", "\"num_hidden_layers\": 49", "num_hidden_layers");
  refused(real, "\"linear_num_value_heads\": 48", "\"linear_num_value_heads\": 32", "linear_num_value_heads");
  refused(real, "\"split_ngram_parts\"", "\"seed\": 99, \"split_ngram_parts\"", "seed");   // the descriptor's 1234

  // --- the descriptor from a config: layers and the PLE base read, the rest held -----------------------
  {
    const Qwen4ExpDesc s4 = model::qwen4exp_desc(common::json::parse(synth4_config()), Q4Forms{});
    CHECK_EQ(s4.layers, 4u);
    CHECK_EQ(s4.ple_base, uint64_t(1000));
    CHECK_EQ(s4.qsa_before(4), 1u);
    CHECK(s4.forms.dense == Q4Form::Int4 && s4.forms.expert_group == 64);
    const Qwen4ExpDesc r = model::qwen4exp_desc(common::json::parse(real), Q4Forms{Q4Form::Bf16, Q4Form::Bf16,
                                                                                   Q4Form::Bf16, 128});
    CHECK(r.layers == 48 && r.ple_base == 20000000 && r.forms.expert_group == 128);
    // A seed in the config is read (and held by the check); a synthetic with N = 1 drops the PLE layer.
    const Qwen4ExpDesc sd = model::qwen4exp_desc(
        common::json::parse(replaced(synth4_config(), "\"split_ngram_parts\"", "\"seed\": 99, \"split_ngram_parts\"")),
        Q4Forms{});
    CHECK_EQ(sd.ple_seed, uint64_t(99));
    refused_by(real_config(1, 1000), "num_hidden_layers",
               [](const common::json::Value& c) { model::qwen4exp_desc(c, Q4Forms{}); });
    refused_by(replaced(real, "\"model_type\": \"qwen4_exp\"", "\"model_type\": \"qwen3_5_moe\""), "model_type",
               [](const common::json::Value& c) { model::qwen4exp_desc(c, Q4Forms{}); });
    CHECK(model::is_qwen4exp_model_type("qwen4_exp") && model::is_qwen4exp_model_type("qwen4_exp_text") &&
          !model::is_qwen4exp_model_type("qwen3_5_moe"));
  }

  // --- placements (Kolibri's rules) -----------------------------------------------------------------
  {
    const Q4Placement one = Q4Placement::one(d), two = Q4Placement::two(d, 24);
    model::validate(one, d);
    model::validate(two, d);
    CHECK(one.devices == 1 && one.first(0) == 0 && one.end(0) == 48 && one.count(0) == 48 && one.device_of(47) == 0);
    CHECK(two.device_of(23) == 0 && two.device_of(24) == 1 && two.first(1) == 24 && two.end(1) == 48);
    CHECK(two.count(0) == 24 && two.count(1) == 24);
    CHECK(throws_invalid([&] { model::validate(Q4Placement::two(d, 0), d); }));
    CHECK(throws_invalid([&] { model::validate(Q4Placement::two(d, 48), d); }));
    CHECK(throws_invalid([&] { model::validate(Q4Placement{1, 24, 48}, d); }));
    CHECK(throws_invalid([&] { model::validate(Q4Placement{3, 24, 48}, d); }));
    Qwen4ExpDesc s4 = d;
    s4.layers = 4;
    CHECK(throws_invalid([&] { model::validate(one, s4); }));   // placed 48, the descriptor has 4
  }
  std::printf("qwen4exp_test OK: 48 layers (36 GDN / 12 QSA), hidden 2560 x 4 HC streams, 512 experts top-10, "
              "router 528, PLE 16 x 160 at layer 1, every structural key refused by name\n");
  return 0;
}
