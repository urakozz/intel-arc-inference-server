#include "model/inkling2.h"

#include <cmath>
#include <stdexcept>
#include <string_view>
#include "common/expert_fp_config.h"

namespace model {
namespace {
using J = common::json::Value;
using D = Inkling2Desc;

[[noreturn]] void unsupported(std::string_view key, const std::string& expected) {
  throw std::runtime_error("Inkling 2 Mini: " + std::string(key) + " must be " + expected +
                           ". Use the fixed-loop Inkling 2 Mini checkpoint.");
}

void expect(const J& j, std::initializer_list<std::pair<const char*, uint32_t>> fields) {
  for (const auto& [key, value] : fields)
    if (j.at(key).num() != value) unsupported(key, std::to_string(value));
}

void expect_bool(const J& j, const char* key, bool value) {
  if (j.at(key).boolean() != value) unsupported(key, value ? "true" : "false");
}

float positive(const J& j, const char* key) {
  const float n = float(j.at(key).num());
  if (!std::isfinite(n) || n <= 0) unsupported(key, "finite and positive");
  return n;
}

}  // namespace

Inkling2Desc inkling2_desc_from_config(const J& config, const J& quant) {
  if (!is_inkling2_model_type(config.at("model_type").str())) unsupported("model_type", "inkling2_loop");
  const auto& t = config.at("text_config");
  if (t.at("model_type").str() != "inkling2_loop_text") unsupported("text_config.model_type", "inkling2_loop_text");
  const auto& loop = t.at("loop_config");
  expect_bool(loop, "is_varloop", false);
  expect(loop, {{"core_start", D::core_start}, {"core_end", D::core_end}, {"num_passes", D::passes},
                {"gate_rank", D::controller_rank}});
  expect(t, {{"num_hidden_layers", D::layers}, {"hidden_size", D::hidden}, {"num_attention_heads", D::q_heads},
             {"num_key_value_heads", D::kv_heads}, {"head_dim", D::head_dim}, {"swa_head_dim", D::head_dim},
             {"swa_num_attention_heads", D::q_heads}, {"swa_num_key_value_heads", D::kv_heads},
             {"d_rel", D::relative_dim}, {"rel_extent", D::relative_extent}, {"sliding_window_size", D::window},
             {"sconv_kernel_size", D::conv_taps}, {"n_layers_per_attn_res_block", D::block_layers},
             {"n_routed_experts", D::experts}, {"num_experts_per_tok", D::top_k}, {"n_shared_experts", 0},
             {"intermediate_size", D::inter}, {"route_scale", D::routed_scale}, {"vocab_size", D::vocab},
             {"unpadded_vocab_size", D::vocab_used}, {"model_max_length", D::trained_max_len},
             {"mlp_gate_softcap", uint32_t(D::gate_cap)}, {"mlp_up_softcap", uint32_t(D::up_cap)},
             {"final_logit_softcapping", uint32_t(D::logit_cap)}, {"log_scaling_n_floor", 8192}});
  expect_bool(t, "q_bias", false);
  expect_bool(t, "o_bias", false);
  expect_bool(t, "use_value_norm_no_gamma", true);
  expect_bool(t, "attn_res_normalize_values", true);
  expect_bool(t, "use_gate_bias", true);
  if (const auto* score = t.find("scoring_func"); score && score->str() != "sigmoid")
    unsupported("scoring_func", "sigmoid");
  D d;
  d.eps = positive(t, "rms_norm_eps");
  d.qk_clip = positive(t, "qk_l2_clip_threshold");
  d.relative_clip = positive(t, "rel_l2_clip_threshold");
  d.log_alpha = positive(t, "log_scaling_alpha");
  d.log_floor = uint32_t(t.at("log_scaling_n_floor").num());
  const auto eos = config.at("eos_token_id").num();
  if (!(eos >= 0 && eos < D::vocab_used && std::floor(eos) == eos)) unsupported("eos_token_id", "a live token ID");
  d.eos = uint32_t(eos);
  for (const auto& val : t.at("local_layer_ids").arr()) {
    const auto n = val.num();
    if (!(n >= 0 && n < D::layers && std::floor(n) == n) || d.local[uint32_t(n)])
      unsupported("local_layer_ids", "distinct layer indices");
    d.local[uint32_t(n)] = true;
  }
  const auto formats = common::expert_fp_formats(quant);
  for (uint32_t l = 0; l < D::layers; ++l) {
    if (d.local[l] != ((l + 1) % D::block_layers != 0))
      unsupported("local_layer_ids", "sliding attention except layers 5, 11, 17 and 23");
    const auto prefix = inkling2_layer_prefix(l) + "mlp.experts.";
    const auto up = formats.find(prefix + "w13_weight"), down = formats.find(prefix + "w2_weight");
    if (up == formats.end() || down == formats.end() || up->second != down->second)
      unsupported(prefix + "weight_transform", "matching NVFP4 or MXFP8 formats");
    d.expert_format[l] = up->second;
  }
  auto add = [&](uint32_t first, uint32_t last, uint32_t pass, bool controller) {
    for (uint32_t l = first; l < last; ++l) {
      auto& slot = d.local[l] ? d.window_visits : d.full_visits;
      d.visits.push_back({l, pass, d.local[l] ? D::window : 0, slot++, uint32_t(d.visits.size()),
                          controller, l % D::block_layers == 0});
    }
  };
  add(0, D::core_start, 0, false);
  for (uint32_t p = 0; p < D::passes; ++p) add(D::core_start, D::core_end, p, p != 0);
  add(D::core_end, D::layers, D::passes - 1, false);

  const auto& m = config.at("mtp_config");
  expect(m, {{"num_hidden_layers", D::draft_layers}, {"hidden_size", D::hidden}, {"num_attention_heads", D::q_heads},
             {"num_key_value_heads", D::kv_heads}, {"head_dim", D::head_dim}, {"intermediate_size", D::draft_inter},
             {"sliding_window_size", D::window}, {"markov_rank", D::markov_rank}, {"block_size", 8}, {"rope_theta", 1000000}});
  expect_bool(m, "q_bias", false);
  expect_bool(m, "o_bias", false);
  if (m.at("markov_head_type").str() != "rnn_hidden") unsupported("mtp_config.markov_head_type", "rnn_hidden");
  if (m.find("max_position_embeddings")) expect(m, {{"max_position_embeddings", D::trained_max_len}});
  auto& draft = d.draft;
  draft.form = DFlashForm::InklingDSpark;
  draft.architecture = "Inkling 2 Mini DSpark";
  draft.version = 1;
  draft.layers = D::draft_layers;
  draft.hidden = D::hidden;
  draft.q_heads = D::q_heads;
  draft.kv_heads = D::kv_heads;
  draft.head_dim = D::head_dim;
  draft.inter = D::draft_inter;
  draft.vocab = D::head_rows;
  draft.num_target_layers = D::visit_count;
  draft.window = {D::window, 0, D::window};
  draft.causal = {false, false, false};
  draft.block = 8;
  draft.sel_rank = D::markov_rank;
  draft.softcap = D::logit_cap;
  draft.theta = m.at("rope_theta").num();
  draft.eps = positive(m, "rms_norm_eps");
  if (draft.eps != 1e-6f) unsupported("mtp_config.rms_norm_eps", "1e-6");
  const auto& local = m.at("local_layer_ids").arr();
  if (local.size() != 2 || local[0].num() != 0 || local[1].num() != 2)
    unsupported("mtp_config.local_layer_ids", "[0, 2]");
  const auto& taps = m.at("target_layer_ids").arr();
  constexpr std::array<uint32_t, 4> expected{5, 23, 41, 48};
  if (taps.size() != expected.size()) unsupported("mtp_config.target_layer_ids", "[5, 23, 41, 48]");
  for (size_t i = 0; i < expected.size(); ++i) {
    if (taps[i].num() != expected[i]) unsupported("mtp_config.target_layer_ids", "[5, 23, 41, 48]");
    draft.taps.push_back(expected[i]);
  }
  return d;
}

}  // namespace model
