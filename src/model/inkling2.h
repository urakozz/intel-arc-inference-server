#pragma once
#include <array>
#include <string>
#include <vector>

#include "common/expert_fp.h"
#include "common/json.h"
#include "model/dflash.h"

namespace model {

inline constexpr char kInkling2Prefix[] = "model.llm.";
inline std::string inkling2_layer_prefix(uint32_t layer) {
  return std::string(kInkling2Prefix) + "layers." + std::to_string(layer) + ".";
}

struct Inkling2Visit {
  uint32_t layer, pass, window, kv_slot, conv_slot;
  bool controller, block_boundary;
};

struct Inkling2Desc {
  // The fixed release geometry. The parser holds the config to the compiled
  // rows before the loader or the planner uses any of these widths.
  static constexpr uint32_t layers = 24, hidden = 2304, q_heads = 24, kv_heads = 8, head_dim = 128;
  static constexpr uint32_t q_n = q_heads * head_dim, kv_n = kv_heads * head_dim;
  static constexpr uint32_t window = 1024, relative_dim = 16, relative_extent = 1024, conv_taps = 4;
  static constexpr uint32_t experts = 128, top_k = 8, inter = 1152, controller_rank = 64;
  static constexpr uint32_t routed_scale = 8, vocab = 201024, vocab_used = 200058;
  static constexpr uint32_t head_rows = (vocab_used + 63) / 64 * 64;
  static constexpr uint32_t core_start = 6, core_end = 18, passes = 3, block_layers = 6;
  static constexpr uint32_t visit_count = layers + (passes - 1) * (core_end - core_start);
  static constexpr uint32_t trained_max_len = 1048576, draft_layers = 3, draft_inter = 9216, markov_rank = 256;
  static constexpr float gate_cap = 4.f, up_cap = 25.f, logit_cap = 30.f;
  float eps = 0, qk_clip = 0, relative_clip = 0, log_alpha = 0;
  uint32_t log_floor = 0, eos = 0, full_visits = 0, window_visits = 0;
  std::array<bool, layers> local{};
  std::array<common::ExpertFpFormat, layers> expert_format{};
  std::vector<Inkling2Visit> visits;
  DFlashDesc draft;
};

Inkling2Desc inkling2_desc_from_config(const common::json::Value& config, const common::json::Value& quant);

}  // namespace model
