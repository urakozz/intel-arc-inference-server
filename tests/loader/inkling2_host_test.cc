#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "cli/inkling2_decode.h"
#include "loader/inkling2_loader.h"
#include "runtime/inkling2/inkling2_sizes.h"

int main(int argc, char** argv) {
  CHECK_EQ(argc, 2);
  CHECK(model::is_inkling2_model_type("inkling2_loop"));
  CHECK(!model::is_inkling2_model_type("inkling2"));
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  CHECK(checkpoint.has_drafter());
  const auto& d = checkpoint.desc();
  CHECK_EQ(d.visits.size(), 48u);
  std::vector<uint32_t> global;
  for (uint32_t v = 0; v < d.visits.size(); ++v) {
    const auto& visit = d.visits[v];
    CHECK_EQ(visit.conv_slot, v);
    if (!visit.window) global.push_back(v);
    CHECK_EQ(visit.controller, v >= 18 && v < 42);
  }
  CHECK_EQ(global, (std::vector<uint32_t>{5, 11, 17, 23, 29, 35, 41, 47}));
  CHECK(d.draft.form == model::DFlashForm::InklingDSpark);
  CHECK_EQ(d.draft.causal, (std::vector<bool>{false, false, false}));
  CHECK_EQ(d.draft.taps, (std::vector<uint32_t>{5, 23, 41, 48}));
  CHECK_EQ(d.draft.window, (std::vector<uint32_t>{1024, 0, 1024}));
  CHECK_EQ(std::count(d.expert_format.begin(), d.expert_format.end(), common::ExpertFpFormat::Nvfp4), 20);
  CHECK_EQ(std::count(d.expert_format.begin(), d.expert_format.end(), common::ExpertFpFormat::Mxfp8), 4);

  namespace ink = runtime::inkling2;
  const auto& weights = checkpoint.layout();
  const auto& head = weights.weights.at("model.llm.unembed.weight");
  CHECK_EQ(head.N, d.head_rows);
  CHECK_EQ(head.K, d.hidden);
  CHECK_EQ(head.bytes, size_t(d.head_rows) * d.hidden);
  for (const auto& [name, w] : weights.weights) {
    (void)name;
    CHECK(w.offset % 64 == 0 && w.offset + w.bytes <= weights.bytes[w.arena]);
    if (w.scales) CHECK(w.scales % 64 == 0 && w.scales + w.N * sizeof(float) <= weights.bytes[w.arena]);
  }
  CHECK_EQ(cli::inkling2::plan(argv[1], cli::parse_max_len("auto"), "auto", false), 0);
  CHECK_EQ(cli::inkling2::plan(argv[1], cli::parse_max_len("auto"), "auto", true), 0);
  const auto s = ink::sizes(d, weights, 4096, true);
  CHECK_EQ(s.persistent.full_k, size_t(8) * 4096 * 1024 * 2);
  CHECK_EQ(s.persistent.ring_k, size_t(40) * ink::kRing * 1024 * 2);
  CHECK_EQ(s.persistent.conv_k, size_t(48) * 3 * 1024 * 4);
  CHECK_EQ(s.persistent.conv_mlp, size_t(48) * 3 * 2304 * 4);
  CHECK_EQ(s.draft.draft_full_k, size_t(4096) * 1024 * 2);
  CHECK_EQ(s.draft.draft_ring_k, size_t(2) * ink::kDraftRing * 1024 * 2);
  CHECK_EQ(s.weights.layers + s.weights.model + s.weights.draft, weights.total(true));
  CHECK(weights.total(true) > 18'300'000'000ull && weights.total(true) < 18'650'000'000ull);
  constexpr size_t capacity = 32'000'000'000ull, reserve = 1'500'000'000ull;
  const uint32_t fit = ink::max_len_that_fits(d, weights, capacity, reserve, true);
  const uint32_t plain = ink::max_len_that_fits(d, weights, capacity, reserve, false);
  CHECK(fit >= 262144 && fit < plain && plain < d.trained_max_len);
  CHECK_EQ(fit % runtime::kMaxLenQuantum, 0u);
  CHECK(ink::sizes(d, weights, fit, true).memory().total() + reserve <= capacity);
  CHECK(ink::sizes(d, weights, fit + runtime::kMaxLenQuantum, true).memory().total() + reserve > capacity);
  CHECK_EQ(ink::max_len_that_fits(d, weights, reserve, reserve, true), 0u);
  // `--kv-cache int8` (PERF-102): 1,024 + 16 B a full visit's K (or V) row instead of 2,048; spec 41 §3.2's ~450k.
  const auto s8 = ink::sizes(d, weights, 4096, true, 0, true);
  CHECK_EQ(s8.persistent.full_k, size_t(8) * 4096 * 1024);
  CHECK_EQ(s8.persistent.full_k_scale, size_t(8) * 4096 * 8 * 2);
  CHECK_EQ(s8.persistent.ring_k, s.persistent.ring_k);
  CHECK_EQ(s8.draft.draft_full_k, s.draft.draft_full_k);
  const uint32_t fit8 = ink::max_len_that_fits(d, weights, capacity, reserve, true, 0, true);
  std::printf("Inkling one-card context with the drafter: %u bf16, %u int8 cache\n", fit, fit8);
  CHECK(fit8 > fit * 3 / 2 && fit8 <= d.trained_max_len);
  const auto refusal = cli::inkling2::no_fit(d, weights, capacity, reserve, true, 0, fit8, false);
  CHECK(refusal.find("--kv-cache int8") != std::string::npos);
  const auto no_draft = ink::sizes(d, weights, 4096, false);
  CHECK_EQ(no_draft.memory().drafter, 0u);
  CHECK_EQ(no_draft.scratch.draft_taps, 0u);

  const auto cfg = loader::read_json(checkpoint.directory() + "config.json");
  const auto quant = loader::read_json(checkpoint.directory() + "hf_quant_config.json");
  auto bad = cfg.obj();
  auto text = bad.at("text_config").obj();
  auto loop = text.at("loop_config").obj();
  loop["is_varloop"] = common::json::Value::make_bool(true);
  text["loop_config"] = common::json::Value::make_obj(std::move(loop));
  bad["text_config"] = common::json::Value::make_obj(std::move(text));
  try {
    (void)model::inkling2_desc_from_config(common::json::Value::make_obj(std::move(bad)), quant);
    CHECK(false);
  } catch (const std::runtime_error& err) {
    CHECK(std::string(err.what()).find("is_varloop") != std::string::npos);
  }
  std::printf("Inkling headers: %zu tensors, %zu packed weight bytes, %u / %u positions with / without drafter\n",
              checkpoint.tensor_count(), weights.total(true), fit, plain);
}
