// Spec 24e (plan 24e Task 3, decisions 1 and 2): the repetition penalty on the host sampler and the sampling defaults
// LFM2.5-24B-A2B's generation_config.json resolves to. Host only.
//   lfm2moe_sampling_test <tests/model/lfm2moe>
//   - apply_repetition_penalty against hand-computed rows: positive and negative logits (`l < 0 ? l * p : l / p`, HF's
//     RepetitionPenaltyLogitsProcessor), an id listed twice counted once, ids at or above vocab_used ignored, zero and
//     negative zero, penalty 1.0 a bitwise no-op, penalty < 1 (HF allows it: it favours repeats);
//   - SeenIds: distinct ids in insertion order, the 128000-id range;
//   - the ORDER: penalty, then temperature, top-k, top-p. The row is built so that a penalty applied after top-k would
//     keep a candidate the right order drops, and one applied before would not - the result is compared with a reference
//     written out step by step in the test;
//   - generation_config.json of LFM2.5 resolves to {sampled, T 0.1, top-k 50, top-p 1.0 (absent: HF's default), penalty
//     1.05}; Kolibri's and Muse's configs resolve exactly as they did before this plan (every field they name is read as
//     before; none of them leaves one to the defaults); an absent field takes HF's GenerationConfig default (T 1.0,
//     top-k 50, top-p 1.0, penalty 1.0), not Sampling{}'s (top-k 20, top-p 0.95).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "server/deps.h"
#include "server/openai.h"
#include "server/spec_accept.h"

namespace {

using json = nlohmann::json;

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void case_penalty_rows() {
  // HF: score = gather(logits, ids); score = where(score < 0, score * p, score / p); scatter.
  std::vector<float> row = {2.0f, -3.0f, 0.5f, 4.0f, 1.0f, 0.0f, -0.0f, -1.5f};
  server::apply_repetition_penalty(row.data(), 8, {0, 1, 1, 3, 5, 6, 7, 99}, 2.0f);
  // 0: 2/2; 1: -3*2 once, though listed twice; 2 untouched; 3: 4/2; 4 untouched; 5: 0/2; 6: -0.0 is not < 0, /2; 7: -1.5*2;
  // 99 is outside vocab_used and ignored.
  const std::vector<float> want = {1.0f, -6.0f, 0.5f, 2.0f, 1.0f, 0.0f, -0.0f, -3.0f};
  CHECK(bits_equal(row, want));

  // A penalty below 1 favours repeats: positive logits grow, negative ones shrink in magnitude.
  std::vector<float> low = {2.0f, -2.0f, 3.0f};
  server::apply_repetition_penalty(low.data(), 3, {0, 1}, 0.5f);
  CHECK(bits_equal(low, {4.0f, -1.0f, 3.0f}));

  // 1.0 changes nothing, bit for bit (-0.0 and NaN payloads included).
  std::vector<float> same = {1.0f, -2.5f, -0.0f, 7.0f};
  const std::vector<float> keep = same;
  server::apply_repetition_penalty(same.data(), 4, {0, 1, 2, 3}, 1.0f);
  CHECK(bits_equal(same, keep));

  // No ids: nothing; ids at the vocab_used edge: id vocab_used - 1 is penalised, vocab_used is not.
  std::vector<float> edge = {1.0f, 2.0f, 3.0f};
  server::apply_repetition_penalty(edge.data(), 3, {}, 2.0f);
  CHECK(bits_equal(edge, {1.0f, 2.0f, 3.0f}));
  server::apply_repetition_penalty(edge.data(), 3, {2, 3}, 2.0f);
  CHECK(bits_equal(edge, {1.0f, 2.0f, 1.5f}));
  std::printf("penalty rows: ok\n");
}

void case_seen_ids() {
  server::SeenIds seen;
  CHECK(seen.distinct().empty());
  for (uint32_t id : {5u, 3u, 5u, 127999u, 0u, 3u, 5u}) seen.add(id);
  CHECK((seen.distinct() == std::vector<uint32_t>{5, 3, 127999, 0}));
  CHECK(seen.contains(127999) && seen.contains(0) && !seen.contains(1) && !seen.contains(128000));
  bool threw = false;
  try {
    seen.add(1u << 30);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("seen ids: ok\n");
}

// HF's order, written step by step: repetition penalty on the row, then temperature, then top-k, then top-p.
std::vector<std::pair<uint32_t, double>> reference(std::vector<float> logits, const std::vector<uint32_t>& seen,
                                                   float penalty, float temperature, uint32_t top_k, float top_p) {
  std::vector<uint32_t> distinct = seen;
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  for (uint32_t id : distinct) logits[id] = logits[id] < 0 ? logits[id] * penalty : logits[id] / penalty;
  std::vector<double> z(logits.size());
  for (size_t i = 0; i < z.size(); ++i) z[i] = double(logits[i]) / temperature;
  std::vector<uint32_t> order(z.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return z[a] > z[b]; });
  order.resize(std::min<size_t>(top_k, order.size()));
  double sum = 0;
  std::vector<double> p;
  for (uint32_t id : order) {
    p.push_back(std::exp(z[id] - z[order[0]]));
    sum += p.back();
  }
  for (double& v : p) v /= sum;
  std::vector<std::pair<uint32_t, double>> out;
  double cumulative = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    out.push_back({order[i], p[i]});
    cumulative += p[i];
    if (cumulative >= top_p) break;
  }
  double mass = 0;
  for (auto& [id, v] : out) mass += v;
  for (auto& [id, v] : out) v /= mass;
  return out;
}

void case_order() {
  // Token 0 leads (8.0) but was seen: 8 / 4 = 2.0, below tokens 1 (5.0), 2 (4.0), 3 (3.0). With top-k 2 the right order
  // keeps {1, 2}; a penalty applied after top-k would keep {0, 1} and then drop 0 to 2.0 inside the pair.
  std::vector<float> logits = {8.0f, 5.0f, 4.0f, 3.0f, -1.0f, -2.0f};
  const std::vector<uint32_t> seen = {0, 5};
  const float penalty = 4.0f;
  server::Sampling s;
  s.greedy = false;
  s.temperature = 0.7f;
  s.top_k = 2;
  s.top_p = 1.0f;
  s.repetition_penalty = penalty;
  std::vector<float> row = logits;
  server::apply_repetition_penalty(row.data(), uint32_t(row.size()), seen, penalty);
  const server::FilteredProbs f = server::filter_probs(row.data(), uint32_t(row.size()), s);
  const auto ref = reference(logits, seen, penalty, 0.7f, 2, 1.0f);
  CHECK_EQ(f.ids.size(), ref.size());
  CHECK_EQ(f.ids.size(), 2u);
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK_EQ(f.ids[i], ref[i].first);
    CHECK_NEAR(f.probs[i] / f.mass, ref[i].second, 1e-6);   // filter_probs works in fp32
  }
  CHECK_EQ(f.ids[0], 1u);
  CHECK_EQ(f.ids[1], 2u);
  // The wrong order (top-k on the raw row, penalty after) would have kept token 0: show that the test would notice.
  server::Sampling raw = s;
  const server::FilteredProbs wrong = server::filter_probs(logits.data(), uint32_t(logits.size()), raw);
  CHECK_EQ(wrong.ids[0], 0u);
  // top-p after the penalty too: with top_p 0.5 the leader of the penalised row alone is kept.
  s.top_k = 4;
  s.top_p = 0.5f;
  const server::FilteredProbs g = server::filter_probs(row.data(), uint32_t(row.size()), s);
  const auto ref2 = reference(logits, seen, penalty, 0.7f, 4, 0.5f);
  CHECK_EQ(g.ids.size(), ref2.size());
  for (size_t i = 0; i < ref2.size(); ++i) CHECK_EQ(g.ids[i], ref2[i].first);
  std::printf("order: ok\n");
}

void case_defaults(const std::string& dir) {
  const json lfm = json::parse(slurp(dir + "/generation_config.json"));
  const server::Sampling s = server::generation_sampling(lfm);
  CHECK(!s.greedy);
  CHECK_NEAR(s.temperature, 0.1, 1e-7);
  CHECK_EQ(s.top_k, 50u);
  CHECK_NEAR(s.top_p, 1.0, 0.0);   // absent: HF's default, not Sampling{}'s 0.95
  CHECK_NEAR(s.repetition_penalty, 1.05, 1e-7);
  CHECK(s.seen == nullptr);
  CHECK_EQ(server::generation_eos_ids(lfm), std::vector<uint32_t>{124900});

  // Kolibri's and Muse's generation_config.json name every sampling field: resolved as before.
  const server::Sampling kol = server::generation_sampling(
      {{"temperature", 1.0}, {"top_p", 0.97}, {"top_k", 128}, {"do_sample", true}, {"eos_token_id", {127906, 127901}}});
  CHECK(!kol.greedy);
  CHECK_NEAR(kol.temperature, 1.0, 0.0);
  CHECK_NEAR(kol.top_p, 0.97, 1e-7);
  CHECK_EQ(kol.top_k, 128u);
  CHECK_NEAR(kol.repetition_penalty, 1.0, 0.0);
  const server::Sampling muse = server::generation_sampling({{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95}, {"top_k", 64}});
  CHECK_NEAR(muse.top_p, 0.95, 1e-7);
  CHECK_EQ(muse.top_k, 64u);

  // An absent field takes HF's GenerationConfig default.
  const server::Sampling bare = server::generation_sampling({{"do_sample", true}});
  CHECK(!bare.greedy);
  CHECK_NEAR(bare.temperature, 1.0, 0.0);
  CHECK_EQ(bare.top_k, 50u);
  CHECK_NEAR(bare.top_p, 1.0, 0.0);
  CHECK_NEAR(bare.repetition_penalty, 1.0, 0.0);
  CHECK(server::generation_sampling(json::object()).greedy);   // do_sample absent: greedy, as before

  // A penalty outside (0, 2] or not a number is refused naming the key.
  for (const json& bad : {json(0), json(-1.0), json(2.5), json("1.05"), json(nullptr), json(true)}) {
    bool threw = false;
    try {
      (void)server::generation_sampling({{"do_sample", true}, {"repetition_penalty", bad}});
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("repetition_penalty") != std::string::npos;
    }
    CHECK(threw);
  }
  CHECK_NEAR(server::generation_sampling({{"repetition_penalty", 2.0}}).repetition_penalty, 2.0, 0.0);
  std::printf("defaults: ok\n");
}

void case_eos_and_stop() {
  CHECK_EQ(server::generation_eos_ids({{"eos_token_id", 124900}}), std::vector<uint32_t>{124900});
  CHECK((server::generation_eos_ids({{"eos_token_id", {1, 250019}}}) == std::vector<uint32_t>{1, 250019}));
  for (const json& bad : {json::object(), json({{"eos_token_id", json::array()}}), json({{"eos_token_id", "x"}}),
                          json({{"eos_token_id", -1}}), json({{"eos_token_id", {1, "a"}}}),
                          json({{"eos_token_id", 4294967296ull}})}) {
    bool threw = false;
    try {
      (void)server::generation_eos_ids(bad);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("eos_token_id") != std::string::npos;
    }
    CHECK(threw);
  }
  CHECK((server::generation_stop_strings({{"stop_strings", "END"}}) == std::vector<std::string>{"END"}));
  CHECK((server::generation_stop_strings({{"stop_strings", {"a", "b"}}}) == std::vector<std::string>{"a", "b"}));
  CHECK(server::generation_stop_strings(json::object()).empty());
  std::printf("eos / stop_strings: ok\n");
}

void case_request_field() {
  const auto parse = [](const json& extra) {
    json body = {{"model", "m"}, {"messages", {{{"role", "user"}, {"content", "hi"}}}}};
    for (const auto& [k, v] : extra.items()) body[k] = v;
    return server::parse_request(body.dump(), true);
  };
  const auto rejected = [&](const json& extra) {
    try {
      (void)parse(extra);
    } catch (const server::BadRequest& e) {
      return std::string(e.what()).find("repetition_penalty") != std::string::npos;
    }
    return false;
  };
  CHECK_NEAR(parse({{"repetition_penalty", 1.1}}).sampling.repetition_penalty, 1.1, 1e-7);
  CHECK_NEAR(parse({{"repetition_penalty", 2}}).sampling.repetition_penalty, 2.0, 0.0);
  CHECK_NEAR(parse({{"repetition_penalty", 0.5}}).sampling.repetition_penalty, 0.5, 0.0);
  CHECK_NEAR(parse(json::object()).sampling.repetition_penalty, 1.0, 0.0);
  for (const json& bad : {json(0), json(-0.5), json(3), json(2.0001), json("1.1"), json(nullptr), json(true)})
    CHECK(rejected({{"repetition_penalty", bad}}));
  // A request overrides the model's default; a request without it keeps the default.
  server::Sampling defaults;
  defaults.greedy = false;
  defaults.repetition_penalty = 1.05f;
  json body = {{"model", "m"}, {"messages", {{{"role", "user"}, {"content", "hi"}}}}};
  CHECK_NEAR(server::parse_request(body.dump(), true, defaults).sampling.repetition_penalty, 1.05, 1e-7);
  body["repetition_penalty"] = 1.3;
  CHECK_NEAR(server::parse_request(body.dump(), true, defaults).sampling.repetition_penalty, 1.3, 1e-7);
  std::printf("request field: ok\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/model/lfm2moe";
  case_penalty_rows();
  case_seen_ids();
  case_order();
  case_defaults(dir);
  case_eos_and_stop();
  case_request_field();
  std::printf("lfm2moe_sampling_test OK\n");
  return 0;
}
