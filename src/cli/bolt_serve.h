#pragma once
// Spec 25e: Nemotron 3.5 Bolt behind b70-serve - runtime::bolt::BoltEngine as a server::EngineIface (LFM2's Lfm2MoeEngineAdapterT
// shape, src/cli/lfm2moe_serve.h, for a hybrid Mamba / attention / MoE model). b70-serve instantiates it over BoltEngine (two
// cards by default, Decision 4: the engine holds the placement); the template exists so the host test
// (tests/runtime/bolt_snapshot_test.cc) runs the same adapter over a device-free engine with Bolt's real state and KV layouts.
// This header names no Level Zero type.
//
// What the adapter forwards, and what Bolt does not have:
//   - prefill / step / reset / pos / max_len / ingest: BoltEngine's (25d's chunked prefill on l0; greedy = the device argmax;
//     sampled = the logits row read back and filtered as every model's - server::filter_probs at the tokenizer's id count,
//     temperature, top-k (none by default), top-p -, the chosen id written by set_token on every device);
//   - every row of the head is a token (131072: tokenizer.json defines all of them), so no id is ever masked on the release;
//     `vocab_used` guards a tokenizer that defines fewer (a greedy argmax on a row without a token takes the masked row's argmax);
//   - the repetition penalty: a SAMPLED request with Sampling::repetition_penalty != 1 has the logits row penalised over the
//     request's distinct prompt and output ids (server::apply_repetition_penalty, HF's rule) before the filters; a greedy request
//     keeps the device argmax and ignores it (LFM2's rule). NVIDIA's defaults carry none;
//   - a SEEDED sampled request is reproducible: the RNG is re-seeded from the request's seed at its first step (the first step
//     after a prefill, an ingest, a restore or a reset - every request begins with one of them); an unseeded request re-seeds from
//     the random device, so it never continues an earlier seeded request's stream;
//   - the next id is chosen at the START of step(), from the logits row of the previous replay (or of the prefill's head);
//   - spec 7's prefix cache (spec 25 §6.3): per Mamba block the SSM state and the three raw-xBC conv columns before the position
//     as the state, the 8 attention blocks' KV as the blocks (runtime/bolt/bolt_sizes.h's state_runs / kv_runs), the block hook on
//     BoltEngine::prefill (chunks end at block ends), ingest for --prefix-split-last; the store keyed by Bolt's own root
//     (kBoltStoreKey, the bf16-state arm another one: its snapshots are half the size), so no entry another model's engine made
//     - or another state precision's - can ever hit;
//   - no speculative decoding: mtp_k() and verify_k() stay 0 (no MTP head, no drafter) - b70-serve refuses --mtp and --spec
//     by name.
//
// Also here, host-side and device-free: the generation defaults (Decision 12) and the flags b70-serve refuses for this model.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/deps.h"
#include "server/openai.h"
#include "server/spec_accept.h"

namespace cli::bolt {

// The prefix cache's root for Bolt (PrefixCache's kv_form): "NEMOBOLT". Qwen3.8 / Agnes / Ornith keep 0 / 1, the other models
// their own keys: no chain crosses models; the entries' bytes are Bolt's SSM states, conv columns and attention KV (bf16 only),
// meaningless to any other engine. The bf16-state arm (B70_BOLT_SSM=bf16) flips the low bit: its states are half the size.
inline constexpr uint64_t kBoltStoreKey = 0x4E454D4F424F4C54ull;
inline constexpr uint64_t store_key(bool bf16_state) { return kBoltStoreKey ^ (bf16_state ? 1ull : 0ull); }

// Bolt's chat template (chat_template.jinja, 12,514 B; byte-identical in NVIDIA's and SergiioB's repos) - what the startup line
// compares the template it reads against. Not a refusal: --chat-template FILE and a newer snapshot template are the operator's.
inline constexpr const char* kTemplateSha256 = "6521afc75afb0661e3558e7e6dee5f2d15de8be1c4dece8171935e778ea283f4";

// ---- generation defaults (Decision 12) ---------------------------------------------------------------------------------------
// NVIDIA's published values (their generation_config.json, the card's "Recommended Sampling: temperature 1.0, top_p 0.95"):
// sampled, temperature 1.0, top-p 0.95, EOS [2, 11] (`</s>` and `<|im_end|>`). SergiioB's export ships a generation_config.json
// with EOS 11 only and no sampling keys - it gives the published set, not "greedy, EOS 11". The card names no top-k and vLLM
// (the reference server) applies none, so an absent top_k is NONE here (0), not Hugging Face's 50 that server::generation_sampling
// assumes for a field a file leaves out.
inline constexpr uint32_t kEosEnd = 11, kEosSentence = 2;

struct Generation {
  server::Sampling sampling;       // what a request that names no temperature / top_p / top_k samples with
  std::vector<uint32_t> eos;       // sorted, distinct
  std::vector<std::string> stops;  // generation_config.json's stop_strings (none in either published file)
  std::string source;              // "published defaults" | "<path> (the snapshot's file carries sampling keys)" | "--generation-config <path>"
};

inline bool has_sampling_keys(const nlohmann::json& g) {
  if (!g.is_object()) return false;
  for (const char* k : {"do_sample", "temperature", "top_p", "top_k", "repetition_penalty"})
    if (g.contains(k)) return true;
  return false;
}

inline Generation published_generation() {
  Generation out;
  out.sampling = server::generation_sampling(nlohmann::json{{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95}});
  out.sampling.top_k = 0;
  out.eos = {kEosSentence, kEosEnd};
  out.source = "published defaults (NVIDIA's generation_config.json: sampled, T 1.0, top-p 0.95, EOS [2, 11])";
  return out;
}

// `snapshot` is the snapshot's generation_config.json (null when it has none), `explicit_file` the parsed --generation-config FILE
// (null when not given; `explicit_path` names it for the startup line). Rules:
//   - an explicit file is the operator's word: its sampling fields (a field it leaves out takes the published value) and its
//     eos_token_id (absent: the published [2, 11]) are used as they are;
//   - otherwise the snapshot's file is used only when it carries sampling keys (NVIDIA's own does: do_sample, temperature,
//     top_p, eos [2, 11]); a file without them (SergiioB's: eos 11, nothing else) gives the published defaults, and its
//     eos_token_id is ADDED to the published set (an export listing another end id keeps it).
// Throws std::invalid_argument naming the key of a malformed field.
inline Generation resolve_generation(const nlohmann::json* snapshot, const nlohmann::json* explicit_file,
                                     const std::string& explicit_path = "", const std::string& snapshot_path = "") {
  Generation out = published_generation();
  const auto apply_sampling = [&](const nlohmann::json& g) {
    // The file's keys OVERLAY the published values (review I2): {"temperature": 0.6} keeps do_sample true and top_p 0.95; a
    // field neither names is not HF's default (top-k 50, greedy) but the published one.
    nlohmann::json merged = {{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95}};
    if (!g.is_object()) throw std::invalid_argument("generation_config.json is not an object");
    for (auto it = g.begin(); it != g.end(); ++it) merged[it.key()] = it.value();
    const server::Sampling s = server::generation_sampling(merged);
    out.sampling = s;
    if (!g.contains("top_k")) out.sampling.top_k = 0;
  };
  const auto eos_of = [](const nlohmann::json& g) {
    return g.is_object() && g.contains("eos_token_id") ? server::generation_eos_ids(g) : std::vector<uint32_t>{};
  };
  std::set<uint32_t> eos(out.eos.begin(), out.eos.end());
  if (explicit_file != nullptr) {
    apply_sampling(*explicit_file);
    const std::vector<uint32_t> e = eos_of(*explicit_file);
    if (!e.empty()) eos = std::set<uint32_t>(e.begin(), e.end());
    out.stops = server::generation_stop_strings(*explicit_file);
    out.source = "--generation-config " + explicit_path;
  } else if (snapshot != nullptr) {
    const std::vector<uint32_t> e = eos_of(*snapshot);
    eos.insert(e.begin(), e.end());
    out.stops = server::generation_stop_strings(*snapshot);
    if (has_sampling_keys(*snapshot)) {
      apply_sampling(*snapshot);
      out.source = snapshot_path + " (the snapshot's file carries sampling keys)";
    } else {
      out.source += "; the snapshot's generation_config.json names no sampling key" +
                    std::string(e.empty() ? "" : " (its eos_token_id is added)");
    }
  }
  out.eos.assign(eos.begin(), eos.end());
  return out;
}

// ---- what b70-serve refuses before the device -----------------------------------------------------------------------------------
// What b70-serve parsed that Bolt has no use for (the rest - --mtp, --spec, --kv-cache int8, the prefill backends, --device with
// --pp 2, a split without --pp 2 - is cli/bolt_decode.h's check_args, shared with b70-decode), each refusal by name
// (tests/server/bolt_request_test.cc, and cli_reject_serve_bolt_* through the binary on the box).
struct ServeFlags {
  bool mtp = false;              // --mtp N | auto
  std::string spec;              // --spec's value ("" when not given): off | mtp | lookup | dflash
  bool proposer_tuning = false;  // --mtp-max, --mtp-cost, --spec-min-match, --spec-history, --spec-max, --spec-cost,
                                 // --draft-model, --draft-format
  bool draft_vocab = false;      // --draft-vocab, --draft-vocab-ids
  bool qwen4exp_flags = false;   // --layers, --ple-dir, --tokenizer
};

inline void check_serve_flags(const ServeFlags& f) {
  if (f.mtp || f.spec == "mtp")
    throw std::runtime_error("Nemotron 3.5 Bolt has no MTP head (spec 25 §0: num_nextn_predict_layers 0): drop --mtp / --spec mtp");
  if (f.spec == "dflash" || f.spec == "lookup")
    throw std::runtime_error("--spec " + f.spec + ": Nemotron 3.5 Bolt has no speculative decoding (no MTP head, no drafter)");
  if (f.proposer_tuning)
    throw std::runtime_error("--mtp-max, --mtp-cost, --spec-min-match, --spec-history, --spec-max, --spec-cost, --draft-model "
                             "and --draft-format belong to a speculative proposer; Nemotron 3.5 Bolt has none (no MTP "
                             "head, no drafter)");
  if (f.draft_vocab)
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids narrow an MTP head's or a drafter's vocabulary; Nemotron 3.5 "
                             "Bolt has neither");
  if (f.qwen4exp_flags)
    throw std::runtime_error("--layers, --ple-dir and --tokenizer belong to Qwen3.8-Flash-Next (model_type qwen4_exp, spec 21e); "
                             "this checkpoint is Nemotron 3.5 Bolt (b70-decode --layers is its development mode)");
}

// The served context is capped at the trained 262144 (Decision 14) - refused before the device (the planner refuses it too, after
// the cards are open).
inline void check_serve_max_len(bool is_auto, uint32_t value, uint32_t trained_max_len) {
  if (!is_auto && value > trained_max_len)
    throw std::runtime_error("--max-len " + std::to_string(value) + " exceeds Nemotron 3.5 Bolt's trained context " +
                             std::to_string(trained_max_len) + " (max_position_embeddings; spec 25 Decision 14)");
}

// ---- the top-p filter without a top-k (review I1) -------------------------------------------------------------------------------
// With no top-k (Bolt's default) server::filter_probs partial_sorts the WHOLE 131072-row vocabulary for every sampled id: measured
// 17.2 ms on the Mac (k 50: 0.15 ms) against a ~16.5 ms decode step. This is the same filter without the full sort: the softmax
// is normalised once over every row (one pass, no sort; rows more than 40 nats under the top are not added: < 6e-13 of the sum), then the head is sorted in growing prefixes (64, 256, 1024 ...) until the
// cumulative mass reaches top_p. Same ids in the same descending order and the same kept count as filter_probs on distinct
// logits (its probabilities differ only in the last bits: the normaliser is summed in id order, not sorted order); among EQUAL
// logits the order is ascending id (filter_probs' partial_sort leaves it unspecified). It is Bolt's alone - the shared
// filter_probs and every other model's sampling are untouched. Only for top_k 0 and no softcap / multiplier; else filter_probs.
inline server::FilteredProbs filter_probs_top_p(const float* logits, uint32_t vocab_used, const server::Sampling& s) {
  if (s.top_k != 0 || s.softcap > 0.0f || s.logit_mult != 1.0f) return server::filter_probs(logits, vocab_used, s);
  const float temperature = s.temperature > 0.0f ? s.temperature : 1.0f;
  float top = logits[0];
  for (uint32_t i = 1; i < vocab_used; ++i) top = std::max(top, logits[i]);
  double sum = 0.0;
  for (uint32_t i = 0; i < vocab_used; ++i) {
    const double x = static_cast<double>((logits[i] - top) / temperature);
    if (x > -40.0) sum += std::exp(x);   // a row 40 nats under the top adds < 4.3e-18: dropped (all of them < 6e-13 of the sum)
  }
  const double top_p = std::clamp(static_cast<double>(s.top_p), 0.0, 1.0);
  std::vector<uint32_t> idx(vocab_used);
  for (uint32_t i = 0; i < vocab_used; ++i) idx[i] = i;
  const auto before = [logits](uint32_t a, uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  server::FilteredProbs f;
  size_t sorted = 0;
  for (size_t n = 64;; n *= 4) {
    n = std::min<size_t>(n, vocab_used);
    std::partial_sort(idx.begin() + sorted, idx.begin() + n, idx.end(), before);   // [0, sorted) are already the head
    sorted = n;
    double cumulative = 0.0;
    uint32_t kept = 0;
    f.probs.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      f.probs[i] = std::exp(static_cast<double>((logits[idx[i]] - top) / temperature)) / sum;   // (kept rows are never pruned out of the sum above: the nucleus holds the mass)
      cumulative += f.probs[i];
      if (cumulative >= top_p) {
        kept = uint32_t(i + 1);
        break;
      }
    }
    if (kept == 0 && n == vocab_used) kept = uint32_t(n);
    if (kept != 0) {
      f.probs.resize(kept);
      f.ids.assign(idx.begin(), idx.begin() + kept);
      break;
    }
  }
  f.mass = 0.0;
  for (double p : f.probs) f.mass += p;
  return f;
}

// ---- the adapter ------------------------------------------------------------------------------------------------------------
template <class Engine>
struct BoltEngineAdapterT : server::EngineIface {
  // `vocab_used`: the tokenizer's id count (131072); ids at or above it are never emitted.
  BoltEngineAdapterT(Engine& engine, uint32_t vocab_used) : eng(engine), vocab_used_(vocab_used), host_logits_(engine.vocab()) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("BoltEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Nemotron 3.5 Bolt's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
  }

  void reset() override {
    eng.reset();
    new_request_ = true;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's prefill starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    eng.prefill(ids);
    new_request_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    // The pending id is the argmax of the last replay (or of the prefill's head row). A sampled request replaces it with a draw
    // from that same row; a greedy one keeps it unless it is a row the tokenizer does not define.
    const bool penalised = sampling.repetition_penalty != 1.0f && sampling.seen != nullptr;
    if (new_request_) {
      rng_.seed(sampling.has_seed ? sampling.seed : uint64_t(std::random_device{}()));
      new_request_ = false;
    }
    if (!sampling.greedy) choose_on_host(sampling, penalised);
    else mask_pending();
    const uint32_t id = eng.generate(1)[0];
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  // --- spec 7 (plan 25e Review Focus 4) -----------------------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return store_key(eng.bf16_state()); }
  void save_state(void* host) override { eng.save_state(host); }
  void load_state(const void* host, uint32_t p) override {
    eng.load_state(host, p);
    new_request_ = true;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override { eng.save_kv(b, e, host); }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override {
    eng.ingest(ids);
    new_request_ = true;
  }

  // The pending id for a sampled request, chosen from the logits row the engine left: penalised first (HF's order: the repetition
  // penalty is a logits processor, temperature / top-k / top-p the warpers after it), then drawn.
  void choose_on_host(const server::Sampling& sampling, bool penalised) {
    eng.read_logits_into(host_logits_.data());
    if (penalised)
      server::apply_repetition_penalty(host_logits_.data(), vocab_used_, sampling.seen->distinct(), sampling.repetition_penalty);
    eng.set_token(server::draw(filter_probs_top_p(host_logits_.data(), vocab_used_, sampling), rng_));
    ++sampled_;
  }

  // Greedy: a device argmax on an id the tokenizer does not define becomes the masked row's argmax (the lowest id among equal
  // maxima, the device argmax's tie rule).
  void mask_pending() {
    if (eng.pending() < vocab_used_) return;
    eng.read_logits_into(host_logits_.data());
    const float* row = host_logits_.data();
    eng.set_token(uint32_t(std::max_element(row, row + vocab_used_) - row));
    ++masked_;
  }

  void report() {
    if (gen_tokens_ == 0) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  uint64_t masked() const { return masked_; }     // greedy ids moved off an undefined row (tests)
  uint64_t sampled() const { return sampled_; }   // rows drawn on the host (tests)

  Engine& eng;

 private:
  uint32_t vocab_used_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool new_request_ = true;   // the next step() is a request's first
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0, sampled_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
};

}  // namespace cli::bolt
