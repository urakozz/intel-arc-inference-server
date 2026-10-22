#pragma once
// Spec 24e: LFM2.5-24B-A2B behind b70-serve - runtime::lfm2moe::Lfm2MoeEngine as a server::EngineIface, the LFM2 counterpart
// of cli::kolibri::KolibriEngineAdapterT (src/cli/kolibri_serve.h). b70-serve instantiates it over Lfm2MoeEngine (one card
// or two: the engine holds the placement); the template exists so the host test (tests/runtime/lfm2moe_snapshot_test.cc)
// runs the same adapter over a device-free engine with LFM2's real conv-ring and KV layouts. This header names no Level
// Zero type.
//
// What it forwards, and what LFM2.5 does not have:
//   - prefill / step / reset / pos / max_len / ingest: Lfm2MoeEngine's (24d's chunked prefill on l0; greedy = the device
//     argmax; sampled = the logits row read back and filtered as every model's (server::filter_probs at the tokenizer's id
//     count), the chosen id written by set_token on every device);
//   - the ids the tokenizer does not define: the head has 128000 rows, tokenizer.json defines 125017 ids (0..125016; rows
//     125017..127999 have no token - spec 24 §1). Sampling masks them (vocab_used = the tokenizer's count). LFM's device argmax
//     is built with VOCAB_USED = 125017, so it already ranks only the defined rows; mask_pending() is the fallback for a
//     B70_LFM_TOKENIZER_JSON that defines fewer ids than that (a greedy argmax landing on an undefined id is replaced by the
//     host argmax of the masked row), so the server never streams an id it cannot detokenise. No kernel changes;
//   - the repetition penalty (spec 24 decision 1): a SAMPLED request with Sampling::repetition_penalty != 1 has the logits row
//     penalised over the request's distinct prompt and output ids (server::apply_repetition_penalty, HF's rule) before
//     temperature / top-k / top-p. A greedy request (temperature 0, the gates) keeps the device argmax and ignores the
//     penalty - a documented deviation from vLLM, which penalises greedy requests too - unless `penalty_greedy`
//     (B70_LFM_PENALTY_GREEDY=1, the operator's A/B) routes it through the host: the row is read back, penalised and its
//     argmax (the lowest id among equal maxima, the device argmax's tie rule) becomes the next id;
//   - a SEEDED sampled request is reproducible: the RNG is re-seeded from the request's seed at its first step (the first step
//     after a prefill, an ingest, a restore or a reset - every request begins with one of them), so the same request again on a
//     long-running server draws the same ids; an unseeded request re-seeds from the random device at its first step, so
//     it never continues an earlier seeded request's stream. (EngineAdapter, the pipeline, K2 and Kolibri adapters follow the same
//     per-request rule since 2026-10-23; they seeded once per process before.)
//   - the next id is chosen at the START of step(), from the logits row of the previous replay (or of the prefill's head):
//     the server adds each id it consumes to the request's SeenIds before it calls step() again, so the penalty sees the
//     prompt and every earlier output id, and the first id after a prefill is sampled too when the request samples;
//   - spec 7's prefix cache: the two conv-ring rows of positions pos - 1 and pos - 2 per conv layer as the state, the
//     attention layers' KV as the blocks (runtime/lfm2moe/lfm2moe_sizes.h), the block hook on Lfm2MoeEngine::prefill (chunks
//     end at block ends), ingest for --prefix-split-last; the store keyed by LFM2's own root (kLfm2MoeStoreKey), so no entry
//     another model's engine made can ever hit;
//   - no speculative decoding: mtp_k() and verify_k() stay 0 (no MTP head, spec 24 §11; --spec lookup needs verify lists this
//     engine does not capture) - b70-serve refuses --mtp and --spec by name.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "server/deps.h"
#include "server/spec_accept.h"

namespace cli::lfm2moe {

// The prefix cache's root for LFM2.5 (PrefixCache's kv_form): "LFM2MOE\0". Qwen3.8 / Agnes / Ornith keep 0 / 1, K2 its
// "K2HORIZ" key, Kolibri "KOLIBRI\0", Muse its own: no chain crosses models; the entries' bytes are LFM2's conv-ring rows and
// attention-layer KV (bf16 only), meaningless to any other engine.
inline constexpr uint64_t kLfm2MoeStoreKey = 0x4C464D324D4F4500ull;

template <class Engine>
struct Lfm2MoeEngineAdapterT : server::EngineIface {
  // `vocab_used`: the tokenizer's id count (125017); ids at or above it are never emitted. `penalty_greedy`: route a greedy
  // request that carries a penalty through the host (B70_LFM_PENALTY_GREEDY=1); off by default.
  Lfm2MoeEngineAdapterT(Engine& engine, uint32_t vocab_used, bool penalty_greedy = false)
      : eng(engine), vocab_used_(vocab_used), penalty_greedy_(penalty_greedy), host_logits_(engine.vocab()) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("Lfm2MoeEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside LFM2.5-24B-A2B's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
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
    // The pending id is the argmax of the last replay (or of the prefill's head row). A sampled request replaces it with a
    // draw from that same row; a greedy one keeps it unless it is a row the tokenizer does not define.
    const bool penalised = sampling.repetition_penalty != 1.0f && sampling.seen != nullptr;
    if (new_request_) {
      rng_.seed(sampling.has_seed ? sampling.seed : uint64_t(std::random_device{}()));
      new_request_ = false;
    }
    if (!sampling.greedy || (penalised && penalty_greedy_)) choose_on_host(sampling, penalised);
    else mask_pending();
    const uint32_t id = eng.generate(1)[0];
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  // --- spec 7 (plan 24e Review Focus 5) -----------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kLfm2MoeStoreKey; }
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

  // The pending id for this request, chosen from the logits row the engine left: penalised first (HF's order: the repetition
  // penalty is a logits processor, temperature / top-k / top-p the warpers after it), then drawn (sampled) or argmaxed
  // (greedy + B70_LFM_PENALTY_GREEDY). The RNG is seeded at a request's first step (step()): from the request's seed, else from the
  // random device.
  void choose_on_host(const server::Sampling& sampling, bool penalised) {
    eng.read_logits_into(host_logits_.data());
    if (penalised)
      server::apply_repetition_penalty(host_logits_.data(), vocab_used_, sampling.seen->distinct(),
                                       sampling.repetition_penalty);
    if (sampling.greedy) {
      const float* row = host_logits_.data();
      eng.set_token(uint32_t(std::max_element(row, row + vocab_used_) - row));
      ++penalised_greedy_;
      return;
    }
    eng.set_token(server::draw(server::filter_probs(host_logits_.data(), vocab_used_, sampling), rng_));
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
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  uint64_t masked() const { return masked_; }                    // greedy ids moved off an undefined row (tests)
  uint64_t penalised_greedy() const { return penalised_greedy_; }  // greedy steps routed through the host (tests)

  Engine& eng;

 private:
  uint32_t vocab_used_;
  bool penalty_greedy_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool new_request_ = true;   // the next step() is a request's first
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0, penalised_greedy_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
};

}  // namespace cli::lfm2moe
