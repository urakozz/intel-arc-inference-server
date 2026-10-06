#pragma once
// Spec 20e: Kolibri-1 behind b70-serve - runtime::kolibri::KolibriEngine as a server::EngineIface, the
// Kolibri counterpart of cli::k2::K2EngineAdapterT (src/cli/k2_serve_adapter.h). b70-serve instantiates it
// over KolibriEngine (one card or two: the engine holds the placement); the template exists so the host
// test (tests/runtime/kolibri_snapshot_test.cc) runs the same adapter over a device-free engine with
// Kolibri's real ring and KV layouts. This header names no Level Zero type.
//
// What it forwards, and what Kolibri does not have:
//   - prefill / step / reset / pos / max_len / ingest: KolibriEngine's (20d's chunked prefill on l0; greedy
//     = the device argmax; sampled = the logits row read back and filtered as every model's
//     (server::filter_probs at the tokenizer's id count), the drawn id written by set_token on every device);
//   - the ids the tokenizer does not define: the head has 128000 rows, tokenizer.json defines 127998 ids
//     (0..127997; rows 127998 and 127999 have no token - measured on the release's tokenizer.json, 20e
//     Task 1). Sampling masks them (vocab_used = the tokenizer's count); a greedy device argmax that lands
//     on one is replaced by the host argmax of the masked row, so the server never streams an id it cannot
//     detokenise (plan 20e Review Focus 3). No kernel changes: the device argmax still ranks every row;
//   - the FIRST id after a prefill is sampled too when the request samples (the logits row the prefill's
//     head left), not left at the device argmax - Kolibri samples by default (T 1.0, top-p 0.97, top-k 128);
//   - spec 7's prefix cache: the rings' last 512 positions as the state, the full layers' KV as the blocks
//     (runtime/kolibri/kolibri_sizes.h), the block hook on KolibriEngine::prefill (chunks end at block
//     ends), ingest for --prefix-split-last; the store keyed by Kolibri's own root (kKolibriStoreKey), so no
//     entry another model's engine made can ever hit;
//   - no speculative decoding: mtp_k() and verify_k() stay 0 (no MTP head, spec 20 §1; --spec lookup needs
//     verify lists KolibriEngine does not capture) - b70-serve refuses --mtp, --spec mtp|lookup by name.
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

namespace cli::kolibri {

// The prefix cache's root for Kolibri (PrefixCache's kv_form): "KOLIBRI\0". Qwen3.8 / Agnes / Ornith keep
// 0 / 1 and K2 its "K2HORIZ" key, so no chain crosses models; the entries' bytes are Kolibri's ring and
// full-layer layouts (bf16 only), meaningless to any other engine.
inline constexpr uint64_t kKolibriStoreKey = 0x4B4F4C4942524900ull;

template <class Engine>
struct KolibriEngineAdapterT : server::EngineIface {
  // `vocab_used`: the tokenizer's id count (127998); ids at or above it are never emitted.
  KolibriEngineAdapterT(Engine& engine, uint32_t vocab_used)
      : eng(engine), vocab_used_(vocab_used), host_logits_(engine.vocab()) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("KolibriEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Kolibri-1's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
  }

  void reset() override {
    eng.reset();
    fresh_ = false;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's
  // prefill starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    eng.prefill(ids);
    fresh_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    // The first id after a prefill / ingest: the head's row of the last prompt position is still in the
    // logits buffer - sample it rather than keep the device argmax.
    if (fresh_) {
      if (sampling.greedy) mask_pending();
      else sample_into_pending(sampling);
      fresh_ = false;
    }
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_pending(sampling);
    else mask_pending();
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  // --- spec 7 (plan 20e Review Focus 4) -------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kKolibriStoreKey; }
  void save_state(void* host) override { eng.save_state(host); }
  void load_state(const void* host, uint32_t p) override {
    eng.load_state(host, p);
    fresh_ = false;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override { eng.save_kv(b, e, host); }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override {
    eng.ingest(ids);
    fresh_ = true;
  }

  // The logits row of the step just run (the next token's distribution), filtered and drawn as
  // EngineAdapter's sample(); the id replaces the device argmax as the pending id. The RNG seeds itself
  // once, from the first sampled request's seed (EngineAdapter's rule).
  void sample_into_pending(const server::Sampling& sampling) {
    if (!rng_seeded_) {
      rng_.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded_ = true;
    }
    eng.read_logits_into(host_logits_.data());
    eng.set_token(server::draw(server::filter_probs(host_logits_.data(), vocab_used_, sampling), rng_));
  }

  // Greedy: a device argmax on an id the tokenizer does not define becomes the masked row's argmax (the
  // lowest id among equal maxima, the device argmax's tie rule).
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

  uint64_t masked() const { return masked_; }   // greedy ids moved off an undefined row (tests)

  Engine& eng;

 private:
  uint32_t vocab_used_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  bool fresh_ = false;   // a prefill / ingest left the pending id; no step since
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
};

}  // namespace cli::kolibri
