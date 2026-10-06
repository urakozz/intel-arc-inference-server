#pragma once
// Spec 18d: K2-Horizon behind b70-serve - runtime::k2::K2Engine as a server::EngineIface, the
// K2 counterpart of EngineAdapter (src/cli/serve_adapters.h). b70-serve instantiates it as
// K2EngineAdapter (= K2EngineAdapterT<runtime::k2::K2Engine>); the template exists so the
// host test (tests/server/k2_serve_test.cc) runs the same adapter over a device-free engine
// with K2's real KV layout. This header names no Level Zero type.
//
// What it forwards, and what K2 does not have:
//   - prefill / step / reset / pos / max_len: K2Engine's (spec 18c prefill on the l0 backend;
//     greedy = the device argmax; sampled = the logits row read back and filtered as every
//     model's (server::filter_probs), the drawn id written to cur_token[0] - EngineAdapter's
//     protocol);
//   - spec 7's prefix cache: KV-only snapshots (state_bytes() == 0; runtime/k2/k2_sizes.h
//     kv_snapshot_runs), the block hook on K2Engine::prefill, ingest for --prefix-split-last;
//     the store keyed by K2's own root (k2_store_key), so no entry made by another model's
//     engine - or by K2 in the other KV form - can ever hit;
//   - no speculative decoding: mtp_k() and verify_k() stay 0 (K2 has no MTP head, spec 18 §9;
//     --spec lookup needs verify lists K2Engine does not capture) - b70-serve refuses --mtp,
//     --spec mtp|lookup and --draft-vocab for K2 by name before the device.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/buffer_sizes.h"   // runtime::KvCache (device-free)
#include "server/deps.h"
#include "server/spec_accept.h"

namespace cli::k2 {

// The prefix cache's root for K2 (PrefixCache's kv_form, EngineIface::kv_form): every hash
// chain of the store starts here. Qwen3.8 / Agnes / Ornith keep 0 (bf16) / 1 (int8), so their
// chains are unchanged; K2's are "K2HORIZ" plus its form, a disjoint key - the entries' bytes
// are K2's KV layout in that form (48 layers x 8 heads x 128), meaningless to any other engine.
inline constexpr uint64_t kK2StoreKey = 0x4B32484F52495A00ull;   // "K2HORIZ\0"
inline uint64_t k2_store_key(runtime::KvCache kv) {
  return kK2StoreKey | (kv == runtime::KvCache::Int8 ? 1u : 0u);
}

template <class Engine>
struct K2EngineAdapterT : server::EngineIface {
  // `vocab_used`: the sampler's mask (the tokenizer's count; 250,624 = K2's whole vocabulary).
  K2EngineAdapterT(Engine& engine, uint32_t vocab_used)
      : eng(engine), vocab_used_(vocab_used), host_logits_(engine.vocab()) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("K2EngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside K2's vocabulary (" + std::to_string(engine.vocab()) + ")");
  }

  void reset() override { eng.reset(); }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next
  // request's prefill starts (the first moment the adapter knows it is over).
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    eng.prefill(ids);
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_pending(sampling);
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  // --- spec 7, KV-only (spec 18d Review Focus 3) -------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return k2_store_key(eng.kv_cache()); }
  void save_state(void* host) override { eng.save_state(host); }
  void load_state(const void* host, uint32_t p) override { eng.load_state(host, p); }
  void save_kv(uint32_t b, uint32_t e, void* host) override { eng.save_kv(b, e, host); }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override { eng.ingest(ids); }

  // The logits row of the step just run (the next token's distribution), filtered and drawn
  // as EngineAdapter's sample(); the id replaces the device argmax as the pending id. The RNG
  // seeds itself once, from the first sampled request's seed (EngineAdapter's rule).
  void sample_into_pending(const server::Sampling& sampling) {
    if (!rng_seeded_) {
      rng_.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded_ = true;
    }
    eng.read_logits_into(host_logits_.data());
    eng.set_pending(server::draw(server::filter_probs(host_logits_.data(), vocab_used_, sampling), rng_));
  }

  void report() {
    if (gen_tokens_ == 0) return;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  Engine& eng;

 private:
  uint32_t vocab_used_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  uint32_t gen_tokens_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
};

}  // namespace cli::k2
