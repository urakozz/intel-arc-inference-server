#pragma once
// Spec 23e: Muse-Glimmer-30B behind b70-serve - runtime::muse::MuseEngine as a server::EngineIface (Kolibri's
// KolibriEngineAdapterT shape, src/cli/kolibri_serve.h). b70-serve instantiates it over MuseEngine (one card, the
// default, or two: the engine holds the placement); the template exists so the host test
// (tests/runtime/muse_snapshot_test.cc) runs the same adapter over a device-free engine with Muse's real ring and KV
// layouts. This header names no Level Zero type.
//
// What it forwards, and what Muse does not have:
//   - prefill / step / reset / pos / max_len / ingest: MuseEngine's (23d's chunked prefill on l0; greedy = the device
//     argmax over the RAW fp32 logits - the softcap is monotone, spec 23 §1.1); SAMPLED = the head's raw row read back
//     and drawn through server::filter_probs with the model's logit multiplier and softcap (spec 23 §6.4: z' = 20
//     tanh(z x 0.196.../20) in fp32 on the host, before temperature / top-k / top-p) - the adapter writes the model's
//     two constants into the Sampling it filters with, so the cap is applied exactly once whatever the request's
//     defaults carried; the drawn id is written by set_token on every device;
//   - the FIRST id after a prefill is drawn from the prefill's head row when the request samples (Muse samples by
//     default: T 1.0, top-k 64, top-p 0.95 - decision 4), not left at the device argmax;
//   - ids the tokenizer does not define are never emitted (vocab_used; Muse's tokenizer defines all 202048 rows, so
//     this never fires on the release - kept for a checkpoint whose head is wider than its tokenizer);
//   - spec 7's prefix cache: the rings' last 2047 positions as the state, the full layers' KV as the blocks
//     (runtime/muse/muse_sizes.h), the block hook on MuseEngine::prefill (chunks end at block ends), ingest for
//     --prefix-split-last; the store keyed by Muse's own root (kMuseStoreKey), so no entry another model's engine made
//     can ever hit;
//   - no speculative decoding: mtp_k() and verify_k() stay 0 (no MTP head; the DFlash drafter is spec 23f) - b70-serve
//     refuses --mtp, --spec by name.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <sstream>

#include "server/deps.h"
#include "server/spec_accept.h"
#include "tokenizer/chat_template.h"

namespace cli::muse {

// ---- the chat template (spec 23 decision 1) -------------------------------------------------------------------------
// The ORIGINAL's chat_template.jinja (meta-models/Muse-Glimmer-30B @ a4e59da5, 9992 B; vendored at
// tests/tokenizer/muse/chat_template.jinja) - and the older one both Vishva007 repos (and the tiny test model) ship
// (7167 B, one line: no "Reasoning effort" normalisation, no skip of a system prompt's own reasoning line, another
// model's name in its error text).
inline constexpr const char* kOriginalTemplateSha256 = "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678";
inline constexpr const char* kOlderTemplateSha256 = "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e";

struct TemplateChoice {
  std::string file;     // the template chat::Template reads ("" = the snapshot's chat_template.jinja)
  std::string sha256;   // of what is read
  bool original = false;
};

// Before the device: `--chat-template FILE` (b70-serve) wins - any file, its sha256 printed (the operator's way to use
// an older template deliberately); otherwise <snapshot>/chat_template.jinja is read only when it IS the original's,
// else refused naming --chat-template (the GPTQ / -AutoRound repos ship the older one: pass the original's).
inline TemplateChoice check_serve_template(const std::string& snapshot_dir, const std::string& file) {
  const auto slurp = [](const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
  };
  TemplateChoice c;
  std::string src;
  if (!file.empty()) {
    if (!slurp(file, src)) throw std::runtime_error("--chat-template: cannot read '" + file + "'");
    c.file = file;
    c.sha256 = chat::sha256_hex(src);
    c.original = c.sha256 == kOriginalTemplateSha256;
    return c;
  }
  const std::string path = snapshot_dir + (snapshot_dir.empty() || snapshot_dir.back() == '/' ? "" : "/") +
                           "chat_template.jinja";
  if (!slurp(path, src))
    throw std::runtime_error("Muse-Glimmer-30B's chat template: " + path + " is absent - pass --chat-template FILE with "
                             "the original's chat_template.jinja (meta-models/Muse-Glimmer-30B, sha256 " +
                             std::string(kOriginalTemplateSha256).substr(0, 8) + "...; spec 23 decision 1)");
  c.sha256 = chat::sha256_hex(src);
  c.original = c.sha256 == kOriginalTemplateSha256;
  if (!c.original)
    throw std::runtime_error("Muse-Glimmer-30B's chat template: " + path + " has sha256 " + c.sha256.substr(0, 8) +
                             "..." + (c.sha256 == kOlderTemplateSha256 ? " (the older template the Vishva007 repos ship)" : "") +
                             ", not the original's " + std::string(kOriginalTemplateSha256).substr(0, 8) +
                             "... (spec 23 decision 1) - pass --chat-template FILE with meta-models/Muse-Glimmer-30B's "
                             "chat_template.jinja (vendored at tests/tokenizer/muse/chat_template.jinja)");
  return c;
}

// The prefix cache's root for Muse (PrefixCache's kv_form): "MUSEGLIM". Qwen3.8 / Agnes / Ornith keep 0 / 1, K2 its
// "K2HORIZ" key, Kolibri "KOLIBRI": no chain crosses models; the entries' bytes are Muse's ring and full-layer layouts
// (bf16 only), meaningless to any other engine.
inline constexpr uint64_t kMuseStoreKey = 0x4D555345474C494Dull;

template <class Engine>
struct MuseEngineAdapterT : server::EngineIface {
  // `vocab_used`: the tokenizer's id count (202048); `logit_mult` / `softcap`: the model's (model::MuseGlimmerDesc's
  // out_mult 0.19611613513818404 and softcap 20).
  MuseEngineAdapterT(Engine& engine, uint32_t vocab_used, float logit_mult, float softcap)
      : eng(engine), vocab_used_(vocab_used), logit_mult_(logit_mult), softcap_(softcap), host_logits_(engine.vocab()) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("MuseEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Muse-Glimmer's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
  }

  void reset() override {
    eng.reset();
    fresh_ = false;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's prefill
  // starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    eng.prefill(ids);
    fresh_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    // The first id after a prefill / ingest: the head's row of the last prompt position is still in the logits
    // buffer - sample it rather than keep the device argmax.
    if (fresh_) {
      // A request's seed holds per request (Muse samples by default): its first sampled step reseeds the RNG, so the
      // same seeded request draws the same ids on a long-running server. Unseeded requests continue the stream.
      if (!sampling.greedy && sampling.has_seed) {
        rng_.seed(sampling.seed);
        rng_seeded_ = true;
      }
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

  // --- spec 7 -----------------------------------------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kMuseStoreKey; }
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

  // The logits row of the step just run (the next token's distribution), filtered with the model's cap and drawn as
  // EngineAdapter's sample(); the id replaces the device argmax as the pending id. An unseeded server's RNG seeds itself
  // once (EngineAdapter's rule); a seeded request reseeds it at its first sampled step (step()).
  void sample_into_pending(const server::Sampling& sampling) {
    if (!rng_seeded_) {
      rng_.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded_ = true;
    }
    server::Sampling s = sampling;
    s.logit_mult = logit_mult_;
    s.softcap = softcap_;
    eng.read_logits_into(host_logits_.data());
    eng.set_token(server::draw(server::filter_probs(host_logits_.data(), vocab_used_, s), rng_));
    ++sampled_;
  }

  // Greedy: a device argmax on an id the tokenizer does not define becomes the masked row's argmax (the lowest id
  // among equal maxima, the device argmax's tie rule).
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

  uint64_t masked() const { return masked_; }     // greedy ids moved off an undefined row (tests)
  uint64_t sampled() const { return sampled_; }   // rows drawn on the host (tests)

  Engine& eng;

 private:
  uint32_t vocab_used_;
  float logit_mult_, softcap_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  bool fresh_ = false;   // a prefill / ingest left the pending id; no step since
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0, sampled_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
};

}  // namespace cli::muse
