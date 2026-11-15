#pragma once
// Spec 20e: Kolibri-1 behind b70-serve - runtime::kolibri::KolibriEngine as a server::EngineIface, the
// adapter of a family with its own engine. b70-serve instantiates it
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
//   - Kolibri DSpark: speculative decoding with `spec_k` > 0 - --spec dflash with the DSpark drafter (step_many: a block
//     through KolibriDraftHooks, verified at M = K + 1 by KolibriEngine's verify lists, accepted on the host). Greedy: the
//     device's Markov walk, then the longest prefix the target's argmax agrees with (the output is plain decode's); a bonus
//     id on a row the tokenizer does not define takes the masked row's argmax, as a plain step's does. Sampled: the walk
//     Gumbel-samples at the request's T with a seed drawn from the request's RNG, q_j = softmax(scores_j / T) over row j's
//     16 candidates (server::dflash_q), then server::accept_sparse against the filtered p - lossless whatever q is. The
//     commit is deferred to the next call that touches the engine (truncate_to may commit fewer). Without spec_k every path
//     is 20e's (mtp_k() 0); there is no MTP head (--spec mtp refused) and no prompt lookup (--spec lookup refused).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "runtime/dflash/dflash_control.h"   // Kolibri DSpark: DFlashDraftOut (a plain struct)
#include "server/deps.h"
#include "server/spec_accept.h"

namespace cli::kolibri {

// The prefix cache's root for Kolibri (PrefixCache's kv_form): "KOLIBRI\0". Qwen3.8 / Ornith keep
// 0 / 1, so no chain crosses models; the entries' bytes are Kolibri's ring and
// full-layer layouts (bf16 only), meaningless to any other engine.
inline constexpr uint64_t kKolibriStoreKey = 0x4B4F4C4942524900ull;

// ---- Kolibri DSpark: the drafter as the adapter sees it ----------------------------------------------------------------
// b70-serve binds a runtime::dflash::DSparkDrafter beside the engine (cli/kolibri_spec.h); the host test a fake. One block at
// the engine's pos (the anchor its pending id): the Markov walk greedy at temperature 0, else Gumbel at `temperature` keyed by
// `seed`; the readback carries every row's 16 candidates and the walk's realized scores (the sampled q).
struct KolibriDraftHooks {
  std::function<const runtime::dflash::DFlashDraftOut&(uint32_t k, float temperature, uint64_t seed)> draft;
  bool on() const { return static_cast<bool>(draft); }
};
// An engine with KolibriEngine's speculative calls (verify / commit / ...): runtime::kolibri::KolibriEngine and the spec
// tests' fakes; 20e's snapshot fakes have none, and the adapter steps them plainly.
template <class E, class = void>
struct HasVerify : std::false_type {};
template <class E>
struct HasVerify<E, std::void_t<decltype(std::declval<E&>().verify(static_cast<const uint32_t*>(nullptr), 0u)),
                                decltype(std::declval<E&>().commit(0u, 0u))>> : std::true_type {};

template <class Engine>
struct KolibriEngineAdapterT : server::EngineIface {
  static constexpr bool kSpec = HasVerify<Engine>::value;
  static constexpr uint32_t kMaxK = 7;   // DSpark's block of 8 - 1 (runtime::kolibri::kVerifyRows - 1)
  // `vocab_used`: the tokenizer's id count (127998); ids at or above it are never emitted. Kolibri DSpark: `spec_k` the
  // deepest K (--spec-max) with `drafter` the DSpark drafter's hooks.
  KolibriEngineAdapterT(Engine& engine, uint32_t vocab_used, uint32_t spec_k = 0, KolibriDraftHooks drafter = {})
      : eng(engine), vocab_used_(vocab_used), host_logits_(engine.vocab()), k_(spec_k), df_(std::move(drafter)) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("KolibriEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Kolibri-1's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
    if (k_ > kMaxK) throw std::runtime_error("KolibriEngineAdapter: K " + std::to_string(k_) + " exceeds 7 (DSpark's block - 1)");
    if (k_ > 0 && !df_.on()) throw std::runtime_error("KolibriEngineAdapter: K > 0 without the DSpark drafter");
    if (k_ > 0 && !kSpec) throw std::runtime_error("KolibriEngineAdapter: K > 0 over an engine without verify lists");
    if (k_ > 0) spec_logits_.resize(size_t(kMaxK + 1) * engine.vocab());
  }

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    eng.reset();
    fresh_ = false;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's
  // prefill starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    flush_commit();
    rng_seeded_ = false;   // a new request: its first sampled step reseeds
    eng.prefill(ids);
    fresh_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    flush_commit();
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
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }

  // --- Kolibri DSpark: speculative decoding ------------------------------------------------------------------------------
  uint32_t mtp_k() override { return df_.on() ? k_ : 0; }

  // One DSpark iteration at pos = n with x_n pending: the drafter's block, verify at M = K + 1, accept on the host; returns
  // x_n, d_1 .. d_j. The commit is deferred. (An engine without KolibriEngine's verify / commit steps plainly.)
  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    if constexpr (kSpec) {
      return step_many_spec(sampling, k_req);
    } else {
      (void)k_req;
      return {step(sampling)};
    }
  }

  // Keep x_n, d_1 .. d_m of the last iteration (pos = n + m + 1), pending d_{m+1} (or the run's next id when m = j).
  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.pos()) return;
      throw std::logic_error("KolibriEngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("KolibriEngineAdapter::truncate_to: pos " + std::to_string(p) + " is outside the last run [" +
                             std::to_string(pending_pos_ + 1) + ", " + std::to_string(pending_pos_ + pending_j_ + 1) + "]");
    const uint32_t m = p - pending_pos_ - 1;
    gen_tokens_ -= pending_j_ - m;
    pending_ = false;
    if constexpr (kSpec) eng.commit(m, m < pending_j_ ? pending_drafts_[m] : pending_next_);
  }

  void flush_commit() {
    if (!pending_) return;
    pending_ = false;
    if constexpr (kSpec) eng.commit(pending_j_, pending_next_);
  }

  uint64_t drafter_key() override {
    if constexpr (kSpec) return eng.drafter_key();
    else return 0;
  }
  uint64_t spec_iterations() const { return iters_; }   // tests
  uint64_t spec_accepted() const { return accepted_; }
  uint64_t spec_drafted() const { return drafted_; }

  // --- spec 7 (plan 20e Review Focus 4) -------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kKolibriStoreKey; }
  // Kolibri DSpark: a deferred commit is made before a save (the request-end snapshot follows a run the server kept whole); a
  // restore discards it, as reset() does.
  void save_state(void* host) override {
    flush_commit();
    eng.save_state(host);
  }
  void load_state(const void* host, uint32_t p) override {
    pending_ = false;
    eng.load_state(host, p);
    fresh_ = false;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override {
    flush_commit();
    eng.save_kv(b, e, host);
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override {
    flush_commit();
    rng_seeded_ = false;
    eng.ingest(ids);
    fresh_ = true;
  }

  // The logits row of the step just run (the next token's distribution), filtered and drawn as
  // EngineAdapter's sample(); the id replaces the device argmax as the pending id. The RNG seeds itself
  // once per request, from the request's seed (EngineAdapter's rule).
  void sample_into_pending(const server::Sampling& sampling) {
    seed_rng(sampling);
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
    if (k_ > 0 && iters_ > 0) {   // Kolibri DSpark: the previous request's speculative line
      std::print(stderr, "dspark: K {:d}, {:d} iterations, {:d}/{:d} drafts accepted ({:.3f}), {:.3f} tokens a verify, iterations "
                 "at K 0..7:", k_, (unsigned long long)iters_, (unsigned long long)accepted_,
                 (unsigned long long)drafted_, drafted_ ? double(accepted_) / double(drafted_) : 0.0,
                 1.0 + double(accepted_) / double(iters_));
      for (uint32_t k = 0; k <= kMaxK; ++k) std::print(stderr, "{}{:d}", k ? "/" : " ", (unsigned long long)iters_at_k_[k]);
      std::print(stderr, "\n");
    }
    iters_ = accepted_ = drafted_ = 0;
    for (auto& c : iters_at_k_) c = 0;
    if (gen_tokens_ == 0) return;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::print(stderr, "gen: {:d} tokens, {:.3f} ms, {:.4f} ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  uint64_t masked() const { return masked_; }   // greedy ids moved off an undefined row (tests)

  Engine& eng;

 private:
  // The request's RNG, seeded once a request from its seed (EngineAdapter's rule).
  void seed_rng(const server::Sampling& sampling) {
    if (!rng_seeded_) {
      rng_.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded_ = true;
    }
  }

  std::vector<uint32_t> step_many_spec(const server::Sampling& sampling, uint32_t k_req) {
    flush_commit();
    uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    if (!df_.on() || k == 0) {
      ++iters_at_k_[0];
      return {step(sampling)};
    }
    // The pending id x_n is settled as step() settles it (after a prefill / ingest: sampled or masked on the host).
    if (fresh_) {
      if (sampling.greedy) mask_pending();
      else sample_into_pending(sampling);
      fresh_ = false;
    }
    const uint32_t n = eng.pos(), x = eng.pending();
    float T = 0.0f;
    uint64_t seed = 0;
    if (!sampling.greedy) {   // RNG order: the walk's device seed, then the acceptance's uniforms and its one draw
      seed_rng(sampling);
      T = server::dflash_temperature(sampling);
      seed = rng_();
    }
    const runtime::dflash::DFlashDraftOut& o = df_.draft(k, T, seed);
    std::vector<uint32_t> drafts(k);
    for (uint32_t j = 0; j < k; ++j) drafts[j] = o.ids[j];
    for (uint32_t j = 0; j < k; ++j)
      if (drafts[j] >= vocab_used_) {   // never a draft the tokenizer does not define (the top-k leaves them out)
        k = j;
        break;
      }
    ++iters_at_k_[k];
    if (k == 0) return {step(sampling)};
    drafts.resize(k);
    eng.verify(drafts.data(), k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
      if (a.next_token >= vocab_used_) {   // the bonus on an undefined row: the masked row's argmax (mask_pending's rule)
        eng.read_verify_logits(spec_logits_.data(), a.accepted + 1);
        const float* row = spec_logits_.data() + size_t(a.accepted) * eng.vocab();
        a.next_token = uint32_t(std::max_element(row, row + vocab_used_) - row);
        ++masked_;
      }
    } else {
      server::SparseQ q[kMaxK];
      for (uint32_t j = 0; j < k; ++j)
        q[j] = server::dflash_q(o.cand[j], o.scores[j], runtime::dflash::DFlashDraftOut::kTopK, T);
      eng.read_verify_logits(spec_logits_.data(), k + 1);
      a = server::accept_sparse(spec_logits_.data(), q, drafts.data(), k, vocab_used_, eng.vocab(), sampling, rng_);
    }
    pending_ = true;
    pending_pos_ = n;
    pending_j_ = a.accepted;
    pending_next_ = a.next_token;
    pending_drafts_ = drafts;
    ++iters_;
    drafted_ += k;
    accepted_ += a.accepted;
    std::vector<uint32_t> out{x};
    out.insert(out.end(), drafts.begin(), drafts.begin() + a.accepted);
    gen_tokens_ += static_cast<uint32_t>(out.size());
    return out;
  }

  uint32_t vocab_used_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  bool fresh_ = false;   // a prefill / ingest left the pending id; no step since
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  // Kolibri DSpark.
  uint32_t k_ = 0;
  KolibriDraftHooks df_;
  std::vector<float> spec_logits_;   // [K + 1][vocab]: the verify rows' raw logits
  bool pending_ = false;             // a run's commit is deferred
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t iters_ = 0, drafted_ = 0, accepted_ = 0;
  uint64_t iters_at_k_[kMaxK + 1] = {};
};

}  // namespace cli::kolibri
