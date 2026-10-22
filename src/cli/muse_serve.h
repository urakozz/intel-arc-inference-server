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
//   - spec 23f: speculative decoding on one card - `--spec dflash` (the Muse-Glimmer assistant, mtp_k() = --spec-max:
//     step_many() drafts a block through MuseDraftHooks, verifies at M = K + 1 and accepts on the host) or `--spec
//     lookup` (verify_k(): step_drafts() over the server's prompt-lookup drafts). Greedy: the longest prefix the
//     target's argmax agrees with (D2: the output is plain decode's). Sampled: the drafts are DRAWN HERE from the
//     drafter's q (q_j = softmax(scores_j / T) over draft row j's 16 candidates - server::dflash_q; the scores raw or
//     capped per --draft-q), then server::accept_sparse against the target's filtered p (the model's cap applied
//     exactly once) - lossless whatever q is (D4); lookup's point mass through server::accept_point_mass. The commit is
//     deferred to the next call that touches the engine (truncate_to may commit fewer), as EngineAdapter's. No MTP
//     head: --mtp / --spec mtp stay refused.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fstream>
#include <sstream>

#include "runtime/dflash/dflash_control.h"   // spec 23f: MuseDraftOut (a plain struct)
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

// Spec 23f: the drafter as the adapter sees it (b70-serve binds a runtime::dflash::MuseDrafter beside the engine:
// cli/muse_spec.h; the host test a fake) - one block at the engine's pos (the anchor its pending id), the readback's
// first k rows read. The block is always the drafter's whole 16 rows (MuseDrafter::draft).
struct MuseDraftHooks {
  std::function<const runtime::dflash::MuseDraftOut&(uint32_t k)> draft;
  bool on() const { return static_cast<bool>(draft); }
};

template <class Engine>
struct MuseEngineAdapterT : server::EngineIface {
  // `vocab_used`: the tokenizer's id count (202048); `logit_mult` / `softcap`: the model's (model::MuseGlimmerDesc's
  // out_mult 0.19611613513818404 and softcap 20). Spec 23f: `spec_k` > 0 - speculative decoding up to K drafts an
  // iteration, through `drafter` (--spec dflash) or the server's lookup drafts (`lookup`, --spec lookup).
  MuseEngineAdapterT(Engine& engine, uint32_t vocab_used, float logit_mult, float softcap, uint32_t spec_k = 0,
                     bool lookup = false, MuseDraftHooks drafter = {})
      : eng(engine), vocab_used_(vocab_used), logit_mult_(logit_mult), softcap_(softcap), host_logits_(engine.vocab()),
        k_(spec_k), lookup_(lookup), df_(std::move(drafter)) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("MuseEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Muse-Glimmer's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
    if (k_ > kMaxK) throw std::runtime_error("MuseEngineAdapter: K " + std::to_string(k_) + " exceeds 15");
    if (lookup_ && df_.on()) throw std::runtime_error("MuseEngineAdapter: --spec dflash and --spec lookup: pick one");
    if (k_ > 0 && !lookup_ && !df_.on())
      throw std::runtime_error("MuseEngineAdapter: K > 0 without a proposer (--spec dflash or --spec lookup)");
    if (k_ > 0 && k_ > engine.spec_rows() - 1)
      throw std::runtime_error("MuseEngineAdapter: K " + std::to_string(k_) + " exceeds the engine's verify lists (" +
                               std::to_string(engine.spec_rows()) + " rows)");
    if (k_ > 0) spec_logits_.resize(size_t(kMaxK + 1) * engine.vocab());
  }

  static constexpr uint32_t kMaxK = 15;   // the assistant's block of 16 - 1

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    eng.reset();
    fresh_ = false;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's prefill
  // starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    flush_commit();
    eng.prefill(ids);
    fresh_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    flush_commit();
    settle_fresh(sampling);
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_pending(sampling);
    else mask_pending();
    ++gen_tokens_;
    return id;
  }

  // The first id after a prefill / ingest: the head's row of the last prompt position is still in the logits buffer -
  // sample it (or mask it) rather than keep the device argmax.
  void settle_fresh(const server::Sampling& sampling) {
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
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }

  // --- spec 23f: speculative decoding -------------------------------------------------------------------------------
  uint32_t mtp_k() override { return df_.on() ? k_ : 0; }
  uint32_t verify_k() override { return lookup_ ? k_ : 0; }

  // One DFlash iteration at pos = n with x_n pending: the drafter's block (greedy: its device picks; sampled: drawn
  // here from its q), verify at M = K + 1, accept on the host; returns x_n, d_1 .. d_j. The commit is deferred.
  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    flush_commit();
    settle_fresh(sampling);
    uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    if (!df_.on() || vocab_used_ < eng.vocab()) k = 0;   // a head wider than the tokenizer: plain (masked) steps
    ++iters_at_k_[k];
    if (k == 0) return {step(sampling)};
    const uint32_t n = eng.pos(), x = eng.pending();
    const runtime::dflash::MuseDraftOut& o = df_.draft(k);
    std::vector<uint32_t> drafts(k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      for (uint32_t j = 0; j < k; ++j) drafts[j] = o.ids[j];
      eng.verify(drafts.data(), k);
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
    } else {
      seed_rng(sampling);
      // RNG order: the k draft draws, then accept_sparse's uniforms, then its one correction / bonus draw.
      const float T = server::dflash_temperature(sampling);
      server::SparseQ q[kMaxK];
      for (uint32_t j = 0; j < k; ++j) {
        q[j] = server::dflash_q(o.cand[j], o.scores[j], runtime::dflash::MuseDraftOut::kTopK, T);
        drafts[j] = draw_sparse(q[j]);
      }
      eng.verify(drafts.data(), k);
      eng.read_verify_logits(spec_logits_.data(), k + 1);
      a = server::accept_sparse(spec_logits_.data(), q, drafts.data(), k, vocab_used_, eng.vocab(), capped(sampling),
                                rng_);
    }
    return keep(n, x, drafts, k, a);
  }

  // One lookup iteration (spec 19e's contract): propose(x_n) -> d_1 .. d_k, verify, accept (greedy: argmax; sampled:
  // the point mass, lossless).
  std::vector<uint32_t> step_drafts(const server::Sampling& sampling, const Proposer& propose) override {
    flush_commit();
    settle_fresh(sampling);
    const uint32_t x = eng.pending();
    const std::vector<uint32_t> d = propose(x);
    uint32_t k = std::min<uint32_t>({uint32_t(d.size()), k_, eng.max_verify_k()});
    if (!lookup_ || vocab_used_ < eng.vocab()) k = 0;
    ++iters_at_k_[k];
    if (k == 0) return {step(sampling)};
    const uint32_t n = eng.pos();
    std::vector<uint32_t> drafts(d.begin(), d.begin() + k);
    eng.verify(drafts.data(), k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
    } else {
      seed_rng(sampling);
      eng.read_verify_logits(spec_logits_.data(), k + 1);
      a = server::accept_point_mass(spec_logits_.data(), drafts.data(), k, vocab_used_, eng.vocab(), capped(sampling),
                                    rng_);
    }
    return keep(n, x, drafts, k, a);
  }

  // Keep x_n, d_1 .. d_m of the last iteration (pos = n + m + 1), pending d_{m+1} (or the run's next id when m = j).
  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.pos()) return;
      throw std::logic_error("MuseEngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("MuseEngineAdapter::truncate_to: pos " + std::to_string(p) + " is outside the last run [" +
                             std::to_string(pending_pos_ + 1) + ", " + std::to_string(pending_pos_ + pending_j_ + 1) +
                             "]");
    const uint32_t m = p - pending_pos_ - 1;
    gen_tokens_ -= pending_j_ - m;
    pending_ = false;
    eng.commit(m, m < pending_j_ ? pending_drafts_[m] : pending_next_);
  }

  void flush_commit() {
    if (!pending_) return;
    pending_ = false;
    eng.commit(pending_j_, pending_next_);
  }

  uint64_t drafter_key() override { return eng.drafter_key(); }
  uint64_t spec_iterations() const { return iters_; }   // tests
  uint64_t spec_accepted() const { return accepted_; }
  uint64_t spec_drafted() const { return drafted_; }

  // --- spec 7 -----------------------------------------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kMuseStoreKey; }
  // Spec 23f: a deferred commit is made before a save (the request-end snapshot follows a run the server kept whole);
  // a restore discards it, as reset() does.
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
    eng.ingest(ids);
    fresh_ = true;
  }

  // The logits row of the step just run (the next token's distribution), filtered with the model's cap and drawn as
  // EngineAdapter's sample(); the id replaces the device argmax as the pending id. An unseeded request continues the
  // stream (seeded once); a seeded request reseeds it at its first sampled step (step()).
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
    if (k_ > 0 && iters_ > 0) {   // spec 23f: the previous request's speculative line
      std::fprintf(stderr, "%s: K %u, %llu iterations, %llu/%llu drafts accepted (%.3f), iterations at K 0..15:",
                   lookup_ ? "lookup" : "dflash", k_, (unsigned long long)iters_, (unsigned long long)accepted_,
                   (unsigned long long)drafted_, drafted_ ? double(accepted_) / double(drafted_) : 0.0);
      for (uint32_t k = 0; k <= kMaxK; ++k) std::fprintf(stderr, "%s%llu", k ? "/" : " ", (unsigned long long)iters_at_k_[k]);
      std::fprintf(stderr, "\n");
    }
    iters_ = accepted_ = drafted_ = 0;
    for (auto& c : iters_at_k_) c = 0;
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
  // The request's Sampling with the model's cap (the filter applies it exactly once).
  server::Sampling capped(const server::Sampling& s) const {
    server::Sampling c = s;
    c.logit_mult = logit_mult_;
    c.softcap = softcap_;
    return c;
  }
  void seed_rng(const server::Sampling& s) {
    if (!rng_seeded_) {
      rng_.seed(s.has_seed ? s.seed : std::random_device{}());
      rng_seeded_ = true;
    }
  }
  // One id from a sparse q (its candidates, in order; the last positive one on a rounding shortfall).
  uint32_t draw_sparse(const server::SparseQ& q) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    double acc = 0.0;
    uint32_t last = q.ids[0];
    for (uint32_t c = 0; c < q.n; ++c) {
      if (!(q.probs[c] > 0.0)) continue;
      last = q.ids[c];
      acc += q.probs[c];
      if (u < acc) return q.ids[c];
    }
    return last;
  }
  // The iteration's bookkeeping: the deferred commit and the ids returned.
  std::vector<uint32_t> keep(uint32_t n, uint32_t x, const std::vector<uint32_t>& drafts, uint32_t k,
                             const server::AcceptResult& a) {
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
  float logit_mult_, softcap_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  bool fresh_ = false;   // a prefill / ingest left the pending id; no step since
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0, sampled_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  // Spec 23f.
  uint32_t k_ = 0;
  bool lookup_ = false;
  MuseDraftHooks df_;
  std::vector<float> spec_logits_;   // [K + 1][vocab]: the verify rows' raw logits (sampled acceptance)
  bool pending_ = false;
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t iters_ = 0, drafted_ = 0, accepted_ = 0;
  uint64_t iters_at_k_[kMaxK + 1] = {};
};

}  // namespace cli::muse
