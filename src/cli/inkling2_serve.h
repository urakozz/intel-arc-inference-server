#pragma once
// Plan 41f Task 5: Inkling 2 Mini behind b70-serve - runtime::inkling2::Engine as a server::EngineIface. b70-serve
// instantiates it over the engine (one card; `replica` one a card); it is a template so a test can run it over another
// engine with the same calls. This header names no Level Zero type.
//
// What it forwards:
//   - prefill / step / reset / pos / max_len: the engine's prefill (4096-row chunks, the planned width) and its
//     one-row verify + commit. Greedy is the device argmax over the live vocabulary (200058 ids: the head's six
//     padding rows are never ranked); sampled reads the step's logits row (the release's softcap applied, as
//     read_logits returns it) and draws on the host (server::filter_probs), the drawn id the next one fed;
//   - the first id after a prefill or ingest is settled from the prompt's last logits row: kept (greedy) or drawn
//     (sampled), once a request, by whichever call reads it first - pending_id() too (PERF-91);
//   - the DSpark drafter (`spec_k` > 0, the checkpoint's mtp.safetensors): step_many proposes K drafts from the
//     pending id (Engine::propose, greedy over the drafter's Markov head), verifies them at K + 1 rows and keeps the
//     longest agreeing prefix (greedy) or the point-mass rule (sampled: the drafts are deterministic,
//     server::accept_point_mass - lossless). The commit is deferred to the next call that touches the engine, so
//     truncate_to may keep fewer;
//   - the DFlash 2 drafter (`--spec dflash`): the same round over its draft pass, with two differences. Of the K
//     drafts the window asked for, the round verifies the count with the most expected ids per cost: draft j's odds of
//     being kept are its selector probability (softmax of its 16 scores at the walk's temperature, T 1 for greedy)
//     calibrated at its position by the drafts verified so far (server::DraftCalib, one per engine), their running
//     product the odds the first j are all kept, the cost the round costs the server measures at the position
//     (set_live_costs: each verify, draft pass and plain step timed into server::LiveCost, seeded at load by
//     measure_costs) - so
//     an unsure stretch costs fewer verify rows where rows are dear, and few are cut where they are cheap (drafted()
//     says how many were verified).
//     A sampled request's walk draws its drafts from that softmax (a fresh device key a round from the request's
//     RNG) and keeps them by server::accept_sparse, the exact rule: lossless whatever the drafter proposes;
//   - prompt lookup inside it (PERF-55, dflash_window(): the drafter's window): step_drafts_or_many verifies the
//     matcher's drafts through the same verify rows (up to 7: Inkling's verify lists hold 8 rows, so no wide lookup
//     verify), the drafter's draft pass skipped - its context projection runs on every commit, whoever proposed;
//   - the prefix cache: the engine's snapshot calls (Engine::state_bytes ... set_block_hook), keyed by Inkling's own
//     root (kInkling2StoreKey, the int8 cache another key) and the drafter; PERF-92's head image: the prompt's last
//     logits row and its first id, so a whole repeated prompt is restored with nothing to prefill.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/bench_prompt.h"
#include "runtime/dflash/dflash_control.h"   // DFlashDraftOut: a plain struct
#include "server/adaptive_k.h"
#include "server/deps.h"
#include "server/dflash2_calib_seed.h"
#include "server/spec_accept.h"
#include "server/step_k.h"

namespace cli::inkling2 {

// The prefix cache's root for Inkling (PrefixCache's kv_form): "INKLING\0", the int8 cache "INKLING\1". Qwen3.8 /
// Ornith keep 0 / 1 and Kolibri "KOLIBRI\0", so no chain crosses models or KV forms.
inline constexpr uint64_t kInkling2StoreKey = 0x494E4B4C494E4700ull;
// The DFlash 2 drafter's repositories (plan 41k), in b70-serve's --spec auto order: our packed int4 export (L2) when it
// is downloaded, else the bf16 release (its linears int8 at load).
inline constexpr const char* kDflashInt4Repo = "urakozz/Inkling-2-Mini-DFlash2-W4A16-g64-AutoRound";
inline constexpr const char* kDflashRepo = "incoai/Inkling-2-Mini-DFlash2";

template <class Engine>
struct Inkling2EngineAdapterT : server::EngineIface {
  static constexpr uint32_t kMaxK = 7;   // the verify lists' 8 rows - 1
  // `vocab_used`: the tokenizer's id count (200058 = the head's live rows). `spec_k`: the deepest DSpark K
  // (--spec-max, 1..7; 0 = the drafter off); `chunk` the prefill's planned chunk width; `lookup`: prompt lookup inside
  // the drafter (dflash_window()).
  Inkling2EngineAdapterT(Engine& engine, uint32_t vocab_used, uint32_t chunk, uint32_t spec_k, bool lookup)
      : eng(engine), vocab_used_(vocab_used), chunk_(chunk), host_logits_(vocab_used),
        spec_logits_(size_t(kMaxK + 1) * vocab_used), k_(spec_k), lookup_(lookup) {
    if (k_ > kMaxK) throw std::runtime_error("--spec-max " + std::to_string(k_) + ": Inkling's drafter verifies 1..7");
    if (k_ > 0 && !eng.drafter()) throw std::runtime_error("Inkling's drafter is not loaded (mtp.safetensors)");
  }

  // The round costs this adapter times its calls into and the DFlash 2 round's stop reads; none: nothing is timed and
  // every draft the window asked for is verified.
  void set_live_costs(server::LiveCost* live) { live_ = live; }

  // The round's costs measured before the server takes requests: a 4,096-id prompt (the bench prompt cycled), each
  // verify width 1..8 with its commit (a round's verify and its deferred commit; the greedy plain step's append is
  // both in one submission) and the draft pass (K 7) timed kTimed times after one untimed, into `live` (LiveCost::seed), the
  // engine reset after. Returns the mean times in ms: the widths', then the draft pass's.
  std::pair<std::array<double, kMaxK + 1>, double> measure_costs(server::LiveCost& live) {
    constexpr uint32_t kDepth = server::LiveCost::kFirstDepth, kTimed = 8;
    std::vector<uint32_t> ids(kDepth);
    for (uint32_t i = 0; i < kDepth; ++i) ids[i] = kBenchPrompt[i % kBenchPromptLen];
    reset();
    prefill(ids);
    std::array<double, kMaxK + 1> ms{};
    double draft_ms = 0;
    const auto since = [](auto t) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    for (uint32_t rows = 1; rows <= kMaxK + 1; ++rows)
      for (uint32_t i = 0; i <= kTimed; ++i) {
        std::array<uint32_t, kMaxK + 1> input{};
        for (uint32_t j = 0; j < rows; ++j) input[j] = kBenchPrompt[(i + j) % kBenchPromptLen];
        const auto t = std::chrono::steady_clock::now();
        eng.verify({input.data(), rows});
        eng.commit(1);
        if (i) ms[rows - 1] += since(t) / kTimed;
      }
    for (uint32_t i = 0; k_ > 0 && i <= kTimed; ++i) {
      const auto t = std::chrono::steady_clock::now();
      (void)eng.propose(kBenchPrompt[i % kBenchPromptLen], kMaxK);
      if (i) draft_ms += since(t) / kTimed;
    }
    reset();
    gen_tokens_ = 0;
    if (k_ > 0) live.seed(kDepth, ms, draft_ms);
    return {ms, draft_ms};
  }

  // POST /v1/control/reconfigure: the drafter on (K) or off (0) between requests; lookup only beside it.
  void set_spec(uint32_t k, bool lookup) {
    if (k > kMaxK || (k > 0 && !eng.drafter())) throw std::runtime_error("Inkling's drafter: K is 0..7, loaded at start");
    flush_commit();
    k_ = k;
    lookup_ = lookup && k > 0;
  }

  void reset() override {
    pending_ = false;
    eng.reset();
    fresh_ = false;
  }

  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    flush_commit();
    rng_seeded_ = false;
    if constexpr (requires { Engine::chunk_rows(ids.size()); })   // pp: the prompt's chunk up to the planned width
      next_ = eng.prefill(ids, std::min(chunk_, Engine::chunk_rows(ids.size())));
    else
      next_ = eng.prefill(ids, chunk_);
    fresh_ = true;
    head_host_ = false;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    flush_commit();
    settle_first(sampling);
    const uint32_t id = next_;
    const uint32_t n = eng.position();
    const auto t = std::chrono::steady_clock::now();
    if (sampling.greedy) {
      next_ = eng.append(id);   // the greedy plain step (its candidate head), committed in the same submission
      time_verify(n, 1, t);
    } else {
      eng.verify({&id, 1});
      eng.read_logits(0, host_logits_);
      next_ = draw(sampling, host_logits_.data(), {&id, 1});
      eng.commit(1);
      time_verify(n, 1, t);
    }
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.position(); }
  std::optional<uint32_t> pending_id(const server::Sampling& sampling) override {
    if (pending_) return std::nullopt;
    settle_first(sampling);
    return next_;
  }

  // --- the DSpark drafter ----------------------------------------------------------------------------------------
  uint32_t mtp_k() override { return k_; }

  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    flush_commit();
    settle_first(sampling);
    const uint32_t n = eng.position();
    uint32_t k = std::min({k_req, k_, room()});
    verified_ = k;
    if (k == 0) {
      ++iters_at_k_[0];
      return {step(sampling)};
    }
    if (!eng.dflash()) {
      ++iters_at_k_[k];
      const auto t = std::chrono::steady_clock::now();
      const auto proposed = eng.propose(next_, k);
      time_draft(n, k, t);
      return verify_run(sampling, n, std::vector<uint32_t>(proposed.begin(), proposed.end()));
    }
    float T = 0.f;
    uint64_t seed = 0;
    if (!sampling.greedy) {   // RNG order: the walk's device key, then the acceptance's uniforms and its one draw
      seed_rng(sampling);
      T = server::dflash_temperature(sampling);
      seed = rng_();
    }
    const auto t = std::chrono::steady_clock::now();
    const auto proposed = eng.propose(next_, k, T, seed);
    time_draft(n, k, t);
    const runtime::dflash::DFlashDraftOut& o = eng.draft_out();
    server::SparseQ q[kMaxK];
    float p[kMaxK];
    for (uint32_t j = 0; j < k; ++j) {
      q[j] = server::dflash_q(o.cand[j], o.scores[j], runtime::dflash::DFlashDraftOut::kTopK, T > 0.f ? T : 1.f);
      p[j] = float(q[j].prob(proposed[j]));
    }
    if (live_) {   // E(K) = 1 + c_1 + c_1 c_2 + ... over verify[K] + the draft pass, the best K <= k
      const server::MtpCost c = live_->at(n);
      double kept = 1.0, ids = 1.0, best = 0.0;
      uint32_t at = k;
      for (uint32_t j = 0; j < k; ++j) {
        kept *= calib_.rate(j, p[j]);
        ids += kept;
        const double rate = ids / (c.verify[j + 1] + c.draft[j]);
        if (rate > best) {
          best = rate;
          at = j + 1;
        }
      }
      k = at;
    }
    verified_ = k;
    ++iters_at_k_[k];
    auto out = verify_run(sampling, n, std::vector<uint32_t>(proposed.begin(), proposed.begin() + k), T > 0.f ? q : nullptr);
    const uint32_t kept = uint32_t(out.size() - 1);
    for (uint32_t j = 0; j < k && j <= kept; ++j) calib_.observe(j, p[j], j < kept);   // up to the first rejection
    return out;
  }
  uint32_t drafted(uint32_t k) override { return std::min(k, verified_); }

  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.position()) return;
      throw std::logic_error("Inkling2EngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("Inkling2EngineAdapter::truncate_to: pos " + std::to_string(p) + " is outside the last run");
    const uint32_t m = p - pending_pos_ - 1;
    gen_tokens_ -= pending_j_ - m;
    pending_ = false;
    timed_commit(m + 1);
    next_ = m < pending_j_ ? pending_drafts_[m] : pending_next_;
  }

  void flush_commit() {
    if (!pending_) return;
    pending_ = false;
    timed_commit(pending_j_ + 1);
    next_ = pending_next_;
  }

  // PERF-55: prompt lookup inside the drafter.
  uint32_t dflash_window() override { return lookup_ && k_ > 0 ? kLookupWindow : 0; }
  std::vector<uint32_t> step_drafts_or_many(const server::Sampling& sampling, uint32_t k_draft,
                                            const Proposer& propose) override {
    flush_commit();
    settle_first(sampling);
    std::vector<uint32_t> d = propose(next_);
    if (d.empty()) return step_many(sampling, k_draft);
    verified_ = 0;   // the proposer's drafts, not the drafter's
    d.resize(std::min<size_t>(d.size(), std::min(kMaxK, room())));
    ++lookup_iters_;
    if (d.empty()) return {step(sampling)};
    return verify_run(sampling, eng.position(), d);
  }

  uint64_t drafter_key() override {
    if (!eng.drafter()) return 0;
    if (!eng.dflash()) return kDrafterKey;
    const uint64_t packed = eng.packed_draft_key();
    return packed ? packed : kDflashKey;
  }

  // --- the prefix cache --------------------------------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return kInkling2StoreKey | (eng.kv8() ? 1 : 0); }
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
  // PERF-92: the prompt's last logits row (fp32, the live vocabulary, softcapped) and its first id, read in the
  // prompt-end hook (the engine's prefill has chosen it).
  size_t head_bytes() override { return size_t(vocab_used_) * sizeof(float) + sizeof(uint32_t); }
  void save_head(void* host) override {
    auto* h = static_cast<std::byte*>(host);
    eng.read_prefill_logits({reinterpret_cast<float*>(h), vocab_used_});
    const uint32_t id = eng.head_id();
    std::memcpy(h + size_t(vocab_used_) * sizeof(float), &id, sizeof(id));
  }
  void load_head(const void* host) override {
    report();
    rng_seeded_ = false;
    const auto* h = static_cast<const std::byte*>(host);
    std::memcpy(host_logits_.data(), h, size_t(vocab_used_) * sizeof(float));
    std::memcpy(&next_, h + size_t(vocab_used_) * sizeof(float), sizeof(next_));
    fresh_ = head_host_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }
  // --prefix-split-last: the last prompt id through one decode replay.
  void ingest(const std::vector<uint32_t>& ids) override {
    flush_commit();
    rng_seeded_ = false;
    for (size_t i = 0; i < ids.size(); ++i) {
      next_ = eng.verify({&ids[i], 1})[0];
      if (i + 1 == ids.size()) eng.read_logits(0, host_logits_);
      eng.commit(1);
    }
    fresh_ = head_host_ = true;
  }

  void report() {
    if (iters_ > 0 || lookup_iters_ > 0) {
      std::print(stderr, "dspark: K {:d}, {:d} iterations ({:d} prompt lookup), {:d}/{:d} drafts accepted ({:.3f}), "
                 "{:.3f} ids a verify, iterations at K 0..7:", k_, (unsigned long long)iters_,
                 (unsigned long long)lookup_iters_, (unsigned long long)accepted_, (unsigned long long)drafted_,
                 drafted_ ? double(accepted_) / double(drafted_) : 0.0,
                 iters_ ? 1.0 + double(accepted_) / double(iters_) : 0.0);
      for (uint32_t k = 0; k <= kMaxK; ++k) std::print(stderr, "{}{:d}", k ? "/" : " ", (unsigned long long)iters_at_k_[k]);
      std::print(stderr, "\n");
    }
    iters_ = lookup_iters_ = accepted_ = drafted_ = 0;
    for (auto& c : iters_at_k_) c = 0;
    if (gen_tokens_ == 0) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::print(stderr, "gen: {:d} tokens, {:.3f} ms, {:.4f} ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  uint64_t spec_iterations() const { return iters_; }   // tests
  uint64_t spec_accepted() const { return accepted_; }

  Engine& eng;

 private:
  // The drafter's window reach (its two window layers; prompt lookup's ENTER rule wants a source older than it).
  static constexpr uint32_t kLookupWindow = 1024;
  static constexpr uint64_t kDrafterKey = 0x494E4B4453504B31ull;   // "INKDSPK1": the checkpoint's own drafter
  static constexpr uint64_t kDflashKey = 0x494E4B4446324231ull;    // "INKDF2B1": the DFlash 2 release loaded as int8

  // The drafts a verify may add here: the context left after the pending id, at most kMaxK.
  uint32_t room() const {
    const uint32_t left = eng.max_len() - eng.position();
    return left > 1 ? std::min(kMaxK, left - 1) : 0;
  }

  void seed_rng(const server::Sampling& sampling) {
    if (!rng_seeded_) {
      rng_.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded_ = true;
    }
  }
  uint32_t draw(const server::Sampling& sampling, const float* row, std::span<const uint32_t> tail = {}) {
    seed_rng(sampling);
    return server::draw(server::filter_probs(row, vocab_used_, sampling, tail), rng_);
  }

  // The first id after a prefill / ingest / head restore: the prompt's last row, kept (greedy: the device argmax)
  // or drawn. `head_host_`: that row is in host_logits_ already.
  void settle_first(const server::Sampling& sampling) {
    if (!fresh_) return;
    fresh_ = false;
    if (sampling.greedy) return;
    if (!std::exchange(head_host_, false)) eng.read_prefill_logits(host_logits_);
    next_ = draw(sampling, host_logits_.data());
  }

  void time_verify(uint32_t pos, uint32_t rows, std::chrono::steady_clock::time_point t, double more_ms = 0) {
    if (live_)
      live_->verify(pos, rows,
                    more_ms + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
  }
  // A drafted round's deferred commit, its time added to the round's verify (pending_ms_) - the round's whole cost.
  void timed_commit(uint32_t rows) {
    const auto t = std::chrono::steady_clock::now();
    eng.commit(rows);
    time_verify(pending_pos_, pending_rows_, t, pending_ms_);
  }
  void time_draft(uint32_t pos, uint32_t k, std::chrono::steady_clock::time_point t) {
    if (live_)
      live_->draft(pos, k, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
  }

  // x = next_ and `drafts` verified at positions n .. n + k; the commit deferred. `q`: the drafts were drawn from these
  // sparse rows (the DFlash 2 walk), the exact rule; none: deterministic drafts, the point-mass rule.
  std::vector<uint32_t> verify_run(const server::Sampling& sampling, uint32_t n, const std::vector<uint32_t>& drafts,
                                   const server::SparseQ* q = nullptr) {
    const uint32_t k = uint32_t(drafts.size()), x = next_;
    std::vector<uint32_t> rows{x};
    rows.insert(rows.end(), drafts.begin(), drafts.end());
    // greedy: the plain step's candidate head (the same pick whenever the full head's winner is among its candidates);
    // tp's head is split across the cards, whole.
    std::span<const uint32_t> pred;
    const auto t = std::chrono::steady_clock::now();
    if constexpr (requires { eng.verify(rows, true); })
      pred = eng.verify(rows, sampling.greedy);
    else
      pred = eng.verify(rows);
    server::AcceptResult a{};
    if (sampling.greedy) {
      a = server::accept_greedy(pred.data(), drafts.data(), k);
    } else {
      seed_rng(sampling);
      for (uint32_t r = 0; r <= k; ++r)
        eng.read_logits(r, {spec_logits_.data() + size_t(r) * vocab_used_, vocab_used_});
      const auto history = sampling.seen ? std::span<const uint32_t>(rows) : std::span<const uint32_t>{};
      a = q ? server::accept_sparse(spec_logits_.data(), q, drafts.data(), k, vocab_used_, vocab_used_, sampling, rng_,
                                    history)
            : server::accept_point_mass(spec_logits_.data(), drafts.data(), k, vocab_used_, vocab_used_, sampling, rng_,
                                        history);
    }
    pending_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    pending_rows_ = uint32_t(rows.size());
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
    gen_tokens_ += uint32_t(out.size());
    return out;
  }

  uint32_t vocab_used_, chunk_;
  std::vector<float> host_logits_, spec_logits_;
  std::mt19937_64 rng_;
  bool rng_seeded_ = false;
  uint32_t next_ = 0;         // the pending id: what the next step returns
  bool fresh_ = false;        // a prefill / ingest / head restore left next_; its row not yet settled
  bool head_host_ = false;    // host_logits_ holds that row
  uint32_t gen_tokens_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  uint32_t k_;
  uint32_t verified_ = 0;     // the drafts the last step_many verified
  server::LiveCost* live_ = nullptr;   // set_live_costs
  // The selector's probabilities against the drafts kept, by position. The DFlash 2 drafter starts from the
  // measured per-(position, bin) rates (kDflash2CalibSeed) as its prior at weight 100, not the stated
  // probability - chosen on the held-out sets by tools/oracle/price_stop.py.
  server::DraftCalib calib_{server::kDflash2CalibSeed, 100.0};
  bool lookup_;
  bool pending_ = false;      // a run's commit is deferred
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  uint32_t pending_rows_ = 0;   // its verify's rows and time, timed into live_ with its commit
  double pending_ms_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t iters_ = 0, lookup_iters_ = 0, drafted_ = 0, accepted_ = 0;
  uint64_t iters_at_k_[kMaxK + 1] = {};
};

}  // namespace cli::inkling2
