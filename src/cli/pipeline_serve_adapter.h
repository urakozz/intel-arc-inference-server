#pragma once
// Spec 16d: b70-serve --parallel pp - runtime::PipelineEngine (the model's layers over two B70s) as a
// server::EngineIface, the two-card counterpart of EngineAdapter (src/cli/serve_adapters.h).
// b70-serve instantiates it as PipelineEngineAdapterT<runtime::PipelineEngine>; the template
// exists so the host test (tests/server/pp_serve_test.cc) runs the SAME adapter over a
// device-free two-device engine. This header names no Level Zero type.
//
// It is EngineAdapter's logic, line for line, over the pipeline engine's host-side calls
// instead of a Control pointer and an immediate list:
//   - prefill / step / reset / pos / max_len: PipelineEngine's (spec 16c's two-card prefill;
//     greedy = device 1's argmax; sampled = device 1's logits row read back and filtered by
//     server::filter_probs, the drawn id written on both devices with set_token);
//   - spec 7's prefix cache: PipelineEngine's snapshots, which are the ONE-CARD host layouts
//     (16b's design; with MTP the head's hidden row and KV layer too, spec 16d), so the store
//     keys them exactly as EngineAdapter does - kv_form() 0 (bf16) / 1 (int8): an entry is
//     the same bytes whichever of --parallel 1 / 2 wrote it. The block hook on the pipeline's
//     prefill (from the shadows mid-prompt), ingest for --prefix-split-last;
//   - spec 8's MTP (`--spec mtp`, `--spec mtp`): draft on device 1, verify on both, the
//     host's acceptance (greedy argmax match; sampled: server::accept_sampled over the q rows
//     of the drafts and the p rows of the verify), the commit deferred exactly as
//     EngineAdapter defers it (truncate_to commits fewer);
//   - spec 19e's external prompt lookup: the proposer's drafts written on both
//     devices (set_draft_input), the same verify lists, server::accept_point_mass;
//   - spec 19d's DFlash drafter on device 1 (`--spec dflash --parallel pp`): draft_tap (the drafter's block, its sparse q
//     in the readback), the tap verify lists on both devices, the host's acceptance (greedy argmax match; sampled:
//     server::accept_sparse) - EngineAdapter::step_dflash line for line; the commit replays the drafter's projection;
//     the snapshots carry its ring and the store keys it (drafter_key).
//
// The engine's interface (what this calls; runtime::PipelineEngine and the test's fake):
//   reset, prefill(ids), ingest(ids), generate(1), pos, max_len, kBlock, kMaxDraft,
//   state_bytes, kv_bytes(n), kv_cache, save_state, load_state, save_kv, load_kv,
//   set_block_hook, pending, set_token(id), read_logits_into(host, rows),
//   read_draft_logits_into(host, i), host_rows(floats), mtp, max_verify_k, draft(k, pick),
//   draft_ids, verify(k), verify_ids, commit(j, t), set_draft_input(i, id); spec 19d: max_draft, draft_tap(k, T,
//   seed), tap_samples, drafter_key.
#include <algorithm>
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
#include <type_traits>
#include <utility>
#include <vector>

#include "runtime/buffer_sizes.h"   // runtime::KvCache (device-free)
#include "runtime/dflash/dflash_control.h"   // spec 19d: DFlashDraftOut (a plain struct)
#include "server/deps.h"
#include "server/spec_accept.h"

namespace cli::pp {

// Spec 19d: whether an engine has the tap drafter's calls (runtime::PipelineEngine does; Qwen4ExpEngine and the
// spec 16d test fake do not) - the DFlash path is compiled only for those that do.
template <class E, class = void>
struct HasTapDrafter : std::false_type {};
template <class E>
struct HasTapDrafter<E, std::void_t<decltype(std::declval<E&>().draft_tap(1u, 0.0f, uint64_t(0))),
                                    decltype(std::declval<const E&>().max_draft()),
                                    decltype(std::declval<const E&>().tap_samples()),
                                    decltype(std::declval<const E&>().drafter_key())>> : std::true_type {};

// Open-work 13: engines whose draft lists can stop on the head's probability (runtime::TpEngine; open-work 49: both
// Qwen3.8-Flash-Next engines) - the per-step K is compiled only for those.
template <class E, class = void>
struct HasDraftProbs : std::false_type {};
template <class E>
struct HasDraftProbs<E, std::void_t<decltype(std::declval<const E&>().draft_probs())>> : std::true_type {};

// Spec 19 v3c: engines whose commit can defer the drafter's projection to the next submit (runtime::TpEngine) - the
// DFlash iteration's commit then rides with the next draft.
template <class E, class = void>
struct HasDeferredCommit : std::false_type {};
template <class E>
struct HasDeferredCommit<E, std::void_t<decltype(std::declval<E&>().commit(0u, 0u, true))>> : std::true_type {};

// Spec 19 v3c: engines whose drafter can verify prompt-lookup drafts on its tap lists (runtime::TpEngine) - prompt lookup
// inside DFlash (spec 19f) is compiled only for those.
template <class E, class = void>
struct HasDrafterWindow : std::false_type {};
template <class E>
struct HasDrafterWindow<E, std::void_t<decltype(std::declval<const E&>().drafter_window())>> : std::true_type {};

template <class Engine>
struct PipelineEngineAdapterT : server::EngineIface {
  static constexpr bool kTap = HasTapDrafter<Engine>::value;

  // `vocab` (the sampler's mask: the tokenizer's count) and `V` (the logits row's width: the
  // model's whole vocabulary, model::Qwen35::kVocab). `mtp_k` > 0 needs the head loaded;
  // `lookup` verifies prompt-lookup drafts through the same verify lists (spec 19e).
  // `dflash` (spec 19d): draft with the engine's drafter on device 1, `mtp_k` its deepest K (--spec-max, 1..7).
  // `dflash_lookup` (spec 19 v3c, an engine with HasDrafterWindow): prompt lookup inside DFlash (spec 19f) -
  // EngineAdapter::step_drafts_or_many line for line.
  PipelineEngineAdapterT(Engine& engine, uint32_t vocab, uint32_t V, uint32_t mtp_k = 0, bool lookup = false,
                         bool dflash = false, bool dflash_lookup = false)
      : eng(engine), vocab_used(vocab), V_(V), k_(mtp_k), lookup_(lookup), df_(dflash), dflash_lookup_(dflash_lookup) {
    if (dflash_lookup_ && (!df_ || !HasDrafterWindow<Engine>::value))
      throw std::runtime_error("prompt lookup inside DFlash: this engine's drafter does not verify lookup drafts");
    if (vocab_used == 0 || vocab_used > V_)
      throw std::runtime_error("PipelineEngineAdapter: vocab_used " + std::to_string(vocab_used) +
                               " is outside the logits row (" + std::to_string(V_) + ")");
    if (df_) {
      if (lookup_) throw std::runtime_error("the DFlash drafter and external prompt-lookup drafts are one proposer each; pick one");
      if constexpr (kTap) {
        if (eng.max_draft() == 0 || eng.mtp())
          throw std::runtime_error("--spec dflash: the pipeline was not built with a drafter");
        if (k_ == 0 || k_ > eng.max_draft())
          throw std::runtime_error("--spec-max " + std::to_string(k_) + ": the drafter verifies K in 1.." +
                                   std::to_string(eng.max_draft()));
      } else {
        throw std::runtime_error("--spec dflash: this engine has no tap drafter");
      }
    } else if (k_ > Engine::kMaxDraft) {
      throw std::runtime_error("--spec mtp " + std::to_string(k_) + ": K is 0..3");
    }
    if (k_ > 0 && !df_ && !eng.mtp()) throw std::runtime_error("--spec mtp needs the model's MTP head loaded");
    if (lookup_ && k_ > 0) throw std::runtime_error("external prompt-lookup drafts and the MTP head are one proposer each; pick one");
    if (lookup_ && !eng.mtp())
      throw std::runtime_error("external prompt-lookup drafts need the verify lists, which the pipeline captures "
                               "with the MTP head loaded; this model has none");
    // Pinned readback rows (EngineAdapter's): q [kMaxDraft][V], then p [kMaxDraft + 1][V]; spec 19d: DFlash's p rows
    // alone, up to [8][V] (its q is the 16-candidate readback).
    rows_ = eng.host_rows(size_t(std::max<uint32_t>(2 * Engine::kMaxDraft + 1, kMaxIterK + 1)) * V_);
  }

  // The deepest K any proposer runs (spec 19d: DFlash's 7) - the per-K counters' size.
  static constexpr uint32_t kMaxIterK = 7;

  // b70-serve's reconfigure in place (cli/serve_reconfigure.h), between requests: the proposer this adapter was built
  // with at a K up to the one it was built with, or none (every argument 0 / false).
  void set_spec(uint32_t mtp_k, bool lookup, bool dflash, bool dflash_lookup) {
    k_ = mtp_k;
    lookup_ = lookup;
    df_ = dflash;
    dflash_lookup_ = dflash_lookup;
  }

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    first_ = false;
    eng.reset();
  }

  // EngineAdapter's measurement lines: the previous request's, when the next one starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    begin_request();
    eng.prefill(ids);
    first_ = true;
    first_host_ = false;
    gen_start_ = std::chrono::steady_clock::now();
  }

  // A new request (prefill(), or load_head() in its place): the previous one's summary lines, the counters reset.
  void begin_request() {
    rng_seeded = false;   // a new request: its first sampled step reseeds (EngineAdapter's rule)
    if (gen_tokens_ > 0) {
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
      std::print(stderr, "gen: {:d} tokens, {:.3f} ms, {:.4f} ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
      std::fflush(stderr);
    }
    if ((k_ > 0 || lookup_) && (mtp_iters > 0 || lk_iters_ > 0)) {
      std::print(stderr,
                 "{}: K {:d}, {:d} iterations, {:d}/{:d} drafts accepted ({:.3f}), iterations at "
                 "K 0/1/2/3: {:d}/{:d}/{:d}/{:d}",
                 lookup_ ? "lookup" : df_ ? "dflash" : "mtp", lookup_ ? Engine::kMaxDraft : k_,
                 (unsigned long long)mtp_iters, (unsigned long long)mtp_accepted,
                 (unsigned long long)mtp_drafted,
                 mtp_drafted ? double(mtp_accepted) / double(mtp_drafted) : 0.0,
                 (unsigned long long)mtp_iters_at_k[0], (unsigned long long)mtp_iters_at_k[1],
                 (unsigned long long)mtp_iters_at_k[2], (unsigned long long)mtp_iters_at_k[3]);
      if (df_)   // spec 19d: K 4..7 as well
        std::print(stderr, ", K 4/5/6/7: {:d}/{:d}/{:d}/{:d}", (unsigned long long)mtp_iters_at_k[4],
                   (unsigned long long)mtp_iters_at_k[5], (unsigned long long)mtp_iters_at_k[6],
                   (unsigned long long)mtp_iters_at_k[7]);
      if (dflash_lookup_)   // spec 19f: the lookup iterations (not in the counts above) and the ids they produced
        std::print(stderr, "; lookup: {:d} iterations, {:d}/{:d} drafts accepted, {:d} of {:d} ids",
                   (unsigned long long)lk_iters_, (unsigned long long)lk_accepted_,
                   (unsigned long long)lk_drafted_, (unsigned long long)lk_ids_, gen_tokens_);
      std::print(stderr, "\n");
    }
    mtp_iters = mtp_drafted = mtp_accepted = 0;
    lk_iters_ = lk_drafted_ = lk_accepted_ = lk_ids_ = 0;
    for (auto& n : mtp_iters_at_k) n = 0;
    gen_tokens_ = 0;
    flush_commit();
  }

  uint32_t step(const server::Sampling& sampling) override {
    sample_first(sampling);
    flush_commit();
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_pending(sampling, false, {&id, 1});   // id: not yet in `seen`
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }
  // PERF-91: EngineAdapter::pending_id - with no deferred commit, the engine's pending id (what step() and every
  // iteration below return first); a sampled request's first id is drawn here, as the call would draw it.
  std::optional<uint32_t> pending_id(const server::Sampling& sampling) override {
    if (pending_) return std::nullopt;
    sample_first(sampling);
    return eng.pending();
  }

  // --- spec 8: one speculative iteration (EngineAdapter::step_many) -------------------------
  uint32_t mtp_k() override { return k_; }

  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    sample_first(sampling);
    if constexpr (kTap) {
      if (df_) {   // spec 19d; spec 19 v3c: the commit's projection rides with the next draft where it can
        flush_commit(HasDeferredCommit<Engine>::value);
        return step_dflash(sampling, k_req);
      }
    }
    flush_commit();
    const uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    ++mtp_iters_at_k[k];
    if (k == 0) return {step(sampling)};   // K = 0, or the last positions of max_len: plain
    const uint32_t n = eng.pos();
    const uint32_t x = eng.pending();
    server::AcceptResult a{};
    if (sampling.greedy) {
      eng.draft(k);
      eng.verify(k);
      a = server::accept_greedy(eng.verify_ids(), eng.draft_ids().data(), k);
    } else {
      seed_rng(sampling);
      float* q = rows_;
      float* p = q + size_t(Engine::kMaxDraft) * V_;
      // RNG order (spec 8 Review Focus 4): the k draft samples, then accept_sampled's
      // uniforms, then its one correction / bonus draw.
      std::vector<uint32_t> hist{x};   // the pending id and the drafts so far: the repetition penalty's tail
      eng.draft(k, [&](uint32_t i) {
        eng.read_draft_logits_into(q + size_t(i) * V_, i);
        hist.push_back(server::sample_draft(q + size_t(i) * V_, vocab_used, sampling, rng, hist));
        return hist.back();
      });
      eng.verify(k);
      eng.read_logits_into(p, k + 1);
      a = server::accept_sampled(p, q, eng.draft_ids().data(), k, vocab_used, V_, sampling, rng, hist);
    }
    return spec_run(n, x, k, a, eng.draft_ids());
  }

  // Open-work 13: EngineAdapter::step_chosen - the head's drafts chosen one at a time, greedy.
  bool chooses_drafts() override {
    if constexpr (HasDraftProbs<Engine>::value) return !df_ && k_ > 0 && eng.draft_probs();
    return false;
  }
  std::vector<uint32_t> step_chosen(const server::Sampling& sampling, uint32_t k_req,
                                    const DraftChoice& choose) override {
    if constexpr (HasDraftProbs<Engine>::value) {
      if (!sampling.greedy) throw std::logic_error("PipelineEngineAdapterT::step_chosen: greedy requests only");
      flush_commit();
      const uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
      if (k == 0) {
        ++mtp_iters_at_k[0];
        return {step(sampling)};
      }
      const uint32_t n = eng.pos();
      const uint32_t x = eng.pending();
      int v = -1;
      eng.draft(k, {}, [&](uint32_t i, float p) {
        v = choose(i + 1, p);
        return v < 0;
      });
      const uint32_t kv = v < 0 ? static_cast<uint32_t>(eng.draft_ids().size()) : static_cast<uint32_t>(v);
      ++mtp_iters_at_k[kv];
      eng.verify(kv);
      const server::AcceptResult a = server::accept_greedy(eng.verify_ids(), eng.draft_ids().data(), kv);
    return spec_run(n, x, kv, a, eng.draft_ids());
    } else {
      (void)choose;
      return step_many(sampling, k_req);   // chooses_drafts() is false: the server never calls this
    }
  }

  // The iteration's deferred commit and counters - an MTP or DFlash iteration's, or (`lookup`) a lookup run's: `drafts`'
  // first k are the run, its first a.accepted are kept. Returns x and the kept drafts.
  std::vector<uint32_t> spec_run(uint32_t n, uint32_t x, uint32_t k, const server::AcceptResult& a,
                                 const std::vector<uint32_t>& drafts, bool lookup = false) {
    pending_ = true;
    pending_pos_ = n;
    pending_j_ = a.accepted;
    pending_next_ = a.next_token;
    pending_drafts_.assign(drafts.begin(), drafts.begin() + k);
    std::vector<uint32_t> out{x};
    out.insert(out.end(), drafts.begin(), drafts.begin() + a.accepted);
    gen_tokens_ += static_cast<uint32_t>(out.size());
    if (lookup) {
      lk_drafted_ += k;
      lk_accepted_ += a.accepted;
      lk_ids_ += out.size();
      pending_lookup_ = true;   // a truncate_to of this run takes its dropped ids off lk_ids_
    } else {
      ++mtp_iters;
      mtp_drafted += k;
      mtp_accepted += a.accepted;
    }
    return out;
  }

  // --- spec 19d: one DFlash iteration across the split (EngineAdapter::step_dflash) ---------------------------
  // A member template, so an explicit instantiation of the adapter over an engine without a drafter
  // (tests/cli/qwen4exp_adapter_compile.cc) does not instantiate it.
  template <class E = Engine>
  std::vector<uint32_t> step_dflash(const server::Sampling& sampling, uint32_t k_req) {
    static_assert(HasTapDrafter<E>::value, "step_dflash needs an engine with a tap drafter");
    uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    if (!sampling.greedy && !eng.tap_samples()) k = 0;   // a v1 drafter drafts greedily only
    ++mtp_iters_at_k[k];
    if (k == 0) return {step(sampling)};
    const uint32_t n = eng.pos();
    const uint32_t x = eng.pending();
    server::AcceptResult a{};
    std::vector<uint32_t> drafts;
    // Spec 19 v3c: the readback is read after verify(k) - under TP the drafter's tails (the selector) ride with it.
    if (sampling.greedy) {
      const runtime::dflash::DFlashDraftOut& o = eng.draft_tap(k, 0.0f, 0);
      eng.verify(k);
      drafts.assign(o.ids, o.ids + k);
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
    } else {
      // EngineAdapter::dflash_sampled's RNG order: the walk's Gumbel key, the uniforms, the correction draw.
      seed_rng(sampling);
      const uint64_t seed = rng();
      const float T = server::dflash_temperature(sampling);
      const runtime::dflash::DFlashDraftOut& o = eng.draft_tap(k, T, seed);
      eng.verify(k);
      drafts.assign(o.ids, o.ids + k);
      server::SparseQ q[kMaxIterK];
      for (uint32_t j = 0; j < k; ++j)
        q[j] = server::dflash_q(o.cand[j], o.scores[j], runtime::dflash::DFlashDraftOut::kTopK, T);
      float* p = rows_;
      eng.read_logits_into(p, k + 1);
      std::vector<uint32_t> hist{x};
      hist.insert(hist.end(), drafts.begin(), drafts.end());
      a = server::accept_sparse(p, q, drafts.data(), k, vocab_used, V_, sampling, rng, hist);
    }
    return spec_run(n, x, k, a, drafts);
  }

  // --- spec 19 v3c: prompt lookup inside DFlash (EngineAdapter::step_drafts_or_many, spec 19f) ----------------
  // Lookup drafts d_1..d_k (k <= --spec-max and max_verify_k) go into both cards' cur_token[1..k] and through the tap
  // verify list (the drafter's draft list does not run); the commit replays the drafter's projection of the committed
  // rows, so the ring stays current and the exit to DFlash is warm. No drafts: one DFlash iteration.
  uint32_t dflash_window() override {
    if constexpr (HasDrafterWindow<Engine>::value) return dflash_lookup_ ? eng.drafter_window() : 0;
    return 0;
  }
  std::vector<uint32_t> step_drafts_or_many(const server::Sampling& sampling, uint32_t k_dflash,
                                            const Proposer& propose) override {
    if constexpr (HasDrafterWindow<Engine>::value && kTap) {
      if (!dflash_lookup_) throw std::logic_error("PipelineEngineAdapter::step_drafts_or_many without prompt lookup inside DFlash");
      sample_first(sampling);
      flush_commit(HasDeferredCommit<Engine>::value);
      const uint32_t x = eng.pending();
      const std::vector<uint32_t> d = propose(x);
      if (d.empty()) return step_dflash(sampling, k_dflash);
      const uint32_t k = std::min<uint32_t>({static_cast<uint32_t>(d.size()), k_, eng.max_verify_k()});
      ++lk_iters_;
      if (k == 0) {   // the last positions of max_len: one plain step
        ++lk_ids_;
        return {step(sampling)};
      }
      const uint32_t n = eng.pos();
      for (uint32_t i = 0; i < k; ++i) eng.set_draft_input(i, d[i]);
      eng.verify(k);
      server::AcceptResult a{};
      if (sampling.greedy) {
        a = server::accept_greedy(eng.verify_ids(), d.data(), k);
      } else {
        seed_rng(sampling);
        float* p = rows_;
        eng.read_logits_into(p, k + 1);
        std::vector<uint32_t> hist{x};
        hist.insert(hist.end(), d.begin(), d.begin() + k);
        a = server::accept_point_mass(p, d.data(), k, vocab_used, V_, sampling, rng, hist);
      }
      return spec_run(n, x, k, a, d, true);
    } else {
      throw std::logic_error("PipelineEngineAdapter::step_drafts_or_many: no tap drafter");
    }
  }

  // --- spec 19e: drafts proposed outside the engine (EngineAdapter::step_drafts) ------------
  uint32_t verify_k() override { return lookup_ ? Engine::kMaxDraft : 0; }

  std::vector<uint32_t> step_drafts(const server::Sampling& sampling, const Proposer& propose) override {
    sample_first(sampling);
    flush_commit();
    const uint32_t x = eng.pending();
    const std::vector<uint32_t> d = propose(x);
    const uint32_t k = std::min<uint32_t>({static_cast<uint32_t>(d.size()), Engine::kMaxDraft,
                                           eng.max_verify_k()});
    ++mtp_iters_at_k[k];
    if (k == 0) return {step(sampling)};   // no match, or the last positions of max_len
    const uint32_t n = eng.pos();
    for (uint32_t i = 0; i < k; ++i) eng.set_draft_input(i, d[i]);
    eng.verify(k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      a = server::accept_greedy(eng.verify_ids(), d.data(), k);
    } else {
      seed_rng(sampling);
      float* p = rows_ + size_t(Engine::kMaxDraft) * V_;
      eng.read_logits_into(p, k + 1);
      std::vector<uint32_t> hist{x};
      hist.insert(hist.end(), d.begin(), d.begin() + k);
      a = server::accept_point_mass(p, d.data(), k, vocab_used, V_, sampling, rng, hist);
    }
    return spec_run(n, x, k, a, d);
  }

  // Keep x_n, d_1 .. d_m of the last iteration (EngineAdapter::truncate_to).
  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.pos()) return;
      throw std::logic_error("PipelineEngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("PipelineEngineAdapter::truncate_to: pos " + std::to_string(p) +
                             " is outside the last run [" + std::to_string(pending_pos_ + 1) + ", " +
                             std::to_string(pending_pos_ + pending_j_ + 1) + "]");
    const uint32_t m = p - pending_pos_ - 1;
    gen_tokens_ -= pending_j_ - m;
    if (pending_lookup_) lk_ids_ -= pending_j_ - m;   // spec 19f: the lookup share counts kept ids only
    pending_ = false;
    pending_lookup_ = false;
    eng.commit(m, m < pending_j_ ? pending_drafts_[m] : pending_next_);
  }

  // `defer` (spec 19 v3c, an engine with HasDeferredCommit): the drafter's projection stays pending for the next submit.
  void flush_commit(bool defer = false) {
    if (!pending_) return;
    pending_ = false;
    pending_lookup_ = false;
    if constexpr (HasDeferredCommit<Engine>::value)
      eng.commit(pending_j_, pending_next_, defer);
    else
      eng.commit(pending_j_, pending_next_);
  }

  void seed_rng(const server::Sampling& sampling) {
    if (!rng_seeded) {
      rng.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded = true;
    }
  }

  // --- spec 7: the snapshots, forwarded (a deferred commit first, as EngineAdapter) ---------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return eng.kv_cache() == runtime::KvCache::Int8 ? 1 : 0; }
  uint64_t drafter_key() override {   // spec 19d: 0 without a drafter
    if constexpr (kTap) return eng.drafter_key();
    return 0;
  }
  void save_state(void* host) override {
    flush_commit();
    eng.save_state(host);
  }
  void load_state(const void* host, uint32_t p) override {
    pending_ = false;
    first_ = false;
    eng.load_state(host, p);
  }
  // PERF-92 (EngineAdapter's): the prompt pass's last logits row (fp32, V) then the pending id - read in the prompt-end
  // hook, after the head on the last card (pp) / both cards' halves merged (tp) has left both. load_head puts the id back
  // on every card and keeps the row on the host for sample_first. What MTP drafts from (the head's hidden row and KV
  // layer) and the DFlash drafter's ring are in the state and the KV already.
  size_t head_bytes() override { return size_t(V_) * sizeof(float) + sizeof(uint32_t); }
  void save_head(void* host) override {
    auto* h = static_cast<uint8_t*>(host);
    eng.read_logits_into(reinterpret_cast<float*>(h), 1);
    const uint32_t id = eng.pending();
    std::memcpy(h + size_t(V_) * sizeof(float), &id, sizeof(uint32_t));
  }
  void load_head(const void* host) override {
    begin_request();
    const auto* h = static_cast<const uint8_t*>(host);
    std::memcpy(rows_ + size_t(Engine::kMaxDraft) * V_, h, size_t(V_) * sizeof(float));   // sample_into_pending's row
    uint32_t id;
    std::memcpy(&id, h + size_t(V_) * sizeof(float), sizeof(uint32_t));
    eng.set_token(id);
    first_ = first_host_ = true;
    gen_start_ = std::chrono::steady_clock::now();
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override {
    flush_commit();
    eng.save_kv(b, e, host);
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override {
    rng_seeded = false;
    flush_commit();
    eng.ingest(ids);
    first_ = true;
    first_host_ = false;
  }

  // EngineAdapter::sample_first: after a prefill or ingest the pending id is the argmax of the prompt's last row, which
  // the engine leaves in its logits row (read_logits_into(host, 1)); a sampled request draws it from that row.
  // PERF-92: after load_head the row is already on the host (`first_host_`).
  void sample_first(const server::Sampling& sampling) {
    if (!first_) return;
    first_ = false;
    const bool host = std::exchange(first_host_, false);
    if (!sampling.greedy) sample_into_pending(sampling, host);
  }

  // The step's logits row (device 1's), filtered and drawn as EngineAdapter's sample(); the
  // id replaces the argmax as the pending id on both devices. The RNG seeds itself once per request.
  // `host`: the row is in place already (load_head's).
  void sample_into_pending(const server::Sampling& sampling, bool host = false, std::span<const uint32_t> tail = {}) {
    seed_rng(sampling);
    float* row = rows_ + size_t(Engine::kMaxDraft) * V_;
    if (!host) eng.read_logits_into(row, 1);
    eng.set_token(server::draw(server::filter_probs(row, vocab_used, sampling, tail), rng));
  }

  Engine& eng;
  uint32_t vocab_used;

 private:
  uint32_t V_;
  float* rows_ = nullptr;   // the engine's pinned rows: q [kMaxDraft][V] then p [kMaxDraft + 1][V]
  std::mt19937_64 rng;
  bool rng_seeded = false;
  bool first_ = false;   // the pending id is still the prompt pass's argmax
  bool first_host_ = false;   // PERF-92: that row is in the sample row of rows_ (load_head)
  uint32_t gen_tokens_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  uint32_t k_ = 0;
  bool lookup_ = false;
  bool df_ = false;   // spec 19d: step_many drafts with the engine's drafter
  bool dflash_lookup_ = false, pending_lookup_ = false;   // spec 19 v3c: prompt lookup inside DFlash (spec 19f)
  uint64_t lk_iters_ = 0, lk_drafted_ = 0, lk_accepted_ = 0, lk_ids_ = 0;
  bool pending_ = false;
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t mtp_iters = 0, mtp_drafted = 0, mtp_accepted = 0;
  uint64_t mtp_iters_at_k[kMaxIterK + 1] = {};
};

}  // namespace cli::pp
