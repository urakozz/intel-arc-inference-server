#pragma once
// Spec 16d: b70-serve --pp 2 - runtime::PipelineEngine (the model's layers over two B70s) as a
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
//     the same bytes whichever of --pp 1 / 2 wrote it. The block hook on the pipeline's
//     prefill (from the shadows mid-prompt), ingest for --prefix-split-last;
//   - spec 8's MTP (`--mtp K|auto`, `--spec mtp`): draft on device 1, verify on both, the
//     host's acceptance (greedy argmax match; sampled: server::accept_sampled over the q rows
//     of the drafts and the p rows of the verify), the commit deferred exactly as
//     EngineAdapter defers it (truncate_to commits fewer);
//   - spec 19e's prompt lookup (`--spec lookup`): the proposer's drafts written on both
//     devices (set_draft_input), the same verify lists, server::accept_point_mass.
//
// The engine's interface (what this calls; runtime::PipelineEngine and the test's fake):
//   reset, prefill(ids), ingest(ids), generate(1), pos, max_len, kBlock, kMaxDraft,
//   state_bytes, kv_bytes(n), kv_cache, save_state, load_state, save_kv, load_kv,
//   set_block_hook, pending, set_token(id), read_logits_into(host, rows),
//   read_draft_logits_into(host, i), host_rows(floats), mtp, max_verify_k, draft(k, pick),
//   draft_ids, verify(k), verify_ids, commit(j, t), set_draft_input(i, id).
#include <algorithm>
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

namespace cli::pp {

template <class Engine>
struct PipelineEngineAdapterT : server::EngineIface {
  // `vocab` (the sampler's mask: the tokenizer's count) and `V` (the logits row's width: the
  // model's whole vocabulary, model::Qwen35::kVocab). `mtp_k` > 0 needs the head loaded;
  // `lookup` verifies prompt-lookup drafts through the same verify lists (spec 19e).
  PipelineEngineAdapterT(Engine& engine, uint32_t vocab, uint32_t V, uint32_t mtp_k = 0, bool lookup = false)
      : eng(engine), vocab_used(vocab), V_(V), k_(mtp_k), lookup_(lookup) {
    if (vocab_used == 0 || vocab_used > V_)
      throw std::runtime_error("PipelineEngineAdapter: vocab_used " + std::to_string(vocab_used) +
                               " is outside the logits row (" + std::to_string(V_) + ")");
    if (k_ > Engine::kMaxDraft)
      throw std::runtime_error("--mtp " + std::to_string(k_) + ": K is 0..3");
    if (k_ > 0 && !eng.mtp()) throw std::runtime_error("--mtp needs the model's MTP head loaded");
    if (lookup_ && k_ > 0) throw std::runtime_error("--spec lookup and --mtp: pick one proposer");
    if (lookup_ && !eng.mtp())
      throw std::runtime_error("--spec lookup needs the verify lists, which the pipeline captures "
                               "with the MTP head loaded; this model has none");
    // Pinned readback rows (EngineAdapter's): q [kMaxDraft][V], then p [kMaxDraft + 1][V].
    rows_ = eng.host_rows(size_t(2 * Engine::kMaxDraft + 1) * V_);
  }

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    eng.reset();
  }

  // EngineAdapter's measurement lines: the previous request's, when the next one starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    if (gen_tokens_ > 0) {
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
      std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
      std::fflush(stderr);
    }
    if ((k_ > 0 || lookup_) && mtp_iters > 0)
      std::fprintf(stderr,
                   "%s: K %u, %llu iterations, %llu/%llu drafts accepted (%.3f), iterations at "
                   "K 0/1/2/3: %llu/%llu/%llu/%llu\n",
                   lookup_ ? "lookup" : "mtp", lookup_ ? Engine::kMaxDraft : k_,
                   (unsigned long long)mtp_iters, (unsigned long long)mtp_accepted,
                   (unsigned long long)mtp_drafted,
                   mtp_drafted ? double(mtp_accepted) / double(mtp_drafted) : 0.0,
                   (unsigned long long)mtp_iters_at_k[0], (unsigned long long)mtp_iters_at_k[1],
                   (unsigned long long)mtp_iters_at_k[2], (unsigned long long)mtp_iters_at_k[3]);
    mtp_iters = mtp_drafted = mtp_accepted = 0;
    for (auto& n : mtp_iters_at_k) n = 0;
    gen_tokens_ = 0;
    flush_commit();
    eng.prefill(ids);
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    flush_commit();
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_pending(sampling);
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }

  // --- spec 8: one speculative iteration (EngineAdapter::step_many) -------------------------
  uint32_t mtp_k() override { return k_; }

  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
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
      eng.draft(k, [&](uint32_t i) {
        eng.read_draft_logits_into(q + size_t(i) * V_, i);
        return server::sample_draft(q + size_t(i) * V_, vocab_used, sampling, rng);
      });
      eng.verify(k);
      eng.read_logits_into(p, k + 1);
      a = server::accept_sampled(p, q, eng.draft_ids().data(), k, vocab_used, V_, sampling, rng);
    }
    pending_ = true;
    pending_pos_ = n;
    pending_j_ = a.accepted;
    pending_next_ = a.next_token;
    pending_drafts_ = eng.draft_ids();
    ++mtp_iters;
    mtp_drafted += k;
    mtp_accepted += a.accepted;
    std::vector<uint32_t> out{x};
    out.insert(out.end(), pending_drafts_.begin(), pending_drafts_.begin() + a.accepted);
    gen_tokens_ += static_cast<uint32_t>(out.size());
    return out;
  }

  // --- spec 19e: drafts proposed outside the engine (EngineAdapter::step_drafts) ------------
  uint32_t verify_k() override { return lookup_ ? Engine::kMaxDraft : 0; }

  std::vector<uint32_t> step_drafts(const server::Sampling& sampling, const Proposer& propose) override {
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
      a = server::accept_point_mass(p, d.data(), k, vocab_used, V_, sampling, rng);
    }
    pending_ = true;
    pending_pos_ = n;
    pending_j_ = a.accepted;
    pending_next_ = a.next_token;
    pending_drafts_.assign(d.begin(), d.begin() + k);
    ++mtp_iters;
    mtp_drafted += k;
    mtp_accepted += a.accepted;
    std::vector<uint32_t> out{x};
    out.insert(out.end(), d.begin(), d.begin() + a.accepted);
    gen_tokens_ += static_cast<uint32_t>(out.size());
    return out;
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
    pending_ = false;
    eng.commit(m, m < pending_j_ ? pending_drafts_[m] : pending_next_);
  }

  void flush_commit() {
    if (!pending_) return;
    pending_ = false;
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
  void save_state(void* host) override {
    flush_commit();
    eng.save_state(host);
  }
  void load_state(const void* host, uint32_t p) override {
    pending_ = false;
    eng.load_state(host, p);
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
  }

  // The step's logits row (device 1's), filtered and drawn as EngineAdapter's sample(); the
  // id replaces the argmax as the pending id on both devices. The RNG seeds itself once.
  void sample_into_pending(const server::Sampling& sampling) {
    seed_rng(sampling);
    float* row = rows_ + size_t(Engine::kMaxDraft) * V_;
    eng.read_logits_into(row, 1);
    eng.set_token(server::draw(server::filter_probs(row, vocab_used, sampling), rng));
  }

  Engine& eng;
  uint32_t vocab_used;

 private:
  uint32_t V_;
  float* rows_ = nullptr;   // the engine's pinned rows: q [kMaxDraft][V] then p [kMaxDraft + 1][V]
  std::mt19937_64 rng;
  bool rng_seeded = false;
  uint32_t gen_tokens_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  uint32_t k_ = 0;
  bool lookup_ = false;
  bool pending_ = false;
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t mtp_iters = 0, mtp_drafted = 0, mtp_accepted = 0;
  uint64_t mtp_iters_at_k[4] = {};
};

}  // namespace cli::pp
