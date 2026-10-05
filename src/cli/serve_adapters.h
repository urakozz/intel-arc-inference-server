#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/prefill/backend.h"
#include "server/deps.h"
#include "server/prefix_cache.h"
#include "l0/memory.h"
#include "server/spec_accept.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"

struct TokAdapter : server::TokIface {
  explicit TokAdapter(const std::string& json) : t(json) {}

  std::vector<uint32_t> encode(std::string_view text) override { return t.encode(text, false); }
  std::string decode(const std::vector<uint32_t>& ids) override { return t.decode(ids, false); }

  struct S : server::StreamerIface {
    explicit S(const tok::Tokenizer& tokenizer) : st(tokenizer) {}

    std::string push(uint32_t id) override { return st.push(id); }
    std::string flush() override { return st.flush(); }

    tok::Streamer st;
  };

  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(t); }
  uint32_t vocab_used() override { return t.vocab_size(); }

  tok::Tokenizer t;
};

struct TemplateAdapter : server::TemplateIface {
  explicit TemplateAdapter(const std::string& dir) : tmpl(dir) {}

  std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                     bool enable_thinking) override {
    return tmpl.render(messages, tools, enable_thinking);
  }

  chat::Template tmpl;
};

// Host sampling (spec §3.5, plan 7d Task 5): temperature/top-k/top-p over one
// replay's logits row, seeded and reproducible. The filter (top-k, softmax at
// temperature, top-p, the `vocab_used` mask) is `server::filter_probs`
// (src/server/spec_accept.h), shared with the speculative acceptance rule (spec 8
// plan 8c) so p and q are filtered exactly as this row is;
// `std::discrete_distribution` draws inside the kept prefix.
inline uint32_t sample(const float* logits, uint32_t vocab_used, const server::Sampling& s,
                       std::mt19937_64& rng) {
  return server::draw(server::filter_probs(logits, vocab_used, s), rng);
}

struct EngineAdapter : server::EngineIface {
  // `imm` and `host_logits` are created ONCE here, not per token (Task 5
  // Step 3): the immediate command list and the readback buffer are reused
  // across every sampled step for the life of the adapter.
  // `mtp_k` > 0 (spec 8, `b70-serve --mtp K`) needs an engine with the head loaded.
  explicit EngineAdapter(runtime::Engine& engine, uint32_t vocab, uint32_t mtp_k = 0)
      : eng(engine),
        vocab_used(vocab),
        imm(l0::CmdList::immediate(engine.context())),
        host_logits(model::Qwen35::kVocab),
        k_(mtp_k) {
    if (k_ > runtime::Engine::kMaxDraft)
      throw std::runtime_error("--mtp " + std::to_string(k_) + ": K is 0..3");
    if (k_ > 0 && !eng.mtp()) throw std::runtime_error("--mtp needs the model's MTP head loaded");
    if (k_ > 0) {
      // Pinned readback rows (probe-mtp §3: pageable costs 2.5x): q [K][V], p [K+1][V].
      spec_logits_ = std::make_unique<l0::Mem>(
          engine.context(), l0::MemKind::Host,
          size_t(2 * runtime::Engine::kMaxDraft + 1) * model::Qwen35::kVocab * sizeof(float));
    }
  }

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    eng.reset();
  }

  // Task 5 Step 4's measurement hook. `src/server` is protected (it has no
  // "generation finished" callback in Deps), so the only place left to time
  // just the generation loop -- prefill excluded, HTTP excluded -- is here,
  // where every request's `prefill()` and `step()` calls actually land. A
  // request's summary line is printed at the START of the NEXT request's
  // prefill() (the earliest point at which this adapter knows the previous
  // one's generation is over: nothing calls prefill() again until the
  // previous request's last step() has returned to server.cc's loop and a
  // new one has been accepted). One request is therefore always exactly one
  // line, delayed by one request -- `tools/serve_bench.sh`'s host-sampling
  // measurement sends one trailing no-op request to flush the last line.
  void prefill(const std::vector<uint32_t>& ids) override {
    if (gen_tokens_ > 0) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - gen_start_)
                            .count();
      std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms,
                   ms / gen_tokens_);
      std::fflush(stderr);
    }
    if (k_ > 0 && mtp_iters > 0)
      std::fprintf(stderr,
                   "mtp: K %u, %llu iterations, %llu/%llu drafts accepted (%.3f), iterations at "
                   "K 0/1/2/3: %llu/%llu/%llu/%llu\n",
                   k_, (unsigned long long)mtp_iters, (unsigned long long)mtp_accepted,
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
    if (!sampling.greedy) sample_into_control(sampling);
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }

  // --- spec 8 §3.3 / §3.6: one speculative iteration (plan 8c Task 2) ------------------
  //
  // At pos = n with x_n pending: draft(k) -> verify(k) -> accept on the host. Returns
  // x_n, d_1 .. d_j. The commit (pos = n + j + 1, pending = next) is DEFERRED to the
  // next call that touches the engine, so that truncate_to() can still commit fewer
  // (the verify rows' GDN slots and hidden rows stay valid until the next verify).
  uint32_t mtp_k() override { return k_; }

  // `k_req` (spec 8 §10): the iteration's K, <= mtp_k(); `--mtp auto` varies it per
  // iteration, 0 is one plain step.
  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    flush_commit();
    const uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    ++mtp_iters_at_k[k];
    if (k == 0) return {step(sampling)};   // K = 0, or the last positions of max_len: plain
    const uint32_t n = eng.pos();
    auto* control = eng.buffers().control.as<runtime::Control>();
    const uint32_t x = control->cur_token[0];
    server::AcceptResult a{};
    constexpr size_t V = model::Qwen35::kVocab;
    if (sampling.greedy) {
      eng.draft(k);
      eng.verify(k);
      a = server::accept_greedy(eng.verify_ids(), eng.draft_ids().data(), k);
    } else {
      seed_rng(sampling);
      float* q = spec_logits_->as<float>();
      float* p = q + runtime::Engine::kMaxDraft * V;
      // RNG order (Review Focus 4): the k draft samples, then accept_sampled's
      // uniforms, then its one correction/bonus draw.
      eng.draft(k, [&](uint32_t i) {
        imm.copy(q + i * V, eng.mtp_logits_device() + i * V, V * sizeof(float));
        return server::sample_draft(q + i * V, vocab_used, sampling, rng);
      });
      eng.verify(k);
      imm.copy(p, eng.verify_logits_device(), (k + 1) * V * sizeof(float));
      a = server::accept_sampled(p, q, eng.draft_ids().data(), k, vocab_used, V, sampling, rng);
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

  // Keep x_n, d_1 .. d_m of the last iteration (pos = n + m + 1), pending d_{m+1} (or the
  // accepted run's next id when m = j): the session is what m + 1 plain steps leave.
  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.pos()) return;
      throw std::logic_error("EngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("EngineAdapter::truncate_to: pos " + std::to_string(p) +
                             " is outside the last run [" + std::to_string(pending_pos_ + 1) +
                             ", " + std::to_string(pending_pos_ + pending_j_ + 1) + "]");
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

  // Spec 7 (plan 7c): plan 7b's snapshot calls, forwarded. With MTP (spec 8) a
  // speculative run's commit may still be deferred: a save or a replay commits it first
  // (the request-end snapshot follows a run the server kept whole), a restore discards it
  // as reset() does.
  uint32_t block() override { return runtime::Engine::kBlock; }
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

  // Reads back the replay's fp32 logits row (`eng.buffers().logits`, [1][
  // model::Qwen35::kVocab] = 993 KB), samples over the first `vocab_used` of
  // them, and writes the sampled id into `cur_token[0]` -- exactly the ingest
  // protocol `Control` already supports (spec §3.5) -- so the NEXT
  // `generate(1)` embeds it and returns it as the following step's pending
  // id. `generate(1)` above already returned the id the *previous* step
  // sampled; this call replaces the argmax the device just wrote for the
  // step after this one.
  //
  // The RNG seeds itself once, lazily, on the first sampled token this
  // adapter ever serves: from the request's `seed` if it gave one, else from
  // `std::random_device`. It is not reseeded per request after that (the
  // server is single-stream, so "per request" would mean "every token of
  // every request after the first restarts the same draw", which is not what
  // real sampling wants); a `seed`d request run first after server start is
  // reproducible from that seed, which is what `sampling_test.cc` grades at
  // the `sample()` level.
  void sample_into_control(const server::Sampling& sampling) {
    seed_rng(sampling);
    imm.copy(host_logits.data(), eng.buffers().logits.ptr(),
             model::Qwen35::kVocab * sizeof(float));
    const uint32_t id = sample(host_logits.data(), vocab_used, sampling, rng);
    eng.buffers().control.as<runtime::Control>()->cur_token[0] = id;
  }

  runtime::Engine& eng;
  uint32_t vocab_used;
  l0::CmdList imm;
  std::vector<float> host_logits;
  std::mt19937_64 rng;
  bool rng_seeded = false;
  uint32_t gen_tokens_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  // Spec 8.
  uint32_t k_ = 0;
  std::unique_ptr<l0::Mem> spec_logits_;   // pinned q rows then p rows, MTP only
  bool pending_ = false;
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t mtp_iters = 0, mtp_drafted = 0, mtp_accepted = 0;   // since the last prefill
  uint64_t mtp_iters_at_k[runtime::Engine::kMaxDraft + 1] = {};  // step_many calls per K
};

// Spec 7 §3.1: the prefix cache's pinned host memory. ONE l0::MemKind::Host allocation of
// the whole budget at server start (the P0 probe measured ~100 ms per GiB to allocate, too
// slow per entry), carved first-fit in 64 KiB units with coalescing frees. Host-resident
// and device-visible, so the engine's snapshot copies run at the probe's 12-14 GB/s.
struct PinnedAlloc : server::HostAlloc {
  static constexpr size_t kUnit = size_t(64) << 10;
  PinnedAlloc(l0::Context& ctx, size_t bytes)
      : mem(ctx, l0::MemKind::Host, bytes / kUnit * kUnit, kUnit) {
    free_[0] = mem.size();
  }
  void* alloc(size_t bytes) override {
    const size_t n = (bytes + kUnit - 1) / kUnit * kUnit;
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second < n) continue;
      const size_t off = it->first, len = it->second;
      free_.erase(it);
      if (len > n) free_[off + n] = len - n;
      return mem.as<uint8_t>() + off;
    }
    return nullptr;
  }
  void free(void* p, size_t bytes) override {
    const size_t n = (bytes + kUnit - 1) / kUnit * kUnit;
    size_t off = size_t(static_cast<uint8_t*>(p) - mem.as<uint8_t>()), len = n;
    auto next = free_.lower_bound(off);
    if (next != free_.end() && next->first == off + len) {
      len += next->second;
      next = free_.erase(next);
    }
    if (next != free_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == off) {
        off = prev->first;
        len += prev->second;
        free_.erase(prev);
      }
    }
    free_[off] = len;
  }
  l0::Mem mem;
  std::map<size_t, size_t> free_;   // offset -> length
};
