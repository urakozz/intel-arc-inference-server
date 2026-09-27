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
// replay's logits row, seeded and reproducible. `logits` holds `vocab_used`
// or more entries; only the first `vocab_used` are ever looked at, which is
// the mask "ids >= vocab_used never sampled" the spec asks for -- there is no
// separate masking step because the candidate set never includes them.
//
// - top-k: `std::partial_sort` picks the k highest-logit ids (k = min(top_k,
//   vocab_used); top_k == 0 means "no cap", i.e. k = vocab_used).
// - softmax over exactly those k, at `temperature` (<= 0 treated as 1.0, the
//   checkpoint's own default, rather than dividing by zero).
// - top-p: walk the (already logit-sorted, so probability-sorted) k in order,
//   accumulate, cut after the first prefix whose cumulative mass >= top_p;
//   always keeps at least one candidate.
// - `std::discrete_distribution` draws the index inside the kept prefix.
inline uint32_t sample(const float* logits, uint32_t vocab_used, const server::Sampling& s,
                       std::mt19937_64& rng) {
  const uint32_t k = std::min(s.top_k == 0 ? vocab_used : s.top_k, vocab_used);
  std::vector<uint32_t> idx(vocab_used);
  for (uint32_t i = 0; i < vocab_used; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [logits](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
  idx.resize(k);

  const float temperature = s.temperature > 0.0f ? s.temperature : 1.0f;
  const float top_logit = logits[idx[0]];
  std::vector<double> probs(k);
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) {
    probs[i] = std::exp(static_cast<double>((logits[idx[i]] - top_logit) / temperature));
    sum += probs[i];
  }
  for (double& p : probs) p /= sum;

  double cumulative = 0.0;
  uint32_t kept = k;
  const double top_p = std::clamp(static_cast<double>(s.top_p), 0.0, 1.0);
  for (uint32_t i = 0; i < k; ++i) {
    cumulative += probs[i];
    if (cumulative >= top_p) {
      kept = i + 1;
      break;
    }
  }
  kept = std::max<uint32_t>(kept, 1);
  probs.resize(kept);

  std::discrete_distribution<uint32_t> draw(probs.begin(), probs.end());
  return idx[draw(rng)];
}

struct EngineAdapter : server::EngineIface {
  // `imm` and `host_logits` are created ONCE here, not per token (Task 5
  // Step 3): the immediate command list and the readback buffer are reused
  // across every sampled step for the life of the adapter.
  explicit EngineAdapter(runtime::Engine& engine, uint32_t vocab)
      : eng(engine),
        vocab_used(vocab),
        imm(l0::CmdList::immediate(engine.context())),
        host_logits(model::Qwen35::kVocab) {}

  void reset() override { eng.reset(); }

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
    gen_tokens_ = 0;
    eng.prefill(ids);
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_control(sampling);
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  // Spec 7 (plan 7c): plan 7b's snapshot calls, forwarded.
  uint32_t block() override { return runtime::Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  void save_state(void* host) override { eng.save_state(host); }
  void load_state(const void* host, uint32_t p) override { eng.load_state(host, p); }
  void save_kv(uint32_t b, uint32_t e, void* host) override { eng.save_kv(b, e, host); }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  uint32_t pending() override {
    return eng.buffers().control.as<runtime::Control>()->cur_token[0];
  }
  void set_pending(uint32_t id) override {
    eng.buffers().control.as<runtime::Control>()->cur_token[0] = id;
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
    if (!rng_seeded) {
      rng.seed(sampling.has_seed ? sampling.seed : std::random_device{}());
      rng_seeded = true;
    }
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
