#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "server/batch_engine.h"
#include "server/deps.h"

struct MockTok : server::TokIface {
  std::vector<std::string> vocab;
  std::map<std::string, uint32_t> index;

  uint32_t id_of(const std::string& word) {
    if (word == "<|im_end|>") return 248046;
    const auto it = index.find(word);
    if (it != index.end()) return it->second;
    const uint32_t id = 1000 + static_cast<uint32_t>(vocab.size());
    vocab.push_back(word);
    index[word] = id;
    return id;
  }

  std::vector<uint32_t> encode(std::string_view text) override {
    std::vector<uint32_t> out;
    std::string word;
    for (char c : text) {
      if (c == ' ' || c == '\n') {
        if (!word.empty()) out.push_back(id_of(word));
        word.clear();
        if (c == '\n') out.push_back(id_of("\n"));
      } else {
        word += c;
      }
    }
    if (!word.empty()) out.push_back(id_of(word));
    return out;
  }

  // Raw pieces decode byte-exact (no trailing space): scripted generated text.
  std::map<uint32_t, std::string> raw_pieces;
  std::vector<uint32_t> raw(const std::string& text, size_t piece_bytes) {
    std::vector<uint32_t> out;
    for (size_t at = 0; at < text.size(); at += piece_bytes) {
      const uint32_t id = 200000 + static_cast<uint32_t>(raw_pieces.size());
      raw_pieces[id] = text.substr(at, piece_bytes);
      out.push_back(id);
    }
    return out;
  }

  std::string piece(uint32_t id) {
    if (id == 248046) return "<|im_end|>";
    const auto raw_it = raw_pieces.find(id);
    if (raw_it != raw_pieces.end()) return raw_it->second;
    const std::string& word = vocab.at(id - 1000);
    return word == "\n" ? "\n" : word + " ";
  }

  std::string decode(const std::vector<uint32_t>& ids) override {
    std::string text;
    for (uint32_t id : ids) text += piece(id);
    return text;
  }

  struct Streamer : server::StreamerIface {
    explicit Streamer(MockTok& tokenizer) : tokenizer(tokenizer) {}
    std::string push(uint32_t id) override { return tokenizer.piece(id); }
    std::string flush() override { return ""; }
    MockTok& tokenizer;
  };

  std::unique_ptr<server::StreamerIface> streamer() override {
    return std::make_unique<Streamer>(*this);
  }

  uint32_t vocab_used() override { return 248077; }
};

struct MockTemplate : server::TemplateIface {
  bool think_tag = false;  // end a thinking prompt in "<think>\n" as the real template does
  std::string render(const nlohmann::json& messages, const nlohmann::json&, bool think) override {
    std::string text;
    for (const auto& message : messages) {
      text += message.at("role").get<std::string>() + ": " +
              message.at("content").get<std::string>() + "\n";
    }
    if (think_tag && think) return text + "assistant:\n<think>\n";
    return text + (think ? "assistant(think): " : "assistant: ");
  }
};

struct MockEngine : server::EngineIface {
  std::vector<uint32_t> script;
  size_t at = 0;
  int step_ms = 0;
  std::vector<uint32_t> last_prompt;
  std::vector<std::chrono::steady_clock::time_point> step_times;
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};

  void reset() override { at = 0; }
  void prefill(const std::vector<uint32_t>& ids) override { last_prompt = ids; }

  uint32_t step(const server::Sampling&) override {
    const int now_active = ++active;
    max_active = std::max(max_active.load(), now_active);
    if (step_ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    step_times.push_back(std::chrono::steady_clock::now());
    const uint32_t id = at < script.size() ? script[at++] : 248046;
    --active;
    return id;
  }

  uint32_t max_len() override { return 1024; }
  uint32_t pos() override { return static_cast<uint32_t>(last_prompt.size() + at); }

  // Spec 8 (plan 8c): a scripted speculative engine. With mtp > 0, step_many(s, k)
  // returns the next `bursts[i % bursts.size()]` ids of the script (clamped to
  // 1..k + 1: the burst is how many drafts the "head" would get right), and
  // truncate_to() rewinds `at`; every truncation and every requested k is recorded.
  uint32_t mtp = 0;
  std::vector<uint32_t> bursts{4};
  size_t burst_calls = 0;
  std::vector<uint32_t> truncations;
  std::vector<uint32_t> ks;
  uint32_t mtp_k() override { return mtp; }
  std::vector<uint32_t> step_many(const server::Sampling& s, uint32_t k) override {
    if (k > mtp) throw std::logic_error("mock: step_many past mtp_k()");
    ks.push_back(k);
    uint32_t n = bursts[burst_calls++ % bursts.size()];
    n = std::max<uint32_t>(1, std::min<uint32_t>(n, k + 1));
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < n; ++i) out.push_back(step(s));
    return out;
  }
  void truncate_to(uint32_t p) override {
    truncations.push_back(p);
    at = p - last_prompt.size();
  }
};

// Spec 7 (plan 7c): an engine with a real session for the prefix cache tests. The state is
// a hash of every id consumed so far, the KV cache is the ids per position, and the next
// id is a function of BOTH (the state and a hash over KV [0, pos)), so a wrong state or a
// stale KV position after a restore generates different ids. It follows runtime::Engine's
// protocol: prefill leaves the first generated id pending, step() returns the pending id
// and consumes it (its KV is written, pos advances), the block hook fires at every block
// end inside a prefill and at its end.
struct StateMockEngine : server::EngineIface {
  uint32_t blk = 4;
  uint32_t maxlen = 1024;
  std::vector<uint32_t> words;   // what it generates: ids the MockTok knows
  uint32_t bad_id = 0xFFFFFFFFu; // prefill throws when it meets it (after writing the ids before)
  std::vector<uint32_t> kv = std::vector<uint32_t>(1024, 0);
  uint64_t state = 0;
  uint32_t p = 0, cur = 0;
  BlockHook hook;
  uint64_t rng = 0;
  size_t prefilled = 0, ingested = 0, resets = 0, state_loads = 0, kv_loaded = 0;

  static uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h * 0xBF58476D1CE4E5B9ull;
  }
  uint32_t next() const {
    uint64_t h = state;
    for (uint32_t i = 0; i < p; ++i) h = mix(h, kv[i]);
    return words[h % words.size()];
  }
  void consume(uint32_t id) {
    if (p >= maxlen) throw std::runtime_error("mock: past max_len");
    kv[p++] = id;
    state = mix(state, id);
  }

  void reset() override {
    ++resets;
    std::fill(kv.begin(), kv.end(), 0);
    state = 0;
    p = 0;
    cur = 0;
  }
  void prefill(const std::vector<uint32_t>& ids) override {
    if (ids.empty()) throw std::runtime_error("mock: empty prefill");
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == bad_id) throw std::runtime_error("mock: id out of vocabulary");
      consume(ids[i]);
      ++prefilled;
      const bool last = i + 1 == ids.size();
      if (last) cur = next();
      if (hook && (last || p % blk == 0)) hook(p, p % blk == 0);
    }
  }
  uint32_t step(const server::Sampling& s) override {
    const uint32_t id = cur;
    consume(id);
    cur = next();
    if (!s.greedy) {   // "sampling": a different pick than the argmax, seeded
      if (rng == 0) rng = s.has_seed ? s.seed + 1 : 12345;
      rng = mix(rng, 7);
      cur = words[(rng >> 7) % words.size()];
    }
    return id;
  }
  uint32_t max_len() override { return maxlen; }
  uint32_t pos() override { return p; }

  uint32_t block() override { return blk; }
  size_t state_bytes() override { return sizeof(uint64_t); }
  size_t kv_bytes(uint32_t n) override { return size_t(n) * sizeof(uint32_t); }
  void save_state(void* host) override { std::memcpy(host, &state, sizeof state); }
  void load_state(const void* host, uint32_t at) override {
    std::memcpy(&state, host, sizeof state);
    p = at;
    ++state_loads;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override {
    std::memcpy(host, kv.data() + b, size_t(e - b) * 4);
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) override {
    std::memcpy(kv.data() + b, host, size_t(e - b) * 4);
    kv_loaded += e - b;
  }
  void set_block_hook(BlockHook h) override { hook = std::move(h); }
  void ingest(const std::vector<uint32_t>& ids) override {
    for (uint32_t id : ids) {
      if (id == bad_id) throw std::runtime_error("mock: id out of vocabulary");
      consume(id);
      ++ingested;
    }
    cur = next();
  }
};

// Spec 13 (plan 13c Task 1): a batched engine with `n_slots` independent sessions, each a
// StateMockEngine: the next id is a function of the slot's state hash AND its KV ids, so a
// wrong restore or a row mixed up with another slot's generates different ids. Sampled rows
// draw from the row's own generator only. `clock` counts cost in mock ticks: a decode step
// costs `step_cost` whatever its row count (decode is bandwidth-bound, spec 13 §1), a
// prefill call costs its ids / `prefill_rate`. `events` logs the calls for ordering checks.
struct MockBatchEngine : server::BatchEngineIface {
  struct Slot {
    std::vector<uint32_t> kv;
    uint64_t state = 0;
    uint32_t p = 0, cur = 0, prompt_end = 0;
    bool pending = false;
  };
  explicit MockBatchEngine(uint32_t n_slots = 4, uint32_t len = 1024) : maxlen(len) {
    slot.resize(n_slots);
    for (auto& s : slot) s.kv.assign(maxlen, 0);
    for (uint32_t i = 0; i < 24; ++i) words.push_back(1000 + i);
  }
  uint32_t maxlen;
  uint32_t blk = 4;
  std::vector<Slot> slot;
  std::vector<uint32_t> words;
  uint32_t eos = 248046;
  // A session whose first id is `key` generates EOS as its n-th generated id (n >= 1).
  std::map<uint32_t, uint32_t> eos_at;
  uint32_t bad_id = 0xFFFFFFFFu;   // prefill throws when it meets it
  double clock = 0, step_cost = 1, prefill_rate = 32;
  int step_us = 0;                 // wall-clock sleep per step (threaded tests)
  std::vector<std::string> events;
  std::atomic<uint64_t> steps{0};

  uint32_t next(const Slot& s) const {
    const uint32_t generated = s.p - s.prompt_end + 1;   // the pending id's index, 1-based
    const auto it = eos_at.find(s.kv[0]);
    if (it != eos_at.end() && generated == it->second) return eos;
    uint64_t h = s.state;
    for (uint32_t i = 0; i < s.p; ++i) h = StateMockEngine::mix(h, s.kv[i]);
    return words[h % words.size()];
  }
  void consume(Slot& s, uint32_t id) {
    if (s.p >= maxlen) throw std::runtime_error("mock: past max_len");
    s.kv[s.p++] = id;
    s.state = StateMockEngine::mix(s.state, id);
  }
  static std::string ev(const std::string& what, uint32_t i, uint32_t at) {
    return what + " " + std::to_string(i) + " @" + std::to_string(at);
  }

  uint32_t slots() override { return static_cast<uint32_t>(slot.size()); }
  uint32_t max_len() override { return maxlen; }
  uint32_t pos(uint32_t i) override { return slot.at(i).p; }
  void reset(uint32_t i) override {
    Slot& s = slot.at(i);
    std::fill(s.kv.begin(), s.kv.end(), 0);
    s.state = 0;
    s.p = s.cur = s.prompt_end = 0;
    s.pending = false;
    events.push_back(ev("reset", i, 0));
  }
  void prefill(const std::vector<Chunk>& chunks) override {
    std::string e = "prefill";
    uint32_t total = 0;
    for (const Chunk& c : chunks) {
      if (c.n == 0) throw std::runtime_error("mock: empty chunk");
      Slot& s = slot.at(c.slot);
      e += " " + std::to_string(c.slot) + ":" + std::to_string(c.n);
      for (uint32_t k = 0; k < c.n; ++k) {
        if (c.ids[k] == bad_id) throw std::runtime_error("mock: id out of vocabulary");
        consume(s, c.ids[k]);
      }
      total += c.n;
      if (c.last) {
        s.prompt_end = s.p;
        s.cur = next(s);
        s.pending = true;
      }
    }
    clock += total / prefill_rate;
    events.push_back(e);
  }
  std::vector<uint32_t> step_batch(const std::vector<Row>& rows) override {
    if (step_us != 0) std::this_thread::sleep_for(std::chrono::microseconds(step_us));
    std::vector<uint32_t> out;
    std::string e = "step";
    for (const Row& r : rows) {
      Slot& s = slot.at(r.slot);
      if (!s.pending) throw std::logic_error("mock: step on a slot with nothing pending");
      e += " " + std::to_string(r.slot);
      const uint32_t id = s.cur;
      consume(s, id);
      s.cur = next(s);
      if (!r.sampling->greedy && s.cur != eos) s.cur = words[(*r.rng)() % words.size()];
      out.push_back(id);
    }
    clock += step_cost;
    events.push_back(e);
    ++steps;
    return out;
  }

  uint32_t block() override { return blk; }
  size_t state_bytes() override { return sizeof(uint64_t); }
  size_t kv_bytes(uint32_t n) override { return size_t(n) * sizeof(uint32_t); }
  void save_state(uint32_t i, void* host) override {
    std::memcpy(host, &slot.at(i).state, sizeof(uint64_t));
    events.push_back(ev("save_state", i, slot[i].p));
  }
  void load_state(uint32_t i, const void* host, uint32_t at) override {
    Slot& s = slot.at(i);
    std::memcpy(&s.state, host, sizeof(uint64_t));
    s.p = at;
    s.pending = false;
    events.push_back(ev("load_state", i, at));
  }
  void save_kv(uint32_t i, uint32_t b, uint32_t e, void* host) override {
    std::memcpy(host, slot.at(i).kv.data() + b, size_t(e - b) * 4);
  }
  void load_kv(uint32_t i, uint32_t b, uint32_t e, const void* host) override {
    std::memcpy(slot.at(i).kv.data() + b, host, size_t(e - b) * 4);
  }
};
