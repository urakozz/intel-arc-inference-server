#pragma once
// Spec 13 §10 / §11 (plan 13b Task 3): b70-serve --parallel replica - two of today's single-sequence engines, one
// per card, behind server::BatchEngineIface: a two-slot batched engine with no new kernels. Slot i IS card i's
// engine (a server::EngineIface: EngineAdapter over a runtime::Engine on the box, a mock in the host test).
//
// Threads (Review Focus 2): every call on a card's engine runs on that card's own worker thread, so the engines
// never share a thread and each sees exactly the call sequence the serial server would make. A call over several
// slots (step_batch, step_spec, prefill) is posted to every card's worker and joined; a call that failed on some
// cards throws server::SlotsFailed naming those rows only, with the others' results. prefill_begin / prefill_end
// (overlaps_prefill() is true) let the scheduler run a prefill on one card beside a step on the other.
//
// Bitwise per card (R1): the engines are independent - nothing is shared but the host prefix store (the scheduler's
// PrefixSlots: a snapshot saved on card 0 restores on card 1, the host layout being the same - checked at
// construction). The prompt is cut where the serial server's one Engine::prefill call cuts it (prefill_cut,
// runtime::prefill_chunk_rows with the card's chunk cap, hooked when the prefix cache is on - the serial server's
// PrefixSession hooks its engine then). The request's random generator is the card adapter's own, seeded from the
// request's seed at its first sampled id exactly as in the serial server (EngineAdapter::seed_rng); the scheduler's
// per-row generator (Row::rng) is not used.
//
// Host only: this header includes no device code (tests/cli/replica_engine_test.cc runs it on the Mac).
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "runtime/prefill_chunks.h"
#include "server/batch_engine.h"
#include "server/deps.h"

namespace cli {

// One thread that runs the tasks posted to it, in order.
class CardWorker {
 public:
  CardWorker() : thread_([this] { loop(); }) {}
  ~CardWorker() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  CardWorker(const CardWorker&) = delete;
  CardWorker& operator=(const CardWorker&) = delete;

  std::future<void> post(std::function<void()> fn) {
    std::packaged_task<void()> task(std::move(fn));
    std::future<void> done = task.get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      tasks_.push_back(std::move(task));
    }
    cv_.notify_all();
    return done;
  }
  std::thread::id id() const { return thread_.get_id(); }

 private:
  void loop() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty()) return;   // stopping, nothing left
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();   // an exception is stored in the task's future
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::packaged_task<void()>> tasks_;
  bool stopping_ = false;
  std::thread thread_;   // last: started once the queue exists
};

class ReplicaEngine : public server::BatchEngineIface {
 public:
  struct Card {
    server::EngineIface* engine = nullptr;
    uint32_t chunk_cap = 2048;   // the card's runtime::Engine::prefill_chunk_cap()
  };
  // `hooked`: the prefix cache is on (the serial server then hooks its engine, which cuts every chunk at a stride
  // end); `wide_stride`: the model's prefill passes it to runtime::prefill_chunk_rows (Qwen3.8: yes).
  ReplicaEngine(std::vector<Card> cards, bool hooked, bool wide_stride)
      : cards_(std::move(cards)), hooked_(hooked), wide_(wide_stride) {
    if (cards_.size() < 2) throw std::invalid_argument("ReplicaEngine: two cards or more");
    for (const Card& c : cards_)
      if (c.engine == nullptr) throw std::invalid_argument("ReplicaEngine: a card without an engine");
    for (size_t i = 0; i < cards_.size(); ++i) workers_.push_back(std::make_unique<CardWorker>());
    // What the scheduler and the shared host store read once, from each card on its own thread.
    for (uint32_t i = 0; i < slots(); ++i) {
      Facts f;
      on(i, [&] {
        server::EngineIface& e = *cards_[i].engine;
        f.max_len = e.max_len();
        f.mtp_k = e.mtp_k();
        f.verify_k = e.verify_k();
        f.block = e.block();
        f.kv_form = e.kv_form();
        f.drafter_key = e.drafter_key();
      });
      facts_.push_back(f);
    }
    for (uint32_t i = 1; i < slots(); ++i) {
      const Facts &a = facts_[0], &b = facts_[i];
      if (a.mtp_k != b.mtp_k || a.verify_k != b.verify_k || a.block != b.block || a.kv_form != b.kv_form ||
          a.drafter_key != b.drafter_key)
        throw std::invalid_argument("ReplicaEngine: card " + std::to_string(i) +
                                    "'s engine differs from card 0's (proposer, block, KV form or drafter)");
    }
  }
  ~ReplicaEngine() override {
    for (auto& f : inflight_)
      if (f.valid()) f.wait();
    workers_.clear();   // joins every card's thread
  }

  // The worker thread of card `i` (tests).
  std::thread::id thread_of(uint32_t i) const { return workers_.at(i)->id(); }

  uint32_t slots() override { return static_cast<uint32_t>(cards_.size()); }
  uint32_t max_len() override {
    uint32_t m = 0;
    for (const Facts& f : facts_) m = std::max(m, f.max_len);
    return m;
  }
  uint32_t slot_max_len(uint32_t slot) override { return facts_.at(slot).max_len; }
  uint32_t pos(uint32_t slot) override {
    uint32_t p = 0;
    on(slot, [&] { p = card(slot).pos(); });
    return p;
  }
  void reset(uint32_t slot) override {
    on(slot, [&] { card(slot).reset(); });
  }

  uint32_t prefill_cut(uint32_t slot, uint32_t pos, uint32_t remaining) override {
    return runtime::prefill_chunk_rows(pos, remaining, cards_.at(slot).chunk_cap, hooked_, facts_.at(slot).block,
                                       wide_);
  }
  bool overlaps_prefill() override { return true; }
  void prefill(const std::vector<Chunk>& chunks) override {
    prefill_begin(chunks);
    prefill_end();
  }
  void prefill_begin(const std::vector<Chunk>& chunks) override {
    if (!inflight_.empty()) throw std::logic_error("ReplicaEngine: a prefill is already in flight");
    for (const Chunk& c : chunks) {
      const uint32_t slot = c.slot;
      std::vector<uint32_t> ids(c.ids, c.ids + c.n);
      const bool replay = c.replay;
      inflight_.push_back(workers_.at(slot)->post([this, slot, ids = std::move(ids), replay] {
        server::EngineIface& e = card(slot);
        if (replay) {   // --prefix-split-last: PrefixSession::begin's ingest of the last id
          e.ingest(ids);
          return;
        }
        // One Engine::prefill call per card chunk: the cut the serial server's one call makes (bitwise, R1).
        size_t at = 0;
        while (at < ids.size()) {
          const uint32_t n = prefill_cut(slot, e.pos(), static_cast<uint32_t>(ids.size() - at));
          e.prefill(std::vector<uint32_t>(ids.begin() + long(at), ids.begin() + long(at + n)));
          at += n;
        }
      }));
    }
  }
  void prefill_end() override {
    std::vector<std::future<void>> done = std::move(inflight_);
    inflight_.clear();
    join(done, nullptr);
  }

  std::vector<uint32_t> step_batch(const std::vector<Row>& rows) override {
    std::vector<std::vector<uint32_t>> runs(rows.size());
    std::vector<std::future<void>> done;
    for (size_t r = 0; r < rows.size(); ++r) {
      const Row row = rows[r];
      done.push_back(workers_.at(row.slot)->post([this, row, &runs, r] { runs[r] = {card(row.slot).step(*row.sampling)}; }));
    }
    join(done, &runs);
    std::vector<uint32_t> ids;
    for (const auto& run : runs) ids.push_back(run.at(0));
    return ids;
  }

  // MTP (the head on each card), DFlash (each card's drafter) or prompt lookup (a row with a proposer).
  uint32_t spec_k() override { return facts_[0].mtp_k; }
  uint32_t verify_k() override { return facts_[0].verify_k; }
  std::vector<std::vector<uint32_t>> step_spec(const std::vector<SpecRow>& rows) override {
    std::vector<std::vector<uint32_t>> runs(rows.size());
    std::vector<std::future<void>> done;
    for (size_t r = 0; r < rows.size(); ++r) {
      const SpecRow row = rows[r];
      done.push_back(workers_.at(row.slot)->post([this, row, &runs, r] {
        server::EngineIface& e = card(row.slot);
        if (row.propose != nullptr) runs[r] = e.step_drafts(*row.sampling, *row.propose);
        else if (facts_[row.slot].mtp_k > 0) runs[r] = e.step_many(*row.sampling, row.k);
        else runs[r] = {e.step(*row.sampling)};
      }));
    }
    join(done, &runs);
    return runs;
  }
  void truncate_to(uint32_t slot, uint32_t p) override {
    on(slot, [&] { card(slot).truncate_to(p); });
  }

  // Spec 7, per card; the host layouts are the cards' own (equal: checked above, state_bytes / kv_bytes below).
  uint32_t block() override { return facts_[0].block; }
  size_t state_bytes() override {
    size_t b0 = 0;
    for (uint32_t i = 0; i < slots(); ++i) {
      size_t b = 0;
      on(i, [&] { b = card(i).state_bytes(); });
      if (i == 0) b0 = b;
      else if (b != b0) throw std::invalid_argument("ReplicaEngine: the cards' state snapshots differ in size");
    }
    return b0;
  }
  size_t kv_bytes(uint32_t n) override {
    size_t b = 0;
    on(0, [&] { b = card(0).kv_bytes(n); });
    return b;
  }
  uint64_t kv_form() override { return facts_[0].kv_form; }
  uint64_t drafter_key() override { return facts_[0].drafter_key; }
  void save_state(uint32_t slot, void* host) override {
    on(slot, [&] { card(slot).save_state(host); });
  }
  void load_state(uint32_t slot, const void* host, uint32_t p) override {
    on(slot, [&] { card(slot).load_state(host, p); });
  }
  void save_kv(uint32_t slot, uint32_t b, uint32_t e, void* host) override {
    on(slot, [&] { card(slot).save_kv(b, e, host); });
  }
  void load_kv(uint32_t slot, uint32_t b, uint32_t e, const void* host) override {
    on(slot, [&] { card(slot).load_kv(b, e, host); });
  }

 private:
  struct Facts {
    uint32_t max_len = 0, mtp_k = 0, verify_k = 0, block = 0;
    uint64_t kv_form = 0, drafter_key = 0;
  };
  server::EngineIface& card(uint32_t slot) { return *cards_.at(slot).engine; }
  // Runs fn on card `slot`'s thread and waits; rethrows what it threw.
  void on(uint32_t slot, std::function<void()> fn) { workers_.at(slot)->post(std::move(fn)).get(); }
  // Waits for every posted call; the failed ones become one SlotsFailed (per call, in order) with the completed
  // ones' results - nothing returns before every card is done (no task outlives the call's locals).
  static void join(std::vector<std::future<void>>& done, std::vector<std::vector<uint32_t>>* runs) {
    std::vector<std::string> errors(done.size());
    bool failed = false;
    for (size_t i = 0; i < done.size(); ++i) {
      try {
        done[i].get();
      } catch (const std::exception& e) {
        errors[i] = e.what();
        if (errors[i].empty()) errors[i] = "a card failed";
        failed = true;
      } catch (...) {
        errors[i] = "a card failed";
        failed = true;
      }
    }
    if (!failed) return;
    std::vector<std::vector<uint32_t>> kept;
    if (runs != nullptr) {
      kept = *runs;
      for (size_t i = 0; i < kept.size(); ++i)
        if (!errors[i].empty()) kept[i].clear();
    }
    throw server::SlotsFailed(std::move(errors), std::move(kept));
  }

  std::vector<Card> cards_;
  bool hooked_, wide_;
  std::vector<Facts> facts_;
  std::vector<std::future<void>> inflight_;          // the prefill_begin calls not yet joined
  std::vector<std::unique_ptr<CardWorker>> workers_;   // one thread per card
};

}  // namespace cli
