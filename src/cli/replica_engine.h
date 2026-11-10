#pragma once
// Spec 13 §10 / §11 (plan 13b Task 3): b70-serve --parallel replica - two of today's single-sequence engines, one
// per card, behind server::BatchEngineIface: a two-slot batched engine with no new kernels. Slot i IS card i's
// engine (a server::EngineIface: EngineAdapter over a runtime::Engine on the box, a mock in the host test).
//
// Threads (Review Focus 2): every call on a card's engine runs on that card's own worker thread, so the engines
// never share a thread and each sees exactly the call sequence the serial server would make. A call over several
// slots (step_batch, step_spec) is posted to every card's worker and joined; a call that failed on some cards throws
// server::SlotsFailed naming those rows only, with the others' results. A prefill (overlaps_prefill() is true) is
// started per card by prefill_begin and joined per card by prefill_end(slot) once prefill_ready(slot): the scheduler
// keeps stepping the other card meanwhile, and both cards may prefill at once.
//
// Bitwise per card (R1): the engines are independent - nothing is shared but the host prefix store (the scheduler's
// PrefixSlots: a snapshot saved on card 0 restores on card 1, the host layout being the same - checked at
// construction). The prompt is cut where the serial server's one Engine::prefill call cuts it (prefill_cut,
// runtime::prefill_chunk_rows with the card's chunk cap, hooked when the prefix cache is on - the serial server's
// PrefixSession hooks its engine then). The request's random generator is the card adapter's own, seeded from the
// request's seed at its first sampled id exactly as in the serial server (EngineAdapter::seed_rng); the scheduler's
// per-row generator (Row::rng) is not used.
//
// Replica x slots (operator 2026-11-10): a card whose --max-len leaves spare memory serves several requests - its
// Card is then a server::BatchEngineIface (cli::SlotsEngine over the card's runtime::Engine with plan 13c's slots, a
// mock in the host test) instead of one EngineIface. The scheduler's slots are the cards' slots in card order (card
// 0's first; card_of says which card): a call over rows of one card is ONE call of that card's batched engine on that
// card's thread (its own rows, its own slot numbers, in the scheduler's order), so each card sees exactly the calls
// 13c's one-card server makes for the same slot schedule (R1 per card), and a card that throws fails its rows only
// (its own SlotsFailed is mapped to those rows). The scheduler keeps one chunk in flight per card and leaves a busy
// card alone (BatchEngineIface::card_of); each card's step has its own row budget (card_row_budget). A card with one
// slot is today's EngineIface path, call for call.
//
// Host only: this header includes no device code (tests/cli/replica_engine_test.cc runs it on the Mac).
#include <algorithm>
#include <chrono>
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
  // Exactly one of `engine` (one slot: today's single-sequence engine) and `slots` (several: the card's batched
  // engine, cli::SlotsEngine) is set.
  struct Card {
    server::EngineIface* engine = nullptr;
    uint32_t chunk_cap = 2048;   // the card's runtime::Engine::prefill_chunk_cap()
    server::BatchEngineIface* slots = nullptr;
  };
  // `hooked`: the prefix cache is on (the serial server then hooks its engine, which cuts every chunk at a stride
  // end); `wide_stride`: the model's prefill passes it to runtime::prefill_chunk_rows (Qwen3.8: yes).
  ReplicaEngine(std::vector<Card> cards, bool hooked, bool wide_stride)
      : cards_(std::move(cards)), hooked_(hooked), wide_(wide_stride) {
    if (cards_.size() < 2) throw std::invalid_argument("ReplicaEngine: two cards or more");
    for (const Card& c : cards_)
      if ((c.engine == nullptr) == (c.slots == nullptr))
        throw std::invalid_argument("ReplicaEngine: a card needs one engine (single-sequence or slots)");
    for (size_t i = 0; i < cards_.size(); ++i) workers_.push_back(std::make_unique<CardWorker>());
    // The slots, card by card; what the scheduler and the shared host store read once, from each card on its own
    // thread.
    for (uint32_t c = 0; c < cards_.size(); ++c) {
      uint32_t n = 1;
      if (cards_[c].slots != nullptr) on_card(c, [&] { n = cards_[c].slots->slots(); });
      if (n == 0) throw std::invalid_argument("ReplicaEngine: card " + std::to_string(c) + " has no slot");
      for (uint32_t l = 0; l < n; ++l) {
        Facts f;
        on_card(c, [&] {
          if (server::EngineIface* e = cards_[c].engine) {
            f.max_len = e->max_len();
            f.mtp_k = e->mtp_k();
            f.verify_k = e->verify_k();
            f.block = e->block();
            f.kv_form = e->kv_form();
            f.drafter_key = e->drafter_key();
          } else {
            server::BatchEngineIface& b = *cards_[c].slots;
            f.max_len = b.slot_max_len(l);
            f.mtp_k = b.spec_k();
            f.verify_k = b.verify_k();
            f.block = b.block();
            f.kv_form = b.kv_form();
            f.drafter_key = b.drafter_key();
          }
        });
        facts_.push_back(f);
        slot_card_.push_back(c);
        slot_local_.push_back(l);
      }
    }
    inflight_.resize(facts_.size());
    for (uint32_t i = 1; i < slots(); ++i) {
      const Facts &a = facts_[0], &b = facts_[i];
      if (a.mtp_k != b.mtp_k || a.verify_k != b.verify_k || a.block != b.block || a.kv_form != b.kv_form ||
          a.drafter_key != b.drafter_key)
        throw std::invalid_argument("ReplicaEngine: card " + std::to_string(slot_card_[i]) +
                                    "'s engine differs from card 0's (proposer, block, KV form or drafter)");
    }
  }
  ~ReplicaEngine() override {
    for (auto& f : inflight_)
      if (f.valid()) f.wait();   // a prefill still on a card finishes before the threads go
    workers_.clear();   // joins every card's thread
  }

  // The worker thread of card `i` (tests).
  std::thread::id thread_of(uint32_t i) const { return workers_.at(i)->id(); }

  uint32_t slots() override { return static_cast<uint32_t>(facts_.size()); }
  uint32_t card_of(uint32_t slot) override { return slot_card_.at(slot); }
  uint32_t max_len() override {
    uint32_t m = 0;
    for (const Facts& f : facts_) m = std::max(m, f.max_len);
    return m;
  }
  uint32_t slot_max_len(uint32_t slot) override { return facts_.at(slot).max_len; }
  uint32_t pos(uint32_t slot) override {
    uint32_t p = 0;
    on(slot, [&] { p = single(slot) ? card(slot).pos() : batch(slot).pos(local(slot)); });
    return p;
  }
  void reset(uint32_t slot) override {
    on(slot, [&] {
      if (single(slot)) card(slot).reset();
      else batch(slot).reset(local(slot));
    });
  }

  // A card with slots cuts as cli::SlotsEngine::prefill_cut does (the same rule over its Card::chunk_cap - the card
  // engine's prefill_chunk_cap() - and runtime::Engine::kBlock), without a call on the card.
  uint32_t prefill_cut(uint32_t slot, uint32_t pos, uint32_t remaining) override {
    return runtime::prefill_chunk_rows(pos, remaining, cards_.at(slot_card_.at(slot)).chunk_cap, hooked_,
                                       facts_.at(slot).block, wide_);
  }
  bool overlaps_prefill() override { return true; }
  // Synchronous: every chunk's card started, then each joined (a failure names its chunk only).
  void prefill(const std::vector<Chunk>& chunks) override {
    prefill_begin(chunks);
    std::vector<std::string> errors(chunks.size());
    bool failed = false;
    for (size_t i = 0; i < chunks.size(); ++i) {
      try {
        prefill_end(chunks[i].slot);
      } catch (const std::exception& e) {
        errors[i] = e.what()[0] != '\0' ? e.what() : "a card failed";
        failed = true;
      }
    }
    if (failed) throw server::SlotsFailed(std::move(errors), {});
  }
  // Starts each chunk on its card's thread and returns. Checked before anything is posted: one chunk per slot, and
  // no slot with a prefill still in flight - so a throw here started nothing. Two chunks on one card run one after
  // the other on its thread (the scheduler starts one per card).
  void prefill_begin(const std::vector<Chunk>& chunks) override {
    std::vector<bool> seen(facts_.size(), false);
    for (const Chunk& c : chunks) {
      if (c.slot >= facts_.size()) throw std::out_of_range("ReplicaEngine: no slot " + std::to_string(c.slot));
      if (seen[c.slot]) throw std::logic_error("ReplicaEngine: two chunks of slot " + std::to_string(c.slot));
      if (inflight_[c.slot].valid())
        throw std::logic_error("ReplicaEngine: slot " + std::to_string(c.slot) + " has a prefill in flight");
      seen[c.slot] = true;
    }
    for (const Chunk& c : chunks) {
      const uint32_t slot = c.slot;
      std::vector<uint32_t> ids(c.ids, c.ids + c.n);
      const bool replay = c.replay, last = c.last;
      if (!single(slot)) {   // the card's batched engine, its own slot number, cut as the card's one call cuts it
        inflight_[slot] = worker(slot).post([this, slot, ids = std::move(ids), replay, last] {
          server::BatchEngineIface& b = batch(slot);
          const uint32_t l = local(slot);
          if (replay) {
            b.prefill({{l, ids.data(), static_cast<uint32_t>(ids.size()), last, true}});
            return;
          }
          size_t at = 0;
          while (at < ids.size()) {
            const uint32_t n = prefill_cut(slot, b.pos(l), static_cast<uint32_t>(ids.size() - at));
            b.prefill({{l, ids.data() + at, n, last && at + n == ids.size(), false}});
            at += n;
          }
        });
        continue;
      }
      inflight_[slot] = worker(slot).post([this, slot, ids = std::move(ids), replay] {
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
      });
    }
  }
  bool prefill_ready(uint32_t slot) override {
    const std::future<void>& f = inflight_.at(slot);
    return !f.valid() || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }
  // Joins the slot's prefill (waiting if it still runs) and throws what it threw; nothing in flight: nothing.
  void prefill_end(uint32_t slot) override {
    std::future<void>& f = inflight_.at(slot);
    if (!f.valid()) return;
    std::future<void> done = std::move(f);   // the slot is free again whatever get() throws
    done.get();
  }

  std::vector<uint32_t> step_batch(const std::vector<Row>& rows) override {
    std::vector<std::vector<uint32_t>> runs(rows.size());
    std::vector<std::string> errors(rows.size());
    const std::vector<std::vector<size_t>> groups = batched_rows(rows);   // every slot checked before any post
    std::vector<Post> done;
    for (size_t r = 0; r < rows.size(); ++r) {
      const Row row = rows[r];
      if (!single(row.slot)) continue;
      done.push_back({worker(row.slot).post([this, row, &runs, r] { runs[r] = {card(row.slot).step(*row.sampling)}; }),
                      {r}});
    }
    for (const std::vector<size_t>& mine : groups)   // one call per card with slots
      done.push_back(post_card(rows, mine, runs, errors, [](server::BatchEngineIface& b, const std::vector<Row>& own) {
        std::vector<std::vector<uint32_t>> got;
        for (uint32_t id : b.step_batch(own)) got.push_back({id});
        return got;
      }));
    join(done, errors, &runs);
    std::vector<uint32_t> ids;
    for (const auto& run : runs) ids.push_back(run.at(0));
    return ids;
  }

  // MTP (the head on each card), DFlash (each card's drafter) or prompt lookup (a row with a proposer).
  uint32_t spec_k() override { return facts_[0].mtp_k; }
  uint32_t verify_k() override { return facts_[0].verify_k; }
  // Replica x slots: each card's own budget for its rows (a single-sequence card: none, as before).
  uint32_t card_row_budget(uint32_t c, uint32_t active) override {
    return cards_.at(c).slots != nullptr ? cards_[c].slots->row_budget(active) : 0;
  }
  std::vector<std::vector<uint32_t>> step_spec(const std::vector<SpecRow>& rows) override {
    std::vector<std::vector<uint32_t>> runs(rows.size());
    std::vector<std::string> errors(rows.size());
    const std::vector<std::vector<size_t>> groups = batched_rows(rows);   // every slot checked before any post
    std::vector<Post> done;
    for (size_t r = 0; r < rows.size(); ++r) {
      const SpecRow row = rows[r];
      if (!single(row.slot)) continue;
      done.push_back({worker(row.slot).post([this, row, &runs, r] {
                        server::EngineIface& e = card(row.slot);
                        if (row.propose != nullptr) runs[r] = e.step_drafts(*row.sampling, *row.propose);
                        else if (facts_[row.slot].mtp_k > 0) runs[r] = e.step_many(*row.sampling, row.k);
                        else runs[r] = {e.step(*row.sampling)};
                      }),
                      {r}});
    }
    for (const std::vector<size_t>& mine : groups)   // one call per card with slots
      done.push_back(post_card(rows, mine, runs, errors,
                               [](server::BatchEngineIface& b, const std::vector<SpecRow>& own) {
                                 return b.step_spec(own);
                               }));
    join(done, errors, &runs);
    return runs;
  }
  void truncate_to(uint32_t slot, uint32_t p) override {
    on(slot, [&] {
      if (single(slot)) card(slot).truncate_to(p);
      else batch(slot).truncate_to(local(slot), p);
    });
  }

  // Spec 7, per card; the host layouts are the cards' own (equal: checked above, state_bytes / kv_bytes below).
  uint32_t block() override { return facts_[0].block; }
  size_t state_bytes() override {
    size_t b0 = 0;
    for (uint32_t c = 0; c < cards_.size(); ++c) {
      size_t b = 0;
      on_card(c, [&] { b = cards_[c].engine ? cards_[c].engine->state_bytes() : cards_[c].slots->state_bytes(); });
      if (c == 0) b0 = b;
      else if (b != b0) throw std::invalid_argument("ReplicaEngine: the cards' state snapshots differ in size");
    }
    return b0;
  }
  size_t kv_bytes(uint32_t n) override {
    size_t b = 0;
    on_card(0, [&] { b = cards_[0].engine ? cards_[0].engine->kv_bytes(n) : cards_[0].slots->kv_bytes(n); });
    return b;
  }
  uint64_t kv_form() override { return facts_[0].kv_form; }
  uint64_t drafter_key() override { return facts_[0].drafter_key; }
  void save_state(uint32_t slot, void* host) override {
    on(slot, [&] {
      if (single(slot)) card(slot).save_state(host);
      else batch(slot).save_state(local(slot), host);
    });
  }
  void load_state(uint32_t slot, const void* host, uint32_t p) override {
    on(slot, [&] {
      if (single(slot)) card(slot).load_state(host, p);
      else batch(slot).load_state(local(slot), host, p);
    });
  }
  void save_kv(uint32_t slot, uint32_t b, uint32_t e, void* host) override {
    on(slot, [&] {
      if (single(slot)) card(slot).save_kv(b, e, host);
      else batch(slot).save_kv(local(slot), b, e, host);
    });
  }
  void load_kv(uint32_t slot, uint32_t b, uint32_t e, const void* host) override {
    on(slot, [&] {
      if (single(slot)) card(slot).load_kv(b, e, host);
      else batch(slot).load_kv(local(slot), b, e, host);
    });
  }

 private:
  struct Facts {
    uint32_t max_len = 0, mtp_k = 0, verify_k = 0, block = 0;
    uint64_t kv_form = 0, drafter_key = 0;
  };
  // A posted call and the rows (of the multi-slot call) it serves.
  struct Post {
    std::future<void> done;
    std::vector<size_t> rows;
  };
  bool single(uint32_t slot) const { return cards_.at(slot_card_.at(slot)).engine != nullptr; }
  uint32_t local(uint32_t slot) const { return slot_local_.at(slot); }
  server::EngineIface& card(uint32_t slot) { return *cards_.at(slot_card_.at(slot)).engine; }
  server::BatchEngineIface& batch(uint32_t slot) { return *cards_.at(slot_card_.at(slot)).slots; }
  CardWorker& worker(uint32_t slot) { return *workers_.at(slot_card_.at(slot)); }
  // Runs fn on slot `slot`'s card's thread (card `c`'s) and waits; rethrows what it threw.
  void on(uint32_t slot, std::function<void()> fn) { worker(slot).post(std::move(fn)).get(); }
  void on_card(uint32_t c, std::function<void()> fn) { workers_.at(c)->post(std::move(fn)).get(); }
  // The rows on cards with slots, grouped per card in the call's order.
  template <class R>
  std::vector<std::vector<size_t>> batched_rows(const std::vector<R>& rows) const {
    std::vector<std::vector<size_t>> per(cards_.size());
    for (size_t r = 0; r < rows.size(); ++r) {
      if (rows[r].slot >= facts_.size())
        throw std::out_of_range("ReplicaEngine: no slot " + std::to_string(rows[r].slot));
      if (!single(rows[r].slot)) per[slot_card_[rows[r].slot]].push_back(r);
    }
    std::vector<std::vector<size_t>> out;
    for (auto& p : per)
      if (!p.empty()) out.push_back(std::move(p));
    return out;
  }
  // One call of a card's batched engine over the call's rows `mine` (all on that card, in the call's order), posted
  // to its thread: the rows with the card's own slot numbers; its runs, or a SlotsFailed of its, onto those rows.
  template <class R, class Call>
  Post post_card(const std::vector<R>& rows, const std::vector<size_t>& mine, std::vector<std::vector<uint32_t>>& runs,
                 std::vector<std::string>& errors, Call call) {
    const uint32_t c = slot_card_[rows[mine[0]].slot];
    std::vector<R> own;
    for (size_t r : mine) {
      own.push_back(rows[r]);
      own.back().slot = local(rows[r].slot);
    }
    std::future<void> done = workers_[c]->post([this, c, own = std::move(own), mine, &runs, &errors, call] {
      try {
        std::vector<std::vector<uint32_t>> got = call(*cards_[c].slots, own);
        if (got.size() != mine.size())
          throw std::logic_error("ReplicaEngine: card " + std::to_string(c) + " returned " +
                                 std::to_string(got.size()) + " results for " + std::to_string(mine.size()) + " rows");
        for (size_t i = 0; i < mine.size(); ++i) runs[mine[i]] = std::move(got[i]);
      } catch (const server::SlotsFailed& f) {
        scatter(f, mine, &runs, &errors);
      }
    });
    return {std::move(done), mine};
  }
  // A card's SlotsFailed onto the call's rows `mine` (its rows in order): its failed rows' errors, the others' runs.
  static void scatter(const server::SlotsFailed& f, const std::vector<size_t>& mine,
                      std::vector<std::vector<uint32_t>>* runs, std::vector<std::string>* errors) {
    for (size_t i = 0; i < mine.size(); ++i) {
      std::string e = i < f.errors.size() ? f.errors[i] : std::string(f.what());
      if (e.empty() && (i >= f.runs.size() || f.runs[i].empty())) e = f.what();
      if (!e.empty()) (*errors)[mine[i]] = e;
      else (*runs)[mine[i]] = f.runs[i];
    }
  }
  // Waits for every posted call; a call that threw fails its rows, and with the rows a card's SlotsFailed named
  // (`errors`, per row) they become one SlotsFailed (per row, in order) with the completed rows' results - nothing
  // returns before every card is done (no task outlives the call's locals).
  // `errors` is read only after every call is done (the cards' tasks write into it).
  static void join(std::vector<Post>& done, std::vector<std::string>& errors,
                   std::vector<std::vector<uint32_t>>* runs) {
    for (Post& p : done) p.done.wait();
    bool failed = false;
    for (const std::string& e : errors)
      if (!e.empty()) failed = true;
    for (Post& p : done) {
      std::string error;
      try {
        p.done.get();
      } catch (const std::exception& e) {
        error = e.what();
        if (error.empty()) error = "a card failed";
      } catch (...) {
        error = "a card failed";
      }
      if (error.empty()) continue;
      for (size_t r : p.rows) errors[r] = error;
      failed = true;
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
  std::vector<uint32_t> slot_card_, slot_local_;      // per slot: its card, its number on the card
  std::vector<std::future<void>> inflight_;          // per slot: the prefill started, not yet joined
  std::vector<std::unique_ptr<CardWorker>> workers_;   // one thread per card
};

}  // namespace cli
