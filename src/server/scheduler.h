#pragma once
// Spec 13 §3 and §8 (plan 13c Task 1): the batching scheduler, host only. It owns a
// BatchEngineIface (server/batch_engine.h) and serves up to slots() requests at once: a
// request is queued, granted a slot (its spec 7 prefix plan is taken then, not at arrival),
// prefilled, then decoded one id per tick in a step shared with every other decoding slot.
//
// One tick (Scheduler::tick):
//   1. cancellations: a cancelled request leaves the queue or frees its slot (a client
//      disconnect never stalls the others: its row is simply not in the next step);
//   2. grants: waiting requests take free slots in arrival order (Admission::Batched holds
//      them up to `burst_hold_s` after the first arrival, or until they fill the free slots);
//   3. at most one prefill call:
//        Admission::Interleave (the default) - one chunk of the oldest prefilling request,
//          at most `chunk` ids while other rows are decoding (the whole rest otherwise);
//        Admission::Batched - every granted prompt, whole, in one call (spec 13 §8: one
//          admission prefill for a burst; it stalls the decoding rows for that call);
//   4. one decode step over every decoding slot; per row, server.cc's per-id rules: EOS
//      (unless ignore_eos or min_tokens not reached) ends the request without emitting it,
//      otherwise the id goes to on_token (which may end it: a stop string on the text),
//      and max_tokens steps end it with Length. A finishing slot's final state is
//      snapshotted (SlotCacheIface::end) before anything else can take the slot.
//
// Threading: submit() and cancel() may be called from any thread; engine calls and the
// callbacks run on the thread calling tick() (the start() loop in the server, the test's
// own thread in tests) with no scheduler lock held.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "server/batch_engine.h"
#include "server/deps.h"
#include "server/prefix_cache.h"

namespace server {

enum class Admission { Interleave, Batched };

struct SchedulerOptions {
  Admission admission = Admission::Interleave;
  uint32_t chunk = 512;          // Interleave: prompt ids per tick while rows decode (decide, 13a)
  double burst_hold_s = 0.005;   // Batched: admission waits this long after the first arrival
  size_t queue_depth = 4;        // requests waiting beyond the free slots; more get nullptr
  std::vector<uint32_t> eos_ids;
  std::function<double()> clock; // seconds; empty = std::chrono::steady_clock
};

struct Finish {
  enum Reason { Stop, Length, Cancelled, Error } reason = Length;
  std::string error;             // Error only
  uint32_t prompt_tokens = 0;
  uint32_t completion_tokens = 0;   // ids passed to on_token
  uint32_t cached_tokens = 0;       // prompt positions the slot grant reused (spec 7)
  int slot = -1;                    // -1: never granted
};

struct SchedRequest {
  std::vector<uint32_t> prompt;     // non-empty, prompt.size() + 1 <= max_len
  uint32_t max_tokens = UINT32_MAX; // clamped to max_len - prompt.size()
  uint32_t min_tokens = 0;
  bool ignore_eos = false;
  Sampling sampling;
  // Tick thread, in order: every id the request keeps (an EOS that ends it is not passed).
  // Return false to end the request with Stop (a stop string found in the text).
  std::function<bool(uint32_t)> on_token;
  std::function<void(const Finish&)> on_done;   // exactly once, after the last on_token
};

// A submitted request. Held by the caller and the scheduler; finished() is safe anywhere.
class Job {
 public:
  bool finished() const { return finished_.load(); }

 private:
  friend class Scheduler;
  enum class State { Waiting, Prefilling, Decoding };
  SchedRequest req;
  std::atomic<bool> cancelled_{false};
  std::atomic<bool> finished_{false};
  std::mt19937_64 rng;       // the request's own generator (Review Focus 4)
  double arrived = 0;
  uint64_t order = 0;        // arrival order
  // Tick thread only.
  State state = State::Waiting;
  int slot = -1;
  uint32_t max_tokens = 0;
  uint32_t prefilled = 0;    // prompt ids the slot holds
  uint32_t steps = 0;        // decode steps taken (the EOS that ends it included)
  Finish fin;
};

// Spec 7 per slot: the scheduler's prefix-cache calls. PrefixSlots is the implementation
// over PrefixCache; a null cache means reset() and a full prefill per request.
struct SlotCacheIface {
  virtual ~SlotCacheIface() = default;
  // Positions of `prompt` the slot's resident session already holds (slot affinity).
  virtual uint32_t match(uint32_t slot, const std::vector<uint32_t>& prompt) = 0;
  // At the slot grant: plan `prompt` against the slot's resident session and the host
  // store, restore into the slot; returns the positions reused (< prompt.size()).
  virtual uint32_t begin(uint32_t slot, const std::vector<uint32_t>& prompt) = 0;
  // After a prefill call: the slot holds prompt[0:end); `last` when the prompt is in.
  virtual void prefilled(uint32_t slot, uint32_t end, bool last) = 0;
  virtual void fed(uint32_t slot, uint32_t id) = 0;   // every id a step consumed
  virtual void end(uint32_t slot) = 0;   // request end: snapshot the slot's final state
  virtual void fail(uint32_t slot) = 0;  // the slot's session is unknown after an error
};

// The prefix cache per slot: one host store (PrefixCache) shared by the slots, one resident
// session per slot. Blocks and the prompt-end snapshot are stored after the prefill call
// that reaches a block end or the prompt's end (no in-prefill hook: a chunk that is not a
// block multiple stores its crossed blocks at the next such call); `--prefix-split-last`
// is not supported here.
class PrefixSlots : public SlotCacheIface {
 public:
  PrefixSlots(BatchEngineIface& engine, size_t budget_bytes, HostAlloc& alloc);
  uint32_t match(uint32_t slot, const std::vector<uint32_t>& prompt) override;
  uint32_t begin(uint32_t slot, const std::vector<uint32_t>& prompt) override;
  void prefilled(uint32_t slot, uint32_t end, bool last) override;
  void fed(uint32_t slot, uint32_t id) override;
  void end(uint32_t slot) override;
  void fail(uint32_t slot) override;

  PrefixCache& cache() { return cache_; }
  PrefixCache::Plan::Kind last_kind(uint32_t slot) const { return slots_.at(slot).kind; }

 private:
  struct Slot {
    std::vector<uint32_t> resident;   // ids the slot's session holds (valid only)
    std::vector<uint32_t> prompt;     // the prompt being prefilled
    bool valid = false;
    PrefixCache::Plan::Kind kind = PrefixCache::Plan::Cold;
  };
  void store(uint32_t slot, uint32_t end);

  BatchEngineIface& engine_;
  PrefixCache cache_;
  std::vector<Slot> slots_;
};

class Scheduler {
 public:
  Scheduler(BatchEngineIface& engine, SchedulerOptions opts, SlotCacheIface* cache = nullptr);
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // Thread-safe. nullptr when the queue is full (the server's 503: more than queue_depth
  // requests would wait beyond the free slots). Throws std::invalid_argument for an empty
  // prompt or one that leaves no room to generate.
  std::shared_ptr<Job> submit(SchedRequest r);
  // Thread-safe: a client disconnect. The request ends with Cancelled at the next tick.
  void cancel(const std::shared_ptr<Job>& job);

  // One scheduling round on the calling thread. False when there was nothing to do.
  bool tick();
  // A thread that ticks while there is work and sleeps otherwise; stop() joins it.
  void start();
  void stop();

  size_t waiting() const;    // queued, not yet granted
  size_t active() const;     // holding a slot (tick thread's view, read under the lock)
  // The seconds until a held burst is admitted; < 0 when nothing is held.
  double hold_remaining() const;

 private:
  struct Impl;
  void grant(const std::shared_ptr<Job>& job, uint32_t slot);
  bool prefill_round();
  bool decode_round();
  void finish(const std::shared_ptr<Job>& job, Finish::Reason reason, std::string error = {});
  uint32_t pick_slot(const Job& job) const;
  double now() const;

  BatchEngineIface& engine_;
  SchedulerOptions opts_;
  SlotCacheIface* cache_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
