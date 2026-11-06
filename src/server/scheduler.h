#pragma once
// Spec 13 §3 and §8 (plan 13b Task 1): the batching scheduler, host only. It owns a
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
//      Spec 19d: with a speculative engine (BatchEngineIface::spec_k() > 0) the step is one
//      draft / verify / accept iteration (step_spec): each row asks for K drafts from its
//      request's own policy (server::AdaptiveK over SchedulerOptions::spec, fed only by that
//      request's acceptance - or spec_fixed_k), gets back 1..K + 1 ids and runs them through
//      the same per-id rules in order; ids past the one that ends the request are handed
//      back with truncate_to(slot, pos) BEFORE the request-end snapshot, so the slot holds
//      exactly the prompt and the ids consumed, as plain steps would leave it.
//
// Plan 13b (spec 13 §11, the server's features since Task 1 carried through the scheduler):
//   - per-slot lengths: a request is granted a free slot it fits (slot_max_len), its
//     max_tokens clamped to that slot's room;
//   - the prefill cut: an engine with prefill_cut() decides every chunk (each call = one of
//     its card's own prefill chunks, so a replica engine's prompt is cut where the serial
//     server's one call cuts it); with overlaps_prefill() (one card per slot) every prefilling
//     slot has its next chunk in flight at once, polled each tick and joined when done
//     (prefill_ready / prefill_end), so a card's prefill never slows the other card's steps and
//     two prompts prefill on the two cards together;
//   - --prefix-split-last (SchedRequest::split_last, with a cache): the prompt is prefilled
//     to len - 1 (its prompt-end snapshot there) and its last id goes through the decode
//     replay (Chunk::replay) - PrefixSession::begin(prompt, true)'s order, per slot;
//   - prompt lookup (SchedulerOptions::lookup, an engine with verify_k() > 0): each request's
//     drafts come from its slot's matcher (server::PromptLookup over the request's prompt and
//     every id it consumed, plus with lookup_history the slot's earlier requests' generated
//     ids) - never another slot's (Review Focus 4); its K from its own AdaptiveK over
//     lookup_adaptive, fed (drafts proposed, drafts kept) as server.cc's loop feeds it;
//   - per-slot failures: an engine that throws SlotsFailed fails only the rows it names; a
//     callback (on_token / on_done) that throws ends its own request with Error and nothing
//     else - the other rows of the step still get their ids, the engine stays usable.
//
// Threading: submit() and cancel() may be called from any thread; engine calls and the
// callbacks run on the thread calling tick() (the start() loop in the server, the test's
// own thread in tests) with no scheduler lock held.
#include <atomic>
#include <cstddef>
#include <exception>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "server/adaptive_k.h"
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
  // Spec 19d, a speculative engine only: each request's K policy (max_k clamped to the
  // engine's spec_k(); the cost table is the proposer's, e.g. MtpCost::dflash), or a fixed K
  // for every iteration when spec_fixed_k > 0.
  AdaptiveKOptions spec;
  uint32_t spec_fixed_k = 0;
  // Plan 13b: prompt lookup (`--spec lookup`, spec 19e) per request - server::Options'
  // spec_lookup / lookup_min_match / lookup_history / lookup_adaptive, with the server's
  // meaning, per slot. Needs an engine with verify_k() > 0 and spec_k() == 0.
  bool lookup = false;
  uint32_t lookup_min_match = 3;
  uint32_t lookup_history = 0;
  AdaptiveKOptions lookup_adaptive;
};

struct Finish {
  enum Reason { Stop, Length, Cancelled, Error } reason = Length;
  std::string error;             // Error only
  uint32_t prompt_tokens = 0;
  uint32_t completion_tokens = 0;   // ids passed to on_token
  uint32_t cached_tokens = 0;       // prompt positions the slot grant reused (spec 7)
  int slot = -1;                    // -1: never granted
  // Spec 19d: speculative iterations, drafts asked for, drafts kept (consumed ids only).
  uint32_t spec_iters = 0, spec_drafted = 0, spec_accepted = 0;
};

struct SchedRequest {
  std::vector<uint32_t> prompt;     // non-empty, prompt.size() + 1 <= max_len
  uint32_t max_tokens = UINT32_MAX; // clamped to max_len - prompt.size()
  uint32_t min_tokens = 0;
  bool ignore_eos = false;
  // Plan 13b: --prefix-split-last (a chat request): prefill to len - 1, the last id by the
  // decode replay, so the prompt-end snapshot sits at len - 1. Only with a cache.
  bool split_last = false;
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
  std::unique_ptr<AdaptiveK> policy;   // spec 19d: this request's K (speculative engines)
  bool split = false;        // plan 13b: split_last honoured (a cache is present)
  // Plan 13b, prompt lookup: the row's proposer over the slot's matcher, this iteration's K,
  // the drafts it proposed, whether the pending id went into the matcher, and the ids kept
  // (for lookup_history).
  BatchEngineIface::Proposer propose;
  uint32_t iter_k = 0, proposed = 0;
  bool head_indexed = false;
  std::vector<uint32_t> out;
  // Plan 13b, an overlapping engine: a chunk of this request is in flight (prefill_begin'd, not
  // yet joined), and which.
  bool inflight = false;
  BatchEngineIface::Chunk chunk{0, nullptr, 0, false, false};
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
  // Plan 13b: positions the slot's resident session holds (what a grant there would evict);
  // the grant prefers, among equal matches, the slot that holds least.
  virtual uint32_t held(uint32_t /*slot*/) { return 0; }
};

// The prefix cache per slot: one host store (PrefixCache) shared by the slots, one resident
// session per slot. Blocks and the prompt-end snapshot are stored after the prefill call
// that reaches a block end or the prompt's end (no in-prefill hook: a chunk that is not a
// block multiple stores its crossed blocks at the next such call). `--prefix-split-last`
// (plan 13b): the scheduler calls prefilled(slot, len - 1, true) after the prefill to len - 1
// (the prompt-end snapshot there) and fed(slot, last id) after the replay.
class PrefixSlots : public SlotCacheIface {
 public:
  PrefixSlots(BatchEngineIface& engine, size_t budget_bytes, HostAlloc& alloc);
  uint32_t match(uint32_t slot, const std::vector<uint32_t>& prompt) override;
  uint32_t begin(uint32_t slot, const std::vector<uint32_t>& prompt) override;
  void prefilled(uint32_t slot, uint32_t end, bool last) override;
  void fed(uint32_t slot, uint32_t id) override;
  void end(uint32_t slot) override;
  void fail(uint32_t slot) override;
  uint32_t held(uint32_t slot) override;

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
  struct PrefillCall;   // plan 13b: one prefill call's requests and chunks
  void grant(const std::shared_ptr<Job>& job, uint32_t slot);
  BatchEngineIface::Chunk next_chunk(const Job& job, bool cut);
  bool prefill_plan(PrefillCall& call);
  void prefill_done(PrefillCall& call, std::exception_ptr error);
  bool overlapped_prefills();   // plan 13b: join the finished chunks, start the next ones
  void lookup_begin(uint32_t slot, const std::vector<uint32_t>& prompt);   // plan 13b
  void lookup_end(uint32_t slot, const std::vector<uint32_t>& out);
  bool decode_round();
  bool spec_round();   // spec 19d
  void finish(const std::shared_ptr<Job>& job, Finish::Reason reason, std::string error = {});
  int pick_slot(const Job& job) const;   // -1: no free slot the prompt fits
  double now() const;

  BatchEngineIface& engine_;
  SchedulerOptions opts_;
  SlotCacheIface* cache_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
