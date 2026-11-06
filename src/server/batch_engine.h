#pragma once
// Spec 13 (plan 13b Task 1): what the batching scheduler (server/scheduler.h) needs from a
// batched engine. Plan 13c implements it on runtime::Engine (per-slot state, KV base and
// Control; one captured decode list per batch size); tests/server/mock.h has a host mock
// (MockBatchEngine) the scheduler is written and tested against.
//
// Every slot follows runtime::Engine's single-sequence protocol:
//   - a prefill chunk appends ids to the slot's session at pos(slot); the chunk that ends
//     the prompt (`last`) leaves the slot's first generated id pending;
//   - a decode step returns the slot's pending id and consumes it (its KV is written,
//     pos advances by one) and leaves the next id pending.
// Calls come from one thread (the scheduler's tick loop); nothing here is thread-safe.
//
// Plan 13b (spec 13 §11, the two-engine mode): an engine whose slots are separate devices
// (cli::ReplicaEngine: one runtime::Engine per card) also says how a prompt is cut between
// ticks (prefill_cut), runs a prefill call beside a step (overlaps_prefill), fails one slot
// without the others (SlotsFailed), verifies drafts proposed outside it per row
// (SpecRow::propose, prompt lookup) and feeds --prefix-split-last's last prompt id through
// the decode replay (Chunk::replay).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "server/deps.h"

namespace server {

// Plan 13b: some rows (or chunks) of one call failed and the others completed - an engine
// whose slots are independent devices throws this, so that a failure on one card ends that
// slot's request only (Review Focus 2). `errors` is per row / chunk in call order (empty: it
// completed); `runs` per row the result it would have returned (step_batch: one id; step_spec:
// the run; prefill: nothing), empty for a failed row.
struct SlotsFailed : std::runtime_error {
  SlotsFailed(std::vector<std::string> e, std::vector<std::vector<uint32_t>> r)
      : std::runtime_error(first(e)), errors(std::move(e)), runs(std::move(r)) {}
  std::vector<std::string> errors;
  std::vector<std::vector<uint32_t>> runs;

 private:
  static std::string first(const std::vector<std::string>& e) {
    for (const std::string& s : e)
      if (!s.empty()) return s;
    return "a slot failed";
  }
};

struct BatchEngineIface {
  virtual ~BatchEngineIface() = default;

  virtual uint32_t slots() = 0;              // B: sequences one decode step can carry
  virtual uint32_t max_len() = 0;            // positions per slot (plan 13b: the largest slot's)
  // Plan 13b: the positions slot `slot` holds (a replica engine's cards may differ).
  virtual uint32_t slot_max_len(uint32_t /*slot*/) { return max_len(); }
  virtual uint32_t pos(uint32_t slot) = 0;   // ids the slot's session holds
  virtual void reset(uint32_t slot) = 0;     // empty the slot (pos 0, nothing pending)

  // One prefill call over one or more slots: chunk interleaving passes one chunk of one
  // slot; batched admission (spec 13 §8) passes every admitted prompt in one call.
  struct Chunk {
    uint32_t slot;
    const uint32_t* ids;
    uint32_t n;      // > 0
    bool last;       // the prompt ends here: the first generated id is then pending
    // Plan 13b (--prefix-split-last): the ids go through the decode replay one each
    // (EngineIface::ingest), not the prefill - the last prompt id, after the prefill to
    // len - 1 whose end the prompt-end snapshot holds.
    bool replay = false;
  };
  virtual void prefill(const std::vector<Chunk>& chunks) = 0;
  // Plan 13b: how many of `remaining` prompt ids the slot's next prefill call takes at `pos`;
  // 0 = the scheduler's own rule (SchedulerOptions::chunk). A replica engine returns its
  // card's own cut (runtime::prefill_chunk_rows, the cut one runtime::Engine::prefill call
  // makes): the chunk boundaries are part of what a prefill computes, so a prompt cut between
  // ticks must be cut where the serial server's one call cuts it (bitwise per card, R1).
  virtual uint32_t prefill_cut(uint32_t /*slot*/, uint32_t /*pos*/, uint32_t /*remaining*/) { return 0; }
  // Plan 13b: true when a prefill call and a step over OTHER slots can run at once (one card
  // per slot). The scheduler then starts the prefill (prefill_begin), runs the tick's step and
  // joins the prefill (prefill_end, which throws what the prefill threw) - so a long prompt's
  // chunk stalls only its own card. The defaults run the prefill in prefill_begin.
  virtual bool overlaps_prefill() { return false; }
  virtual void prefill_begin(const std::vector<Chunk>& chunks) { prefill(chunks); }
  virtual void prefill_end() {}

  // One decode step over the active rows (distinct slots, each with an id pending). Returns,
  // per row in order, the slot's pending id (consumed). The next pending id is the argmax,
  // or with !sampling->greedy a draw from the filtered row (server::filter_probs) with
  // *rng - the request's own generator, so a seeded request draws the same ids whatever
  // else shares the step (plan 13b Review Focus 4).
  struct Row {
    uint32_t slot;
    const Sampling* sampling;
    std::mt19937_64* rng;
  };
  virtual std::vector<uint32_t> step_batch(const std::vector<Row>& rows) = 0;

  // Spec 19d (spec 19 §5 "Batching": one drafter ring per slot, the draft list batching slots as the verify does):
  // speculative decoding under the scheduler. spec_k() == 0 (the default) means the scheduler calls step_batch()
  // only, exactly as before. Otherwise step_spec(rows) runs ONE draft / verify / accept iteration over every row
  // (distinct slots, each with an id pending): row r drafts up to rows[r].k ids (<= spec_k(); 0 = a plain step; the
  // engine may draft fewer near max_len) from ITS SLOT'S proposer state alone - a DFlash drafter's ring per slot,
  // which no other slot's row reads or writes - verifies them, and keeps the longest acceptable prefix (greedy: the
  // target's argmax; sampled: the lossless rule with the row's own *rng, so a seeded request draws the same ids
  // whatever shares the step). It returns per row the pending id then the kept drafts (1 .. k + 1 ids); pos(slot)
  // counts them all and the next id is pending. truncate_to(slot, pos) keeps fewer of the slot's last run (an EOS,
  // a stop or max_tokens inside it): the slot is then exactly what that many plain steps leave - EngineIface's
  // truncate_to, per slot. The slot's snapshots (save_state) carry its ring, keyed by drafter_key().
  //
  // Plan 13b (prompt lookup per request, Review Focus 4): a row with `propose` set verifies
  // drafts proposed OUTSIDE the engine, as EngineIface::step_drafts does: the engine calls
  // propose(x) exactly once with the slot's pending id x, verifies what it answers (at most
  // verify_k() ids; the proposer has already cut them to the row's k) and returns x and the
  // kept drafts. verify_k() == 0 (the default): no row ever carries a proposer.
  using Proposer = std::function<std::vector<uint32_t>(uint32_t pending)>;
  struct SpecRow {
    uint32_t slot;
    const Sampling* sampling;
    std::mt19937_64* rng;
    uint32_t k;      // drafts asked for this iteration (the request's own policy, server::AdaptiveK)
    const Proposer* propose = nullptr;   // plan 13b: the request's own drafts (prompt lookup)
  };
  virtual uint32_t spec_k() { return 0; }
  virtual uint32_t verify_k() { return 0; }
  virtual std::vector<std::vector<uint32_t>> step_spec(const std::vector<SpecRow>& rows) {
    std::vector<Row> plain;
    for (const SpecRow& r : rows) plain.push_back({r.slot, r.sampling, r.rng});
    std::vector<std::vector<uint32_t>> out;
    for (uint32_t id : step_batch(plain)) out.push_back({id});
    return out;
  }
  virtual void truncate_to(uint32_t slot, uint32_t pos) {
    if (pos != this->pos(slot))
      throw std::logic_error("BatchEngineIface::truncate_to: no speculative run to shorten on this engine");
  }

  // Spec 7 (plan 7b's snapshot calls), per slot. The host layouts are EngineIface's: one
  // state blob of state_bytes(), KV in save_kv's layout, kv_bytes(n) for n positions.
  // Defaults throw, so an engine without them runs with the prefix cache off only.
  virtual uint32_t block() { return 2048; }
  virtual size_t state_bytes() { return unsupported<size_t>(); }
  virtual size_t kv_bytes(uint32_t) { return unsupported<size_t>(); }
  // Spec 12b: the KV cache's form (0 = bf16, 1 = int8), which keys the prefix cache so a
  // bf16 entry is never restored into an int8 engine (EngineIface::kv_form's twin).
  virtual uint64_t kv_form() { return 0; }
  // Spec 19d: EngineIface::drafter_key's twin (every slot's ring is the same drafter's).
  virtual uint64_t drafter_key() { return 0; }
  virtual void save_state(uint32_t /*slot*/, void*) { unsupported<int>(); }
  virtual void load_state(uint32_t /*slot*/, const void*, uint32_t /*pos*/) { unsupported<int>(); }
  virtual void save_kv(uint32_t /*slot*/, uint32_t, uint32_t, void*) { unsupported<int>(); }
  virtual void load_kv(uint32_t /*slot*/, uint32_t, uint32_t, const void*) { unsupported<int>(); }

 private:
  template <class T>
  static T unsupported() {
    throw std::logic_error("BatchEngineIface: this engine does not support the prefix cache");
  }
};

}  // namespace server
