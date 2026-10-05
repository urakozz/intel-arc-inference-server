#pragma once
// Spec 13 (plan 13c Task 1): what the batching scheduler (server/scheduler.h) needs from a
// batched engine. Plan 13b implements it on runtime::Engine (per-slot state, KV base and
// Control; one captured decode list per batch size); tests/server/mock.h has a host mock
// (MockBatchEngine) the scheduler is written and tested against.
//
// Every slot follows runtime::Engine's single-sequence protocol:
//   - a prefill chunk appends ids to the slot's session at pos(slot); the chunk that ends
//     the prompt (`last`) leaves the slot's first generated id pending;
//   - a decode step returns the slot's pending id and consumes it (its KV is written,
//     pos advances by one) and leaves the next id pending.
// Calls come from one thread (the scheduler's tick loop); nothing here is thread-safe.
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "server/deps.h"

namespace server {

struct BatchEngineIface {
  virtual ~BatchEngineIface() = default;

  virtual uint32_t slots() = 0;              // B: sequences one decode step can carry
  virtual uint32_t max_len() = 0;            // positions per slot
  virtual uint32_t pos(uint32_t slot) = 0;   // ids the slot's session holds
  virtual void reset(uint32_t slot) = 0;     // empty the slot (pos 0, nothing pending)

  // One prefill call over one or more slots: chunk interleaving passes one chunk of one
  // slot; batched admission (spec 13 §8) passes every admitted prompt in one call.
  struct Chunk {
    uint32_t slot;
    const uint32_t* ids;
    uint32_t n;      // > 0
    bool last;       // the prompt ends here: the first generated id is then pending
  };
  virtual void prefill(const std::vector<Chunk>& chunks) = 0;

  // One decode step over the active rows (distinct slots, each with an id pending). Returns,
  // per row in order, the slot's pending id (consumed). The next pending id is the argmax,
  // or with !sampling->greedy a draw from the filtered row (server::filter_probs) with
  // *rng - the request's own generator, so a seeded request draws the same ids whatever
  // else shares the step (plan 13c Review Focus 4).
  struct Row {
    uint32_t slot;
    const Sampling* sampling;
    std::mt19937_64* rng;
  };
  virtual std::vector<uint32_t> step_batch(const std::vector<Row>& rows) = 0;

  // Spec 7 (plan 7b's snapshot calls), per slot. The host layouts are EngineIface's: one
  // state blob of state_bytes(), KV in save_kv's layout, kv_bytes(n) for n positions.
  // Defaults throw, so an engine without them runs with the prefix cache off only.
  virtual uint32_t block() { return 2048; }
  virtual size_t state_bytes() { return unsupported<size_t>(); }
  virtual size_t kv_bytes(uint32_t) { return unsupported<size_t>(); }
  // Spec 12b: the KV cache's form (0 = bf16, 1 = int8), which keys the prefix cache so a
  // bf16 entry is never restored into an int8 engine (EngineIface::kv_form's twin).
  virtual uint64_t kv_form() { return 0; }
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
