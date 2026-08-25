#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"

namespace runtime {

// The decode loop, and nothing else. It owns the weights, the buffers and the
// one captured command list, and a token is a replay of that list: the host
// writes at most four bytes of shared memory (the ingested id), submits, waits
// on the fence, reads four bytes back. No kernel is appended, no argument is
// rebound and no branch in the loop depends on a value the device produced -
// that is what makes a replay bitwise reproducible, and
// tests/runtime/replay_determinism_test proves it for the raw loop this class
// wraps.
//
// Not thread-safe and deliberately single-queue: one Engine drives one device
// context, and `generate()` blocks on every token.
class Engine {
 public:
  // debug_resid=true captures the per-layer tap (golden gate); costs 64 device
  // copies per token - off for benchmarking. `max_len` must equal the max_len
  // `model` was loaded with (the RoPE table, the KV allocation and the
  // attention variants' baked MAXLEN are one number - runtime::build throws if
  // they disagree). The constructor captures the list and then reset()s, so a
  // fresh Engine starts from an explicitly zeroed state.
  Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len,
         bool debug_resid = false);

  // Zeroes exactly the persistent group - control, gdn_state, conv_ring, kv_k,
  // kv_v. Scratch is deliberately NOT zeroed: no step may read scratch it has
  // not first written, and replay_determinism_test's run C is the standing
  // proof of that. Zeroing scratch here would mask a regression of it.
  void reset();

  // Feeds prompt ids one per replay (argmax result overwritten by the next
  // write; pos advances in-kernel). After ingest, control.pos == ids.size()
  // for a freshly reset Engine - ingest is incremental, so it may be called
  // repeatedly to extend a session.
  void ingest(const std::vector<uint32_t>& ids);

  // Greedy-generates n ids; on_token is called after each fence (host side,
  // overlaps nothing in v1). Returns the ids.
  //
  // Token i is read out of the control block *before* replay i - it is what
  // the previous fence (the last ingest step, or generation step i-1) sampled.
  // Replay i then embeds it and samples token i+1, so `n` tokens cost `n`
  // replays, `pos` advances by `n`, and the (n+1)-th token is left pending in
  // the control block: two back-to-back generate() calls are one continuous
  // stream.
  std::vector<uint32_t> generate(uint32_t n,
                                 const std::function<void(uint32_t)>& on_token = {});

  // Over the last generate() call, wall clock around the whole loop (callback
  // included). 0 before the first call.
  double last_tok_per_s() const { return last_tok_per_s_; }
  // The same call, split for Task 9's breakdown: wall time, and the part of it
  // spent inside execute+fence.wait(). Both milliseconds, both measured.
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  // Golden-gate accessors: the per-layer resid tap for the last replay, read
  // back in debug buffer order [layer][m][5120] bf16. What tap[L] *means* is
  // documented in exactly one place - runtime/capture.h's `debug_resid`
  // paragraph - and is not restated here.
  std::vector<uint16_t> read_debug_resid();  // throws unless debug_resid
  bool debug_resid() const { return tap_ != nullptr; }

  l0::Context& context() const { return ctx_; }
  const loader::LoadedModel& model() const { return model_; }
  DecodeBuffers& buffers() { return buffers_; }
  const CapturedStep& step() const { return step_; }
  // Position of the next token, i.e. how many tokens this session has consumed.
  uint32_t pos() const { return control_->pos; }
  uint32_t max_len() const { return buffers_.max_len; }

 private:
  // One replay: the precondition, the submit, the wait. Every attention
  // kernel's bound assumes `pos + n_active <= max_len`, and the caller of the
  // list is what guarantees it (Task 5 ruling) - the Engine is that caller, so
  // the check lives here permanently and throws rather than replaying past the
  // KV cache and the RoPE table.
  void replay();

  l0::Context& ctx_;
  loader::LoadedModel model_;
  DecodeBuffers buffers_;
  std::unique_ptr<l0::Mem> tap_;   // null unless debug_resid
  CapturedStep step_;
  l0::Queue queue_;
  l0::Fence fence_;
  l0::CmdList imm_;                // uploads and readbacks only, never a token
  Control* control_;               // shared memory, inside buffers_.control
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
