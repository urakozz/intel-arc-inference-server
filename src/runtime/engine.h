#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
#include "runtime/prefill_backend.h"

namespace runtime {

// The prefill path's per-engine state: `runtime::prefill::Context` (the L0
// immediate list and the SYCL queue) and its `KernelCache`. **Incomplete here
// on purpose.** Everything that touches it -- `Engine::prefill`,
// `prefill_launches`, and the deleter -- is compiled into `b70_prefill_host`
// and linked only by a target that calls `b70_link_prefill()`. `b70_runtime`
// is linked by every decode binary and test, and none of those may acquire a
// dependency on libb70_prefill.so (cmake/prefill.cmake's whole argument), so
// the member below is a `unique_ptr` with a FUNCTION-POINTER deleter: that
// instantiation needs no complete type and emits no reference to any prefill
// symbol in a translation unit that never calls `prefill()`.
struct PrefillEngine;

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

  // Prefill `ids` in chunks of at most `chunk` positions starting at the
  // current pos; on return pos == old pos + ids.size(), the persistent state
  // holds those positions, and control_->cur_token[0] is the argmax of the LAST
  // position's logits (the first generated id) -- exactly as ingest() would
  // have left it, which is what tests/prefill/prefill_consistency_test.cc
  // grades. Default chunk = PrefillScratch::kC (ruling A7/A13, 2048).
  //
  // Throws if `ids` is empty, if `chunk > PrefillScratch::kC`, if any id is
  // outside the vocabulary (`pf_embed_gather` has no `Control::debug_flag`
  // channel, so the host is the only bound), or if pos + ids.size() > max_len.
  // Allocates `PrefillScratch` and the prefill `Context` on the FIRST call
  // (ruling R7): a decode-only Engine's device residency is unchanged.
  //
  // **Defined in b70_prefill_host, not in b70_runtime** -- see PrefillEngine
  // above. A binary that never calls this links exactly what it linked before.
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Everything prefill() sets up before its first chunk, done now: the prefill
  // scratch and Context, and on the l0-int8 backend every int4 linear's rotated
  // column scales (spec 5 T2, one pf_colmax_rot pass per linear). Idempotent;
  // prefill() calls it too. A CLI that serves prompts calls it right after
  // load, so the one-time scale pass (about 140 ms) never lands inside a
  // request. Decode-only engines never call it (ruling R7 stays true).
  void prepare_prefill();

  // Non-null only after the first prefill(); for the tests and the CLI report.
  const PrefillScratch* prefill_scratch() const { return pf_.get(); }
  // Context::launches() -- the L0 launches this engine's prefill path has
  // appended since the first prefill(). 0 before it. The SYCL GEMMs are not on
  // that list and are NOT counted; `runtime::prefill::step_chunk_gemms()` is.
  size_t prefill_launches() const;

  // Spec 2.1: the backend prefill() runs. Unset means the build's default
  // (runtime::prefill::default_prefill_backend(), defined in b70_prefill_host). It may change
  // between prefill() calls -- persistent state is shared, scratch is per backend.
  void set_prefill_backend(PrefillBackend b) { pf_backend_ = b; }
  PrefillBackend prefill_backend() const;          // defined in engine_prefill.cc
  // True once this engine's prefill Context has built its SYCL side. The L0 backend never
  // does; the equivalence test asserts it (spec §5 S3).
  bool prefill_sycl_side_created() const;         // defined in engine_prefill.cc

  // Experimental pure-L0 whole-chunk replay. Off by default; an explicit
  // setting overrides B70_PREFILL_REPLAY=1. Recordings retain frozen (pos,C)
  // arguments, not input contents, and survive reset() with a bounded cache.
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled; }

  // Spec 6 (plan 6b): one line summing the device memory this engine holds --
  //   memory: model <GB>, kv <GB>, decode state <GB>, prefill scratch <GB>, int8 <GB>,
  //   total <GB> of <device GB>
  // model = the loader report's total (RoPE included); kv = kv_k + kv_v; decode state =
  // the rest of PersistentBuffers plus DecodeScratch; prefill scratch = PrefillScratch's
  // bytes() + lazy_bytes() as allocated NOW; int8 = Int8State::bytes(). The CLIs print it
  // after prepare_prefill(), before the first prefill. Defined in engine_prefill.cc.
  std::string memory_line() const;

  // Spec 7 §3.4: snapshots of the session for prefix caching. The GDN state and the
  // conv ring are valid only at the exact position they were computed to, so a restore
  // point needs both; the KV cache of positions [0, pos) is copied by range. Every call
  // is blocking and runs on `imm_` (a synchronous immediate list) outside the captured
  // decode list, which reads `pos` and the persistent buffers these calls write in
  // place. Host pointers are l0::MemKind::Host allocations (device-visible). The caller
  // must not have a prefill or a replay in flight -- none of the public calls leaves one.
  static constexpr uint32_t kBlock = 2048;
  size_t state_bytes() const;                 // gdn_state + conv_ring (166.72 MB)
  size_t kv_bytes(uint32_t n_pos) const;      // n_pos * 16 * 4 * 256 * 2 B, K and V
  // Layout: gdn_state then conv_ring. conv_ring is a ring indexed by pos % kConvRing;
  // the whole ring is copied, so a restore at any pos % 16 is exact.
  void save_state(void* host) const;
  // Writes both, then control.pos = pos, control.n_active = 0. cur_token is NOT
  // restored: prefill at least one id after a restore (spec 7 §3.3 step 4).
  void load_state(const void* host, uint32_t pos);
  // Positions [begin, end) of kv_k and kv_v. Host layout: K [16][end-begin][4][256]
  // bf16, then V the same. begin == end copies nothing (host may be null). Throws
  // unless begin <= end <= max_len.
  void save_kv(uint32_t begin, uint32_t end, void* host) const;
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // Spec 7 §3.2: called on the host after every prefill chunk whose end position is a
  // multiple of kBlock, with the device idle and the persistent state exactly at
  // `end_pos` (control.pos == end_pos, n_active == 0; cur_token is not yet the argmax),
  // and after the LAST chunk of every prefill (the prompt end), whatever its end, once
  // the first generated id is in cur_token. `is_block_end` tells whether end_pos is a
  // multiple of kBlock. A prompt that ends on a block end gets ONE call. With a hook set,
  // every chunk ends at a block end or at the prompt end: chunk rows are
  // min(chunk, kBlock - pos % kBlock, remaining). Without one (an empty function, the
  // default), chunking is exactly as before. If the hook throws, the exception propagates
  // with pos == end_pos: the state holds exactly the chunks written.
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

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
  // The two owned groups, then the view over them. Declaration order is the
  // construction order and `buffers_` binds references into the two above it,
  // so this order is load-bearing, not stylistic. `pf_` stays null until the
  // first prefill() (ruling R7): a decode-only Engine allocates exactly what
  // it allocated before the split, byte for byte.
  PersistentBuffers persist_;
  DecodeScratch decode_scratch_;
  DecodeBuffers buffers_;
  std::unique_ptr<PrefillScratch> pf_;
  // The function-pointer deleter is what keeps `PrefillEngine` incomplete here;
  // both halves stay null until the first prefill().
  std::unique_ptr<PrefillEngine, void (*)(PrefillEngine*)> pfx_{nullptr, nullptr};
  std::optional<PrefillBackend> pf_backend_;   // unset = default_prefill_backend()
  std::optional<bool> pf_replay_;
  BlockHook block_hook_;
  std::unique_ptr<l0::Mem> tap_;   // null unless debug_resid
  CapturedStep step_;
  l0::Queue queue_;
  l0::Fence fence_;
  // Uploads, readbacks and snapshot copies only, never a token. `mutable`: the
  // save_* calls are const on the session and still append to it.
  mutable l0::CmdList imm_;
  Control* control_;               // shared memory, inside buffers_.control
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
