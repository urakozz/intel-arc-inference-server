#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "loader/loader.h"
#include "runtime/buffers.h"

namespace runtime {

// Profiling state for ONE captured list: the timestamp pool and one event per
// launch, in walk order (event i is launch i, and `CapturedStep::labels[i]`
// names it). Caller-owned like the residual tap, and for the same reason -
// `build` only borrows it, so the caller decides whether the instrumentation
// outlives the list and how many replays it accumulates.
//
// The pool is created at `kProfileCapacity`, not at the launch count: the walk
// has not run yet when the caller constructs this, and a pool cannot grow. The
// `events` vector is filled by `build` to exactly `kernel_count` entries, so
// `events.size()` is the honest count and the unused pool slots cost nothing.
//
// `std::vector<l0::Event>` beside the pool it draws from is safe: an `Event`
// holds a handle and a copy of the pool's `TimerCalib`, never a reference to
// the pool, so a reallocation that moves the vector moves handles and leaves
// the pool alone. (The constructor reserves anyway - no reallocation happens.)
//
// Events must be `reset()` before EVERY replay: re-signalling an un-signalled-
// again event is undefined, and `duration_us()` throws on an unsignalled one.
// Both are the caller's business, not `build`'s - the list is captured once and
// replayed by whoever owns the queue.
struct ProfileEvents {
  l0::EventPool pool;              // capacity kProfileCapacity
  std::vector<l0::Event> events;   // one per launch, walk order
  // 774 today (spec §9.1's 645, plus the 129 launches spec 1.5's lever L1 added
  // by splitting every `prep_res_norm` site in two); the headroom is for plan
  // 4's remaining levers, which move the launch count in both directions.
  // `build` throws rather than overrun it.
  static constexpr uint32_t kProfileCapacity = 1024;
  explicit ProfileEvents(l0::Context& ctx);
};

// The decode step, captured once. Owns every Module/Kernel the list references
// (Level Zero resolves a launch's arguments at append time - proven by
// tests/l0/arg_capture_test.cc - but the *handles* must outlive the list) and
// the closed list itself. Replay is `Queue::execute(step.list, &fence)`; the
// host mutates nothing in it, ever.
struct CapturedStep {
  l0::CmdList list;                                            // closed, in-order
  size_t kernel_count = 0;                                     // 774 at M = 1 (spec §9.1 + L1)
  std::map<std::string, std::unique_ptr<l0::Module>> modules;  // by variant name
  std::vector<std::unique_ptr<l0::Kernel>> kernels;            // append order
  // One per launch, walk order: "L<layer> <kernel> <variant>", and
  // "-- <kernel> <variant>" for the six token-boundary launches that belong to
  // no layer (embed_gather, the final norm's two launches, lm_head, and the
  // two argmax stages). ALWAYS filled - 774 small strings built on the host during a
  // capture that already opens 19 device binaries, so there is no profiling
  // switch on them and no way for a profiled walk to be described differently
  // from a plain one.
  std::vector<std::string> labels;
};

// Builds the whole per-token list against `m`'s weights and `b`'s allocations
// and closes it. Throws (std::runtime_error / l0::Error / std::out_of_range) on
// any mismatch it can see at capture: a missing device binary, a buffer whose
// size disagrees with the per-layer slice arithmetic, a `max_len` that is not
// the one the attention variants were compiled for.
//
// debug_resid: when non-null (bf16 [64][M][5120]), append a device-to-device
// copy of `resid` after every layer - the golden gate's per-layer tap. Copies
// are commands, not kernels, so `kernel_count` does not count them and
// determinism is unaffected. The tap runs after the layer's LAST kernel, and
// the residual stream is only advanced by `prep_res_fold` - whose per-element
// arithmetic is the single-work-group `prep_res_norm`'s, unchanged, so one
// launch of it writes the same `resid` bytes for the same inputs.
//
// **That is a per-launch property and NOT a claim that the tap tensors are
// stable across a numerics change**, which is what an earlier draft of this
// comment asserted about spec 1.5's lever L1. L1 reordered the norm's global
// Σx², which moves `x` in the last bf16 ulp and therefore moves every
// subsequent layer's input; the taps this parameter produces were **measured**
// to move in both directions across it (`code`'s worst cosine 0.829 → 0.914,
// `prose`'s 0.99945 → 0.99481) while the gate's token ids stayed 96/96
// element-exact. Read the taps as diagnostics of one engine build, never as a
// contract between two (docs/14, "The diagnostics move with every lever").
//
// So tap[L]
// holds the hidden state with layer L's *mixer* contribution folded in and
// layer L's MLP contribution still sitting un-folded in `partials` (layer
// L+1's leading `prep_res_fold` folds it). No tap holds the final hidden state
// - but after a step's fence, `b.resid` holds it pre-norm (the final
// `prep_res_fold` folds layer 63's MLP into `resid` first, like every prep,
// before its `prep_norm_finish` writes the normalised row to `x`) and `b.x`
// holds its final-normalised form; both are readable without any capture
// change.
//
// prof: when non-null, launch `i` signals `prof->events[i]` - the events are
// created here, one per launch, so the caller never has to know the count in
// advance. `build` throws if the walk has more launches than
// `ProfileEvents::kProfileCapacity`. The signal observes only: nothing in the
// list waits on an event, so a profiled list appends the same commands in the
// same order with the same arguments as a plain one and replays to the same
// tokens (tests/runtime/profile_capture_test.cc asserts exactly that). It is
// not free, though - each signal carries a host-scope flush at kernel
// completion - so a profiled list is never a bench list (spec §3.3): it prices
// shares and per-kernel deltas, not the absolute step.
//
// M: the rows in flight (consecutive positions), default 1 - what ships. M > 1
// binds the `_M<M>` variants, which a default build does not compile (spec 8a:
// B70_DECODE_EXTRA_M builds M = 2..4 at max_len 16384 for the MTP probe).
CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid = nullptr, ProfileEvents* prof = nullptr,
                   uint32_t M = 1);

// Spec 8 (plan 8b), MTP on only (`m.mtp` loaded, `mtp` allocated).
//
// build_verify: the decode list at M rows (positions pos .. pos + M - 1, ids
// cur_token[0..M)) with gdn_step's SPEC_SLOTS build - row m's GDN state into slot
// (gdn_live + m) % kSlots - then the rows' post-final-norm hidden into MtpBuffers::hh
// rows 1..M and the MTP head's K/V fill of M rows at hctl.pos (the caller sets
// hctl.pos = pos - 1, hctl.n_active = M, hctl.cur_token = cur_token): row r is the pair
// (hh[r], cur_token[r]), i.e. (h_{pos-1+r}, x[pos+r]). argmax ids land in out_token[0..M),
// logits in `logits` [M][kVocab]; argmax_stage2 advances pos by M (the caller rewinds it
// at commit). 774 + 10 launches at every M.
//
// build_draft: the MTP head at M = 1 on (hctl.cur_token[0], MtpBuffers::dh) at hctl.pos;
// its post-mtp.norm hidden back into dh, logits into MtpBuffers::logits row `i`, the
// argmax into hctl.out_token[0] AND hctl.cur_token[0], hctl.pos += 1 - so replaying the
// lists for i = 0, 1, ... chains the drafts.
CapturedStep build_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                          const MtpBuffers& mtp, uint32_t M);
CapturedStep build_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const MtpBuffers& mtp, uint32_t i);

}  // namespace runtime
