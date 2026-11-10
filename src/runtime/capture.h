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
#include "runtime/gemv_kind.h"
#include "runtime/pipeline_plan.h"
#include "runtime/verify_attn.h"

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
// binds the `_M<M>` variants (B70_MTP builds M = 2..4 at max_len 16384).
// Spec 10 (plan 10b): every capture binds decode attention v2 (attn_v2.cl; v1 was removed in
// the zero-config cleanup).

// Spec 28b: at M >= 2 every tuned int4 / int8 linear binds gemm_sm (runtime::gemv_kind, runtime/gemv_kind.h);
// `opt` (tests and probes only) forces gemv or gemm_sm at every M - build, build_verify and build_verify_taps
// take it, every other builder follows the rule.
// Spec 28c: at M >= 2 a v2 list's attention binds attn_verify_dpas where runtime/verify_attn.h's measured table
// says so; `opt` (tests and probes only) forces v2 or DPAS at every M >= 2 - build, build_verify and
// build_verify_taps take it, every other builder follows the rule.
//
// Spec 13c: `slot` - the sequence slot whose persistent state the list reads and writes, for an engine with several
// (runtime::Engine's slots). Every pointer null and kv_base 0 (the default) binds exactly what the list always bound:
// the same commands with the same arguments. build, build_verify and build_draft take it.
struct SlotBind {
  void* control = nullptr;     // the slot's Control (null: DecodeBuffers::control)
  void* gdn_state = nullptr;   // its GDN state 0, one whole-model state (null: DecodeBuffers::gdn_state)
  void* gdn_spec = nullptr;    // its kSlots - 1 speculative GDN states, slot-major (null: MtpBuffers::gdn_spec)
  void* conv_ring = nullptr;   // its conv ring, one whole-model ring (null: DecodeBuffers::conv_ring)
  uint32_t kv_base = 0;        // its first position in the KV allocations, the main cache's and the head's
  void* hctl = nullptr;        // its head Control (null: MtpBuffers::hctl)
  void* hh = nullptr;          // its hh rows, bf16 [kM + 1][hidden] (null: MtpBuffers::hh)
};
CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid = nullptr, ProfileEvents* prof = nullptr,
                   uint32_t M = 1, const CaptureOptions& opt = {}, const SlotBind& slot = {});

// The decode list's launch count (`CapturedStep::kernel_count` of `build`), which the
// walk asserts on itself: embed_gather, 12 per dense layer (the input norm pair, the
// mixer's 5, the post norm pair, gate||up + SiLU + down) or 13 per MoE layer (spec 15c:
// the router GEMV, moe_route, moe_gate_up, moe_down in place of the dense MLP's 3), and
// 5 at the boundary (the final norm pair, lm_head, the two argmax stages). Qwen3.8 774,
// Ornith 526.
size_t decode_launches(const model::ModelDesc& d);
// Spec 15e: the MTP lists' launch counts, which their walks assert on themselves too.
// verify_launches: decode_launches + the head's 10-launch KV fill, at every M (Qwen3.8
// 784, Ornith 536). draft_launches: 23 with a dense head, 24 with a MoE head
// (its FFN is the 4-launch MoE block in place of gate||up, SiLU, down) - a draft
// vocabulary swaps 3 launches for 3.
size_t verify_launches(const model::ModelDesc& d);
size_t draft_launches(const model::ModelDesc& d);

// Spec 8 (plan 8b), MTP on only (`m.mtp` loaded, `mtp` allocated).
//
// build_verify: the decode list at M rows (positions pos .. pos + M - 1, ids
// cur_token[0..M)) with gdn_step's SPEC_SLOTS build - row m's GDN state into slot
// (gdn_live + m) % kSlots - then the rows' post-final-norm hidden into MtpBuffers::hh
// rows 1..M and the MTP head's K/V fill of M rows at hctl.pos (the caller sets
// hctl.pos = pos - 1, hctl.n_active = M, hctl.cur_token = cur_token): row r is the pair
// (hh[r], cur_token[r]), i.e. (h_{pos-1+r}, x[pos+r]). argmax ids land in out_token[0..M),
// logits in `logits` [M][kVocab]; argmax_stage2 advances pos by M (the caller rewinds it
// at commit). 774 + 10 launches at every M (verify_launches; Ornith 526 + 10, spec 15e:
// its MoE blocks at M rows run moe.cl per row, routing every row independently).
// M is 1..mtp.slots: kSlots = 4 on every engine; spec 19a's probe (probe_mtp_steps, max M
// 8) builds M = 5..8 over an 8-slot MtpBuffers, binding `gdn_step_slots_M<M>_N8` and the
// B70_VERIFY_M8 binaries (src/kernels/CMakeLists.txt; Qwen3.8 at the int8 head only).
//
// build_draft: the MTP head at M = 1 on (hctl.cur_token[0], MtpBuffers::dh) at hctl.pos;
// its post-mtp.norm hidden back into dh, logits into MtpBuffers::logits row `i`, the
// argmax into hctl.out_token[0] AND hctl.cur_token[0], hctl.pos += 1 - so replaying the
// lists for i = 0, 1, ... chains the drafts.
// `prof` (spec 28a, probe_verify_m --verify8): per-launch events exactly as build's - an
// observer only; null (every engine call) appends the identical list it always did.
CapturedStep build_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                          const MtpBuffers& mtp, uint32_t M, ProfileEvents* prof = nullptr,
                          const CaptureOptions& opt = {}, const SlotBind& slot = {});
CapturedStep build_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const MtpBuffers& mtp, uint32_t i, const SlotBind& slot = {});

// --- spec 19c: a tap drafter's verify list (DFlash) ------------------------------------
//
// build_verify_taps: the decode list at M rows (positions pos .. pos + M - 1, ids cur_token[0..M)), with
// gdn_step's SPEC_SLOTS build over the drafter's `slots` GDN state slots - row m's state into slot
// (gdn_live + m) % slots, binaries gdn_step_slots_M<M>_N8 -, and the TAPS: after the input fold of layer
// i + 1 (or the final norm's fold, for the last layer) `resid` holds layer i's output, and for every i in
// `layers` each row's [hidden] bf16 goes to taps[row][t][hidden] (t: i's index in `layers`) by a device
// copy (5 x M on Qwen3.8's drafter). argmax ids land in out_token[0..M), logits in `logits` [M][kVocab],
// argmax_stage2 advances pos by M (the caller rewinds it at commit) - exactly build_verify's contract
// without the MTP head's KV fill: decode_launches(d) launches at every M. runtime/tap_drafter.h says what
// the Engine does with the taps.
struct TapVerify {
  const l0::Mem* gdn_spec = nullptr;   // slots 1..slots-1, slot-major, one gdn_state each (slot 0 = gdn_state)
  uint32_t slots = 0;                  // TapVerifyDims::kSlots
  l0::Mem* taps = nullptr;             // bf16 [DecodeBuffers::kM][layers.size()][hidden]
  std::vector<uint32_t> layers;        // the tapped layers' OUTPUTS: ascending, each < the model's layers
};
CapturedStep build_verify_taps(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                               const TapVerify& tv, uint32_t M, const CaptureOptions& opt = {});

// (spec 28c: the stage verify builders below take CaptureOptions too - PipelineEngine's deep lists force DPAS.)
// --- spec 16b: pipeline parallel decode ----------------------------------------------
//
// One stage's decode list: the walk above restricted to layers [st.first, st.last), on the
// device `ctx` / `b` / `m` belong to (`m` holds only those layers' weights - see
// runtime/pipeline_place.h - and `b` only their state, PersistentBuffers' stage
// constructor). Same kernels, same arguments, same order as the single-card list, so the
// two stages together compute exactly what it computes (spec 16 §4 P1):
//
//   stage 0  embed_gather, layers [0, s), then layer s's prep_res_fold - the cut: the fold
//            reads no weight, so this device runs it and hands off the FOLDED residual and
//            its norm sums (`resid` M rows + `norm_sumsq`, 10.9 KB on Qwen3.8) rather than
//            the residual plus down's split-K partials (90 KB) - then the hand-off out.
//   stage 1  the hand-off in, then layer s from its prep_norm_finish on, layers (s, L), the
//            head (final norm, lm_head, the two argmax stages: the token lands in THIS
//            device's control block).
//
// The hand-off (`link.mode`, spec 16 §2 decision 3):
//   copy  out: two device-to-device copies into the landing buffer (peer writes by the copy
//              path), then a barrier signalling `link.event`; in: a wait on `link.event`, then
//              two local copies from the landing buffer into `resid` / `norm_sumsq`.
//   peer  out: pp_send (pp_handoff.cl) as the last launch; in: pp_recv as the first.
// The per-kind state indices are the stage's own (GDN slice g of `b.gdn_state` is the
// model's GDN layer st.gdn_first + g), and `m.layer_small` / `m.moe` are indexed by
// layer - st.first. The labels keep the model's layer numbers ("L32 ..."), the hand-off's
// launches are "-- pp_send" / "-- pp_recv". No residual tap, no profiling events.
struct StageLink {
  PpHandoff mode = PpHandoff::Copy;
  void* land_resid = nullptr;   // the landing buffer's rows (the next device's memory)
  void* land_sumsq = nullptr;   // its norm sums; the stamp word follows them
  void* flag = nullptr;         // peer: the flag word, a page of its own
  size_t resid_cap = 0, sumsq_cap = 0;   // what the landing regions hold
  void* send_seq = nullptr;     // peer, out: pp_send's counter (this device's memory)
  void* recv_state = nullptr;   // peer, in: pp_recv's state words (this device's memory)
  uint32_t spin_limit = 0;      // peer, in: pp_recv's bound on flag loads
  ze_event_handle_t event = nullptr;   // copy: the cross-device event (l0::SyncEvent)
};
CapturedStep build_stage(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const PpStage& st, const StageLink& link);

// --- spec 16d: MTP across the split ----------------------------------------------------
//
// build_stage_verify: build_verify restricted to one stage, exactly as build_stage restricts
// build - the M rows' ids from THIS device's Control (cur_token[0..M), which the host writes
// on both devices), gdn_step's SPEC_SLOTS build on the stage's GDN layers (row m's state into
// slot (gdn_live + m) % kSlots: slot 0 is the stage's gdn_state, slots 1..3 live in
// `gdn_spec`, sized pp_gdn_spec_bytes for the stage - the binary's slot stride is the whole
// model's, the stage binds its own slice g), and the hand-off of M rows (the landing buffer
// holds kM). The last stage then runs build_verify's tail on `mtp` (its device's MtpBuffers):
// the rows' post-final-norm hidden into hh rows 1..M and the head's KV fill (its embed_gather
// reads the device's replica of the embedding, spec 16d placement). `mtp` is null for the
// other stage. Launches: pp_stage_verify_launches (+ pp_send / pp_recv under peer); the two
// stages add up to verify_launches.
//
// build_stage_draft: build_draft on the last stage's device - the head at M = 1 over its own
// buffers, the stage model's head weights, lm_head and embedding replica (no layer of the
// main model runs; the stage's buffers are only its scratch). draft_launches(d) launches.
CapturedStep build_stage_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                                const PpStage& st, const StageLink& link, uint32_t M,
                                const MtpBuffers* mtp, const l0::Mem& gdn_spec, const CaptureOptions& opt = {});
CapturedStep build_stage_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                               const PpStage& st, const MtpBuffers& mtp, uint32_t i);

// --- spec 19d: a tap drafter across the split (--spec dflash --parallel pp) -------------------------
//
// build_stage_verify_taps: build_verify_taps restricted to one stage, exactly as build_stage_verify restricts
// build_verify - the stage's layers at M rows, gdn_step_slots_M<M>_N8 over `tv.gdn_spec` (the stage's slots,
// pp_gdn_spec_bytes(d, stage gdn, 8): the binary's slot stride is the whole model's, the stage binds its slice g),
// the hand-off of M rows, and the taps of THIS stage's sites (a tap whose site - the input fold of the tapped
// layer's successor - lies in the stage) copied into `tv.taps`, the drafter's buffer on the last device: from device
// 0 that is a device-to-device copy into the other card's memory, appended before the hand-off out and so ordered
// before the cross-device event (the copy hand-off only - capture refuses peer). Launches: pp_stage_launches (the
// taps are copies); the two stages' lists add up to build_verify_taps' decode_launches.
CapturedStep build_stage_verify_taps(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                                     const PpStage& st, const StageLink& link, const TapVerify& tv, uint32_t M,
                                     const CaptureOptions& opt = {});

// --- spec 13c: rows of different sequences (spec 13a's probe build, promoted) -----------------------------------------
//
// build_rows: the decode list at M rows whose per-sequence sites read the row table (runtime/batch_rows.h) - gdn_step,
// attn_prep and decode attention v2 bound to their `_ROWS` builds (kernels/batch_kernels.h), every row-agnostic kernel
// at M rows exactly as build() binds it (spec 28b's gemm_sm at M >= 2). Row m is (position, KV base, GDN src / dst,
// slot) from `rows`; the KV caches are ONE allocation each (DecodeBuffers::kv_k / kv_v at the slots' summed length), in
// which a slot is a position range; GDN state index 0 is DecodeBuffers::gdn_state and i >= 1 is `state_pool`'s
// whole-model state i - 1; slot 0's conv ring is DecodeBuffers::conv_ring and slot s >= 1's is `ring_pool`'s ring s - 1.
// The host writes the table, and the list's own Control (`control`: n_active = M, cur_token[0..M)), before every
// replay; nothing in the list reads a Control's pos. argmax_stage2's in-list advance lands in that Control too, where
// nobody reads it: the host commits each slot (runtime::Engine::verify_rows).
//
// With `mtp` the MTP head's KV fill follows over the same rows (build_verify's tail): the rows' post-final-norm hidden
// copied to `saved`, then ONE embed_gather over `gctl` - cur_token[m] the row of `table` that is row m's input hidden:
// its slot's h_{pos-1} (the slot's hh row 0) at a run's first row, the saved hidden of row m - 1 otherwise - into
// `hin`, and the head's front over `hin` with `hctl` (n_active = M, cur_token the rows' ids) and its attn_prep at
// `head_rows` (each row's position - 1 and its slot's KV base, in the head's KV layer). So one list serves every shape
// of M rows: M = 2..8, one list each. rows_launches(d, head) launches.
struct BatchRowsBind {
  void* rows = nullptr;                 // runtime::BatchRowTable, shared
  void* control = nullptr;              // the list's Control, shared
  const void* state_pool = nullptr;     // GDN indices 1..: one whole-model state each (null: index 0 only)
  const void* ring_pool = nullptr;      // conv rings of slots 1..: one whole-model ring each (null: slot 0 only)
  const MtpBuffers* mtp = nullptr;      // the head's buffers, or null: the decode list alone
  void* head_rows = nullptr;            // runtime::BatchRowTable for the head's attn_prep, shared
  void* hctl = nullptr;                 // the head's Control, shared
  void* gctl = nullptr;                 // the gather's Control, shared
  const void* table = nullptr;          // bf16 rows of hidden the gather reads (embed_gather's `embed`)
  void* saved = nullptr;                // bf16 [M][hidden], rows of `table`: the rows' hidden
  void* hin = nullptr;                  // bf16 [M][hidden]: the head's input rows
};
CapturedStep build_rows(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const BatchRowsBind& br,
                        uint32_t M);
// decode_launches(d), or with the head verify_launches(d) + 1 (the gather).
size_t rows_launches(const model::ModelDesc& d, bool head);

// build_draft_rows: the MTP head's draft list (build_draft's walk) at M rows of DIFFERENT sequences - one draft index
// of M slots in one replay, the head's weights and lm_head read once for all of them (spec 13 §9.1 P7's second form).
// Row m is (hctl.cur_token[m], dh row m) at the head row table's row m (its draft position, its slot's KV base): the
// head's attn_prep and decode attention are their `_ROWS` builds over the head's KV layer; every other launch is
// build_draft's at M rows. The rows' post-mtp.norm hidden go back into dh rows 0..M-1, their logits into `logits`
// [M][kVocab] fp32 and their argmax into hctl.out_token[0..M) (argmax_stage2's in-list advance lands in hctl, which
// the host rewrites before every replay). draft_launches(d) launches. Dense heads only (Qwen3.8).
struct DraftRowsBind {
  void* hctl = nullptr;        // the head Control, shared: n_active = M, cur_token[0..M) the rows' input ids
  void* head_rows = nullptr;   // runtime::BatchRowTable, shared: each row's draft position, KV base, slot
  void* dh = nullptr;          // bf16 [M][hidden]: in, each row's chain hidden; out, the next
  void* logits = nullptr;      // fp32 [M][kVocab]
};
CapturedStep build_draft_rows(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                              const DraftRowsBind& dr, uint32_t M);

// --- spec 17b: tensor parallel decode (`--parallel tp`) ---------------------------------------------------------
//
// build_tp: ONE card's decode list under TP - build()'s walk at the card's shapes (`m` loaded by loader::load_tp, `b`
// sized by model::tp_card_desc), with three differences (src/kernels/tp.cl has the kernels, runtime/tp_protocol.h the
// protocol):
//   - after each row-parallel linear (out_proj, o_proj, down) a tp_send pushes this card's part into the PEER's
//     landing buffer (128 launches a token on Qwen3.8);
//   - the residual fold that consumes such a linear is tp_res_fold: it waits for the peer's part in this card's own
//     landing buffer and folds both cards' parts in one fixed order (the slices form: prep_res_fold's one-card chain);
//   - the head is the card's vocabulary half (gemv_i8w at its 124160 rows) and tp_argmax's two stages: the global id
//     and its value into this card's Control (out_token, out_value); the host merges the two cards (TpEngine).
// Launches: tp_launches(d). No profiling events; the residual tap (debug_resid) as build()'s.
struct TpLink {
  uint32_t rank = 0;
  bool folded = false;              // model::TpCrossing::Folded
  void* peer_land = nullptr;        // the peer's landing buffer (this card's sends write it)
  void* land = nullptr;             // this card's landing buffer (its folds read it)
  void* send_ctr = nullptr;         // device, tp::Layout::counter_bytes() each
  void* fold_ctr = nullptr;
  void* own = nullptr;              // folded: this card's folded row, fp32 [kM][hidden]
  void* state = nullptr;            // shared, tp::kStateWords: the first failure
  uint32_t spin_limit = 0;
  uint32_t vocab_used = 0;          // the real rows of this card's head half (its first id: rank x rows)
  const model::ModelDesc* head = nullptr;   // spec 17d: the MTP head's descriptor (the one-card one), or null
};
CapturedStep build_tp(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const TpLink& link,
                      l0::Mem* debug_resid = nullptr);
// decode_launches(d) + one tp_send per row-parallel linear (2 a layer): Qwen3.8's card 774 + 128 = 902.
size_t tp_launches(const model::ModelDesc& d);

// --- spec 17d: MTP under tensor parallel ---------------------------------------------------------------------------
//
// The MTP head is WHOLE on each card (loader::load_tp's `mtp`: one card's shapes, `link.head` its descriptor), runs
// in the card's decode scratch (which holds its rows at M <= 4 - checked at capture) and keeps its own KV layer on
// each card (MtpBuffers' TP form: the GDN slots at the card's shapes, the head's layer at its own). Both cards hold
// the same residual, so both run the same head on the same input and write the same head KV.
//
// build_tp_verify: build_verify's contract on one TP card - build_tp's walk at M rows (positions pos .. pos + M - 1,
// ids cur_token[0..M) of the card's Control; the crossings at M rows, whose counters keep every row's count level),
// gdn_step's SPEC_SLOTS build at the card's heads over mtp.gdn_spec, then tp.cl's argmax at M rows (each row's winner
// of the card's half and its value: the host merges, TpEngine), the rows' post-final-norm hidden into hh rows 1..M
// and the head's KV fill of M rows at hctl.pos - spec 8's tail at the head's shapes. tp_verify_launches(d) launches.
// `opt` as build_verify's (spec 28c's deep list: CaptureOptions::force_attn).
//
// build_tp_draft: build_draft on one TP card - the head at M = 1 on (hctl.cur_token[0], dh) at hctl.pos at its own
// shapes; its lm_head is the card's int8 vocabulary half into MtpBuffers::logits row `i` (the card's rows only) and
// tp.cl's argmax into hctl (out_token, out_value, cur_token, pos + 1). The host merges the two cards' winners into
// both hctl.cur_token[0] before the next draft list (TpEngine::draft). draft_launches(head) launches.
CapturedStep build_tp_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                             const TpLink& link, uint32_t M, const CaptureOptions& opt = {});
CapturedStep build_tp_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                            const TpLink& link, uint32_t i);
// tp_launches(d) + the head's 10-launch KV fill: Qwen3.8's card 912 at every M.
size_t tp_verify_launches(const model::ModelDesc& d);

}  // namespace runtime
