#pragma once
#include <algorithm>
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
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/batch_rows.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/kv_pool.h"
#include "runtime/memory_plan.h"
#include "runtime/prefill_backend.h"
#include "runtime/prefill_chunks.h"
#include "runtime/tap_drafter.h"

namespace runtime {

// The prefill path's per-engine state: `runtime::prefill::Context` (the L0
// immediate list) and its `KernelCache`. **Incomplete here on purpose.** Everything
// that touches it -- `Engine::prefill`, `prefill_launches`, and the deleter -- is
// compiled into `b70_prefill_host`. `b70_runtime` is linked by every decode binary
// and test, and none of those needs the prefill walk, so the member below is a
// `unique_ptr` with a FUNCTION-POINTER deleter: that
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
  //
  // `kv` (spec 12b, `--kv-cache`): the KV cache's form for the session's lifetime - bf16
  // (today's engine, bit for bit) or int8 (rotkv: int8 rows + fp16 scales, the writers
  // and readers of src/kernels/kv8.cl). The default is runtime::default_kv_cache()
  // (B70_KV_CACHE, unset = bf16), which is how the gate tests run over the int8 cache.
  // int8 needs decode attention v2 and, for prefill(), the L0 backends' flash attention;
  // anything else throws (at capture, or at the first prefill) rather than reading int8
  // rows as bf16.
  //
  // `drafter` (spec 19c, `--spec dflash`): a proposer fed by the target's taps (runtime/tap_drafter.h), borrowed - it
  // must outlive the Engine. The Engine then allocates 8 GDN state slots (TapVerifyDims) and captures the verify
  // lists at M = 1 .. drafter->max_k() + 1 with the taps (build_verify_taps); ingest(), generate() and commit() keep
  // the drafter's ring filled (see "spec 19c" below). Not with the MTP head (one proposer per engine): throws.
  //
  // `slot_lens` (spec 13c) and `pool_pages` (spec 31, the KV page pool): empty and 0 (the default) - or one length,
  // max_len - is one sequence of max_len: today's engine, allocation for allocation and list for list. Several
  // sequences share a KV page pool: `pool_pages` pages of kKvPage positions (max_len = pool_pages x kKvPage) that they
  // take as they grow, and `slot_lens` names every sequence's cap, all equal (2..SlotDims::kMaxSeqs of them) - see
  // "spec 13c" and "spec 31" below. Several sequences without a pool throw.
  Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len,
         bool debug_resid = false, KvCache kv = default_kv_cache(), TapDrafter* drafter = nullptr,
         std::vector<uint32_t> slot_lens = {}, uint32_t pool_pages = 0);

  // Zeroes exactly the persistent group - control, gdn_state, conv_ring, kv_k,
  // kv_v. Scratch is deliberately NOT zeroed: no step may read scratch it has
  // not first written, and replay_determinism_test's run C is the standing
  // proof of that. Zeroing scratch here would mask a regression of it.
  // Spec 13c: with several sequence slots, the SELECTED slot's persistent state only - its Control and head
  // Control, its GDN states, conv ring, hh rows and KV range (main and head); no other slot's byte moves.
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
  // grades. Default chunk = the session's chunk cap (prefill_chunk_cap(): 2048 unless set,
  // ruling A7/A13; plan 5d: 4096 by set_prefill_chunk_cap).
  //
  // Throws if `ids` is empty, if `chunk > prefill_chunk_cap()`, if any id is
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
  // appended since the first prefill(). 0 before it.
  size_t prefill_launches() const;

  // Spec 2.1: the backend prefill() runs. Unset means the default, l0
  // (runtime::prefill::default_prefill_backend()). It may change
  // between prefill() calls -- persistent state is shared, scratch is per backend.
  void set_prefill_backend(PrefillBackend b) { pf_backend_ = b; }
  PrefillBackend prefill_backend() const;          // defined in engine_prefill.cc

  // Plan 5d: the session's prefill chunk cap - the most rows a chunk has, and so the rows of
  // every per-chunk buffer (PrefillScratch, Int8State's xq / xs, MTP's prefill hidden rows, a
  // drafter's tap staging). 2048 (PrefillScratchDims::kCDefault) or 4096 (kCMax); the CLIs
  // set the one their memory plan counted (runtime::prefill_chunk_cap_that_fits). Unset means
  // runtime::default_prefill_chunk_cap() (2048; a test may move it). Throws
  // std::invalid_argument for a cap check_prefill_chunk_cap refuses (4096 on a MoE model, by
  // name), and std::logic_error once the prefill scratch exists at another cap (the buffers
  // are sized once; set it before prepare_prefill()). Defined in engine_prefill.cc.
  void set_prefill_chunk_cap(uint32_t cap);
  uint32_t prefill_chunk_cap() const;
  // Open-work 38: hold a MoE model's mixer linears unpacked in bf16 for the session (prefill::make_dense_slabs,
  // runtime::dense_slab_bytes), made by the next prepare_prefill(); the CLIs set it when their memory plan has room
  // (PrefillPath::dense_bf16). Off by default: every chunk unpacks them. Bitwise either way; no effect on a dense
  // model. Once made they stay for the engine's life.
  void set_prefill_dense_bf16(bool on) { pf_dense_bf16_ = on; }
  // Open-work 18: the positions at a hooked prefill's end cut at kBlock under a 4096-row chunk
  // (runtime::prefill_chunk_rows' `tail`): Qwen3.8 kPrefillTail, a MoE model 0.
  uint32_t prefill_tail() const { return runtime::prefill_tail(model_.desc->is_moe()); }

  // Spec 6 (plan 6b): one line summing the device memory this engine holds --
  //   memory: model <GB>, kv <GB>, decode state <GB>, prefill scratch <GB>, int8 <GB>,
  //   total <GB> of <device GB>
  // model = the loader report's total (RoPE included); kv = kv_k + kv_v; decode state =
  // the rest of PersistentBuffers plus DecodeScratch; prefill scratch = PrefillScratch's
  // bytes() + lazy_bytes() as allocated NOW; int8 = Int8State::bytes(). With a draft
  // vocabulary (spec 8 §11) a `draft vocab <GB>,` term (LoadReport::draft_vocab_bytes)
  // precedes the total; without one the line is unchanged. The CLIs print it
  // after prepare_prefill(), before the first prefill. Defined in engine_prefill.cc.
  std::string memory_line() const;
  // The five figures memory_line() prints, in bytes (spec 6 §10): what
  // runtime::plan() predicts for this model, max_len and path, and what
  // tests/runtime/memory_plan_box_test.cc holds it to. Defined in engine_prefill.cc.
  MemoryComponents memory_use() const;

  // Spec 7 §3.4: snapshots of the session for prefix caching. The GDN state and the
  // conv ring are valid only at the exact position they were computed to, so a restore
  // point needs both; the KV cache of positions [0, pos) is copied by range. Every call
  // is blocking and runs on `imm_` (a synchronous immediate list) outside the captured
  // decode list, which reads `pos` and the persistent buffers these calls write in
  // place. Host pointers are l0::MemKind::Host allocations (device-visible). The caller
  // must not have a prefill or a replay in flight -- none of the public calls leaves one.
  // PERF-89: the exception is a mid-prompt block hook (below): there save_kv's device-to-host
  // copies, and with the MTP head save_state's (from a device copy of the state in the
  // sequence's verify slots, idle during a prefill), go on the copy engine and run while the
  // next chunk computes; the host bytes are complete once prefill() returns or throws.
  static constexpr uint32_t kBlock = 2048;
  // Spec 8 §3.5 (plan 8b Review Focus 4): with MTP on, the state also carries the
  // head's input hidden h_{pos-1} (MtpBuffers::hh row 0, 10240 B) and the KV the head's
  // own layer (a 17th KV layer); both sizes grow accordingly.
  // Spec 19d (spec 19 decision 5): with a drafter, its whole ring as well (TapDrafter::ring_bytes, 41.94 MB for
  // Qwen3.8's) - a restore then resumes the drafter where the snapshot found it, not cold.
  size_t state_bytes() const;                 // gdn_state + conv_ring (166.72 MB) [+ hh0] [+ the drafter's ring]
  // n_pos * (fa_layers [+1]) * KvLayout::pos_bytes(), K and V: 4 x 256 x 2 B per layer and
  // position at bf16; 4 x (256 + 2) B at int8 (spec 12b: the rows and their scales).
  size_t kv_bytes(uint32_t n_pos) const;
  // Layout: gdn_state (the LIVE slot) then conv_ring [then hh row 0 | then the drafter's ring]. conv_ring is a
  // ring indexed by pos % kConvRing; the whole ring is copied, so a restore at any
  // pos % 16 is exact.
  void save_state(void* host) const;
  // Writes both (and the hidden row / the drafter's ring), then control.pos = pos, control.n_active = 0.
  // cur_token is NOT restored: prefill at least one id after a restore (spec 7 §3.3 step 4).
  void load_state(const void* host, uint32_t pos)
    pre(pos <= max_len());
  // Positions [begin, end) of kv_k and kv_v. Host layout: K [16][end-begin][4][256]
  // bf16, then V the same; with MTP on, [17] layers each, the head's cache last. begin == end copies nothing (host
  // may be null). begin <= end <= max_len - for save_kv with a KV page pool, <= the positions its pages hold.
  // Spec 12b, int8: K is [layers][end-begin][4][256] int8 rows, then [layers][end-begin][4]
  // fp16 scales (the head's layer last in both), then V the same - kv_bytes(end - begin)
  // bytes. A store holds one form only: the bytes are not convertible, so the prefix cache
  // keys its entries by the engine's form as well (server/prefix_cache.h).
  void save_kv(uint32_t begin, uint32_t end, void* host) const
    pre(begin <= end && end <= (pool_ ? pages_held(cur_) * kKvPage : max_len()));
  void load_kv(uint32_t begin, uint32_t end, const void* host)
    pre(begin <= end && end <= max_len());

  // Spec 7 §3.2: called on the host after every prefill chunk whose end position is a
  // multiple of kBlock, with the device idle and the persistent state exactly at
  // `end_pos` (control.pos == end_pos, n_active == 0; cur_token is not yet the argmax),
  // and after the LAST chunk of every prefill (the prompt end), whatever its end, once
  // the first generated id is in cur_token. `is_block_end` tells whether end_pos is a
  // multiple of kBlock. A prompt that ends on a block end gets ONE call. With a hook set,
  // every chunk ends at a block end or at the prompt end: chunk rows are
  // min(chunk, S - pos % S, remaining), S = chunk rounded up to whole kBlocks
  // (runtime/prefill_chunks.h) - kBlock itself up to 2048 rows, 4096 at a 4096-row chunk
  // (plan 5d, ruling A: the hook then fires at every second block end), and kBlock again over
  // the last prefill_tail() positions of the call (open-work 18). S follows the call's
  // `chunk`, not the cap: prefill(ids, 1024) at cap 4096 still hooks every 2048. Without one (an empty function, the
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
  // back in debug buffer order [layer][m][hidden] bf16. What tap[L] *means* is
  // documented in exactly one place - runtime/capture.h's `debug_resid`
  // paragraph - and is not restated here.
  std::vector<uint16_t> read_debug_resid();  // throws unless debug_resid
  bool debug_resid() const { return own_.tap != nullptr; }

  // --- spec 8: MTP speculative decoding (plan 8b) --------------------------------
  //
  // On iff the model was loaded with its MTP head (`loader::load(..., mtp = true)`).
  // Off, none of this allocates or captures anything and every call below throws.
  // On, the engine also captures three draft lists and a verify list per M = 1..4.
  //
  // One iteration at pos = n, pending id x_n in cur_token[0]:
  //   draft(k)        the head drafts d_1..d_k (greedy, on the card) at positions
  //                   n-1 .. n+k-2 from (h_{n-1}, x_n) and its own chained hidden.
  //                   The ids go to cur_token[1..k] (the verify inputs) and draft_ids();
  //                   draft i's logits (q_i) are row i of mtp_logits_device().
  //                   A host sampler may overwrite cur_token[1..k] before verify().
  //   verify(k)       the main model at M = k + 1 rows (x_n, d_1..d_k at n..n+k): row r's
  //                   argmax in verify_ids()[r], logits in verify_logits_device() [k+1][V],
  //                   the GDN state after every row kept (Control index, no copy), and the
  //                   head's KV rewritten at n-1..n+k-1 from the main hidden.
  //   commit(j, t)    j accepted drafts (0 <= j <= k): pos = n + j + 1, row j's GDN state
  //                   and hidden become live, t is the pending id. Rows past j are stale
  //                   and never read (the causal bound; the conv ring's positions).
  // verify(0) + commit(0, verify_ids()[0]) is one plain token; with MTP on, ingest() and
  // generate() run exactly that (at pos 0, the plain list), so the head's KV stays
  // filled whatever mix of plain and speculative steps the caller runs.
  //
  // Positions: verify(k) needs 1 <= pos and pos + k + 1 <= max_len; max_verify_k() is the
  // largest k allowed now (0 in the last positions: the M = 1 fallback). draft(k) needs
  // the same k.
  static constexpr uint32_t kMaxDraft = MtpBuffers::kMaxK;   // 3
  bool mtp() const { return mtp_ != nullptr; }
  // `pick` (plan 8c, sampled drafting): called on the host after draft list i with the
  // device idle and q_i in mtp_logits_device() row i; its return value replaces the
  // on-card argmax as d_{i+1} (the next draft's input, cur_token[1 + i], draft_ids()).
  // `more` (open-work 13, the serial server's per-step K): called after draft list i with the device idle and the
  // head's probability for draft i (its softmax at the drafted id, draft_prob.cl); false stops drafting there, so
  // draft_ids() may hold fewer than k. Needs draft_probs().
  void draft(uint32_t k, const std::function<uint32_t(uint32_t i)>& pick = {},
             const std::function<bool(uint32_t i, float p)>& more = {});
  // Open-work 13: the selected slot has the draft lists with the probability (one-slot engines with the head).
  bool draft_probs() const;
  const std::vector<uint32_t>& draft_ids() const { return seq().draft_ids; }
  const float* mtp_logits_device() const;      // [kMaxDraft][kVocab] fp32, device memory
  // Spec 8 §11: |V'| when the model was loaded with a draft vocabulary (the draft lists
  // then run the compact head and draft only V' ids; mtp_logits_device() rows hold -inf
  // outside V'), 0 when the full head drafts. The verify lists are the same either way.
  uint32_t draft_vocab() const { return model_.draft_vocab ? model_.draft_vocab->size() : 0; }
  // A tap drafter may supply its list: pending projection, draft and verify run in one submit.
  void verify(uint32_t k, l0::CmdList* draft = nullptr);
  // PERF-123: on, the next verifies at 2..8 rows take the main head as the 2-bit candidate pass + the exact rescore
  // where the lists carry it (Qwen3.8's int8 head with loader::LoadedModel::cand_head): verify_ids() is the same argmax,
  // but verify_logits_device() then holds the 2-bit scores - a request that reads the logits (sampled) keeps it off.
  void greedy_head(bool on) { greedy_head_ = on; }
  const uint32_t* verify_ids() const {
    const uint32_t k = seq().verify_k;
    return k == kWideK ? control_->wide_out : k == kDeepK32 || k == kDeepK64 ? control_->deep_out : control_->out_token;
  }
  // Open-work 28: the one-card wide verify list - prompt lookup's long runs, checked 16 rows at once. Built with the
  // MTP head on one sequence slot (the deferred GDN build), the bf16 KV cache and the int8 head; its drafts are an
  // external proposer's (no draft list goes this deep). The caller writes the 15 drafts into control()->wide_cur[1..15]
  // and calls verify(kWideK) (row 0 is cur_token[0]); verify_ids() then reads Control::wide_out, and commit(j, t)
  // takes j up to 15 as after any verify. max_wide_k(): kWideK where the list is built and the rows fit, else 0.
  static constexpr uint32_t kWideK = DecodeScratchDims::kWideM - 1;   // 15
  // Spec 32: the deep lists beside it - 32 and 64 rows (31 / 63 drafts in control()->deep_cur[1..], ids in
  // Control::deep_out), built where the 16-row list is. max_wide_k() is then the deepest k (63, 31, 15) whose list
  // is built, whose rows fit before max_len and that B70_SPEC_LOOKUP_ROWS (a test handle: 16, 32 or 64) allows - the
  // 64-row list up to kDeep64MaxKeys keys (runtime/verify_attn.h, spec 32e's per-depth cap).
  static constexpr uint32_t kDeepK32 = 31, kDeepK64 = 63;
  uint32_t max_wide_k() const;
  bool wide_built() const { return seq().wide != nullptr; }
  bool deep_built(uint32_t k) const { return (k == kDeepK32 ? seq().deep32 : seq().deep64) != nullptr; }
  // The list verify(k) replays for k = kWideK / kDeepK32 / kDeepK64 (probes read its launch labels).
  const CapturedStep* wide_step(uint32_t k) const {
    return k == kWideK ? seq().wide.get() : k == kDeepK32 ? seq().deep32.get() : seq().deep64.get();
  }
  // The deepest k max_wide_k() can return at any position (0 without the 16-row list).
  uint32_t widest_k() const {
    if (!wide_built()) return 0;
    for (const uint32_t k : {kDeepK64, kDeepK32})
      if (deep_built(k) && k + 1 <= deep_rows_) return k;
    return kWideK;
  }
  const float* verify_logits_device() const;   // [k + 1][kVocab] fp32, device memory
  void commit(uint32_t j, uint32_t next_token, bool defer_projection = false)
    pre(next_token < model::Qwen35::kVocab);
  void flush_projection() const;
  uint64_t queue_submissions() const { return queue_.submissions(); }
  // verify-cost-m8, tests and probes: whether the verify lists commit by replay (gdn_step's DEFERRED build), whether
  // committed rows are pending (a state read flushes them: one more submission), and the committed GDN state.
  bool gdn_deferred() const { return seq().pend != nullptr; }
  bool gdn_rows_pending() const { return seq().pend && (control_->gdn_pend & 0xFFu) != 0; }
  const void* gdn_state_now() const { return live_gdn_slot(); }
  uint32_t max_verify_k() const;
  // Tests and the record: the MTP buffers and lists (null / throw when MTP is off).
  MtpBuffers* mtp_buffers() { return mtp_.get(); }
  // The last prefill chunk's main hidden rows: bf16 [chunk_cap + 1][hidden], row 1 + r = position
  // (last chunk base + r) post-final-norm (step_mtp_kv). Null before an MTP prefill.
  const l0::Mem* mtp_prefill_hidden() const { return pf_own_.mtp_hid.get(); }
  const CapturedStep& verify_step(uint32_t M) const;
  const CapturedStep& draft_step(uint32_t i) const;
  // Spec 28c, tests and probes only (no engine flag, no environment variable): capture the verify lists again under
  // `opt` (CaptureOptions::force_attn: the verify attention v2 or DPAS at every M >= 2 instead of the measured
  // rule) - the MTP head's or the drafter's, whichever this Engine has. Session state is untouched.
  void recapture_verify(const CaptureOptions& opt);
  // Spec 28c: the verify lists captured twice (by depth) and their capture time - printed at engine start.
  uint32_t verify_deep_lists() const;
  double verify_deep_capture_ms() const { return verify_deep_ms_; }
  // Spec 32e: the 32- and 64-row lookup verify lists' capture time (their binaries loaded and the lists recorded).
  const double* deep_capture_ms() const { return deep_capture_ms_; }
  // The list verify(k) submits at row 0's key count `keys` (pos + 1): tests and probes.
  const CapturedStep& verify_step_at(uint32_t M, uint32_t keys) const;

  // --- spec 19c: a tap drafter (DFlash) -------------------------------------------------
  //
  // On iff the Engine was built with a TapDrafter. Then verify(k) / commit(j, t) / max_verify_k() are spec 8's
  // contract with three differences:
  //   - k goes up to max_draft() = drafter->max_k() (7 for DFlash 2's block of 8) and the verify list is
  //     build_verify_taps' at M = k + 1 over TapVerifyDims::kSlots (8) GDN slots: row m's state into slot
  //     (live + m) % 8, so commit(j, t) makes row j's state live by setting Control::gdn_live = (live + j) % 8 -
  //     the GDN state is then exactly the state after the last accepted row, rows past j are stale slots that no
  //     list reads before a later verify overwrites them (and the conv ring's positions past n + j are rewritten
  //     before any row reads them - gdn_step.cl's slot disjointness at ring depth 16 >= M + 3);
  //   - the drafts are whatever the caller (the drafter's draft list) wrote into cur_token[1..k]; there is no
  //     draft() / draft_ids() (those are the MTP head's);
  //   - commit(j, t) then replays the drafter's projection of verify rows 0..j - the committed positions
  //     n .. n + j - into its ring (TapDrafter::projection). With defer_projection it runs at the front of the next
  //     batched verify, or flushes before snapshots, plain steps and prefill. Reset/restore discard it. Otherwise
  //     commit projects before returning. Rejected rows are never projected; t has no tap (the next row 0).
  // verify(0) + commit(0, verify_ids()[0]) is one plain token and, with a drafter, is what ingest() and generate()
  // run (pos 0 included: nothing here needs h_{-1}). prefill() fills the ring from the prompt's last window()
  // positions (spec 19c Task 3).
  bool has_drafter() const { return drafter_ != nullptr; }
  TapDrafter* tap_drafter() const { return drafter_; }
  // The deepest k verify() takes: kMaxDraft with the MTP head, drafter->max_k() with a drafter, 0 otherwise.
  // The selected slot's (plan 40a Task 2: every slot of several keeps its rows pending, so the drafter's whole block).
  uint32_t max_draft() const;
  // The GDN slot count of the verify lists: MtpBuffers::kSlots (MTP) or TapVerifyDims::kSlots (drafter) - the
  // selected slot's (plan 40a Task 2: 1 for a slot of several beside a drafter - its rows pending).
  uint32_t gdn_slots() const { return seq().states; }
  // Spec 19c: the drafter's GDN state slots 1..7 (slot-major, one gdn_state each) of a one-slot engine; null without a
  // drafter or with several slots (plan 40a Task 2: their rows pending, each slot one state in SlotBuffers).
  const l0::Mem* drafter_slots() const { return own_.df_slots.get(); }
  // Replays a list that is not the Engine's own (the drafter's draft and projection lists) on this engine's queue
  // and waits for it: nothing of the Engine's is in flight before or after. The list must only read Control and
  // the buffers it was captured against.
  void replay_aux(l0::CmdList& list);

  // --- spec 13c: sequence slots - several requests on one card ------------------------------------------------
  //
  // `slot_lens` (the constructor's): the engine's sequence slots (one: today's engine). Slot q is up to slot_len(q)
  // positions of the KV caches (the main one and the MTP head's), with its own GDN states (gdn_slots() of them: the
  // live one and, with MTP, three speculative), conv ring, Control, head Control and hh rows, and its own copy of every
  // captured list (plain, verify and draft) bound to them - so a slot alone runs exactly today's arithmetic over its
  // own state: bitwise what a one-slot engine computes for that session.
  //
  // Plan 19v3b Task 0: with several slots those positions are KV pages of the pool (runtime/buffer_sizes.h, kKvPage;
  // "spec 31" below): slot q's page p is the cache's page kv_page_table()[slot_base(q) / kKvPage + p] of every FA
  // layer, main and head, and every slot's lists - slot 0's too - and the row lists bind the paged binaries over that
  // table (capture.h, SlotBind). Snapshots (save_kv / load_kv) keep their format whatever the table.
  //
  // Every single-sequence call above and below (reset, ingest, prefill, generate, draft, verify, commit,
  // max_verify_k, pos, max_len, the snapshots, control) acts on the SELECTED slot - select(q); 0 at construction.
  //
  // verify_rows(runs) is ONE decode step over several slots' rows (spec 13 §9's row table; runtime::build_rows, the
  // small-batch kernels): each run is a slot's pending id and its k drafts (cur_token[0..k] of that slot's Control -
  // draft(k) with the slot selected writes them, as for verify) at the slot's pos .. pos + k; at least two rows, at
  // most kMaxRows, each slot at most once, at pos >= 1 with MTP. rows_ids() / rows_logits_device() then hold every
  // row's argmax and logits in run order, and per run select(slot) + commit(j, t) commits that slot exactly as after
  // its own verify(k) (row j's GDN state live, pos + j + 1, row j's hidden its h_{pos-1}) - or not yet: as after
  // verify(), the run stays committable until the slot's next verify. Rows of one step never read each other's slot
  // (gdn_step's state indices, the KV bases), and a rejected row writes only its own slot's speculative state, ring
  // positions and KV rows past the commit, which nothing reads (spec 13 §9: B3).
  static constexpr uint32_t kMaxRows = DecodeScratchDims::kM;   // 8
  // Plan 40a Task 3: with the MTP head beside several Qwen3.8 sequences (the deferred delta-net rows, the int8 head)
  // the row lists also exist at 16 and 32 rows - a step of 9..16 or 17..32 rows runs the wider list, the rows past its
  // n_active the table's padding. max_rows() is a step's ceiling: kMaxWideRows there, kMaxRows elsewhere; every run at
  // most kMaxRows rows (a sequence with pending rows verifies up to kMaxRows - 1 drafts in a row step).
  static constexpr uint32_t kMaxWideRows = DecodeScratchDims::kWideRowsM;   // 32
  uint32_t max_rows() const { return rows_wide_[1] ? kMaxWideRows : kMaxRows; }
  // PERF-120 (plan 40a Task 7): the 16- and 32-row lists' attention at 4 key blocks a run (BatchRowsBind::attn_blocks)
  // - fewer, longer key walks: the 32-row step 67.9 / 74.4 / 86.4 ms at 32 blocks, 59.6 / 64.7 / 76.4 at 8, 55.8 /
  // 60.8 / 72.4 at 4 (depth 4k / 12k / 30k, 4 sequences, int8 KV, interleaved; 64 and 128 blocks slower still).
  static constexpr uint32_t kRowsAttnBlocks = 4;
  // PERF-120: a step of 8 rows whose deepest row is at kRowsWideDepth or past it runs the 16-row list (its DPAS
  // attention; rows 8.. padding) - 4 sequences x K 1, int8 KV: 43.9 -> 44.7 ms at 4k (kept on its own list), 52.2 ->
  // 49.9 at 12k, 68.8 -> 61.4 at 30k; a 4-row step loses on the 16-row list at every depth (37.7 -> 43.3 at 4k, 46.3 ->
  // 60.5 at 30k); 5..7 rows not measured (their own lists).
  static constexpr uint32_t kRowsWideDepth = 8192;
  // Tests and probes: every step of m..8 rows on the 16-row list (0: the rule above).
  void probe_rows_wide_from(uint32_t m) { rows_wide_from_ = m; }
  // PERF-120, tests and probes: the 16- and 32-row lists built again with the MoE union read or the per-row MoE block
  // (BatchRowsBind::moe_union; the engine's own: union on a MoE model). Only where the wide lists exist.
  void probe_rows_moe_union(bool on);
  uint32_t slots() const { return static_cast<uint32_t>(seqs_.size()); }
  void select(uint32_t slot)
    pre(slot < slots());
  uint32_t selected() const { return cur_; }
  uint32_t slot_len(uint32_t slot) const { return seqs_.at(slot).len; }
  uint32_t slot_base(uint32_t slot) const { return seqs_.at(slot).base; }
  // Plan 19v3b Task 0 - the KV page table's host copy (empty with one slot).
  const std::vector<uint32_t>& kv_page_table() const { return pages_; }
  // Probes (probe_pool_speed --profile-rows): what the row lists bind, to build them again with events.
  const BatchRowsBind& rows_bind() const { return rows_bind_; }
  // The cache row (every FA layer's, main and head) of slot q's position p: below the positions its pages hold with a
  // KV page pool, max_len with one sequence.
  uint32_t kv_row(uint32_t slot, uint32_t p) const
    pre(slot < slots())
    pre(p < (pool_ ? pages_held(slot) * kKvPage : slot_len(slot)));
  // --- spec 31: the KV page pool ----------------------------------------------------------------------------------
  //
  // With several sequences (the constructor's `pool_pages`) every sequence is alike: slot_len = the per-request cap, slot_base its
  // region of the page table (kv_pages(cap) entries a sequence), and its KV pages are taken from the pool as it grows
  // (runtime::KvPool on the host, each new table entry written to the card between submits). Every call that writes
  // KV takes the pages it writes first - prefill (the call's positions), a plain step, draft(k), verify(k),
  // verify_rows and draft_rows (up to pos + k + 1), load_kv - and throws, with nothing written, when the pool cannot
  // cover them; max_verify_k() counts the pages the sequence could still take. reset() gives the selected sequence's
  // pages back and zeroes only their rows below its written bound (PERF-90's bound, kept per sequence). The kernels
  // are the paged builds (plan 19v3b Task 0). A table entry no page backs reads page 0 and
  // is never read or written (every read and write is below the positions the sequence holds).
  uint32_t kv_pool_pages() const { return pool_ ? pool_->pages_total() : 0; }
  uint32_t pages_free() const { return pool_ ? pool_->pages_free() : 0; }
  uint32_t pages_held(uint32_t slot) const { return pool_ ? uint32_t(pool_->pages_of(slot).size()) : 0; }
  // The physical pages slot `slot` holds, its positions' order (empty without a pool).
  std::vector<uint32_t> pool_pages_of(uint32_t slot) const { return pool_ ? pool_->pages_of(slot) : std::vector<uint32_t>{}; }
  // The selected sequence's pages up to position `upto` (exclusive), taken now: false, and nothing changed, when the
  // free pages do not cover them. A pool, and at most the per-request cap.
  bool take_pages(uint32_t upto)
    pre(pool_ != nullptr) { return take_pages_of(cur_, upto, kv_write_from(cur_)); }
  // Spec 31 §9 (shared prefix pages): slot's next page becomes `from_slot`'s page `index` - its positions [index
  // kKvPage, (index + 1) kKvPage), read-only for both until one writes there (then that one gets a private copy, see
  // take_pages). The caller vouches that both hold the same KV there (the same ids, the page whole). False - nothing
  // changed - unless `slot` holds exactly `index` pages and `from_slot` more.
  bool share_page(uint32_t slot, uint32_t index, uint32_t from_slot);
  struct Run {
    uint32_t slot = 0;
    uint32_t k = 0;   // drafts in the slot's cur_token[1..k]
  };
  void verify_rows(const std::vector<Run>& runs)
    pre(slots() >= 2)
    pre(runs.size() <= max_rows());
  // The MTP drafts of several slots at once (runtime::build_draft_rows: the head and lm_head read once a draft index
  // for all of them): runs[r] = (slot, k), 2 <= runs <= kMaxDraftRows, every k in 1..max_verify_k() of its slot and
  // every slot room for max(k). Each run drafts max(k) ids and keeps its first k: they land in its cur_token[1..k] and
  // its draft_ids(), as draft(k) with the slot selected leaves them. `pick(r, i, argmax)` (sampled drafting): called
  // after draft index i, per run, with the device idle and run r's draft logits in rows_logits_device() row r; its
  // return replaces the row's argmax as d_{i+1} (draft()'s `pick`).
  static constexpr uint32_t kMaxDraftRows = 4;
  void draft_rows(const std::vector<Run>& runs,
                  const std::function<uint32_t(size_t r, uint32_t i, uint32_t argmax)>& pick = {});
  const uint32_t* rows_ids() const            // [rows] the argmax of every row, run order
    pre(slots() >= 2);
  const float* rows_logits_device() const;     // [rows][kVocab] fp32, device memory (verify_logits_device()'s)
  // The selected slot's Control (shared memory): the pending id cur_token[0], the drafts cur_token[1..k].
  Control* control() const { return control_; }
  // Tests and probes: slot q's device state - its GDN states (state0, then gdn_slots() - 1 at spec), conv ring, hh
  // rows (MTP; [kM + 1][hidden]) - its Control and its KV range.
  struct SlotView {
    const uint8_t* state0 = nullptr;
    const uint8_t* spec = nullptr;
    uint32_t states = 1;
    const uint8_t* ring = nullptr;
    const uint8_t* hh = nullptr;
    Control* control = nullptr;
    uint32_t base = 0, len = 0;
  };
  SlotView slot_view(uint32_t q) const {
    const Seq& s = seqs_.at(q);
    return {s.state0, s.spec, s.states, s.ring, s.hh, s.control, s.base, s.len};
  }

  l0::Context& context() const { return ctx_; }
  const loader::LoadedModel& model() const { return model_; }
  KvCache kv_cache() const { return persist_.kv_lay.form; }   // spec 12b
  DecodeBuffers& buffers() { return buffers_; }
  const CapturedStep& step() const { return plain(); }
  // Position of the next token, i.e. how many tokens this session has consumed.
  uint32_t pos() const { return control_->pos; }
  // Spec 13c: the selected slot's positions (max_len with one slot).
  uint32_t max_len() const { return seqs_[cur_].len; }

 private:
  // One replay: the precondition, the submit, the wait. Every attention
  // kernel's bound assumes `pos + n_active <= max_len`, and the caller of the
  // list is what guarantees it (Task 5 ruling) - the Engine is that caller, so
  // the check lives here permanently and throws rather than replaying past the
  // KV cache and the RoPE table.
  void replay();
  // Spec 8: one plain token with MTP on (verify(0) + commit(0), or the plain list at
  // pos 0 plus the hidden bookkeeping) - what ingest() and generate() run then. Spec 19c:
  // with a drafter, verify(0) + commit(0) at every pos.
  void mtp_step1();
  void require_mtp(const char* what) const;
  void require_spec(const char* what) const;   // spec 19c: MTP or a drafter
  // Spec 8: before anything that reads gdn_state directly (prefill, save_state's
  // callers see the live slot): copy the live slot into slot 0 and make 0 live.
  // Spec 19c: from the drafter's slots as well.
  void mtp_normalise_live();
  // The live GDN slot's device address (slot 0 = gdn_state; MTP's or the drafter's slots) - the selected sequence's.
  // verify-cost-m8: with the pending rows, flushes them first (the state is then gdn_state, slot 0).
  const void* live_gdn_slot() const;
  // verify-cost-m8: the committed rows the next verify would replay (Control::gdn_pend), into gdn_state now - the
  // flush list at n_active 0. Every reader of the GDN state calls it first; nothing to do without pending rows.
  void flush_gdn() const { flush_gdn(seq()); }
  // PERF-89: a mid-prompt hook's snapshot copies, appended to the copy engine's list (made on first use), and the
  // wait for every one appended so far.
  void snap_copy(void* host, const void* dev, size_t bytes) const;
  void snap_wait() const;
  // Spec 19c (defined in engine_prefill.cc): the prefill tail of the drafter's ring - the n staged tap rows
  // (pf_own_.df_taps rows 0..n-1) at positions pos .. pos + n - 1, projected at prefill width (TapDrafter::project_wide)
  // or, where the drafter has no such path, 8 rows a replay.
  void drafter_project_staged(uint32_t pos, uint32_t n);

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
  std::optional<uint32_t> pf_chunk_cap_;       // plan 5d; unset = default_prefill_chunk_cap()
  bool pf_dense_bf16_ = false;                 // open-work 38
  BlockHook block_hook_;
  // Spec 39: Engine's own allocations (EngineSizes: the debug tap, the drafter's slots, the deferred GDN rows, the deep
  // chain scratch), each null until made.
  EngineBuffers own_;
  // Spec 8: null unless the model carries the MTP head. Declared before step_ so it
  // exists when the lists are captured (constructor body).
  std::unique_ptr<MtpBuffers> mtp_;
  // Spec 13c: the further sequence slots' state (null with one slot).
  std::unique_ptr<SlotBuffers> slot_bufs_;
  CapturedStep step_;   // slot 0's plain list (the other slots' are Seq::plain)

  // Spec 13c: one sequence slot - its region of the KV page table, the device addresses of its state, its Control blocks
  // and every list captured for it (slot 0's: the buffers' own state, today's lists; Seq::bind all null).
  static constexpr uint32_t kNoVerify = 0xFFFFFFFFu;
  struct Seq {
    uint32_t base = 0, len = 0;
    SlotBind bind;                  // what its lists were captured with
    Control* control = nullptr;
    Control* hctl = nullptr;        // MTP
    uint8_t* state0 = nullptr;      // GDN state 0, one whole-model state
    uint8_t* spec = nullptr;        // states 1..: MTP's (kSlots - 1) or the drafter's (7 slot 0, 3 a further slot)
    uint32_t states = 1;            // its GDN states (gdn_slots())
    uint32_t first = 0;             // the GDN index of its state 0 (the row lists' pool numbering, gdn_index)
    TapVerify tv;                   // spec 19c / plan 19v3b: what its verify lists bind beside a drafter
    // A deferred commit's projection of its drafter session (spec 19c; plan 19v3b: one a slot).
    mutable l0::CmdList* pending = nullptr;
    // Plan 19v3b: its last verify was a verify_rows step's run at rows row0 .. (its taps in the rows' tap buffer).
    bool from_rows = false;
    uint32_t rows_row0 = 0;
    uint8_t* ring = nullptr;
    uint8_t* hh = nullptr;          // MTP: [kM + 1] rows ([kWideM + 1] on one slot), row 0 = h_{pos-1}
    std::vector<CapturedStep> plain;   // the plain list of each slot of several (one slot's is step_)
    std::vector<CapturedStep> draft_steps;    // [kMaxDraft], MTP on only
    std::vector<CapturedStep> draft_prob_steps;   // open-work 13: the same ending in draft_prob (plan 31: every slot)
    std::vector<CapturedStep> verify_steps;   // [M - 1] for M = 1..kSlots
    // Spec 28c (§3, decided 2026-10-31: two lists per M are acceptable): where the measured
    // verify-attention table switches an M to DPAS at a depth (runtime/verify_attn.h, a finite dpas_from > 1), the
    // list at M is captured twice - verify_steps by the rule at row 0's key count 1 (v2) and verify_deep with DPAS
    // bound - and verify() submits the deep one when row 0's keys (pos + 1) reach verify_deep_from. Null / never
    // where one list serves every depth.
    std::vector<std::unique_ptr<CapturedStep>> verify_deep;   // [M - 1], parallel to verify_steps
    std::vector<uint32_t> verify_deep_from;
    std::unique_ptr<CapturedStep> wide;   // open-work 28: the 16-row list (slot 0 of one slot), null where not built
    std::unique_ptr<CapturedStep> deep32, deep64;   // spec 32: the 32- and 64-row lists, where the 16-row one is
    std::vector<uint32_t> draft_ids;
    uint32_t verify_k = kNoVerify, verify_pos = 0;
    // verify-cost-m8 (plan 31 Task 13: every slot of several with the MTP head): its slice of own_.gdn_pend (two
    // parities of kernels::gdn_pend_parity_bytes) its verify lists' DEFERRED build replays, and the list that applies
    // them where the state is read. Null elsewhere: the slot builds, gdn_live moved at commit.
    uint8_t* pend = nullptr;
    std::unique_ptr<CapturedStep> flush;
    // Where commit(j) takes row j's hidden: hh row 1 after verify(), the row list's saved row r0 after verify_rows.
    const uint8_t* verify_hidden = nullptr;
    uint32_t kv_end = 0;   // spec 31: PERF-90's written bound for this sequence (note_kv), what reset() zeroes
  };
  std::vector<Seq> seqs_;
  uint32_t cur_ = 0;
  Seq& seq() { return seqs_[cur_]; }
  const Seq& seq() const { return seqs_[cur_]; }
  void flush_gdn(const Seq& s) const;   // flush_gdn() for sequence `s` (verify_rows: every run's slot)
  // Slot 0 of several is paged (plan 19v3b Task 0): its own plain list, like every further slot's.
  const CapturedStep& plain() const { return seqs_[cur_].plain.empty() ? step_ : seqs_[cur_].plain.at(0); }
  CapturedStep& plain() { return seqs_[cur_].plain.empty() ? step_ : seqs_[cur_].plain.at(0); }
  // Spec 13c: the row lists at M = 2..kMaxRows (index M - 2) and what they read - several slots only.
  std::vector<CapturedStep> rows_steps_;
  std::unique_ptr<CapturedStep> rows_wide_[2];   // plan 40a Task 3: the 16- and 32-row lists, where built
  void build_rows_wide(bool moe_union);
  uint32_t rows_wide_from_ = 0;
  uint32_t rows_w_ = 0;                          // the last row step's list width (its ids' Control fields)
  bool rows_defer_ = false;                      // the last row step left its delta-net rows pending (PERF-120)
  BatchRowsBind rows_bind_;
  std::vector<CapturedStep> draft_rows_steps_;   // MTP: the row draft lists at M = 2..min(slots, kMaxDraftRows)
  DraftRowsBind draft_bind_;
  BatchLayout layout_;
  void setup_slots(const std::vector<uint32_t>& lens, uint32_t pool_pages);
  void capture_slot(uint32_t q);
  uint32_t gdn_index(uint32_t slot, uint32_t k) const { return seqs_[slot].first + k; }
  uint32_t max_draft_of(const Seq& s) const;
  void drop_rows_commits();   // the selected slot's unprojected rows commits (reset / restore discard them)
  void zero_slot();
  // PERF-90, one slot: a bound on the KV positions written since the last zero (all of them until the constructor's
  // reset). reset() zeroes positions [0, kv_end_) only - the rest is still zero, so the bytes match a whole fill. It is
  // raised before pos is lowered (load_state, reset) and by the host's own KV writes (prefill chunks, load_kv);
  // kKvAhead covers the rows a decode, draft or verify step writes past the pos it leaves (a 64-row verify keeps at
  // least one row, a draft chain writes the head's rows up to 3 ahead).
  static constexpr uint32_t kKvAhead = 2 * DecodeScratchDims::kDeepM;
  uint32_t kv_end_ = UINT32_MAX;
  void note_kv(uint32_t end) {
    kv_end_ = std::max(kv_end_, end);
    seqs_[cur_].kv_end = std::max(seqs_[cur_].kv_end, end);   // spec 31: the pool's per-sequence bound
  }
  // Spec 31: the pool (null with one sequence), a sequence's pages taken (the table entries written), and the same
  // that throws for `what` when the pool is full.
  std::unique_ptr<KvPool> pool_;
  // Pages up to `upto`, and those from `from` on made the sequence's own (a shared one copied first).
  bool take_pages_of(uint32_t slot, uint32_t upto, uint32_t from);
  void need_pages(uint32_t slot, uint32_t from, uint32_t upto, const char* what);
  uint32_t kv_write_from(uint32_t slot) const;   // the first position a step of the slot writes
  void write_table(uint32_t slot, uint32_t index, uint32_t n);   // its entries [index, index + n) to the card
  void copy_rows(uint32_t src_row, uint32_t dst_row, uint32_t n);   // every FA layer's KV rows, main and head
  // Plan 19v3b Task 0: the KV page table's host copy (empty with one slot) and the cache rows of slot `s`'s positions
  // [begin, end) as (first row, rows) runs - consecutive pages merged, so one run unless the table scatters them.
  std::vector<uint32_t> pages_;
  std::vector<std::pair<uint32_t, uint32_t>> kv_rows(const Seq& s, uint32_t begin, uint32_t end) const;

  double verify_deep_ms_ = 0;   // the deep lists' capture time
  double deep_capture_ms_[2] = {0, 0};   // spec 32e: the 32- and 64-row lists'
  void capture_verify(uint32_t M, const CaptureOptions& opt);
  void capture_wide(const CaptureOptions& opt);   // open-work 28: the 16-row list where it is built
  // Spec 39: the prefill's own allocations (EnginePrefillSizes: the MTP head's hidden rows, a drafter's tap staging).
  EnginePrefillBuffers pf_own_;
  // Spec 19c: the borrowed drafter (its GDN slots: own_.df_slots), and the prefill's tap staging (bf16
  // [chunk_cap][taps][hidden], lazy: the first prefill).
  TapDrafter* drafter_ = nullptr;
  // verify-cost-m8: with the pending GDN rows (own_.gdn_pend; plan 31 Task 13: a slice for every slot of several
  // with the MTP head, Seq::pend) the verify lists bind gdn_step's DEFERRED build (no per-row state; the committed rows
  // are replayed by the next launch), and the slot's flush list (Seq::flush) applies them where the state is read.
  // Null elsewhere: the slot builds, gdn_live moved at commit.
  // Spec 32: the ring copy a commit of rows 16.. runs (gdn_ring_commit.cl) beside the deep lists' chain scratch
  // (own_.gdn_x / conv_x); deep_rows_ the widest rows B70_SPEC_LOOKUP_ROWS allows (64 unset).
  std::unique_ptr<l0::Module> ring_mod_;
  bool greedy_head_ = false;   // PERF-123: verify() writes it into own_.head_gate (head_gate.h; 1: the candidate pass)
  std::unique_ptr<l0::Kernel> ring_k_;
  std::unique_ptr<l0::CmdList> ring_list_;
  uint32_t deep_rows_ = 64;
  // Plan 19v3b: the commits of verify_rows runs not yet projected - one rows projection replay for all of them.
  mutable std::vector<TapDrafter::RowsCommit> rows_commits_;
  l0::CmdList* profile_projection_ = nullptr;   // CaptureOptions test/probe override, borrowed
  // Snapshots materialize a deferred projection without changing the logical session.
  mutable l0::Queue queue_;
  mutable l0::Fence fence_;
  // Uploads, readbacks and snapshot copies only, never a token. `mutable`: the
  // save_* calls are const on the session and still append to it.
  mutable l0::CmdList imm_;
  // PERF-89: the copy engine's list for a mid-prompt hook's snapshot copies (null until the first), `snap_async_` set
  // while such a hook runs, `snap_pending_` while copies may be in flight.
  mutable std::unique_ptr<l0::CmdList> snap_bcs_;
  bool snap_async_ = false;
  mutable bool snap_pending_ = false;
  Control* control_;               // shared memory: the selected slot's (slot 0's is buffers_.control)
  Control* hctl_ = nullptr;        // spec 8: the selected slot's head control block (slot 0's: MtpBuffers::hctl)
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
