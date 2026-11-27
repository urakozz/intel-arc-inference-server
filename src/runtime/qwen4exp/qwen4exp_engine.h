#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/qwen4exp_loader.h"
#include "runtime/buffer_sizes.h"   // KvCache (device-free)
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_engine.h"   // PipelineOptions, PipelineLink (spec 16b's, unchanged)
#include "runtime/qwen4exp/qwen4exp_buffers.h"
#include "runtime/qwen4exp/qwen4exp_capture.h"   // ListSpec, MtpBinding (spec 21-QFN-v1e)

// Spec 21-QFN-v1c: Qwen3.8-Flash-Next's decode loop - runtime::Engine's contract (KolibriEngine's arrangement) on one
// card or, spec 16b's pieces, two: it owns the weights (each layer on its device: loader::load_qwen4exp's parts,
// the PLE table in host USM), one Qwen4ExpBuffers and one captured list per device, and a token is a replay of
// the lists. A prompt is fed one id per replay (`ingest`); 21-QFN-v1d adds the prefill, 21-QFN-v1e the snapshots and MTP.
//
// **At construction:** the PLE hash constants (multipliers, head sizes, offsets - the descriptor's, which 21-QFN-v1b's
// loader held equal to the checkpoint's I64 tensors) are written to the PLE device, and the PLE table's host
// ranges are read back BY THE DEVICE (q4_ple_check: the first u64 of every 2 MiB page through the pointer table)
// against the page words the loader recorded - a mismatch throws naming the range and the page (an aliased or
// unmapped host page: the xe hazard, spec 22 §1). With the expert tier's table (plan 21-QFN-v2a Task 8), each device
// reads sampled words of every routed block through its table (q4_expert_check: every 2 MiB page of every VRAM arena
// and host mirror) against the checkpoint's - a mismatch throws naming the device, layer, expert and block.
//
// **Two cards** (spec 16 §8's rules, KolibriEngine's): layers [0, s) on device 0 with the embedding, [s, L) plus
// the head on device 1. The host submits list 0 then list 1 and waits on device 1's fence (bounded); the lists
// order themselves through the hand-off - the MATERIALISED 4-stream H crosses (20 KB). The token's way back:
// device 1's argmax writes the id and advances pos; after the fence the host copies device 1's Control into
// device 0's. set_token writes both. A lost hand-off host-signals the event (so the queue drains), throws naming
// it, and marks the engine until reset().
//
// **The injected run** (spec 21-QFN-v1 F3's debug input): set_injected_selection(true) switches every replay to a
// second capture per device whose QSA layers read their lists from host-USM rows the caller writes before each
// replay (injected_list), without q4_qsa_score / _select; its launch count is injected_launches().
namespace runtime::qwen4exp {

// Spec 21-QFN-v1d: prefill() and its accessors are DEFINED in b70_qwen4exp_prefill (runtime/qwen4exp/
// qwen4exp_prefill_engine.cc), not in b70_qwen4exp_runtime - KolibriEngine's arrangement (runtime/kolibri/
// kolibri_engine.h): a target that never prefills links what it always linked; one that does links
// b70_qwen4exp_prefill. The prefill state is lazy (allocated by the first prefill or prepare_prefill), so a
// decode-only engine holds exactly what it did.
struct Qwen4ExpPrefillState;   // runtime/qwen4exp/qwen4exp_prefill_engine.cc

class Qwen4ExpEngine {
 public:
  // `devices[i]` is the context of placement device i (device 1 a view of device 0's context); `model` was
  // loaded onto them at `max_len`. debug_tap: the per-layer H copy the partial-forward gate reads. Two devices:
  // throws when device 0 cannot access device 1's memory (zeDeviceCanAccessPeer).
  // `cap` (spec 28d): the verify lists' capture override - tests and probes only (CaptureOptions::force_linear Gemv
  // binds the pre-28d gemv / gemv_i8w at M >= 2; the default binds gemm_sm there, runtime::gemv_kind).
  // cpu_experts (PERF-124, b70-decode / b70-serve --host-experts auto|gpu|cpu): Adaptive (auto) computes the host-memory
  // experts on the CPU where the startup measurement says it is faster; All (cpu) on the CPU always; Off (gpu) reads them
  // over PCIe.
  Qwen4ExpEngine(std::vector<l0::Context*> devices, loader::Q4LoadedModel model, uint32_t max_len,
                 bool debug_tap = false, const PipelineOptions& opt = {}, const CaptureOptions& cap = {},
                 CpuExperts::Mode cpu_experts = CpuExperts::Mode::Off);
  ~Qwen4ExpEngine();
  Qwen4ExpEngine(const Qwen4ExpEngine&) = delete;
  Qwen4ExpEngine& operator=(const Qwen4ExpEngine&) = delete;

  // Zeroes every device's persistent group (control, KV, indexer keys and tails, GDN state and rings, the PLE
  // rings) and the hand-off buffers; clears a failed hand-off.
  void reset();
  // The host-memory experts' CPU route (--host-experts auto or cpu, where some layer has a host mirror); null without.
  CpuExperts* cpu_experts() { return cpu_.get(); }
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is left
  // pending in cur_token).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t vocab() const { return model_.desc.vocab; }

  // --- the injected run (spec 21-QFN-v1 F3, debug) -----------------------------------------------------------------
  // On: every following replay runs the injected capture (built on first use); off: the normal one.
  void set_injected_selection(bool on);
  bool injected() const { return injected_; }
  // QSA layer qsa_index's host-USM row (kM = 1): positions [0, count), the count at word kCountWord. The caller
  // writes it before each replay.
  uint32_t* injected_list(uint32_t qsa_index);
  size_t injected_launches() const;

  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  // bf16 [layers][10240] of the last replay, each layer's row from the device holding it (debug_tap).
  std::vector<uint16_t> read_debug_H();
  // fp32 [layers][router_n] of the last replay: each MoE layer's router GEMV output before q4_route's bf16 rounding
  // (debug_tap; a MoE model).
  std::vector<float> read_debug_router();
  // u32 [layers][32]: every layer's route row of the last replay (kernels::qwen4exp::route).
  std::vector<uint32_t> read_routes();
  // Plan 21-QFN-v3a Task 0: of the last plain step's routed experts, those read from host memory (step_misses over
  // every device's part) - a sample off the timed path.
  uint32_t read_step_misses();
  // QSA layer `layer`'s selection of the last replay: positions [0, count) then the count (count + 1 words).
  std::vector<uint32_t> read_selection(uint32_t layer);
  // fp32 [qsa layers][2]: the 512th / 513th block scores of the last replay (+INF / -INF without a cut).
  std::vector<float> read_selection_diag();
  // fp32 [max_len / 4]: row 0's block scores of the last replay in the LAST QSA layer (a device's QSA layers share the
  // scores row, so only the last one's survive) - the prefill test's selection judge.
  std::vector<float> read_last_qsa_scores();
  // u64 [16]: the PLE row ids of the last replay (the golden set's ple.ids row).
  std::vector<uint64_t> read_ple_ids();
  // fp32 [vocab]: the last replay's logits (the last device's).
  std::vector<float> read_logits();
  void read_logits_into(float* host);
  // bf16 [count][2][256] of QSA layer `layer`'s K (or V) for positions [first, first + count) (--kv-cache int8: the
  // int8 codes, two a word).
  std::vector<uint16_t> read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v);
  // bf16 [count][128] of QSA layer `layer`'s compressed indexer keys for blocks [first_block, + count).
  std::vector<uint16_t> read_idx_keys(uint32_t layer, uint32_t first_block, uint32_t count);
  // Every persistent byte of layer `layer` (QSA: K, V, the indexer keys and the tail ring; GDN: the state and
  // the conv ring) and, on the PLE layer, the PLE rings after them - the same bytes on one card and two (the
  // F4 / Review Focus 6 comparisons).
  std::vector<uint8_t> read_layer_state(uint32_t layer);
  Control control(uint32_t dev) const;

  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line() const;   // one line per device

  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  size_t launches() const;           // every device's list
  size_t launches(uint32_t dev) const;
  uint32_t devices() const { return uint32_t(st_.size()); }
  uint32_t split() const { return model_.placement.split; }
  Q4Attn attention() const { return attn_; }
  const loader::Q4LoadedModel& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  // Probes only, one card: the decode list rebuilt with a timestamp event a launch, replayed `replays` times (each
  // replay decodes one id, pos advances); returns each launch's mean device time in us, walk order (step(0).labels).
  std::vector<double> profile_decode(uint32_t replays);
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that never
  // arrives. The step must throw within the bounds, never hang.
  void drop_next_handoff() { drop_next_ = true; }
  // The PLE table's device read-back at construction: pages checked (q4_ple_check).
  size_t ple_pages_checked() const { return ple_pages_checked_; }
  // The expert tier's device read-back at construction (the table form): block words read through each device's
  // table and held to the checkpoint's (q4_expert_check over loader::Q4DevicePart::check_*); 0 in the direct form.
  size_t expert_words_checked() const { return expert_words_checked_; }

  // --- spec 21-QFN-v1d: prefill (defined in b70_qwen4exp_prefill) --------------------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = model().pf_rows: 2048 or 4096) through qwen4exp_prefill.h's
  // walk on every device, then decode's head on the last row: leaves pos += ids.size() on every device and the first
  // generated id pending in cur_token (Engine::prefill's contract). The state it writes (KV, compressed keys and tail
  // rings, GDN state and conv rings, the PLE rings) is decode's to a cosine bar (the GEMMs' sum orders, the chunked GDN,
  // the flash walks), not bitwise (plan 21-QFN-v1d Review Focus 5). A prompt may be continued by another prefill. Two
  // devices: each chunk runs layers [0, s) on device 0, crosses by spec 16b's hand-off at chunk size (a second
  // PipelineLink a slot), then layers [s, L) on device 1, device 0 running chunk j + 1 while device 1 runs chunk j
  // (qwen4exp_prefill_engine.cc: the overlap; bitwise the sequential order's). A lost hand-off host-signals the event, throws naming it and marks
  // the engine until reset(). B70_Q4_ATTN=eager runs the eager sparse attention in prefill too (one switch, read at
  // construction). With injected() on, the sparse rows read the lists set_prefill_injector's callback writes before
  // each chunk (q4_qsa_score / _select not launched).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every binary the
  // walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // u32 [layers][pf_rows][32]: every layer's route rows of the last chunk (rows [0, C) written; pf_route_at), each
  // layer's from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  // QSA layer `layer`'s selection rows of the last chunk, u32 [pf_rows][kListRow] (positions [0, count), the count at
  // kCountWord; rows [0, C) written, the dense rows' identity lists only when the chunk had sparse rows - or the
  // injected rows when injected() is on).
  std::vector<uint32_t> read_prefill_selection(uint32_t layer);
  // fp32 [pf_rows][max_len / 4]: the last chunk's rows' block scores in the LAST QSA layer (read_last_qsa_scores' rule;
  // rows [0, C) written when the chunk had sparse rows).
  std::vector<float> read_prefill_last_qsa_scores();
  // The injected run on prefill (spec 21-QFN-v1 F3's debug input): with injected() on, before each chunk the engine calls
  // `f(qsa_index, pos, rows, lists)` for every QSA layer, `lists` the host-USM rows [rows][kListRow] the chunk's sparse
  // rows read (positions [0, count), count at kCountWord).
  using PrefillInjector = std::function<void(uint32_t qsa_index, uint32_t pos, uint32_t rows, uint32_t* lists)>;
  void set_prefill_injector(PrefillInjector f) { pf_inject_ = std::move(f); }
  // Spec 7's block (runtime::Engine::kBlock); spec 21-QFN-v1e's prefix cache hooks into the chunk walk: with a hook set every
  // chunk ends at a block end or at the prompt end (runtime::prefill_chunk_rows), and the hook is called on the host
  // after each chunk whose end is a multiple of kBlock (pos == end_pos, n_active 0 on every device; on two cards device
  // 0 may already run the next chunk - save_state / save_kv read the block end's state all the same) and once at the
  // prompt end (the first generated id in cur_token). PERF-30: on two cards the chunks run in pairs, and only a pair's
  // end is hooked (a block end inside a pair is not: no moment holds its state), as under tp.
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

  // --- spec 21-QFN-v1e: spec 7's prefix-cache snapshots ---------------------------------------------------------------
  // The state at pos (the GDN states and conv rows, the PLE history, the QSA layers' open-block raw keys; with the MTP
  // head its tail and R_{pos-1}) and the blocks (K / V rows and the complete blocks' compressed keys; the head's last):
  // runtime/qwen4exp/qwen4exp_sizes.h's state_runs / kv_runs give both HOST layouts, the same under --parallel 1 and --parallel pp
  // (spec 16b's rule). state_bytes() is 115,651,592 B at the real 48 layers and kv_bytes(n) the blocks of n positions
  // from a block start (24,792 B a position; derived). save_state copies the state at pos() (the zero runs' history:
  // zeros, the PLE ids as EOS; the rows a commit left replayed first); load_state writes it back and sets pos on every
  // device (n_active 0; the GDN state into gdn_state with no committed rows over it; the prefill that follows every
  // restore, spec 7 §3.3 step 4, sets the pending id). save_kv / load_kv copy [begin, end), begin a multiple of 4 (the
  // cache's is of kBlock); begin == end copies nothing. Every
  // call runs on the devices' immediate lists with no step in flight (a failed hand-off is reset by load_state,
  // which overwrites everything a session reads). kv_cache() is the descriptor's KV form (plan 21-QFN-v3a: Int8 with
  // --kv-cache int8, its scales runs of their own), the prefix cache's kv_form 0 / 1.
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  KvCache kv_cache() const { return model_.desc.kv8 ? KvCache::Int8 : KvCache::Bf16; }
  void save_state(void* host);
  void load_state(const void* host, uint32_t pos)
    pre(pos <= max_len_);
  void save_kv(uint32_t begin, uint32_t end, void* host);
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // --- spec 21-QFN-v1e: the MTP head - spec 8's draft / verify / commit (runtime/engine.h's contracts) -----------------
  //
  // On iff the model was loaded with its head (loader::load_qwen4exp(..., mtp = true)); off, none of this allocates
  // or captures anything and every call below throws. On, every device's scratch holds kVerifyRows rows and its GDN
  // layers their pending rows (PERF-51), the last device the head's buffers (runtime/qwen4exp/qwen4exp_sizes.h), and
  // the engine captures a verify list per M = 1..4 and a GDN flush list on every device and kMaxDraft draft lists on
  // the last.
  //
  // One iteration at pos = n, pending id x_n in cur_token[0] (every device's):
  //   draft(k)       the head drafts d_1..d_k on the last device: step i at position n - 1 + i on (R, t) - step 0 on
  //                  (R_{n-1}, x_n), step i on (its own pre-mixer H, d_i); step 0 selects its QSA blocks, later steps
  //                  attend step 0's list (decision 5). The ids go to
  //                  every device's cur_token[1..k] and draft_ids(); draft i's logits (q_i) are the head's row i
  //                  (read_draft_logits_into). `pick(i)` (the sampled path) replaces the on-card argmax as d_{i+1}.
  //                  `more(i, p)` (open-work 49, greedy): p the head's softmax probability at draft i's argmax (fp32,
  //                  max-subtracted, draft_prob.cl); false stops drafting there, so draft_ids() may hold fewer than k.
  //   verify(k)      the main model at M = k + 1 rows (x_n, d_1..d_k at n..n+k) on every device: row r's argmax in
  //                  verify_ids()[r], its logits row r (read_logits_into(host, k + 1)), its GDN recurrence inputs into
  //                  the pending rows (PERF-51; no per-row state), every row its own selection and top-10; then the
  //                  head's KV pass over the rows (positions n - 1 .. n + k - 1 from (R_{n-1+r}, x_{n+r})).
  //   commit(j, t)   j accepted drafts (0 <= j <= k): pos = n + j + 1 on every device, rows 0..j left for the next
  //                  launch to replay into the GDN state (Control::gdn_pend), R_{n+j} the next draft's R, t the pending
  //                  id. Rows past j are stale and never read (plan 21-QFN-v1e Review Focus 2).
  // verify(0) + commit(0, verify_ids()[0]) is one plain token; with the head on, ingest() and generate() run exactly
  // that (pos 0: the plain list, then its pre-mixer H into the head's R row), so the head's KV stays filled whatever
  // mix of plain and speculative steps runs; prefill() runs the head's pass per chunk (Review Focus 4).
  // verify(k) needs 1 <= pos and pos + k + 1 <= max_len; max_verify_k() is the largest k allowed now.
  static constexpr uint32_t kMaxDraft = runtime::qwen4exp::kMaxDraft;   // 3: verify at M <= 4
  bool mtp() const { return mtp_ != nullptr; }
  uint32_t max_verify_k() const;
  void draft(uint32_t k, const std::function<uint32_t(uint32_t i)>& pick = {},
             const std::function<bool(uint32_t i, float p)>& more = {});
  bool draft_probs() const { return mtp(); }   // every draft step ends with its probability (cli::pp's step-K choice)
  const std::vector<uint32_t>& draft_ids() const { return draft_ids_; }
  void verify(uint32_t k);
  const uint32_t* verify_ids() const;   // the last device's out_token[0..k]
  void commit(uint32_t j, uint32_t next_token)
    pre(next_token < model_.desc.vocab);
  // Spec 19e's prompt lookup: draft i's id from outside (every device's cur_token[1 + i]) before verify(k).
  void set_draft_input(uint32_t i, uint32_t id);
  // The last verify's (or plain step's) logits rows [0, rows) - fp32 [rows][vocab] - and draft step i's head row.
  void read_logits_into(float* host, uint32_t rows);
  void read_draft_logits_into(float* host, uint32_t i);
  // Pinned host rows for the sampled path (EngineAdapter's): `floats` fp32, allocated once (grown on demand).
  float* host_rows(size_t floats);
  // u32 [layers][M][32]: the last verify's route rows (M = its k + 1), each layer's from the device holding it - the
  // per-layer union of experts over the verify rows is spec 22 P0.6's "MTP verify" term.
  std::vector<uint32_t> read_verify_routes();
  uint32_t last_verify_rows() const { return last_verify_m_; }
  // QSA layer `layer`'s selection of the last verify's row r (positions, then the count) - M2's per-row compare.
  std::vector<uint32_t> read_verify_selection(uint32_t layer, uint32_t r);
  // fp32 [48][128][128] of GDN layer `layer`'s committed state (PERF-51: the rows the last commit left replayed first).
  std::vector<float> read_gdn_state(uint32_t layer);
  // u32 [count + 1] of draft step 0's selection (positions, then the count) and step i's route row (the head's MoE).
  std::vector<uint32_t> read_draft_selection();
  std::vector<uint32_t> read_draft_routes(uint32_t i);
  // bf16 [10240] of the head's R row 0 (R_{pos-1}) and of the last draft step's pre-mixer H (the next step's R).
  std::vector<uint16_t> read_mtp_R();
  std::vector<uint16_t> read_draft_H();
  // M1's test hook: replace R_{pos-1} (hh row 0) - the head then drafts on a reference's R (bf16 [10240]).
  void write_mtp_R(const std::vector<uint16_t>& R);
  MtpNorm mtp_norm() const { return mtp_norm_; }
  size_t verify_list_launches(uint32_t M) const;   // every device's verify list at M
  size_t draft_list_launches(uint32_t i) const;
  // The head's KV / compressed keys of positions [first, first + count) (bf16 [count][2][256] - int8: the codes, two
  // a word - / blocks [count][128]).
  std::vector<uint16_t> read_mtp_kv(uint32_t first, uint32_t count, bool v);
  std::vector<uint16_t> read_mtp_idx_keys(uint32_t first_block, uint32_t count);

 private:
  struct Stage;
  // Spec 21-QFN-v1d: a device's pieces for the prefill half (b70_qwen4exp_prefill), which cannot see Stage.
  l0::Context& stage_ctx(uint32_t dev) const;
  Qwen4ExpBuffers& stage_buffers(uint32_t dev) const;
  Control* stage_control(uint32_t dev) const;
  l0::CmdList& stage_imm(uint32_t dev) const;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;
  void build_injected();
  void ple_init();
  void expert_check();
  CapturedStep& active(uint32_t dev) const;
  // One step over every device's list `lists[dev]` (the hand-off, the bounds, the Control mirror) - step_once's
  // body; the plain step, the verify lists.
  void submit(const std::vector<CapturedStep*>& lists);
  // Spec 21-QFN-v1e: a snapshot run's device bytes (the GDN state's live slot with the MTP head).
  uint8_t* snap_ptr(const SnapRun& r) const;
  void settle_all(const char* what);
  // Spec 21-QFN-v1e: the MTP head's plain token (verify(0) + commit(0), or at pos 0 the plain list and R_0 into hh row 0),
  // what ingest() and generate() run with the head on; PERF-51: the rows the last commit left replayed into gdn_state
  // on every device (the flush lists, at n_active 0; before anything that reads gdn_state directly: a snapshot, a
  // prefill); the check every MTP call starts with.
  void mtp_step1();
  void flush_gdn();
  void require_mtp(const char* what) const;
  Control* hctl() const;

  loader::Q4LoadedModel model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  CaptureOptions cap_;   // spec 28d: the verify lists' override (tests only)
  Q4Attn attn_;
  // PERF-124 step 4: the host-RAM experts on the CPU (null: not asked for, or none in host RAM). Before st_:
  // the lists wait on its events and read its host buffers, so it is destroyed after them.
  std::unique_ptr<CpuExperts> cpu_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  StageLink binding_;
  [[=gs::Outside("the debug tap, a device each: memory_use and the plan's debug_tap term count it")]]
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  [[=gs::Outside("the injected selection rows, a device each: host memory, outside the device plan")]]
  std::vector<std::unique_ptr<l0::Mem>> inj_rows_;
  bool broken_ = false, drop_next_ = false, injected_ = false;
  std::vector<bool> pending_;
  size_t ple_pages_checked_ = 0, expert_words_checked_ = 0;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  BlockHook block_hook_;           // spec 21-QFN-v1d: empty = no hook (uniform chunks)
  PrefillInjector pf_inject_;      // spec 21-QFN-v1d: the injected run's lists on prefill
  // Spec 21-QFN-v1e: the MTP head (null without it), its binding into the lists, the iteration's bookkeeping.
  std::unique_ptr<Qwen4ExpMtpBuffers> mtp_;
  MtpBinding mtp_bind_;
  MtpNorm mtp_norm_ = MtpNorm::Single;
  static constexpr uint32_t kNoVerify = ~0u;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0, last_verify_m_ = 0;
  std::vector<uint32_t> draft_ids_;
  [[=gs::Outside("pinned host memory: outside the device plan")]] std::unique_ptr<l0::Mem> host_rows_;
  // Spec 21-QFN-v1d: set by prepare_prefill (b70_qwen4exp_prefill) - releases a prefill list waiting on a hand-off that will
  // not come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Plan 21-QFN-v2b Task 7 Step 2: set by prepare_prefill - points the table rows at the experts copied into the idle
  // prefill scratch once that copy has finished (Q4LendState::lend); submit() calls it before each step.
  std::function<void()> pf_lend_;
  // Inside a mid-prompt block hook on two cards: device 0's state runs at the block end (host; save_state reads them
  // there, and nothing settles the prefill lists - device 0 may run the next chunk). Null otherwise.
  const uint8_t* pf_shadow_ = nullptr;
  // Spec 21-QFN-v1d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is destroyed
  // before the buffers it points into. The deleter is set where the state is made (b70_qwen4exp_prefill), so this
  // header names no prefill symbol. pf_dev_bytes_: each device's prefill scratch + prefill link, for memory_use().
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  std::unique_ptr<Qwen4ExpPrefillState, void (*)(Qwen4ExpPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::qwen4exp
