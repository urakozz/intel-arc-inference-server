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
#include "runtime/qwen4exp/qwen4exp_capture.h"   // ListSpec, MtpBinding (spec 21e)

// Spec 21c: Qwen3.8-Flash-Next's decode loop - runtime::Engine's contract (KolibriEngine's arrangement) on one
// card or, spec 16b's pieces, two: it owns the weights (each layer on its device: loader::load_qwen4exp's parts,
// the PLE table in host USM), one Qwen4ExpBuffers and one captured list per device, and a token is a replay of
// the lists. A prompt is fed one id per replay (`ingest`); 21d adds the prefill, 21e the snapshots and MTP.
//
// **At construction:** the PLE hash constants (multipliers, head sizes, offsets - the descriptor's, which 21b's
// loader held equal to the checkpoint's I64 tensors) are written to the PLE device, and the PLE table's host
// ranges are read back BY THE DEVICE (q4_ple_check: the first u64 of every 2 MiB page through the pointer table)
// against the page words the loader recorded - a mismatch throws naming the range and the page (an aliased or
// unmapped host page: the xe hazard, spec 22 §1).
//
// **Two cards** (spec 16 §8's rules, KolibriEngine's): layers [0, s) on device 0 with the embedding, [s, L) plus
// the head on device 1. The host submits list 0 then list 1 and waits on device 1's fence (bounded); the lists
// order themselves through the hand-off - the MATERIALISED 4-stream H crosses (20 KB). The token's way back:
// device 1's argmax writes the id and advances pos; after the fence the host copies device 1's Control into
// device 0's. set_token writes both. A lost hand-off host-signals the event (so the queue drains), throws naming
// it, and marks the engine until reset().
//
// **The injected run** (spec 21 F3's debug input): set_injected_selection(true) switches every replay to a
// second capture per device whose QSA layers read their lists from host-USM rows the caller writes before each
// replay (injected_list), without q4_qsa_score / _select; its launch count is injected_launches().
namespace runtime::qwen4exp {

// Spec 21d: prefill() and its accessors are DEFINED in b70_qwen4exp_prefill (runtime/qwen4exp/
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
  Qwen4ExpEngine(std::vector<l0::Context*> devices, loader::Q4LoadedModel model, uint32_t max_len,
                 bool debug_tap = false, const PipelineOptions& opt = {});
  ~Qwen4ExpEngine();
  Qwen4ExpEngine(const Qwen4ExpEngine&) = delete;
  Qwen4ExpEngine& operator=(const Qwen4ExpEngine&) = delete;

  // Zeroes every device's persistent group (control, KV, indexer keys and tails, GDN state and rings, the PLE
  // rings) and the hand-off buffers; clears a failed hand-off.
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is left
  // pending in cur_token).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t vocab() const { return model_.desc.vocab; }

  // --- the injected run (spec 21 F3, debug) -----------------------------------------------------------------
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
  // u32 [layers][32]: every layer's route row of the last replay (kernels::qwen4exp::route).
  std::vector<uint32_t> read_routes();
  // QSA layer `layer`'s selection of the last replay: positions [0, count) then the count (count + 1 words).
  std::vector<uint32_t> read_selection(uint32_t layer);
  // fp32 [qsa layers][2]: the 512th / 513th block scores of the last replay (+INF / -INF without a cut).
  std::vector<float> read_selection_diag();
  // u64 [16]: the PLE row ids of the last replay (the golden set's ple.ids row).
  std::vector<uint64_t> read_ple_ids();
  // fp32 [vocab]: the last replay's logits (the last device's).
  std::vector<float> read_logits();
  void read_logits_into(float* host);
  // bf16 [count][2][256] of QSA layer `layer`'s K (or V) for positions [first, first + count).
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
  PpHandoff handoff() const { return opt_.handoff; }
  Q4Attn attention() const { return attn_; }
  const loader::Q4LoadedModel& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that never
  // arrives. The step must throw within the bounds, never hang.
  void drop_next_handoff() { drop_next_ = true; }
  // The PLE table's device read-back at construction: pages checked (q4_ple_check).
  size_t ple_pages_checked() const { return ple_pages_checked_; }

  // --- spec 21d: prefill (defined in b70_qwen4exp_prefill) --------------------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = kPfC 2048) through runtime/qwen4exp/qwen4exp_prefill.h's
  // walk on every device, then decode's head on the last row: leaves pos += ids.size() on every device and the first
  // generated id pending in cur_token (Engine::prefill's contract). The state it writes (KV, compressed keys and tail
  // rings, GDN state and conv rings, the PLE rings) is decode's to a cosine bar (the GEMMs' sum orders, the chunked GDN,
  // the flash walks), not bitwise (plan 21d Review Focus 5). A prompt may be continued by another prefill. Two
  // devices: each chunk runs layers [0, s) on device 0, crosses by spec 16b's `copy` hand-off at chunk size (a second
  // PipelineLink; the chunk always crosses by copy, whatever --pipeline-handoff says for decode: pp_handoff.cl's peer
  // kernels are sized for a 20 KB row, not 42 MB), then layers [s, L) on device 1 - sequentially (spec 16c's
  // overlapped chunk pipeline is a later lever). A lost hand-off host-signals the event, throws naming it and marks
  // the engine until reset(). B70_Q4_ATTN=eager runs the eager sparse attention in prefill too (one switch, read at
  // construction). With injected() on, the sparse rows read the lists set_prefill_injector's callback writes before
  // each chunk (q4_qsa_score / _select not launched).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every binary the
  // walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement (Kolibri's): record each chunk's list once per (pos, rows, injected) and device
  // and replay it; the hand-off's copies and event stay on the immediate lists (never inside a recording).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // u32 [layers][kPfC][32]: every layer's route rows of the last chunk (rows [0, C) written; pf_route_at), each
  // layer's from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  // QSA layer `layer`'s selection rows of the last chunk, u32 [kPfC][kListRow] (positions [0, count), the count at
  // kCountWord; rows [0, C) written, the dense rows' identity lists only when the chunk had sparse rows - or the
  // injected rows when injected() is on).
  std::vector<uint32_t> read_prefill_selection(uint32_t layer);
  // The injected run on prefill (spec 21 F3's debug input): with injected() on, before each chunk the engine calls
  // `f(qsa_index, pos, rows, lists)` for every QSA layer, `lists` the host-USM rows [rows][kListRow] the chunk's sparse
  // rows read (positions [0, count), count at kCountWord).
  using PrefillInjector = std::function<void(uint32_t qsa_index, uint32_t pos, uint32_t rows, uint32_t* lists)>;
  void set_prefill_injector(PrefillInjector f) { pf_inject_ = std::move(f); }
  // Spec 7's block (runtime::Engine::kBlock); spec 21e's prefix cache hooks into the chunk walk: with a hook set every
  // chunk ends at a block end or at the prompt end (runtime::prefill_chunk_rows), and the hook is called on the host
  // after each chunk whose end is a multiple of kBlock (devices idle, pos == end_pos, n_active 0 on every device) and
  // once at the prompt end (the first generated id in cur_token).
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

  // --- spec 21e: spec 7's prefix-cache snapshots ---------------------------------------------------------------
  // The state at pos (the GDN states and conv rows, the PLE history, the QSA layers' open-block raw keys; with the MTP
  // head its tail and R_{pos-1}) and the blocks (K / V rows and the complete blocks' compressed keys; the head's last):
  // runtime/qwen4exp/qwen4exp_sizes.h's state_runs / kv_runs give both HOST layouts, the same under --pp 1 and --pp 2
  // (spec 16b's rule). state_bytes() is 115,651,592 B at the real 48 layers and kv_bytes(n) the blocks of n positions
  // from a block start (25,344 B a position; derived). save_state copies the state at pos() (the zero runs' history:
  // zeros, the PLE ids as EOS); load_state writes it back and sets pos on every device (n_active 0; the GDN state into
  // slot 0, made live; the prefill that follows every restore, spec 7 §3.3 step 4, sets the pending id). save_kv /
  // load_kv copy [begin, end), begin a multiple of 4 (the cache's is of kBlock); begin == end copies nothing. Every
  // call runs on the devices' immediate lists with no step in flight (a failed hand-off is reset by load_state,
  // which overwrites everything a session reads). KV bf16 (spec 21 decision 8): kv_cache() is KvCache::Bf16, the
  // prefix cache's kv_form 0.
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  KvCache kv_cache() const { return KvCache::Bf16; }
  void save_state(void* host);
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host);
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // --- spec 21e: the MTP head - spec 8's draft / verify / commit (runtime/engine.h's contracts) -----------------
  //
  // On iff the model was loaded with its head (loader::load_qwen4exp(..., mtp = true)); off, none of this allocates
  // or captures anything and every call below throws. On, every device's scratch holds kVerifyRows rows and its GDN
  // layers three more state slots, the last device the head's buffers (runtime/qwen4exp/qwen4exp_sizes.h), and the
  // engine captures a verify list per M = 1..4 on every device and kMaxDraft draft lists on the last.
  //
  // One iteration at pos = n, pending id x_n in cur_token[0] (every device's):
  //   draft(k)       the head drafts d_1..d_k on the last device: step i at position n - 1 + i on (R, t) - step 0 on
  //                  (R_{n-1}, x_n), step i on (its own pre-mixer H, d_i); step 0 selects its QSA blocks, later steps
  //                  attend step 0's list (decision 5; B70_Q4_MTP_SELECT=fresh selects on every step). The ids go to
  //                  every device's cur_token[1..k] and draft_ids(); draft i's logits (q_i) are the head's row i
  //                  (read_draft_logits_into). `pick(i)` (the sampled path) replaces the on-card argmax as d_{i+1}.
  //   verify(k)      the main model at M = k + 1 rows (x_n, d_1..d_k at n..n+k) on every device: row r's argmax in
  //                  verify_ids()[r], its logits row r (read_logits_into(host, k + 1)), its GDN state into slot
  //                  (live + r) % 4, every row its own selection and top-10; then the head's KV pass over the rows
  //                  (positions n - 1 .. n + k - 1 from (R_{n-1+r}, x_{n+r})).
  //   commit(j, t)   j accepted drafts (0 <= j <= k): pos = n + j + 1 on every device, row j's GDN slot live, R_{n+j}
  //                  the next draft's R, t the pending id. Rows past j are stale and never read (plan 21e Review Focus 2).
  // verify(0) + commit(0, verify_ids()[0]) is one plain token; with the head on, ingest() and generate() run exactly
  // that (pos 0: the plain list, then its pre-mixer H into the head's R row), so the head's KV stays filled whatever
  // mix of plain and speculative steps runs; prefill() runs the head's pass per chunk (Review Focus 4).
  // verify(k) needs 1 <= pos and pos + k + 1 <= max_len; max_verify_k() is the largest k allowed now.
  static constexpr uint32_t kMaxDraft = runtime::qwen4exp::kMaxDraft;   // 3: verify at M <= 4
  bool mtp() const { return mtp_ != nullptr; }
  uint32_t max_verify_k() const;
  void draft(uint32_t k, const std::function<uint32_t(uint32_t i)>& pick = {});
  const std::vector<uint32_t>& draft_ids() const { return draft_ids_; }
  void verify(uint32_t k);
  const uint32_t* verify_ids() const;   // the last device's out_token[0..k]
  void commit(uint32_t j, uint32_t next_token);
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
  // u32 [count + 1] of draft step 0's selection (positions, then the count) and step i's route row (the head's MoE).
  std::vector<uint32_t> read_draft_selection();
  std::vector<uint32_t> read_draft_routes(uint32_t i);
  // bf16 [10240] of the head's R row 0 (R_{pos-1}) and of the last draft step's pre-mixer H (the next step's R).
  std::vector<uint16_t> read_mtp_R();
  std::vector<uint16_t> read_draft_H();
  // M1's test hook: replace R_{pos-1} (hh row 0) - the head then drafts on a reference's R (bf16 [10240]).
  void write_mtp_R(const std::vector<uint16_t>& R);
  MtpNorm mtp_norm() const { return mtp_norm_; }
  MtpSelect mtp_select() const { return mtp_select_; }
  size_t verify_list_launches(uint32_t M) const;   // every device's verify list at M
  size_t draft_list_launches(uint32_t i) const;
  // The head's KV / compressed keys of positions [first, first + count) (bf16 [count][2][256] / blocks [count][128]).
  std::vector<uint16_t> read_mtp_kv(uint32_t first, uint32_t count, bool v);
  std::vector<uint16_t> read_mtp_idx_keys(uint32_t first_block, uint32_t count);

 private:
  struct Stage;
  // Spec 21d: a device's pieces for the prefill half (b70_qwen4exp_prefill), which cannot see Stage.
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
  CapturedStep& active(uint32_t dev) const;
  // One step over every device's list `lists[dev]` (the hand-off, the bounds, the Control mirror) - step_once's
  // body; the plain step, the verify lists.
  void submit(const std::vector<CapturedStep*>& lists);
  // Spec 21e: a snapshot run's device bytes (the GDN state's live slot with the MTP head).
  uint8_t* snap_ptr(const SnapRun& r) const;
  void settle_all(const char* what);
  // Spec 21e: the MTP head's plain token (verify(0) + commit(0), or at pos 0 the plain list and R_0 into hh row 0),
  // what ingest() and generate() run with the head on; the live GDN slot copied into slot 0 (before anything that
  // reads gdn_state directly: prefill); the check every MTP call starts with.
  void mtp_step1();
  void mtp_normalise_live();
  void require_mtp(const char* what) const;
  Control* hctl() const;

  loader::Q4LoadedModel model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  Q4Attn attn_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  StageLink binding_;
  std::vector<std::unique_ptr<l0::Mem>> taps_, inj_rows_;
  bool broken_ = false, drop_next_ = false, injected_ = false;
  std::vector<bool> pending_;
  size_t ple_pages_checked_ = 0;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  BlockHook block_hook_;           // spec 21d: empty = no hook (uniform chunks)
  PrefillInjector pf_inject_;      // spec 21d: the injected run's lists on prefill
  // Spec 21e: the MTP head (null without it), its binding into the lists, the iteration's bookkeeping.
  std::unique_ptr<Qwen4ExpMtpBuffers> mtp_;
  MtpBinding mtp_bind_;
  MtpNorm mtp_norm_ = MtpNorm::Single;
  MtpSelect mtp_select_ = MtpSelect::Reuse;
  static constexpr uint32_t kNoVerify = ~0u;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0, last_verify_m_ = 0;
  std::vector<uint32_t> draft_ids_;
  std::unique_ptr<l0::Mem> host_rows_;
  // Spec 21d: set by prepare_prefill (b70_qwen4exp_prefill) - releases a prefill list waiting on a hand-off that will
  // not come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 21d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is destroyed
  // before the buffers it points into. The deleter is set where the state is made (b70_qwen4exp_prefill), so this
  // header names no prefill symbol. pf_dev_bytes_: each device's prefill scratch + prefill link, for memory_use().
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  int pf_replay_ = -1;   // -1: B70_PREFILL_REPLAY decides
  std::unique_ptr<Qwen4ExpPrefillState, void (*)(Qwen4ExpPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::qwen4exp
