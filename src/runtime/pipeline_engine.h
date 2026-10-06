#pragma once
#include <array>
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
#include "l0/sync_event.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill_backend.h"

// Spec 16b: decode with the model's layers split over two B70s (`b70-decode --pp 2`).
//
//   device 0  embedding, layers [0, s), their GDN state / conv ring / KV, decode list 0
//   device 1  layers [s, L), their state, final norm, lm_head, argmax, decode list 1
//
// One token is one replay of each list. The host submits list 0 and list 1 back to back
// and waits on device 1's fence (then device 0's, already done): list 0 ends with the
// hand-off out, list 1 starts with the hand-off in (runtime::build_stage, capture.h), so the
// two devices order themselves - no host round trip between them. `--pipeline-handoff copy`
// (default) or `peer` - spec 16 §2 option (a) or (b); 16a's probe, which would have chosen,
// has not run.
//
// **The token's way back (spec 16b Review Focus 2).** argmax_stage2 on device 1 writes the
// sampled id and advances `pos` in device 1's Control. After the fence the host copies
// device 1's 128-byte Control block into device 0's (both are shared allocations the host
// reads anyway - Engine::generate reads cur_token after every fence). That is the one host
// action per token beyond the fence wait the single-card engine already has; no extra wait,
// no extra submission. A host sampler writes its id with set_token(), into BOTH blocks.
//
// **pos in step (Review Focus 3).** Device 0's kernels read pos / n_active / cur_token from
// device 0's Control, device 1's from device 1's; nothing on device 0 writes its block. So
// the two blocks are equal at every step boundary: reset() zeroes both, ingest/generate set
// n_active and cur_token in both, every step ends with the mirror, load_state() sets pos in
// both. tests/runtime/pp_decode_test.cc checks the equality after every phase.
//
// **Bounded (P4).** Every fence wait is bounded (PipelineOptions::timeout_ms); `peer`'s
// pp_recv bounds its spin and reports in its state words. A failed hand-off throws, names
// what did not arrive, releases a list still waiting on the copy event (host-signals it so
// the queue drains), and marks the engine: every later step throws until reset().
//
// **Prefill (spec 16c)** - prefill() below: the chunk pipeline of
// runtime/pipeline_prefill_plan.h. Each device has its own prefill context (immediate list,
// kernel cache, PrefillScratch, Int8State); device 0 runs a chunk's layers [0, s) and hands
// its rows into one of two landing slots on device 1, device 1 runs layers [s, L) on them
// while device 0 starts the next chunk. Defined in b70_prefill_host
// (runtime/prefill/pipeline_prefill.cc), as Engine::prefill is, so a decode-only binary
// links no prefill symbol: this class reaches that half only through PipelinePrefillBase's
// virtual calls.
//
// Not here: the residual tap, profiling. Snapshots (spec 7) ARE here, in the single-card
// host layout, so a snapshot moves between --pp 1 and 2 unchanged.
//
// **MTP across the split (spec 16d)** - draft() / verify() / commit() below, Engine's
// contract. The head, its buffers and its draft lists live on device 1 with lm_head, and
// device 1 holds a replica of the embedding for them (runtime/pipeline_place.h). A verify at
// M = k + 1 rows is one replay of each device's verify list (runtime::build_stage_verify:
// the stage's layers at M rows with the per-row GDN slots, M rows handed off together, the
// head's KV fill on device 1); a draft is a replay of device 1's draft list alone. gdn_live
// and pos move on BOTH Control blocks at commit, as pos does after every step. With MTP on,
// ingest() and generate() run verify(0) + commit(0) per id (Engine's mtp_step1), and
// prefill() fills the head's KV on device 1 after each chunk's layers (step_mtp_kv). The
// snapshots carry the head's hidden row and KV layer in Engine's MTP layouts.
namespace runtime {

struct PipelineOptions {
  PpHandoff handoff = kDefaultPpHandoff;
  uint32_t timeout_ms = 30000;      // one step's fence wait; a Qwen3.8 step is ~36 ms
  uint32_t spin_limit = 1u << 24;   // peer: pp_recv's flag loads before it gives up
  // Spec 16c: one prefill wait's bound - the host waits for a device to finish a chunk it
  // appended up to two chunks earlier. A one-card 2048-row chunk is ~1 s at 4k and a few s
  // near 256k (derived from docs/BENCHMARKS.md's pp rows); the bound is for a hang, not a
  // slow chunk.
  uint32_t prefill_timeout_ms = 120000;
  // Spec 16c, a TEST hook (pp_prefill_fail_test's back-pressure case): every device-1 chunk
  // first waits on an event a host thread signals this many ms after the chunk is appended,
  // so device 1 is artificially slow and device 0 must wait for its slots. 0: off. Also
  // set_prefill_hold_ms() between prefills.
  uint32_t prefill_hold_ms = 0;
};

// Spec 16c: the prefill half's state (both prefill contexts, scratches, the landing slots,
// the events), built by prepare_prefill() in b70_prefill_host. PipelineEngine (b70_runtime)
// only settles, zeroes, measures and destroys it - through these virtual calls, so this
// archive names no prefill symbol (Engine's PrefillEngine rule, by another means: reset()
// must reach it).
struct PipelinePrefillBase {
  virtual ~PipelinePrefillBase() = default;
  // Releases a list waiting on a hand-off that will not come and waits (bounded) for both
  // prefill lists to go idle: true when they did.
  virtual bool settle(uint32_t timeout_ms) = 0;
  // The prefill hand-off's state back to a fresh session's: events reset, peer counters 0.
  virtual void zero() = 0;
  // Its device bytes, in memory_line()'s components (prefill scratch, int8, decode state).
  virtual MemoryComponents memory(uint32_t dev) const = 0;
};

// The hand-off buffers (pipeline_plan.h: pp_landing_layout).
struct PipelineLink {
  PipelineLink(l0::Context& d0, l0::Context& d1, const model::ModelDesc& d, PpHandoff mode);
  // Spec 20c: from a layout (pp_landing_layout(resid_bytes, sumsq_bytes)) - Kolibri-1's engine,
  // which has no ModelDesc. The form above delegates to this one.
  PipelineLink(l0::Context& d0, l0::Context& d1, const PpLandingLayout& layout, PpHandoff mode);
  PpHandoff mode;
  PpLandingLayout layout;
  l0::Mem landing;      // device 1, written only by device 0
  l0::Mem send_seq;     // device 0: pp_send's counter
  l0::Mem recv_state;   // device 1, shared: pp_recv's state words, read by the host
  std::unique_ptr<l0::SyncEvent> event;   // copy only
  StageLink binding(uint32_t spin_limit) const;
  size_t bytes(uint32_t device) const;
  // Zeroes the landing buffer, both counters and the flag; resets the event.
  void zero(l0::CmdList& imm0, l0::CmdList& imm1);
};

class PipelineEngine {
 public:
  // `stages` is runtime::place_stages()'s pair (device 0's model, device 1's), both loaded at
  // `max_len`. `d1` is a view of `d0`'s context. Throws when device 0 cannot access device
  // 1's memory (zeDeviceCanAccessPeer) - both hand-offs write it.
  PipelineEngine(l0::Context& d0, l0::Context& d1, std::vector<loader::LoadedModel> stages,
                 uint32_t max_len, const PipelineOptions& opt = {},
                 KvCache kv = default_kv_cache());
  ~PipelineEngine();
  PipelineEngine(const PipelineEngine&) = delete;
  PipelineEngine& operator=(const PipelineEngine&) = delete;

  // Engine's contract, on both devices: zeroes both persistent groups (both Controls) and
  // the hand-off buffers; clears a failed hand-off.
  void reset();
  void ingest(const std::vector<uint32_t>& ids);
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  // The pending id, written where both devices read it (spec 16b Review Focus 2).
  void set_token(uint32_t id);
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  uint32_t split() const { return stage(0).last; }
  const PpStage& stage(uint32_t dev) const;
  PpHandoff handoff() const { return opt_.handoff; }
  KvCache kv_cache() const { return kv_; }
  const CapturedStep& step(uint32_t dev) const;
  DecodeBuffers& buffers(uint32_t dev);
  const Control& control(uint32_t dev) const;
  const loader::LoadedModel& model(uint32_t dev) const;
  l0::Context& context(uint32_t dev) const;

  // Spec 7, in Engine's host layouts exactly: save_state is every GDN layer's state then
  // every conv ring - device 0's slices then device 1's, which IS the single-card order -
  // and save_kv per tensor (K, V) every layer's rows [then at int8 every layer's scales],
  // device 0's layers first (runtime::pp_kv_runs). Spec 16d, MTP on: the state is the LIVE
  // verify slot of each device's layers, then the head's hidden row h_{pos-1} last, and the
  // KV has the head's layer after the last FA layer - Engine's MTP layouts. load_state sets
  // pos in both Controls (and gdn_live 0).
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host) const;
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host) const;
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // Device 1's logits row 0 after the last step (fp32 [kVocab]) - P1's comparison.
  std::vector<float> read_logits() const;

  // --- spec 16d: MTP across the split (Engine's contract, engine.h "spec 8") ---------------
  //
  // On iff device 1's stage model carries the head (place_stages of a model loaded with
  // mtp = true). Off, nothing below allocates and every call throws.
  //   draft(k, pick)   device 1's draft lists i = 0..k-1 from (h_{pos-1}, the pending id);
  //                    d_i into draft_ids() and cur_token[1 + i] of BOTH Control blocks.
  //   verify(k)        both devices' verify lists at M = k + 1 (bounded, as a step); row r's
  //                    argmax in verify_ids()[r], its logits in device 1's logits row r.
  //   commit(j, t)     gdn_live += j and pos = n + j + 1 on both devices, hh row 1 + j -> 0
  //                    on device 1, t pending.
  static constexpr uint32_t kMaxDraft = MtpBuffers::kMaxK;   // 3
  bool mtp() const;
  uint32_t max_verify_k() const;
  void draft(uint32_t k, const std::function<uint32_t(uint32_t i)>& pick = {});
  const std::vector<uint32_t>& draft_ids() const { return draft_ids_; }
  void verify(uint32_t k);
  const uint32_t* verify_ids() const;
  void commit(uint32_t j, uint32_t next_token);
  uint32_t draft_vocab() const;   // |V'| (spec 8 §11), 0 = the full head drafts
  const CapturedStep& verify_step(uint32_t dev, uint32_t M) const;
  const CapturedStep& draft_step(uint32_t i) const;
  MtpBuffers* mtp_buffers();   // device 1's, null when MTP is off

  // --- spec 16d: the server's host reads (b70-serve --pp 2, cli/pipeline_serve_adapter.h) ---
  //
  // The pending id (device 1's Control, mirrored on device 0 at every step boundary).
  uint32_t pending() const;
  // d_{i+1} as the next verify's input row 1 + i, on both devices (a host-proposed draft:
  // spec 19e's lookup, or a sampled draft overriding the on-card argmax).
  void set_draft_input(uint32_t i, uint32_t id);
  // Rows [0, rows) of device 1's logits (the last step's row 0, or a verify's k + 1 rows) /
  // the head's draft logits row i, fp32 [kVocab] each, into `host` (any host memory; the
  // pinned host_rows() buffer is the fast one).
  void read_logits_into(float* host, uint32_t rows = 1) const;
  void read_draft_logits_into(float* host, uint32_t i) const;
  // `floats` of pinned host memory (device 1's context), allocated on the first call and
  // kept: the sampler's readback rows (probe-mtp §3: pageable costs 2.5x).
  float* host_rows(size_t floats);

  // Spec 6's memory line per device ("memory, device 0: model ..., total ... of ...").
  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line(uint32_t dev) const;

  // P4's test hook: the next step submits device 1's list only - a hand-off that never
  // arrives. The step must throw within the bounds, never hang. Spec 16c: and the next
  // prefill's first chunk is not handed over (device 0 neither copies nor signals it).
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 16c: prefill across the two cards ------------------------------------------
  //
  // Engine::prefill's contract, on two devices (runtime/pipeline_prefill_plan.h has the
  // order): `ids` in chunks of at most `chunk` (0 = PrefillScratch::kC) from the current
  // pos, the same chunks one card cuts; on return pos == old pos + ids.size() in both
  // Controls and cur_token[0] is the first generated id. KV, GDN state, conv ring and the
  // logits row are bitwise one card's (P1). Every wait is bounded (prefill_timeout_ms); a
  // failure throws and marks the engine (reset() recovers), as a failed step does.
  // Defined in b70_prefill_host (runtime/prefill/pipeline_prefill.cc).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Everything prefill() sets up before its first chunk (each device's scratch, context,
  // and on l0-int8 its own linears' column scales). Idempotent. Refuses, by name, what has
  // no two-card walk: sycl-tla, B70_PREFILL_ATTN=composed, B70_PREFILL_REPLAY=1,
  // B70_PREFILL_PROFILE=1.
  void prepare_prefill();
  void set_prefill_backend(PrefillBackend b) { pf_backend_ = b; pf_backend_set_ = true; }
  PrefillBackend prefill_backend() const;   // defined in pipeline_prefill.cc
  // Spec 7 §3.2's hook, as Engine's: after every mid-prompt chunk that ends on a block end
  // (chunks then end at block ends) and after the prompt's end. Mid-prompt, the hook runs
  // once BOTH devices have finished the block - the next chunk may already be running - and
  // save_state() inside it reads the block end's shadows (each device copies its GDN state
  // and conv ring there after a hooked chunk); save_kv() reads [0, end), which later chunks
  // do not write. pos() reads the block end. A hook that throws leaves the state at its end
  // (the lists drain, the shadows go back), as on one card.
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }
  static constexpr uint32_t kBlock = 2048;   // Engine::kBlock
  // The last prefill(): wall time and each device's busy time (its chunks' walks, from
  // device timestamps around them), so a run shows which card the pipeline waited for
  // (spec 16 plan 16c Review Focus 5).
  struct PrefillStats {
    uint32_t chunks = 0;
    double wall_ms = 0;
    std::array<double, kPpDevices> busy_ms{}, busy_max_ms{};
    std::array<size_t, kPpDevices> launches{};
    // What the walk should have appended (step_stage_launches per chunk, the peer hand-off's
    // pp_send / pp_recv, the head) - the box test holds `launches` to it.
    std::array<size_t, kPpDevices> expected_launches{};
  };
  const PrefillStats& last_prefill() const { return pf_stats_; }
  // The test hook above, between prefills.
  void set_prefill_hold_ms(uint32_t ms) { opt_.prefill_hold_ms = ms; }
  void set_prefill_timeout_ms(uint32_t ms) { opt_.prefill_timeout_ms = ms; }
  // Launches each device's prefill list has appended (0 before prepare_prefill(); the
  // l0-int8 column-scale passes count, as Engine::prefill_launches does).
  size_t prefill_launches(uint32_t dev) const;   // defined in pipeline_prefill.cc
  // The last prefill's logits row (device 1's prefill scratch, fp32 [kVocab]) - P1's
  // comparison with one card's prefill_scratch()->logits.
  std::vector<float> read_prefill_logits() const;   // defined in pipeline_prefill.cc

 private:
  struct Stage;
  struct PrefillDriver;   // spec 16c: pp_prefill_run's driver (pipeline_prefill.cc)
  void step_once();
  // Spec 16d: one replay of a list on each device (step_once's protocol and bounds), and of
  // one list on device 1 alone (a draft). step_once is run_pair over the decode lists.
  void run_pair(l0::CmdList& list0, l0::CmdList& list1, const char* what);
  void run_one(uint32_t dev, l0::CmdList& list, const char* what);
  void require_mtp(const char* what) const;
  void mtp_step1();             // Engine::mtp_step1: one plain token with MTP on
  void mtp_normalise_live();    // Engine::mtp_normalise_live, on both devices
  // Defined in pipeline_engine.cc, called by prefill(): the live slot into slot 0, no
  // verify pending.
  void mtp_before_prefill();
  size_t slot_bytes() const;    // one whole-model verify slot (SPEC_SLOT_STRIDE x 4)
  const l0::Mem& live_gdn(uint32_t dev) const;   // a device's GDN state as save_state reads it
  // Waits (bounded) for device `dev`'s outstanding step, if any; true when none is left.
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;

  uint32_t max_len_;
  PipelineOptions opt_;
  KvCache kv_;
  std::array<std::unique_ptr<Stage>, kPpDevices> stages_;
  std::unique_ptr<PipelineLink> link_;
  // Spec 16c: null until prepare_prefill(). Declared after stages_ and link_, so it goes
  // first (its lists settle before the buffers they use are freed).
  std::unique_ptr<PipelinePrefillBase> pf_;
  PrefillBackend pf_backend_ = PrefillBackend::L0Int8;
  bool pf_backend_set_ = false;
  BlockHook block_hook_;
  PrefillStats pf_stats_;
  // Set only while a mid-prompt block hook runs: save_state reads these (each device's
  // GDN state / conv ring shadow at the block end) instead of the live state.
  std::array<const l0::Mem*, kPpDevices> snap_gdn_{}, snap_conv_{};
  // Spec 16d: and device 1's shadow of the head's hidden row (MTP on).
  const l0::Mem* snap_hh_ = nullptr;
  std::vector<uint32_t> draft_ids_;
  static constexpr uint32_t kNoVerify = 0xFFFFFFFFu;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0;
  std::unique_ptr<l0::Mem> host_rows_;   // host_rows(): pinned, device 1's context
  bool broken_ = false, drop_next_ = false;
  std::array<bool, kPpDevices> pending_{};   // a submitted step whose fence is not yet seen
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
