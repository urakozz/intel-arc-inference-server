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
#include "loader/kolibri1_loader.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_buffers.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_engine.h"   // PipelineOptions, PipelineLink (spec 16b's, unchanged)
#include "runtime/tap_drafter.h"      // Kolibri DSpark: TapSink (an interface: no link dependency)

// Spec 20c: Kolibri-1's decode loop - runtime::Engine's contract on one card
// or, spec 16b's pieces, two: it owns the weights (each layer on its device: loader::load_kolibri1's
// parts), one KolibriBuffers and one captured list per device, and a token is a replay of the lists.
// A prompt is fed one id per replay (`ingest`) or, spec 20d, in chunks (`prefill`, below). No MTP
// (Kolibri has no head). Spec 20e: spec 7's prefix-cache snapshots (the rings' last 512 positions as the
// state, the full layers' KV as the blocks - runtime/kolibri/kolibri_sizes.h's layouts) and the server's
// calls (pending, read_logits_into, vocab).
//
// **Two cards** (spec 16 §8's rules, PipelineEngine's, here for a model PipelineEngine cannot run):
// layers [0, s) on device 0 with the embedding, [s, L) plus the head on device 1 (KolPlacement). The
// host submits list 0 then list 1 and waits on device 1's fence (bounded: PipelineOptions::timeout_ms);
// the lists order themselves through the hand-off (a device-to-device copy and a cross-device event).
// **The token's way back:** argmax on device 1 writes the id and
// advances pos in device 1's Control; after the fence the host copies device 1's Control block into
// device 0's. set_token writes both. **Bounded:** a lost hand-off host-signals the event (so the queue
// drains), throws naming it, and marks the engine until reset().
namespace runtime::kolibri {

// Spec 20d: prefill() and its accessors are DEFINED in b70_kolibri_prefill
// (runtime/kolibri/kolibri_prefill_engine.cc), not in b70_kolibri_runtime, so
// a target that never prefills links what it always linked; one that does links
// b70_kolibri_prefill. The prefill state is lazy (allocated by the first prefill or prepare_prefill), so a
// decode-only engine holds exactly what it did.
struct KolibriPrefillState;   // runtime/kolibri/kolibri_prefill_engine.cc

class KolibriEngine {
 public:
  // `devices[i]` is the context of placement device i (device 1 a view of device 0's context,
  // l0::Context(const Context&, 1)); `model` was loaded onto them at `max_len`. debug_tap: the
  // per-layer resid copy the partial-forward gate reads (off for speed). Two devices: throws when
  // device 0 cannot access device 1's memory (zeDeviceCanAccessPeer) - the hand-off writes it.
  KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len,
                bool debug_tap = false, const PipelineOptions& opt = {});
  // The debug tap's full form (a test's: on, and its point - kolibri_sizes.h DebugTap).
  KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len, DebugTap tap,
                const PipelineOptions& opt = {});
  ~KolibriEngine();
  KolibriEngine(const KolibriEngine&) = delete;
  KolibriEngine& operator=(const KolibriEngine&) = delete;

  // Zeroes every device's persistent group (control, full KV, rings) and the hand-off buffers; clears
  // a failed hand-off.
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is
  // left pending in cur_token).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  // The pending id, written where every device reads it (PipelineEngine::set_token's rule).
  void set_token(uint32_t id);
  // Spec 20e: the pending id (the last device's cur_token[0]: the argmax of the last replay or prefill,
  // or what set_token wrote) and the head's rows.
  uint32_t pending() const;
  uint32_t vocab() const { return model_.desc.vocab; }

  // --- spec 20e: spec 7's prefix-cache snapshots ---------------------------------------------------
  // Kolibri's "state" is the sliding rings' last state_positions (512) positions - what the next query's
  // window reads - and its blocks are the full layers' KV: runtime/kolibri/kolibri_sizes.h's state_runs /
  // kv_runs give both host layouts, which are the same under --parallel 1 and --parallel pp (spec 16b's rule: layers in
  // layer order, whichever device holds them). state_bytes() is 41,943,040 B and kv_bytes(n) n x 20,480 B
  // at the real shapes (derived). save_state copies the rings' rows of [pos - 512, pos) (zeros for
  // positions before 0); load_state writes them back into their slots (p & 4095, the zeros into the slots
  // of positions before 0) and sets pos on every device (n_active 0; the cur_token is the prefill's that
  // follows every restore, spec 7 §3.3 step 4). save_kv / load_kv copy positions [begin, end) of every full
  // layer; begin == end copies nothing; throws unless begin <= end <= max_len. Every call runs on the
  // devices' immediate lists with no step in flight (a hand-off that failed is reset by load_state, which
  // overwrites everything a session reads).
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host);
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host);
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  // bf16 [layers][hidden] of the last replay, each layer's row from the device holding it (debug_tap).
  std::vector<uint16_t> read_debug_resid();
  // u32 [layers][32]: every layer's route row of the last replay (kernels::kolibri::route).
  std::vector<uint32_t> read_routes();
  // fp32 [vocab]: the last replay's logits (the last device's) - or the last prefill's head row.
  std::vector<float> read_logits();
  void read_logits_into(float* host);   // the same row into `host` (vocab floats), no allocation
  // bf16 [count][kv_heads][128] of layer `layer`'s K (or V) for absolute positions [first, first +
  // count): a full layer's rows, a sliding layer's through the ring (row p & (kRing - 1)) - the caller
  // asks only for positions the ring still holds.
  std::vector<uint16_t> read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v);
  // A device's whole Control block, for the mirror's tests.
  Control control(uint32_t dev) const;

  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line() const;   // one line per device

  uint32_t devices() const { return uint32_t(st_.size()); }
  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  size_t launches() const;           // every device's list
  size_t launches(uint32_t dev) const;
  uint32_t split() const { return model_.placement.split; }
  KolAttn attention() const { return attn_; }
  const loader::KolLoadedModel& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that
  // never arrives. The step must throw within the bounds, never hang. Spec 20d: the next prefill CHUNK
  // likewise (device 0's half not appended), unless prefills are replayed.
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 20d: prefill (defined in b70_kolibri_prefill) ---------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = kPfC 2048) through runtime/kolibri/
  // kolibri_prefill.h's walk on every device, then the head on the last row: leaves pos += ids.size() on
  // every device and the first generated id pending in cur_token (Engine::prefill's contract). The KV and
  // ring rows it writes are decode's to a cosine bar (the GEMMs' sum order), not bitwise (plan 20d Review
  // Focus 4). A prompt may be continued by another prefill. Two devices: each chunk runs layers [0, s) on
  // device 0, crosses by spec 16b's hand-off at chunk size (a second PipelineLink), then layers [s, L) on
  // device 1 - sequentially
  // (spec 16c's overlapped chunk pipeline is a later lever). A lost hand-off host-signals the event, throws
  // naming it and marks the engine until reset(). B70_KOLIBRI_ATTN=eager runs the eager prefill attention
  // too (the engine's attention(), read once at construction).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every
  // binary the walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  // u32 [layers][kPfC][32]: every layer's route rows of the last chunk (rows [0, C) written; pf_route_at),
  // each layer's from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  // debug_tap: the last chunk's per-layer output rows, [layers][kPfC][hidden] bf16 (row t of the chunk at t).
  std::vector<uint16_t> read_prefill_tap();
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // Spec 7's block (runtime::Engine::kBlock); spec 20e's prefix cache hooks into the chunk walk: with a hook
  // set every chunk ends at a block end or at the prompt end (runtime::prefill_chunk_rows), and the hook is
  // called on the host after each chunk whose end is a multiple of kBlock (devices idle, pos == end_pos,
  // n_active 0 on every device) and once at the prompt end (the first generated id in cur_token).
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

  // --- Kolibri DSpark: speculative decoding (--spec dflash with the DSpark drafter) -----------
  // enable_spec allocates every device's KolibriVerifyBuffers (the verify scratch at 8 rows, control_hi) and, two cards,
  // the verify link (a copy hand-off sized for 8 rows); the verify lists are captured on first use per M
  // (runtime/kolibri/kolibri_capture.h build_verify). Before it nothing of this exists: a plain run allocates, captures and
  // replays exactly 20c's. B70_KOLIBRI_ATTN=eager is refused (the verify lists are flash only). Two cards: the taps ride on
  // the hand-off's barrier. At pos = n with x_n pending:
  //   verify(drafts, k)  rows [x_n, d_1 .. d_k] at n .. n + k (device 0's Control::cur_token[1..k] = the drafts, n_active
  //                      k + 1 on both cards; control_hi = {n + 4, k - 3} when k + 1 > 4); row r's argmax in
  //                      verify_ids()[r], its raw fp32 logits row in the last device's verify scratch
  //                      (read_verify_logits); every row's full-layer KV and ring slot written; with a drafter, every row's
  //                      five taps in its taps(). pos is NOT advanced (the last device's Control is put back).
  //   commit(j, t)       j accepted drafts (0 <= j <= k): pos = n + j + 1, pending t, n_active 1 on every card; with a
  //                      drafter, its projection of rows 0..j (positions n .. n + j) into the ring, replayed on the last
  //                      device now (a rejected row's taps never reach the ring). The rejected rows' KV rows and ring slots
  //                      sit past pos: no key bound reads them before a later step rewrites them (the ring's slot of a
  //                      position p is rewritten by p itself; 4096 slots against the 513 window).
  // verify(nullptr, 0) + commit(0, verify_ids()[0]) is one plain token; with the verify lists on, ingest() and generate()
  // run exactly that (a drafter's ring stays filled). Positions: verify(k) needs pos + k + 1 <= max_len (max_verify_k()).
  // attach_drafter: the drafter (runtime::dflash::DSparkDrafter behind TapSink) on the LAST device, borrowed - it must
  // outlive the engine; after enable_spec, before the first verify; its max_k() <= 7, its tap layers < the target's. It
  // captures every verify list (M = 1..8) with the taps, so a missing binary fails there, at startup.
  // reset() resets it; state_bytes / save_state / load_state carry its ring after the target's state (19d's rule);
  // drafter_key() names it; prefill() stages each chunk's tap rows inside its window (drafter_prefill_r0) and projects them
  // 8 a replay.
  void enable_spec();
  bool spec_on() const { return !vb_.empty(); }
  void attach_drafter(TapSink& d);
  uint64_t drafter_key() const { return drafter_ ? drafter_->key() : 0; }
  uint32_t max_verify_k() const;
  void verify(const uint32_t* drafts, uint32_t k);
  const uint32_t* verify_ids() const { return verify_ids_; }
  void read_verify_logits(float* host, uint32_t rows);   // rows <= the last verify's k + 1
  void commit(uint32_t j, uint32_t next_token);
  size_t verify_launches(uint32_t M) const;   // a captured verify list's, every device (0 before its first use)
  // Runs a list that is not the engine's own (the drafter's draft / projection lists) on the LAST device's queue and waits.
  void replay_aux(l0::CmdList& list);
  // What the drafter binds: the last device's context and Control block, the target's lm_head, and device 0's embedding
  // (the CLI copies it into the drafter's replica on the last device).
  l0::Context& context(uint32_t dev) const { return stage_ctx(dev); }
  const l0::Mem& control_mem(uint32_t dev) const;
  const loader::DeviceWeight& lm_head() const;
  const l0::Mem& embedding() const;

 private:
  struct Stage;
  // Spec 20d: a device's pieces for the prefill half (b70_kolibri_prefill), which cannot see Stage.
  l0::Context& stage_ctx(uint32_t dev) const;
  KolibriBuffers& stage_buffers(uint32_t dev) const;
  Control* stage_control(uint32_t dev) const;
  l0::CmdList& stage_imm(uint32_t dev) const;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;
  l0::Mem& snap_mem(uint32_t dev, SnapTensor t) const;   // spec 20e: a snapshot run's allocation
  void settle_all(const char* what);
  void run_verify(uint32_t M);                           // Kolibri DSpark: one verify replay on every device
  void capture_verify(uint32_t M);                       // Kolibri DSpark: device lists at M not captured yet
  void drafter_project_staged(uint32_t pos, uint32_t n);   // Kolibri DSpark: the prefill tail's staged rows, 8 a replay

  loader::KolLoadedModel model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  KolAttn attn_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  [[=gs::Outside("the debug tap, a device each: memory_use and the plan's debug_tap term count it")]]
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  TapAt tap_at_ = TapAt::LayerOut;
  bool broken_ = false, drop_next_ = false;
  std::vector<bool> pending_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  BlockHook block_hook_;   // spec 20d: empty = no hook (uniform chunks)
  // Kolibri DSpark: the verify buffers (one a device), the verify link (two cards), the verify lists [device][M - 1]
  // (captured on first use; declared after the buffers they bind, so destroyed first), the borrowed drafter and its prefill
  // staging (lazy, on the last device), the last verify's (k, pos) a commit must match and its ids, read_logits' row.
  std::vector<std::unique_ptr<KolibriVerifyBuffers>> vb_;
  std::unique_ptr<PipelineLink> vlink_;
  std::vector<std::vector<std::optional<CapturedStep>>> vsteps_;
  TapSink* drafter_ = nullptr;
  [[=gs::Outside("a drafter's prefill tap staging: memory_use counts it in prefill scratch")]]
  std::unique_ptr<l0::Mem> pf_taps_;
  static constexpr uint32_t kNoVerify = 0xFFFFFFFFu;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0;
  uint32_t verify_ids_[kVerifyRows] = {};
  int logits_row_ = -1;   // -1 the decode scratch's row, j the verify scratch's row j (the last commit's)
  // Spec 20d: set by prepare_prefill (b70_kolibri_prefill) - releases a prefill list waiting on a hand-off
  // that will not come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 20d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is
  // destroyed before the buffers it points into. The deleter is set where the state is made
  // (b70_kolibri_prefill), so this header names no prefill symbol. pf_dev_bytes_: each device's prefill
  // scratch + prefill link, for memory_use() (empty = none).
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  std::unique_ptr<KolibriPrefillState, void (*)(KolibriPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::kolibri
