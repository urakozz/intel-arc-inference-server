#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/bolt_loader.h"
#include "runtime/bolt/bolt_buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_engine.h"   // PipelineOptions, PipelineLink (spec 16b's, unchanged)
#include "runtime/tap_drafter.h"      // spec 26c: TapSink (an interface: no link dependency)

// Spec 25c: Nemotron 3.5 Bolt's decode loop - LFM2's / Kolibri's engine (runtime/lfm2moe/lfm2moe_engine.h) for this family:
// it owns the weights (each block on its device: loader::load_bolt's parts), one BoltBuffers and one captured list per
// device, and a token is a replay of the lists. A prompt is fed one id per replay (`ingest`; 25d adds the chunked prefill,
// 25e the snapshots and the server's calls). No MTP (the model has none).
//
// **The list** (runtime/bolt/bolt_capture.h): 653 launches a token on the real model, 645 under B70_BOLT_ATTN=eager, 533
// with every fused arm of B70_BOLT_FUSE (each arm bitwise the as-built list); B70_BOLT_SSM=bf16 keeps the SSM state in bf16
// (Decision 2's arm). All read at construction (the CLI) or given (tests).
// **The profile** (`profile`): the list is built with a kernel-timestamp event per launch (docs/15's instrument,
// runtime::ProfileEvents, capacity 1024); every replay accumulates the per-launch durations and the fence wall, and
// profile_table() prints the anatomy - never a bench list (a profiled launch carries a host-scope flush).
// **Two cards** (spec 16b's pieces, Decision 4 - the real model's default): blocks [0, s) and the embedding on device 0,
// [s, L), norm_f and the head on device 1; the host submits list 0 then list 1 and waits on device 1's fence (bounded:
// PipelineOptions::timeout_ms); the hand-off is h and block s-1's pending slices. The token's way back: device 1's Control is
// copied into device 0's after the fence. A lost hand-off host-signals the event, throws naming it, and marks the engine
// until reset().
namespace runtime::bolt {

// Spec 25d: prefill() and its accessors are DEFINED in b70_bolt_prefill (runtime/bolt/bolt_prefill_engine.cc), not in
// b70_bolt_runtime - LFM2's / Kolibri's arrangement (runtime/lfm2moe/lfm2moe_engine.h): a target that never prefills links
// what it always linked; one that does links b70_bolt_prefill. The prefill state is lazy (made by the first prefill or
// prepare_prefill), so a decode-only engine holds exactly what it did.
struct BoltPrefillState;   // runtime/bolt/bolt_prefill_engine.cc

class BoltEngine {
 public:
  // `devices[i]` is the context of placement device i (device 1 a view of device 0's context); `model` was loaded onto them
  // at `max_len`. debug_tap: the per-block residual copy the golden gate reads (off for speed). profile: ProfileEvents.
  BoltEngine(std::vector<l0::Context*> devices, loader::BoltLoaded model, uint32_t max_len, bool debug_tap = false,
             const PipelineOptions& opt = {}, bool profile = false);
  // The same with an explicit attention form, arm set and SSM state precision (tests; the CLI reads the environment).
  BoltEngine(std::vector<l0::Context*> devices, loader::BoltLoaded model, uint32_t max_len, bool debug_tap,
             const PipelineOptions& opt, bool profile, BoltAttn attn, BoltFuse fuse, BoltState state);
  ~BoltEngine();
  BoltEngine(const BoltEngine&) = delete;
  BoltEngine& operator=(const BoltEngine&) = delete;

  // Zeroes every device's persistent group (control, KV, conv rings, SSM states) and the hand-off buffers; clears a failed
  // hand-off. pos 0.
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is left pending).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t vocab() const { return model_.desc.vocab; }

  // bf16 [layers][hidden] of the last replay: block l's output (its residual after the next site's fold), from the device
  // that folded it (debug_tap).
  std::vector<uint16_t> read_debug_h();
  // u32 [moe blocks][kRouteWords]: every MoE block's route row of the last replay (kernels::bolt::route).
  std::vector<uint32_t> read_routes();
  // bf16 [16][10240]: the mamba_index-th Mamba block's raw-xBC ring (position p at slot p % 16).
  std::vector<uint16_t> read_conv_ring(uint32_t mamba_index);
  // fp32 [128][64][128]: the mamba_index-th Mamba block's SSM state (the bf16 arm's widened).
  std::vector<float> read_ssm_state(uint32_t mamba_index);
  // bf16 [count][2][128] of the attn_index-th attention block's K (or V), positions [first, first + count).
  std::vector<uint16_t> read_kv(uint32_t attn_index, uint32_t first, uint32_t count, bool v);
  // Both residual buffers of device `dev` (bf16 [2][hidden]: h0 then h1) - the ping-pong's state.
  std::vector<uint16_t> read_residuals(uint32_t dev);
  // fp32 [vocab]: the last replay's logits (the last device's) - spec 26c: after a commit(j, t), the verify's row j (the row t
  // came from); after a plain step or a prefill, the decode scratch's.
  std::vector<float> read_logits();
  void read_logits_into(float* host);
  Control control(uint32_t dev) const;

  // --profile: the anatomy of every profiled replay since construction (or clear_profile): per launch family (the kernel
  // and its binary) the in-situ us, launches a step, Σ, the fence wall, the gap per launch. MEASURED, profiled.
  std::string profile_table() const;
  void clear_profile();
  size_t profiled_replays() const { return prof_replays_; }

  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line() const;

  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  size_t launches() const;
  size_t launches(uint32_t dev) const;
  uint32_t devices() const { return uint32_t(st_.size()); }
  uint32_t split() const { return model_.place.split; }
  PpHandoff handoff() const { return opt_.handoff; }
  BoltAttn attention() const { return attn_; }
  BoltFuse fuse() const { return fuse_; }
  BoltState state() const { return state_; }
  const loader::BoltLoaded& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  // What device dev allocated beyond its weights, against runtime::bolt::plan's terms (bolt_decode_test: plan == allocation).
  size_t persistent_bytes(uint32_t dev) const;
  size_t scratch_bytes(uint32_t dev) const;
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that never arrives.
  // Spec 25d: the next prefill CHUNK likewise (device 0's half not appended), unless prefills are replayed.
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 25d: prefill (defined in b70_bolt_prefill) ---------------------------------------------------------------
  // Runs `ids` from the current pos in chunks of at most `chunk` (0 = kPfC 2048; a multiple of 64) through runtime/bolt/
  // bolt_prefill.h's walk on every device, every chunk ending on the absolute 64-grid or at the prompt end
  // (runtime::bolt::prefill_chunk_rows), then the head on the last row: leaves pos += ids.size() on every device and the
  // first generated id pending in cur_token (Engine::prefill's contract); decode continues from the KV, the conv rings and
  // the SSM states it wrote. Those are decode's fill to cosine bars (the GEMMs' sum order, the SSD form), not bitwise (spec
  // 25 §6.2); a prompt cut at multiples of 64 - any chunk width this takes - is bitwise one chunk (Review Focus 1: every
  // kernel is row-local or keyed to absolute positions). A prompt may be continued by another prefill or by decode, and a
  // prefill may follow decode steps (its first chunk then starts off the grid: its first sub-chunk is partial). Two devices
  // (Decision 4): each chunk runs blocks [0, s) on device 0, crosses by a second PipelineLink in copy mode at chunk size
  // (h [C][4096] and block s - 1's pending GEMM rows), then blocks [s, L) and the head on device 1 - sequentially (Kolibri's
  // hand-off). A lost hand-off host-signals the event, throws naming it and marks the engine until reset().
  // B70_BOLT_ATTN=eager runs the eager prefill attention; B70_BOLT_SSM=bf16 the bf16-state scan (rounded once a chunk).
  // With debug_tap, the last chunk's last row of every block lands in the tap (read_debug_h).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every binary the walk
  // and the head bind exists (naming a missing one); prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement (Kolibri's): record each chunk's list once per (pos, rows) and device and replay it;
  // the hand-off's copies and event stay on the immediate lists (never inside a recording; no taps under replay).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  // u32 [layers][kPfC][48]: every MoE block's route rows of the last chunk (rows [0, C) written; pf_route_at), each block's
  // from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // Spec 7's block (runtime::Engine::kBlock) for 25e's prefix cache: with a hook set every chunk also ends at a block end,
  // and the hook is called on the host after each chunk whose end is a multiple of kBlock (devices idle, pos == end_pos,
  // n_active 0 on every device) and once at the prompt end (the first generated id in cur_token). Defined with prefill().
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

  // --- spec 25e: prefix-cache snapshots (spec 7's engine calls; runtime/bolt/bolt_sizes.h's state_runs / kv_runs give both host
  // layouts, one for --pp 1 and --pp 2) ------------------------------------------------------------------------------------------
  // The state is what a session leaves behind besides `pos` and the KV: per Mamba block the SSM state (fp32 or bf16 - the arm
  // B70_BOLT_SSM) and the raw-xBC columns of positions pos - 1 .. pos - 3, written back at their ring slots by load_state (a
  // position below 0 a zero row). state_bytes() is 170,229,760 B (86,343,680 with the bf16 arm) and kv_bytes(n) n x 8,192 B at the
  // real shapes (derived). save_state copies the state of the CURRENT pos (every device idle: the hook runs at a chunk end / the
  // prompt end, a request-end snapshot after the last fence); load_state writes it and sets pos on every device (n_active 0).
  // save_kv / load_kv copy positions [begin, end) of every attention block; begin == end copies nothing; throws unless begin <=
  // end <= max_len. A snapshot is valid only for the SSM state precision it was made with (bf16_state(): the adapter's store
  // key carries it). A restore after a failed hand-off resets first (it rewrites everything a session reads).
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  bool bf16_state() const { return state_ == BoltState::Bf16; }
  void save_state(void* host);
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host);
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // --- spec 26c: speculative decoding (--spec dflash / lookup; spec 26 §5.3-§5.5) ----------------------------------------------
  // enable_spec allocates every device's BoltVerifyBuffers (the verify scratch at 8 rows, the SSM state slots 1..7 of each
  // Mamba block, the union MoE's down rows) and, two cards, the verify link (a copy hand-off sized for 8 rows); the verify lists
  // are captured on first use per M (runtime/bolt/bolt_capture.h build_verify). Before it nothing of this exists: a plain run
  // allocates, captures and replays exactly 25c's (D0). `moe`: the routed experts at M > 1 (B70_BOLT_VERIFY_MOE: union the
  // default, Decision 10). Two cards: the copy hand-off only (peer is refused by name - the taps ride on its barrier).
  // At pos = n with x_n pending:
  //   verify(drafts, k)  rows [x_n, d_1 .. d_k] at n .. n + k (device 0's Control::cur_token[1..k] = the drafts, n_active
  //                      k + 1 on both cards); row r's argmax in verify_ids()[r], its raw fp32 logits row in the last
  //                      device's verify scratch (read_verify_logits); every row's KV rows, ring columns and SSM state slot
  //                      (live + r) % 8 written; with a drafter, every row's six taps in its taps() (taps 1 / 18 / 35 copied
  //                      by device 0 before the hand-off). pos is NOT advanced (the last device's Control is put back).
  //   commit(j, t)       j accepted drafts (0 <= j <= k): pos = n + j + 1, pending t, n_active 1, Control::gdn_live = (live
  //                      + j) % 8 on both cards - the live slot then holds the state after row j (Review Focus 1); with a
  //                      drafter, its projection of rows 0..j (positions n .. n + j) into the ring and the anchor, replayed on
  //                      the last device now (Review Focus 3: a rejected row's taps never reach the ring). The rejected rows'
  //                      KV rows, ring columns and slots are rewritten before anything reads them.
  // verify(nullptr, 0) + commit(0, verify_ids()[0]) is one plain token; with the verify lists on, ingest() and generate() run
  // exactly that (the slots stay the state's home and a drafter's ring stays filled). A plain list (25c's, in place on slot 0)
  // and every prefill / snapshot first move the live slot back to slot 0 (canonical_ssm: one copy a Mamba block when live != 0).
  // Positions: verify(k) needs pos + k + 1 <= max_len (max_verify_k()).
  // attach_drafter: the drafter (runtime::dflash::LiliDrafter behind TapSink) on the LAST device, borrowed - it must outlive the
  // engine; after enable_spec, before the first verify; its max_k() <= 7, its tap blocks < the target's. reset() resets it;
  // state_bytes / save_state / load_state carry its ring and anchor after the target's state (spec 26 §6, 19d's rule);
  // drafter_key() names it; prefill() stages each chunk's tap rows inside its window (lili_prefill_r0: Decision 7) and projects
  // them 8 a replay (the GEMM path is not built: the replay fallback, plan 26c Task 5).
  void enable_spec(VerifyMoe moe);
  bool spec_on() const { return !vb_.empty(); }
  VerifyMoe verify_moe() const { return vmoe_; }
  void attach_drafter(TapSink& d);
  bool has_drafter() const { return drafter_ != nullptr; }
  TapSink* drafter() const { return drafter_; }
  uint64_t drafter_key() const { return drafter_ ? drafter_->key() : 0; }
  uint32_t max_verify_k() const;
  void verify(const uint32_t* drafts, uint32_t k);
  const uint32_t* verify_ids() const { return verify_ids_; }
  void read_verify_logits(float* host, uint32_t rows);   // rows <= the last verify's k + 1
  void commit(uint32_t j, uint32_t next_token);
  uint32_t ssm_live() const;   // Control::gdn_live: the live SSM slot (0 without the verify lists)
  size_t verify_launches(uint32_t M) const;   // a captured verify list's, every device (0 before its first use)
  // Runs a list that is not the engine's own (the drafter's draft / projection lists) on the LAST device's queue and waits.
  void replay_aux(l0::CmdList& list);
  // What the drafter binds: the last device's context and Control block, the target's lm_head (the drafter's embedding is
  // its replica, the CLI's).
  l0::Context& context(uint32_t dev) const { return stage_ctx(dev); }
  const l0::Mem& control_mem(uint32_t dev) const;
  const loader::DeviceWeight& lm_head() const;
  // fp32 [128][64][128]: the mamba_index-th Mamba block's state in slot `slot` (0 the block's own, 1..7 the verify copies) -
  // the tests' (bolt_commit_test); read_ssm_state reads the LIVE slot.
  std::vector<float> read_ssm_slot(uint32_t mamba_index, uint32_t slot);

 private:
  void settle_all(const char* what);                   // spec 25e: every device idle, or throws naming `what`
  void canonical_ssm();                                // spec 26c: the live slot back into slot 0 (live 0), devices idle
  void run_verify(uint32_t M);                         // spec 26c: one verify replay on every device (captured on first use)
  void drafter_project_staged(uint32_t pos, uint32_t n);   // spec 26c: the prefill tail's staged rows, 8 a replay
  l0::Mem& snap_mem(uint32_t dev, int tensor) const;   // spec 25e: a SnapTensor's device allocation
  struct Stage;
  // Spec 25d: a device's pieces for the prefill half (b70_bolt_prefill), which cannot see Stage.
  l0::Context& stage_ctx(uint32_t dev) const;
  BoltBuffers& stage_buffers(uint32_t dev) const;
  Control* stage_control(uint32_t dev) const;
  l0::CmdList& stage_imm(uint32_t dev) const;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;
  void accumulate_profile(double fence_us);

  loader::BoltLoaded model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  BoltAttn attn_;
  BoltFuse fuse_;
  BoltState state_;
  bool profile_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  bool broken_ = false, drop_next_ = false;
  std::vector<bool> pending_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  // --profile: per label (the launch's kernel and binary, its block stripped) Σ us and launches a step; the walls.
  std::map<std::string, std::pair<double, size_t>> prof_us_;
  std::vector<double> prof_wall_us_, prof_sum_us_;
  size_t prof_replays_ = 0;
  BlockHook block_hook_;   // spec 25d (for 25e): empty = no hook
  // Spec 26c: the verify buffers (one a device), the verify link (two cards), the verify lists [device][M - 1] (captured on
  // first use; declared after the buffers they bind, so destroyed first), the routed experts' form, the borrowed drafter and
  // its prefill staging (lazy, on the last device), the last verify's (k, pos) a commit must match and its ids.
  std::vector<std::unique_ptr<BoltVerifyBuffers>> vb_;
  std::unique_ptr<PipelineLink> vlink_;
  std::vector<std::vector<std::optional<CapturedStep>>> vsteps_;
  VerifyMoe vmoe_ = VerifyMoe::Union;
  TapSink* drafter_ = nullptr;
  std::unique_ptr<l0::Mem> pf_taps_;
  static constexpr uint32_t kNoVerify = 0xFFFFFFFFu;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0;
  uint32_t verify_ids_[kVerifyRows] = {};
  int logits_row_ = -1;   // read_logits' row: -1 the decode scratch's, j the verify scratch's row j (the last commit's)
  // Spec 25d: set by prepare_prefill (b70_bolt_prefill) - releases a prefill list waiting on a hand-off that will not come
  // and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 25d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is destroyed before
  // the buffers it points into. The deleter is set where the state is made (b70_bolt_prefill), so this header names no
  // prefill symbol. pf_dev_bytes_: each device's prefill scratch + prefill link, for memory_use().
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  int pf_replay_ = -1;   // -1: B70_PREFILL_REPLAY decides
  std::unique_ptr<BoltPrefillState, void (*)(BoltPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::bolt
