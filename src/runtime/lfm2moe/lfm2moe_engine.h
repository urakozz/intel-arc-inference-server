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
#include "loader/lfm2moe_loader.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/lfm2moe/lfm2moe_buffers.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_engine.h"   // PipelineOptions, PipelineLink (spec 16b's, unchanged)

// Spec 24c: LFM2.5-24B-A2B's decode loop - Kolibri's engine (runtime/kolibri/kolibri_engine.h, runtime::Engine's
// contract) for this family: it owns the weights (each layer on its device: loader::load_lfm2moe's parts), one
// Lfm2MoeBuffers and one captured list per device, and a token is a replay of the lists. A prompt is fed one id per
// replay (`ingest`) or, spec 24d, in chunks (`prefill`, below). No MTP (the model has no head). 24e adds the snapshots
// and the server's calls.
//
// **The list** (runtime/lfm2moe/lfm2moe_capture.h): 383 launches a token on one card, 403 under B70_LFM_ATTN=eager,
// 277 with every fused arm of B70_LFM_FUSE (each arm bitwise the as-built list); both read at construction.
// **The profile** (`profile`): the list is built with a kernel-timestamp event per launch (docs/15's instrument,
// runtime::ProfileEvents); every replay accumulates the per-launch durations and the fence wall, and
// profile_table() prints the anatomy - never a bench list (a profiled launch carries a host-scope flush).
// **Two cards** (spec 16b's pieces, decision 10): layers [0, s) and the embedding on device 0, [s, L) and the head on
// device 1; the host submits list 0 then list 1 and waits on device 1's fence (bounded: PipelineOptions::timeout_ms);
// the hand-off is h (and, at a dense cut, the pending slices). The token's way back: device 1's Control is copied into
// device 0's after the fence. A lost hand-off host-signals the event, throws naming it, and marks the engine until
// reset().
namespace runtime::lfm2moe {

// Spec 24d: prefill() and its accessors are DEFINED in b70_lfm2moe_prefill (runtime/lfm2moe/lfm2moe_prefill_engine.cc),
// not in b70_lfm2moe_runtime - Kolibri's arrangement (runtime/kolibri/kolibri_engine.h): a target that never prefills
// links what it always linked; one that does links b70_lfm2moe_prefill. The prefill state is lazy (allocated by the
// first prefill or prepare_prefill), so a decode-only engine holds exactly what it did.
struct Lfm2MoePrefillState;   // runtime/lfm2moe/lfm2moe_prefill_engine.cc

class Lfm2MoeEngine {
 public:
  // `devices[i]` is the context of placement device i (device 1 a view of device 0's context); `model` was loaded
  // onto them at `max_len` - with conv_tiles exactly when B70_LFM_FUSE has the arm conv. debug_tap: the per-layer
  // residual copy the golden gate reads (off for speed). profile: the list built with ProfileEvents.
  Lfm2MoeEngine(std::vector<l0::Context*> devices, loader::LfmLoaded model, uint32_t max_len, bool debug_tap = false,
                const PipelineOptions& opt = {}, bool profile = false);
  // The same with an explicit attention form and arm set (tests; the CLI reads the environment).
  Lfm2MoeEngine(std::vector<l0::Context*> devices, loader::LfmLoaded model, uint32_t max_len, bool debug_tap,
                const PipelineOptions& opt, bool profile, LfmAttn attn, LfmFuse fuse);
  ~Lfm2MoeEngine();
  Lfm2MoeEngine(const Lfm2MoeEngine&) = delete;
  Lfm2MoeEngine& operator=(const Lfm2MoeEngine&) = delete;

  // Zeroes every device's persistent group (control, KV, conv rings) and the hand-off buffers; clears a failed
  // hand-off. pos 0: the conv rings need no clearing (positions < 0 read 0).
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is left pending).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t vocab() const { return model_.desc.vocab; }

  // bf16 [layers][hidden] of the last replay, each layer's row from the device holding it (debug_tap).
  std::vector<uint16_t> read_debug_h();
  // u32 [moe layers][kRouteWords]: every MoE layer's route row of the last replay (kernels::lfm2moe::route).
  std::vector<uint32_t> read_routes();
  // bf16 [16][hidden]: the conv_index-th conv layer's ring (Bx at pos % 16), from the device holding it.
  std::vector<uint16_t> read_conv_ring(uint32_t conv_index);
  // bf16 [count][kv_heads][64] of the attn_index-th attention layer's K (or V), positions [first, first + count).
  std::vector<uint16_t> read_kv(uint32_t attn_index, uint32_t first, uint32_t count, bool v);
  // Both residual buffers of device `dev` (bf16 [2][hidden]: h0 then h1) - the ping-pong's state.
  std::vector<uint16_t> read_residuals(uint32_t dev);
  // fp32 [vocab]: the last replay's logits (the last device's).
  std::vector<float> read_logits();
  void read_logits_into(float* host);
  Control control(uint32_t dev) const;

  // --profile: the anatomy of every profiled replay since construction (or clear_profile): per launch family (the
  // kernel and its binary) the in-situ us, launches a step, Σ, the fence wall, the gap per launch. MEASURED, profiled.
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
  LfmAttn attention() const { return attn_; }
  LfmFuse fuse() const { return fuse_; }
  const loader::LfmLoaded& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that never arrives.
  // Spec 24d: the next prefill CHUNK likewise (device 0's half not appended), unless prefills are replayed.
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 24d: prefill (defined in b70_lfm2moe_prefill) ---------------------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = kPfC 2048) through runtime/lfm2moe/lfm2moe_prefill.h's
  // walk on every device, then the head on the last row: leaves pos += ids.size() on every device and the first
  // generated id pending in cur_token (Engine::prefill's contract); decode continues from the KV and the conv rings it
  // wrote. The conv rings, KV, routes and logits are decode's to a cosine bar (the GEMMs' sum order), not bitwise
  // (plan 24d Review Focus 2); a prompt split into chunks at any offset is bitwise one chunk (Review Focus 1: every
  // kernel is row-local or keyed to ABSOLUTE positions - the GEMM rows, the conv's per-row chain with its seed, the
  // sorted MoE rows, the flash tiles from key 0, whose wholly masked tiles add exact zeros, RoPE, the ring slots). A
  // prompt may be continued by another prefill or by decode, and a prefill
  // may follow decode steps. Two devices (decision 10): each chunk runs layers [0, s) on device 0, crosses by spec
  // 16b's `copy` hand-off at chunk size (a second PipelineLink: h [C][2048], and at a dense cut the down's pending
  // rows), then layers [s, L) and the head on device 1 - sequentially. A lost hand-off host-signals the event, throws
  // naming it and marks the engine until reset(). B70_LFM_ATTN=eager runs the eager prefill attention too (the
  // engine's attention(), read at construction); B70_LFM_FUSE's arm conv reads in_proj in the conv-tile order (the
  // _T conv build). With debug_tap, the last chunk's last row of every layer lands in the tap (read_debug_h).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every binary the
  // walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement (Kolibri's): record each chunk's list once per (pos, rows) and device and replay
  // it; the hand-off's copies and event stay on the immediate lists (never inside a recording; no taps under replay).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  // u32 [layers][kPfC][16]: every MoE layer's route rows of the last chunk (rows [0, C) written; pf_route_at), each
  // layer's from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  bool prefill_ready() const { return pf_bytes_ != 0; }

 private:
  struct Stage;
  // Spec 24d: a device's pieces for the prefill half (b70_lfm2moe_prefill), which cannot see Stage.
  l0::Context& stage_ctx(uint32_t dev) const;
  Lfm2MoeBuffers& stage_buffers(uint32_t dev) const;
  Control* stage_control(uint32_t dev) const;
  l0::CmdList& stage_imm(uint32_t dev) const;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;
  void accumulate_profile(double fence_us);

  loader::LfmLoaded model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  LfmAttn attn_;
  LfmFuse fuse_;
  bool profile_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  bool broken_ = false, drop_next_ = false;
  std::vector<bool> pending_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  // --profile: per label (the launch's kernel and binary, its layer stripped) Σ us and launches a step; the walls.
  std::map<std::string, std::pair<double, size_t>> prof_us_;
  std::vector<double> prof_wall_us_, prof_sum_us_;
  size_t prof_replays_ = 0;
  // Spec 24d: set by prepare_prefill (b70_lfm2moe_prefill) - releases a prefill list waiting on a hand-off that will
  // not come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 24d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is destroyed
  // before the buffers it points into. The deleter is set where the state is made (b70_lfm2moe_prefill), so this
  // header names no prefill symbol. pf_dev_bytes_: each device's prefill scratch + prefill link, for memory_use().
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  int pf_replay_ = -1;   // -1: B70_PREFILL_REPLAY decides
  std::unique_ptr<Lfm2MoePrefillState, void (*)(Lfm2MoePrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::lfm2moe
