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

// Spec 20c: Kolibri-1's decode loop - runtime::Engine's contract (K2Engine's arrangement) on one card
// or, spec 16b's pieces, two: it owns the weights (each layer on its device: loader::load_kolibri1's
// parts), one KolibriBuffers and one captured list per device, and a token is a replay of the lists.
// A prompt is fed one id per replay (`ingest`; spec 20d prefills in chunks). No MTP (Kolibri has no
// head), no prefill, no snapshots (spec 20e: the ring is part of the snapshot).
//
// **Two cards** (spec 16 §8's rules, PipelineEngine's, here for a model PipelineEngine cannot run):
// layers [0, s) on device 0 with the embedding, [s, L) plus the head on device 1 (KolPlacement). The
// host submits list 0 then list 1 and waits on device 1's fence (bounded: PipelineOptions::timeout_ms);
// the lists order themselves through the hand-off (`copy`: a device-to-device copy and a cross-device
// event; `peer`: pp_send / pp_recv). **The token's way back:** argmax on device 1 writes the id and
// advances pos in device 1's Control; after the fence the host copies device 1's Control block into
// device 0's. set_token writes both. **Bounded:** a lost hand-off host-signals the event (so the queue
// drains), throws naming it, and marks the engine until reset().
namespace runtime::kolibri {

// Spec 20d: prefill() and its accessors are DEFINED in b70_kolibri_prefill
// (runtime/kolibri/kolibri_prefill_engine.cc), not in b70_kolibri_runtime - K2Engine's arrangement
// (runtime/k2/k2_engine.h): a target that never prefills links what it always linked; one that does links
// b70_kolibri_prefill. The prefill state is lazy (allocated by the first prefill or prepare_prefill), so a
// decode-only engine holds exactly what it did.
struct KolibriPrefillState;   // runtime/kolibri/kolibri_prefill_engine.cc

class KolibriEngine {
 public:
  // `devices[i]` is the context of placement device i (device 1 a view of device 0's context,
  // l0::Context(const Context&, 1)); `model` was loaded onto them at `max_len`. debug_tap: the
  // per-layer resid copy the partial-forward gate reads (off for speed). Two devices: throws when
  // device 0 cannot access device 1's memory (zeDeviceCanAccessPeer) - both hand-offs write it.
  KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len,
                bool debug_tap = false, const PipelineOptions& opt = {});
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

  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  // bf16 [layers][hidden] of the last replay, each layer's row from the device holding it (debug_tap).
  std::vector<uint16_t> read_debug_resid();
  // u32 [layers][32]: every layer's route row of the last replay (kernels::kolibri::route).
  std::vector<uint32_t> read_routes();
  // fp32 [vocab]: the last replay's logits (the last device's).
  std::vector<float> read_logits();
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
  PpHandoff handoff() const { return opt_.handoff; }
  KolAttn attention() const { return attn_; }
  const loader::KolLoadedModel& model() const { return model_; }
  const CapturedStep& step(uint32_t dev) const;
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that
  // never arrives. The step must throw within the bounds, never hang.
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 20d: prefill (defined in b70_kolibri_prefill) ---------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = kPfC 2048) through runtime/kolibri/
  // kolibri_prefill.h's walk on every device, then the head on the last row: leaves pos += ids.size() on
  // every device and the first generated id pending in cur_token (Engine::prefill's contract). The KV and
  // ring rows it writes are decode's to a cosine bar (the GEMMs' sum order), not bitwise (plan 20d Review
  // Focus 4). A prompt may be continued by another prefill. Two devices: each chunk runs layers [0, s) on
  // device 0, crosses by spec 16b's `copy` hand-off at chunk size (a second PipelineLink; the chunk always
  // crosses by copy, whatever --pipeline-handoff says for decode: pp_handoff.cl's peer kernels are one
  // 256-lane work-group sized for a 5 KB row, not 10.5 MB), then layers [s, L) on device 1 - sequentially
  // (spec 16c's overlapped chunk pipeline is a later lever). A lost hand-off host-signals the event, throws
  // naming it and marks the engine until reset(). B70_KOLIBRI_ATTN=eager runs the eager prefill attention
  // too (the engine's attention(), read once at construction).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every
  // binary the walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement (K2's): record each chunk's list once per (pos, rows) and device and
  // replay it; the hand-off's copies and event stay on the immediate lists (never inside a recording).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  // u32 [layers][kPfC][32]: every layer's route rows of the last chunk (rows [0, C) written; pf_route_at),
  // each layer's from the device holding it. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // Spec 7's block (runtime::Engine::kBlock); spec 20e's prefix cache hooks into the chunk walk: with a hook
  // set every chunk ends at a block end or at the prompt end (runtime::prefill_chunk_rows), and the hook is
  // called on the host after each chunk whose end is a multiple of kBlock (devices idle, pos == end_pos,
  // n_active 0 on every device) and once at the prompt end (the first generated id in cur_token).
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

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

  loader::KolLoadedModel model_;
  uint32_t max_len_;
  PipelineOptions opt_;
  KolAttn attn_;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  bool broken_ = false, drop_next_ = false;
  std::vector<bool> pending_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  BlockHook block_hook_;   // spec 20d: empty = no hook (uniform chunks)
  // Spec 20d: set by prepare_prefill (b70_kolibri_prefill) - releases a prefill list waiting on a hand-off
  // that will not come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 20d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is
  // destroyed before the buffers it points into. The deleter is set where the state is made
  // (b70_kolibri_prefill), so this header names no prefill symbol. pf_dev_bytes_: each device's prefill
  // scratch + prefill link, for memory_use() (empty = none).
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  int pf_replay_ = -1;   // -1: B70_PREFILL_REPLAY decides
  std::unique_ptr<KolibriPrefillState, void (*)(KolibriPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::kolibri
