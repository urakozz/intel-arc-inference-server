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

 private:
  struct Stage;
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
};

}  // namespace runtime::kolibri
