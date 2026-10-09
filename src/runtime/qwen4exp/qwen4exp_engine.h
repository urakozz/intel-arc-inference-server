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
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_engine.h"   // PipelineOptions, PipelineLink (spec 16b's, unchanged)
#include "runtime/qwen4exp/qwen4exp_buffers.h"

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

 private:
  struct Stage;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;
  void build_injected();
  void ple_init();
  CapturedStep& active(uint32_t dev) const;

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
};

}  // namespace runtime::qwen4exp
