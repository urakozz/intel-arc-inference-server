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
  // fp32 [vocab]: the last replay's logits (the last device's).
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
  void drop_next_handoff() { drop_next_ = true; }

 private:
  struct Stage;
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
};

}  // namespace runtime::bolt
