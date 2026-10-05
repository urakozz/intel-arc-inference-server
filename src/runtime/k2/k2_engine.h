#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/k2_loader.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/k2/k2_buffers.h"
#include "runtime/memory_plan.h"

// Spec 18b: K2-Horizon's decode loop - runtime::Engine's contract (engine.h) for K2: it owns
// the weights, the buffers and the one captured list, and a token is a replay of that list
// (the host writes the ingested id, submits, waits, reads the sampled id back). Decode only:
// a prompt is fed one id per replay (`ingest`); prefill is spec 18c and is refused with that
// name by the CLIs. No MTP (K2 has no head, spec 18 §2), bf16 KV only (int8 KV is 18e).
namespace runtime::k2 {

class K2Engine {
 public:
  // `max_len` must be the one the model was loaded with (its RoPE table). debug_tap: the
  // per-layer resid copy the golden gate reads (48 copies a token - off for speed).
  K2Engine(l0::Context& ctx, loader::K2LoadedModel model, uint32_t max_len, bool debug_tap = false);

  // Zeroes the persistent group (control, kv_k, kv_v); scratch is never zeroed (no step
  // reads scratch it has not written first; the replay test's second run is the proof).
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the
  // (n+1)-th is left pending in cur_token).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});

  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  // The per-layer tap of the last replay, bf16 [layers][hidden]; throws unless debug_tap.
  std::vector<uint16_t> read_debug_resid();
  // Every layer's route rows of the last replay: u32 [layers][2 (MoVA, MoE)][32]
  // (k2_sizes.h; kernels::k2::route has the word layout). Dense layers' rows are not written.
  std::vector<uint32_t> read_routes();
  // The last replay's logits row, fp32 [vocab].
  std::vector<float> read_logits();

  // memory_line()'s five figures (runtime::format_memory): model = the loader's total (RoPE
  // included), kv, decode state = control + scratch (+ the tap), no prefill, no int8.
  MemoryComponents memory_use() const;
  std::string memory_line() const;

  const CapturedStep& step() const { return step_; }
  K2Buffers& buffers() { return buffers_; }
  const loader::K2LoadedModel& model() const { return model_; }
  uint32_t pos() const { return control_->pos; }
  uint32_t max_len() const { return buffers_.max_len; }
  l0::Context& context() const { return ctx_; }

 private:
  void replay();

  l0::Context& ctx_;
  loader::K2LoadedModel model_;
  K2Buffers buffers_;
  std::unique_ptr<l0::Mem> tap_;
  CapturedStep step_;
  l0::Queue queue_;
  l0::Fence fence_;
  mutable l0::CmdList imm_;
  Control* control_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime::k2
