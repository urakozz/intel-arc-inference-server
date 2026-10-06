#pragma once
#include <array>
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
#include "l0/sync_event.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"

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
// Not here (spec 16b scope): prefill (16c - b70-decode ingests through the decode lists),
// MTP and the server (16d), the residual tap, profiling. Snapshots (spec 7) ARE here, in
// the single-card host layout, so a snapshot moves between --pp 1 and 2 unchanged.
namespace runtime {

struct PipelineOptions {
  PpHandoff handoff = kDefaultPpHandoff;
  uint32_t timeout_ms = 30000;      // one step's fence wait; a Qwen3.8 step is ~36 ms
  uint32_t spin_limit = 1u << 24;   // peer: pp_recv's flag loads before it gives up
};

// The hand-off buffers (pipeline_plan.h: pp_landing_layout).
struct PipelineLink {
  PipelineLink(l0::Context& d0, l0::Context& d1, const model::ModelDesc& d, PpHandoff mode);
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

  // Spec 7, in Engine's host layouts exactly (no MTP): save_state is every GDN layer's state
  // then every conv ring - device 0's slices then device 1's, which IS the single-card
  // order - and save_kv per tensor (K, V) every layer's rows [then at int8 every layer's
  // scales], device 0's layers first. load_state sets pos in both Controls.
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host) const;
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host) const;
  void load_kv(uint32_t begin, uint32_t end, const void* host);

  // Device 1's logits row 0 after the last step (fp32 [kVocab]) - P1's comparison.
  std::vector<float> read_logits() const;

  // Spec 6's memory line per device ("memory, device 0: model ..., total ... of ...").
  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line(uint32_t dev) const;

  // P4's test hook: the next step submits device 1's list only - a hand-off that never
  // arrives. The step must throw within the bounds, never hang.
  void drop_next_handoff() { drop_next_ = true; }

 private:
  struct Stage;
  void step_once();
  // Waits (bounded) for device `dev`'s outstanding step, if any; true when none is left.
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);
  Stage& st(uint32_t dev) const;

  uint32_t max_len_;
  PipelineOptions opt_;
  KvCache kv_;
  std::array<std::unique_ptr<Stage>, kPpDevices> stages_;
  std::unique_ptr<PipelineLink> link_;
  bool broken_ = false, drop_next_ = false;
  std::array<bool, kPpDevices> pending_{};   // a submitted step whose fence is not yet seen
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
