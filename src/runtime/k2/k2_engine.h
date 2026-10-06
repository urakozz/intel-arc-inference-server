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
// (the host writes the ingested id, submits, waits, reads the sampled id back). A prompt is
// fed one id per replay (`ingest`) or, spec 18c, in chunks (`prefill`). No MTP (K2 has no
// head, spec 18 §2). The KV cache is bf16 or, spec 18e, int8 (rotkv at head_dim 128).
//
// Spec 18c: prefill() and its accessors are DEFINED in b70_k2_prefill
// (runtime/k2/k2_prefill_engine.cc), not in b70_k2_runtime - Engine::prefill's arrangement
// (runtime/prefill/engine_prefill.cc): a target that never prefills links what it always
// linked; one that does links b70_k2_prefill. The prefill state is lazy (allocated by the
// first prefill or prepare_prefill), so a decode-only engine holds exactly what it did.
namespace runtime::k2 {

struct K2PrefillState;   // runtime/k2/k2_prefill_engine.cc

class K2Engine {
 public:
  // `max_len` must be the one the model was loaded with (its RoPE table). debug_tap: the
  // per-layer resid copy the golden gate reads (48 copies a token - off for speed). `kv`
  // (spec 18e): the KV cache's form, default runtime::default_kv_cache() (B70_KV_CACHE, unset =
  // bf16) - the decode list and the prefill walk bind the form's binaries.
  K2Engine(l0::Context& ctx, loader::K2LoadedModel model, uint32_t max_len, bool debug_tap = false,
           KvCache kv = default_kv_cache());

  // Zeroes the persistent group (control, kv_k, kv_v); scratch is never zeroed (no step
  // reads scratch it has not written first; the replay test's second run is the proof).
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the
  // (n+1)-th is left pending in cur_token).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});

  // --- spec 18c: prefill (defined in b70_k2_prefill) ------------------------------------
  // The prompt in chunks of `chunk` positions (0 = kPfC) from the current pos: every layer's
  // K / V rows written, then the head on the last row - cur_token[0] holds the first
  // generated id and pos = old pos + ids.size(), exactly what ingest(ids) leaves (to the
  // GEMMs' rounding: prefill's KV is decode's to a cosine bar, spec 18c Review Focus 1).
  // A prompt may be continued by another prefill (no state but the KV and pos). The L0
  // backend only (K2's hidden 2560 is not whole 1024-k Hadamard blocks: no h8).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates the prefill scratch and the prefill Context, and checks every binary the walk
  // binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement (engine_prefill.cc): record each chunk's list once per
  // (pos, rows, attention variant) and replay it. Off by default (unset = the environment).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches the prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  // The last chunk's route rows, u32 [layers][2 (MoVA, MoE)][kPfC][32] (k2_sizes.h
  // pf_route_at); rows [0, C) of the last chunk are written. Throws before the first prefill.
  std::vector<uint32_t> read_prefill_routes();
  bool prefill_ready() const { return pf_bytes_ != 0; }

  // --- spec 18d: the prefix cache's engine calls (spec 7, runtime::Engine's contract) -----
  // K2 has no recurrent state, so a snapshot is KV only (runtime/k2/k2_sizes.h
  // kv_snapshot_runs): state_bytes() is 0, save_state copies nothing and load_state restores
  // `pos` alone (control.pos = pos, n_active = 0; cur_token is NOT restored - prefill at least
  // one id after a restore, spec 7 §3.3 step 4). save_kv / load_kv copy positions [begin, end)
  // of every layer's K and V (and at int8 their scales) in kv_snapshot_runs' host layout,
  // kv_bytes(end - begin) bytes; begin == end copies nothing; throws unless begin <= end <=
  // max_len. Every call is blocking on `imm_`, outside the decode list and the prefill
  // Context; the caller has no replay or prefill in flight (no public call leaves one).
  static constexpr uint32_t kBlock = 2048;   // runtime::Engine::kBlock: the store's block
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  size_t state_bytes() const { return 0; }
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host) const;
  void load_state(const void* host, uint32_t pos);
  void save_kv(uint32_t begin, uint32_t end, void* host) const;
  void load_kv(uint32_t begin, uint32_t end, const void* host);
  // runtime::Engine::set_block_hook's contract, on K2Engine::prefill: called on the host after
  // every chunk whose end is a multiple of kBlock (device idle, control.pos == end_pos,
  // n_active == 0) and after the prompt's last chunk once the first generated id is in
  // cur_token; with a hook every chunk ends at a block end or at the prompt end
  // (runtime::prefill_chunk_rows). Without one, chunking is exactly as before. A hook that
  // throws propagates with pos == end_pos. ingest() (one replay per id) never calls it.
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }
  // Host sampling (b70-serve, spec 18d): the id the next generate() embeds - cur_token[0], the
  // ingest protocol Control already has - replacing the argmax the last replay wrote.
  void set_pending(uint32_t id);
  // The last replay's logits row into `host` (vocab floats) - read_logits() without the
  // allocation, for the sampler's per-token readback.
  void read_logits_into(float* host);

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
  // included), kv, decode state = control + scratch (+ the tap), prefill scratch once a
  // prefill was prepared (spec 18c; runtime::k2::prefill_sizes), no int8.
  MemoryComponents memory_use() const;
  std::string memory_line() const;

  const CapturedStep& step() const { return step_; }
  K2Buffers& buffers() { return buffers_; }
  const loader::K2LoadedModel& model() const { return model_; }
  uint32_t pos() const { return control_->pos; }
  uint32_t max_len() const { return buffers_.max_len; }
  uint32_t vocab() const { return model_.desc->vocab; }   // the logits row's length (250,624)
  KvCache kv_cache() const { return buffers_.kv_cache(); }   // spec 18e
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
  BlockHook block_hook_;   // spec 18d: empty = no hook (prefill chunks as before)
  // Spec 18c: the prefill state, lazy; declared LAST so it (its Context, kernels and
  // recordings, which name the buffers and the weights) is destroyed first. The deleter is
  // set where the state is made (b70_k2_prefill), so this header names no prefill symbol.
  std::unique_ptr<K2PrefillState, void (*)(K2PrefillState*)> pf_{nullptr, nullptr};
  size_t pf_bytes_ = 0;   // the prefill scratch's bytes, for memory_use() (0 = none)
  int pf_replay_ = -1;    // -1: B70_PREFILL_REPLAY decides
};

}  // namespace runtime::k2
