#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/muse_loader.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/muse/muse_buffers.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/pipeline_engine.h"   // PipelineLink (spec 16b's, unchanged)

// Spec 23c: Muse-Glimmer-30B's decode loop - runtime::Engine's contract (KolibriEngine's arrangement) on one card
// (the default, spec 23 decision 7) or, spec 16b's pieces, two: it plans (runtime::muse::plan_muse, the decode
// scratch reserved), loads (loader::load_muse_glimmer: each layer straight onto its device), and owns one MuseBuffers
// and one captured list per device; a token is a replay of the lists. A prompt is fed one id per replay (`ingest`);
// the chunked prefill is spec 23d's.
//
// **Two cards** (spec 16 §8's rules, Kolibri's): layers [0, s) on device 0 with the embedding, [s, L) plus the head
// on device 1. The host submits list 0 then list 1 and waits on device 1's fence (bounded: MuseOptions::timeout_ms);
// the lists order themselves through the hand-off (`copy`: a device-to-device copy and a cross-device event; `peer`:
// pp_send / pp_recv). The token's way back: argmax on device 1 writes the id and advances pos in device 1's Control;
// after the fence the host copies device 1's Control block into device 0's. set_token writes both. A lost hand-off
// host-signals the event (so the queue drains), throws naming device 1, and marks the engine until reset().
//
// **Spec 23d: prefill() and its accessors are DEFINED in b70_muse_prefill** (runtime/muse/muse_prefill_engine.cc), not
// in b70_muse_runtime - Kolibri's arrangement (runtime/kolibri/kolibri_engine.h): a target that never prefills links
// what it always linked; one that does links b70_muse_prefill. The prefill state is lazy (allocated by the first
// prefill or prepare_prefill), so a decode-only engine holds exactly what it did.
namespace runtime::muse {

struct MusePrefillState;   // runtime/muse/muse_prefill_engine.cc

struct MuseOptions {
  uint32_t max_len = 0;                  // 0 = auto: the largest that fits every device, at most 131072 (decision 6)
  bool int8_head = true;                 // spec 9's int8 lm_head (the default) or the bf16 one
  PpHandoff handoff = PpHandoff::Copy;   // --pipeline-handoff (two devices)
  uint32_t split = 0;                    // two devices: the first layer of device 1; 0 = by bytes (pp_balance)
  uint32_t layers = 0;                   // development mode: the checkpoint's first N layers (0: all)
  bool debug_tap = false;                // the per-layer residual tap (MG3's partial forward)
  uint32_t timeout_ms = 30000;           // a step's fence wait (two devices)
  uint32_t spin_limit = 1u << 24;        // peer: pp_recv's flag loads before it gives up
  bool prefill = false;                  // spec 23d: plan the prefill scratch (and its link) into max_len and the fit
  uint32_t prefill_timeout_ms = 30000;   // spec 23d: a prefill chunk's / the head's wait (bounded; under eager per
                                         // 2048 keys of the chunk's deepest row - runtime::muse::prefill_wait_ms)
};

class MuseEngine {
 public:
  // `ctx[i]` is the context of placement device i (device 1 a view of device 0's context, l0::Context(const
  // Context&, 1)). Plans before the first allocation: max_len (auto or checked against 131072 - decision 6 -, the
  // decode scratch reserved), the split; the load prints its report to stdout (the CLI redirects it). Throws when
  // the plan does not fit, naming the bytes; two devices: when device 0 cannot access device 1's memory.
  MuseEngine(std::vector<l0::Context*> ctx, const std::string& snapshot, const MuseOptions& o);
  ~MuseEngine();
  MuseEngine(const MuseEngine&) = delete;
  MuseEngine& operator=(const MuseEngine&) = delete;

  // Zeroes every device's persistent group (control, full KV, rings) and the hand-off buffers; clears a failed
  // hand-off.
  void reset();
  // One replay per id; ids must be < vocab. pos advances by ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids (Engine::generate's contract: id i is the previous fence's, the (n+1)-th is left pending
  // in cur_token). Greedy is the device argmax over the RAW fp32 logits (spec 23 §1.1: the softcap is monotone).
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  // The pending id, written where every device reads it.
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  uint32_t vocab() const { return w_.desc.vocab; }

  // fp32 [vocab]: the last replay's RAW logits (lm_head's row: no multiplier, no softcap - the host caps for
  // sampling, spec 23 §6.4).
  std::vector<float> read_logits();
  void read_logits_into(float* host);
  // bf16 [count][kv_heads][128] of layer `layer`'s K (or V) for absolute positions [first, first + count): a full
  // layer's rows, a sliding layer's through the ring (row p & 4095) - the caller asks only for positions the ring
  // still holds (count <= 4096).
  std::vector<uint16_t> read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v);
  // A sliding layer's whole ring (K or V), bf16 [4096][kv_heads][128] - the ring-addressing tests.
  std::vector<uint16_t> read_ring(uint32_t layer, bool v);
  // bf16 [layers][hidden] of the last replay, each layer's row from the device holding it (debug_tap).
  std::vector<uint16_t> read_debug_resid();
  Control control(uint32_t dev) const;

  size_t launches() const;   // every device's list
  size_t launches(uint32_t dev) const;
  MemoryComponents memory_use(uint32_t dev) const;
  std::string memory_line() const;   // one line per device
  const MusePlan& plan() const { return plan_; }   // the plan the load ran (at this run's max_len)
  const loader::MuseWeights& weights() const { return w_; }
  const MuseBuffers& buffers(uint32_t dev) const;
  const CapturedStep& step(uint32_t dev) const;
  uint32_t devices() const { return uint32_t(st_.size()); }
  uint32_t split() const { return w_.placement.split; }
  PpHandoff handoff() const { return opt_.handoff; }
  MuseAttn attention() const { return attn_; }
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }
  // P4's test hook (PipelineEngine's): the next step submits device 1's list only - a hand-off that never arrives.
  // The step must throw within the bound, never hang.
  // Spec 23d: the next prefill CHUNK likewise (device 0's half not appended), unless prefills are replayed.
  void drop_next_handoff() { drop_next_ = true; }

  // --- spec 23d: prefill (defined in b70_muse_prefill) ---------------------------------------------------------
  // Runs `ids` from the current pos in chunks of `chunk` (0 = kPfC 2048; at most 2048 - the 4096-slot ring holds a
  // chunk and the window before it) through runtime/muse/muse_prefill.h's walk on every device, then decode's head on
  // the last row: leaves pos += ids.size() on every device and the first generated id pending in cur_token (the
  // device argmax over the raw fp32 row, as every decode step). The KV and ring rows it writes are decode's to a cosine
  // bar (the slab GEMMs' sum order, the flash tiles), not bitwise (plan 23d Review Focus 5). A prompt may be continued
  // by another prefill. Two devices: each chunk runs layers [0, s) on device 0, crosses by a `copy` hand-off at chunk
  // size (a second PipelineLink; whatever --pipeline-handoff says for decode), then layers [s, L) on device 1 -
  // sequentially. A lost hand-off host-signals the event, throws naming device 1 and marks the engine until reset().
  // B70_MUSE_ATTN=eager runs the eager prefill attention too (attention(), read once at construction).
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates every device's prefill scratch, Context and (two devices) the prefill link, and checks every binary the
  // walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  // B70_PREFILL_REPLAY's arrangement: record each chunk's list once per (pos, rows) and device and replay it; the
  // hand-off's copies and event stay on the immediate lists (never inside a recording).
  void set_prefill_replay(bool enabled) { pf_replay_ = enabled ? 1 : 0; }
  // Launches every device's prefill Context has appended since it was made (0 before the first prefill).
  size_t prefill_launches() const;
  size_t prefill_launches(uint32_t dev) const;
  bool prefill_ready() const { return pf_bytes_ != 0; }
  // Spec 7's block; spec 23e's prefix cache hooks into the chunk walk: with a hook set every chunk ends at a block end
  // or at the prompt end (runtime::prefill_chunk_rows), and the hook is called on the host after each chunk whose end
  // is a multiple of kBlock (devices idle, pos == end_pos, n_active 0 on every device) and once at the prompt end (the
  // first generated id in cur_token) - Kolibri's contract (spec 20e).
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

 private:
  struct Stage;
  // Spec 23d: a device's pieces for the prefill half (b70_muse_prefill), which cannot see Stage.
  l0::Context& stage_ctx(uint32_t dev) const;
  MuseBuffers& stage_buffers(uint32_t dev) const;
  Control* stage_control(uint32_t dev) const;
  l0::CmdList& stage_imm(uint32_t dev) const;
  Stage& st(uint32_t dev) const;
  void step_once();
  bool settle(uint32_t dev);
  [[noreturn]] void fail(const std::string& what);

  MuseOptions opt_;
  MuseAttn attn_;
  MusePlan plan_;
  loader::MuseWeights w_;
  uint32_t max_len_ = 0;
  std::vector<std::unique_ptr<Stage>> st_;
  std::unique_ptr<PipelineLink> link_;
  std::vector<std::unique_ptr<l0::Mem>> taps_;
  bool broken_ = false, drop_next_ = false;
  std::vector<bool> pending_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
  BlockHook block_hook_;   // spec 23d: empty = no hook (uniform chunks)
  // Spec 23d: set by prepare_prefill (b70_muse_prefill) - releases a prefill list waiting on a hand-off that will not
  // come and waits (bounded) for every prefill list to go idle; reset() calls it.
  std::function<bool()> pf_settle_;
  // Spec 23d: the prefill state, lazy; declared LAST so it (its Contexts, kernels, recordings and link) is destroyed
  // before the buffers it points into. The deleter is set where the state is made (b70_muse_prefill), so this header
  // names no prefill symbol. pf_dev_bytes_: each device's prefill scratch + prefill link, for memory_use().
  std::vector<size_t> pf_dev_bytes_;
  size_t pf_bytes_ = 0;
  int pf_replay_ = -1;   // -1: B70_PREFILL_REPLAY decides
  std::unique_ptr<MusePrefillState, void (*)(MusePrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::muse
