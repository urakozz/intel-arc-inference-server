#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "runtime/dflash/dflash_control.h"   // DFlashDraftOut: a plain struct, no link dependency

// Spec 19c: a speculative proposer fed by the target's own hidden states - DFlash (spec 19 §2, §5). The Engine
// (runtime/engine.h) knows a drafter only through this interface, because the drafter's code (runtime/dflash,
// b70_dflash_runtime) links b70_runtime and not the other way round. runtime::dflash::DFlashDrafter implements it.
//
// What the Engine does with one (spec 19 §5 "Per iteration", "Prefill"):
//   - its verify lists at M = k + 1 rows also copy the TAPS of every row - the target's residual stream at the output
//     of each layer in tap_layers() (the input of the layer after it: vLLM's target_layer_id + 1), bf16, into taps()
//     laid out [row][tap][hidden] (runtime/capture.h, build_verify_taps);
//   - Engine::commit(j, t) replays projection(pos, j + 1): rows 0..j of taps() - the committed positions - into the
//     drafter's ring. Rows past j (the rejected drafts) are never projected; the newest token (t) has no tap yet - it
//     is the next block's row 0;
//   - Engine::prefill writes the taps of the prompt positions inside window() of its end and projects them, at most
//     kernels' 8 rows a replay, through the same projection(pos, n) with taps() refilled per group;
//   - Engine::reset() calls reset(): the drafter's ring holds no position afterwards (a cold drafter);
//   - spec 19d (decision 5, as ruled there): Engine::save_state / load_state carry the ring - save_ring / load_ring,
//     ring_bytes() of it after the target's state - so a prefix-cache restore resumes the drafter exactly where the
//     snapshot left it; and key() names the drafter in the prefix store's keys (server::store_root).
// Everything runs on the Engine's queue (Engine::replay_aux), synchronously, between the Engine's own replays.
namespace runtime {

class Engine;

class TapDrafter {
 public:
  virtual ~TapDrafter() = default;
  // The largest K one draft makes (the verify lists then go up to M = max_k() + 1 <= 8 rows).
  virtual uint32_t max_k() const = 0;
  // The target layers whose OUTPUT is tapped (ascending, each < the target's layer count), and the tap buffer:
  // bf16 [8][tap_layers().size()][target hidden].
  virtual const std::vector<uint32_t>& tap_layers() const = 0;
  virtual l0::Mem& taps() = 0;
  // Positions the drafter reads back from the newest one (its ring); prefill taps only the prompt's last window().
  virtual uint32_t window() const = 0;
  // Rows 0..n-1 of taps() are positions pos .. pos + n - 1 (1 <= n <= 8): writes the projection's control block and
  // returns its captured list, which the caller replays next (and before anything rewrites taps()).
  virtual l0::CmdList& projection(uint32_t pos, uint32_t n) = 0;
  // A cold drafter: the ring and the scratch zeroed (Engine::reset).
  virtual void reset(l0::CmdList& imm) = 0;
  // Device bytes the drafter holds - its weights and its buffers (Engine::memory_use's `drafter`).
  virtual size_t bytes() const = 0;

  // Spec 19d (spec 19 decision 5): the drafter's share of a session snapshot - its whole ring (every slot of every
  // layer's K and V, as the device holds it), so the ring after load_ring is bitwise the ring at save_ring. Both copy
  // on `imm` (a synchronous immediate list) with nothing in flight; host memory of ring_bytes().
  virtual size_t ring_bytes() const = 0;
  virtual void save_ring(l0::CmdList& imm, void* host) const = 0;
  virtual void load_ring(l0::CmdList& imm, const void* host) = 0;
  // A 64-bit name for what the ring's bytes mean - the drafter's shapes, format and norm weights - which the prefix
  // store folds into every key, so a snapshot written beside one drafter is never restored beside another. Never 0.
  virtual uint64_t key() const = 0;

  // The proposer's side, for the serving loop (cli EngineAdapter, b70-decode): one block of k drafts at the
  // engine's pos into Control.cur_token[1..k] - greedy at temperature 0, sampled at `temperature` with `seed` -
  // and the readback (per draft row its id, its 16 candidates and their scores: the draft distribution q).
  // Throws unless the drafter can draft k at e's pos (k <= e.max_verify_k()).
  virtual const dflash::DFlashDraftOut& draft(Engine& e, uint32_t k, float temperature, uint64_t seed) = 0;
  // False when the drafter drafts greedily only (a DFlash v1 drafter: its sampled q is a full-vocabulary softmax
  // the readback cannot carry) - a sampled request then decodes plainly.
  virtual bool samples() const = 0;
};

}  // namespace runtime
