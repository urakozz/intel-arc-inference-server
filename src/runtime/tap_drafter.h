#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/memory.h"

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
//   - Engine::reset() calls reset(): the drafter's ring holds no position afterwards (a cold drafter).
// Everything runs on the Engine's queue (Engine::replay_aux), synchronously, between the Engine's own replays.
namespace runtime {

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
};

}  // namespace runtime
