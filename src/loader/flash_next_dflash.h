#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "loader/dflash_layout.h"
#include "loader/dflash_repack.h"
#include "model/dflash.h"

// Plan 21-QFN-v3e Task 1 (host arm): Flash-Next's DFlash 2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2) through the
// generic DFlash loader, at kernels::dflash::kFlashNext's shape set. The checkpoint's tensor names are
// Qwen3.8's drafter's (loader::DFlashCheckpoint's), so the pieces build exactly as Qwen3.8's and Inkling's build
// them. The bf16 release loads as Int8 - every linear quantised at load, one fp32 scale per output row (the target
// head's rule), the selector's codebooks and the norms at source precision; a packed W4A16 g64 export loads as
// Int4G64 (its own packed words, under the same contract checks Qwen3.8's and Inkling's packed drafters use).
// Host only - Task 2 wires the round and the lists.
namespace loader::flash_next {

// Opens the drafter's checkpoint (headers only until a piece is built) and resolves its load format: a packed
// W4A16 g64 export -> Int4G64, the bf16 release -> Int8. `require_packed_g64`: a quantised folder must ship every
// linear packed (Inkling's rule).
DFlashCheckpoint open(const std::string& snapshot_or_repo, DraftFormat& format, bool require_packed_g64 = false);
// Holds a descriptor to kernels::dflash::kFlashNext and to the Flash-Next target - the engine / CLI call this
// before any piece is built (the host test calls it against the published config and, on the box, the release).
void check(const model::DFlashDesc& d, const model::DFlashTargetShape& target);
// The same against an opened checkpoint, plus fc's K = taps x the target's hidden.
void check(const DFlashCheckpoint& ck, const model::DFlashTargetShape& target);
// Every linear's {K, N} at the descriptor's shape, in load order: fc; per layer q||k||v, o, gate||up, down, the
// attention and the MLP conv kernel projections; the selector's projection; the context k||v reads q||k||v's
// rows [q_n, qkv_n) in place (no pair of its own). At kFlashNext these are the shape set's pairs that
// kernels::dflash::flash_next_draft_variants names in the binaries.
std::vector<std::pair<uint32_t, uint32_t>> linear_shapes(const model::DFlashDesc& d, uint32_t target_hidden);

}  // namespace loader::flash_next
