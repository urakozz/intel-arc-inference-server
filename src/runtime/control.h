#pragma once
#include <cstdint>

namespace runtime {
// The ONLY mutable state the host and the captured list share. Lives in
// zeMemAllocShared memory; every kernel that needs position or the current
// token reads it; argmax_stage2 writes out_token/cur_token and advances pos.
// The host writes cur_token[0] before a replay only during prompt ingestion,
// and reads out_token[0] after the fence. Nothing else moves per token.
// 128 B = two cache lines (the spec's §8.2 sketch said 64 B but its own
// fields need 96 - corrected here, noted in the spec by Task 6).
struct Control {
  uint32_t pos;            // KV slot / position of token 0 of this step
  uint32_t n_active;       // tokens in this step (1 in this plan)
  uint32_t cur_token[8];   // input ids for this step
  uint32_t out_token[8];   // argmax outputs
  uint32_t debug_flag;     // check_finite writes first bad layer+1 here (debug builds)
  // Spec 8 (plan 8b): the live GDN state slot, 0..MtpBuffers::kSlots-1. Only the
  // SPEC_SLOTS gdn_step variants (the MTP verify lists) read it; everything else
  // uses slot 0 (`gdn_state`) and never looks. 0 whenever MTP is off.
  uint32_t gdn_live;
  uint32_t pad[12];        // pad to 128 B
};
static_assert(sizeof(Control) == 128, "control block is two cache lines");
}  // namespace runtime
