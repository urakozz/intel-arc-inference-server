#pragma once
#include <cstdint>

namespace runtime {
// The ONLY mutable state the host and the captured list share. Lives in
// zeMemAllocShared memory; every kernel that needs position or the current
// token reads it; argmax_stage2 writes out_token/cur_token and advances pos.
// The host writes cur_token[0] before a replay only during prompt ingestion,
// and reads out_token[0] after the fence. Nothing else moves per token.
// 128 B = two cache lines (the spec's §8.2 sketch said 64 B but its own
// fields need 96 - corrected here, noted in the spec by Task 6); 256 B since open-work 28 (the wide rows' ids), 768 B
// since spec 32 (the deep rows' ids).
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
  // Spec 17b (tensor parallel): each card's argmax over its vocabulary half writes the winner's fp32 bits here
  // (tp.cl's tp_argmax_stage2, CTRL_OUTV); the host merges the two cards' rows. Untouched by every other list.
  uint32_t out_value[8];
  // Open-work 13: a draft list's softmax max and normaliser (fp32 bits, draft_prob.cl) - the head's probability for
  // its draft is 1 / normaliser. Written only by the draft lists captured with the probability (runtime::build_draft).
  uint32_t draft_stat[2];
  // verify-cost-m8: the one-card verify lists' committed rows (gdn_step.cl, DEFERRED): bits 0..7 the rows the next
  // launch replays, bit 8 the parity they sit in. Engine::commit sets it, Engine::verify clears the count. 0 elsewhere.
  uint32_t gdn_pend;
  uint32_t pad[1];
  // Open-work 28: the one-card 16-row verify list's ids (its builds read CTRL_CUR 32 and write CTRL_OUT 48, so the
  // 8-id fields above keep their offsets for every other list): wide_cur[0..15] in, wide_out[0..15] out.
  uint32_t wide_cur[16];
  uint32_t wide_out[16];
  // Spec 32: the 32- and 64-row verify lists' ids (their embed_gather reads CTRL_CUR 64, their argmax writes
  // CTRL_OUT 128; every other list keeps its offsets): deep_cur[0..63] in, deep_out[0..63] out.
  uint32_t deep_cur[64];
  uint32_t deep_out[64];
};
static_assert(sizeof(Control) == 768, "control block is twelve cache lines");
}  // namespace runtime
