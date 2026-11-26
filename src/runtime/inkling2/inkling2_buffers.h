#pragma once
#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "runtime/inkling2/inkling2_sizes.h"
#include "runtime/inkling2/inkling2_tp_split.h"

namespace runtime::inkling2 {

struct WeightBuffers {
  std::unique_ptr<l0::Mem> model, draft, prompt_int8, candidate_head;
  std::array<std::unique_ptr<l0::Mem>, model::Inkling2Desc::layers> layers;
  const l0::Mem& arena(uint32_t index) const;
};
struct PersistentBuffers {
  l0::Mem control, full_k, full_v, ring_k, ring_v, conv_k, conv_v, conv_attn, conv_mlp;
  std::unique_ptr<l0::Mem> full_k_scale, full_v_scale;   // `--kv-cache int8` only
  PersistentBuffers(l0::Context& ctx, const PersistentSizes& sizes);
};
struct ScratchBuffers {
  std::unique_ptr<l0::Mem> residual, block_bank, messages, controller, inputs, projections, full_taps, attention_parts,
      attn_output, routes, expert_union, expert_hidden, expert_output, pending_conv, logits, argmax_parts, draft_taps,
      head_candidates, glue_parts, step_io;
  ScratchBuffers(l0::Context& ctx, const ScratchSizes& sizes);
};
struct DraftBuffers {
  std::unique_ptr<l0::Mem> rope, draft_control, draft_full_k, draft_full_v, draft_ring_k, draft_ring_v, draft_tap,
      block_hidden, block_projections, draft_parts, context_mix, markov, draft_logits;
  DraftBuffers(l0::Context& ctx, const DraftSizes& sizes);
};
struct PrefillBuffers {
  std::unique_ptr<l0::Mem> ids, residual, block_bank, messages, controller, inputs, projections, conv, full_taps,
      attention, split_parts, linear_slab, linear_tail_input, linear_tail_output, route_logits, routes, expert_counts,
      expert_header, expert_rows, expert_pairs,
      expert_items, expert_inputs, expert_hidden, expert_output, draft_taps, draft_projection, prelude_trace,
      int8_rows, int8_scales, tp_part, tp_landing;
  PrefillBuffers(l0::Context& ctx, const PrefillSizes& sizes);
};
struct HandoffBuffers {
  std::unique_ptr<l0::Mem> landing, prefill_landing, prefill_outbox, snapshot_shadow;
  HandoffBuffers(l0::Context& ctx, const HandoffSizes& sizes);
};
struct TpBuffers {
  std::unique_ptr<l0::Mem> tp_landing, tp_send_ctr, tp_sum_ctr, tp_state, tp_part;
  TpBuffers(l0::Context& ctx, const TpSizes& sizes);
};
static_assert(gs::buffers_problem<l0::Mem, WeightBuffers, WeightSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, TpBuffers, TpSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, HandoffBuffers, HandoffSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, PersistentBuffers, PersistentSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, ScratchBuffers, ScratchSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, DraftBuffers, DraftSizes>().empty());
static_assert(gs::buffers_problem<l0::Mem, PrefillBuffers, PrefillSizes>().empty());

// The byte plan is checked before creating any allocation. Contents are
// uploaded by a separate stage, so a failed load can unwind the arenas.
struct Buffers {
  PersistentBuffers persistent;
  ScratchBuffers scratch;
  DraftBuffers draft;
  PrefillBuffers prefill;
  PrefillBuffers prefill_b;   // `--parallel tp`: the second chunk in flight's
  HandoffBuffers handoff;
  TpBuffers tp;
  WeightBuffers weights;
  // share: the card's visits (stage_share); only its layers' arenas are allocated.
  Buffers(l0::Context& ctx, const Sizes& sizes, const loader::Inkling2WeightLayout& layout, const StageShare& share);
  void upload(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, unsigned threads = 0);
  // `--parallel pp`: one pass over the checkpoint into every card's arenas (a card skips the layers it does not hold).
  static void upload(std::span<const std::pair<l0::Context*, Buffers*>> cards,
                     const loader::Inkling2Checkpoint& checkpoint, unsigned threads = 0);
  // `--parallel tp`: one pass over the checkpoint into both cards, each its part (`split`: card i's tp_weights).
  static void upload(std::span<const std::pair<l0::Context*, Buffers*>> cards, std::span<const TpWeights* const> split,
                     const loader::Inkling2Checkpoint& checkpoint, unsigned threads = 0)
      pre(cards.size() == split.size());
  // The drafter's RoPE table on each card that holds the drafter.
  static void upload_rope(std::span<const std::pair<l0::Context*, Buffers*>> cards, std::vector<l0::CmdList>& lists,
                          const loader::Inkling2Checkpoint& checkpoint);
  // Every zero_on_reset buffer; the position-indexed caches (the full visits' K / V and scales, the drafter's full
  // layer) only in positions [0, written) - nothing past what a session wrote is ever non-zero (PERF-90).
  void zero(l0::Context& ctx, uint32_t written, uint32_t max_len, uint32_t full_slots);
};

}  // namespace runtime::inkling2
