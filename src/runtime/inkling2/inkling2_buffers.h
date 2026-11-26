#pragma once
#include "l0/memory.h"
#include "runtime/inkling2/inkling2_sizes.h"

namespace runtime::inkling2 {

struct WeightBuffers {
  std::unique_ptr<l0::Mem> model, draft;
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
      attn_output, routes, expert_union, expert_hidden, expert_output, pending_conv, logits, argmax_parts, draft_taps;
  ScratchBuffers(l0::Context& ctx, const ScratchSizes& sizes);
};
struct DraftBuffers {
  std::unique_ptr<l0::Mem> rope, draft_control, draft_full_k, draft_full_v, draft_ring_k, draft_ring_v,
      block_hidden, block_projections, draft_parts, context_mix, markov, draft_logits;
  DraftBuffers(l0::Context& ctx, const DraftSizes& sizes);
};
struct PrefillBuffers {
  std::unique_ptr<l0::Mem> ids, residual, block_bank, messages, controller, inputs, projections, conv, full_taps,
      attention, linear_slab, linear_tail_input, linear_tail_output, route_logits, routes, expert_counts,
      expert_header, expert_rows, expert_pairs,
      expert_items, expert_inputs, expert_hidden, expert_output, draft_taps, draft_projection, prelude_trace;
  PrefillBuffers(l0::Context& ctx, const PrefillSizes& sizes);
};
static_assert(gs::buffers_problem<l0::Mem, WeightBuffers, WeightSizes>().empty());
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
  WeightBuffers weights;
  Buffers(l0::Context& ctx, const Sizes& sizes, const loader::Inkling2WeightLayout& layout);
  void upload(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, unsigned threads = 0);
  void zero(l0::Context& ctx);
};

}  // namespace runtime::inkling2
