#pragma once
#include "loader/inkling2_loader.h"
#include "runtime/gpu_state.h"
#include "runtime/memory_plan.h"

namespace runtime::inkling2 {

inline constexpr uint32_t kRows = 8, kRing = 4096, kDraftRing = 2048, kPartitions = 32;
enum class SnapTensor { FullK, FullV, RingK, RingV, ConvK, ConvV, ConvAttn, ConvMlp,
                        DraftFullK, DraftFullV, DraftRingK, DraftRingV };
using Snap = gs::Snapshot<SnapTensor>;

struct WeightSizes {
  [[=gs::constant]] size_t model = 0;
  [[=gs::constant]] size_t layers = 0;
  [[=gs::constant]] size_t draft = 0;
};
struct PersistentSizes {
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("restoring the sequence position reconstructs control")]]
  size_t control = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullK}]] size_t full_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullV}]] size_t full_v = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::RingK}]] size_t ring_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::RingV}]] size_t ring_v = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvK}]] size_t conv_k = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvV}]] size_t conv_v = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvAttn}]] size_t conv_attn = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvMlp}]] size_t conv_mlp = 0;
};
struct ScratchSizes {
  [[=gs::scratch]] size_t residual = 0;
  [[=gs::scratch]] size_t block_bank = 0;
  [[=gs::scratch]] size_t messages = 0;
  [[=gs::scratch]] size_t controller = 0;
  [[=gs::scratch]] size_t inputs = 0;
  [[=gs::scratch]] size_t projections = 0;
  [[=gs::scratch]] size_t full_taps = 0;
  [[=gs::scratch]] size_t attention_parts = 0;
  [[=gs::scratch]] size_t attn_output = 0;
  [[=gs::scratch]] size_t routes = 0;
  [[=gs::scratch]] size_t expert_union = 0;
  [[=gs::scratch]] size_t expert_hidden = 0;
  [[=gs::scratch]] size_t expert_output = 0;
  [[=gs::scratch]] size_t pending_conv = 0;
  [[=gs::scratch]] size_t logits = 0;
  [[=gs::scratch]] size_t argmax_parts = 0;
  [[=gs::scratch]] size_t draft_taps = 0;
};
struct DraftSizes {
  [[=gs::constant]] size_t rope = 0;
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("each draft resets the recurrence from its anchor")]]
  size_t draft_control = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftFullK}]] size_t draft_full_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftFullV}]] size_t draft_full_v = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftRingK}]] size_t draft_ring_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftRingV}]] size_t draft_ring_v = 0;
  [[=gs::scratch]] size_t block_hidden = 0;
  [[=gs::scratch]] size_t block_projections = 0;
  [[=gs::scratch]] size_t draft_parts = 0;
  [[=gs::scratch]] size_t context_mix = 0;
  [[=gs::scratch]] size_t markov = 0;
  [[=gs::scratch]] size_t draft_logits = 0;
};
static_assert(gs::sizes_problem<WeightSizes, SnapTensor>().empty(), gs::sizes_problem<WeightSizes, SnapTensor>());
static_assert(gs::sizes_problem<PersistentSizes, SnapTensor>().empty(), gs::sizes_problem<PersistentSizes, SnapTensor>());
static_assert(gs::sizes_problem<ScratchSizes, SnapTensor>().empty(), gs::sizes_problem<ScratchSizes, SnapTensor>());
static_assert(gs::sizes_problem<DraftSizes, SnapTensor>().empty(), gs::sizes_problem<DraftSizes, SnapTensor>());
static_assert(gs::snapshot_problem<SnapTensor, PersistentSizes, DraftSizes>().empty());

struct Sizes {
  WeightSizes weights;
  PersistentSizes persistent;
  ScratchSizes scratch;
  DraftSizes draft;
  MemoryComponents memory() const;
};

Sizes sizes(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, uint32_t max_len, bool draft);
uint32_t max_len_that_fits(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout,
                           size_t capacity, size_t reserve, bool draft);

}  // namespace runtime::inkling2
