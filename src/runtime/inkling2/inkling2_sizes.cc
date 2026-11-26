#include "runtime/inkling2/inkling2_sizes.h"

#include <stdexcept>

#include "runtime/control.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/inkling2/inkling2_prefill.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
using L = loader::Inkling2WeightLayout;
}  // namespace

MemoryComponents Sizes::memory() const {
  MemoryComponents m;
  m.model = weights.model + weights.layers;
  m.kv = gs::bytes(persistent, gs::Group::Kv);
  m.decode_state = gs::bytes(persistent, gs::Group::State) + gs::bytes(scratch);
  m.prefill_scratch = gs::bytes(prefill);
  m.drafter = weights.draft + gs::bytes(draft);
  return m;
}

Sizes sizes(const D& d, const L& layout, uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8) {
  if (!max_len || max_len > D::trained_max_len)
    throw std::invalid_argument("Inkling --max-len must be within its trained context");
  if (draft && !layout.bytes[L::kDraft])
    throw std::invalid_argument("Inkling drafter weights are missing. Use --spec off or restore mtp.safetensors.");

  Sizes out;
  if (prefill_rows) out.prefill = prefill_sizes(max_len, prefill_rows, draft);
  out.weights = {layout.bytes[L::kModel], layout.layer_bytes(), draft ? layout.bytes[L::kDraft] : 0};
  auto& p = out.persistent;
  constexpr size_t kv_row = D::kv_n * 2;
  constexpr size_t conv_rows = (D::conv_taps - 1) * sizeof(float);
  p.control = sizeof(Control);
  p.full_k = p.full_v = size_t(d.full_visits) * max_len * (kv8 ? D::kv_n : kv_row);
  if (kv8) p.full_k_scale = p.full_v_scale = size_t(d.full_visits) * max_len * D::kv_heads * 2;
  p.ring_k = p.ring_v = size_t(d.window_visits) * kRing * kv_row;
  p.conv_k = p.conv_v = d.visits.size() * D::kv_n * conv_rows;
  p.conv_attn = p.conv_mlp = d.visits.size() * D::hidden * conv_rows;

  // The developer prompt path uses verify-width rows. Pending convolution
  // inputs are separate from the committed three-row histories.
  auto& s = out.scratch;
  constexpr size_t m = kRows;
  s.residual = m * D::hidden * 6 * 4;
  s.block_bank = m * D::hidden * (D::layers / D::block_layers + 1) * 4;
  s.messages = m * (D::core_end - D::core_start) * 4 * D::hidden * 4;
  s.controller = m * (D::hidden * 12 * sizeof(float) + 3 * D::controller_rank * sizeof(uint16_t));
  s.inputs = m * (3 * D::hidden + D::q_n) * 2;
  s.projections = m * (2 * D::q_n + 2 * D::kv_n + D::q_heads * D::relative_dim + 64) * 4;
  s.full_taps = m * 2 * 64 * 4;
  s.attention_parts = m * D::q_heads * kPartitions * (D::head_dim + 2) * 4;
  s.attn_output = m * D::q_n * 2;
  s.routes = d.visits.size() * m * 32 * sizeof(uint32_t);
  s.expert_union = m * D::top_k * 2 * sizeof(uint32_t);
  s.expert_hidden = m * D::top_k * D::inter * 2;
  s.expert_output = m * D::top_k * D::hidden * 4;
  s.pending_conv = d.visits.size() * m * 2 * (D::kv_n + D::hidden) * 4;
  s.logits = m * D::head_rows * 4;
  s.argmax_parts = m * ((D::head_rows + 1023) / 1024) * 2 * sizeof(uint32_t);
  if (draft) {
    s.draft_taps = (m + m) * d.draft.taps.size() * D::hidden * 2;
    auto& x = out.draft;
    x.draft_control = sizeof(dflash::DFlashCtl) + sizeof(dflash::DFlashDraftOut);
    x.draft_ring_k = x.draft_ring_v = 2 * kDraftRing * kv_row;
    x.draft_full_k = x.draft_full_v = size_t(max_len) * kv_row;
    x.rope = (size_t(max_len) + d.draft.block) * D::head_dim * 4;
    x.block_hidden = m * (D::hidden * 3 + D::draft_inter) * 4;
    x.block_projections = m * (D::q_n + 2 * D::kv_n + 2 * D::draft_inter) * 4;
    x.draft_parts = m * D::q_heads * kPartitions * (D::head_dim + 2) * 4;
    x.context_mix = m * (D::hidden + 2 * D::kv_n) * 4;
    x.markov = (m * 3 + 3) * D::markov_rank * 4;
    x.draft_logits = (m + 2) * D::head_rows * 4;
  }
  return out;
}

uint32_t max_len_that_fits(const D& d, const L& layout, size_t capacity, size_t reserve, bool draft,
                           uint32_t prefill_rows, bool kv8) {
  if (capacity < reserve) return 0;
  const auto fits = [&](uint32_t pages) {
    return sizes(d, layout, pages * kMaxLenQuantum, draft, prefill_rows, kv8).memory().total() <= capacity - reserve;
  };
  uint32_t lo = kMinAutoMaxLen / kMaxLenQuantum, hi = D::trained_max_len / kMaxLenQuantum;
  if (!fits(lo)) return 0;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid)) lo = mid;
    else hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}

}  // namespace runtime::inkling2
