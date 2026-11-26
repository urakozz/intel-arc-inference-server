#include "runtime/inkling2/inkling2_prefill.h"

#include <algorithm>

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
static_assert(kPrefillAttentionRows + D::window - 1 <= kRing && kPrefillAttentionRows == kDraftRing);
}  // namespace

std::vector<PrefillChunk> prefill_chunks(const D& desc, uint32_t position, uint32_t rows, uint32_t cap) {
  std::vector<PrefillChunk> chunks;
  chunks.reserve((rows + cap - 1) / cap);
  const uint32_t end = position + rows;
  while (position < end) {
    const uint32_t n = std::min(cap, end - position);
    chunks.push_back({position, n, desc.visits});
    position += n;
  }
  return chunks;
}

PrefillSizes prefill_sizes(uint32_t max_len, uint32_t rows, bool draft, bool taps, const Tp& tp, bool dflash) {
  const size_t m = rows;
  constexpr size_t bf16 = sizeof(uint16_t), f32 = sizeof(float), word = sizeof(uint32_t);
  const size_t pairs = m * D::top_k;
  // Each expert's sorted rows pad to a multiple of the GEMM's 32-row tile.
  const size_t sorted = (pairs + D::experts * (kPrefillExpertTile - 1) + kPrefillExpertTile - 1) /
                        kPrefillExpertTile * kPrefillExpertTile;
  PrefillSizes s;
  s.ids = size_t(max_len) * word;
  s.residual = 6 * m * D::hidden * f32;
  s.block_bank = (D::layers / D::block_layers) * m * D::hidden * f32;
  s.messages = 4 * m * D::hidden * f32;
  s.controller = m * (12 * D::hidden * f32 + 3 * D::controller_rank * bf16);
  s.inputs = 3 * m * D::hidden * bf16;
  s.projections = m * (2 * tp.q_n() + tp.rel_n() + 64) * f32;
  s.conv = 2 * m * (tp.kv_n() + D::hidden) * f32;
  s.full_taps = 2 * m * 64 * f32;
  s.attention = m * tp.q_n() * bf16;
  for (uint32_t n = 1; n <= std::min<uint32_t>(rows, kPrefillAttentionRows); ++n)   // PERF-12's partials
    s.split_parts = std::max(s.split_parts,
                             size_t(attention_spans(n, UINT32_MAX)) * n * (tp.q_n() / D::head_dim) * (D::head_dim + 2) * f32);
  // the widest K a prompt linear reads: the controllers' 2 x hidden, or the DFlash 2 drafter's fc over four taps
  const size_t widest = (draft && dflash ? 4 : 2) * size_t(D::hidden);
  s.linear_slab = widest * size_t(kPrefillSlab) * bf16;
  s.linear_tail_input = 256 * widest * bf16;
  s.linear_tail_output = 256 * kPrefillSlab * f32;
  s.route_logits = m * D::experts * f32;
  s.routes = m * 32 * word;
  s.expert_counts = D::experts * ((m + kPrefillExpertTile - 1) / kPrefillExpertTile) * word;
  s.expert_header = (4 + D::experts + 1) * word;
  s.expert_rows = sorted * word;
  s.expert_pairs = pairs * word;
  s.expert_items = (sorted / kPrefillExpertTile) * 4 * word;
  s.expert_inputs = sorted * D::hidden * bf16;
  s.expert_hidden = sorted * tp.inter() * bf16;
  s.expert_output = sorted * D::hidden * f32;
  s.prelude_trace = D::core_start * D::hidden * f32;
  if (taps) s.draft_taps = 4 * m * D::hidden * bf16;
  // DFlash 2 (plan 41k): the chunk's in-window rows through fc (fp32), hidden_norm (bf16) and a layer's k / v (fp32)
  if (draft) s.draft_projection = m * (D::hidden * (dflash ? f32 + bf16 : bf16) + 2 * D::kv_n * f32);
  if (tp.on()) {
    s.tp_part = m * D::hidden * f32;
    s.tp_landing = 2 * m * D::hidden * f32;
  }
  return s;
}

}  // namespace runtime::inkling2
