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

PrefillSizes prefill_sizes(uint32_t max_len, uint32_t rows, bool draft) {
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
  s.projections = m * (2 * D::q_n + D::q_heads * D::relative_dim + 64) * f32;
  s.conv = 2 * m * (D::kv_n + D::hidden) * f32;
  s.full_taps = 2 * m * 64 * f32;
  s.attention = m * D::q_n * bf16;
  s.linear_slab = 2 * D::hidden * size_t(kPrefillSlab) * bf16;
  s.linear_tail_input = 256 * 2 * D::hidden * bf16;
  s.linear_tail_output = 256 * kPrefillSlab * f32;
  s.route_logits = m * D::experts * f32;
  s.routes = m * 32 * word;
  s.expert_counts = D::experts * ((m + kPrefillExpertTile - 1) / kPrefillExpertTile) * word;
  s.expert_header = (4 + D::experts + 1) * word;
  s.expert_rows = sorted * word;
  s.expert_pairs = pairs * word;
  s.expert_items = (sorted / kPrefillExpertTile) * 4 * word;
  s.expert_inputs = sorted * D::hidden * bf16;
  s.expert_hidden = sorted * D::inter * bf16;
  s.expert_output = sorted * D::hidden * f32;
  s.prelude_trace = D::core_start * D::hidden * f32;
  if (draft) {
    s.draft_taps = 4 * m * D::hidden * bf16;
    s.draft_projection = m * (D::hidden * bf16 + 2 * D::kv_n * f32);
  }
  return s;
}

}  // namespace runtime::inkling2
