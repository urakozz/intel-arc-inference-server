#pragma once
#include <span>
#include <vector>

#include "runtime/inkling2/inkling2_sizes.h"

namespace runtime::inkling2 {

struct PrefillChunk {
  uint32_t position, rows;
  std::span<const model::Inkling2Visit> visits;
};

// The caller records all visits of each chunk in order before the next chunk.
// The descriptor must outlive the returned walk.
std::vector<PrefillChunk> prefill_chunks(const model::Inkling2Desc& desc, uint32_t position,
                                        uint32_t rows, uint32_t cap)
    pre(cap > 0 && cap <= kPrefillRows)
    pre(rows > 0 && position < model::Inkling2Desc::trained_max_len &&
        rows <= model::Inkling2Desc::trained_max_len - position)
    pre(desc.visits.size() == model::Inkling2Desc::visit_count);

PrefillSizes prefill_sizes(uint32_t max_len, uint32_t rows, bool draft)
    pre(max_len > 0 && max_len <= model::Inkling2Desc::trained_max_len)
    pre(rows > 0 && rows <= kPrefillRows);

}  // namespace runtime::inkling2
