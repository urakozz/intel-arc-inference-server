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

// PERF-12: a full visit's prompt attention over `rows` rows at `pos`, in this many key spans (split-K; 1: one walk).
// The PAIR grid is ceil(rows / 16) x 8 kv heads work-groups; under half of kSplitGroups the keys split towards
// kSplitGroups work-groups, a span at least kSplitKeys keys.
inline constexpr uint32_t kSplitGroups = 256, kSplitKeys = 4096;
constexpr uint32_t attention_spans(uint32_t rows, uint32_t pos) {
  const uint32_t groups = (rows + 15) / 16 * model::Inkling2Desc::kv_heads;
  if (groups >= kSplitGroups / 2) return 1;
  const uint32_t want = (kSplitGroups + groups - 1) / groups, most = pos / kSplitKeys;
  return want < most ? want : most > 1 ? most : 1;
}

// taps: the card writes the drafter's taps (the drafter's card, or under `--parallel pp` the card before it).
// tp: a `--parallel tp` card's (its heads' and experts' widths, the all-reduce's part and landing).
PrefillSizes prefill_sizes(uint32_t max_len, uint32_t rows, bool draft, bool taps, const Tp& tp = {},
                           bool dflash = false)
    pre(max_len > 0 && max_len <= model::Inkling2Desc::trained_max_len)
    pre(rows > 0 && rows <= kPrefillRows);

}  // namespace runtime::inkling2
