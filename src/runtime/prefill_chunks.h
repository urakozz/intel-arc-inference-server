#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

// Where a prefill cuts a prompt into chunks - Engine::prefill's rule, header-only and
// dependency-free so the two-card pipeline's planner (runtime/pipeline_prefill_plan.h, spec
// 16c) cuts a prompt with the SAME function one card does. The chunk boundaries are part of
// what a prefill computes - the gated delta rule cuts its 64-position chunks relative to a
// chunk's first id - so a pipeline that cut elsewhere could not be bitwise one card (P1).
namespace runtime {

// The rows of the chunk that starts at absolute position `pos` with `remaining` ids left:
// at most `chunk`, and with a block hook (spec 7 §3.2) never past the next multiple of
// `block`, so every chunk ends at a block end or at the prompt end.
inline uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked,
                                   uint32_t block) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (hooked) c = std::min(c, block - pos % block);
  return c;
}

}  // namespace runtime
