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
// at most `chunk`, and with a block hook (spec 7 §3.2) never past the next multiple of the
// hook's stride, so every chunk ends at a stride end (a block end) or at the prompt end. The
// stride is the block - the rule of every engine - unless `wide_stride`: then `chunk` rounded
// up to whole blocks when it is wider than one. Only Qwen3.8's prefill (Engine::prefill
// and the two-card pp_prefill_chunks) pass it - plan 5d, operator ruling (A): a 4096-row chunk
// over 2048-position blocks hooks every 4096, the blocks between stored as KV by the hook's
// store call without a state snapshot of their own; up to 2048 rows the stride is the block,
// exactly the old rule. The other engines (K2, Kolibri, Flash-Next) and their mocks, whose
// chunks may span several small blocks, keep a hook at every block end.
inline uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked,
                                   uint32_t block, bool wide_stride = false) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (hooked) {
    const uint32_t stride =
        wide_stride && chunk > block ? (chunk + block - 1) / block * block : block;
    c = std::min(c, stride - pos % stride);
  }
  return c;
}

}  // namespace runtime
