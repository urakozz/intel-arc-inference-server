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

// The hook's stride for a prefill unit of `rows` (one chunk, or a TP pair of two): the block - the rule of every
// engine - unless `wide_stride`: then `rows` rounded up to whole blocks when it is wider than one. Only Qwen3.8's
// prefills (Engine::prefill, the two-card pp_prefill_chunks and tp_prefill_chunks) pass it - plan 5d, ruling
// (A): a 4096-row unit over 2048-position blocks hooks every 4096, the blocks between stored as KV by the hook's store
// call without a state snapshot of their own; up to 2048 rows the stride is the block, exactly the old rule. The other
// engines (Kolibri, Flash-Next) and their mocks, whose chunks may span several small blocks, keep a hook at every
// block end.
inline uint32_t prefill_hook_stride(uint32_t rows, uint32_t block, bool wide_stride) {
  return wide_stride && rows > block ? (rows + block - 1) / block * block : block;
}

// Open-work 29: the one rule for where a hooked prefill stores a snapshot - the next snapshot point after `pos` of a
// prompt that ends at `end`: the next multiple of `stride` (prefill_hook_stride), or `end`. Every cutter (one card,
// replica and slots through prefill_chunk_rows, the pipeline's pp_prefill_chunks, TP's tp_prefill_chunks) fills the
// space between two points with its own unit (a chunk, a pipeline pass, a pair) and never crosses one.
// `tail` (open-work 18): under a wide stride, the positions from (end - tail) rounded down to a block go back to the
// block stride - the hook then fires at every block end there, so a follow-up turn that leaves the prompt within its
// last `tail` positions restarts at most a block back (30,720 after a 32k turn, not 28,672); that point itself is a
// snapshot point. `end` must be the prompt's end: a call that holds one unit of a longer prompt passes tail 0.
inline uint64_t prefill_snapshot_point(uint32_t pos, uint64_t end, uint32_t stride, uint32_t block, uint32_t tail) {
  uint64_t next = std::min<uint64_t>(end, (uint64_t(pos) / stride + 1) * stride);
  if (stride > block && tail != 0) {
    const uint64_t from = end > tail ? (end - tail) / block * block : 0;
    if (pos >= from) next = std::min<uint64_t>(end, (uint64_t(pos) / block + 1) * block);
    else next = std::min(next, from);
  }
  return next;
}

// The rows of the chunk that starts at absolute position `pos` with `remaining` ids left: at most `chunk`, and with a
// block hook (spec 7 §3.2) never past the next snapshot point (prefill_snapshot_point at the chunk's
// prefill_hook_stride), so every chunk ends at a snapshot point, a block end, or at the prompt end. The caller must
// pass the prompt's whole rest as `remaining` when it passes a `tail`.
inline uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked,
                                   uint32_t block, bool wide_stride = false, uint32_t tail = 0) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (hooked) {
    const uint64_t at = prefill_snapshot_point(pos, uint64_t(pos) + remaining,
                                               prefill_hook_stride(chunk, block, wide_stride), block, tail);
    c = uint32_t(std::min<uint64_t>(c, at - pos));
  }
  return c;
}

// Open-work 18: Qwen3.8's tail - the last 6144 positions of a hooked prefill
// (6144 to 8191 after the round-down) in block-wide chunks. A follow-up turn usually leaves the
// previous prompt a few ids before its end (the template's "<think>\n" becomes "<think>\n\n</think>");
// 6144 also keeps the 2048 restart when a client rewrites up to ~6k ids at the end (the last tool
// result or user message). A MoE model (Ornith) keeps the wide stride to its end (0).
inline constexpr uint32_t kPrefillTail = 6144;
inline uint32_t prefill_tail(bool moe) { return moe ? 0 : kPrefillTail; }

}  // namespace runtime
