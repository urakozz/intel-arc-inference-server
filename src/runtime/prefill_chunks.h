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
//
// `tail` (open-work 18): under a wide stride, the positions from (end - tail) rounded down to a
// block, end = pos + remaining (the prompt's end), go back to the block stride - the hook then
// fires at every block end there, so a follow-up turn that leaves the prompt within its last
// `tail` positions restarts at most a block back (30,720 after a 32k turn, not 28,672). A wide
// chunk before that point stops at it. The caller must pass the prompt's whole rest as
// `remaining`: a call that holds one chunk of a longer prompt passes tail 0 (its cut is made).
inline uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked,
                                   uint32_t block, bool wide_stride = false, uint32_t tail = 0) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (hooked) {
    uint32_t stride =
        wide_stride && chunk > block ? (chunk + block - 1) / block * block : block;
    if (stride > block && tail != 0) {
      const uint64_t end = uint64_t(pos) + remaining;
      const uint64_t from = end > tail ? (end - tail) / block * block : 0;
      if (pos >= from) stride = block;
      else c = uint32_t(std::min<uint64_t>(c, from - pos));
    }
    c = std::min(c, stride - pos % stride);
  }
  return c;
}

// Open-work 18 (operator 2026-11-14): Qwen3.8's tail - the last 6144 positions of a hooked prefill
// (6144 to 8191 after the round-down) in block-wide chunks. A follow-up turn usually leaves the
// previous prompt a few ids before its end (the template's "<think>\n" becomes "<think>\n\n</think>");
// 6144 also keeps the 2048 restart when a client rewrites up to ~6k ids at the end (the last tool
// result or user message). A MoE model (Ornith) keeps the wide stride to its end (0).
inline constexpr uint32_t kPrefillTail = 6144;
inline uint32_t prefill_tail(bool moe) { return moe ? 0 : kPrefillTail; }

}  // namespace runtime
