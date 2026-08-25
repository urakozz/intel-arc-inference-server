// embed_gather.cl - the first kernel of a decode step: copy each active token's
// embedding row into the residual stream.
//
//   row              = ctrl[CTRL_CUR + m]          (Control::cur_token[m])
//   resid[m][0..K)   = embed[row][0..K)            (bf16, copied verbatim)
//
// **Why the id comes from the control block and not from a kernel argument.**
// The decode list is captured ONCE and replayed once per token, and Level Zero
// resolves a kernel's arguments at `zeCommandListAppendLaunchKernel` time, not
// at execute time (tests/l0/arg_capture_test.cc measures exactly that). An id
// passed as an argument would therefore be frozen into the list at capture, and
// every token would gather the same row. The control block is
// `zeMemAllocShared` memory whose *contents* are read when the kernel runs, so
// the host (prompt ingestion) or `argmax_stage2` (every token after the first)
// can move the id between replays without re-recording anything. This kernel is
// the only reader of `cur_token` in the list; `argmax_stage2` is its only
// writer. tests/kernels/embed_gather_test.cc replays one closed list with two
// different ids and requires two different rows - that is the property the
// whole capture-once design rests on.
//
// No rounding happens here and none may: the embedding table is already bf16
// and the residual stream is bf16, so the row is copied as raw `ushort`. (The
// reference does `embed_tokens(ids)`, a gather, not an arithmetic op.)
//
// Work assignment: grid (1, M), one work-group of 256 per token, HIDDEN/256 =
// 20 elements per work-item, lane `i` taking k = i, i+256, … so every iteration
// is a fully coalesced 512-byte read and a 512-byte write per subgroup.

#ifndef M
#define M 1
#endif
#if M > 8
// cur_token[] holds 8 ids (runtime::Control); a larger M has nowhere to read from.
#error "embed_gather: M > 8 exceeds Control::cur_token[]"
#endif
#ifndef HIDDEN
#define HIDDEN 5120        /* model::Qwen35::kHidden */
#endif
#ifndef VOCAB
#define VOCAB 248320       /* model::Qwen35::kVocab - the table's row count */
#endif

#define WG_EMBED 256

__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void embed_gather(__global uint* restrict ctrl,
                           __global const ushort* restrict embed,
                           __global ushort* restrict resid) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint row = ctrl[CTRL_CUR + m];

  // An id outside the table cannot be produced by this engine: argmax masks at
  // kVocabUsed and the tokenizer's ids are smaller still. So it does not mean
  // "clamp me" - it means the control block is wrong, i.e. an engine bug, and
  // clamping would hide it behind a plausible-looking token. Write nothing,
  // leave the residual stream as it was, and report through Control::debug_flag
  // (the same channel check_finite uses in debug builds) so the host can see
  // *which* step went wrong instead of chasing a garbage continuation.
  if (row >= VOCAB) {
    if (lid == 0) ctrl[CTRL_DEBUG] = 0xDEAD0001u;
    return;
  }

  __global const ushort* restrict src = embed + (size_t)row * HIDDEN;
  __global ushort* restrict dst = resid + (size_t)m * HIDDEN;
  for (uint k = lid; k < HIDDEN; k += WG_EMBED) dst[k] = src[k];
}
