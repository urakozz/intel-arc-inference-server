// pf_embed.cl -- the first kernel of a prefill chunk: copy each of the chunk's
// token embeddings into the residual stream. Grid (1, M), work-group 256.
//
//   resid[m][0..HIDDEN) = embed[ids[m]][0..HIDDEN)        (bf16, copied verbatim)
//
// `src/kernels/embed_gather.cl:45-68` transcribed with three changes and
// nothing else:
//
//   1. **The id comes from a buffer, not from the control block.** Decode reads
//      `ctrl[CTRL_CUR + m]` because its list is captured once and replayed per
//      token, and Level Zero freezes kernel arguments at append time
//      (tests/l0/arg_capture_test.cc measures exactly that). A prefill chunk is
//      not a replayed list and `Control::cur_token[8]` holds eight ids where a
//      chunk holds up to `PrefillScratch::kC` = 2048 (interfaces.md ruling
//      A13), so the ids are uploaded as a device buffer and passed as a
//      pointer.
//   2. **The `#if M > 8` guard is deleted** with the `cur_token` limit it
//      guarded. `m_count` is a runtime argument; the grid's y extent IS M.
//   3. **An out-of-range id returns silently.** There is no `ctrl` argument any
//      more, so this kernel cannot report through `Control::debug_flag` the way
//      decode's does. The HOST validates the ids before uploading them (the
//      same bound `src/cli/b70_decode.cc`'s `read_ids` applies, `id < kVocab`);
//      the check here only stops the gather from reading past the table.
//
// No rounding happens here and none may: the embedding table is bf16 and the
// residual stream is bf16, so the row is copied as raw `ushort` (the reference
// does `embed_tokens(ids)`, a gather, not an arithmetic op).

#ifndef HIDDEN
#define HIDDEN 5120        /* model::Qwen35::kHidden */
#endif
#ifndef VOCAB
#define VOCAB 248320       /* model::Qwen35::kVocab -- the table's row count */
#endif

#define WG_EMBED 256

__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void pf_embed_gather(__global const uint* restrict ids,
                              __global const ushort* restrict embed,
                              __global ushort* restrict resid, uint m_count) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= m_count) return;          // uniform across the work-group
  const uint row = ids[m];
  if (row >= VOCAB) return;          // the host validated these; see (3) above

  __global const ushort* restrict src = embed + (size_t)row * HIDDEN;
  __global ushort* restrict dst = resid + (size_t)m * HIDDEN;
  for (uint k = lid; k < HIDDEN; k += WG_EMBED) dst[k] = src[k];
}
