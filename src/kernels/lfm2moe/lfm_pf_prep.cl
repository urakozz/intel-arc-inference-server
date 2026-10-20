// lfm_pf_prep.cl - spec 24d: the first launch of an LFM2.5-24B-A2B prefill chunk - the chunk's token rows out of the
// tiled embedding into the residual h:
//
//   lfm_pf_embed_gather(ids, tiles, h, m_count)     grid (1, C), WG 256
//
// 24c's lfm_embed_gather (lfm_prep.cl) with prefill/pf_embed.cl's one change: the id comes from a buffer (`ids`, the
// chunk's ids the host uploaded), not from Control::cur_token (eight slots where a chunk holds up to 2048). Row id of
// the bf16 [vocab][K] embedding sits in gemv_bf16's tiles {K, EMBED_V} - the ONE copy that also serves the bf16 head
// (spec 24 §4.4) - at ((id / 16) x K/8 + k / 8) x 8 + k % 8) x 16 + id % 16 (loader::lfm::lfm_embed_row_index). An id
// >= EMBED_V reads row EMBED_V - 1 (never past the allocation; the host refuses such ids before the chunk). A copy: no
// rounding (the reference's embed_tokens(ids) is a gather). Plain OpenCL C (tools/mac/clrun/lfm2moe_run.cc runs it).
#if !defined(K) || !defined(EMBED_V)
#error "lfm_pf_embed_gather: K and EMBED_V must be defined"
#endif
#define WG_EMBED 256

__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void lfm_pf_embed_gather(__global const uint* restrict ids, __global const ushort* restrict tiles,
                                  __global ushort* restrict h, uint m_count) {
  const uint m = get_group_id(1);
  if (m >= m_count) return;   // uniform across the work-group
  uint id = ids[m];
  if (id >= EMBED_V) id = EMBED_V - 1;
  const size_t base = (size_t)(id / 16) * (K / 8);
  for (uint k = get_local_id(0); k < K; k += WG_EMBED)
    h[(size_t)m * K + k] = tiles[((base + k / 8) * 8 + k % 8) * 16 + id % 16];
}
