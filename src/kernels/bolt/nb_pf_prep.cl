// nb_pf_prep.cl - spec 25d: the first launch of a Nemotron 3.5 Bolt prefill chunk - the chunk's token rows out of the tiled
// embedding into the residual h:
//
//   nb_pf_embed_gather(ids, tiles, h, m_count)     grid (1, C), WG 256
//
// 25c's nb_embed_gather (nb_prep.cl) with one change (lfm_pf_prep.cl's, spec 24d): the id comes from a buffer (`ids`, the
// chunk's ids the host uploaded), not from Control::cur_token (eight slots where a chunk holds up to 2048). Row id of the
// bf16 [vocab][K] embedding sits in gemv_bf16's tiles {K, EMBED_V} at ((id / 16) x K/8 + k / 8) x 8 + k % 8) x 16 +
// id % 16 (loader::bolt::embed_row_index). An id >= EMBED_V reads row EMBED_V - 1 (never past the allocation; the host
// refuses such ids before the chunk). A copy: no rounding.
//
// The other per-row kernels of a prefill chunk are 25c's own binaries at M = 2048 (the norm sites `_NF` / `_S1` /
// `_S1_T1`, the router, the route, the shared expert's relu2 and the latent, the attention prep - kernels::bolt::pf_*):
// decode's chains row for row, so the prefill's per-row arithmetic is decode's by construction. Plain OpenCL C
// (tools/mac/clrun/bolt_pf_run.cc runs it).
#if !defined(K) || !defined(EMBED_V)
#error "nb_pf_embed_gather: K and EMBED_V must be defined"
#endif
#define WG_EMBED 256

__attribute__((reqd_work_group_size(WG_EMBED, 1, 1)))
__kernel void nb_pf_embed_gather(__global const uint* restrict ids, __global const ushort* restrict tiles,
                                 __global ushort* restrict h, uint m_count) {
  const uint m = get_group_id(1);
  if (m >= m_count) return;   // uniform across the work-group
  uint id = ids[m];
  if (id >= EMBED_V) id = EMBED_V - 1;
  const size_t base = (size_t)(id / 16) * (K / 8);
  for (uint k = get_local_id(0); k < K; k += WG_EMBED)
    h[(size_t)m * K + k] = tiles[((base + k / 8) * 8 + k % 8) * 16 + id % 16];
}
