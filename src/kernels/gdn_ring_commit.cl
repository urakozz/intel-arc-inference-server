// gdn_ring_commit.cl - spec 32 §5.4: after a 32- or 64-row verify, its accepted rows 16.. into the GDN conv ring.
//
//   gdn_ring_commit(ctrl, conv_ring, conv_x)          grid (CONV_ROWS / 256, 3, 48 GDN layers), WG 256
//
// The ring is 32 deep, so the chain's links past the first (gdn_step.cl, ROW0) write their rows' conv inputs to the
// conv scratch `conv_x` [48 layers][48 rows][CONV_ROWS] (row r of the list at r - 16) and never to the ring. A commit
// of j accepted drafts leaves pos = verify pos + j + 1, and the next step reads its window at rows j - 2 .. j: the
// ones of them at 16 or later are copied here into the ring at their positions % 32 (bf16 words, unchanged). The
// host sets ctrl[CTRL_RC] = j + 1 (Control::ring_commit) after ctrl[CTRL_POS]; 0 copies nothing.
#ifndef CTRL_POS
#error "gdn_ring_commit: CTRL_POS must be defined"
#endif
#ifndef CTRL_RC
#error "gdn_ring_commit: CTRL_RC must be defined (Control::ring_commit)"
#endif
#ifndef CONV_ROWS
#define CONV_ROWS 10240   /* Qwen3.8's GDN conv channels (gdn_step.cl); PERF-56: Ornith's 8192 by define */
#endif
#define RING 32           /* runtime::PersistentDims::kConvRing */
#define XROWS 48          /* kernels::kGdnConvXRows */

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void gdn_ring_commit(__global const uint* restrict ctrl, __global ushort* restrict conv_ring,
                              __global const ushort* restrict conv_x) {
  const uint n = ctrl[CTRL_RC];
  if (n == 0u) return;
  const uint j = n - 1u;
  if (j + get_group_id(1) < 2u) return;                   // row j - 2 + y below 0
  const uint r = j - 2u + get_group_id(1);                // list rows j - 2 .. j
  if (r < 16u) return;                                    // in the ring already (the chain's first link)
  const uint vpos = ctrl[CTRL_POS] - n;                   // the verify's position
  const uint g = get_group_id(2), ch = get_group_id(0) * 256u + get_local_id(0);
  conv_ring[((size_t)g * RING + (vpos + r) % RING) * CONV_ROWS + ch] =
      conv_x[((size_t)g * XROWS + (r - 16u)) * CONV_ROWS + ch];
}
