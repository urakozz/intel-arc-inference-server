// probe_tp_pf.cl - spec 17c step 1 (plan 17c, Review 2026-11-10 point 3): the kernel arms of the 2048-row crossing
// rate. A PROBE kernel: no runtime path binds it.
//
//   ptpf_copy   n4 uint4 words from `src` to `dst` with a grid-stride loop. Launched on the card that owns `src`
//               with `dst` in the peer's VRAM it is a kernel P2P push (plain stores over PCIe); launched on the card
//               that owns `dst` with `src` in the peer's VRAM it is a pull (loads over PCIe).
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void ptpf_copy(__global const uint4* restrict src, __global uint4* restrict dst, uint n4) {
  for (size_t i = get_global_id(0); i < n4; i += get_global_size(0)) dst[i] = src[i];
}
