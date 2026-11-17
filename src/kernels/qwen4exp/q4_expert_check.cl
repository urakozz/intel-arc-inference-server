// q4_expert_check.cl - plan 21-QFN-v2a Task 8: the expert tier's load check, the device half (q4_ple_check's
// arrangement). Every routed expert block is reached the way the _TBL MoE kernels reach it - through its device's
// table of block addresses (u64 [layers][1024]: gate||up of expert e at [e], down at [512 + e]), each address in the
// layer's VRAM arena or its pinned host mirror - and sampled words of it are read back:
//
//   q4_expert_check(rows, idx, off, out, n)    grid (ceil(n / 64)), WG 64
//   out[g] = the u64 at rows[idx[g]] + off[g]  (g < n)
//
// The loader picks the samples (loader::Q4DevicePart::check_*: each block's first and last word and one every 1 MiB,
// so every 2 MiB page of every arena and mirror is read) and keeps the checkpoint's words beside them; the engine
// compares. Runs once, at engine construction, with the host indirect-access flag the MoE launches set.
//
// Plain OpenCL C (no sub-group functions).
#define WG_CHECK 64
__attribute__((reqd_work_group_size(WG_CHECK, 1, 1)))
__kernel void q4_expert_check(__global const ulong* restrict rows, __global const uint* restrict idx,
                              __global const uint* restrict off, __global ulong* restrict out, uint n) {
  const uint g = get_global_id(0);
  if (g >= n) return;
  out[g] = *(__global const ulong*)(intptr_t)(rows[idx[g]] + off[g]);
}
