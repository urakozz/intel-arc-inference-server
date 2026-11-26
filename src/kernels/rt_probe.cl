// rt_spin: a dependent integer chain of n steps, one work-item: the GPU-side stand-in for the card's own (VRAM) experts
// in tools/probe/probe_cpu_roundtrip.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void rt_spin(__global uint* restrict out, uint n) {
  if (get_global_id(0) == 0) {
    uint x = n;
    for (uint k = 0; k < n; ++k) x = x * 1664525u + 1013904223u;
    out[0] = x;
  }
}
