// noop: the smallest kernel. Used by the L0 smoke test and by probe_replay,
// where ~700 of these in one command list measure per-kernel fixed cost.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void noop(__global uint* restrict out) {
  if (get_global_id(0) == 0) out[0] = 42u;
}
