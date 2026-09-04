// ctrl_read: the shape of every decode kernel's first instruction - read one
// dword of the shared-memory control block. probe_replay times this against
// noop to price a control-block read per kernel.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void ctrl_read(__global const uint* restrict ctrl, __global uint* restrict out) {
  if (get_global_id(0) == 0) out[0] = ctrl[0] + 1u;
}
