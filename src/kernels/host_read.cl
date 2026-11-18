// host_read: one dword read through a pointer table, the way the PLE gather reads host memory
// (the address comes from memory, not from an argument). host_lock_test runs it on both cards
// against memory locked against swap; the kernel needs l0::Kernel::indirect_host().
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void host_read(__global const ulong* restrict table, __global uint* restrict out) {
  const uint i = get_global_id(0);
  out[i] = ((__global const uint*)table[0])[i * 1024u];
}
