// Exercise the production decode reader's native widening, not the probe copy.
#include "../../src/kernels/kv8.cl"
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kv8_decode_scale_probe(__global const ushort* p, __global float* out, uint count) {
  const uint i = get_global_id(0);
  if (i < count) out[i] = kv8_decode_scale(p, i);
}
