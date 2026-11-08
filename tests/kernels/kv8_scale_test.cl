// Exercise the production raw-flash scale reader on every finite fp16 code.
#include "../../src/kernels/kv8.cl"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kv8_scale_test(__global const ushort* p, __global float* out, uint count) {
  const uint i = get_global_id(0);
  if (i < count) out[i] = kv8_flash_scale(p, i);
}
