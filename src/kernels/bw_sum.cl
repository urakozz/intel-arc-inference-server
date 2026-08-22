// bw_sum: read-only bandwidth probe. Every work-item strides over uint4s and
// folds them with XOR; the result is written only if it equals a constant the
// data almost surely never produces, which keeps the loads live without atomics.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void bw_sum(__global const uint4* restrict in, ulong n_vec,
                     __global uint* restrict out) {
  uint acc = 0u;
  for (ulong i = get_global_id(0); i < n_vec; i += get_global_size(0)) {
    uint4 v = in[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[get_group_id(0)] = acc;
}
