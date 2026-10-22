// nb_probe.cl - spec 25c P0 (plan 25c Task 1): the SSM state's traffic floor on its own, for tools/probe/
// probe_bolt_step.cc's `state` list. No decode list binds it.
//
//   nb_state_rmw(state, n4)      grid (ceil(n4 / 256), 1), WG 256
//
// One read-modify-write of a Mamba block's whole SSM state in place - each element read once and written once, as
// nb_ssm's step does (4 MiB fp32 a block; STATE_BF16 1: 2 MiB, the bf16-state arm of Decision 2). `n4` counts
// 4-element vectors. The arithmetic is a stand-in the compiler cannot drop (s x 0.999 + 0.001); bw_sum.cl's
// grid-stride pattern, plain OpenCL C.
#pragma OPENCL FP_CONTRACT OFF

#ifndef STATE_BF16
#error "nb_state_rmw: STATE_BF16 (0 fp32, 1 bf16) must be defined"
#endif

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_state_rmw(
#if STATE_BF16
    __global ushort4* restrict state,
#else
    __global float4* restrict state,
#endif
    uint n4) {
  for (uint i = get_global_id(0); i < n4; i += get_global_size(0)) {
#if STATE_BF16
    const ushort4 h = state[i];
    float4 v = (float4)(as_float((uint)h.x << 16), as_float((uint)h.y << 16), as_float((uint)h.z << 16),
                        as_float((uint)h.w << 16));
    v = v * 0.999f + 0.001f;
    state[i] = (ushort4)((ushort)(as_uint(v.x) >> 16), (ushort)(as_uint(v.y) >> 16), (ushort)(as_uint(v.z) >> 16),
                         (ushort)(as_uint(v.w) >> 16));
#else
    state[i] = state[i] * 0.999f + 0.001f;
#endif
  }
}
