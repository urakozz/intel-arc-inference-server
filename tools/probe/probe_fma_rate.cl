// probe_fma_rate.cl - open-work 3m: the B70's sustained fp32 VECTOR fma rate, the compute bound
// of every GDN prefill kernel that runs plain fp32 vector code (record:
// docs/probes/probe-gdn-prefill-2026-11-10.md). A PROBE kernel: no runtime path binds it, and the
// value it computes is meaningless.
//
// Each work-item carries NACC independent fp32 accumulators through `acc = fma(acc, a, b)`,
// UNROLL rounds per loop iteration, so the loop body is UNROLL * NACC independent-in-a-round
// `mad` instructions on SIMD16 (16 lanes each). `a` sits just below 1 and `b` is small, so every
// accumulator converges to b / (1 - a) ~ 1: no overflow, no denormal, nothing special.
//
// `iters` is a kernel argument, so the trip count cannot be evaluated at compile time; `fma` is
// never contracted or reassociated (no fast-math in the project's ocloc options); the sum of the
// accumulators is stored so nothing is dead.
#define NACC 16
#define UNROLL 4
#define WGS 256

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fma_rate(__global const float* restrict src, __global float* restrict dst,
                       uint iters) {
  const uint gid = get_global_id(0);
  const float a = 0.999f + 1e-6f * src[gid & 4095u];
  const float b = 1e-3f * src[(gid * 7u + 1u) & 4095u];
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = src[(gid + 131u * (uint)i) & 4095u];

#pragma unroll 1
  for (uint t = 0; t < iters; ++t) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
#pragma unroll
      for (int i = 0; i < NACC; ++i) acc[i] = fma(acc[i], a, b);
  }

  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  dst[gid] = s;
}
