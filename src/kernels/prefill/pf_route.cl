// pf_route.cl - phase 2 Task 6 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): the prefill's
// bf16 GEMVs over a chunk (the MoE router || shared gate, the GDN a||b) with MORE ROWS and COLUMNS a work-group,
// BITWISE pf_gemv_bf16.cl's pf_ab_proj (and so decode's gemv_bf16 row for row, spec 15c).
//
// What is kept, term for term: KSPLIT = 16 slices of K, sub-group q owns slice q; inside a slice the k8 groups
// ascend and each adds its eight products in order with `a += bf16f(w) * bf16f(x)` (the same expression, so the
// same contraction); the slices meet in pf_ab_proj's pairwise tree (stride 8, 4, 2, 1) through SLM. What moves is
// only the tiling: a work-group covers TC 16-column tiles and MT rows (pf_ab_proj: 1 and 8), so a weight block is
// read once per MT rows and an activation row once per TC tiles.
//
// Grid (ceil(N / 16 / TC), ceil(M / MT)); work-group 256 = 16 sub-groups (the K slices) x 16 lanes (columns).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef K
#error "pf_route: K must be defined"
#endif
#ifndef N
#error "pf_route: N must be defined"
#endif
#ifndef MT
#define MT 16
#endif
#ifndef TC
#define TC 2
#endif
#define SG 16
#define KSPLIT 16
#define K8 (K / 8)
#define K8_PER_SG (K8 / KSPLIT)
#define NTILES (N / SG)
#if (N % SG) != 0 || (K8 % KSPLIT) != 0
#error "pf_route: N must be whole 16-column tiles and K / 8 whole slices"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG * KSPLIT, 1, 1)))
__kernel void pf_gemv_rows(__global const ushort* restrict w,
                           __global const ushort* restrict x,
                           __global float* restrict out, uint m_count) {
  __local float red[KSPLIT][TC][MT][SG];
  const uint lane = get_sub_group_local_id();
  const uint q = get_sub_group_id();
  const uint t0 = get_group_id(0) * TC;
  const uint m0 = get_group_id(1) * MT;
  const uint k8_0 = q * K8_PER_SG;

  float acc[TC][MT];
#pragma unroll
  for (uint t = 0; t < TC; ++t)
#pragma unroll
    for (uint i = 0; i < MT; ++i) acc[t][i] = 0.f;

  for (uint k8 = k8_0; k8 < k8_0 + K8_PER_SG; ++k8) {
    ushort8 wv[TC];
#pragma unroll
    for (uint t = 0; t < TC; ++t) {
      const uint nt = min(t0 + t, (uint)NTILES - 1);   // a past-the-end tile reads tile NTILES-1, never stored
      wv[t] = intel_sub_group_block_read_us8(w + (size_t)nt * K8 * 128 + (size_t)k8 * 128);
    }
#pragma unroll
    for (uint i = 0; i < MT; ++i) {
      const uint m = m0 + i;
      const uint mr = m < m_count ? m : m0;
      const ushort8 xv = vload8(0, x + (size_t)mr * K + k8 * 8);
#pragma unroll
      for (uint t = 0; t < TC; ++t) {
        float a = acc[t][i];
        a += bf16f(wv[t].s0) * bf16f(xv.s0); a += bf16f(wv[t].s1) * bf16f(xv.s1);
        a += bf16f(wv[t].s2) * bf16f(xv.s2); a += bf16f(wv[t].s3) * bf16f(xv.s3);
        a += bf16f(wv[t].s4) * bf16f(xv.s4); a += bf16f(wv[t].s5) * bf16f(xv.s5);
        a += bf16f(wv[t].s6) * bf16f(xv.s6); a += bf16f(wv[t].s7) * bf16f(xv.s7);
        acc[t][i] = a;
      }
    }
  }
#pragma unroll
  for (uint t = 0; t < TC; ++t)
#pragma unroll
    for (uint i = 0; i < MT; ++i) red[q][t][i][lane] = acc[t][i];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = KSPLIT / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (uint t = 0; t < TC; ++t)
        for (uint i = 0; i < MT; ++i) red[q][t][i][lane] += red[q + stride][t][i][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // The tree's result in red[0]; the 16 sub-groups store its rows (sub-group q: rows q, q + 16, ..).
  for (uint e = q; e < TC * MT; e += KSPLIT) {
    const uint t = e / MT, i = e % MT;
    const uint m = m0 + i, nt = t0 + t;
    if (m < m_count && nt < NTILES) out[(size_t)m * N + nt * SG + lane] = red[0][t][i][lane];
  }
}

#ifdef ROUTE_SLAB
// pf_route_slab (phase 3 Task 2; built only with ROUTE_SLAB = the padded width NP, kernels::pf_route_slab_variant) -
// the weight pf_gemv_rows reads (gemv_bf16's tiled [N][K]: tile n / 16, k8 group, lane n % 16, eight k) re-laid as
// pf_gemm's untransposed B, bf16 [K][NP] (ldb NP), columns N..NP-1 zero, so the dense GEMM writes the logits (or
// a||b) rows [M][NP] at row pitch NP. Its sums are DPAS's, not the GEMV's order: judged, not bitwise.
//
// Grid (NP / 256, K / 8); work-group 256: one column n and eight k a work-item (the stores coalesce along n).
#define NP ROUTE_SLAB
#if NP % 256 != 0 || NP < N
#error "pf_route_slab: ROUTE_SLAB must be whole 256-column GEMM tiles covering N"
#endif
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void pf_route_slab(__global const ushort* restrict w, __global ushort* restrict slab) {
  const uint n = get_global_id(0);
  const uint k8 = get_group_id(1);
  ushort8 v = (ushort8)(0);
  if (n < N) v = vload8(0, w + (size_t)(n / SG) * K8 * 128 + (size_t)k8 * 128 + (n % SG) * 8);
  __global ushort* restrict o = slab + (size_t)k8 * 8 * NP + n;
  o[0] = v.s0; o[NP] = v.s1; o[2 * NP] = v.s2; o[3 * NP] = v.s3;
  o[4 * NP] = v.s4; o[5 * NP] = v.s5; o[6 * NP] = v.s6; o[7 * NP] = v.s7;
}
#endif
