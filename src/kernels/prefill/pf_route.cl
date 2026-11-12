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

#ifdef ROUTE_DPAS
// ---------------------------------------------------------------------------------------------------------
// pf_route_dpas - operator decision D3 (2026-11-12, only if the f32 judge passes): the same GEMV as bf16 DPAS -
// x [M][K] bf16 times the pf_ab_proj tile layout's W - fp32 logits whose LAST BITS differ from decode's (the
// products are exact, the sum order is the DPAS's). Grid (ceil(N / 16 / 16), ceil(M / 32)); sub-group s of a
// work-group owns the 16-column tile 16 gx + s (tiles past N return) and 32 rows (four 8-row DPAS tiles).
// B fragment: lane n = column n, VNNI pairs along k - two k8 groups of the tile layout's block reads.
__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG * 16, 1, 1)))
__kernel void pf_route_dpas(__global const ushort* restrict w,
                            __global const ushort* restrict x,
                            __global float* restrict out, uint m_count) {
  const uint lane = get_sub_group_local_id();
  const uint nt = get_group_id(0) * 16 + get_sub_group_id();
  if (nt >= NTILES) return;                       // uniform across the sub-group; no barrier below
  const uint m0 = get_group_id(1) * 32;
  __global const ushort* tile = w + (size_t)nt * K8 * 128;
  float8 acc[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
  size_t row[4][8];
  for (uint a = 0; a < 4; ++a)
    for (uint r = 0; r < 8; ++r) row[a][r] = (size_t)min(m0 + a * 8 + r, m_count - 1) * K + lane;
  for (uint kb = 0; kb < K / 16; ++kb) {
    const ushort8 w0 = intel_sub_group_block_read_us8(tile + (size_t)(2 * kb) * 128);
    const ushort8 w1 = intel_sub_group_block_read_us8(tile + (size_t)(2 * kb + 1) * 128);
    const int8 b = (int8)(as_int((ushort2)(w0.s0, w0.s1)), as_int((ushort2)(w0.s2, w0.s3)),
                          as_int((ushort2)(w0.s4, w0.s5)), as_int((ushort2)(w0.s6, w0.s7)),
                          as_int((ushort2)(w1.s0, w1.s1)), as_int((ushort2)(w1.s2, w1.s3)),
                          as_int((ushort2)(w1.s4, w1.s5)), as_int((ushort2)(w1.s6, w1.s7)));
    for (uint a = 0; a < 4; ++a) {
      short av[8];
      for (uint r = 0; r < 8; ++r) av[r] = (short)x[row[a][r] + kb * 16];
      acc[a] = intel_sub_group_bf16_bf16_matrix_mad_k16(vload8(0, av), b, acc[a]);
    }
  }
  for (uint a = 0; a < 4; ++a) {
    float f[8];
    vstore8(acc[a], 0, f);
    for (uint r = 0; r < 8; ++r) {
      const uint m = m0 + a * 8 + r;
      if (m < m_count) out[(size_t)m * N + nt * SG + lane] = f[r];
    }
  }
}
#endif
