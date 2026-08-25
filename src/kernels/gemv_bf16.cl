// gemv_bf16.cl - bf16 weights x bf16 activations, M in [1,8]. Used for lm_head
// (N = 248320, 2.5 GB read per token) and the padded a||b projection (N = 128).
// Same lane-per-n structure as gemv.cl. Weights are in the canonical bf16 tile
// layout [n_tile][k_octet][8 k][16 n], so one intel_sub_group_block_read_us8
// per 8 k streams 256 contiguous bytes per subgroup.
//
// Two compile-time knobs shape the grid, and both exist for ONE shape - `a‖b`,
// which at the default tiling is **8 subgroups for the whole launch** and takes
// 48.8 µs to read 1.31 MB (26.9 GB/s, 4.6% of the device; docs/15 §L2).
//
// **COLS_PER_WG** - output columns per work-group, i.e. the work-group size. A
// subgroup owns one 16-column tile; COLS_PER_WG / 16 tiles share a work-group,
// so the grid is N / COLS_PER_WG work-groups. 64 is the default and what
// lm_head uses (3880 work-groups, 98.5% of measured bandwidth in situ).
//
// **KSPLIT** - how many subgroups split one tile's K. Each takes a contiguous
// K/KSPLIT slice into its own fp32 accumulator and the work-group merges them
// through SLM. This is the same idea as `gemv.cl`'s `S`, moved inside the
// work-group: `gemv` writes S partial slices for `prep_res_norm` to fold, which
// this kernel cannot do - `a‖b`'s output is read directly by `gdn_step`, and a
// separate merge would be a 49th launch per layer.
//
// **KSPLIT is the knob that worked, and it worked because it is the only one
// that changes the number of hardware threads.** Measured in situ at `a‖b`
// (docs/15 §L2), KSPLIT 1 / 4 / 16 at COLS_PER_WG 16 - 8 / 32 / 128 subgroups -
// read **48.774 / 13.115 / 5.340 µs** per launch, i.e. 26.9 / 99.9 / 245 GB/s
// and 3.36 / 3.12 / 1.92 GB/s **per subgroup**. A subgroup pulls what it pulls;
// the launch was slow because it only ever had eight of them. The shipped
// tiling is `{16, 16}`, which stops 2.4x above this shape's 2.22 µs traffic
// floor - there is at most 0.15 ms/token left in the whole kernel.
//
// **The merge order is fixed and is part of the kernel's contract.** Subgroup
// `q` of a tile holds k in [q·K/KSPLIT, (q+1)·K/KSPLIT), writes it to SLM, and
// a fixed pairwise tree collapses the slices: for stride = KSPLIT/2, …, 1,
// slice `q < stride` does `red[q] += red[q + stride]` with a barrier after each
// step - prep.cl's tree, the project's one reduction idiom. At KSPLIT = 4 the
// sum would be **(p0 + p2) + (p1 + p3)**; at the shipped KSPLIT = 16 it is that
// tree four levels deep, strides 8, 4, 2, 1. Either way it is NOT the unsplit
// build's single ascending chain: KSPLIT > 1 changes the summation order and
// the golden gate is what arbitrates it (it did: 96/96, spec 1.5 lever L2).
// There is no atomic and no data-dependent branch, so a replayed list still
// gives the same bits every time.
//
// **What was measured and rejected, in situ, 2026-08-25** (docs/15 §L2) - both
// were bit-identical and both were worth nothing, which is why this file does
// not carry them:
//
//   - COLS_PER_WG 64 -> 16 (2 work-groups -> 8): 48.774 -> 49.127 µs/launch.
//     It re-spreads the same 8 subgroups over 8 work-groups; the launch is not
//     short of work-groups.
//   - a `KU` knob issuing 4 block reads into registers before adding any of
//     them (deeper memory-level parallelism per subgroup): 49.052 µs/launch,
//     and it cost lm_head +1.3% by perturbing its codegen.
//
// Neither changes the number of hardware threads doing the work, and that - 8
// of them - turned out to be the whole of it. Do not re-derive either idea from
// first principles: they were tried, in situ, and they are worth nothing here.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#ifndef COLS_PER_WG
#define COLS_PER_WG 64
#endif
#ifndef KSPLIT
#define KSPLIT 1
#endif
#define SG 16
#define TILES_PER_WG (COLS_PER_WG / SG)
#define SG_PER_WG (TILES_PER_WG * KSPLIT)
#define WG_N (SG * SG_PER_WG)
#define K8 (K / 8)
#define K8_PER_SG (K8 / KSPLIT)
#if (COLS_PER_WG % SG) != 0 || COLS_PER_WG < SG
#error "COLS_PER_WG must be a positive multiple of the SIMD16 subgroup width"
#endif
#if (N % COLS_PER_WG) != 0
#error "N must be a multiple of COLS_PER_WG: the grid is N / COLS_PER_WG"
#endif
#if (K8 % KSPLIT) != 0 || (KSPLIT & (KSPLIT - 1)) != 0
#error "KSPLIT must be a power of two dividing K/8 (the merge is a pairwise tree)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv_bf16(__global const ushort* restrict w,
                        __global const ushort* restrict x,
                        __global float* restrict out) {
#if KSPLIT > 1
  // [subgroup][m][lane]: one fp32 partial per column per K slice. 256 B at the
  // a‖b tiling. Declared here because OpenCL wants __local at kernel scope.
  __local float red[SG_PER_WG][M][SG];
#endif
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile_idx = sg / KSPLIT;  // which 16-column tile of this work-group
  const uint q = sg % KSPLIT;         // which K slice of that tile
  const uint n_tile = get_group_id(0) * TILES_PER_WG + tile_idx;
  const uint n = n_tile * SG + lane;
  const uint k8_0 = q * K8_PER_SG;
  __global const ushort* tile = w + (size_t)n_tile * K8 * 128 + (size_t)k8_0 * 128;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint k8 = k8_0; k8 < k8_0 + K8_PER_SG; ++k8, tile += 128) {
    ushort8 wv = intel_sub_group_block_read_us8(tile);
    for (int m = 0; m < M; ++m) {
      ushort8 xv = vload8(0, x + (size_t)m * K + k8 * 8);
      float a = acc[m];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[m] = a;
    }
  }
#if KSPLIT == 1
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m];
#else
  for (int m = 0; m < M; ++m) red[sg][m][lane] = acc[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  // The fixed tree. `q < stride` keeps `sg + stride` inside this tile's own
  // KSPLIT slices, so the tiles of a wide work-group never touch each other.
  for (uint stride = KSPLIT / 2; stride > 0; stride >>= 1) {
    if (q < stride)
      for (int m = 0; m < M; ++m) red[sg][m][lane] += red[sg + stride][m][lane];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (q == 0)
    for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = red[sg][m][lane];
#endif
}
