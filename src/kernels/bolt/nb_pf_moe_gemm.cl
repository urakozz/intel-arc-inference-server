// nb_pf_moe_gemm - spec 25d: the GROUPED expert GEMM of a Nemotron 3.5 Bolt prefill chunk - a family copy (Decision 19) of
// spec 15d's prefill/pf_moe_gemm.cl bf16 form (I8 0), whose mainloop it keeps statement for statement (the sub-group tile
// 32 m x 64 n, the 2D block A read and the VNNI-transform B read, the 32-k dpas body; 4 sub-groups a work-group), with
// Bolt's two epilogues in place of the gated SiLU one (Bolt's routed experts are NOT gated: up only, relu2 - spec 25 §2.4):
//
//   RELU2_EPI = 1   the up GEMM (K 1024 the latent, N = I): a = rne(relu(rne(acc))²) - relu2 on the bf16 value, the square
//                   in fp32, one rounding (decode's nb_moe_up epilogue, torch.square(F.relu) on bf16)
//   RELU2_EPI = 0   the down GEMM (K = I, N 1024): y = rne(acc) - the one rounding every consumer applies first (decode's
//                   nb_moe_down rounds its sum the same way before the combine)
//
// ONE launch covers every expert's tiles of a weight batch: work-group (tile, n-block) reads its tile's expert and first
// sorted row from the table nb_pf_moe.cl's nb_pf_sort wrote and computes that tile's TM = 32 rows x 256 columns. A padding
// tile (expert NONE), and a tile whose expert is outside this launch's batch [b0, b1), returns at once (no barrier in this
// kernel), so the grid stays a function of (C, E). B = nb_pf_dequant_up / _dn's bf16 blocks, expert e at (e - b0) x K x N,
// row-major [K][N].
//
// **Row independence** (pf_moe_gemm.cl's argument, unchanged): DPAS reduces along K only and the epilogue is per element, so
// a row's output does not depend on which other rows share its tile - bolt_pf_kernels_test holds the grouped rows of an
// expert BITWISE pf_gemm_T0's dense GEMM over the same rows and the same bf16 block (+ the same epilogue on the host).
// **Bounds**: every tile is a whole TM rows of its own expert (the sort pads), so the 2D block surfaces are exactly one tile
// tall and every read and write stays inside it.
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !defined(K) || !defined(N) || !defined(TM) || !defined(RELU2_EPI)
#error "nb_pf_moe_gemm: K, N (one expert block), TM and RELU2_EPI must be defined (src/kernels/CMakeLists.txt, spec 25d)"
#endif
#if TM != 32
#error "nb_pf_moe_gemm: TM must be 32 - one sub-group's row tile is the whole expert tile"
#endif
#define SG 16
#define NONE 0xFFFFFFFFu
#define WG_N 256
#define SGS 4          /* 4 sub-groups x 64 columns */
#define KSTEP 32       /* pf_gemm's WG_K */
#if N % WG_N != 0
#error "nb_pf_moe_gemm: N must be whole 256-column work-group blocks"
#endif
#if K % KSTEP != 0
#error "nb_pf_moe_gemm: K must be whole k-steps"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
#if RELU2_EPI
inline ushort relu2(float acc) {
  float u = bf16f(rne_bf16(acc));
  u = u > 0.0f ? u : 0.0f;
  return rne_bf16(u * u);
}
#define EPI(v) relu2(v)
#else
#define EPI(v) rne_bf16(v)
#endif

// A: bf16 [rows][K] (the gathered latent, or the up activations); B: the batch's bf16 blocks; out: bf16 [rows][N].
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void nb_pf_moe_gemm(__global const uint* restrict tiles, __global const ushort* restrict A,
                             __global const uint* restrict B, __global ushort* restrict out, uint b0, uint b1) {
  const uint tile = get_group_id(0);
  const uint e = tiles[2 * tile];
  if (e == NONE || e < b0 || e >= b1) return;      // uniform; no barrier below
  const uint row0 = tiles[2 * tile + 1];
  const uint n0 = get_group_id(1) * WG_N;
  const uint s = get_sub_group_id();

  __global const ushort* restrict Ab = A + (size_t)row0 * K;
  __global const ushort* restrict Bb = (__global const ushort*)B + (size_t)(e - b0) * K * N;
  const int a_w = K * 2, a_h = TM, a_p = K * 2;
  const int b_w = N * 2, b_h = K, b_p = N * 2;
  const uint nsg = n0 + 64u * s;                   // this sub-group's 64 columns

  float8 acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);

  for (uint k0 = 0; k0 < K; k0 += KSTEP) {
    // A: 32 m x 32 k, one message. afrag[c*32 + r] = A[row0 + r][k0 + 16c + lane].
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)Ab, a_w, a_h, a_p, (int2)((int)k0, 0), afrag);
    // B: 32 k x 64 n, two VNNI-transform messages.
    uint bfrag[2][32];
    intel_sub_group_2d_block_read_transform_16b_32r16x2c((__global void*)Bb, b_w, b_h, b_p, (int2)((int)nsg, (int)k0),
                                                         bfrag[0]);
    intel_sub_group_2d_block_read_transform_16b_32r16x2c((__global void*)Bb, b_w, b_h, b_p,
                                                         (int2)((int)(nsg + 32u), (int)k0), bfrag[1]);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int h = b >> 1, c = b & 1;
        const int8 bv = as_int8(vload8(0, bfrag[h] + (uint)(c * 16 + 8 * ks)));
#pragma unroll
        for (int a = 0; a < 4; ++a) acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(af[a], bv, acc[a][b]);
      }
    }
  }

  // out = EPI(acc): acc[a][b].sR is row 8a + R, column nsg + 16b + lane.
  __global ushort* restrict Y = out + (size_t)row0 * N;
  const int y_w = N * 2, y_h = TM, y_p = N * 2;
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float8 v = acc[a][b];
      ushort yv[8];
      yv[0] = EPI(v.s0); yv[1] = EPI(v.s1); yv[2] = EPI(v.s2); yv[3] = EPI(v.s3);
      yv[4] = EPI(v.s4); yv[5] = EPI(v.s5); yv[6] = EPI(v.s6); yv[7] = EPI(v.s7);
      intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)Y, y_w, y_h, y_p, (int2)((int)(nsg + 16u * (uint)b), 8 * a),
                                                 yv);
    }
}
