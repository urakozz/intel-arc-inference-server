// W4A8 probe kernels (pre-registration:
// docs/specs/2026-09-22-w4a8-dpas-probe-design.md).
//
// TWO entry points in one binary:
//   pw4a8_quant  -- dynamic per-token symmetric int8 quantisation of the bf16
//                   activations (design §3: xscale[m] = max|x[m][:]| / 127,
//                   values rounded RNE). Timed on its own; it is part of the
//                   price.
//   pw4a8_gemm   -- the W4A8 GEMM: int4 symmetric g64 weights STRAIGHT from the
//                   checkpoint's layout-0 words into the DPAS's 4-bit operand,
//                   int32 accumulation inside a group of 64 (exactly two k32
//                   instructions), and an fp32 rescale at EVERY group boundary.
//
// These are PROBE kernels. No runtime path binds either entry point, no
// production file is touched, and the weights are not repacked, re-quantised or
// otherwise changed -- the bytes the loader put on the device are the bytes the
// dpas reads.
//
// ---------------------------------------------------------------------------
// THE ONE DEVIATION FROM DESIGN §3, STATED HERE AND IN THE RECORD.
//
// §3's pseudocode spells the instruction `intel_sub_group_i4_i8_matrix_mad_k32
// (w_s4, x_s8, acc)`, i.e. weights in the *A* operand slot. This kernel issues
// `intel_sub_group_i8_i4_matrix_mad_k32(x_s8, w_s4, acc)` -- the same mixed
// 4-bit x 8-bit k32 DPAS with the weights in the *B* slot. §3's prose
// requirement ("the weights ... fed as the 4-bit operand") is met either way,
// and the rate probe measured the two at 366.89 and 366.90 TIOP/s -- the same
// number to 0.01% -- and named this one explicitly:
//
//   "in a GEMM the weights are the B operand, so the builtin a W4A8 kernel
//    would actually issue is `i8_i4_k32`, and it measures the same."
//        -- docs/probe-dpas-rates-2026-09-22.md §7
//
// It is not a free choice. The DPAS B (Src1) operand is column-per-lane with K
// packed into dwords, so for 4-bit B one lane's 32-k fragment is FOUR
// CONSECUTIVE qweight words of its own column -- the checkpoint's layout-0
// bytes verbatim, one `xor` away from signed nibbles. The A (Src2) operand is
// distributed the other way (lane = k), so weights in the A slot would need a
// nibble transpose across lanes, and this probe would then measure that
// transpose instead of the rescale it exists to price. The break-even is
// unaffected: both spellings run at the same measured 2.000x of bf16.
// ---------------------------------------------------------------------------
//
// OPERAND LAYOUTS, derived from the DPAS ISA note (Src0/Src2/Dst row-major,
// Src1 in the packed "columns into GRF columns" form -- intel-graphics-compiler
// documentation/visa/instructions/DPAS.md) and cross-checked against the two
// mappings this tree already proves on device:
//
//   A (Src2), int8 k32: `short8`, component r = output row m0+r (the dpas
//     repeat index); within a component the LOW byte is k = k0 + 2*lane and the
//     HIGH byte is k = k0 + 2*lane + 1. At bf16 k16 the same rule degenerates to
//     "component = row, lane = k", which is exactly the mapping
//     src/kernels/prefill/pf_gemm.cl feeds its bitwise-equal-to-sycl-tla dpas.
//     Consequence: viewing the int8 [M][K] activations as ushort [M][K/2], the
//     A fragment is ONE `..._16b_32r16x2c` read -- the same message pf_gemm
//     already uses, at the same coordinates, covering 64 k instead of 32.
//
//   B (Src1), int4 k32: `int4`, lane = column n, dword d = k0+8d .. k0+8d+7 with
//     nibble j of that dword being k0+8d+j. GPTQ layout 0 stores nibble i of
//     qweight[r][n] at k = 8r+i, so dword d IS qweight[k0/8+d][n]. No repack.
//
//   C (Dst): `int8`, component r = row m0+r, lane = column n. Same as bf16.
//
// The `^ 0x88888888` is the production dequant's own zero-point fold
// (src/kernels/prefill/pf_dequant_slab.cl): for a GPTQ v1 symmetric nibble
// q in [0,15] with zero point 8, (q ^ 8) read as a signed 4-bit value IS q - 8.
// The dpas sign-extends `:s4` itself, so that xor is the whole conversion.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

#define SG 16
#define GROUP 64          // the checkpoint's int4 group -- 2 x k32 per group
#define QWG 256           // pw4a8_quant work-group: one token row per group

// pw4a8_gemm tile. WG = 512 work-items = 32 sub-groups, 8 (m) x 4 (n).
// Per sub-group: 32 m rows (4 dpas repeat-blocks of 8) x 32 n (2 atoms of 16).
// That is FOUR int32 accumulators and FOUR fp32 accumulators per n-atom; both
// must be live at once because the int32 one is reset at every group boundary
// and the fp32 one is not. 8 x (int8 + float8) = 128 GRF of accumulator, which
// is why the tile is 256x128 and not pf_gemm's 256x256 (that would be 256 GRF
// of accumulator alone, before a single operand register).
#define WG_M 256
#define WG_N 128
#define SG_M 32
#define SG_N 32
#define NA 4              // dpas repeat-blocks of 8 rows
#define NB 2              // n-atoms of 16 columns

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// --- the activation quantiser ----------------------------------------------
// One work-group per token row. Design §3: xscale[m] = max|x[m][:]| / 127, and
// xq = RNE(x / xscale). The division is the literal expression and the build
// carries -cl-fp32-correctly-rounded-divide-sqrt, so it is the correctly
// rounded one. An all-zero row would divide by zero; it takes scale 0 and
// stores zeros, which is the only value that reconstructs it.
__attribute__((reqd_work_group_size(QWG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw4a8_quant(__global const ushort* restrict x,
                          __global char* restrict xq,
                          __global float* restrict xs,
                          uint M, uint K, uint ldx) {
  const uint m = get_group_id(0);
  const uint lid = get_local_id(0);
  const uint per = K / QWG;                     // 5120 / 256 = 20
  __global const ushort* restrict row = x + (size_t)m * ldx;
  __global char* restrict out = xq + (size_t)m * K;
  __local float red[QWG / SG];

  float v[32];
  float amax = 0.0f;
  for (uint j = 0; j < per; ++j) {
    v[j] = bf16f(row[lid + QWG * j]);
    amax = fmax(amax, fabs(v[j]));
  }
  amax = sub_group_reduce_max(amax);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = 0; i < QWG / SG; ++i) amax = fmax(amax, red[i]);

  const float scale = amax / 127.0f;
  if (lid == 0) xs[m] = scale;
  if (scale == 0.0f) {
    for (uint j = 0; j < per; ++j) out[lid + QWG * j] = (char)0;
    return;
  }
  for (uint j = 0; j < per; ++j)
    out[lid + QWG * j] = convert_char_sat_rte(v[j] / scale);
}

// --- the W4A8 GEMM ----------------------------------------------------------
//   C[m][n] fp32 = xs[m] * SUM_g  ws[g][n] * SUM_{k in g} xq[m][k] * w4(k, n)
// written exactly as design §3 writes it: the inner sum is an int32 dpas
// accumulation over the group's two k32 halves, and the group's contribution is
// folded into fp32 at the boundary with the product (ws[g][n] * xs[m]) as the
// multiplier -- NOT with xs[m] factored out to the epilogue, which would be a
// different (cheaper) expression than the one that was pre-registered.
//
// `xq16` is the int8 [M][K] activation matrix VIEWED as ushort [M][K/2]; `ldxq`
// is its pitch in ushorts. `sig` takes the dispatch signature (see the record's
// dispatch proof): one work-item of one work-group writes it, outside the
// mainloop.
__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pw4a8_gemm(__global const ushort* restrict xq16,
                         __global const float* restrict xs,
                         __global const uint* restrict qw,
                         __global const half* restrict wscale,
                         __global float* restrict C,
                         __global uint* restrict sig,
                         uint M, uint K, uint N, uint ldxq, uint ldc) {
  const uint gx = get_group_id(0);        // M axis
  const uint gy = get_group_id(1);        // N axis
  const uint m0 = gx * WG_M;
  const uint n0 = gy * WG_N;
  const uint s = get_sub_group_id();      // 0..31
  const uint sm = s >> 2;                 // 0..7
  const uint sn = s & 3u;                 // 0..3
  const uint mb = m0 + SG_M * sm;         // this sub-group's first row
  const uint nb = n0 + SG_N * sn;         // this sub-group's first column

  if (get_global_linear_id() == 0) sig[0] = 0x57344138u;   // 'W4A8'

  // 2D block descriptors (width bytes, height rows, pitch bytes).
  const int x_w = (int)(K), x_h = (int)M, x_p = (int)(ldxq * 2u);   // ushort view
  const int q_w = (int)(N * 4u), q_h = (int)(K / 8u), q_p = (int)(N * 4u);
  const int c_w = (int)(N * 4u), c_h = (int)M, c_p = (int)(ldc * 4u);
  const uint G = K / GROUP;

  // xs[m] for this sub-group's 32 rows: sub-group UNIFORM scalars, hoisted out
  // of the group loop (they do not depend on g).
  float xsv[NA][8];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int r = 0; r < 8; ++r) xsv[a][r] = xs[mb + 8u * (uint)a + (uint)r];

  float8 accf[NA][NB];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b) accf[a][b] = (float8)(0.0f);

#pragma unroll
  for (int pf = 0; pf < 2; ++pf) {
    const int kpf = pf * GROUP;
    intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                                  (int2)(kpf / 2, (int)(m0 + 8u * s)));
    if (s < 16u)
      intel_sub_group_2d_block_prefetch_32b_4r16x1c(
          (__global void*)qw, q_w, q_h, q_p,
          (int2)((int)(n0 + 16u * (s >> 1)), kpf / 8 + 4 * (int)(s & 1u)));
  }

  for (uint g = 0; g < G; ++g) {
    const uint k0 = g * GROUP;
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    // A: 32 rows x 32 ushorts = 32 m x 64 k of int8, ONE message.
    // afrag[c*32 + r] = X16[mb + r][k0/2 + 16c + lane], i.e. the k32 half `c`
    // of row r, low byte k0+32c+2*lane, high byte +1.
    ushort afrag[64];
    intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                               (int2)((int)(k0 / 2u), (int)mb), afrag);

    // B: 8 qweight words x 16 columns per n-atom = 64 k, ONE message each.
    // wv[b][d] = qw[k0/8 + d][nb + 16b + lane] -- k0+8d .. k0+8d+7 of column n.
    uint wv[NB][8];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)qw, q_w, q_h, q_p,
                                                (int2)((int)(nb + 16u * (uint)b),
                                                       (int)(k0 / 8u)),
                                                wv[b]);

    // The group's weight scales: one f16 per column, one message per n-atom.
    float wsc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      wsc[b] = (float)as_half(intel_sub_group_block_read_us(
          (__global const ushort*)(wscale + (size_t)g * N + nb + 16u * (uint)b)));

    if (g + 2u < G) {
      const int kpf = (int)((g + 2u) * GROUP);
      intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)xq16, x_w, x_h, x_p,
                                                    (int2)(kpf / 2, (int)(m0 + 8u * s)));
      if (s < 16u)
        intel_sub_group_2d_block_prefetch_32b_4r16x1c(
            (__global void*)qw, q_w, q_h, q_p,
            (int2)((int)(n0 + 16u * (s >> 1)), kpf / 8 + 4 * (int)(s & 1u)));
    }

    // --- the group: int32 accumulation, exactly two k32 dpas per accumulator
    int8 acci[NA][NB];
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int b = 0; b < NB; ++b) acci[a][b] = (int8)(0);

#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      short8 af[NA];
#pragma unroll
      for (int a = 0; a < NA; ++a)
        af[a] = as_short8(vload8(0, afrag + (uint)(ks * 32 + 8 * a)));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const int4 bv = as_int4(vload4(0, wv[b] + (uint)(4 * ks)) ^ (uint4)(0x88888888u));
#pragma unroll
        for (int a = 0; a < NA; ++a)
          acci[a][b] = intel_sub_group_i8_i4_matrix_mad_k32(af[a], bv, acci[a][b]);
      }
    }

    // --- the group boundary: the rescale design §3 warns is the cost
#define PW_RESCALE(a, b, r)                                                  \
  accf[a][b].s##r = fma((float)acci[a][b].s##r, wsc[b] * xsv[a][r], accf[a][b].s##r)
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        PW_RESCALE(a, b, 0); PW_RESCALE(a, b, 1); PW_RESCALE(a, b, 2); PW_RESCALE(a, b, 3);
        PW_RESCALE(a, b, 4); PW_RESCALE(a, b, 5); PW_RESCALE(a, b, 6); PW_RESCALE(a, b, 7);
      }
#undef PW_RESCALE
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
  }

  // epilogue: 8 stores per sub-group, 8 rows x 64 B each, no C read.
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int b = 0; b < NB; ++b)
      intel_sub_group_2d_block_write_32b_8r16x1c(
          (__global void*)C, c_w, c_h, c_p,
          (int2)((int)(nb + 16u * (uint)b), (int)(mb + 8u * (uint)a)),
          (__private uint*)&accf[a][b]);
}
