// probe_dpas_rate - a DPAS *issue-rate* micro-benchmark, one data type per binary.
//
// The question (docs/probe-dpas-rates-2026-09-22.md): IGC declares matrix_mad
// builtins for bf16/f16 (k16), i8/i4_i8 (k32), i4/i2/e2m1 (k64) and scaled
// forms. A builtin existing in IGC is NOT proof the silicon runs it at the
// K-implied rate -- IGC serves many platforms. This kernel measures the rate.
//
// It is NOT a GEMM. There is no tiling, no memory loop, no barrier. Every
// operand is loaded ONCE into registers before the loop; the loop body is a
// straight chain of `UNROLL * NACC` independent `matrix_mad` calls on
// register-resident operands, so the only thing that can limit it is the DPAS
// pipe. Memory traffic over the whole kernel is ~1 KB of operand loads plus a
// 32-byte store per work-item, against >= 10^6 dpas per subgroup.
//
// THE HAZARD this kernel is built against: a loop the compiler deleted reports
// an impressive and meaningless number. Three things stop that, and the .asm
// dump checked in the record is the proof, not the intent:
//   1. the operands come from `src` (a runtime pointer), so nothing is a
//      compile-time constant and nothing can be folded;
//   2. every accumulator is summed and STORED to `dst` unconditionally, so no
//      dpas is dead;
//   3. `iters` is a kernel argument, so the trip count is unknown at compile
//      time and the loop cannot be unrolled away or evaluated.
// The accumulator chain is the one real dependency: NACC independent
// accumulators are round-robined so that `NACC` dpas separate any two writes
// to the same accumulator, which covers the systolic pipe's latency.
//
// Per-call work (the number every ops/s in the record is derived from):
//   ops = 2 * M * N * K   with M = 8 (repeat count), N = 16 (sub-group width),
//   K = the builtin's k-suffix. bf16 k16 -> 4096, i8 k32 -> 8192,
//   i4 k64 -> 16384. "2 *" counts the multiply and the add.
//
// Operand shapes (read off IGC's own CTHeader.h via ocloc's overload
// diagnostics, 2026-09-22, ocloc 26.35.39758.10):
//   A is M x K distributed over 16 lanes -> M*K/16 elements per lane;
//   B is K x 16, one column per lane -> K elements per lane;
//   C is M x 16 -> M elements per lane.
// Hence A is short8 for every 16-bit-per-lane-row case (bf16/f16 k16, i8 k32,
// i4 k64, e2m1 k64) and char8 for i2 k64; B is int8 (32 B/lane) except i2's
// int4; C is float8 or int8.
//
// This is a PROBE kernel. No runtime path binds it, no production file is
// touched, and it computes a deliberately meaningless dot product.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef DT
#error "probe_dpas_rate: DT must be defined (1..9, see the table below)"
#endif
#ifndef NACC
#define NACC 8        // independent accumulators -- covers the dpas latency
#endif
#ifndef UNROLL
#define UNROLL 8      // dpas per accumulator per loop body -> 64 dpas/iteration
#endif
#ifndef WGS
#define WGS 256       // 16 sub-groups per work-group
#endif
#define SG 16
#define NA 4          // distinct A registers held live
#define NB 4          // distinct B registers held live

// --- one stanza per data type ----------------------------------------------
// A_LOAD/B_LOAD/C_LOAD take a `__global const uint*` and produce the operand.
#if DT == 1   // bf16_bf16_k16 -- the baseline, the builtin src/kernels/prefill/pf_gemm.cl uses
#define TYPE_NAME "bf16_bf16_k16"
#define KDEPTH 16
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16((a), (b), (c))

#elif DT == 2   // f16_f16_k16
#define TYPE_NAME "f16_f16_k16"
#define KDEPTH 16
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16((a), (b), (c))

#elif DT == 3   // i8_i8_k32
#define TYPE_NAME "i8_i8_k32"
#define KDEPTH 32
#define A_T short8
#define B_T int8
#define C_T int8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (c))

#elif DT == 4   // i4_i8_k32 -- the W4A8 shape (4-bit weights, 8-bit activations)
#define TYPE_NAME "i4_i8_k32"
#define KDEPTH 32
#define A_T char8
#define B_T int8
#define C_T int8
#define A_LOAD(p) as_char8(vload2(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_i4_i8_matrix_mad_k32((a), (b), (c))

#elif DT == 5   // i4_i4_k64 -- the W4A4 shape
#define TYPE_NAME "i4_i4_k64"
#define KDEPTH 64
#define A_T short8
#define B_T int8
#define C_T int8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_i4_i4_matrix_mad_k64((a), (b), (c))

#elif DT == 6   // i2_i2_k64 -- B is int4 here, not int8 (64 x 2 bits = 16 B/lane)
#define TYPE_NAME "i2_i2_k64"
#define KDEPTH 64
#define A_T char8
#define B_T int4
#define C_T int8
#define A_LOAD(p) as_char8(vload2(0, (p)))
#define B_LOAD(p) as_int4(vload4(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_i2_i2_matrix_mad_k64((a), (b), (c))

#elif DT == 7   // e2m1_e2m1_k64 -- FP4, unscaled
#define TYPE_NAME "e2m1_e2m1_k64"
#define KDEPTH 64
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_e2m1_e2m1_matrix_mad_k64((a), (b), (c))

#elif DT == 8   // hf8_hf8_scaled_k32 -- FP8 (E4M3). The ONLY FP8 form this
                // compiler declares in OpenCL C: the unscaled
                // intel_sub_group_e4m3_e4m3_matrix_mad_k32 is an undeclared
                // identifier (recorded in the doc).
#define TYPE_NAME "hf8_hf8_scaled_k32"
#define KDEPTH 32
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define SCALED 1
#define DPAS(a, b, c) \
  intel_sub_group_hf8_hf8_scaled_matrix_mad_k32((a), (b), (c), sa, sb)

#elif DT == 9   // e2m1_e2m1_scaled_k64 -- the microscaling (MXFP4/NVFP4) form.
                // Two E8M0 scale bytes per operand, one per 32-deep half.
#define TYPE_NAME "e2m1_e2m1_scaled_k64"
#define KDEPTH 64
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define SCALED 2
#define DPAS(a, b, c) \
  intel_sub_group_e2m1_e2m1_scaled_matrix_mad_k64((a), (b), (c), sa2, sb2)

#elif DT == 10  // bf16_bf16_scaled_k16 -- the `scaled_matrix_mad` form the task
                // asks for, on the one input type this device's backend
                // accepts. Same K as DT 1, so DT10/DT1 is the cost of the
                // E8M0 scale operands alone.
#define TYPE_NAME "bf16_bf16_scaled_k16"
#define KDEPTH 16
#define A_T short8
#define B_T int8
#define C_T float8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int8(vload8(0, (p)))
#define C_LOAD(p) as_float8(vload8(0, (p)))
#define SCALED 1
#define DPAS(a, b, c) \
  intel_sub_group_bf16_bf16_scaled_matrix_mad_k16((a), (b), (c), sa, sb)

#elif DT == 11  // i8_i4_k32 -- the other W4A8 operand order (8-bit A, 4-bit B).
                // In a GEMM the weights are the B operand, so THIS is the one
                // a W4A8 prefill kernel would actually issue.
#define TYPE_NAME "i8_i4_k32"
#define KDEPTH 32
#define A_T short8
#define B_T int4
#define C_T int8
#define A_LOAD(p) as_short8(vload4(0, (p)))
#define B_LOAD(p) as_int4(vload4(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_i8_i4_matrix_mad_k32((a), (b), (c))

#elif DT == 12  // u4_u4_k64 -- unsigned 4-bit, the packing an asymmetric
                // zero-point int4 checkpoint actually stores.
#define TYPE_NAME "u4_u4_k64"
#define KDEPTH 64
#define A_T ushort8
#define B_T uint8
#define C_T int8
#define A_LOAD(p) as_ushort8(vload4(0, (p)))
#define B_LOAD(p) as_uint8(vload8(0, (p)))
#define C_LOAD(p) as_int8(vload8(0, (p)))
#define DPAS(a, b, c) intel_sub_group_u4_u4_matrix_mad_k64((a), (b), (c))

#else
#error "probe_dpas_rate: unknown DT"
#endif

// `iters` is a kernel argument -- the trip count is not a compile-time
// constant, so the loop cannot be unrolled away, evaluated, or deleted.
// `scale_bits` carries the two E8M0 scale bytes for the scaled forms (the host
// passes 127 = 2^0); it is a runtime value for the same reason.
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void dpas_rate(__global const uint* restrict src, __global uint* restrict dst,
                        uint iters, uint scale_bits) {
  // Lane-varying operand addresses: every lane of every sub-group starts at a
  // different word of `src`, so no operand register is uniform and none can be
  // recognised as a duplicate of another.
  const uint base = (get_global_id(0) * 37u) & 4095u;
  const __global uint* p = src + base;

#if SCALED == 1
  const uchar sa = (uchar)(scale_bits & 0xffu);
  const uchar sb = (uchar)((scale_bits >> 8) & 0xffu);
#elif SCALED == 2
  const uchar2 sa2 = (uchar2)((uchar)(scale_bits & 0xffu), (uchar)(scale_bits & 0xffu));
  const uchar2 sb2 = (uchar2)((uchar)((scale_bits >> 8) & 0xffu),
                              (uchar)((scale_bits >> 8) & 0xffu));
#endif

  A_T a[NA];
  B_T b[NB];
  C_T acc[NACC];
#pragma unroll
  for (int i = 0; i < NA; ++i) a[i] = A_LOAD(p + 8 * i);
#pragma unroll
  for (int i = 0; i < NB; ++i) b[i] = B_LOAD(p + 64 + 8 * i);
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = C_LOAD(p + 128 + 8 * i);

  // The measured loop. UNROLL * NACC dpas per iteration, all independent
  // within a round of NACC; `a`/`b` are loop-invariant register operands, the
  // accumulators are the only carried dependency.
#pragma unroll 1
  for (uint t = 0; t < iters; ++t) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
#pragma unroll
      for (int i = 0; i < NACC; ++i)
        acc[i] = DPAS(a[(i + u) & (NA - 1)], b[(i ^ u) & (NB - 1)], acc[i]);
  }

  // Unconditional sink: every accumulator reaches memory, so no dpas is dead.
  C_T s = acc[0];
#pragma unroll
  for (int i = 1; i < NACC; ++i) s += acc[i];
  vstore8(as_uint8(s), 0, dst + 8 * get_global_id(0));
}
