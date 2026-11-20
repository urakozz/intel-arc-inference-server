// bw_sum: read-only bandwidth probe. Every work-item strides over uint4s and
// folds them with XOR; the result is written only if it equals a constant the
// data almost surely never produces, which keeps the loads live without atomics.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void bw_sum(__global const uint4* restrict in, ulong n_vec,
                     __global uint* restrict out) {
  uint acc = 0u;
  for (ulong i = get_global_id(0); i < n_vec; i += get_global_size(0)) {
    uint4 v = in[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[get_group_id(0)] = acc;
}

#if BW_SWEEP || BW_LSC
// The read-form sweep (PERF-111): BW_SWEEP=1 builds module bw_sum_sweep, BW_LSC=1 builds bw_sum_lsc (the
// cache-control loads, kept apart so a driver without those builtins only loses its own rows). The default
// build is the kernel above alone.
//
// One kernel per read form: bw_<form>_w<W>_u<U>_s<SIMD>.
//   form v  plain vector load, lane l reads W contiguous bytes at l * W (uint4 / uint8 / uint16 / 2 x uint16)
//   form b  intel_sub_group_block_read4 (W 16) or W / 32 x intel_sub_group_block_read8
//   form d  one intel_sub_group_2d_block_read_16b_<R>r16x2c: R = W / 4 rows of 64 B (SIMD16 only)
//   form l<cc>  __builtin_IB_lsc_load_global_uint4 / uint8 at LSC cache policy cc (module bw_sum_lsc)
//   W  bytes one lane receives per load; U independent loads issued before any is folded
//
// The address pattern is the host's. The buffer is a surface of rows `pitch` bytes apart; one load covers LR rows
// x LW bytes (LR = 1 and LW = SIMD * W, except form d: LR = R, LW = 64) and the pitch is a multiple of LW, so
// there are cols = pitch / LW columns. Sub-group gid owns column gid % cols and share s = gid / cols of the
// column's S * U * k row-blocks:
//   mode 0  U streams per sub-group, each walking its own k consecutive row-blocks (a gemv's tile walk)
//   mode 1  grid stride: row-block (it * U + u) * S + s (the base kernel's interleave)
// A contiguous read is pitch = LW (one column). Every row-block of the covered prefix is read exactly once, so
// the sub-groups' XOR, which the host folds and compares with its own XOR of those bytes, proves a form read the
// bytes it was meant to (a builtin that reads other addresses fails the check rather than timing well).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define BW_F4(v) ((v).s0 ^ (v).s1 ^ (v).s2 ^ (v).s3)
#define BW_F8(v) (BW_F4((v).lo) ^ BW_F4((v).hi))
#define BW_F16(v) (BW_F8((v).lo) ^ BW_F8((v).hi))

#define BW_LD_V(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  _Pragma("unroll") for (uint j = 0; j < NSUB; ++j) d[j] = ((const __global RAWT*)((p) + (l) * W))[j];
#define BW_LD_B4(d, p, l, W, SGS, pitch, RAWT, NSUB) d[0] = intel_sub_group_block_read4((const __global uint*)(p));
#define BW_LD_B8(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  _Pragma("unroll") for (uint j = 0; j < NSUB; ++j)  \
      d[j] = intel_sub_group_block_read8((const __global uint*)((p) + j * SGS * 32));
#define BW_LD_D8(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)(p), 64, 8, (int)(pitch), (int2)(0, 0), d);
#define BW_LD_D16(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  intel_sub_group_2d_block_read_16b_16r16x2c((__global void*)(p), 64, 16, (int)(pitch), (int2)(0, 0), d);
#define BW_LD_D32(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(p), 64, 32, (int)(pitch), (int2)(0, 0), d);

#define BW_FD(acc, d, NSUB, l, FT) _Pragma("unroll") for (uint j = 0; j < NSUB; ++j) acc ^= FT(d[j]);
// A 16b 2D read hands lane l columns l and l + 16 of each 64 B row: ushorts of one parity, which are the low
// (l even) or high (l odd) halves of the row's 32-bit words.
#define BW_FD_D(acc, d, NSUB, l, FT)                               \
  {                                                                \
    uint x = 0u;                                                   \
    _Pragma("unroll") for (uint j = 0; j < NSUB; ++j) x ^= d[j];   \
    acc ^= ((l) & 1u) ? x << 16 : x;                               \
  }

#define BW_K(NAME, SGS, U, RAWT, NSUB, LR, LW, LOADU, FOLDU, FT, W)                                              \
  __attribute__((intel_reqd_sub_group_size(SGS))) __kernel void NAME(                                           \
      __global const uchar* restrict in, ulong pitch, uint cols, uint S, uint k, uint mode,                     \
      __global uint* restrict out) {                                                                            \
    const uint gid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();                               \
    const uint l = get_sub_group_local_id();                                                                    \
    const uint s = gid / cols;                                                                                  \
    const ulong blk = (ulong)(LR) * pitch;                                                                      \
    __global const uchar* col = in + (ulong)(gid % cols) * (LW);                                                \
    uint acc = 0u;                                                                                              \
    for (uint it = 0; it < k; ++it) {                                                                           \
      RAWT r[U][NSUB];                                                                                          \
      _Pragma("unroll") for (uint u = 0; u < U; ++u) {                                                          \
        const ulong rb = mode ? (ulong)(it * U + u) * S + s : (ulong)(s * U + u) * k + it;                      \
        LOADU(r[u], col + rb * blk, l, W, SGS, pitch, RAWT, NSUB)                                               \
      }                                                                                                         \
      _Pragma("unroll") for (uint u = 0; u < U; ++u) FOLDU(acc, r[u], NSUB, l, FT)                              \
    }                                                                                                           \
    _Pragma("unroll") for (uint m = SGS / 2; m > 0; m >>= 1) acc ^= intel_sub_group_shuffle_xor(acc, m);        \
    if (l == 0) out[gid] = acc;                                                                                 \
  }
#endif

#if BW_SWEEP
// Lanes x W x U stays within 256 B a lane at SIMD16 and 128 B at SIMD32 (half the register file), so no form
// spills.
#define BW_V(W, U, SGS, RAWT, NSUB, FT) \
  BW_K(bw_v_w##W##_u##U##_s##SGS, SGS, U, RAWT, NSUB, 1, SGS * W, BW_LD_V, BW_FD, FT, W)
#define BW_B(W, U, SGS, NSUB) \
  BW_K(bw_b_w##W##_u##U##_s##SGS, SGS, U, uint8, NSUB, 1, SGS * W, BW_LD_B8, BW_FD, BW_F8, W)
#define BW_B16(U, SGS) BW_K(bw_b_w16_u##U##_s##SGS, SGS, U, uint4, 1, 1, SGS * 16, BW_LD_B4, BW_FD, BW_F4, 16)
#define BW_D(W, U, R) BW_K(bw_d_w##W##_u##U##_s16, 16, U, ushort, W / 2, R, 64, BW_LD_D##R, BW_FD_D, , W)

BW_V(16, 1, 16, uint4, 1, BW_F4) BW_V(16, 2, 16, uint4, 1, BW_F4)
BW_V(16, 4, 16, uint4, 1, BW_F4) BW_V(16, 8, 16, uint4, 1, BW_F4)
BW_V(32, 1, 16, uint8, 1, BW_F8) BW_V(32, 2, 16, uint8, 1, BW_F8)
BW_V(32, 4, 16, uint8, 1, BW_F8) BW_V(32, 8, 16, uint8, 1, BW_F8)
BW_V(64, 1, 16, uint16, 1, BW_F16) BW_V(64, 2, 16, uint16, 1, BW_F16) BW_V(64, 4, 16, uint16, 1, BW_F16)
BW_V(128, 1, 16, uint16, 2, BW_F16) BW_V(128, 2, 16, uint16, 2, BW_F16)
BW_V(16, 1, 32, uint4, 1, BW_F4) BW_V(16, 2, 32, uint4, 1, BW_F4)
BW_V(16, 4, 32, uint4, 1, BW_F4) BW_V(16, 8, 32, uint4, 1, BW_F4)
BW_V(32, 1, 32, uint8, 1, BW_F8) BW_V(32, 2, 32, uint8, 1, BW_F8) BW_V(32, 4, 32, uint8, 1, BW_F8)
BW_V(64, 1, 32, uint16, 1, BW_F16) BW_V(64, 2, 32, uint16, 1, BW_F16)
BW_V(128, 1, 32, uint16, 2, BW_F16)

BW_B16(1, 16) BW_B16(2, 16) BW_B16(4, 16) BW_B16(8, 16)
BW_B(32, 1, 16, 1) BW_B(32, 2, 16, 1) BW_B(32, 4, 16, 1) BW_B(32, 8, 16, 1)
BW_B(64, 1, 16, 2) BW_B(64, 2, 16, 2) BW_B(64, 4, 16, 2)
BW_B(128, 1, 16, 4) BW_B(128, 2, 16, 4)
BW_B16(1, 32) BW_B16(2, 32) BW_B16(4, 32) BW_B16(8, 32)
BW_B(32, 1, 32, 1) BW_B(32, 2, 32, 1) BW_B(32, 4, 32, 1)
BW_B(64, 1, 32, 2) BW_B(64, 2, 32, 2)
BW_B(128, 1, 32, 4)

BW_D(32, 1, 8) BW_D(32, 2, 8) BW_D(32, 4, 8) BW_D(32, 8, 8)
BW_D(64, 1, 16) BW_D(64, 2, 16) BW_D(64, 4, 16)
BW_D(128, 1, 32) BW_D(128, 2, 32)
#endif

#if BW_LSC
// IGC's LSC cache-control enum and its per-lane 32-bit vector loads. Declared, not included: ocloc ships no header
// for them (tools/probe/probe_gemv_loads.cl found the 8b / 16b forms refused; these are the D32 forms).
enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0, LSC_LDCC_L1UC_L3UC = 1, LSC_LDCC_L1UC_L3C = 2,
  LSC_LDCC_L1C_L3UC = 3, LSC_LDCC_L1C_L3C = 4, LSC_LDCC_L1S_L3UC = 5,
  LSC_LDCC_L1S_L3C = 6, LSC_LDCC_L1IAR_L3C = 7
};
uint4 __builtin_IB_lsc_load_global_uint4(const __global uint4* base, int immElemOff, enum LSC_LDCC cacheOpt);
uint8 __builtin_IB_lsc_load_global_uint8(const __global uint8* base, int immElemOff, enum LSC_LDCC cacheOpt);

#define BW_LD_L4(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  d[0] = __builtin_IB_lsc_load_global_uint4((const __global uint4*)((p) + (l) * 16), 0, (enum LSC_LDCC)BW_CC);
#define BW_LD_L8(d, p, l, W, SGS, pitch, RAWT, NSUB) \
  d[0] = __builtin_IB_lsc_load_global_uint8((const __global uint8*)((p) + (l) * 32), 0, (enum LSC_LDCC)BW_CC);
#define BW_L(CC, U)                                                                                    \
  BW_K(bw_l##CC##_w16_u##U##_s16, 16, U, uint4, 1, 1, 16 * 16, BW_LD_L4, BW_FD, BW_F4, 16)             \
  BW_K(bw_l##CC##_w32_u##U##_s16, 16, U, uint8, 1, 1, 16 * 32, BW_LD_L8, BW_FD, BW_F8, 32)

// cc 0 (the default policy) is the control: it must time as form v at the same W and U.
#define BW_CC 0
BW_L(0, 1) BW_L(0, 4)
#undef BW_CC
#define BW_CC 1
BW_L(1, 1) BW_L(1, 4)
#undef BW_CC
#define BW_CC 2
BW_L(2, 1) BW_L(2, 4)
#undef BW_CC
#define BW_CC 3
BW_L(3, 1) BW_L(3, 4)
#undef BW_CC
#define BW_CC 5
BW_L(5, 1) BW_L(5, 4)
#undef BW_CC
#endif
