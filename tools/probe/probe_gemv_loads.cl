// probe_gemv_loads.cl - the load-path variants of the production int4 GEMV
// (spec 1.7 §3, P1). One tag = one `-D` set = one binary = one row of the
// battery's table, exactly as `probe_attn.cl` works.
//
// **`src/kernels/gemv.cl` is not touched by any of this.** The `PGL_*` knobs
// below all default to 0, and with every knob at 0 this file is that kernel
// verbatim - same entry point `gemv`, same four arguments, same grid, same
// `reqd_work_group_size`, same accumulation order - so `base` is a *control*
// and not a paraphrase. The harness (`tests/kernels/gemv_harness.h`) binds any
// variant here with no change beyond the module name.
//
// -------------------------------------------------------------------------
// The message-count model this file exists to falsify
// -------------------------------------------------------------------------
// Per subgroup (16 lanes, 16 consecutive `n`) per k-group (64 K elements,
// 544 weight bytes), counting the LSC messages the kernel issues:
//
//   LAYOUT 1 (canonical, what the engine binds)
//     1 x intel_sub_group_block_read8      512 B nibbles, contiguous
//     1 x intel_sub_group_block_read_us     32 B scales, contiguous with them
//     8 x vload8(ushort8)                 8 x 16 B activations, subgroup-UNIFORM
//                                         address (all 16 lanes read the same
//                                         bytes), L1/L2-resident, 0 DRAM bytes
//     = 10 messages / 544 DRAM bytes = 54.4 DRAM B per message.
//     The 544 B are one contiguous run: the tile is nibbles-then-scales.
//
//   LAYOUT 0 (GPTQ native, no repack)
//     8 x wp[j * N]                       8 x 64 B nibbles, rows N*4 B apart
//     1 x scales[g * N + n]                32 B
//     8 x vload8                          the same 128 B of activations
//     = 17 messages / 544 DRAM bytes = 32.0 DRAM B per message.
//
//   gemv_bf16 at lm_head - the 584 GB/s shape the plateau is measured against
//     1 x intel_sub_group_block_read_us8  256 B, contiguous
//     1 x vload8                           16 B activations
//     = 2 messages / 256 DRAM bytes = 128 DRAM B per message.
//
// So the int4 GEMV issues 2.35x the messages per weight byte that the shape at
// 97% of peak does. If message ISSUE is what holds the int4 shapes at
// 533-559 GB/s, removing messages must pay. Each knob below removes a
// countable number of them and changes nothing else.
//
// The standing matrix already argues the other way and this file is how that
// argument gets tested rather than asserted: layout 0 issues **1.70x** layout
// 1's messages for byte-identical traffic and identical arithmetic, and
// docs/probe-gemv-2026-08-24.md has layout 0 *ahead* at three of the five
// shapes (577 vs 540 at q||k||v, 567 vs 531 at down, 549 vs 533 at out/o_proj).
// A message-issue-bound GEMV cannot produce that table.
//
// -------------------------------------------------------------------------
// The knobs
// -------------------------------------------------------------------------
//   PGL_XWIDE     1: the 8 activation loads per k-group become 4 `ushort16`
//                    loads of 32 B. Messages 10 -> 6 (layout 1). At M = 1 the
//                    dot8 calls keep their exact order, so the output is
//                    BIT-IDENTICAL to `base`; at M > 1 the m/j nesting swaps
//                    and it is not, which is why the battery is M = 1 only
//                    (and every production shape is).
//   PGL_BLK2D     8|16: the layout-0 weight load becomes ONE
//                    `intel_sub_group_2d_block_read_32b_<R>r16x1c` covering
//                    R/8 k-groups. Messages 17 -> 10 (R = 8) or 34 -> 19
//                    (R = 16). Layout 0 ONLY -- see the note below.
//   PGL_PREFETCH  D: OpenCL `prefetch()` of the weight tile D k-groups ahead.
//                    A prefetch returns no data, so it costs NO registers --
//                    which is the whole point: the attention probe's prefetch
//                    row was confounded by +32 floats per lane (docs/15, M1),
//                    and this form cannot be.
//   PGL_PFBUF     1: the register form -- next group's `block_read8` issued
//                    before this group's arithmetic. +8 u32 per lane live.
//   PGL_BALLAST   1: the register-pressure CONTROL for PGL_PFBUF. Carries the
//                    same 8 u32 per lane across the same loop through a
//                    non-reassociable recurrence and issues no extra load. It
//                    costs 8 integer mads per k-group that PGL_PFBUF does not,
//                    so it is a CONSERVATIVE control: it can only make
//                    `pfbuf - ballast` look better than the truth, never worse.
//   PGL_CACHECTL  **MEASURED BROKEN 2026-08-26 - do not read its GB/s.** The
//                    builtin below compiles and runs, but under the signature
//                    declared here it does NOT reproduce
//                    `intel_sub_group_block_read8`'s data mapping: all three
//                    policies differed from `base` in EVERY cell (max abs err
//                    52-94 against a 0.005 bar) and timed at 129-136% of the
//                    measured 590 GB/s, which is what reading the wrong,
//                    cache-resident bytes looks like. The rows were removed
//                    from the battery rather than reported. Reviving this needs
//                    IGC's actual declaration, not a plausible one -- and
//                    whatever replaces it must be byte-checked before its GB/s
//                    is quoted, which is how this was caught.
//   PGL_CACHECTL  1..7: the 512 B weight read becomes
//                    `__builtin_IB_simd_block_read_8_global_cacheopts` at that
//                    LSC cache policy -- same message, same data mapping, same
//                    bytes, different cache behaviour. The weight stream is
//                    read exactly once per token, so caching it is pure
//                    pollution of an L1 the activation vector wants; 1 is
//                    L1UC_L3UC (bypass both), 2 is L1UC_L3C, 5 is L1S_L3UC.
//   PGL_DEQ_SHIFT 1: dequant by `w ^ 0x88888888` then a 4-bit arithmetic shift
//                    per nibble, instead of mask-and-subtract-8. Identical
//                    VALUES -- signext(q ^ 8) == q - 8 for every q in [0,16) --
//                    and identical accumulation order, so bit-identical
//                    outputs. ALU only, zero message change: it is here to ask
//                    whether the plateau shapes have any ALU term at all. It
//                    turned out to be worth +1.5% mean, positive at six shapes
//                    of six (2026-08-26), so the answer is "yes, a small one".
//
//                    **Note for whoever promotes this into src/kernels/gemv.cl.**
//                    It leans on `>>` over a NEGATIVE `int` being an ARITHMETIC
//                    shift. OpenCL C requires that (it is not C's
//                    implementation-defined behaviour: OpenCL fixes the sign
//                    fill), and it is verified empirically here -- every cell of
//                    every `deq` row came back bit-identical to `base` on this
//                    driver. A future compiler that disagreed would produce
//                    wrong logits at every token, which the golden gate catches
//                    immediately and loudly. Keep the gate in that lever's
//                    acceptance; do not promote it on a unit test alone.
//
// **Why there is no layout-1 2D-block variant.** The canonical tile is 136
// u32 = **544 bytes**, and 544 is not a multiple of 64, so consecutive tiles
// alternate 64 B / 32 B alignment. `intel_sub_group_2d_block_read_*` requires
// a 64-byte-aligned base. Beyond that there would be nothing to win if it did
// fit: layout 1's weight load is ALREADY one 512 B message per k-group, which
// is the widest an untyped block read gets. The 2D builtin's only GEMV
// application here is layout 0's strided [K/8][N] matrix, where it collapses
// eight 64 B loads into one message -- and that is the variant this file
// builds. Recorded here because "try 2D block loads on the canonical layout"
// is the obvious next idea and it is arithmetically dead.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#ifndef PGL_XWIDE
#define PGL_XWIDE 0
#endif
#ifndef PGL_BLK2D
#define PGL_BLK2D 0
#endif
#ifndef PGL_PREFETCH
#define PGL_PREFETCH 0
#endif
#ifndef PGL_PFBUF
#define PGL_PFBUF 0
#endif
#ifndef PGL_BALLAST
#define PGL_BALLAST 0
#endif
#ifndef PGL_DEQ_SHIFT
#define PGL_DEQ_SHIFT 0
#endif
#ifndef PGL_CACHECTL
#define PGL_CACHECTL 0
#endif

#if PGL_BLK2D
// No `#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io` here: measured
// 2026-08-26, our ocloc answers that pragma with "unknown or does not require
// pragma - ignoring" and declares the builtins unconditionally from CTHeader.h.
// `base_address` is a NON-const `__global void*` there, so the cast below drops
// the const rather than keeping it.
#if LAYOUT != 0
#error "probe_gemv_loads: PGL_BLK2D is layout 0 only (layout 1's 544 B tile stride is not 64 B aligned)"
#endif
#if PGL_BLK2D != 8 && PGL_BLK2D != 16
#error "probe_gemv_loads: PGL_BLK2D must be 8 (one k-group) or 16 (two)"
#endif
#endif
#if (PGL_PFBUF || PGL_PREFETCH) && LAYOUT != 1
#error "probe_gemv_loads: the prefetch variants price the PRODUCTION load path (layout 1)"
#endif
#if PGL_PFBUF && PGL_BALLAST
#error "probe_gemv_loads: PGL_BALLAST is PGL_PFBUF's control, not its companion"
#endif
#if PGL_XWIDE && M != 1
#error "probe_gemv_loads: PGL_XWIDE keeps bit-identity only at M = 1"
#endif
#if PGL_CACHECTL
#if LAYOUT != 1 || PGL_PFBUF
#error "probe_gemv_loads: PGL_CACHECTL replaces layout 1's plain block read"
#endif
// IGC's LSC cache-control enum, and the one block-read builtin that takes it.
// Declared rather than included: ocloc has no header for these, but it does
// have the builtins (probed 2026-08-26 -- this form compiles, and the
// per-operand form for the ACTIVATIONS does not: 16-bit LSC loads answer
// "__builtin_IB_lsc_load_global_ushort8: 8b and 16b not supported", so only the
// weight operand of the coordinator's per-operand idea is expressible here).
enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0, LSC_LDCC_L1UC_L3UC = 1, LSC_LDCC_L1UC_L3C = 2,
  LSC_LDCC_L1C_L3UC = 3, LSC_LDCC_L1C_L3C = 4, LSC_LDCC_L1S_L3UC = 5,
  LSC_LDCC_L1S_L3C = 6, LSC_LDCC_L1IAR_L3C = 7
};
uint8 __builtin_IB_simd_block_read_8_global_cacheopts(const __global uint*, enum LSC_LDCC);
#endif

#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define TILE_U32 136   /* 128 u32 of nibbles + 8 u32 of scales = 544 B */

#if PGL_BLK2D == 16 && (G_PER_S % 2) != 0
#error "probe_gemv_loads: PGL_BLK2D=16 needs an even number of k-groups per slice"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// 8 nibbles of one u32 word times 8 consecutive activations. gemv.cl's, verbatim.
inline float dot8(uint word, ushort8 xv) {
#if PGL_DEQ_SHIFT
  // signext_4(q ^ 8) == q - 8 for every q in [0, 16): q = 0 -> 8 -> -8,
  // q = 8 -> 0 -> 0, q = 15 -> 7 -> 7. One XOR for the whole word, then a
  // shift pair per nibble instead of a shift/mask/subtract triple. Same
  // values, same order, so the same bits come out.
  const uint u = word ^ 0x88888888u;
  float a = 0.f;
  a += (float)(((int)(u << 28)) >> 28) * bf16f(xv.s0);
  a += (float)(((int)(u << 24)) >> 28) * bf16f(xv.s1);
  a += (float)(((int)(u << 20)) >> 28) * bf16f(xv.s2);
  a += (float)(((int)(u << 16)) >> 28) * bf16f(xv.s3);
  a += (float)(((int)(u << 12)) >> 28) * bf16f(xv.s4);
  a += (float)(((int)(u <<  8)) >> 28) * bf16f(xv.s5);
  a += (float)(((int)(u <<  4)) >> 28) * bf16f(xv.s6);
  a += (float)(((int)(u      )) >> 28) * bf16f(xv.s7);
  return a;
#else
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
#endif
}

// One k-group's arithmetic, factored so every variant runs the identical
// sequence of dot8 calls over the identical activation bytes. `PGL_XWIDE`
// changes only how many messages the ushorts arrive in.
#define PGL_GROUP_MATH(wv, xbase, gacc)                                            \
  do {                                                                             \
    for (int m = 0; m < M; ++m) gacc[m] = 0.f;                                      \
    for (int j = 0; j < 8; ++j) {                                                  \
      for (int m = 0; m < M; ++m) {                                                \
        ushort8 xv = vload8(0, (xbase) + (size_t)m * K + j * 8);                   \
        gacc[m] += dot8(wv[j], xv);                                                \
      }                                                                            \
    }                                                                              \
  } while (0)

#define PGL_GROUP_MATH_XWIDE(wv, xbase, gacc)                                      \
  do {                                                                             \
    for (int m = 0; m < M; ++m) gacc[m] = 0.f;                                      \
    for (int jj = 0; jj < 4; ++jj) {                                               \
      for (int m = 0; m < M; ++m) {                                                \
        ushort16 xv = vload16(0, (xbase) + (size_t)m * K + jj * 16);               \
        gacc[m] += dot8(wv[2 * jj], xv.lo);                                        \
        gacc[m] += dot8(wv[2 * jj + 1], xv.hi);                                    \
      }                                                                            \
    }                                                                              \
  } while (0)

#if PGL_XWIDE
#define PGL_MATH(wv, xbase, gacc) PGL_GROUP_MATH_XWIDE(wv, xbase, gacc)
#else
#define PGL_MATH(wv, xbase, gacc) PGL_GROUP_MATH(wv, xbase, gacc)
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv(__global const uint* restrict w,
                   __global const half* restrict scales,
                   __global const ushort* restrict x,
                   __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  const uint s = get_group_id(1);
  const uint g0 = s * G_PER_S;
  const uint g1 = g0 + G_PER_S;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

#if PGL_BALLAST
  // The register-pressure control: 8 u32 per lane carried across the whole
  // k-loop through eight independent multiply-accumulate recurrences, which no
  // reassociation can collapse into fewer registers. No extra load is issued.
  uint ball[8];
  for (int j = 0; j < 8; ++j) ball[j] = 0x9E3779B9u + (uint)j;
#endif
#if PGL_PFBUF
  // The register form of prefetch: the first tile is read before the loop and
  // every iteration reads the NEXT one before doing this one's arithmetic.
  // +8 u32 per lane live across the arithmetic, which is exactly the confound
  // PGL_BALLAST is here to price.
  uint8 cur = intel_sub_group_block_read8(w + ((size_t)n_tile * G + g0) * TILE_U32);
#endif
#if PGL_BLK2D == 16
  uint blk16[16];
#endif

  for (uint g = g0; g < g1; ++g) {
    uint wv[8];
    float scale;
#if LAYOUT == 0
#if PGL_BLK2D
    // The whole [K/8][N] u32 matrix as one surface: width and pitch N*4 bytes,
    // height K/8 rows. One message brings this subgroup's 16 columns for
    // PGL_BLK2D consecutive k-octet rows -- 512 B (R = 8) or 1024 B (R = 16) --
    // against the eight 64 B strided loads the layout-0 baseline issues.
#if PGL_BLK2D == 8
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)(__global uint*)w, (int)(N * 4u),
                                              (int)(K / 8u), (int)(N * 4u),
                                              (int2)((int)(n_tile * SG), (int)(g * 8u)), wv);
#else
    if (((g - g0) & 1u) == 0u)
      intel_sub_group_2d_block_read_32b_16r16x1c((__global void*)(__global uint*)w, (int)(N * 4u),
                                                 (int)(K / 8u), (int)(N * 4u),
                                                 (int2)((int)(n_tile * SG), (int)(g * 8u)), blk16);
    for (int j = 0; j < 8; ++j) wv[j] = blk16[(((g - g0) & 1u) ? 8 : 0) + j];
#endif
#else
    __global const uint* wp = w + (size_t)(g * 8) * N + n;
    for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
#endif
    scale = (float)scales[(size_t)g * N + n];
#else
    __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
#if PGL_PREFETCH
    // Clamped so the address always lands inside this n_tile's own run of G
    // tiles: a prefetch may not fault, and an out-of-range one on the last
    // work-group would be reaching past the allocation.
    {
      const uint gp = min(g + (uint)PGL_PREFETCH, (uint)(G - 1u));
      prefetch(w + ((size_t)n_tile * G + gp) * TILE_U32, 136);
    }
#endif
#if PGL_PFBUF
    uint8 blk = cur;
    {
      const uint gn = min(g + 1u, (uint)(G - 1u));
      cur = intel_sub_group_block_read8(w + ((size_t)n_tile * G + gn) * TILE_U32);
    }
#elif PGL_CACHECTL
    uint8 blk = __builtin_IB_simd_block_read_8_global_cacheopts(tile, (enum LSC_LDCC)PGL_CACHECTL);
#else
    uint8 blk = intel_sub_group_block_read8(tile);
#endif
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
    ushort sh = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
    scale = (float)as_half(sh);
#endif
#if PGL_BALLAST
    for (int j = 0; j < 8; ++j) ball[j] = ball[j] * 1664525u + wv[j];
#endif
    float gacc[M];
    PGL_MATH(wv, x + g * GROUP, gacc);
    for (int m = 0; m < M; ++m) acc[m] += gacc[m] * scale;
  }

#if PGL_BALLAST
  // Consume the ballast on a branch the hardware never takes -- grid dim 1 is
  // exactly S work-groups, so get_group_id(1) < S always -- and which the
  // compiler cannot fold away, because it does not know the grid.
  {
    uint bs = 0;
    for (int j = 0; j < 8; ++j) bs ^= ball[j];
    if (get_group_id(1) == (uint)S) out[0] = (float)bs;
  }
#endif
  for (int m = 0; m < M; ++m) out[((size_t)s * M + m) * N + n] = acc[m];
}
