// lfm_pf_conv.cl - spec 24d: LFM2.5-24B-A2B's short conv over a PREFILL chunk (spec 24 §2.5, §4.1; plan 24d "The conv
// over a chunk"), after in_proj's slab GEMMs wrote the chunk's rows. One binary (kernels::lfm2moe::pf_conv_variant), two
// entry points:
//
//   lfm_pf_conv_seed(ctrl, ring, seed)                  grid (HIDDEN / 256), WG 256
//   lfm_pf_conv(ctrl, inproj, seed, taps, ring, y)      grid (HIDDEN / 256, ceil(C / 16)), WG 256
//
// **No recurrence.** Output row t needs Bx[t - 2 .. t] only, and Bx is a function of in_proj's row alone, so each
// work-item (channel c, rows r0 .. r0 + 15) RECOMPUTES the Bx of its first row's two predecessors from in_proj's
// output - the same rounding, rne(rne(0 + B) · rne(0 + x)), so bitwise - instead of carrying it across work-groups.
// Only rows 0 and 1 of the chunk need the previous chunk: lfm_pf_conv_seed, a PRIOR launch of the in-order list, copies
// the ring slots of positions p0 - 2 and p0 - 1 into `seed` [2][HIDDEN] (0 below position 0 - the ring is never
// cleared: a fresh sequence never reads it). pf_gdn_conv.cl's argument: that read happened in a previous launch, so no
// work-group of lfm_pf_conv reads a slot another one writes, whatever C is (at C = 2048 the slots (p0 + C - 2) % 16
// and (p0 - 2) % 16 coincide - reading the ring in the launch that writes it would race). lfm_pf_conv never reads the
// ring; it writes the Bx of the chunk's last min(C, 2) rows into their slots (p0 + t) % 16, and decode continues from
// the ring exactly as after decode steps.
//
// Per row t (lfm_conv's chain - tests/kernels/lfm2moe_pf_ref.h conv_chunk is the twin; EDIT THE TWO TOGETHER):
//   B, C, x = rne(0 + in[t][col_B]), rne(0 + in[t][col_C]), rne(0 + in[t][col_x])   decode's split-K sum at S 1
//   Bx      = rne(f32(B) · f32(x))
//   acc     = fma(w2, Bx[t], fma(w1, Bx[t - 1], fma(w0, Bx[t - 2], 0)))      taps oldest first (tap 0 meets Bx[t-2])
//   y[t][c] = rne(f32(C) · f32(rne(acc)))
// `inproj` is the in_proj GEMM's fp32 rows [C][3 x HIDDEN] (pitch 3 x HIDDEN): [B | C | x], or - CONV_TILES 1, the arm
// conv's in_proj load (loader::repack_in_proj conv_tiles) - 16-channel tiles [B 16 | C 16 | x 16]. `taps` fp32
// [HIDDEN][3] (loader::lfm::small_block's conv_taps). Plain OpenCL C (the Mac runs it, tools/mac/clrun/lfm2moe_run.cc);
// no FP_CONTRACT pragma, as lfm_conv.cl: every product is of two bf16 values and the sums are explicit fma.
#if !defined(CTRL_POS) || !defined(CTRL_NACT) || !defined(HIDDEN) || !defined(PF_C) || !defined(CONV_TILES)
#error "lfm_pf_conv: CTRL_POS / CTRL_NACT, HIDDEN, PF_C and CONV_TILES must be defined"
#endif
#define RING 16
#define WG_CONV 256
#define ROWS 16
#define IN_N (3 * HIDDEN)
#if HIDDEN % WG_CONV != 0 || HIDDEN % 16 != 0
#error "lfm_pf_conv: HIDDEN is whole 256-lane work-groups"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
// in_proj's column of channel c's part (0 B, 1 C, 2 x).
inline size_t pf_conv_col(uint c, uint part) {
#if CONV_TILES
  return ((size_t)3 * (c / 16) + part) * 16 + c % 16;
#else
  return (size_t)part * HIDDEN + c;
#endif
}
// The in_proj value at (row t, column col) as decode's conv reads it: the split-K sum from 0.0f, one slice.
inline ushort pf_conv_in(__global const float* restrict inproj, uint t, size_t col) {
  float v = 0.0f;
  v += inproj[(size_t)t * IN_N + col];
  return rne_bf16(v);
}
inline ushort pf_conv_bx(__global const float* restrict inproj, uint t, uint c) {
  const ushort B = pf_conv_in(inproj, t, pf_conv_col(c, 0));
  const ushort X = pf_conv_in(inproj, t, pf_conv_col(c, 2));
  return rne_bf16(bf16f(B) * bf16f(X));
}

__attribute__((reqd_work_group_size(WG_CONV, 1, 1)))
__kernel void lfm_pf_conv_seed(__global const uint* restrict ctrl, __global const ushort* restrict ring,
                               __global ushort* restrict seed) {
  const uint c = get_group_id(0) * WG_CONV + get_local_id(0);
  const int p0 = (int)ctrl[CTRL_POS];
  for (int j = 0; j < 2; ++j) {
    const int p = p0 - 2 + j;
    seed[(size_t)j * HIDDEN + c] = p < 0 ? (ushort)0 : ring[(size_t)((uint)p % RING) * HIDDEN + c];
  }
}

__attribute__((reqd_work_group_size(WG_CONV, 1, 1)))
__kernel void lfm_pf_conv(__global const uint* restrict ctrl, __global const float* restrict inproj,
                          __global const ushort* restrict seed, __global const float* restrict taps,
                          __global ushort* restrict ring, __global ushort* restrict y) {
  const uint c = get_group_id(0) * WG_CONV + get_local_id(0);
  const uint r0 = get_group_id(1) * ROWS;
  const uint p0 = ctrl[CTRL_POS];
  uint C = ctrl[CTRL_NACT];
  if (C > PF_C) C = PF_C;
  if (r0 >= C) return;
  const uint r1 = min(r0 + ROWS, C);
  const float w0 = taps[(size_t)c * 3 + 0], w1 = taps[(size_t)c * 3 + 1], w2 = taps[(size_t)c * 3 + 2];
  // Bx of rows r0 - 2 and r0 - 1 (r0 is 0 or >= 16): the seed's two rows for the chunk's first rows, else recomputed
  // from in_proj.
  float f2 = r0 == 0 ? bf16f(seed[c]) : bf16f(pf_conv_bx(inproj, r0 - 2, c));
  float f1 = r0 == 0 ? bf16f(seed[(size_t)HIDDEN + c]) : bf16f(pf_conv_bx(inproj, r0 - 1, c));
  for (uint t = r0; t < r1; ++t) {
    const ushort bx = pf_conv_bx(inproj, t, c);
    const ushort Cv = pf_conv_in(inproj, t, pf_conv_col(c, 1));
    const float acc = fma(w2, bf16f(bx), fma(w1, f1, fma(w0, f2, 0.0f)));
    y[(size_t)t * HIDDEN + c] = rne_bf16(bf16f(Cv) * bf16f(rne_bf16(acc)));
    if (t + 2 >= C) ring[(size_t)((p0 + t) % RING) * HIDDEN + c] = bx;
    f2 = f1;
    f1 = bf16f(bx);
  }
}
