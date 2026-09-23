// pf_gdn_scan.cl - `gdn_chunk` part C: the chunk-to-chunk sequential state
// scan. Three entry points share one file: `pf_gdn_scan` (ruling A27's vector
// kernel, the default), `pf_gdn_scan_dpas` (ruling A28 option (c)'s DPAS
// kernel - BUILT, MEASURED and REVERTED, kept in-tree and launched by nothing),
// and the separate D3 experiment `pf_gdn_scan_dpas_split`, selected only by
// `B70_PREFILL_GDN_SCAN=dpas_split`.
// It is the only kernel in the GDN family that touches `gdn_state`.
//
// ---------------------------------------------------------------------------
// THE DPAS REWRITE WAS BUILT, MEASURED AND REVERTED (2026-09-05)
// ---------------------------------------------------------------------------
// A27's verdict on this kernel was "under 1.5 ms needs DPAS, not another vector
// mapping" (27% FMA density x 0.52 instructions/XVE/clock). Ruling A28 option
// (c) made that its own task; `pf_gdn_scan_dpas` below is the result and
// docs/prefill-gdn-scan-dpas-2026-09-05.md is its record.
//
// **It works, and it is fast: 3.3937 -> 0.8491 ms per GDN layer per chunk
// (3.997x, 11.38 TFLOP/s), against a pre-registered bar of <= 1.0 - HIT.** Every
// band bar it was pre-registered against was met, `gdn_chunk_test`'s three
// structural bit-identity cases stayed bit-identical, determinism stayed 9 x 3
// bitwise and the self-consistency gate stayed 18/18 on 1101 determined rows.
//
// **It was reverted because it costs a TOKEN.** The pre-registration made the
// golden gate the arbiter and fixed "any gate failure -> revert" in advance, and
// two of its bars fell:
//   * `prefill_gate_rtn_test`: **92 of 93 determined rows exact, was 93/93.**
//     The prose prompt's generated position 18 flips 11362 -> 10932 on a row
//     whose golden argmax is unique. Not a tie, not a judgement call.
//   * the pre-registered "every `gdn_state` cosine > 0.999" - Vishva's `code`
//     prompt reads **0.998499499 at L60**, where the vector kernel reads
//     0.999780657 in the same session. `1 - cos` grew 6.84x, i.e. the state's
//     L2 relative error ~2.6x, against a predicted 1.5x. Five of the six
//     (prompt, checkpoint) cosines did not move at all; this is a single-point
//     sensitivity, as is the token flip on the OTHER checkpoint.
//
// **The cause is D2's own arithmetic and not a defect.** DPAS takes bf16
// operands, and the fp32 quantities this kernel contracts are the recurrent
// state and the intermediates built from it, so feeding them to DPAS means
// rounding them: R1 (`S` into `w.S`), R2 (`vn` into the A2 term), R3 (`vn *
// exp(gl-gc)` into the state update) and R4 (`A2`). "DPAS only where both
// operands are already bf16" describes NO work on this kernel - every stage has
// exactly one bf16 operand and one fp32 one - so there is no cheaper variant of
// the same idea. What the measurement adds to the pre-registration is that the
// synthetic fixture under-reports the cost: `gdn_chunk_test`'s band barely moved
// (max rel 3.506e-02 -> 3.534e-02, +0.8%) while one real prompt's 60th GDN layer
// lost 6.84x of `1 - cos` and one token flipped on the other checkpoint.
//
// The separately pre-registered D3 split-bf16 experiment carries `S`, `D` and
// (since the 2026-09-23 fix) `A2` as bf16 hi/lo pairs (`hi = rne(x)`,
// `lo = rne(x - hi)`) so the DPAS operands retain residual information.  Its
// bounded opt-in arithmetic, gates and evidence are in
// docs/prefill-gdn-scan-split-2026-09-20.md and
// docs/prefill-gdn-scan-split-fix-2026-09-23.md.  **R2, the single bf16
// rounding of `vn` in the `A2*vn` term, is the one D2 rounding D3 keeps** - a
// 2x2 on the golden gate measured its effect as nil while `A2`'s was the whole
// failure (entry (3)'s block below).  None of this alters the D2 result or the
// vector default, which is still what `gdn.cc` binds.
//
// ---------------------------------------------------------------------------
// The tile mapping of `pf_gdn_scan` - ruling A25/A27, unchanged
// ---------------------------------------------------------------------------
// The grid is (48 v-heads, 4 state-column chunks) = 192 work-groups of 256 = 16
// subgroups of 16 lanes. Work-group `(h, c)` owns state columns `[32c, 32c+32)`
// of head `h`, and work-item `(sgid, lane)` owns **16 fp32 of state in
// registers** - k-rows `8*sgid .. 8*sgid+7`, columns `32c+lane` and
// `32c+lane+16`. The state stays k-major, so the 128 v of a row are contiguous
// and a subgroup's 16 lanes read one 64 B line, exactly as decode does.
//
// As delivered by L1-core, stages 1 and 2 were decode's `gdn_step` mapping
// (`gdn_step.cl:26-65`) widened one position at a time: 64 sequential 256-lane
// band-tree reductions each, five barriers per tree, with only `sgid == 0` - 16
// of 256 work-items - running the epilogue. That is 640 barriers per 64-position
// sub-chunk and it measured **15.071 ms per GDN layer per chunk**, 35.1% of the
// whole `--pp 4096` walk. Ruling A25's rewrite:
//
//   * the chunk-START `S` tile is staged **once per sub-chunk** into
//     `Ss[128][32]` fp32 (16 KB SLM), written straight out of the work-items'
//     own state registers - never re-read from global. It is constant through
//     stages 1-2, which both read the chunk-start state.
//   * every work-item gets an **output tile of 8**: 64 positions x 32 columns
//     = 2048 outputs / 256 lanes. Subgroup `sgid` owns positions
//     `i in [4*sgid, 4*sgid+4)`; lane `l` owns columns `l` and `l+16` - the
//     same two columns it already owns in registers.
//   * each of those 8 outputs is a **private ascending-k fp32 `fma`
//     accumulation over all 128 k**. No `red[]`, no tree, no idle epilogue.
//   * `A2` is read from global at a subgroup-uniform address rather than
//     staged: under this mapping each of the 64 `A2` rows is read by exactly
//     one subgroup, so staging it was pure overhead. SLM per work-group
//     therefore FALLS, 26.5 KB -> 24.5 KB.
//   * **barriers per sub-chunk: 640 -> 3.**
//
// Stage 3 is untouched: it was already per-work-item with no barriers, and it
// keeps `S` in registers. `Ss` is a read-only copy of the chunk-start tile, so
// nothing is written back to it.
//
// ---------------------------------------------------------------------------
// The algebra, and the ONE reassociation relative to decode
// ---------------------------------------------------------------------------
// Per 64-chunk `t`, with `S` the state as it stands at the chunk's FIRST
// position (transcribed in tests/prefill/gdn_chunk_ref.h, which must be edited
// with this file):
//
//   vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x]
//   o[i][x]  = ( SUM_k q[i][k] * S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
//   S[k][x] <- S[k][x] * exp(gl) + SUM_i k[i][k] * ( vn[i][x] * exp(gl - gc[i]) )
//
// **`o` reads the CHUNK-START `S`, not the updated one.** Computing `vn` and
// `o` before touching `S` is what removes FLA's `[C/64][48][128][128]` fp32 `h`
// buffer (100,663,296 B at C = 2048, derived) entirely - we never materialise a
// per-chunk state snapshot because the scan is fused.
//
// **The reassociation.** Decode interleaves the decay with the rank-1 update
// per position: `S *= exp(g_i)` then `S += kf_i (x) delta_i`, 64 times
// (gdn_step.cl:318-349). The chunk applies `exp(gl)` ONCE and then adds
// `SUM_i kf_i (x) (vn_i * exp(gl - gc_i))`. Algebraically identical; in fp32 it
// replaces 64 sequential multiplies by one, and `exp(gl - gc_i)` by a product
// of `exp(g_j)`. This is a named source of the state band that
// `tests/prefill/gdn_chunk_test.cc` records, and it is a far smaller term than
// Q1-Q4's four bf16 roundings in `pf_gdn_wu`
// (docs/prefill-l1-preregistration-2026-09-05.md §2.2).
//
// **The SECOND source, new with A25 and pre-registered before it was built**
// (docs/prefill-gdn-scan-2026-09-05.md §1.3): both 128-term contractions above
// now run as ONE ascending-k `fma` chain per output instead of
// `gdn_step.cl:52-65`'s 16-band tree. Algebraically identical, differently
// rounded, and it is why A22's 3.506e-02 is not this kernel's bar. The token
// gate is the arbiter; the band is re-measured and recorded.
//
// **Nothing here rounds** (gdn_step.cl:317-356, P9) and `gdn_o` is written fp32
// (gdn_step.cl:368-370, P10) - `pf_gated_head`, `gdn_chunk`'s tenth launch,
// does that rounding. P6 (the q-scale in fp32, after the bf16 read) and P7
// (bf16 `k`) are unchanged and sit where they always did.
//
// SLM: Ss[128][32] (16 KB) + vn[64][32] (8 KB) + gcv[64] / expg[64] (512 B)
//      = 24.5 KB.

#define HEADS 48
#define DIM 128
#define CONV_ROWS 10240
#define Q_OFF 0
#define K_OFF 2048
#define CT 64             /* the FLA intra-chunk size (PrefillScratch::kGdnChunk) */
#define SG 16             /* SIMD16: 16 subgroups of 16 lanes */
#define BAND_K 8          /* k-rows per subgroup band: 128 / 16 */
#define CHUNK_V 32        /* state columns per work-group: 128 / 4 */
#define VPW 2             /* v columns per work-item: 32 / 16 */
#define PPW 4             /* positions per work-item in stages 1-2: 64 / 16 */
#define WG_SCAN 256
#define Q_SCALE 0.08838834764831845f   /* 1/sqrt(128), applied to q in fp32 */

/* `pf_gdn_scan_dpas` only: the DPAS atom and the SLM row strides its fragments
   are read at. See that kernel's own block below. */
#define DM 8              /* the atom's M */
#define DK 16             /* the atom's K */
#define KBLKS (DIM / DK)  /* 8 k-blocks of 16 kdim */
#define PBLKS (CT / DK)   /* 4 position-blocks of 16 */
#define SB_LD 17          /* SbW / VNbW row stride in UINTS (34 ushorts): odd */
#define DT_LD 33          /* DtW row stride in UINTS (66 ushorts): odd */
#define A2_BLKS 10        /* the live (nt, kb) blocks of A2^T: kb <= nt */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// ---------------------------------------------------------------------------
// (3) `pf_gdn_scan_dpas_split` - D3's opt-in split-BF16 probe.
//
// `S`, the update `D` and (since 2026-09-23) the intra-chunk `A2` each have
// independent hi/lo BF16 DPAS chains, while the master state and every chain
// combination remain fp32.  This is approximate arithmetic, not an
// fp32-equivalence implementation; see docs/prefill-gdn-scan-split-2026-09-20.md
// for the original experiment and docs/prefill-gdn-scan-split-fix-2026-09-23.md
// for the fix below.
//
// **Why `A2` is split and `vn` is not (MEASURED, not argued).**  As shipped on
// 2026-09-20 this entry left `A2*vn` as D2's single-BF16 "controlled variable",
// both operands rounded once.  That term is a DIRECT additive contributor to
// `o` (the formula ~line 99), and `o` is what every later layer reads, so it
// fails the golden gate: `prefill_gate_l0_test` 92/93 with the code prompt's
// L60 `gdn_state` cosine at 0.996344994 against a > 0.999 bar (device 0,
// 2026-09-23, `$HOME/split-dev0.log`).  Four throwaway diagnostic entries then
// computed that one term in scalar fp32 with each operand independently either
// fp32 or `bf16f(rne_bf16(.))`, changing nothing else - a 2x2 on the same gate
// (commit 694b724, removed after it was recorded; logs `$HOME/fix-diag-*.log`):
//
//   A2 bf16, vn bf16   code L60 0.997517446   92/93   (control: reproduces it)
//   A2 bf16, vn fp32   code L60 0.997517446   92/93   (vn changes NOTHING)
//   A2 fp32, vn bf16   code L60 0.999189068   93/93
//   A2 fp32, vn fp32   code L60 0.999189068   93/93
//
// The two fp32-`A2` rows are the vector kernel's own cosines to nine digits.
// `A2`'s rounding is the entire error and `vn`'s is invisible in it, so `A2`
// gets the limbs and `vn` keeps its single BF16 - two DPAS chains where the
// prescribed both-operand split would have cost three, 4,352 B less SLM and
// 2.5 fewer DPAS per work-item per sub-chunk.  `A2_hi*vn + A2_lo*vn` is exact
// in each product (bf16 x bf16 -> fp32) and the two chains are combined once,
// in fp32, after the last position block - the treatment `S` and `D` get.
//
// SLM: SbHi/SbLo 17,408 + VNb 4,352 + DtHi/DtLo 8,448 + A2bHi/A2bLo 10,240
//      = 40,448 B.  DPAS per work-item per sub-chunk: 53.0 average (16 + 21 +
//      16; stage 2's A2 chains are 2(nt+1), i.e. 2..8 by subgroup).
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_SCAN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_scan_dpas_split(__global const ushort* restrict xb,
                                     __global const ushort* restrict w,
                                     __global const ushort* restrict u,
                                     __global const float* restrict A2,
                                     __global const float* restrict g_cum,
                                     __global float* restrict state,
                                     __global float* restrict gdn_o, uint c_count) {
  const uint h = get_group_id(0), c = get_group_id(1), lid = get_local_id(0);
  const uint sg = lid / SG, lane = lid % SG, kh = h / 3;
  const uint mt = sg >> 2, nt = sg & 3u, col0 = mt * DM, pos_l = nt * DK + lane;

  __local uint SbHiW[DIM * SB_LD], SbLoW[DIM * SB_LD];
  __local uint VNbW[CT * SB_LD];
  __local uint DtHiW[CHUNK_V * DT_LD], DtLoW[CHUNK_V * DT_LD];
  __local uint A2bHiW[A2_BLKS * SG * DM], A2bLoW[A2_BLKS * SG * DM];
  __local ushort* const SbHi = (__local ushort*)SbHiW;
  __local ushort* const SbLo = (__local ushort*)SbLoW;
  __local ushort* const DtHi = (__local ushort*)DtHiW;
  __local ushort* const DtLo = (__local ushort*)DtLoW;

  const size_t sbase =
      (size_t)h * DIM * DIM + (size_t)(sg * DM) * DIM + c * CHUNK_V + lane;
  float8 S0, S1;
  {
    float f0[DM], f1[DM];
#pragma unroll
    for (uint m = 0; m < DM; ++m) {
      f0[m] = state[sbase + (size_t)m * DIM];
      f1[m] = state[sbase + (size_t)m * DIM + SG];
    }
    S0 = vload8(0, f0);
    S1 = vload8(0, f1);
  }

  const uint nch = (c_count + CT - 1) / CT;
  for (uint t = 0; t < nch; ++t) {
    const uint base_m = t * CT, L = min((uint)CT, c_count - base_m), ilast = L - 1;
    const uint posc = min(pos_l, ilast);

    // S is split only at the DPAS boundary.  S0/S1 remain the fp32 master.
    {
      float f0[DM], f1[DM];
      vstore8(S0, 0, f0);
      vstore8(S1, 0, f1);
      __local ushort* restrict dhi = SbHi + (size_t)(sg * DM) * (2 * SB_LD) + lane;
      __local ushort* restrict dlo = SbLo + (size_t)(sg * DM) * (2 * SB_LD) + lane;
#pragma unroll
      for (uint m = 0; m < DM; ++m) {
        const ushort h0 = rne_bf16(f0[m]), h1 = rne_bf16(f1[m]);
        dhi[(size_t)m * (2 * SB_LD)] = h0;
        dhi[(size_t)m * (2 * SB_LD) + SG] = h1;
        dlo[(size_t)m * (2 * SB_LD)] = rne_bf16(f0[m] - bf16f(h0));
        dlo[(size_t)m * (2 * SB_LD) + SG] = rne_bf16(f1[m] - bf16f(h1));
      }
    }
    {
      __global const float* restrict At = A2 + ((size_t)(t * HEADS + h) * CT) * CT;
      for (uint p = lid; p < A2_BLKS * SG * DM; p += WG_SCAN) {
        const uint blk = p / (SG * DM), rest = p - blk * (SG * DM);
        const uint ln = rest / DM, r = rest - ln * DM;
        const uint bnt = blk < 1 ? 0u : (blk < 3 ? 1u : (blk < 6 ? 2u : 3u));
        const uint bkb = blk - ((bnt * (bnt + 1)) >> 1);
        const float2 v = vload2(0, At + (size_t)(bnt * DK + ln) * CT + bkb * DK + 2 * r);
        const ushort h0 = rne_bf16(v.s0), h1 = rne_bf16(v.s1);
        A2bHiW[p] = ((uint)h1 << 16) | (uint)h0;
        A2bLoW[p] = ((uint)rne_bf16(v.s1 - bf16f(h1)) << 16) |
                    (uint)rne_bf16(v.s0 - bf16f(h0));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float gcp = g_cum[(size_t)(base_m + posc) * HEADS + h];
    const float gl = g_cum[(size_t)(base_m + ilast) * HEADS + h];

    // 1. Two separate chains preserve S's BF16 residual until the fp32 sum.
    {
      float8 hi = (float8)(0.0f), lo = (float8)(0.0f);
      __global const uint* restrict wp =
          (__global const uint*)(w + ((size_t)(base_m + posc) * HEADS + h) * DIM);
      for (uint kb = 0; kb < KBLKS; ++kb) {
        const int8 b = as_int8(vload8(0, wp + kb * (DK / 2)));
        hi = intel_sub_group_bf16_bf16_matrix_mad_k16(
            as_short8(vload4(0, SbHiW + (kb * DK + lane) * SB_LD + mt * 4)), b, hi);
        lo = intel_sub_group_bf16_bf16_matrix_mad_k16(
            as_short8(vload4(0, SbLoW + (kb * DK + lane) * SB_LD + mt * 4)), b, lo);
      }
      const uint4 uw = vload4(0, (__global const uint*)(
          u + ((size_t)(base_m + posc) * HEADS + h) * DIM + c * CHUNK_V + col0));
      ushort ua[DM], vb[DM];
      float hf[DM], lf[DM];
      vstore8(as_ushort8(uw), 0, ua);
      vstore8(hi, 0, hf);
      vstore8(lo, 0, lf);
      const bool live = pos_l < L;
      const float sc = exp(gl - gcp);
      __local ushort* restrict dthi = DtHi + (size_t)col0 * (2 * DT_LD) + pos_l;
      __local ushort* restrict dtlo = DtLo + (size_t)col0 * (2 * DT_LD) + pos_l;
#pragma unroll
      for (uint m = 0; m < DM; ++m) {
        const float vnv = live ? (bf16f(ua[m]) - (hf[m] + lf[m])) : 0.0f;
        const float d = vnv * sc;
        const ushort dh = rne_bf16(d);
        vb[m] = rne_bf16(vnv);  // controlled D2 A2*vn operand
        dthi[(size_t)m * (2 * DT_LD)] = dh;
        dtlo[(size_t)m * (2 * DT_LD)] = rne_bf16(d - bf16f(dh));
      }
      vstore4(as_uint4(vload8(0, vb)), 0, VNbW + pos_l * SB_LD + mt * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Q.S is a split pair, and so is A2 - the operand the 2026-09-23 2x2
    //    above found responsible for the whole gate failure. `vn` stays single
    //    BF16 because moving it to fp32 changed no printed cosine at all.
    {
      float8 hi = (float8)(0.0f), lo = (float8)(0.0f);
      __global const uint* restrict qp = (__global const uint*)(
          xb + (size_t)(base_m + posc) * CONV_ROWS + Q_OFF + kh * DIM);
      for (uint kb = 0; kb < KBLKS; ++kb) {
        const int8 b = as_int8(vload8(0, qp + kb * (DK / 2)));
        hi = intel_sub_group_bf16_bf16_matrix_mad_k16(
            as_short8(vload4(0, SbHiW + (kb * DK + lane) * SB_LD + mt * 4)), b, hi);
        lo = intel_sub_group_bf16_bf16_matrix_mad_k16(
            as_short8(vload4(0, SbLoW + (kb * DK + lane) * SB_LD + mt * 4)), b, lo);
      }
      float8 o = (hi + lo) * (Q_SCALE * exp(gcp));
      // The high chain accumulates on top of the q.S term; the low chain starts
      // at zero so A2's residual is not swamped by it, and they meet once below.
      float8 olo = (float8)(0.0f);
      const uint blk0 = (nt * (nt + 1)) >> 1;
      for (uint kb = 0; kb <= nt; ++kb) {
        const short8 a = as_short8(vload4(0, VNbW + (kb * DK + lane) * SB_LD + mt * 4));
        const int8 bh = as_int8(vload8(0, A2bHiW + (blk0 + kb) * (SG * DM) + lane * DM));
        const int8 bl = as_int8(vload8(0, A2bLoW + (blk0 + kb) * (SG * DM) + lane * DM));
        o = intel_sub_group_bf16_bf16_matrix_mad_k16(a, bh, o);
        olo = intel_sub_group_bf16_bf16_matrix_mad_k16(a, bl, olo);
      }
      o += olo;
      if (pos_l < L)
        vstore8(o, 0, gdn_o + ((size_t)(base_m + pos_l) * HEADS + h) * DIM +
                            c * CHUNK_V + col0);
    }

    // 3. The high chain begins with decayed fp32 master state; the low chain
    // begins at zero.  Combine exactly once after all position blocks.
    {
      const float dl = exp(gl);
      float8 lo0 = (float8)(0.0f), lo1 = (float8)(0.0f);
      S0 *= dl;
      S1 *= dl;
      __global const ushort* restrict kp = xb + K_OFF + kh * DIM + sg * DM;
      for (uint kb = 0; kb < PBLKS; ++kb) {
        const uint p = min(kb * DK + lane, ilast);
        const short8 a = as_short8(vload4(0, (__global const uint*)(
            kp + (size_t)(base_m + p) * CONV_ROWS)));
        const int8 b0h = as_int8(vload8(0, DtHiW + (size_t)lane * DT_LD + kb * (DK / 2)));
        const int8 b1h = as_int8(vload8(0, DtHiW + (size_t)(SG + lane) * DT_LD + kb * (DK / 2)));
        const int8 b0l = as_int8(vload8(0, DtLoW + (size_t)lane * DT_LD + kb * (DK / 2)));
        const int8 b1l = as_int8(vload8(0, DtLoW + (size_t)(SG + lane) * DT_LD + kb * (DK / 2)));
        S0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b0h, S0);
        S1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b1h, S1);
        lo0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b0l, lo0);
        lo1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b1l, lo1);
      }
      S0 += lo0;
      S1 += lo1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  {
    float f0[DM], f1[DM];
    vstore8(S0, 0, f0);
    vstore8(S1, 0, f1);
#pragma unroll
    for (uint m = 0; m < DM; ++m) {
      state[sbase + (size_t)m * DIM] = f0[m];
      state[sbase + (size_t)m * DIM + SG] = f1[m];
    }
  }
}

__attribute__((reqd_work_group_size(WG_SCAN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_scan(__global const ushort* restrict xb,
                          __global const ushort* restrict w,
                          __global const ushort* restrict u,
                          __global const float* restrict A2,
                          __global const float* restrict g_cum,
                          __global float* restrict state,
                          __global float* restrict gdn_o, uint c_count) {
  const uint h = get_group_id(0);          // v-head
  const uint c = get_group_id(1);          // state-column chunk
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;              // owns k-rows [8*sgid, 8*sgid+8)
  const uint lane = lid % SG;              // owns columns 32c+lane and 32c+lane+16
  const uint kh = h / 3;                   // repeat_interleave(., 3)
  const uint col0 = c * CHUNK_V + lane, col1 = col0 + SG;
  const uint i0 = sgid * PPW;              // this subgroup's four positions

  __local float Ss[DIM][CHUNK_V];          // the chunk-START state tile
  __local float vn[CT][CHUNK_V];
  __local float gcv[CT], expg[CT];

  // The work-item's 8x2 state tile into registers (gdn_step.cl:273-278).
  const size_t sbase = (size_t)h * DIM * DIM + (size_t)(sgid * BAND_K) * DIM + col0;
  float S[BAND_K][VPW];
  for (uint kk = 0; kk < BAND_K; ++kk) {
    S[kk][0] = state[sbase + (size_t)kk * DIM];
    S[kk][1] = state[sbase + (size_t)kk * DIM + SG];
  }

  const uint nch = (c_count + CT - 1) / CT;
  for (uint t = 0; t < nch; ++t) {
    const uint base_m = t * CT;
    const uint L = min((uint)CT, c_count - base_m);
    const uint ilast = L - 1;

    // --- stage the gate, and the chunk-START S tile into SLM ------------------
    if (lid < L) {
      const float g = g_cum[(size_t)(base_m + lid) * HEADS + h];
      gcv[lid] = g;
      expg[lid] = exp(g);
    }
#pragma unroll
    for (uint kk = 0; kk < BAND_K; ++kk) {
      Ss[sgid * BAND_K + kk][lane] = S[kk][0];
      Ss[sgid * BAND_K + kk][lane + SG] = S[kk][1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float gl = gcv[L - 1];

    // --- 1. vn[i][x] = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x] -----------
    // Four positions x two columns per work-item; ascending k, one fma chain
    // per output. The `min(., ilast)` clamp keeps the w reads of a short final
    // sub-chunk in bounds; those lanes' results are simply not stored.
    {
      __global const ushort* restrict wp[PPW];
      float a0[PPW], a1[PPW];
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        wp[p] = w + ((size_t)(base_m + min(i0 + p, ilast)) * HEADS + h) * DIM;
        a0[p] = 0.0f;
        a1[p] = 0.0f;
      }
      for (uint k = 0; k < DIM; ++k) {          // ascending k over all 128
        const float s0 = Ss[k][lane], s1 = Ss[k][lane + SG];
#pragma unroll
        for (uint p = 0; p < PPW; ++p) {
          const float wv = bf16f(wp[p][k]);
          a0[p] = fma(s0, wv, a0[p]);
          a1[p] = fma(s1, wv, a1[p]);
        }
      }
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        const uint i = i0 + p;
        if (i < L) {
          const size_t ub = ((size_t)(base_m + i) * HEADS + h) * DIM;
          vn[i][lane] = bf16f(u[ub + col0]) - a0[p];
          vn[i][lane + SG] = bf16f(u[ub + col1]) - a1[p];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // vn is now live for every position

    // --- 2. o[i][x], from the CHUNK-START S ----------------------------------
    {
      __global const float* restrict At = A2 + ((size_t)(t * HEADS + h) * CT) * CT;
      __global const ushort* restrict qp[PPW];
      float a0[PPW], a1[PPW];
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        qp[p] = xb + (size_t)(base_m + min(i0 + p, ilast)) * CONV_ROWS + Q_OFF + kh * DIM;
        a0[p] = 0.0f;
        a1[p] = 0.0f;
      }
      for (uint k = 0; k < DIM; ++k) {          // ascending k over all 128
        const float s0 = Ss[k][lane], s1 = Ss[k][lane + SG];
#pragma unroll
        for (uint p = 0; p < PPW; ++p) {
          const float qv = bf16f(qp[p][k]) * Q_SCALE;   // P6
          a0[p] = fma(qv, s0, a0[p]);
          a1[p] = fma(qv, s1, a1[p]);
        }
      }
#pragma unroll
      for (uint p = 0; p < PPW; ++p) {
        const uint i = i0 + p;
        if (i < L) {
          float o0 = a0[p] * expg[i], o1 = a1[p] * expg[i];
          __global const float* restrict Ai = At + (size_t)i * CT;
          for (uint j = 0; j <= i; ++j) {              // j ascending, explicit fma
            const float a = Ai[j];
            o0 = fma(a, vn[j][lane], o0);
            o1 = fma(a, vn[j][lane + SG], o1);
          }
          __global float* restrict op =
              gdn_o + ((size_t)(base_m + i) * HEADS + h) * DIM + c * CHUNK_V;
          op[lane] = o0;                               // P10: fp32
          op[lane + SG] = o1;
        }
      }
    }

    // --- 3. the state update. THIS is the reassociation named in the header --
    // Unchanged from L1-core: already per-work-item, no barriers, S in
    // registers. `k[i][kk]` is subgroup-uniform and `vn[i][.]` is two SLM
    // reads per position for 16 fma.
    const float dl = exp(gl);
    for (uint kk = 0; kk < BAND_K; ++kk) {
      S[kk][0] *= dl;
      S[kk][1] *= dl;
    }
    for (uint i = 0; i < L; ++i) {                    // i ascending
      const float sc = exp(gl - gcv[i]);
      const float d0 = vn[i][lane] * sc, d1 = vn[i][lane + SG] * sc;
      const size_t kbase = (size_t)(base_m + i) * CONV_ROWS + K_OFF + kh * DIM;
      for (uint kk = 0; kk < BAND_K; ++kk) {          // kk ascending
        const float kfv = bf16f(xb[kbase + sgid * BAND_K + kk]);            // P7
        S[kk][0] = fma(kfv, d0, S[kk][0]);
        S[kk][1] = fma(kfv, d1, S[kk][1]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next chunk reuses Ss, gcv, expg, vn
  }

  // The tile goes back once, after every chunk of this call (gdn_step.cl:378-381).
  for (uint kk = 0; kk < BAND_K; ++kk) {
    state[sbase + (size_t)kk * DIM] = S[kk][0];
    state[sbase + (size_t)kk * DIM + SG] = S[kk][1];
  }
}

// ---------------------------------------------------------------------------
// (2) `pf_gdn_scan_dpas` - ruling A28 option (c). BUILT, MEASURED, REVERTED,
//     and kept here because a measured design is worth more in the tree than in
//     a commit message. **Nothing launches it**: `gdn.cc` binds `pf_gdn_scan`
//     above, and the header block at the top of this file records why.
//
//     Measured: **0.8491 ms per GDN layer per chunk** (3.997x the vector
//     kernel's 3.3937, 11.38 TFLOP/s on 9.66 GFLOP) against a pre-registered
//     bar of <= 1.0 - the time bar was HIT. `dpas.8x8` emitted (28 of them),
//     SIMD16, 128 GRF, 22,400 B of SLM, no spill. It fell on the token gate:
//     `prefill_gate_rtn_test` 92/93 determined rows and Vishva's `code` prompt
//     at `gdn_state` cosine 0.998499499 (L60) against a pre-registered 0.999
//     and this session's control of 0.999780657.
//
// --- the builtin, and its three fragment layouts (MEASURED) ----------------
// `intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc)` is an
// unconditional IGC builtin that lowers to `dpas.8x8`: bf16 x bf16 -> fp32,
// M = 8, K = 16, N = 16, sub-group 16. Its fragments, confirmed bit-exactly by a
// standalone probe against two deliberately-wrong controls
// (docs/prefill-gdn-scan-dpas-2026-09-05.md §3.1):
//
//   A (M x K = 8 x 16)   lane `l` holds A[0..7][l]  -- one K COLUMN per lane,
//                        register index = the M index.
//   B (K x N = 16 x 16)  lane `n` holds column `n`, VNNI-packed along K:
//                        int r = (B[2r+1][n] << 16) | B[2r][n].
//                        **Even k in the LOW half** -- which is exactly what a
//                        little-endian `uint` read of a k-contiguous bf16 row
//                        gives for free, and the whole B-operand design rests
//                        on that.
//   C (M x N = 8 x 16)   lane `n` holds C[0..7][n], register index = the M index.
//
// --- D1 is EMPTY; this is D2 ----------------------------------------------
// Every stage has exactly one bf16 operand and one fp32 one, and the fp32 one is
// always the state or an intermediate built from it:
//
//   1  vn[64][32] = u - W[64][128] . S[128][32]      W, u bf16 | **S fp32**
//   2  o [64][32] = (Q.S) * expg + A2[64][64] . vn   Q bf16    | **S, A2, vn fp32**
//   3  S [128][32] <- S*dl + K^T[128][64] . D[64][32]  K bf16  | **D fp32**
//
// So "DPAS only where both operands are already bf16" describes no work at all
// unless `S` is CARRIED in bf16, which is a truncation of the model's recurrent
// memory rather than an optimisation. D2 rounds `S`, `vn`, `D` and `A2` to bf16
// **at the DPAS inputs** and keeps the fp32 master state.
//
// --- the grid, unchanged --------------------------------------------------
// (48 v-heads, 4 state-column chunks) = 192 work-groups of 256 = 16 subgroups of
// 16 lanes, as above. At 256 work-items an Xe-core holds 4 work-groups (64 of
// its 64 thread slots) and the launch needs 6 per core, so this is **1.5 waves
// -- exactly `pf_gdn_scan`'s wave structure**, and the two kernels therefore
// differ in one variable. Every grid that removes the 1.5 doubles or quadruples
// the per-lane accumulator count, and the state alone is already 16 fp32 per
// work-item.
//
// --- the orientation, which is the one non-obvious choice ------------------
// **Stages 1 and 2 are computed TRANSPOSED and stage 3 is not**, because that is
// what makes every fragment a single contiguous per-lane load:
//
//   * stage 1 as `vn^T[32][64] = u^T - S^T[32][128] . W^T[128][64]`:
//     A = S^T (M = 32 state columns, K = 128 kdim), B = W^T (K = kdim,
//     N = 64 positions). The B fragment is then `W[position][16 consecutive
//     kdim]`, which is **already the VNNI order in memory** -- one `vload8` of
//     `uint` straight off `w`, no repack anywhere. The A fragment is 8
//     consecutive state COLUMNS at one kdim row: one `vload4` of `uint` out of
//     `Sb`.
//   * stage 2 is the same shape for `Q.S`, plus `o^T += vn^T . A2^T`, whose A
//     operand is `vn^T` -- read out of `VNb` in exactly the layout stage 1 wrote
//     it -- and whose B operand is `A2^T`, VNNI-packed once per sub-chunk.
//   * stage 3 in the natural orientation: A = K^T (M = kdim, K = position) is
//     `xb`'s k row for one position, 8 consecutive kdim per lane, again one
//     vector load; B = D (K = position, N = column) needs D TRANSPOSED in SLM,
//     which costs 8 scattered 16-bit SLM stores per work-item and buys a
//     contiguous VNNI-ready `vload8`. The transposed orientation for stage 3 was
//     priced and rejected: its B operand would put positions on the K axis of
//     `k`, needing `k` staged transposed in 16 KB more SLM, which drops
//     residency to 3 work-groups per Xe-core = 2 waves.
//
// --- tile ownership --------------------------------------------------------
//   stages 1-2   C tile = 8 columns x 16 positions; 4 x 4 = 16 tiles, ONE per
//                subgroup: `mt = sg >> 2` owns columns `8mt..8mt+7`,
//                `nt = sg & 3` owns positions `16nt..16nt+15`; lane `n` is
//                position `16nt+n`.
//   stage 3      C tile = 8 kdim x 16 columns; 16 x 2 = 32 tiles, TWO per
//                subgroup: M-tile `sg` (kdim `8sg..8sg+7`), both N-tiles; lane
//                `n` is column `16*n3 + n`.
//
// **Stage 3's C fragments ARE the state**: 2 x float8 = 16 fp32 per work-item,
// exactly what `pf_gdn_scan` holds in registers, and the global load and store
// are the same 16 coalesced 64 B lines.
//
// --- SLM: 22,400 B (measured), below `pf_gdn_scan`'s 25,088 ----------------
//   SbW   uint[128][17]   bf16 S[k][x]                          8,704 B
//   VNbW  uint[64][17]    bf16 vn[i][x]                         4,352 B
//   DtW   uint[32][33]    bf16 D[i][x], TRANSPOSED to [x][i]    4,224 B
//   A2bW  uint[10][16][8] bf16 A2^T, VNNI, the 10 live blocks   5,120 B
// The three row strides are ODD in dwords (17, 17, 33) on purpose: the 16 lanes
// of a fragment read one row each, so an odd dword stride puts them on 16
// distinct SLM banks. 6 of A2's 16 (nt, kb) blocks are entirely above the
// diagonal and are neither stored nor multiplied.
//
// **Barriers per sub-chunk: 3**, as `pf_gdn_scan`.
// **`dpas.8x8` per work-item per sub-chunk: 26.5** = 8 + 8 + 2.5 + 8.
//
// --- the five new bf16 rounding points, and what each reaches --------------
//   R1  S -> bf16 for stage 1's w.S    -> vn, so gdn_state AND gdn_o
//   R1' the same Sb for stage 2's q.S  -> gdn_o only
//   R2  vn -> bf16, stage 2's A operand-> gdn_o only
//   R3  vn*exp(gl-gc) -> bf16, stage 3's B operand -> **gdn_state directly**
//   R4  A2 -> bf16, stage 2's B operand-> gdn_o only
// Decode has none of these and neither does the chunked CPU reference. The
// master state stays fp32: stage 3 accumulates in fp32 out of DPAS, the register
// state and `gdn_state` are fp32, and nothing is carried in bf16.
//
// **One rounding point MOVES, in the safe direction.** P6 -- the `1/sqrt(128)`
// q-scale -- was applied PER TERM inside the k-loop (128 fp32 multiplies per
// output). The DPAS A operand must be the bf16 word itself, so the scale is
// folded onto the ACCUMULATED dot: one fp32 multiply where there were 128. It
// stays in fp32 and after the bf16 read, exactly as A6 requires, and it REMOVES
// roundings.
//
// **The short final sub-chunk.** Every global read whose row could pass `L` is
// clamped with `min(., L-1)` -- always a live row -- and its result is not
// stored; `VNb` and `Dt` are written **exactly 0.0f** for positions `>= L`,
// which is what makes stage 3's clamped `K^T` reads contribute exactly nothing.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(WG_SCAN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_scan_dpas(__global const ushort* restrict xb,
                               __global const ushort* restrict w,
                               __global const ushort* restrict u,
                               __global const float* restrict A2,
                               __global const float* restrict g_cum,
                               __global float* restrict state,
                               __global float* restrict gdn_o, uint c_count) {
  const uint h = get_group_id(0);          // v-head
  const uint c = get_group_id(1);          // state-column chunk
  const uint lid = get_local_id(0);
  const uint sg = lid / SG;
  const uint lane = lid % SG;
  const uint kh = h / 3;                   // repeat_interleave(., 3)

  // Stages 1-2: this subgroup's 8-column x 16-position tile of vn^T / o^T.
  const uint mt = sg >> 2;                 // columns 8mt .. 8mt+7 of the WG's 32
  const uint nt = sg & 3u;                 // positions 16nt .. 16nt+15
  const uint col0 = mt * DM;
  const uint pos_l = nt * DK + lane;       // this lane's position in the sub-chunk

  __local uint SbW[DIM * SB_LD];           // bf16 S[k][x], 34-ushort rows
  __local uint VNbW[CT * SB_LD];           // bf16 vn[i][x], 34-ushort rows
  __local uint DtW[CHUNK_V * DT_LD];       // bf16 D[i][x] TRANSPOSED, 66-ushort rows
  __local uint A2bW[A2_BLKS * SG * DM];    // bf16 A2^T, VNNI, 10 live blocks
  __local ushort* const Sb = (__local ushort*)SbW;
  __local ushort* const Dt = (__local ushort*)DtW;

  // Stage 3's C fragments ARE the state: M-tile `sg` = kdim [8sg, 8sg+8), both
  // N-tiles, lane `n` = column `16*n3 + n` (gdn_step.cl:273-278's 16 fp32).
  const size_t sbase =
      (size_t)h * DIM * DIM + (size_t)(sg * DM) * DIM + c * CHUNK_V + lane;
  float8 S0, S1;
  {
    float f0[DM], f1[DM];
#pragma unroll
    for (uint m = 0; m < DM; ++m) {
      f0[m] = state[sbase + (size_t)m * DIM];
      f1[m] = state[sbase + (size_t)m * DIM + SG];
    }
    S0 = vload8(0, f0);
    S1 = vload8(0, f1);
  }

  const uint nch = (c_count + CT - 1) / CT;
  for (uint t = 0; t < nch; ++t) {
    const uint base_m = t * CT;
    const uint L = min((uint)CT, c_count - base_m);
    const uint ilast = L - 1;
    const uint posc = min(pos_l, ilast);        // always a live row

    // --- stage 0a: the chunk-START S into SLM as bf16 -----------------------
    {
      float f0[DM], f1[DM];
      vstore8(S0, 0, f0);
      vstore8(S1, 0, f1);
      __local ushort* restrict d = Sb + (size_t)(sg * DM) * (2 * SB_LD) + lane;
#pragma unroll
      for (uint m = 0; m < DM; ++m) {
        d[(size_t)m * (2 * SB_LD)] = rne_bf16(f0[m]);            // column `lane`
        d[(size_t)m * (2 * SB_LD) + SG] = rne_bf16(f1[m]);       // column `lane+16`
      }
    }

    // --- stage 0b: A2 into SLM as bf16, VNNI-packed along j ------------------
    // Only the 10 blocks with `kb <= nt` exist: A2[i][j] is exactly 0.0f for
    // j > i, so blocks entirely above the diagonal are neither stored nor
    // multiplied. Rows and columns past L are exactly 0.0f in global already.
    {
      __global const float* restrict At = A2 + ((size_t)(t * HEADS + h) * CT) * CT;
      for (uint p = lid; p < A2_BLKS * SG * DM; p += WG_SCAN) {
        const uint blk = p / (SG * DM), rest = p - blk * (SG * DM);
        const uint ln = rest / DM, r = rest - ln * DM;
        const uint bnt = blk < 1 ? 0u : (blk < 3 ? 1u : (blk < 6 ? 2u : 3u));
        const uint bkb = blk - ((bnt * (bnt + 1)) >> 1);
        const float2 v =
            vload2(0, At + (size_t)(bnt * DK + ln) * CT + bkb * DK + 2 * r);
        A2bW[p] = ((uint)rne_bf16(v.s1) << 16) | (uint)rne_bf16(v.s0);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float gcp = g_cum[(size_t)(base_m + posc) * HEADS + h];
    const float gl = g_cum[(size_t)(base_m + ilast) * HEADS + h];

    // --- 1. vn^T[32][64] = u^T - S^T[32][128] . W^T[128][64] ----------------
    // A = S^T: lane `l` holds S[16kb+l][8mt .. 8mt+7], one `vload4` of uint.
    // B = W^T: lane `n` holds w[pos][16kb .. 16kb+15], already VNNI in memory.
    {
      float8 acc = (float8)(0.0f);
      __global const uint* restrict wp =
          (__global const uint*)(w + ((size_t)(base_m + posc) * HEADS + h) * DIM);
      for (uint kb = 0; kb < KBLKS; ++kb) {
        const short8 a = as_short8(vload4(0, SbW + (kb * DK + lane) * SB_LD + mt * 4));
        const int8 b = as_int8(vload8(0, wp + kb * (DK / 2)));
        acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
      }

      const uint4 uw = vload4(
          0, (__global const uint*)(u + ((size_t)(base_m + posc) * HEADS + h) * DIM +
                                    c * CHUNK_V + col0));
      ushort ua[DM];
      vstore8(as_ushort8(uw), 0, ua);
      float af[DM];
      vstore8(acc, 0, af);
      const bool live = pos_l < L;
      const float sc = exp(gl - gcp);
      ushort vb[DM];
      __local ushort* restrict dt = Dt + (size_t)col0 * (2 * DT_LD) + pos_l;
#pragma unroll
      for (uint m = 0; m < DM; ++m) {
        const float vnv = live ? (bf16f(ua[m]) - af[m]) : 0.0f;
        vb[m] = rne_bf16(vnv);                                   // R2
        dt[(size_t)m * (2 * DT_LD)] = rne_bf16(vnv * sc);        // R3
      }
      vstore4(as_uint4(vload8(0, vb)), 0, VNbW + pos_l * SB_LD + mt * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // VNb and Dt are live for every position

    // --- 2. o^T = (S^T . Q^T) * (Q_SCALE * expg) + vn^T . A2^T --------------
    {
      float8 o = (float8)(0.0f);
      __global const uint* restrict qp =
          (__global const uint*)(xb + (size_t)(base_m + posc) * CONV_ROWS + Q_OFF +
                                 kh * DIM);
      for (uint kb = 0; kb < KBLKS; ++kb) {
        const short8 a = as_short8(vload4(0, SbW + (kb * DK + lane) * SB_LD + mt * 4));
        const int8 b = as_int8(vload8(0, qp + kb * (DK / 2)));
        o = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, o);
      }
      o *= Q_SCALE * exp(gcp);              // P6, folded onto the accumulated dot
      const uint blk0 = (nt * (nt + 1)) >> 1;
      for (uint kb = 0; kb <= nt; ++kb) {   // subgroup-uniform: kb <= nt only
        const short8 a =
            as_short8(vload4(0, VNbW + (kb * DK + lane) * SB_LD + mt * 4));
        const int8 b = as_int8(vload8(0, A2bW + (blk0 + kb) * (SG * DM) + lane * DM));
        o = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, o);
      }
      if (pos_l < L)
        vstore8(o, 0,
                gdn_o + ((size_t)(base_m + pos_l) * HEADS + h) * DIM + c * CHUNK_V +
                    col0);                  // P10: fp32
    }

    // --- 3. S[128][32] <- S*dl + K^T[128][64] . D[64][32] -------------------
    // A = K^T: lane `l` holds xb's k row of position `16kb+l`, kdim 8sg..8sg+7.
    // B = D: lane `n` holds Dt[16*n3+n][16kb .. 16kb+15], VNNI-ready.
    {
      const float dl = exp(gl);
      S0 *= dl;
      S1 *= dl;
      __global const ushort* restrict kp = xb + K_OFF + kh * DIM + sg * DM;
      for (uint kb = 0; kb < PBLKS; ++kb) {
        const uint p = min(kb * DK + lane, ilast);
        const short8 a = as_short8(vload4(
            0, (__global const uint*)(kp + (size_t)(base_m + p) * CONV_ROWS)));
        const int8 b0 = as_int8(vload8(0, DtW + (size_t)lane * DT_LD + kb * (DK / 2)));
        const int8 b1 =
            as_int8(vload8(0, DtW + (size_t)(SG + lane) * DT_LD + kb * (DK / 2)));
        S0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b0, S0);
        S1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b1, S1);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // the next sub-chunk overwrites Sb/A2b/VNb/Dt
  }

  // The tile goes back once, after every chunk of this call (gdn_step.cl:378-381).
  {
    float f0[DM], f1[DM];
    vstore8(S0, 0, f0);
    vstore8(S1, 0, f1);
#pragma unroll
    for (uint m = 0; m < DM; ++m) {
      state[sbase + (size_t)m * DIM] = f0[m];
      state[sbase + (size_t)m * DIM + SG] = f1[m];
    }
  }
}
