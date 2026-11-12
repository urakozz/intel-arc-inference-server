// pf_gdn_scan_hl.cl - phase 2 Task 3 under operator decision D2 (2026-11-12): `pf_gdn_scan_dpas_split`
// (pf_gdn_scan.cl entry (3), the walk's scan) reading w and u as hi / lo bf16 planes - the fp32 Q3 / Q4 that
// pf_gdn_intra.cl's WU_HILO form writes - instead of single bf16 words. Two edits and nothing else:
//   * stage 1's W.S gets a third DPAS chain, S_hi x w_lo, into the low chain (S_lo x w_lo is below both);
//   * vn = (f32(u_hi) + f32(u_lo)) - (hi + lo).
// Stages 2 and 3, the state, A2 and every other operand are the split scan's, line for line. A new file so
// pf_gdn_scan.cl's binaries stay byte for byte (the plan's additive rule).

#ifndef GDN_V_HEADS
#define HEADS 48
#define DIM 128
#define CONV_ROWS 10240
#define Q_OFF 0
#define K_OFF 2048
#define V_PER_K 3         /* repeat_interleave(., 48 / 16) */
#else
#ifndef GDN_K_HEADS
#error "pf_gdn_scan: GDN_V_HEADS needs GDN_K_HEADS"
#endif
#define HEADS GDN_V_HEADS
#define DIM 128
#define CONV_ROWS ((2 * GDN_K_HEADS + GDN_V_HEADS) * DIM)
#define Q_OFF 0
#define K_OFF (GDN_K_HEADS * DIM)
#define V_PER_K (GDN_V_HEADS / GDN_K_HEADS)
#if GDN_V_HEADS % GDN_K_HEADS != 0
#error "pf_gdn_scan: v-heads must be a multiple of k-heads"
#endif
#endif
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
/* Open-work 3n: `pf_gdn_A2_hilo`'s packed A2 per (sub-chunk, head), in uints - the hi words of the
   A2_BLKS blocks, then the lo words. pf_gdn_wy.cl and src/runtime/prefill/gdn_a2_pack.h say the same. */
#define A2HL_LO (A2_BLKS * SG * DM)       /* 1,280: the lo words' offset */
#define A2HL_WORDS (2 * A2HL_LO)          /* 2,560 uints = 10,240 B */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((reqd_work_group_size(WG_SCAN, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_scan_hl(__global const ushort* restrict xb,
                             __global const ushort* restrict w,
                             __global const ushort* restrict wl,
                             __global const ushort* restrict u,
                             __global const ushort* restrict ul,
                                     __global const uint* restrict A2hl,
                                     __global const float* restrict g_cum,
                                     __global float* restrict state,
                                     __global float* restrict gdn_o, uint c_count) {
  const uint h = get_group_id(0), c = get_group_id(1), lid = get_local_id(0);
  const uint sg = lid / SG, lane = lid % SG, kh = h / V_PER_K;
  const uint mt = sg >> 2, nt = sg & 3u, col0 = mt * DM, pos_l = nt * DK + lane;

  __local uint SbHiW[DIM * SB_LD], SbLoW[DIM * SB_LD];
  __local uint VNbW[CT * SB_LD];
  __local uint DtHiW[CHUNK_V * DT_LD], DtLoW[CHUNK_V * DT_LD];
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
    barrier(CLK_LOCAL_MEM_FENCE);

    const float gcp = g_cum[(size_t)(base_m + posc) * HEADS + h];
    const float gl = g_cum[(size_t)(base_m + ilast) * HEADS + h];

    // 1. Two separate chains preserve S's BF16 residual until the fp32 sum.
    {
      float8 hi = (float8)(0.0f), lo = (float8)(0.0f);
      __global const uint* restrict wp =
          (__global const uint*)(w + ((size_t)(base_m + posc) * HEADS + h) * DIM);
      __global const uint* restrict wlp =
          (__global const uint*)(wl + ((size_t)(base_m + posc) * HEADS + h) * DIM);
      for (uint kb = 0; kb < KBLKS; ++kb) {
        const int8 b = as_int8(vload8(0, wp + kb * (DK / 2)));
        const int8 bl = as_int8(vload8(0, wlp + kb * (DK / 2)));
        const short8 sh = as_short8(vload4(0, SbHiW + (kb * DK + lane) * SB_LD + mt * 4));
        hi = intel_sub_group_bf16_bf16_matrix_mad_k16(sh, b, hi);
        lo = intel_sub_group_bf16_bf16_matrix_mad_k16(
            as_short8(vload4(0, SbLoW + (kb * DK + lane) * SB_LD + mt * 4)), b, lo);
        lo = intel_sub_group_bf16_bf16_matrix_mad_k16(sh, bl, lo);   // w's lo limb
      }
      const uint4 uw = vload4(0, (__global const uint*)(
          u + ((size_t)(base_m + posc) * HEADS + h) * DIM + c * CHUNK_V + col0));
      const uint4 ulw = vload4(0, (__global const uint*)(
          ul + ((size_t)(base_m + posc) * HEADS + h) * DIM + c * CHUNK_V + col0));
      ushort ua[DM], ula[DM], vb[DM];
      vstore8(as_ushort8(ulw), 0, ula);
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
        const float vnv = live ? ((bf16f(ua[m]) + bf16f(ula[m])) - (hf[m] + lf[m])) : 0.0f;
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
      __global const uint* restrict a2t =
          A2hl + (size_t)(t * HEADS + h) * A2HL_WORDS + (size_t)blk0 * (SG * DM) + lane * DM;
      for (uint kb = 0; kb <= nt; ++kb) {
        const short8 a = as_short8(vload4(0, VNbW + (kb * DK + lane) * SB_LD + mt * 4));
        const int8 bh = as_int8(vload8(0, a2t + kb * (SG * DM)));
        const int8 bl = as_int8(vload8(0, a2t + A2HL_LO + kb * (SG * DM)));
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

