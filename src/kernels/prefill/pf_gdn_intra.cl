// pf_gdn_intra.cl - phase 2 Task 3 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md):
// `gdn_chunk`'s intra-chunk WY stages in ONE launch - A, T = (I + A)^-1, W / U and A2 - one
// work-group per (v-head, 64-position sub-chunk), every sub-chunk in parallel. It replaces
// pf_gdn_A, pf_gdn_solve, pf_gdn_wu and pf_gdn_A2_hilo (pf_gdn_wy.cl) and writes the scan's
// inputs in their layouts - w, u bf16 [C][heads][128] and A2's packed hi / lo words
// (src/runtime/prefill/gdn_a2_pack.h) - so `pf_gdn_scan_dpas_split` reads them unchanged.
//
// The algebra is pf_gdn_wy.cl's header and tests/prefill/gdn_chunk_ref.h, unchanged:
//   A[i][j]  = beta[i] (k_i . k_j) exp(gc[i] - gc[j])          i > j
//   T        = (I + A)^-1
//   vb[j][x] = rne(v[j][x] beta[j])                             Q1
//   kb[j][k] = rne(k[j][k] beta[j] exp(gc[j]))                  Q2
//   u[i][x]  = rne(SUM_j T[i][j] vb[j][x])                      Q3
//   w[i][k]  = rne(SUM_j T[i][j] kb[j][k])                      Q4
//   A2[i][j] = (q_i . k_j) Q_SCALE exp(gc[i] - gc[j])           j <= i
// What moves is the arithmetic's ORDER and form, so the kernel is JUDGED (the f32 judge), not
// bitwise:
//   * A's and A2's dots are bf16 DPAS (k and q are bf16 words already: the operands are exact,
//     only the sum order moves); Q_SCALE multiplies the fp32 dot instead of every q term.
//   * T is solved in 16 x 16 blocks, fp32 throughout: the four diagonal blocks by forward
//     substitution (each column's chain is pf_gdn_solve's, term for term), the off-diagonal
//     blocks as T_ij = -T_ii (SUM_{j<=k<i} A_ik T_kj), fp32 vector FMAs.
//   * W / U are DPAS with T carried as a hi / lo bf16 pair (hi = rne(T), lo = rne(T - hi)),
//     two chains summed once in fp32; vb / kb are bf16 by Q1 / Q2, so they are exact operands.
//     No fp32 quantity is a single bf16 DPAS operand (the 2026-09-23 lesson, pf_gdn_scan.cl).
// The GDN state is not touched here (the scan owns it, fp32). Deterministic: no atomics, fixed
// orders.
//
// DPAS: `intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 c)`, SIMD16, M 8,
// K 16, N 16 (pf_gdn_scan.cl entry (2) has the measured fragment layouts): A - lane l holds
// A[0..7][l]; B - lane n holds column n, VNNI pairs along K (even k in the low half); C - lane n
// holds C[0..7][n].
//
// SLM 32,768 B (4 work-groups an Xe-core): As fp32 [64][64] (A; its strictly-upper 16 x 16
// blocks hold the solve's temporaries) and Ts fp32 [64][64] (T); then the hi / lo T words
// [64][TLD] replace them for the W / U stage.
//
// -DINTRA_DBG=1 (the probe's binary only) adds two fp32 [nch][heads][64][64] outputs, A and T,
// for tools/probe/probe_gdn_intra.cc's per-stage error against fp64.

#ifndef GDN_V_HEADS
#define HEADS 48
#define DIM 128
#define CONV_ROWS 10240
#define Q_OFF 0
#define K_OFF 2048
#define V_OFF 4096
#define V_PER_K 3
#else
#ifndef GDN_K_HEADS
#error "pf_gdn_intra: GDN_V_HEADS needs GDN_K_HEADS"
#endif
#define HEADS GDN_V_HEADS
#define DIM 128
#define CONV_ROWS ((2 * GDN_K_HEADS + GDN_V_HEADS) * DIM)
#define Q_OFF 0
#define K_OFF (GDN_K_HEADS * DIM)
#define V_OFF (2 * GDN_K_HEADS * DIM)
#define V_PER_K (GDN_V_HEADS / GDN_K_HEADS)
#if GDN_V_HEADS % GDN_K_HEADS != 0
#error "pf_gdn_intra: v-heads must be a multiple of k-heads"
#endif
#endif
#define CT 64                          /* the FLA intra-chunk size (PrefillScratch::kGdnChunk) */
#define Q_SCALE 0.08838834764831845f   /* 1/sqrt(128) */
#define SG 16
#define NSG 16                         /* sub-groups a work-group: 256 lanes */
#define TB 16                          /* T's block edge */
#define TLD 66                         /* the hi / lo T rows' stride in ushorts: 33 uints, odd */
#define A2_BLKS 10                     /* pf_gdn_scan.cl's packed A2 (gdn_a2_pack.h) */
#define A2HL_LO (A2_BLKS * 16 * 8)
#define A2HL_WORDS (2 * A2HL_LO)
#define NTILE 20                       /* the 8 x 16 tiles of A / A2 on or below the diagonal */
#ifndef INTRA_DBG
#define INTRA_DBG 0
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

// Tile ti (0..19) of the 8 x 4 grid of 8-row x 16-column tiles with a row i >= some column j:
// M-tile mi = rows 8mi.., N-tile nj = columns 16nj.., nj <= (8mi + 7) / 16.
inline uint2 tile_of(uint ti) {
  // cumulative counts per mi: 1 1 2 2 3 3 4 4 -> starts 0 1 2 4 6 9 12 16
  const uint mi = ti < 1 ? 0 : ti < 2 ? 1 : ti < 4 ? 2 : ti < 6 ? 3 : ti < 9 ? 4 : ti < 12 ? 5 : ti < 16 ? 6 : 7;
  const uint start = mi == 0 ? 0 : mi == 1 ? 1 : mi == 2 ? 2 : mi == 3 ? 4 : mi == 4 ? 6 : mi == 5 ? 9 : mi == 6 ? 12 : 16;
  return (uint2)(mi, ti - start);
}

__attribute__((reqd_work_group_size(NSG * SG, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gdn_intra(__global const ushort* restrict xb,
                           __global const float* restrict g_cum,
                           __global const float* restrict beta,
                           __global ushort* restrict w,
                           __global ushort* restrict u,
                           __global uint* restrict A2hl,
                           uint c_count
#if INTRA_DBG
                           , __global float* restrict Adbg, __global float* restrict Tdbg
#endif
                           ) {
#pragma OPENCL FP_CONTRACT OFF
  const uint h = get_group_id(0), t = get_group_id(1);
  const uint kh = h / V_PER_K;
  const uint base_m = t * CT;
  const uint L = min((uint)CT, c_count - base_m);
  const uint lid = get_local_id(0);
  const uint sg = lid / SG, lane = lid % SG;

  __local float As[CT * CT];
  __local float Ts[CT * CT];

  // ---- 1. A and A2: bf16 DPAS over the 128 k, the gate in fp32 --------------------------
  {
    __global ushort* restrict Ht = (__global ushort*)(A2hl + (size_t)(t * HEADS + h) * A2HL_WORDS);
    for (uint ti = sg; ti < NTILE; ti += NSG) {
      const uint2 tl = tile_of(ti);
      const uint i0 = tl.x * 8, j0 = tl.y * 16;
      // B: k row j0 + lane (clamped to a live row), 16 k a block as 8 VNNI uints.
      const uint jr = min(j0 + lane, L - 1);
      __global const uint* restrict kbrow =
          (__global const uint*)(xb + (size_t)(base_m + jr) * CONV_ROWS + K_OFF + kh * DIM);
      size_t rk[8], rq[8];
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const size_t row = (size_t)(base_m + min(i0 + r, L - 1)) * CONV_ROWS;
        rk[r] = row + K_OFF + kh * DIM + lane;
        rq[r] = row + Q_OFF + kh * DIM + lane;
      }
      float8 accA = (float8)(0.0f), accQ = (float8)(0.0f);
#pragma unroll 2
      for (uint kb = 0; kb < DIM / 16; ++kb) {
        const int8 b = as_int8(vload8(0, kbrow + kb * 8));
        short ak[8], aq[8];
#pragma unroll
        for (uint r = 0; r < 8; ++r) {
          ak[r] = (short)xb[rk[r] + kb * 16];
          aq[r] = (short)xb[rq[r] + kb * 16];
        }
        accA = intel_sub_group_bf16_bf16_matrix_mad_k16(vload8(0, ak), b, accA);
        accQ = intel_sub_group_bf16_bf16_matrix_mad_k16(vload8(0, aq), b, accQ);
      }
      float fa[8], fq[8];
      vstore8(accA, 0, fa);
      vstore8(accQ, 0, fq);
      const uint j = j0 + lane;
      const float gj = j < L ? g_cum[(size_t)(base_m + j) * HEADS + h] : 0.0f;
      const uint bkb = j0 / 16;
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const uint i = i0 + r;
        const bool live = i < L && j < L;
        const float gi = i < L ? g_cum[(size_t)(base_m + i) * HEADS + h] : 0.0f;
        const float bi = i < L ? beta[(size_t)(base_m + i) * HEADS + h] : 0.0f;
        const float e = exp(gi - gj);
        // NOTE the mask pair (pf_gdn_wy.cl's header): A strict, A2 with the diagonal.
        As[i * CT + j] = (live && i > j) ? bi * fa[r] * e : 0.0f;
        const float v = (live && j <= i) ? (fq[r] * Q_SCALE) * e : 0.0f;
        const ushort hv = rne_bf16(v);
        const uint bnt = i / 16;
        const size_t p = (size_t)(((bnt * (bnt + 1)) >> 1) + bkb) * 256 + (i % 16) * 16 + (j % 16);
        Ht[p] = hv;
        Ht[2 * A2HL_LO + p] = rne_bf16(v - bf16f(hv));
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#if INTRA_DBG
  {
    __global float* restrict Ad = Adbg + ((size_t)(t * HEADS + h) * CT) * CT;
    for (uint p = lid; p < CT * CT; p += NSG * SG)
      Ad[p] = (p % CT) / TB <= (p / CT) / TB ? As[p] : 0.0f;
  }
#endif

  // ---- 2. T = (I + A)^-1 in 16 x 16 blocks ----------------------------------------------
  // (a) the diagonal blocks: sub-group b owns block b, lane c its column c. The column's chain
  //     is pf_gdn_solve's: T[r][c] = -A[r][c] + SUM_{c<l<r} (-A[r][l]) T[l][c], l ascending.
  if (sg < CT / TB) {
    const uint b0 = sg * TB, c = lane;
    float tc[TB];
#pragma unroll
    for (uint r = 0; r < TB; ++r) {
      float acc = r == c ? 1.0f : 0.0f;
      if (r > c) {
        acc = -As[(b0 + r) * CT + b0 + c];
#pragma unroll
        for (uint l = 0; l < r; ++l)
          if (l > c) acc = fma(-As[(b0 + r) * CT + b0 + l], tc[l], acc);
      }
      tc[r] = acc;
    }
#pragma unroll
    for (uint r = 0; r < TB; ++r) Ts[(b0 + r) * CT + b0 + c] = tc[r];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // (b) the off-diagonal blocks by their distance d from the diagonal. Lane (r, c) = (sg, lane)
  //     of each block; X = SUM_{16j <= l < 16i} A[R][l] T[l][C] is parked in As's strictly-upper
  //     block (j, i), which nothing else reads, then T_ij = -(T_ii X).
  for (uint d = 1; d < CT / TB; ++d) {
    const uint nb = CT / TB - d;
    for (uint q = 0; q < nb; ++q) {
      const uint bi = q + d, bj = q;
      const uint R = bi * TB + sg, Cc = bj * TB + lane;
      float acc = 0.0f;
      for (uint l = bj * TB; l < bi * TB; ++l) acc = fma(As[R * CT + l], Ts[l * CT + Cc], acc);
      As[(bj * TB + sg) * CT + bi * TB + lane] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint q = 0; q < nb; ++q) {
      const uint bi = q + d, bj = q;
      const uint R = bi * TB + sg, Cc = bj * TB + lane;
      float acc = 0.0f;
#pragma unroll
      for (uint l = 0; l < TB; ++l)
        acc = fma(Ts[R * CT + bi * TB + l], As[(bj * TB + l) * CT + bi * TB + lane], acc);
      Ts[R * CT + Cc] = -acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#if INTRA_DBG
  {
    __global float* restrict Td = Tdbg + ((size_t)(t * HEADS + h) * CT) * CT;
    for (uint p = lid; p < CT * CT; p += NSG * SG) Td[p] = (p % CT) <= (p / CT) ? Ts[p] : 0.0f;
  }
#endif

  // ---- 3. T as hi / lo bf16 words [64][TLD]: hi over As, lo over Ts ---------------------
  float tv[CT * CT / (NSG * SG)];
#pragma unroll
  for (uint q = 0; q < CT * CT / (NSG * SG); ++q) {
    const uint p = lid + q * NSG * SG;
    tv[q] = (p % CT) <= (p / CT) ? Ts[p] : 0.0f;   // T is lower triangular; the upper blocks are scratch
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __local ushort* const Th = (__local ushort*)As;
  __local ushort* const Tl = (__local ushort*)Ts;
#pragma unroll
  for (uint q = 0; q < CT * CT / (NSG * SG); ++q) {
    const uint p = lid + q * NSG * SG;
    const ushort hv = rne_bf16(tv[q]);
    Th[(p / CT) * TLD + p % CT] = hv;
    Tl[(p / CT) * TLD + p % CT] = rne_bf16(tv[q] - bf16f(hv));
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // ---- 4. u^T = vb^T T^T and w^T = kb^T T^T by DPAS (M = 8 channels, K = 16 positions j,
  //         N = 16 positions i); sub-group sg owns channels 8sg..8sg+7 of both ----------------
  {
    const uint x0 = sg * 8;
    // A operands, one per K-block kb: lane l = position j = 16kb + l (a live row, zero past L).
    short8 av[CT / 16], ak[CT / 16];
#pragma unroll
    for (uint kb = 0; kb < CT / 16; ++kb) {
      const uint j = kb * 16 + lane;
      const uint jr = min(j, L - 1);
      const size_t row = (size_t)(base_m + jr) * CONV_ROWS;
      const size_t gi = (size_t)(base_m + jr) * HEADS + h;
      const float bt = beta[gi];
      const float eg = exp(g_cum[gi]);
      const ushort8 vw = vload8(0, xb + row + V_OFF + h * DIM + x0);
      const ushort8 kw = vload8(0, xb + row + K_OFF + kh * DIM + x0);
      ushort vv[8], kv[8], vo[8], ko[8];
      vstore8(vw, 0, vv);
      vstore8(kw, 0, kv);
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        vo[r] = j < L ? rne_bf16(bf16f(vv[r]) * bt) : (ushort)0;          // Q1
        ko[r] = j < L ? rne_bf16(bf16f(kv[r]) * bt * eg) : (ushort)0;     // Q2
      }
      av[kb] = as_short8(vload8(0, vo));
      ak[kb] = as_short8(vload8(0, ko));
    }
    __local const uint* const ThW = (__local const uint*)Th;
    __local const uint* const TlW = (__local const uint*)Tl;
#pragma unroll
    for (uint ni = 0; ni < CT / 16; ++ni) {
      const uint i = ni * 16 + lane;            // this lane's output position
      float8 uh = (float8)(0.0f), ul = (float8)(0.0f), wh = (float8)(0.0f), wl = (float8)(0.0f);
#pragma unroll
      for (uint kb = 0; kb <= ni; ++kb) {
        const int8 bh = as_int8(vload8(0, ThW + (i * TLD + kb * 16) / 2));
        const int8 bl = as_int8(vload8(0, TlW + (i * TLD + kb * 16) / 2));
        uh = intel_sub_group_bf16_bf16_matrix_mad_k16(av[kb], bh, uh);
        ul = intel_sub_group_bf16_bf16_matrix_mad_k16(av[kb], bl, ul);
        wh = intel_sub_group_bf16_bf16_matrix_mad_k16(ak[kb], bh, wh);
        wl = intel_sub_group_bf16_bf16_matrix_mad_k16(ak[kb], bl, wl);
      }
      if (i < L) {
        float fu[8], fw[8];
        vstore8(uh + ul, 0, fu);
        vstore8(wh + wl, 0, fw);
        ushort ou[8], ow[8];
#pragma unroll
        for (uint r = 0; r < 8; ++r) {
          ou[r] = rne_bf16(fu[r]);                                         // Q3
          ow[r] = rne_bf16(fw[r]);                                         // Q4
        }
        const size_t out = ((size_t)(base_m + i) * HEADS + h) * DIM + x0;
        vstore8(vload8(0, ou), 0, u + out);
        vstore8(vload8(0, ow), 0, w + out);
      }
    }
  }
}
