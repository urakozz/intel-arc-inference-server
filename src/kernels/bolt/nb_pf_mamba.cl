// nb_pf_mamba.cl - spec 25d: Nemotron 3.5 Bolt's Mamba-2 mixer between its two GEMMs over a PREFILL chunk of C <= 2048 rows
// at absolute positions pos .. pos + C - 1 (spec 25 §6.2; plan 25d's SSD form). in_proj's ONE unsplit GEMM leaves fp32 rows
// P [C][18688] (pitch pad256(18560): pf_gemm's zero-padded tail slab) - z [0, 8192), xBC [8192, 18432), dt [18432, 18560);
// every consumer below sums a column as decode sums its one slice (0.0f + v) and rounds once. tests/kernels/bolt_pf_ref.h
// repeats every chain (seed, conv_chunk, dt_cum, cb, ssd, gnorm_chunk) - EDIT THE TWO TOGETHER. One binary per tiling (the
// define in its name), entries:
//
//   nb_pf_seed(ring, seed, pos)                                         grid (10240 / 256), WG 256
//   nb_pf_conv(P, seed, cw, cb, ring, xc, pos, C)                       grid (10240 / 256, ceil(C / CONV_ROWS)), WG 256
//   nb_pf_dt(P, A, dt_bias, dtv, cum, pos, C)                           grid (1, nsub), WG 128
//   nb_pf_cb(xc, cbo, pos, C)                                           grid (nsub, 8), WG 256 (plain) | 64 (CB_DPAS)
//   nb_pf_ssd(xc, dtv, cum, cbo, D, state, y, pos, C)                   grid (128 x 64 / SSD_W), WG 256
//   nb_pf_gnorm(P, y, w, out)                                           grid (8, C), WG 256
// with nsub = ceil((pos % 64 + C) / 64) - the chunk's ABSOLUTE 64-row sub-chunks (Review Focus 1): sub-chunk k holds the
// chunk's rows whose position lies in one 64-aligned window; the first is partial when pos % 64 != 0, so a chunk boundary
// on the 64-grid is a sub-chunk boundary and a split there changes no sub-chunk (runtime::bolt::prefill_chunk_rows cuts
// only there or at the prompt end).
//
// **The conv** (causal_conv1d over the chunk, decode's chain row by row): nb_pf_seed copies the ring's raw columns of
// positions pos - 3 .. pos - 1 (0 below position 0 - the ring is never cleared) into `seed` in a launch BEFORE the conv,
// because the conv's writeback (the chunk's last min(C, 3) raw columns into slots (pos + t) % 16) can reach the very slots
// the seed reads (C >= 14). nb_pf_conv never reads the ring: work-item (c, block) recomputes the raw x of its rows' three
// predecessors from P (rne(0 + P), the same rounding) or reads the seed below row 0; acc = bias, + w_j x_j OLDEST first,
// xc = rne(silu(acc)) - decode's conv_step, bitwise.
// **dt** (nb_pf_dt): dt = softplus(f32(rne(0 + P[t][18432 + h])) + dt_bias) (decode's), a = A dt, cum = the sequential
// fp32 prefix sum of a inside each sub-chunk from its first row IN THIS CHUNK.
// **C B^T** (nb_pf_cb): per (sub-chunk, group) the 64 x 64 dots of C_t with B_s over the 128 state entries, [nsub][8][64]
// [64] fp32 at the rows' absolute-local indices (pos + t) % 64, 0 where a row lies outside the chunk. Both operands are
// the conv's bf16 outputs, so the products are exact: the plain build sums n ascending from 0.0f (the twin's order);
// CB_DPAS = 1 is the same dots by DPAS (bf16 x bf16 -> fp32, intel_sub_group_bf16_bf16_matrix_mad_k16) - the hardware's
// sum order, held to the plain build by a bar (bolt_pf_kernels_test), never bitwise. Nothing else meets DPAS: the fp32
// state never does (the GDN record: pf_gdn_scan.cl's DPAS scan rounded the state to bf16 and cost a token).
// **The scan** (nb_pf_ssd): work-group (h, q) owns head h's dims p in [q SSD_W, (q + 1) SSD_W); work-item (pl, lane)
// holds S[p][n] for n = lane + SSD_LN j in registers (fp32; SSM_B16: read widened from bf16 and rounded ONCE at the end of
// the chunk - Decision 2's prefill rule), read once, written once. Per sub-chunk in order (L its last row in the chunk):
//   M[t][s] = (CB[t][s] · exp(cum_t - cum_s)) · dt_s for s <= t, else 0    the MASK BEFORE THE EXP: never exp of > 0
//   y_t[p]  = rne((Yd + Yo) + x_t[p] · D)                                  per output, no cross-lane reduction:
//             Yd = M[t][s0] x_s0[p] + M[t][s0+1] x_s0+1[p] + ...             sequential from the first term
//             Yo = tree_n(S[p][n] · C_t[n]) · exp(cum_t)                    the halving tree (decode's y), S from SLM
//   S[p][n] = S[p][n] · exp(cum_L) + acc, acc = (B_s0[n] w_s0) x_s0[p] + ..., w_s = exp(cum_L - cum_s) · dt_s
// so a one-row sub-chunk updates the state bitwise as decode's nb_ssm step does (exp(0) = 1: w = dt).
// **The gated norm** (nb_pf_gnorm): nb_gnorm's chain (one work-group a 1024-group) at the prefill pitch.
//
// Tilings (Review Focus 4: every build finite and equal, bitwise, ten times over): SSD_W (dims p a work-group: 16 or 8 -
// the work-item's state values 8 or 4) and CONV_ROWS (rows a conv work-item: any >= 1) change only who computes what, never
// an operation's operands or order. Plain OpenCL C but the CB_DPAS build (every cross-lane step through SLM), so the file
// also runs on the Mac's GPU (tools/mac/clrun/bolt_pf_run.cc).
#pragma OPENCL FP_CONTRACT OFF

#include "nb_common.h"

#ifndef PF_SUB
#error "nb_pf_mamba: PF_SUB (the sub-chunk length) must be defined"
#endif
#if PF_SUB != 64
#error "nb_pf_mamba: the sub-chunk is 64 rows (spec 25d as built; M's SLM tile and the walk's grid both assume it)"
#endif
#ifndef CB_DPAS
#define CB_DPAS 0
#endif

#define NB_PF_LD 18688           /* pad256(18560) */
#define NB_XBC_OFF 8192
#define NB_DT_OFF 18432
#define NB_XBC 10240
#define NB_INNER 8192
#define NB_HEADS 128
#define NB_P 64
#define NB_N 128
#define NB_G 8
#define NB_B_OFF 8192            /* within xc: x [0, 8192), B [8192, 9216), C [9216, 10240) */
#define NB_C_OFF 9216
#define NB_RING 16
#define NB_TAPS 4

// The sub-chunk k of a chunk at pos with C rows: rows [r0, r1), r0's absolute-local index u0.
inline uint pf_nsub(uint pos, uint C) { return (pos % PF_SUB + C + PF_SUB - 1) / PF_SUB; }
inline void pf_span(uint pos, uint C, uint k, uint* r0, uint* r1, uint* u0) {
  const uint off = pos % PF_SUB;
  *r0 = k == 0 ? 0u : k * PF_SUB - off;
  *r1 = min(C, (k + 1) * PF_SUB - off);
  *u0 = k == 0 ? off : 0u;
}
inline size_t pf_cb_at(uint k, uint g, uint i, uint j) {
  return (((size_t)k * NB_G + g) * PF_SUB + i) * PF_SUB + j;
}

#if CB_DPAS
// =====================================================================================================================
// The DPAS build of nb_pf_cb only: work-group (k, g) of 4 sub-groups; sub-group s computes rows i in [16 s, 16 s + 16)
// (two 8-row blocks) x the 64 columns (four 16-column blocks), K = 128 in 8 steps of 16. Operands by plain loads: a.s_r =
// C_{row i0 + r}[16 ks + lane] (bf16 bits), b.s_v = the pair (B_{row j0 + lane}[16 ks + 2 v], [.. + 1]) as one 32-bit
// word (B is n-contiguous, so the pair is adjacent: the VNNI word itself). A row outside the chunk reads zeros.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
__attribute__((reqd_work_group_size(64, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void nb_pf_cb(__global const ushort* restrict xc, __global float* restrict cbo, uint pos, uint C) {
  const uint k = get_group_id(0), g = get_group_id(1);
  const uint s = get_sub_group_id(), lane = get_sub_group_local_id();
  const int off = (int)(pos % PF_SUB);
  float8 acc[2][4];
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 4; ++b) acc[a][b] = (float8)(0.0f);
  for (uint ks = 0; ks < NB_N / 16; ++ks) {
    short8 av[2];
    for (int a = 0; a < 2; ++a) {
      short tmp[8];
      for (int r = 0; r < 8; ++r) {
        const int ti = (int)(k * PF_SUB) + (int)(16 * s + 8 * a + r) - off;
        tmp[r] = (ti >= 0 && ti < (int)C)
                     ? as_short(xc[(size_t)ti * NB_XBC + NB_C_OFF + g * NB_N + 16 * ks + lane])
                     : (short)0;
      }
      av[a] = (short8)(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
    }
    for (int b = 0; b < 4; ++b) {
      const int tj = (int)(k * PF_SUB) + (int)(16 * b + lane) - off;
      int tmp[8];
      for (int v = 0; v < 8; ++v)
        tmp[v] = (tj >= 0 && tj < (int)C)
                     ? as_int(((__global const uint*)(xc + (size_t)tj * NB_XBC + NB_B_OFF + g * NB_N + 16 * ks))[v])
                     : 0;
      const int8 bv = (int8)(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]);
      for (int a = 0; a < 2; ++a) acc[a][b] = intel_sub_group_bf16_bf16_matrix_mad_k16(av[a], bv, acc[a][b]);
    }
  }
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 4; ++b) {
      const float vals[8] = {acc[a][b].s0, acc[a][b].s1, acc[a][b].s2, acc[a][b].s3,
                             acc[a][b].s4, acc[a][b].s5, acc[a][b].s6, acc[a][b].s7};
      for (int r = 0; r < 8; ++r) cbo[pf_cb_at(k, g, 16 * s + 8 * a + r, 16 * b + lane)] = vals[r];
    }
}

#else  // !CB_DPAS
// =====================================================================================================================
#if !defined(SSD_W) || !defined(CONV_ROWS) || !defined(SSM_B16)
#error "nb_pf_mamba: SSD_W, CONV_ROWS and SSM_B16 must be defined"
#endif
#if SSD_W != 16 && SSD_W != 8
#error "nb_pf_mamba: SSD_W is 16 or 8 dims a work-group (256 work-items x 128 / (256 / SSD_W) state values)"
#endif
#if CONV_ROWS < 1
#error "nb_pf_mamba: CONV_ROWS >= 1"
#endif
#define SSD_LN (256 / SSD_W)     /* lanes a dim p */
#define SSD_PER (NB_N / SSD_LN)  /* state values a work-item */
#if SSM_B16
#define STATE_T ushort
#define STATE_LD(v) bf16f(v)
#define STATE_ST(f) rne_bf16(f)
#else
#define STATE_T float
#define STATE_LD(v) (v)
#define STATE_ST(f) (f)
#endif

// Column `col` of P's row t as decode sums its one slice.
inline float pf_col(__global const float* restrict P, uint t, uint col) {
  float v = 0.0f;
  v += P[(size_t)t * NB_PF_LD + col];
  return v;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_pf_seed(__global const ushort* restrict ring, __global ushort* restrict seed, uint pos) {
  const uint c = get_group_id(0) * 256 + get_local_id(0);
  if (c >= NB_XBC) return;
  for (uint j = 0; j < NB_TAPS - 1; ++j) {
    const int p = (int)pos - (int)(NB_TAPS - 1) + (int)j;
    seed[(size_t)j * NB_XBC + c] = p < 0 ? (ushort)0 : ring[(size_t)((uint)p % NB_RING) * NB_XBC + c];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_pf_conv(__global const float* restrict P, __global const ushort* restrict seed,
                         __global const float* restrict cw, __global const float* restrict cb,
                         __global ushort* restrict ring, __global ushort* restrict xc, uint pos, uint C) {
  const uint c = get_group_id(0) * 256 + get_local_id(0);
  const uint t0 = get_group_id(1) * CONV_ROWS;
  if (c >= NB_XBC || t0 >= C) return;
  const uint t1 = min(C, t0 + CONV_ROWS);
  float hist[NB_TAPS - 1];
  for (uint i = 0; i < NB_TAPS - 1; ++i) {
    const int t = (int)t0 - (int)(NB_TAPS - 1) + (int)i;
    hist[i] = t < 0 ? bf16f(seed[(size_t)(t + (int)(NB_TAPS - 1)) * NB_XBC + c])
                    : bf16f(rne_bf16(pf_col(P, (uint)t, NB_XBC_OFF + c)));
  }
  for (uint t = t0; t < t1; ++t) {
    const ushort raw = rne_bf16(pf_col(P, t, NB_XBC_OFF + c));
    float acc = cb[c];
    acc = acc + cw[(size_t)c * NB_TAPS + 0] * hist[0];
    acc = acc + cw[(size_t)c * NB_TAPS + 1] * hist[1];
    acc = acc + cw[(size_t)c * NB_TAPS + 2] * hist[2];
    acc = acc + cw[(size_t)c * NB_TAPS + 3] * bf16f(raw);
    xc[(size_t)t * NB_XBC + c] = rne_bf16(nb_silu_conv(acc));
    if (t + (NB_TAPS - 1) >= C) ring[(size_t)((pos + t) % NB_RING) * NB_XBC + c] = raw;   // the chunk's last three
    hist[0] = hist[1];
    hist[1] = hist[2];
    hist[2] = bf16f(raw);
  }
}

__attribute__((reqd_work_group_size(NB_HEADS, 1, 1)))
__kernel void nb_pf_dt(__global const float* restrict P, __global const float* restrict A,
                       __global const float* restrict dt_bias, __global float* restrict dtv,
                       __global float* restrict cum, uint pos, uint C) {
  const uint h = get_local_id(0);
  uint r0, r1, u0;
  pf_span(pos, C, get_group_id(1), &r0, &r1, &u0);
  if (r0 >= C) return;
  const float Ah = A[h], bh = dt_bias[h];
  float cs = 0.0f;
  for (uint t = r0; t < r1; ++t) {
    const float dt = nb_softplus(rf(pf_col(P, t, NB_DT_OFF + h)) + bh);
    const float a = Ah * dt;
    cs = t == r0 ? a : cs + a;
    dtv[(size_t)t * NB_HEADS + h] = dt;
    cum[(size_t)t * NB_HEADS + h] = cs;
  }
}

// The plain C B^T: 16 outputs a work-item, n ascending from 0.0f.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_pf_cb(__global const ushort* restrict xc, __global float* restrict cbo, uint pos, uint C) {
  const uint k = get_group_id(0), g = get_group_id(1), lid = get_local_id(0);
  const int off = (int)(pos % PF_SUB);
  for (uint q = 0; q < PF_SUB * PF_SUB / 256; ++q) {
    const uint o = lid + 256 * q, i = o / PF_SUB, j = o % PF_SUB;
    const int ti = (int)(k * PF_SUB + i) - off, tj = (int)(k * PF_SUB + j) - off;
    float acc = 0.0f;
    if (ti >= 0 && ti < (int)C && tj >= 0 && tj < (int)C) {
      const __global ushort* Cr = xc + (size_t)ti * NB_XBC + NB_C_OFF + g * NB_N;
      const __global ushort* Br = xc + (size_t)tj * NB_XBC + NB_B_OFF + g * NB_N;
      for (uint n = 0; n < NB_N; ++n) acc = acc + bf16f(Cr[n]) * bf16f(Br[n]);
    }
    cbo[pf_cb_at(k, g, i, j)] = acc;
  }
}

// The halving tree over the 128 products S[n] · C[n] (bolt_mamba_ops.tree_sum, decode's y): level 3's 16 partials, each the
// 8-term tree of n = i + 16 m, then 16 -> 1 halving.
inline float pf_tree128(__local const float* S, __global const ushort* C) {
  float v[16];
  for (uint i = 0; i < 16; ++i) {
    const float p0 = S[i] * bf16f(C[i]), p64 = S[i + 64] * bf16f(C[i + 64]);
    const float p32 = S[i + 32] * bf16f(C[i + 32]), p96 = S[i + 96] * bf16f(C[i + 96]);
    const float p16 = S[i + 16] * bf16f(C[i + 16]), p80 = S[i + 80] * bf16f(C[i + 80]);
    const float p48 = S[i + 48] * bf16f(C[i + 48]), p112 = S[i + 112] * bf16f(C[i + 112]);
    v[i] = ((p0 + p64) + (p32 + p96)) + ((p16 + p80) + (p48 + p112));
  }
  for (uint h = 8; h > 0; h >>= 1)
    for (uint i = 0; i < h; ++i) v[i] = v[i] + v[i + h];
  return v[0];
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_pf_ssd(__global const ushort* restrict xc, __global const float* restrict dtv,
                        __global const float* restrict cum, __global const float* restrict cbo,
                        __global const float* restrict D, __global STATE_T* restrict state,
                        __global ushort* restrict y, uint pos, uint C) {
  const uint h = get_group_id(0) / (NB_P / SSD_W);
  const uint q = get_group_id(0) % (NB_P / SSD_W);
  const uint lid = get_local_id(0);
  const uint pl = lid / SSD_LN, lane = lid % SSD_LN;
  const uint g = h / (NB_HEADS / NB_G);
  __local float Ms[PF_SUB * PF_SUB];
  __local float Ss[SSD_W * NB_N];
  __local float xs[PF_SUB * SSD_W];
  __local float cums[PF_SUB], dts[PF_SUB], ws[PF_SUB];
  const size_t row = ((size_t)h * NB_P + q * SSD_W + pl) * NB_N;
  float st[SSD_PER];
  for (uint j = 0; j < SSD_PER; ++j) {
    st[j] = STATE_LD(state[row + lane + SSD_LN * j]);
    Ss[pl * NB_N + lane + SSD_LN * j] = st[j];
  }
  const float Dh = D[h];
  const uint ns = pf_nsub(pos, C);
  for (uint k = 0; k < ns; ++k) {
    uint r0, r1, u0;
    pf_span(pos, C, k, &r0, &r1, &u0);
    const uint nr = r1 - r0, iL = u0 + nr - 1;
    barrier(CLK_LOCAL_MEM_FENCE);   // the previous sub-chunk's readers of xs / Ms / cums are done
    if (lid < nr) {
      cums[u0 + lid] = cum[(size_t)(r0 + lid) * NB_HEADS + h];
      dts[u0 + lid] = dtv[(size_t)(r0 + lid) * NB_HEADS + h];
    }
    for (uint o = lid; o < nr * SSD_W; o += 256) {
      const uint tl = o / SSD_W, pp = o % SSD_W;
      xs[(u0 + tl) * SSD_W + pp] = bf16f(xc[(size_t)(r0 + tl) * NB_XBC + (size_t)h * NB_P + q * SSD_W + pp]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = lid; o < PF_SUB * PF_SUB; o += 256) {
      const uint i = o / PF_SUB, j = o % PF_SUB;
      float m = 0.0f;
      if (j <= i && j >= u0 && i <= iL) m = (cbo[pf_cb_at(k, g, i, j)] * nb_exp(cums[i] - cums[j])) * dts[j];
      Ms[o] = m;
    }
    if (lid < nr) ws[u0 + lid] = nb_exp(cums[iL] - cums[u0 + lid]) * dts[u0 + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = lid; o < nr * SSD_W; o += 256) {   // y: one output (row, dim) a work-item at a time
      const uint tl = o / SSD_W, pp = o % SSD_W, i = u0 + tl, t = r0 + tl;
      float yd = Ms[i * PF_SUB + u0] * xs[u0 * SSD_W + pp];
      for (uint j = u0 + 1; j <= i; ++j) yd = yd + Ms[i * PF_SUB + j] * xs[j * SSD_W + pp];
      const float yo = pf_tree128(Ss + pp * NB_N, xc + (size_t)t * NB_XBC + NB_C_OFF + g * NB_N) * nb_exp(cums[i]);
      y[(size_t)t * NB_INNER + (size_t)h * NB_P + q * SSD_W + pp] = rne_bf16((yd + yo) + xs[i * SSD_W + pp] * Dh);
    }
    {   // the state, in registers: S · exp(cum_L) + Σ_s (B_s w_s) x_s, sequential from the first term
      const float decay = nb_exp(cums[iL]);
      for (uint j = 0; j < SSD_PER; ++j) {
        const uint n = lane + SSD_LN * j;
        const __global ushort* Bc = xc + NB_B_OFF + g * NB_N + n;
        float acc = (bf16f(Bc[(size_t)r0 * NB_XBC]) * ws[u0]) * xs[u0 * SSD_W + pl];
        for (uint s = u0 + 1; s <= iL; ++s)
          acc = acc + (bf16f(Bc[(size_t)(r0 + s - u0) * NB_XBC]) * ws[s]) * xs[s * SSD_W + pl];
        st[j] = st[j] * decay + acc;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // every y of this sub-chunk read the old Ss
    for (uint j = 0; j < SSD_PER; ++j) Ss[pl * NB_N + lane + SSD_LN * j] = st[j];
  }
  for (uint j = 0; j < SSD_PER; ++j) state[row + lane + SSD_LN * j] = STATE_ST(st[j]);
}

#define GN_GROUP 1024
#define GN_PER (GN_GROUP / 256)
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nb_pf_gnorm(__global const float* restrict P, __global const ushort* restrict y,
                          __global const float* restrict w, __global ushort* restrict out) {
  const uint g = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  __local float red[256];
  float v[GN_PER], sq[GN_PER];
  for (uint j = 0; j < GN_PER; ++j) {
    const uint c = g * GN_GROUP + lid + j * 256;
    v[j] = bf16f(y[(size_t)t * NB_INNER + c]) * nb_silu_gate(rf(pf_col(P, t, c)));
    sq[j] = v[j] * v[j];
  }
  for (uint hv = GN_PER / 2; hv > 0; hv >>= 1)
    for (uint j = 0; j < hv; ++j) sq[j] = sq[j] + sq[j + hv];
  red[lid] = sq[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] = red[lid] + red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)GN_GROUP + 1e-5f);
  for (uint j = 0; j < GN_PER; ++j) {
    const uint c = g * GN_GROUP + lid + j * 256;
    out[(size_t)t * NB_INNER + c] = rne_bf16((v[j] * rstd) * w[c]);
  }
}
#endif  // CB_DPAS
