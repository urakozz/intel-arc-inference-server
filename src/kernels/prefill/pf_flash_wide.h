// pf_flash_wide.h - spec 38 (PERF-112): the wide form of pf_flash_attn.cl's prompt attention, included by it under
// WIDE=1. Four sub-groups own 32 query rows of one head, HPW heads a work-group; grid (ceil(C / 32), kv-heads,
// GQA / HPW). Each sub-group owns a quarter of O (32 rows x 64 dims, dim atoms 4w .. 4w + 3), so every V operand feeds
// four row blocks (the PAIR form: two). The score product, by WIDE_AHEAD:
//   1  tile pairs: sub-group w = 2u + k computes S of tile t + 64 u, key atoms 2k, 2k + 1, all 32 rows (a 32 x 32
//      block: each Q operand feeds two key atoms, each K operand four row blocks); every sub-group then runs P V over
//      tile t, then tile t + 1. ~232 B of operands a DPAS against the PAIR form's 336 (spec 38 §5.3).
//   0  one tile at a time: sub-group w = 2 rh + k computes S of rows 16 rh .. 16 rh + 15, key atoms 2k, 2k + 1 - the
//      PAIR form's block. ~293 B a DPAS (the fallback, spec 38 §5.6).
// Who computes what changes; no value, operand or order does, so the output is bitwise the RPW 8 kernel's (the
// contract, spec 38 §3):
//   S       each score one DPAS chain over dim atoms 0..15 from 0, then the scale and the mask, as RPW 8.
//   max     per tile mnew = fmax(m, fmax(max of atoms 0-1, max of atoms 2-3)); the second tile of a pair takes the
//           first's mnew as its m. The half maxima meet in SLM (hmax); m itself lives in SLM, double-buffered by the
//           iteration's parity (read sm[pp], written sm[pp ^ 1] by the k = 0 owner of the iteration's last tile).
//   l       per lane ((p0 + p1) + p2) + p3, then sub_group_reduce_add: the k = 0 owner publishes p0 + p1 (spart), the
//           k = 1 owner adds p2, p3, reduces and publishes rsum. One sub-group (w = 3) owns l in SLM and applies
//           l = l * corr + rsum tile by tile in order, one iteration late (after the next barrier 1), and once more
//           after the walk.
//   P V     o *= corr, then o += P V over key atoms 0..3 in order, tile after tile; P (bf16) through SLM (sp), corr
//           through SLM (scorr, parity-buffered), both written by the k = 0 owners.
// A row block may walk tiles wholly masked for it (corr 1, P 0: its numbers stay the same, as in PAIR); a tile past
// the work-group's last key is skipped uniformly. Barriers: 2 an iteration (1 a tile at WIDE_AHEAD 1), 1 before the
// walk, 2 after it.
//
// FA_SKELETON=1 (probe-only, spec 38a Task 1; pf_flash_attn.cl): EXPF and the two sub-group reductions become
// identities, so the arm times this form's loads, DPAS, SLM traffic and barriers without the softmax's arithmetic.
// Its output is not attention.
#if KT != 64 || RPW != 32 || SPLIT || QREG || defined(KV_PAGED) || GATE_EPI || PAIR
#error "pf_flash_attn WIDE: KT 64, RPW 32, no SPLIT / QREG / KV_PAGED / GATE_EPI / PAIR"
#endif
#ifndef WIDE_AHEAD
#error "pf_flash_attn WIDE: WIDE_AHEAD (1: tile pairs, 0: one tile) must be defined"
#endif
#ifndef WIDE_Q1
#define WIDE_Q1 0
#endif
#if WIDE_Q1 && !WIDE_AHEAD
#error "pf_flash_attn WIDE_Q1 (one 32-row Q read a dim atom) needs WIDE_AHEAD 1"
#endif
#if WIDE_AHEAD
#define W_NT 2u                       /* tiles an iteration (full) */
#define W_NRB 4                       /* row blocks of a sub-group's S block */
#else
#define W_NT 1u
#define W_NRB 2
#endif
__attribute__((reqd_work_group_size(SG * 4 * HPW, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_flash_attn(__global const ushort* restrict Q, __global const ushort* restrict Kc,
                  __global const ushort* restrict Vc, __global float* restrict O,
                  uint pos, uint C, uint rows) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint hp = s >> 2, w = s & 3u, k = w & 1u;                  // head of the work-group; sub-group; key half
#if WIDE_AHEAD
  const uint tu = w >> 1, rb0 = 0u;                                // its tile of the pair; its first row block
#else
  const uint tu = 0u, rb0 = 2u * (w >> 1);
#endif
  const uint j = get_group_id(1);                                  // kv head
  const uint h = j * FA_GQA + get_group_id(2) * HPW + hp;          // q head
  const uint r0 = get_group_id(0) * 32u;                           // first of 32 rows
  if (r0 >= C) return;                                             // uniform over the work-group
  const uint depth = pos + C;
  const int q_w = FA_QH * HD * 2, q_h = (int)C, q_p = FA_QH * HD * 2;
  const int kt_w = FA_KVH * HD * 2, kt_h = (int)depth, kt_p = FA_KVH * HD * 2;
  const int v_w = FA_KVH * HD * 2, v_h = (int)depth, v_p = FA_KVH * HD * 2;
  __local float hmax[HPW * 128];     // [head][tile][k][32 rows]: a tile's row maxima of key half k
  __local float sm[HPW * 64];        // [head][parity][32 rows]: m
  __local float scorr[HPW * 128];    // [head][parity][tile][32 rows]: corr
  __local float srs[HPW * 128];      // [head][parity][tile][32 rows]: rsum
  __local float sl[HPW * 32];        // [head][32 rows]: l (w = 3 only)
  __local float spart[HPW * 1024];   // [head][tile][rb][r * 16 + lane]: the k = 0 owner's p0 + p1
  __local ushort sp[HPW * 4096];     // [head][tile][rb * 4 + atom][r * 16 + lane]: P, bf16
  __local float* const Hm = hmax + hp * 128u;
  __local float* const Sm = sm + hp * 64u;
  __local float* const Sc = scorr + hp * 128u;
  __local float* const Sr = srs + hp * 128u;
  __local float* const Sl = sl + hp * 32u;
  __local float* const Spart = spart + hp * 1024u;
  __local ushort* const Sp = sp + hp * 4096u;

  if (w == 0u && l < 8u) {           // m = -INF, l = 0 for the 32 rows (lanes 0-7, four rows each)
    vstore4((float4)(-INFINITY), 0, Sm + 4u * l);
    vstore4((float4)(0.0f), 0, Sl + 4u * l);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float8 o[4][4];
#pragma unroll
  for (uint rb = 0; rb < 4u; ++rb)
#pragma unroll
    for (uint dd = 0; dd < 4u; ++dd) o[rb][dd] = (float8)(0.0f);
  const uint last = min(depth, pos + r0 + 32u);
  uint pp = 0u, nt = W_NT;
  for (uint t0 = 0; t0 < last; t0 += 64u * W_NT, pp ^= 1u) {
    nt = t0 + 64u * (W_NT - 1u) < last ? W_NT : 1u;               // uniform: the last pair may hold one tile
    const bool mine = tu < nt;                                     // this sub-group computes a score block
    const uint ts = t0 + 64u * tu;                                 // its tile's first key
    // --- S: rows 8 (rb0 + i) + r, keys ts + 16 (2k + bb) + lane ---
    float8 sacc[W_NRB][2];
#pragma unroll
    for (uint i = 0; i < W_NRB; ++i) sacc[i][0] = sacc[i][1] = (float8)(0.0f);
    if (mine) {
#if WIDE_Q1
#pragma unroll
      for (uint kk = 0; kk < NDA; ++kk) {
        ushort t[32];
        intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)Q, q_w, q_h, q_p,
                                                   (int2)((int)(h * HD + 16u * kk), (int)r0), t);
        short8 qa[4];
#pragma unroll
        for (uint i = 0; i < 4u; ++i) qa[i] = as_short8(vload8(0, t + 8u * i));
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          uint kb[8];
          intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
              (__global void*)Kc, kt_w, kt_h, kt_p, (int2)((int)((j * HD + 16u * kk) / 2u), (int)(ts + 16u * (2u * k + bb))), kb);
#pragma unroll
          for (uint i = 0; i < 4u; ++i)
            sacc[i][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[i], as_int8(vload8(0, kb)), sacc[i][bb]);
        }
      }
#else
#pragma unroll
      for (uint kk = 0; kk < NDA; kk += 2) {
        short8 qa[W_NRB][2];
#pragma unroll
        for (uint i = 0; i < W_NRB; ++i) {
          ushort t[16];
          intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)Q, q_w, q_h, q_p,
                                                    (int2)((int)(h * HD + 16u * kk), (int)(r0 + 8u * (rb0 + i))), t);
          qa[i][0] = as_short8(vload8(0, t));
          qa[i][1] = as_short8(vload8(0, t + 8));
        }
#pragma unroll
        for (uint kq = 0; kq < 2u; ++kq)
#pragma unroll
          for (uint bb = 0; bb < 2u; ++bb) {
            uint kb[8];
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (__global void*)Kc, kt_w, kt_h, kt_p,
                (int2)((int)((j * HD + 16u * (kk + kq)) / 2u), (int)(ts + 16u * (2u * k + bb))), kb);
#pragma unroll
            for (uint i = 0; i < W_NRB; ++i)
              sacc[i][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[i][kq], as_int8(vload8(0, kb)), sacc[i][bb]);
          }
      }
#endif
      // scale, mask, this half's row maxima
#pragma unroll
      for (uint i = 0; i < W_NRB; ++i) {
        const uint rr = r0 + 8u * (rb0 + i);
        float8 tmax = (float8)(-INFINITY), red;
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          const uint key = ts + 16u * (2u * k + bb) + l;
#define MASK(r) sacc[i][bb].s##r = (key <= pos + rr + r && key < depth) ? sacc[i][bb].s##r * SCORE_SCALE : -INFINITY;
          MASK(0) MASK(1) MASK(2) MASK(3) MASK(4) MASK(5) MASK(6) MASK(7)
#undef MASK
          tmax = fmax(tmax, sacc[i][bb]);
        }
#define RED(r) red.s##r = FA_RMAX(tmax.s##r);
        RED(0) RED(1) RED(2) RED(3) RED(4) RED(5) RED(6) RED(7)
#undef RED
        if (l == 0) vstore8(red, 0, Hm + tu * 64u + k * 32u + 8u * (rb0 + i));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);                                  // barrier 1
    // --- the previous iteration's l, in tile order (w = 3; always W_NT tiles: only the last can be short) ---
    if (w == 3u && t0 != 0u) {
#pragma unroll
      for (uint rb = 0; rb < 4u; ++rb) {
        float8 lsum = vload8(0, Sl + 8u * rb);
#pragma unroll
        for (uint tt = 0; tt < W_NT; ++tt)
          lsum = lsum * vload8(0, Sc + (pp ^ 1u) * 64u + tt * 32u + 8u * rb) + vload8(0, Sr + (pp ^ 1u) * 64u + tt * 32u + 8u * rb);
        if (l == 0) vstore8(lsum, 0, Sl + 8u * rb);
      }
    }
    // --- softmax of this sub-group's block ---
    if (mine) {
#pragma unroll
      for (uint i = 0; i < W_NRB; ++i) {
        const uint rb = rb0 + i;
        const float8 m = vload8(0, Sm + pp * 32u + 8u * rb);
        const float8 mn0 = fmax(m, fmax(vload8(0, Hm + 8u * rb), vload8(0, Hm + 32u + 8u * rb)));
#if WIDE_AHEAD
        const float8 mprev = tu ? mn0 : m;
        const float8 mnew = tu ? fmax(mn0, fmax(vload8(0, Hm + 64u + 8u * rb), vload8(0, Hm + 96u + 8u * rb))) : mn0;
#else
        const float8 mprev = m, mnew = mn0;
#endif
        const float8 corr = EXPF(mprev - mnew);
        float8 part = (float8)(0.0f);
#pragma unroll
        for (uint bb = 0; bb < 2u; ++bb) {
          sacc[i][bb] = EXPF(sacc[i][bb] - mnew);   // p, kept for the k = 1 owner's sum
          part += sacc[i][bb];
          short8 pa;
#define CV(r) pa.s##r = as_short(rne_bf16(sacc[i][bb].s##r));
          CV(0) CV(1) CV(2) CV(3) CV(4) CV(5) CV(6) CV(7)
#undef CV
          intel_sub_group_block_write_us8(Sp + tu * 2048u + (rb * 4u + 2u * k + bb) * 128u, as_ushort8(pa));
        }
        if (k == 0) {
          intel_sub_group_block_write8((__local uint*)(Spart + tu * 512u + rb * 128u), as_uint8(part));
          if (l == 0) {
            vstore8(corr, 0, Sc + pp * 64u + tu * 32u + 8u * rb);
            if (tu + 1u == nt) vstore8(mnew, 0, Sm + (pp ^ 1u) * 32u + 8u * rb);
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);                                  // barrier 2
    // --- rsum of this block's rows (k = 1 owners) ---
    if (mine && k == 1u) {
#pragma unroll
      for (uint i = 0; i < W_NRB; ++i) {
        const uint rb = rb0 + i;
        float8 psum = as_float8(intel_sub_group_block_read8((__local const uint*)(Spart + tu * 512u + rb * 128u)));
        psum += sacc[i][0];
        psum += sacc[i][1];
        float8 rsum;
#define SUM(r) rsum.s##r = FA_RADD(psum.s##r);
        SUM(0) SUM(1) SUM(2) SUM(3) SUM(4) SUM(5) SUM(6) SUM(7)
#undef SUM
        if (l == 0) vstore8(rsum, 0, Sr + pp * 64u + tu * 32u + 8u * rb);
      }
    }
    // --- O += P V over this sub-group's dims 64 w .. 64 w + 63, tile after tile, key atoms in order ---
#pragma unroll
    for (uint tt = 0; tt < W_NT; ++tt) {
      if (tt < nt) {
#pragma unroll
        for (uint rb = 0; rb < 4u; ++rb) {
          const float8 corr = vload8(0, Sc + pp * 64u + tt * 32u + 8u * rb);
#pragma unroll
          for (uint dd = 0; dd < 4u; ++dd) o[rb][dd] *= corr;
        }
#pragma unroll
        for (uint b = 0; b < 4u; b += 2) {
          short8 pa[4][2];
#pragma unroll
          for (uint rb = 0; rb < 4u; ++rb)
#pragma unroll
            for (uint ks = 0; ks < 2u; ++ks)
              pa[rb][ks] = as_short8(intel_sub_group_block_read_us8(Sp + tt * 2048u + (rb * 4u + b + ks) * 128u));
#pragma unroll
          for (uint dd = 0; dd < 4u; dd += 2) {
            uint vb[32];
            intel_sub_group_2d_block_read_transform_16b_32r16x2c(
                (__global void*)Vc, v_w, v_h, v_p,
                (int2)((int)(j * HD + 16u * (4u * w + dd)), (int)(t0 + 64u * tt + 16u * b)), vb);
#pragma unroll
            for (uint c = 0; c < 2u; ++c)
#pragma unroll
              for (uint ks = 0; ks < 2u; ++ks)
#pragma unroll
                for (uint rb = 0; rb < 4u; ++rb)
                  o[rb][dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                      pa[rb][ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[rb][dd + c]);
          }
        }
      }
    }
  }
  // --- the last iteration's l (nt tiles), then O / l ---
  barrier(CLK_LOCAL_MEM_FENCE);
  if (w == 3u) {
    pp ^= 1u;                                                      // the last iteration's parity
#pragma unroll
    for (uint rb = 0; rb < 4u; ++rb) {
      float8 lsum = vload8(0, Sl + 8u * rb);
#pragma unroll
      for (uint tt = 0; tt < W_NT; ++tt)
        if (tt < nt) lsum = lsum * vload8(0, Sc + pp * 64u + tt * 32u + 8u * rb) + vload8(0, Sr + pp * 64u + tt * 32u + 8u * rb);
      if (l == 0) vstore8(lsum, 0, Sl + 8u * rb);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float* Oh = O + (size_t)h * rows * HD;
#pragma unroll
  for (uint rb = 0; rb < 4u; ++rb) {
    const uint rr = r0 + 8u * rb;
    if (rr >= C) break;                                            // a block wholly past C is not written
    const float8 inv = 1.0f / vload8(0, Sl + 8u * rb);
#pragma unroll
    for (uint dd = 0; dd < 4u; ++dd) {
      float8 wv = o[rb][dd] * inv;
      intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, (int)rows, HD * 4,
                                                 (int2)((int)(16u * (4u * w + dd)), (int)rr), (__private uint*)&wv);
    }
  }
}
#undef W_NT
#undef W_NRB
