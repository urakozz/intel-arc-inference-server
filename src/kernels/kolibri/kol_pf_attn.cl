// kol_pf_attn.cl - spec 20d: Kolibri-1's prefill attention - spec 6's flash
// attention carried to head_dim 128, at GQA 12, with Kolibri's two layer kinds and no output gate:
//
//   kol_pf_flash_attn(q, kv_k, kv_v, out, pos, C)   grid (ceil(C / RPW), KV_HEADS, GQA / HPW), 256 GRF
//     q      fp32 [kPfC][Q_HEADS][128]   kol_attn_prep's q (bf16 values: q/k head norm, RoPE in a sliding
//                                       layer) - 20c's decode prep kernel at M = kPfC, S = 1
//     kv_k   bf16 [rows][KV_HEADS][128]  this layer's keys: SLIDING the 4096-slot ring (key k at row
//     kv_v                              k & (RING - 1)), else the growing cache (key k at row k); the
//                                       chunk's own rows were written by kol_attn_prep first
//     out    bf16 [C][Q_HEADS x 128]     rne(o / l) - o_proj's A operand (no gate: Kolibri has none)
//
// **The window** (spec 20 §1, 20a pinned: 513 keys INCLUDING the query). Row t of a chunk at `pos` sits
// at p = pos + t and sees keys k with k <= p and, SLIDING, k > p - WINDOW (i.e. k >= p - 512) - some
// from the ring (positions before pos, the previous chunks' writes) and some from this chunk. A sub-
// group's 8 rows start at p0 = pos + r0; its key tiles run from KT x floor(max(0, p0 - (WINDOW - 1)) /
// KT) (SLIDING; 0 for a full layer) up to its last row: ABSOLUTE multiples of KT, and RING % KT == 0, so
// every tile is KT contiguous ring rows and no 2D block read crosses the ring's end. A chunk of kPfC =
// 2048 rows needs keys [pos - 512, pos + 2047]: 2560 < 4096 slots, so this chunk's writes never
// overwrite a key one of its rows still reads (the reason for 20c's 4096-slot ring).
//
// **A wholly masked tile adds exact zeros.** Every tile a row walks lies within its group's span, but a
// row's FIRST tile may hold none of ITS keys (the group's first row reaches further back): the online
// softmax runs with m_safe = (m_new == -INF ? 0 : m_new), so such a tile leaves (m, l, o) exactly as
// they were (corr = exp(-INF) = 0 on zeros, p = exp(-INF) = 0); once a key was seen the update is
// the plain flash update (m_safe = m_new). So a row's result depends on the tiles that hold its keys -
// a function of its ABSOLUTE position alone, whatever chunk or row group it rides in: the prefill
// split argument (a split at a multiple of 64 is bitwise).
//
// **The flash walk at head_dim 128:** a sub-group owns 8 query rows of one head; O is NDA
// float8 (8 rows x 128 dims) in registers; S for one KV tile is KT/16 float8 with the row in the
// component and the key in the lane - the bf16 DPAS A-operand layout, so P goes to P·V with no data
// movement; K^T through the transposed 32-bit 2D read, V through the VNNI-transform read; the fp32
// online softmax; P as the P·V operand (the default path: two bf16 halves, KOL_PV_HL; eager: the reference's
// bf16 p); q read from the prep's fp32 rows once (a row
// past C reads row C - 1: finite, never written); the scale 1 / sqrt(128).
//
// **EAGER (B70_KOLIBRI_ATTN=eager, the switch 20c's decode reads too - one parser,
// runtime::kolibri::kolibri_attn()).** Two passes: pass 1 the running max and sum (fp32,
// online), pass 2 p = rne(exp(s - m) / l) - the softmax's bf16 output - and o += p·V; s = rne(f32(rne(
// q·k)) x 128^-0.5); the epilogue rounds o once (p already normalised). The reference's ROUNDING POINTS
// (tests/kernels/kolibri_pf_ref.h eager_window), not its sum orders (DPAS, the tile walk).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
// KV8 (PERF-102, `--kv-cache int8`; undefined on every other command line): Inkling's FULL layer over the int8 cache
// (kv8_scheme.h at head_dim 128, written by ink_prep.cl's INK_KV8 prefill build): Kc / Vc int8 [rows][KV_HEADS][128],
// their fp16 scales Ks / Vs [rows][KV_HEADS] after them; Q is the ROTATED q. kv8.cl's flash rule: K's int8 is the bf16
// DPAS operand as is (exact) and the dot is multiplied by the key's scale before the score scale; V's int8 likewise,
// its scale folded into P (P s_v, then the hi / lo halves); the row sum is P's, unscaled. The epilogue un-rotates
// each row's O / l (FWHT across the lanes, then the dim atoms) before the gate.
#ifndef KV8
#define KV8 0
#endif
#if KV8
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#endif
#ifndef KT
#error "kol_pf_attn: KT (KV positions per tile, a multiple of 32) must be defined"
#endif
#if KT % 32
#error "kol_pf_attn: KT must be a multiple of 32 (the P·V loop takes key atoms in pairs)"
#endif
#if !defined(RPW) || RPW % 8
#error "kol_pf_attn: RPW (rows per work-group per head) must be a multiple of 8"
#endif
#if !defined(Q_HEADS) || !defined(KV_HEADS) || !defined(HPW)
#error "kol_pf_attn: Q_HEADS, KV_HEADS and HPW must be defined"
#endif
#if Q_HEADS % KV_HEADS != 0 || (Q_HEADS / KV_HEADS) % HPW != 0
#error "kol_pf_attn: q-heads must be a multiple of kv-heads, and the GQA group of HPW"
#endif
#if !defined(EXP2) || !defined(EAGER) || !defined(WINDOW) || !defined(RING)
#error "kol_pf_attn: EXP2, EAGER, WINDOW and RING must be defined (WINDOW 0 / RING 0: a full layer)"
#endif
#if EAGER && EXP2
#error "kol_pf_attn: the eager-mirroring path uses natural exp (EXP2 0), as the reference's softmax"
#endif
#define SLIDING (WINDOW > 0)
#if SLIDING && (RING == 0 || (RING & (RING - 1)) != 0 || RING % KT != 0 || RING < 2048 + WINDOW - 1)
#error "kol_pf_attn: the ring is a power of two of whole KT tiles holding a chunk and its window"
#endif
#if !SLIDING && RING != 0
#error "kol_pf_attn: a full layer has no ring (RING 0)"
#endif
#if KV8 && (SLIDING || EAGER || !INK_PREFILL)
#error "kol_pf_attn KV8: Inkling's full layers (INK_PREFILL, WINDOW 0, the default path)"
#endif
#ifndef PAIR
#define PAIR 0
#endif
#if PAIR && (EAGER || !INK_PREFILL || KT != 64 || RPW != 16)
#error "kol_pf_attn PAIR: Inkling's default path (INK_PREFILL, not EAGER), KT 64, RPW 16"
#endif
#ifndef KV8_SHARE
#define KV8_SHARE 0
#endif
#if KV8_SHARE && !(PAIR && KV8 && HPW == 3)
#error "kol_pf_attn KV8_SHARE: the KV8 PAIR form at HPW 3 (six sub-groups widen each tile once)"
#endif
#define GQA (Q_HEADS / KV_HEADS)
#define SG 16
#define HD 128
#define NDA (HD / 16)                  /* 8 dim-atoms of O */
#define NKA (KT / 16)                  /* key atoms of S per tile */
#define SGS (HPW * RPW / 8)            /* sub-groups per work-group */
#define SSCALE 0.08838834764831845f    /* 1 / sqrt(128), as fp32 (the reference's d ** -0.5) */
#if EXP2
#define SCORE_SCALE (SSCALE * M_LOG2E_F)   /* scores in the log2 domain */
#define EXPF(x) exp2(x)
#else
#define SCORE_SCALE SSCALE
#define EXPF(x) exp(x)
#endif

#include "bf16.h"
#if KV8
#include "kv8_scheme.h"
#define KV_T char
// int8 -> bf16 bits, exact (|q| <= 127).
inline uint i8_bf16(char c) { return as_uint((float)c) >> 16; }
#else
#define KV_T ushort
#endif
// One score as the walk uses it: the default path scales the fp32 dot; the eager one rounds the dot (the
// bf16 matmul's output) and the scaled score (the bf16 multiply).
inline float score_of(float dot) {
#if EAGER
  return bf16f(rne_bf16(bf16f(rne_bf16(dot)) * SSCALE));
#else
  return dot * SCORE_SCALE;
#endif
}
// The online softmax's reference point: m itself, or 0 while a row has seen no key (m = -INF).
inline float8 m_safe(float8 m) { return select(m, (float8)(0.0f), isequal(m, (float8)(-INFINITY))); }

#if PAIR
// PAIR=1 (PERF-10 / PERF-11 at head_dim 128, plan 41d Task 3; bf16 or KV8): pf_flash_attn.cl's PAIR form - two
// sub-groups share 16 query rows of one head, so every K / V operand (and KV8's widening of it) feeds two row blocks'
// DPAS instead of one. Who computes what changes; no value, operand or order does (bitwise the 8-row form below):
//   S     sub-group k scores key atoms 2k, 2k + 1 for both row blocks: the same DPAS chain over the dims (KV8: then the
//         key's scale), the relative bias from the same table columns and fma chain, the same mask; the row maxima
//         meet in SLM, mnew = fmax(m, fmax(max of atoms 0-1, max of atoms 2-3)) - a max of the same set.
//   l     psum keeps its order ((p0 + p1) + p2) + p3: sub-group 0 publishes p0 + p1, sub-group 1 adds p2, p3, reduces
//         and owns l (from the sink's 1); it hands l over once, after the walk.
//   P V   both publish their atoms' P halves (KV8: P times the key's V scale first) in SLM; sub-group k owns dim atoms
//         4k .. 4k + 3 of O for both row blocks over all four key atoms in order (hi then lo).
//   KV8   the un-rotation's last dim-atom stage pairs atom dd with dd + 4: the halves meet in SLM, sub-group 0 keeps
//         a + c, sub-group 1 a - c.
// Row block 0 may walk one tile more than in the 8-row form (and row block 1, SLIDING, one before its window), wholly
// masked for it: m starts at the sink, so corr is 1 and P 0, and its numbers stay the same. A work-group is HPW pairs
// (q heads of one kv head) on the same 16 rows, so the barriers are uniform. Grid (ceil(C / 16), KV_HEADS, GQA / HPW).
__attribute__((reqd_work_group_size(SG * 2 * HPW, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void kol_pf_flash_attn(__global const float* restrict Q, __global const KV_T* restrict Kc,
                                __global const KV_T* restrict Vc,
#if KV8
                                __global const ushort* restrict Ks, __global const ushort* restrict Vs,
#endif
                                __global ushort* restrict out, uint pos, uint C,
                                __global const float* restrict relative, __global const float* restrict table,
                                __global const float* restrict sinks, __global const float* restrict gate) {
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint k = s & 1u, hp = s >> 1;                                   // key half / dim half; the pair
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + hp;                  // q head
  const uint r0 = get_group_id(0) * 16u;                                // first of 16 rows
  if (r0 >= C) return;                                                  // uniform over the work-group
  const uint depth = pos + C;
  const uint p0 = pos + r0;                                             // row block rb starts at p0 + 8 rb
#if SLIDING
  const uint tstart = p0 > (uint)(WINDOW - 1) ? (p0 - (uint)(WINDOW - 1)) / KT * KT : 0u;
#define KROW(t) ((t) & (uint)(RING - 1))
#else
  const uint tstart = 0u;
#define KROW(t) (t)
#endif
  __local float smax[HPW * 32];      // [pair][k][16 rows]: the tile's row maxima of key half k
  __local float sbuf[HPW * 1280];    // [pair]: sa, then sp
  __local float* const sa = sbuf + hp * 1280u;   // [rb][r * 16 + lane]: sub-group 0's p0 + p1
  __local ushort* const sp = (__local ushort*)(sa + 256);   // [half][rb * 4 + b][r * 16 + lane]: P's hi / lo, bf16
  __local float sl[HPW * 16];        // [pair][16 rows]: l after the walk
#if KV8_SHARE
  // KV8_SHARE=1 (plan 41d, after Task 3): the work-group's six sub-groups widen each K / V tile ONCE into SLM for the
  // three heads (bitwise: the same bf16 words, the same DPAS order). A widened unit is 16 words a lane, [z][lane]:
  // K unit 4 p + a = key atom a x dims 32 p .. 32 p + 31 (units 0-7 in kvw, 8-15 in sbuf, free until the softmax);
  // V unit 8 hv + da = dim atom da x keys 32 hv .. 32 hv + 31, all in kvw (sbuf holds P then). 16 KB more SLM: 32 KB a
  // work-group, four resident an Xe-core (128 KB) instead of five - still faster than an 8 KB kvw taking V one key
  // half at a time (two more barriers a tile; 64k full flash 10.54 s vs 10.26 s, card 0).
  __local uint kvw[4096];
#define KV8_UNIT(u) (((u) < 8u ? kvw : (__local uint*)sbuf) + ((u) & 7u) * 256u)
#endif

  // Q, 16 rows x 128 dims: qa[rb][kk].s_r = q[r0 + 8 rb + r][h][16 kk + lane].
  short8 qa[2][NDA];
  float8 m[2], lsum[2] = {(float8)(1.0f), (float8)(1.0f)};
  float rqv[2][8];   // row 8 rb + r's relative query, lane d holding component d
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk) {
      ushort t[8];
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const uint row = min(r0 + 8u * rb + r, C - 1u);
        t[r] = rne_bf16(Q[((size_t)row * Q_HEADS + h) * HD + 16u * kk + l]);
      }
      qa[rb][kk] = as_short8(vload8(0, t));
    }
    float sink[8];
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint row = min(r0 + 8u * rb + r, C - 1u);
      sink[r] = sinks[(size_t)row * 64 + h];
      rqv[rb][r] = relative[((size_t)row * Q_HEADS + h) * 16 + l];
    }
    m[rb] = vload8(0, sink);
  }
  float8 o[2][NDA / 2];
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] = (float8)(0.0f);
  const uint last = min(depth, p0 + 16u);

  for (uint t0 = tstart; t0 < last; t0 += KT) {
    // S for this sub-group's key atoms b = 2k + bb (lane = key t0 + 16 b + l), both row blocks.
    float8 sacc[2][2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) sacc[rb][0] = sacc[rb][1] = (float8)(0.0f);
#if KV8_SHARE
    barrier(CLK_LOCAL_MEM_FENCE);                                       // the last tile's V and P reads are done
    for (uint u = s; u < 16u; u += SGS) {
      uint k4[8], kb[16];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV_HEADS * HD, (int)depth,
          KV_HEADS * HD, (int2)((int)((j * HD + 32u * (u >> 2)) / 4u), (int)(t0 + 16u * (u & 3u))), k4);
#pragma unroll
      for (uint z = 0; z < 8u; ++z) {
        const char4 c4 = as_char4(k4[z]);
        kb[2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;
        kb[2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;
      }
      intel_sub_group_block_write8(KV8_UNIT(u), vload8(0, kb));
      intel_sub_group_block_write8(KV8_UNIT(u) + 128u, vload8(1, kb));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2)
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint u = 2u * kk + 2u * k + bb;                           // dims 16 kk .., key atom 2 k + bb
        const int8 k0 = as_int8(intel_sub_group_block_read8(KV8_UNIT(u)));
        const int8 k1 = as_int8(intel_sub_group_block_read8(KV8_UNIT(u) + 128u));
#pragma unroll
        for (uint rb = 0; rb < 2u; ++rb) {
          sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kk], k0, sacc[rb][bb]);
          sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kk + 1], k1, sacc[rb][bb]);
        }
      }
#pragma unroll
    for (uint bb = 0; bb < 2u; ++bb) {
      const float ksc = vload_half((size_t)min(t0 + 16u * (2u * k + bb) + l, depth - 1) * KV_HEADS + j,
                                   (__global const half*)Ks);
      sacc[0][bb] *= ksc;
      sacc[1][bb] *= ksc;
    }
#elif KV8
#pragma unroll
    for (uint kk = 0; kk < NDA; kk += 2)
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        uint k4[8], kb[16];
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV_HEADS * HD, (int)depth,
            KV_HEADS * HD, (int2)((int)((j * HD + 16u * kk) / 4u), (int)(t0 + 16u * (2u * k + bb))), k4);
#pragma unroll
        for (uint z = 0; z < 8u; ++z) {
          const char4 c4 = as_char4(k4[z]);
          kb[2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;
          kb[2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;
        }
#pragma unroll
        for (uint rb = 0; rb < 2u; ++rb) {
          sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kk], as_int8(vload8(0, kb)), sacc[rb][bb]);
          sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kk + 1], as_int8(vload8(1, kb)), sacc[rb][bb]);
        }
      }
#pragma unroll
    for (uint bb = 0; bb < 2u; ++bb) {
      const float ksc = vload_half((size_t)min(t0 + 16u * (2u * k + bb) + l, depth - 1) * KV_HEADS + j,
                                   (__global const half*)Ks);
      sacc[0][bb] *= ksc;
      sacc[1][bb] *= ksc;
    }
#else
#pragma unroll
    for (uint kk = 0; kk < NDA; ++kk)
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint row = KROW(min(t0 + 16u * (2u * k + bb) + l, depth - 1));
        const __global uint* words = (const __global uint*)(Kc + (size_t)row * KV_HEADS * HD + j * HD + 16u * kk);
        const int8 kw = as_int8(vload8(0, words));
#pragma unroll
        for (uint rb = 0; rb < 2u; ++rb)
          sacc[rb][bb] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[rb][kk], kw, sacc[rb][bb]);
      }
#endif
    // The relative bias (the 8-row form's KOL_BIAS for this sub-group's atoms: atom b's previous column is b - 1's),
    // the mask, this half's row maxima.
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      const uint q0 = p0 + 8u * rb, rr = r0 + 8u * rb;
      float8 bias[2] = {(float8)(0.0f), (float8)(0.0f)};
      if ((int)q0 - (int)(t0 + KT - 1u) < 1024) {
        const int dbase = (int)q0 - (int)t0 - (int)l;
#pragma unroll
        for (uint d = 0; d < 16; ++d) {
          float prev = table[d * 1024 + clamp(dbase + 16 - 32 * (int)k, 0, 1023)];
#pragma unroll
          for (uint bb = 0; bb < 2u; ++bb) {
            const float cur = table[d * 1024 + clamp(dbase - 16 * (int)(2u * k + bb), 0, 1023)];
#define PAIR_BIAS_ROW(r) \
  bias[bb].s##r = fma(sub_group_broadcast(rqv[rb][r], d), intel_sub_group_shuffle_up(prev, cur, (uint)(r)), bias[bb].s##r);
            PAIR_BIAS_ROW(0) PAIR_BIAS_ROW(1) PAIR_BIAS_ROW(2) PAIR_BIAS_ROW(3)
            PAIR_BIAS_ROW(4) PAIR_BIAS_ROW(5) PAIR_BIAS_ROW(6) PAIR_BIAS_ROW(7)
#undef PAIR_BIAS_ROW
            prev = cur;
          }
        }
      }
      float8 tmax = (float8)(-INFINITY), red;
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint key = t0 + 16u * (2u * k + bb) + l;
#if SLIDING
#define PAIR_VISIBLE(r) (key <= q0 + r && key < depth && key + (uint)WINDOW > q0 + r)
#else
#define PAIR_VISIBLE(r) (key <= q0 + r && key < depth)
#endif
#define PAIR_MASK(r)                                                                                        \
  sacc[rb][bb].s##r = PAIR_VISIBLE(r) && rr + r < C                                                         \
      ? score_of(sacc[rb][bb].s##r) + (q0 + r - key < 1024u ? bias[bb].s##r : 0.f) : -INFINITY;
        PAIR_MASK(0) PAIR_MASK(1) PAIR_MASK(2) PAIR_MASK(3) PAIR_MASK(4) PAIR_MASK(5) PAIR_MASK(6) PAIR_MASK(7)
#undef PAIR_MASK
#undef PAIR_VISIBLE
        tmax = fmax(tmax, sacc[rb][bb]);
      }
#define PAIR_RED(r) red.s##r = sub_group_reduce_max(tmax.s##r);
      PAIR_RED(0) PAIR_RED(1) PAIR_RED(2) PAIR_RED(3) PAIR_RED(4) PAIR_RED(5) PAIR_RED(6) PAIR_RED(7)
#undef PAIR_RED
      if (l == 0) vstore8(red, 0, smax + hp * 32u + k * 16u + 8u * rb);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#if KV8_SHARE
    // V's tile widened into kvw: unit 8 hv + da = key atoms 2 hv, 2 hv + 1 x dim atom da, the 8-row form's VNNI words.
#define KV8_SHARE_V()                                                                                       \
  for (uint u = s; u < 2u * NDA; u += SGS) {                                                                \
    uint v4[8], vb[16];                                                                                     \
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((__global void*)Vc, KV_HEADS * HD, (int)depth,      \
        KV_HEADS * HD, (int2)((int)(j * HD + 16u * (u & 7u)), (int)(t0 + 32u * (u >> 3))), v4);              \
    _Pragma("unroll") for (uint z = 0; z < 8u; ++z) {                                                       \
      const char4 c4 = as_char4(v4[z]);                                                                     \
      vb[2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;                                                   \
      vb[2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;                                              \
    }                                                                                                       \
    intel_sub_group_block_write8(kvw + u * 256u, vload8(0, vb));                                            \
    intel_sub_group_block_write8(kvw + u * 256u + 128u, vload8(1, vb));                                     \
  }
    KV8_SHARE_V()
#endif
    float8 corr[2];
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb) {
      const float8 mnew =
          fmax(m[rb], fmax(vload8(0, smax + hp * 32u + 8u * rb), vload8(0, smax + hp * 32u + 16u + 8u * rb)));
      const float8 msafe = m_safe(mnew);
      corr[rb] = EXPF(m[rb] - msafe);
      m[rb] = mnew;
      float8 part = (float8)(0.0f);
#pragma unroll
      for (uint bb = 0; bb < 2u; ++bb) {
        const uint b = 2u * k + bb;
        const float8 p = EXPF(sacc[rb][bb] - msafe);
        sacc[rb][bb] = p;                       // kept for sub-group 1's sum
        part += p;
#if KV8
        const float8 pv = p * vload_half((size_t)min(t0 + 16u * b + l, depth - 1) * KV_HEADS + j,
                                         (__global const half*)Vs);
#else
        const float8 pv = p;
#endif
        short8 hi, lo;
#define PAIR_CV(r)                                                                                          \
  {                                                                                                         \
    const ushort hb = rne_bf16(pv.s##r);                                                                    \
    hi.s##r = as_short(hb);                                                                                 \
    lo.s##r = as_short(rne_bf16(pv.s##r - bf16f(hb)));                                                      \
  }
        PAIR_CV(0) PAIR_CV(1) PAIR_CV(2) PAIR_CV(3) PAIR_CV(4) PAIR_CV(5) PAIR_CV(6) PAIR_CV(7)
#undef PAIR_CV
        intel_sub_group_block_write_us8(sp + (rb * 4u + b) * 128u, as_ushort8(hi));
        intel_sub_group_block_write_us8(sp + 1024u + (rb * 4u + b) * 128u, as_ushort8(lo));
      }
      if (k == 0) intel_sub_group_block_write8((__local uint*)(sa + rb * 128u), as_uint8(part));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 1) {
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        float8 psum = as_float8(intel_sub_group_block_read8((__local const uint*)(sa + rb * 128u)));
        psum += sacc[rb][0];
        psum += sacc[rb][1];
        float8 rsum;
#define PAIR_SUM(r) rsum.s##r = sub_group_reduce_add(psum.s##r);
        PAIR_SUM(0) PAIR_SUM(1) PAIR_SUM(2) PAIR_SUM(3) PAIR_SUM(4) PAIR_SUM(5) PAIR_SUM(6) PAIR_SUM(7)
#undef PAIR_SUM
        lsum[rb] = lsum[rb] * corr[rb] + rsum;
      }
    }
#pragma unroll
    for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd) o[rb][dd] *= corr[rb];
    // O += P V over this sub-group's dim atoms 4k + dd, key atoms in order.
#if KV8_SHARE
#pragma unroll
    for (uint b = 0; b < NKA; b += 2) {
      short8 pa[2][2], pl[2][2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
        for (uint ks = 0; ks < 2u; ++ks) {
          pa[rb][ks] = as_short8(intel_sub_group_block_read_us8(sp + (rb * 4u + b + ks) * 128u));
          pl[rb][ks] = as_short8(intel_sub_group_block_read_us8(sp + 1024u + (rb * 4u + b + ks) * 128u));
        }
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd)
#pragma unroll
        for (uint ks = 0; ks < 2u; ++ks) {
          const int8 vv = as_int8(intel_sub_group_block_read8(kvw + (4u * b + 4u * k + dd) * 256u + 128u * ks));
#pragma unroll
          for (uint rb = 0; rb < 2u; ++rb) {
            o[rb][dd] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[rb][ks], vv, o[rb][dd]);
            o[rb][dd] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[rb][ks], vv, o[rb][dd]);
          }
        }
    }
#undef KV8_SHARE_V
#elif KV8
#pragma unroll
    for (uint b = 0; b < NKA; b += 2) {
      short8 pa[2][2], pl[2][2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb)
#pragma unroll
        for (uint ks = 0; ks < 2u; ++ks) {
          pa[rb][ks] = as_short8(intel_sub_group_block_read_us8(sp + (rb * 4u + b + ks) * 128u));
          pl[rb][ks] = as_short8(intel_sub_group_block_read_us8(sp + 1024u + (rb * 4u + b + ks) * 128u));
        }
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; dd += 2) {
        uint vb[32];
#pragma unroll
        for (uint c = 0; c < 2u; ++c) {
          uint v4[8];
          intel_sub_group_2d_block_read_transform_8b_32r16x1c((__global void*)Vc, KV_HEADS * HD, (int)depth,
              KV_HEADS * HD, (int2)((int)(j * HD + 16u * (4u * k + dd + c)), (int)(t0 + 16u * b)), v4);
#pragma unroll
          for (uint z = 0; z < 8u; ++z) {
            const char4 c4 = as_char4(v4[z]);
            vb[c * 16u + 2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;
            vb[c * 16u + 2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;
          }
        }
#pragma unroll
        for (uint c = 0; c < 2u; ++c)
#pragma unroll
          for (uint ks = 0; ks < 2u; ++ks) {
            const int8 vv = as_int8(vload8(0, vb + c * 16u + 8u * ks));
#pragma unroll
            for (uint rb = 0; rb < 2u; ++rb) {
              o[rb][dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[rb][ks], vv, o[rb][dd + c]);
              o[rb][dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[rb][ks], vv, o[rb][dd + c]);
            }
          }
      }
    }
#else
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      short8 pa[2], pl[2];
#pragma unroll
      for (uint rb = 0; rb < 2u; ++rb) {
        pa[rb] = as_short8(intel_sub_group_block_read_us8(sp + (rb * 4u + b) * 128u));
        pl[rb] = as_short8(intel_sub_group_block_read_us8(sp + 1024u + (rb * 4u + b) * 128u));
      }
      ushort vk[16][4];   // key 16 b + kk's dims 64 k + 16 c + lane
#pragma unroll
      for (uint kk = 0; kk < 16; ++kk) {
        const uint row = KROW(min(t0 + 16u * b + kk, depth - 1));
        vstore4(intel_sub_group_block_read_us4(Vc + (size_t)row * KV_HEADS * HD + j * HD + 64u * k), 0, vk[kk]);
      }
#pragma unroll
      for (uint c = 0; c < 4u; ++c) {
        uint pairs[8];
#pragma unroll
        for (uint kk = 0; kk < 8; ++kk) pairs[kk] = (uint)vk[2 * kk][c] | ((uint)vk[2 * kk + 1][c] << 16);
        const int8 vv = as_int8(vload8(0, pairs));
#pragma unroll
        for (uint rb = 0; rb < 2u; ++rb) {
          o[rb][c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[rb], vv, o[rb][c]);
          o[rb][c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[rb], vv, o[rb][c]);
        }
      }
    }
#endif
  }
#undef KROW
#if KV8_SHARE
#undef KV8_UNIT
#endif
  if (k == 1 && l == 0) {
    vstore8(lsum[0], 0, sl + hp * 16u);
    vstore8(lsum[1], 0, sl + hp * 16u + 8u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#if KV8
  // The 8-row form's KV8 epilogue on this sub-group's dim atoms: O / l, the signs, the FWHT across the lanes, the
  // dim-atom stages 16 and 32 here, 64 with the other half through SLM, x 1/8, the gate.
  float8 u[2][NDA / 2];
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    const float8 ls = vload8(0, sl + hp * 16u + 8u * rb);
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      const uint di = 16u * (4u * k + dd) + l;
      float8 w = o[rb][dd] / ls;
      w.s0 = kv8_flip(w.s0, di); w.s1 = kv8_flip(w.s1, di); w.s2 = kv8_flip(w.s2, di); w.s3 = kv8_flip(w.s3, di);
      w.s4 = kv8_flip(w.s4, di); w.s5 = kv8_flip(w.s5, di); w.s6 = kv8_flip(w.s6, di); w.s7 = kv8_flip(w.s7, di);
      u[rb][dd] = w;
    }
#pragma unroll
    for (uint hs = 1u; hs < 16u; hs <<= 1)
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd) {
        const float8 v = u[rb][dd];
        float8 w;
        w.s0 = intel_sub_group_shuffle_xor(v.s0, hs); w.s1 = intel_sub_group_shuffle_xor(v.s1, hs);
        w.s2 = intel_sub_group_shuffle_xor(v.s2, hs); w.s3 = intel_sub_group_shuffle_xor(v.s3, hs);
        w.s4 = intel_sub_group_shuffle_xor(v.s4, hs); w.s5 = intel_sub_group_shuffle_xor(v.s5, hs);
        w.s6 = intel_sub_group_shuffle_xor(v.s6, hs); w.s7 = intel_sub_group_shuffle_xor(v.s7, hs);
        u[rb][dd] = (l & hs) ? w - v : v + w;
      }
#pragma unroll
    for (uint ka = 1u; ka < NDA / 2; ka <<= 1)
#pragma unroll
      for (uint dd = 0; dd < NDA / 2; ++dd)
        if (!(dd & ka)) {
          const float8 a = u[rb][dd], c = u[rb][dd + ka];
          u[rb][dd] = a + c;
          u[rb][dd + ka] = a - c;
        }
  }
  // The halves meet one row block at a time in the walk's SLM (sa / sp, free now): [k][dd][r * 16 + lane].
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    const uint rr = r0 + 8u * rb;
    if (rb) barrier(CLK_LOCAL_MEM_FENCE);                               // row block 0's reads are done
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd)
      intel_sub_group_block_write8((__local uint*)(sa + (k * 4u + dd) * 128u), as_uint8(u[rb][dd]));
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rr >= C) continue;                                              // a block wholly past C is not written
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      const float8 x = as_float8(intel_sub_group_block_read8((__local const uint*)(sa + ((1u - k) * 4u + dd) * 128u)));
      const float8 w = k == 0 ? u[rb][dd] + x : x - u[rb][dd];
      float wv[8];
      vstore8(w * 0.125f, 0, wv);
      for (uint r = 0; r < 8; ++r) {
        const uint row = rr + r;
        if (row >= C) break;
        const size_t i = ((size_t)row * Q_HEADS + h) * HD + 16u * (4u * k + dd) + l;
        out[i] = rne_bf16(wv[r] / (1.f + exp(-gate[i])));
      }
    }
  }
#else
#pragma unroll
  for (uint rb = 0; rb < 2u; ++rb) {
    const uint rr = r0 + 8u * rb;
    if (rr >= C) break;                                                 // a block wholly past C is not written
    const float8 ls = vload8(0, sl + hp * 16u + 8u * rb);
#pragma unroll
    for (uint dd = 0; dd < NDA / 2; ++dd) {
      const float8 w = o[rb][dd] / ls;
      float wv[8];
      vstore8(w, 0, wv);
      for (uint r = 0; r < 8; ++r) {
        const uint row = rr + r;
        if (row >= C) break;
        const size_t i = ((size_t)row * Q_HEADS + h) * HD + 16u * (4u * k + dd) + l;
        out[i] = rne_bf16(wv[r] / (1.f + exp(-gate[i])));
      }
    }
  }
#endif
}
#else
__attribute__((reqd_work_group_size(SG * SGS, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void kol_pf_flash_attn(__global const float* restrict Q, __global const KV_T* restrict Kc,
                                __global const KV_T* restrict Vc,
#if KV8
                                __global const ushort* restrict Ks, __global const ushort* restrict Vs,
#endif
                                __global ushort* restrict out, uint pos,
#if INK_PREFILL
                                uint C, __global const float* restrict relative,
                                __global const float* restrict table, __global const float* restrict sinks,
                                __global const float* restrict gate) {
#else
                                uint C) {
#endif
  const uint s = get_sub_group_id(), l = get_sub_group_local_id();
  const uint j = get_group_id(1);                                       // kv head
  const uint h = j * GQA + get_group_id(2) * HPW + s / (RPW / 8u);      // q head
  const uint r0 = get_group_id(0) * RPW + 8u * (s % (RPW / 8u));        // first of 8 rows
  if (r0 >= C) return;                                                  // no barriers below
  const uint depth = pos + C;
  const uint p0 = pos + r0;                                             // the group's first position
#if SLIDING
  const int rows_h = RING;                                              // the ring's rows
  const uint tstart = p0 > (uint)(WINDOW - 1) ? (p0 - (uint)(WINDOW - 1)) / KT * KT : 0u;
#define KROW(t) ((t) & (uint)(RING - 1))
#else
  const int rows_h = (int)depth;                                        // rows past depth read as zero
  const uint tstart = 0u;
#define KROW(t) (t)
#endif
  const int kt_w = KV_HEADS * HD * 2, kt_p = KV_HEADS * HD * 2;
  const int v_w = KV_HEADS * HD * 2, v_p = KV_HEADS * HD * 2;

  // Q, 8 rows x 128 dims as NDA A operands: qa[kk].s_r = q[r0 + r][h][16 kk + lane].
  short8 qa[NDA];
#pragma unroll
  for (uint kk = 0; kk < NDA; ++kk) {
    ushort t[8];
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint row = min(r0 + r, C - 1u);
      t[r] = rne_bf16(Q[((size_t)row * Q_HEADS + h) * HD + 16u * kk + l]);
    }
    qa[kk] = as_short8(vload8(0, t));
  }

  float8 o[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) o[dd] = (float8)(0.0f);
#if INK_PREFILL
  float8 m, lsum = (float8)(1.0f);
  float sink[8];
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    sink[r] = sinks[(size_t)min(r0 + r, C - 1u) * 64 + h];
  m = vload8(0, sink);
  float rqv[8];   // row r's relative query, lane d holding component d (KOL_BIAS broadcasts it)
#pragma unroll
  for (uint r = 0; r < 8; ++r) rqv[r] = relative[((size_t)min(r0 + r, C - 1u) * Q_HEADS + h) * 16 + l];
#else
  float8 m = (float8)(-INFINITY), lsum = (float8)(0.0f);
#endif
  const uint last = min(depth, p0 + 8u);                // keys this SG's rows can see

  // The score tile at absolute key t0: S = Q K^T over KT keys, scored (score_of) and masked. Component
  // r = row r0 + r; lane = key t0 + 16 b + lane; the 2D read at ring row KROW(t0) + 16 b.
#if INK_PREFILL
#define KOL_SCORES(t0)                                                                                      \
  float8 sacc[NKA];                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk)                                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                                        \
    const uint row = KROW(min((t0) + 16u * b + l, depth - 1));                                              \
    const __global uint* words = (const __global uint*)                                                     \
        (Kc + (size_t)row * KV_HEADS * HD + j * HD + 16u * kk);                                             \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, words)), sacc[b]);         \
  }                                                                                                         \
  KOL_BIAS(t0)                                                                                              \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                                        \
    const uint key = (t0) + 16u * b + l;                                                                    \
    KOL_MASK(0) KOL_MASK(1) KOL_MASK(2) KOL_MASK(3) KOL_MASK(4) KOL_MASK(5) KOL_MASK(6) KOL_MASK(7)         \
  }
// The relative bias of a tile: row r's distance to key t0 + 16 b + l is row 0's to lane l - r, so each table
// column is read once a tile (NKA + 1 lane segments, one a key atom and one before) and shifted to the rows by a
// shuffle. The same fma chain a score (d ascending from 0); out-of-range distances read a clamped column and are
// dropped by KOL_MASK. A tile with no key within 1,024 of its rows (a full layer's early keys) skips it.
#define KOL_BIAS(t0)                                                                                        \
  float8 bias[NKA];                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) bias[b] = (float8)(0.0f);                                \
  if ((int)p0 - (int)((t0) + KT - 1u) < 1024) {                                                             \
    const int dbase = (int)p0 - (int)(t0) - (int)l;                                                          \
    _Pragma("unroll") for (uint d = 0; d < 16; ++d) {                                                       \
      float prev = table[d * 1024 + clamp(dbase + 16, 0, 1023)];                                            \
      _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                                    \
        const float cur = table[d * 1024 + clamp(dbase - 16 * (int)b, 0, 1023)];                            \
        KOL_BIAS_ROW(0) KOL_BIAS_ROW(1) KOL_BIAS_ROW(2) KOL_BIAS_ROW(3)                                     \
        KOL_BIAS_ROW(4) KOL_BIAS_ROW(5) KOL_BIAS_ROW(6) KOL_BIAS_ROW(7)                                     \
        prev = cur;                                                                                         \
      }                                                                                                     \
    }                                                                                                       \
  }
#define KOL_BIAS_ROW(r)                                                                                     \
  bias[b].s##r = fma(sub_group_broadcast(rqv[r], d), intel_sub_group_shuffle_up(prev, cur, (uint)(r)), bias[b].s##r);
#else
#define KOL_SCORES(t0)                                                                      \
  float8 sacc[NKA];                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; ++kk)                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    uint kb[8];                                                                             \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c(                                    \
        (__global void*)Kc, kt_w, rows_h, kt_p,                                             \
        (int2)((int)((j * HD + 16u * kk) / 2u), (int)(KROW(t0) + 16u * b)), kb);            \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]); \
  }                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                        \
    const uint key = (t0) + 16u * b + l;                                                    \
    KOL_MASK(0) KOL_MASK(1) KOL_MASK(2) KOL_MASK(3) KOL_MASK(4) KOL_MASK(5) KOL_MASK(6) KOL_MASK(7) \
  }
#endif
#if KV8
#undef KOL_SCORES
// KV8: a key atom's 32 dims (two dim atoms) in one transposed 2D read of the int8 rows (lane = key, uint z = dims
// 4 z .. 4 z + 3), widened to bf16 words (exact), the two DPAS in the dim order; then each lane's key scale on its
// scores (key in the lane). Rows past depth read as zeros; their scores are masked.
#define KOL_SCORES(t0)                                                                                      \
  float8 sacc[NKA];                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) sacc[b] = (float8)(0.0f);                                \
  _Pragma("unroll") for (uint kk = 0; kk < NDA; kk += 2)                                                    \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                                        \
    uint k4[8], kb[16];                                                                                     \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV_HEADS * HD, (int)depth,      \
        KV_HEADS * HD, (int2)((int)((j * HD + 16u * kk) / 4u), (int)((t0) + 16u * b)), k4);                 \
    _Pragma("unroll") for (uint z = 0; z < 8u; ++z) {                                                       \
      const char4 c4 = as_char4(k4[z]);                                                                     \
      kb[2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;                                                   \
      kb[2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;                                              \
    }                                                                                                       \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk], as_int8(vload8(0, kb)), sacc[b]);            \
    sacc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(qa[kk + 1], as_int8(vload8(1, kb)), sacc[b]);        \
  }                                                                                                         \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b)                                                          \
    sacc[b] *= vload_half((size_t)min((t0) + 16u * b + l, depth - 1) * KV_HEADS + j, (__global const half*)Ks); \
  KOL_BIAS(t0)                                                                                              \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) {                                                        \
    const uint key = (t0) + 16u * b + l;                                                                    \
    KOL_MASK(0) KOL_MASK(1) KOL_MASK(2) KOL_MASK(3) KOL_MASK(4) KOL_MASK(5) KOL_MASK(6) KOL_MASK(7)         \
  }
#endif
#if SLIDING
#define KOL_VISIBLE(r) (key <= p0 + r && key < depth && key + (uint)WINDOW > p0 + r)
#else
#define KOL_VISIBLE(r) (key <= p0 + r && key < depth)
#endif
#if INK_PREFILL
#define KOL_MASK(r)                                                                         \
  sacc[b].s##r = KOL_VISIBLE(r) && r0 + r < C                                                   \
      ? score_of(sacc[b].s##r) + (p0 + r - key < 1024u ? bias[b].s##r : 0.f) : -INFINITY;
#else
#define KOL_MASK(r) sacc[b].s##r = KOL_VISIBLE(r) ? score_of(sacc[b].s##r) : -INFINITY;
#endif

  // P·V for one tile: A = pa[b] (8 rows x 16 keys), B = V (16 keys x 16 dims, VNNI).
#define KOL_PV(t0)                                                                          \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                  \
    uint vb[32];                                                                            \
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(                                   \
        (__global void*)Vc, v_w, rows_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(KROW(t0) + 16u * b)), vb); \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks)                                      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(                                 \
          pa[b + ks], as_int8(vload8(0, vb + c * 16u + 8u * ks)), o[dd + c]);               \
  }
  // The default path's P·V: P = hi + lo, both bf16 (hi = rne(p), lo = rne(p - hi): p to ~16 bits), each V
  // tile read once and multiplied by both halves - fp32 P to within 2^-17 of each p instead of bf16's 2^-9.
#if INK_PREFILL
// A key's 64 dims in one block read (lane l: dims 16 c + l, c < 4), 16 keys a half: 32 reads a key atom,
// not 128. The same words and DPAS order (atom b, then dim block dd, the hi half then lo).
#define KOL_PV_HL(t0)                                                                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b)                                                          \
  _Pragma("unroll") for (uint hv = 0; hv < 2u; ++hv) {                                                \
    ushort vk[16][4];                                                                                       \
    _Pragma("unroll") for (uint k = 0; k < 16; ++k) {                                                       \
      const uint row = KROW(min((t0) + 16u * b + k, depth - 1));                                            \
      vstore4(intel_sub_group_block_read_us4(Vc + (size_t)row * KV_HEADS * HD + j * HD + 64u * hv), 0, vk[k]); \
    }                                                                                                       \
    _Pragma("unroll") for (uint c = 0; c < 4u; ++c) {                                                       \
      const uint dd = 4u * hv + c;                                                                        \
      uint pairs[8];                                                                                        \
      _Pragma("unroll") for (uint k = 0; k < 8; ++k) pairs[k] = (uint)vk[2 * k][c] | ((uint)vk[2 * k + 1][c] << 16); \
      const int8 vv = as_int8(vload8(0, pairs));                                                            \
      o[dd] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[b], vv, o[dd]);                                   \
      o[dd] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[b], vv, o[dd]);                                   \
    }                                                                                                       \
  }
#else
#define KOL_PV_HL(t0)                                                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                  \
    uint vb[32];                                                                            \
    intel_sub_group_2d_block_read_transform_16b_32r16x2c(                                   \
        (__global void*)Vc, v_w, rows_h, v_p, (int2)((int)(j * HD + 16u * dd), (int)(KROW(t0) + 16u * b)), vb); \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks) {                                    \
      const int8 vv = as_int8(vload8(0, vb + c * 16u + 8u * ks));                           \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[b + ks], vv, o[dd + c]);      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[b + ks], vv, o[dd + c]);      \
    }                                                                                       \
  }
#endif
#if KV8
#undef KOL_PV_HL
// KV8: two key atoms x one dim atom in one transform read of the int8 rows (lane = dim, uint z = keys 4 z .. 4 z + 3),
// widened to the bf16 VNNI words (keys 2 i, 2 i + 1); the DPAS order of each O atom is the one-row reads' (atom b's
// hi then lo, then b + 1's).
#define KOL_PV_HL(t0)                                                                                       \
  _Pragma("unroll") for (uint b = 0; b < NKA; b += 2)                                                       \
  _Pragma("unroll") for (uint dd = 0; dd < NDA; dd += 2) {                                                  \
    uint vb[32];                                                                                            \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c) {                                                       \
      uint v4[8];                                                                                           \
      intel_sub_group_2d_block_read_transform_8b_32r16x1c((__global void*)Vc, KV_HEADS * HD, (int)depth,    \
          KV_HEADS * HD, (int2)((int)(j * HD + 16u * (dd + c)), (int)((t0) + 16u * b)), v4);                \
      _Pragma("unroll") for (uint z = 0; z < 8u; ++z) {                                                     \
        const char4 c4 = as_char4(v4[z]);                                                                   \
        vb[c * 16u + 2u * z] = i8_bf16(c4.s0) | i8_bf16(c4.s1) << 16;                                       \
        vb[c * 16u + 2u * z + 1u] = i8_bf16(c4.s2) | i8_bf16(c4.s3) << 16;                                  \
      }                                                                                                     \
    }                                                                                                       \
    _Pragma("unroll") for (uint c = 0; c < 2u; ++c)                                                         \
    _Pragma("unroll") for (uint ks = 0; ks < 2u; ++ks) {                                                    \
      const int8 vv = as_int8(vload8(0, vb + c * 16u + 8u * ks));                                           \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[b + ks], vv, o[dd + c]);                      \
      o[dd + c] = intel_sub_group_bf16_bf16_matrix_mad_k16(pl[b + ks], vv, o[dd + c]);                      \
    }                                                                                                       \
  }
#endif

#define KOL_ROWMAX()                                                                        \
  float8 tmax = (float8)(-INFINITY);                                                        \
  _Pragma("unroll") for (uint b = 0; b < NKA; ++b) tmax = fmax(tmax, sacc[b]);              \
  float8 mnew;                                                                              \
  mnew.s0 = fmax(m.s0, sub_group_reduce_max(tmax.s0));                                      \
  mnew.s1 = fmax(m.s1, sub_group_reduce_max(tmax.s1));                                      \
  mnew.s2 = fmax(m.s2, sub_group_reduce_max(tmax.s2));                                      \
  mnew.s3 = fmax(m.s3, sub_group_reduce_max(tmax.s3));                                      \
  mnew.s4 = fmax(m.s4, sub_group_reduce_max(tmax.s4));                                      \
  mnew.s5 = fmax(m.s5, sub_group_reduce_max(tmax.s5));                                      \
  mnew.s6 = fmax(m.s6, sub_group_reduce_max(tmax.s6));                                      \
  mnew.s7 = fmax(m.s7, sub_group_reduce_max(tmax.s7));                                      \
  const float8 msafe = m_safe(mnew);
#define KOL_ROWSUM(psum, rsum)                                                              \
  float8 rsum;                                                                              \
  rsum.s0 = sub_group_reduce_add(psum.s0); rsum.s1 = sub_group_reduce_add(psum.s1);         \
  rsum.s2 = sub_group_reduce_add(psum.s2); rsum.s3 = sub_group_reduce_add(psum.s3);         \
  rsum.s4 = sub_group_reduce_add(psum.s4); rsum.s5 = sub_group_reduce_add(psum.s5);         \
  rsum.s6 = sub_group_reduce_add(psum.s6); rsum.s7 = sub_group_reduce_add(psum.s7);
#define KOL_CV(r) pa[b].s##r = as_short(rne_bf16(p.s##r));
#define KOL_CV_HL(r)                                                                        \
  {                                                                                         \
    const ushort hb = rne_bf16(p.s##r);                                                     \
    pa[b].s##r = as_short(hb);                                                              \
    pl[b].s##r = as_short(rne_bf16(p.s##r - bf16f(hb)));                                    \
  }
#if KV8
#undef KOL_CV_HL
// KV8: P times the lane's key's V scale (one fp32 rounding), then the halves; psum stays P's.
#define KOL_CV_HL(r)                                                                                        \
  {                                                                                                         \
    const float pv = p.s##r * vload_half((size_t)min(t0 + 16u * b + l, depth - 1) * KV_HEADS + j,           \
                                         (__global const half*)Vs);                                         \
    const ushort hb = rne_bf16(pv);                                                                         \
    pa[b].s##r = as_short(hb);                                                                              \
    pl[b].s##r = as_short(rne_bf16(pv - bf16f(hb)));                                                        \
  }
#endif

#if !EAGER
  // The default path: the online softmax, P = EXPF(s - m) as hi + lo bf16 halves for P·V (KOL_PV_HL), O / l
  // last. A single bf16 P (2^-9 per weight against the fp32 l) moved Kolibri-1's real-checkpoint prefill off
  // decode's fp32-P attention: the code prompt's layer-16 attention rows to cosine 0.974, the residual medians
  // to 0.962 of f32 by layer 24 (decode 0.994), the routing 8762 sets off f32 against the bf16 golden's 7981
  // (kolibri_golden_prefill_test, 2026-11-01); hi + lo: 7963 (7792 with the halves' P·V in two passes - the
  // count's own noise), the code prompt's layer-24 median 0.995. Cost: pp32768 2304 against 2380 t/s (-3 %).
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    KOL_ROWMAX()
    const float8 corr = EXPF(m - msafe);         // m = -INF before a row's first key: EXPF(-INF) = 0
    float8 psum = (float8)(0.0f);
    short8 pa[NKA], pl[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = EXPF(sacc[b] - msafe);    // masked: EXPF(-INF) = 0 (hi and lo both 0)
      psum += p;
      KOL_CV_HL(0) KOL_CV_HL(1) KOL_CV_HL(2) KOL_CV_HL(3) KOL_CV_HL(4) KOL_CV_HL(5) KOL_CV_HL(6) KOL_CV_HL(7)
    }
    KOL_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) o[dd] *= corr;
    KOL_PV_HL(t0)
  }
#else
  // Pass 1: the row's max and Σ exp(s - max) over its keys (fp32, online).
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    KOL_ROWMAX()
    const float8 corr = exp(m - msafe);
    float8 psum = (float8)(0.0f);
#pragma unroll
    for (uint b = 0; b < NKA; ++b) psum += exp(sacc[b] - msafe);
    KOL_ROWSUM(psum, rsum)
    lsum = lsum * corr + rsum;
    m = mnew;
  }
  // Pass 2: p = rne(exp(s - m) / l), the softmax's bf16 output; O = Σ p v in fp32. Every row saw its own
  // key, so m is finite here.
  for (uint t0 = tstart; t0 < last; t0 += KT) {
    KOL_SCORES(t0)
    short8 pa[NKA];
#pragma unroll
    for (uint b = 0; b < NKA; ++b) {
      const float8 p = exp(sacc[b] - m) / lsum;
      KOL_CV(0) KOL_CV(1) KOL_CV(2) KOL_CV(3) KOL_CV(4) KOL_CV(5) KOL_CV(6) KOL_CV(7)
    }
    KOL_PV(t0)
  }
#endif
#undef KOL_SCORES
#undef KOL_BIAS
#undef KOL_BIAS_ROW
#undef KOL_VISIBLE
#undef KOL_MASK
#undef KOL_PV
#undef KOL_PV_HL
#undef KOL_ROWMAX
#undef KOL_ROWSUM
#undef KOL_CV
#undef KOL_CV_HL
#undef KROW

#if KV8
  // KV8's epilogue: O / l un-rotated per row (kv8.h unrotate_n: the signs, the FWHT's stages h = 1..8 across the
  // lanes, then 16, 32, 64 across the dim atoms, x 1/8), then the gate, as below.
  float8 u[NDA];
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    u[dd] = o[dd] / lsum;
    u[dd].s0 = kv8_flip(u[dd].s0, 16u * dd + l); u[dd].s1 = kv8_flip(u[dd].s1, 16u * dd + l);
    u[dd].s2 = kv8_flip(u[dd].s2, 16u * dd + l); u[dd].s3 = kv8_flip(u[dd].s3, 16u * dd + l);
    u[dd].s4 = kv8_flip(u[dd].s4, 16u * dd + l); u[dd].s5 = kv8_flip(u[dd].s5, 16u * dd + l);
    u[dd].s6 = kv8_flip(u[dd].s6, 16u * dd + l); u[dd].s7 = kv8_flip(u[dd].s7, 16u * dd + l);
  }
#pragma unroll
  for (uint hs = 1u; hs < 16u; hs <<= 1)
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd) {
      const float8 v = u[dd];
      float8 w;
      w.s0 = intel_sub_group_shuffle_xor(v.s0, hs); w.s1 = intel_sub_group_shuffle_xor(v.s1, hs);
      w.s2 = intel_sub_group_shuffle_xor(v.s2, hs); w.s3 = intel_sub_group_shuffle_xor(v.s3, hs);
      w.s4 = intel_sub_group_shuffle_xor(v.s4, hs); w.s5 = intel_sub_group_shuffle_xor(v.s5, hs);
      w.s6 = intel_sub_group_shuffle_xor(v.s6, hs); w.s7 = intel_sub_group_shuffle_xor(v.s7, hs);
      u[dd] = (l & hs) ? w - v : v + w;
    }
#pragma unroll
  for (uint k = 1u; k < NDA; k <<= 1)
#pragma unroll
    for (uint dd = 0; dd < NDA; ++dd)
      if (!(dd & k)) {
        const float8 a = u[dd], c = u[dd + k];
        u[dd] = a + c;
        u[dd + k] = a - c;
      }
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
    float wv[8];
    vstore8(u[dd] * 0.125f, 0, wv);
    for (uint r = 0; r < 8; ++r) {
      const uint row = r0 + r;
      if (row >= C) break;
      const size_t i = ((size_t)row * Q_HEADS + h) * HD + 16u * dd + l;
      out[i] = rne_bf16(wv[r] / (1.f + exp(-gate[i])));
    }
  }
#else
  // The epilogue: per element (row r0 + r, dim 16 dd + lane), bf16 [C][Q_HEADS x 128].
#pragma unroll
  for (uint dd = 0; dd < NDA; ++dd) {
#if EAGER
    const float8 w = o[dd];                      // already normalised
#else
    const float8 w = o[dd] / lsum;
#endif
    float wv[8];
    vstore8(w, 0, wv);
    for (uint r = 0; r < 8; ++r) {
      const uint row = r0 + r;
      if (row >= C) break;
#if INK_PREFILL
      const size_t i = ((size_t)row * Q_HEADS + h) * HD + 16u * dd + l;
      out[i] = rne_bf16(wv[r] / (1.f + exp(-gate[i])));
#else
      out[((size_t)row * Q_HEADS + h) * HD + 16u * dd + l] = rne_bf16(wv[r]);
#endif
    }
  }
#endif
}
#endif  // PAIR
