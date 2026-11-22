// q4_qsa.cl - spec 21-QFN-v1c: Qwen3.8-Flash-Next's QSA indexer and block selection (spec 21-QFN-v1 §2.3, §4.2; M:665-771),
// per QSA layer and row p = pos + m. Three kernels in one binary per M (attn.cl's arrangement):
//
//   q4_qsa_prep(ctrl, idx_f32, small, rope, idx_q, tail, idx_keys)         grid (5, M),                 WG 128
//   q4_qsa_score(ctrl, idx_q, idx_keys, scores, stride)                    grid (max_len / 4 / 256, M), WG 256
//   q4_qsa_select(ctrl, scores, stride, list, diag)                        grid (1, M),                 WG SEL_WG
//
// **q4_qsa_prep** over the indexer GEMV's fp32 row idx_f32 [M][640] (4 query heads x 128 | 1 raw key x 128):
//   work-groups 0..3 (the query heads): x = rne(idx_f32); rstd over the head (lane i: x_i^2, a plain multiply; the
//     tree 64 .. 1; 1 / sqrt); n = rne(x x rstd x (1 + w_q)); RoPE on dims [0, 64) the way torch runs it in bf16:
//     cos / sin = rne(the loader's fp32 table at p), out_i = rne(rne(n_i c) + rne(rot_i s)) (rotate_half);
//     idx_q[m][h][i] = f32(out) (bf16 values)
//   work-group 4 (the raw key): raw = rne(idx_f32[512 + i]) -> tail[p % 8] (the open block's raw keys: un-normed,
//     un-roped, bf16); when (p + 1) % 4 == 0 row p completes block b = (p + 1) / 4 - 1 and forms its compressed key
//     ONCE, from the raw keys of positions 4b .. 4b + 3 (this launch's rows straight from idx_f32, older ones from
//     the ring): the fp32 mean (((k0 + k1) + k2) + k3) / 4 -> bf16 -> k_layernorm (the 128-lane norm, (1 + w_k)) ->
//     RoPE at 4b (bf16 as above) -> idx_keys[b] (M:733-742). EIGHT tail slots (Review Focus 3): rows of one launch
//     write pos .. pos + 3 while a completing row reads pos - 3 .. pos - 1 - seven consecutive positions, distinct
//     mod 8; four slots (spec 21-QFN-v1 §4.2) would collide at a 4-row launch.
// **q4_qsa_score**: row m's n = (p + 1) / 4 complete blocks; work-group g owns blocks [256 g, 256 g + 256) and
//   exits whole past n (the grid spans max_len / 4 blocks under replay). score_b = (((r0 + r1) + r2) + r3) / sqrt(128),
//   r_h = max(sum_i q[h][i] k_b[i], +0) (one fma chain, i ascending: bf16 x bf16 products are exact, so the chain
//   is contraction-proof) - M:744-747, fp32. scores fp32 [M][stride], stride = max_len / 4. The prefill form
//   (QSA_PF) holds SCORE_R rows a work-group, grid (ceil(n / 256), ceil(M / SCORE_R)): lane b reads its key once (8
//   elements a load) for the 4 x SCORE_R chains, each chain the same fma order - bitwise the one-row form.
// **q4_qsa_select** (decision 3: EXACT ties to the lower block): n <= 512 - every visible position, the list 0 ..
//   p (QSA is plain causal attention below 2052 visible positions); else the top 512 blocks by (score desc, block
//   asc), written ASCENDING, each expanded to its 4 positions, then the open block's tail 4n .. p. The order
//   key of a block is its score's bits (every score is >= +0, so the unsigned bits order as the floats do): an
//   MSB-first radix select (4 passes of 8 bits; a local-memory histogram of integer counts - exact, so the
//   result does not depend on which lane counted first) finds the 512th score T and how many of the blocks equal
//   to T are taken; the lowest-index ones are, through an exclusive scan over contiguous per-lane chunks. A row
//   list (u32 words [M][LIST_ROW]): positions [0, count), count at word COUNT_W = 2048 + (p + 1 - 4n) (or p + 1);
//   diag [M][2] = {the 512th score, the 513th} ({+INF, -INF} without a cut) - the near-tie diagnostic (gate S).
//
// Plain OpenCL C (no sub-group functions).
//
// -DQSA_I4H=1 (`_I4H`, the engine's form) scores in int4 x int4 on DPAS after the 128-point Hadamard; the selection rule and
// the exact attention after it are unchanged. Unset, the bf16 form (the kernel test's and bench's reference), the source
// below is token for token what it was.
//   q4_qsa_prep quantises each query head and each compressed key once: the bf16-valued vector, its plain 128-point
//     Hadamard first (FWHT stages h = 1 .. 64 ascending, (i, i + h) -> (a + c, a - c), unscaled - kv8.cl's butterfly
//     without the signs, DeepSeek-V3.2's indexer rotation), in two groups of 64 (one k64 DPAS each): amax =
//     max |y|, s = amax / 7, code = s == 0 ? 0 : clamp(rint(y / s), -7, 7), element 2j the low nibble of byte j.
//     Query row m in idx_q's fp32 row footprint: nibbles [h][64] at byte 0, scales fp32 [h][2] at byte 64 IDX_HEADS.
//     Block b in idx_keys' I4_KEY_BYTES slot (72 B): nibbles [64] at byte 0, scales fp32 [2] at byte 64.
//   q4_qsa_score: sub-group = 16 blocks (lane = block = DPAS column), A = 8 query vectors (row j, head h at
//     8 t + r = j IDX_HEADS + h), one intel_sub_group_i4_i4_matrix_mad_k64 per group -> exact s32 dots d0, d1;
//     a = ((float)d0 x (sq0 x sk0)) + ((float)d1 x (sq1 x sk1)), x 2^-7 (H H^T = 128 I), then the bf16
//     form's relu, left fold over the heads and / sqrt(128). Same grids as the bf16 form (decode one row a
//     work-group-row, prefill SCORE_R).
#pragma OPENCL FP_CONTRACT OFF
#ifdef QSA_I4H
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_qsa: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TOPB) || !defined(SEL_WG) || !defined(LIST_ROW) || !defined(COUNT_W)
#error "q4_qsa: TOPB, SEL_WG, LIST_ROW and COUNT_W must be defined (kernels::qwen4exp)"
#endif
// Spec 21-QFN-v1d: -DQSA_PF builds the PREFILL form at M = C (kernels::qwen4exp::kPfC): the indexer GEMM's rows arrive at
// the prefill GEMM's pitch IDX_LD = pf_ld(640) = 768, and q4_qsa_prep writes NO tail-ring slot - a 2048-row launch's
// rows would overwrite slots (p % 8) its first rows still read for the block that straddles pos - so the ring is
// updated by its own launch, q4_qsa_ring, after the prep: the last min(8, C) raw keys, what decode's ring holds
// after the same ids. Unset (every 21-QFN-v1c binary), IDX_LD is IDX_N and the tail write is the prep's, token for token
// (commit 6499948's rule: the preprocessed source of every existing binary is unchanged).
#ifdef QSA_PF
#if M < 1 || M > 4096
#error "q4_qsa: the prefill form is M = 1..4096 (kernels::qwen4exp::kPfCWide)"
#endif
#ifndef SCORE_R
#error "q4_qsa: the prefill form's SCORE_R (kernels::qwen4exp::kPfScoreRows) must be defined"
#endif
#elif M < 1 || M > 4
#error "q4_qsa: M is 1..4 (the 8-slot tail ring holds a 4-row launch)"
#endif
#define IDX_HEADS 4
#define IDX_DIM 128
#define IDX_N 640
#ifdef QSA_PF
#define IDX_LD 768          /* pf_ld(640): runtime::qwen4exp::pf_ld */
#else
#define IDX_LD IDX_N
#endif
#define BLK 4
#define TAIL 8
#define ROT_HALF 32
#define IDX_QNORM_OFF 512   /* the QSA small block in floats (loader/qwen4exp_layout.h Q4QsaSmall) */
#define IDX_KNORM_OFF 640
#define SQRT_IDX 11.313708498984761f

#include "bf16.h"
inline float rf(float f) { return bf16f(rne_bf16(f)); }
inline uint n_rows(__global const uint* restrict ctrl) {
  const uint n = ctrl[CTRL_NACT];
  return n > M ? M : n;
}

// ---------------------------------------------------------------------------------------------------------------
#define WG_PREP IDX_DIM
// The 128-lane head norm and torch's bf16 RoPE of one head held in v (lane i: v[i] = x_i, bf16 values) at the
// table row cs ([2][32] cos then sin, fp32): returns lane i's output (bf16 value).
inline float norm_rope(__local float* v, __local float* red, float x, __global const float* restrict w,
                       __global const float* restrict cs, uint i) {
  red[i] = x * x;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG_PREP / 2; stride > 0; stride >>= 1) {
    if (i < stride) red[i] += red[i + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rstd = 1.0f / sqrt(red[0] / (float)IDX_DIM + 1e-6f);
  v[i] = rf(x * rstd * w[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float o = v[i];
  if (i < 2 * ROT_HALF) {
    const uint ii = i % ROT_HALF;
    const float c = rf(cs[ii]), s = rf(cs[ROT_HALF + ii]);
    const float rot = i < ROT_HALF ? -v[i + ROT_HALF] : v[i - ROT_HALF];
    o = rf(rf(v[i] * c) + rf(rot * s));
  }
  barrier(CLK_LOCAL_MEM_FENCE);   // v and red are reused
  return o;
}

#ifdef QSA_I4H
#define I4_GROUP 64
#define I4_ROW_BYTES (IDX_HEADS * IDX_DIM * 4)   /* a row's footprint in idx_q (fp32 in the bf16 form) */
#define I4_NIB (IDX_DIM / 2)                     /* nibble bytes of one vector */
#define I4_KEY_BYTES (I4_NIB + 8)                /* a block's slot in idx_keys: nibbles + two fp32 scales (bf16 form: IDX_DIM * 2) */
// One vector (lane i: x, a bf16 value) -> its int4 codes and two group scales (the header's rule). Every lane calls it.
inline void i4_pack(__local float* v, __local float* red, float x, uint i, __global uchar* restrict nib,
                    __global float* restrict sc) {
  v[i] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint hb = 1; hb < IDX_DIM; hb <<= 1) {
    const float a = v[i & ~hb], c = v[i | hb];
    barrier(CLK_LOCAL_MEM_FENCE);
    v[i] = (i & hb) ? a - c : a + c;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  x = v[i];
  red[i] = fabs(x);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = I4_GROUP / 2; s > 0; s >>= 1) {
    if (i % I4_GROUP < s) red[i] = fmax(red[i], red[i + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float scale = red[i / I4_GROUP * I4_GROUP] / 7.0f;
  const float code = scale == 0.0f ? 0.0f : clamp(rint(x / scale), -7.0f, 7.0f);
  barrier(CLK_LOCAL_MEM_FENCE);   // every lane has its group's amax before v is reused
  v[i] = code;
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((i & 1u) == 0) nib[i / 2] = (uchar)(((int)v[i] & 15) | (((int)v[i + 1] & 15) << 4));
  if (i % I4_GROUP == 0) sc[i / I4_GROUP] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

//   idx_f32 fp32 [M][640]; small fp32: the layer's QSA block (idx q norm at 512, k norm at 640: (1 + w));
//   rope fp32 [max_len][2][32]; idx_q fp32 [M][4][128]; tail bf16 [8][128]; idx_keys bf16 [max_len / 4][128]
__attribute__((reqd_work_group_size(WG_PREP, 1, 1)))
__kernel void q4_qsa_prep(__global const uint* restrict ctrl, __global const float* restrict idx_f32,
                          __global const float* restrict small, __global const float* restrict rope,
                          __global float* restrict idx_q, __global ushort* restrict tail,
                          __global ushort* restrict idx_keys) {
  const uint h = get_group_id(0);
  const uint m = get_group_id(1);
  const uint i = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  __local float v[IDX_DIM], red[IDX_DIM];
  const uint pos = ctrl[CTRL_POS];
  const uint p = pos + m;
  if (h < IDX_HEADS) {
    const float x = rf(idx_f32[(size_t)m * IDX_LD + h * IDX_DIM + i]);
#ifdef QSA_I4H
    const float o = norm_rope(v, red, x, small + IDX_QNORM_OFF, rope + (size_t)p * 2 * ROT_HALF, i);
    __global uchar* restrict row = (__global uchar*)idx_q + (size_t)m * I4_ROW_BYTES;
    i4_pack(v, red, o, i, row + h * I4_NIB, (__global float*)(row + IDX_HEADS * I4_NIB) + h * 2);
#else
    idx_q[((size_t)m * IDX_HEADS + h) * IDX_DIM + i] =
        norm_rope(v, red, x, small + IDX_QNORM_OFF, rope + (size_t)p * 2 * ROT_HALF, i);
#endif
    return;
  }
  const ushort raw = rne_bf16(idx_f32[(size_t)m * IDX_LD + IDX_HEADS * IDX_DIM + i]);
  if ((p + 1) % BLK == 0) {   // uniform: row p completes block b
    const uint b = (p + 1) / BLK - 1;
    float k4[BLK];
    for (uint j = 0; j < BLK; ++j) {
      const uint q = b * BLK + j;
      k4[j] = q >= pos ? rf(idx_f32[(size_t)(q - pos) * IDX_LD + IDX_HEADS * IDX_DIM + i])
                       : bf16f(tail[(size_t)(q % TAIL) * IDX_DIM + i]);
    }
    const float mean = rf((((k4[0] + k4[1]) + k4[2]) + k4[3]) / 4.0f);
    const float o = norm_rope(v, red, mean, small + IDX_KNORM_OFF, rope + (size_t)(b * BLK) * 2 * ROT_HALF, i);
#ifdef QSA_I4H
    __global uchar* restrict slot = (__global uchar*)idx_keys + (size_t)b * I4_KEY_BYTES;
    i4_pack(v, red, o, i, slot, (__global float*)(slot + I4_NIB));   // o is the bf16 key's value
#else
    idx_keys[(size_t)b * IDX_DIM + i] = rne_bf16(o);
#endif
  }
#ifndef QSA_PF
  tail[(size_t)(p % TAIL) * IDX_DIM + i] = raw;
#else
  (void)raw;   // q4_qsa_ring writes the ring after this launch
#endif
}

#ifdef QSA_PF
// q4_qsa_ring(ctrl, idx_f32, tail)   grid (1, TAIL), WG 128 - the prefill form's ring update: work-group y writes the
// y-th of the chunk's last TAIL rows (fewer rows: only the last C), tail[(pos + m) % 8] = rne(idx_f32[m][512 + i]).
__attribute__((reqd_work_group_size(WG_PREP, 1, 1)))
__kernel void q4_qsa_ring(__global const uint* restrict ctrl, __global const float* restrict idx_f32,
                          __global ushort* restrict tail) {
  const uint y = get_group_id(1);
  const uint i = get_local_id(0);
  const uint C = n_rows(ctrl);
  if (C + y < TAIL) return;   // uniform
  const uint m = C + y - TAIL;
  tail[(size_t)((ctrl[CTRL_POS] + m) % TAIL) * IDX_DIM + i] =
      rne_bf16(idx_f32[(size_t)m * IDX_LD + IDX_HEADS * IDX_DIM + i]);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
#define WG_SCORE 256
#ifdef QSA_I4H
#ifdef QSA_PF
#define I4_ROWS SCORE_R
#else
#define I4_ROWS 1
#endif
#define I4_QV (I4_ROWS * IDX_HEADS)   /* query vectors a work-group */
#define I4_QT ((I4_QV + 7) / 8)       /* DPAS A tiles of 8 */
// Rows m0 .. m0 + I4_ROWS - 1 over blocks [256 g, 256 g + 256), the bf16 forms' grids and early-outs.
__attribute__((reqd_work_group_size(WG_SCORE, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void q4_qsa_score(__global const uint* restrict ctrl, __global const float* restrict idx_q,
                           __global const ushort* restrict idx_keys, __global float* restrict scores, uint stride) {
  const uint g = get_group_id(0);
  const uint m0 = get_group_id(1) * I4_ROWS;
  const uint lid = get_local_id(0);
  const uint C = n_rows(ctrl);
  if (m0 >= C) return;   // uniform
  const uint rows = min((uint)I4_ROWS, C - m0), p0 = ctrl[CTRL_POS] + m0;
  const uint nl = (p0 + rows) / BLK;   // the last row's complete blocks
  if (g * WG_SCORE >= nl) return;      // uniform
  // A rows: query vector v's group gg, lane l's 4 nibbles (bytes 2l, 2l + 1); vectors past the rows are zero
  __local ushort qa[I4_QT * 8][2][16];
  __local float qs[I4_QT * 8][2];
  const __global uchar* restrict qb = (const __global uchar*)idx_q + (size_t)m0 * I4_ROW_BYTES;
  for (uint t = lid; t < I4_QT * 8 * 32; t += WG_SCORE) {
    const uint v = t / 32, gg = t / 16 % 2, l = t % 16;
    const __global uchar* src = qb + (size_t)(v / IDX_HEADS) * I4_ROW_BYTES + v % IDX_HEADS * I4_NIB + gg * 32;
    qa[v][gg][l] = v < rows * IDX_HEADS ? ((const __global ushort*)src)[l] : (ushort)0;
  }
  for (uint t = lid; t < I4_QT * 8 * 2; t += WG_SCORE) {
    const uint v = t / 2;
    const __global float* src = (const __global float*)(qb + (size_t)(v / IDX_HEADS) * I4_ROW_BYTES + IDX_HEADS * I4_NIB);
    qs[v][t % 2] = v < rows * IDX_HEADS ? src[v % IDX_HEADS * 2 + t % 2] : 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint l = get_sub_group_local_id();
  const uint b = g * WG_SCORE + lid;   // lane l of sub-group s: block 256 g + 16 s + l
  const bool live = b < nl;
  const __global uchar* restrict key = (const __global uchar*)idx_keys + (size_t)(live ? b : 0) * I4_KEY_BYTES;
  int d[I4_QT][2][8];
  for (uint gg = 0; gg < 2; ++gg) {   // every lane runs the DPAS (a sub-group operation); dead lanes' columns are 0
    const int8 kb = live ? as_int8(vload8(0, (const __global uint*)(key + gg * 32))) : (int8)(0);
    for (uint t = 0; t < I4_QT; ++t) {
      ushort a8[8];
      for (uint r = 0; r < 8; ++r) a8[r] = qa[t * 8 + r][gg][l];
      vstore8(intel_sub_group_i4_i4_matrix_mad_k64(as_short8(vload8(0, a8)), kb, (int8)(0)), 0, d[t][gg]);
    }
  }
  if (!live) return;
  const float ks0 = ((const __global float*)(key + I4_NIB))[0], ks1 = ((const __global float*)(key + I4_NIB))[1];
  for (uint j = 0; j < rows; ++j) {
    if (b >= (p0 + j + 1) / BLK) continue;   // past row j's complete blocks
    float s = 0.0f;                          // + 0 then the heads: the left fold ((r0 + r1) + r2) + ...
    for (uint hh = 0; hh < IDX_HEADS; ++hh) {
      const uint v = j * IDX_HEADS + hh;
      float a0 = qs[v][0] * ks0;
      a0 = (float)d[v / 8][0][v % 8] * a0;
      float a1 = qs[v][1] * ks1;
      a1 = (float)d[v / 8][1][v % 8] * a1;
      float a = a0 + a1;
      a = a * 0.0078125f;
      s += a > 0.0f ? a : 0.0f;
    }
    scores[(size_t)(m0 + j) * stride + b] = s / SQRT_IDX;
  }
}
#elif defined(QSA_PF)
// Rows m0 .. m0 + SCORE_R - 1 (the chunk's rows past n_rows: none) over blocks [256 g, 256 g + 256): the group exits
// whole past its last row's complete blocks; row j writes block b < its own n.
__attribute__((reqd_work_group_size(WG_SCORE, 1, 1)))
__kernel void q4_qsa_score(__global const uint* restrict ctrl, __global const float* restrict idx_q,
                           __global const ushort* restrict idx_keys, __global float* restrict scores, uint stride) {
  const uint g = get_group_id(0);
  const uint m0 = get_group_id(1) * SCORE_R;
  const uint lid = get_local_id(0);
  const uint C = n_rows(ctrl);
  if (m0 >= C) return;   // uniform
  const uint rows = min((uint)SCORE_R, C - m0), p0 = ctrl[CTRL_POS] + m0;
  const uint nl = (p0 + rows) / BLK;   // the last row's complete blocks
  if (g * WG_SCORE >= nl) return;      // uniform
  __local float q[SCORE_R * IDX_HEADS * IDX_DIM];
  for (uint t = lid; t < SCORE_R * IDX_HEADS * IDX_DIM; t += WG_SCORE)
    q[t] = t < rows * IDX_HEADS * IDX_DIM ? idx_q[(size_t)m0 * IDX_HEADS * IDX_DIM + t] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint b = g * WG_SCORE + lid;
  if (b >= nl) return;
  __global const ushort* restrict key = idx_keys + (size_t)b * IDX_DIM;
  float a[SCORE_R][IDX_HEADS];
  for (uint j = 0; j < SCORE_R; ++j)
    for (uint hh = 0; hh < IDX_HEADS; ++hh) a[j][hh] = 0.0f;
  for (uint t = 0; t < IDX_DIM; t += 8) {
    const ushort8 k8 = vload8(0, key + t);
    const float k[8] = {bf16f(k8.s0), bf16f(k8.s1), bf16f(k8.s2), bf16f(k8.s3),
                        bf16f(k8.s4), bf16f(k8.s5), bf16f(k8.s6), bf16f(k8.s7)};
    for (uint e = 0; e < 8; ++e)
      for (uint j = 0; j < SCORE_R; ++j)
        for (uint hh = 0; hh < IDX_HEADS; ++hh)
          a[j][hh] = fma(q[(j * IDX_HEADS + hh) * IDX_DIM + t + e], k[e], a[j][hh]);
  }
  for (uint j = 0; j < rows; ++j) {
    if (b >= (p0 + j + 1) / BLK) continue;   // past row j's complete blocks
    float r[IDX_HEADS];
    for (uint hh = 0; hh < IDX_HEADS; ++hh) r[hh] = a[j][hh] > 0.0f ? a[j][hh] : 0.0f;
    scores[(size_t)(m0 + j) * stride + b] = (((r[0] + r[1]) + r[2]) + r[3]) / SQRT_IDX;
  }
}
#else
__attribute__((reqd_work_group_size(WG_SCORE, 1, 1)))
__kernel void q4_qsa_score(__global const uint* restrict ctrl, __global const float* restrict idx_q,
                           __global const ushort* restrict idx_keys, __global float* restrict scores, uint stride) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  const uint n = (ctrl[CTRL_POS] + m + 1) / BLK;
  if (g * WG_SCORE >= n) return;   // uniform: past the row's complete blocks
  __local float q[IDX_HEADS * IDX_DIM];
  for (uint t = lid; t < IDX_HEADS * IDX_DIM; t += WG_SCORE) q[t] = idx_q[(size_t)m * IDX_HEADS * IDX_DIM + t];
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint b = g * WG_SCORE + lid;
  if (b >= n) return;
  __global const ushort* restrict key = idx_keys + (size_t)b * IDX_DIM;
  float r[IDX_HEADS];
  for (uint hh = 0; hh < IDX_HEADS; ++hh) {
    float a = 0.0f;
    for (uint t = 0; t < IDX_DIM; ++t) a = fma(q[hh * IDX_DIM + t], bf16f(key[t]), a);
    r[hh] = a > 0.0f ? a : 0.0f;
  }
  scores[(size_t)m * stride + b] = (((r[0] + r[1]) + r[2]) + r[3]) / SQRT_IDX;
}
#endif

// ---------------------------------------------------------------------------------------------------------------
#if (SEL_WG & (SEL_WG - 1)) != 0 || SEL_WG < 256
#error "q4_qsa_select: SEL_WG is a power of two >= 256 (the 256-bin histogram)"
#endif
__attribute__((reqd_work_group_size(SEL_WG, 1, 1)))
__kernel void q4_qsa_select(__global const uint* restrict ctrl, __global const float* restrict scores, uint stride,
                            __global uint* restrict list, __global float* restrict diag) {
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  if (m >= n_rows(ctrl)) return;   // uniform
  const uint p = ctrl[CTRL_POS] + m;
  const uint n = (p + 1) / BLK;
  __global uint* restrict row = list + (size_t)m * LIST_ROW;
  if (n <= TOPB) {   // every visible position: the identity list
    for (uint q = lid; q <= p; q += SEL_WG) row[q] = q;
    if (lid == 0) {
      row[COUNT_W] = p + 1;
      diag[m * 2] = INFINITY;
      diag[m * 2 + 1] = -INFINITY;
    }
    return;
  }
  __global const uint* restrict key = (__global const uint*)(scores + (size_t)m * stride);
  __local uint hist[256];
  __local uint sh[4];   // [0] prefix, [1] need, [2] the 513th's bits, [3] spare
  __local uint cnt[SEL_WG];
  // 1. the 512th largest key T and `need` - how many keys equal to T are taken - by MSB-first radix select
  uint prefix = 0u, mask = 0u, need = TOPB;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint d = lid; d < 256; d += SEL_WG) hist[d] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = lid; b < n; b += SEL_WG) {
      const uint k = key[b];
      if ((k & mask) == prefix) atomic_inc(&hist[(k >> shift) & 255u]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
      uint cum = 0u, dg = 255u;
      for (int d = 255; d >= 0; --d) {
        if (cum + hist[d] >= need) {
          dg = (uint)d;
          break;
        }
        cum += hist[d];
      }
      sh[0] = prefix | (dg << shift);
      sh[1] = need - cum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix = sh[0];
    need = sh[1];
    mask |= 255u << shift;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const uint T = prefix;
  // 2. contiguous chunks per lane: the ties each lane holds, their exclusive scan (lane 0, ascending lanes)
  const uint chunk = (n + SEL_WG - 1) / SEL_WG;
  const uint b0 = min(lid * chunk, n), b1 = min(b0 + chunk, n);
  uint ties = 0u;
  for (uint b = b0; b < b1; ++b) ties += key[b] == T ? 1u : 0u;
  cnt[lid] = ties;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    uint run = 0u;
    for (uint l = 0; l < SEL_WG; ++l) {
      const uint t = cnt[l];
      cnt[l] = run;
      run += t;
    }
    sh[2] = run;   // every key equal to T
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint tie_before = cnt[lid], all_ties = sh[2];
  // 3. the lane's selected blocks: key > T, or a tie among the `need` lowest-index ones
  uint sel = 0u, t_seen = tie_before;
  for (uint b = b0; b < b1; ++b) {
    const uint k = key[b];
    if (k > T) ++sel;
    else if (k == T && t_seen++ < need) ++sel;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  cnt[lid] = sel;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    uint run = 0u;
    for (uint l = 0; l < SEL_WG; ++l) {
      const uint t = cnt[l];
      cnt[l] = run;
      run += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint at = cnt[lid] * BLK;
  t_seen = tie_before;
  for (uint b = b0; b < b1; ++b) {
    const uint k = key[b];
    const bool take = k > T || (k == T && t_seen++ < need);
    if (take) {
      for (uint j = 0; j < BLK; ++j) row[at + j] = b * BLK + j;
      at += BLK;
    }
  }
  // 4. the open block's tail, the count, the diagnostic (the 513th: T again when a tie was left, else the
  //    largest key below T - an integer max, exact in any order)
  const uint tail0 = n * BLK, ntail = p + 1 - tail0;
  if (lid < ntail) row[TOPB * BLK + lid] = tail0 + lid;
  uint below = 0u;
  for (uint b = lid; b < n; b += SEL_WG) {
    const uint k = key[b];
    if (k < T) below = max(below, k);
  }
  cnt[lid] = below;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = SEL_WG / 2; s > 0; s >>= 1) {
    if (lid < s) cnt[lid] = max(cnt[lid], cnt[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    row[COUNT_W] = TOPB * BLK + ntail;
    diag[m * 2] = as_float(T);
    diag[m * 2 + 1] = all_ties > need ? as_float(T) : as_float(cnt[0]);
  }
}
