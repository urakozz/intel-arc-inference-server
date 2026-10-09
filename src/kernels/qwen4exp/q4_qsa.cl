// q4_qsa.cl - spec 21c: Qwen3.8-Flash-Next's QSA indexer and block selection (spec 21 §2.3, §4.2; M:665-771),
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
//     mod 8; four slots (spec 21 §4.2) would collide at a 4-row launch.
// **q4_qsa_score**: row m's n = (p + 1) / 4 complete blocks; work-group g owns blocks [256 g, 256 g + 256) and
//   exits whole past n (the grid spans max_len / 4 blocks under replay). score_b = (((r0 + r1) + r2) + r3) / sqrt(128),
//   r_h = max(sum_i q[h][i] k_b[i], +0) (one fma chain, i ascending: bf16 x bf16 products are exact, so the chain
//   is contraction-proof) - M:744-747, fp32. scores fp32 [M][stride], stride = max_len / 4.
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
// Plain OpenCL C (no sub-group functions): the Mac's OpenCL runs it.
#pragma OPENCL FP_CONTRACT OFF

#ifndef M
#define M 1
#endif
#if !defined(CTRL_POS) || !defined(CTRL_NACT)
#error "q4_qsa: CTRL_POS / CTRL_NACT must be defined (src/kernels/CMakeLists.txt)"
#endif
#if !defined(TOPB) || !defined(SEL_WG) || !defined(LIST_ROW) || !defined(COUNT_W)
#error "q4_qsa: TOPB, SEL_WG, LIST_ROW and COUNT_W must be defined (kernels::qwen4exp)"
#endif
#if M < 1 || M > 4
#error "q4_qsa: M is 1..4 (the 8-slot tail ring holds a 4-row launch)"
#endif
#define IDX_HEADS 4
#define IDX_DIM 128
#define IDX_N 640
#define BLK 4
#define TAIL 8
#define ROT_HALF 32
#define IDX_QNORM_OFF 512   /* the QSA small block in floats (loader/qwen4exp_layout.h Q4QsaSmall) */
#define IDX_KNORM_OFF 640
#define SQRT_IDX 11.313708498984761f

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
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
    const float x = rf(idx_f32[(size_t)m * IDX_N + h * IDX_DIM + i]);
    idx_q[((size_t)m * IDX_HEADS + h) * IDX_DIM + i] =
        norm_rope(v, red, x, small + IDX_QNORM_OFF, rope + (size_t)p * 2 * ROT_HALF, i);
    return;
  }
  const ushort raw = rne_bf16(idx_f32[(size_t)m * IDX_N + IDX_HEADS * IDX_DIM + i]);
  if ((p + 1) % BLK == 0) {   // uniform: row p completes block b
    const uint b = (p + 1) / BLK - 1;
    float k4[BLK];
    for (uint j = 0; j < BLK; ++j) {
      const uint q = b * BLK + j;
      k4[j] = q >= pos ? rf(idx_f32[(size_t)(q - pos) * IDX_N + IDX_HEADS * IDX_DIM + i])
                       : bf16f(tail[(size_t)(q % TAIL) * IDX_DIM + i]);
    }
    const float mean = rf((((k4[0] + k4[1]) + k4[2]) + k4[3]) / 4.0f);
    const float o = norm_rope(v, red, mean, small + IDX_KNORM_OFF, rope + (size_t)(b * BLK) * 2 * ROT_HALF, i);
    idx_keys[(size_t)b * IDX_DIM + i] = rne_bf16(o);
  }
  tail[(size_t)(p % TAIL) * IDX_DIM + i] = raw;
}

// ---------------------------------------------------------------------------------------------------------------
#define WG_SCORE 256
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
