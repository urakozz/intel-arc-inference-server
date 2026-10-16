// df_select.cl - spec 19b: DFlash 2's candidate selector walk (spec 19 §2 "The walk"), or DFlash v1's per-row pick,
// over df_topk's candidates - one work-group, K sequential rows. Twin: tests/kernels/dflash_ref.h edge_scores() /
// choose(); the semantics are dflash_ref.py's score_edges / selector_walk (vLLM dflash2/speculator.py).
//
//   df_select  grid (1), WG TOPK·QS (256): for j = 0 .. K-1 (K = dctl.k):
//     prev  = j == 0 ? the anchor (ctrl.cur_token[0]) : the id chosen at row j - 1        (a real id, never an index)
//     V2:   S[c] = unary[j][c] + Σ_q (Σ_{r in slice q} (pred[prev][r] · hid[j + 1][r]) · succ[cand[j][c]][r]),
//           lane (c, q) = (lid / QS, lid % QS) sums its RANK / QS slice ascending, then lane c adds the QS slices
//           ascending; V1: S[c] = unary[j][c]
//     pick  T = dctl.temperature: T == 0 -> argmax (ties: the lower c; NaN never wins); else Gumbel-argmax of
//           S[c] / T - log(-log1p(-u)), u = murmur3(seed, p + j + 2^30, cand id) (vLLM gumbel.py's keying: the same
//           (seed, position, token) always draws the same uniform; the salt keeps draft noise off the target's)
//     write the readback (`out`, runtime/dflash/dflash_control.h's DFlashDraftOut words): ids[j], chosen[j], the
//           row's candidates, unary and realized scores S; and ctrl.cur_token[1 + j] = the id - the verify list's
//           draft rows (spec 8 / 19e: Engine::verify reads cur_token[1..k]).
//
// hid is the selector's hidden_projection over the block's rows, [M][RANK]: row j + 1 is mask row j's.
#pragma OPENCL FP_CONTRACT OFF

#ifndef TOPK
#define TOPK 16
#endif
#ifndef QS
#define QS 16
#endif
#ifndef MAXR
#define MAXR 8
#endif
#ifndef V2
#define V2 1
#endif
#define WG (TOPK * QS)
#define NO_IDX 0x7FFFFFFFu
#define SALT 1073741824u
// DFlashDraftOut's words (runtime/dflash/dflash_control.h static_asserts the same numbers).
#define OUT_K 0
#define OUT_ANCHOR 1
#define OUT_POS 2
#define OUT_IDS 4
#define OUT_CHOSEN (OUT_IDS + MAXR)
#define OUT_CAND (OUT_CHOSEN + MAXR)
#define OUT_UNARY (OUT_CAND + MAXR * TOPK)
#define OUT_SCORES (OUT_UNARY + MAXR * TOPK)
#if V2 && ((RANK % QS) != 0 || !defined(VROWS))
#error "V2 needs VROWS (the codebooks' rows) and RANK a multiple of QS"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

inline uint rotl32(uint v, uint s) { return (v << s) | (v >> (32u - s)); }
inline uint mm_mix(uint h, uint key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
inline float murmur3_uniform(uint seed_lo, uint seed_hi, uint pos, uint key) {
  uint h = 0;
  h = mm_mix(h, seed_lo);
  h = mm_mix(h, seed_hi);
  h = mm_mix(h, pos);
  h = mm_mix(h, key);
  h ^= 16u;
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return (float)(h >> 16) * 0x1p-16f + ((float)(h & 0xFFFFu) + 0.5f) * 0x1p-32f;
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void df_select(__global uint* restrict ctrl, __global const uint* restrict dctl,
                        __global const uint* restrict cand, __global const float* restrict unary,
                        __global const float* restrict hid, __global const ushort* restrict pred,
                        __global const ushort* restrict succ, __global uint* restrict out) {
  __local float part[TOPK * QS];
  __local float s[TOPK];
  __local uint prev_l;
  const uint lid = get_local_id(0);
  const uint p = ctrl[CTRL_POS], anchor = ctrl[CTRL_CUR];
  uint K = dctl[DCTL_K];
  if (K > MAXR - 1) K = MAXR - 1;
  const float T = as_float(dctl[DCTL_TEMP]);
  const uint seed_lo = dctl[DCTL_SEED_LO], seed_hi = dctl[DCTL_SEED_HI];
  if (lid == 0) {
    prev_l = anchor;
    out[OUT_K] = K;
    out[OUT_ANCHOR] = anchor;
    out[OUT_POS] = p;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint j = 0; j < K; ++j) {
    const uint prev = prev_l;
    __global const uint* cj = cand + (size_t)j * TOPK;
    __global const float* uj = unary + (size_t)j * TOPK;
#if V2
    {
      const uint c = lid / QS, q = lid % QS, per = RANK / QS;
      const uint id = cj[c];
      __global const float* hr = hid + (size_t)(j + 1) * RANK;
      float a = 0.0f;
      if (prev < VROWS)   // an engine bug (an id past the codebooks) scores 0, never reads past them
        for (uint r = q * per; r < (q + 1) * per; ++r) {
          const float P = bf16f(pred[(size_t)prev * RANK + r]) * hr[r];
          a = a + P * (id >= VROWS ? 0.0f : bf16f(succ[(size_t)id * RANK + r]));
        }
      part[c * QS + q] = a;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < TOPK) {
      float t = 0.0f;
      for (uint q = 0; q < QS; ++q) t = t + part[lid * QS + q];
      s[lid] = uj[lid] + t;
    }
#else
    if (lid < TOPK) s[lid] = uj[lid];
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
      uint best = 0;
      float bv = -INFINITY;
      int any = 0;
      for (uint c = 0; c < TOPK; ++c) {
        float x = s[c];
#if V2
        if (T != 0.0f) {
          const float u = murmur3_uniform(seed_lo, seed_hi, p + j + SALT, cj[c]);
          x = x / T - log(-log1p(-u));
        }
#endif
        if (!any ? !isnan(x) : x > bv) {
          bv = x;
          best = c;
          any = 1;
        }
      }
      const uint id = cj[best];
      out[OUT_IDS + j] = id;
      out[OUT_CHOSEN + j] = best;
      for (uint c = 0; c < TOPK; ++c) {
        out[OUT_CAND + j * TOPK + c] = cj[c];
        out[OUT_UNARY + j * TOPK + c] = as_uint(uj[c]);
        out[OUT_SCORES + j * TOPK + c] = as_uint(s[c]);
      }
      ctrl[CTRL_CUR + 1 + j] = id;
      prev_l = id;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
