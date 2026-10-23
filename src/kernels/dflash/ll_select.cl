// ll_select.cl - spec 26b: the LiLiCorr walk (spec 26 §2.4; vLLM dflash2/speculator.py's candidate walk, as dflash_ref.py
// lilicorr_walk ports it) over the lattice ll_lattice.cl scored - df_select.cl's walk with the edge term replaced by a
// lattice lookup (a family copy; df_select.cl is not edited). One work-group. Twin: tests/kernels/lilicorr_ref.h walk().
//
//   ll_select  grid (1), WG 64: previous = 0 (slot 0's lattice row is the start scores broadcast over previous, so index 0
//              IS the anchor); for s = 0 .. K - 1 (K = dctl.k, at most S): scores = lattice[s][previous][0..KC) - THE LATTICE
//              ALONE (the log-probs entered only as features) -, the pick: T = dctl.temperature 0 -> argmax (ties: the lower
//              index; NaN never wins), else Gumbel-argmax of scores / T - log(-log1p(-u)), u = murmur3(seed ^ (s << 32), p +
//              s + 2^30, candidate id) - vLLM 3901d56800's spec_step_seed keying (26a ruling 2: the seed's high word xor s),
//              the position the predicted one less one, the draft salt (fp32, df_select's arithmetic; vLLM's is fp32 too);
//              previous = the pick; ctrl.cur_token[1 + s] = cand[s][pick] - the verify list's draft rows.
//              The readback (`out`, runtime/dflash/dflash_control.h LiliDraftOut's words): k, the anchor, p, anchor_valid,
//              per walked slot its id, chosen index and realized scores, and for EVERY slot its candidates, their
//              log-probs and raw logits and its lattice rows (the host's q is softmax(scores / T) on the candidates).
#pragma OPENCL FP_CONTRACT OFF

#ifndef KC
#define KC 8
#endif
#ifndef S
#define S 7
#endif
#define WGS 64
#define SALT 1073741824u
// LiliDraftOut's words (runtime/dflash/dflash_control.h static_asserts the same numbers).
#define OUT_K 0
#define OUT_ANCHOR 1
#define OUT_POS 2
#define OUT_VALID 3
#define OUT_IDS 4
#define OUT_CHOSEN (OUT_IDS + S + 1)
#define OUT_CAND (OUT_CHOSEN + S + 1)
#define OUT_LP (OUT_CAND + S * KC)
#define OUT_LOGIT (OUT_LP + S * KC)
#define OUT_SCORES (OUT_LOGIT + S * KC)
#define OUT_LAT (OUT_SCORES + S * KC)

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

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void ll_select(__global uint* restrict ctrl, __global const uint* restrict dctl, __global const uint* restrict cand,
                        __global const float* restrict lp, __global const float* restrict logit,
                        __global const float* restrict lat, __global uint* restrict out) {
  const uint lid = get_local_id(0);
  const uint p = ctrl[CTRL_POS], anchor = ctrl[CTRL_CUR];
  uint K = dctl[DCTL_K];
  if (K > S) K = S;
  for (uint e = lid; e < S * KC; e += WGS) {
    out[OUT_CAND + e] = cand[e];
    out[OUT_LP + e] = as_uint(lp[e]);
    out[OUT_LOGIT + e] = as_uint(logit[e]);
  }
  for (uint e = lid; e < S * KC * KC; e += WGS) out[OUT_LAT + e] = as_uint(lat[e]);
  if (lid != 0) return;
  out[OUT_K] = K;
  out[OUT_ANCHOR] = anchor;
  out[OUT_POS] = p;
  out[OUT_VALID] = dctl[DCTL_AVALID];
  const float T = as_float(dctl[DCTL_TEMP]);
  const uint seed_lo = dctl[DCTL_SEED_LO], seed_hi = dctl[DCTL_SEED_HI];
  uint prev = 0;
  for (uint s = 0; s < K; ++s) {
    __global const float* row = lat + ((size_t)s * KC + prev) * KC;
    __global const uint* cs = cand + (size_t)s * KC;
    uint best = 0;
    float bv = -INFINITY;
    int any = 0;
    for (uint c = 0; c < KC; ++c) {
      float x = row[c];
      if (T != 0.0f) {
        const float u = murmur3_uniform(seed_lo, seed_hi ^ s, p + s + SALT, cs[c]);
        x = x / T - log(-log1p(-u));
      }
      if (!any ? !isnan(x) : x > bv) {
        bv = x;
        best = c;
        any = 1;
      }
    }
    out[OUT_IDS + s] = cs[best];
    out[OUT_CHOSEN + s] = best;
    for (uint c = 0; c < KC; ++c) out[OUT_SCORES + s * KC + c] = as_uint(row[c]);
    ctrl[CTRL_CUR + 1 + s] = cs[best];
    prev = best;
  }
}
