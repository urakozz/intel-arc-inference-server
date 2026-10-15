#pragma once
// CPU references for spec 23c's Muse-Glimmer decode kernels (src/kernels/muse/mg_*.cl) - the same op chains in the
// same orders, every line with a twin in a kernel; edit the two together. The chains are transformers 5.19.0's
// muse_glimmer modules (spec 23 §1.1, modeling_muse_glimmer.py M:101-136, M:330-358, M:384-406, M:451-457);
// tests/kernels/muse_ref_test.cc holds them to those modules' own outputs (tests/kernels/muse_fixture.h,
// tools/oracle/muse_fixture.py) and to the semantics' properties (plan 23c Review Focus 1-5).
//
//   embed_norm()      mg_embed: the gathered bf16 row through the WEIGHT-LESS norm, `x * pow(ms + 1e-5, -0.5)`,
//                     one rounding (pow(x, -0.5) = rsqrt = 1 / sqrt on torch's CPU, 23a: the correctly rounded
//                     1 / sqrt); its output IS the residual, and the per-chunk Σ² layer 0's mg_norm reads
//   centered_norm()   mg_norm: `x * rsqrt(ms + eps) * (1 + w)` in fp32 (three products, no contraction), ONE
//                     rounding - not Kolibri's two (kol_norm_elem rounds x̂ first) and not plain w
//   post_add()        mg_post_add: the sub-block's OWN bf16 row normalised (centered, eps 1e-8), rounded, then
//                     `residual + that` rounded; the new residual's chunk Σ² for the next norm
//   final_norm()      mg_final_norm: plain w, `x * pow(ms + 1e-5, -0.5) * w`, one rounding
//   qk_prep()         mg_attn_prep: per head the weight-less norm (Σ² by a 128-lane tree), q̂ = bf16, q =
//                     bf16(q̂ x 3.87f) (two roundings); k̂ = bf16; RoPE (sliding only):
//                     bf16(bf16(y·cos) + bf16(rot(y)·sin)) with the table's bf16 cos / sin
//   gate_bf16() / gate_product()   bf16(sigmoid(g)) and bf16(attn x that) - M:358
//   silu_mul()        bf16(bf16(silu(g)) x u) - M:154
//   attention_eager() the reference's bf16 eager chain (k2_ref's score / softmax), keys [key_lo, pos] through
//                     ring_row; P·V in eager_block-key blocks (the device's); the gate after
//   attention_fp64()  the same in fp64 (flash's tolerance reference)
//   softcap_row()     spec 23 §6.4's host cap in fp32: 20 tanh(z x 0.19611613513818404 / 20)
//
// The Σ² trees: a row's chunk sums are prep.cl prep_res_fold's (G chunks of ceil(K / G), 256 lanes, fma, then
// 256 -> 1 pairwise); a norm adds the G sums ascending, / K, + eps, 1 / sqrt. At hidden 6656 G = 26 (chunks of 256).
// What is NOT bit-exact between these and the device: nothing in the exp-free chains; `exp` is exp_torch on both
// sides (k2_ref's Sleef expf u10); the GEMVs' multiply-adds are the GEMV's.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"   // exp_torch, eager_score, eager_softmax (torch's fp32 softmax), lowbias32

namespace museref {

constexpr uint32_t kHidden = 6656, kHd = 128, kQHeads = 32, kKvHeads = 2, kQN = kQHeads * kHd, kKvN = kKvHeads * kHd,
                   kQgkvN = 2 * kQN + 2 * kKvN, kInter = 19968, kVocab = 202048, kWindow = 2048, kRing = 4096,
                   kG = 26, kWg = 256, kAttnTgt = 64;
constexpr float kEps = 1e-5f, kPostEps = 1e-8f, kQkScale = 3.87f, kOutMult = 0.19611613513818404f, kSoftcap = 20.0f;
constexpr float kAttnScale = k2_ref::kAttnScale;   // float(128 ** -0.5)
// Column offsets inside the fused q || gate || k || v GEMV (model::MuseGlimmerDesc::qgkv_off_*).
constexpr uint32_t kOffGate = kQN, kOffK = 2 * kQN, kOffV = 2 * kQN + kKvN;

inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float bf(uint16_t h) { return common::bf16_to_f32(h); }
inline float rf(float f) { return bf(rne(f)); }
inline float exp_torch(float x) { return k2_ref::exp_torch(x); }
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

// --- the fixture's inputs ------------------------------------------------------------------------
// tools/oracle/muse_fixture.py val(): element i of input tensor t. coarse: (h % 17 - 8) x step; fine:
// ((h >> 8) x 2^-23 - 1) x amp, rounded to bf16.
inline float fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float step = 0.125f, float amp = 1.0f) {
  const uint32_t h = k2_ref::lowbias32(k2_ref::lowbias32(seed * 3 + t) ^ i);
  if (coarse) return float(int(h % 17) - 8) * step;
  return rf((float(h >> 8) * 0x1p-23f - 1.0f) * amp);
}
inline std::vector<uint16_t> hex16(const char* s) {
  std::vector<uint16_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 4 <= h.size(); i += 4) v.push_back(uint16_t(std::stoul(h.substr(i, 4), nullptr, 16)));
  return v;
}

// --- the Σ² and the norms --------------------------------------------------------------------------
inline uint32_t groups_of(uint32_t K) { return (K + kWg - 1) / kWg; }
// prep_res_fold's chunk trees over a bf16 row: sumsq[g] for g < G.
inline void chunk_sumsq(const uint16_t* x, uint32_t K, uint32_t G, float* sumsq) {
  const uint32_t chunk = (K + G - 1) / G;
  float red[kWg];
  for (uint32_t g = 0; g < G; ++g) {
    const uint32_t k0 = g * chunk, k1 = std::min(K, k0 + chunk);
    for (uint32_t i = 0; i < kWg; ++i) {
      float acc = 0.f;
      for (uint32_t k = k0 + i; k < k1; k += kWg) acc = std::fma(bf(x[k]), bf(x[k]), acc);
      red[i] = acc;
    }
    for (uint32_t stride = kWg / 2; stride > 0; stride >>= 1)
      for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
    sumsq[g] = red[0];
  }
}
// rstd from G chunk sums, ascending: 1 / sqrt(Σ / K + eps) (never rsqrt; = torch's rsqrt / pow(-0.5) on its CPU).
inline float rstd_of(const float* sumsq, uint32_t G, uint32_t K, float eps) {
  float total = 0.f;
  for (uint32_t g = 0; g < G; ++g) total += sumsq[g];
  return 1.0f / std::sqrt(total / float(K) + eps);
}
inline float rstd_row(const uint16_t* x, uint32_t K, float eps) {
  std::vector<float> s(groups_of(K));
  chunk_sumsq(x, K, groups_of(K), s.data());
  return rstd_of(s.data(), groups_of(K), K, eps);
}
// One element of the centered norm: (x · rstd) · (1 + w), each product fp32, one rounding.
inline uint16_t centered_elem(float x, float rstd, float w) { return rne((x * rstd) * (1.0f + w)); }

inline void embed_norm(const uint16_t* row, uint32_t K, uint16_t* out) {
  const float r = rstd_row(row, K, kEps);
  for (uint32_t k = 0; k < K; ++k) out[k] = rne(bf(row[k]) * r);
}
// mg_embed's whole output: the residual row and its chunk sums (G = groups_of(K)).
inline void embed_norm_sums(const uint16_t* row, uint32_t K, uint16_t* resid, float* sumsq) {
  embed_norm(row, K, resid);
  chunk_sumsq(resid, K, groups_of(K), sumsq);
}
// mg_norm over the chunk sums the previous launch wrote.
inline void centered_norm_from(const float* sumsq, uint32_t G, const uint16_t* x, const float* w, float eps, uint32_t K,
                               uint16_t* out) {
  const float r = rstd_of(sumsq, G, K, eps);
  for (uint32_t k = 0; k < K; ++k) out[k] = centered_elem(bf(x[k]), r, w[k]);
}
inline void centered_norm(const uint16_t* x, const float* w, float eps, uint32_t K, uint16_t* out) {
  const float r = rstd_row(x, K, eps);
  for (uint32_t k = 0; k < K; ++k) out[k] = centered_elem(bf(x[k]), r, w[k]);
}
// Review Focus 1's wrong forms: Kolibri's (x̂ rounded to bf16 first) and plain w.
inline void centered_norm_kolibri_form(const uint16_t* x, const float* w, float eps, uint32_t K, uint16_t* out) {
  const float r = rstd_row(x, K, eps);
  for (uint32_t k = 0; k < K; ++k) out[k] = rne(rf(bf(x[k]) * r) * (1.0f + w[k]));
}
inline void final_norm_eps(const uint16_t* x, const float* w, float eps, uint32_t K, uint16_t* out) {
  const float r = rstd_row(x, K, eps);
  for (uint32_t k = 0; k < K; ++k) out[k] = rne((bf(x[k]) * r) * w[k]);
}
inline void final_norm(const uint16_t* x, const float* w, uint32_t K, uint16_t* out) { final_norm_eps(x, w, kEps, K, out); }
inline void final_norm_from(const float* sumsq, uint32_t G, const uint16_t* x, const float* w, uint32_t K, uint16_t* out) {
  const float r = rstd_of(sumsq, G, K, kEps);
  for (uint32_t k = 0; k < K; ++k) out[k] = rne((bf(x[k]) * r) * w[k]);
}

// The sandwich: resid = rne(resid + rne(centered(sub, eps 1e-8))). The device form takes the sub-block's chunk
// sums (prep_res_fold ..._Z wrote them) and writes the new residual's.
inline void post_add_sums(const float* sumsq_sub, uint32_t G, const uint16_t* sub, const float* w, uint16_t* resid,
                          float* sumsq_out, uint32_t K) {
  const float r = rstd_of(sumsq_sub, G, K, kPostEps);
  for (uint32_t k = 0; k < K; ++k) resid[k] = rne(bf(resid[k]) + bf(centered_elem(bf(sub[k]), r, w[k])));
  if (sumsq_out) chunk_sumsq(resid, K, G, sumsq_out);
}
inline void post_add(uint16_t* resid, const uint16_t* sub, const float* w_post, uint32_t K) {
  std::vector<float> s(groups_of(K));
  chunk_sumsq(sub, K, groups_of(K), s.data());
  post_add_sums(s.data(), groups_of(K), sub, w_post, resid, nullptr, K);
}

// --- the attention prep ---------------------------------------------------------------------------
// One head's weight-less norm: Σ² by a 128-lane tree (lane d: fma(x, x, 0), then 64, 32, ... pairwise).
inline float head_rstd(const float* xb) {
  float red[kHd];
  for (uint32_t d = 0; d < kHd; ++d) red[d] = std::fma(xb[d], xb[d], 0.0f);
  for (uint32_t stride = kHd / 2; stride > 0; stride >>= 1)
    for (uint32_t d = 0; d < stride; ++d) red[d] += red[d + stride];
  return 1.0f / std::sqrt(red[0] / float(kHd) + kEps);
}
// RoPE on one head of bf16 values y with the table's row at `pos` ([2][64]: cos then sin, bf16 values).
inline void rope_head(const float* y, const float* cs, uint16_t* out) {
  for (uint32_t d = 0; d < kHd; ++d) {
    const uint32_t ii = d % (kHd / 2);
    const float c = cs[ii], s = cs[kHd / 2 + ii];
    const float rot = d < kHd / 2 ? -y[d + kHd / 2] : y[d - kHd / 2];
    out[d] = rne(rf(y[d] * c) + rf(rot * s));
  }
}
// q [32][128] and k [2][128]: the GEMV's bf16 outputs. `tab`: the RoPE table [max][2][64] (read when rope).
inline void qk_prep(const uint16_t* q, const uint16_t* k, uint32_t pos, bool rope, const float* tab, uint16_t* q_out,
                    uint16_t* k_out) {
  float xb[kHd], y[kHd];
  const float* cs = rope ? tab + size_t(pos) * kHd : nullptr;
  for (uint32_t h = 0; h < kQHeads + kKvHeads; ++h) {
    const bool isq = h < kQHeads;
    const uint16_t* src = isq ? q + size_t(h) * kHd : k + size_t(h - kQHeads) * kHd;
    uint16_t* dst = isq ? q_out + size_t(h) * kHd : k_out + size_t(h - kQHeads) * kHd;
    for (uint32_t d = 0; d < kHd; ++d) xb[d] = bf(src[d]);
    const float r = head_rstd(xb);
    for (uint32_t d = 0; d < kHd; ++d) {
      const float hat = rf(xb[d] * r);
      y[d] = isq ? rf(hat * kQkScale) : hat;
    }
    if (cs) {
      rope_head(y, cs, dst);
    } else {
      for (uint32_t d = 0; d < kHd; ++d) dst[d] = rne(y[d]);
    }
  }
}
// Review Focus 2's wrong form: q x 3.87 folded into the norm's one rounding.
inline void q_prep_one_rounding(const uint16_t* q, uint16_t* out) {
  float xb[kHd];
  for (uint32_t h = 0; h < kQHeads; ++h) {
    for (uint32_t d = 0; d < kHd; ++d) xb[d] = bf(q[size_t(h) * kHd + d]);
    const float r = head_rstd(xb);
    for (uint32_t d = 0; d < kHd; ++d) out[size_t(h) * kHd + d] = rne(xb[d] * r * kQkScale);
  }
}
// The fused GEMV's bf16 output column `col` of row m: Σ_{s ascending} partials[s][m][col], rounded once.
inline uint16_t partials_col(const float* partials, uint32_t S, uint32_t M, uint32_t m, uint32_t N, size_t col) {
  float v = 0.f;
  for (uint32_t s = 0; s < S; ++s) v += partials[(size_t(s) * M + m) * N + col];
  return rne(v);
}

// --- the gate and the MLP's activation ---------------------------------------------------------------
inline uint16_t gate_bf16(uint16_t g) { return rne(sigmoid_t(bf(g))); }
inline uint16_t gate_product(uint16_t a, uint16_t g_bf16) { return rne(bf(a) * bf(g_bf16)); }
inline void silu_mul(const uint16_t* g, const uint16_t* u, uint32_t I, uint16_t* out) {
  for (uint32_t i = 0; i < I; ++i) out[i] = rne(rf(silu_t(bf(g[i]))) * bf(u[i]));
}

// --- the window and the ring --------------------------------------------------------------------------
// The first visible key of the query at `pos`: pos - 2047 (2048 keys including the query: i - 2048 < j <= i,
// MU:93-102) in a sliding layer, 0 in a full one.
inline uint32_t key_lo(uint32_t pos, bool sliding) { return sliding && pos + 1 > kWindow ? pos + 1 - kWindow : 0; }
inline uint32_t ring_row(uint32_t p, bool sliding) { return sliding ? (p & (kRing - 1)) : p; }
// v2's ppw over the step's key span (the eager kernels' P·V block).
inline uint32_t eager_block(uint32_t span, uint32_t tgt = kAttnTgt) {
  uint32_t p = (span + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}

// --- eager decode attention -----------------------------------------------------------------------------
// q [32][128] (bf16), caches [rows][2][128] read at row p & (ring - 1) when sliding and ring != 0, else row p.
struct EagerHead {
  std::vector<float> s, p, o;   // scores and probabilities (bf16 values) over keys [lo, pos], o's fp32 sums
};
inline size_t cache_row(uint32_t p, bool sliding, uint32_t ring) { return sliding && ring ? (p & (ring - 1)) : p; }
inline EagerHead attention_eager_head(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos,
                                      bool sliding, uint32_t h, uint32_t ring = kRing) {
  const uint32_t lo = key_lo(pos, sliding), L = pos - lo + 1, j = h / (kQHeads / kKvHeads);
  const uint32_t blk = eager_block(L);
  float qf[kHd];
  for (uint32_t d = 0; d < kHd; ++d) qf[d] = bf(q[size_t(h) * kHd + d]);
  EagerHead r;
  r.s.resize(L);
  r.p.resize(L);
  r.o.assign(kHd, 0.0f);
  for (uint32_t i = 0; i < L; ++i)
    r.s[i] = k2_ref::eager_score(qf, kc + (cache_row(lo + i, sliding, ring) * kKvHeads + j) * kHd, kHd);
  k2_ref::eager_softmax(r.s.data(), L, r.p.data());
  for (uint32_t d = 0; d < kHd; ++d) {
    float o = 0.0f;
    for (uint32_t b0 = 0; b0 < L; b0 += blk) {
      float acc = 0.0f;
      for (uint32_t i = b0; i < std::min(L, b0 + blk); ++i)
        acc = std::fma(r.p[i], bf(vc[(cache_row(lo + i, sliding, ring) * kKvHeads + j) * kHd + d]), acc);
      o += acc;
    }
    r.o[d] = o;
  }
  return r;
}
// Every head: out [32][128] = rne(o), then (gate non-null: bf16(sigmoid(g)) [32][128]) rne(rne(o) x gate).
inline void attention_eager(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, bool sliding,
                            uint32_t ring, const uint16_t* gate, uint16_t* out) {
  for (uint32_t h = 0; h < kQHeads; ++h) {
    const EagerHead e = attention_eager_head(q, kc, vc, pos, sliding, h, ring);
    for (uint32_t d = 0; d < kHd; ++d) {
      const uint16_t o = rne(e.o[d]);
      out[size_t(h) * kHd + d] = gate ? gate_product(o, gate[size_t(h) * kHd + d]) : o;
    }
  }
}
inline void attention_fp64(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, bool sliding,
                           uint32_t ring, const uint16_t* gate, double* out) {
  const uint32_t lo = key_lo(pos, sliding), L = pos - lo + 1;
  std::vector<double> sc(L);
  for (uint32_t h = 0; h < kQHeads; ++h) {
    const uint32_t j = h / (kQHeads / kKvHeads);
    double mx = -INFINITY;
    for (uint32_t i = 0; i < L; ++i) {
      const size_t rw = (cache_row(lo + i, sliding, ring) * kKvHeads + j) * kHd;
      double a = 0;
      for (uint32_t d = 0; d < kHd; ++d) a += double(bf(q[size_t(h) * kHd + d])) * bf(kc[rw + d]);
      sc[i] = a / std::sqrt(double(kHd));
      mx = std::max(mx, sc[i]);
    }
    double z = 0;
    for (double& s : sc) z += (s = std::exp(s - mx));
    for (uint32_t d = 0; d < kHd; ++d) {
      double o = 0;
      for (uint32_t i = 0; i < L; ++i) o += sc[i] / z * bf(vc[(cache_row(lo + i, sliding, ring) * kKvHeads + j) * kHd + d]);
      out[size_t(h) * kHd + d] = gate ? o * bf(gate[size_t(h) * kHd + d]) : o;
    }
  }
}

// --- flash decode attention's arithmetic on the host (mg_attn.cl, M = 1) ------------------------------------
// mg_attn_decode's blocks and waves and mg_attn_reduce's merge, step for step but for the score dot's order (one
// fma chain here; the card's is 8 fmas a lane and a 16-lane shuffle tree) and `exp` (std::exp here): a tolerance
// twin (fp64 at cosine 0.99999) that runs the block / window / ring arithmetic of Review Focus 3 on any host.
//   SLIDING: base = key_lo(pos) & ~63, absolute 64-key blocks b = 0.. while base + 64 b <= pos; FULL: base 0,
//   v2's ppw(pos + 1) keys a block. Per block: waves of 16 from its start; a key outside [lo, pos] scores -INF
//   (weight 0); online softmax (nmx, resc, wl; sums ascending); the reduce merges blocks with bmx > -INF.
inline uint32_t flash_ppw(uint32_t len, uint32_t tgt = kAttnTgt) { return eager_block(len, tgt); }
inline void attention_flash_sim(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, bool sliding,
                                uint32_t ring, const uint16_t* gate, uint16_t* out) {
  const uint32_t lo = key_lo(pos, sliding), end = pos + 1;
  const uint32_t base = sliding ? (lo & ~63u) : 0u, s = sliding ? 64u : flash_ppw(pos + 1);
  const uint32_t nb = sliding ? (pos - base) / 64u + 1u : pos / s + 1u;
  for (uint32_t h = 0; h < kQHeads; ++h) {
    const uint32_t j = h / (kQHeads / kKvHeads);
    std::vector<float> bmx(nb), bsm(nb), bacc(size_t(nb) * kHd);
    for (uint32_t b = 0; b < nb; ++b) {
      const uint32_t bstart = base + b * s, bend = std::min(bstart + s, end);
      float mx = -INFINITY, sm = 0.0f;
      std::vector<float> acc(kHd, 0.0f);
      for (uint32_t p0 = bstart; p0 < bend; p0 += 16) {
        float sc[16];
        for (uint32_t l = 0; l < 16; ++l) {
          const uint32_t p = p0 + l;
          if (p > pos || p < lo) {
            sc[l] = -INFINITY;
            continue;
          }
          float a = 0.0f;
          const size_t rw = (cache_row(p, sliding, ring) * kKvHeads + j) * kHd;
          for (uint32_t d = 0; d < kHd; ++d) a = std::fma(bf(q[size_t(h) * kHd + d]), bf(kc[rw + d]), a);
          sc[l] = a * kAttnScale;
        }
        float wmx = -INFINITY;
        for (float x : sc) wmx = std::max(wmx, x);
        const float nmx = std::max(mx, wmx);
        float resc = 1.0f, wl[16];
        if (nmx > -INFINITY) {
          resc = std::exp(mx - nmx);
          for (uint32_t l = 0; l < 16; ++l) wl[l] = std::exp(sc[l] - nmx);
        } else {
          for (float& w : wl) w = 0.0f;
        }
        float ssum = 0.0f;
        for (uint32_t l = 0; l < 16; ++l) ssum += wl[l];
        for (uint32_t d = 0; d < kHd; ++d) {
          float tsum = 0.0f;
          for (uint32_t l = 0; l < 16; ++l) {
            const uint32_t p = p0 + l;
            const float v = (p <= pos && p >= lo) ? bf(vc[(cache_row(p, sliding, ring) * kKvHeads + j) * kHd + d]) : 0.0f;
            tsum = std::fma(wl[l], v, tsum);
          }
          acc[d] = std::fma(acc[d], resc, tsum);
        }
        if (nmx > -INFINITY) {
          sm = std::fma(sm, resc, ssum);
          mx = nmx;
        }
      }
      bmx[b] = mx;
      bsm[b] = sm;
      std::copy(acc.begin(), acc.end(), bacc.begin() + size_t(b) * kHd);
    }
    for (uint32_t d = 0; d < kHd; ++d) {
      float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
      for (uint32_t b = 0; b < nb; ++b) {
        if (!(bmx[b] > -INFINITY)) continue;
        const float nmx = std::max(mx, bmx[b]);
        const float a = std::exp(mx - nmx), bs = std::exp(bmx[b] - nmx);
        sm = std::fma(sm, a, bsm[b] * bs);
        acc = std::fma(acc, a, bacc[size_t(b) * kHd + d] * bs);
        mx = nmx;
      }
      const uint16_t o = rne(acc / sm);
      out[size_t(h) * kHd + d] = gate ? gate_product(o, gate[size_t(h) * kHd + d]) : o;
    }
  }
}

// --- the head --------------------------------------------------------------------------------------------
inline void softcap_row(const float* z, uint32_t V, float* out) {
  for (uint32_t i = 0; i < V; ++i) out[i] = kSoftcap * std::tanh((z[i] * kOutMult) / kSoftcap);
}

}  // namespace museref
