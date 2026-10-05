#pragma once
// CPU references for spec 18b's K2-Horizon kernels (src/kernels/k2/*.cl) - the same op chains
// in the same orders, every line with a twin in the kernel; edit the two together.
//
//   norm_finish()   k2_norm_finish: per-GROUP rstd from prep_res_fold's chunk sums (the
//                   chunks of a group summed ascending), x = rne(r · rstd_g · w), plain w
//   silu_mul()      k2_silu_mul at K2's intermediate and split-K
//   softplus()      F.softplus(beta = ln 2, threshold = 20), torch's CPU expression
//   route()         k2_route: Σ slices -> rne -> sigmoid; sel = s + bias; rank with ties to
//                   the lower id over the REAL experts only; Σ s in rank order; slots in
//                   ascending id; w = rne((s / Σ) x scale)
//   combine()       k2_moe_down's epilogue: the ascending-id bf16 chain, the ungated shared
//                   expert, the residual add
//   moe_down()      the whole down + combine over layout-1 blocks
//   mova_value()    k2_mova_value: GEMV, SiLU, the ascending-id bf16 chain
//   attn_prep()     k2_attn_prep: the bf16 RoPE chain (rotate_half over all dims), gate, v
//   attention()     decode attention in fp64 (a tolerance reference, not the kernel's order)
//   attn_gate()     k2_attn_reduce's last line: rne(f32(rne(o)) · f32(rne(softplus(g))))
//   attention_eager()  B70_K2_ATTN=eager (k2_attn_eager.cl): the reference's bf16 eager chain,
//                   the kernel's orders, torch's softmax bit for bit (the section has the account)
//
// What is NOT bit-exact between these and the device: `exp` / `log1p` (OpenCL's ulp
// bounds: the sigmoid, SiLU, softplus, the softmax) and the GEMVs' multiply-adds (OpenCL may
// contract them). The orders and the rounding points are exact here; tests/kernels/
// k2_ref_test.cc checks them on the host against independent double-precision formulas
// and against the reference semantics of spec 18 §3.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "kernels/moe_ref.h"   // f32 / rne / rf, silu, sigmoid, the layout-1 tile_dot / split_dot

namespace k2_ref {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;
using moe_ref::sigmoid_f32;
using moe_ref::silu_f32;
using moe_ref::split_dot;

// The route row (k2_moe.cl R_*, kernels::k2::route).
constexpr uint32_t kWords = 32, kIds = 0, kWeights = 8, kSel = 16, kNext = 24, kSum = 25;

// --- the grouped norm ------------------------------------------------------------------
// `sumsq` [G][M] from prep_ref::res_fold (stage A, unchanged); `groups` norm groups of K /
// groups, each a whole number of the G chunks.
inline void norm_finish(const float* sumsq, const uint16_t* resid, const float* w, uint16_t* x,
                        uint32_t M, uint32_t K, uint32_t G, uint32_t groups) {
  const uint32_t gs = K / groups, cpg = G / groups;
  for (uint32_t m = 0; m < M; ++m) {
    std::vector<float> rstd(groups);
    for (uint32_t gi = 0; gi < groups; ++gi) {
      float total = 0.f;
      for (uint32_t g = gi * cpg; g < (gi + 1) * cpg; ++g) total += sumsq[size_t(g) * M + m];
      rstd[gi] = 1.0f / std::sqrt(total / float(gs) + 1e-6f);
    }
    for (uint32_t k = 0; k < K; ++k)
      x[size_t(m) * K + k] = rne(f32(resid[size_t(m) * K + k]) * rstd[k / gs] * w[k]);
  }
}

// --- SiLU x up over gate||up (interleave16) --------------------------------------------
inline void silu_mul(const float* partials, uint16_t* x, uint32_t M, uint32_t I, uint32_t S) {
  const size_t FN = size_t(2) * I;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < I; ++k) {
      const size_t gflat = size_t(k / 16) * 32 + (k % 16), uflat = gflat + 16;
      float ga = 0.f, ua = 0.f;
      for (uint32_t s = 0; s < S; ++s) {
        const size_t base = (size_t(s) * M + m) * FN;
        ga += partials[base + gflat];
        ua += partials[base + uflat];
      }
      const uint16_t g_b = rne(ga), u_b = rne(ua);
      x[size_t(m) * I + k] = rne(f32(rne(silu_f32(f32(g_b)))) * f32(u_b));
    }
}

// --- softplus ------------------------------------------------------------------------------
constexpr float kSpBeta = 0.6931471805599453f, kSpThreshold = 20.0f;
inline float softplus(float x) {
  const float bx = x * kSpBeta;
  return bx > kSpThreshold ? x : std::log1p(std::exp(bx)) / kSpBeta;
}

// --- the sigmoid router ----------------------------------------------------------------------
struct Route {
  uint32_t ids[8] = {};
  float w[8] = {};     // bf16 values, slot order (ascending id)
  float sel[8] = {};   // s + bias, slot order
  float next = -INFINITY;   // sel of rank top_k
  float sum = 0;       // Σ s of the top-k, rank order
  uint32_t rank_ids[9] = {};   // the ranked ids (diagnostics), rank order
};

// Token m's route from `logits` read as `ls` slices of [ls][M][ln] from column `loff`.
inline Route route(const float* logits, uint32_t M, uint32_t m, uint32_t ln, uint32_t loff,
                   uint32_t ls, const float* bias, uint32_t E, uint32_t K, float scale) {
  std::vector<float> s(E), sel(E);
  for (uint32_t e = 0; e < E; ++e) {
    float acc = 0.0f;
    for (uint32_t q = 0; q < ls; ++q) acc += logits[(size_t(q) * M + m) * ln + loff + e];
    s[e] = sigmoid_f32(rf(acc));
    sel[e] = s[e] + bias[e];
  }
  std::vector<uint32_t> top_id(K + 1);
  std::vector<float> top_s(K + 1, 0.0f), top_sel(K + 1, -INFINITY);
  for (uint32_t k = 0; k <= K; ++k) top_id[k] = k;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < E; ++j) rank += (sel[j] > sel[e] || (sel[j] == sel[e] && j < e)) ? 1u : 0u;
    if (rank <= K) {
      top_id[rank] = e;
      top_s[rank] = s[e];
      top_sel[rank] = sel[e];
    }
  }
  Route r;
  for (uint32_t k = 0; k < K; ++k) r.sum += top_s[k];
  for (uint32_t k = 0; k <= K; ++k) r.rank_ids[k] = top_id[k];
  for (uint32_t j = 0; j < K; ++j)
    for (uint32_t k = 0; k < K; ++k) {
      uint32_t smaller = 0;
      for (uint32_t k2 = 0; k2 < K; ++k2) smaller += top_id[k2] < top_id[k] ? 1u : 0u;
      if (smaller != j) continue;
      const float t = top_s[k] / r.sum;
      const float w = t * scale;
      r.ids[j] = top_id[k];
      r.w[j] = rf(w);
      r.sel[j] = top_sel[k];
    }
  r.next = top_sel[K];
  return r;
}

// --- the MoE combine -------------------------------------------------------------------------
// `down` [K + 1]: the fp32 down sums in slot order, the shared expert last.
inline uint16_t combine(const float* down, const Route& r, uint32_t K, uint16_t resid) {
  float acc = 0.0f;
  for (uint32_t j = 0; j < K; ++j) {
    const uint16_t d_b = rne(down[j]);
    const uint16_t t_b = rne(f32(d_b) * r.w[j]);
    acc = rf(acc + f32(t_b));
  }
  const uint16_t sh_b = rne(down[K]);
  const uint16_t o_b = rne(acc + f32(sh_b));
  return rne(f32(resid) + f32(o_b));
}

// moe_ref's Shape / Route from K2's (the gate||up kernel is moe.cl's, ids from the row).
inline moe_ref::Shape moe_shape(uint32_t E, uint32_t K, uint32_t hidden, uint32_t inter) {
  return {E, K, hidden, inter, 0};
}
inline moe_ref::Route as_moe(const Route& r, uint32_t K) {
  moe_ref::Route m;
  for (uint32_t j = 0; j < K; ++j) m.ids[j] = r.ids[j];
  return m;
}

// The residual row after k2_moe_down.
inline std::vector<uint16_t> moe_down(const Route& r, const std::vector<uint16_t>& h,
                                      const uint32_t* dn, const moe_ref::Shape& s,
                                      const uint16_t* resid, uint32_t dn_ks) {
  std::vector<uint16_t> out(s.hidden);
  std::vector<float> d(s.slots());
  for (uint32_t n = 0; n < s.hidden; ++n) {
    for (uint32_t slot = 0; slot < s.slots(); ++slot) {
      const uint32_t e = slot < s.top_k ? r.ids[slot] : s.shared_block();
      d[slot] = split_dot(dn + e * s.down_words(), s.inter, n, h.data() + size_t(slot) * s.inter, dn_ks);
    }
    out[n] = combine(d.data(), r, s.top_k, resid[n]);
  }
  return out;
}

// --- MoVA ----------------------------------------------------------------------------------------
// v [N] bf16 from the K selected value experts (layout-1 blocks of D x N).
inline std::vector<uint16_t> mova_value(const Route& r, const uint16_t* x, const uint32_t* w,
                                        uint32_t D, uint32_t N, uint32_t K, uint32_t ks) {
  const size_t blk = size_t(N / 16) * (D / 64) * 136;
  std::vector<uint16_t> v(N);
  for (uint32_t n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < K; ++j) {
      const uint16_t v_b = rne(split_dot(w + r.ids[j] * blk, D, n, x, ks));
      const uint16_t a_b = rne(silu_f32(f32(v_b)));
      const uint16_t t_b = rne(f32(a_b) * r.w[j]);
      acc = rf(acc + f32(t_b));
    }
    v[n] = rne(acc);
  }
  return v;
}

// --- the attention prep ----------------------------------------------------------------------
struct Prep {
  std::vector<float> q, gate;      // [q_heads][hd] (bf16 values)
  std::vector<uint16_t> k, v;      // [kv_heads][hd]; v empty unless `v_from`
};
// Token m of a [S][M][n] fp32 fused row; `cs` = the RoPE table's row at pos + m.
inline Prep attn_prep(const float* partials, uint32_t M, uint32_t m, uint32_t n, uint32_t S,
                      const float* cs, uint32_t qh, uint32_t kvh, uint32_t hd, bool v_from) {
  const uint32_t k_off = qh * hd, gate_off = k_off + kvh * hd, v_off = gate_off + qh * hd;
  const uint32_t half = hd / 2;
  auto sum = [&](size_t col) {
    float v = 0.f;
    for (uint32_t s = 0; s < S; ++s) v += partials[(size_t(s) * M + m) * n + col];
    return v;
  };
  auto rope_head = [&](size_t base, uint16_t* out) {
    std::vector<float> xs(hd);
    for (uint32_t i = 0; i < hd; ++i) xs[i] = rf(sum(base + i));
    for (uint32_t i = 0; i < hd; ++i) {
      const float c = cs[i % half], sn = cs[half + i % half];
      const float rot = i < half ? -xs[i + half] : xs[i - half];
      out[i] = rne(f32(rne(xs[i] * c)) + f32(rne(rot * sn)));
    }
  };
  Prep p;
  p.q.resize(size_t(qh) * hd);
  p.gate.resize(size_t(qh) * hd);
  p.k.resize(size_t(kvh) * hd);
  std::vector<uint16_t> tmp(hd);
  for (uint32_t h = 0; h < qh; ++h) {
    rope_head(size_t(h) * hd, tmp.data());
    for (uint32_t i = 0; i < hd; ++i) {
      p.q[size_t(h) * hd + i] = f32(tmp[i]);
      p.gate[size_t(h) * hd + i] = rf(sum(gate_off + size_t(h) * hd + i));
    }
  }
  for (uint32_t j = 0; j < kvh; ++j) rope_head(k_off + size_t(j) * hd, p.k.data() + size_t(j) * hd);
  if (v_from) {
    p.v.resize(size_t(kvh) * hd);
    for (uint32_t i = 0; i < kvh * hd; ++i) p.v[i] = rne(sum(v_off + i));
  }
  return p;
}

// --- decode attention (fp64, a tolerance reference) -----------------------------------------
// One q head (`q` [hd] fp32) over keys 0..len-1 of kv head j in a [max][kvh][hd] cache.
inline std::vector<double> attention(const float* q, const uint16_t* kv_k, const uint16_t* kv_v,
                                     uint32_t len, uint32_t j, uint32_t kvh, uint32_t hd) {
  std::vector<double> sc(len);
  double mx = -INFINITY;
  for (uint32_t p = 0; p < len; ++p) {
    double a = 0;
    for (uint32_t d = 0; d < hd; ++d) a += double(q[d]) * f32(kv_k[(size_t(p) * kvh + j) * hd + d]);
    sc[p] = a / std::sqrt(double(hd));
    mx = std::max(mx, sc[p]);
  }
  double z = 0;
  for (double& s : sc) z += (s = std::exp(s - mx));
  std::vector<double> o(hd, 0.0);
  for (uint32_t p = 0; p < len; ++p)
    for (uint32_t d = 0; d < hd; ++d) o[d] += sc[p] / z * f32(kv_v[(size_t(p) * kvh + j) * hd + d]);
  return o;
}
inline uint16_t attn_gate(float o, float gate) {
  return rne(f32(rne(o)) * f32(rne(softplus(gate))));
}

// --- eager decode attention (B70_K2_ATTN=eager: src/kernels/k2/k2_attn_eager.cl) ------------
// transformers' bf16 eager path as tools/oracle/k2_ref.py attention() runs it, one q head over
// keys 0..len-1, every rounding point at the reference's place:
//
//   s_p = rne(f32(rne(Σ_d q_d·k_pd)) · scale)    the bf16 GEMM's one rounding, then `* scaling`
//                                               on the bf16 tensor (fp32 inside, rounded)
//   p_p = rne(e_p · (1 / Σ e))  e_p = exp(s_p - max)   torch's fp32 softmax on the CPU
//         (SoftMaxKernel.cpp _vec_softmax_lastdim at AVX2: Sleef_expf8_u10, the 8-lane sum,
//         one reciprocal), then .to(bf16)
//   o_d = Σ_p p_p·v_pd                           the bf16 GEMM: fp32 sum, rounded by attn_gate()
//
// What is bitwise against torch, and why (tests/kernels/k2_attn_eager_ref_test.cc measures each
// against tools/oracle/k2_attn_eager_fixture.py's dump):
//   * the softmax: exact. exp_torch() is Sleef's expf u10 step for step (fma as the AVX2 build
//     has it), the sum is reduce_all's order (8 lane accumulators, then the permute tree), the
//     reciprocal and the multiply are torch's. Probed on 22,760 probabilities: 0 fp32
//     differences (std::exp: 2,101; a sequential sum: 20,069).
//   * the score's rounding points: exact. The dot's ACCUMULATION ORDER is ours (one fma chain,
//     d ascending; every product of two bf16 values is exact in fp32, so fma = mul + add and
//     the chain is contraction-proof); torch's GEMM order is its own and differs between the
//     prompt pass and a decode step. The two round apart only where the fp32 dot sits within
//     its accumulation error of a bf16 rounding boundary (~2^-16 x a few per score).
//   * P·V: same - our order (block sums ascending, each an fma chain), one rounding.
// The device kernel computes exactly these functions in exactly these orders (exp_torch is
// in the kernel too), so kernel against this reference is bitwise.
constexpr float kAttnScale = 0.08838834764831845f;   // float(128 ** -0.5): torch's scalar cast

// Sleef_expf8_u10 (sleefsimdsp.c xexpf at AVX2 + FMA): torch's Vectorized<float>::exp, which is
// what the fp32 softmax evaluates (torch.exp itself goes to MKL's vsExp and is NOT this).
inline float exp_torch(float d) {
  const float kRLn2 = 1.442695040888963407359924681001892137426645954152985934135449406931f;
  const float kL2U = 0.693145751953125f, kL2L = 1.428606765330187045e-06f;
  const int q = int(std::nearbyint(d * kRLn2));   // cvtps2dq: nearest-even
  const float qf = float(q);
  float s = std::fma(qf, -kL2U, d);
  s = std::fma(qf, -kL2L, s);
  float u = 0.000198527617612853646278381f;
  u = std::fma(u, s, 0.00139304355252534151077271f);
  u = std::fma(u, s, 0.00833336077630519866943359f);
  u = std::fma(u, s, 0.0416664853692054748535156f);
  u = std::fma(u, s, 0.166666671633720397949219f);
  u = std::fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + std::fma(ss, u, s);
  const auto pow2i = [](int e) {   // vpow2i_vf_vi2: exponent field only (e in -75 .. 75 here)
    const uint32_t b = uint32_t(e + 127) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  if (d < -104.0f) return 0.0f;   // before ldexp: q would leave pow2i's range
  if (100.0f < d) return INFINITY;
  u = u * pow2i(q >> 1);
  return u * pow2i(q - (q >> 1));
}

inline float eager_score(const float* q, const uint16_t* k, uint32_t hd) {
  float a = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) a = std::fma(q[d], f32(k[d]), a);
  return rf(rf(a) * kAttnScale);
}

// p[0..len) from s[0..len) (bf16 values). The sum is reduce_all's at a row of >= 8 entries;
// a row shorter than 8 is summed as the reference's PROMPT pass sums it (padded with exact
// zeros to the prompt's length, >= 8 for every golden prompt) - a decode-only row of < 8 keys
// would be sequential in torch, which the engine never meets after a prompt of >= 8 tokens.
inline void eager_softmax(const float* s, uint32_t len, float* p) {
  float mx = -INFINITY;
  for (uint32_t i = 0; i < len; ++i) mx = std::max(mx, s[i]);
  float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < len; ++i) {
    p[i] = exp_torch(s[i] - mx);
    lane[i % 8] += p[i];   // lane i % 8's chain, ascending i (0 + e = e exactly)
  }
  const float sum = ((lane[0] + lane[4]) + (lane[2] + lane[6])) + ((lane[1] + lane[5]) + (lane[3] + lane[7]));
  const float inv = 1.0f / sum;
  for (uint32_t i = 0; i < len; ++i) p[i] = rf(p[i] * inv);
}

// Σ_p p_p·v_pd over keys [0, len) of kv head j (a [max][kvh][hd] cache), in `blk`-key blocks:
// each block an fma chain ascending from 0, the blocks added ascending from 0 (k2_attn_eager_pv
// then k2_attn_eager_reduce). blk >= len is one chain.
inline float eager_pv(const float* p, const uint16_t* kv_v, uint32_t len, uint32_t j, uint32_t kvh,
                      uint32_t hd, uint32_t d, uint32_t blk) {
  float o = 0.0f;
  for (uint32_t b0 = 0; b0 < len; b0 += blk) {
    float acc = 0.0f;
    for (uint32_t pp = b0; pp < std::min(len, b0 + blk); ++pp)
      acc = std::fma(p[pp], f32(kv_v[(size_t(pp) * kvh + j) * hd + d]), acc);
    o += acc;
  }
  return o;
}

// One q head over keys 0..len-1: s and p [len] (bf16 values), o [hd] the fp32 sums attn_gate()
// rounds. `blk`: the device's block (eager_block); len for one chain.
struct EagerHead {
  std::vector<float> s, p, o;
};
inline EagerHead attention_eager(const float* q, const uint16_t* kv_k, const uint16_t* kv_v,
                                 uint32_t len, uint32_t j, uint32_t kvh, uint32_t hd, uint32_t blk) {
  EagerHead r;
  r.s.resize(len);
  r.p.resize(len);
  r.o.resize(hd);
  for (uint32_t pp = 0; pp < len; ++pp) r.s[pp] = eager_score(q, kv_k + (size_t(pp) * kvh + j) * hd, hd);
  eager_softmax(r.s.data(), len, r.p.data());
  for (uint32_t d = 0; d < hd; ++d) r.o[d] = eager_pv(r.p.data(), kv_v, len, j, kvh, hd, d, blk);
  return r;
}

// The device's key block for a step at `pos` with `n_act` rows: v2's ppw over the longest row
// (k2_attn_eager.cl eager_ppw), so no row has more than tgt blocks.
inline uint32_t eager_block(uint32_t pos, uint32_t n_act, uint32_t tgt) {
  const uint32_t len = pos + n_act;
  uint32_t p = (len + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}

// tools/oracle/k2_attn_eager_fixture.py val(): element i of tensor t (0 q, 1 k, 2 v).
inline uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}
inline float eager_fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed) {
  const uint32_t h = lowbias32(lowbias32(seed * 3 + t) ^ i);
  if (coarse) return float(int(h % 17) - 8) * (t == 0 ? 0.5f : 0.125f);
  const float f = float(h >> 8) * 0x1p-23f - 1.0f;   // exact
  return rf(f * (t == 0 ? 4.0f : 1.0f));
}

}  // namespace k2_ref
