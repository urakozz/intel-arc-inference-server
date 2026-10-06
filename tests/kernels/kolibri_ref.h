#pragma once
// CPU references for spec 20c's Kolibri-1 kernels (src/kernels/kolibri/*.cl) - the same op chains in
// the same orders, every line with a twin in a kernel; edit the two together. The chains are
// tools/oracle/kolibri_ref.py's (spec 20a), op for op; tests/kernels/kolibri_ref_test.cc holds them to
// that reference's own outputs (tests/kernels/kolibri_fixture.h) and to the semantics' properties.
//
//   norm_finish()   kol_norm_finish: rstd from prep_res_fold's chunk sums (ascending), x̂ rounded to
//                   bf16, THEN × the plain w (two roundings: `w * x_hat.to(bf16)`)
//   post_add()      kol_post_add: the sub-block's own bf16 output normalised (its Σ² from
//                   prep_res_fold ..._Z), × the post norm's w, added to the residual (rounded), and
//                   the new residual's per-chunk Σ² in prep_res_fold's tree (the sandwich)
//   attn_prep()     kol_attn_prep: q / k per-head RMSNorm (Σ² over the head by a 128-lane tree), RoPE
//                   only in sliding layers (rne(y·cos) + rne(rot(y)·sin)), v the linear's output;
//                   K / V rows at (pos + m) & (RING - 1) (sliding) or pos + m (full)
//   route()         kol_route: sel = logit + bias over the 384 real experts (padded slots never
//                   ranked), rank with ties to the lower id, ids ascending, w = sigmoid(logit) fp32
//   combine()       kol_moe_down's epilogue: Σ_j fp32(rne(down_j)) × w_j in ascending id (mul rounded,
//                   then add), + fp32(rne(shared down)), one rounding; NOT folded into the residual
//   gate_up() / down()   the whole MoE block over layout-1 blocks and the bf16 shared slot
//   attention_eager()    kol_attn_eager.cl: k2_attn_eager's chain without the gate, over the row's
//                   keys [lo, hi] indexed from lo (the cached decode pass's row)
//   attention()     decode attention in fp64 over [lo, hi] (a tolerance reference for flash)
//   ring_row()      the sliding ring's key -> row map
//
// What is NOT bit-exact between these and the device: `exp` (sigmoid, SiLU: OpenCL's ulp bounds) and
// the GEMVs' multiply-adds (OpenCL may contract them). The orders and rounding points are exact.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"     // exp_torch, eager_softmax (torch's fp32 softmax), the eager block
#include "kernels/moe_ref.h"    // f32 / rne / rf, silu, the layout-1 tile_dot / split_dot
#include "kernels/prep_ref.h"   // res_fold (stage A, unchanged)

namespace kolibri_ref {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;
using moe_ref::silu_f32;
using moe_ref::split_dot;

constexpr float kEps = 1e-6f;
constexpr uint32_t kHidden = 2560, kHd = 128, kQHeads = 48, kKvHeads = 4, kExperts = 384, kTopK = 6,
                   kRouterN = 512, kInter = 512, kWindow = 513, kRing = 4096, kG = 20;
// The route row (kol_moe.cl R_*, kernels::kolibri::route).
constexpr uint32_t kWords = 32, kIds = 0, kWeights = 8, kSel = 16, kNext = 24, kLogitMin = 25;
constexpr float kAttnScale = k2_ref::kAttnScale;   // float(128 ** -0.5)

// --- the fixture's inputs and its hex arrays ------------------------------------------------
// tools/oracle/kolibri_fixture.py val(): element i of input tensor t. coarse: (h % 17 - 8) x step;
// fine: ((h >> 8) x 2^-23 - 1) x amp, rounded to bf16.
inline float fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float step = 0.125f, float amp = 1.0f) {
  const uint32_t h = k2_ref::lowbias32(k2_ref::lowbias32(seed * 3 + t) ^ i);
  if (coarse) return float(int(h % 17) - 8) * step;
  const float f = (float(h >> 8) * 0x1p-23f - 1.0f) * amp;
  return rf(f);
}
inline std::vector<uint16_t> hex16(const char* s) {
  std::vector<uint16_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 4 <= h.size(); i += 4) v.push_back(uint16_t(std::stoul(h.substr(i, 4), nullptr, 16)));
  return v;
}
inline std::vector<uint32_t> hex32(const char* s) {
  std::vector<uint32_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 8 <= h.size(); i += 8) v.push_back(uint32_t(std::stoul(h.substr(i, 8), nullptr, 16)));
  return v;
}
inline float as_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint32_t as_u32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// --- the norms --------------------------------------------------------------------------------
// rstd from `G` chunk sums [G][M] (prep_res_fold's), ascending; never rsqrt.
inline float rstd_of(const float* sumsq, uint32_t M, uint32_t m, uint32_t K, uint32_t G) {
  float total = 0.f;
  for (uint32_t g = 0; g < G; ++g) total += sumsq[size_t(g) * M + m];
  return 1.0f / std::sqrt(total / float(K) + kEps);
}
// One element of Kolibri's RMSNorm: rne(f32(rne(x · rstd)) · w) - kolibri_ref.rms_norm's
// `w * h.to(bf16)` (x̂ rounded, then the bf16 product).
inline uint16_t norm_elem(float x, float rstd, float w) { return rne(rf(x * rstd) * w); }

inline void norm_finish(const float* sumsq, const uint16_t* resid, const float* w, uint16_t* x, uint32_t M,
                        uint32_t K = kHidden, uint32_t G = kG) {
  for (uint32_t m = 0; m < M; ++m) {
    const float rstd = rstd_of(sumsq, M, m, K, G);
    for (uint32_t k = 0; k < K; ++k) x[size_t(m) * K + k] = norm_elem(f32(resid[size_t(m) * K + k]), rstd, w[k]);
  }
}

// kol_post_add: `a` [M][K] (the sub-block's bf16 output) with its chunk sums `sumsq_a` [G][M];
// resid += rne(rf(a · rstd_a) · w); sumsq_out [G][M] = the new resid's chunk trees (256 lanes:
// lane i of chunk g takes k = g·chunk + i, + 256, ... with fma; then 256 -> 1 pairwise).
inline void post_add(const float* sumsq_a, const uint16_t* a, const float* w, uint16_t* resid, float* sumsq_out,
                     uint32_t M, uint32_t K = kHidden, uint32_t G = kG) {
  const uint32_t chunk = (K + G - 1) / G, wg = 256;
  std::vector<float> red(wg);
  for (uint32_t m = 0; m < M; ++m) {
    const float rstd = rstd_of(sumsq_a, M, m, K, G);
    uint16_t* rp = resid + size_t(m) * K;
    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t k0 = g * chunk, k1 = std::min(K, k0 + chunk);
      for (uint32_t i = 0; i < wg; ++i) {
        float acc2 = 0.f;
        for (uint32_t k = k0 + i; k < k1; k += wg) {
          const uint16_t n = norm_elem(f32(a[size_t(m) * K + k]), rstd, w[k]);
          const uint16_t r = rne(f32(rp[k]) + f32(n));
          rp[k] = r;
          const float v = f32(r);
          acc2 = std::fma(v, v, acc2);
        }
        red[i] = acc2;
      }
      for (uint32_t stride = wg / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
      sumsq_out[size_t(g) * M + m] = red[0];
    }
  }
}

// --- the attention prep ---------------------------------------------------------------------------
// One head of 128 bf16 values x_b: the head RMSNorm with w (Σ² by a 128-lane tree: lane d holds
// fma(x, x, 0), then 64, 32, ... pairwise), then (sliding) RoPE with the bf16-valued cos / sin row
// `cs` [2][64] (cos then sin; null = NoPE).
inline void head_norm_rope(const float* xb, const float* w, const float* cs, uint16_t* out) {
  float red[kHd];
  for (uint32_t d = 0; d < kHd; ++d) red[d] = std::fma(xb[d], xb[d], 0.0f);
  for (uint32_t stride = kHd / 2; stride > 0; stride >>= 1)
    for (uint32_t d = 0; d < stride; ++d) red[d] += red[d + stride];
  const float rstd = 1.0f / std::sqrt(red[0] / float(kHd) + kEps);
  float y[kHd];
  for (uint32_t d = 0; d < kHd; ++d) y[d] = f32(norm_elem(xb[d], rstd, w[d]));
  for (uint32_t d = 0; d < kHd; ++d) {
    if (!cs) {
      out[d] = rne(y[d]);
      continue;
    }
    const uint32_t ii = d % (kHd / 2);
    const float c = cs[ii], s = cs[kHd / 2 + ii];
    const float rot = d < kHd / 2 ? -y[d + kHd / 2] : y[d - kHd / 2];
    out[d] = rne(rf(y[d] * c) + rf(rot * s));
  }
}

struct Prep {
  std::vector<float> q;        // [q_heads][128] (bf16 values)
  std::vector<uint16_t> k, v;  // [kv_heads][128]
};
// Token m of the fused q||k||v GEMV's [S][M][7168] fp32 partials; `qkn` = q_norm [128] then k_norm
// [128] (fp32 plain w); `cs` the RoPE row at pos + m (null: a full, NoPE layer).
inline Prep attn_prep(const float* partials, uint32_t M, uint32_t m, uint32_t S, const float* qkn, const float* cs) {
  const uint32_t N = kQHeads * kHd + 2 * kKvHeads * kHd;
  auto col = [&](size_t c) {
    float v = 0.f;
    for (uint32_t s = 0; s < S; ++s) v += partials[(size_t(s) * M + m) * N + c];
    return rf(v);
  };
  Prep p;
  p.q.resize(size_t(kQHeads) * kHd);
  p.k.resize(size_t(kKvHeads) * kHd);
  p.v.resize(size_t(kKvHeads) * kHd);
  float xb[kHd];
  uint16_t out[kHd];
  for (uint32_t h = 0; h < kQHeads; ++h) {
    for (uint32_t d = 0; d < kHd; ++d) xb[d] = col(size_t(h) * kHd + d);
    head_norm_rope(xb, qkn, cs, out);
    for (uint32_t d = 0; d < kHd; ++d) p.q[size_t(h) * kHd + d] = f32(out[d]);
  }
  for (uint32_t j = 0; j < kKvHeads; ++j) {
    for (uint32_t d = 0; d < kHd; ++d) xb[d] = col(size_t(kQHeads) * kHd + size_t(j) * kHd + d);
    head_norm_rope(xb, qkn + kHd, cs, p.k.data() + size_t(j) * kHd);
    for (uint32_t d = 0; d < kHd; ++d)
      p.v[size_t(j) * kHd + d] = rne(col(size_t(kQHeads + kKvHeads) * kHd + size_t(j) * kHd + d));
  }
  return p;
}

// The sliding ring: key p lives at row p & (RING - 1); a full layer's at row p.
inline uint32_t ring_row(uint32_t p, bool sliding) { return sliding ? (p & (kRing - 1)) : p; }
// The visible keys of query `pos`: [lo, pos], lo = pos - 512 (clamped) in a sliding layer (513 keys
// including the query: i - 513 < j <= i), 0 in a full one.
inline uint32_t key_lo(uint32_t pos, bool sliding) { return sliding && pos + 1 > kWindow ? pos + 1 - kWindow : 0; }
// The rows a query at `pos` reads, in key order.
inline std::vector<uint32_t> ring_rows(uint32_t pos, bool sliding) {
  std::vector<uint32_t> r;
  for (uint32_t p = key_lo(pos, sliding); p <= pos; ++p) r.push_back(ring_row(p, sliding));
  return r;
}

// --- the router ----------------------------------------------------------------------------------
struct Route {
  uint32_t ids[kTopK] = {};
  float w[kTopK] = {};       // sigmoid(logit), fp32, slot order (ascending id)
  float sel[kTopK] = {};     // logit + bias, slot order
  float next = -INFINITY;    // sel of rank top_k (the first not taken)
  float logit_min = 0;       // the smallest logit among the selected (diagnostics)
  uint32_t rank_ids[kTopK + 1] = {};
};
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
// `logits` fp32 [router_n] (the GEMV's row; padded entries ignored), `bias` fp32 [>= E].
inline Route route(const float* logits, const float* bias, uint32_t E = kExperts, uint32_t K = kTopK) {
  std::vector<float> sel(E);
  for (uint32_t e = 0; e < E; ++e) sel[e] = logits[e] + bias[e];
  std::vector<uint32_t> top_id(K + 1);
  std::vector<float> top_sel(K + 1, -INFINITY);
  for (uint32_t k = 0; k <= K; ++k) top_id[k] = k;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < E; ++j) rank += (sel[j] > sel[e] || (sel[j] == sel[e] && j < e)) ? 1u : 0u;
    if (rank <= K) {
      top_id[rank] = e;
      top_sel[rank] = sel[e];
    }
  }
  Route r;
  for (uint32_t k = 0; k <= K; ++k) r.rank_ids[k] = top_id[k];
  std::vector<uint32_t> ids(top_id.begin(), top_id.begin() + K);
  std::sort(ids.begin(), ids.end());
  r.logit_min = INFINITY;
  for (uint32_t j = 0; j < K; ++j) {
    r.ids[j] = ids[j];
    r.w[j] = sigmoid(logits[ids[j]]);
    r.sel[j] = sel[ids[j]];
    r.logit_min = std::min(r.logit_min, logits[ids[j]]);
  }
  r.next = top_sel[K];
  return r;
}
// The selection score gap at the cut (the reference's route.moe.gap): sel of rank K-1 - sel of rank K.
inline float gap(const float* logits, const float* bias, const Route& r) {
  return (logits[r.rank_ids[kTopK - 1]] + bias[r.rank_ids[kTopK - 1]]) - r.next;
}

// --- the MoE block ---------------------------------------------------------------------------------
// One bf16-tiled column n ({K, N} gemv_bf16 tiles) over `ks` K slices, each an ascending k chain
// (a += w · x), merged by the pairwise tree - kol_moe.cl's shared slot.
inline float bf16_split_dot(const uint16_t* tiles, uint32_t K, uint32_t n, const uint16_t* x, uint32_t ks) {
  const uint32_t K8 = K / 8, per = K8 / ks;
  std::vector<float> r(ks);
  for (uint32_t q = 0; q < ks; ++q) {
    float a = 0.f;
    for (uint32_t k8 = q * per; k8 < (q + 1) * per; ++k8)
      for (uint32_t i = 0; i < 8; ++i) a += f32(tiles[((size_t(n / 16) * K8 + k8) * 8 + i) * 16 + n % 16]) * f32(x[k8 * 8 + i]);
    r[q] = a;
  }
  for (uint32_t stride = ks / 2; stride > 0; stride >>= 1)
    for (uint32_t q = 0; q < stride; ++q) r[q] += r[q + stride];
  return r[0];
}

constexpr uint32_t kSlots = kTopK + 1;   // 6 routed + the shared expert (slot 6)
inline size_t gate_up_block_words() { return size_t(2 * kInter / 16) * (kHidden / 64) * 136; }
inline size_t down_block_words() { return size_t(kHidden / 16) * (kInter / 64) * 136; }

// h [7][512] bf16: slot j < 6 the routed expert ids[j] (layout-1 blocks), slot 6 the shared expert
// (bf16 tiles {hidden, 2I}); h = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up))).
inline std::vector<uint16_t> gate_up(const uint32_t* ids, const uint16_t* x, const uint32_t* gu,
                                     const uint16_t* sh_gu, uint32_t up_ks) {
  std::vector<uint16_t> h(size_t(kSlots) * kInter);
  for (uint32_t slot = 0; slot < kSlots; ++slot)
    for (uint32_t i = 0; i < kInter; ++i) {
      const uint32_t gc = moe_ref::gate_col(i), uc = moe_ref::up_col(i);
      float g, u;
      if (slot < kTopK) {
        const uint32_t* blk = gu + ids[slot] * gate_up_block_words();
        g = split_dot(blk, kHidden, gc, x, up_ks);
        u = split_dot(blk, kHidden, uc, x, up_ks);
      } else {
        g = bf16_split_dot(sh_gu, kHidden, gc, x, up_ks);
        u = bf16_split_dot(sh_gu, kHidden, uc, x, up_ks);
      }
      h[size_t(slot) * kInter + i] = moe_ref::silu_up(g, u);
    }
  return h;
}

// kol_moe_down's epilogue for one hidden column: down[7] fp32 sums (slot order, shared last).
inline uint16_t combine(const float* down, const float* w) {
  float acc = 0.0f;
  for (uint32_t j = 0; j < kTopK; ++j) {
    const float t = f32(rne(down[j])) * w[j];   // the product, rounded to fp32
    acc = acc + t;                             // then the add (no fma: FP_CONTRACT OFF)
  }
  return rne(acc + f32(rne(down[kTopK])));
}

// mo [2560] bf16 (NOT added to the residual: post_ffn_norm comes first).
inline std::vector<uint16_t> down(const uint32_t* ids, const float* w, const std::vector<uint16_t>& h,
                                  const uint32_t* dn, const uint16_t* sh_dn, uint32_t dn_ks) {
  std::vector<uint16_t> out(kHidden);
  float d[kSlots];
  for (uint32_t n = 0; n < kHidden; ++n) {
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      const uint16_t* hs = h.data() + size_t(slot) * kInter;
      d[slot] = slot < kTopK ? split_dot(dn + ids[slot] * down_block_words(), kInter, n, hs, dn_ks)
                             : bf16_split_dot(sh_dn, kInter, n, hs, dn_ks);
    }
    out[n] = combine(d, w);
  }
  return out;
}

// --- eager decode attention (kol_attn_eager.cl) ---------------------------------------------------
// One q head over the keys [lo, hi] of kv head j in a [rows][kv_heads][128] cache read through
// ring_row: s_i = rne(f32(rne(q·k)) · scale) for i = p - lo; torch's fp32 softmax over the row in
// k2_ref::eager_softmax's order (8 lanes over i); P·V in `blk`-key blocks from i = 0.
struct EagerHead {
  std::vector<float> s, p, o;
};
inline EagerHead attention_eager(const float* q, const uint16_t* kv_k, const uint16_t* kv_v, uint32_t lo,
                                 uint32_t hi, bool sliding, uint32_t j, uint32_t blk) {
  const uint32_t L = hi - lo + 1;
  EagerHead r;
  r.s.resize(L);
  r.p.resize(L);
  r.o.assign(kHd, 0.0f);
  for (uint32_t i = 0; i < L; ++i)
    r.s[i] = k2_ref::eager_score(q, kv_k + (size_t(ring_row(lo + i, sliding)) * kKvHeads + j) * kHd, kHd);
  k2_ref::eager_softmax(r.s.data(), L, r.p.data());
  for (uint32_t d = 0; d < kHd; ++d) {
    float o = 0.0f;
    for (uint32_t b0 = 0; b0 < L; b0 += blk) {
      float acc = 0.0f;
      for (uint32_t i = b0; i < std::min(L, b0 + blk); ++i)
        acc = std::fma(r.p[i], f32(kv_v[(size_t(ring_row(lo + i, sliding)) * kKvHeads + j) * kHd + d]), acc);
      o += acc;
    }
    r.o[d] = o;
  }
  return r;
}
// The device's key block at a step: v2's ppw over the step's key span (pos + n_act - lo of row 0).
inline uint32_t eager_block(uint32_t span, uint32_t tgt = 32) {
  uint32_t p = (span + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}

// --- decode attention in fp64 (flash's tolerance reference) ---------------------------------------
inline std::vector<double> attention(const float* q, const uint16_t* kv_k, const uint16_t* kv_v, uint32_t lo,
                                     uint32_t hi, bool sliding, uint32_t j) {
  const uint32_t L = hi - lo + 1;
  std::vector<double> sc(L);
  double mx = -INFINITY;
  for (uint32_t i = 0; i < L; ++i) {
    const size_t row = (size_t(ring_row(lo + i, sliding)) * kKvHeads + j) * kHd;
    double a = 0;
    for (uint32_t d = 0; d < kHd; ++d) a += double(q[d]) * f32(kv_k[row + d]);
    sc[i] = a / std::sqrt(double(kHd));
    mx = std::max(mx, sc[i]);
  }
  double z = 0;
  for (double& s : sc) z += (s = std::exp(s - mx));
  std::vector<double> o(kHd, 0.0);
  for (uint32_t i = 0; i < L; ++i) {
    const size_t row = (size_t(ring_row(lo + i, sliding)) * kKvHeads + j) * kHd;
    for (uint32_t d = 0; d < kHd; ++d) o[d] += sc[i] / z * f32(kv_v[row + d]);
  }
  return o;
}

}  // namespace kolibri_ref
