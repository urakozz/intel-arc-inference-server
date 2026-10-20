#pragma once
// CPU references for spec 24c's LFM2.5-24B-A2B decode kernels (src/kernels/lfm2moe/*.cl) - the same op chains in
// the same orders, every line with a twin in a kernel; EDIT THE TWO TOGETHER. The chains are spec 24 §2's (24a's
// tools/oracle/lfm2moe_ref.py restates transformers 5.19.0 op for op); tests/kernels/lfm2moe_ref_test.cc holds
// them to that reference's own outputs (tests/kernels/lfm2moe_fixture.h) and to the semantics' properties
// (plan 24c Review Focus 1-3, 5). Used by the kernel tests (Tasks 3, 5) and by 24d / 24e.
//
//   res_norm()    lfm_res_norm: fold the preceding GEMV's S split-K slices into the residual
//                 (h_out = rne(h_in + rne(Σ_s slices)); S = 0 "_NF": no fold), Σ r² by ONE fixed tree (256 lanes,
//                 lane i: fma over k = i, i + 256, ... ascending; then 128 .. 1 pairwise), rstd = 1 / sqrt(Σ / 2048 +
//                 1e-5) (never rsqrt), x = rne(f32(rne(r · rstd)) · w) - spec 24 §2.2's two roundings
//   conv_step()   lfm_conv: B, C, x = rne(Σ_s slices) at columns c, 2048 + c, 4096 + c; Bx = rne(B · x) into ring
//                 slot (pos + m) % 16; acc = fma(w2, Bx[p], fma(w1, Bx[p-1], fma(w0, Bx[p-2], 0))) - taps oldest first,
//                 positions < 0 read 0 (the ring is never cleared: a fresh sequence never reads it); y = rne(C · rne(acc))
//   router()      lfm_router (and the _R arm): fp32 logits of 64 bf16 rows over gemv_bf16's tiles - per row 8 lanes of
//                 256 consecutive k (ascending, exact bf16 products), then the 8 partials by the pairwise tree
//                 (stride 4, 2, 1); NEVER rounded here (lfm_route rounds)
//   route()       lfm_route: spec 24 §2.3 - l_b = rne(logit); s_b = rne(sigmoid(l_b)); sel = s_b + bias (fp32);
//                 rank with exact ties to the LOWER id; ids in RANK order; Σ = ((s0 + s1) + s2) + s3 (fp32, rank
//                 order); d = rne(rne(Σ) + 1e-6); w_j = rne(s_j / d); the 5th sel kept (the near-tie margin)
//   moe()         lfm_moe_gate_up + lfm_moe_down: per slot (rank order) gate||up over the expert's layout-1 block,
//                 h = rne(rne(silu(rne(g))) · rne(u)); d = rne(down); t_k = rne(d_k · w_k); y = rne(((t0 + t1) + t2)
//                 + t3) in fp32 (24a: 5.19.0 runs grouped_mm, slot order); h_res = rne(h_res + y) (the fold)
//   attn_prep()   lfm_attn_prep: q / k per-head RMSNorm over 64 (64-lane tree, eps 1e-5, plain w), RoPE on every
//                 attention layer (rotate-half, rne(rne(y · cos) + rne(rot(y) · sin)), the [max_len][2][32] table),
//                 v the linear's output; K / V rows at pos + m (the growing cache)
//   attention_eager()  lfm_attn_eager.cl: k2_attn_eager's chain at head_dim 64 (scale 0.125: exact)
//   attention()   decode attention in fp64 (flash's tolerance reference)
//   silu_mul()    lfm_silu_mul: prep_silu_mul's chain at I 11776 (gate||up interleaved in 16s)
//   argmax()      argmax.cl's comparator at VOCAB 128000, VOCAB_USED 125017: ids >= vocab_used never win
//   embed_row()   lfm_embed_gather: the token's row out of gemv_bf16's tiles (loader::lfm::lfm_embed_row_index)
//
// What is NOT bit-exact between these and the device: `exp` (sigmoid, SiLU: OpenCL's ulp bounds against libm's)
// and nothing else - every sum's order and every rounding point is the kernels'.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"    // exp_torch, eager_score / eager_softmax (torch's fp32 softmax), lowbias32
#include "kernels/moe_ref.h"   // f32 / rne / rf, silu, the layout-1 tile_dot / split_dot, gate_col / up_col

namespace lfmref {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;
using moe_ref::silu_f32;
using moe_ref::split_dot;

constexpr float kEps = 1e-5f;
constexpr uint32_t kHidden = 2048, kQHeads = 32, kKvHeads = 8, kHd = 64, kQkvN = 3072, kInProjN = 6144;
constexpr uint32_t kExperts = 64, kTopK = 4, kInter = 1536, kDenseInter = 11776, kVocab = 128000, kVocabUsed = 125017;
constexpr uint32_t kRing = 16;                      // loader::lfm::kConvRing
constexpr uint32_t kNormWg = 256;                   // lfm_res_norm's lanes (every work-group computes the same Σ)
constexpr uint32_t kRouterLanes = 8;                // lfm_router: 8 partials of 256 k a row
constexpr float kAttnScale = 0.125f;                // 64 ** -0.5, exact
// The route row (lfm_moe.cl R_*, kernels::lfm2moe::route).
constexpr uint32_t kWords = 16, kIds = 0, kW = 4, kSel = 8, kSum = 13;

// --- the fixture's inputs and hex arrays (tools/oracle/lfm2moe_fixture.py val()) ----------------------------------
// coarse: (h % 17 - 8) x step (exact sums in any order); fine: ((h >> 8) x 2^-23 - 1) x amp, rounded to bf16.
inline float fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float step = 0.125f, float amp = 1.0f) {
  const uint32_t h = k2_ref::lowbias32(k2_ref::lowbias32(seed * 3 + t) ^ i);
  if (coarse) return float(int(h % 17) - 8) * step;
  const float f = (float(h >> 8) * 0x1p-23f - 1.0f) * amp;
  return rf(f);
}
inline uint32_t fixture_hash(uint32_t t, uint32_t i, uint32_t seed) {
  return k2_ref::lowbias32(k2_ref::lowbias32(seed * 3 + t) ^ i);
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

// --- the norm (lfm_res_norm) -----------------------------------------------------------------------------------
// The split-K sum of column `col` of row m over S slices [S][M][N]: 0.0f + slice 0 + slice 1 ... (the order every
// consumer of a GEMV's slices uses - lfm_conv_in's epilogue starts its single slice from the same 0.0f).
inline float slice_sum(const float* slices, uint32_t S, uint32_t M, uint32_t m, uint32_t N, size_t col) {
  float v = 0.0f;
  for (uint32_t s = 0; s < S; ++s) v += slices[(size_t(s) * M + m) * N + col];
  return v;
}
// Σ r² of one row by lfm_res_norm's tree (identical in every work-group).
inline float norm_sumsq(const uint16_t* r, uint32_t K = kHidden) {
  std::vector<float> red(kNormWg, 0.0f);
  for (uint32_t i = 0; i < kNormWg; ++i) {
    float acc = 0.0f;
    for (uint32_t k = i; k < K; k += kNormWg) {
      const float v = f32(r[k]);
      acc = std::fma(v, v, acc);
    }
    red[i] = acc;
  }
  for (uint32_t stride = kNormWg / 2; stride > 0; stride >>= 1)
    for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
  return red[0];
}
inline float rstd_of(float sumsq, uint32_t K = kHidden) { return 1.0f / std::sqrt(sumsq / float(K) + kEps); }
// One element of the two-rounding norm: rne(f32(rne(r · rstd)) · w).
inline uint16_t norm_elem(float r, float rstd, float w) { return rne(rf(r * rstd) * w); }

struct NormOut {
  std::vector<uint16_t> h_out, x;   // [M][K] each; h_out == h_in under S = 0
};
// `slices` [S][M][K] fp32 (ignored at S = 0), `h_in` [M][K] bf16, `w` fp32 [K] (bf16 values).
inline NormOut res_norm(const uint16_t* h_in, const float* slices, uint32_t S, uint32_t M, const float* w,
                        uint32_t K = kHidden) {
  NormOut o;
  o.h_out.resize(size_t(M) * K);
  o.x.resize(size_t(M) * K);
  for (uint32_t m = 0; m < M; ++m) {
    uint16_t* r = o.h_out.data() + size_t(m) * K;
    for (uint32_t k = 0; k < K; ++k) {
      const uint16_t hv = h_in[size_t(m) * K + k];
      r[k] = S ? rne(f32(hv) + f32(rne(slice_sum(slices, S, M, m, K, k)))) : hv;
    }
    const float rstd = rstd_of(norm_sumsq(r, K), K);
    for (uint32_t k = 0; k < K; ++k) o.x[size_t(m) * K + k] = norm_elem(f32(r[k]), rstd, w[k]);
  }
  return o;
}

// --- the short conv (lfm_conv) ----------------------------------------------------------------------------------
// `slices` [S][M][6144] (in_proj's: B | C | x), `ring` bf16 [16][2048] (Bx at slot p % 16), `taps` fp32 [2048][3]
// (tap 0 multiplies Bx[p - 2], tap 2 the current Bx), the rows' positions pos .. pos + M - 1 (M <= 14). Writes
// the M new Bx into `ring` and returns y [M][2048].
inline std::vector<uint16_t> conv_step(const float* slices, uint32_t S, uint16_t* ring, const float* taps,
                                       uint32_t pos, uint32_t M, uint32_t H = kHidden) {
  std::vector<uint16_t> y(size_t(M) * H);
  const uint32_t N = 3 * H;
  for (uint32_t c = 0; c < H; ++c) {
    const float w0 = taps[size_t(c) * 3 + 0], w1 = taps[size_t(c) * 3 + 1], w2 = taps[size_t(c) * 3 + 2];
    const int p2 = int(pos) - 2, p1 = int(pos) - 1;
    float f2 = p2 < 0 ? 0.0f : f32(ring[size_t(uint32_t(p2) % kRing) * H + c]);
    float f1 = p1 < 0 ? 0.0f : f32(ring[size_t(uint32_t(p1) % kRing) * H + c]);
    for (uint32_t m = 0; m < M; ++m) {
      const uint16_t B = rne(slice_sum(slices, S, M, m, N, c));
      const uint16_t C = rne(slice_sum(slices, S, M, m, N, H + c));
      const uint16_t X = rne(slice_sum(slices, S, M, m, N, 2 * H + c));
      const uint16_t bx = rne(f32(B) * f32(X));
      ring[size_t((pos + m) % kRing) * H + c] = bx;
      const float acc = std::fma(w2, f32(bx), std::fma(w1, f1, std::fma(w0, f2, 0.0f)));
      y[size_t(m) * H + c] = rne(f32(C) * f32(rne(acc)));
      f2 = f1;
      f1 = f32(bx);
    }
  }
  return y;
}

// --- the router (lfm_router, the _R arm) -------------------------------------------------------------------------
// gemv_bf16's tiles {K, N}: element (k, n) at ((n / 16) * K/8 + k / 8) * 8 + k % 8) * 16 + n % 16.
inline size_t tile_index(uint32_t K, uint32_t k, uint32_t n) {
  return ((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16;
}
// fp32 logits [E] of one row x [K] (bf16) over the router's tiles {K, E}.
inline std::vector<float> router(const uint16_t* x, const uint16_t* tiles, uint32_t E = kExperts, uint32_t K = kHidden) {
  std::vector<float> out(E);
  const uint32_t span = K / kRouterLanes;
  for (uint32_t n = 0; n < E; ++n) {
    float p[kRouterLanes];
    for (uint32_t j = 0; j < kRouterLanes; ++j) {
      float a = 0.0f;
      for (uint32_t k = j * span; k < (j + 1) * span; ++k) a += f32(tiles[tile_index(K, k, n)]) * f32(x[k]);
      p[j] = a;
    }
    for (uint32_t stride = kRouterLanes / 2; stride > 0; stride >>= 1)
      for (uint32_t j = 0; j < stride; ++j) p[j] += p[j + stride];
    out[n] = p[0];
  }
  return out;
}

// --- the route (lfm_route) ------------------------------------------------------------------------------------
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
struct Route {
  uint32_t ids[kTopK] = {};   // RANK order (descending sel, exact ties to the lower id) - the combine's order
  float w[kTopK] = {};        // rne(s_j / d), bf16 values
  float s[kTopK] = {};        // s_b of the selected, bf16 values
  float sel[kTopK + 1] = {};  // sel of ranks 0 .. 4 (rank 4: the first expert not taken)
  float sum = 0;              // d = rne(rne(Σ s) + 1e-6), a bf16 value
  uint32_t row[kWords] = {};  // the route row as lfm_route writes it
};
inline Route route(const float* logits, const float* bias, uint32_t E = kExperts) {
  std::vector<float> s(E), sel(E);
  for (uint32_t e = 0; e < E; ++e) {
    s[e] = rf(sigmoid(rf(logits[e])));
    sel[e] = s[e] + bias[e];
  }
  Route r;
  for (uint32_t k = 0; k <= kTopK; ++k) r.sel[k] = -INFINITY;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < E; ++j) rank += (sel[j] > sel[e] || (sel[j] == sel[e] && j < e)) ? 1u : 0u;
    if (rank < kTopK) {
      r.ids[rank] = e;
      r.s[rank] = s[e];
    }
    if (rank <= kTopK) r.sel[rank] = sel[e];
  }
  float sum = 0.0f;
  for (uint32_t k = 0; k < kTopK; ++k) sum = sum + r.s[k];   // ((s0 + s1) + s2) + s3, fp32
  r.sum = rf(rf(sum) + 1e-6f);
  for (uint32_t k = 0; k < kTopK; ++k) r.w[k] = rf(r.s[k] / r.sum);
  for (uint32_t k = 0; k < kTopK; ++k) {
    r.row[kIds + k] = r.ids[k];
    r.row[kW + k] = as_u32(r.w[k]);
  }
  for (uint32_t k = 0; k <= kTopK; ++k) r.row[kSel + k] = as_u32(r.sel[k]);
  r.row[kSum] = as_u32(r.sum);
  return r;
}
// The selection gap at the cut (24a's route.gap): sel of rank 3 - sel of rank 4.
inline float gap(const Route& r) { return r.sel[kTopK - 1] - r.sel[kTopK]; }

// --- the MoE block (lfm_moe_gate_up, lfm_moe_down) -----------------------------------------------------------------
constexpr uint32_t kUpKs = 4, kDnKs = 2;   // kernels::lfm2moe::kUpKs / kDnKs (PROVISIONAL)
inline size_t gate_up_block_words(uint32_t H = kHidden, uint32_t I = kInter) { return size_t(2 * I / 16) * (H / 64) * 136; }
inline size_t down_block_words(uint32_t H = kHidden, uint32_t I = kInter) { return size_t(H / 16) * (I / 64) * 136; }
// h [4][I] bf16 for the route's slots (rank order): expert ids[j]'s gate||up block at gu + id x gate_up_block_words.
inline std::vector<uint16_t> gate_up(const uint32_t* ids, const uint16_t* x, const uint32_t* gu, uint32_t up_ks = kUpKs,
                                     uint32_t H = kHidden, uint32_t I = kInter) {
  std::vector<uint16_t> h(size_t(kTopK) * I);
  for (uint32_t j = 0; j < kTopK; ++j) {
    const uint32_t* blk = gu + ids[j] * gate_up_block_words(H, I);
    for (uint32_t i = 0; i < I; ++i) {
      const float g = split_dot(blk, H, moe_ref::gate_col(i), x, up_ks);
      const float u = split_dot(blk, H, moe_ref::up_col(i), x, up_ks);
      h[size_t(j) * I + i] = moe_ref::silu_up(g, u);
    }
  }
  return h;
}
// The combine of one hidden column: d[4] the slots' fp32 down sums (rank order), w[4] their weights.
//   t_k = rne(f32(rne(d_k)) · w_k);  y = rne(((t0 + t1) + t2) + t3)   (fp32, no fma, rank order: grouped_mm's)
inline uint16_t combine(const float* d, const float* w) {
  float acc = f32(rne(f32(rne(d[0])) * w[0]));
  for (uint32_t k = 1; k < kTopK; ++k) {
    const float t = f32(rne(f32(rne(d[k])) * w[k]));
    acc = acc + t;
  }
  return rne(acc);
}
// The same combine from the experts' bf16 outputs (the fixture's torch proj rows).
inline uint16_t combine_proj(const uint16_t* proj, const float* w) {
  float acc = f32(rne(f32(proj[0]) * w[0]));
  for (uint32_t k = 1; k < kTopK; ++k) acc = acc + f32(rne(f32(proj[k]) * w[k]));
  return rne(acc);
}
struct MoeOut {
  std::vector<uint16_t> h, y, h_res;   // gate||up [4][I], the combine [H], the folded residual [H]
};
inline MoeOut moe(const Route& r, const uint16_t* x, const uint32_t* gu, const uint32_t* dn, const uint16_t* h_res,
                  uint32_t up_ks = kUpKs, uint32_t dn_ks = kDnKs, uint32_t H = kHidden, uint32_t I = kInter) {
  MoeOut o;
  o.h = gate_up(r.ids, x, gu, up_ks, H, I);
  o.y.resize(H);
  o.h_res.resize(H);
  for (uint32_t n = 0; n < H; ++n) {
    float d[kTopK];
    for (uint32_t j = 0; j < kTopK; ++j)
      d[j] = split_dot(dn + r.ids[j] * down_block_words(H, I), I, n, o.h.data() + size_t(j) * I, dn_ks);
    o.y[n] = combine(d, r.w);
    o.h_res[n] = rne(f32(h_res[n]) + f32(o.y[n]));
  }
  return o;
}

// The fixture's int4 experts (tools/oracle/lfm2moe_fixture.py expert_weight): matrix `mat` (0 w1 / gate, 1 w3 / up,
// 2 w2 / down) of expert e as GPTQ qweight [K/8][N] + f16 scales [K/64][N] - nibble q(k, n) = hash(100 + 4e + mat,
// k x N + n, 8) % 16, scale 2^-7 x (1 + ((hash(100 + 4e + mat, g x N + n, 9) >> 4) % 8) / 8).
struct Gptq {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;
};
inline Gptq expert_gptq(uint32_t e, uint32_t mat, uint32_t K, uint32_t N) {
  Gptq g;
  g.K = K;
  g.N = N;
  g.qweight.assign(size_t(K / 8) * N, 0u);
  g.scales.resize(size_t(K / 64) * N);
  const uint32_t t = 100 + 4 * e + mat;
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t q = fixture_hash(t, k * N + n, 8) % 16;
      g.qweight[size_t(k / 8) * N + n] |= q << (4 * (k % 8));
    }
  for (uint32_t gg = 0; gg < K / 64; ++gg)
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t j = (fixture_hash(t, gg * N + n, 9) >> 4) % 8;
      g.scales[size_t(gg) * N + n] = common::f32_to_f16((1.0f + float(j) / 8.0f) * 0x1p-7f);
    }
  return g;
}

// --- the attention prep (lfm_attn_prep) ---------------------------------------------------------------------------
// One head of 64 bf16 values x_b: the head RMSNorm (Σ² by a 64-lane tree: lane d holds fma(x, x, 0), then 32, 16,
// ... pairwise), then RoPE with the bf16-valued row `cs` [2][32] (cos then sin) - every attention layer rotates.
inline void head_norm_rope(const float* xb, const float* w, const float* cs, uint16_t* out) {
  float red[kHd];
  for (uint32_t d = 0; d < kHd; ++d) red[d] = std::fma(xb[d], xb[d], 0.0f);
  for (uint32_t stride = kHd / 2; stride > 0; stride >>= 1)
    for (uint32_t d = 0; d < stride; ++d) red[d] += red[d + stride];
  const float rstd = 1.0f / std::sqrt(red[0] / float(kHd) + kEps);
  float y[kHd];
  for (uint32_t d = 0; d < kHd; ++d) y[d] = f32(norm_elem(xb[d], rstd, w[d]));
  for (uint32_t d = 0; d < kHd; ++d) {
    const uint32_t ii = d % (kHd / 2);
    const float c = cs[ii], s = cs[kHd / 2 + ii];
    const float rot = d < kHd / 2 ? -y[d + kHd / 2] : y[d - kHd / 2];
    out[d] = rne(rf(y[d] * c) + rf(rot * s));
  }
}
struct Prep {
  std::vector<float> q;        // [32][64] (bf16 values)
  std::vector<uint16_t> k, v;  // [8][64]
};
// Row m of the fused q||k||v GEMV's [S][M][3072] fp32 slices; `qkn` = q_norm [64] then k_norm [64] (fp32); `cs`
// the RoPE row at pos + m ([2][32]: loader::lfm_rope_table's).
inline Prep attn_prep(const float* slices, uint32_t S, uint32_t M, uint32_t m, const float* qkn, const float* cs) {
  auto col = [&](size_t c) { return rf(slice_sum(slices, S, M, m, kQkvN, c)); };
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

// --- eager decode attention (lfm_attn_eager.cl) -------------------------------------------------------------------
// One q head over keys [0, L) of kv head j in a [rows][8][64] cache: s_i = rne(f32(rne(q·k)) · 0.125); torch's fp32
// softmax over the row (k2_ref::eager_softmax: 8 lanes over i); P·V in `blk`-key blocks (an fma chain each,
// ascending), the blocks added ascending; o fp32 (the reduce rounds once).
struct EagerHead {
  std::vector<float> s, p, o;
};
inline uint32_t eager_block(uint32_t span, uint32_t tgt = 32) {
  uint32_t p = (span + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}
inline EagerHead attention_eager(const float* q, const uint16_t* kv_k, const uint16_t* kv_v, uint32_t L, uint32_t j,
                                 uint32_t blk) {
  EagerHead r;
  r.s.resize(L);
  r.p.resize(L);
  r.o.assign(kHd, 0.0f);
  for (uint32_t i = 0; i < L; ++i) {
    float a = 0.0f;
    const uint16_t* k = kv_k + (size_t(i) * kKvHeads + j) * kHd;
    for (uint32_t d = 0; d < kHd; ++d) a = std::fma(q[d], f32(k[d]), a);
    r.s[i] = rf(rf(a) * kAttnScale);
  }
  k2_ref::eager_softmax(r.s.data(), L, r.p.data());
  for (uint32_t d = 0; d < kHd; ++d) {
    float o = 0.0f;
    for (uint32_t b0 = 0; b0 < L; b0 += blk) {
      float acc = 0.0f;
      for (uint32_t i = b0; i < std::min(L, b0 + blk); ++i)
        acc = std::fma(r.p[i], f32(kv_v[(size_t(i) * kKvHeads + j) * kHd + d]), acc);
      o += acc;
    }
    r.o[d] = o;
  }
  return r;
}
// Decode attention in fp64 over keys [0, L) (flash's tolerance reference, spec 6 K1).
inline std::vector<double> attention(const float* q, const uint16_t* kv_k, const uint16_t* kv_v, uint32_t L, uint32_t j) {
  std::vector<double> sc(L);
  double mx = -INFINITY;
  for (uint32_t i = 0; i < L; ++i) {
    const size_t row = (size_t(i) * kKvHeads + j) * kHd;
    double a = 0;
    for (uint32_t d = 0; d < kHd; ++d) a += double(q[d]) * f32(kv_k[row + d]);
    sc[i] = a * 0.125;
    mx = std::max(mx, sc[i]);
  }
  double z = 0;
  for (double& s : sc) z += (s = std::exp(s - mx));
  std::vector<double> o(kHd, 0.0);
  for (uint32_t i = 0; i < L; ++i) {
    const size_t row = (size_t(i) * kKvHeads + j) * kHd;
    for (uint32_t d = 0; d < kHd; ++d) o[d] += sc[i] / z * f32(kv_v[row + d]);
  }
  return o;
}

// --- SiLU x up (lfm_silu_mul) ------------------------------------------------------------------------------------
// `slices` [S][M][2I] over gate||up interleaved in 16-column blocks; out [M][I].
inline std::vector<uint16_t> silu_mul(const float* slices, uint32_t S, uint32_t M, uint32_t I = kDenseInter) {
  std::vector<uint16_t> out(size_t(M) * I);
  const uint32_t FN = 2 * I;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t i = 0; i < I; ++i) {
      const uint16_t g = rne(slice_sum(slices, S, M, m, FN, moe_ref::gate_col(i)));
      const uint16_t u = rne(slice_sum(slices, S, M, m, FN, moe_ref::up_col(i)));
      out[size_t(m) * I + i] = rne(f32(rne(silu_f32(f32(g)))) * f32(u));
    }
  return out;
}

// --- the head's argmax and the gather ---------------------------------------------------------------------------
// argmax.cl's comparator: strictly greater wins, an exact tie goes to the lower index; ids >= vocab_used never win.
inline uint32_t argmax(const float* logits, uint32_t vocab = kVocab, uint32_t vocab_used = kVocabUsed) {
  float bv = -INFINITY;
  uint32_t bi = 0x7FFFFFFFu;
  for (uint32_t i = 0; i < vocab; ++i) {
    const float v = i < vocab_used ? logits[i] : -INFINITY;
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  return bi;
}
// Row `id` of the bf16 embedding [vocab][K] out of gemv_bf16's tiles {K, vocab}.
inline std::vector<uint16_t> embed_row(const uint16_t* tiles, uint32_t id, uint32_t K = kHidden) {
  std::vector<uint16_t> r(K);
  for (uint32_t k = 0; k < K; ++k) r[k] = tiles[tile_index(K, k, id)];
  return r;
}

}  // namespace lfmref
