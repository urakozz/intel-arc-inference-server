#pragma once
// CPU references for spec 25c's Nemotron 3.5 Bolt decode kernels (src/kernels/bolt/nb_*.cl) - the same op chains in the
// same orders, every line with a twin in a kernel; EDIT THE TWO TOGETHER. The chains are spec 25 §2's reading 1 as 25a
// pinned them (tools/oracle/bolt_ref.py's restated ops, tools/oracle/bolt_mamba_ops.py's training-kernel functions - the
// orders spec 25 §13 records); tests/kernels/bolt_ref_test.cc holds them to those functions' own outputs
// (tests/kernels/bolt_fixture.h, tools/oracle/bolt_fixture.py) and to the semantics' properties (plan 25c Review Focus
// 1-3). Used by the kernel tests (Tasks 3, 5) and by 25d / 25e.
//
//   res_norm()    nb_res_norm: fold the pending split-K slices into the residual - one source (an out_proj / o_proj:
//                 h = rne(h + rne(Σ s1))), two (a MoE block: h = rne(h + rne(rne(Σ fc2) + rne(Σ shared_down)))) or none
//                 (`_NF`) - Σ r² by the HALVING tree over the 4096 squares (v[:n/2] + v[n/2:], 25a's tree), rstd = 1 /
//                 sqrt(Σ / 4096 + 1e-5) (never rsqrt), x = rne((r · rstd) · w) - ONE rounding (spec 25 §2.1)
//   conv_step()   nb_conv: raw xBC = rne(Σ_s in_proj slices) at 8192 + c into ring slot (pos + m) % 16; acc = bias, then
//                 + w_j · x_j for the taps OLDEST FIRST (positions < 0 read 0; the ring is never cleared), xc =
//                 rne(silu(acc)), silu(v) = v / (1 + exp(-v)) - ONE rounding (25a: the bias seeds the accumulator)
//   ssm_step()    nb_ssm: dt = softplus(f32(rne(Σ_s slices at 18432 + h)) + dt_bias) (threshold 20, NO clamp); dA =
//                 exp(A · dt); dB = B · dt; S = S · dA + dB · x (products and adds separately rounded, no fma); y =
//                 rne(tree(S · C) + x · D) with the halving tree over the 128 state entries - mamba_ssm's order (25a);
//                 the bf16-state arm (Decision 2) rounds S to bf16 after every row, y from the fp32 update
//   gnorm()       nb_gnorm: v = f32(y) · silu_gate(f32(rne(Σ_s z slices))), silu_gate(z) = z · (1 / (1 + exp(-z)));
//                 per group of 1024 ms = tree(v²) / 1024 (halving tree), rstd = 1 / sqrt(ms + 1e-5), out = rne((v ·
//                 rstd) · w) (rmsnorm_fn, norm_before_gate false)
//   router()      nb_router (and the _R arm): fp32 logits of E bf16 rows over gemv_bf16's tiles - per row 8 spans of 512
//                 consecutive k (ascending, exact bf16 products), then the 8 partials by the stride 4 / 2 / 1 tree; never
//                 rounded
//   route()       nb_route: s = sigmoid(l) (torch's vectorised form: 1 / (exp(0 - l) + 1)); sel = s + bias; rank with
//                 exact ties to the LOWER id; the selected set rank < 9; the route row: ids ASCENDING, w = (s / (Σ +
//                 1e-20)) · 5 with Σ the nine s added in ascending id (25a: torch.sum's order has no name - the twin's
//                 own fixed order), sel of ranks 0..9 (rank 9 the near-tie margin), Σ, the smallest selected sel
//   sh_act()      nb_sh_act: lat = rne(Σ_s fc1||shared_up slices [0, 1024)); sh = relu2(rne(Σ_s [1024, 6400))),
//                 relu2(u) = rne(relu(u)²) on bf16
//   moe_up() / moe_down() / combine()   nb_moe_up / nb_moe_down: per slot (ascending id) u = rne(up GEMV over the
//                 latent), a = relu2(u); d = rne(down GEMV); acc = 0, acc = acc + (f32(d_k) · w_k) ascending - the
//                 product rounded to fp32, then the add, ONE rounding (spec 25 §2.4, Decision 10)
//   attn_prep()   nb_attn_prep: q / k / v = rne(Σ_s q||k||v slices), no norm, no position encoding (NoPE); q as fp32 of
//                 the bf16 value (mg_attn's input), k / v rows at pos + m
//   attention_eager() / attention_fp64()   Muse's twin (tests/kernels/muse_ref.h, Decision 15: the engine binds Muse's
//                 mg_attn / mg_attn_eager at Bolt's shape) on a full NoPE layer with no gate (a ones gate is exact)
//   argmax()      argmax.cl's comparator at 131072 = vocab_used: strictly greater wins, a tie to the lower id
//   embed_row()   nb_embed_gather: the token's row out of gemv_bf16's tiles
//
// The transcendentals are the kernels' own on both sides, so device and twin are bitwise everywhere: exp is exp_t
// (src/common/exp_torch.h, Sleef's expf u10 - the loader's A uses it too), log1p is log1p_t (fdlibm's s_log1pf in plain
// fp32 operations). Against the reference (torch.exp: MKL's vsExp; torch.log1p) they differ by ulps; the fixture test
// counts them.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/repack.h"
#include "kernels/moe_ref.h"    // f32 / rne / rf, the layout-1 tile_dot / split_dot
#include "kernels/muse_ref.h"   // the attention twin (Muse's mg_attn at GQA 16 / head_dim 128, Decision 15)

namespace boltref {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;
using moe_ref::split_dot;

constexpr float kEps = 1e-5f;
constexpr uint32_t kHidden = 4096, kHeads = 128, kHeadDim = 64, kGroups = 8, kState = 128, kConvTaps = 4;
constexpr uint32_t kInner = 8192, kXbc = 10240, kInProjN = 18560, kXbcOff = 8192, kDtOff = 18432;
constexpr uint32_t kBOff = 8192, kCOff = 9216;   // within xc: x [0, 8192), B [8192, 9216), C [9216, 10240)
constexpr uint32_t kQHeads = 32, kKvHeads = 2, kHd = 128, kQkvN = 4608;
constexpr uint32_t kLatent = 1024, kShared = 5376, kFc1N = 6400, kTopK = 9, kVocab = 131072;
constexpr uint32_t kRing = 16, kGnormGroup = 1024, kRouterLanes = 8;
constexpr float kRoutedScale = 5.0f;
// The route row (nb_moe.cl R_*, kernels::bolt::route): 48 u32 words per (layer, token).
constexpr uint32_t kWords = 48, kIds = 0, kW = 9, kSel = 18, kSum = 28, kSelMin = 29;
constexpr uint32_t kMoeUpKs = 4, kMoeDnKs = 2;   // kernels::bolt's (PROVISIONAL)

// --- the fixture's inputs and hex arrays (tools/oracle/bolt_fixture.py) ------------------------------------------------
inline uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}
inline uint32_t fixture_hash(uint32_t t, uint32_t i, uint32_t seed) { return lowbias32(lowbias32(seed * 3 + t) ^ i); }
// coarse: (h % 17 - 8) x step (exact sums in any order); fine: ((h >> 8) x 2^-23 - 1) x amp, rounded to bf16.
inline float fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float step = 0.125f, float amp = 1.0f) {
  const uint32_t h = fixture_hash(t, i, seed);
  if (coarse) return float(int(h % 17) - 8) * step;
  return rf((float(h >> 8) * 0x1p-23f - 1.0f) * amp);
}
// fp32, not rounded to bf16 (the SSM state, the router bias).
inline float fixture_f32(uint32_t t, uint32_t i, uint32_t seed, float amp = 1.0f) {
  return (float(fixture_hash(t, i, seed) >> 8) * 0x1p-23f - 1.0f) * amp;
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

// --- the transcendentals (the kernels' nb_exp / nb_log1p, line for line) ------------------------------------------------
inline float exp_t(float x) { return common::exp_torch(x); }
// fdlibm's s_log1pf.c (FreeBSD) in plain fp32 operations - every operation an IEEE fp32 op, so the kernel's copy gives the
// same bits. Defined for x > -1 finite (the softplus feeds it exp(v) >= 0); bolt_ref_test holds it within 1 ulp of libm.
inline float log1p_t(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f;
  const float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f, Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f,
              Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f, Lp7 = 1.4798198640e-01f;
  int32_t hx;
  std::memcpy(&hx, &x, 4);
  const int32_t ax = hx & 0x7fffffff;
  int32_t k = 1, hu = 0;
  float f = 0.0f, c = 0.0f;
  if (hx < 0x3ed413d0) {                  // 1 + x < sqrt(2)+
    if (ax >= 0x3f800000) return x == -1.0f ? -INFINITY : NAN;
    if (ax < 0x38000000) {                // |x| < 2^-15
      if (ax < 0x33800000) return x;      // |x| < 2^-24
      return x - (x * x) * 0.5f;
    }
    if (hx > 0 || hx <= int32_t(0xbe95f619)) {   // sqrt(2)/2- <= 1 + x < sqrt(2)+
      k = 0;
      f = x;
      hu = 1;
    }
  }
  if (hx >= 0x7f800000) return x + x;
  if (k != 0) {
    float u;
    if (hx < 0x5a000000) {
      u = 1.0f + x;
      std::memcpy(&hu, &u, 4);
      k = (hu >> 23) - 127;
      c = k > 0 ? 1.0f - (u - x) : x - (u - 1.0f);   // the correction term
      c = c / u;
    } else {
      u = x;
      std::memcpy(&hu, &u, 4);
      k = (hu >> 23) - 127;
      c = 0.0f;
    }
    hu &= 0x007fffff;
    if (hu < 0x3504f4) {                  // u < sqrt(2): normalise u
      const int32_t b = hu | 0x3f800000;
      std::memcpy(&u, &b, 4);
    } else {                              // normalise u / 2
      k += 1;
      const int32_t b = hu | 0x3f000000;
      std::memcpy(&u, &b, 4);
      hu = (0x00800000 - hu) >> 2;
    }
    f = u - 1.0f;
  }
  const float hfsq = (0.5f * f) * f;
  const float fk = float(k);
  if (hu == 0) {                          // |f| < 2^-20
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      c = c + fk * ln2_lo;
      return fk * ln2_hi + c;
    }
    const float R = hfsq * (1.0f - 0.6666666666666666f * f);
    if (k == 0) return f - R;
    return fk * ln2_hi - ((R - (fk * ln2_lo + c)) - f);
  }
  const float s = f / (2.0f + f);
  const float z = s * s;
  const float R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
  if (k == 0) return f - (hfsq - s * (hfsq + R));
  return fk * ln2_hi - ((hfsq - (s * (hfsq + R) + (fk * ln2_lo + c))) - f);
}
// torch's vectorised fp32 sigmoid (at::native sigmoid's Vectorized path: 0 - x, exp, + 1, reciprocal).
inline float sigmoid_t(float x) { return 1.0f / (exp_t(0.0f - x) + 1.0f); }
// The conv's SiLU (causal_conv1d's form) and the gated norm's (layernorm_gated's z · sigmoid(z)).
inline float silu_conv(float v) { return v / (1.0f + exp_t(-v)); }
inline float silu_gate(float z) { return z * (1.0f / (1.0f + exp_t(-z))); }
// mamba_ssm's softplus: v <= 20 ? log1p(exp(v)) : v. No clamp (Decision 1).
inline float softplus_t(float v) { return v <= 20.0f ? log1p_t(exp_t(v)) : v; }

// The halving tree over n (a power of two) fp32 values: v[i] += v[i + n/2] for i < n/2, repeated (bolt_mamba_ops.tree_sum).
inline float tree_halving(std::vector<float> v) {
  for (size_t n = v.size(); n > 1; n /= 2)
    for (size_t i = 0; i < n / 2; ++i) v[i] = v[i] + v[i + n / 2];
  return v[0];
}

// The split-K sum of column `col` of row m over S slices [S][M][N]: 0.0f + slice 0 + slice 1 ... (every consumer of a
// GEMV's slices sums them this way).
inline float slice_sum(const float* slices, uint32_t S, uint32_t M, uint32_t m, uint32_t N, size_t col) {
  float v = 0.0f;
  for (uint32_t s = 0; s < S; ++s) v += slices[(size_t(s) * M + m) * N + col];
  return v;
}

// --- the norm (nb_res_norm) --------------------------------------------------------------------------------------
// Row m's folded value at k: S1 = 0 none (`_NF`); S2 = 0 one source; else the MoE block's two (spec 25 §2.4's three
// roundings: rne(Σ fc2), rne(Σ shared_down), their bf16 sum, then the residual add).
inline uint16_t fold(const uint16_t* h_in, const float* s1, uint32_t S1, const float* s2, uint32_t S2, uint32_t M,
                     uint32_t m, uint32_t k, uint32_t K = kHidden) {
  const uint16_t h = h_in[size_t(m) * K + k];
  if (!S1) return h;
  const uint16_t a = rne(slice_sum(s1, S1, M, m, K, k));
  if (!S2) return rne(f32(h) + f32(a));
  const uint16_t b = rne(slice_sum(s2, S2, M, m, K, k));
  return rne(f32(h) + f32(rne(f32(a) + f32(b))));
}
inline float norm_sumsq(const uint16_t* r, uint32_t K = kHidden) {
  std::vector<float> sq(K);
  for (uint32_t k = 0; k < K; ++k) sq[k] = f32(r[k]) * f32(r[k]);
  return tree_halving(std::move(sq));
}
inline float rstd_of(float sumsq, uint32_t K = kHidden) { return 1.0f / std::sqrt(sumsq / float(K) + kEps); }
inline uint16_t norm_elem(float r, float rstd, float w) { return rne((r * rstd) * w); }
struct NormOut {
  std::vector<uint16_t> h_out, x;   // [M][K] each; h_out == h_in under `_NF`
};
inline NormOut res_norm(const uint16_t* h_in, const float* s1, uint32_t S1, const float* s2, uint32_t S2, uint32_t M,
                        const float* w, uint32_t K = kHidden) {
  NormOut o;
  o.h_out.resize(size_t(M) * K);
  o.x.resize(size_t(M) * K);
  for (uint32_t m = 0; m < M; ++m) {
    uint16_t* r = o.h_out.data() + size_t(m) * K;
    for (uint32_t k = 0; k < K; ++k) r[k] = fold(h_in, s1, S1, s2, S2, M, m, k, K);
    const float rstd = rstd_of(norm_sumsq(r, K), K);
    for (uint32_t k = 0; k < K; ++k) o.x[size_t(m) * K + k] = norm_elem(f32(r[k]), rstd, w[k]);
  }
  return o;
}

// --- the conv (nb_conv) ------------------------------------------------------------------------------------------
// `slices` [S][M][18560] (in_proj's), `ring` bf16 [16][10240] (the raw xBC of position p at slot p % 16), `w` fp32
// [10240][4] (tap 0 the OLDEST position), `b` fp32 [10240], the rows' positions pos .. pos + M - 1 (M <= 13). Writes the
// M raw columns into `ring` and returns xc [M][10240].
inline std::vector<uint16_t> conv_step(const float* slices, uint32_t S, uint32_t M, uint16_t* ring, const float* w,
                                       const float* b, uint32_t pos) {
  std::vector<uint16_t> xc(size_t(M) * kXbc);
  for (uint32_t c = 0; c < kXbc; ++c) {
    float hist[kConvTaps - 1];
    for (uint32_t j = 0; j < kConvTaps - 1; ++j) {   // positions pos - 3, pos - 2, pos - 1
      const int p = int(pos) - int(kConvTaps - 1) + int(j);
      hist[j] = p < 0 ? 0.0f : f32(ring[size_t(uint32_t(p) % kRing) * kXbc + c]);
    }
    for (uint32_t m = 0; m < M; ++m) {
      const uint16_t raw = rne(slice_sum(slices, S, M, m, kInProjN, kXbcOff + c));
      ring[size_t((pos + m) % kRing) * kXbc + c] = raw;
      float acc = b[c];
      for (uint32_t j = 0; j < kConvTaps - 1; ++j) acc = acc + w[size_t(c) * kConvTaps + j] * hist[j];
      acc = acc + w[size_t(c) * kConvTaps + kConvTaps - 1] * f32(raw);
      xc[size_t(m) * kXbc + c] = rne(silu_conv(acc));
      for (uint32_t j = 0; j + 1 < kConvTaps - 1; ++j) hist[j] = hist[j + 1];
      hist[kConvTaps - 2] = f32(raw);
    }
  }
  return xc;
}

// --- the SSM update (nb_ssm) -------------------------------------------------------------------------------------
// `slices` [S][M][18560] (dt at 18432 + h), `xc` [M][10240], A / D / dt_bias fp32 [128], `state` [128][64][128] fp32 (or,
// `state16` non-null, bf16 - the arm: S read widened, rounded back after every row). Returns y [M][8192].
inline std::vector<uint16_t> ssm_step(const float* slices, uint32_t S, uint32_t M, const uint16_t* xc, const float* A,
                                      const float* D, const float* dt_bias, float* state, uint16_t* state16 = nullptr) {
  std::vector<uint16_t> y(size_t(M) * kInner);
  std::vector<float> st(kState), prod(kState);
  for (uint32_t h = 0; h < kHeads; ++h) {
    const uint32_t g = h / (kHeads / kGroups);
    for (uint32_t p = 0; p < kHeadDim; ++p) {
      const size_t row = (size_t(h) * kHeadDim + p) * kState;
      for (uint32_t n = 0; n < kState; ++n) st[n] = state16 ? f32(state16[row + n]) : state[row + n];
      for (uint32_t m = 0; m < M; ++m) {
        const float dt = softplus_t(f32(rne(slice_sum(slices, S, M, m, kInProjN, kDtOff + h))) + dt_bias[h]);
        const float dA = exp_t(A[h] * dt);
        const uint16_t* xr = xc + size_t(m) * kXbc;
        const float x = f32(xr[size_t(h) * kHeadDim + p]);
        for (uint32_t n = 0; n < kState; ++n) {
          const float dB = f32(xr[kBOff + g * kState + n]) * dt;
          st[n] = st[n] * dA + dB * x;
          prod[n] = st[n] * f32(xr[kCOff + g * kState + n]);
        }
        y[size_t(m) * kInner + size_t(h) * kHeadDim + p] = rne(tree_halving(prod) + x * D[h]);
        if (state16)
          for (uint32_t n = 0; n < kState; ++n) st[n] = rf(st[n]);
      }
      for (uint32_t n = 0; n < kState; ++n) {
        if (state16)
          state16[row + n] = rne(st[n]);
        else
          state[row + n] = st[n];
      }
    }
  }
  return y;
}

// --- the gated norm (nb_gnorm) -----------------------------------------------------------------------------------
// `slices` [S][M][18560] (z at [0, 8192)), y [M][8192], w fp32 [8192]. Returns out [M][8192].
inline std::vector<uint16_t> gnorm(const float* slices, uint32_t S, uint32_t M, const uint16_t* y, const float* w) {
  std::vector<uint16_t> out(size_t(M) * kInner);
  std::vector<float> v(kGnormGroup), sq(kGnormGroup);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t g = 0; g < kInner / kGnormGroup; ++g) {
      for (uint32_t i = 0; i < kGnormGroup; ++i) {
        const uint32_t c = g * kGnormGroup + i;
        v[i] = f32(y[size_t(m) * kInner + c]) * silu_gate(f32(rne(slice_sum(slices, S, M, m, kInProjN, c))));
        sq[i] = v[i] * v[i];
      }
      const float ms = tree_halving(sq) / float(kGnormGroup);
      const float rstd = 1.0f / std::sqrt(ms + kEps);
      for (uint32_t i = 0; i < kGnormGroup; ++i) out[size_t(m) * kInner + g * kGnormGroup + i] = rne((v[i] * rstd) * w[g * kGnormGroup + i]);
    }
  return out;
}

// --- the router (nb_router, the _R arm) ---------------------------------------------------------------------------
// gemv_bf16's tiles {K, N}: element (k, n) at ((n / 16) x K/8 + k / 8) x 8 + k % 8) x 16 + n % 16.
inline size_t tile_index(uint32_t K, uint32_t k, uint32_t n) {
  return ((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16;
}
// fp32 logits [E] of one row x [K] (bf16) over the router's tiles {K, E}.
inline std::vector<float> router(const uint16_t* x, const uint16_t* tiles, uint32_t E, uint32_t K = kHidden) {
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

// --- the route (nb_route) ----------------------------------------------------------------------------------------
struct Route {
  uint32_t ids[kTopK] = {};     // ASCENDING - the combine's order
  float w[kTopK] = {};          // (s / (Σ + 1e-20)) x 5 at ids[j], fp32, never rounded
  float s[kTopK] = {};          // sigmoid(l) at ids[j]
  float sel[kTopK + 1] = {};    // sel of ranks 0 .. 9 (rank 9: the first expert not taken - the near-tie margin)
  float sum = 0, sel_min = 0;   // Σ of the nine s (ascending id), the smallest selected sel (rank 8)
  uint32_t row[kWords] = {};    // the route row as nb_route writes it
};
inline Route route(const float* logits, const float* bias, uint32_t E) {
  std::vector<float> s(E), sel(E);
  for (uint32_t e = 0; e < E; ++e) {
    s[e] = sigmoid_t(logits[e]);
    sel[e] = s[e] + bias[e];
  }
  Route r;
  uint32_t top_id[kTopK + 1] = {};
  for (uint32_t k = 0; k <= kTopK; ++k) r.sel[k] = -INFINITY;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < E; ++j) rank += (sel[j] > sel[e] || (sel[j] == sel[e] && j < e)) ? 1u : 0u;
    if (rank <= kTopK) {
      top_id[rank] = e;
      r.sel[rank] = sel[e];
    }
  }
  std::vector<uint32_t> ids(top_id, top_id + kTopK);
  std::sort(ids.begin(), ids.end());
  float sum = 0.0f;
  for (uint32_t k = 0; k < kTopK; ++k) {
    r.ids[k] = ids[k];
    r.s[k] = s[ids[k]];
    sum = sum + r.s[k];   // ascending id, sequential, fp32
  }
  r.sum = sum;
  for (uint32_t k = 0; k < kTopK; ++k) r.w[k] = (r.s[k] / (sum + 1e-20f)) * kRoutedScale;
  r.sel_min = r.sel[kTopK - 1];
  for (uint32_t k = 0; k < kTopK; ++k) {
    r.row[kIds + k] = r.ids[k];
    r.row[kW + k] = as_u32(r.w[k]);
  }
  for (uint32_t k = 0; k <= kTopK; ++k) r.row[kSel + k] = as_u32(r.sel[k]);
  r.row[kSum] = as_u32(r.sum);
  r.row[kSelMin] = as_u32(r.sel_min);
  return r;
}
// The selection gap at the cut (25a's route.gap): sel of rank 8 - sel of rank 9.
inline float gap(const Route& r) { return r.sel[kTopK - 1] - r.sel[kTopK]; }

// --- the latent MoE (nb_sh_act, nb_moe_up, nb_moe_down) --------------------------------------------------------------
inline uint16_t relu2(uint16_t u) {
  const float v = f32(u) > 0.0f ? f32(u) : 0.0f;
  return rne(v * v);
}
// `slices` [S][M][6400] (fc1 || shared_up): lat [M][1024], sh [M][5376].
inline void sh_act(const float* slices, uint32_t S, uint32_t M, uint16_t* lat, uint16_t* sh) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t c = 0; c < kLatent; ++c) lat[size_t(m) * kLatent + c] = rne(slice_sum(slices, S, M, m, kFc1N, c));
    for (uint32_t i = 0; i < kShared; ++i)
      sh[size_t(m) * kShared + i] = relu2(rne(slice_sum(slices, S, M, m, kFc1N, kLatent + i)));
  }
}
// One expert block (loader/bolt_layout.h block(I)): up (K 1024, N I) then down (K I, N 1024), both layout 1.
inline size_t expert_up_words(uint32_t I) { return size_t(I / 16) * (kLatent / 64) * 136; }
inline size_t expert_block_words(uint32_t I) { return 2 * expert_up_words(I); }
// a [9][I] for the route's slots (ascending id): expert ids[j]'s up block at blocks + id x block(I).
inline std::vector<uint16_t> moe_up(const uint32_t* ids, const uint16_t* lat, const uint32_t* blocks, uint32_t I,
                                    uint32_t up_ks = kMoeUpKs) {
  std::vector<uint16_t> a(size_t(kTopK) * I);
  for (uint32_t j = 0; j < kTopK; ++j) {
    const uint32_t* up = blocks + ids[j] * expert_block_words(I);
    for (uint32_t i = 0; i < I; ++i) a[size_t(j) * I + i] = relu2(rne(split_dot(up, kLatent, i, lat, up_ks)));
  }
  return a;
}
// The combine of one latent column: d[9] the slots' bf16 down outputs (ascending id), w[9] their weights.
inline uint16_t combine(const uint16_t* d, const float* w) {
  float acc = 0.0f;
  for (uint32_t k = 0; k < kTopK; ++k) {
    const float t = f32(d[k]) * w[k];
    acc = acc + t;
  }
  return rne(acc);
}
struct MoeOut {
  std::vector<uint16_t> a, d, r;   // up's relu2 [9][I], the down outputs [9][1024], the combine [1024]
};
inline MoeOut moe(const Route& rt, const uint16_t* lat, const uint32_t* blocks, uint32_t I, uint32_t up_ks = kMoeUpKs,
                  uint32_t dn_ks = kMoeDnKs) {
  MoeOut o;
  o.a = moe_up(rt.ids, lat, blocks, I, up_ks);
  o.d.resize(size_t(kTopK) * kLatent);
  o.r.resize(kLatent);
  for (uint32_t n = 0; n < kLatent; ++n) {
    uint16_t d[kTopK];
    for (uint32_t j = 0; j < kTopK; ++j) {
      const uint32_t* dn = blocks + rt.ids[j] * expert_block_words(I) + expert_up_words(I);
      d[j] = rne(split_dot(dn, I, n, o.a.data() + size_t(j) * I, dn_ks));
      o.d[size_t(j) * kLatent + n] = d[j];
    }
    o.r[n] = combine(d, rt.w);
  }
  return o;
}

// --- the fixture's int4 linears (tools/oracle/bolt_fixture.py int4_weight) ----------------------------------------------
// Linear t as GPTQ qweight [K/8][N] + f16 scales [K/64][N]: nibble q(k, n) = hash(t, k x N + n, 8) % 16, scale (g, n) =
// (-1)^(hash(t, g x N + n, 9) & 1) x (1 + ((hash >> 4) % 8) / 8) x 2^-shift - SIGNED, every (q - 8) x scale a bf16 value.
struct Gptq {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;
};
inline Gptq gptq_of(uint32_t t, uint32_t K, uint32_t N, int shift) {
  Gptq g;
  g.K = K;
  g.N = N;
  g.qweight.assign(size_t(K / 8) * N, 0u);
  g.scales.resize(size_t(K / 64) * N);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t q = fixture_hash(t, k * N + n, 8) % 16;
      g.qweight[size_t(k / 8) * N + n] |= q << (4 * (k % 8));
    }
  for (uint32_t gg = 0; gg < K / 64; ++gg)
    for (uint32_t n = 0; n < N; ++n) {
      const uint32_t hs = fixture_hash(t, gg * N + n, 9);
      const float s = (1.0f + float((hs >> 4) % 8) / 8.0f) * std::ldexp(1.0f, -shift) * ((hs & 1) ? -1.0f : 1.0f);
      g.scales[size_t(gg) * N + n] = common::f32_to_f16(s);
    }
  return g;
}
// gemv.cl's order on GPTQ weights (layout-independent: the same nibbles, scales and sums), S split-K slices: slice s over
// k-groups [s G/S, (s + 1) G/S) - per group 8 dot8 (8 nibbles each, ascending) into a group sum, then acc += gsum x scale.
// Returns [S][N] for one row x [K]. (On the card gemv.cl may contract acc += gsum x scale into an fma: the int4 GEMVs are
// held to this by tolerance, never bitwise.)
inline std::vector<float> gemv_int4(const Gptq& w, const uint16_t* x, uint32_t S = 1) {
  const uint32_t G = w.K / 64, per = G / S;
  std::vector<float> out(size_t(S) * w.N);
  for (uint32_t s = 0; s < S; ++s)
    for (uint32_t n = 0; n < w.N; ++n) {
      float acc = 0.0f;
      for (uint32_t g = s * per; g < (s + 1) * per; ++g) {
        float gacc = 0.0f;
        for (uint32_t j = 0; j < 8; ++j) {
          const uint32_t word = w.qweight[size_t(g * 8 + j) * w.N + n];
          float a = 0.0f;
          for (uint32_t i = 0; i < 8; ++i) a += float(int((word >> (4 * i)) & 0xFu) - 8) * f32(x[g * 64 + j * 8 + i]);
          gacc += a;
        }
        acc += gacc * common::f16_to_f32(w.scales[size_t(g) * w.N + n]);
      }
      out[size_t(s) * w.N + n] = acc;
    }
  return out;
}
// One expert's device block (up then down, both layout 1) from its two GPTQ linears.
inline void expert_block(const Gptq& up, const Gptq& dn, uint32_t* out) {
  common::repack_int4_layout1(up.qweight.data(), up.scales.data(), up.K, up.N, out);
  common::repack_int4_layout1(dn.qweight.data(), dn.scales.data(), dn.K, dn.N, out + expert_up_words(up.N));
}

// --- the attention prep (nb_attn_prep) ----------------------------------------------------------------------------
struct Prep {
  std::vector<float> q;        // [32][128] (bf16 values: mg_attn reads fp32)
  std::vector<uint16_t> k, v;  // [2][128]
};
// Row m of the fused q||k||v GEMV's [S][M][4608] fp32 slices: no norm, no position encoding.
inline Prep attn_prep(const float* slices, uint32_t S, uint32_t M, uint32_t m) {
  Prep p;
  p.q.resize(size_t(kQHeads) * kHd);
  p.k.resize(size_t(kKvHeads) * kHd);
  p.v.resize(size_t(kKvHeads) * kHd);
  for (uint32_t i = 0; i < kQHeads * kHd; ++i) p.q[i] = rf(slice_sum(slices, S, M, m, kQkvN, i));
  for (uint32_t i = 0; i < kKvHeads * kHd; ++i) {
    p.k[i] = rne(slice_sum(slices, S, M, m, kQkvN, kQHeads * kHd + i));
    p.v[i] = rne(slice_sum(slices, S, M, m, kQkvN, (kQHeads + kKvHeads) * kHd + i));
  }
  return p;
}

// --- attention (Muse's twin on a full NoPE layer, no gate) ------------------------------------------------------------
// q [32][128] bf16, caches [rows][2][128] bf16, the query at `pos` (keys 0 .. pos): out [32][128] = rne(o).
inline void attention_eager(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, uint16_t* out) {
  museref::attention_eager(q, kc, vc, pos, false, 0, nullptr, out);
}
inline museref::EagerHead attention_eager_head(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos,
                                               uint32_t h) {
  return museref::attention_eager_head(q, kc, vc, pos, false, h, 0);
}
inline void attention_fp64(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, double* out) {
  museref::attention_fp64(q, kc, vc, pos, false, 0, nullptr, out);
}

// --- the head's argmax and the gather ---------------------------------------------------------------------------
inline uint32_t argmax(const float* logits, uint32_t vocab = kVocab) {
  float bv = -INFINITY;
  uint32_t bi = 0x7FFFFFFFu;
  for (uint32_t i = 0; i < vocab; ++i) {
    const float v = logits[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  return bi;
}
inline std::vector<uint16_t> embed_row(const uint16_t* tiles, uint32_t id, uint32_t K = kHidden) {
  std::vector<uint16_t> r(K);
  for (uint32_t k = 0; k < K; ++k) r[k] = tiles[tile_index(K, k, id)];
  return r;
}

}  // namespace boltref
