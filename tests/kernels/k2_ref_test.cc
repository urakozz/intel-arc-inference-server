// Spec 18b Task 2: the K2 kernels' host reference (tests/kernels/k2_ref.h) against spec 18
// §3's semantics, each by an INDEPENDENT formula in double. Host only. These are the
// Review Focus items of plan 18b:
//
//   1. the two norm traps - two groups of 1280 each with its own mean of squares, and the
//      plain weight (`w · x̂`, not `(1 + w) · x̂`);
//   2. the softplus threshold - `beta·x > 20 -> x` exactly, log1p(exp) below it;
//   4. the router's padding - the 100 real experts on 128 lanes; a padded row is never
//      selected, even when every real score is far below the padded rows' logits;
//   plus the router formula (sigmoid, selection-only bias, x 2.5 after normalising, ties
//   to the lower id), the ASCENDING-ID bf16 combine (order-sensitive inputs that the rank
//   order would round differently), the RoPE chain (rotate_half over all 128 dims), MoVA's
//   value chain and the SiLU x up at K2's split-K.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/k2_ref.h"
#include "kernels/prep_ref.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"

namespace {

using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

std::mt19937 rng(18);

float bf(float v) { return rf(v); }

// 1. the grouped norm, through stage A (prep_ref::res_fold, unchanged) and k2_ref's stage B.
void test_norm() {
  const uint32_t K = 2560, G = 20, groups = 2, GS = 1280;
  std::vector<uint16_t> resid(K);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  // Group 0 small, group 1 ~100x larger: one shared mean would crush group 0.
  for (uint32_t k = 0; k < K; ++k) resid[k] = rne(d(rng) * (k < GS ? 0.01f : 1.0f));
  std::vector<float> w(K);
  for (float& x : w) x = bf(0.25f + 0.5f * (d(rng) + 1.f));   // plain w in [0.25, 1.25]
  std::vector<float> sumsq(G);
  std::vector<uint16_t> r = resid;
  prep_ref::res_fold(nullptr, r.data(), sumsq.data(), 1, K, 0, G);
  CHECK(r == resid);   // SP0 folds nothing
  std::vector<uint16_t> x(K);
  k2_ref::norm_finish(sumsq.data(), r.data(), w.data(), x.data(), 1, K, G, groups);
  // Independent: each group's own mean of squares, in double; x = w * (r * rsqrt).
  double worst = 0;
  for (uint32_t gi = 0; gi < groups; ++gi) {
    double ss = 0;
    for (uint32_t k = gi * GS; k < (gi + 1) * GS; ++k) ss += double(f32(resid[k])) * f32(resid[k]);
    const double rs = 1.0 / std::sqrt(ss / GS + 1e-6);
    for (uint32_t k = gi * GS; k < (gi + 1) * GS; ++k) {
      const double want = double(w[k]) * f32(resid[k]) * rs;
      const double err = std::fabs(f32(x[k]) - want);
      worst = std::max(worst, err / (std::fabs(want) + 1e-30));
      CHECK(err <= std::fabs(want) * 8e-3 + 1e-30);   // one bf16 rounding + fp32 noise
    }
  }
  // Trap 1: a single mean over the row would put group 0's outputs ~70x too small.
  {
    double ss = 0;
    for (uint32_t k = 0; k < K; ++k) ss += double(f32(resid[k])) * f32(resid[k]);
    const double rs_one = 1.0 / std::sqrt(ss / K + 1e-6);
    const double one = double(w[3]) * f32(resid[3]) * rs_one;
    CHECK(std::fabs(f32(x[3])) > 10.0 * std::fabs(one) || resid[3] == 0);
  }
  // Trap 2: a (1 + w) weight - Qwen's bake - would move most outputs; the check above held
  // them to the plain-w formula.
  std::vector<float> w1(K);
  for (uint32_t k = 0; k < K; ++k) w1[k] = 1.0f + w[k];
  std::vector<uint16_t> x1(K);
  k2_ref::norm_finish(sumsq.data(), r.data(), w1.data(), x1.data(), 1, K, G, groups);
  uint32_t differ = 0;
  for (uint32_t k = 0; k < K; ++k) differ += x1[k] != x[k] ? 1u : 0u;
  CHECK(differ > K / 2);
  std::printf("norm: two groups of 1280, own means, plain w - worst %.2e relative to the double "
              "formula; a (1 + w) weight moves %u of %u outputs\n", worst, differ, K);
}

// 2. softplus, both sides of the threshold.
void test_softplus() {
  const double beta = std::log(2.0);
  for (float x : {-30.f, -3.f, 0.f, 0.5f, 7.f, 25.f, 28.85f}) {   // beta x <= 20
    CHECK(x * k2_ref::kSpBeta <= 20.0f);
    const double want = std::log1p(std::exp(beta * x)) / beta;
    CHECK(std::fabs(k2_ref::softplus(x) - want) <= 2e-6 * std::max(1.0, std::fabs(want)));
  }
  for (float x : {28.86f, 29.f, 40.f, 1000.f}) {   // beta x > 20: x itself, bit for bit
    CHECK(x * k2_ref::kSpBeta > 20.0f);
    CHECK(k2_ref::softplus(x) == x);
  }
  // Near the threshold the two branches agree to fp32 (log1p(e^20) - 20 = 2e-9); well below
  // it the formula is far from x: softplus(1) = log2(3) = 1.585, softplus(-3) = 0.1699.
  CHECK(std::fabs(k2_ref::softplus(1.0f) - 1.5849625f) < 1e-6f);
  CHECK(std::fabs(k2_ref::softplus(-3.0f) - 0.169925f) < 1e-6f);
  // The gate chain: o x softplus(g), both rounded to bf16 first.
  const uint16_t got = k2_ref::attn_gate(0.3f, 40.0f);
  CHECK(got == rne(f32(rne(0.3f)) * 40.0f));
  std::puts("softplus: log1p(exp(beta x)) / beta below beta x = 20, x itself above it");
}

// 3/4. the router.
void test_route() {
  const uint32_t E = 100, K = 8, LN = 128;
  std::uniform_real_distribution<float> d(-3.f, 3.f);
  std::vector<float> logits(LN, 0.0f), bias(E, 0.0f);
  for (uint32_t e = 0; e < E; ++e) logits[e] = d(rng);
  for (uint32_t e = 0; e < E; ++e) bias[e] = bf(d(rng) * 0.05f);
  const k2_ref::Route r = k2_ref::route(logits.data(), 1, 0, LN, 0, 1, bias.data(), E, K, 2.5f);
  // Independent: sigmoid of the bf16 logit in double, select by s + bias (descending, ties
  // to the lower id), weights from the UNBIASED s, normalised then x 2.5, ascending id.
  std::vector<double> s(E), sel(E);
  for (uint32_t e = 0; e < E; ++e) {
    s[e] = 1.0 / (1.0 + std::exp(-double(bf(logits[e]))));
    sel[e] = s[e] + bias[e];
  }
  std::vector<uint32_t> ord(E);
  std::iota(ord.begin(), ord.end(), 0u);
  std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return sel[a] > sel[b]; });
  std::vector<uint32_t> top(ord.begin(), ord.begin() + K);
  std::sort(top.begin(), top.end());
  double sum = 0;
  for (uint32_t e : top) sum += s[e];
  double wsum = 0;
  for (uint32_t j = 0; j < K; ++j) {
    CHECK_EQ(r.ids[j], top[j]);   // ascending id
    if (j) CHECK(r.ids[j] > r.ids[j - 1]);
    const double want = s[top[j]] / sum * 2.5;
    CHECK(std::fabs(r.w[j] - want) <= want * 8e-3);   // one bf16 rounding
    CHECK(r.w[j] == rf(r.w[j]));
    wsum += r.w[j];
  }
  CHECK(std::fabs(wsum - 2.5) < 0.05);   // x 2.5 AFTER normalising
  CHECK(r.next == float(s[ord[K]] + bias[ord[K]]) || std::fabs(r.next - sel[ord[K]]) < 1e-6);

  // The bias selects but does not weigh: a large bias on expert 5 puts it in the top-8 with
  // the weight of its own (small) sigmoid.
  {
    std::vector<float> l2 = logits, b2 = bias;
    l2[5] = -6.0f;
    b2[5] = 4.0f;
    const k2_ref::Route r2 = k2_ref::route(l2.data(), 1, 0, LN, 0, 1, b2.data(), E, K, 2.5f);
    const uint32_t* at = std::find(r2.ids, r2.ids + K, 5u);
    CHECK(at != r2.ids + K);
    CHECK(r2.w[at - r2.ids] < 0.01f);   // sigmoid(-6) / Σ x 2.5
  }
  // Ties go to the lower id.
  {
    std::vector<float> l3(LN, -2.0f), b3(E, 0.0f);
    const k2_ref::Route r3 = k2_ref::route(l3.data(), 1, 0, LN, 0, 1, b3.data(), E, K, 2.5f);
    for (uint32_t j = 0; j < K; ++j) CHECK_EQ(r3.ids[j], j);
    CHECK(r3.w[0] == rf(2.5f / 8.0f));
  }
  // Review Focus 4 / the coordinator's padding check: every real score far below the padded
  // rows' logits (which a 128-lane top-k over the whole GEMV row would take), the real
  // biases negative - only real experts are selected.
  {
    std::vector<float> l4(LN, 0.0f), b4(E, -0.9f);
    for (uint32_t e = 0; e < E; ++e) l4[e] = -30.0f - float(e % 7);
    for (uint32_t e = E; e < LN; ++e) l4[e] = 80.0f;
    const k2_ref::Route r4 = k2_ref::route(l4.data(), 1, 0, LN, 0, 1, b4.data(), E, K, 2.5f);
    for (uint32_t j = 0; j < K; ++j) CHECK(r4.ids[j] < E);
    for (uint32_t k = 0; k <= K; ++k) CHECK(r4.rank_ids[k] < E);
    CHECK(std::isfinite(r4.sum) && r4.sum > 0.0f);
  }
  // MoVA's form: two split-K slices of the fused row from column 9216, summed then rounded.
  {
    const uint32_t LN2 = 9280, OFF = 9216, S = 2, E2 = 64, K2 = 4;
    std::vector<float> p(size_t(S) * LN2, 0.0f), b2(E2, 0.0f);
    for (uint32_t e = 0; e < E2; ++e) {
      p[OFF + e] = d(rng);
      p[LN2 + OFF + e] = d(rng);
      b2[e] = common::f16_to_f32(common::f32_to_f16(d(rng) * 0.1f));
    }
    const k2_ref::Route rm = k2_ref::route(p.data(), 1, 0, LN2, OFF, S, b2.data(), E2, K2, 2.5f);
    std::vector<double> sm(E2);
    std::vector<uint32_t> o2(E2);
    for (uint32_t e = 0; e < E2; ++e) {
      sm[e] = 1.0 / (1.0 + std::exp(-double(rf(p[OFF + e] + p[LN2 + OFF + e])))) + b2[e];
      o2[e] = e;
    }
    std::stable_sort(o2.begin(), o2.end(), [&](uint32_t a, uint32_t b) { return sm[a] > sm[b]; });
    std::vector<uint32_t> t2(o2.begin(), o2.begin() + K2);
    std::sort(t2.begin(), t2.end());
    for (uint32_t j = 0; j < K2; ++j) CHECK_EQ(rm.ids[j], t2[j]);
  }
  std::puts("route: sigmoid, selection-only bias, x 2.5 after normalising, ascending id, ties "
            "to the lower id, padded rows never selected, the MoVA slices summed");
}

// The ascending-id bf16 combine: inputs where the order of the bf16 adds matters.
void test_combine() {
  k2_ref::Route r;
  const uint32_t K = 8;
  for (uint32_t j = 0; j < K; ++j) {
    r.ids[j] = j * 11;
    r.w[j] = 1.0f;
  }
  // 256 then seven 1s: in ascending-id order (256 first) each +1 is lost to bf16's 8 bits
  // (256 + 1 rounds to 256); summed small-first in fp32 the 7 survive.
  std::vector<float> down = {256.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f};
  const uint16_t got = k2_ref::combine(down.data(), r, K, rne(0.0f));
  CHECK(f32(got) == 256.0f);
  float fp32_sum = 0;
  for (uint32_t j = 0; j < K; ++j) fp32_sum += down[j];
  CHECK(f32(rne(fp32_sum)) == 264.0f);   // moe.cl's chain would give 264
  // Reversed (as a descending-id order would add them), the 1s accumulate first.
  std::vector<float> rev = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 256.f, 0.f};
  CHECK(f32(k2_ref::combine(rev.data(), r, K, rne(0.0f))) == 264.0f);
  // The shared expert is added ungated, after the routed sum, then the residual - each a
  // bf16 add: rne(256 + 3) = 260 (a tie, to even), rne(1 + 260) = 260 (a tie, to even).
  down[K] = 3.0f;
  CHECK(f32(k2_ref::combine(down.data(), r, K, rne(1.0f))) == 260.0f);
  std::puts("combine: ascending-id bf16 adds (index_add_ into a bf16 tensor), shared ungated, "
            "then the residual");
}

// RoPE: rotate_half over all 128 dims, bf16 per op.
void test_rope() {
  const model::K2Desc& d = model::k2();
  const std::vector<float> tab = loader::k2_rope_table(d, 64);
  const uint32_t hd = 128, qh = 2, kvh = 1, n = 2 * qh * hd + 2 * kvh * hd, S = 2;
  std::vector<float> part(size_t(S) * n);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (float& v : part) v = u(rng);
  // Position 0: cos 1, sin 0 - the output is the linear's bf16 output itself.
  const k2_ref::Prep p0 = k2_ref::attn_prep(part.data(), 1, 0, n, S, tab.data(), qh, kvh, hd, true);
  for (uint32_t i = 0; i < hd; ++i) CHECK(p0.q[i] == rf(part[i] + part[n + i]));
  // Position 37: independent double rotation of the bf16 inputs, rounded per torch op.
  const float* cs = tab.data() + size_t(37) * 2 * 64;
  const k2_ref::Prep p = k2_ref::attn_prep(part.data(), 1, 0, n, S, cs, qh, kvh, hd, true);
  for (uint32_t h = 0; h < qh + kvh; ++h) {
    const size_t base = h < qh ? size_t(h) * hd : size_t(qh) * hd + size_t(h - qh) * hd;
    for (uint32_t i = 0; i < hd; ++i) {
      const float x = rf(part[base + i] + part[n + base + i]);
      const uint32_t pi = i < 64 ? i + 64 : i - 64;
      const float xp = rf(part[base + pi] + part[n + base + pi]);
      const float c = cs[i % 64], s = cs[64 + i % 64];
      const float rot = i < 64 ? -xp : xp;   // rotate_half: cat(-x2, x1)
      const float want = rf(rf(x * c) + rf(rot * s));
      const float got = h < qh ? p.q[size_t(h) * hd + i] : f32(p.k[i]);
      CHECK(got == want);
    }
  }
  // The gate and v columns.
  CHECK(p.gate[5] == rf(part[size_t(qh) * hd + kvh * hd + 5] + part[n + size_t(qh) * hd + kvh * hd + 5]));
  const uint32_t voff = 2 * qh * hd + kvh * hd;
  CHECK(f32(p.v[9]) == rf(part[voff + 9] + part[n + voff + 9]));
  std::puts("rope: rotate_half pairs (i, i + 64) over all 128 dims, bf16 per op, cos / sin from "
            "the fp32-step table");
}

// MoVA value chain on layout-1 blocks against a double GEMV of the dequantised weights.
void test_mova() {
  const uint32_t D = 128, N = 32, E = 6, K = 3, ks = 2;
  const size_t blk = size_t(N / 16) * (D / 64) * 136;
  std::vector<uint32_t> w(blk * E);
  for (size_t b = 0; b < E; ++b)
    for (size_t t = 0; t < blk / 136; ++t) {
      uint32_t* tile = w.data() + b * blk + t * 136;
      for (uint32_t i = 0; i < 128; ++i) tile[i] = rng();
      for (uint32_t i = 0; i < 8; ++i)
        tile[128 + i] = uint32_t(common::f32_to_f16(0.01f)) | (uint32_t(common::f32_to_f16(0.02f)) << 16);
    }
  std::vector<uint16_t> x(D);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (uint16_t& v : x) v = rne(u(rng));
  k2_ref::Route r;
  r.ids[0] = 1;
  r.ids[1] = 4;
  r.ids[2] = 5;
  r.w[0] = rf(0.5f);
  r.w[1] = rf(1.25f);
  r.w[2] = rf(0.75f);
  const std::vector<uint16_t> v = k2_ref::mova_value(r, x.data(), w.data(), D, N, K, ks);
  for (uint32_t n = 0; n < N; ++n) {
    double want = 0;
    for (uint32_t j = 0; j < K; ++j) {
      const uint32_t* b = w.data() + r.ids[j] * blk;
      double dot = 0;
      for (uint32_t k = 0; k < D; ++k) {
        const uint32_t* tile = b + (size_t(n / 16) * (D / 64) + k / 64) * 136;
        const uint32_t word = tile[((k % 64) / 8) * 16 + n % 16];
        const int q = int((word >> (4 * (k % 8))) & 15u) - 8;
        const double sc = (n % 2 == 0) ? 0.01 : 0.02;
        dot += double(q) * sc * f32(x[k]);
      }
      want += dot / (1.0 + std::exp(-dot)) * r.w[j];
    }
    CHECK(std::fabs(f32(v[n]) - want) <= 0.03 * std::max(1.0, std::fabs(want)));
  }
  std::puts("mova: value experts by id from flat layout-1 blocks, SiLU, weighted ascending-id "
            "bf16 combine");
}

void test_silu() {
  const uint32_t I = 6144, S = 4;
  std::vector<float> part(size_t(S) * 2 * I);
  std::uniform_real_distribution<float> u(-2.f, 2.f);
  for (float& v : part) v = u(rng);
  std::vector<uint16_t> x(I);
  k2_ref::silu_mul(part.data(), x.data(), 1, I, S);
  for (uint32_t k : {0u, 15u, 16u, 6143u}) {
    const size_t gflat = size_t(k / 16) * 32 + k % 16;
    double g = 0, up = 0;
    for (uint32_t s = 0; s < S; ++s) {
      g += part[s * 2 * I + gflat];
      up += part[s * 2 * I + gflat + 16];
    }
    const double want = g / (1.0 + std::exp(-g)) * up;
    CHECK(std::fabs(f32(x[k]) - want) <= 0.03 * std::max(1.0, std::fabs(want)));
  }
  std::puts("silu: gate||up interleave16 at I 6144, S 4");
}

}  // namespace

int main() {
  test_norm();
  test_softplus();
  test_route();
  test_combine();
  test_rope();
  test_mova();
  test_silu();
  std::puts("k2_ref_test OK");
  return 0;
}
