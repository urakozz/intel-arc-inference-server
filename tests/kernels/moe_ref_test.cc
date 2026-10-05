// Spec 15c: the MoE host reference (tests/kernels/moe_ref.h) against the spec's
// formula, on the host - the reference the device test (moe_test) holds the kernels to
// must itself be the reference model's maths. Host only, no device.
//
//   1. top-k ties go to the LOWER expert id - inside the top-k (slot order) and at the
//      boundary (the k-th vs the (k+1)-th); p9 is the first expert not taken;
//   2. the weights are the top-k p renormalised to sum 1, each rounded to bf16;
//   3. the slot sum is in FIXED ascending slot order: on terms where the order changes
//      the bf16 result, combine() gives the ascending-order one;
//   4. SiLU x up's bf16 chain against double precision (<= 1 bf16 ulp);
//   5. the layout-1 GEMV (split_dot) against a dequantised double dot, for every K split;
//   6. the whole block against an independent double-precision MoE (softmax / sort /
//      renormalise / dequantised experts / shared expert by its sigmoid gate / residual),
//      built from the GPTQ tensors, not the tiles: same expert sets, every output within
//      2 bf16 ulps.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/repack.h"
#include "moe_ref.h"

namespace {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;

// bf16 ulps between two bf16 values of the same sign (or both ~0).
int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// --- 1, 2: the route -------------------------------------------------------------
void test_route_ties() {
  const moe_ref::Shape s{16, 4, 128, 64, 32};
  std::vector<float> lg(s.router_n, 0.0f);
  // distinct, well separated bf16 logits ...
  for (uint32_t e = 0; e < s.experts; ++e) lg[e] = -4.0f + 0.25f * float((e * 7) % 16);
  // ... then two exact ties: experts 12 and 2 share the highest logit (slot order must
  // be 2 then 12), and experts 3 and 9 share the logit at the top-4 boundary (3 is
  // taken, 9 is the first not taken).
  lg[12] = lg[2] = 3.0f;
  lg[5] = 2.5f;
  lg[3] = lg[9] = 2.0f;
  lg[s.experts] = 0.75f;   // the shared gate's logit
  const moe_ref::Route r = moe_ref::route(lg.data(), s);
  CHECK_EQ(r.ids[0], 2u);
  CHECK_EQ(r.ids[1], 12u);
  CHECK_EQ(r.ids[2], 5u);
  CHECK_EQ(r.ids[3], 3u);
  CHECK(r.p[0] == r.p[1]);
  CHECK(r.p9 == r.p[3]);          // expert 9: tied with the last taken, not taken
  // 2: renormalised weights, bf16 each, summing to ~1.
  const double s4 = double(r.p[0]) + r.p[1] + r.p[2] + r.p[3];
  double wsum = 0;
  for (uint32_t k = 0; k < s.top_k; ++k) {
    CHECK(r.w[k] == rf(r.p[k] / float(r.p[0] + r.p[1] + r.p[2] + r.p[3])));
    CHECK_NEAR(r.w[k], r.p[k] / s4, 4e-3);   // half a bf16 ulp near 0.5, and more
    wsum += r.w[k];
  }
  CHECK_NEAR(wsum, 1.0, 1e-2);
  CHECK(r.sg == rf(1.0f / (1.0f + std::exp(-0.75f))));
  // A permutation of equal-logit experts never changes WHICH are taken: all 16 equal.
  std::vector<float> eq(s.router_n, 1.0f);
  const moe_ref::Route re = moe_ref::route(eq.data(), s);
  for (uint32_t k = 0; k < s.top_k; ++k) {
    CHECK_EQ(re.ids[k], k);       // the lowest ids, in id order
    CHECK(re.w[k] == rf(0.25f));
  }
}

// --- 3: the fixed slot order ------------------------------------------------------
void test_slot_order() {
  // Terms whose fp32 sum depends on the order: a large pair that cancels and two small
  // terms that one order absorbs into the large ones and the other does not. Ascending
  // slot order: 1024 + 2^-15 = 1024 (below half an fp32 ulp), - 1024 = 0, + 2^-15 ->
  // 2^-15. Descending: 2^-15 + -1024 = -1024, + 2^-15 = -1024, + 1024 = 0.
  const moe_ref::Shape s{16, 8, 128, 64, 32};
  const float tiny = std::ldexp(1.0f, -15);
  const float cases[2][9] = {{1024.f, tiny, -1024.f, tiny, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 1024.f, tiny, -1024.f, tiny, 0}};
  for (const float* d : cases) {
    float fwd = 0.f, rev = 0.f;
    for (int k = 0; k < 8; ++k) fwd += d[k];
    for (int k = 7; k >= 0; --k) rev += d[k];
    CHECK(rne(fwd) != rne(rev));                  // the order matters for these terms
    moe_ref::Route r;
    for (int k = 0; k < 8; ++k) r.w[k] = 1.0f;   // the terms ARE the d's: rne(d x 1) = d
    r.sg = 0.0f;
    const uint16_t out = moe_ref::combine(d, r, s, /*resid=*/0);
    CHECK_EQ(out, rne(tiny));                     // ascending slot order: 2^-15
    CHECK_EQ(out, rne(f32(rne(fwd))));
  }
}

// --- 4: SiLU x up ------------------------------------------------------------------
void test_silu_up() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> v(-6.0f, 6.0f);
  int worst = 0;
  for (int i = 0; i < 20000; ++i) {
    const float g = v(rng), u = v(rng);
    const uint16_t got = moe_ref::silu_up(g, u);
    const double gb = f32(rne(g)), ub = f32(rne(u));
    const double sb = f32(rne(float(gb / (1.0 + std::exp(-gb)))));
    const uint16_t want = rne(float(sb * ub));
    worst = std::max(worst, ulps(got, want));
  }
  CHECK(worst <= 1);
}

// --- 5: the layout-1 GEMV ------------------------------------------------------------
void test_split_dot() {
  const uint32_t K = 2048, N = 64;
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 5);
  const std::vector<uint32_t> tiles = w.tiled();
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> xv(-1.f, 1.f);
  std::vector<uint16_t> x(K);
  for (uint16_t& e : x) e = rne(xv(rng));
  for (uint32_t ks : {1u, 2u, 4u, 8u, 32u})
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0, mag = 0;
      for (uint32_t k = 0; k < K; ++k) {
        ref += double(w.at(k, n)) * f32(x[k]);
        mag += std::fabs(double(w.at(k, n)) * f32(x[k]));
      }
      const float got = moe_ref::split_dot(tiles.data(), K, n, x.data(), ks);
      CHECK_NEAR(got, ref, 1e-5 * mag + 1e-6);
    }
}

// --- 6: the block against an independent double-precision MoE --------------------
struct Experts {   // GPTQ sources, one per block (routed + the shared last)
  std::vector<common::Int4Gptq> gate, up, down;
};

std::vector<uint32_t> tile_gate_up(const moe_ref::Shape& s, const Experts& x) {
  std::vector<uint32_t> out(s.gate_up_words() * (s.experts + 1));
  for (uint32_t b = 0; b <= s.experts; ++b) {
    const common::Part g{x.gate[b].qweight.data(), x.gate[b].scales.data(), s.inter};
    const common::Part u{x.up[b].qweight.data(), x.up[b].scales.data(), s.inter};
    common::repack_int4_layout1_cols(s.hidden, 2 * s.inter, common::cols_interleave16(g, u),
                                     out.data() + b * s.gate_up_words());
  }
  return out;
}
std::vector<uint32_t> tile_down(const moe_ref::Shape& s, const Experts& x) {
  std::vector<uint32_t> out(s.down_words() * (s.experts + 1));
  for (uint32_t b = 0; b <= s.experts; ++b)
    common::repack_int4_layout1(x.down[b].qweight.data(), x.down[b].scales.data(), s.inter,
                                s.hidden, out.data() + b * s.down_words());
  return out;
}

// The spec's formula (spec 15 §4.1 / moe.cl's header), double precision between the
// bf16 roundings torch makes; experts dequantised from their GPTQ tensors.
std::vector<uint16_t> naive_block(const moe_ref::Shape& s, const std::vector<float>& logits,
                                  const std::vector<uint16_t>& x, const Experts& ex,
                                  const std::vector<uint16_t>& resid, std::vector<uint32_t>& ids) {
  const uint32_t E = s.experts;
  std::vector<double> p(E);
  double mx = -1e300;
  for (uint32_t e = 0; e < E; ++e) mx = std::max(mx, double(rf(logits[e])));
  double z = 0;
  for (uint32_t e = 0; e < E; ++e) z += (p[e] = std::exp(double(rf(logits[e])) - mx));
  for (double& v : p) v /= z;
  std::vector<uint32_t> order(E);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return p[a] > p[b]; });
  ids.assign(order.begin(), order.begin() + s.top_k);
  double ps = 0;
  for (uint32_t k = 0; k < s.top_k; ++k) ps += p[ids[k]];
  auto expert_out = [&](uint32_t b) {   // bf16 [hidden]: the expert's SwiGLU, torch's roundings
    std::vector<uint16_t> h(s.inter), o(s.hidden);
    for (uint32_t i = 0; i < s.inter; ++i) {
      double g = 0, u = 0;
      for (uint32_t k = 0; k < s.hidden; ++k) {
        g += double(ex.gate[b].at(k, i)) * f32(x[k]);
        u += double(ex.up[b].at(k, i)) * f32(x[k]);
      }
      const double gb = f32(rne(float(g))), ub = f32(rne(float(u)));
      h[i] = rne(float(double(f32(rne(float(gb / (1 + std::exp(-gb)))))) * ub));
    }
    for (uint32_t n = 0; n < s.hidden; ++n) {
      double d = 0;
      for (uint32_t i = 0; i < s.inter; ++i) d += double(ex.down[b].at(i, n)) * f32(h[i]);
      o[n] = rne(float(d));
    }
    return o;
  };
  std::vector<std::vector<uint16_t>> outs;
  for (uint32_t k = 0; k < s.top_k; ++k) outs.push_back(expert_out(ids[k]));
  const std::vector<uint16_t> sh = expert_out(E);
  const double sgl = f32(rne(logits[E]));
  const double sg = f32(rne(float(1.0 / (1.0 + std::exp(-sgl)))));
  std::vector<uint16_t> out(s.hidden);
  for (uint32_t n = 0; n < s.hidden; ++n) {
    double sum = 0;
    for (uint32_t k = 0; k < s.top_k; ++k) {
      const double w = f32(rne(float(p[ids[k]] / ps)));
      sum += f32(rne(float(f32(outs[k][n]) * w)));
    }
    const double r_b = f32(rne(float(sum)));
    const double sh_b = f32(rne(float(f32(sh[n]) * sg)));
    out[n] = rne(float(f32(resid[n]) + f32(rne(float(r_b + sh_b)))));
  }
  return out;
}

void test_block() {
  const moe_ref::Shape s{32, 4, 256, 128, 48};
  std::mt19937 rng(2026);
  Experts ex;
  for (uint32_t b = 0; b <= s.experts; ++b) {
    ex.gate.push_back(common::Int4Gptq::random(s.hidden, s.inter, 100 + 3 * b));
    ex.up.push_back(common::Int4Gptq::random(s.hidden, s.inter, 101 + 3 * b));
    ex.down.push_back(common::Int4Gptq::random(s.inter, s.hidden, 102 + 3 * b));
  }
  const std::vector<uint32_t> gu = tile_gate_up(s, ex), dn = tile_down(s, ex);
  std::uniform_real_distribution<float> xv(-0.08f, 0.08f), lv(-3.f, 3.f);
  int worst = 0;
  size_t exact = 0, total = 0;
  double bare_max = 0;   // the largest |output| of a zero-residual token
  for (int tok = 0; tok < 6; ++tok) {
    std::vector<uint16_t> x(s.hidden), resid(s.hidden);
    for (uint16_t& e : x) e = rne(xv(rng));
    // Odd tokens add the block to a zero residual, so its output is compared bare.
    for (uint16_t& e : resid) e = tok % 2 ? uint16_t(0) : rne(lv(rng));
    std::vector<float> logits(s.router_n, 0.f);
    for (uint32_t e = 0; e <= s.experts; ++e) logits[e] = lv(rng);
    const moe_ref::Route r = moe_ref::route(logits.data(), s);
    const std::vector<uint16_t> h = moe_ref::gate_up(r, x.data(), gu.data(), s, 4);
    const std::vector<uint16_t> got = moe_ref::down(r, h, dn.data(), s, resid.data(), 2);
    std::vector<uint32_t> ids;
    const std::vector<uint16_t> want = naive_block(s, logits, x, ex, resid, ids);
    for (uint32_t k = 0; k < s.top_k; ++k) CHECK_EQ(r.ids[k], ids[k]);
    for (uint32_t n = 0; n < s.hidden; ++n) {
      const int u = ulps(got[n], want[n]);
      worst = std::max(worst, u);
      exact += u == 0;
      ++total;
      if (tok % 2) bare_max = std::max(bare_max, std::fabs(double(f32(got[n]))));
    }
  }
  std::printf("  block vs double-precision MoE: %zu / %zu bf16 outputs exact, worst %d ulp"
              " (bare block outputs up to %.3f)\n",
              exact, total, worst, bare_max);
  CHECK(bare_max > 0.05);   // the compared outputs are the block's, not a dominant residual
  CHECK(worst <= 2);
  CHECK(exact * 100 >= total * 95);
}

}  // namespace

int main() {
  test_route_ties();
  test_slot_order();
  test_silu_up();
  test_split_dot();
  test_block();
  std::puts("moe_ref_test OK");
  return 0;
}
