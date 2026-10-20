// Spec 24c Task 2: tests/kernels/lfm2moe_ref.h (the LFM2.5-24B-A2B decode kernels' host twin) against
// tools/oracle/lfm2moe_ref.py's restated ops and transformers 5.19.0's own pieces (tests/kernels/lfm2moe_fixture.h),
// and against the semantics' properties (plan 24c Review Focus 1-3, 5). Host only.
//
//   (a) the norm: coarse rows (incl. a huge and a tiny Σ) bitwise; fine rows within 1 ulp (Σx²'s order: torch's
//       reduction is not the kernel's tree) and the count printed; the fold's h_out bitwise; the two roundings
//       (x̂ rounded before the w product) are not the one-rounding chain
//   (b) the conv ring (Review Focus 1): positions 0 .. 17 (the ring wraps at 16) bitwise from a ring full of garbage
//       (positions < 0 read 0, never the ring), as M = 1 steps, as M = 4 launches and as a 1 / 2 / 3 / 4 mix; the
//       ring holds Bx at pos % 16
//   (c) the route (Review Focus 2): the router's fp32 order against torch's bf16 linear (printed; within 1 ulp);
//       the route on torch's logits: ids, w, the 5th sel, the gap bitwise; crafted rows - a tie at the 4th / 5th
//       goes to the lower id (torch's tie order is undetermined: the set is compared beyond the tie), all-equal
//       logits, a bias that reorders without moving a weight, one dominant expert
//   (d) the MoE: the combine bitwise from torch's expert outputs (grouped_mm's slot order); the whole block
//       from the int4 experts (gate||up exact up to SiLU's exp, down within ulps: the GEMV's order)
//   (e) the attention prep: q / k head norm + RoPE bitwise at 0, 1, 4095, 131071; eager attention's scores and
//       probabilities bitwise, outputs within 1 ulp (P·V's fp32 order); the fp64 form close to eager
//   (f) silu_mul at 11776 (SiLU's exp printed)
//   (g) the argmax mask (Review Focus 5): 127999 never wins over the 125017 real ids; ties to the lower id
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include "check.h"
#include "common/repack.h"
#include "kernels/lfm2moe_fixture.h"
#include "kernels/lfm2moe_ref.h"
#include "loader/lfm2moe_repack.h"   // lfm_rope_table
#include "model/lfm2moe.h"

namespace {

namespace fx = lfm2moe_fixture;
namespace lr = lfmref;
using lr::f32;
using lr::rf;
using lr::rne;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

std::vector<float> vals(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step = 0.125f, float amp = 1.0f,
                        uint32_t base = 0) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = lr::fixture_val(coarse, t, base + i, seed, step, amp);
  return v;
}
std::vector<uint16_t> bf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = rne(v[i]);
  return o;
}

// --- (a) -----------------------------------------------------------------------------------------------------
void test_norm() {
  const uint32_t H = lr::kHidden;
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  const std::vector<std::vector<float>> rows = {
      vals(true, 0, H, 10, 0.125f),         vals(false, 0, H, 11, 0.125f, 4.0f), vals(true, 0, H, 12, 128.0f),
      vals(true, 0, H, 13, 0x1p-20f),       vals(false, 0, H, 14, 0.125f, 2.0f), vals(false, 0, H, 15, 0.125f, 2.0f),
      vals(false, 0, H, 16, 0.125f, 2.0f),  vals(false, 0, H, 17, 0.125f, 2.0f)};
  const std::vector<uint16_t> want_x = lr::hex16(fx::kNormX), want_h = lr::hex16(fx::kNormH);
  CHECK_EQ(want_x.size(), size_t(8) * H);
  CHECK_EQ(want_h.size(), size_t(4) * H);
  size_t fine_diff = 0, two_vs_one = 0;
  for (uint32_t r = 0; r < 8; ++r) {
    const std::vector<uint16_t> h_in = bf(rows[r]);
    std::vector<float> slices;
    uint32_t S = 0;
    if (r >= 4) {
      S = 2;
      const std::vector<float> s0 = vals(true, 2, H, 20 + r, 0.125f), s1 = vals(true, 3, H, 20 + r, 0.125f);
      slices = s0;
      slices.insert(slices.end(), s1.begin(), s1.end());
    }
    const lr::NormOut o = lr::res_norm(h_in.data(), S ? slices.data() : nullptr, S, 1, w.data());
    if (r >= 4)
      for (uint32_t k = 0; k < H; ++k) CHECK_EQ(o.h_out[k], want_h[size_t(r - 4) * H + k]);   // the fold: bitwise
    else
      CHECK(o.h_out == h_in);   // _NF: the residual untouched
    const bool coarse = r == 0 || r == 2 || r == 3;
    for (uint32_t k = 0; k < H; ++k) {
      const uint16_t got = o.x[k], want = want_x[size_t(r) * H + k];
      if (coarse) {
        if (got != want) std::fprintf(stderr, "norm row %u k %u: %04x != torch's %04x\n", r, k, got, want);
        CHECK_EQ(got, want);
      } else {
        CHECK(ulps16(got, want) <= 1);
        fine_diff += got != want;
      }
    }
    // The two roundings matter: the one-rounding chain rne(r · rstd · w) differs somewhere on the fine rows.
    if (!coarse) {
      const float rstd = lr::rstd_of(lr::norm_sumsq(o.h_out.data()));
      for (uint32_t k = 0; k < H; ++k) two_vs_one += rne(f32(o.h_out[k]) * rstd * w[k]) != o.x[k];
    }
  }
  CHECK(two_vs_one > 0);
  std::printf("  (a) norm: coarse rows bitwise (incl. a huge and a tiny Σ), the fold bitwise; fine rows %zu of %u "
              "elements one ulp apart (Σx²'s order); the two-rounding chain differs from one rounding at %zu\n",
              fine_diff, 5 * H, two_vs_one);
}

// --- (b) -----------------------------------------------------------------------------------------------------
void test_conv() {
  const uint32_t H = lr::kHidden, T = fx::kConvT, N = 3 * H;
  const std::vector<uint16_t> want_bx = lr::hex16(fx::kConvBx), want_y = lr::hex16(fx::kConvY);
  CHECK_EQ(want_y.size(), size_t(T) * H);
  const std::vector<uint16_t> B = bf(vals(false, 20, T * H, 5, 0.125f, 2.0f)), C = bf(vals(false, 21, T * H, 5, 0.125f, 2.0f)),
                              X = bf(vals(false, 22, T * H, 5, 0.125f, 2.0f));
  const std::vector<float> taps = [&] {
    std::vector<float> t = vals(false, 23, H * 3, 5, 0.125f, 0.5f);
    return t;   // bf16 values, [2048][3]: tap 0 multiplies Bx[p - 2]
  }();
  // in_proj's output as one slice per row: [M][6144] = B | C | x (an fp32 value of each bf16: rne(0 + v) = v)
  auto slice_rows = [&](uint32_t p0, uint32_t M) {
    std::vector<float> s(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t c = 0; c < H; ++c) {
        const size_t i = size_t(p0 + m) * H + c;
        s[size_t(m) * N + c] = f32(B[i]);
        s[size_t(m) * N + H + c] = f32(C[i]);
        s[size_t(m) * N + 2 * H + c] = f32(X[i]);
      }
    return s;
  };
  auto run = [&](const std::vector<uint32_t>& ms, const char* what) {
    std::vector<uint16_t> ring(size_t(lr::kRing) * H);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage: never read
    uint32_t p = 0;
    size_t steps = 0;
    for (size_t i = 0; p < T; ++i) {
      const uint32_t M = std::min(ms[i % ms.size()], T - p);
      const std::vector<float> s = slice_rows(p, M);
      const std::vector<uint16_t> y = lr::conv_step(s.data(), 1, ring.data(), taps.data(), p, M);
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t c = 0; c < H; ++c) {
          if (y[size_t(m) * H + c] != want_y[size_t(p + m) * H + c])
            std::fprintf(stderr, "conv (%s) position %u channel %u: %04x != torch's %04x\n", what, p + m, c,
                         y[size_t(m) * H + c], want_y[size_t(p + m) * H + c]);
          CHECK_EQ(y[size_t(m) * H + c], want_y[size_t(p + m) * H + c]);
          CHECK_EQ(ring[size_t((p + m) % lr::kRing) * H + c], want_bx[size_t(p + m) * H + c]);   // the ring holds Bx
        }
      p += M;
      ++steps;
    }
    std::printf("  (b) conv %s: positions 0 .. %u bitwise in %zu launches (the ring wraps at 16)\n", what, T - 1, steps);
  };
  run({1}, "M = 1");
  run({4}, "M = 4");
  run({1, 2, 3, 4}, "M = 1 / 2 / 3 / 4");
  run({14}, "M = 14");
}

// --- (c) -----------------------------------------------------------------------------------------------------
void test_route() {
  const uint32_t H = lr::kHidden, E = lr::kExperts;
  const std::vector<uint16_t> x = bf(vals(false, 30, 64 * H, 6)), gw = bf(vals(false, 31, E * H, 6, 0.125f, 0.05f));
  std::vector<float> bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = (float(lr::fixture_hash(32, e, 6) >> 8) * 0x1p-23f - 1.0f) * 0.05f;
  std::vector<uint16_t> tiles(size_t(H) * E);
  common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
  const std::vector<uint16_t> want_l = lr::hex16(fx::kRouteLogits);
  const std::vector<uint32_t> ids = lr::hex32(fx::kRouteIds), wv = lr::hex32(fx::kRouteW), s5 = lr::hex32(fx::kRouteSel5),
                              gp = lr::hex32(fx::kRouteGap);
  size_t ldiff = 0;
  for (uint32_t r = 0; r < 64; ++r) {
    const std::vector<float> lg = lr::router(x.data() + size_t(r) * H, tiles.data());
    for (uint32_t e = 0; e < E; ++e) {
      CHECK(ulps16(rne(lg[e]), want_l[size_t(r) * E + e]) <= 1);
      ldiff += rne(lg[e]) != want_l[size_t(r) * E + e];
    }
    // The route on torch's own bf16 logits: ids in rank order, w, the 5th sel, the gap - bitwise (no exact tie
    // in these rows: every gap > 0).
    std::vector<float> tl(E);
    for (uint32_t e = 0; e < E; ++e) tl[e] = f32(want_l[size_t(r) * E + e]);
    const lr::Route rt = lr::route(tl.data(), bias.data());
    CHECK(lr::as_f32(gp[r]) > 0.0f);
    for (uint32_t k = 0; k < lr::kTopK; ++k) {
      CHECK_EQ(rt.ids[k], ids[size_t(r) * 4 + k]);
      CHECK_EQ(lr::as_u32(rt.w[k]), wv[size_t(r) * 4 + k]);
    }
    for (uint32_t k = 0; k <= lr::kTopK; ++k) CHECK_EQ(lr::as_u32(rt.sel[k]), s5[size_t(r) * 5 + k]);
    CHECK_EQ(lr::as_u32(lr::gap(rt)), gp[r]);
    // The route row as lfm_route writes it.
    CHECK_EQ(rt.row[lr::kIds], rt.ids[0]);
    CHECK_EQ(rt.row[lr::kSum], lr::as_u32(rt.sum));
    CHECK_EQ(rt.row[14], 0u);
  }
  std::printf("  (c) route: 64 rows - ids, w, the 5th sel, the gap bitwise on torch's logits; the router's fp32 order "
              "%zu of %u logits one bf16 ulp from torch's bf16 linear\n", ldiff, 64 * E);

  // Crafted rows.
  const std::vector<uint32_t> cl = lr::hex32(fx::kCraftLogits), cb = lr::hex32(fx::kCraftBias), cid = lr::hex32(fx::kCraftIds),
                              cw = lr::hex32(fx::kCraftW), cs = lr::hex32(fx::kCraftSel5), cg = lr::hex32(fx::kCraftGap);
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> l(E), b(E);
    for (uint32_t e = 0; e < E; ++e) {
      l[e] = lr::as_f32(cl[size_t(r) * E + e]);
      b[e] = lr::as_f32(cb[size_t(r) * E + e]);
    }
    const lr::Route rt = lr::route(l.data(), b.data());
    std::multiset<uint32_t> mw, tw;
    for (uint32_t k = 0; k < 4; ++k) {
      mw.insert(lr::as_u32(rt.w[k]));
      tw.insert(cw[size_t(r) * 4 + k]);
    }
    CHECK(mw == tw);   // the weights as a multiset (torch's order among exact ties is its own)
    for (uint32_t k = 0; k <= 4; ++k) CHECK_EQ(lr::as_u32(rt.sel[k]), cs[size_t(r) * 5 + k]);
    CHECK_EQ(lr::as_u32(lr::gap(rt)), cg[r]);
    std::printf("    crafted %u: ours {%u %u %u %u}, torch's {%u %u %u %u}, gap %g\n", r, rt.ids[0], rt.ids[1], rt.ids[2],
                rt.ids[3], cid[r * 4], cid[r * 4 + 1], cid[r * 4 + 2], cid[r * 4 + 3], double(lr::gap(rt)));
    if (r == 0) {   // 3, 9, 20 tie for the top (lower id first), 31 / 40 tie for the 4th: the lower id, 31
      CHECK(rt.ids[0] == 3 && rt.ids[1] == 9 && rt.ids[2] == 20 && rt.ids[3] == 31);
      CHECK_EQ(lr::gap(rt), 0.0f);
    } else if (r == 1) {   // every sel ties: ids 0 .. 3, w = rne(0.5 / rne(rne(2) + 1e-6)) = 0.25
      CHECK(rt.ids[0] == 0 && rt.ids[1] == 1 && rt.ids[2] == 2 && rt.ids[3] == 3);
      CHECK_EQ(rt.w[0], 0.25f);
    } else if (r == 2) {   // the bias lifts 20..23 over 10..13; their weights are s(0.5)'s, not sel's
      CHECK(rt.ids[0] == 20 && rt.ids[3] == 23);
      CHECK_EQ(rt.s[0], rf(lr::sigmoid(0.5f)));
    } else {               // one dominant expert first, then the lowest ids of the tied rest
      CHECK(rt.ids[0] == 7 && rt.ids[1] == 0 && rt.ids[2] == 1 && rt.ids[3] == 2);
      CHECK_EQ(cid[r * 4], 7u);
      CHECK(rt.w[0] > 0.99f);
    }
  }
}

// --- (d) -----------------------------------------------------------------------------------------------------
void test_moe() {
  const uint32_t H = lr::kHidden, I = lr::kInter;
  const std::vector<uint16_t> proj = lr::hex16(fx::kMoeProj), want_y = lr::hex16(fx::kMoeY), act0 = lr::hex16(fx::kMoeAct0);
  const std::vector<uint32_t> wbits = lr::hex32(fx::kMoeW);
  // The combine, from torch's own expert outputs: bitwise.
  for (uint32_t r = 0; r < 4; ++r)
    for (uint32_t n = 0; n < H; ++n) {
      uint16_t p[4];
      float w[4];
      for (uint32_t k = 0; k < 4; ++k) {
        p[k] = proj[(size_t(r) * 4 + k) * H + n];
        w[k] = lr::as_f32(wbits[r * 4 + k]);
      }
      CHECK_EQ(lr::combine_proj(p, w), want_y[size_t(r) * H + n]);
    }
  // The whole block from the int4 experts (blocks only for the experts used, ids remapped).
  std::vector<uint32_t> used;
  for (const auto& row : fx::kMoeIds)
    for (uint32_t e : row)
      if (std::find(used.begin(), used.end(), e) == used.end()) used.push_back(e);
  std::sort(used.begin(), used.end());
  std::vector<uint32_t> gu(used.size() * lr::gate_up_block_words()), dn(used.size() * lr::down_block_words());
  for (size_t u = 0; u < used.size(); ++u) {
    const lr::Gptq g = lr::expert_gptq(used[u], 0, H, I), up = lr::expert_gptq(used[u], 1, H, I),
                   d = lr::expert_gptq(used[u], 2, I, H);
    const std::vector<common::ColSource> cols =
        common::cols_interleave16({g.qweight.data(), g.scales.data(), I}, {up.qweight.data(), up.scales.data(), I});
    common::repack_int4_layout1_cols(H, 2 * I, cols, gu.data() + u * lr::gate_up_block_words());
    common::repack_int4_layout1(d.qweight.data(), d.scales.data(), I, H, dn.data() + u * lr::down_block_words());
  }
  size_t ydiff = 0, adiff = 0;
  int ymax = 0;
  for (uint32_t r = 0; r < 4; ++r) {
    const std::vector<uint16_t> x = bf(vals(true, 40, H, 7, 0.125f, 1.0f, r * H));
    lr::Route rt;
    for (uint32_t k = 0; k < 4; ++k) {
      rt.ids[k] = uint32_t(std::find(used.begin(), used.end(), fx::kMoeIds[r][k]) - used.begin());
      rt.w[k] = lr::as_f32(wbits[r * 4 + k]);
    }
    const std::vector<uint16_t> h0(H, 0);
    const lr::MoeOut o = lr::moe(rt, x.data(), gu.data(), dn.data(), h0.data());
    if (r == 0)
      for (size_t i = 0; i < size_t(4) * I; ++i) {
        CHECK(ulps16(o.h[i], act0[i]) <= 1);   // gate / up exact (coarse x, bf16 weights); SiLU's exp
        adiff += o.h[i] != act0[i];
      }
    for (uint32_t n = 0; n < H; ++n) {
      const int u = ulps16(o.y[n], want_y[size_t(r) * H + n]);
      ymax = std::max(ymax, u);
      ydiff += u != 0;
      CHECK_EQ(o.h_res[n], rne(0.0f + f32(o.y[n])));   // the fold onto a zero residual
    }
  }
  CHECK(ymax <= 2);
  CHECK(ydiff * 20 <= size_t(4) * H);   // >= 95 % bitwise: the down GEMV's fp32 order is the only difference
  std::printf("  (d) MoE: the combine bitwise from torch's expert outputs (rank order, fp32, one rounding); row 0's "
              "activations %zu of %u one ulp off (SiLU's exp); y %zu of %u differ, at most %d ulp (the down GEMV's order)\n",
              adiff, 4 * I, ydiff, 4 * H, ymax);
}

// --- (e) -----------------------------------------------------------------------------------------------------
void test_attention() {
  const uint32_t HD = lr::kHd;
  const std::vector<float> qh = vals(true, 60, HD, 3, 0.25f), kh = vals(true, 61, HD, 3, 0.25f);
  std::vector<float> qkn(2 * HD);
  for (uint32_t d = 0; d < HD; ++d) {
    qkn[d] = rf(1.0f + lr::fixture_val(false, 62, d, 3, 0.125f, 0.5f));
    qkn[HD + d] = rf(1.0f + lr::fixture_val(false, 63, d, 3, 0.125f, 0.5f));
  }
  // The normed heads (cos 1, sin 0: RoPE the identity).
  std::vector<float> ident(HD, 0.0f);
  for (uint32_t i = 0; i < HD / 2; ++i) ident[i] = 1.0f;
  uint16_t out[64];
  lr::head_norm_rope(qh.data(), qkn.data(), ident.data(), out);
  CHECK(std::vector<uint16_t>(out, out + HD) == lr::hex16(fx::kQNormed));
  lr::head_norm_rope(kh.data(), qkn.data() + HD, ident.data(), out);
  CHECK(std::vector<uint16_t>(out, out + HD) == lr::hex16(fx::kKNormed));
  const std::vector<float> table = loader::lfm_rope_table(model::lfm2moe(), 131072);
  const std::vector<uint16_t> qr = lr::hex16(fx::kQRoped), kr = lr::hex16(fx::kKRoped);
  for (uint32_t i = 0; i < fx::kRopeN; ++i) {
    const float* cs = table.data() + size_t(fx::kRopePos[i]) * HD;
    lr::head_norm_rope(qh.data(), qkn.data(), cs, out);
    for (uint32_t d = 0; d < HD; ++d) CHECK_EQ(out[d], qr[size_t(i) * HD + d]);
    lr::head_norm_rope(kh.data(), qkn.data() + HD, cs, out);
    for (uint32_t d = 0; d < HD; ++d) CHECK_EQ(out[d], kr[size_t(i) * HD + d]);
  }
  std::printf("  (e) prep: q / k head norm bitwise, RoPE bitwise at 0, 1, 4095, 131071\n");

  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const uint32_t L = fx::kAttnL[ci];
    const std::vector<float> q = vals(true, 70, lr::kQHeads * HD, ci, 0.5f);
    const std::vector<uint16_t> kk = bf(vals(true, 71, L * lr::kKvHeads * HD, 9)), vv = bf(vals(true, 72, L * lr::kKvHeads * HD, 9));
    const std::vector<uint16_t> ws = lr::hex16(fx::kAttnS[ci]), wp = lr::hex16(fx::kAttnP[ci]), wo = lr::hex16(fx::kAttnO[ci]);
    size_t pdiff = 0, odiff = 0;
    double worst_cos = 1.0;
    for (uint32_t h = 0; h < lr::kQHeads; ++h) {
      const lr::EagerHead e = lr::attention_eager(q.data() + size_t(h) * HD, kk.data(), vv.data(), L, h / 4,
                                                  lr::eager_block(L));
      if (h == 0)
        for (uint32_t i = 0; i < L; ++i) {
          CHECK_EQ(rne(e.s[i]), ws[i]);
          CHECK(ulps16(rne(e.p[i]), wp[i]) <= (L < 8 ? 1 : 0));   // a row shorter than 8: torch sums it sequentially
          pdiff += rne(e.p[i]) != wp[i];
        }
      const std::vector<double> f = lr::attention(q.data() + size_t(h) * HD, kk.data(), vv.data(), L, h / 4);
      double dot = 0, na = 0, nb = 0;
      for (uint32_t d = 0; d < HD; ++d) {
        const uint16_t got = rne(e.o[d]), want = wo[size_t(h) * HD + d];
        CHECK(ulps16(got, want) <= 1);
        odiff += got != want;
        dot += double(e.o[d]) * f[d];
        na += double(e.o[d]) * e.o[d];
        nb += f[d] * f[d];
      }
      worst_cos = std::min(worst_cos, dot / std::sqrt(na * nb + 1e-300));
    }
    CHECK(worst_cos >= 0.9999);
    std::printf("  (e) eager attention at %u keys: scores bitwise, probabilities %zu off by one ulp, outputs %zu of %u "
                "one ulp apart (P·V's order); fp64 cosine >= %.6f\n", L, pdiff, odiff, lr::kQHeads * HD, worst_cos);
  }
}

// --- (f), (g) -----------------------------------------------------------------------------------------------------
void test_silu_argmax() {
  const uint32_t I = lr::kDenseInter;
  const std::vector<float> g = vals(false, 80, I, 4, 0.125f, 4.0f), u = vals(false, 81, I, 4, 0.125f, 4.0f);
  std::vector<float> slice(2 * I);
  for (uint32_t i = 0; i < I; ++i) {
    slice[moe_ref::gate_col(i)] = g[i];
    slice[moe_ref::up_col(i)] = u[i];
  }
  const std::vector<uint16_t> got = lr::silu_mul(slice.data(), 1, 1), want = lr::hex16(fx::kSiluOut);
  size_t diff = 0;
  for (uint32_t i = 0; i < I; ++i) {
    CHECK(ulps16(got[i], want[i]) <= 1);
    diff += got[i] != want[i];
  }
  std::printf("  (f) silu_mul at 11776: %zu of %u one ulp off (SiLU's exp: libm against Sleef)\n", diff, I);

  std::vector<float> lg = vals(false, 90, lr::kVocab, 4);
  lg[125016] = 50.0f;
  lg[127999] = 100.0f;
  CHECK_EQ(lr::argmax(lg.data()), fx::kArgmax);
  CHECK_EQ(lr::argmax(lg.data()), 125016u);
  lg[100] = lg[200] = 60.0f;   // an exact tie at the top: the lower id
  CHECK_EQ(lr::argmax(lg.data()), 100u);
  CHECK_EQ(lr::argmax(lg.data(), lr::kVocab, lr::kVocab), 127999u);   // unmasked, 127999 would win
  std::printf("  (g) argmax: 125016 over the planted 127999 (masked), ties to the lower id\n");
}

}  // namespace

int main() {
  test_norm();
  test_conv();
  test_route();
  test_moe();
  test_attention();
  test_silu_argmax();
  std::printf("lfm2moe_ref_test OK\n");
  return 0;
}
