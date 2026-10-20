// lfm2moe_run - spec 24c's portable LFM2.5-24B-A2B decode kernels on the Mac's OpenCL GPU against
// tests/kernels/lfm2moe_ref.h and torch's own outputs (tests/kernels/lfm2moe_fixture.h) (INDICATIVE ONLY: clrun.h
// says what this can and cannot show).
//
//   lfm2moe_run      no arguments; LFM2's real shapes (hidden 2048, 32 / 8 heads x 64, 64 experts top 4 x 1536,
//                    dense 11776, vocab 128000)
//
// What runs, with the CMake block's defines: src/kernels/lfm2moe/lfm_prep.cl (lfm_res_norm _NF and _S1 on 8
// work-groups - every work-group's Σ the same tree -, lfm_silu_mul at 11776, lfm_attn_prep at positions 0, 1, 4095,
// 131071 against the loader's RoPE table, lfm_embed_gather from gemv_bf16's tiles), src/kernels/lfm2moe/lfm_conv.cl
// (the fixture's 18 positions as M = 1 steps from a garbage ring - the ring wraps at 16), src/kernels/lfm2moe/
// lfm_moe.cl on its plain-load path (lfm_router, lfm_route on the fixture's 64 rows + the four crafted ones,
// lfm_moe_gate_up / lfm_moe_down over the fixture's int4 experts) and src/kernels/lfm2moe/lfm_attn_eager.cl (the
// fixture's 6 / 300 / 3000-key cases). lfm_attn.cl is not run: its decode is shuffles, sub-group reductions and
// 2D block reads.
//
// Bars: the exp-free chains (the norm, the conv, the prep, the gather, the router, the eager scores) bitwise; the
// route ids exact; the rest within 2 bf16 ulps (OpenCL's exp against libm's; Apple's compiler is not told
// -cl-fp32-correctly-rounded-divide-sqrt, so a norm value may move with its 1/sqrt - reported, bar 1).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/lfm2moe_fixture.h"
#include "kernels/lfm2moe_pf_ref.h"
#include "kernels/lfm2moe_ref.h"
#include "loader/lfm2moe_repack.h"
#include "model/lfm2moe.h"

namespace {

namespace lr = lfmref;
namespace fx = lfm2moe_fixture;
using lr::f32;
using lr::rf;
using lr::rne;

const std::vector<std::string> kCtrlDefs = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
int ulps(uint16_t a, uint16_t b) {
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

int failures = 0;
// "exact" / "agrees" / the first differing index
template <class T>
void report(const char* what, const std::vector<T>& got, const std::vector<T>& want, int bar) {
  int worst = 0;
  size_t first = got.size() == want.size() ? size_t(-1) : 0;
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    int u;
    if constexpr (sizeof(T) == 2) {
      u = ulps(uint16_t(got[i]), uint16_t(want[i]));
    } else {
      u = got[i] == want[i] ? 0 : 1000;
    }
    if (u > worst) worst = u;
    if (u > bar && first == size_t(-1)) first = i;
  }
  if (first == size_t(-1)) {
    std::printf("lfm2moe_run: %-56s %zu values, %s\n", what, got.size(),
                worst == 0 ? "exact" : ("agrees (worst " + std::to_string(worst) + " ulp, bar " + std::to_string(bar) + ")").c_str());
  } else {
    std::printf("lfm2moe_run: %-56s DISAGREES at index %zu (worst %d, bar %d)\n", what, first, worst, bar);
    ++failures;
  }
}

constexpr uint32_t H = lr::kHidden, HD = lr::kHd, QH = lr::kQHeads, KVH = lr::kKvHeads, QN = QH * HD, KVN = KVH * HD;

void prep_checks(clrun::Device& dev) {
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  {   // _NF: the fixture's coarse and fine rows; _S1 / _S2: a fold over one / two slices
    clrun::Program nf(dev, "src/kernels/lfm2moe/lfm_prep.cl", {"M=1", "K=2048", "NORM_WGS=8", "NORM_EPS=1e-5f", "NORM_S=0"});
    const std::vector<uint16_t> want = lr::hex16(fx::kNormX);
    for (uint32_t r : {0u, 1u, 2u, 3u}) {
      const std::vector<std::vector<float>> rows = {vals(true, 0, H, 10, 0.125f), vals(false, 0, H, 11, 0.125f, 4.0f),
                                                    vals(true, 0, H, 12, 128.0f), vals(true, 0, H, 13, 0x1p-20f)};
      const std::vector<uint16_t> h = bf(rows[r]);
      clrun::Buffer hb(dev, h), sb(dev, 16), wb(dev, w), ob(dev, H * 2), xb(dev, H * 2);
      nf.run("lfm_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, wb, ob, xb);
      const std::vector<uint16_t> x = xb.read<uint16_t>();
      report(("lfm_res_norm _NF (fixture row " + std::to_string(r) + ")").c_str(), x,
             std::vector<uint16_t>(want.begin() + size_t(r) * H, want.begin() + size_t(r + 1) * H), r == 1 ? 1 : 0);
    }
    for (uint32_t S : {1u, 2u}) {
      clrun::Program sp(dev, "src/kernels/lfm2moe/lfm_prep.cl",
                        {"M=1", "K=2048", "NORM_WGS=8", "NORM_EPS=1e-5f", "NORM_S=" + std::to_string(S)});
      const std::vector<uint16_t> h = bf(vals(false, 0, H, 14, 0.125f, 2.0f));
      std::vector<float> sl;
      for (uint32_t s = 0; s < S; ++s) {
        const std::vector<float> v = vals(true, 2 + s, H, 24, 0.125f);
        sl.insert(sl.end(), v.begin(), v.end());
      }
      clrun::Buffer hb(dev, h), sb(dev, sl), wb(dev, w), ob(dev, H * 2), xb(dev, H * 2);
      sp.run("lfm_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, wb, ob, xb);
      const lr::NormOut want = lr::res_norm(h.data(), sl.data(), S, 1, w.data());
      report(("lfm_res_norm _S" + std::to_string(S) + ": h_out (the fold)").c_str(), ob.read<uint16_t>(), want.h_out, 0);
      report(("lfm_res_norm _S" + std::to_string(S) + ": x").c_str(), xb.read<uint16_t>(), want.x, 1);
      if (S == 2) {   // the fixture's row 4 (slices of seed 24): torch's own h and x
        const std::vector<uint16_t> fh = lr::hex16(fx::kNormH), fxx = lr::hex16(fx::kNormX);
        report("lfm_res_norm _S2 vs torch (fixture row 4)", ob.read<uint16_t>(),
               std::vector<uint16_t>(fh.begin(), fh.begin() + H), 0);
        report("lfm_res_norm _S2 x vs torch (fixture row 4)", xb.read<uint16_t>(),
               std::vector<uint16_t>(fxx.begin() + 4 * size_t(H), fxx.begin() + 5 * size_t(H)), 1);
      }
    }
  }
  {   // silu_mul at 11776, S 2 (the slices' sum then the chain)
    const uint32_t I = lr::kDenseInter;
    clrun::Program p(dev, "src/kernels/lfm2moe/lfm_prep.cl", {"M=1", "SILU_I=11776", "SILU_S=2"});
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-3.f, 3.f);
    std::vector<float> sl(size_t(2) * 2 * I);
    for (float& v : sl) v = d(rng);
    clrun::Buffer sb(dev, sl), ob(dev, size_t(I) * 2);
    p.run("lfm_silu_mul", {size_t((I + 255) / 256) * 256, 1}, {256, 1}, sb, ob);
    report("lfm_silu_mul at 11776 (S 2)", ob.read<uint16_t>(), lr::silu_mul(sl.data(), 2, 1), 2);
  }
  {   // the attention prep at 0, 1, 4095, 131071 (S 2), random slices
    const std::vector<float> table = loader::lfm_rope_table(model::lfm2moe(), 131072);
    clrun::Program p(dev, "src/kernels/lfm2moe/lfm_prep.cl",
                     with(kCtrlDefs, {"M=1", "QKV_S=2", "Q_HEADS=32", "KV_HEADS=8", "HD=64", "PREP_EPS=1e-5f"}));
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> sl(size_t(2) * lr::kQkvN), qkn(2 * HD);
    for (float& v : sl) v = d(rng);
    for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
    clrun::Buffer rb(dev, table), sb(dev, sl), qb(dev, qkn);
    bool ok = true;
    for (uint32_t pos : {0u, 1u, 4095u, 131071u}) {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      const uint32_t rows = pos + 1;
      clrun::Buffer cb(dev, ctrl), aq(dev, QN * 4), kb(dev, size_t(rows) * KVN * 2), vb(dev, size_t(rows) * KVN * 2);
      p.run("lfm_attn_prep", {size_t(QH + 2 * KVH) * 64, 1}, {64, 1}, cb, sb, qb, rb, aq, kb, vb);
      const lr::Prep want = lr::attn_prep(sl.data(), 2, 1, 0, qkn.data(), table.data() + size_t(pos) * HD);
      const std::vector<uint16_t> k = kb.read<uint16_t>(), v = vb.read<uint16_t>();
      ok = ok && aq.read<float>() == want.q && std::equal(want.k.begin(), want.k.end(), k.begin() + size_t(pos) * KVN) &&
           std::equal(want.v.begin(), want.v.end(), v.begin() + size_t(pos) * KVN);
    }
    std::printf("lfm2moe_run: %-56s %s\n", "lfm_attn_prep at 0, 1, 4095, 131071 (S 2)", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
  {   // the gather: rows 0, 7, 125016, 127999 out of a random tiled table
    const uint32_t V = lr::kVocab;
    std::vector<uint16_t> tiles(size_t(H) * V);
    std::mt19937 rng(13);
    for (uint16_t& t : tiles) t = uint16_t(rng());
    clrun::Program p(dev, "src/kernels/lfm2moe/lfm_prep.cl", with(kCtrlDefs, {"M=1", "K=2048", "EMBED_V=128000"}));
    clrun::Buffer tb(dev, tiles);
    bool ok = true;
    for (uint32_t id : {0u, 7u, 125016u, 127999u}) {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[1] = 1;
      ctrl[2] = id;
      clrun::Buffer cb(dev, ctrl), hb(dev, H * 2);
      p.run("lfm_embed_gather", {256, 1}, {256, 1}, cb, tb, hb);
      ok = ok && hb.read<uint16_t>() == lr::embed_row(tiles.data(), id);
    }
    std::printf("lfm2moe_run: %-56s %s\n", "lfm_embed_gather (rows 0, 7, 125016, 127999)", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
}

void conv_checks(clrun::Device& dev) {
  const uint32_t T = fx::kConvT, N = 3 * H;
  clrun::Program p(dev, "src/kernels/lfm2moe/lfm_conv.cl", with(kCtrlDefs, {"M=1", "HIDDEN=2048", "CONV_S=1"}));
  const std::vector<uint16_t> B = bf(vals(false, 20, T * H, 5, 0.125f, 2.0f)), C = bf(vals(false, 21, T * H, 5, 0.125f, 2.0f)),
                              X = bf(vals(false, 22, T * H, 5, 0.125f, 2.0f));
  const std::vector<float> taps = vals(false, 23, H * 3, 5, 0.125f, 0.5f);
  const std::vector<uint16_t> want_y = lr::hex16(fx::kConvY);
  std::vector<uint16_t> ring(size_t(lr::kRing) * H);
  for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage: never read
  clrun::Buffer rb(dev, ring), tb(dev, taps), yb(dev, H * 2);
  std::vector<uint16_t> y_all;
  for (uint32_t pos = 0; pos < T; ++pos) {
    std::vector<float> s(N);
    for (uint32_t c = 0; c < H; ++c) {
      s[c] = f32(B[size_t(pos) * H + c]);
      s[H + c] = f32(C[size_t(pos) * H + c]);
      s[2 * H + c] = f32(X[size_t(pos) * H + c]);
    }
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = 1;
    clrun::Buffer cb(dev, ctrl), sb(dev, s);
    p.run("lfm_conv", {H, 1}, {256, 1}, cb, sb, tb, rb, yb);
    const std::vector<uint16_t> y = yb.read<uint16_t>();
    y_all.insert(y_all.end(), y.begin(), y.end());
  }
  report("lfm_conv: positions 0 .. 17 vs torch (ring wraps at 16)", y_all, want_y, 0);
}

void moe_checks(clrun::Device& dev) {
  const uint32_t E = lr::kExperts, I = lr::kInter;
  {   // the router on the fixture's 64 rows, then the route on its logits and on torch's crafted rows
    const std::vector<uint16_t> x = bf(vals(false, 30, 64 * H, 6)), gw = bf(vals(false, 31, E * H, 6, 0.125f, 0.05f));
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
    clrun::Program rp(dev, "src/kernels/lfm2moe/lfm_moe.cl", {"M=1", "ROUTER_E=64", "ROUTER_K=2048"});
    clrun::Program tp(dev, "src/kernels/lfm2moe/lfm_moe.cl", {"M=1", "ROUTE_E=64"});
    std::vector<float> bias(E);
    for (uint32_t e = 0; e < E; ++e) bias[e] = (float(lr::fixture_hash(32, e, 6) >> 8) * 0x1p-23f - 1.0f) * 0.05f;
    clrun::Buffer tb(dev, tiles), bb(dev, bias);
    std::vector<uint32_t> got_rows, want_rows;
    std::vector<uint32_t> lg_got, lg_want;
    for (uint32_t r = 0; r < 64; ++r) {
      const std::vector<uint16_t> xr(x.begin() + size_t(r) * H, x.begin() + size_t(r + 1) * H);
      clrun::Buffer xb(dev, xr), lb(dev, E * 4), ro(dev, lr::kWords * 4);
      rp.run("lfm_router", {4 * 128, 1}, {128, 1}, xb, tb, lb);
      const std::vector<float> lg = lb.read<float>(), want = lr::router(xr.data(), tiles.data());
      for (uint32_t e = 0; e < E; ++e) {
        lg_got.push_back(lr::as_u32(lg[e]));
        lg_want.push_back(lr::as_u32(want[e]));
      }
      tp.run("lfm_route", {64, 1}, {64, 1}, lb, bb, ro);
      const std::vector<uint32_t> row = ro.read<uint32_t>();
      const lr::Route rt = lr::route(want.data(), bias.data());
      got_rows.insert(got_rows.end(), row.begin(), row.end());
      want_rows.insert(want_rows.end(), rt.row, rt.row + lr::kWords);
    }
    report("lfm_router (64 rows, fp32 logits, the 8-span tree)", lg_got, lg_want, 0);
    // The ids exactly; the weights' bits through sigmoid's exp (bf16-rounded: almost always exact).
    size_t wdiff = 0;
    bool ids_ok = true;
    for (uint32_t r = 0; r < 64; ++r)
      for (uint32_t k = 0; k < lr::kWords; ++k) {
        const uint32_t g = got_rows[r * lr::kWords + k], w = want_rows[r * lr::kWords + k];
        if (k < 4) ids_ok = ids_ok && g == w;
        else wdiff += g != w;
      }
    std::printf("lfm2moe_run: %-56s ids %s, %zu other words differ (sigmoid's exp)\n", "lfm_route (64 fixture rows)",
                ids_ok ? "exact" : "DISAGREE", wdiff);
    if (!ids_ok) ++failures;
    const std::vector<uint32_t> cl = lr::hex32(fx::kCraftLogits), cb = lr::hex32(fx::kCraftBias);
    bool cr_ok = true;
    for (uint32_t r = 0; r < 4; ++r) {
      std::vector<float> l(E), b(E);
      for (uint32_t e = 0; e < E; ++e) {
        l[e] = lr::as_f32(cl[size_t(r) * E + e]);
        b[e] = lr::as_f32(cb[size_t(r) * E + e]);
      }
      clrun::Buffer lb(dev, l), bb2(dev, b), ro(dev, lr::kWords * 4);
      tp.run("lfm_route", {64, 1}, {64, 1}, lb, bb2, ro);
      const std::vector<uint32_t> row = ro.read<uint32_t>();
      const lr::Route rt = lr::route(l.data(), b.data());
      for (uint32_t k = 0; k < lr::kWords; ++k) cr_ok = cr_ok && row[k] == rt.row[k];
    }
    std::printf("lfm2moe_run: %-56s %s\n", "lfm_route (tie at the 4th/5th, all-equal, bias, dominant)",
                cr_ok ? "exact" : "DISAGREES");
    if (!cr_ok) ++failures;
  }
  {   // the MoE block over the fixture's int4 experts (ids remapped onto the used blocks)
    std::vector<uint32_t> used;
    for (const auto& row : fx::kMoeIds)
      for (uint32_t e : row)
        if (std::find(used.begin(), used.end(), e) == used.end()) used.push_back(e);
    std::sort(used.begin(), used.end());
    std::vector<uint32_t> gu(used.size() * lr::gate_up_block_words()), dn(used.size() * lr::down_block_words());
    for (size_t u = 0; u < used.size(); ++u) {
      const lr::Gptq g = lr::expert_gptq(used[u], 0, H, I), up = lr::expert_gptq(used[u], 1, H, I),
                     d = lr::expert_gptq(used[u], 2, I, H);
      common::repack_int4_layout1_cols(
          H, 2 * I, common::cols_interleave16({g.qweight.data(), g.scales.data(), I}, {up.qweight.data(), up.scales.data(), I}),
          gu.data() + u * lr::gate_up_block_words());
      common::repack_int4_layout1(d.qweight.data(), d.scales.data(), I, H, dn.data() + u * lr::down_block_words());
    }
    clrun::Program mp(dev, "src/kernels/lfm2moe/lfm_moe.cl",
                      {"M=1", "MOE_E=64", "MOE_K=4", "HIDDEN=2048", "INTER=1536", "UP_KS=4", "DN_KS=2"});
    const std::vector<uint32_t> wbits = lr::hex32(fx::kMoeW);
    const std::vector<uint16_t> want_y = lr::hex16(fx::kMoeY);
    clrun::Buffer gb(dev, gu), db(dev, dn);
    std::vector<uint16_t> h_all, h_ref_all, y_all, y_ref_all, ytorch;
    for (uint32_t r = 0; r < 4; ++r) {
      std::vector<uint32_t> row(lr::kWords, 0);
      lr::Route rt;
      for (uint32_t k = 0; k < 4; ++k) {
        rt.ids[k] = uint32_t(std::find(used.begin(), used.end(), fx::kMoeIds[r][k]) - used.begin());
        rt.w[k] = lr::as_f32(wbits[r * 4 + k]);
        row[lr::kIds + k] = rt.ids[k];
        row[lr::kW + k] = wbits[r * 4 + k];
      }
      const std::vector<uint16_t> x = bf(vals(true, 40, H, 7, 0.125f, 1.0f, r * H));
      const std::vector<uint16_t> h0 = bf(vals(false, 42, H, 7, 0.125f, 1.0f));   // a residual to fold into
      clrun::Buffer rb(dev, row), xb(dev, x), hb(dev, size_t(4) * I * 2), ob(dev, h0);
      mp.run("lfm_moe_gate_up", {size_t(4 * 48) * 256, 1}, {256, 1}, rb, xb, gb, hb);
      mp.run("lfm_moe_down", {size_t(H / 16) * 128, 1}, {128, 1}, rb, hb, db, ob);
      const lr::MoeOut want = lr::moe(rt, x.data(), gu.data(), dn.data(), h0.data());
      const std::vector<uint16_t> h = hb.read<uint16_t>(), o = ob.read<uint16_t>();
      h_all.insert(h_all.end(), h.begin(), h.end());
      h_ref_all.insert(h_ref_all.end(), want.h.begin(), want.h.end());
      y_all.insert(y_all.end(), o.begin(), o.end());
      y_ref_all.insert(y_ref_all.end(), want.h_res.begin(), want.h_res.end());
    }
    report("lfm_moe_gate_up (4 slots, rank order, int4 experts)", h_all, h_ref_all, 2);
    report("lfm_moe_down + the rank-order fp32 combine + the fold", y_all, y_ref_all, 2);
  }
}

void attention_checks(clrun::Device& dev) {
  const uint32_t TGT = 32;
  clrun::Program p(dev, "src/kernels/lfm2moe/lfm_attn_eager.cl",
                   with(kCtrlDefs, {"M=1", "TGT=32", "Q_HEADS=32", "KV_HEADS=8"}));
  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const uint32_t L = fx::kAttnL[ci], pos = L - 1;
    const std::vector<float> q = vals(true, 70, QN, ci, 0.5f);
    const std::vector<uint16_t> K = bf(vals(true, 71, L * KVN, 9)), V = bf(vals(true, 72, L * KVN, 9));
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = 1;
    const uint32_t stride = L;
    clrun::Buffer cb(dev, ctrl), qb(dev, q), kb(dev, K), vb(dev, V), sb(dev, size_t(QH) * stride * 4);
    clrun::Buffer part(dev, size_t(QH) * TGT * (HD + 2) * 4), ob(dev, QN * 2);
    p.run("lfm_attn_eager_score", {size_t(KVH) * 64, TGT}, {64, 1}, cb, qb, kb, sb, stride);
    std::vector<float> s = sb.read<float>();
    p.run("lfm_attn_eager_softmax", {size_t(QH) * 64, 1}, {64, 1}, cb, sb, stride);
    std::vector<float> pr = sb.read<float>();
    p.run("lfm_attn_eager_pv", {size_t(KVH) * 64, TGT}, {64, 1}, cb, sb, vb, part, stride);
    p.run("lfm_attn_eager_reduce", {size_t(QH) * 64, 1}, {64, 1}, cb, part, ob);
    const std::vector<uint16_t> o = ob.read<uint16_t>();
    std::vector<uint16_t> s0(L), p0(L), oref(QN);
    for (uint32_t i = 0; i < L; ++i) {
      s0[i] = rne(s[i]);
      p0[i] = rne(pr[i]);
    }
    for (uint32_t h = 0; h < QH; ++h) {
      const lr::EagerHead e = lr::attention_eager(q.data() + size_t(h) * HD, K.data(), V.data(), L, h / 4, lr::eager_block(L));
      for (uint32_t dd = 0; dd < HD; ++dd) oref[size_t(h) * HD + dd] = rne(e.o[dd]);
    }
    const std::string what = "lfm_attn_eager at " + std::to_string(L) + " keys";
    report((what + ": head 0 scores vs torch").c_str(), s0, lr::hex16(fx::kAttnS[ci]), 0);
    report((what + ": head 0 probabilities vs torch").c_str(), p0, lr::hex16(fx::kAttnP[ci]), 1);
    report((what + ": output vs lfm2moe_ref.h").c_str(), o, oref, 1);
    report((what + ": output vs torch").c_str(), o, lr::hex16(fx::kAttnO[ci]), 1);
  }
}

// Spec 24c Task 5: the fused arms against their as-built pairs (bitwise is the claim; on this device too).
void arm_checks(clrun::Device& dev) {
  const uint32_t E = lr::kExperts;
  std::mt19937 rng(21);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  {   // router: lfm_res_norm _S1 _R against _S1 + lfm_router
    std::vector<float> sl(H), w(H);
    for (float& v : sl) v = u(rng);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.5f + float(k % 9) * 0.1f);
    std::vector<uint16_t> h(H), rw(size_t(H) * E), tiles(size_t(H) * E);
    for (uint16_t& v : h) v = rne(2.f * u(rng));
    for (uint16_t& v : rw) v = rne(0.05f * u(rng));
    common::repack_bf16_tiled(rw.data(), H, E, tiles.data());
    clrun::Program pn(dev, "src/kernels/lfm2moe/lfm_prep.cl", {"M=1", "K=2048", "NORM_WGS=8", "NORM_EPS=1e-5f", "NORM_S=1"});
    clrun::Program pf(dev, "src/kernels/lfm2moe/lfm_prep.cl",
                      {"M=1", "K=2048", "NORM_WGS=8", "NORM_EPS=1e-5f", "NORM_S=1", "ROUTER_FUSED=1", "ROUTER_E=64"});
    clrun::Program pr(dev, "src/kernels/lfm2moe/lfm_moe.cl", {"M=1", "ROUTER_E=64", "ROUTER_K=2048"});
    clrun::Buffer hb(dev, h), sb(dev, sl), wb(dev, w), tb(dev, tiles), o1(dev, H * 2), x1(dev, H * 2), o2(dev, H * 2),
        x2(dev, H * 2), l1(dev, E * 4), l2(dev, E * 4);
    pn.run("lfm_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, wb, o1, x1);
    pr.run("lfm_router", {4 * 128, 1}, {128, 1}, x1, tb, l1);
    pf.run("lfm_res_norm", {12 * 256, 1}, {256, 1}, hb, sb, wb, o2, x2, tb, l2);
    const bool ok = o1.read<uint16_t>() == o2.read<uint16_t>() && x1.read<uint16_t>() == x2.read<uint16_t>() &&
                    l1.read<uint32_t>() == l2.read<uint32_t>();
    std::printf("lfm2moe_run: %-56s %s\n", "arm router: lfm_res_norm _R vs _S1 + lfm_router", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
  {   // route: lfm_moe _R's gate||up vs lfm_route + lfm_moe_gate_up
    std::vector<float> lg(E), bias(E);
    for (float& v : lg) v = 3.f * u(rng);
    for (float& v : bias) v = 0.05f * u(rng);
    std::vector<uint32_t> gu(size_t(E) * lr::gate_up_block_words());
    std::mt19937 wr(32);
    for (size_t t = 0; t < gu.size() / 136; ++t) {
      for (int i = 0; i < 128; ++i) gu[t * 136 + i] = wr();
      for (int i = 0; i < 8; ++i) gu[t * 136 + 128 + i] = 0x24002400u;   // f16 0.015625 x 2
    }
    std::vector<uint16_t> x(H);
    for (uint16_t& v : x) v = rne(0.5f * u(rng));
    clrun::Program pt(dev, "src/kernels/lfm2moe/lfm_moe.cl", {"M=1", "ROUTE_E=64"});
    const std::vector<std::string> moe = {"M=1", "MOE_E=64", "MOE_K=4", "HIDDEN=2048", "INTER=1536", "UP_KS=4", "DN_KS=2"};
    clrun::Program pm(dev, "src/kernels/lfm2moe/lfm_moe.cl", moe);
    clrun::Program pf(dev, "src/kernels/lfm2moe/lfm_moe.cl", with(moe, {"ROUTE_FUSED=1"}));
    clrun::Buffer lb(dev, lg), bb(dev, bias), gb(dev, gu), xb(dev, x), r1(dev, lr::kWords * 4), r2(dev, lr::kWords * 4),
        h1(dev, size_t(4) * lr::kInter * 2), h2(dev, size_t(4) * lr::kInter * 2);
    pt.run("lfm_route", {64, 1}, {64, 1}, lb, bb, r1);
    pm.run("lfm_moe_gate_up", {size_t(4 * 48) * 256, 1}, {256, 1}, r1, xb, gb, h1);
    pf.run("lfm_moe_gate_up", {size_t(4 * 48) * 256, 1}, {256, 1}, lb, bb, r2, xb, gb, h2);
    const bool ok = r1.read<uint32_t>() == r2.read<uint32_t>() && h1.read<uint16_t>() == h2.read<uint16_t>();
    std::printf("lfm2moe_run: %-56s %s\n", "arm route: lfm_moe _R vs lfm_route + lfm_moe_gate_up", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
  for (uint32_t L : {1u, 0u}) {   // conv: lfm_conv_in on the conv-tile in_proj, both layouts
    // gemv.cl does not build on Apple's OpenCL (as_half without cl_khr_fp16), so the as-built pair here is the host's
    // gemv order (moe_ref::tile_dot over the whole K: gemv.cl's at split-K 1) + lfm_conv on the device; the device's
    // own pair is bitwise on the card (lfm2moe_kernels_test 7). Bar 1 ulp: the host does not contract
    // acc + gacc x scale, a device compiler may.
    const uint32_t K = H, N = 3 * H;
    std::vector<uint32_t> qw(size_t(K / 8) * N);
    std::mt19937 wr(41 + L);
    for (uint32_t& v : qw) v = wr();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    for (uint16_t& v : sc) v = common::f32_to_f16(0.002f + 0.018f * float(wr() % 1000) / 1000.f);
    const loader::HostLinear plain1 = loader::repack_in_proj(qw.data(), sc.data(), K, N, false, 1),
                             tiled = loader::repack_in_proj(qw.data(), sc.data(), K, N, true, L);
    std::vector<float> taps(size_t(H) * 3);
    for (float& v : taps) v = rf(0.5f * u(rng));
    std::vector<uint16_t> ring(size_t(lr::kRing) * H);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);
    clrun::Program pc(dev, "src/kernels/lfm2moe/lfm_conv.cl", with(kCtrlDefs, {"M=1", "HIDDEN=2048", "CONV_S=1"}));
    clrun::Program pi(dev, "src/kernels/lfm2moe/lfm_conv.cl",
                      with(kCtrlDefs, {"M=1", "HIDDEN=2048", "CONV_IN_L=" + std::to_string(L)}));
    std::vector<uint16_t> no_scales(2, 0);
    clrun::Buffer wt(dev, tiled.words), st(dev, L == 0 ? tiled.scales : no_scales), tb(dev, taps), rp(dev, ring),
        rt(dev, ring), yp(dev, H * 2), yt(dev, H * 2);
    int worst = 0;
    for (uint32_t pos = 0; pos < 6; ++pos) {
      std::vector<uint16_t> x(H);
      for (uint16_t& v : x) v = rne(u(rng));
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      std::vector<float> s(N);
      for (uint32_t n = 0; n < N; ++n) s[n] = moe_ref::tile_dot(plain1.words.data(), K, n, x.data(), 0, K / 64);
      clrun::Buffer cb(dev, ctrl), xb(dev, x), sl(dev, s);
      pc.run("lfm_conv", {H, 1}, {256, 1}, cb, sl, tb, rp, yp);
      pi.run("lfm_conv_in", {size_t(H / 16) * 48, 1}, {48, 1}, cb, wt, L == 0 ? st : wt, xb, tb, rt, yt);
      const std::vector<uint16_t> a = yp.read<uint16_t>(), c = yt.read<uint16_t>(), ra = rp.read<uint16_t>(),
                                  rc = rt.read<uint16_t>();
      for (uint32_t i = 0; i < H; ++i) worst = std::max(worst, ulps(a[i], c[i]));
      for (size_t i = 0; i < ra.size(); ++i) worst = std::max(worst, ulps(ra[i], rc[i]));
    }
    const std::string what = "arm conv (layout " + std::to_string(L) + "): lfm_conv_in vs gemv order + lfm_conv";
    std::printf("lfm2moe_run: %-56s %s (worst %d ulp over y and the ring, positions 0..5, bar 1)\n", what.c_str(),
                worst <= 1 ? (worst == 0 ? "exact" : "agrees") : "DISAGREES", worst);
    if (worst > 1) ++failures;
  }
}

// --- spec 24d: the plain-OpenCL prefill kernels (lfm_pf_prep.cl, lfm_pf_conv.cl, lfm_pf_moe.cl) -------------------
// lfm_pf_attn.cl is not run: it is DPAS, 2D block reads and sub-group reductions.
std::vector<float> rnd(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = d(g);
  return v;
}
std::vector<uint16_t> rnd16(size_t n, float lo, float hi, uint32_t seed) { return bf(rnd(n, lo, hi, seed)); }

void prefill_checks(clrun::Device& dev) {
  namespace lp = lfmpfref;
  {   // the chunk's rows out of the tiles
    const uint32_t C = 40;
    const std::vector<uint16_t> tiles = rnd16(size_t(H) * 32, -1.f, 1.f, 5);
    std::vector<uint32_t> ids(C);
    for (uint32_t m = 0; m < C; ++m) ids[m] = (m * 7 + 3) % 32;
    clrun::Program p(dev, "src/kernels/lfm2moe/lfm_pf_prep.cl", {"K=2048", "EMBED_V=128000"});
    clrun::Buffer ib(dev, ids), tb(dev, tiles), hb(dev, size_t(C) * H * 2);
    p.run("lfm_pf_embed_gather", {256, C}, {256, 1}, ib, tb, hb, C);
    std::vector<uint16_t> want;
    for (uint32_t m = 0; m < C; ++m) {
      const std::vector<uint16_t> r = lr::embed_row(tiles.data(), ids[m]);
      want.insert(want.end(), r.begin(), r.end());
    }
    report("lfm_pf_embed_gather (40 rows from the tiles)", hb.read<uint16_t>(), want, 0);
  }
  {   // the conv over a chunk: the seed (a prior launch), then the conv; plain and conv-tile in_proj
    std::vector<float> taps(size_t(H) * 3);
    {
      const std::vector<uint16_t> t = rnd16(taps.size(), -0.6f, 0.6f, 12);
      for (size_t i = 0; i < t.size(); ++i) taps[i] = f32(t[i]);
    }
    const std::vector<uint16_t> garbage = rnd16(size_t(lp::kRing) * H, -50.f, 50.f, 13);
    for (int tiled = 0; tiled < 2; ++tiled) {
      clrun::Program p(dev, "src/kernels/lfm2moe/lfm_pf_conv.cl",
                       with(kCtrlDefs, {"HIDDEN=2048", "PF_C=2048", std::string("CONV_TILES=") + (tiled ? "1" : "0")}));
      for (const auto& [p0, C] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 2}, {5, 17}, {30, 64}, {4097, 300}}) {
        const std::vector<float> plain = rnd(size_t(C) * 3 * H, -3.f, 3.f, 20 + p0 + C);
        std::vector<float> in = plain;
        if (tiled)
          for (uint32_t t = 0; t < C; ++t)
            for (uint32_t c = 0; c < H; ++c)
              for (uint32_t part = 0; part < 3; ++part)
                in[size_t(t) * 3 * H + lp::conv_col(c, part, H, true)] = plain[size_t(t) * 3 * H + lp::conv_col(c, part, H, false)];
        std::vector<uint32_t> ctl(32, 0);
        ctl[0] = p0;
        ctl[1] = C;
        clrun::Buffer cb(dev, ctl), ib(dev, in), tb(dev, taps), rb(dev, garbage), sb(dev, size_t(2) * H * 2),
            yb(dev, size_t(C) * H * 2);
        p.run("lfm_pf_conv_seed", {H}, {256}, cb, rb, sb);
        p.run("lfm_pf_conv", {H, (C + 15) / 16}, {256, 1}, cb, ib, sb, tb, rb, yb);
        std::vector<uint16_t> ring = garbage;
        const std::vector<uint16_t> seed = lp::conv_seed(ring.data(), p0);
        const std::vector<uint16_t> y = lp::conv_chunk(in.data(), seed.data(), taps.data(), ring.data(), p0, C, H, tiled);
        const std::string what = std::string("lfm_pf_conv") + (tiled ? " _T" : "") + " (p0 " + std::to_string(p0) +
                                 ", C " + std::to_string(C) + "): ";
        report((what + "seed").c_str(), sb.read<uint16_t>(), seed, 0);
        report((what + "y").c_str(), yb.read<uint16_t>(), y, 0);
        report((what + "the ring").c_str(), rb.read<uint16_t>(), ring, 0);
      }
    }
  }
  {   // the MoE block's plain entry points over C = 64: sort, gather, two dequant batches, the combine + fold
    const uint32_t C = 64, E = lr::kExperts, I = lr::kInter, T = lp::tmax(lp::kC);
    std::mt19937 g(40);
    std::uniform_real_distribution<float> dl(-4.f, 4.f), db(-0.05f, 0.05f);
    std::vector<float> bias(E);
    for (float& b : bias) b = db(g);
    std::vector<uint32_t> route(size_t(C) * lr::kWords);
    for (uint32_t t = 0; t < C; ++t) {
      std::vector<float> lg(E);
      for (float& v : lg) v = dl(g);
      const lr::Route r = lr::route(lg.data(), bias.data());
      std::copy(r.row, r.row + lr::kWords, route.begin() + size_t(t) * lr::kWords);
    }
    const lp::Sorted st = lp::sort(route.data(), C);
    clrun::Program p(dev, "src/kernels/lfm2moe/lfm_pf_moe.cl",
                     {"PF_E=64", "PF_K=4", "PF_WG=64", "PF_D=2048", "PF_INTER=1536", "TM=32", "KC=2048"});
    clrun::Buffer rb(dev, route), hdr(dev, size_t(lp::hdr_words()) * 4), tiles(dev, size_t(T) * 2 * 4),
        row_tok(dev, size_t(T) * lp::kTm * 4), pair_row(dev, size_t(C) * 4 * 4);
    const uint32_t tm = lp::tmax(C);
    p.run("lfm_pf_sort", {64}, {64}, rb, hdr, tiles, row_tok, pair_row, C, tm);
    const std::vector<uint32_t> hd = hdr.read<uint32_t>();
    report("lfm_pf_sort: the header", std::vector<uint32_t>(hd.begin(), hd.begin() + lp::kHdrCount + E),
           std::vector<uint32_t>(st.hdr.begin(), st.hdr.begin() + lp::kHdrCount + E), 0);
    const std::vector<uint32_t> tl = tiles.read<uint32_t>();
    report("lfm_pf_sort: the tile table", std::vector<uint32_t>(tl.begin(), tl.begin() + size_t(tm) * 2), st.tiles, 0);
    const std::vector<uint32_t> rt = row_tok.read<uint32_t>();
    report("lfm_pf_sort: row_tok", std::vector<uint32_t>(rt.begin(), rt.begin() + st.rows_used()),
           std::vector<uint32_t>(st.row_tok.begin(), st.row_tok.begin() + st.rows_used()), 0);
    report("lfm_pf_sort: pair_row", pair_row.read<uint32_t>(), st.pair_row, 0);
    const std::vector<uint16_t> x = rnd16(size_t(C) * H, -1.f, 1.f, 41);
    clrun::Buffer xb(dev, x), xg(dev, size_t(st.rows_used()) * H * 2);
    p.run("lfm_pf_gather", {size_t(st.rows_used()) * 64}, {64}, xb, hdr, row_tok, xg);
    report("lfm_pf_gather: the sorted A rows", xg.read<uint16_t>(), lp::gather(st, x.data()), 0);
    std::vector<uint32_t> gu(size_t(E) * lr::gate_up_block_words()), dn(size_t(E) * lr::down_block_words());
    {
      std::mt19937 rg(42);
      std::uniform_real_distribution<float> ds(0.01f, 0.05f);
      for (std::vector<uint32_t>* blocks : {&gu, &dn})
        for (size_t t = 0; t < blocks->size() / 136; ++t) {
          uint32_t* tile = blocks->data() + t * 136;
          for (int i = 0; i < 128; ++i) tile[i] = rg();
          for (int i = 0; i < 8; ++i)
            tile[128 + i] = uint32_t(common::f32_to_f16(ds(rg))) | (uint32_t(common::f32_to_f16(ds(rg))) << 16);
        }
    }
    clrun::Buffer gb(dev, gu), dnb(dev, dn);
    for (const auto& [b0, b1] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 4}, {60, 64}}) {
      clrun::Buffer og(dev, size_t(b1 - b0) * H * 2 * I * 2), od(dev, size_t(b1 - b0) * I * H * 2);
      p.run("lfm_pf_dequant_gu", {size_t(2 * I / 16) * 16, H / 64, b1 - b0}, {16, 1, 1}, gb, hdr, og, b0, b1);
      p.run("lfm_pf_dequant_dn", {size_t(H / 16) * 16, I / 64, b1 - b0}, {16, 1, 1}, dnb, hdr, od, b0, b1);
      const std::vector<uint16_t> ogv = og.read<uint16_t>(), odv = od.read<uint16_t>();
      for (uint32_t e = b0; e < b1; ++e) {
        const std::string tag = " (expert " + std::to_string(e) + (st.hdr[lp::kHdrCount + e] ? "" : ", unrouted: skipped") + ")";
        const size_t gn = size_t(H) * 2 * I, dnn = size_t(I) * H;
        const std::vector<uint16_t> want_g = st.hdr[lp::kHdrCount + e] ? lp::dequant_gu(gu.data(), e) : std::vector<uint16_t>(gn, 0);
        const std::vector<uint16_t> want_d = st.hdr[lp::kHdrCount + e] ? lp::dequant_dn(dn.data(), e) : std::vector<uint16_t>(dnn, 0);
        report(("lfm_pf_dequant_gu" + tag).c_str(),
               std::vector<uint16_t>(ogv.begin() + (e - b0) * gn, ogv.begin() + (e - b0 + 1) * gn), want_g, 0);
        report(("lfm_pf_dequant_dn" + tag).c_str(),
               std::vector<uint16_t>(odv.begin() + (e - b0) * dnn, odv.begin() + (e - b0 + 1) * dnn), want_d, 0);
      }
    }
    const std::vector<uint16_t> y = rnd16(size_t(st.rows_used()) * H, -2.f, 2.f, 43), h0 = rnd16(size_t(C) * H, -2.f, 2.f, 44);
    clrun::Buffer yb(dev, y), hb(dev, h0);
    p.run("lfm_pf_combine", {H, C}, {256, 1}, rb, pair_row, yb, hb, C);
    std::vector<uint16_t> hh = h0;
    lp::combine(route.data(), st, y.data(), hh.data(), C);
    report("lfm_pf_combine: rank order, rounded products, the fold", hb.read<uint16_t>(), hh, 0);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("lfm2moe_run: device %s\n", dev.name().c_str());
    prep_checks(dev);
    conv_checks(dev);
    moe_checks(dev);
    attention_checks(dev);
    arm_checks(dev);
    prefill_checks(dev);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lfm2moe_run: %s\n", e.what());
    return 1;
  }
  if (failures) {
    std::printf("lfm2moe_run: %d check(s) DISAGREE\n", failures);
    return 1;
  }
  std::printf("lfm2moe_run: every check agrees\n");
  return 0;
}
