// moe_run - src/kernels/moe.cl (spec 15c's decode MoE block) on the Mac's OpenCL GPU at
// Ornith's shape (256 experts, top-8, hidden 2048, expert 512), for spec 15e's M > 1 rows
// (the MTP verify lists): INDICATIVE ONLY (clrun.h says what this can and cannot show).
//
//   moe_run
//
// moe.cl builds as OpenCL 1.2 without the Intel subgroup extension (plain loads of the
// same elements, every cross-lane step through SLM), so all three kernels run here with
// the CMake block's defines except ONE: moe_down at DN_KS 1 (144 lanes), because the Mac's
// GPUs cap a work-group at 256 and the B70 build's DN_KS 2 is 288 (k2_run's arrangement).
//
//   1. M = 1 against tests/kernels/moe_ref.h: moe_route's ids exact (except a near-tie at
//      the top-8 boundary), weights / gate within a bf16 ulp; moe_gate_up and moe_down
//      within 2 bf16 ulps (the host and the device round the GEMV's sums differently);
//   2. M = 4 (the verify list's largest), rows distinct, one with exact ties: every row's
//      route row, h and folded resid BITWISE the M = 1 binary's on that row alone, and the
//      same with the rows reversed - the property spec 8's M2 needs (moe_m_test pins it
//      on the card).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "kernels/moe_ref.h"

namespace {

const moe_ref::Shape kS = moe_ref::ornith_shape();
constexpr uint32_t kUpKs = 4, kDnKs = 1;   // DN_KS 1 here (the header)
int g_bad = 0;

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
void expect(bool ok, const char* what) {
  std::printf("  %s: %s\n", ok ? "ok  " : "BAD ", what);
  if (!ok) ++g_bad;
}

std::vector<std::string> defines(uint32_t M) {
  return {"M=" + std::to_string(M), "EXPERTS=256", "TOP_K=8", "HIDDEN=2048", "INTER=512",
          "ROUTER_N=272", "UP_KS=" + std::to_string(kUpKs), "DN_KS=" + std::to_string(kDnKs)};
}

std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {   // moe_test's
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) |
                      (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}

struct Out {
  std::vector<uint32_t> route;
  std::vector<uint16_t> h, resid;
};

// The capture's three launches (capture.cc moe_block) over `M` rows.
Out run(clrun::Device& dev, clrun::Program& p, uint32_t M, const std::vector<float>& logits,
        const std::vector<uint16_t>& x, const std::vector<uint16_t>& resid, clrun::Buffer& gu,
        clrun::Buffer& dn) {
  clrun::Buffer lb(dev, logits), xb(dev, x), rb(dev, resid);
  clrun::Buffer route(dev, size_t(M) * moe_ref::kWords * 4);
  clrun::Buffer h(dev, size_t(M) * kS.slots() * kS.inter * 2);
  const size_t up_groups = size_t(kS.slots()) * (2 * kS.inter / 16 / 4);
  const size_t dn_wg = size_t(16) * kS.slots() * kDnKs;
  p.run("moe_route", {kS.experts, M}, {kS.experts, 1}, lb, route);
  p.run("moe_gate_up", {up_groups * 64 * kUpKs, M}, {64 * kUpKs, 1}, route, xb, gu, h);
  p.run("moe_down", {size_t(kS.hidden / 16) * dn_wg, M}, {dn_wg, 1}, route, h, dn, rb);
  return {route.read<uint32_t>(), h.read<uint16_t>(), rb.read<uint16_t>()};
}

template <class T>
std::vector<T> row_of(const std::vector<T>& v, size_t width, uint32_t m) {
  return std::vector<T>(v.begin() + m * width, v.begin() + (m + 1) * width);
}

moe_ref::Route as_route(const std::vector<uint32_t>& row) {
  moe_ref::Route r;
  for (uint32_t k = 0; k < kS.top_k; ++k) {
    r.ids[k] = row[moe_ref::kIds + k];
    std::memcpy(&r.w[k], &row[moe_ref::kWeights + k], 4);
  }
  std::memcpy(&r.sg, &row[moe_ref::kSharedGate], 4);
  return r;
}

}  // namespace

int main() {
  clrun::Device dev;
  std::printf("moe_run on %s (moe.cl at Ornith's shape, DN_KS 1)\n", dev.name().c_str());
  const std::vector<uint32_t> guw = random_blocks(kS.gate_up_words() * (kS.experts + 1), 3);
  const std::vector<uint32_t> dnw = random_blocks(kS.down_words() * (kS.experts + 1), 4);
  clrun::Buffer gu(dev, guw), dn(dev, dnw);
  std::mt19937 rng(1505);
  std::uniform_real_distribution<float> ld(-3.f, 3.f), xd(-0.1f, 0.1f), rd(-2.f, 2.f);
  constexpr uint32_t kM = 4;
  const uint32_t LW = kS.router_n, RW = moe_ref::kWords, HW = kS.slots() * kS.inter;
  std::vector<float> logits(size_t(kM) * LW);
  for (float& v : logits) v = ld(rng);
  for (uint32_t e : {9u, 40u, 41u, 77u, 130u, 131u, 200u, 201u, 250u}) logits[size_t(1) * LW + e] = 5.f;
  std::vector<uint16_t> x(size_t(kM) * kS.hidden), resid(size_t(kM) * kS.hidden);
  for (uint16_t& v : x) v = clrun::f32_to_bf16(xd(rng));
  for (uint16_t& v : resid) v = clrun::f32_to_bf16(rd(rng));

  clrun::Program p1(dev, "src/kernels/moe.cl", defines(1));
  std::vector<Out> one;
  for (uint32_t m = 0; m < kM; ++m)
    one.push_back(run(dev, p1, 1, row_of(logits, LW, m), row_of(x, kS.hidden, m),
                      row_of(resid, kS.hidden, m), gu, dn));

  // 1. M = 1 against the host reference, row by row.
  bool ids_ok = true, w_ok = true;
  int h_worst = 0, r_worst = 0;
  for (uint32_t m = 0; m < kM; ++m) {
    const moe_ref::Route want = moe_ref::route(logits.data() + size_t(m) * LW, kS);
    const moe_ref::Route got = as_route(one[m].route);
    const float pk = want.p[kS.top_k - 1];
    const bool near_tie = pk != want.p9 && pk - want.p9 <= 1e-5f * want.p9;
    for (uint32_t k = 0; k < kS.top_k; ++k) {
      if (near_tie && k == kS.top_k - 1) continue;
      ids_ok = ids_ok && got.ids[k] == want.ids[k];
      w_ok = w_ok && ulps(moe_ref::rne(got.w[k]), moe_ref::rne(want.w[k])) <= 1;
    }
    w_ok = w_ok && ulps(moe_ref::rne(got.sg), moe_ref::rne(want.sg)) <= 1;
    // gate||up and down against the reference on the DEVICE's route (so a near-tie id
    // does not count against the GEMVs) and, for down, the device's h.
    const std::vector<uint16_t> hr =
        moe_ref::gate_up(got, x.data() + size_t(m) * kS.hidden, guw.data(), kS, kUpKs);
    for (size_t i = 0; i < hr.size(); ++i) h_worst = std::max(h_worst, ulps(one[m].h[i], hr[i]));
    const std::vector<uint16_t> rr = moe_ref::down(got, one[m].h, dnw.data(), kS,
                                                   resid.data() + size_t(m) * kS.hidden, kDnKs);
    for (uint32_t n = 0; n < kS.hidden; ++n) r_worst = std::max(r_worst, ulps(one[m].resid[n], rr[n]));
  }
  const uint32_t tie[8] = {9, 40, 41, 77, 130, 131, 200, 201};
  bool tie_ok = true;
  for (uint32_t k = 0; k < 8; ++k) tie_ok = tie_ok && one[1].route[moe_ref::kIds + k] == tie[k];
  expect(ids_ok, "M = 1 moe_route: ids = moe_ref's (near-tie boundary slot excepted)");
  expect(tie_ok, "M = 1 moe_route: exact ties take the lower ids, in id order");
  expect(w_ok, "M = 1 moe_route: weights and shared gate within 1 bf16 ulp");
  std::printf("  moe_gate_up worst %d ulps, moe_down worst %d ulps (bar 2)\n", h_worst, r_worst);
  expect(h_worst <= 2 && r_worst <= 2, "M = 1 moe_gate_up / moe_down within 2 bf16 ulps of moe_ref");

  // 2. M = 4: every row bitwise the M = 1 binary's on that row, in order and reversed.
  clrun::Program p4(dev, "src/kernels/moe.cl", defines(kM));
  for (bool reversed : {false, true}) {
    std::vector<float> lg;
    std::vector<uint16_t> xs, rs;
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < kM; ++i) order.push_back(reversed ? kM - 1 - i : i);
    for (uint32_t m : order) {
      const auto a = row_of(logits, LW, m);
      const auto b = row_of(x, kS.hidden, m);
      const auto c = row_of(resid, kS.hidden, m);
      lg.insert(lg.end(), a.begin(), a.end());
      xs.insert(xs.end(), b.begin(), b.end());
      rs.insert(rs.end(), c.begin(), c.end());
    }
    const Out got = run(dev, p4, kM, lg, xs, rs, gu, dn);
    bool same = true;
    for (uint32_t i = 0; i < kM; ++i) {
      const uint32_t m = order[i];
      same = same && row_of(got.route, RW, i) == one[m].route && row_of(got.h, HW, i) == one[m].h &&
             row_of(got.resid, kS.hidden, i) == one[m].resid;
    }
    expect(same, reversed ? "M = 4, rows reversed: every row bitwise M = 1's on that row"
                          : "M = 4: every row's route, h and resid bitwise M = 1's on that row");
  }
  std::printf("moe_run: %s\n", g_bad ? "DISAGREES" : "agrees");
  return g_bad ? 1 : 0;
}
