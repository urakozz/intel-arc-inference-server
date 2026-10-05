// k2_run - spec 18b's portable K2-Horizon kernels on the Mac's OpenCL GPU against
// tests/kernels/k2_ref.h (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   k2_run        no arguments; K2's real shapes (hidden 2560, 100 + 1 experts x 768,
//                 64 value experts x 1024, 32 q / 8 kv heads x 128)
//
// What runs: src/kernels/k2/k2_prep.cl (k2_norm_finish after prep.cl's prep_res_fold,
// k2_silu_mul, both k2_attn_prep builds) and src/kernels/k2/k2_moe.cl (k2_route - the MoE
// router with its 28 padded lanes and MoVA's over two slices of the fused row -,
// k2_moe_gate_up, k2_moe_down, k2_mova_value), with the CMake block's defines except ONE:
// k2_moe_down runs at DN_KS 1 here (144 lanes) because the Mac's GPUs cap a work-group at
// 256 and the B70 build's DN_KS 2 is 288. k2_attn.cl is not run: its decode is shuffles,
// sub-group reductions and 2D block reads, which the emulation leaves undeclared on purpose.
//
// Bars: the exp-free chains (the norm, the RoPE prep) bit-exact; the rest within 2 bf16
// ulps (OpenCL's exp, the GEMVs' fma contraction), the route ids exact.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "kernels/k2_ref.h"
#include "kernels/prep_ref.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"

namespace {

using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}
std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(rng);
  return v;
}
std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) | (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}

int failures = 0;
void report(const char* what, int worst, int bar, size_t n) {
  const bool ok = worst <= bar;
  std::printf("  %-44s %zu values, worst %d bf16 ulps (bar %d) %s\n", what, n, worst, bar,
              ok ? "agrees" : "DISAGREES");
  if (!ok) ++failures;
}
void report_bool(const char* what, bool ok) {
  std::printf("  %-44s %s\n", what, ok ? "agrees" : "DISAGREES");
  if (!ok) ++failures;
}

}  // namespace

int main() {
  const model::K2Desc& m = model::k2();
  const uint32_t H = m.hidden;
  try {
    clrun::Device dev;
    std::printf("k2_run on %s (INDICATIVE: Mac OpenCL, emulated sub-groups)\n", dev.name().c_str());

    // --- the grouped norm: prep_res_fold (SP4) + k2_norm_finish ------------------------------
    {
      const uint32_t G = 20, S = 4;
      std::vector<uint16_t> resid = random_bf16(H, -1.f, 1.f, 1);
      for (uint32_t k = 0; k < H / 2; ++k) resid[k] = rne(f32(resid[k]) * 0.01f);
      const std::vector<float> part = random_f32(size_t(S) * H, -0.5f, 0.5f, 2);
      std::vector<float> w(H);
      for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
      std::vector<uint16_t> r_ref = resid, x_ref(H);
      std::vector<float> ss_ref(G);
      prep_ref::res_fold(part.data(), r_ref.data(), ss_ref.data(), 1, H, S, G);
      k2_ref::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), 1, H, G, 2);
      clrun::Program fold(dev, "src/kernels/prep.cl", {"M=1", "K=2560", "S_PREV=4", "FOLD_G=20"});
      clrun::Program norm(dev, "src/kernels/k2/k2_prep.cl",
                          {"M=1", "K=2560", "NORM_G=20", "NORM_WGS=20", "NORM_GROUPS=2"});
      clrun::Buffer pb(dev, part), rb(dev, resid), wb(dev, w), ss(dev, G * 4), xb(dev, H * 2);
      fold.run("prep_res_fold", {20 * 256, 1}, {256, 1}, pb, rb, ss);
      norm.run("k2_norm_finish", {20 * 256, 1}, {256, 1}, ss, rb, wb, xb);
      const std::vector<uint16_t> r = rb.read<uint16_t>(), x = xb.read<uint16_t>();
      int worst = 0;
      for (uint32_t k = 0; k < H; ++k) worst = std::max(worst, std::max(ulps(r[k], r_ref[k]), ulps(x[k], x_ref[k])));
      report("grouped norm (resid, x)", worst, 0, 2 * H);
    }
    // --- SiLU x up ---------------------------------------------------------------------------
    {
      const uint32_t I = 6144, S = 4;
      const std::vector<float> part = random_f32(size_t(S) * 2 * I, -1.f, 1.f, 3);
      std::vector<uint16_t> x_ref(I);
      k2_ref::silu_mul(part.data(), x_ref.data(), 1, I, S);
      clrun::Program p(dev, "src/kernels/k2/k2_prep.cl", {"M=1", "SILU_I=6144", "SILU_S=4"});
      clrun::Buffer pb(dev, part), xb(dev, I * 2);
      p.run("k2_silu_mul", {2 * 256, 1}, {256, 1}, pb, xb);
      const std::vector<uint16_t> x = xb.read<uint16_t>();
      int worst = 0;
      for (uint32_t k = 0; k < I; ++k) worst = std::max(worst, ulps(x[k], x_ref[k]));
      report("silu x up (I 6144, S 4)", worst, 2, I);
    }
    // --- the attention prep, both builds -------------------------------------------------------
    const std::vector<float> rope = loader::k2_rope_table(m, 64);
    for (int dense = 1; dense >= 0; --dense) {
      const uint32_t N = dense ? m.attn_dense_n() : m.attn_sparse_n(), S = 2, pos = 37;
      const std::vector<float> part = random_f32(size_t(S) * N, -1.f, 1.f, 10 + dense);
      const k2_ref::Prep want = k2_ref::attn_prep(part.data(), 1, 0, N, S, rope.data() + size_t(pos) * 128,
                                                  m.q_heads, m.kv_heads, 128, dense);
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      clrun::Program p(dev, "src/kernels/k2/k2_prep.cl",
                       {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                        "M=1", "QKV_S=2", "Q_HEADS=32", "KV_HEADS=8", "HD=128",
                        "QKV_N=" + std::to_string(N), std::string("V_FROM_PARTIALS=") + (dense ? "1" : "0")});
      clrun::Buffer cb(dev, ctrl), pb(dev, part), rb(dev, rope), aq(dev, m.q_n() * 4), ag(dev, m.q_n() * 4);
      clrun::Buffer kk(dev, size_t(64) * m.kv_n() * 2), kv(dev, size_t(64) * m.kv_n() * 2);
      p.run("k2_attn_prep", {size_t(m.q_heads + m.kv_heads) * 128, 1}, {128, 1}, cb, pb, rb, aq, ag, kk, kv);
      const std::vector<float> q = aq.read<float>(), g = ag.read<float>();
      const std::vector<uint16_t> kr = kk.read<uint16_t>(), vr = kv.read<uint16_t>();
      bool ok = q == want.q && g == want.gate &&
                std::equal(want.k.begin(), want.k.end(), kr.begin() + size_t(pos) * m.kv_n());
      if (dense) ok = ok && std::equal(want.v.begin(), want.v.end(), vr.begin() + size_t(pos) * m.kv_n());
      report_bool(dense ? "attn prep, dense (q, gate, k, v bit-exact)" : "attn prep, MoVA (q, gate, k bit-exact)", ok);
    }
    // --- the routers -----------------------------------------------------------------------------
    {
      const std::vector<float> bias = random_f32(128, -0.05f, 0.05f, 20);
      std::vector<float> lg = random_f32(128, -3.f, 3.f, 21);
      for (uint32_t e = 100; e < 128; ++e) lg[e] = 0.0f;   // the GEMV's zero rows
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "ROUTE_E=100", "ROUTE_K=8", "ROUTE_WG=128", "ROUTE_LN=128",
                        "ROUTE_LOFF=0", "ROUTE_LS=1", "ROUTE_SCALE=2.5f"});
      clrun::Buffer lb(dev, lg), bb(dev, bias), rt(dev, 32 * 4);
      p.run("k2_route", {128, 1}, {128, 1}, lb, bb, rt);
      const k2_ref::Route want = k2_ref::route(lg.data(), 1, 0, 128, 0, 1, bias.data(), 100, 8, 2.5f);
      const std::vector<uint32_t> row = rt.read<uint32_t>();
      bool ok = true;
      int worst = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        ok = ok && row[j] == want.ids[j];
        float w;
        std::memcpy(&w, &row[8 + j], 4);
        worst = std::max(worst, ulps(rne(w), rne(want.w[j])));
      }
      report_bool("MoE route: ids (ascending)", ok);
      report("MoE route: weights", worst, 1, 8);
      // The padded lanes at +80, the real scores far below: never selected.
      for (uint32_t e = 0; e < 100; ++e) lg[e] = -30.0f - float(e % 5);
      for (uint32_t e = 100; e < 128; ++e) lg[e] = 80.0f;
      std::vector<float> nb(128, -0.9f);
      lb.write(lg.data(), lg.size() * 4);
      bb.write(nb.data(), nb.size() * 4);
      p.run("k2_route", {128, 1}, {128, 1}, lb, bb, rt);
      const std::vector<uint32_t> r2 = rt.read<uint32_t>();
      bool pad_ok = true;
      for (uint32_t j = 0; j < 8; ++j) pad_ok = pad_ok && r2[j] < 100;
      report_bool("MoE route: the 28 padded lanes never selected", pad_ok);
    }
    {
      const uint32_t N = 9280, S = 2;
      const std::vector<float> part = random_f32(size_t(S) * N, -2.f, 2.f, 22);
      const std::vector<float> bias = random_f32(64, -0.05f, 0.05f, 23);
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "ROUTE_E=64", "ROUTE_K=4", "ROUTE_WG=64", "ROUTE_LN=9280",
                        "ROUTE_LOFF=9216", "ROUTE_LS=2", "ROUTE_SCALE=2.5f"});
      clrun::Buffer pb(dev, part), bb(dev, bias), rt(dev, 32 * 4);
      p.run("k2_route", {64, 1}, {64, 1}, pb, bb, rt);
      const k2_ref::Route want = k2_ref::route(part.data(), 1, 0, N, 9216, S, bias.data(), 64, 4, 2.5f);
      const std::vector<uint32_t> row = rt.read<uint32_t>();
      bool ok = true;
      for (uint32_t j = 0; j < 4; ++j) ok = ok && row[j] == want.ids[j];
      report_bool("MoVA route over two slices of the fused row", ok);
    }
    // --- the MoE block (DN_KS 1 here, see the header) ---------------------------------------------
    {
      const moe_ref::Shape ms = k2_ref::moe_shape(100, 8, H, 768);
      const std::vector<uint32_t> gu = random_blocks(ms.gate_up_words() * 101, 30);
      const std::vector<uint32_t> dn = random_blocks(ms.down_words() * 101, 31);
      k2_ref::Route forced;
      const uint32_t ids[8] = {0, 3, 17, 42, 63, 77, 98, 99};
      const float ws[8] = {0.5f, 0.375f, 0.3125f, 0.25f, 0.4375f, 0.1875f, 0.25f, 0.1875f};
      std::vector<uint32_t> row(32, 0);
      for (uint32_t j = 0; j < 8; ++j) {
        forced.ids[j] = ids[j];
        forced.w[j] = ws[j];
        row[j] = ids[j];
        std::memcpy(&row[8 + j], &ws[j], 4);
      }
      const std::vector<uint16_t> xs = random_bf16(H, -0.1f, 0.1f, 32);
      const std::vector<uint16_t> resid0 = random_bf16(H, -2.f, 2.f, 33);
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "MOE_E=100", "MOE_K=8", "HIDDEN=2560", "INTER=768", "UP_KS=4", "DN_KS=1"});
      clrun::Buffer rb(dev, row), xb(dev, xs), gb(dev, gu), db(dev, dn), hb(dev, size_t(9) * 768 * 2), sb(dev, resid0);
      p.run("k2_moe_gate_up", {size_t(9) * (2 * 768 / 16 / 4) * 256, 1}, {256, 1}, rb, xb, gb, hb);
      const std::vector<uint16_t> hd = hb.read<uint16_t>();
      const std::vector<uint16_t> hr = moe_ref::gate_up(k2_ref::as_moe(forced, 8), xs.data(), gu.data(), ms, 4);
      int worst = 0;
      for (size_t i = 0; i < hd.size(); ++i) worst = std::max(worst, ulps(hd[i], hr[i]));
      report("MoE gate||up (8 slots + shared block 100)", worst, 2, hd.size());
      p.run("k2_moe_down", {size_t(H / 16) * 144, 1}, {144, 1}, rb, hb, db, sb);
      const std::vector<uint16_t> rd = sb.read<uint16_t>();
      const std::vector<uint16_t> rr = k2_ref::moe_down(forced, hd, dn.data(), ms, resid0.data(), 1);
      worst = 0;
      for (uint32_t n = 0; n < H; ++n) worst = std::max(worst, ulps(rd[n], rr[n]));
      report("MoE down + ascending-id combine + residual", worst, 2, H);

      // --- MoVA's value experts ------------------------------------------------------------------
      const uint32_t N = 1024;
      const size_t vblk = size_t(N / 16) * (H / 64) * 136;
      const std::vector<uint32_t> vw = random_blocks(vblk * 64, 40);
      k2_ref::Route vr;
      const uint32_t vids[4] = {2, 31, 32, 63};
      const float vws[4] = {0.625f, 0.75f, 0.5f, 0.625f};
      std::vector<uint32_t> vrow(32, 0);
      for (uint32_t j = 0; j < 4; ++j) {
        vr.ids[j] = vids[j];
        vr.w[j] = vws[j];
        vrow[j] = vids[j];
        std::memcpy(&vrow[8 + j], &vws[j], 4);
      }
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = 5;
      ctrl[1] = 1;
      clrun::Program pv(dev, "src/kernels/k2/k2_moe.cl",
                        {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                         "M=1", "MOVA_E=64", "MOVA_K=4", "MOVA_D=2560", "MOVA_N=1024", "MOVA_KS=4"});
      clrun::Buffer cb(dev, ctrl), vrb(dev, vrow), vwb(dev, vw), kv(dev, size_t(8) * N * 2);
      pv.run("k2_mova_value", {size_t(N / 16) * 256, 1}, {256, 1}, cb, vrb, xb, vwb, kv);
      const std::vector<uint16_t> v = kv.read<uint16_t>();
      const std::vector<uint16_t> want = k2_ref::mova_value(vr, xs.data(), vw.data(), H, N, 4, 4);
      worst = 0;
      for (uint32_t n = 0; n < N; ++n) worst = std::max(worst, ulps(v[size_t(5) * N + n], want[n]));
      report("MoVA value experts into V[pos 5]", worst, 2, N);
    }
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "k2_run: %s\n", e.what());
    return 1;
  }
  if (failures) {
    std::printf("DISAGREES with the host reference in %d check(s) (indicative: read the kernel "
                "before the box)\n", failures);
    return 1;
  }
  std::printf("agrees with the host reference (indicative only; the B70 is the real test)\n");
  return 0;
}
