// kolibri_run - spec 20c's portable Kolibri-1 kernels on the Mac's OpenCL GPU against
// tests/kernels/kolibri_ref.h and torch's own outputs (tests/kernels/kolibri_fixture.h)
// (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   kolibri_run      no arguments; Kolibri's real shapes (hidden 2560, 48 / 4 heads x 128, 384 experts
//                    top 6 x 512 + the bf16 shared expert, window 513 on a 4096-slot ring)
//
// What runs, with the CMake block's defines: src/kernels/kolibri/kol_prep.cl (kol_norm_finish after
// prep.cl's prep_res_fold SP0, kol_post_add after prep_res_fold SP4 _Z, kol_attn_prep sliding (ring)
// and full at S 2), src/kernels/kolibri/kol_moe.cl (kol_route on 256 lanes x 2 - random and the
// fixture's four rows -, kol_moe_gate_up at UP_KS 4 = 256 lanes, kol_moe_down at DN_KS 2 = 224
// lanes: both within the Mac's 256 cap, so the B70 build's defines run unchanged) and
// src/kernels/kolibri/kol_attn_eager.cl (both builds; the fixture's five cases and a wrapped ring).
// kol_attn.cl is not run: its decode is shuffles, sub-group reductions and 2D block reads.
//
// Spec 20d (prefill_checks, against tests/kernels/kolibri_pf_ref.h): kol_pf_moe.cl - kol_pf_sort on
// 256 lanes at C = 2048 (random, tied at the cut, all to 6 experts, the tile bound reached), kol_pf_gather,
// kol_pf_dequant_gu / _dn (a routed block, an expert with no row skipped, block 384 the shared expert's
// bf16 tiles), kol_pf_moe_combine; kol_pf_linear.cl's bf16 slab (o_proj's widths and a zero-padded
// tail); 18c's k2_pf_dequant_slab at q||k||v's width (the int4 arm); prefill/pf_prep.cl's pf_res_fold
// (SP0, _Z) with kol_prep.cl at M = 2048 (the norm, the sandwich) over a whole chunk - all exact, except
// that over 2048 rows a handful of norm / post-add values move with Apple's 1/sqrt (not correctly
// rounded here): each must be the host chain at an rstd within 4 fp32 ulps (measured on this Mac: 11 and
// 16 of 5.2M). Not run: kol_pf_attn.cl (DPAS, 2D block reads, sub-group reductions) and pf_moe_gemm
// (DPAS): box only.
//
// Bars: the exp-free chains (the norm, the sandwich, the prep, the eager scores) bitwise; the
// route ids exact; the rest within 2 bf16 ulps (OpenCL's exp, the GEMVs' fma contraction, and Apple's
// compiler is not told -cl-fp32-correctly-rounded-divide-sqrt).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/kolibri_fixture.h"
#include "kernels/kolibri_pf_ref.h"
#include "kernels/kolibri_ref.h"
#include "kernels/prep_ref.h"
#include "loader/kolibri1_repack.h"
#include "model/kolibri1.h"

namespace {

namespace kr = kolibri_ref;
namespace fx = kolibri_fixture;
using kr::f32;
using kr::rf;
using kr::rne;

const std::vector<std::string> kCtrlDefs = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

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
  std::printf("kolibri_run: %-46s %zu values, worst %d bf16 ulps (bar %d) %s\n", what, n, worst, bar,
              ok ? (worst == 0 ? "exact" : "agrees") : "DISAGREES");
  if (!ok) ++failures;
}
void report_bool(const char* what, bool ok) {
  std::printf("kolibri_run: %-46s %s\n", what, ok ? "exact" : "DISAGREES");
  if (!ok) ++failures;
}

const uint32_t H = kr::kHidden, HD = kr::kHd, QH = kr::kQHeads, KVH = kr::kKvHeads, QN = QH * HD, KVN = KVH * HD;

void prep_checks(clrun::Device& dev) {
  const uint32_t G = kr::kG;
  {   // the norm: prep_res_fold SP0 + kol_norm_finish
    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 1);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
    std::vector<uint16_t> r_ref = resid, x_ref(H);
    std::vector<float> ss_ref(G);
    prep_ref::res_fold(nullptr, r_ref.data(), ss_ref.data(), 1, H, 0, G);
    kr::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), 1);
    clrun::Program fold(dev, "src/kernels/prep.cl", {"M=1", "K=2560", "S_PREV=0", "FOLD_G=20"});
    clrun::Program norm(dev, "src/kernels/kolibri/kol_prep.cl", {"M=1", "K=2560", "NORM_G=20", "NORM_WGS=20"});
    clrun::Buffer rb(dev, resid), wb(dev, w), ss(dev, G * 4), xb(dev, H * 2);
    fold.run("prep_res_fold", {20 * 256, 1}, {256, 1}, rb, rb, ss);
    norm.run("kol_norm_finish", {20 * 256, 1}, {256, 1}, ss, rb, wb, xb);
    const std::vector<uint16_t> x = xb.read<uint16_t>();
    int worst = 0;
    for (uint32_t k = 0; k < H; ++k) worst = std::max(worst, ulps(x[k], x_ref[k]));
    report("kol_norm_finish (x-hat rounded, then plain w)", worst, 0, H);
  }
  {   // the sandwich: prep_res_fold SP4 _Z + kol_post_add
    const std::vector<float> part = random_f32(size_t(4) * H, -0.5f, 0.5f, 2);
    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 3);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(1.5f + float(k % 7) * 0.2f);
    std::vector<uint16_t> a_ref(H), r_ref = resid;
    for (uint32_t k = 0; k < H; ++k) {
      float acc = 0.f;
      for (uint32_t s = 0; s < 4; ++s) acc += part[size_t(s) * H + k];
      a_ref[k] = rne(acc);
    }
    std::vector<float> sa(G), so(G);
    {
      std::vector<uint16_t> t = a_ref;
      prep_ref::res_fold(nullptr, t.data(), sa.data(), 1, H, 0, G);
    }
    kr::post_add(sa.data(), a_ref.data(), w.data(), r_ref.data(), so.data(), 1);
    clrun::Program fz(dev, "src/kernels/prep.cl", {"M=1", "K=2560", "S_PREV=4", "FOLD_G=20", "ZERO_RESID=1"});
    clrun::Program pa(dev, "src/kernels/kolibri/kol_prep.cl", {"M=1", "K=2560", "POST_G=20"});
    clrun::Buffer pb(dev, part), ab(dev, H * 2), sab(dev, G * 4), wb(dev, w), rb(dev, resid), sob(dev, G * 4);
    fz.run("prep_res_fold", {20 * 256, 1}, {256, 1}, pb, ab, sab);
    pa.run("kol_post_add", {20 * 256, 1}, {256, 1}, sab, ab, wb, rb, sob);
    const std::vector<uint16_t> r = rb.read<uint16_t>(), a = ab.read<uint16_t>();
    int worst = 0;
    for (uint32_t k = 0; k < H; ++k) worst = std::max({worst, ulps(r[k], r_ref[k]), ulps(a[k], a_ref[k])});
    report("prep_res_fold _Z + kol_post_add (the sandwich)", worst, 0, 2 * H);
  }
  // the attention prep, sliding (pos 4097: ring row 1) and full (pos 37)
  const std::vector<float> rope = loader::kol_rope_table(model::kolibri1(), 4100);
  for (int sliding = 1; sliding >= 0; --sliding) {
    const uint32_t pos = sliding ? 4097 : 37, rows = sliding ? kr::kRing : 64;
    const std::vector<float> part = random_f32(size_t(2) * 7168, -1.f, 1.f, 10 + sliding);
    std::vector<float> qkn(2 * HD);
    for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = 1;
    clrun::Program p(dev, "src/kernels/kolibri/kol_prep.cl",
                     with(kCtrlDefs, {"M=1", "QKV_N=7168", "QKV_S=2", "Q_HEADS=48", "KV_HEADS=4", "HD=128",
                                      sliding ? "SLIDING=1" : "SLIDING=0", sliding ? "RING=4096" : "RING=0"}));
    clrun::Buffer cb(dev, ctrl), pb(dev, part), qb(dev, qkn), rb(dev, rope), aq(dev, QN * 4);
    clrun::Buffer kb(dev, size_t(rows) * KVN * 2), vb(dev, size_t(rows) * KVN * 2);
    p.run("kol_attn_prep", {size_t(QH + 2 * KVH) * 128, 1}, {128, 1}, cb, pb, qb, rb, aq, kb, vb);
    const kr::Prep want = kr::attn_prep(part.data(), 1, 0, 2, qkn.data(), sliding ? rope.data() + size_t(pos) * HD : nullptr);
    const size_t row = kr::ring_row(pos, sliding);
    const std::vector<uint16_t> k = kb.read<uint16_t>(), v = vb.read<uint16_t>();
    const bool ok = aq.read<float>() == want.q && std::equal(want.k.begin(), want.k.end(), k.begin() + row * KVN) &&
                    std::equal(want.v.begin(), want.v.end(), v.begin() + row * KVN);
    report_bool(sliding ? "kol_attn_prep sliding (pos 4097 -> ring row 1)" : "kol_attn_prep full (NoPE, pos 37)", ok);
  }
}

void moe_checks(clrun::Device& dev) {
  clrun::Program rp(dev, "src/kernels/kolibri/kol_moe.cl",
                    {"M=1", "ROUTE_E=384", "ROUTE_K=6", "ROUTE_WG=256", "ROUTE_EPL=2", "ROUTE_LN=512"});
  const auto route = [&](const std::vector<float>& lg, const std::vector<float>& bs) {
    clrun::Buffer lb(dev, lg), bb(dev, bs), rt(dev, 32 * 4);
    rp.run("kol_route", {256, 1}, {256, 1}, lb, bb, rt);
    return rt.read<uint32_t>();
  };
  {   // random logits and the fixture's four rows (tie, bias reorder, all -1e30)
    std::vector<float> lg = random_f32(512, -3.f, 3.f, 20), bs = random_f32(512, -0.5f, 0.5f, 21);
    for (uint32_t e = 384; e < 512; ++e) {
      lg[e] = 80.0f;   // padded slots: never ranked
      bs[e] = 0.0f;
    }
    bool ok = true;
    int worst = 0;
    const auto cmp = [&](const std::vector<uint32_t>& row, const kr::Route& want) {
      for (uint32_t j = 0; j < 6; ++j) {
        ok = ok && row[j] == want.ids[j];
        worst = std::max(worst, int(std::abs(int64_t(row[8 + j]) - int64_t(kr::as_u32(want.w[j])))));
      }
    };
    cmp(route(lg, bs), kr::route(lg.data(), bs.data()));
    const std::vector<uint32_t> fl = kr::hex32(fx::kRouteLogits), fb = kr::hex32(fx::kRouteBias), fi = kr::hex32(fx::kRouteIds);
    for (uint32_t r = 0; r < 4; ++r) {
      std::vector<float> l(512, 80.0f), b(512, 0.0f);
      for (uint32_t e = 0; e < 384; ++e) {
        l[e] = kr::as_f32(fl[size_t(r) * 384 + e]);
        b[e] = kr::as_f32(fb[size_t(r) * 384 + e]);
      }
      const std::vector<uint32_t> row = route(l, b);
      for (uint32_t j = 0; j < 6; ++j) ok = ok && row[j] == fi[size_t(r) * 6 + j];
      cmp(row, kr::route(l.data(), b.data()));
    }
    report_bool("kol_route ids (random + torch's 4 rows, padded +80)", ok);
    std::printf("kolibri_run: %-46s worst %d fp32 ulps (bar 2) %s\n", "kol_route weights (sigmoid)", worst,
                worst <= 2 ? "agrees" : "DISAGREES");
    if (worst > 2) ++failures;
  }
  {   // the MoE block at the B70 build's defines
    const size_t gub = kr::gate_up_block_words(), dnb = kr::down_block_words();
    const std::vector<uint32_t> gu = random_blocks(gub * 384, 30), dn = random_blocks(dnb * 384, 31);
    const std::vector<uint16_t> sgr = random_bf16(size_t(1024) * H, -0.03f, 0.03f, 32);
    const std::vector<uint16_t> sdr = random_bf16(size_t(H) * 512, -0.05f, 0.05f, 33);
    std::vector<uint16_t> sg(sgr.size()), sd(sdr.size());
    common::repack_bf16_tiled(sgr.data(), H, 1024, sg.data());
    common::repack_bf16_tiled(sdr.data(), 512, H, sd.data());
    const std::vector<uint16_t> x = random_bf16(H, -0.5f, 0.5f, 34);
    const uint32_t ids[6] = {0, 17, 128, 255, 300, 383};
    const float ws[6] = {0.73f, 0.51f, 0.95f, 0.12f, 0.66f, 0.88f};
    std::vector<uint32_t> row(32, 0);
    for (uint32_t j = 0; j < 6; ++j) {
      row[j] = ids[j];
      row[8 + j] = kr::as_u32(ws[j]);
    }
    clrun::Program mp(dev, "src/kernels/kolibri/kol_moe.cl",
                      {"M=1", "MOE_E=384", "MOE_K=6", "HIDDEN=2560", "INTER=512", "UP_KS=4", "DN_KS=2"});
    clrun::Buffer rb(dev, row), xb(dev, x), gb(dev, gu), db(dev, dn), sgb(dev, sg), sdb(dev, sd);
    clrun::Buffer hb(dev, size_t(7) * 512 * 2), ob(dev, H * 2);
    mp.run("kol_moe_gate_up", {size_t(7 * 16) * 256, 1}, {256, 1}, rb, xb, gb, sgb, hb);
    const std::vector<uint16_t> h = hb.read<uint16_t>();
    const std::vector<uint16_t> h_ref = kr::gate_up(ids, x.data(), gu.data(), sg.data(), 4);
    int worst = 0;
    for (size_t i = 0; i < h.size(); ++i) worst = std::max(worst, ulps(h[i], h_ref[i]));
    report("kol_moe_gate_up (6 routed + bf16 shared)", worst, 2, h.size());
    mp.run("kol_moe_down", {size_t(H / 16) * 224, 1}, {224, 1}, rb, hb, db, sdb, ob);
    const std::vector<uint16_t> mo = ob.read<uint16_t>(), mo_ref = kr::down(ids, ws, h, dn.data(), sd.data(), 2);
    worst = 0;
    for (uint32_t n = 0; n < H; ++n) worst = std::max(worst, ulps(mo[n], mo_ref[n]));
    report("kol_moe_down + the fp32 ascending-id combine", worst, 2, H);
  }
}

void attention_checks(clrun::Device& dev) {
  const uint32_t TGT = 32, ML = 1024;
  for (int sliding = 1; sliding >= 0; --sliding) {
    clrun::Program p(dev, "src/kernels/kolibri/kol_attn_eager.cl",
                     with(kCtrlDefs, {"M=1", "TGT=32", "Q_HEADS=48", "KV_HEADS=4", sliding ? "WINDOW=513" : "WINDOW=0",
                                      sliding ? "RING=4096" : "RING=0"}));
    // the fixture's cases of this kind (one kv head of 12 q heads, coarse inputs) and a wrapped ring
    std::vector<uint32_t> cases;
    for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci)
      if ((fx::kAttn[ci].sliding != 0) == bool(sliding)) cases.push_back(ci);
    if (sliding) cases.push_back(99);   // pos 4103, random
    for (uint32_t ci : cases) {
      const bool fixture = ci != 99;
      const uint32_t pos = fixture ? fx::kAttn[ci].query : 4103, lo = kr::key_lo(pos, sliding);
      const uint32_t rows = sliding ? kr::kRing : std::max(pos + 1, ML);
      std::vector<uint16_t> K(size_t(rows) * KVN, 0), V(size_t(rows) * KVN, 0);
      std::vector<float> q(QN, 0.0f);
      if (fixture) {
        for (uint32_t pp = lo; pp <= pos; ++pp)
          for (uint32_t dd = 0; dd < HD; ++dd) {
            const size_t at = size_t(kr::ring_row(pp, sliding)) * KVN + dd;   // kv head 0
            K[at] = rne(kr::fixture_val(true, 21, pp * 128 + dd, 9, 0.125f));
            V[at] = rne(kr::fixture_val(true, 22, pp * 128 + dd, 9, 0.125f));
          }
        for (uint32_t i = 0; i < 12 * HD; ++i) q[i] = kr::fixture_val(true, 20, i, ci, 0.5f);
      } else {
        K = random_bf16(K.size(), -1.f, 1.f, 50);
        V = random_bf16(V.size(), -1.f, 1.f, 51);
        const std::vector<uint16_t> qb = random_bf16(QN, -4.f, 4.f, 52);
        for (uint32_t i = 0; i < QN; ++i) q[i] = f32(qb[i]);
      }
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      const uint32_t stride = std::max(ML, pos + 1);
      clrun::Buffer cb(dev, ctrl), qb(dev, q), kb(dev, K), vb(dev, V), sb(dev, size_t(QH) * stride * 4);
      clrun::Buffer part(dev, size_t(QH) * TGT * 130 * 4), ob(dev, QN * 2);
      p.run("kol_attn_eager_score", {size_t(KVH) * 128, TGT}, {128, 1}, cb, qb, kb, sb, stride);
      const std::vector<float> s = sb.read<float>();
      p.run("kol_attn_eager_softmax", {size_t(QH) * 128, 1}, {128, 1}, cb, sb, stride);
      const std::vector<float> pr = sb.read<float>();
      p.run("kol_attn_eager_pv", {size_t(KVH) * 128, TGT}, {128, 1}, cb, sb, vb, part, stride);
      p.run("kol_attn_eager_reduce", {size_t(QH) * 128, 1}, {128, 1}, cb, part, ob);
      const std::vector<uint16_t> o = ob.read<uint16_t>();
      const uint32_t L = pos + 1 - lo, blk = kr::eager_block(L);
      size_t s_bad = 0;
      int p_worst = 0, o_worst = 0, f_worst = 0;
      const std::vector<uint16_t> fo = fixture ? kr::hex16(fx::kAttn[ci].o) : std::vector<uint16_t>();
      for (uint32_t h = 0; h < (fixture ? 12u : QH); ++h) {
        const kr::EagerHead e = kr::attention_eager(q.data() + size_t(h) * HD, K.data(), V.data(), lo, pos, sliding,
                                                    h / (QH / KVH), blk);
        for (uint32_t i = 0; i < L; ++i) {
          s_bad += s[size_t(h) * stride + i] != e.s[i];
          p_worst = std::max(p_worst, ulps(rne(pr[size_t(h) * stride + i]), rne(e.p[i])));
        }
        for (uint32_t dd = 0; dd < HD; ++dd) {
          o_worst = std::max(o_worst, ulps(o[size_t(h) * HD + dd], rne(e.o[dd])));
          if (fixture) f_worst = std::max(f_worst, ulps(o[size_t(h) * HD + dd], fo[size_t(h) * HD + dd]));
        }
      }
      char what[96];
      std::snprintf(what, sizeof what, "kol_attn_eager %s at %u (%u keys)", sliding ? "sliding" : "full", pos, L);
      report_bool((std::string(what) + ": scores").c_str(), s_bad == 0);
      report((std::string(what) + ": probabilities").c_str(), p_worst, 1, size_t(L) * QH);
      report((std::string(what) + ": output").c_str(), o_worst, 1, QN);
      if (fixture) report((std::string(what) + ": output vs torch").c_str(), f_worst, 1, 12 * HD);
    }
  }
}

// ---- spec 20d: the portable prefill kernels (kol_pf_moe.cl, kol_pf_linear.cl, kol_prep.cl at M 2048) ----
namespace kp = kolibri_pf_ref;
constexpr uint32_t kC = kp::kC;

// kol_route-shaped rows: random (kolibri_ref::route), coarse (exact ties at the cut), or from an expert
// multiset laid out column-major (the adversaries; kolibri_pf_ref_test's from_list).
std::vector<uint32_t> pf_routes(uint32_t C, uint32_t kind, uint32_t seed) {
  std::vector<uint32_t> rows(size_t(C) * kr::kWords, 0);
  std::mt19937 rng(seed);
  if (kind <= 1) {
    std::uniform_real_distribution<float> u(-3.f, 3.f);
    std::uniform_int_distribution<int> lq(-4, 4), bq(0, 3);
    std::vector<float> bias(kr::kRouterN, 0.0f), lg(kr::kRouterN, 1e30f);
    for (uint32_t e = 0; e < kr::kExperts; ++e) bias[e] = kind ? 0.25f * float(bq(rng)) : 0.1f * u(rng);
    for (uint32_t t = 0; t < C; ++t) {
      for (uint32_t e = 0; e < kr::kExperts; ++e) lg[e] = kind ? 0.5f * float(lq(rng)) : u(rng);
      const kr::Route r = kr::route(lg.data(), bias.data());
      for (uint32_t j = 0; j < 6; ++j) {
        rows[size_t(t) * kr::kWords + j] = r.ids[j];
        rows[size_t(t) * kr::kWords + 8 + j] = kr::as_u32(r.w[j]);
      }
    }
    return rows;
  }
  std::vector<uint32_t> list;
  if (kind == 2) {   // every token to the same six experts (the lane boundaries)
    for (uint32_t e : {0u, 127u, 255u, 256u, 382u, 383u}) list.insert(list.end(), C, e);
  } else {           // the tile bound reached: 372 experts x 33 rows, 12 x 1 (C = 2048)
    for (uint32_t e = 0; e < kr::kExperts; ++e) list.insert(list.end(), e < 372 ? 33u : 1u, e);
  }
  for (uint32_t t = 0; t < C; ++t) {
    uint32_t ids[6];
    for (uint32_t j = 0; j < 6; ++j) ids[j] = list[size_t(j) * C + t];
    std::sort(ids, ids + 6);
    for (uint32_t j = 0; j < 6; ++j) {
      rows[size_t(t) * kr::kWords + j] = ids[j];
      rows[size_t(t) * kr::kWords + 8 + j] = kr::as_u32(0.05f + 0.15f * float(j));
    }
  }
  return rows;
}

void prefill_checks(clrun::Device& dev) {
  const std::vector<std::string> moe_defs = {"PF_E=384", "PF_K=6", "PF_WG=256", "PF_EPL=2", "PF_D=2560",
                                             "PF_INTER=512", "TM=32", "KC=2048"};
  clrun::Program p(dev, "src/kernels/kolibri/kol_pf_moe.cl", moe_defs);
  const uint32_t T = kp::tmax(kC), R = T * kp::kTm;
  const std::vector<uint16_t> x = random_bf16(size_t(kC) * H, -1.f, 1.f, 70);
  clrun::Buffer xb(dev, x);
  const char* names[] = {"random", "tied at the cut", "all to 6 experts", "the bound reached (820 tiles)"};
  std::vector<uint32_t> keep_routes;
  kp::Sorted keep;
  for (uint32_t kind = 0; kind < 4; ++kind) {   // kol_pf_sort + kol_pf_gather at C = 2048
    const std::vector<uint32_t> rt = pf_routes(kC, kind, 71 + kind);
    const kp::Sorted st = kp::sort(rt.data(), kC);
    clrun::Buffer rb(dev, rt), hdr(dev, kp::hdr_words() * 4), tiles(dev, size_t(T) * 8), row_tok(dev, size_t(R) * 4),
        pair_row(dev, size_t(kC) * 6 * 4), xg(dev, size_t(R) * H * 2);
    p.run("kol_pf_sort", {256}, {256}, rb, hdr, tiles, row_tok, pair_row, kC, T);
    const std::vector<uint32_t> h = hdr.read<uint32_t>(), ti = tiles.read<uint32_t>(), rtok = row_tok.read<uint32_t>(),
                                pr = pair_row.read<uint32_t>();
    const bool sort_ok = std::equal(st.hdr.begin(), st.hdr.begin() + kp::kHdrCount + kp::kE + 1, h.begin()) &&
                         ti == st.tiles && std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), rtok.begin()) &&
                         pr == st.pair_row;
    p.run("kol_pf_gather", {size_t(R) * 64}, {64}, xb, hdr, row_tok, xg);
    const std::vector<uint16_t> g = xg.read<uint16_t>(), gw = kp::gather(st, x.data());
    const bool gather_ok = std::equal(gw.begin(), gw.end(), g.begin());
    char what[96];
    std::snprintf(what, sizeof what, "kol_pf_sort C 2048, %s (%u tiles)", names[kind], st.tiles_used);
    report_bool(what, sort_ok);
    std::snprintf(what, sizeof what, "kol_pf_gather C 2048, %s", names[kind]);
    report_bool(what, gather_ok);
    if (kind == 0) {
      keep_routes = rt;
      keep = st;
    }
  }
  {   // kol_pf_moe_combine over random y on the random chunk's sorted rows
    const std::vector<uint16_t> y = random_bf16(size_t(keep.rows_used()) * H, -1.f, 1.f, 80);
    std::vector<uint16_t> want(size_t(kC) * H);
    kp::combine(keep_routes.data(), keep, y.data(), want.data(), kC);
    clrun::Buffer rb(dev, keep_routes), hb(dev, keep.hdr), pb(dev, keep.pair_row), yb(dev, y), mo(dev, size_t(kC) * H * 2);
    p.run("kol_pf_moe_combine", {size_t(H), kC}, {256, 1}, rb, hb, pb, yb, mo, kC);
    report_bool("kol_pf_moe_combine C 2048 (ascending id, + shared, 1 rounding)", mo.read<uint16_t>() == want);
  }
  {   // kol_pf_dequant_gu / _dn over blocks [382, 385): 382 routed (rows), 383 routed with NO row (skipped:
      // its output keeps the sentinel), 384 the shared expert's bf16 tiles copied
    const size_t gub = kr::gate_up_block_words(), dnb = kr::down_block_words();
    std::vector<uint32_t> gu(gub * 384, 0), dn(dnb * 384, 0);
    const std::vector<uint32_t> g382 = random_blocks(gub, 81), d382 = random_blocks(dnb, 82);
    std::copy(g382.begin(), g382.end(), gu.begin() + gub * 382);
    std::copy(d382.begin(), d382.end(), dn.begin() + dnb * 382);
    const std::vector<uint16_t> sgr = random_bf16(size_t(1024) * H, -0.03f, 0.03f, 83), sdr = random_bf16(size_t(H) * 512, -0.05f, 0.05f, 84);
    std::vector<uint16_t> sg(sgr.size()), sd(sdr.size());
    common::repack_bf16_tiled(sgr.data(), H, 1024, sg.data());
    common::repack_bf16_tiled(sdr.data(), 512, H, sd.data());
    std::vector<uint32_t> hdr(kp::hdr_words(), 0);
    hdr[kp::kHdrCount + 382] = 7;   // 383: 0 rows
    const size_t gsz = size_t(H) * 1024, dsz = size_t(512) * H;
    clrun::Buffer gb(dev, gu), db(dev, dn), sgb(dev, sg), sdb(dev, sd), hb(dev, hdr);
    clrun::Buffer og(dev, std::vector<uint16_t>(3 * gsz, 0x1234)), od(dev, std::vector<uint16_t>(3 * dsz, 0x1234));
    const uint32_t b0 = 382, b1 = 385;
    p.run("kol_pf_dequant_gu", {1024, H / 64, 3}, {16, 1, 1}, gb, sgb, hb, og, b0, b1);
    p.run("kol_pf_dequant_dn", {size_t(H), 512 / 64, 3}, {16, 1, 1}, db, sdb, hb, od, b0, b1);
    const std::vector<uint16_t> g = og.read<uint16_t>(), d = od.read<uint16_t>();
    const std::vector<uint16_t> g0 = kp::dequant_gu(gu.data(), sg.data(), 382), g2 = kp::dequant_gu(gu.data(), sg.data(), 384);
    const std::vector<uint16_t> d0 = kp::dequant_dn(dn.data(), sd.data(), 382), d2 = kp::dequant_dn(dn.data(), sd.data(), 384);
    const bool skip = std::all_of(g.begin() + gsz, g.begin() + 2 * gsz, [](uint16_t v) { return v == 0x1234; }) &&
                      std::all_of(d.begin() + dsz, d.begin() + 2 * dsz, [](uint16_t v) { return v == 0x1234; });
    report_bool("kol_pf_dequant_gu / _dn: routed block exact (int4)",
                std::equal(g0.begin(), g0.end(), g.begin()) && std::equal(d0.begin(), d0.end(), d.begin()));
    report_bool("kol_pf_dequant_gu / _dn: an expert with no row skipped", skip);
    report_bool("kol_pf_dequant_gu / _dn: block 384 = the bf16 shared tiles",
                std::equal(g2.begin(), g2.end(), g.begin() + 2 * gsz) && std::equal(d2.begin(), d2.end(), d.begin() + 2 * dsz));
  }
  {   // kol_pf_bf16_slab: o_proj 6144 x 2560 (1024, 1024, 512) and a synthetic 256 x 2400 (a zero tail)
    struct Shape { uint32_t K, N; };
    for (const Shape s : {Shape{6144, 2560}, Shape{256, 2400}}) {
      const std::vector<uint16_t> wr = random_bf16(size_t(s.N) * s.K, -1.f, 1.f, s.N);
      std::vector<uint16_t> wt(wr.size());
      common::repack_bf16_tiled(wr.data(), s.K, s.N, wt.data());
      clrun::Program sp(dev, "src/kernels/kolibri/kol_pf_linear.cl", {"K=" + std::to_string(s.K), "N=" + std::to_string(s.N)});
      clrun::Buffer wb(dev, wt), out(dev, size_t(s.K) * 1024 * 2);
      bool ok = true;
      for (uint32_t n0 = 0; n0 < s.N;) {
        const uint32_t ns = s.N - n0 >= 1024 ? 1024 : (s.N - n0 + 255) / 256 * 256;
        sp.run("kol_pf_bf16_slab", {size_t(ns), size_t(s.K / 8)}, {16, 1}, wb, out, n0, ns);
        const std::vector<uint16_t> got = out.read<uint16_t>(), want = kp::bf16_slab(wt.data(), s.K, s.N, n0, ns);
        ok = ok && std::equal(want.begin(), want.end(), got.begin());
        n0 += ns;
      }
      char what[96];
      std::snprintf(what, sizeof what, "kol_pf_bf16_slab K %u N %u (slabs + tail)", s.K, s.N);
      report_bool(what, ok);
    }
  }
  {   // the int4 arm's slab: 18c's k2_pf_dequant_slab at q||k||v 2560 x 7168, the last slab
    const uint32_t K = 2560, N = 7168, n0 = 6144, ns = 1024;
    std::mt19937 rng(90);
    std::vector<uint32_t> words(size_t(K / 8) * N);
    for (uint32_t& w : words) w = rng();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    for (uint16_t& v : sc) v = common::f32_to_f16(0.01f + 0.04f * float(rng() % 1000) / 1000.f);
    clrun::Program sp(dev, "src/kernels/k2/k2_pf_linear.cl", {"K=2560", "N=7168"});
    clrun::Buffer wb(dev, words), sb(dev, sc), out(dev, size_t(K) * ns * 2);
    sp.run("k2_pf_dequant_slab", {size_t(ns), size_t(K / 64)}, {16, 1}, wb, sb, out, n0, ns);
    report_bool("k2_pf_dequant_slab K 2560 N 7168 (the int4 arm)",
                out.read<uint16_t>() == k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, ns));
  }
  {   // the grouped norm at M 2048: pf_res_fold SP0 + kol_norm_finish; the sandwich: pf_res_fold _Z (o_proj's
      // one slice) + kol_post_add - every row of the chunk, sums at pitch kPfC
    const uint32_t G = kr::kG, C = kC;
    const std::vector<uint16_t> resid = random_bf16(size_t(C) * H, -2.f, 2.f, 91);
    const std::vector<float> part = random_f32(size_t(C) * H, -0.5f, 0.5f, 92);
    std::vector<float> w(H), w2(H);
    for (uint32_t k = 0; k < H; ++k) {
      w[k] = rf(0.3f + float(k % 11) * 0.1f);
      w2[k] = rf(1.5f + float(k % 7) * 0.2f);
    }
    std::vector<uint16_t> r_ref = resid, x_ref(size_t(C) * H), a_ref(size_t(C) * H);
    std::vector<float> ss_ref(size_t(G) * C), sa(size_t(G) * C), so(size_t(G) * C);
    prep_ref::res_fold(nullptr, r_ref.data(), ss_ref.data(), C, H, 0, G);
    kr::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), C);
    for (size_t i = 0; i < a_ref.size(); ++i) a_ref[i] = rne(part[i]);
    {
      std::vector<uint16_t> t = a_ref;
      prep_ref::res_fold(nullptr, t.data(), sa.data(), C, H, 0, G);
    }
    std::vector<uint16_t> r2_ref = r_ref;
    kr::post_add(sa.data(), a_ref.data(), w2.data(), r2_ref.data(), so.data(), C);
    clrun::Program f0(dev, "src/kernels/prefill/pf_prep.cl", {"K=2560", "S_PREV=0", "FOLD_G=20"});
    clrun::Program fz(dev, "src/kernels/prefill/pf_prep.cl", {"K=2560", "S_PREV=1", "FOLD_G=20", "ZERO_RESID=1"});
    clrun::Program nm(dev, "src/kernels/kolibri/kol_prep.cl", {"M=2048", "K=2560", "NORM_G=20", "NORM_WGS=20"});
    clrun::Program pa(dev, "src/kernels/kolibri/kol_prep.cl", {"M=2048", "K=2560", "POST_G=20"});
    clrun::Buffer rb(dev, resid), pb(dev, part), wb(dev, w), w2b(dev, w2), ss(dev, size_t(G) * C * 4), xb2(dev, size_t(C) * H * 2),
        ab(dev, size_t(C) * H * 2), sab(dev, size_t(G) * C * 4), sob(dev, size_t(G) * C * 4);
    // Apple's OpenCL is not given -cl-fp32-correctly-rounded-divide-sqrt (the B70 build is), so a row's
    // rstd = 1 / sqrt(Σ / K + eps) can land a few fp32 ulps off - over a whole chunk a handful of values
    // then round apart (the residual add amplifies one ulp of n where it cancels). Every differing value
    // must be EXACTLY the host's chain at an rstd within 4 fp32 ulps of the correctly rounded one, and
    // the sums (no sqrt) bitwise; kolibri_pf_kernels_test holds the card to bitwise.
    const auto explained = [&](const std::vector<uint16_t>& got, const std::vector<uint16_t>& want, const float* sums,
                               const std::function<uint16_t(size_t, float)>& chain, size_t* nd) {
      bool ok = true;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i]) continue;
        ++*nd;
        const uint32_t m = uint32_t(i / H);
        float r = kr::rstd_of(sums, C, m, H, G), lo = r, hi = r;
        bool hit = false;
        for (int s = 0; s < 4 && !hit; ++s) {
          lo = std::nextafter(lo, 0.0f);
          hi = std::nextafter(hi, INFINITY);
          hit = chain(i, lo) == got[i] || chain(i, hi) == got[i];
        }
        ok = ok && hit;
      }
      return ok && *nd * 10000 < got.size();
    };
    f0.run("pf_res_fold", {20 * 256, C}, {256, 1}, pb, rb, ss, C);
    nm.run("kol_norm_finish", {20 * 256, C}, {256, 1}, ss, rb, wb, xb2);
    report_bool("pf_res_fold SP0 M 2048: resid and its 20 sums", rb.read<uint16_t>() == r_ref && ss.read<float>() == ss_ref);
    size_t nd = 0;
    const bool nok = explained(xb2.read<uint16_t>(), x_ref, ss_ref.data(),
                               [&](size_t i, float rs) { return kr::norm_elem(f32(r_ref[i]), rs, w[i % H]); }, &nd);
    report_bool("kol_norm_finish M 2048 (2048 rows)", nok);
    std::printf("kolibri_run:   (%zu of %zu values differ, each the chain at an rstd a few fp32 ulps off)\n", nd,
                size_t(C) * H);
    fz.run("pf_res_fold", {20 * 256, C}, {256, 1}, pb, ab, sab, C);
    pa.run("kol_post_add", {20 * 256, C}, {256, 1}, sab, ab, w2b, rb, sob);
    report_bool("pf_res_fold _Z M 2048: o_proj's row and its sums", ab.read<uint16_t>() == a_ref && sab.read<float>() == sa);
    nd = 0;
    const bool pok = explained(rb.read<uint16_t>(), r2_ref, sa.data(),
                               [&](size_t i, float rs) {
                                 return rne(f32(r_ref[i]) + f32(kr::norm_elem(f32(a_ref[i]), rs, w2[i % H])));
                               },
                               &nd);
    report_bool("kol_post_add M 2048 (the sandwich's residual)", pok);
    std::printf("kolibri_run:   (%zu of %zu values differ, same cause)\n", nd, size_t(C) * H);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("kolibri_run on %s (INDICATIVE: Mac OpenCL, emulated sub-groups)\n", dev.name().c_str());
    prep_checks(dev);
    moe_checks(dev);
    attention_checks(dev);
    prefill_checks(dev);
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "kolibri_run: %s\n", e.what());
    return 1;
  }
  if (failures) {
    std::printf("DISAGREES with the host reference in %d check(s) (indicative: read the kernel before the box)\n",
                failures);
    return 1;
  }
  std::printf("agrees with the host reference (indicative only; the B70 is the real test)\n");
  return 0;
}
