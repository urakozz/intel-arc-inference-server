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
// Bars: the exp-free chains (the norm, the sandwich, the prep, the eager scores) bitwise; the
// route ids exact; the rest within 2 bf16 ulps (OpenCL's exp, the GEMVs' fma contraction, and Apple's
// compiler is not told -cl-fp32-correctly-rounded-divide-sqrt).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/kolibri_fixture.h"
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

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("kolibri_run on %s (INDICATIVE: Mac OpenCL, emulated sub-groups)\n", dev.name().c_str());
    prep_checks(dev);
    moe_checks(dev);
    attention_checks(dev);
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
