// muse_run - spec 23c's portable Muse-Glimmer kernels on the Mac's OpenCL GPU against tests/kernels/muse_ref.h (the
// host twin, itself bitwise to transformers 5.19.0: tests/kernels/muse_ref_test.cc) (INDICATIVE ONLY: clrun.h says
// what this can and cannot show).
//
//   muse_run      no arguments; Muse-Glimmer-30B's real shapes (hidden 6656, 32 / 2 heads x 128, FF 19968, window
//                 2048 on a 4096-slot ring); the embedding table is 8 rows here (the B70 build's is 202048)
//
// What runs, with the CMake block's defines: src/kernels/muse/mg_prep.cl - mg_embed (the weight-less norm, the
// placeholders' row 0, an id past the table flagged), mg_norm (centered, eps 1e-5) and mg_final_norm (plain w) over
// the residual's chunk sums, mg_post_add after prep.cl's prep_res_fold SP4 _Z at G 26 (the sandwich), mg_attn_prep
// sliding (pos 4097: ring row 1) and full (pos 37) at S 2 with the gate's sigmoid, mg_silu_mul at S 8 over the
// concatenated gate||up; src/kernels/muse/mg_attn_eager.cl (both builds) - see attention_checks.
// mg_attn.cl (flash) is not run: its decode is shuffles, sub-group reductions and 2D block reads (box only:
// muse_kernels_test; tests/kernels/muse_ref_test.cc runs its block / wave / merge arithmetic on the host).
//
// Spec 23d (prefill_checks, against tests/kernels/muse_pf_ref.h and muse_ref.h): the prefill's portable kernels at the
// B70 build's defines - src/kernels/muse/mg_pf_prep.cl (mg_pf_embed over a chunk's ids, mg_pf_attn_gate), mg_prep.cl
// at M 2048 (rows at the chunk's pitch: mg_norm, pf_res_fold _Z K 6656 + mg_post_add, mg_attn_prep at S 1 through the
// ring's wrap, mg_silu_mul at S 1), 18c's k2_pf_dequant_slab at K 6656 / 19968 (exact), and mg_pf_attn.cl's EAGER build
// (plain OpenCL) against musepf::eager_window - decode's eager rows - over a chunk crossing the ring's wrap and a full
// layer. mg_pf_attn.cl's flash build is not run (DPAS, 2D block reads, sub-group reductions: box only,
// muse_pf_kernels_test; tests/kernels/muse_pf_ref_test.cc runs its tile walk on the host).
//
// Bars: the exp-free chains bitwise, except that a value may move with Apple's 1 / sqrt and divide (not correctly
// rounded here; the B70 build is -cl-fp32-correctly-rounded-divide-sqrt): each such value must be the host chain at an
// rstd within 4 fp32 ulps (counted); the exp chains (the gate's sigmoid, SiLU) within 2 bf16 ulps (Apple's divide).
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
#include "kernels/k2_pf_ref.h"
#include "kernels/muse_pf_ref.h"
#include "kernels/muse_ref.h"
#include "runtime/muse/muse_rope.h"

namespace {

namespace mr = museref;
using mr::bf;
using mr::rf;
using mr::rne;

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
float step_ulps(float r, int k) {
  float x = r;
  for (int i = 0; i < std::abs(k); ++i) x = std::nextafter(x, k > 0 ? INFINITY : 0.0f);
  return x;
}

int failures = 0;
void report_bool(const char* what, bool ok, const std::string& detail = "") {
  std::printf("muse_run: %-58s %s%s\n", what, ok ? "exact" : "DISAGREES", detail.c_str());
  if (!ok) ++failures;
}
void report(const char* what, int worst, int bar, size_t n) {
  const bool ok = worst <= bar;
  std::printf("muse_run: %-58s %zu values, worst %d bf16 ulps (bar %d) %s\n", what, n, worst, bar,
              ok ? (worst == 0 ? "exact" : "agrees") : "DISAGREES");
  if (!ok) ++failures;
}
// Every device value equals the host chain `f(rstd')` at rstd' = rstd moved by at most 4 fp32 ulps; most at rstd.
void report_rstd(const char* what, const std::vector<uint16_t>& got, size_t n,
                 const std::function<uint16_t(size_t, float)>& f, const std::function<float(size_t)>& rstd) {
  size_t moved = 0;
  bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    const float r = rstd(i);
    if (got[i] == f(i, r)) continue;
    bool any = false;
    for (int k = -4; k <= 4 && !any; ++k) any = got[i] == f(i, step_ulps(r, k));
    if (!any) {
      if (ok) std::printf("muse_run:   %s: value %zu = %04x, the chain gives %04x\n", what, i, got[i], f(i, r));
      ok = false;
    }
    ++moved;
  }
  char buf[96];
  std::snprintf(buf, sizeof buf, " (%zu of %zu at an rstd within 4 ulps: Apple's 1/sqrt)", moved, n);
  report_bool(what, ok, moved ? buf : "");
}

const uint32_t H = mr::kHidden, HD = mr::kHd, QN = mr::kQN, KVN = mr::kKvN, G = mr::kG;

void prep_checks(clrun::Device& dev) {
  {   // mg_embed: an 8-row table; id 3, the image placeholder (row 0), an id past the table (flagged)
    const uint32_t V = 8;
    const std::vector<uint16_t> table = random_bf16(size_t(V) * H, -3.f, 3.f, 1);
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl",
                     with(kCtrlDefs, {"M=1", "K=6656", "EMBED_G=26", "VOCAB=8", "IMG_TOKEN=5", "VID_TOKEN=6"}));
    bool ok = true;
    size_t moved_total = 0;
    for (uint32_t id : {3u, 5u, 6u}) {
      const uint32_t row = id == 5 || id == 6 ? 0 : id;
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[2] = id;
      clrun::Buffer cb(dev, ctrl), tb(dev, table), rb(dev, H * 2), sb(dev, G * 4);
      p.run("mg_embed", {256, 1}, {256, 1}, cb, tb, rb, sb);
      const std::vector<uint16_t> r = rb.read<uint16_t>();
      const std::vector<float> ss = sb.read<float>();
      const uint16_t* src = table.data() + size_t(row) * H;
      const float rs = mr::rstd_row(src, H, mr::kEps);
      for (uint32_t k = 0; k < H; ++k) {
        if (r[k] == rne(bf(src[k]) * rs)) continue;
        bool any = false;
        for (int q = -4; q <= 4 && !any; ++q) any = r[k] == rne(bf(src[k]) * step_ulps(rs, q));
        ok = ok && any;
        ++moved_total;
      }
      std::vector<float> want(G);
      mr::chunk_sumsq(r.data(), H, G, want.data());   // the sums of what it wrote: exact
      ok = ok && ss == want;
    }
    {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[2] = 9;   // past the table: nothing written, the debug flag set
      const std::vector<uint16_t> sentinel(H, 0x1234);
      clrun::Buffer cb(dev, ctrl), tb(dev, table), rb(dev, sentinel), sb(dev, G * 4);
      p.run("mg_embed", {256, 1}, {256, 1}, cb, tb, rb, sb);
      ok = ok && rb.read<uint16_t>() == sentinel && cb.read<uint32_t>()[18] == 0xE0000000u;
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%zu of %u values at an rstd within 4 ulps)", moved_total, 3 * H);
    report_bool("mg_embed (weight-less norm, placeholders -> row 0, bad id)", ok, moved_total ? buf : "");
  }
  {   // mg_norm (centered) and mg_final_norm (plain w) over the chunk sums of a random residual
    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 2);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(-0.4f + float(k % 11) * 0.08f);
    std::vector<float> ss(G);
    mr::chunk_sumsq(resid.data(), H, G, ss.data());
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl", {"M=1", "K=6656", "NORM_G=26", "NORM_WGS=26", "NORM_EPS=1e-5f"});
    clrun::Buffer sb(dev, ss), rb(dev, resid), wb(dev, w), xb(dev, H * 2), fb(dev, H * 2);
    p.run("mg_norm", {26 * 256, 1}, {256, 1}, sb, rb, wb, xb);
    p.run("mg_final_norm", {26 * 256, 1}, {256, 1}, sb, rb, wb, fb);
    const float rs = mr::rstd_of(ss.data(), G, H, mr::kEps);
    report_rstd("mg_norm (centered: (x rstd)(1 + w), one rounding)", xb.read<uint16_t>(), H,
                [&](size_t k, float r) { return mr::centered_elem(bf(resid[k]), r, w[k]); }, [&](size_t) { return rs; });
    report_rstd("mg_final_norm (plain w)", fb.read<uint16_t>(), H,
                [&](size_t k, float r) { return rne((bf(resid[k]) * r) * w[k]); }, [&](size_t) { return rs; });
  }
  {   // the sandwich: prep_res_fold SP4 _Z at G 26 (o_proj's / down's row and Σ²) + mg_post_add (eps 1e-8)
    const std::vector<float> part = random_f32(size_t(4) * H, -0.5f, 0.5f, 3);
    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 4);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.5f + float(k % 7) * 0.2f);
    std::vector<uint16_t> a_ref(H);
    for (uint32_t k = 0; k < H; ++k) a_ref[k] = mr::partials_col(part.data(), 4, 1, 0, H, k);
    clrun::Program fz(dev, "src/kernels/prep.cl", {"M=1", "K=6656", "S_PREV=4", "FOLD_G=26", "ZERO_RESID=1"});
    clrun::Program pa(dev, "src/kernels/muse/mg_prep.cl", {"M=1", "K=6656", "POST_G=26", "POST_EPS=1e-8f"});
    clrun::Buffer pb(dev, part), ab(dev, H * 2), sab(dev, G * 4), wb(dev, w), rb(dev, resid), sob(dev, G * 4);
    fz.run("prep_res_fold", {26 * 256, 1}, {256, 1}, pb, ab, sab);
    pa.run("mg_post_add", {26 * 256, 1}, {256, 1}, sab, ab, wb, rb, sob);
    const std::vector<uint16_t> a = ab.read<uint16_t>(), r = rb.read<uint16_t>();
    std::vector<float> sa_ref(G);
    mr::chunk_sumsq(a_ref.data(), H, G, sa_ref.data());
    report_bool("prep_res_fold SP4 _Z G26 (the sub-block's row and Σ²)", a == a_ref && sab.read<float>() == sa_ref);
    const float rs = mr::rstd_of(sa_ref.data(), G, H, mr::kPostEps);
    report_rstd("mg_post_add (centered eps 1e-8, + resid)", r, H,
                [&](size_t k, float rr) { return rne(bf(resid[k]) + bf(mr::centered_elem(bf(a_ref[k]), rr, w[k]))); },
                [&](size_t) { return rs; });
    std::vector<float> so_ref(G);
    mr::chunk_sumsq(r.data(), H, G, so_ref.data());
    report_bool("mg_post_add's Σ² of the new residual (26 chunk trees)", sob.read<float>() == so_ref);
  }
  // the attention prep, sliding (pos 4097: ring row 1) and full (pos 37), S 2
  const std::vector<float> rope = runtime::muse::rope_table(HD, 5e5f, 4100);
  for (int sliding = 1; sliding >= 0; --sliding) {
    const uint32_t pos = sliding ? 4097 : 37, rows = sliding ? mr::kRing : 64;
    const std::vector<float> part = random_f32(size_t(2) * mr::kQgkvN, -1.f, 1.f, 10 + sliding);
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = 1;
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl",
                     with(kCtrlDefs, {"M=1", "QGKV_N=8704", "QGKV_S=2", "Q_HEADS=32", "KV_HEADS=2", "HD=128",
                                      "QK_SCALE=3.87f", sliding ? "SLIDING=1" : "SLIDING=0", sliding ? "RING=4096" : "RING=0"}));
    clrun::Buffer cb(dev, ctrl), pb(dev, part), rb(dev, rope), aq(dev, QN * 4), gb(dev, QN * 2);
    clrun::Buffer kb(dev, size_t(rows) * KVN * 2), vb(dev, size_t(rows) * KVN * 2);
    p.run("mg_attn_prep", {size_t(2 * 32 + 2 * 2) * 128, 1}, {128, 1}, cb, pb, rb, aq, gb, kb, vb);
    // the host chain: the GEMV's bf16 outputs, then museref::qk_prep
    std::vector<uint16_t> q(QN), k(KVN), g(QN), v(KVN), qo(QN), ko(KVN);
    for (uint32_t i = 0; i < QN; ++i) {
      q[i] = mr::partials_col(part.data(), 2, 1, 0, mr::kQgkvN, i);
      g[i] = mr::partials_col(part.data(), 2, 1, 0, mr::kQgkvN, mr::kOffGate + i);
    }
    for (uint32_t i = 0; i < KVN; ++i) {
      k[i] = mr::partials_col(part.data(), 2, 1, 0, mr::kQgkvN, mr::kOffK + i);
      v[i] = mr::partials_col(part.data(), 2, 1, 0, mr::kQgkvN, mr::kOffV + i);
    }
    mr::qk_prep(q.data(), k.data(), pos, sliding != 0, rope.data(), qo.data(), ko.data());
    const size_t row = mr::ring_row(pos, sliding != 0);
    const std::vector<float> aqv = aq.read<float>();
    const std::vector<uint16_t> kd = kb.read<uint16_t>(), vd = vb.read<uint16_t>(), gd = gb.read<uint16_t>();
    size_t qk_diff = 0;
    int qk_worst = 0;
    for (uint32_t i = 0; i < QN; ++i) {
      const int u = ulps(rne(aqv[i]), qo[i]);
      qk_diff += u != 0;
      qk_worst = std::max(qk_worst, u);
    }
    for (uint32_t i = 0; i < KVN; ++i) {
      const int u = ulps(kd[row * KVN + i], ko[i]);
      qk_diff += u != 0;
      qk_worst = std::max(qk_worst, u);
    }
    const bool v_ok = std::equal(v.begin(), v.end(), vd.begin() + row * KVN);
    int g_worst = 0;
    for (uint32_t i = 0; i < QN; ++i) g_worst = std::max(g_worst, ulps(gd[i], mr::gate_bf16(g[i])));
    char buf[160];
    std::snprintf(buf, sizeof buf, "mg_attn_prep %s: q / k (%zu moved, Apple's 1/sqrt)",
                  sliding ? "sliding (pos 4097 -> ring row 1)" : "full (NoPE, pos 37)", qk_diff);
    report(buf, qk_worst, 1, QN + KVN);
    report_bool(sliding ? "mg_attn_prep sliding: v at ring row 1" : "mg_attn_prep full: v at row 37", v_ok);
    report(sliding ? "mg_attn_prep sliding: gate rne(sigmoid(g))" : "mg_attn_prep full: gate rne(sigmoid(g))", g_worst, 2, QN);
  }
  {   // SiLU x up over the concatenated gate||up, S 8
    const uint32_t I = mr::kInter;
    const std::vector<float> part = random_f32(size_t(8) * 2 * I, -1.f, 1.f, 20);
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl", {"M=1", "SILU_I=19968", "SILU_S=8"});
    clrun::Buffer pb(dev, part), hb(dev, size_t(I) * 2);
    p.run("mg_silu_mul", {size_t(78) * 256, 1}, {256, 1}, pb, hb);
    const std::vector<uint16_t> h = hb.read<uint16_t>();
    std::vector<uint16_t> g(I), u(I), want(I);
    for (uint32_t i = 0; i < I; ++i) {
      g[i] = mr::partials_col(part.data(), 8, 1, 0, 2 * I, i);
      u[i] = mr::partials_col(part.data(), 8, 1, 0, 2 * I, I + i);
    }
    mr::silu_mul(g.data(), u.data(), I, want.data());
    int worst = 0;
    for (uint32_t i = 0; i < I; ++i) worst = std::max(worst, ulps(h[i], want[i]));
    report("mg_silu_mul (gate||up concatenated, exp_torch)", worst, 2, I);
  }
}

// mg_attn_eager.cl (one launch a layer) against museref::attention_eager, bitwise (no Apple sqrt / divide in its
// exp-free part; the softmax's reciprocal is 1 / sum - Apple's divide may move a probability, counted with the
// output): sliding through the ring at 5, 2046, 2047, 2048, 4095, 4096, 4097, 9000 (Review Focus 3's rows: the
// window's edge, the ring's wrap) and at 4904 (= 9000 - 4096: the same ring rows, the same output), full at 3000;
// random caches and q, the gate applied.
void attention_checks(clrun::Device& dev) {
  const uint32_t ML = 3001;
  std::mt19937 rng(40);
  std::uniform_real_distribution<float> u(-1.f, 1.f), uq(-4.f, 4.f);
  std::vector<uint16_t> rk(size_t(mr::kRing) * KVN), rv(rk.size()), fk(size_t(ML) * KVN), fv(fk.size()), q(QN), g(QN);
  for (auto* v : {&rk, &rv, &fk, &fv})
    for (uint16_t& e : *v) e = rne(u(rng));
  for (uint16_t& e : q) e = rne(uq(rng));
  for (uint16_t& e : g) e = mr::gate_bf16(rne(uq(rng)));
  std::vector<float> qf(QN);
  for (uint32_t i = 0; i < QN; ++i) qf[i] = bf(q[i]);
  std::vector<uint16_t> o4904;
  for (int sliding = 1; sliding >= 0; --sliding) {
    clrun::Program p(dev, "src/kernels/muse/mg_attn_eager.cl",
                     with(kCtrlDefs, {"M=1", "TGT=64", "Q_HEADS=32", "KV_HEADS=2", sliding ? "WINDOW=2048" : "WINDOW=0",
                                      sliding ? "RING=4096" : "RING=0"}));
    const std::vector<uint16_t>& kc = sliding ? rk : fk;
    const std::vector<uint16_t>& vc = sliding ? rv : fv;
    for (uint32_t pos : sliding ? std::vector<uint32_t>{5, 2046, 2047, 2048, 4095, 4096, 4097, 4904, 9000}
                                : std::vector<uint32_t>{3000}) {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      clrun::Buffer cb(dev, ctrl), qb(dev, qf), kb(dev, kc), vb(dev, vc), gb(dev, g), sb(dev, size_t(QN / HD) * 9001 * 4),
          ob(dev, size_t(QN) * 2);
      p.run("mg_attn_eager", {size_t(32) * 128, 1}, {128, 1}, cb, qb, kb, vb, gb, sb, ob, uint32_t(9001));
      const std::vector<uint16_t> o = ob.read<uint16_t>();
      std::vector<uint16_t> want(QN);
      mr::attention_eager(q.data(), kc.data(), vc.data(), pos, sliding != 0, mr::kRing, g.data(), want.data());
      int worst = 0;
      size_t diff = 0;
      for (uint32_t i = 0; i < QN; ++i) {
        worst = std::max(worst, ulps(o[i], want[i]));
        diff += o[i] != want[i];
      }
      if (sliding && pos == 4904) o4904 = o;
      if (sliding && pos == 9000) report_bool("mg_attn_eager: query 9000 == query 4904 (the same ring rows)", o == o4904);
      char what[120];
      std::snprintf(what, sizeof what, "mg_attn_eager %s at %u (%zu differ: Apple's 1/sum)", sliding ? "sliding" : "full",
                    pos, diff);
      report(what, worst, 1, QN);
    }
  }
}

// ---- spec 23d: the prefill's portable kernels ---------------------------------------------------------------------
namespace mp = musepf;
const uint32_t PC = 2048;   // the chunk pitch the M 2048 binaries bake

void slab_check(clrun::Device& dev, uint32_t K, uint32_t N, uint32_t n0, uint32_t ns) {
  std::mt19937 rng(90 + K);
  std::vector<uint32_t> words(size_t(K / 8) * N);
  for (uint32_t& w : words) w = rng();
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  for (uint16_t& v : sc) v = common::f32_to_f16(0.01f + 0.04f * float(rng() % 1000) / 1000.f);
  clrun::Program sp(dev, "src/kernels/k2/k2_pf_linear.cl", {"K=" + std::to_string(K), "N=" + std::to_string(N)});
  clrun::Buffer wb(dev, words), sb(dev, sc), out(dev, size_t(K) * ns * 2);
  sp.run("k2_pf_dequant_slab", {size_t(ns), size_t(K / 64)}, {16, 1}, wb, sb, out, n0, ns);
  char what[96];
  std::snprintf(what, sizeof what, "k2_pf_dequant_slab K %u N %u (columns %u..%u)", K, N, n0, n0 + ns - 1);
  report_bool(what, out.read<uint16_t>() == k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, ns));
}

void prefill_checks(clrun::Device& dev) {
  slab_check(dev, 6656, 8704, 8192, 512);   // q||gate||k||v's last slab (8 x 1024 + 512)
  slab_check(dev, 19968, 6656, 0, 1024);    // down's first slab (K 19968)
  {   // mg_pf_embed over a chunk of 3 ids (row 3, the image placeholder -> row 0, row 7), sums at the pitch 2048
    const uint32_t V = 8, C = 3;
    const std::vector<uint16_t> table = random_bf16(size_t(V) * H, -3.f, 3.f, 101);
    const std::vector<uint32_t> ids = {3, 5, 7};
    clrun::Program p(dev, "src/kernels/muse/mg_pf_prep.cl",
                     {"M=2048", "K=6656", "EMBED_G=26", "VOCAB=8", "IMG_TOKEN=5", "VID_TOKEN=6"});
    clrun::Buffer ib(dev, ids), tb(dev, table), rb(dev, size_t(PC) * H * 2), sb(dev, size_t(G) * PC * 4);
    p.run("mg_pf_embed", {256, C}, {256, 1}, ib, tb, rb, sb);
    const std::vector<uint16_t> r = rb.read<uint16_t>();
    const std::vector<float> ss = sb.read<float>();
    bool ok = true;
    size_t moved = 0;
    for (uint32_t m = 0; m < C; ++m) {
      const uint16_t* src = table.data() + size_t(ids[m] == 5 ? 0 : ids[m]) * H;
      const float rs = mr::rstd_row(src, H, mr::kEps);
      for (uint32_t k = 0; k < H; ++k) {
        const uint16_t got = r[size_t(m) * H + k];
        if (got == rne(bf(src[k]) * rs)) continue;
        bool any = false;
        for (int q = -4; q <= 4 && !any; ++q) any = got == rne(bf(src[k]) * step_ulps(rs, q));
        ok = ok && any;
        ++moved;
      }
      std::vector<float> want(G);
      mr::chunk_sumsq(r.data() + size_t(m) * H, H, G, want.data());
      for (uint32_t g = 0; g < G; ++g) ok = ok && ss[size_t(g) * PC + m] == want[g];
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%zu of %u values at an rstd within 4 ulps)", moved, C * H);
    report_bool("mg_pf_embed (3 ids, placeholder -> row 0, sums at pitch 2048)", ok, moved ? buf : "");
  }
  {   // mg_norm at M 2048 over rows 0..3; the sandwich at M 2048: pf_res_fold _Z (K 6656, m_count 2048) + mg_post_add
    const uint32_t C = 4;
    const std::vector<uint16_t> resid = random_bf16(size_t(PC) * H, -2.f, 2.f, 102);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(-0.4f + float(k % 11) * 0.08f);
    std::vector<float> ss(size_t(G) * PC, 0.f);
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<float> s1(G);
      mr::chunk_sumsq(resid.data() + size_t(m) * H, H, G, s1.data());
      for (uint32_t g = 0; g < G; ++g) ss[size_t(g) * PC + m] = s1[g];
    }
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl", {"M=2048", "K=6656", "NORM_G=26", "NORM_WGS=26", "NORM_EPS=1e-5f"});
    clrun::Buffer sb(dev, ss), rb(dev, resid), wb(dev, w), xb(dev, size_t(PC) * H * 2);
    p.run("mg_norm", {26 * 256, C}, {256, 1}, sb, rb, wb, xb);
    const std::vector<uint16_t> x = xb.read<uint16_t>();
    for (uint32_t m = 0; m < C; m += 3) {
      std::vector<float> s1(G);
      for (uint32_t g = 0; g < G; ++g) s1[g] = ss[size_t(g) * PC + m];
      const float rs = mr::rstd_of(s1.data(), G, H, mr::kEps);
      const std::vector<uint16_t> row(x.begin() + size_t(m) * H, x.begin() + size_t(m + 1) * H);
      report_rstd(m == 0 ? "mg_norm M 2048 (row 0)" : "mg_norm M 2048 (row 3: the pitch)", row, H,
                  [&](size_t k, float r) { return mr::centered_elem(bf(resid[size_t(m) * H + k]), r, w[k]); },
                  [&](size_t) { return rs; });
    }
    const std::vector<float> part = random_f32(size_t(PC) * H, -0.5f, 0.5f, 103);
    clrun::Program fz(dev, "src/kernels/prefill/pf_prep.cl", {"K=6656", "S_PREV=1", "FOLD_G=26", "ZERO_RESID=1"});
    clrun::Program pa(dev, "src/kernels/muse/mg_prep.cl", {"M=2048", "K=6656", "POST_G=26", "POST_EPS=1e-8f"});
    clrun::Buffer pb(dev, part), ab(dev, size_t(PC) * H * 2), sab(dev, size_t(G) * PC * 4), rb2(dev, resid),
        sob(dev, size_t(G) * PC * 4);
    fz.run("pf_res_fold", {26 * 256, C}, {256, 1}, pb, ab, sab, PC);
    pa.run("mg_post_add", {26 * 256, C}, {256, 1}, sab, ab, wb, rb2, sob);
    const std::vector<uint16_t> a = ab.read<uint16_t>(), r2 = rb2.read<uint16_t>();
    const std::vector<float> sa = sab.read<float>(), so = sob.read<float>();
    bool fold_ok = true, sums_ok = true, post_ok = true;
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<uint16_t> a_ref(H);
      for (uint32_t k = 0; k < H; ++k) a_ref[k] = rne(part[size_t(m) * H + k]);
      std::vector<float> sa_ref(G);
      mr::chunk_sumsq(a_ref.data(), H, G, sa_ref.data());
      fold_ok = fold_ok && std::equal(a_ref.begin(), a_ref.end(), a.begin() + size_t(m) * H);
      for (uint32_t g = 0; g < G; ++g) fold_ok = fold_ok && sa[size_t(g) * PC + m] == sa_ref[g];
      const float rs = mr::rstd_of(sa_ref.data(), G, H, mr::kPostEps);
      for (uint32_t k = 0; k < H && post_ok; ++k) {
        const uint16_t got = r2[size_t(m) * H + k];
        bool any = false;
        for (int q = -4; q <= 4 && !any; ++q)
          any = got == rne(bf(resid[size_t(m) * H + k]) + bf(mr::centered_elem(bf(a_ref[k]), step_ulps(rs, q), w[k])));
        post_ok = any;
      }
      std::vector<float> so_ref(G);
      mr::chunk_sumsq(r2.data() + size_t(m) * H, H, G, so_ref.data());
      for (uint32_t g = 0; g < G; ++g) sums_ok = sums_ok && so[size_t(g) * PC + m] == so_ref[g];
    }
    report_bool("pf_res_fold K6656 SP1 G26 _Z (4 rows, pitch 2048)", fold_ok);
    report_bool("mg_post_add M 2048 (4 rows; rstd within 4 ulps)", post_ok);
    report_bool("mg_post_add M 2048: the new residual's sums at pitch 2048", sums_ok);
  }
  // mg_attn_prep at M 2048, S 1: 3 rows from 4095 (sliding: slots 4095, 0, 1) and from 37 (full)
  const std::vector<float> rope = runtime::muse::rope_table(HD, 5e5f, 4100);
  for (int sliding = 1; sliding >= 0; --sliding) {
    const uint32_t pos = sliding ? 4095 : 37, rows = sliding ? mr::kRing : 64, C = 3;
    const std::vector<float> part = random_f32(size_t(PC) * mr::kQgkvN, -1.f, 1.f, 110 + sliding);
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = C;
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl",
                     with(kCtrlDefs, {"M=2048", "QGKV_N=8704", "QGKV_S=1", "Q_HEADS=32", "KV_HEADS=2", "HD=128",
                                      "QK_SCALE=3.87f", sliding ? "SLIDING=1" : "SLIDING=0", sliding ? "RING=4096" : "RING=0"}));
    clrun::Buffer cb(dev, ctrl), pb(dev, part), rb(dev, rope), aq(dev, size_t(PC) * QN * 4), gb(dev, size_t(PC) * QN * 2);
    clrun::Buffer kb(dev, size_t(rows) * KVN * 2), vb(dev, size_t(rows) * KVN * 2);
    p.run("mg_attn_prep", {size_t(2 * 32 + 2 * 2) * 128, C}, {128, 1}, cb, pb, rb, aq, gb, kb, vb);
    const std::vector<float> aqv = aq.read<float>();
    const std::vector<uint16_t> kd = kb.read<uint16_t>(), vd = vb.read<uint16_t>(), gd = gb.read<uint16_t>();
    int qk_worst = 0, g_worst = 0;
    bool v_ok = true;
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<uint16_t> q(QN), k(KVN), g(QN), v(KVN), qo(QN), ko(KVN);
      for (uint32_t i = 0; i < QN; ++i) {
        q[i] = mr::partials_col(part.data(), 1, PC, m, mr::kQgkvN, i);
        g[i] = mr::partials_col(part.data(), 1, PC, m, mr::kQgkvN, mr::kOffGate + i);
      }
      for (uint32_t i = 0; i < KVN; ++i) {
        k[i] = mr::partials_col(part.data(), 1, PC, m, mr::kQgkvN, mr::kOffK + i);
        v[i] = mr::partials_col(part.data(), 1, PC, m, mr::kQgkvN, mr::kOffV + i);
      }
      mr::qk_prep(q.data(), k.data(), pos + m, sliding != 0, rope.data(), qo.data(), ko.data());
      const size_t row = mr::ring_row(pos + m, sliding != 0);
      for (uint32_t i = 0; i < QN; ++i) {
        qk_worst = std::max(qk_worst, ulps(rne(aqv[size_t(m) * QN + i]), qo[i]));
        g_worst = std::max(g_worst, ulps(gd[size_t(m) * QN + i], mr::gate_bf16(g[i])));
      }
      for (uint32_t i = 0; i < KVN; ++i) qk_worst = std::max(qk_worst, ulps(kd[row * KVN + i], ko[i]));
      v_ok = v_ok && std::equal(v.begin(), v.end(), vd.begin() + row * KVN);
    }
    report(sliding ? "mg_attn_prep M 2048 sliding: 3 rows from 4095 (slots 4095, 0, 1)"
                   : "mg_attn_prep M 2048 full: 3 rows from 37 (NoPE)", qk_worst, 1, size_t(C) * (QN + KVN));
    report_bool(sliding ? "mg_attn_prep M 2048 sliding: v at the ring's slots" : "mg_attn_prep M 2048 full: v rows",
                v_ok);
    report(sliding ? "mg_attn_prep M 2048 sliding: gate" : "mg_attn_prep M 2048 full: gate", g_worst, 2, size_t(C) * QN);
  }
  {   // mg_silu_mul at M 2048, S 1, rows 0 and 1
    const uint32_t I = mr::kInter, C = 2;
    const std::vector<float> part = random_f32(size_t(PC) * 2 * I, -1.f, 1.f, 120);
    clrun::Program p(dev, "src/kernels/muse/mg_prep.cl", {"M=2048", "SILU_I=19968", "SILU_S=1"});
    clrun::Buffer pb(dev, part), hb(dev, size_t(PC) * I * 2);
    p.run("mg_silu_mul", {size_t(78) * 256, C}, {256, 1}, pb, hb);
    const std::vector<uint16_t> h = hb.read<uint16_t>();
    int worst = 0;
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<uint16_t> g(I), u(I), want(I);
      for (uint32_t i = 0; i < I; ++i) {
        g[i] = mr::partials_col(part.data(), 1, PC, m, 2 * I, i);
        u[i] = mr::partials_col(part.data(), 1, PC, m, 2 * I, I + i);
      }
      mr::silu_mul(g.data(), u.data(), I, want.data());
      for (uint32_t i = 0; i < I; ++i) worst = std::max(worst, ulps(h[size_t(m) * I + i], want[i]));
    }
    report("mg_silu_mul M 2048 S 1 (2 rows)", worst, 2, size_t(C) * I);
  }
  {   // mg_pf_attn_gate: rne(rne(o) x g) in place, 2 rows
    const uint32_t C = 2;
    std::vector<uint16_t> a = random_bf16(size_t(C) * QN, -2.f, 2.f, 130);
    const std::vector<uint16_t> g = random_bf16(size_t(C) * QN, 0.f, 1.f, 131);
    clrun::Program p(dev, "src/kernels/muse/mg_pf_prep.cl", {"M=2048", "GATE_N=4096"});
    clrun::Buffer gb(dev, g), ab(dev, a);
    p.run("mg_pf_attn_gate", {size_t(16) * 256, C}, {256, 1}, gb, ab);
    mp::gate_window(a.data(), g.data(), C);
    report_bool("mg_pf_attn_gate (decode's gate chain)", ab.read<uint16_t>() == a);
  }
  // mg_pf_attn.cl EAGER against musepf::eager_window (decode's eager rows): a sliding chunk of 24 rows at 4090 (the
  // ring's wrap inside the chunk; every row's window 2048 keys), a full chunk of 12 at 2990.
  for (int sliding = 1; sliding >= 0; --sliding) {
    const uint32_t c0 = sliding ? 4090 : 2990, C = sliding ? 24 : 12, n = c0 + C;
    std::mt19937 rng(140 + sliding);
    std::uniform_real_distribution<float> u(-1.f, 1.f), uq(-4.f, 4.f);
    const uint32_t slots = sliding ? mr::kRing : n;
    std::vector<uint16_t> kc(size_t(slots) * KVN), vc(kc.size());
    for (uint32_t p = 0; p < n; ++p) {   // positions written in order: a slot holds the last position mapping to it
      const size_t s = size_t(mr::ring_row(p, sliding != 0)) * KVN;
      for (uint32_t i = 0; i < KVN; ++i) {
        kc[s + i] = rne(u(rng));
        vc[s + i] = rne(u(rng));
      }
    }
    std::vector<uint16_t> q(size_t(C) * QN);
    for (uint16_t& e : q) e = rne(uq(rng));
    std::vector<float> qf(q.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = bf(q[i]);
    clrun::Program p(dev, "src/kernels/muse/mg_pf_attn.cl",
                     {"TGT=64", "Q_HEADS=32", "KV_HEADS=2", "EAGER=1", sliding ? "WINDOW=2048" : "WINDOW=0",
                      sliding ? "RING=4096" : "RING=0"});
    clrun::Buffer qb(dev, qf), kb(dev, kc), vb(dev, vc), ob(dev, size_t(C) * QN * 2);
    p.run("mg_pf_flash_attn", {size_t(32) * 128, C}, {128, 1}, qb, kb, vb, ob, c0, C);
    const std::vector<uint16_t> o = ob.read<uint16_t>();
    std::vector<uint16_t> want(o.size());
    mp::eager_window(q.data(), kc.data(), vc.data(), c0, C, sliding != 0, want.data());
    int worst = 0;
    size_t diff = 0;
    for (size_t i = 0; i < o.size(); ++i) {
      worst = std::max(worst, ulps(o[i], want[i]));
      diff += o[i] != want[i];
    }
    char what[120];
    std::snprintf(what, sizeof what, "mg_pf_flash_attn EAGER %s, %u rows from %u (%zu differ: Apple's 1/sum)",
                  sliding ? "sliding" : "full", C, c0, diff);
    report(what, worst, 1, o.size());
    // the same rows through DECODE's eager kernel on this device (mg_attn_eager M 1, one row a step, the gate 1.0):
    // prefill's eager rows are decode's bit for bit - whatever this device's divide does, it does it to both
    clrun::Program dp(dev, "src/kernels/muse/mg_attn_eager.cl",
                      with(kCtrlDefs, {"M=1", "TGT=64", "Q_HEADS=32", "KV_HEADS=2", sliding ? "WINDOW=2048" : "WINDOW=0",
                                       sliding ? "RING=4096" : "RING=0"}));
    const std::vector<uint16_t> one(QN, 0x3F80);
    bool same = true;
    for (uint32_t t = 0; t < C; ++t) {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = c0 + t;
      ctrl[1] = 1;
      const std::vector<float> qrow(qf.begin() + size_t(t) * QN, qf.begin() + size_t(t + 1) * QN);
      clrun::Buffer cb(dev, ctrl), qb1(dev, qrow), gb(dev, one), sb(dev, size_t(32) * n * 4), ob1(dev, size_t(QN) * 2);
      dp.run("mg_attn_eager", {size_t(32) * 128, 1}, {128, 1}, cb, qb1, kb, vb, gb, sb, ob1, n);
      const std::vector<uint16_t> d1 = ob1.read<uint16_t>();
      same = same && std::equal(d1.begin(), d1.end(), o.begin() + size_t(t) * QN);
    }
    std::snprintf(what, sizeof what, "mg_pf_flash_attn EAGER %s == mg_attn_eager's rows on this device", sliding ? "sliding" : "full");
    report_bool(what, same);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("muse_run: device %s (INDICATIVE: Mac OpenCL, emulated subgroups)\n", dev.name().c_str());
    prep_checks(dev);
    attention_checks(dev);
    prefill_checks(dev);
  } catch (const std::exception& e) {
    std::printf("muse_run: FAILED: %s\n", e.what());
    return 1;
  }
  std::printf("muse_run: %s\n", failures ? "DISAGREES" : "all agree");
  return failures ? 1 : 0;
}
