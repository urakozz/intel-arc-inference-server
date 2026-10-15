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

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("muse_run: device %s (INDICATIVE: Mac OpenCL, emulated subgroups)\n", dev.name().c_str());
    prep_checks(dev);
  } catch (const std::exception& e) {
    std::printf("muse_run: FAILED: %s\n", e.what());
    return 1;
  }
  std::printf("muse_run: %s\n", failures ? "DISAGREES" : "all agree");
  return failures ? 1 : 0;
}
