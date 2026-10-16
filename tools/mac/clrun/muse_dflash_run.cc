// muse_dflash_run - spec 23f: Muse-Glimmer's DFlash drafter lists (spec 19b's df_* kernels at Muse's defines plus
// src/kernels/muse/mg_dflash.cl) on the Mac's OpenCL GPU at the Muse fixture's tiny shapes
// (tests/loader/muse_dflash_fixture.safetensors: hidden 64, 2 layers, 8 / 2 heads x 16 (GQA 4), 5 taps, block 16,
// window 16 inclusive as WIN 17 over a 32-slot ring, vocab 512), against the host twin tests/kernels/dflash_ref.h in
// its Muse form (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   muse_dflash_run      no arguments
//
// What runs, in runtime/dflash/dflash_capture.cc's order for a Muse-form drafter: the projection list (fc,
// mg_df_norm with hidden_norm, per layer the k||v rows of q||k||v, df_ctx_store) in chunks of up to 16 positions, and
// the draft list over the whole 16-row block (dctl.k = 15) - df_embed (the raw rows), per layer mg_df_norm,
// q||k||v, df_qkv_prep, df_attn_part + df_attn_reduce, o, df_add, mg_df_norm, gate||up, df_silu_mul, down, df_add;
// the final mg_df_norm, the int8 head over rows 1..15 (gemv_i8w M 15), df_topk_stage1/2 (raw: mult 1, no cap), and
// mg_df_select writing the 804-word readback (runtime::dflash::MuseDraftOut) and the verify's ids vids[16]. Every
// fixture case (p 1 .. 39: the window's edge for row 0 and row 15, the ring's wrap), int8 drafter + int8 head; then
// the int4 g64 drafter (gemv.cl LAYOUT 1, S 1) when gemv.cl builds on this device. Also df_attn alone at Muse's
// geometry (p 1 .. 100, K 15) and mg_df_norm alone against the twin's norm2.
//
// Bars (against the twin in Engine numerics; Apple's exp / sqrt / divide are not correctly rounded): rows cosine
// >= 0.99999 (all 16), the candidates' raw logits cosine >= 0.99999, the 15 greedy ids equal, vids = [anchor, ids].
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/repack.h"
#include "kernels/dflash_fixture_io.h"
#include "kernels/dflash_ref.h"

namespace {

using Strs = std::vector<std::string>;
const Strs kCtrl = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                    "DCTL_K=0",   "DCTL_PPOS=1", "DCTL_PN=2",  "DCTL_TEMP=3", "DCTL_SEED_LO=4", "DCTL_SEED_HI=5"};
Strs with(Strs a, const Strs& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
std::string n_(uint32_t v) { return std::to_string(v); }

constexpr uint32_t M = 16, R = 15, kTopkCh = 256;
// runtime::dflash::MuseDraftOut's words (mg_dflash.cl OUT_* at MAXR 16)
constexpr uint32_t OUT_IDS = 4, OUT_CHOSEN = OUT_IDS + 16, OUT_CAND = OUT_CHOSEN + 16, OUT_UNARY = OUT_CAND + 256,
                   OUT_SCORES = OUT_UNARY + 256, OUT_WORDS = OUT_SCORES + 256;

struct Gpu {
  clrun::Device dev;
  dfref::Dims d;
  std::unique_ptr<clrun::Program> prep, attn, topk, mg;
  std::vector<std::unique_ptr<clrun::Program>> gemvs;
  std::vector<std::string> gemv_keys;

  explicit Gpu(const dfref::Dims& dd) : d(dd) {
    prep = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_prep.cl",
        with(kCtrl, {"M=16", "H=" + n_(d.H), "WG=256", "EPS=1e-05f", "VOCAB=" + n_(d.V), "INTER=" + n_(d.I),
                     "NH=" + n_(d.NH), "NKV=" + n_(d.NKV), "HD=" + n_(d.HD), "RING=" + n_(d.ring)}));
    attn = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_attn.cl",
        with(kCtrl, {"M=16", "NH=" + n_(d.NH), "NKV=" + n_(d.NKV), "HD=" + n_(d.HD), "RING=" + n_(d.ring),
                     "WIN=" + n_(d.win), "PB=" + n_(d.pb), "WGA=16", "CAUSAL=0"}));
    topk = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_topk.cl",
        Strs{"R=15", "N=" + n_(d.V), "USED=" + n_(d.used), "TOPK=16", "CH=" + n_(kTopkCh), "COMPACT=0"});
    mg = std::make_unique<clrun::Program>(
        dev, "src/kernels/muse/mg_dflash.cl",
        with(kCtrl, {"H=" + n_(d.H), "WG=256", "EPS=1e-05f", "SELECT=1", "MAXR=16", "TOPK=16"}));
  }
  clrun::Program* gemv(bool int4, uint32_t m, uint32_t K, uint32_t N) {
    const std::string key = std::string(int4 ? "q4_" : "i8_") + n_(m) + "_" + n_(K) + "_" + n_(N);
    for (size_t i = 0; i < gemv_keys.size(); ++i)
      if (gemv_keys[i] == key) return gemvs[i].get();
    if (int4)
      gemvs.push_back(std::make_unique<clrun::Program>(
          dev, "src/kernels/gemv.cl", Strs{"M=" + n_(m), "K=" + n_(K), "N=" + n_(N), "S=1", "LAYOUT=1"}));
    else
      gemvs.push_back(std::make_unique<clrun::Program>(dev, "src/kernels/gemv_i8w.cl",
                                                       Strs{"M=" + n_(m), "K=" + n_(K), "N=" + n_(N)}));
    gemv_keys.push_back(key);
    return gemvs.back().get();
  }
  // y [m][N'] = lin(x [m][K] bf16) for columns [n_lo, n_hi) (whole 16-column tiles), x rows from row0.
  std::vector<float> lin(const dfref::Lin& L, const std::vector<uint16_t>& x, uint32_t m, uint32_t row0 = 0,
                         uint32_t n_lo = 0, uint32_t n_hi = 0) {
    if (n_hi == 0) n_hi = L.N;
    const uint32_t N = n_hi - n_lo;
    std::vector<uint16_t> xs(x.begin() + size_t(row0) * L.K, x.begin() + size_t(row0 + m) * L.K);
    clrun::Buffer bx(dev, xs), by(dev, size_t(m) * N * 4);
    if (!L.q4.empty()) {
      // the loader's device form: layout 1 tiles of the whole weight, the column range a contiguous tile range
      std::vector<uint32_t> tiles(size_t(L.N / 16) * (L.K / 64) * 136);
      common::repack_int4_layout1(L.q4.data(), L.s4.data(), L.K, L.N, tiles.data());
      std::vector<uint32_t> sub(tiles.begin() + size_t(n_lo / 16) * (L.K / 64) * 136,
                                tiles.begin() + size_t(n_hi / 16) * (L.K / 64) * 136);
      clrun::Buffer bw(dev, sub);
      gemv(true, m, L.K, N)->run("gemv", {size_t(N), 1}, {64, 1}, bw, bw, bx, by);
    } else {
      std::vector<int8_t> q(L.i8.begin() + size_t(n_lo) * L.K, L.i8.begin() + size_t(n_hi) * L.K);
      std::vector<float> s(L.scales.begin() + n_lo, L.scales.begin() + n_hi);
      clrun::Buffer bq(dev, q), bs(dev, s);
      gemv(false, m, L.K, N)->run("gemv_i8w", {size_t(N)}, {64}, bq, bs, bx, by);
    }
    return by.read<float>();
  }
};

double maxrel(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0, s = 0;
  for (float x : b) s = std::fmax(s, std::fabs(x));
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(double(a[i]) - b[i]));
  return s == 0 ? m : m / s;
}

struct Ring {
  std::unique_ptr<clrun::Buffer> k, v;
};

// The projection list over positions [pos0, pos0 + n), n <= 16.
void project(Gpu& g, const dfref::Weights& w, const std::vector<uint16_t>& taps, uint32_t pos0, uint32_t n,
             clrun::Buffer& brope, Ring& ring) {
  const dfref::Dims& d = w.d;
  std::vector<uint16_t> t(size_t(M) * d.fck(), 0);
  std::copy(taps.begin() + size_t(pos0) * d.fck(), taps.begin() + size_t(pos0 + n) * d.fck(), t.begin());
  clrun::Buffer bf(g.dev, g.lin(w.fc, t, M)), bw(g.dev, w.hidden_norm), bh(g.dev, size_t(M) * d.H * 4),
      bx(g.dev, size_t(M) * d.H * 2);
  g.mg->run("mg_df_norm", {256, M}, {256, 1}, bf, bw, bh, bx);
  const std::vector<uint16_t> x = bx.read<uint16_t>();
  std::vector<float> kv, qkn;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::vector<float> o = g.lin(w.layers[l].qkv, x, M, 0, d.qn(), d.qkvn());
    kv.insert(kv.end(), o.begin(), o.end());
    qkn.insert(qkn.end(), w.layers[l].q_norm.begin(), w.layers[l].q_norm.end());
    qkn.insert(qkn.end(), w.layers[l].k_norm.begin(), w.layers[l].k_norm.end());
  }
  std::vector<uint32_t> dctl(16, 0);
  dctl[1] = pos0;
  dctl[2] = n;
  clrun::Buffer bd(g.dev, dctl), bkv(g.dev, kv), bqk(g.dev, qkn);
  g.prep->run("df_ctx_store", {size_t(d.layers) * 2 * d.NKV * d.HD, M}, {d.HD, 1}, bd, bkv, bqk, brope, *ring.k,
              *ring.v);
}

struct Out {
  std::vector<float> hidden;
  std::vector<uint32_t> words, vids;
};

// The draft list over the whole block (dctl.k = 15).
Out draft(Gpu& g, const dfref::Weights& w, const Ring& ring, clrun::Buffer& brope, uint32_t anchor, uint32_t p) {
  const dfref::Dims& d = w.d;
  const uint32_t H = d.H;
  std::vector<uint32_t> ctrl(32, 0), dctl(16, 0);
  ctrl[0] = p;
  ctrl[2] = anchor;
  dctl[0] = R;
  clrun::Buffer bctrl(g.dev, ctrl), bdctl(g.dev, dctl), bemb(g.dev, w.embed), bmask(g.dev, w.mask_row);
  clrun::Buffer resid(g.dev, size_t(M) * H * 4), h(g.dev, size_t(M) * H * 4), x(g.dev, size_t(M) * H * 2);
  g.prep->run("df_embed", {256, M}, {256, 1}, bctrl, bemb, bmask, 1.0f, resid);
  const size_t lay = size_t(d.ring) * d.kvn();
  const std::vector<uint16_t> rk = ring.k->read<uint16_t>(), rv = ring.v->read<uint16_t>();
  for (uint32_t l = 0; l < d.layers; ++l) {
    const dfref::Layer& L = w.layers[l];
    clrun::Buffer bin(g.dev, L.in_norm), bpost(g.dev, L.post_norm);
    std::vector<float> qkn = L.q_norm;
    qkn.insert(qkn.end(), L.k_norm.begin(), L.k_norm.end());
    clrun::Buffer bqk(g.dev, qkn);
    g.mg->run("mg_df_norm", {256, M}, {256, 1}, resid, bin, h, x);
    clrun::Buffer qkv(g.dev, g.lin(L.qkv, x.read<uint16_t>(), M));
    clrun::Buffer q(g.dev, size_t(M) * d.qn() * 4), kb(g.dev, size_t(M) * d.kvn() * 4), vb(g.dev, size_t(M) * d.kvn() * 4);
    g.prep->run("df_qkv_prep", {size_t(d.NH + 2 * d.NKV) * d.HD, M}, {d.HD, 1}, bctrl, qkv, bqk, brope, q, kb, vb);
    clrun::Buffer lk(g.dev, std::vector<uint16_t>(rk.begin() + l * lay, rk.begin() + (l + 1) * lay)),
        lv(g.dev, std::vector<uint16_t>(rv.begin() + l * lay, rv.begin() + (l + 1) * lay));
    const uint32_t tgt = d.ring / d.pb;
    clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
    g.attn->run("df_attn_part", {size_t(d.NKV) * 16, tgt}, {16, 1}, bctrl, bdctl, q, lk, lv, d.scale, part);
    g.attn->run("df_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bctrl, bdctl, part, q, kb, vb, d.scale, ao);
    clrun::Buffer oo(g.dev, g.lin(L.o, ao.read<uint16_t>(), M));
    g.prep->run("df_add", {256, M}, {256, 1}, oo, resid);
    g.mg->run("mg_df_norm", {256, M}, {256, 1}, resid, bpost, h, x);
    clrun::Buffer gu(g.dev, g.lin(L.gate_up, x.read<uint16_t>(), M)), a(g.dev, size_t(M) * d.I * 2);
    g.prep->run("df_silu_mul", {256 * ((d.I + 255) / 256), M}, {256, 1}, gu, a);
    clrun::Buffer dn(g.dev, g.lin(L.down, a.read<uint16_t>(), M));
    g.prep->run("df_add", {256, M}, {256, 1}, dn, resid);
  }
  clrun::Buffer bfin(g.dev, w.final_norm);
  g.mg->run("mg_df_norm", {256, M}, {256, 1}, resid, bfin, h, x);
  const std::vector<uint16_t> xr = x.read<uint16_t>();
  clrun::Buffer logits(g.dev, g.lin(w.head, xr, R, 1));
  const uint32_t chunks = (d.V + kTopkCh - 1) / kTopkCh;
  clrun::Buffer tpart(g.dev, size_t(R) * chunks * 16 * 2 * 4), cand(g.dev, size_t(R) * 16 * 4),
      unary(g.dev, size_t(R) * 16 * 4), ids(g.dev, std::vector<uint32_t>{0});
  g.topk->run("df_topk_stage1", {size_t(chunks) * 256, R}, {256, 1}, logits, tpart);
  g.topk->run("df_topk_stage2", {size_t(R) * 256}, {256}, tpart, ids, 1.0f, 0.0f, cand, unary);
  clrun::Buffer out(g.dev, size_t(OUT_WORDS) * 4), vids(g.dev, size_t(16) * 4);
  g.mg->run("mg_df_select", {16}, {16}, bctrl, bdctl, cand, unary, out, vids);
  Out o;
  o.hidden = h.read<float>();
  o.words = out.read<uint32_t>();
  o.vids = vids.read<uint32_t>();
  return o;
}

float as_f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// df_attn alone at Muse's geometry on random q / block K, V / ring rows at p, K 15, against dfref::attention.
int attn_check(Gpu& g, uint32_t p) {
  const dfref::Dims& d = g.d;
  uint64_t s = 0x9E3779B97F4A7C15ull + p * 131;
  auto rnd = [&]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return float(int32_t(s & 0xFFFF) - 32768) / 32768.0f;
  };
  std::vector<float> q(size_t(M) * d.qn()), kb(size_t(M) * d.kvn()), vb(kb.size());
  for (float& x : q) x = rnd();
  for (float& x : kb) x = rnd();
  for (float& x : vb) x = rnd();
  std::vector<uint16_t> rk(size_t(d.ring) * d.kvn()), rv(rk.size());
  std::vector<float> fk(rk.size()), fv(rk.size());
  for (size_t i = 0; i < rk.size(); ++i) {
    rk[i] = dfref::rne(rnd());
    rv[i] = dfref::rne(rnd());
    fk[i] = dfref::bf(rk[i]);
    fv[i] = dfref::bf(rv[i]);
  }
  std::vector<uint32_t> ctrl(32, 0), dctl(16, 0);
  ctrl[0] = p;
  dctl[0] = R;
  clrun::Buffer bc(g.dev, ctrl), bd(g.dev, dctl), bq(g.dev, q), bkb(g.dev, kb), bvb(g.dev, vb), brk(g.dev, rk),
      brv(g.dev, rv);
  const uint32_t tgt = d.ring / d.pb;
  clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
  g.attn->run("df_attn_part", {size_t(d.NKV) * 16, tgt}, {16, 1}, bc, bd, bq, brk, brv, d.scale, part);
  g.attn->run("df_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bc, bd, part, bq, bkb, bvb, d.scale, ao);
  std::vector<float> want(size_t(M) * d.qn());
  dfref::attention(d.attn(), q.data(), kb.data(), vb.data(), fk.data(), fv.data(), p, R, false, want.data());
  const std::vector<uint16_t> got = ao.read<uint16_t>();
  int over = 0;
  double worst = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double e = std::fabs(dfref::bf(got[i]) - dfref::rf(want[i]));
    worst = std::fmax(worst, e);
    if (e > 1.0 / 128 * std::fmax(1e-3, std::fabs(want[i]))) ++over;
  }
  std::printf("muse_dflash_run: attention alone at p %u (16 rows, window 16 inclusive): worst |diff| %.3g, %d over 1 "
              "bf16 ulp\n", p, worst, over);
  return over != 0;
}

int norm_check(Gpu& g, const dfref::Weights& w) {
  const uint32_t H = w.d.H;
  std::vector<float> src(size_t(M) * H);
  for (size_t i = 0; i < src.size(); ++i) src[i] = std::sin(float(i) * 0.37f) * (1.0f + float(i % 7));
  clrun::Buffer bs(g.dev, src), bw(g.dev, w.layers[0].in_norm), bh(g.dev, size_t(M) * H * 4), bx(g.dev, size_t(M) * H * 2);
  g.mg->run("mg_df_norm", {256, M}, {256, 1}, bs, bw, bh, bx);
  const std::vector<uint16_t> x = bx.read<uint16_t>();
  const std::vector<float> h = bh.read<float>();
  int exact = 0, ulp1 = 0, worse = 0, hbad = 0;
  for (uint32_t r = 0; r < M; ++r) {
    std::vector<uint16_t> t(H);
    dfref::norm2(&src[size_t(r) * H], w.layers[0].in_norm.data(), H, w.d.eps, nullptr, t.data());
    for (uint32_t i = 0; i < H; ++i) {
      const int dd = std::abs(int(x[size_t(r) * H + i]) - int(t[i]));
      exact += dd == 0;
      ulp1 += dd == 1;
      worse += dd > 1;
      hbad += h[size_t(r) * H + i] != dfref::bf(x[size_t(r) * H + i]);
    }
  }
  std::printf("muse_dflash_run: mg_df_norm vs the twin's norm2: %d exact, %d 1-ulp, %d worse; h = f32(x) %s\n", exact,
              ulp1, worse, hbad ? "DIFFERS" : "everywhere");
  return worse != 0 || hbad != 0;
}

// Every fixture case through both lists for the drafter `w`; returns the failures.
int run_cases(Gpu& g, const dffix::Fixture& fx, const dfref::Weights& w, const char* arm, clrun::Buffer& brope,
              const std::vector<float>& rope) {
  const dfref::Dims& d = w.d;
  const std::vector<uint16_t> taps = fx.bf16("taps");
  int bad = 0;
  const std::vector<dffix::Case> cs = dffix::cases(fx);
  for (size_t ci = 0; ci < cs.size(); ++ci) {
    const dffix::Case& c = cs[ci];
    Ring ring{std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2),
              std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2)};
    dfref::Ring tw = dfref::make_ring(d);
    for (uint32_t pos = 0; pos < c.p; pos += M) {
      const uint32_t n = std::min<uint32_t>(M, c.p - pos);
      project(g, w, taps, pos, n, brope, ring);
      dfref::project(w, dfref::Numerics::Engine, &taps[size_t(pos) * d.fck()], n, pos, rope, tw);
    }
    {
      const std::vector<uint16_t> rk = ring.k->read<uint16_t>();
      std::vector<float> fk(rk.size());
      for (size_t i = 0; i < rk.size(); ++i) fk[i] = dfref::bf(rk[i]);
      const double e = maxrel(fk, tw.k);
      if (e > 1e-2) {
        std::printf("  %s case %zu: ring k rows differ from the twin (max rel %.3g)\n", arm, ci, e);
        ++bad;
      }
    }
    const Out o = draft(g, w, ring, brope, c.anchor, c.p);
    const dfref::DraftOut t = dfref::draft(w, dfref::Numerics::Engine, tw, rope, c.anchor, c.p, R);
    double hc = 1;
    for (uint32_t r = 0; r < M; ++r)
      hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * d.H], &t.hidden[size_t(r) * d.H], d.H));
    std::vector<float> un(size_t(R) * 16);
    for (size_t i = 0; i < un.size(); ++i) un[i] = as_f(o.words[OUT_UNARY + i]);
    const double ucos = dfref::cosine(un.data(), t.unary.data(), un.size());
    bool ids = o.words[0] == R && o.words[1] == c.anchor && o.words[2] == c.p && o.vids[0] == c.anchor;
    for (uint32_t j = 0; j < R; ++j) {
      ids &= o.words[OUT_IDS + j] == t.ids[j] && o.vids[1 + j] == t.ids[j];
      ids &= o.words[OUT_SCORES + j * 16] == o.words[OUT_UNARY + j * 16];
    }
    const std::vector<int32_t> pids = fx.i32("muse.case" + std::to_string(ci) + ".ids");
    uint32_t py_eq = 0;
    for (uint32_t j = 0; j < c.K; ++j) py_eq += o.words[OUT_IDS + j] == uint32_t(pids[j]);
    const bool ok = hc >= 0.99999 && ucos >= 0.99999 && ids;
    std::printf("  %s case %zu (p %2u): rows cos %.7f, logits cos %.7f, 15 ids / vids %s | dflash_ref.py ids %u / %u\n",
                arm, ci, c.p, hc, ucos, ids ? "=" : "DIFF", py_eq, c.K);
    if (!ok) ++bad;
  }
  return bad;
}

}  // namespace

int main() {
  try {
    const dffix::Fixture fx(clrun::repo_root() + "/tests/loader/muse_dflash_fixture.safetensors");
    const dfref::Weights w = dffix::muse_weights(fx, dffix::Form::Int8, dffix::Form::Int8, 16);
    Gpu g(w.d);
    const dfref::Dims& d = w.d;
    std::printf("muse_dflash_run on %s: hidden %u, %u layers, %u / %u heads x %u, window WIN %u over ring %u, block %u, "
                "vocab %u\n", g.dev.name().c_str(), d.H, d.layers, d.NH, d.NKV, d.HD, d.win, d.ring, d.M, d.V);
    const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
    clrun::Buffer brope(g.dev, rope);
    int bad = norm_check(g, w);
    for (uint32_t p : {1u, 16u, 17u, 18u, 33u, 40u, 100u}) bad += attn_check(g, p);
    bad += run_cases(g, fx, w, "int8", brope, rope);
    // the int4 g64 drafter through gemv.cl LAYOUT 1 (the loader's form); the head stays int8
    const dfref::Weights w4 = dffix::muse_weights(fx, dffix::Form::Int4, dffix::Form::Int8, 16);
    bool built = true;
    try {
      g.gemv(true, M, d.H, d.qkvn());
    } catch (const std::exception& e) {
      built = false;
      const std::string why = e.what();
      std::printf("muse_dflash_run: int4 arm NOT RUN - gemv.cl does not build on this device (%s); the twin's "
                  "gemv_int4 (dflash_ref_test) and the box's D1 cover it\n", why.substr(0, why.find('\n')).c_str());
    }
    if (built) bad += run_cases(g, fx, w4, "int4", brope, rope);
    if (bad) {
      std::printf("DISAGREES with the host twin (indicative: read the kernels before the box)\n");
      return 1;
    }
    std::printf("agrees with the host twin (indicative only; the B70 is the real test)\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "muse_dflash_run: %s\n", e.what());
    return 1;
  }
}
