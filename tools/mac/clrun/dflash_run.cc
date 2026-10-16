// dflash_run - spec 19b's DFlash drafter kernels (src/kernels/dflash/df_*.cl) and the reused gemv_i8w on the Mac's
// OpenCL GPU at the fixture's tiny shapes (tests/loader/dflash_ref_fixture.safetensors: hidden 64, 2 layers, 4 / 2
// heads x 16, window 16 over a 16-slot ring, vocab 512 with 508 tokens), against the host twin
// tests/kernels/dflash_ref.h (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   dflash_run      no arguments
//
// What runs: the projection list (fc, hidden_norm, per layer the k||v rows of q||k||v, df_ctx_store) in chunks of up
// to 8 positions into the ring, and the draft list - df_embed, per layer df_norm, the conv kernel projection,
// df_conv_prep, q||k||v, df_qkv_prep, df_attn_part + df_attn_reduce, o, df_conv_add, df_norm, df_conv_prep, gate||up,
// df_silu_mul, down, df_conv_add; the final df_norm, the int8 head over rows 1..7, df_topk_stage1/2, the selector
// projection, df_select - in the capture's order (runtime/dflash/dflash_capture.cc) for every fixture case (p 1 .. 39:
// the window's edge and the ring's wrap; K 7 .. 1; greedy and two sampled walks), with the int8 drafter and the int8
// head (the fixture's GEMV widths are multiples of 64; the bf16 GEMV is the shipped gemv_bf16.cl). Also: planted top-k
// cases (ties, NaN, the vocab mask, -inf rows) exact; df_attn alone at the window's edge and past the ring's wrap
// (p 1 .. 123, K 7 / 5 / 3 / 6) exact; every case again over a 256-id draft vocabulary (the compact int8 head
// gathered by loader::gather_int8_tiled_rows, df_topk COMPACT) - bitwise the full head's draft when V' holds every
// candidate, different when it lacks row 0's best (Review Focus 3); DFlash v1 (no convs, df_add, df_select V2=0).
//
// Bars: against the twin in Engine numerics (the same arithmetic; Apple's exp / log / sqrt / divide are not
// correctly rounded): the rows' cosine >= 0.99999, the scores' cosine >= 0.99999, the walk's ids equal; the top-16
// order is printed (an rstd an ulp off moved one case's values by ~1e-3 and swapped two near candidates). Against
// dflash_ref.py's own outputs: printed (the tiny random drafter's margins are not the D1 checkpoint's).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "clrun.h"
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

constexpr uint32_t kTopkCh = 256;   // the tiny head's chunk (the B70 build's is 4096)
constexpr uint32_t OUT_IDS = 4, OUT_CHOSEN = 12, OUT_CAND = 20, OUT_UNARY = 20 + 128, OUT_SCORES = 20 + 256;

struct Gpu {
  clrun::Device dev;
  dfref::Weights w;
  dfref::Dims d;
  std::unique_ptr<clrun::Program> prep, attn, topk, select;
  std::vector<std::unique_ptr<clrun::Program>> gemvs;   // by (M, K, N), see gemv()
  std::vector<std::string> gemv_keys;

  explicit Gpu(const dfref::Weights& ww) : w(ww), d(ww.d) {
    prep = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_prep.cl",
        with(kCtrl, {"M=8", "H=" + n_(d.H), "WG=256", "EPS=1e-06f", "VOCAB=" + n_(d.V), "GROUP=" + n_(d.group),
                     "CTAPS=" + n_(d.ctaps), "INTER=" + n_(d.I), "NH=" + n_(d.NH), "NKV=" + n_(d.NKV),
                     "HD=" + n_(d.HD), "RING=" + n_(d.ring)}));
    attn = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_attn.cl",
        with(kCtrl, {"M=8", "NH=" + n_(d.NH), "NKV=" + n_(d.NKV), "HD=" + n_(d.HD), "RING=" + n_(d.ring),
                     "WIN=" + n_(d.win), "PB=" + n_(d.pb), "WGA=16", "CAUSAL=0"}));
    topk = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_topk.cl",
        Strs{"R=7", "N=" + n_(d.V), "USED=" + n_(d.used), "TOPK=16", "CH=" + n_(kTopkCh), "COMPACT=0"});
    select = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_select.cl",
        with(kCtrl, {"MAXR=8", "TOPK=16", "QS=16", "V2=1", "RANK=" + n_(d.rank), "VROWS=" + n_(d.V)}));
    select_v1 = std::make_unique<clrun::Program>(dev, "src/kernels/dflash/df_select.cl",
                                                 with(kCtrl, {"MAXR=8", "TOPK=16", "QS=16", "V2=0"}));
  }
  std::unique_ptr<clrun::Program> select_v1, topk_c;   // v1's pick; the compact head's top-16 (COMPACT=1)
  clrun::Program& topk_compact(uint32_t nv) {
    if (!topk_c)
      topk_c = std::make_unique<clrun::Program>(
          dev, "src/kernels/dflash/df_topk.cl",
          Strs{"R=7", "N=" + n_(nv), "USED=" + n_(nv), "TOPK=16", "CH=" + n_(kTopkCh), "COMPACT=1"});
    return *topk_c;
  }
  clrun::Program& gemv(uint32_t M, uint32_t K, uint32_t N) {
    const std::string key = n_(M) + "_" + n_(K) + "_" + n_(N);
    for (size_t i = 0; i < gemv_keys.size(); ++i)
      if (gemv_keys[i] == key) return *gemvs[i];
    gemvs.push_back(std::make_unique<clrun::Program>(dev, "src/kernels/gemv_i8w.cl",
                                                     Strs{"M=" + n_(M), "K=" + n_(K), "N=" + n_(N)}));
    gemv_keys.push_back(key);
    return *gemvs.back();
  }
  // y [M][N] = lin(x [M][K] bf16) through gemv_i8w; `rows` of x from `x0`.
  std::vector<float> lin(const dfref::Lin& L, const std::vector<uint16_t>& x, uint32_t M, uint32_t row0 = 0,
                         uint32_t n_lo = 0, uint32_t n_hi = 0) {
    if (n_hi == 0) n_hi = L.N;
    const uint32_t N = n_hi - n_lo;
    // the tiles of columns [n_lo, n_hi): whole 16-column tiles, contiguous (n-tile-major)
    std::vector<int8_t> q(L.i8.begin() + size_t(n_lo) * L.K, L.i8.begin() + size_t(n_hi) * L.K);
    std::vector<float> s(L.scales.begin() + n_lo, L.scales.begin() + n_hi);
    std::vector<uint16_t> xs(x.begin() + size_t(row0) * L.K, x.begin() + size_t(row0 + M) * L.K);
    clrun::Buffer bq(dev, q), bs(dev, s), bx(dev, xs), by(dev, size_t(M) * N * 4);
    gemv(M, L.K, N).run("gemv_i8w", {size_t(N)}, {64}, bq, bs, bx, by);
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

// The projection list over positions [pos0, pos0 + n), n <= 8.
void project(Gpu& g, const std::vector<uint16_t>& taps, uint32_t pos0, uint32_t n, const std::vector<float>& rope,
             clrun::Buffer& brope, Ring& ring) {
  const dfref::Dims& d = g.d;
  std::vector<uint16_t> t(size_t(8) * d.fck(), 0);
  std::copy(taps.begin() + size_t(pos0) * d.fck(), taps.begin() + size_t(pos0 + n) * d.fck(), t.begin());
  const std::vector<float> f = g.lin(g.w.fc, t, 8);
  clrun::Buffer bf(g.dev, f), bw(g.dev, g.w.hidden_norm), bh(g.dev, size_t(8) * d.H * 4), bx(g.dev, size_t(8) * d.H * 2);
  g.prep->run("df_norm", {256, 8}, {256, 1}, bf, bw, bh, bx);
  const std::vector<uint16_t> x = bx.read<uint16_t>();
  std::vector<float> kv;   // [L][8][2 kvn]
  std::vector<float> qkn;  // [L][q | k][HD]
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::vector<float> o = g.lin(g.w.layers[l].qkv, x, 8, 0, d.qn(), d.qkvn());
    kv.insert(kv.end(), o.begin(), o.end());
    qkn.insert(qkn.end(), g.w.layers[l].q_norm.begin(), g.w.layers[l].q_norm.end());
    qkn.insert(qkn.end(), g.w.layers[l].k_norm.begin(), g.w.layers[l].k_norm.end());
  }
  std::vector<uint32_t> dctl(16, 0);
  dctl[1] = pos0;
  dctl[2] = n;
  clrun::Buffer bd(g.dev, dctl), bkv(g.dev, kv), bqk(g.dev, qkn);
  g.prep->run("df_ctx_store", {size_t(d.layers) * 2 * d.NKV * d.HD, 8}, {d.HD, 1}, bd, bkv, bqk, brope, *ring.k,
              *ring.v);
  (void)rope;
}

struct Out {
  std::vector<float> hidden;
  std::vector<uint32_t> words;
  std::vector<uint32_t> ctrl;   // runtime::Control's words after the list
};

// The draft list. `w` is the drafter (v2 or v1: the convs and the selector by its version); `cids` non-empty: the
// head is w.head over a draft vocabulary of those ids (ascending), its top-16 COMPACT.
Out draft(Gpu& g, const dfref::Weights& w, const Ring& ring, clrun::Buffer& brope, uint32_t anchor, uint32_t p,
          uint32_t K, float T, uint64_t seed, const std::vector<uint32_t>& cids = {}) {
  const dfref::Dims& d = w.d;
  const bool v2 = d.version == 2;
  const uint32_t M = 8, H = d.H;
  std::vector<uint32_t> ctrl(32, 0), dctl(16, 0);
  ctrl[0] = p;
  ctrl[2] = anchor;
  dctl[0] = K;
  std::memcpy(&dctl[3], &T, 4);
  dctl[4] = uint32_t(seed);
  dctl[5] = uint32_t(seed >> 32);
  clrun::Buffer bctrl(g.dev, ctrl), bdctl(g.dev, dctl), bemb(g.dev, w.embed), bmask(g.dev, w.mask_row);
  clrun::Buffer resid(g.dev, size_t(M) * H * 4), h(g.dev, size_t(M) * H * 4), x(g.dev, size_t(M) * H * 2);
  g.prep->run("df_embed", {256, M}, {256, 1}, bctrl, bemb, bmask, d.in_scale, resid);
  const size_t lay = size_t(d.ring) * d.kvn();
  for (uint32_t l = 0; l < d.layers; ++l) {
    const dfref::Layer& L = w.layers[l];
    std::vector<float> base_a(1, 0.0f), base_m(1, 0.0f);   // v1: unused (a buffer must not be empty)
    if (v2) {
      base_a.assign(L.conv_base.begin(), L.conv_base.begin() + 2 * d.ctaps * H);
      base_m.assign(L.conv_base.begin() + 2 * d.ctaps * H, L.conv_base.end());
    }
    clrun::Buffer bba(g.dev, base_a), bbm(g.dev, base_m), bin(g.dev, L.in_norm), bpost(g.dev, L.post_norm);
    std::vector<float> qkn = L.q_norm;
    qkn.insert(qkn.end(), L.k_norm.begin(), L.k_norm.end());
    clrun::Buffer bqk(g.dev, qkn);
    // attention sub-block
    g.prep->run("df_norm", {256, M}, {256, 1}, resid, bin, h, x);
    std::unique_ptr<clrun::Buffer> co;
    clrun::Buffer xc(g.dev, size_t(M) * H * 2);
    if (v2) {
      co = std::make_unique<clrun::Buffer>(g.dev, g.lin(L.akp, x.read<uint16_t>(), M));
      g.prep->run("df_conv_prep", {256, M}, {256, 1}, h, *co, bba, xc);
    }
    clrun::Buffer qkv(g.dev, g.lin(L.qkv, (v2 ? xc : x).read<uint16_t>(), M));
    clrun::Buffer q(g.dev, size_t(M) * d.qn() * 4), kb(g.dev, size_t(M) * d.kvn() * 4), vb(g.dev, size_t(M) * d.kvn() * 4);
    g.prep->run("df_qkv_prep", {size_t(d.NH + 2 * d.NKV) * d.HD, M}, {d.HD, 1}, bctrl, qkv, bqk, brope, q, kb, vb);
    // this layer's ring rows
    const std::vector<uint16_t> rk = ring.k->read<uint16_t>(), rv = ring.v->read<uint16_t>();
    clrun::Buffer lk(g.dev, std::vector<uint16_t>(rk.begin() + l * lay, rk.begin() + (l + 1) * lay)),
        lv(g.dev, std::vector<uint16_t>(rv.begin() + l * lay, rv.begin() + (l + 1) * lay));
    const uint32_t tgt = d.ring / d.pb;
    clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
    g.attn->run("df_attn_part", {size_t(d.NKV) * 16, tgt}, {16, 1}, bctrl, bdctl, q, lk, lv, d.scale, part);
    g.attn->run("df_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bctrl, bdctl, part, q, kb, vb, d.scale, ao);
    clrun::Buffer oo(g.dev, g.lin(L.o, ao.read<uint16_t>(), M));
    if (v2)
      g.prep->run("df_conv_add", {256, M}, {256, 1}, oo, *co, bba, resid);
    else
      g.prep->run("df_add", {256, M}, {256, 1}, oo, resid);
    // MLP sub-block
    g.prep->run("df_norm", {256, M}, {256, 1}, resid, bpost, h, x);
    std::unique_ptr<clrun::Buffer> co2;
    if (v2) {
      co2 = std::make_unique<clrun::Buffer>(g.dev, g.lin(L.mkp, x.read<uint16_t>(), M));
      g.prep->run("df_conv_prep", {256, M}, {256, 1}, h, *co2, bbm, xc);
    }
    clrun::Buffer gu(g.dev, g.lin(L.gate_up, (v2 ? xc : x).read<uint16_t>(), M)), a(g.dev, size_t(M) * d.I * 2);
    g.prep->run("df_silu_mul", {256, M}, {256, 1}, gu, a);
    clrun::Buffer dn(g.dev, g.lin(L.down, a.read<uint16_t>(), M));
    if (v2)
      g.prep->run("df_conv_add", {256, M}, {256, 1}, dn, *co2, bbm, resid);
    else
      g.prep->run("df_add", {256, M}, {256, 1}, dn, resid);
  }
  clrun::Buffer bfin(g.dev, w.final_norm);
  g.prep->run("df_norm", {256, M}, {256, 1}, resid, bfin, h, x);
  const std::vector<uint16_t> xr = x.read<uint16_t>();
  const uint32_t NV = cids.empty() ? d.V : uint32_t(cids.size());
  clrun::Buffer logits(g.dev, g.lin(w.head, xr, 7, 1));
  const uint32_t chunks = (NV + kTopkCh - 1) / kTopkCh;
  clrun::Buffer tpart(g.dev, size_t(7) * chunks * 16 * 2 * 4), cand(g.dev, size_t(7) * 16 * 4),
      unary(g.dev, size_t(7) * 16 * 4), ids(g.dev, cids.empty() ? std::vector<uint32_t>{0} : cids);
  clrun::Program& tk = cids.empty() ? *g.topk : g.topk_compact(NV);
  tk.run("df_topk_stage1", {size_t(chunks) * 256, 7}, {256, 1}, logits, tpart);
  tk.run("df_topk_stage2", {size_t(7) * 256}, {256}, tpart, ids, d.out_mult, d.softcap, cand, unary);
  clrun::Buffer out(g.dev, size_t(404) * 4);
  if (v2) {
    clrun::Buffer hid(g.dev, g.lin(w.sel, xr, M)), pred(g.dev, w.pred), succ(g.dev, w.succ);
    g.select->run("df_select", {256}, {256}, bctrl, bdctl, cand, unary, hid, pred, succ, out);
  } else {
    clrun::Buffer none(g.dev, size_t(4));
    g.select_v1->run("df_select", {256}, {256}, bctrl, bdctl, cand, unary, none, none, none, out);
  }
  Out o;
  o.hidden = h.read<float>();
  o.words = out.read<uint32_t>();
  const std::vector<uint32_t> c = bctrl.read<uint32_t>();
  for (uint32_t j = 0; j < o.words[0] && j < K; ++j)
    if (c[3 + j] != o.words[OUT_IDS + j]) throw clrun::Error("ctrl.cur_token[1 + j] != the readback's id");
  o.ctrl = c;
  return o;
}

// df_attn_part + df_attn_reduce alone on random q / block K, V / ring rows at (p, K) against dfref::attention.
int attn_check(Gpu& g, uint32_t p, uint32_t K) {
  const dfref::Dims& d = g.d;
  const uint32_t M = 8;
  uint64_t s = 0x1234567ull + p * 31 + K;
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
  dctl[0] = K;
  clrun::Buffer bc(g.dev, ctrl), bd(g.dev, dctl), bq(g.dev, q), bkb(g.dev, kb), bvb(g.dev, vb), brk(g.dev, rk),
      brv(g.dev, rv);
  const uint32_t tgt = d.ring / d.pb;
  clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
  g.attn->run("df_attn_part", {size_t(d.NKV) * 16, tgt}, {16, 1}, bc, bd, bq, brk, brv, d.scale, part);
  g.attn->run("df_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bc, bd, part, bq, bkb, bvb, d.scale, ao);
  std::vector<float> want(size_t(M) * d.qn());
  dfref::attention(d.attn(), q.data(), kb.data(), vb.data(), fk.data(), fv.data(), p, K, false, want.data());
  const std::vector<uint16_t> got = ao.read<uint16_t>();
  int over = 0;
  double worst = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double e = std::fabs(dfref::bf(got[i]) - dfref::rf(want[i]));
    worst = std::fmax(worst, e);
    if (e > 1.0 / 128 * std::fmax(1e-3, std::fabs(want[i]))) ++over;
  }
  std::printf("dflash_run: attention alone at p %u, K %u: worst |diff| %.3g, %d over 1 bf16 ulp\n", p, K, worst, over);
  return over != 0;
}

float as_f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int planted_topk(Gpu& g) {
  // ties at the top, a NaN, the masked tail (>= used) the largest, a row of -inf
  const uint32_t V = g.d.V, used = g.d.used;
  std::vector<float> lg(size_t(7) * V);
  for (uint32_t j = 0; j < 7; ++j)
    for (uint32_t i = 0; i < V; ++i) lg[size_t(j) * V + i] = float((i * 37 + j * 11) % 101) * 0.01f;
  for (uint32_t i = used; i < V; ++i) lg[i] = 100.0f;                     // row 0: the mask must hold
  lg[V + 7] = lg[V + 300] = lg[V + 9] = 5.0f;                               // row 1: a three-way tie
  lg[2 * V + 3] = NAN;                                                      // row 2: a NaN never wins
  for (uint32_t i = 0; i < V; ++i) lg[size_t(3) * V + i] = -INFINITY;       // row 3: all -inf
  lg[size_t(3) * V + 5] = 1.0f;
  const uint32_t chunks = (V + kTopkCh - 1) / kTopkCh;
  clrun::Buffer bl(g.dev, lg), tpart(g.dev, size_t(7) * chunks * 16 * 2 * 4), cand(g.dev, size_t(7) * 16 * 4),
      unary(g.dev, size_t(7) * 16 * 4), ids(g.dev, size_t(4));
  g.topk->run("df_topk_stage1", {size_t(chunks) * 256, 7}, {256, 1}, bl, tpart);
  g.topk->run("df_topk_stage2", {size_t(7) * 256}, {256}, tpart, ids, 1.5f, 3.0f, cand, unary);
  const std::vector<uint32_t> c = cand.read<uint32_t>();
  const std::vector<float> u = unary.read<float>();
  int bad = 0;
  for (uint32_t j = 0; j < 7; ++j) {
    const std::vector<dfref::Cand> want = dfref::topk(&lg[size_t(j) * V], used, 16, 1.5f, 3.0f);
    for (uint32_t k = 0; k < 16; ++k)
      if (c[j * 16 + k] != want[k].i || !(u[j * 16 + k] == want[k].v || std::fabs(u[j * 16 + k] - want[k].v) < 1e-6f))
        ++bad;
  }
  std::printf("dflash_run: planted top-16 (mask, a three-way tie, NaN, -inf rows): %s\n", bad ? "DIFFERS" : "exact");
  return bad;
}

}  // namespace

int main() {
  try {
    const dffix::Fixture fx(clrun::repo_root() + "/tests/loader/dflash_ref_fixture.safetensors");
    const dfref::Weights w = dffix::weights(fx, 2, dffix::Form::Int8, dffix::Form::Int8, 8);
    Gpu g(w);
    std::printf("dflash_run on %s: int8 drafter + int8 head, hidden %u, %u layers, window %u / ring %u, vocab %u\n",
                g.dev.name().c_str(), w.d.H, w.d.layers, w.d.win, w.d.ring, w.d.V);
    const dfref::Dims& d = w.d;
    const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
    clrun::Buffer brope(g.dev, rope);
    const std::vector<uint16_t> taps = fx.bf16("taps");
    int bad = planted_topk(g), dv_ok = 0;
    for (auto [p, K] : {std::pair<uint32_t, uint32_t>{1, 7}, {15, 7}, {16, 7}, {17, 5}, {17, 7}, {40, 3}, {123, 6}})
      bad += attn_check(g, p, K);
    const std::vector<dffix::Case> cs = dffix::cases(fx);
    for (size_t ci = 0; ci < cs.size(); ++ci) {
      const dffix::Case& c = cs[ci];
      Ring ring{std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2),
                std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2)};
      dfref::Ring tw = dfref::make_ring(d);
      for (uint32_t pos = 0; pos < c.p; pos += 8) {
        const uint32_t n = std::min<uint32_t>(8, c.p - pos);
        project(g, taps, pos, n, rope, brope, ring);
        dfref::project(w, dfref::Numerics::Engine, &taps[size_t(pos) * d.fck()], n, pos, rope, tw);
      }
      // the ring's rows: the twin's (bf16-widened) against the device's
      {
        const std::vector<uint16_t> rk = ring.k->read<uint16_t>();
        std::vector<float> fk(rk.size());
        for (size_t i = 0; i < rk.size(); ++i) fk[i] = dfref::bf(rk[i]);
        const double e = maxrel(fk, tw.k);
        if (e > 1e-2) {
          std::printf("  case %zu: ring k rows differ from the twin (max rel %.3g)\n", ci, e);
          ++bad;
        }
      }
      const Out o = draft(g, w, ring, brope, c.anchor, c.p, c.K, c.T, c.seed);
      const dfref::DraftOut t = dfref::draft(w, dfref::Numerics::Engine, tw, rope, c.anchor, c.p, c.K, c.T, c.seed);
      double hc = 1;
      for (uint32_t r = 0; r <= c.K; ++r)
        hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * d.H], &t.hidden[size_t(r) * d.H], d.H));
      std::vector<float> sc(size_t(c.K) * 16);
      std::vector<uint32_t> cand(size_t(c.K) * 16);
      for (size_t i = 0; i < sc.size(); ++i) {
        sc[i] = as_f(o.words[OUT_SCORES + i]);
        cand[i] = o.words[OUT_CAND + i];
      }
      const double scos = dfref::cosine(sc.data(), t.scores.data(), sc.size());
      bool walk = true, order = true;
      for (uint32_t j = 0; j < c.K; ++j) {
        walk &= o.words[OUT_IDS + j] == t.ids[j] && o.words[OUT_CHOSEN + j] == t.chosen[j];
        for (uint32_t k = 0; k < 16; ++k)
          if (cand[j * 16 + k] != t.cand[j * 16 + k]) {
            // a swap inside a near tie is not a disagreement
            const float a = as_f(o.words[OUT_UNARY + j * 16 + k]), b = t.unary[j * 16 + k];
            if (std::fabs(a - b) > 1e-4f * std::fmax(1.0f, std::fabs(b))) order = false;
          }
      }
      // against dflash_ref.py's own outputs (printed)
      const std::vector<int32_t> pids = fx.i32("v2.case" + std::to_string(ci) + ".ids");
      uint32_t py_eq = 0;
      for (uint32_t j = 0; j < c.K; ++j) py_eq += o.words[OUT_IDS + j] == uint32_t(pids[j]);
      // The order is reported, not gated: Apple's 1 / sqrt and divide are not correctly rounded (the B70 build's
      // are), so a norm's rstd can sit an ulp off and move later values by ~1e-3 (measured: one case of nine).
      const bool ok = hc >= 0.99999 && scos >= 0.99999 && walk;
      if (!order)
        for (uint32_t j = 0; j < c.K; ++j)
          for (uint32_t k = 0; k < 16; ++k)
            if (cand[j * 16 + k] != t.cand[j * 16 + k])
              std::printf("    row %u slot %u: device id %u unary %.7g, twin id %u unary %.7g\n", j, k, cand[j * 16 + k],
                          double(as_f(o.words[OUT_UNARY + j * 16 + k])), t.cand[j * 16 + k], double(t.unary[j * 16 + k]));
      std::printf("  case %zu (p %2u, K %u, T %.1f): rows cos %.7f, scores cos %.7f, top-16 %s, walk %s | "
                  "dflash_ref.py walk %u / %u\n",
                  ci, c.p, c.K, double(c.T), hc, scos, order ? "=" : "DIFF", walk ? "=" : "DIFF", py_eq, c.K);
      if (!ok) ++bad;

      // Review Focus 3: the same block over a draft vocabulary V' (ascending ids, 256 rows gathered from the int8
      // head by the loader's own gather) holding every full-head candidate of the K rows: candidates, unary, scores
      // and the walk BITWISE the full head's - the selector's codebooks, the walk's predecessor and the readback see
      // real ids. Then V' without row 0's best candidate: the rows differ (the comparison can fail).
      for (int drop = 0; drop < 2; ++drop) {
        std::vector<uint32_t> vp;
        for (uint32_t i = 0; i < c.K * 16; ++i) vp.push_back(cand[i]);
        std::sort(vp.begin(), vp.end());
        vp.erase(std::unique(vp.begin(), vp.end()), vp.end());
        if (drop) vp.erase(std::find(vp.begin(), vp.end(), cand[0]));
        for (uint32_t id = 0; vp.size() < 256 && id < d.used; ++id)
          if (!std::binary_search(vp.begin(), vp.end(), id) && (!drop || id != cand[0])) {
            vp.insert(std::upper_bound(vp.begin(), vp.end(), id), id);
          }
        dfref::Weights wc = w;
        wc.head.N = uint32_t(vp.size());
        wc.head.i8.assign(size_t(wc.head.N) * d.H, 0);
        wc.head.scales.assign(wc.head.N, 0.0f);
        loader::gather_int8_tiled_rows(w.head.i8.data(), w.head.scales.data(), d.H, d.V, vp.data(), wc.head.N,
                                       wc.head.i8.data(), wc.head.scales.data(), 1);
        const Out oc = draft(g, wc, ring, brope, c.anchor, c.p, c.K, c.T, c.seed, vp);
        bool same = true;
        for (uint32_t i = 0; i < c.K * 16; ++i)
          same &= oc.words[OUT_CAND + i] == o.words[OUT_CAND + i] && oc.words[OUT_UNARY + i] == o.words[OUT_UNARY + i] &&
                  oc.words[OUT_SCORES + i] == o.words[OUT_SCORES + i];
        for (uint32_t j = 0; j < c.K; ++j) same &= oc.words[OUT_IDS + j] == o.words[OUT_IDS + j];
        if (same == bool(drop)) {
          std::printf("    case %zu: V' (%zu ids) %s: the draft %s the full head's\n", ci, vp.size(),
                      drop ? "without row 0's best" : "holding every candidate", same ? "EQUALS" : "DIFFERS from");
          ++bad;
        } else {
          ++dv_ok;
        }
      }
    }
    std::printf("dflash_run: draft vocabulary (256 ids, compact int8 head, COMPACT top-16): %d / %zu checks - with every "
                "candidate in V' the draft is bitwise the full head's, without row 0's best it differs\n",
                dv_ok, 2 * cs.size());
    {
      // DFlash v1 (the same weights without the convolutions and the selector): greedy per row, against the twin.
      const dfref::Weights w1 = dffix::weights(fx, 1, dffix::Form::Int8, dffix::Form::Int8, 8);
      int v1_bad = 0;
      for (const dffix::Case& c : cs) {
        if (c.T > 0) continue;
        Ring ring{std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2),
                  std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2)};
        dfref::Ring tw = dfref::make_ring(d);
        for (uint32_t pos = 0; pos < c.p; pos += 8) {
          const uint32_t n = std::min<uint32_t>(8, c.p - pos);
          project(g, taps, pos, n, rope, brope, ring);
          dfref::project(w1, dfref::Numerics::Engine, &taps[size_t(pos) * d.fck()], n, pos, rope, tw);
        }
        const Out o = draft(g, w1, ring, brope, c.anchor, c.p, c.K, 0.0f, 0);
        const dfref::DraftOut t = dfref::draft(w1, dfref::Numerics::Engine, tw, rope, c.anchor, c.p, c.K);
        for (uint32_t j = 0; j < c.K; ++j)
          if (o.words[OUT_IDS + j] != t.ids[j] || o.words[OUT_CHOSEN + j] != 0) ++v1_bad;
        double hc = 1;
        for (uint32_t r = 0; r <= c.K; ++r)
          hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * d.H], &t.hidden[size_t(r) * d.H], d.H));
        if (hc < 0.99999) ++v1_bad;
        // v1 at T != 0 (an engine bug past set_draft_params' refusal): no draft at all - readback k 0 and
        // ctrl.cur_token[1..] untouched - rather than greedy drafts under a sampled request's acceptance.
        const Out s = draft(g, w1, ring, brope, c.anchor, c.p, c.K, 0.7f, 5);
        if (s.words[0] != 0 || s.ctrl[3] != 0) ++v1_bad;
      }
      std::printf("dflash_run: DFlash v1 (no convs, no selector), greedy, %zu cases: %s\n", cs.size() - 2,
                  v1_bad ? "DIFFERS" : "rows and ids = the twin");
      bad += v1_bad;
    }
    if (bad) {
      std::printf("DISAGREES with the host twin (indicative: read the kernels before the box)\n");
      return 1;
    }
    std::printf("agrees with the host twin (indicative only; the B70 is the real test)\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dflash_run: %s\n", e.what());
    return 1;
  }
}
