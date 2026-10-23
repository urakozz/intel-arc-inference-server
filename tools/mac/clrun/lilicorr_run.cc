// lilicorr_run - spec 26b's LiLiCorr drafter kernels (src/kernels/dflash/ll_*.cl) and the reused df_prep / GEMVs on the
// Mac's OpenCL GPU at the fixture's tiny shapes (tests/loader/lilicorr_dflash_fixture.safetensors: hidden 128, 2 layers,
// 4 / 1 heads x 32, FF 256, 6 taps, window 16 over a 16-slot ring in two PB 8 blocks, sinks, the head 64 wide / 2 layers
// / 4 heads / factor 64, top-8, vocab 1024), against the host twin tests/kernels/lilicorr_ref.h (INDICATIVE ONLY:
// clrun.h says what this can and cannot show). The drafter is the fixture's own NVFP4 checkpoint read by the loader's host
// half (loader::LiliCheckpoint) - the device forms the card will hold.
//
//   lilicorr_run      no arguments
//
// What runs, in runtime/dflash/lilicorr_drafter.cc's order: the projection list (fc, df_norm with hidden_norm, ll_anchor,
// per layer the k||v rows of q||k||v, df_ctx_store) in chunks of up to 8 positions, and the draft list over the whole
// 8-row block - df_embed, per layer df_norm, the conv kernel projection, df_conv_prep, q||k||v, df_qkv_prep, ll_attn_part
// + ll_attn_reduce (the sinks), o, df_conv_add, df_norm, the projection, df_conv_prep, gate||up, df_silu_mul, down,
// df_conv_add; the final df_norm, the target head over rows 1..7, ll_topk_stage1/2, the LiLiCorr head (ll_gemm_gather,
// ll_gemm, ll_feat, ll_sum, ll_valid, ll_rmsnorm, ll_lattn, ll_factor, ll_l2norm, ll_scores) and ll_select - for every
// fixture case in three arms: the exact drafter (ll_gemv_bf16_g for its NVFP4 linears, gemv_bf16 for the BF16 ones) with
// the bf16 head, int8 (gemv_i8w) with the int8 head, int4 g64 (gemv.cl LAYOUT 1, S 1, when it builds here) with the int8
// head. Also: ll_attn alone at the window's edge and past the ring's wrap; ll_topk alone on planted rows (ties, NaN, a -inf
// chunk, a huge value, three chunks) exact.
//
// Bars (against the twin in Engine numerics; Apple's exp / log / sqrt / divide are not correctly rounded): the rows'
// cosine >= 0.99999, the candidates equal, the lattice's cosine >= 0.99999, the walk's ids equal, ctrl.cur_token[1 + s] =
// the readback's ids, the anchor = the projection's last row.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "clrun.h"
#include "kernels/lilicorr_fixture_io.h"
#include "kernels/lilicorr_ref.h"

namespace {

using Strs = std::vector<std::string>;
const Strs kCtrl = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2",  "CTRL_OUT=10",    "CTRL_DEBUG=18",   "DCTL_K=0",
                    "DCTL_PPOS=1", "DCTL_PN=2",  "DCTL_TEMP=3", "DCTL_SEED_LO=4", "DCTL_SEED_HI=5", "DCTL_AVALID=6"};
Strs with(Strs a, const Strs& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
std::string n_(uint32_t v) { return std::to_string(v); }
constexpr uint32_t M = 8, OUT_IDS = 4, OUT_CAND = 20, OUT_LAT = 244, OUT_WORDS = 692;

float as_f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct HeadDev {
  std::unique_ptr<clrun::Buffer> wt, gb;
  uint32_t K = 0, N = 0;
};

struct Gpu {
  clrun::Device dev;
  llref::Dims d;
  std::unique_ptr<clrun::Program> prep, attn, topk, lattice, select;
  std::map<std::string, std::unique_ptr<clrun::Program>> gemvs;

  explicit Gpu(const llref::Dims& dd) : d(dd) {
    prep = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/df_prep.cl",
        with(kCtrl, {"M=8", "H=" + n_(d.H), "WG=256", "EPS=1e-06f", "VOCAB=" + n_(d.V), "GROUP=" + n_(d.group),
                     "CTAPS=" + n_(d.ctaps), "INTER=" + n_(d.I), "NH=" + n_(d.NH), "NKV=" + n_(d.NKV),
                     "HD=" + n_(d.HD), "RING=" + n_(d.ring)}));
    attn = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/ll_attn.cl",
        with(kCtrl, {"M=8", "NH=" + n_(d.NH), "NKV=" + n_(d.NKV), "HD=" + n_(d.HD), "RING=" + n_(d.ring),
                     "WIN=" + n_(d.win), "PB=" + n_(d.pb), "HPW=" + n_(hpw()), "WGA=16"}));
    topk = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/ll_topk.cl", Strs{"R=7", "N=" + n_(d.V), "USED=" + n_(d.used), "TOPK=8", "CH=4096"});
    lattice = std::make_unique<clrun::Program>(
        dev, "src/kernels/dflash/ll_lattice.cl",
        with(kCtrl, {"S=7", "KC=8", "HL=" + n_(d.Hl), "LHEADS=" + n_(d.lheads), "BLK=8", "RMAX=64", "GW=64", "KT=16",
                     "EPS=1e-06f", "FEPS=1e-05f"}));
    select = std::make_unique<clrun::Program>(dev, "src/kernels/dflash/ll_select.cl", with(kCtrl, {"S=7", "KC=8"}));
  }
  uint32_t hpw() const { return std::max(1u, d.NH / d.NKV / 2); }   // two part groups a kv head, as the card's 16 / 8
  clrun::Program& program(const std::string& key, const std::string& src, const Strs& defs) {
    auto it = gemvs.find(key);
    if (it == gemvs.end()) it = gemvs.emplace(key, std::make_unique<clrun::Program>(dev, src, defs)).first;
    return *it->second;
  }
  // y [m][n_hi - n_lo] = L(x [m][K] bf16) for whole 16-column tiles [n_lo, n_hi) - the capture's binding for L's form.
  std::vector<float> lin(const loader::DFlashHostLinear& L, const std::vector<uint16_t>& x, uint32_t m, uint32_t row0 = 0,
                         uint32_t n_lo = 0, uint32_t n_hi = 0) {
    if (n_hi == 0) n_hi = L.N;
    const uint32_t N = n_hi - n_lo, K = L.K;
    std::vector<uint16_t> xs(x.begin() + size_t(row0) * K, x.begin() + size_t(row0 + m) * K);
    clrun::Buffer bx(dev, xs), by(dev, size_t(m) * N * 4);
    const std::string mkn = "M=" + n_(m);
    if (L.kind == model::WeightKind::Int4) {
      std::vector<uint32_t> sub(L.int4.begin() + size_t(n_lo / 16) * (K / 64) * 136,
                                L.int4.begin() + size_t(n_hi / 16) * (K / 64) * 136);
      clrun::Buffer bw(dev, sub);
      program("q4_" + n_(m) + "_" + n_(K) + "_" + n_(N), "src/kernels/gemv.cl",
              Strs{mkn, "K=" + n_(K), "N=" + n_(N), "S=1", "LAYOUT=1"})
          .run("gemv", {size_t(N), 1}, {64, 1}, bw, bw, bx, by);
    } else if (L.kind == model::WeightKind::Int8) {
      std::vector<int8_t> q(L.i8.begin() + size_t(n_lo) * K, L.i8.begin() + size_t(n_hi) * K);
      std::vector<float> s(L.scales.begin() + n_lo, L.scales.begin() + n_hi);
      clrun::Buffer bq(dev, q), bs(dev, s);
      program("i8_" + n_(m) + "_" + n_(K) + "_" + n_(N), "src/kernels/gemv_i8w.cl", Strs{mkn, "K=" + n_(K), "N=" + n_(N)})
          .run("gemv_i8w", {size_t(N)}, {64}, bq, bs, bx, by);
    } else {
      const uint32_t ks = llfix::bf16_ksplit(N), cols = ks == 1 ? 64 : 16;
      std::vector<uint16_t> w(L.bf16.begin() + size_t(n_lo) * K, L.bf16.begin() + size_t(n_hi) * K);
      clrun::Buffer bw(dev, w);
      Strs defs{mkn, "K=" + n_(K), "N=" + n_(N)};
      if (cols != 64) defs.push_back("COLS_PER_WG=" + n_(cols));
      if (ks != 1) defs.push_back("KSPLIT=" + n_(ks));
      if (!L.gs.empty()) {
        std::vector<float> gs(L.gs.begin() + n_lo, L.gs.begin() + n_hi);
        clrun::Buffer bg(dev, gs);
        program("g_" + n_(m) + "_" + n_(K) + "_" + n_(N), "src/kernels/dflash/ll_gemv_bf16_g.cl", defs)
            .run("ll_gemv_bf16_g", {size_t(N) * ks}, {size_t(cols) * ks}, bw, bx, bg, by);
      } else {
        program("b_" + n_(m) + "_" + n_(K) + "_" + n_(N), "src/kernels/gemv_bf16.cl", defs)
            .run("gemv_bf16", {size_t(N) * ks}, {size_t(cols) * ks}, bw, bx, by);
      }
    }
    return by.read<float>();
  }
  // ll_gemm over an fp32 buffer: y [rows][N] (mode 1: added into y).
  void gemm(const clrun::Buffer& x, uint32_t xoff, uint32_t ldx, const HeadDev& w, clrun::Buffer& y, uint32_t rows,
            uint32_t mode, uint32_t act) {
    lattice->run("ll_gemm", {size_t(w.N)}, {64}, x, xoff, ldx, *w.wt, *w.gb, y, rows, w.K, w.N, mode, act);
  }
};

HeadDev head_dev(Gpu& g, const loader::LiliHostHead& h) {
  HeadDev o;
  o.K = h.K;
  o.N = h.N;
  o.wt = std::make_unique<clrun::Buffer>(g.dev, h.wt);
  std::vector<float> gb(h.gs);
  gb.insert(gb.end(), h.bias.begin(), h.bias.end());
  o.gb = std::make_unique<clrun::Buffer>(g.dev, gb);
  return o;
}

// The drafter in one arm on the device: the backbone's linears (the loader's own pieces), the head, the small tensors.
struct Arm {
  const char* name;
  llfix::Form f;
  llfix::HeadForm hf;
  loader::DFlashHostLinear fc;
  std::vector<std::vector<loader::DFlashHostLinear>> layer;   // [l][qkv, o, gate_up, down, akp, mkp]
  loader::DFlashHostLinear head;                              // the target's
  llref::Weights tw;                                          // the twin's
};

Arm make_arm(const char* name, const dffix::Fixture& fx, const loader::LiliCheckpoint& ck, llfix::Form f, llfix::HeadForm hf) {
  Arm a{name, f, hf, {}, {}, {}, llfix::weights(fx, ck, f, hf, 8)};
  const loader::DraftFormat df = llfix::device_format(f);
  a.fc = ck.fc(df, 1);
  using DL = loader::DFlashLinear;
  for (uint32_t l = 0; l < ck.desc().layers; ++l) {
    std::vector<loader::DFlashHostLinear> v;
    for (DL w : {DL::Qkv, DL::O, DL::GateUp, DL::Down, DL::AttnKproj, DL::MlpKproj}) v.push_back(ck.layer_linear(l, w, df, 1));
    a.layer.push_back(std::move(v));
  }
  const dfref::Lin& h = a.tw.head;
  a.head.K = h.K;
  a.head.N = h.N;
  if (!h.i8.empty()) {
    a.head.kind = model::WeightKind::Int8;
    a.head.i8 = h.i8;
    a.head.scales = h.scales;
  } else {
    a.head.kind = model::WeightKind::Bf16;
    a.head.bf16 = h.bf16;
  }
  return a;
}

struct Ring {
  std::unique_ptr<clrun::Buffer> k, v;
};

// The projection list over positions [pos0, pos0 + n), n <= 8; the anchor buffer gets row n - 1 (ll_anchor).
void project(Gpu& g, const Arm& a, const loader::LiliCheckpoint& ck, const std::vector<uint16_t>& taps, uint32_t pos0,
             uint32_t n, clrun::Buffer& brope, Ring& ring, clrun::Buffer& anchor) {
  const llref::Dims& d = g.d;
  std::vector<uint16_t> t(size_t(M) * d.fck(), 0);
  std::copy(taps.begin() + size_t(pos0) * d.fck(), taps.begin() + size_t(pos0 + n) * d.fck(), t.begin());
  clrun::Buffer bf(g.dev, g.lin(a.fc, t, M)), bw(g.dev, ck.hidden_norm()), bh(g.dev, size_t(M) * d.H * 4),
      bx(g.dev, size_t(M) * d.H * 2);
  g.prep->run("df_norm", {256, M}, {256, 1}, bf, bw, bh, bx);
  std::vector<uint32_t> dctl(16, 0);
  dctl[1] = pos0;
  dctl[2] = n;
  clrun::Buffer bd(g.dev, dctl);
  g.lattice->run("ll_anchor", {size_t((d.H + 255) / 256) * 256}, {256}, bd, bh, d.H, anchor);
  const std::vector<uint16_t> x = bx.read<uint16_t>();
  std::vector<float> kv, qkn;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::vector<float> o = g.lin(a.layer[l][0], x, M, 0, d.qn(), d.qkvn());
    kv.insert(kv.end(), o.begin(), o.end());
    qkn.insert(qkn.end(), a.tw.layers[l].q_norm.begin(), a.tw.layers[l].q_norm.end());
    qkn.insert(qkn.end(), a.tw.layers[l].k_norm.begin(), a.tw.layers[l].k_norm.end());
  }
  clrun::Buffer bkv(g.dev, kv), bqk(g.dev, qkn);
  g.prep->run("df_ctx_store", {size_t(d.layers) * 2 * d.NKV * d.HD, M}, {d.HD, 1}, bd, bkv, bqk, brope, *ring.k, *ring.v);
}

struct Out {
  std::vector<float> hidden;
  std::vector<uint32_t> words, ctrl;
};

Out draft(Gpu& g, const Arm& a, const loader::LiliCheckpoint& ck, const std::vector<uint16_t>& embed, const Ring& ring,
          clrun::Buffer& brope, clrun::Buffer& anchor, bool valid, uint32_t anchor_id, uint32_t p, uint32_t K, float T,
          uint64_t seed) {
  const llref::Dims& d = g.d;
  const uint32_t H = d.H, S = d.S, KC = d.k, R = d.R(), Hl = d.Hl;
  std::vector<uint32_t> ctrl(32, 0), dctl(16, 0);
  ctrl[0] = p;
  ctrl[2] = anchor_id;
  dctl[0] = 7;   // the whole block (spec 26 §12 ruling 1); the walk takes dctl.k = K below
  std::memcpy(&dctl[3], &T, 4);
  dctl[4] = uint32_t(seed);
  dctl[5] = uint32_t(seed >> 32);
  dctl[6] = valid ? 1 : 0;
  clrun::Buffer bctrl(g.dev, ctrl), bdctl(g.dev, dctl), bemb(g.dev, embed), bmask(g.dev, a.tw.mask_row);
  clrun::Buffer resid(g.dev, size_t(M) * H * 4), h(g.dev, size_t(M) * H * 4), x(g.dev, size_t(M) * H * 2);
  g.prep->run("df_embed", {256, M}, {256, 1}, bctrl, bemb, bmask, d.in_scale, resid);
  const size_t lay = size_t(d.ring) * d.kvn();
  const std::vector<float> sinks = ck.sinks();
  for (uint32_t l = 0; l < d.layers; ++l) {
    const llref::Layer& L = a.tw.layers[l];
    const std::vector<loader::DFlashHostLinear>& W = a.layer[l];
    std::vector<float> base_a(L.conv_base.begin(), L.conv_base.begin() + 2 * d.ctaps * H),
        base_m(L.conv_base.begin() + 2 * d.ctaps * H, L.conv_base.end()), qkn = L.q_norm;
    qkn.insert(qkn.end(), L.k_norm.begin(), L.k_norm.end());
    clrun::Buffer bba(g.dev, base_a), bbm(g.dev, base_m), bin(g.dev, L.in_norm), bpost(g.dev, L.post_norm), bqk(g.dev, qkn),
        bsink(g.dev, std::vector<float>(sinks.begin() + l * d.NH, sinks.begin() + (l + 1) * d.NH));
    g.prep->run("df_norm", {256, M}, {256, 1}, resid, bin, h, x);
    clrun::Buffer co(g.dev, g.lin(W[4], x.read<uint16_t>(), M)), xc(g.dev, size_t(M) * H * 2);
    g.prep->run("df_conv_prep", {256, M}, {256, 1}, h, co, bba, xc);
    clrun::Buffer qkv(g.dev, g.lin(W[0], xc.read<uint16_t>(), M));
    clrun::Buffer q(g.dev, size_t(M) * d.qn() * 4), kb(g.dev, size_t(M) * d.kvn() * 4), vb(g.dev, size_t(M) * d.kvn() * 4);
    g.prep->run("df_qkv_prep", {size_t(d.NH + 2 * d.NKV) * d.HD, M}, {d.HD, 1}, bctrl, qkv, bqk, brope, q, kb, vb);
    const std::vector<uint16_t> rk = ring.k->read<uint16_t>(), rv = ring.v->read<uint16_t>();
    clrun::Buffer lk(g.dev, std::vector<uint16_t>(rk.begin() + l * lay, rk.begin() + (l + 1) * lay)),
        lv(g.dev, std::vector<uint16_t>(rv.begin() + l * lay, rv.begin() + (l + 1) * lay));
    const uint32_t tgt = d.ring / d.pb, groups = d.NKV * (d.NH / d.NKV / g.hpw());
    clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
    g.attn->run("ll_attn_part", {size_t(groups) * 16, tgt}, {16, 1}, bctrl, bdctl, q, lk, lv, d.scale, part);
    g.attn->run("ll_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bctrl, bdctl, part, q, kb, vb, bsink, d.scale, ao);
    clrun::Buffer oo(g.dev, g.lin(W[1], ao.read<uint16_t>(), M));
    g.prep->run("df_conv_add", {256, M}, {256, 1}, oo, co, bba, resid);
    g.prep->run("df_norm", {256, M}, {256, 1}, resid, bpost, h, x);
    clrun::Buffer co2(g.dev, g.lin(W[5], x.read<uint16_t>(), M));
    g.prep->run("df_conv_prep", {256, M}, {256, 1}, h, co2, bbm, xc);
    clrun::Buffer gu(g.dev, g.lin(W[2], xc.read<uint16_t>(), M)), am(g.dev, size_t(M) * d.I * 2);
    g.prep->run("df_silu_mul", {256, M}, {256, 1}, gu, am);
    clrun::Buffer dn(g.dev, g.lin(W[3], am.read<uint16_t>(), M));
    g.prep->run("df_conv_add", {256, M}, {256, 1}, dn, co2, bbm, resid);
  }
  clrun::Buffer bfin(g.dev, a.tw.final_norm);
  g.prep->run("df_norm", {256, M}, {256, 1}, resid, bfin, h, x);
  // the target head over rows 1..7, the top-8 with log-probs
  clrun::Buffer logits(g.dev, g.lin(a.head, x.read<uint16_t>(), S, 1));
  const uint32_t chunks = (d.V + 4095) / 4096;
  clrun::Buffer tpart(g.dev, size_t(S) * chunks * (2 * KC + 2) * 4), cand(g.dev, size_t(S) * KC * 4),
      blogit(g.dev, size_t(S) * KC * 4), blp(g.dev, size_t(S) * KC * 4);
  g.topk->run("ll_topk_stage1", {size_t(chunks) * 256, S}, {256, 1}, logits, tpart);
  g.topk->run("ll_topk_stage2", {size_t(S) * 256}, {256}, tpart, cand, blogit, blp);
  // the LiLiCorr head
  using LH = loader::LiliHead;
  const HeadDev tok = head_dev(g, ck.head_linear(LH::TokenProj)), pas = head_dev(g, ck.head_linear(LH::PassHidden)),
                fup = head_dev(g, ck.head_linear(LH::FeatUp)), fdn = head_dev(g, ck.head_linear(LH::FeatDown)),
                ctx = head_dev(g, ck.head_linear(LH::ContextProj)), fac = head_dev(g, ck.head_linear(LH::Factor)),
                oin = head_dev(g, ck.head_linear(LH::OutIn)), aout = head_dev(g, ck.head_linear(LH::AnchorOut));
  const loader::LiliHeadSmall so = loader::lili_head_small(ck.desc());
  clrun::Buffer small(g.dev, ck.head_small());
  clrun::Buffer btok(g.dev, size_t(R) * Hl * 4), bpas(g.dev, size_t(S) * Hl * 4), bf(g.dev, size_t(R) * 5 * 4),
      bfu(g.dev, size_t(R) * Hl * 4), bfm(g.dev, size_t(R) * Hl * 4), bxh(g.dev, size_t(R) * Hl * 4),
      bxn(g.dev, size_t(R) * Hl * 4), bqkv(g.dev, size_t(R) * 3 * Hl * 4), bo(g.dev, size_t(R) * Hl * 4),
      bu(g.dev, size_t(R) * d.lmlp * 4), ba(g.dev, size_t(Hl) * 4), ban(g.dev, size_t(Hl) * 4),
      bfh(g.dev, size_t(R) * Hl * 4), bov(g.dev, size_t(R) * 2 * d.F * 4), bao(g.dev, size_t(d.F) * 4),
      blat(g.dev, size_t(S) * KC * KC * 4);
  g.lattice->run("ll_gemm_gather", {size_t(tok.N)}, {64}, bemb, cand, *tok.wt, *tok.gb, btok, R, tok.K, tok.N, d.V);
  g.gemm(h, H, H, pas, bpas, S, 0, 0);
  g.lattice->run("ll_feat", {64}, {64}, blp, small, uint32_t(so.fn_w), bf);
  g.gemm(bf, 0, 5, fup, bfu, R, 0, 1);
  g.gemm(bfu, 0, Hl, fdn, bfm, R, 0, 0);
  g.lattice->run("ll_sum", {size_t((Hl + 255) / 256) * 256, R}, {256, 1}, btok, bpas, bfm, small, uint32_t(so.slot),
                 uint32_t(so.rank), bxh);
  g.gemm(anchor, 0, H, ctx, ba, 1, 0, 0);
  g.lattice->run("ll_valid", {size_t((Hl + 255) / 256) * 256}, {256}, bdctl, ba);
  const float lscale = float(1.0 / std::sqrt(double(d.lhd())));
  for (uint32_t l = 0; l < d.llayers; ++l) {
    const HeadDev in = head_dev(g, ck.head_linear(LH::InProj, l)), op = head_dev(g, ck.head_linear(LH::OutProj, l)),
                  up = head_dev(g, ck.head_linear(LH::MlpUp, l)), dw = head_dev(g, ck.head_linear(LH::MlpDown, l));
    g.lattice->run("ll_rmsnorm", {256, R}, {256, 1}, bxh, small, uint32_t(so.attn_norm(l, Hl)), bxn, Hl);
    g.gemm(bxn, 0, Hl, in, bqkv, R, 0, 0);
    g.lattice->run("ll_lattn", {size_t(d.lheads) * 64, R}, {64, 1}, bqkv, small, uint32_t(so.rel), uint32_t(so.same),
                   lscale, bo);
    g.gemm(bo, 0, Hl, op, bxh, R, 1, 0);
    g.lattice->run("ll_rmsnorm", {256, R}, {256, 1}, bxh, small, uint32_t(so.mlp_norm(l, Hl)), bxn, Hl);
    g.gemm(bxn, 0, Hl, up, bu, R, 0, 1);
    g.gemm(bu, 0, d.lmlp, dw, bxh, R, 1, 0);
  }
  g.lattice->run("ll_rmsnorm", {256, R}, {256, 1}, bxh, small, uint32_t(so.out_norm), bxn, Hl);
  g.lattice->run("ll_rmsnorm", {256, 1}, {256, 1}, ba, small, uint32_t(so.anchor_norm), ban, Hl);
  g.lattice->run("ll_factor", {size_t(Hl)}, {64}, bxn, ban, *fac.wt, *fac.gb, bfh);
  g.gemm(bfh, 0, Hl, oin, bov, R, 0, 0);
  g.lattice->run("ll_l2norm", {2 * 256, R}, {256, 1}, bov, d.F, 2 * d.F, d.veps);
  g.gemm(ban, 0, Hl, aout, bao, 1, 0, 0);
  g.lattice->run("ll_l2norm", {256, 1}, {256, 1}, bao, d.F, d.F, d.veps);
  g.lattice->run("ll_scores", {size_t(S) * KC, KC}, {KC, 1}, bov, bao, d.F, d.lscale, blat);
  // the walk
  std::vector<uint32_t> dk = bdctl.read<uint32_t>();
  dk[0] = K;
  bdctl.write(dk.data(), dk.size() * 4);
  clrun::Buffer out(g.dev, size_t(OUT_WORDS) * 4);
  g.select->run("ll_select", {64}, {64}, bctrl, bdctl, cand, blp, blogit, blat, out);
  Out o;
  o.hidden = h.read<float>();
  o.words = out.read<uint32_t>();
  o.ctrl = bctrl.read<uint32_t>();
  for (uint32_t s = 0; s < o.words[0]; ++s)
    if (o.ctrl[3 + s] != o.words[OUT_IDS + s]) throw clrun::Error("ctrl.cur_token[1 + s] != the readback's id");
  return o;
}

int run_arm(Gpu& g, const dffix::Fixture& fx, const loader::LiliCheckpoint& ck, const Arm& a,
            const std::vector<float>& rope, clrun::Buffer& brope) {
  const llref::Dims& d = g.d;
  const std::vector<uint16_t> taps = fx.bf16("taps"), embed = fx.bf16("target.embed");
  int bad = 0;
  for (const dffix::Case& c : dffix::cases(fx)) {
    Ring ring{std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2),
              std::make_unique<clrun::Buffer>(g.dev, size_t(d.layers) * d.ring * d.kvn() * 2)};
    clrun::Buffer anchor(g.dev, size_t(d.H) * 4);
    for (uint32_t pos = 0; pos < c.p; pos += M) project(g, a, ck, taps, pos, std::min(M, c.p - pos), brope, ring, anchor);
    const llfix::RingAt tw = llfix::ring_at(fx, a.tw, dfref::Numerics::Engine, rope, c.p);
    const std::vector<float> ga = anchor.read<float>();
    const double anc = dfref::cosine(ga.data(), tw.anchor.data(), d.H);
    const Out o = draft(g, a, ck, embed, ring, brope, anchor, c.p > 0, c.anchor, c.p, c.K, c.T, c.seed);
    const llref::DraftOut want = llref::draft(a.tw, dfref::Numerics::Engine, tw.ring, rope, tw.anchor, tw.valid, c.anchor,
                                              c.p, c.K, c.T, c.seed);
    double rows = 1;
    for (uint32_t r = 0; r < M; ++r) rows = std::fmin(rows, dfref::cosine(&o.hidden[size_t(r) * d.H], &want.hidden[size_t(r) * d.H], d.H));
    bool cands = true, walk = o.words[0] == c.K;
    for (uint32_t i = 0; i < d.S * d.k; ++i) cands &= o.words[OUT_CAND + i] == want.cand[i];
    std::vector<float> lat(d.S * d.k * d.k);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = as_f(o.words[OUT_LAT + i]);
    const double lc = dfref::cosine(lat.data(), want.lattice.data(), lat.size());
    for (uint32_t s = 0; s < c.K && walk; ++s) walk &= o.words[OUT_IDS + s] == want.walk.ids[s];
    const bool ok = rows >= 0.99999 && cands && lc >= 0.99999 && walk && (c.p == 0 || anc >= 0.999999);
    std::printf("  %s case (p %u, K %u, T %.1f): rows cos %.7f, anchor cos %.7f, candidates %s, lattice cos %.7f, walk %s%s\n",
                a.name, c.p, c.K, double(c.T), rows, c.p ? anc : 1.0, cands ? "=" : "DIFF", lc, walk ? "=" : "DIFF",
                ok ? "" : "  <-- DISAGREES");
    bad += !ok;
  }
  return bad;
}

// ll_attn alone on random q / block K, V / ring rows at (p, K) against llref::attention_sink.
int attn_check(Gpu& g, uint32_t p, uint32_t K) {
  const llref::Dims& d = g.d;
  uint64_t s = 0x2611ull + p * 31 + K;
  auto rnd = [&]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return float(int32_t(s & 0xFFFF) - 32768) / 32768.0f;
  };
  std::vector<float> q(size_t(M) * d.qn()), kb(size_t(M) * d.kvn()), vb(kb.size()), sink(d.NH);
  for (float& x : q) x = rnd() * 3.0f;
  for (float& x : kb) x = rnd();
  for (float& x : vb) x = rnd();
  for (float& x : sink) x = rnd() * 2.0f;
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
      brv(g.dev, rv), bs(g.dev, sink);
  const uint32_t tgt = d.ring / d.pb, groups = d.NKV * (d.NH / d.NKV / g.hpw());
  clrun::Buffer part(g.dev, size_t(d.NH) * tgt * M * (d.HD + 2) * 4), ao(g.dev, size_t(M) * d.qn() * 2);
  g.attn->run("ll_attn_part", {size_t(groups) * 16, tgt}, {16, 1}, bc, bd, bq, brk, brv, d.scale, part);
  g.attn->run("ll_attn_reduce", {size_t(d.NH) * d.HD, M}, {d.HD, 1}, bc, bd, part, bq, bkb, bvb, bs, d.scale, ao);
  std::vector<float> want(size_t(M) * d.qn());
  llref::attention_sink(d.attn(), q.data(), kb.data(), vb.data(), fk.data(), fv.data(), p, K, sink.data(), want.data());
  const std::vector<uint16_t> got = ao.read<uint16_t>();
  int over = 0;
  double worst = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double e = std::fabs(dfref::bf(got[i]) - dfref::rf(want[i]));
    worst = std::fmax(worst, e);
    if (e > 1.0 / 128 * std::fmax(1e-3, std::fabs(want[i]))) ++over;
  }
  std::printf("lilicorr_run: ll_attn alone at p %u, K %u: worst |diff| %.3g, %d over 1 bf16 ulp\n", p, K, worst, over);
  return over != 0;
}

// ll_topk alone on planted rows over three chunks.
int topk_check(Gpu& g) {
  const uint32_t V = 3 * 4096 + 100, R = 7;
  clrun::Program tk(g.dev, "src/kernels/dflash/ll_topk.cl", Strs{"R=7", "N=" + n_(V), "USED=" + n_(V), "TOPK=8", "CH=4096"});
  std::vector<float> lg(size_t(R) * V);
  for (uint32_t j = 0; j < R; ++j)
    for (uint32_t i = 0; i < V; ++i) lg[size_t(j) * V + i] = std::sin(float(i * (j + 1)) * 0.37f) * 4.0f;
  lg[5] = lg[7] = lg[9000] = 9.0f;                                                        // row 0: a three-way tie
  lg[V + 11] = NAN;                                                                       // row 1: NaN
  for (uint32_t i = 4096; i < 8192; ++i) lg[size_t(2) * V + i] = -INFINITY;               // row 2: a -inf chunk
  lg[size_t(3) * V + 3] = 3.0e38f;                                                        // row 3: huge
  for (uint32_t i = 0; i < V; ++i) lg[size_t(4) * V + i] = (i % 3 == 0) ? -INFINITY : 1.0f;   // row 4: many ties
  const uint32_t chunks = (V + 4095) / 4096;
  clrun::Buffer bl(g.dev, lg), part(g.dev, size_t(R) * chunks * 18 * 4), cand(g.dev, size_t(R) * 8 * 4),
      lo(g.dev, size_t(R) * 8 * 4), lp(g.dev, size_t(R) * 8 * 4);
  tk.run("ll_topk_stage1", {size_t(chunks) * 256, R}, {256, 1}, bl, part);
  tk.run("ll_topk_stage2", {size_t(R) * 256}, {256}, part, cand, lo, lp);
  const std::vector<uint32_t> c = cand.read<uint32_t>();
  const std::vector<float> v = lo.read<float>(), l = lp.read<float>();
  int bad = 0;
  double worst = 0;
  for (uint32_t j = 0; j < R; ++j) {
    const llref::TopK t = llref::topk_lse(&lg[size_t(j) * V], V, 8);
    for (uint32_t k = 0; k < 8; ++k) {
      bad += c[j * 8 + k] != t.id[k] || v[j * 8 + k] != t.v[k] || std::isnan(l[j * 8 + k]);
      worst = std::fmax(worst, std::fabs(double(l[j * 8 + k]) - t.lp[k]) / std::fmax(1.0, std::fabs(double(t.lse))));
    }
  }
  bad += worst > 1e-6;
  std::printf("lilicorr_run: planted top-8 (a three-way tie, NaN, a -inf chunk, a huge value, many ties; 3 chunks): ids "
              "and logits %s, log-probs within %.2e of the twin\n", bad ? "DIFFER" : "exact", worst);
  return bad;
}

}  // namespace

int main() {
  try {
    const dffix::Fixture fx(clrun::repo_root() + "/tests/loader/lilicorr_dflash_fixture.safetensors");
    const std::string dir = llfix::write_checkpoint(fx, "clrun");
    const loader::LiliCheckpoint ck(dir);
    std::vector<Arm> arms;
    arms.push_back(make_arm("exact + bf16 head", fx, ck, llfix::Form::Exact, llfix::HeadForm::Bf16));
    arms.push_back(make_arm("int8 + int8 head", fx, ck, llfix::Form::Int8, llfix::HeadForm::Int8));
    Gpu g(arms[0].tw.d);
    const llref::Dims& d = g.d;
    std::printf("lilicorr_run on %s: hidden %u, %u layers, %u / %u heads x %u, window %u over ring %u (PB %u), sinks, "
                "the head %u / %u layers / %u heads, vocab %u\n", g.dev.name().c_str(), d.H, d.layers, d.NH, d.NKV, d.HD,
                d.win, d.ring, d.pb, d.Hl, d.llayers, d.lheads, d.V);
    const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
    clrun::Buffer brope(g.dev, rope);
    int bad = topk_check(g);
    for (uint32_t p : {1u, 15u, 16u, 17u, 31u, 40u, 123u}) bad += attn_check(g, p, 7);
    bad += attn_check(g, 20, 3);
    for (const Arm& a : arms) bad += run_arm(g, fx, ck, a, rope, brope);
    // the int4 g64 drafter through gemv.cl LAYOUT 1 (the loader's form), when gemv.cl builds on this device
    const Arm a4 = make_arm("int4 g64 + int8 head", fx, ck, llfix::Form::Int4, llfix::HeadForm::Int8);
    bool built = true;
    try {
      (void)g.lin(a4.layer[0][0], std::vector<uint16_t>(size_t(M) * d.H, 0), M);
    } catch (const std::exception& e) {
      built = false;
      const std::string why = e.what();
      std::printf("lilicorr_run: int4 arm NOT RUN - gemv.cl does not build on this device (%s); the twin's gemv_int4 "
                  "(lilicorr_ref_test) and the box cover it\n", why.substr(0, why.find('\n')).c_str());
    }
    if (built) bad += run_arm(g, fx, ck, a4, rope, brope);
    std::filesystem::remove_all(dir);
    if (bad) {
      std::printf("DISAGREES with the host twin (indicative: read the kernels before the box)\n");
      return 1;
    }
    std::printf("agrees with the host twin (indicative only; the B70 is the real test)\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lilicorr_run: %s\n", e.what());
    return 1;
  }
}
