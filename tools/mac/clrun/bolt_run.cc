// bolt_run - spec 25c's portable Nemotron 3.5 Bolt decode kernels on the Mac's OpenCL GPU against tests/kernels/bolt_ref.h
// and 25a's reference's own outputs (tests/kernels/bolt_fixture.h) (INDICATIVE ONLY: clrun.h says what this can and cannot
// show).
//
//   bolt_run      no arguments; Bolt's real shapes (hidden 4096, Mamba 128 heads x 64 / 8 groups / state 128, conv 4, the
//                 1024 latent, the 5376 shared expert, top-9 of 240 / 496, I 1536, NoPE GQA 16 at 128)
//
// What runs, with the CMake block's defines (but nb_moe_down's DN_KS: 1 here, the Mac's 256-lane work-group cap):
//   src/kernels/bolt/nb_prep.cl    nb_res_norm _NF on the fixture's rows, _S1 and _S1_T1 folds (the MoE block's two
//                                  sources), the arm router _R against nb_router, nb_sh_act, nb_attn_prep, nb_embed_gather
//   src/kernels/bolt/nb_mamba.cl   nb_conv over the fixture's 18 positions from a garbage ring (M = 1 steps, M = 4 launches);
//                                  nb_ssm fp32 / bf16 state at M = 1 and M = 3 rows, the arm conv `_C` against nb_conv +
//                                  nb_ssm, the SPEC_SLOTS form (4 slots) against the in-place kernel; nb_gnorm
//   src/kernels/bolt/nb_moe.cl     nb_router (E 496 and 240), nb_route on torch's logits and the crafted rows, nb_moe_up /
//                                  _R / nb_moe_down over the fixture's int4 experts at E 240, I 1536
//   src/kernels/bolt/mg_attn_eager.cl   spec 23c's eager attention at Bolt's full NoPE layer with a ones gate (Decision 15)
// mg_attn.cl's flash decode is not run: shuffles, sub-group reductions and 2D block reads.
//
// Bars: every kernel against the twin BITWISE (the twin's transcendentals are the kernels' own: Sleef's exp, fdlibm's
// log1p); against the fixture the twin's own measured distances (bolt_ref_test). Apple's compiler is not told
// -cl-fp32-correctly-rounded-divide-sqrt: a value through a division or a sqrt may move by an ulp - reported per check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/repack.h"
#include "kernels/bolt_fixture.h"
#include "kernels/bolt_ref.h"

namespace {

namespace br = boltref;
namespace fx = bolt_fixture;
using br::f32;
using br::rf;
using br::rne;

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
  for (uint32_t i = 0; i < n; ++i) v[i] = br::fixture_val(coarse, t, base + i, seed, step, amp);
  return v;
}
std::vector<uint16_t> bf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = rne(v[i]);
  return o;
}
std::vector<uint32_t> ctrl_at(uint32_t pos, uint32_t n_act) {
  std::vector<uint32_t> c(32, 0);
  c[0] = pos;
  c[1] = n_act;
  return c;
}

int failures = 0;
// "exact" / "agrees (worst, bar)" / the first differing index. T: uint16_t (bf16, ulps) or uint32_t (bits, exact only).
template <class T>
void report(const std::string& what, const std::vector<T>& got, const std::vector<T>& want, int bar) {
  int worst = 0;
  size_t first = got.size() == want.size() ? size_t(-1) : 0;
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    int u;
    if constexpr (sizeof(T) == 2) {
      u = ulps(uint16_t(got[i]), uint16_t(want[i]));
    } else {
      u = got[i] == want[i] ? 0 : 1000;
    }
    worst = std::max(worst, u);
    if (u > bar && first == size_t(-1)) first = i;
  }
  if (first == size_t(-1)) {
    std::printf("bolt_run: %-62s %zu values, %s\n", what.c_str(), got.size(),
                worst == 0 ? "exact" : ("agrees (worst " + std::to_string(worst) + " ulp, bar " + std::to_string(bar) + ")").c_str());
  } else {
    std::printf("bolt_run: %-62s DISAGREES at index %zu (worst %d, bar %d)\n", what.c_str(), first, worst, bar);
    ++failures;
  }
}
// Route rows (48 words each): the ids BITWISE; the fp32 words (w, the sel of ranks 0..9, Σ, the smallest sel) within 2
// fp32 ulps - sigmoid's 1 / (exp + 1) and w's s / (Σ + 1e-20) are divisions Apple's compiler is not told to round
// correctly (the card's build is: -cl-fp32-correctly-rounded-divide-sqrt; bolt_kernels_test holds them bitwise there).
void report_route(const std::string& what, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want) {
  size_t bad = 0, fdiff = 0;
  int64_t fmax = 0;
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const size_t k = i % br::kWords;
    if (k < br::kIds + br::kTopK) {
      if (got[i] != want[i]) {
        if (bad < 3) std::printf("bolt_run:   row %zu id slot %zu: %u != the twin's %u\n", i / br::kWords, k, got[i], want[i]);
        ++bad;
      }
      continue;
    }
    const int64_t u = std::llabs(int64_t(int32_t(got[i])) - int64_t(int32_t(want[i])));
    fmax = std::max(fmax, u);
    fdiff += u != 0;
    bad += u > 2;
  }
  if (got.size() != want.size() || bad) {
    std::printf("bolt_run: %-62s DISAGREES (%zu words)\n", what.c_str(), bad);
    ++failures;
  } else {
    std::printf("bolt_run: %-62s ids exact; %zu fp32 words off (max %lld ulp: the divisions)\n", what.c_str(), fdiff,
                (long long)fmax);
  }
}
std::vector<uint32_t> bits(const std::vector<float>& v) {
  std::vector<uint32_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = br::as_u32(v[i]);
  return o;
}

constexpr uint32_t H = br::kHidden, NIN = br::kInProjN, XBC = br::kXbc, INNER = br::kInner;
const std::vector<std::string> kNorm = {"K=4096", "NORM_WGS=8", "NORM_EPS=1e-5f"};

void prep_checks(clrun::Device& dev) {
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  clrun::Buffer wb(dev, w);
  {   // _NF on the fixture's rows 0-3 (vs torch's: coarse rows bitwise, the fine one 1 ulp at most - Σx²'s order)
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", with(kNorm, {"M=1", "NORM_S=0", "NORM_S2=0"}));
    const std::vector<uint16_t> want = br::hex16(fx::kNormX);
    const std::vector<std::vector<float>> rows = {vals(true, 0, H, 10), vals(false, 0, H, 11, 0.125f, 4.0f),
                                                  vals(true, 0, H, 12, 128.0f), vals(true, 0, H, 13, 0x1p-20f)};
    for (uint32_t r = 0; r < 4; ++r) {
      const std::vector<uint16_t> h = bf(rows[r]);
      clrun::Buffer hb(dev, h), sb(dev, 16), ob(dev, H * 2), xb(dev, H * 2);
      p.run("nb_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, sb, wb, ob, xb);
      const br::NormOut tw = br::res_norm(h.data(), nullptr, 0, nullptr, 0, 1, w.data());
      report("nb_res_norm _NF row " + std::to_string(r) + " vs the twin", xb.read<uint16_t>(), tw.x, 0);
      report("nb_res_norm _NF row " + std::to_string(r) + " vs torch", xb.read<uint16_t>(),
             std::vector<uint16_t>(want.begin() + size_t(r) * H, want.begin() + size_t(r + 1) * H), r == 1 ? 1 : 0);
    }
  }
  {   // _S1_T1 (the MoE block's two sources) on the fixture's rows 6-7 (vs the twin and torch); _S1 on a random row
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", with(kNorm, {"M=1", "NORM_S=1", "NORM_S2=2"}));
    const std::vector<uint16_t> fh = br::hex16(fx::kNormH), fxx = br::hex16(fx::kNormX);
    for (uint32_t r = 6; r < 8; ++r) {
      const std::vector<uint16_t> h = bf(vals(false, 0, H, 14 + (r - 4), 0.125f, 2.0f));
      const std::vector<float> s1 = vals(true, 2, H, 20 + r);
      std::vector<float> s2 = vals(true, 3, H, 20 + r);
      const std::vector<float> b1 = vals(true, 4, H, 20 + r);
      s2.insert(s2.end(), b1.begin(), b1.end());
      clrun::Buffer hb(dev, h), sb(dev, s1), s2b(dev, s2), ob(dev, H * 2), xb(dev, H * 2);
      p.run("nb_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, s2b, wb, ob, xb);
      const br::NormOut tw = br::res_norm(h.data(), s1.data(), 1, s2.data(), 2, 1, w.data());
      const std::string tag = "nb_res_norm two sources (S 1 + S 2) row " + std::to_string(r);
      report(tag + ": h_out vs torch", ob.read<uint16_t>(),
             std::vector<uint16_t>(fh.begin() + size_t(r - 4) * H, fh.begin() + size_t(r - 3) * H), 0);
      report(tag + ": x vs the twin", xb.read<uint16_t>(), tw.x, 0);
      report(tag + ": x vs torch", xb.read<uint16_t>(),
             std::vector<uint16_t>(fxx.begin() + size_t(r) * H, fxx.begin() + size_t(r + 1) * H), 1);
    }
  }
  {   // the arm router: _S1_R's logits = nb_router's on the same normalised row; its x = _S1's
    const uint32_t E = 496;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> sl(H), gwf(size_t(E) * H);
    for (float& v : sl) v = d(rng);
    for (float& v : gwf) v = d(rng) * 0.05f;
    const std::vector<uint16_t> h = bf(vals(false, 0, H, 77, 0.125f, 2.0f)), gw = bf(gwf);
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
    clrun::Program pr(dev, "src/kernels/bolt/nb_prep.cl", with(kNorm, {"M=1", "NORM_S=1", "NORM_S2=0", "ROUTER_FUSED=1"}));
    clrun::Program pn(dev, "src/kernels/bolt/nb_prep.cl", with(kNorm, {"M=1", "NORM_S=1", "NORM_S2=0"}));
    clrun::Program pt(dev, "src/kernels/bolt/nb_moe.cl", {"M=1", "ROUTER_K=4096"});
    clrun::Buffer hb(dev, h), sb(dev, sl), tb(dev, tiles), o1(dev, H * 2), x1(dev, H * 2), o2(dev, H * 2), x2(dev, H * 2),
        l1(dev, E * 4), l2(dev, E * 4);
    pr.run("nb_res_norm", {(8 + E / 16) * 256, 1}, {256, 1}, hb, sb, sb, wb, o1, x1, tb, l1, E);
    pn.run("nb_res_norm", {8 * 256, 1}, {256, 1}, hb, sb, sb, wb, o2, x2);
    pt.run("nb_router", {(E / 16) * 128, 1}, {128, 1}, x2, tb, l2, E);
    report("nb_res_norm _S1_R: x = _S1's", x1.read<uint16_t>(), x2.read<uint16_t>(), 0);
    report("nb_res_norm _S1_R: h_out = _S1's", o1.read<uint16_t>(), o2.read<uint16_t>(), 0);
    report("nb_res_norm _S1_R: the logits = nb_router's (E 496)", bits(l1.read<float>()), bits(l2.read<float>()), 0);
    const std::vector<uint16_t> xr = x2.read<uint16_t>();
    report("nb_router vs the twin (E 496)", bits(l2.read<float>()), bits(br::router(xr.data(), tiles.data(), E)), 0);
  }
  {   // nb_sh_act at S 2, random slices
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-3.f, 3.f);
    std::vector<float> sl(size_t(2) * br::kFc1N);
    for (float& v : sl) v = d(rng);
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", {"M=1", "ACT_S=2"});
    clrun::Buffer sb(dev, sl), lb(dev, br::kLatent * 2), shb(dev, br::kShared * 2);
    p.run("nb_sh_act", {25 * 256, 1}, {256, 1}, sb, lb, shb);
    std::vector<uint16_t> lat(br::kLatent), sh(br::kShared);
    br::sh_act(sl.data(), 2, 1, lat.data(), sh.data());
    report("nb_sh_act: the latent (S 2)", lb.read<uint16_t>(), lat, 0);
    report("nb_sh_act: relu2 of the shared up (S 2)", shb.read<uint16_t>(), sh, 0);
  }
  {   // nb_attn_prep at positions 0 and 262143 (S 2)
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> sl(size_t(2) * br::kQkvN);
    for (float& v : sl) v = d(rng);
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", with(kCtrlDefs, {"M=1", "QKV_S=2"}));
    clrun::Buffer sb(dev, sl);
    bool ok = true;
    for (uint32_t pos : {0u, 262143u}) {
      const uint32_t rows = pos + 1;
      clrun::Buffer cb(dev, ctrl_at(pos, 1)), qb(dev, 4096 * 4), kb(dev, size_t(rows) * 256 * 2), vb(dev, size_t(rows) * 256 * 2);
      p.run("nb_attn_prep", {18 * 256, 1}, {256, 1}, cb, sb, qb, kb, vb);
      const br::Prep want = br::attn_prep(sl.data(), 2, 1, 0);
      const std::vector<uint16_t> k = kb.read<uint16_t>(), v = vb.read<uint16_t>();
      ok = ok && qb.read<float>() == want.q && std::equal(want.k.begin(), want.k.end(), k.begin() + size_t(pos) * 256) &&
           std::equal(want.v.begin(), want.v.end(), v.begin() + size_t(pos) * 256);
    }
    std::printf("bolt_run: %-62s %s\n", "nb_attn_prep at 0 and 262143 (S 2; [pos][2][128] rows)", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
  {   // the gather: rows 0, 7, 131071 out of a random tiled table
    const uint32_t V = br::kVocab;
    std::vector<uint16_t> tiles(size_t(H) * V);
    std::mt19937 rng(13);
    for (uint16_t& t : tiles) t = uint16_t(rng());
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", with(kCtrlDefs, {"M=1", "K=4096", "EMBED_V=131072"}));
    clrun::Buffer tb(dev, tiles);
    bool ok = true;
    for (uint32_t id : {0u, 7u, 131071u}) {
      std::vector<uint32_t> ctrl = ctrl_at(0, 1);
      ctrl[2] = id;
      clrun::Buffer cb(dev, ctrl), hb(dev, H * 2);
      p.run("nb_embed_gather", {256, 1}, {256, 1}, cb, tb, hb);
      ok = ok && hb.read<uint16_t>() == br::embed_row(tiles.data(), id);
    }
    std::printf("bolt_run: %-62s %s\n", "nb_embed_gather (rows 0, 7, 131071)", ok ? "exact" : "DISAGREES");
    if (!ok) ++failures;
  }
}

// --- the Mamba mixer ---------------------------------------------------------------------------------------------
struct MambaIn {
  std::vector<float> cw, cb, A, D, dt_bias, S0;
  std::vector<float> slices;   // [T][18560]: z, xBC, dt random
};
MambaIn mamba_inputs(uint32_t T) {
  MambaIn in;
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> d(-2.f, 2.f);
  in.cw = vals(false, 21, XBC * 4, 5, 0.125f, 0.5f);
  in.cb = vals(false, 22, XBC, 5, 0.125f, 0.5f);
  in.dt_bias = vals(false, 32, br::kHeads, 6, 0.125f, 4.0f);
  in.dt_bias[0] = -10.5f;
  const std::vector<float> A_log = vals(false, 33, br::kHeads, 6, 0.125f, 4.0f);
  in.A.resize(br::kHeads);
  for (uint32_t h = 0; h < br::kHeads; ++h) in.A[h] = -common::exp_torch(A_log[h]);
  in.D = vals(false, 34, br::kHeads, 6, 0.125f, 2.0f);
  in.S0.resize(size_t(br::kHeads) * br::kHeadDim * br::kState);
  for (size_t i = 0; i < in.S0.size(); ++i) in.S0[i] = br::fixture_f32(35, uint32_t(i), 6);
  in.slices.resize(size_t(T) * NIN);
  for (float& v : in.slices) v = rf(d(rng));
  return in;
}

void mamba_checks(clrun::Device& dev) {
  {   // nb_conv over the fixture's 18 positions, M = 1 steps from a garbage ring, vs torch's conv_seq and the twin
    const uint32_t T = fx::kConvT, SUB = fx::kConvStride, CS = (XBC + SUB - 1) / SUB;
    clrun::Program p(dev, "src/kernels/bolt/nb_mamba.cl", with(kCtrlDefs, {"M=1", "CONV_S=1"}));
    const std::vector<uint16_t> X = bf(vals(false, 20, T * XBC, 5, 0.125f, 2.0f));
    const std::vector<float> w = vals(false, 21, XBC * 4, 5, 0.125f, 0.5f), b = vals(false, 22, XBC, 5, 0.125f, 0.5f);
    std::vector<uint16_t> ring(size_t(br::kRing) * XBC);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage: never read
    clrun::Buffer rb(dev, ring), wb(dev, w), bb(dev, b), xb(dev, XBC * 2);
    std::vector<uint16_t> got, twin, ring_t = ring;
    for (uint32_t pos = 0; pos < T; ++pos) {
      std::vector<float> s(NIN, 0.0f);
      for (uint32_t c = 0; c < XBC; ++c) s[br::kXbcOff + c] = f32(X[size_t(pos) * XBC + c]);
      clrun::Buffer cb(dev, ctrl_at(pos, 1)), sb(dev, s);
      p.run("nb_conv", {XBC, 1}, {256, 1}, cb, sb, wb, bb, rb, xb);
      const std::vector<uint16_t> xc = xb.read<uint16_t>();
      for (uint32_t j = 0; j < CS; ++j) got.push_back(xc[j * SUB]);
      const std::vector<uint16_t> t = br::conv_step(s.data(), 1, 1, ring_t.data(), w.data(), b.data(), pos);
      twin.insert(twin.end(), t.begin(), t.end());
    }
    std::vector<uint16_t> twin_sub;
    for (uint32_t q = 0; q < T; ++q)
      for (uint32_t j = 0; j < CS; ++j) twin_sub.push_back(twin[size_t(q) * XBC + j * SUB]);
    report("nb_conv: 18 positions vs the twin (the SiLU's division: bar 1)", got, twin_sub, 1);
    report("nb_conv: 18 positions vs torch's conv_seq (ring wraps at 16)", got, br::hex16(fx::kConvXc), 1);
    report("nb_conv: the ring after 18 positions (raw xBC) vs the twin", rb.read<uint16_t>(), ring_t, 0);
  }
  const uint32_t T = 3;
  const MambaIn in = mamba_inputs(T);
  // xc of the 3 rows as the twin's conv writes it (from a zero ring at pos 0), then the SSM over them.
  std::vector<uint16_t> ring0(size_t(br::kRing) * XBC, 0);
  std::vector<uint16_t> ring_t = ring0;
  const std::vector<uint16_t> xc = br::conv_step(in.slices.data(), 1, T, ring_t.data(), in.cw.data(), in.cb.data(), 0);
  for (int b16 = 0; b16 < 2; ++b16) {
    const std::string tag = b16 ? "bf16 state" : "fp32 state";
    std::vector<float> st_t = in.S0;
    std::vector<uint16_t> st16_t(st_t.size());
    for (size_t i = 0; i < st_t.size(); ++i) st16_t[i] = rne(st_t[i]);
    const std::vector<uint16_t> y_t = br::ssm_step(in.slices.data(), 1, T, xc.data(), in.A.data(), in.D.data(),
                                                   in.dt_bias.data(), st_t.data(), b16 ? st16_t.data() : nullptr);
    for (int conv = 0; conv < 2; ++conv) {
      for (uint32_t M : {1u, 3u}) {
        clrun::Program p(dev, "src/kernels/bolt/nb_mamba.cl",
                         with(kCtrlDefs, {"M=" + std::to_string(M), "SSM_S=1", "SSM_B16=" + std::to_string(b16),
                                          "SSM_CONV=" + std::to_string(conv)}));
        std::vector<uint16_t> st16(in.S0.size());
        for (size_t i = 0; i < st16.size(); ++i) st16[i] = rne(in.S0[i]);
        clrun::Buffer stb = b16 ? clrun::Buffer(dev, st16) : clrun::Buffer(dev, in.S0);
        clrun::Buffer Ab(dev, in.A), Db(dev, in.D), bb(dev, in.dt_bias), cwb(dev, in.cw), cbb(dev, in.cb), rb(dev, ring0),
            yb(dev, size_t(M) * INNER * 2);
        std::vector<uint16_t> y_all;
        for (uint32_t m0 = 0; m0 < T; m0 += M) {
          clrun::Buffer cb(dev, ctrl_at(m0, M)),
              sb(dev, std::vector<float>(in.slices.begin() + size_t(m0) * NIN, in.slices.begin() + size_t(m0 + M) * NIN));
          if (conv) {
            p.run("nb_ssm", {512 * 256, 1}, {256, 1}, cb, sb, cwb, cbb, rb, Ab, Db, bb, stb, yb);
          } else {
            clrun::Buffer xb(dev, std::vector<uint16_t>(xc.begin() + size_t(m0) * XBC, xc.begin() + size_t(m0 + M) * XBC));
            p.run("nb_ssm", {512 * 256, 1}, {256, 1}, cb, sb, xb, Ab, Db, bb, stb, yb);
          }
          const std::vector<uint16_t> y = yb.read<uint16_t>();
          y_all.insert(y_all.end(), y.begin(), y.end());
        }
        const std::string what = "nb_ssm" + std::string(conv ? "_C" : "") + " " + tag + ", M " + std::to_string(M);
        report(what + ": y (3 rows)", y_all, y_t, 0);
        if (b16)
          report(what + ": the state", stb.read<uint16_t>(), st16_t, 0);
        else
          report(what + ": the state (bits)", bits(stb.read<float>()), bits(st_t), 0);
        if (conv) report(what + ": the ring (raw xBC)", rb.read<uint16_t>(), ring_t, 0);
      }
    }
  }
  {   // SPEC_SLOTS 4 (the slot form): M = 3 rows from live slot 2 - slot (2 + m) % 4 holds the state after row m
    const uint32_t SL = 4, live = 2;
    const size_t n = in.S0.size();
    clrun::Program p(dev, "src/kernels/bolt/nb_mamba.cl",
                     with(kCtrlDefs, {"M=3", "SSM_S=1", "SSM_B16=0", "SSM_CONV=0", "SPEC_SLOTS=4", "CTRL_LIVE=19",
                                      "SPEC_SLOT_STRIDE=" + std::to_string(n)}));
    std::vector<float> slots(n * (SL - 1), 0.0f);
    std::copy(in.S0.begin(), in.S0.end(), slots.begin() + (live - 1) * n);   // slot 2 = spec + 1 x stride
    std::vector<uint32_t> ctrl = ctrl_at(0, 3);
    ctrl[19] = live;
    clrun::Buffer cb(dev, ctrl), sb(dev, in.slices), xb(dev, xc), Ab(dev, in.A), Db(dev, in.D), bb(dev, in.dt_bias),
        s0b(dev, std::vector<float>(n, 0.0f)), spb(dev, slots), yb(dev, size_t(3) * INNER * 2);
    p.run("nb_ssm", {512 * 256, 1}, {256, 1}, cb, sb, xb, Ab, Db, bb, s0b, yb, spb);
    std::vector<float> st = in.S0;
    std::vector<uint16_t> y_t;
    std::vector<std::vector<float>> after;
    for (uint32_t m = 0; m < 3; ++m) {
      std::vector<float> s1(in.slices.begin() + size_t(m) * NIN, in.slices.begin() + size_t(m + 1) * NIN);
      const std::vector<uint16_t> y = br::ssm_step(s1.data(), 1, 1, xc.data() + size_t(m) * XBC, in.A.data(), in.D.data(),
                                                   in.dt_bias.data(), st.data());
      y_t.insert(y_t.end(), y.begin(), y.end());
      after.push_back(st);
    }
    report("nb_ssm SPEC_SLOTS 4, live 2, M 3: y", yb.read<uint16_t>(), y_t, 0);
    const std::vector<float> s0 = s0b.read<float>(), sp = spb.read<float>();
    for (uint32_t m = 0; m < 3; ++m) {
      const uint32_t slot = (live + m) % SL;
      const std::vector<float> got = slot == 0 ? s0 : std::vector<float>(sp.begin() + (slot - 1) * n, sp.begin() + slot * n);
      report("nb_ssm SPEC_SLOTS: slot " + std::to_string(slot) + " = the state after row " + std::to_string(m), bits(got),
             bits(after[m]), 0);
    }
  }
  {   // nb_gnorm on the fixture's 4 rows (vs torch and the twin)
    std::vector<float> y = vals(false, 40, 4 * INNER, 7, 0.125f, 2.0f);
    y[size_t(1) * INNER + 2 * 1024 + 5] = 60.0f;
    const std::vector<uint16_t> yb_ = bf(y), z = bf(vals(false, 41, 4 * INNER, 7, 0.125f, 4.0f));
    std::vector<float> w = vals(false, 42, INNER, 7, 0.125f, 0.5f);
    for (float& x : w) x = rf(1.0f + x);
    std::vector<float> s(size_t(4) * NIN, 0.0f);
    for (uint32_t m = 0; m < 4; ++m)
      for (uint32_t c = 0; c < INNER; ++c) s[size_t(m) * NIN + c] = f32(z[size_t(m) * INNER + c]);
    clrun::Program p(dev, "src/kernels/bolt/nb_mamba.cl", {"M=4", "GN_S=1", "GN_EPS=1e-5f"});
    clrun::Buffer sb(dev, s), ybuf(dev, yb_), wb(dev, w), ob(dev, size_t(4) * INNER * 2);
    p.run("nb_gnorm", {8 * 256, 4}, {256, 1}, sb, ybuf, wb, ob);
    report("nb_gnorm (4 rows, groups of 1024) vs the twin", ob.read<uint16_t>(), br::gnorm(s.data(), 1, 4, yb_.data(), w.data()), 1);
    report("nb_gnorm (4 rows) vs torch's rmsnorm_gated", ob.read<uint16_t>(), br::hex16(fx::kGnorm), 1);
  }
}

// --- the router, the route, the experts ---------------------------------------------------------------------------------
void moe_checks(clrun::Device& dev) {
  {   // nb_route on torch's fp32 logits (the fixture's 32 rows at E 496 and 240) and the crafted rows
    clrun::Program p(dev, "src/kernels/bolt/nb_moe.cl", {"M=1", "ROUTE_K=9"});
    struct C {
      uint32_t E, seed;
      const char* logits;
    };
    for (const C& c : {C{496, 8, fx::kRouteLogits496}, C{240, 9, fx::kRouteLogits240}}) {
      std::vector<float> bias(c.E);
      for (uint32_t e = 0; e < c.E; ++e) bias[e] = br::fixture_f32(52, e, c.seed) * 0.04f + 4.5f;
      const std::vector<uint32_t> lg = br::hex32(c.logits);
      clrun::Buffer bb(dev, bias);
      std::vector<uint32_t> got, want;
      for (uint32_t r = 0; r < 32; ++r) {
        std::vector<float> l(c.E);
        for (uint32_t e = 0; e < c.E; ++e) l[e] = br::as_f32(lg[size_t(r) * c.E + e]);
        clrun::Buffer lb(dev, l), ro(dev, br::kWords * 4);
        p.run("nb_route", {256, 1}, {256, 1}, lb, bb, ro, c.E);
        const std::vector<uint32_t> row = ro.read<uint32_t>();
        const br::Route rt = br::route(l.data(), bias.data(), c.E);
        got.insert(got.end(), row.begin(), row.end());
        want.insert(want.end(), rt.row, rt.row + br::kWords);
      }
      report_route("nb_route E " + std::to_string(c.E) + " (32 fixture rows) vs the twin", got, want);
    }
    const std::vector<uint32_t> cl = br::hex32(fx::kCraftLogits), cbias = br::hex32(fx::kCraftBias);
    std::vector<uint32_t> got, want;
    for (uint32_t r = 0; r < fx::kCraftN; ++r) {
      const uint32_t E = fx::kCraftE[r];
      std::vector<float> l(E), b(E);
      for (uint32_t e = 0; e < E; ++e) {
        l[e] = br::as_f32(cl[size_t(r) * 496 + e]);
        b[e] = br::as_f32(cbias[size_t(r) * 496 + e]);
      }
      clrun::Buffer lb(dev, l), bb(dev, b), ro(dev, br::kWords * 4);
      p.run("nb_route", {256, 1}, {256, 1}, lb, bb, ro, E);
      const std::vector<uint32_t> row = ro.read<uint32_t>();
      const br::Route rt = br::route(l.data(), b.data(), E);
      got.insert(got.end(), row.begin(), row.end());
      want.insert(want.end(), rt.row, rt.row + br::kWords);
    }
    report_route("nb_route crafted (ties at 496 / 240, all-equal, bias, 495)", got, want);
  }
  {   // the experts at E 240, I 1536: up / _R / down against the twin, the fixture's ids and weights (row 0)
    const uint32_t E = fx::kMoeE, I = fx::kMoeI, L = br::kLatent;
    std::set<uint32_t> used(std::begin(fx::kMoeIds[0]), std::end(fx::kMoeIds[0]));
    std::vector<uint32_t> blocks(size_t(E) * br::expert_block_words(I), 0u);
    for (uint32_t e : used)
      br::expert_block(br::gptq_of(400 + 2 * e, L, I, 7), br::gptq_of(401 + 2 * e, I, L, 7),
                       blocks.data() + size_t(e) * br::expert_block_words(I));
    const std::vector<uint16_t> lat_all = br::hex16(fx::kMoeLat), r_all = br::hex16(fx::kMoeR);
    const std::vector<uint16_t> lat(lat_all.begin(), lat_all.begin() + L);
    const std::vector<uint32_t> wbits = br::hex32(fx::kMoeW);
    br::Route rt;
    std::vector<uint32_t> row(br::kWords, 0u);
    for (uint32_t k = 0; k < 9; ++k) {
      rt.ids[k] = fx::kMoeIds[0][k];
      rt.w[k] = br::as_f32(wbits[k]);
      row[br::kIds + k] = rt.ids[k];
      row[br::kW + k] = wbits[k];
    }
    const std::vector<std::string> defs = {"M=1", "MOE_I=1536", "UP_KS=4", "DN_KS=1"};
    clrun::Program pu(dev, "src/kernels/bolt/nb_moe.cl", defs), pd(dev, "src/kernels/bolt/nb_moe.cl", defs);
    clrun::Buffer rb(dev, row), lb(dev, lat), wb(dev, blocks), ab(dev, size_t(9) * I * 2), out(dev, L * 2);
    pu.run("nb_moe_up", {size_t(9 * (I / 64)) * 256, 1}, {256, 1}, rb, lb, wb, ab, E);
    pd.run("nb_moe_down", {64 * 144, 1}, {144, 1}, rb, ab, wb, out, E);
    const br::MoeOut o = br::moe(rt, lat.data(), blocks.data(), I, 4, 1);
    report("nb_moe_up (9 slots, relu2, I 1536) vs the twin", ab.read<uint16_t>(), o.a, 0);
    report("nb_moe_down (the ascending-id combine) vs the twin", out.read<uint16_t>(), o.r, 0);
    report("nb_moe_down vs torch's combine (row 0)", out.read<uint16_t>(),
           std::vector<uint16_t>(r_all.begin(), r_all.begin() + L), 1);
    // The arm route: _R recomputes the route from logits that rank row 0's ids top (planted), WG 0 writes the row.
    std::vector<float> logits(E, -6.0f), bias(E, 0.0f);
    for (uint32_t k = 0; k < 9; ++k) logits[rt.ids[k]] = 1.0f + 0.25f * float(k);
    const br::Route rr = br::route(logits.data(), bias.data(), E);
    clrun::Program pr(dev, "src/kernels/bolt/nb_moe.cl", {"M=1", "MOE_I=1536", "UP_KS=4", "DN_KS=1", "ROUTE_FUSED=1"});
    clrun::Buffer lgb(dev, logits), bsb(dev, bias), rrb(dev, br::kWords * 4), a2(dev, size_t(9) * I * 2);
    pr.run("nb_moe_up", {size_t(9 * (I / 64)) * 256, 1}, {256, 1}, lgb, bsb, rrb, lb, wb, a2, E);
    {   // the arm's row = nb_route's on the same device, bitwise (the same functions; Review Focus 5)
      clrun::Program pt(dev, "src/kernels/bolt/nb_moe.cl", {"M=1", "ROUTE_K=9"});
      clrun::Buffer ro(dev, br::kWords * 4);
      pt.run("nb_route", {256, 1}, {256, 1}, lgb, bsb, ro, E);
      report("nb_moe_up _R: the route row = nb_route's (device, bits)", rrb.read<uint32_t>(), ro.read<uint32_t>(), 0);
    }
    report_route("nb_moe_up _R: the route row vs the twin's", rrb.read<uint32_t>(), std::vector<uint32_t>(rr.row, rr.row + br::kWords));
    report("nb_moe_up _R: a = the twin's on that route", a2.read<uint16_t>(), br::moe_up(rr.ids, lat.data(), blocks.data(), I, 4), 0);
  }
}

// --- spec 23c's eager attention at Bolt's full NoPE layer (Decision 15) -------------------------------------------------------
void attention_checks(clrun::Device& dev) {
  const uint32_t HD = br::kHd, KV = br::kKvHeads, QH = br::kQHeads;
  clrun::Program p(dev, "src/kernels/bolt/mg_attn_eager.cl",
                   with(kCtrlDefs, {"M=1", "TGT=64", "Q_HEADS=32", "KV_HEADS=2", "WINDOW=0", "RING=0"}));
  for (uint32_t ci = 0; ci < 3; ++ci) {
    const uint32_t L = fx::kAttnL[ci];
    const std::vector<uint16_t> q = bf(vals(true, 70, QH * HD, ci, 0.5f));
    std::vector<float> qf(q.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = f32(q[i]);
    const std::vector<uint16_t> kk = bf(vals(true, 71, L * KV * HD, 9)), vv = bf(vals(true, 72, L * KV * HD, 9));
    const std::vector<uint16_t> ones(size_t(QH) * HD, uint16_t(0x3F80));
    clrun::Buffer cb(dev, ctrl_at(L - 1, 1)), qb(dev, qf), kb(dev, kk), vb(dev, vv), gb(dev, ones),
        sb(dev, size_t(QH) * L * 4), ob(dev, size_t(QH) * HD * 2);
    p.run("mg_attn_eager", {QH * 128, 1}, {128, 1}, cb, qb, kb, vb, gb, sb, ob, L);
    std::vector<uint16_t> tw(size_t(QH) * HD);
    br::attention_eager(q.data(), kk.data(), vv.data(), L - 1, tw.data());
    report("mg_attn_eager (a ones gate) at " + std::to_string(L) + " keys vs the twin", ob.read<uint16_t>(), tw, 0);
    report("mg_attn_eager at " + std::to_string(L) + " keys vs torch (R's eager)", ob.read<uint16_t>(), br::hex16(fx::kAttnO[ci]), 1);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("bolt_run: device %s\n", dev.name().c_str());
    prep_checks(dev);
    mamba_checks(dev);
    moe_checks(dev);
    attention_checks(dev);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bolt_run: %s\n", e.what());
    return 1;
  }
  if (failures) {
    std::printf("bolt_run: %d check(s) DISAGREE\n", failures);
    return 1;
  }
  std::printf("bolt_run: every check agrees\n");
  return 0;
}
