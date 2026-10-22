// bolt_pf_run - spec 25d's portable Nemotron 3.5 Bolt PREFILL kernels on the Mac's OpenCL GPU against tests/kernels/
// bolt_pf_ref.h (INDICATIVE ONLY: clrun.h says what this can and cannot show). No arguments; Bolt's real shapes.
//
// What runs, with the CMake block's defines:
//   src/kernels/bolt/nb_pf_mamba.cl   nb_pf_seed + nb_pf_conv (from a garbage ring; the next three ring columns), nb_pf_dt,
//                                     nb_pf_cb (the plain build), nb_pf_ssd (fp32 and the bf16 state), nb_pf_gnorm - at (pos,
//                                     C) = (0, 181) and (2085, 300), in BOTH tilings (W16_R16, W8_R4) ten times each
//                                     (Review Focus 4: every tiling finite and equal)
//   src/kernels/bolt/nb_pf_moe.cl     nb_pf_sort / nb_pf_gather at E 240 over 181 rows (and the tile bound at 2048 rows,
//                                     E 496), nb_pf_dequant_up / _dn (I 1536), nb_pf_combine
//   src/kernels/bolt/nb_pf_prep.cl    nb_pf_embed_gather
//   25c's nb_prep.cl / nb_moe.cl at M = 2048: the norm's two-source fold, the router, the route, the shared act, the
//                                     attention prep over 300 rows
//   src/kernels/muse/mg_pf_attn.cl    the EAGER build at Bolt's full NoPE layer: every row bitwise decode's eager row
// Not run here: nb_pf_cb's DPAS build and nb_pf_moe_gemm (DPAS, 2D block I/O), mg_pf_flash_attn's flash build.
//
// Bars: the twin is fed the device's own upstream outputs, so every kernel is compared on its own inputs - BITWISE where
// no division sits in the chain (the seed, C B^T, the scan, the sort, the gather, the dequant, the combine, the embed), within
// 1-2 ulp where one does (the conv's and the gate's SiLU, softplus' log1p, the norm's rstd, the route's sigmoid): Apple's
// compiler is not told -cl-fp32-correctly-rounded-divide-sqrt (the card's build is; bolt_pf_kernels_test is bitwise there).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "kernels/bolt_pf_ref.h"

namespace {

namespace br = boltref;
namespace bp = boltpf;
using br::f32;
using br::rne;

const std::vector<std::string> kCtrlDefs = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
constexpr uint32_t X = bp::kX, LD = bp::kInLd, IN = bp::kInner, NH = bp::kHeads, H = br::kHidden;

int failures = 0;
int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
int64_t ulps32(float a, float b) {
  int32_t x, y;
  std::memcpy(&x, &a, 4);
  std::memcpy(&y, &b, 4);
  auto key = [](int32_t v) { return v < 0 ? -int64_t(v & 0x7FFFFFFF) : int64_t(v); };
  return std::llabs(key(x) - key(y));
}
void report16(const std::string& what, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want, int bar) {
  int worst = 0;
  size_t first = got.size() == want.size() ? size_t(-1) : 0, diff = 0;
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const int u = ulps16(got[i], want[i]);
    worst = std::max(worst, u);
    diff += u != 0;
    if (u > bar && first == size_t(-1)) first = i;
  }
  if (first == size_t(-1)) {
    std::printf("bolt_pf_run: %-66s %zu values, %s\n", what.c_str(), got.size(),
                worst == 0 ? "exact" : (std::to_string(diff) + " within " + std::to_string(worst) + " ulp (bar " +
                                        std::to_string(bar) + ")").c_str());
  } else {
    std::printf("bolt_pf_run: %-66s DISAGREES at index %zu (worst %d, bar %d)\n", what.c_str(), first, worst, bar);
    ++failures;
  }
}
void report32(const std::string& what, const std::vector<float>& got, const std::vector<float>& want, int64_t bar) {
  int64_t worst = 0;
  size_t diff = 0;
  bool bad = got.size() != want.size();
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const int64_t u = ulps32(got[i], want[i]);
    if (!std::isfinite(got[i])) bad = true;
    worst = std::max(worst, u);
    diff += u != 0;
  }
  bad = bad || worst > bar;
  std::printf("bolt_pf_run: %-66s %zu values, %s\n", what.c_str(), got.size(),
              bad ? "DISAGREES" : worst == 0 ? "exact" : (std::to_string(diff) + " within " + std::to_string(worst) +
                                                          " ulp (bar " + std::to_string(bar) + ")").c_str());
  if (bad) ++failures;
}
void report_eq(const std::string& what, bool ok) {
  std::printf("bolt_pf_run: %-66s %s\n", what.c_str(), ok ? "exact" : "DISAGREES");
  if (!ok) ++failures;
}
float fval(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float amp = 1.0f) {
  return br::fixture_val(coarse, t, i, seed, 0.125f, amp);
}

// --- the Mamba chain ------------------------------------------------------------------------------------------------
struct MambaIn {
  uint32_t rows;
  std::vector<float> P, cw, cb, A, D, dt_bias, gw, S0;
  std::vector<uint16_t> ring;
};
MambaIn mamba_inputs(uint32_t rows) {
  MambaIn m;
  m.rows = rows;
  m.P.assign(size_t(rows) * LD, 0.0f);
  for (uint32_t t = 0; t < rows; ++t) {
    float* r = m.P.data() + size_t(t) * LD;
    for (uint32_t i = 0; i < IN; ++i) r[i] = fval(false, 80, t * IN + i, 13, 4.0f);
    for (uint32_t i = 0; i < X; ++i) r[br::kXbcOff + i] = fval(false, 81, t * X + i, 13, 2.0f);
    for (uint32_t h = 0; h < NH; ++h) r[br::kDtOff + h] = fval(false, 82, t * NH + h, 13, 2.0f);
  }
  m.cw.resize(size_t(X) * 4);
  m.cb.resize(X);
  for (uint32_t i = 0; i < X * 4; ++i) m.cw[i] = fval(false, 83, i, 13, 0.5f);
  for (uint32_t i = 0; i < X; ++i) m.cb[i] = fval(false, 84, i, 13, 0.5f);
  m.A.resize(NH);
  m.D.resize(NH);
  m.dt_bias.resize(NH);
  for (uint32_t h = 0; h < NH; ++h) {
    m.dt_bias[h] = fval(false, 63, h, 12, 4.0f);
    m.A[h] = -br::exp_t(fval(false, 64, h, 12, 4.0f));
    m.D[h] = fval(false, 65, h, 12, 2.0f);
  }
  m.dt_bias[0] = -10.5f;
  m.dt_bias[6] = -11.6875f;
  m.A[3] = -br::exp_t(br::rf(4.41f));
  m.A[4] = -br::exp_t(br::rf(8.81f));
  m.gw.resize(IN);
  for (uint32_t i = 0; i < IN; ++i) m.gw[i] = br::rf(1.0f + fval(false, 85, i, 13, 0.5f));
  m.S0.resize(size_t(NH) * bp::kP * bp::kN);
  for (size_t i = 0; i < m.S0.size(); ++i) m.S0[i] = br::fixture_f32(86, uint32_t(i), 13, 0.5f);
  m.ring.resize(size_t(br::kRing) * X);
  for (size_t i = 0; i < m.ring.size(); ++i) m.ring[i] = rne(fval(false, 77, uint32_t(i), 5, 3.0f));
  return m;
}
struct MambaOut {
  std::vector<uint16_t> xc, ring, y, out, y16;
  std::vector<float> dt, cum, cb, state, state16w;
};
// One chunk through the device's mamba entries in the build `tag` (W, R): seed, conv, dt, cb, ssd (+ the bf16 build), gnorm.
MambaOut mamba_device(clrun::Device& dev, const MambaIn& m, uint32_t pos, uint32_t C, uint32_t W, uint32_t R) {
  const std::vector<std::string> defs = {"PF_SUB=64", "SSD_W=" + std::to_string(W), "CONV_ROWS=" + std::to_string(R)};
  clrun::Program p(dev, "src/kernels/bolt/nb_pf_mamba.cl", with(defs, {"SSM_B16=0"}));
  clrun::Program p16(dev, "src/kernels/bolt/nb_pf_mamba.cl", with(defs, {"SSM_B16=1"}));
  const uint32_t ns = bp::n_sub(pos, C);
  clrun::Buffer P(dev, std::vector<float>(m.P.begin(), m.P.begin() + size_t(C) * LD)), ring(dev, m.ring),
      seed(dev, size_t(3) * X * 2), cw(dev, m.cw), cb(dev, m.cb), xc(dev, size_t(C) * X * 2), A(dev, m.A), Dd(dev, m.D),
      bias(dev, m.dt_bias), dt(dev, size_t(C) * NH * 4), cum(dev, size_t(C) * NH * 4),
      cbo(dev, size_t(ns) * 8 * 64 * 64 * 4), st(dev, m.S0), y(dev, size_t(C) * IN * 2), gw(dev, m.gw),
      out(dev, size_t(C) * IN * 2), y16(dev, size_t(C) * IN * 2);
  std::vector<uint16_t> s16(m.S0.size());
  for (size_t i = 0; i < s16.size(); ++i) s16[i] = rne(m.S0[i]);
  clrun::Buffer st16(dev, s16);
  p.run("nb_pf_seed", {X}, {256}, ring, seed, pos);
  p.run("nb_pf_conv", {X, (C + R - 1) / R}, {256, 1}, P, seed, cw, cb, ring, xc, pos, C);
  p.run("nb_pf_dt", {NH, ns}, {NH, 1}, P, A, bias, dt, cum, pos, C);
  p.run("nb_pf_cb", {ns * 256, 8}, {256, 1}, xc, cbo, pos, C);
  p.run("nb_pf_ssd", {NH * (64 / W) * 256}, {256}, xc, dt, cum, cbo, Dd, st, y, pos, C);
  p16.run("nb_pf_ssd", {NH * (64 / W) * 256}, {256}, xc, dt, cum, cbo, Dd, st16, y16, pos, C);
  p.run("nb_pf_gnorm", {8 * 256, C}, {256, 1}, P, y, gw, out);
  MambaOut o;
  o.xc = xc.read<uint16_t>();
  o.ring = ring.read<uint16_t>();
  o.dt = dt.read<float>();
  o.cum = cum.read<float>();
  o.cb = cbo.read<float>();
  o.state = st.read<float>();
  o.y = y.read<uint16_t>();
  o.y16 = y16.read<uint16_t>();
  o.out = out.read<uint16_t>();
  const std::vector<uint16_t> sw = st16.read<uint16_t>();
  o.state16w.resize(sw.size());
  for (size_t i = 0; i < sw.size(); ++i) o.state16w[i] = f32(sw[i]);
  return o;
}
bool same_out(const MambaOut& a, const MambaOut& b) {
  return a.xc == b.xc && a.ring == b.ring && a.dt == b.dt && a.cum == b.cum && a.cb == b.cb && a.y == b.y &&
         a.y16 == b.y16 && a.out == b.out && std::memcmp(a.state.data(), b.state.data(), a.state.size() * 4) == 0 &&
         a.state16w == b.state16w;
}

void mamba_checks(clrun::Device& dev) {
  struct Case {
    uint32_t pos, C;
  };
  const MambaIn m = mamba_inputs(300);
  for (const Case k : {Case{0, 181}, Case{2085, 300}}) {
    const std::string at = " (pos " + std::to_string(k.pos) + ", C " + std::to_string(k.C) + ")";
    const MambaOut d = mamba_device(dev, m, k.pos, k.C, 16, 16);
    // the conv and the ring: the twin's from the same seed
    std::vector<uint16_t> ring = m.ring;
    const std::vector<uint16_t> sd = bp::seed(ring.data(), k.pos);
    const std::vector<uint16_t> xc = bp::conv_chunk(m.P.data(), k.C, sd.data(), m.cw.data(), m.cb.data(), ring.data(), k.pos);
    report16("nb_pf_seed + nb_pf_conv: xc" + at, d.xc, xc, 1);
    report16("nb_pf_conv: the ring (the next three columns written back)" + at, d.ring, ring, 1);
    // dt / cum
    const bp::DtCum dc = bp::dt_cum(m.P.data(), k.C, k.pos, m.A.data(), m.dt_bias.data());
    report32("nb_pf_dt: dt" + at, d.dt, dc.dt, 2);
    report32("nb_pf_dt: the sub-chunks' cumulative sums" + at, d.cum, dc.cum, 64);
    // C B^T, the scan and the gated norm on the device's own inputs
    report32("nb_pf_cb (plain): C B^T per (sub-chunk, group)" + at, d.cb, bp::cb(d.xc.data(), k.C, k.pos), 0);
    bp::DtCum ddc{d.dt, d.cum};
    std::vector<float> st = m.S0;
    std::vector<uint16_t> y(size_t(k.C) * IN, 0);
    bp::ssd(d.xc.data(), k.C, k.pos, ddc, d.cb.data(), m.D.data(), st.data(), y.data());
    report16("nb_pf_ssd: y (fp32 state)" + at, d.y, y, 0);
    report32("nb_pf_ssd: the state after the chunk" + at, d.state, st, 0);
    std::vector<uint16_t> s16(m.S0.size()), y16(size_t(k.C) * IN, 0);
    for (size_t i = 0; i < s16.size(); ++i) s16[i] = rne(m.S0[i]);
    std::vector<float> dummy;
    bp::ssd(d.xc.data(), k.C, k.pos, ddc, d.cb.data(), m.D.data(), dummy.data(), y16.data(), 0, NH, s16.data());
    report16("nb_pf_ssd _B16: y (the state read from bf16)" + at, d.y16, y16, 0);
    std::vector<float> s16w(s16.size());
    for (size_t i = 0; i < s16.size(); ++i) s16w[i] = f32(s16[i]);
    report32("nb_pf_ssd _B16: the state rounded once at the chunk's end" + at, d.state16w, s16w, 0);
    report16("nb_pf_gnorm" + at, d.out, bp::gnorm_chunk(m.P.data(), k.C, d.y.data(), m.gw.data()), 1);
    // Review Focus 4: both tilings, ten times each, equal
    bool same = true, finite = true;
    for (int rep = 0; rep < 10; ++rep)
      for (const auto& wr : {std::pair<uint32_t, uint32_t>{16, 16}, std::pair<uint32_t, uint32_t>{8, 4}}) {
        const MambaOut r = mamba_device(dev, m, k.pos, k.C, wr.first, wr.second);
        same = same && same_out(r, d);
        for (float v : r.state) finite = finite && std::isfinite(v);
        for (uint16_t v : r.y) finite = finite && std::isfinite(f32(v));
      }
    report_eq("Review Focus 4: W16_R16 and W8_R4, x10 each - finite and equal" + at, same && finite);
  }
}

// --- the grouped MoE's portable entries -------------------------------------------------------------------------------
std::vector<uint32_t> routes_of(uint32_t C, uint32_t E, uint32_t seed, uint32_t id_cap) {
  std::vector<uint32_t> rt(size_t(C) * br::kWords);
  std::vector<float> lg(E), bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = 4.5f + br::fixture_f32(93, e, seed) * 0.04f;
  for (uint32_t t = 0; t < C; ++t) {
    for (uint32_t e = 0; e < E; ++e) lg[e] = e < id_cap ? br::fixture_f32(94, t * E + e, seed, 4.0f) : -1e30f;
    const br::Route r = br::route(lg.data(), bias.data(), E);
    std::copy(r.row, r.row + br::kWords, rt.begin() + size_t(t) * br::kWords);
  }
  return rt;
}
struct SortBufs {
  clrun::Buffer hdr, tiles, row_tok, pair_row;
  SortBufs(clrun::Device& d, uint32_t C, uint32_t E)
      : hdr(d, size_t(bp::kHdrWords) * 4), tiles(d, size_t(2) * bp::tmax(2048, 512) * 4),
        row_tok(d, size_t(bp::tmax(2048, 512)) * 32 * 4), pair_row(d, size_t(C) * br::kTopK * 4) {
    (void)E;
  }
};
bool sort_equal(const SortBufs& b, const bp::Sorted& s, uint32_t C, uint32_t E) {
  const std::vector<uint32_t> hdr = b.hdr.read<uint32_t>(), tiles = b.tiles.read<uint32_t>(),
                              row_tok = b.row_tok.read<uint32_t>(), pair_row = b.pair_row.read<uint32_t>();
  return std::equal(s.hdr.begin(), s.hdr.begin() + bp::kHdrCount + E, hdr.begin()) &&
         std::equal(s.tiles.begin(), s.tiles.end(), tiles.begin()) &&
         std::equal(s.row_tok.begin(), s.row_tok.begin() + s.rows_used(), row_tok.begin()) &&
         std::equal(s.pair_row.begin(), s.pair_row.begin() + size_t(C) * br::kTopK, pair_row.begin());
}
void moe_checks(clrun::Device& dev) {
  const uint32_t I = 1536;
  clrun::Program p(dev, "src/kernels/bolt/nb_pf_moe.cl", {"MOE_I=1536", "TM=32", "PF_KC=2048"});
  {   // E 240 over 181 rows: sort, gather, combine
    const uint32_t C = 181, E = 240;
    const std::vector<uint32_t> rt = routes_of(C, E, 240, E);
    std::vector<uint16_t> lat(size_t(C) * br::kLatent);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = rne(fval(false, 95, uint32_t(i), 240));
    const bp::Sorted s = bp::sort(rt.data(), C, E);
    const uint32_t T = bp::tmax(C, E);
    clrun::Buffer drt(dev, rt), dl(dev, lat), xg(dev, size_t(T) * 32 * br::kLatent * 2);
    SortBufs sb(dev, C, E);
    p.run("nb_pf_sort", {256}, {256}, drt, sb.hdr, sb.tiles, sb.row_tok, sb.pair_row, C, T, E);
    report_eq("nb_pf_sort at E 240, 181 rows: header, tiles, rows, pair rows", sort_equal(sb, s, C, E));
    p.run("nb_pf_gather", {T * 32 * 64}, {64}, dl, sb.hdr, sb.row_tok, xg);
    const std::vector<uint16_t> g = xg.read<uint16_t>(), want = bp::gather(s, lat.data());
    report_eq("nb_pf_gather: the sorted latent rows (zeros for padding)", std::equal(want.begin(), want.end(), g.begin()));
    std::vector<uint16_t> yd(size_t(s.rows_used()) * br::kLatent);
    for (size_t i = 0; i < yd.size(); ++i) yd[i] = rne(fval(false, 96, uint32_t(i), 240, 2.0f));
    clrun::Buffer dy(dev, yd), dr(dev, size_t(C) * br::kLatent * 2);
    p.run("nb_pf_combine", {br::kLatent, C}, {256, 1}, drt, sb.pair_row, dy, dr, C);
    report16("nb_pf_combine: ascending-id fp32 sum, one rounding", dr.read<uint16_t>(), bp::combine(rt.data(), s, yd.data(), C), 0);
  }
  {   // the tile bound at C 2048, E 496 (494 experts at 33 rows, two at 1065)
    const uint32_t C = 2048, E = 496;
    std::vector<uint32_t> list;
    for (uint32_t e = 0; e < E; ++e) list.insert(list.end(), e < 494 ? 33u : 1065u, e);
    std::vector<uint32_t> rt(size_t(C) * br::kWords, 0);
    for (uint32_t j = 0; j < br::kTopK; ++j)
      for (uint32_t t = 0; t < C; ++t) rt[size_t(t) * br::kWords + j] = list[size_t(j) * C + t];
    const bp::Sorted s = bp::sort(rt.data(), C, E);
    clrun::Buffer drt(dev, rt);
    SortBufs sb(dev, C, E);
    p.run("nb_pf_sort", {256}, {256}, drt, sb.hdr, sb.tiles, sb.row_tok, sb.pair_row, C, bp::tmax(C, E), E);
    report_eq("nb_pf_sort at the tile bound (C 2048, E 496: " + std::to_string(s.tiles_used()) + " = tmax tiles)",
              sort_equal(sb, s, C, E) && s.tiles_used() == bp::tmax(C, E));
  }
  {   // the dequant: 32 experts at I 1536, routed over 64 rows (ids < 32), batches [0, 20) and [20, 32)
    const uint32_t C = 64, E = 32;
    const std::vector<uint32_t> rt = routes_of(C, E, 7, E);
    const bp::Sorted s = bp::sort(rt.data(), C, E);
    std::vector<uint32_t> blocks(size_t(E) * br::expert_block_words(I));
    for (size_t i = 0; i < blocks.size(); ++i) {
      const size_t within = (i % br::expert_block_words(I)) % 136;
      if (within < 128) {
        blocks[i] = br::fixture_hash(90, uint32_t(i), 7);
      } else {
        auto sc = [&](uint32_t h) {
          return uint32_t(common::f32_to_f16((1.0f + float((h >> 4) % 8) / 8.0f) * 0.015625f * ((h & 1) ? -1.0f : 1.0f)));
        };
        blocks[i] = sc(br::fixture_hash(91, uint32_t(i), 7)) | (sc(br::fixture_hash(92, uint32_t(i), 7)) << 16);
      }
    }
    clrun::Buffer drt(dev, rt), db(dev, blocks), up(dev, size_t(20) * br::kLatent * I * 2), dn(dev, size_t(20) * I * br::kLatent * 2);
    SortBufs sb(dev, C, E);
    p.run("nb_pf_sort", {256}, {256}, drt, sb.hdr, sb.tiles, sb.row_tok, sb.pair_row, C, bp::tmax(C, E), E);
    bool ok = true;
    uint32_t checked = 0;
    for (uint32_t b0 = 0; b0 < E; b0 += 20) {
      const uint32_t b1 = std::min(E, b0 + 20);
      p.run("nb_pf_dequant_up", {I, br::kLatent / 64, b1 - b0}, {16, 1, 1}, db, sb.hdr, up, b0, b1, E);
      p.run("nb_pf_dequant_dn", {br::kLatent, I / 64, b1 - b0}, {16, 1, 1}, db, sb.hdr, dn, b0, b1, E);
      const std::vector<uint16_t> gu = up.read<uint16_t>(), gd = dn.read<uint16_t>();
      for (uint32_t e = b0; e < b1; ++e) {
        if (!s.hdr[bp::kHdrCount + e]) continue;
        const std::vector<uint16_t> wu = bp::dequant_up(blocks.data(), e, I), wd = bp::dequant_dn(blocks.data(), e, I);
        ok = ok && std::equal(wu.begin(), wu.end(), gu.begin() + size_t(e - b0) * br::kLatent * I) &&
             std::equal(wd.begin(), wd.end(), gd.begin() + size_t(e - b0) * I * br::kLatent);
        ++checked;
      }
    }
    report_eq("nb_pf_dequant_up / _dn: " + std::to_string(checked) + " routed experts' bf16 blocks (two batches)", ok);
  }
}

// --- the gather, 25c's per-row kernels at M = 2048, the eager attention ---------------------------------------------------
void row_checks(clrun::Device& dev) {
  {   // nb_pf_embed_gather: ids < 64 (the first four 16-row tiles of the table)
    const uint32_t C = 37;
    std::vector<uint16_t> tiles(size_t(64) * H);
    for (size_t i = 0; i < tiles.size(); ++i) tiles[i] = uint16_t(br::fixture_hash(97, uint32_t(i), 1));
    std::vector<uint32_t> ids(C);
    for (uint32_t m = 0; m < C; ++m) ids[m] = (m * 7 + 3) % 64;
    clrun::Program p(dev, "src/kernels/bolt/nb_pf_prep.cl", {"K=4096", "EMBED_V=131072"});
    clrun::Buffer di(dev, ids), dt(dev, tiles), dh(dev, size_t(C) * H * 2);
    p.run("nb_pf_embed_gather", {256, C}, {256, 1}, di, dt, dh, C);
    std::vector<uint16_t> want;
    for (uint32_t id : ids) {
      const std::vector<uint16_t> r = br::embed_row(tiles.data(), id);
      want.insert(want.end(), r.begin(), r.end());
    }
    report16("nb_pf_embed_gather (37 rows from the chunk's ids)", dh.read<uint16_t>(), want, 0);
  }
  const uint32_t C = 300;
  const std::vector<std::string> kNorm = {"K=4096", "NORM_WGS=8", "NORM_EPS=1e-5f", "M=2048"};
  {   // nb_res_norm _S1_T1 at M 2048 (the MoE block's two sources at S 1)
    std::vector<float> w(H), s1(size_t(2048) * H, 0.0f), s2(size_t(2048) * H, 0.0f);
    for (uint32_t k = 0; k < H; ++k) w[k] = br::rf(1.0f + fval(false, 1, k, 3, 0.5f));
    std::vector<uint16_t> h(size_t(2048) * H, 0);
    for (size_t i = 0; i < size_t(C) * H; ++i) {
      h[i] = rne(fval(false, 98, uint32_t(i), 3, 2.0f));
      s1[i] = fval(false, 99, uint32_t(i), 3);
      s2[i] = fval(false, 100, uint32_t(i), 3);
    }
    clrun::Program p(dev, "src/kernels/bolt/nb_prep.cl", with(kNorm, {"NORM_S=1", "NORM_S2=1"}));
    clrun::Buffer dh(dev, h), d1(dev, s1), d2(dev, s2), dw(dev, w), ho(dev, size_t(2048) * H * 2), xo(dev, size_t(2048) * H * 2);
    p.run("nb_res_norm", {8 * 256, C}, {256, 1}, dh, d1, d2, dw, ho, xo);
    const br::NormOut tw = br::res_norm(h.data(), s1.data(), 1, s2.data(), 1, 2048, w.data());
    const std::vector<uint16_t> gx = xo.read<uint16_t>(), gh = ho.read<uint16_t>();
    report16("nb_res_norm _S1_T1 at M 2048 (300 rows): h", std::vector<uint16_t>(gh.begin(), gh.begin() + size_t(C) * H),
             std::vector<uint16_t>(tw.h_out.begin(), tw.h_out.begin() + size_t(C) * H), 0);
    report16("nb_res_norm _S1_T1 at M 2048 (300 rows): x", std::vector<uint16_t>(gx.begin(), gx.begin() + size_t(C) * H),
             std::vector<uint16_t>(tw.x.begin(), tw.x.begin() + size_t(C) * H), 1);
  }
  {   // the router and the route at M 2048 (E 240), the shared act, the attention prep
    const uint32_t E = 240;
    std::vector<uint16_t> x(size_t(C) * H), gw(size_t(E) * H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = rne(fval(false, 101, uint32_t(i), 3));
    for (size_t i = 0; i < gw.size(); ++i) gw[i] = rne(fval(false, 102, uint32_t(i), 3, 0.05f));
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
    std::vector<float> bias(E);
    for (uint32_t e = 0; e < E; ++e) bias[e] = 4.5f + br::fixture_f32(103, e, 3) * 0.04f;
    clrun::Program pr(dev, "src/kernels/bolt/nb_moe.cl", {"M=2048", "ROUTER_K=4096"});
    clrun::Program pt(dev, "src/kernels/bolt/nb_moe.cl", {"M=2048", "ROUTE_K=9"});
    clrun::Buffer dx(dev, x), dt(dev, tiles), dl(dev, size_t(C) * E * 4), db(dev, bias), drt(dev, size_t(C) * br::kWords * 4);
    pr.run("nb_router", {(E / 16) * 128, C}, {128, 1}, dx, dt, dl, E);
    pt.run("nb_route", {256, C}, {256, 1}, dl, db, drt, E);
    const std::vector<float> lg = dl.read<float>();
    std::vector<float> want;
    for (uint32_t m = 0; m < C; ++m) {
      const std::vector<float> r = br::router(x.data() + size_t(m) * H, tiles.data(), E);
      want.insert(want.end(), r.begin(), r.end());
    }
    report32("nb_router at M 2048 (300 rows, E 240)", lg, want, 0);
    const std::vector<uint32_t> got = drt.read<uint32_t>();
    bool ids = true;
    int64_t worst = 0;
    for (uint32_t m = 0; m < C; ++m) {
      const br::Route r = br::route(lg.data() + size_t(m) * E, bias.data(), E);
      for (uint32_t k = 0; k < br::kWords; ++k) {
        const uint32_t g = got[size_t(m) * br::kWords + k];
        if (k < br::kTopK) ids = ids && g == r.row[k];
        else if (k < br::kSelMin + 1) worst = std::max(worst, ulps32(br::as_f32(g), br::as_f32(r.row[k])));
      }
    }
    report_eq("nb_route at M 2048: ids exact, fp32 words within " + std::to_string(worst) + " ulp (bar 2)", ids && worst <= 2);
    std::vector<float> sl(size_t(2048) * br::kFc1N, 0.0f), sq(size_t(2048) * br::kQkvN, 0.0f);
    for (size_t i = 0; i < size_t(C) * br::kFc1N; ++i) sl[i] = fval(false, 104, uint32_t(i), 3, 3.0f);
    for (size_t i = 0; i < size_t(C) * br::kQkvN; ++i) sq[i] = fval(false, 105, uint32_t(i), 3);
    clrun::Program pa(dev, "src/kernels/bolt/nb_prep.cl", {"M=2048", "ACT_S=1"});
    clrun::Buffer dsl(dev, sl), dlat(dev, size_t(2048) * br::kLatent * 2), dsh(dev, size_t(2048) * br::kShared * 2);
    pa.run("nb_sh_act", {25 * 256, C}, {256, 1}, dsl, dlat, dsh);
    std::vector<uint16_t> lat(size_t(2048) * br::kLatent), sh(size_t(2048) * br::kShared);
    br::sh_act(sl.data(), 1, 2048, lat.data(), sh.data());
    const std::vector<uint16_t> gl = dlat.read<uint16_t>(), gs = dsh.read<uint16_t>();
    report_eq("nb_sh_act at M 2048 (300 rows): the latent and the shared relu2",
              std::equal(lat.begin(), lat.begin() + size_t(C) * br::kLatent, gl.begin()) &&
                  std::equal(sh.begin(), sh.begin() + size_t(C) * br::kShared, gs.begin()));
    const uint32_t pos = 4090;
    std::vector<uint32_t> ctrl(32, 0);
    ctrl[0] = pos;
    ctrl[1] = C;
    clrun::Program pp(dev, "src/kernels/bolt/nb_prep.cl", with(kCtrlDefs, {"M=2048", "QKV_S=1"}));
    clrun::Buffer dc(dev, ctrl), dsq(dev, sq), dq(dev, size_t(2048) * 4096 * 4), dk(dev, size_t(pos + C) * 256 * 2),
        dv(dev, size_t(pos + C) * 256 * 2);
    pp.run("nb_attn_prep", {18 * 256, C}, {256, 1}, dc, dsq, dq, dk, dv);
    const std::vector<float> q = dq.read<float>();
    const std::vector<uint16_t> k = dk.read<uint16_t>(), v = dv.read<uint16_t>();
    bool ok = true;
    for (uint32_t m = 0; m < C; ++m) {
      const br::Prep w = br::attn_prep(sq.data(), 1, 2048, m);
      ok = ok && std::equal(w.q.begin(), w.q.end(), q.begin() + size_t(m) * 4096) &&
           std::equal(w.k.begin(), w.k.end(), k.begin() + size_t(pos + m) * 256) &&
           std::equal(w.v.begin(), w.v.end(), v.begin() + size_t(pos + m) * 256);
    }
    report_eq("nb_attn_prep at M 2048 (300 rows at pos 4090: q, K / V rows)", ok);
  }
  {   // mg_pf_flash_attn EAGER at Bolt's layer: rows [2990, 3010) - each row decode's eager row (Decision 15, 23d's rule)
    const uint32_t pos = 2990, Cr = 20, L = pos + Cr;
    std::vector<uint16_t> kc(size_t(L) * 256), vc(size_t(L) * 256), q16(size_t(Cr) * 4096);
    for (size_t i = 0; i < kc.size(); ++i) {
      kc[i] = rne(fval(true, 71, uint32_t(i), 9));
      vc[i] = rne(fval(true, 72, uint32_t(i), 9));
    }
    std::vector<float> q(size_t(2048) * 4096, 0.0f);
    for (size_t i = 0; i < q16.size(); ++i) {
      q16[i] = rne(fval(true, 70, uint32_t(i), 9));
      q[i] = f32(q16[i]);
    }
    clrun::Program p(dev, "src/kernels/muse/mg_pf_attn.cl", {"TGT=64", "Q_HEADS=32", "KV_HEADS=2", "EAGER=1", "WINDOW=0", "RING=0"});
    clrun::Buffer dq(dev, q), dk(dev, kc), dv(dev, vc), dout(dev, size_t(Cr) * 4096 * 2);
    p.run("mg_pf_flash_attn", {32 * 128, Cr}, {128, 1}, dq, dk, dv, dout, pos, Cr);
    std::vector<uint16_t> want(size_t(Cr) * 4096);
    for (uint32_t t = 0; t < Cr; ++t) br::attention_eager(q16.data() + size_t(t) * 4096, kc.data(), vc.data(), pos + t,
                                                          want.data() + size_t(t) * 4096);
    // vs the twin: the softmax's 1 / sum is a division Apple's compiler does not round correctly (the card's build does:
    // bolt_pf_kernels_test holds the eager rows bitwise there) - an ulp in 1 / sum moves a p by an ulp, o by a few
    report16("mg_pf_flash_attn EAGER: rows 2990..3009 vs decode's eager twin", dout.read<uint16_t>(), want, 4);
    // vs decode's own eager kernel on this device (the same division): row by row, bitwise (23d's rule, Decision 15)
    clrun::Program pd(dev, "src/kernels/muse/mg_attn_eager.cl",
                      with(kCtrlDefs, {"M=1", "TGT=64", "Q_HEADS=32", "KV_HEADS=2", "WINDOW=0", "RING=0"}));
    const std::vector<uint16_t> ones(4096, uint16_t(0x3F80));
    clrun::Buffer gb(dev, ones), sb(dev, size_t(32) * L * 4), ob(dev, size_t(4096) * 2);
    std::vector<uint16_t> dec;
    for (uint32_t t = 0; t < Cr; ++t) {
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos + t;
      ctrl[1] = 1;
      clrun::Buffer cb(dev, ctrl), qb(dev, std::vector<float>(q.begin() + size_t(t) * 4096, q.begin() + size_t(t + 1) * 4096));
      pd.run("mg_attn_eager", {32 * 128, 1}, {128, 1}, cb, qb, dk, dv, gb, sb, ob, L);
      const std::vector<uint16_t> r = ob.read<uint16_t>();
      dec.insert(dec.end(), r.begin(), r.end());
    }
    report16("mg_pf_flash_attn EAGER: rows 2990..3009 vs decode's mg_attn_eager kernel", dout.read<uint16_t>(), dec, 0);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("bolt_pf_run: device %s\n", dev.name().c_str());
    mamba_checks(dev);
    moe_checks(dev);
    row_checks(dev);
  } catch (const std::exception& e) {
    std::printf("bolt_pf_run: %s\n", e.what());
    return 2;
  }
  if (failures) {
    std::printf("bolt_pf_run: %d check(s) DISAGREE\n", failures);
    return 1;
  }
  std::printf("bolt_pf_run: every check agrees\n");
  return 0;
}
