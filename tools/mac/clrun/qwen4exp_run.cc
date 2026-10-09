// qwen4exp_run - spec 21c's portable Qwen3.8-Flash-Next kernels on the Mac's OpenCL GPU against
// tests/kernels/qwen4exp_ref.h and torch's own outputs (tests/kernels/qwen4exp_fixture.h)
// (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   qwen4exp_run     no arguments; the real shapes (hidden 2560 in 4 streams, low rank 320, QSA 24 / 2 x 256
//                    with the 4 x 128 + 128 indexer, 512 experts top 10 x 640, PLE 16 x 160), the CMake block's
//                    defines (src/kernels/CMakeLists.txt "Spec 21c") except where the Mac's work-group cap says
//                    otherwise (printed: SEL_WG and DN_KS below)
//
// What runs:
//   q4_hc.cl         q4_hc_combine_norm _E, _S4, _S1, _Y, _X, _Y_NN (H exact, xn within the bar); q4_hc_up_mix
//                    _I and plain (x and the inject weights)
//   q4_ple.cl        q4_ple_gather under PLE_DIRECT (one buffer of every head's rows / scales and byte offsets
//                    in place of 21b's host-USM pointer table: OpenCL 1.2 passes no pointers) over a small table
//                    (16 heads, sizes q4_ple_primes(1000, 16, 0), random int8 rows, bf16 and fp32 scales) at
//                    positions 0..20 one launch each (EOS 248044 and id 248319 in the sequence): ids against
//                    loader::q4_ple_ids through q4ref::ple_history, rows against q4ref::ple_row, the id ring;
//                    q4_ple_block over 12 positions through the 16-slot conv ring
//   q4_qsa.cl        q4_qsa_prep at 0..3 and 2044..2051 (blocks 0, 511, 512 completed; the 8-slot tail), the
//                    scores at 600 blocks (and the early-out past n), the selection EXACT (list, count, diag):
//                    n 512 (identity) / 513 / 600 with exact ties planted across the 512 / 513 cut, the cut
//                    inside zeros, the fixture's four crafted rows, p's own block winning, n 65536 once - tails 0..3
//   q4_qsa_attn_eager.cl   over q4ref::qsa_select's lists at p 1, 2050 (identity), 2051 (the first cut), 2400
//   q4_moe.cl        q4_route (random rows, the fixture's four, a tie for the 10th, all -1e30 but ten: ids exact);
//                    q4_moe_gate_up / q4_moe_down, SHARED_BF16 0 and 1, over 16 allocated expert blocks and a
//                    crafted route row (ids < 16): all slots, each routed slot alone (the others' weights zero),
//                    the shared slot alone
//   prep.cl          prep_gated_head _SIG (GDN_GATE_SIGMOID=1), random and the fixture's case
// Not run: q4_qsa_attn.cl (sub-group shuffles and reductions: the box's qwen4exp_kernels_test).
//
// Bars: integer and exp-free / division-free outputs exact (H, the PLE ids and rows, the tail, the selection,
// the route ids, the MoE down + combine from the device's own h); everything else within 2 bf16 ulps - the exps
// are exp_torch's fma chains (exact wherever fma is), but Apple's compiler is not told
// -cl-fp32-correctly-rounded-divide-sqrt, so 1 / sqrt and every divide may move an ulp (counted per line).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "kernels/qwen4exp_fixture.h"
#include "kernels/qwen4exp_ref.h"
#include "loader/qwen4exp_ple_hash.h"

namespace {

namespace qr = q4ref;
namespace fx = qwen4exp_fixture;
using qr::f32;
using qr::rf;
using qr::rne;

const std::vector<std::string> kCtrlDefs = {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18"};
std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
const std::vector<std::string> kListDefs = {"LIST_ROW=2064", "COUNT_W=2052"};
constexpr uint32_t kListRow = 2064, kCountW = 2052;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
int64_t ulps32(float a, float b) {
  auto key = [](float f) {
    const uint32_t u = qr::as_u32(f);
    return (u & 0x80000000u) ? -int64_t(u & 0x7FFFFFFFu) : int64_t(u);
  };
  return std::llabs(key(a) - key(b));
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
std::vector<float> bf16_vals(size_t n, float lo, float hi, uint32_t seed) {   // fp32 holding bf16 values
  const std::vector<uint16_t> b = random_bf16(n, lo, hi, seed);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = f32(b[i]);
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
// One line per check: "exact", "within N ulp (k of n differ)", or the first differing index.
struct Diff {
  size_t n = 0, differ = 0, first = SIZE_MAX;
  int64_t worst = 0;
  uint64_t got_first = 0, want_first = 0;
  void add(size_t i, int64_t d, uint64_t got, uint64_t want) {
    ++n;
    if (d == 0) return;
    ++differ;
    if (first == SIZE_MAX) {
      first = i;
      got_first = got;
      want_first = want;
    }
    worst = std::max(worst, d);
  }
  void add16(size_t i, uint16_t got, uint16_t want) { add(i, ulps16(got, want), got, want); }
  void add32(size_t i, float got, float want) { add(i, ulps32(got, want), qr::as_u32(got), qr::as_u32(want)); }
  void addx(size_t i, uint64_t got, uint64_t want) { add(i, got == want ? 0 : 1, got, want); }   // exact only
};
void report(const std::string& what, const Diff& d, int64_t bar, const char* unit = "bf16 ulp") {
  if (d.differ == 0) {
    std::printf("qwen4exp_run: %s exact (%zu values)\n", what.c_str(), d.n);
  } else if (d.worst <= bar) {
    std::printf("qwen4exp_run: %s within %lld %s (%zu of %zu differ)\n", what.c_str(), (long long)d.worst, unit, d.differ,
                d.n);
  } else {
    std::printf("qwen4exp_run: %s DISAGREES: first differing index %zu (got 0x%llx, want 0x%llx); worst %lld %s > bar "
                "%lld (%zu of %zu differ)\n",
                what.c_str(), d.first, (unsigned long long)d.got_first, (unsigned long long)d.want_first,
                (long long)d.worst, unit, (long long)bar, d.differ, d.n);
    ++failures;
  }
}
void report_bool(const std::string& what, bool ok, const std::string& detail = "") {
  std::printf("qwen4exp_run: %s %s%s\n", what.c_str(), ok ? "exact" : "DISAGREES", detail.c_str());
  if (!ok) ++failures;
}

constexpr uint32_t D = qr::kHidden, HCN = qr::kHcN;

std::vector<uint32_t> ctrl_at(uint32_t pos, uint32_t n_active = 1) {
  std::vector<uint32_t> c(32, 0);
  c[0] = pos;
  c[1] = n_active;
  return c;
}

// ---- q4_hc.cl ------------------------------------------------------------------------------------------------------
void hc_checks(clrun::Device& dev) {
  const std::vector<uint16_t> H0 = random_bf16(HCN, -2.f, 2.f, 1);
  const std::vector<float> w = bf16_vals(HCN, 0.5f, 1.5f, 2);
  const std::vector<float> inj = {rf(0.75f), rf(1.25f), rf(0.5f), rf(1.875f)};
  const std::vector<float> slices = random_f32(size_t(4) * D, -0.5f, 0.5f, 3);
  const std::vector<uint16_t> ybf = random_bf16(D, -1.f, 1.f, 4), emb = random_bf16(D, -0.1f, 0.1f, 5);
  const std::vector<uint32_t> ctrl = ctrl_at(7);
  struct V {
    const char* name;
    std::vector<std::string> defs;
    int src;   // 0 E, 1 S, 2 Y, 3 X
    uint32_t S;
    bool norm;
  };
  const std::vector<V> vs = {
      {"_E", {"HC_SRC=0", "HC_NORM=1"}, 0, 0, true},  {"_S4", {"HC_SRC=1", "SRC_S=4", "HC_NORM=1"}, 1, 4, true},
      {"_S1", {"HC_SRC=1", "SRC_S=1", "HC_NORM=1"}, 1, 1, true}, {"_Y", {"HC_SRC=2", "HC_NORM=1"}, 2, 0, true},
      {"_X", {"HC_SRC=3", "HC_NORM=1"}, 3, 0, true},  {"_Y_NN", {"HC_SRC=2", "HC_NORM=0"}, 2, 0, false},
  };
  for (const V& v : vs) {
    clrun::Program p(dev, "src/kernels/qwen4exp/q4_hc.cl", with(with(kCtrlDefs, {"M=1"}), v.defs));
    clrun::Buffer cb(dev, ctrl), hb(dev, H0), ib(dev, inj), wb(dev, w), xb(dev, size_t(HCN) * 2);
    std::vector<uint16_t> y(D);
    std::unique_ptr<clrun::Buffer> sb;
    if (v.src == 0) {
      sb.reset(new clrun::Buffer(dev, emb));
      y = emb;
    } else if (v.src == 1) {
      sb.reset(new clrun::Buffer(dev, std::vector<float>(slices.begin(), slices.begin() + size_t(v.S) * D)));
      for (uint32_t k = 0; k < D; ++k) y[k] = qr::y_of_slices(slices.data(), v.S, 1, 0, D, k);
    } else if (v.src == 2) {
      sb.reset(new clrun::Buffer(dev, ybf));
      y = ybf;
    } else {
      sb.reset(new clrun::Buffer(dev, 64));   // unread
    }
    p.run("q4_hc_combine_norm", {4 * 256, 1}, {256, 1}, cb, hb, *sb, ib, wb, xb);
    std::vector<uint16_t> Hh = H0, xh(HCN, 0);
    const qr::HcSrc hs = v.src == 0 ? qr::HcSrc::Embed : v.src == 3 ? qr::HcSrc::None : qr::HcSrc::Y;
    qr::hc_combine_norm(Hh.data(), hs, y.data(), inj.data(), w.data(), v.norm ? xh.data() : nullptr);
    const std::vector<uint16_t> Hd = hb.read<uint16_t>(), xd = xb.read<uint16_t>();
    Diff dh, dx;
    for (uint32_t i = 0; i < HCN; ++i) {
      dh.add16(i, Hd[i], Hh[i]);
      dx.add16(i, xd[i], xh[i]);   // _NN: xn untouched (zero)
    }
    report(std::string("q4_hc_combine_norm_M1") + v.name + " H", dh, 0);
    report(std::string("q4_hc_combine_norm_M1") + v.name + (v.norm ? " xn" : " xn untouched"), dx, v.norm ? 2 : 0);
  }
  for (int inject = 1; inject >= 0; --inject) {
    const uint32_t rows = inject ? 336 : 320;
    clrun::Program p(dev, "src/kernels/qwen4exp/q4_hc.cl",
                     with(kCtrlDefs, {"M=1", "UP_DOWN=" + std::to_string(rows), inject ? "UP_INJECT=1" : "UP_INJECT=0"}));
    const std::vector<float> down = random_f32(rows, -6.f, 6.f, 10 + inject);
    const std::vector<uint16_t> up = random_bf16(size_t(qr::kHcLow) * HCN, -0.15f, 0.15f, 12);
    const std::vector<uint16_t> xn = random_bf16(HCN, -2.f, 2.f, 13);
    clrun::Buffer cb(dev, ctrl), db(dev, down), ub(dev, up), xnb(dev, xn), xb(dev, size_t(D) * 2), ib(dev, 16);
    p.run("q4_hc_up_mix", {(D / 16) * 64, 1}, {64, 1}, cb, db, ub, xnb, xb, ib);
    std::vector<uint16_t> xh(D);
    float ih[4] = {0, 0, 0, 0};
    qr::hc_up_mix(down.data(), inject, up.data(), xn.data(), xh.data(), ih);
    const std::vector<uint16_t> xd = xb.read<uint16_t>();
    const std::vector<float> id = ib.read<float>();
    Diff dx, di;
    for (uint32_t c = 0; c < D; ++c) dx.add16(c, xd[c], xh[c]);
    for (uint32_t s = 0; s < 4; ++s) di.add16(s, rne(id[s]), rne(ih[s]));
    const std::string nm = inject ? "q4_hc_up_mix_M1_I" : "q4_hc_up_mix_M1";
    report(nm + " x", dx, 2);
    report(nm + (inject ? " inj" : " inj untouched"), di, inject ? 2 : 0);
  }
}

// ---- q4_ple.cl -----------------------------------------------------------------------------------------------------
void ple_checks(clrun::Device& dev) {
  const std::vector<uint64_t> sizes = loader::q4_ple_primes(1000, 16, 0), offs = loader::q4_ple_offsets(sizes);
  const std::array<uint64_t, 3> mult = {fx::kPleMult[0], fx::kPleMult[1], fx::kPleMult[2]};
  const uint64_t R = loader::q4_ple_total_rows(sizes);
  std::vector<int8_t> q_all(size_t(R) * qr::kPleDim);
  {
    std::mt19937 rng(50);
    for (int8_t& v : q_all) v = int8_t(int(rng() % 255) - 127);
  }
  std::vector<uint64_t> consts(3 + 32);
  for (uint32_t i = 0; i < 3; ++i) consts[i] = mult[i];
  for (uint32_t h = 0; h < 16; ++h) {
    consts[3 + h] = sizes[h];
    consts[3 + 16 + h] = offs[h];
  }
  // positions 0..20: 248319 at 0 and 13, EOS at 4, 9, 10 (EOS as t0, t1 and t2 of later rows)
  std::vector<uint32_t> seq(21);
  {
    std::mt19937 rng(51);
    for (uint32_t& t : seq) t = rng() % 248320;
    seq[0] = seq[13] = 248319;
    seq[4] = seq[9] = seq[10] = qr::kPleEos;
  }
  for (int bf16_scale = 1; bf16_scale >= 0; --bf16_scale) {
    const size_t sb = bf16_scale ? 2 : 4;
    std::vector<uint8_t> s_all(size_t(R) * sb);
    std::vector<float> scale(R);
    {
      const std::vector<float> sc = random_f32(R, 0.001f, 0.05f, 52);
      for (uint64_t r = 0; r < R; ++r) {
        if (bf16_scale) {
          const uint16_t b = rne(sc[r]);
          std::memcpy(&s_all[r * 2], &b, 2);
          scale[r] = f32(b);
        } else {
          std::memcpy(&s_all[r * 4], &sc[r], 4);
          scale[r] = sc[r];
        }
      }
    }
    std::vector<uint64_t> head_off(32);
    for (uint32_t h = 0; h < 16; ++h) {
      head_off[h] = offs[h] * qr::kPleDim;
      head_off[16 + h] = offs[h] * sb;
    }
    clrun::Program p(dev, "src/kernels/qwen4exp/q4_ple.cl",
                     with(kCtrlDefs, {"M=1", "PLE_GATHER=1", "PLE_EOS=248044",
                                      bf16_scale ? "PLE_SCALE_BF16=1" : "PLE_SCALE_BF16=0", "PLE_DIRECT=1"}));
    clrun::Buffer ringb(dev, std::vector<uint32_t>(16, 0xDEADBEEFu)), qb(dev, q_all), sbuf(dev, s_all), ob(dev, head_off),
        kb(dev, consts), eb(dev, size_t(D) * 2), idb(dev, 16 * 8);
    std::vector<uint32_t> ctrl = ctrl_at(0);
    clrun::Buffer cb(dev, ctrl);
    Diff dids, drow, dring;
    for (uint32_t pos = 0; pos <= 20; ++pos) {
      ctrl[0] = pos;
      ctrl[2] = seq[pos];
      cb.write(ctrl.data(), ctrl.size() * 4);
      p.run("q4_ple_gather", {16 * 160, 1}, {160, 1}, cb, ringb, qb, sbuf, ob, kb, eb, idb);
      const loader::Q4PleHistory hh = qr::ple_history(pos, [&](uint32_t q) { return seq[q]; });
      const std::array<uint64_t, 16> want = loader::q4_ple_ids(hh.t0, hh.t1, hh.t2, mult, sizes, offs);
      const std::vector<uint64_t> got = idb.read<uint64_t>();
      const std::vector<uint16_t> e = eb.read<uint16_t>();
      for (uint32_t h = 0; h < 16; ++h) {
        dids.addx(pos * 16 + h, got[h], want[h]);
        uint16_t row[qr::kPleDim];
        qr::ple_row(q_all.data() + want[h] * qr::kPleDim, scale[want[h]], row);
        for (uint32_t i = 0; i < qr::kPleDim; ++i) drow.add16(size_t(pos) * D + h * qr::kPleDim + i, e[h * qr::kPleDim + i], row[i]);
      }
      const std::vector<uint32_t> ring = ringb.read<uint32_t>();
      dring.addx(pos, ring[pos % 16], seq[pos]);
    }
    const std::string nm = bf16_scale ? "q4_ple_gather_M1_BF16 (PLE_DIRECT)" : "q4_ple_gather_M1_F32 (PLE_DIRECT)";
    report(nm + " ids, positions 0..20 x 16 heads", dids, 0, "mismatch");
    report(nm + " rows", drow, 0);
    report(nm + " id ring advance", dring, 0, "mismatch");
  }
  // q4_ple_block: 12 positions through the 16-slot conv ring
  {
    clrun::Program p(dev, "src/kernels/qwen4exp/q4_ple.cl", with(kCtrlDefs, {"M=1", "PLE_BLOCK=1"}));
    std::vector<float> pw = bf16_vals(size_t(3) * HCN, 0.5f, 1.5f, 60);
    const std::vector<float> taps = random_f32(size_t(HCN) * 4, -0.5f, 0.5f, 61);
    pw.insert(pw.end(), taps.begin(), taps.end());
    const float* wk = pw.data();
    const float* wq = pw.data() + HCN;
    const float* wc = pw.data() + 2 * HCN;
    clrun::Buffer pwb(dev, pw), ringb(dev, size_t(16) * HCN * 2), hb(dev, size_t(HCN) * 2), kvb(dev, size_t(12800) * 4);
    std::vector<std::vector<uint16_t>> hring(16, std::vector<uint16_t>(HCN, 0));
    std::vector<uint32_t> ctrl = ctrl_at(0);
    clrun::Buffer cb(dev, ctrl);
    Diff dh, dr;
    for (uint32_t pos = 0; pos < 12; ++pos) {
      const std::vector<float> kv = random_f32(12800, -2.f, 2.f, 70 + pos);
      const std::vector<uint16_t> H = random_bf16(HCN, -2.f, 2.f, 90 + pos);
      ctrl[0] = pos;
      cb.write(ctrl.data(), ctrl.size() * 4);
      hb.write(H.data(), H.size() * 2);
      kvb.write(kv.data(), kv.size() * 4);
      p.run("q4_ple_block", {4 * 256, 1}, {256, 1}, cb, hb, kvb, pwb, ringb);
      std::vector<uint16_t> Hh = H, out(HCN);
      const uint16_t* hist[3] = {pos >= 9 ? hring[(pos - 9) % 16].data() : nullptr,
                                 pos >= 6 ? hring[(pos - 6) % 16].data() : nullptr,
                                 pos >= 3 ? hring[(pos - 3) % 16].data() : nullptr};
      qr::ple_block(Hh.data(), kv.data(), wk, wq, wc, taps.data(), hist, out.data());
      hring[pos % 16] = out;
      const std::vector<uint16_t> Hd = hb.read<uint16_t>(), rd = ringb.read<uint16_t>();
      for (uint32_t c = 0; c < HCN; ++c) {
        dh.add16(size_t(pos) * HCN + c, Hd[c], Hh[c]);
        dr.add16(size_t(pos) * HCN + c, rd[size_t(pos % 16) * HCN + c], out[c]);
      }
    }
    report("q4_ple_block_M1 H, positions 0..11", dh, 2);
    report("q4_ple_block_M1 conv ring rows", dr, 2);
  }
}

// ---- q4_qsa.cl -----------------------------------------------------------------------------------------------------
std::vector<float> rope_table(uint32_t rows) {   // fp32 [rows][2][32], bf16 values (the loader's table's form)
  std::vector<float> t(size_t(rows) * 64);
  for (uint32_t p = 0; p < rows; ++p)
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = double(p) * std::pow(1e7, -double(2 * i) / 64.0);
      t[size_t(p) * 64 + i] = rf(float(std::cos(a)));
      t[size_t(p) * 64 + 32 + i] = rf(float(std::sin(a)));
    }
  return t;
}

uint32_t g_sel_wg = 1024;

// One selection launch (row p over `scores`, `stride` wide) against q4ref::qsa_select: list, count, diag exact.
bool select_case(clrun::Device& dev, clrun::Program& p, const std::vector<float>& scores, uint32_t pos, std::string* why) {
  const uint32_t stride = uint32_t(scores.size());
  clrun::Buffer cb(dev, ctrl_at(pos)), sb(dev, scores), lb(dev, std::vector<uint32_t>(kListRow, 0xFFFFFFFFu)), db(dev, 8);
  p.run("q4_qsa_select", {g_sel_wg, 1}, {g_sel_wg, 1}, cb, sb, stride, lb, db);
  const qr::Selection sel = qr::qsa_select(scores.data(), pos);
  const std::vector<uint32_t> l = lb.read<uint32_t>();
  const std::vector<float> dg = db.read<float>();
  const uint32_t c = sel.count();
  if (l[kCountW] != c) {
    *why = "count " + std::to_string(l[kCountW]) + " want " + std::to_string(c);
    return false;
  }
  for (uint32_t i = 0; i < c; ++i)
    if (l[i] != sel.list[i]) {
      *why = "list[" + std::to_string(i) + "] " + std::to_string(l[i]) + " want " + std::to_string(sel.list[i]);
      return false;
    }
  for (uint32_t i = c; i < kListRow; ++i)
    if (i != kCountW && l[i] != 0xFFFFFFFFu) {
      *why = "word " + std::to_string(i) + " past the count written";
      return false;
    }
  if (qr::as_u32(dg[0]) != qr::as_u32(sel.s512) || qr::as_u32(dg[1]) != qr::as_u32(sel.s513)) {
    *why = "diag {" + std::to_string(dg[0]) + ", " + std::to_string(dg[1]) + "} want {" + std::to_string(sel.s512) + ", " +
           std::to_string(sel.s513) + "}";
    return false;
  }
  return true;
}

// Distinct scores 0.01 x (a permutation of 1..n), then exact ties: the blocks of descending ranks [lo, hi) set to
// the score of rank lo (a tie group straddling the 512 / 513 cut when lo < 512 < hi).
std::vector<float> tied_scores(uint32_t n, uint32_t stride, uint32_t lo, uint32_t hi, uint32_t seed) {
  std::vector<uint32_t> perm(n);
  for (uint32_t b = 0; b < n; ++b) perm[b] = b + 1;
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<float> s(stride, 0.0f);
  for (uint32_t b = 0; b < n; ++b) s[b] = 0.01f * float(perm[b]);
  std::vector<uint32_t> idx(n);
  for (uint32_t b = 0; b < n; ++b) idx[b] = b;
  std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return s[a] > s[b]; });
  const float v = s[idx[lo]];
  for (uint32_t r = lo; r < hi && r < n; ++r) s[idx[r]] = v;
  return s;
}

void qsa_checks(clrun::Device& dev) {
  const std::vector<std::string> defs =
      with(with(kCtrlDefs, {"M=1", "TOPB=512", "SEL_WG=" + std::to_string(g_sel_wg)}), kListDefs);
  clrun::Program p(dev, "src/kernels/qwen4exp/q4_qsa.cl", defs);
  // -- prep: positions 0..3 (block 0) and 2044..2051 (blocks 511, 512), M = 1, the tail ring carried
  {
    const uint32_t ML = 2560;
    const std::vector<float> rope = rope_table(ML);
    std::vector<float> small = bf16_vals(768, 0.5f, 1.5f, 100);
    const float* wq = small.data() + 512;
    const float* wk = small.data() + 640;
    clrun::Buffer rb(dev, rope), smb(dev, small), qb(dev, 512 * 4), tb(dev, 8 * 128 * 2), kb(dev, size_t(ML / 4) * 128 * 2),
        ib(dev, 640 * 4);
    std::vector<uint32_t> ctrl = ctrl_at(0);
    clrun::Buffer cb(dev, ctrl);
    std::vector<std::vector<uint16_t>> raw(ML);
    Diff dq, dt, dk;
    std::vector<uint32_t> blocks;
    for (uint32_t pos : {0u, 1u, 2u, 3u, 2044u, 2045u, 2046u, 2047u, 2048u, 2049u, 2050u, 2051u}) {
      const std::vector<float> idx = random_f32(640, -3.f, 3.f, 200 + pos);
      ctrl[0] = pos;
      cb.write(ctrl.data(), ctrl.size() * 4);
      ib.write(idx.data(), idx.size() * 4);
      p.run("q4_qsa_prep", {5 * 128, 1}, {128, 1}, cb, ib, smb, rb, qb, tb, kb);
      float qh[512];
      qr::qsa_q(idx.data(), wq, rope.data() + size_t(pos) * 64, qh);
      const std::vector<float> qd = qb.read<float>();
      for (uint32_t i = 0; i < 512; ++i) dq.add16(size_t(pos) * 512 + i, rne(qd[i]), rne(qh[i]));
      raw[pos].resize(128);
      qr::qsa_raw_key(idx.data(), raw[pos].data());
      const std::vector<uint16_t> td = tb.read<uint16_t>();
      for (uint32_t i = 0; i < 128; ++i) dt.add16(size_t(pos) * 128 + i, td[(pos % 8) * 128 + i], raw[pos][i]);
      if ((pos + 1) % 4 == 0) {
        const uint32_t b = (pos + 1) / 4 - 1;
        const uint16_t* r4[4] = {raw[4 * b].data(), raw[4 * b + 1].data(), raw[4 * b + 2].data(), raw[4 * b + 3].data()};
        uint16_t key[128];
        qr::qsa_block_key(r4, wk, rope.data() + size_t(4 * b) * 64, key);
        const std::vector<uint16_t> kd = kb.read<uint16_t>();
        for (uint32_t i = 0; i < 128; ++i) dk.add16(size_t(b) * 128 + i, kd[size_t(b) * 128 + i], key[i]);
        blocks.push_back(b);
      }
    }
    report("q4_qsa_prep_M1 idx q (positions 0..3, 2044..2051)", dq, 2);
    report("q4_qsa_prep_M1 raw keys in the 8-slot tail", dt, 0);
    report("q4_qsa_prep_M1 compressed keys (blocks 0, 511, 512)", dk, 2);
  }
  // -- prep at M = 4 (Review Focus 3, 21e's verify rows; not a B70 binary yet): launches of 1..4 rows at every
  //    alignment, a completing row reading its block's older raw keys from the 8-slot tail while the launch's other
  //    rows write theirs
  {
    clrun::Program p4(dev, "src/kernels/qwen4exp/q4_qsa.cl",
                      with(with(kCtrlDefs, {"M=4", "TOPB=512", "SEL_WG=" + std::to_string(g_sel_wg)}), kListDefs));
    const uint32_t ML = 64;
    const std::vector<float> rope = rope_table(ML);
    const std::vector<float> small = bf16_vals(768, 0.5f, 1.5f, 110);
    clrun::Buffer rb(dev, rope), smb(dev, small), qb(dev, 4 * 512 * 4), tb(dev, 8 * 128 * 2), kb(dev, size_t(ML / 4) * 128 * 2),
        ib(dev, 4 * 640 * 4);
    std::vector<std::vector<uint16_t>> raw(ML, std::vector<uint16_t>(128));
    std::vector<std::vector<float>> idx(ML);
    for (uint32_t q = 0; q < ML; ++q) {
      idx[q] = random_f32(640, -3.f, 3.f, 120 + q);
      qr::qsa_raw_key(idx[q].data(), raw[q].data());
    }
    Diff dq, dk;
    uint32_t pos = 0, nkeys = 0;
    for (uint32_t n : {4u, 3u, 4u, 2u, 4u, 1u, 4u, 4u, 3u}) {
      std::vector<float> rows;
      for (uint32_t m = 0; m < n; ++m) rows.insert(rows.end(), idx[pos + m].begin(), idx[pos + m].end());
      rows.resize(4 * 640, 0.0f);
      ib.write(rows.data(), rows.size() * 4);
      clrun::Buffer cb(dev, ctrl_at(pos, n));
      p4.run("q4_qsa_prep", {5 * 128, 4}, {128, 1}, cb, ib, smb, rb, qb, tb, kb);
      const std::vector<float> qd = qb.read<float>();
      const std::vector<uint16_t> kd = kb.read<uint16_t>();
      for (uint32_t m = 0; m < n; ++m) {
        const uint32_t pp = pos + m;
        float qh[512];
        qr::qsa_q(idx[pp].data(), small.data() + 512, rope.data() + size_t(pp) * 64, qh);
        for (uint32_t i = 0; i < 512; ++i) dq.add16(size_t(pp) * 512 + i, rne(qd[m * 512 + i]), rne(qh[i]));
        if ((pp + 1) % 4 == 0) {
          const uint32_t b = (pp + 1) / 4 - 1;
          const uint16_t* r4[4] = {raw[4 * b].data(), raw[4 * b + 1].data(), raw[4 * b + 2].data(), raw[4 * b + 3].data()};
          uint16_t key[128];
          qr::qsa_block_key(r4, small.data() + 640, rope.data() + size_t(4 * b) * 64, key);
          for (uint32_t i = 0; i < 128; ++i) dk.add16(size_t(b) * 128 + i, kd[size_t(b) * 128 + i], key[i]);
          ++nkeys;
        }
      }
      pos += n;
    }
    report("q4_qsa_prep M=4 idx q (launches of 4, 3, 4, 2, 4, 1, 4, 4, 3 rows)", dq, 2);
    report("q4_qsa_prep M=4 compressed keys (" + std::to_string(nkeys) + " blocks, the 8-slot tail across launches)", dk, 2);
  }
  // -- score: 600 complete blocks (p 2399) on a 1024-block stride; nothing past n; n 0 (p 2) writes nothing
  std::vector<float> dev_scores;
  {
    const uint32_t stride = 1024, n = 600, pos = 4 * n - 1;
    const std::vector<float> q = bf16_vals(512, -1.f, 1.f, 300);
    const std::vector<uint16_t> keys = random_bf16(size_t(stride) * 128, -1.f, 1.f, 301);
    clrun::Buffer qb(dev, q), kb(dev, keys), sb(dev, std::vector<float>(stride, -1.0f));
    clrun::Buffer cb(dev, ctrl_at(pos));
    p.run("q4_qsa_score", {(stride / 256) * 256, 1}, {256, 1}, cb, qb, kb, sb, stride);
    dev_scores = sb.read<float>();
    Diff ds;
    bool past = true;
    for (uint32_t b = 0; b < stride; ++b) {
      if (b < n) ds.add32(b, dev_scores[b], qr::qsa_score(q.data(), keys.data() + size_t(b) * 128));
      else past = past && dev_scores[b] == -1.0f;
    }
    report("q4_qsa_score_M1 scores (600 blocks)", ds, 2, "fp32 ulp");
    clrun::Buffer sb0(dev, std::vector<float>(stride, -1.0f)), cb0(dev, ctrl_at(2));
    p.run("q4_qsa_score", {(stride / 256) * 256, 1}, {256, 1}, cb0, qb, kb, sb0, stride);
    const std::vector<float> s0 = sb0.read<float>();
    for (float v : s0) past = past && v == -1.0f;
    report_bool("q4_qsa_score_M1 early-out past n (and n 0)", past);
    for (uint32_t b = n; b < stride; ++b) dev_scores[b] = 0.0f;
    dev_scores.resize(n);
  }
  // -- select: EXACT
  {
    struct Case {
      std::string name;
      std::vector<float> s;
    };
    std::vector<Case> cases;
    cases.push_back({"n 512 (identity)", tied_scores(512, 512, 0, 0, 400)});
    cases.push_back({"n 513, 6 tied across the cut", tied_scores(513, 513, 509, 515, 401)});
    cases.push_back({"n 513, 2 tied at 512 / 513", tied_scores(513, 513, 511, 513, 402)});
    cases.push_back({"n 600, 9 tied across the cut", tied_scores(600, 600, 506, 515, 403)});
    cases.push_back({"n 600, the device's own scores", dev_scores});
    {
      std::vector<float> z = tied_scores(600, 600, 0, 0, 404);   // 500 positive, the cut inside 100 zeros
      std::vector<uint32_t> idx(600);
      for (uint32_t b = 0; b < 600; ++b) idx[b] = b;
      std::shuffle(idx.begin(), idx.end(), std::mt19937(405));
      for (uint32_t i = 0; i < 100; ++i) z[idx[i]] = 0.0f;
      cases.push_back({"n 600, the cut inside zeros", z});
    }
    {
      const std::vector<uint32_t> rows = qr::hex32(fx::kSelRows);
      for (uint32_t r = 0; r < 4; ++r) {
        std::vector<float> s(fx::kSelRowsN[r]);
        for (uint32_t b = 0; b < s.size(); ++b) s[b] = qr::as_f32(rows[size_t(r) * fx::kSelN + b]);
        cases.push_back({"fixture row " + std::to_string(r) + " (n " + std::to_string(s.size()) + ")", s});
      }
    }
    {
      std::vector<float> dec(513);   // strictly decreasing: p's own block (512) loses at 2051; then wins
      for (uint32_t b = 0; b < dec.size(); ++b) dec[b] = 1000.0f - float(b);
      cases.push_back({"n 513, p's own block last", dec});
      dec[512] = 2000.0f;
      cases.push_back({"n 513, p's own block wins", dec});
    }
    for (const Case& c : cases) {
      bool ok = true;
      std::string why;
      for (uint32_t tail = 0; tail < 4 && ok; ++tail) {
        const uint32_t n = uint32_t(c.s.size()), pos = 4 * n - 1 + tail;
        ok = select_case(dev, p, c.s, pos, &why);
        if (!ok) why = " (tail " + std::to_string(tail) + ": " + why + ")";
      }
      report_bool("q4_qsa_select_M1 " + c.name + ", tails 0..3: list, count, diag", ok, why);
    }
    {   // n 65536 (max_len 262144) once: coarse scores, ties everywhere
      std::vector<float> s(65536);
      std::mt19937 rng(406);
      for (float& v : s) v = 0.25f * float(rng() % 4096);
      std::string why;
      const bool ok = select_case(dev, p, s, 4 * 65536 - 1 + 2, &why);
      report_bool("q4_qsa_select_M1 n 65536 (max_len 262144), tail 2: list, count, diag", ok, ok ? "" : " (" + why + ")");
    }
  }
}

// ---- q4_qsa_attn_eager.cl ------------------------------------------------------------------------------------------
void eager_checks(clrun::Device& dev) {
  clrun::Program p(dev, "src/kernels/qwen4exp/q4_qsa_attn_eager.cl", with(with(kCtrlDefs, {"M=1", "LISTMAX=2052"}), kListDefs));
  const uint32_t rows = 2404, KVN = qr::kKvHeads * qr::kHd, QN = qr::kQHeads * qr::kHd;
  const std::vector<uint16_t> k = random_bf16(size_t(rows) * KVN, -1.f, 1.f, 500), v = random_bf16(size_t(rows) * KVN, -1.f, 1.f, 501);
  const std::vector<float> q = random_f32(QN, -2.f, 2.f, 502), gate = random_f32(QN, -4.f, 4.f, 503);
  clrun::Buffer kb(dev, k), vb(dev, v), qb(dev, q), gb(dev, gate);
  const std::vector<float> scores = random_f32(600, 0.f, 4.f, 504);
  for (uint32_t pos : {1u, 2050u, 2051u, 2400u}) {
    const qr::Selection sel = qr::qsa_select(scores.data(), pos);
    std::vector<uint32_t> row(kListRow, 0);
    std::copy(sel.list.begin(), sel.list.end(), row.begin());
    row[kCountW] = sel.count();
    clrun::Buffer cb(dev, ctrl_at(pos)), lb(dev, row), ob(dev, size_t(QN) * 2);
    p.run("q4_qsa_attn_eager", {qr::kQHeads * 256, 1}, {256, 1}, cb, qb, gb, kb, vb, lb, ob);
    const std::vector<uint16_t> o = ob.read<uint16_t>();
    Diff d;
    for (uint32_t h = 0; h < qr::kQHeads; ++h) {
      const qr::EagerHead e = qr::qsa_attn_eager(q.data(), gate.data(), k.data(), v.data(), sel.list.data(), sel.count(), h);
      for (uint32_t dd = 0; dd < qr::kHd; ++dd) d.add16(h * qr::kHd + dd, o[h * qr::kHd + dd], e.out[dd]);
    }
    report("q4_qsa_attn_eager_M1 at p " + std::to_string(pos) + " (" + std::to_string(sel.count()) + " listed)", d, 2);
  }
}

// ---- q4_moe.cl -----------------------------------------------------------------------------------------------------
void route_checks(clrun::Device& dev) {
  clrun::Program p(dev, "src/kernels/qwen4exp/q4_moe.cl", {"M=1", "ROUTE_E=512", "ROUTE_K=10", "ROUTE_WG=256", "ROUTE_LN=528"});
  Diff ids, wts, sgs, pp;
  const auto one = [&](const std::vector<float>& lg, size_t tag) {
    clrun::Buffer lb(dev, lg), rb(dev, 32 * 4);
    p.run("q4_route", {256, 1}, {256, 1}, lb, rb);
    const std::vector<uint32_t> got = rb.read<uint32_t>();
    const qr::Route want = qr::route(lg.data());
    const std::vector<uint32_t> w = qr::route_row(want);
    for (uint32_t k = 0; k < 10; ++k) {
      ids.addx(tag * 16 + k, got[qr::kIds + k], w[qr::kIds + k]);
      wts.add16(tag * 10 + k, rne(qr::as_f32(got[qr::kWeights + k])), rne(want.w[k]));
    }
    sgs.add16(tag, rne(qr::as_f32(got[qr::kSharedGate])), rne(want.sg));
    pp.add32(tag * 2, qr::as_f32(got[qr::kP10]), want.p10);
    pp.add32(tag * 2 + 1, qr::as_f32(got[qr::kP11]), want.p11);
    for (uint32_t k = 10; k < 16; ++k) ids.addx(tag * 16 + k, got[k], 0);
    return got;
  };
  size_t tag = 0;
  for (uint32_t s = 0; s < 8; ++s) {
    std::vector<float> lg = random_f32(qr::kRouterN, -3.f, 3.f, 600 + s);
    for (uint32_t e = qr::kExperts + 1; e < qr::kRouterN; ++e) lg[e] = 0.0f;
    one(lg, tag++);
  }
  const std::vector<uint32_t> L = qr::hex32(fx::kRouteLogits);
  bool fix = true;
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> lg(qr::kRouterN);
    for (uint32_t e = 0; e < qr::kRouterN; ++e) lg[e] = qr::as_f32(L[size_t(r) * qr::kRouterN + e]);
    const std::vector<uint32_t> got = one(lg, tag++);
    for (uint32_t k = 0; k < 10; ++k) fix = fix && got[k] == fx::kRouteRuledIds[r * 10 + k];
  }
  // crafted: nine winners and an exact tie (300, 7) for the 10th - 7 goes in
  bool tie;
  {
    std::vector<float> lg(qr::kRouterN, -4.0f);
    const uint32_t win[9] = {500, 3, 250, 128, 77, 400, 9, 311, 64};
    for (uint32_t j = 0; j < 9; ++j) lg[win[j]] = 2.0f + 0.125f * float(j);
    lg[300] = lg[7] = 1.0f;
    lg[qr::kExperts] = 0.5f;
    const std::vector<uint32_t> got = one(lg, tag++);
    tie = got[9] == 7;
    for (uint32_t j = 0; j < 9; ++j) tie = tie && got[j] == win[8 - j];
  }
  report("q4_route_M1_E512_T10_N528_L256 ids (8 random, the fixture's 4 rows, a tie for the 10th)", ids, 0, "mismatch");
  report_bool("q4_route_M1_E512_T10_N528_L256 ids = the fixture's ruled ids (incl. its tie (99, 431) and the -1e30 row)", fix);
  report_bool("q4_route_M1_E512_T10_N528_L256 crafted tie (300, 7) at the 10th: the lower id", tie);
  report("q4_route_M1_E512_T10_N528_L256 weights", wts, 2);
  report("q4_route_M1_E512_T10_N528_L256 shared gate", sgs, 2);
  report("q4_route_M1_E512_T10_N528_L256 p10 / p11 diagnostics", pp, 8, "fp32 ulp");
}

void moe_checks(clrun::Device& dev, uint32_t dn_ks) {
  const uint32_t kE = 16;
  const size_t gub = qr::gate_up_block_words(), dnb = qr::down_block_words();
  const std::vector<uint32_t> gu = random_blocks(gub * kE, 700), dn = random_blocks(dnb * kE, 701);
  const std::vector<uint32_t> sh_gu4 = random_blocks(gub, 702), sh_dn4 = random_blocks(dnb, 703);
  const std::vector<uint16_t> sh_gub = random_bf16(size_t(D) * 2 * qr::kInter, -0.03f, 0.03f, 704);
  const std::vector<uint16_t> sh_dnb = random_bf16(size_t(qr::kInter) * D, -0.05f, 0.05f, 705);
  const std::vector<uint16_t> x = random_bf16(D, -0.5f, 0.5f, 706);
  const uint32_t ids[10] = {3, 15, 0, 7, 12, 1, 9, 14, 5, 10};
  const float w_all[10] = {rf(0.21875f), rf(0.15625f), rf(0.125f), rf(0.109375f), rf(0.09375f),
                           rf(0.0859375f), rf(0.078125f), rf(0.0625f), rf(0.046875f), rf(0.0390625f)};
  const float sg_all = rf(0.40625f);
  clrun::Buffer xb(dev, x), gub_b(dev, gu), dnb_b(dev, dn);
  for (int shb = 0; shb <= 1; ++shb) {
    clrun::Program p(dev, "src/kernels/qwen4exp/q4_moe.cl",
                     {"M=1", "MOE_E=512", "MOE_K=10", "HIDDEN=2560", "INTER=640", "UP_KS=4", "DN_KS=" + std::to_string(dn_ks),
                      shb ? "SHARED_BF16=1" : "SHARED_BF16=0"});
    std::unique_ptr<clrun::Buffer> sgu, sdn;
    if (shb) {
      sgu.reset(new clrun::Buffer(dev, sh_gub));
      sdn.reset(new clrun::Buffer(dev, sh_dnb));
    } else {
      sgu.reset(new clrun::Buffer(dev, sh_gu4));
      sdn.reset(new clrun::Buffer(dev, sh_dn4));
    }
    const std::string form = shb ? "_SHB" : "_SH4";
    const std::string nm = "q4_moe_M1_E512_T10_D2560_I640" + form;
    // the cases: every slot; routed slot s alone (the others' weights and the shared gate zero); the shared slot alone
    for (int c = -1; c <= 10; ++c) {
      float w[10], sg;
      for (uint32_t k = 0; k < 10; ++k) w[k] = (c < 0 || c == int(k)) ? w_all[k] : 0.0f;
      sg = (c < 0 || c == 10) ? sg_all : 0.0f;
      std::vector<uint32_t> row(32, 0);
      for (uint32_t k = 0; k < 10; ++k) {
        row[qr::kIds + k] = ids[k];
        row[qr::kWeights + k] = qr::as_u32(w[k]);
      }
      row[qr::kSharedGate] = qr::as_u32(sg);
      clrun::Buffer rb(dev, row), hb(dev, size_t(qr::kSlots) * qr::kInter * 2), yb(dev, size_t(D) * 2);
      p.run("q4_moe_gate_up", {qr::kSlots * (2 * qr::kInter / 16 / 4) * 256, 1}, {256, 1}, rb, xb, gub_b, *sgu, hb);
      const size_t wg_dn = 16 * qr::kSlots * dn_ks;
      p.run("q4_moe_down", {(D / 16) * wg_dn, 1}, {wg_dn, 1}, rb, hb, dnb_b, *sdn, yb);
      const std::vector<uint16_t> h = hb.read<uint16_t>(), y = yb.read<uint16_t>();
      const std::vector<uint16_t> y_ref =
          qr::down(ids, w, sg, h, dn.data(), shb ? nullptr : sh_dn4.data(), shb ? sh_dnb.data() : nullptr, dn_ks);
      Diff dy;
      for (uint32_t n = 0; n < D; ++n) dy.add16(n, y[n], y_ref[n]);
      if (c < 0) {
        const std::vector<uint16_t> h_ref =
            qr::gate_up(ids, x.data(), gu.data(), shb ? nullptr : sh_gu4.data(), shb ? sh_gub.data() : nullptr, 4);
        Diff dh;
        for (size_t i = 0; i < h.size(); ++i) dh.add16(i, h[i], h_ref[i]);
        report(nm + " gate_up h [11][640]", dh, 2);
      }
      const std::string what = c < 0 ? "all 11 slots" : c == 10 ? "the shared slot alone" : "routed slot " + std::to_string(c) + " alone (id " + std::to_string(ids[c]) + ")";
      report(nm + " down + combine (from the device's h), " + what, dy, 0);
    }
  }
}

// ---- prep.cl prep_gated_head _SIG -----------------------------------------------------------------------------------
void gated_checks(clrun::Device& dev) {
  clrun::Program p(dev, "src/kernels/prep.cl", {"M=1", "GDN_GATE_SIGMOID=1"});
  const uint32_t ON = qr::kGdnHeads * qr::kGdnHd;
  const auto run = [&](const std::vector<float>& qkvz, const std::vector<float>& o, const std::vector<uint16_t>& gw) {
    clrun::Buffer zb(dev, qkvz), ob(dev, o), wb(dev, gw), xb(dev, size_t(ON) * 2);
    p.run("prep_gated_head", {qr::kGdnHeads * 128, 1}, {128, 1}, zb, ob, wb, xb);
    return xb.read<uint16_t>();
  };
  {
    const std::vector<float> qkvz = random_f32(qr::kQkvzN, -6.f, 6.f, 800), o = random_f32(ON, -2.f, 2.f, 801);
    const std::vector<uint16_t> gw = random_bf16(128, 0.5f, 1.5f, 802);
    const std::vector<uint16_t> got = run(qkvz, o, gw);
    std::vector<uint16_t> want(ON);
    qr::gated_head_sig(qkvz.data(), o.data(), gw.data(), want.data());
    Diff d;
    for (uint32_t i = 0; i < ON; ++i) d.add16(i, got[i], want[i]);
    report("prep_gated_head_M1_SIG random", d, 2);
  }
  {   // the fixture's case (torch's own output)
    const std::vector<uint16_t> xo = qr::fixture_bf16(true, 60, ON, 7, 0.25f), z = qr::fixture_bf16(false, 61, ON, 7, 0.125f, 6.0f),
                                gw = qr::fixture_bf16(false, 62, 128, 7, 0.125f, 1.5f), want = qr::hex16(fx::kGated);
    std::vector<float> o(ON), qkvz(qr::kQkvzN, 0.0f);
    for (uint32_t i = 0; i < ON; ++i) {
      o[i] = f32(xo[i]);
      qkvz[qr::kZOff + i] = f32(z[i]);
    }
    const std::vector<uint16_t> got = run(qkvz, o, gw);
    Diff d;
    for (uint32_t i = 0; i < ON; ++i) d.add16(i, got[i], want[i]);
    report("prep_gated_head_M1_SIG the fixture's case (torch)", d, 2);
  }
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    size_t cap = 0;
    clGetDeviceInfo(dev.id(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof cap, &cap, nullptr);
    // SEL_WG: the largest power of two <= min(cap, 1024), >= 256 (the 256-bin histogram); DN_KS 2 is 352 lanes.
    g_sel_wg = 1024;
    while (g_sel_wg > 256 && g_sel_wg > cap) g_sel_wg /= 2;
    const uint32_t dn_ks = cap >= 16 * qr::kSlots * 2 ? 2 : 1;
    std::printf("qwen4exp_run: device %s, CL_DEVICE_MAX_WORK_GROUP_SIZE %zu -> q4_qsa SEL_WG=%u%s, q4_moe DN_KS=%u%s\n",
                dev.name().c_str(), cap, g_sel_wg, g_sel_wg == 1024 ? " (the B70 build's)" : " (the B70 build: 1024)", dn_ks,
                dn_ks == 2 ? " (the B70 build's)" : " (16 x 11 = 176 lanes; the B70 build: 2, 352 lanes; host twin at dn_ks 1)");
    hc_checks(dev);
    ple_checks(dev);
    qsa_checks(dev);
    eager_checks(dev);
    route_checks(dev);
    moe_checks(dev, dn_ks);
    gated_checks(dev);
  } catch (const std::exception& e) {
    std::printf("qwen4exp_run: ERROR %s\n", e.what());
    return 1;
  }
  std::printf("qwen4exp_run: %s (%d disagreement%s)\n", failures ? "FAIL" : "OK", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
