// PERF-27: Qwen3.8-Flash-Next's MoE prompt-pass glue (Ornith's, merged 41f2a51a) against the forms it replaced, on the
// card with synthetic data (no checkpoint):
//   1. route    the prefill's q4_route (_P768, ROUTE_SG: one sub-group a token) against decode's binary (the 256-lane
//               form) on the same logits - every word of every route row (ids, order, weights, the shared gate, the
//               two diagnostics) at C 2048 and 37; the rows: random, four-level logits (128-way ties), all equal,
//               fifteen equal maxima across lanes and registers (the rank-9 / rank-10 boundary inside a tie), three
//               live experts (the other slots p = 0 ties); the pad columns past the router's 528 are NaN
//   2. gather   q4_pf_gather_routed + the _SHX gate||up (the shared expert's rows from x) against q4_pf_gather + the
//               plain binary, after pf_moe_sort_mw over the routes of 1: every routed row of xg (the shared rows of
//               the new xg stay poisoned) and every row of h up to the shared expert's last token - one card's 2560 x
//               1280 and a tp card's 2560 x 640, direct and _TBL (ids divisible by 5 in pinned host memory)
//   3. combine  q4_pf_moe_combine (16-byte pieces) and q4_pf_moe_combine_f32 (a tp card's) against the one-column
//               builds (_C1): every output word
//   4. plan     PERF-25: the direct expert GEMMs (part 2's _SHX gate||up, one card's bf16 down 640 x 2560 and a tp
//               card's fp32 down 320 x 2560; the _TBL ones keep the old plan) on the spill-free register plan against
//               their _DB twins (the register double buffer they replaced) on the same inputs: every output word
// The builds' spill sizes are printed (spill 0 required of the new route, gather and combines and of the four direct
// expert GEMMs; the plain gate||up, the _TBL ones and the _DB twins spill). `--time` adds each kernel's
// time old against new at C 2048 (interleaved pairs after a warm-up round).
// Needs a B70; exit 77 (SKIP) without a GPU.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {
namespace qk = kernels::qwen4exp;
constexpr uint32_t kD = qk::kHidden, kE = qk::kExperts, kK = qk::kTopK, kRw = qk::route::kWords, kC = qk::kPfC;
constexpr uint32_t kTm = qk::kPfTm, kMi = kernels::pf_moe::kItemM, kLd = qk::kPfRouterPitch, kRn = qk::kRouterN;
constexpr uint32_t kPool = 32, kHostPool = 8;   // distinct VRAM / host blocks behind the 512 ids

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  template <class T>
  l0::Mem up(const std::vector<T>& v, l0::MemKind kind = l0::MemKind::Device) {
    l0::Mem m(ctx, kind, v.size() * sizeof(T));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem filled(size_t bytes, uint32_t pattern) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), pattern, bytes);
    return m;
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n) {
    std::vector<T> v(n);
    imm.copy(v.data(), m.ptr(), n * sizeof(T));
    return v;
  }
  // `add` appends one launch; run it once and wait.
  void run(const std::function<void(l0::CmdList&)>& add) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    add(list);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
  // The median over `reps` executions of a list of 20 back-to-back launches, in us a launch.
  double time(const std::function<void(l0::CmdList&)>& add, int reps = 9) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (int i = 0; i < 20; ++i) add(list);
    list.close();
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      q.execute(list, &f);
      f.wait();
      t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  }
};

// One kernel of one binary, its group size set.
struct K {
  l0::Module mod;
  l0::Kernel k;
  K(Dev& d, const std::string& variant, const char* entry, uint32_t wg)
      : mod(d.ctx, kernels::path(variant)), k(mod, entry) {
    k.group_size(wg);
  }
  uint32_t spill() const {
    ze_kernel_properties_t pr{};
    pr.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    zeKernelGetProperties(k.handle(), &pr);
    return pr.spillMemSize;
  }
  uint32_t priv() const {
    ze_kernel_properties_t pr{};
    pr.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    zeKernelGetProperties(k.handle(), &pr);
    return pr.privateMemSize;
  }
};

// Old against new: interleaved pairs, the first round a warm-up.
void time_pair(Dev& d, const char* what, const std::function<void(l0::CmdList&)>& a,
               const std::function<void(l0::CmdList&)>& b) {
  double ta = 0, tb = 0;
  for (int r = 0; r < 3; ++r) {
    const double x = d.time(a), y = d.time(b);
    if (r > 0) {
      ta += x / 2;
      tb += y / 2;
    }
  }
  std::printf("  time %-44s old %8.1f us, new %8.1f us (x%.3f)\n", what, ta, tb, tb / ta);
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
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}
uint32_t tmax_of(uint32_t C) { return (C * kK + kE * (kTm - 1)) / kTm + (C + kTm - 1) / kTm; }
uint32_t items_of(uint32_t C) { return (C * kK + kE * (kMi - 1)) / kMi + (C + kMi - 1) / kMi; }

// ---- 1. the route ------------------------------------------------------------------------------------------------
// Logits [kC][kLd] (the router GEMM's pitch: columns 0..511 the experts, 512 the shared gate, 513..527 zero, the rest
// NaN) by row kind m % 8.
std::vector<float> logits_rows() {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-4.f, 4.f);
  std::vector<float> lg(size_t(kC) * kLd, std::numeric_limits<float>::quiet_NaN());
  for (uint32_t m = 0; m < kC; ++m) {
    float* r = lg.data() + size_t(m) * kLd;
    for (uint32_t e = 0; e < kE; ++e) r[e] = u(rng);
    switch (m % 8) {
      case 4:   // four levels: 128-way ties
        for (uint32_t e = 0; e < kE; ++e) r[e] = 0.5f * float(rng() % 4);
        break;
      case 5:   // all equal: ids 0..9
        for (uint32_t e = 0; e < kE; ++e) r[e] = 1.0f;
        break;
      case 6: {   // fifteen equal maxima over lanes and registers, a rotating set
        const uint32_t s = m % 13;
        for (uint32_t i = 0; i < 15; ++i) r[(s + 37 * i + 16 * (i % 3)) % kE] = 6.0f;
        break;
      }
      case 7:   // three live experts: slots 3..10 are p = 0 ties
        for (uint32_t e = 0; e < kE; ++e) r[e] = -200.0f;
        r[(m * 7) % kE] = 5.0f;
        r[(m * 7 + 100) % kE] = 4.0f;
        r[(m * 7 + 301) % kE] = 5.0f;
        break;
      default:
        break;
    }
    r[kE] = u(rng);
    for (uint32_t n = kE + 1; n < kRn; ++n) r[n] = 0.0f;
  }
  return lg;
}

std::vector<uint32_t> route(Dev& d, bool timing) {
  const std::vector<float> lg = logits_rows();
  std::vector<float> compact(size_t(kC) * kRn);
  for (uint32_t m = 0; m < kC; ++m) std::copy_n(lg.begin() + size_t(m) * kLd, kRn, compact.begin() + size_t(m) * kRn);
  l0::Mem lp = d.up(lg), lc = d.up(compact), ra = d.filled(size_t(kC) * kRw * 4, 0xDEADBEEFu),
          rb = d.filled(size_t(kC) * kRw * 4, 0xDEADBEEFu);
  K nw(d, qk::pf_route_gemm_variant(), "q4_route", 16), old(d, qk::route_variant(1), "q4_route", qk::kRouteLanes);
  nw.k.arg_ptr(0, lp.ptr());
  nw.k.arg_ptr(1, ra.ptr());
  old.k.arg_ptr(0, lc.ptr());
  old.k.arg_ptr(1, rb.ptr());
  for (uint32_t C : {37u, kC}) {
    d.run([&](l0::CmdList& l) { l.launch(nw.k, 1, C); });
    d.run([&](l0::CmdList& l) { l.launch(old.k, 1, C); });
    const std::vector<uint32_t> a = d.read<uint32_t>(ra, size_t(C) * kRw), b = d.read<uint32_t>(rb, size_t(C) * kRw);
    size_t bad = 0, kinds[8] = {};
    for (uint32_t m = 0; m < C; ++m)
      if (std::memcmp(&a[size_t(m) * kRw], &b[size_t(m) * kRw], kRw * 4) != 0) {
        ++bad;
        ++kinds[m % 8];
      }
    std::printf("1. route C %u: %zu of %u rows differ from decode's (by kind 0..7: %zu %zu %zu %zu %zu %zu %zu %zu)\n", C,
                bad, C, kinds[0], kinds[1], kinds[2], kinds[3], kinds[4], kinds[5], kinds[6], kinds[7]);
    CHECK_EQ(bad, size_t(0));
    // not trivially: the all-equal row took ids 0..9, the weights are written
    for (uint32_t k = 0; k < kK; ++k) CHECK_EQ(a[size_t(5) * kRw + qk::route::kIds + k], k);
    CHECK(a[qk::route::kWeights] != 0u);
  }
  std::printf("   q4_route _P768 (one sub-group): spill %u B, private %u B\n", nw.spill(), nw.priv());
  CHECK_EQ(nw.spill(), 0u);
  if (timing)
    time_pair(d, "q4_route C 2048 (256 lanes -> one sub-group)", [&](l0::CmdList& l) { l.launch(old.k, 1, kC); },
              [&](l0::CmdList& l) { l.launch(nw.k, 1, kC); });
  return d.read<uint32_t>(ra, size_t(kC) * kRw);
}

// ---- 2. the gather and the gate||up ------------------------------------------------------------------------------
// One shape's expert weights (K x N): the 512 ids' blocks direct (every id its own copy of a pool block) and as the
// table row (u64 [512]; ids divisible by 5 in pinned host memory), the shared expert's int4 block.
struct Weights {
  size_t blk;   // u32 words of one expert's block
  l0::Mem all, pool, host, table, shared;
  Weights(Dev& d, uint32_t K, uint32_t N, uint32_t seed)
      : blk(size_t(N / 16) * (K / 64) * 136),
        all(d.ctx, l0::MemKind::Device, blk * kE * 4),
        pool(d.up(random_blocks(blk * kPool, seed))),
        host(d.up(random_blocks(blk * kHostPool, seed + 1), l0::MemKind::Host)),
        table(d.ctx, l0::MemKind::Device, kE * 8),
        shared(d.up(random_blocks(blk, seed + 2))) {
    std::vector<uint64_t> row(kE);
    const auto at = [](const l0::Mem& m, size_t words) { return static_cast<const uint8_t*>(m.ptr()) + words * 4; };
    for (uint32_t e = 0; e < kE; ++e) {
      const uint8_t* g = e % 5 == 0 ? at(host, blk * (e / 5 % kHostPool)) : at(pool, blk * (e % kPool));
      row[e] = reinterpret_cast<uint64_t>(g);
      d.imm.copy(const_cast<uint8_t*>(at(all, blk * e)), g, blk * 4);
    }
    d.imm.copy(table.ptr(), row.data(), row.size() * 8);
  }
  const void* w(bool tbl) const { return tbl ? table.ptr() : all.ptr(); }
};

struct Sorted {
  l0::Mem hdr, tiles, row_tok, pair_row, items;
  std::vector<uint32_t> h;   // the header
};
Sorted sort(Dev& d, const l0::Mem& rt, uint32_t C) {
  const uint32_t T = tmax_of(C), ib = items_of(C), nblk = (C + 63) / 64;
  Sorted s{d.filled(qk::pf_hdr::words() * 4, 0u), d.filled(size_t(T) * 8, 0u), d.filled(size_t(T) * kTm * 4, 0u),
           d.filled(size_t(C) * kK * 4, 0u), d.filled(size_t(ib) * 16, 0u), {}};
  l0::Mem cnt = d.filled(size_t(nblk) * kE * 4, 0u);
  const std::string sv = kernels::pf_moe_sort_mw_variant(kE, kK);
  K hist(d, sv, "pf_moe_hist", 256), scan(d, sv, "pf_moe_scan", 256), scat(d, sv, "pf_moe_scatter", 256);
  hist.k.arg_ptr(0, rt.ptr());
  hist.k.arg_ptr(1, cnt.ptr());
  hist.k.arg(2, C);
  const void* sa[5] = {cnt.ptr(), s.hdr.ptr(), s.tiles.ptr(), s.row_tok.ptr(), s.items.ptr()};
  for (uint32_t i = 0; i < 5; ++i) scan.k.arg_ptr(i, sa[i]);
  scan.k.arg(5, C);
  scan.k.arg(6, T);
  scan.k.arg(7, ib);
  const void* ca[4] = {rt.ptr(), cnt.ptr(), s.row_tok.ptr(), s.pair_row.ptr()};
  for (uint32_t i = 0; i < 4; ++i) scat.k.arg_ptr(i, ca[i]);
  scat.k.arg(4, C);
  d.run([&](l0::CmdList& l) {
    l.launch(hist.k, nblk);
    l.launch(scan.k, 1);
    l.launch(scat.k, nblk);
  });
  s.h = d.read<uint32_t>(s.hdr, qk::pf_hdr::words());
  return s;
}

void gather_gate_up(Dev& d, const Sorted& s, const l0::Mem& x, bool timing) {
  const uint32_t C = kC, T = tmax_of(C), R = T * kTm, ib = items_of(C), Rs = s.h[qk::pf_hdr::kSharedRow];
  CHECK_EQ(s.h[qk::pf_hdr::kC], C);
  const std::string pv = qk::pf_moe_variant();
  K g_old(d, pv, "q4_pf_gather", qk::kPfGatherWg), g_new(d, pv, "q4_pf_gather_routed", qk::kPfGatherWg);
  l0::Mem xa = d.filled(size_t(R) * kD * 2, 0u), xb = d.filled(size_t(R) * kD * 2, 0x7FC07FC0u);
  for (K* g : {&g_old, &g_new}) {
    g->k.arg_ptr(0, x.ptr());
    g->k.arg_ptr(1, s.hdr.ptr());
    g->k.arg_ptr(2, s.row_tok.ptr());
  }
  g_old.k.arg_ptr(3, xa.ptr());
  g_new.k.arg_ptr(3, xb.ptr());
  d.run([&](l0::CmdList& l) { l.launch(g_old.k, R); });
  d.run([&](l0::CmdList& l) { l.launch(g_new.k, R); });
  {
    const std::vector<uint16_t> a = d.read<uint16_t>(xa, size_t(R) * kD), b = d.read<uint16_t>(xb, size_t(R) * kD);
    CHECK(std::memcmp(a.data(), b.data(), size_t(Rs) * kD * 2) == 0);
    size_t poisoned = 0;
    for (size_t i = size_t(Rs) * kD; i < size_t(Rs + C) * kD; ++i) poisoned += b[i] == 0x7FC0;
    std::printf("2. gather C %u: the routed rows 0..%u bitwise q4_pf_gather's, the shared rows untouched (%s)\n", C, Rs - 1,
                poisoned == size_t(C) * kD ? "poison kept" : "WRITTEN");
    CHECK_EQ(poisoned, size_t(C) * kD);
  }
  for (bool tp : {false, true}) {
    const uint32_t inter = tp ? qk::kTpInter : qk::kInter, NG = 2 * inter, wgn = tp ? qk::kTpPfGuWgN : kernels::pf_moe::kQ4WgN;
    const Weights W(d, kD, NG, tp ? 300 : 200);
    for (bool tbl : {false, true}) {
      const std::string vo = kernels::pf_moe_q4gemm_variant(kD, NG, true, tbl);
      const std::string vn = tp ? qk::tp_pf_moe_gu_variant(tbl) : qk::pf_moe_gu_variant(tbl);
      // the work-group: 16 x WGN lanes, WGN = the columns a work-group / 32 (NB 2 atoms of 16 columns a sub-group)
      K o(d, vo, "pf_moe_q4gemm", wgn / 2), n(d, vn, "pf_moe_q4gemm", wgn / 2);
      if (tbl) {
        o.k.indirect_host();
        n.k.indirect_host();
      }
      l0::Mem ha = d.filled(size_t(R) * inter * 2, 0u), hb = d.filled(size_t(R) * inter * 2, 0u);
      const void* oa[5] = {s.items.ptr(), xa.ptr(), W.w(tbl), W.shared.ptr(), ha.ptr()};
      const void* na[7] = {s.items.ptr(), xb.ptr(), W.w(tbl), W.shared.ptr(), hb.ptr(), x.ptr(), s.hdr.ptr()};
      for (uint32_t i = 0; i < 5; ++i) o.k.arg_ptr(i, oa[i]);
      for (uint32_t i = 0; i < 7; ++i) n.k.arg_ptr(i, na[i]);
      d.run([&](l0::CmdList& l) { l.launch(o.k, NG / wgn, ib); });
      d.run([&](l0::CmdList& l) { l.launch(n.k, NG / wgn, ib); });
      const std::vector<uint16_t> a = d.read<uint16_t>(ha, size_t(Rs + C) * inter),
                                  b = d.read<uint16_t>(hb, size_t(Rs + C) * inter);
      size_t bad = 0;
      for (size_t r = 0; r < size_t(Rs) + C; ++r) bad += std::memcmp(&a[r * inter], &b[r * inter], inter * 2) != 0;
      const bool written = std::any_of(a.begin() + size_t(Rs) * inter, a.end(), [](uint16_t v) { return v != 0; });
      std::printf("   %s: %zu of %u rows (routed + the shared expert's %u tokens) differ from %s; spill %u B (old %u B)\n",
                  vn.c_str(), bad, Rs + C, C, vo.c_str(), n.spill(), o.spill());
      CHECK_EQ(bad, size_t(0));
      CHECK(written);
      // (the plain binaries spill - 256 GRF, IGC's retry; the _SHX ones are timed against them below)
      const std::string card = std::string(tp ? "tp card" : "one card") + (tbl ? " _TBL" : "");
      if (timing)
        time_pair(d, (card + " gate||up (_SHX)").c_str(), [&](l0::CmdList& l) { l.launch(o.k, NG / wgn, ib); },
                  [&](l0::CmdList& l) { l.launch(n.k, NG / wgn, ib); });
      // 4. the register plan (direct only): the _SHX binary against its _DB twin, the same arguments
      if (!tbl) {
        K db(d, vn + "_DB", "pf_moe_q4gemm", wgn / 2);
        l0::Mem hc = d.filled(size_t(R) * inter * 2, 0u);
        for (uint32_t i = 0; i < 7; ++i) db.k.arg_ptr(i, i == 4 ? hc.ptr() : na[i]);
        d.run([&](l0::CmdList& l) { l.launch(db.k, NG / wgn, ib); });
        const std::vector<uint16_t> c = d.read<uint16_t>(hc, size_t(R) * inter),
                                    bn = d.read<uint16_t>(hb, size_t(R) * inter);
        size_t bad_db = 0;
        for (size_t i = 0; i < c.size(); ++i) bad_db += c[i] != bn[i];
        std::printf("4. %s: %zu of %zu words differ from the _DB twin; spill %u B (_DB %u B)\n", vn.c_str(), bad_db,
                    c.size(), n.spill(), db.spill());
        CHECK_EQ(bad_db, size_t(0));
        CHECK_EQ(n.spill(), 0u);
        if (timing)
          time_pair(d, (card + " gate||up register plan").c_str(),
                    [&](l0::CmdList& l) { l.launch(db.k, NG / wgn, ib); },
                    [&](l0::CmdList& l) { l.launch(n.k, NG / wgn, ib); });
      }
      if (timing && !tbl) {
        const std::string what = card + " gather + gate||up";
        time_pair(d, what.c_str(),
                  [&](l0::CmdList& l) {
                    l.launch(g_old.k, R);
                    l.launch(o.k, NG / wgn, ib);
                  },
                  [&](l0::CmdList& l) {
                    l.launch(g_new.k, R);
                    l.launch(n.k, NG / wgn, ib);
                  });
      }
    }
  }
  std::printf("   q4_pf_gather_routed: spill %u B\n", g_new.spill());
  CHECK_EQ(g_new.spill(), 0u);
  if (timing)
    time_pair(d, "gather C 2048 (all rows -> routed rows)", [&](l0::CmdList& l) { l.launch(g_old.k, R); },
              [&](l0::CmdList& l) { l.launch(g_new.k, R); });
}

// ---- 3. the combines ---------------------------------------------------------------------------------------------
void combines(Dev& d, const l0::Mem& rt, const Sorted& s, bool timing) {
  const uint32_t C = kC, R = tmax_of(C) * kTm;
  {
    const l0::Mem y = d.up(random_bf16(size_t(R) * kD, -1.f, 1.f, 400));
    l0::Mem oa = d.filled(size_t(C) * kD * 2, 0u), ob = d.filled(size_t(C) * kD * 2, 0u);
    K o(d, qk::pf_moe_variant() + "_C1", "q4_pf_moe_combine", 256), n(d, qk::pf_moe_variant(), "q4_pf_moe_combine", 64);
    for (K* k : {&o, &n}) {
      k->k.arg_ptr(0, rt.ptr());
      k->k.arg_ptr(1, s.hdr.ptr());
      k->k.arg_ptr(2, s.pair_row.ptr());
      k->k.arg_ptr(3, y.ptr());
      k->k.arg(5, C);
    }
    o.k.arg_ptr(4, oa.ptr());
    n.k.arg_ptr(4, ob.ptr());
    d.run([&](l0::CmdList& l) { l.launch(o.k, kD / 256, C); });
    d.run([&](l0::CmdList& l) { l.launch(n.k, kD / qk::kPfCombineCols, C); });
    const std::vector<uint16_t> a = d.read<uint16_t>(oa, size_t(C) * kD), b = d.read<uint16_t>(ob, size_t(C) * kD);
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
    std::printf("3. q4_pf_moe_combine C %u: %zu of %zu words differ from the one-column form; spill %u B\n", C, bad,
                a.size(), n.spill());
    CHECK_EQ(bad, size_t(0));
    CHECK(std::any_of(a.begin(), a.end(), [](uint16_t v) { return v != 0; }));
    CHECK_EQ(n.spill(), 0u);
    if (timing)
      time_pair(d, "q4_pf_moe_combine C 2048", [&](l0::CmdList& l) { l.launch(o.k, kD / 256, C); },
                [&](l0::CmdList& l) { l.launch(n.k, kD / qk::kPfCombineCols, C); });
  }
  {
    std::vector<float> yf(size_t(R) * kD);
    std::mt19937 rng(500);
    std::uniform_real_distribution<float> u(-3.f, 3.f);
    for (float& v : yf) v = u(rng);
    const l0::Mem y = d.up(yf);
    l0::Mem oa = d.filled(size_t(C) * kD * 4, 0u), ob = d.filled(size_t(C) * kD * 4, 0u);
    K o(d, qk::tp_pf_combine_variant() + "_C1", "q4_pf_moe_combine_f32", 256),
        n(d, qk::tp_pf_combine_variant(), "q4_pf_moe_combine_f32", 64);
    for (K* k : {&o, &n}) {
      k->k.arg_ptr(0, rt.ptr());
      k->k.arg_ptr(1, s.hdr.ptr());
      k->k.arg_ptr(2, s.pair_row.ptr());
      k->k.arg_ptr(3, y.ptr());
      k->k.arg(5, C);
    }
    o.k.arg_ptr(4, oa.ptr());
    n.k.arg_ptr(4, ob.ptr());
    d.run([&](l0::CmdList& l) { l.launch(o.k, kD / 256, C); });
    d.run([&](l0::CmdList& l) { l.launch(n.k, kD / qk::kPfCombineF32Cols, C); });
    const std::vector<uint32_t> a = d.read<uint32_t>(oa, size_t(C) * kD), b = d.read<uint32_t>(ob, size_t(C) * kD);
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
    std::printf("   q4_pf_moe_combine_f32 C %u: %zu of %zu words differ from the one-column form; spill %u B\n", C, bad,
                a.size(), n.spill());
    CHECK_EQ(bad, size_t(0));
    CHECK(std::any_of(a.begin(), a.end(), [](uint32_t v) { return v != 0; }));
    CHECK_EQ(n.spill(), 0u);
    if (timing)
      time_pair(d, "q4_pf_moe_combine_f32 C 2048 (tp card)", [&](l0::CmdList& l) { l.launch(o.k, kD / 256, C); },
                [&](l0::CmdList& l) { l.launch(n.k, kD / qk::kPfCombineF32Cols, C); });
  }
}

// ---- 4. the downs on the register plan -----------------------------------------------------------------------------
// One card's bf16 down 640 x 2560 and a tp card's fp32 down 320 x 2560 (direct) against their _DB twins over the same
// h rows (random bf16; every sorted row of the chunk) - every output word.
void downs(Dev& d, const Sorted& s, bool timing) {
  const uint32_t C = kC, R = tmax_of(C) * kTm, ib = items_of(C), wgn = kernels::pf_moe::kQ4WgN;
  for (bool tp : {false, true}) {
    const uint32_t inter = tp ? qk::kTpInter : qk::kInter, ob = tp ? 4 : 2;
    const Weights W(d, inter, kD, tp ? 310 : 210);
    const l0::Mem h = d.up(random_bf16(size_t(R) * inter, -1.f, 1.f, tp ? 710 : 700));
    {
      const std::string vn = tp ? qk::tp_pf_moe_dn_variant(false) : kernels::pf_moe_q4gemm_variant(inter, kD, false);
      K o(d, vn + "_DB", "pf_moe_q4gemm", wgn / 2), n(d, vn, "pf_moe_q4gemm", wgn / 2);
      l0::Mem ya = d.filled(size_t(R) * kD * ob, 0u), yb = d.filled(size_t(R) * kD * ob, 0u);
      const void* args[4] = {s.items.ptr(), h.ptr(), W.w(false), W.shared.ptr()};
      for (uint32_t i = 0; i < 4; ++i) {
        o.k.arg_ptr(i, args[i]);
        n.k.arg_ptr(i, args[i]);
      }
      o.k.arg_ptr(4, ya.ptr());
      n.k.arg_ptr(4, yb.ptr());
      d.run([&](l0::CmdList& l) { l.launch(o.k, kD / wgn, ib); });
      d.run([&](l0::CmdList& l) { l.launch(n.k, kD / wgn, ib); });
      const std::vector<uint32_t> a = d.read<uint32_t>(ya, size_t(R) * kD * ob / 4),
                                  b = d.read<uint32_t>(yb, size_t(R) * kD * ob / 4);
      size_t bad = 0;
      for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
      std::printf("4. %s: %zu of %zu words differ from the _DB twin; spill %u B (_DB %u B)\n", vn.c_str(), bad, a.size(),
                  n.spill(), o.spill());
      CHECK_EQ(bad, size_t(0));
      CHECK(std::any_of(a.begin(), a.end(), [](uint32_t v) { return v != 0; }));
      CHECK_EQ(n.spill(), 0u);
      if (timing) {
        const std::string what = std::string(tp ? "tp card" : "one card") + " down register plan";
        time_pair(d, what.c_str(), [&](l0::CmdList& l) { l.launch(o.k, kD / wgn, ib); },
                  [&](l0::CmdList& l) { l.launch(n.k, kD / wgn, ib); });
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  const bool timing = argc > 1 && std::string(argv[1]) == "--time";
  if (l0::Context::gpu_count() < 1) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  Dev d;
  const std::vector<uint32_t> routes = route(d, timing);
  const l0::Mem rt = d.up(routes), x = d.up(random_bf16(size_t(kC) * kD, -0.5f, 0.5f, 600));
  const Sorted s = sort(d, rt, kC);
  gather_gate_up(d, s, x, timing);
  combines(d, rt, s, timing);
  downs(d, s, timing);
  std::printf("qwen4exp_pf_glue_test OK\n");
  return 0;
}
