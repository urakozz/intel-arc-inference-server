// Open-work item 48: Qwen3.8-Flash-Next's union-read MoE (src/kernels/qwen4exp/q4_moe_union.cl, the _U binaries) is
// BITWISE the per-row form (q4_moe.cl at the same M) at the real shapes - one card's I 640 (_SH4 and _SHB, direct and
// the expert tier's table form) and a tensor-parallel card's I 320 halves (_TP_R0 / _R1, direct and table: the fp32
// slots, the peer's landing and the call counters) - on random int4 expert blocks (layout 1), random rows and routes:
//   same        every row the same ten experts
//   permuted    the same ten in a different slot order a row (an expert at a different rank in each row)
//   disjoint    ten different experts a row
//   one         expert 5 in every row at a different slot, the rest distinct
//   random      ten distinct a row from a pool of 24 (overlapping)
//   repeat      row 0 holds one expert in all ten slots (a NaN row's form: more pairs than rows for one expert)
// In the table forms every expert id divisible by 5 lives in pinned host memory (l0::MemKind::Host), the rest in VRAM.
// The counting builds (_U_CNT, M 4, one card's table form and tp card 0's) read each chosen expert's gate||up tiles
// once per 4-tile block, its down tiles once per (n-tile, K slice), the shared expert likewise, every other expert
// never - each expert read once a verify. `--time` adds the MoE pair's time at M = 4, union against per-row.
// Needs a B70, no checkpoint; exit 77 (SKIP) without a GPU.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
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
constexpr uint32_t kD = qk::kHidden, kE = qk::kExperts, kK = qk::kTopK, kS = qk::kSlots, kRw = qk::route::kWords;
constexpr uint32_t kPool = 48, kHostPool = 8;   // distinct VRAM / host blocks behind the 512 ids

int failures = 0;
void report(const std::string& what, bool ok, const std::string& detail = "") {
  std::printf("  %s%s%s\n", ok ? "" : "FAIL ", what.c_str(), detail.c_str());
  if (!ok) ++failures;
}

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
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m) {
    std::vector<T> v(m.size() / sizeof(T));
    imm.copy(v.data(), m.ptr(), m.size());
    return v;
  }
};

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
float bf(float x) { return common::bf16_to_f32(common::f32_to_bf16(x)); }
uint32_t u32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// One shape's weights: the 512 ids' gate||up and down blocks, direct (every id its own copy of a pool block) and as
// the table row (u64 [1024]: gate||up at 0..511, down at 512..1023; ids divisible by 5 in host memory).
struct Weights {
  size_t gub, dnb;   // u32 words of one expert's blocks
  l0::Mem gu_all, dn_all, gu_pool, dn_pool, gu_host, dn_host, table, sh_gu, sh_dn, shb_gu, shb_dn;
  Weights(Dev& d, uint32_t inter, uint32_t seed)
      : gub(size_t(2 * inter / 16) * (kD / 64) * 136),
        dnb(size_t(kD / 16) * (inter / 64) * 136),
        gu_all(d.ctx, l0::MemKind::Device, gub * kE * 4),
        dn_all(d.ctx, l0::MemKind::Device, dnb * kE * 4),
        gu_pool(d.up(random_blocks(gub * kPool, seed))),
        dn_pool(d.up(random_blocks(dnb * kPool, seed + 1))),
        gu_host(d.up(random_blocks(gub * kHostPool, seed + 2), l0::MemKind::Host)),
        dn_host(d.up(random_blocks(dnb * kHostPool, seed + 3), l0::MemKind::Host)),
        table(d.ctx, l0::MemKind::Device, 2 * kE * 8),
        sh_gu(d.up(random_blocks(gub, seed + 4))),
        sh_dn(d.up(random_blocks(dnb, seed + 5))),
        shb_gu(d.up(random_bf16(size_t(kD) * 2 * inter, -0.03f, 0.03f, seed + 6))),
        shb_dn(d.up(random_bf16(size_t(inter) * kD, -0.05f, 0.05f, seed + 7))) {
    std::vector<uint64_t> row(2 * kE);
    const auto at = [](const l0::Mem& m, size_t words) { return static_cast<const uint8_t*>(m.ptr()) + words * 4; };
    for (uint32_t e = 0; e < kE; ++e) {
      const bool host = e % 5 == 0;
      const uint8_t* g = host ? at(gu_host, gub * (e / 5 % kHostPool)) : at(gu_pool, gub * (e % kPool));
      const uint8_t* n = host ? at(dn_host, dnb * (e / 5 % kHostPool)) : at(dn_pool, dnb * (e % kPool));
      row[e] = reinterpret_cast<uint64_t>(g);
      row[kE + e] = reinterpret_cast<uint64_t>(n);
      d.imm.copy(const_cast<uint8_t*>(at(gu_all, gub * e)), g, gub * 4);
      d.imm.copy(const_cast<uint8_t*>(at(dn_all, dnb * e)), n, dnb * 4);
    }
    d.imm.copy(table.ptr(), row.data(), row.size() * 8);
  }
  const void* gu(bool tbl) const { return tbl ? table.ptr() : gu_all.ptr(); }
  const void* dn(bool tbl) const { return tbl ? static_cast<const uint8_t*>(table.ptr()) + kE * 8 : dn_all.ptr(); }
};

struct Case {
  std::string name;
  std::vector<uint32_t> route;   // [M][32]
};
std::vector<Case> cases(uint32_t M, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> wd(0.02f, 0.3f), gd(0.1f, 0.9f);
  const auto row_of = [&](const std::vector<uint32_t>& ids) {
    std::vector<uint32_t> r(kRw, 0);
    for (uint32_t k = 0; k < kK; ++k) {
      r[qk::route::kIds + k] = ids[k];
      r[qk::route::kWeights + k] = u32(bf(wd(rng)));
    }
    r[qk::route::kSharedGate] = u32(bf(gd(rng)));
    return r;
  };
  const std::vector<uint32_t> base = {511, 0, 17, 128, 255, 256, 300, 383, 400, 480};
  std::vector<std::vector<std::vector<uint32_t>>> rows(6);
  for (uint32_t m = 0; m < M; ++m) {
    rows[0].push_back(base);
    std::vector<uint32_t> p = base;
    std::rotate(p.begin(), p.begin() + (3 * m) % kK, p.end());
    if (m % 2) std::reverse(p.begin(), p.end());
    rows[1].push_back(p);
    std::vector<uint32_t> dj(kK), one(kK);
    for (uint32_t k = 0; k < kK; ++k) dj[k] = 7 + 50 * m + 3 * k;
    rows[2].push_back(dj);
    for (uint32_t k = 0; k < kK; ++k) one[k] = 100 + 40 * m + k;
    one[m % kK] = 5;
    rows[3].push_back(one);
    std::vector<uint32_t> pool(24);
    for (uint32_t i = 0; i < 24; ++i) pool[i] = 5 * i + i % 3;   // every third a host expert
    std::shuffle(pool.begin(), pool.end(), rng);
    rows[4].emplace_back(pool.begin(), pool.begin() + kK);
    rows[5].push_back(m == 0 ? std::vector<uint32_t>(kK, 25) : rows[4].back());
  }
  const char* names[6] = {"same", "permuted", "disjoint", "one", "random", "repeat"};
  std::vector<Case> out;
  for (uint32_t c = 0; c < 6; ++c) {
    Case cs{names[c], {}};
    for (const auto& r : rows[c]) {
      const std::vector<uint32_t> w = row_of(r);
      cs.route.insert(cs.route.end(), w.begin(), w.end());
    }
    out.push_back(std::move(cs));
  }
  return out;
}

struct Out {
  std::vector<uint16_t> h, y;
  std::vector<uint32_t> slots, land, ctr, cnt;
};
// The landing of a tp MoE (tp_wire.h at SLOTS 11, TP_MROWS 8, FOLD_G 160): two data slots, the trailers, the flags.
size_t land_words() {
  const size_t data = size_t(kS) * 8 * kD, j = size_t(kD / 16) * 8;
  const size_t flag_base = (2 * data + 2 * j * 16 + 1023) / 1024 * 1024;
  return flag_base + 2 * j * 16;
}

// gate||up then down of one binary at M rows; tp: the card's fp32 slots, the landing and the counters.
Out run(Dev& d, const std::string& variant, uint32_t M, bool tp, bool tbl, bool shb, bool count, const Weights& W,
        const l0::Mem& rb, const l0::Mem& xb, uint32_t inter) {
  l0::Module mod(d.ctx, kernels::path(variant));
  l0::Mem hb = d.zeros(size_t(M) * kS * inter * 2);
  l0::Mem yb = d.zeros(tp ? size_t(kS) * M * kD * 4 : size_t(M) * kD * 2);
  l0::Mem land = d.zeros(land_words() * 4), ctr = d.zeros(size_t(kD / 16) * 8 * 4), cnt = d.zeros(2 * (kE + 1) * 4);
  const void* sgu = shb ? W.shb_gu.ptr() : W.sh_gu.ptr();
  const void* sdn = shb ? W.shb_dn.ptr() : W.sh_dn.ptr();
  l0::Kernel gu(mod, "q4_moe_gate_up"), dn(mod, "q4_moe_down");
  if (tbl) {
    gu.indirect_host();
    dn.indirect_host();
  }
  gu.group_size(qk::moe_gate_up_wg());
  dn.group_size(tp ? qk::tp_moe_down_wg() : qk::moe_down_wg());
  const void* ga[5] = {rb.ptr(), xb.ptr(), W.gu(tbl), sgu, hb.ptr()};
  for (uint32_t i = 0; i < 5; ++i) gu.arg_ptr(i, ga[i]);
  if (count) gu.arg_ptr(5, cnt.ptr());
  const void* da[7] = {rb.ptr(), hb.ptr(), W.dn(tbl), sdn, yb.ptr(), land.ptr(), ctr.ptr()};
  const uint32_t na = tp ? 7 : 5;
  for (uint32_t i = 0; i < na; ++i) dn.arg_ptr(i, da[i]);
  if (count) dn.arg_ptr(na, cnt.ptr());
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(gu, kS * (2 * inter / 16 / 4), M);
  list.launch(dn, kD / 16, M);
  list.close();
  d.q.execute(list, &d.f);
  d.f.wait();
  Out o;
  o.h = d.read<uint16_t>(hb);
  if (tp) {
    o.slots = d.read<uint32_t>(yb);
    o.land = d.read<uint32_t>(land);
    o.ctr = d.read<uint32_t>(ctr);
  } else {
    o.y = d.read<uint16_t>(yb);
  }
  if (count) o.cnt = d.read<uint32_t>(cnt);
  return o;
}

// Equal, and not trivially (the outputs were written: h, and y or the slots, hold nonzero values).
bool same(const Out& a, const Out& b) {
  const auto written = [](const auto& v) { return std::any_of(v.begin(), v.end(), [](auto x) { return x != 0; }); };
  return a.h == b.h && a.y == b.y && a.slots == b.slots && a.land == b.land && a.ctr == b.ctr && written(a.h) &&
         (written(a.y) || written(a.slots));
}

// Each chosen expert (and the shared one) read once a 4-tile gate||up block and once a down (n-tile, K slice).
bool counts_ok(const std::vector<uint32_t>& cnt, const std::vector<uint32_t>& route, uint32_t M, uint32_t inter,
               uint32_t dn_ks, std::string& detail) {
  std::set<uint32_t> chosen;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < kK; ++k) chosen.insert(std::min(route[m * kRw + k], kE - 1));
  const uint32_t up_want = 2 * inter / 16 / 4, dn_want = (kD / 16) * dn_ks;
  bool ok = cnt[kE] == up_want && cnt[kE + 1 + kE] == dn_want;
  for (uint32_t e = 0; e < kE; ++e) {
    const bool c = chosen.count(e) != 0;
    ok = ok && cnt[e] == (c ? up_want : 0u) && cnt[kE + 1 + e] == (c ? dn_want : 0u);
  }
  detail = ": " + std::to_string(chosen.size()) + " distinct of " + std::to_string(M * kK) + " pairs + the shared";
  return ok;
}

double time_pair(Dev& d, const std::string& variant, uint32_t M, bool tp, bool tbl, const Weights& W, const l0::Mem& rb,
                 const l0::Mem& xb, uint32_t inter, int reps) {
  l0::Module mod(d.ctx, kernels::path(variant));
  l0::Mem hb = d.zeros(size_t(M) * kS * inter * 2), yb = d.zeros(size_t(kS) * M * kD * 4);
  l0::Mem land = d.zeros(land_words() * 4), ctr = d.zeros(size_t(kD / 16) * 8 * 4);
  l0::Kernel gu(mod, "q4_moe_gate_up"), dn(mod, "q4_moe_down");
  if (tbl) {
    gu.indirect_host();
    dn.indirect_host();
  }
  gu.group_size(qk::moe_gate_up_wg());
  dn.group_size(tp ? qk::tp_moe_down_wg() : qk::moe_down_wg());
  const void* ga[5] = {rb.ptr(), xb.ptr(), W.gu(tbl), W.sh_gu.ptr(), hb.ptr()};
  for (uint32_t i = 0; i < 5; ++i) gu.arg_ptr(i, ga[i]);
  const void* da[7] = {rb.ptr(), hb.ptr(), W.dn(tbl), W.sh_dn.ptr(), yb.ptr(), land.ptr(), ctr.ptr()};
  for (uint32_t i = 0; i < (tp ? 7u : 5u); ++i) dn.arg_ptr(i, da[i]);
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  for (int i = 0; i < 50; ++i) {   // 50 MoE blocks back to back (a verify's 48 layers)
    list.launch(gu, kS * (2 * inter / 16 / 4), M);
    list.launch(dn, kD / 16, M);
  }
  list.close();
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    d.q.execute(list, &d.f);
    d.f.wait();
    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  const bool timing = argc > 1 && std::string(argv[1]) == "--time";
  if (l0::Context::gpu_count() < 1) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  Dev d;
  const Weights pp(d, qk::kInter, 900), tpw(d, qk::kTpInter, 950);
  for (uint32_t M = 2; M <= 4; ++M) {
    const l0::Mem xb = d.up(random_bf16(size_t(M) * kD, -0.5f, 0.5f, 1000 + M));
    for (const Case& c : cases(M, 2000 + M)) {
      const l0::Mem rb = d.up(c.route);
      std::string bad;
      for (bool shb : {false, true})
        for (bool tbl : {false, true}) {
          const Out a = run(d, qk::moe_variant(M, shb, tbl), M, false, tbl, shb, false, pp, rb, xb, qk::kInter);
          const Out b = run(d, qk::moe_variant(M, shb, tbl, true), M, false, tbl, shb, false, pp, rb, xb, qk::kInter);
          if (!same(a, b)) bad += " " + qk::moe_variant(M, shb, tbl, true);
        }
      for (unsigned rank : {0u, 1u})
        for (bool tbl : {false, true}) {
          const Out a = run(d, qk::tp_moe_variant(tbl, rank, M), M, true, tbl, false, false, tpw, rb, xb, qk::kTpInter);
          const Out b = run(d, qk::tp_moe_variant(tbl, rank, M, true), M, true, tbl, false, false, tpw, rb, xb, qk::kTpInter);
          if (!same(a, b)) bad += " " + qk::tp_moe_variant(tbl, rank, M, true);
        }
      report("M " + std::to_string(M) + " " + c.name + ": h, y / the tp slots, landing and counters bitwise the per-row form",
             bad.empty(), bad.empty() ? " (4 one-card + 4 tp binaries)" : " - differ:" + bad);
      if (M == 4) {
        std::string det;
        const Out a = run(d, qk::moe_variant(M, false, true), M, false, true, false, false, pp, rb, xb, qk::kInter);
        const Out b = run(d, qk::moe_variant(M, false, true, true) + "_CNT", M, false, true, false, true, pp, rb, xb,
                          qk::kInter);
        report("M 4 " + c.name + ": one card, each chosen expert read once", counts_ok(b.cnt, c.route, M, qk::kInter, qk::kDnKs, det) && same(a, b), det);
        const Out t0 = run(d, qk::tp_moe_variant(true, 0, M), M, true, true, false, false, tpw, rb, xb, qk::kTpInter);
        const Out t1 = run(d, qk::tp_moe_variant(true, 0, M, true) + "_CNT", M, true, true, false, true, tpw, rb, xb,
                           qk::kTpInter);
        report("M 4 " + c.name + ": tp card 0, each chosen expert read once", counts_ok(t1.cnt, c.route, M, qk::kTpInter, 1, det) && same(t0, t1), det);
      }
    }
  }
  if (timing) {
    // M 4 on the random case (about 0.6 distinct experts a pair), every expert in VRAM and with the host ones
    const Case c = cases(4, 2004)[4];
    const l0::Mem rb = d.up(c.route), xb = d.up(random_bf16(size_t(4) * kD, -0.5f, 0.5f, 1004));
    for (bool tp : {false, true})
      for (bool tbl : {false, true}) {
        const Weights& W = tp ? tpw : pp;
        const uint32_t I = tp ? qk::kTpInter : qk::kInter;
        const std::string a = tp ? qk::tp_moe_variant(tbl, 0, 4) : qk::moe_variant(4, false, tbl);
        const std::string b = tp ? qk::tp_moe_variant(tbl, 0, 4, true) : qk::moe_variant(4, false, tbl, true);
        double ta = 0, tb = 0;
        for (int r = 0; r < 3; ++r) {   // interleaved pairs after the first round's warm-up
          const double x = time_pair(d, a, 4, tp, tbl, W, rb, xb, I, 9), y = time_pair(d, b, 4, tp, tbl, W, rb, xb, I, 9);
          if (r > 0) {
            ta += x / 2;
            tb += y / 2;
          }
        }
        std::printf("  time M 4 %s%s: per-row %.1f us, union %.1f us a MoE block (x%.3f)\n", tp ? "tp card 0" : "one card",
                    tbl ? " table (ids % 5 in host memory)" : " direct", ta, tb, tb / ta);
      }
  }
  if (failures) {
    std::printf("qwen4exp_moe_union_test: %d FAILED\n", failures);
    return 1;
  }
  std::printf("qwen4exp_moe_union_test OK\n");
  return 0;
}
