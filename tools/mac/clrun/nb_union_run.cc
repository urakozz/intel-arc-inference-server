// nb_union_run - spec 26c Task 3 (Decision 10): the union-aware M-row MoE (src/kernels/bolt/nb_moe_union.cl) against the
// plain M-row form (nb_moe.cl's nb_moe_up / nb_moe_down at the same M) and the host twin (tests/kernels/bolt_ref.h moe() row
// by row, union_schedule) on the Mac's OpenCL GPU (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
// Random int4 expert blocks at I 1536 (the smallest Bolt I; E 80 blocks so a verify can choose 72 distinct experts), random
// latent rows, route rows written as nb_route writes them (nine ascending distinct ids, fp32 weights). DN_KS 1 for BOTH forms
// (the Mac's 256-lane work-group cap: the plain down is 16 x 9 x DN_KS lanes) - the card's DN_KS 2 is bolt_moe_union_test's.
// For each case the union's a [M][9][I] and r [M][1024] equal the plain kernels' BITWISE (and both the twin's); the counting
// build reads each chosen expert's up tiles once per 4-tile block (I / 64 work-groups) and its down tiles once per latent
// block (16), every other expert's never:
//
//   same       M 8, every row the same nine experts (the union is 9)
//   disjoint   M 8, 72 distinct experts (the union is every pair)
//   one        M 8, expert 5 in every row, the rest distinct (one expert with 8 pairs, 64 with one)
//   random     M 2 .. 8, overlapping random routes
//
//   nb_union_run           (no arguments)
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
#include "kernels/bolt_ref.h"

namespace {

namespace br = boltref;
constexpr uint32_t kE = 80, kI = 1536, kL = 1024, kK = 9, kRw = 48;

int failures = 0;
template <class T>
void same(const std::string& what, const std::vector<T>& got, const std::vector<T>& want) {
  size_t diff = 0, first = 0;
  if (got.size() != want.size()) {
    std::printf("  %s: size %zu vs %zu  <-- MISMATCH\n", what.c_str(), got.size(), want.size());
    ++failures;
    return;
  }
  for (size_t i = 0; i < got.size(); ++i)
    if (std::memcmp(&got[i], &want[i], sizeof(T)) != 0 && diff++ == 0) first = i;
  if (diff) {
    std::printf("  %s: %zu of %zu differ (first at %zu)  <-- MISMATCH\n", what.c_str(), diff, got.size(), first);
    ++failures;
  }
}

// Random layout-1 expert blocks: nibbles uniform, every tile's 16 f16 scales in [2^-9, 2^-8) (finite, positive).
std::vector<uint32_t> make_blocks(std::mt19937& g) {
  const size_t words = size_t(kE) * br::expert_block_words(kI);
  std::vector<uint32_t> w(words);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = w.data() + t * 136;
    for (int j = 0; j < 128; ++j) tile[j] = uint32_t(g());
    for (int j = 0; j < 8; ++j) {
      const uint16_t lo = uint16_t(0x1800 + (g() & 0x3FF)), hi = uint16_t(0x1800 + (g() & 0x3FF));   // f16 2^-9 .. 2^-8
      tile[128 + j] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
  }
  return w;
}

struct Case {
  std::string name;
  uint32_t M;
  std::vector<uint32_t> ids;   // [M][9], each row ascending, distinct
};

std::vector<uint32_t> sorted_row(std::vector<uint32_t> r) {
  std::sort(r.begin(), r.end());
  return r;
}

void run_case(clrun::Device& dev, const Case& c, const std::vector<uint32_t>& blocks, clrun::Buffer& wb, std::mt19937& g) {
  const uint32_t M = c.M;
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  std::vector<uint16_t> lat(size_t(M) * kL);
  for (uint16_t& x : lat) x = br::rne(ud(g) * 2.0f);
  std::vector<uint32_t> route(size_t(M) * kRw, 0u);
  std::vector<br::Route> rt(M);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < kK; ++k) {
      const float w = 0.05f + 0.6f * (ud(g) + 1.0f) / 2.0f;
      rt[m].ids[k] = c.ids[size_t(m) * kK + k];
      rt[m].w[k] = w;
      route[size_t(m) * kRw + k] = rt[m].ids[k];
      std::memcpy(&route[size_t(m) * kRw + 9 + k], &w, 4);
    }
  const std::string Ms = "M=" + std::to_string(M);
  const std::vector<std::string> plain_defs = {Ms, "MOE_I=1536", "UP_KS=4", "DN_KS=1"};
  clrun::Program pu(dev, "src/kernels/bolt/nb_moe.cl", plain_defs), pd(dev, "src/kernels/bolt/nb_moe.cl", plain_defs);
  clrun::Program uu(dev, "src/kernels/bolt/nb_moe_union.cl", {Ms, "MOE_I=1536", "UP_KS=4", "DN_KS=1", "UNION_UP=1"});
  clrun::Program ud_(dev, "src/kernels/bolt/nb_moe_union.cl", {Ms, "MOE_I=1536", "UP_KS=4", "DN_KS=1", "UNION_DN=1"});
  clrun::Program uc(dev, "src/kernels/bolt/nb_moe_union.cl", {Ms, "UNION_COMB=1"});
  clrun::Program cu(dev, "src/kernels/bolt/nb_moe_union.cl",
                    {Ms, "MOE_I=1536", "UP_KS=4", "DN_KS=1", "UNION_UP=1", "UNION_COUNT=1"});
  clrun::Program cd(dev, "src/kernels/bolt/nb_moe_union.cl",
                    {Ms, "MOE_I=1536", "UP_KS=4", "DN_KS=1", "UNION_DN=1", "UNION_COUNT=1"});
  clrun::Buffer rb(dev, route), lb(dev, lat);
  const size_t a_bytes = size_t(M) * kK * kI * 2, r_bytes = size_t(M) * kL * 2;
  // the plain M-row form
  clrun::Buffer pa(dev, a_bytes), pr(dev, r_bytes);
  pu.run("nb_moe_up", {size_t(kK * (kI / 64)) * 256, M}, {256, 1}, rb, lb, wb, pa, kE);
  pd.run("nb_moe_down", {size_t(64) * 144, M}, {144, 1}, rb, pa, wb, pr, kE);
  // the union form
  clrun::Buffer ua(dev, a_bytes), dd(dev, size_t(M) * kK * kL * 2), ur(dev, r_bytes);
  const size_t ugroups = size_t(M) * kK * (kI / 64), dgroups = size_t(M) * kK * (kL / 64);
  uu.run("nb_moe_up_U", {ugroups * 256}, {256}, rb, lb, wb, ua, kE);
  ud_.run("nb_moe_down_U", {dgroups * 64}, {64}, rb, ua, wb, dd, kE);
  uc.run("nb_moe_comb_U", {kL, M}, {256, 1}, rb, dd, ur);
  const std::vector<uint16_t> got_a = ua.read<uint16_t>(), got_r = ur.read<uint16_t>();
  same(c.name + ": union a = plain a", got_a, pa.read<uint16_t>());
  same(c.name + ": union r = plain r", got_r, pr.read<uint16_t>());
  // the twin, row by row
  std::vector<uint16_t> ta, tr, td;
  for (uint32_t m = 0; m < M; ++m) {
    const br::MoeOut o = br::moe(rt[m], lat.data() + size_t(m) * kL, blocks.data(), kI, 4, 1);
    ta.insert(ta.end(), o.a.begin(), o.a.end());
    tr.insert(tr.end(), o.r.begin(), o.r.end());
    td.insert(td.end(), o.d.begin(), o.d.end());
  }
  same(c.name + ": union a = the twin", got_a, ta);
  same(c.name + ": union d = the twin's down rows", dd.read<uint16_t>(), td);
  same(c.name + ": union r = the twin", got_r, tr);
  // the counting build: each chosen expert's blocks once a (expert, 4-tile block), no other expert's
  clrun::Buffer cnt(dev, size_t(2) * 512 * 4), a2(dev, a_bytes), d2(dev, size_t(M) * kK * kL * 2);
  cu.run("nb_moe_up_U", {ugroups * 256}, {256}, rb, lb, wb, a2, kE, cnt);
  cd.run("nb_moe_down_U", {dgroups * 64}, {64}, rb, ua, wb, d2, kE, cnt);
  const std::vector<uint32_t> n = cnt.read<uint32_t>();
  const std::vector<br::UnionExpert> sch = br::union_schedule(c.ids.data(), M, kE);
  std::vector<uint32_t> want(2 * 512, 0u);
  for (const br::UnionExpert& x : sch) {
    want[x.expert] = kI / 64;
    want[512 + x.expert] = kL / 64;
  }
  same(c.name + ": every chosen block read once (counting build)", n, want);
  size_t pairs = 0;
  for (const br::UnionExpert& x : sch) pairs += x.pairs.size();
  if (pairs != size_t(M) * kK) {
    std::printf("  %s: the schedule covers %zu pairs, not %u  <-- MISMATCH\n", c.name.c_str(), pairs, M * kK);
    ++failures;
  }
  std::printf("%-9s M %u: %zu distinct experts of %u pairs - union a / d / r bitwise the plain M-row form and the twin, "
              "each block read once\n",
              c.name.c_str(), M, sch.size(), M * kK);
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::mt19937 g(26);
    const std::vector<uint32_t> blocks = make_blocks(g);
    clrun::Buffer wb(dev, blocks);
    std::vector<Case> cases;
    {   // the same nine experts in every row
      Case c{"same", 8, {}};
      const std::vector<uint32_t> r = {3, 9, 17, 22, 40, 41, 55, 70, 79};
      for (int m = 0; m < 8; ++m) c.ids.insert(c.ids.end(), r.begin(), r.end());
      cases.push_back(c);
    }
    {   // 72 distinct experts
      Case c{"disjoint", 8, {}};
      std::vector<uint32_t> all(kE);
      for (uint32_t e = 0; e < kE; ++e) all[e] = e;
      std::shuffle(all.begin(), all.end(), g);
      for (int m = 0; m < 8; ++m) {
        const std::vector<uint32_t> r = sorted_row({all.begin() + m * 9, all.begin() + m * 9 + 9});
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    {   // expert 5 in every row, the rest distinct
      Case c{"one", 8, {}};
      std::vector<uint32_t> rest;
      for (uint32_t e = 0; e < kE; ++e)
        if (e != 5) rest.push_back(e);
      std::shuffle(rest.begin(), rest.end(), g);
      for (int m = 0; m < 8; ++m) {
        std::vector<uint32_t> r(rest.begin() + m * 8, rest.begin() + m * 8 + 8);
        r.push_back(5);
        r = sorted_row(r);
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    for (uint32_t M = 2; M <= 8; ++M) {   // overlapping random routes from a pool of 24
      Case c{"random", M, {}};
      for (uint32_t m = 0; m < M; ++m) {
        std::vector<uint32_t> pool(24);
        for (uint32_t e = 0; e < 24; ++e) pool[e] = e * 3 + 1;
        std::shuffle(pool.begin(), pool.end(), g);
        const std::vector<uint32_t> r = sorted_row({pool.begin(), pool.begin() + 9});
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    for (const Case& c : cases) run_case(dev, c, blocks, wb, g);
    std::printf("nb_moe_union on %s: %zu cases, %s\n", dev.name().c_str(), cases.size(),
                failures ? "DISAGREE with the plain M-row form (indicative: read the kernel before the box)"
                         : "bitwise the plain M-row form (indicative only; the B70 is the real test)");
    return failures ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "nb_union_run: %s\n", e.what());
    return 1;
  }
}
