// Spec 26c Task 3 (Decision 10, D3's MoE half on the card): the union-aware M-row MoE (src/kernels/bolt/nb_moe_union.cl:
// nb_moe_up_U / nb_moe_down_U / nb_moe_comb_U) is BITWISE the plain M-row form (nb_moe.cl's nb_moe_up / nb_moe_down at the same
// M) at Bolt's real shapes - E 496 random int4 expert blocks (layout 1, finite positive f16 scales), every I (1536 / 2048 /
// 2304 / 2560), the card's DN_KS 2 - on random latent rows and these routes (nine ascending distinct ids a row):
//   same       every row the same nine experts (the union is 9)
//   disjoint   M 8, 72 distinct experts (every pair its own block)
//   one        expert 5 in every row, the rest distinct
//   random     M 2 .. 8, overlapping routes from a pool of 24
// a [M][9][I] and r [M][1024] equal; the twin (tests/kernels/bolt_ref.h moe(), row by row) agrees at I 1536; and the counting
// build (`_CNT`, M 8, I 1536) reads each chosen expert's up tiles once per 4-tile block (I / 64 work-groups) and its down tiles
// once per latent block (16), every other expert's never - each block read once a verify. Needs a B70, no checkpoint.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/bolt_kernels.h"
#include "kernels/bolt_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {
namespace kb = kernels::bolt;
namespace br = boltref;
constexpr uint32_t kE = 496, kL = 1024, kK = 9, kRw = 48;

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<std::unique_ptr<l0::Module>> mods;
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
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
  l0::Module& mod(const std::string& variant) {
    mods.push_back(std::make_unique<l0::Module>(ctx, kernels::path(variant)));
    return *mods.back();
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};
inline void set1(l0::Kernel& k, uint32_t i, const l0::Mem& m) { k.arg_ptr(i, m.ptr()); }
inline void set1(l0::Kernel& k, uint32_t i, uint32_t v) { k.arg<uint32_t>(i, v); }
template <class... A>
void launch(Dev& d, l0::Module& m, const char* entry, uint32_t wg, uint32_t gx, uint32_t gy, const A&... a) {
  l0::Kernel k = m.kernel(entry);
  k.group_size(wg);
  uint32_t i = 0;
  (set1(k, i++, a), ...);
  d.run(k, gx, gy);
}

std::vector<uint32_t> make_blocks(std::mt19937& g, uint32_t I) {
  std::vector<uint32_t> w(size_t(kE) * br::expert_block_words(I));
  for (size_t t = 0; t < w.size() / 136; ++t) {
    uint32_t* tile = w.data() + t * 136;
    for (int j = 0; j < 128; ++j) tile[j] = uint32_t(g());
    for (int j = 0; j < 8; ++j)
      tile[128 + j] = uint32_t(0x1800 + (g() & 0x3FF)) | (uint32_t(0x1800 + (g() & 0x3FF)) << 16);   // f16 2^-9 .. 2^-8
  }
  return w;
}

std::vector<uint32_t> sorted9(std::vector<uint32_t> r) {
  std::sort(r.begin(), r.end());
  return r;
}

int failures = 0;
template <class T>
void same(const std::string& what, const std::vector<T>& got, const std::vector<T>& want) {
  if (got != want) {
    size_t diff = 0;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) diff += std::memcmp(&got[i], &want[i], sizeof(T)) != 0;
    std::printf("  %s: %zu of %zu differ  <-- MISMATCH\n", what.c_str(), diff, got.size());
    ++failures;
  }
}

}  // namespace

int main() {
  Dev d;
  std::mt19937 g(2026);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (uint32_t I : {1536u, 2048u, 2304u, 2560u}) {
    const std::vector<uint32_t> blocks = make_blocks(g, I);
    const l0::Mem wb = d.up(blocks);
    struct Case {
      std::string name;
      uint32_t M;
      std::vector<uint32_t> ids;
    };
    std::vector<Case> cases;
    {
      Case c{"same", 8, {}};
      for (int m = 0; m < 8; ++m) for (uint32_t e : {3u, 9u, 17u, 22u, 140u, 141u, 255u, 370u, 495u}) c.ids.push_back(e);
      cases.push_back(c);
    }
    {
      Case c{"disjoint", 8, {}};
      std::vector<uint32_t> all(kE);
      for (uint32_t e = 0; e < kE; ++e) all[e] = e;
      std::shuffle(all.begin(), all.end(), g);
      for (int m = 0; m < 8; ++m) {
        const std::vector<uint32_t> r = sorted9({all.begin() + m * 9, all.begin() + m * 9 + 9});
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    {
      Case c{"one", 8, {}};
      std::vector<uint32_t> rest;
      for (uint32_t e = 0; e < kE; ++e)
        if (e != 5) rest.push_back(e);
      std::shuffle(rest.begin(), rest.end(), g);
      for (int m = 0; m < 8; ++m) {
        std::vector<uint32_t> r(rest.begin() + m * 8, rest.begin() + m * 8 + 8);
        r.push_back(5);
        r = sorted9(r);
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    for (uint32_t M = 2; M <= 8; ++M) {
      Case c{"random", M, {}};
      for (uint32_t m = 0; m < M; ++m) {
        std::vector<uint32_t> pool(24);
        for (uint32_t e = 0; e < 24; ++e) pool[e] = e * 20 + 3;
        std::shuffle(pool.begin(), pool.end(), g);
        const std::vector<uint32_t> r = sorted9({pool.begin(), pool.begin() + 9});
        c.ids.insert(c.ids.end(), r.begin(), r.end());
      }
      cases.push_back(c);
    }
    for (const Case& c : cases) {
      const uint32_t M = c.M;
      std::vector<uint16_t> lat(size_t(M) * kL);
      for (uint16_t& x : lat) x = br::rne(ud(g) * 2.0f);
      std::vector<uint32_t> route(size_t(M) * kRw, 0u);
      std::vector<br::Route> rt(M);
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < kK; ++k) {
          const float w = 0.05f + 0.3f * (ud(g) + 1.0f);
          rt[m].ids[k] = c.ids[size_t(m) * kK + k];
          rt[m].w[k] = w;
          route[size_t(m) * kRw + k] = rt[m].ids[k];
          std::memcpy(&route[size_t(m) * kRw + 9 + k], &w, 4);
        }
      const l0::Mem rb = d.up(route), lb = d.up(lat);
      l0::Mem pa = d.zeros(size_t(M) * kK * I * 2), pr = d.zeros(size_t(M) * kL * 2);
      l0::Mem ua = d.zeros(size_t(M) * kK * I * 2), dd = d.zeros(size_t(M) * kK * kL * 2), ur = d.zeros(size_t(M) * kL * 2);
      launch(d, d.mod(kb::moe_up_variant(M, I, false)), "nb_moe_up", kb::moe_up_wg(), kb::moe_up_groups(I), M, rb, lb, wb, pa,
             kE);
      launch(d, d.mod(kb::moe_down_variant(M, I)), "nb_moe_down", kb::moe_down_wg(), kb::moe_down_groups(), M, rb, pa, wb, pr,
             kE);
      launch(d, d.mod(kb::moe_up_union_variant(M, I)), "nb_moe_up_U", kb::moe_up_wg(), kb::moe_up_union_groups(M, I), 1, rb, lb,
             wb, ua, kE);
      launch(d, d.mod(kb::moe_down_union_variant(M, I)), "nb_moe_down_U", kb::moe_down_union_wg(), kb::moe_down_union_groups(M),
             1, rb, ua, wb, dd, kE);
      launch(d, d.mod(kb::moe_comb_union_variant(M)), "nb_moe_comb_U", kb::kCombWg, kL / kb::kCombWg, M, rb, dd, ur);
      const std::string w = "I " + std::to_string(I) + " " + c.name + " M " + std::to_string(M);
      same(w + ": union a = plain a", d.read<uint16_t>(ua), d.read<uint16_t>(pa));
      same(w + ": union r = plain r", d.read<uint16_t>(ur), d.read<uint16_t>(pr));
      if (I == 1536) {   // the twin, row by row
        std::vector<uint16_t> ta, tr;
        for (uint32_t m = 0; m < M; ++m) {
          const br::MoeOut o = br::moe(rt[m], lat.data() + size_t(m) * kL, blocks.data(), I);
          ta.insert(ta.end(), o.a.begin(), o.a.end());
          tr.insert(tr.end(), o.r.begin(), o.r.end());
        }
        same(w + ": union a = the twin", d.read<uint16_t>(ua), ta);
        same(w + ": union r = the twin", d.read<uint16_t>(ur), tr);
      }
      if (I == 1536 && M == 8) {   // the counting build: each chosen block read once
        l0::Mem cnt = d.zeros(size_t(2) * 512 * 4), a2 = d.zeros(size_t(M) * kK * I * 2), d2 = d.zeros(size_t(M) * kK * kL * 2);
        launch(d, d.mod("nb_moe_up_U_M8_I1536_L1024_T9_CNT"), "nb_moe_up_U", kb::moe_up_wg(), kb::moe_up_union_groups(8, I), 1,
               rb, lb, wb, a2, kE, cnt);
        launch(d, d.mod("nb_moe_down_U_M8_I1536_L1024_T9_CNT"), "nb_moe_down_U", kb::moe_down_union_wg(),
               kb::moe_down_union_groups(8), 1, rb, ua, wb, d2, kE, cnt);
        std::vector<uint32_t> want(2 * 512, 0u);
        for (const br::UnionExpert& x : br::union_schedule(c.ids.data(), M, kE)) {
          want[x.expert] = I / 64;
          want[512 + x.expert] = kL / 64;
        }
        same(w + ": every chosen block read once (counting build)", d.read<uint32_t>(cnt), want);
      }
      std::printf("%s: %zu distinct experts - union a / r %s the plain M-row form\n", w.c_str(),
                  br::union_schedule(c.ids.data(), M, kE).size(), failures ? "(see above)" : "bitwise");
    }
  }
  CHECK_EQ(failures, 0);
  std::printf("bolt_moe_union_test OK\n");
  return 0;
}
