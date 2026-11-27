// probe_df2_linears - plan 41k: the DFlash 2 drafter's linears at one shape, several builds timed against each other
// (the int8 gemm_sm_i8w's work-group / K split / read-ahead, the packed int4 gemm_sm against gemv), on one card.
//
//   probe_df2_linears <int8|int4|int2> K N M S <variant>:<work-items>:<grid x>:<grid y> ...
//
// Every build gets the same weights (int8: [N/16][K/16][16 k][16 n] tiles + fp32 column scales; int4: layout 1's
// 544-byte tiles, finite f16 scales; int2: the 2-bit head's 288-byte tiles) and the same M bf16 rows, the weights in NB copies cycled past the L2 as a pass
// reads them (once each). Rounds run the builds in a rotating order after a warm-up; a build's line is the median of
// its rounds' us a launch and its GB/s over the weight bytes, and its largest difference from the first build's output
// (the S slices summed in order; S is the output's slices of that build - 1 for an in-work-group split).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

namespace {

struct Build {
  std::string name, entry;
  uint32_t wg, gx, gy;
};

uint16_t bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 7) {
      std::fprintf(stderr, "usage: probe_df2_linears <int8|int4|int2> K N M S <variant>:<wi>:<gx>:<gy> ...\n");
      return 2;
    }
    const bool int8 = std::string(argv[1]) == "int8", int2 = std::string(argv[1]) == "int2";
    const uint32_t K = uint32_t(std::atoi(argv[2])), N = uint32_t(std::atoi(argv[3])), M = uint32_t(std::atoi(argv[4])),
                   S = uint32_t(std::atoi(argv[5]));
    std::vector<Build> builds;
    for (int a = 6; a < argc; ++a) {
      std::string s = argv[a];
      Build b;
      size_t p = s.find(':');
      b.name = s.substr(0, p);
      std::sscanf(s.c_str() + p + 1, "%u:%u:%u", &b.wg, &b.gx, &b.gy);
      b.entry = b.name.find("gemm_sm_i8w") != std::string::npos ? "gemm_sm_i8w"
                : b.name.find("gemm_sm") != std::string::npos   ? "gemm_sm"
                : b.name.find("gemv_i8w") != std::string::npos  ? "gemv_i8w"
                                                                : "gemv";
      builds.push_back(b);
    }
    l0::Context ctx(0);
    l0::Queue q(ctx);
    l0::Fence fence(q);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    std::mt19937 rng(41);
    std::uniform_int_distribution<uint32_t> u32;
    // the weights
    const size_t tile = int2 ? 288 : 544, scale_at = int2 ? 256 : 512;
    const size_t wbytes = int8 ? size_t(K) * N : size_t(N / 16) * (K / 64) * tile;
    const size_t sbytes = int8 ? size_t(N) * 4 : 0;
    std::vector<uint8_t> hw(wbytes);
    for (auto& v : hw) v = uint8_t(u32(rng));
    if (!int8)   // layouts 1 / 3: a tile's 16 f16 scales after its codes - finite and small
      for (size_t t = 0; t < wbytes / tile; ++t)
        for (int c = 0; c < 16; ++c) {
          const uint16_t h = uint16_t(0x1C00 + (u32(rng) & 0x3FF));
          std::memcpy(&hw[t * tile + scale_at + 2 * c], &h, 2);
        }
    std::vector<float> hs(N);
    for (auto& v : hs) v = 1e-3f * float(1 + u32(rng) % 100);
    const size_t copies = std::max<size_t>(2, (96u << 20) / (wbytes + sbytes) + 1);
    std::vector<std::unique_ptr<l0::Mem>> w, sc;
    for (size_t c = 0; c < copies; ++c) {
      w.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, wbytes));
      imm.copy(w.back()->ptr(), hw.data(), wbytes);
      sc.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, std::max<size_t>(sbytes, 4)));
      if (int8) imm.copy(sc.back()->ptr(), hs.data(), sbytes);
    }
    std::vector<uint16_t> hx(size_t(M) * K);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    for (auto& v : hx) v = bf16(ux(rng));
    l0::Mem x(ctx, l0::MemKind::Device, hx.size() * 2);
    imm.copy(x.ptr(), hx.data(), hx.size() * 2);
    const size_t out_n = size_t(std::max(S, 1u)) * M * N;
    l0::Mem out(ctx, l0::MemKind::Device, out_n * 4);

    std::vector<std::unique_ptr<l0::Module>> mods;
    std::vector<std::unique_ptr<l0::Kernel>> ks;
    for (const Build& b : builds) {
      mods.push_back(std::make_unique<l0::Module>(ctx, kernels::path(b.name)));
      ks.push_back(std::make_unique<l0::Kernel>(mods.back()->kernel(b.entry.c_str())));
    }
    const int launches = int(copies) * 4;
    auto record = [&](size_t i, int n) {
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int l = 0; l < n; ++l) {
        l0::Kernel& k = *ks[i];
        const size_t c = size_t(l) % copies;
        k.arg_ptr(0, w[c]->ptr());
        k.arg_ptr(1, int8 ? sc[c]->ptr() : w[c]->ptr());
        k.arg_ptr(2, x.ptr());
        k.arg_ptr(3, out.ptr());
        k.group_size(builds[i].wg);
        list.launch(k, builds[i].gx, builds[i].gy);
      }
      list.close();
      return list;
    };
    // outputs: each build once on copy 0 (S slices summed in order)
    std::vector<std::vector<double>> res;
    for (size_t i = 0; i < builds.size(); ++i) {
      const uint32_t slices = builds[i].gy > 1 && !int8 ? builds[i].gy : 1;
      std::vector<float> nanfill(out_n, std::nanf(""));
      imm.copy(out.ptr(), nanfill.data(), out_n * 4);
      l0::CmdList one = record(i, 1);
      q.execute(one, &fence);
      fence.wait();
      std::vector<float> h(out_n);
      imm.copy(h.data(), out.ptr(), out_n * 4);
      std::vector<double> r(size_t(M) * N, 0.0);
      for (uint32_t s = 0; s < slices; ++s)
        for (size_t e = 0; e < r.size(); ++e) r[e] += h[s * r.size() + e];
      res.push_back(std::move(r));
    }
    std::vector<l0::CmdList> lists;
    for (size_t i = 0; i < builds.size(); ++i) lists.push_back(record(i, launches));
    for (size_t i = 0; i < builds.size(); ++i) {   // warm-up
      q.execute(lists[i], &fence);
      fence.wait();
    }
    const int rounds = 9;
    std::vector<std::vector<double>> us(builds.size());
    for (int r = 0; r < rounds; ++r)
      for (size_t j = 0; j < builds.size(); ++j) {
        const size_t i = (j + size_t(r)) % builds.size();
        Timer t;
        t.start();
        q.execute(lists[i], &fence);
        fence.wait();
        us[i].push_back(t.ms() * 1e3 / launches);
      }
    std::printf("# probe_df2_linears %s K %u N %u M %u: %zu weight copies, %d launches a round, %d rounds\n",
                argv[1], K, N, M, copies, launches, rounds);
    for (size_t i = 0; i < builds.size(); ++i) {
      std::sort(us[i].begin(), us[i].end());
      const double med = us[i][us[i].size() / 2];
      double diff = 0, mag = 0;
      for (size_t e = 0; e < res[i].size(); ++e) {
        diff = std::max(diff, std::fabs(res[i][e] - res[0][e]));
        mag = std::max(mag, std::fabs(res[0][e]));
      }
      std::printf("%-44s wi %4u grid %5u x %2u  %8.2f us  %5.0f GB/s  max |diff| %.3g of %.3g\n", builds[i].name.c_str(),
                  builds[i].wg, builds[i].gx, builds[i].gy, med, double(wbytes + sbytes) / med / 1e3, diff, mag);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_df2_linears FAILED: %s\n", e.what());
    return 1;
  }
}
