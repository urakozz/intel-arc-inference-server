// probe_ink_w4a8 - plan 41e Task 1: Inkling's NVFP4 expert GEMMs at the 4k chunk's shape (4,096 rows x top 8 over 128
// experts, every expert 256 rows: one full work item each), today's bf16 GEMM (pf_moe_q4gemm_*_F32_INK_NVFP4) against
// the int8 forms of probe_ink_w4a8.cl (ARM 1 exact, 2 the refused per-32 fold, 3 the int8 core alone), random codes and
// block scales, random activations. ARM 1 is checked bitwise against a host twin (the same int8 rows, the same fma
// chain) on two experts' rows before any time; ARM 2 / 3 are timing ceilings and not checked.
//
// Timing: a warm-up of every arm, then `rounds` rounds with the arms in a rotating order; a sample is `reps` launches
// between two host syncs. Medians, ms a launch (one visit's gate||up or down).
//
//   probe_ink_w4a8 [--rounds 7] [--reps 8]
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/inkling2_kernels.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"

namespace {
namespace ink = kernels::inkling2;
using D = ink::D;
constexpr uint32_t kRows = 4096 * D::top_k, kItemRows = 256, kItems = kRows / kItemRows;
constexpr uint32_t kTile = 144;

struct Shape {
  const char* name;
  uint32_t K, N;
};
constexpr Shape kShapes[] = {{"gate||up", D::hidden, 2 * D::inter}, {"down", D::inter, D::hidden}};

size_t block_words(const Shape& s) { return size_t(s.N / 16) * (s.K / 64) * kTile + 1; }

float e4m3(uint32_t c) {
  const uint32_t e = (c >> 3) & 15, m = c & 7;
  const float v = e ? std::ldexp(1.f + m / 8.f, int(e) - 7) : std::ldexp(m / 8.f, -6);
  return c & 128 ? -v : v;
}
int e2m1x2(uint32_t code) {
  static constexpr int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  return code & 8 ? -mag[code & 7] : mag[code & 7];
}
uint32_t code_at(const uint32_t* blk, const Shape& s, uint32_t n, uint32_t k) {
  const uint32_t* tile = blk + (size_t(n / 16) * (s.K / 64) + k / 64) * kTile;
  return (tile[((k % 64) / 8) * 16 + n % 16] >> (4 * (k % 8))) & 15;
}
uint32_t scale_at(const uint32_t* blk, const Shape& s, uint32_t n, uint32_t k) {
  const uint32_t* tile = blk + (size_t(n / 16) * (s.K / 64) + k / 64) * kTile;
  return (tile[128 + ((k % 64) / 16) * 4 + (n % 16) / 4] >> (8 * (n % 4))) & 255;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  uint32_t rounds = 7, reps = 8;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--rounds")) rounds = uint32_t(std::stoul(argv[i + 1]));
    else if (!std::strcmp(argv[i], "--reps")) reps = uint32_t(std::stoul(argv[i + 1]));
  }
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate_async(ctx);
  std::mt19937 rng(41);
  const auto upload = [&](const auto& v) {
    l0::Mem mem(ctx, l0::MemKind::Device, v.size() * sizeof(v[0]));
    list.copy(mem.ptr(), v.data(), mem.size());
    list.host_synchronize();
    return mem;
  };
  std::vector<uint32_t> items(4 * kItems);
  for (uint32_t i = 0; i < kItems; ++i) {
    items[4 * i] = i;
    items[4 * i + 1] = i * kItemRows;
    items[4 * i + 2] = kItemRows;
  }
  auto items_d = upload(items);
  std::vector<float> xs(kRows);
  for (auto& v : xs) v = std::ldexp(1.f + float(rng() % 1024) / 1024.f, -int(rng() % 4) - 6);
  auto xs_d = upload(xs);

  bool ok = true;
  for (const auto& s : kShapes) {
    std::vector<uint32_t> w(block_words(s) * D::experts);
    for (uint32_t e = 0; e < D::experts; ++e) {
      uint32_t* blk = w.data() + e * block_words(s);
      for (size_t t = 0; t < size_t(s.N / 16) * (s.K / 64); ++t) {
        uint32_t* tile = blk + t * kTile;
        for (uint32_t j = 0; j < 128; ++j) tile[j] = rng();
        for (uint32_t j = 128; j < kTile; ++j) {
          uint32_t v = 0;
          for (uint32_t b = 0; b < 4; ++b) v |= (0x28u + rng() % 24u) << (8 * b);   // positive, finite E4M3
          tile[j] = v;
        }
      }
      blk[block_words(s) - 1] = std::bit_cast<uint32_t>(std::ldexp(1.f + float(rng() % 256) / 256.f, -4));
    }
    std::vector<uint8_t> a8(size_t(kRows) * s.K);
    std::vector<uint16_t> a16(size_t(kRows) * s.K);
    for (size_t i = 0; i < a8.size(); ++i) {
      const int v = int(rng() % 255) - 127;
      a8[i] = uint8_t(int8_t(v));
      a16[i] = uint16_t(std::bit_cast<uint32_t>(float(v) / 64.f) >> 16);
    }
    auto w_d = upload(w), a8_d = upload(a8), a16_d = upload(a16);
    l0::Mem out_d(ctx, l0::MemKind::Device, size_t(kRows) * s.N * sizeof(float));
    const bool gu = s.K == D::hidden;

    struct Arm {
      std::string tag, binary;
      bool int8;
    };
    const std::string kn = "_K" + std::to_string(s.K);
    const std::vector<Arm> arms = {{"bf16 (today)", ink::prefill_expert(common::ExpertFpFormat::Nvfp4, gu), false},
                                   {"int8 exact (two masked k32 a 32 k)", "probe_ink_w4a8" + kn + "_A1", true},
                                   {"int8 fold per 32 (refused, ceiling)", "probe_ink_w4a8" + kn + "_A2", true},
                                   {"int8 core alone (ceiling)", "probe_ink_w4a8" + kn + "_A3", true}};
    std::vector<std::unique_ptr<l0::Module>> modules;
    std::vector<l0::Kernel> kernels;
    for (const auto& arm : arms) {
      modules.push_back(std::make_unique<l0::Module>(ctx, kernels::path(arm.binary)));
      auto k = modules.back()->kernel("pf_moe_q4gemm");
      k.group_size(256);
      k.arg_ptr(0, items_d.ptr());
      if (arm.int8) {
        k.arg_ptr(1, a8_d.ptr());
        k.arg_ptr(2, xs_d.ptr());
        k.arg_ptr(3, w_d.ptr());
        k.arg_ptr(4, out_d.ptr());
      } else {
        k.arg_ptr(1, a16_d.ptr());
        k.arg_ptr(2, w_d.ptr());
        k.arg_ptr(3, nullptr);
        k.arg_ptr(4, out_d.ptr());
      }
      kernels.push_back(std::move(k));
    }
    const auto run = [&](size_t i) { list.launch(kernels[i], s.N / 128, kItems); };

    // ARM 1 against the host twin on experts 0 and 127.
    run(1);
    std::vector<float> y(size_t(kRows) * s.N);
    list.copy(y.data(), out_d.ptr(), out_d.size());
    list.host_synchronize();
    size_t bad = 0;
    std::vector<int8_t> wq(size_t(s.N) * s.K);
    std::vector<float> ws(size_t(s.N) * (s.K / 16));
    for (uint32_t e : {0u, D::experts - 1}) {
      const uint32_t* blk = w.data() + e * block_words(s);
      const float outer = std::bit_cast<float>(blk[block_words(s) - 1]);
      for (uint32_t n = 0; n < s.N; ++n)
        for (uint32_t k = 0; k < s.K; ++k) {
          wq[size_t(n) * s.K + k] = int8_t(e2m1x2(code_at(blk, s, n, k)));
          if (k % 16 == 0) ws[size_t(n) * (s.K / 16) + k / 16] = e4m3(scale_at(blk, s, n, k));
        }
      for (uint32_t r = e * kItemRows; r < (e + 1) * kItemRows; ++r)
        for (uint32_t n = 0; n < s.N; ++n) {
          const int8_t* x = reinterpret_cast<const int8_t*>(a8.data()) + size_t(r) * s.K;
          const int8_t* c = wq.data() + size_t(n) * s.K;
          float acc = 0.f;
          for (uint32_t k0 = 0; k0 < s.K; k0 += 16) {
            int t = 0;
            for (uint32_t k = k0; k < k0 + 16; ++k) t += int(x[k]) * int(c[k]);
            acc = std::fma(float(t), ws[size_t(n) * (s.K / 16) + k0 / 16], acc);
          }
          const float want = (acc * xs[r]) * outer;
          const float got = y[size_t(r) * s.N + n];
          if (std::bit_cast<uint32_t>(want) != std::bit_cast<uint32_t>(got) && bad++ < 4)
            std::fprintf(stderr, "%s row %u col %u: got %.9g want %.9g\n", s.name, r, n, got, want);
        }
    }
    std::printf("%s K %u N %u: int8 exact vs host twin: %zu of %u words differ\n", s.name, s.K, s.N, bad,
                2 * kItemRows * s.N);
    ok = ok && bad == 0;

    for (size_t i = 0; i < arms.size(); ++i) run(i);   // warm-up
    list.host_synchronize();
    std::vector<std::vector<double>> ms(arms.size());
    for (uint32_t round = 0; round < rounds; ++round)
      for (size_t j = 0; j < arms.size(); ++j) {
        const size_t i = (round + j) % arms.size();
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < reps; ++r) run(i);
        list.host_synchronize();
        ms[i].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
      }
    const double flops = 2.0 * kRows * s.K * s.N;
    const double base = median(ms[0]);
    for (size_t i = 0; i < arms.size(); ++i) {
      const double m = median(ms[i]);
      std::printf("  %-38s %7.3f ms  %6.1f TFLOPS  %.3fx bf16\n", arms[i].tag.c_str(), m, flops / m / 1e9, base / m);
    }
  }
  std::puts(ok ? "probe_ink_w4a8: int8 exact bitwise its host twin" : "probe_ink_w4a8: int8 exact DIFFERS");
  return ok ? 0 : 1;
}
