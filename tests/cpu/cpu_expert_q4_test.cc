// PERF-124 step 4: cpu::q4_expert (the host-RAM expert from its layout-1 blocks, AVX2, and AVX-512 where the CPU has it)
// against the card's host twin (tests/kernels/qwen4exp_ref.h: q4_moe.cl's split_dot at UP_KS 4 / DN_KS 2, silu_up with
// exp_torch) - BITWISE, every down sum of every row, at 1..8 rows, one thread and four. Random blocks (every nibble, scales 2^-9..2^-5) and bf16 x.
// Also a tensor-parallel card's half expert (inter 320, the tp binaries' DN_KS 1: the card's fp32 partial), and two
// experts in one pool run (cpu::q4_experts).
// `--bench [threads]`: also the time an expert, 64 experts cold in RAM back to back (cpu_expert_bench's method), pinned.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cpu/cpu_pick.h"
#include "cpu/expert_q4.h"
#include "kernels/qwen4exp_ref.h"

namespace {
constexpr uint32_t kH = q4ref::kHidden, kI = q4ref::kInter;

std::vector<uint32_t> random_block(uint32_t K, uint32_t N, std::mt19937& rng) {
  std::vector<uint32_t> b(size_t(N / 16) * (K / 64) * 136);
  std::uniform_real_distribution<float> sc(-9.0f, -5.0f);
  for (size_t t = 0; t < b.size(); t += 136) {
    for (uint32_t i = 0; i < 128; ++i) b[t + i] = rng();
    for (uint32_t i = 0; i < 8; ++i) {
      const uint16_t lo = common::f32_to_f16(std::exp2(sc(rng))), hi = common::f32_to_f16(std::exp2(sc(rng)));
      b[t + 128 + i] = uint32_t(lo) | uint32_t(hi) << 16;
    }
  }
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  std::mt19937 rng(124);
  std::vector<std::vector<uint16_t>> x(8, std::vector<uint16_t>(kH));
  std::uniform_real_distribution<float> xv(-2.0f, 2.0f);
  for (auto& r : x)
    for (auto& v : r) v = common::f32_to_bf16(xv(rng));
  int fails = 0;
  std::vector<cpu::Isa> isas{cpu::Isa::Avx2};
  if (cpu::cpu_has_avx512()) isas.push_back(cpu::Isa::Avx512);
  // inter 640 (one card, DN_KS 2) and 320 (a tp card's half, DN_KS 1)
  for (const uint32_t inter : {kI, cpu::kQ4TpInter}) {
    const uint32_t dks = inter == kI ? 2 : 1;
    const std::vector<uint32_t> gu = random_block(kH, 2 * inter, rng), dn = random_block(inter, kH, rng);
    // the twin: each row alone, q4_moe_gate_up's h then q4_moe_down's fp32 sum per column
    std::vector<std::vector<float>> want(8, std::vector<float>(kH));
    for (uint32_t r = 0; r < 8; ++r) {
      std::vector<uint16_t> h(inter);
      for (uint32_t i = 0; i < inter; ++i)
        h[i] = q4ref::silu_up(q4ref::split_dot(gu.data(), kH, moe_ref::gate_col(i), x[r].data(), 4),
                              q4ref::split_dot(gu.data(), kH, moe_ref::up_col(i), x[r].data(), 4));
      for (uint32_t n = 0; n < kH; ++n) want[r][n] = q4ref::split_dot(dn.data(), inter, n, h.data(), dks);
    }
    for (cpu::Isa isa : isas) {
      for (unsigned threads : {1u, 4u}) {
        cpu::Pool pool(threads);
        for (uint32_t n = 1; n <= 8; ++n) {
          std::vector<std::vector<float>> got(n, std::vector<float>(kH));
          const uint16_t* xp[8];
          float* op[8];
          for (uint32_t r = 0; r < n; ++r) {
            xp[r] = x[8 - n + r].data();   // a different row set each n
            op[r] = got[r].data();
          }
          cpu::q4_expert(gu.data(), dn.data(), xp, n, op, &pool, isa, inter);
          uint32_t diff = 0;
          for (uint32_t r = 0; r < n; ++r)
            diff += uint32_t(std::memcmp(got[r].data(), want[8 - n + r].data(), kH * 4) != 0);
          std::printf("inter %u %s threads %u rows %u: %s\n", inter, cpu::isa_name(isa), threads, n,
                      diff ? "DIFFERS" : "bitwise");
          fails += diff != 0;
        }
        // q4_experts: two experts in one pool run a phase (rows 0..n-1 and 8-n..7, each with a row of the other's
        // count), each bitwise its twin
        for (uint32_t n = 1; n <= 8; ++n) {
          std::vector<std::vector<float>> got(2 * n, std::vector<float>(kH));
          cpu::Q4Job jobs[2];
          for (uint32_t b = 0; b < 2; ++b) {
            jobs[b].gu = gu.data();
            jobs[b].dn = dn.data();
            jobs[b].n = n;
            for (uint32_t r = 0; r < n; ++r) {
              jobs[b].x[r] = x[b ? 8 - n + r : r].data();
              jobs[b].out[r] = got[b * n + r].data();
            }
          }
          cpu::q4_experts(jobs, 2, &pool, isa, inter);
          uint32_t diff = 0;
          for (uint32_t b = 0; b < 2; ++b)
            for (uint32_t r = 0; r < n; ++r)
              diff += uint32_t(std::memcmp(got[b * n + r].data(), want[b ? 8 - n + r : r].data(), kH * 4) != 0);
          std::printf("inter %u %s threads %u rows %u, two experts at once: %s\n", inter, cpu::isa_name(isa), threads,
                      n, diff ? "DIFFERS" : "bitwise");
          fails += diff != 0;
        }
      }
    }
  }
  if (argc > 1 && !std::strcmp(argv[1], "--bench")) {
    const unsigned threads = argc > 2 ? unsigned(std::atoi(argv[2])) : cpu::physical_cores();
    cpu::Pool pool(threads, cpu::pick_cpus(100));
    constexpr uint32_t kPool = 64;
    std::vector<std::vector<uint32_t>> g(kPool), d(kPool);
    for (uint32_t e = 0; e < kPool; ++e) {
      g[e] = random_block(kH, 2 * kI, rng);
      d[e] = random_block(kI, kH, rng);
    }
    std::vector<std::vector<float>> o(8, std::vector<float>(kH));
    const uint16_t* xp[8];
    float* op[8];
    for (uint32_t r = 0; r < 8; ++r) {
      xp[r] = x[r].data();
      op[r] = o[r].data();
    }
    for (uint32_t n : {1u, 2u, 4u, 8u}) {
      std::vector<double> us;
      for (int rep = 0; rep < 9; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t e = 0; e < kPool; ++e) cpu::q4_expert(g[e].data(), d[e].data(), xp, n, op, &pool);
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kPool);
      }
      std::sort(us.begin(), us.end());
      std::printf("bench threads %u rows %u: %.1f us an expert (median of 9 passes over %u cold experts)\n", threads, n,
                  us[us.size() / 2], kPool);
    }
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
