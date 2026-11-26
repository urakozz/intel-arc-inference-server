// PERF-124: cpu::expert_forward (AVX2, and AVX-512 where the CPU has it) against the scalar fp32 twin
// (tests/cpu/expert_int4_ref.h), random weights at Flash-Next's expert shape (hidden 2560, inter 640), every M 1..8,
// one thread and four.
//
// Under Intel SDE (the AVX-512 kernel on a CPU without it; slow, the whole run takes minutes):
//   sde64 -skx -- build/tests/cpu_expert_test
// The AVX2 half always runs; the AVX-512 half prints SKIP when the CPU lacks it.
#include <immintrin.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "cpu/expert_int4.h"
#include "expert_int4_ref.h"

namespace {
// out within kOutTol of the twin, per row, relative to that row's largest |twin|: fp32 sums over K = 2560 / 640 in a
// different order (8 / 16 lanes against one) differ by ~1e-6 of the row's scale; a gate / up that sits within that of a
// bf16 rounding boundary flips one h element by 2^-8 of itself, ~1e-4 of one down term at worst. 1e-4 holds both.
constexpr double kOutTol = 1e-4;
constexpr uint32_t kHidden = 2560, kInter = 640;

uint16_t f16(float f) { return _cvtss_sh(f, 0); }

int check(cpu::Isa isa, const cpu::ExpertInt4& e, const std::vector<uint8_t>& gu_c, const std::vector<uint16_t>& gu_s,
          const std::vector<uint8_t>& dn_c, const std::vector<uint16_t>& dn_s, const std::vector<uint16_t>& x, unsigned threads) {
  cpu::Pool pool(threads);
  cpu::ExpertScratch sc;
  int fails = 0;
  double worst = 0;
  for (uint32_t M = 1; M <= 8; ++M) {
    std::vector<float> out(size_t(M) * kHidden);
    cpu::expert_forward(e, x.data(), M, out.data(), sc, &pool, isa);
    const auto ref = cpu_ref::expert(gu_c.data(), gu_s.data(), dn_c.data(), dn_s.data(), x.data(), M, kHidden, kInter);
    size_t hdiff = 0;
    for (size_t i = 0; i < ref.h.size(); ++i) hdiff += ref.h[i] != sc.h[i];
    double mrel = 0;
    for (uint32_t m = 0; m < M; ++m) {
      double amax = 0, dmax = 0;
      for (uint32_t n = 0; n < kHidden; ++n) {
        const size_t i = size_t(m) * kHidden + n;
        amax = std::fmax(amax, std::fabs(ref.out[i]));
        dmax = std::fmax(dmax, std::fabs(double(out[i]) - ref.out[i]));
      }
      mrel = std::fmax(mrel, dmax / amax);
    }
    worst = std::fmax(worst, mrel);
    const bool ok = mrel <= kOutTol;
    fails += !ok;
    std::printf("  %s threads %u M %u: max rel err %.2e  h bf16 elements differing %zu / %zu  %s\n", cpu::isa_name(isa), threads, M,
                mrel, hdiff, ref.h.size(), ok ? "ok" : "FAIL");
  }
  std::printf("  %s threads %u: worst %.2e (bound %.0e)\n", cpu::isa_name(isa), threads, worst, kOutTol);
  return fails;
}
}  // namespace

int main() {
  std::mt19937 rng(7);
  auto codes = [&](size_t n) { std::vector<uint8_t> v(n); for (auto& c : v) c = uint8_t(rng() & 15); return v; };
  std::uniform_real_distribution<float> su(0.004f, 0.02f), xu(-1.f, 1.f);
  auto scales = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& s : v) s = f16(su(rng)); return v; };
  const auto gu_c = codes(size_t(2 * kInter) * kHidden), dn_c = codes(size_t(kHidden) * kInter);
  const auto gu_s = scales(size_t(2 * kInter) * (kHidden / 64)), dn_s = scales(size_t(kHidden) * (kInter / 64));
  std::vector<uint16_t> x(size_t(8) * kHidden);
  for (auto& v : x) v = common::f32_to_bf16(xu(rng));
  cpu::ExpertInt4 e;
  e.gate_up = cpu::pack_int4(gu_c.data(), gu_s.data(), kHidden, 2 * kInter);
  e.down = cpu::pack_int4(dn_c.data(), dn_s.data(), kInter, kHidden);
  std::printf("expert %zu B (int4 + f16 scales)\n", e.bytes());

  int fails = 0;
  for (unsigned t : {1u, 4u}) fails += check(cpu::Isa::Avx2, e, gu_c, gu_s, dn_c, dn_s, x, t);
  if (cpu::cpu_has_avx512())
    for (unsigned t : {1u, 4u}) fails += check(cpu::Isa::Avx512, e, gu_c, gu_s, dn_c, dn_s, x, t);
  else
    std::printf("SKIP avx512: this CPU has no AVX-512 (run under sde64 -skx)\n");
  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
