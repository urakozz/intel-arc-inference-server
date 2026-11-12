// probe_route - phase 2 Task 6 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): pf_route.cl's
// pf_gemv_rows tilings against today's pf_ab_proj binaries at Ornith's two prefill GEMV shapes - the router ||
// shared gate (K 2048, N 272) and a||b (K 2048, N 128) - at M = 2048 rows of random bf16. Every arm memcmp'd
// against the control (it must be bitwise), then timed: interleaved rounds after a warm-up, median, L0 global
// timestamps. Record: docs/probes/probe-prefill-phase2-2026-11-12.md.
//
//   probe_route [--rounds 9] [--rows 2048]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {
using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;
uint16_t bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
struct Arm {
  std::string variant;
  uint32_t mt, tc;   // 0, 0: the control (pf_ab_proj, 16 columns x 8 rows)
};
}  // namespace

int main(int argc, char** argv) {
  int rounds = 9;
  uint32_t M = 2048;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (a == "--rows" && i + 1 < argc) M = uint32_t(std::atoi(argv[++i]));
  }
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  runtime::prefill::KernelCache kc(ctx);
  constexpr uint32_t K = 2048;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> x(size_t(M) * K);
  for (auto& v : x) v = bf(nd(rng));
  l0::Mem d_x(ctx, l0::MemKind::Device, x.size() * 2), d_stamp(ctx, l0::MemKind::Host, 64);
  cx.copy(d_x.ptr(), x.data(), x.size() * 2);
  uint64_t* stamp = d_stamp.as<uint64_t>();
  bool all_ok = true;
  for (uint32_t N : {272u, 128u}) {
    std::vector<uint16_t> w(size_t(N) * K);
    for (auto& v : w) v = bf(nd(rng) * 0.02f);
    l0::Mem d_w(ctx, l0::MemKind::Device, w.size() * 2), d_o(ctx, l0::MemKind::Device, size_t(M) * N * 4);
    cx.copy(d_w.ptr(), w.data(), w.size() * 2);
    cx.wait();
    const std::string sfx = "_K2048_N" + std::to_string(N);
    std::vector<Arm> arms = {{N == 272 ? "pf_moe_router_K2048_N272" : "pf_ab_proj_D2048", 0, 0}};
    for (auto [mt, tc] : std::vector<std::pair<uint32_t, uint32_t>>{{16, 2}, {32, 1}, {32, 2}, {16, 4}, {64, 1}})
      arms.push_back({"pfr" + sfx + "_MT" + std::to_string(mt) + "_TC" + std::to_string(tc), mt, tc});
    auto run = [&](const Arm& a) {
      const char* entry = a.mt ? "pf_gemv_rows" : "pf_ab_proj";
      const uint32_t gx = a.mt ? (N / 16 + a.tc - 1) / a.tc : N / 16, gy = a.mt ? (M + a.mt - 1) / a.mt : (M + 7) / 8;
      cx.timestamp(stamp);
      cx.launch(kc(a.variant, entry), gx, gy, 1, {PtrArg(d_w.ptr()), PtrArg(d_x.ptr()), PtrArg(d_o.ptr()), arg_val(M)});
      cx.timestamp(stamp + 1);
      cx.wait();
      return cx.timestamp_ms(stamp[0], stamp[1]);
    };
    std::vector<float> ref(size_t(M) * N), got(ref.size());
    run(arms[0]);
    cx.copy(ref.data(), d_o.ptr(), ref.size() * 4);
    cx.wait();
    for (size_t i = 1; i < arms.size(); ++i) {
      std::vector<float> junk(ref.size(), -1.0f);
      cx.copy(d_o.ptr(), junk.data(), junk.size() * 4);
      run(arms[i]);
      cx.copy(got.data(), d_o.ptr(), got.size() * 4);
      cx.wait();
      size_t diff = 0;
      for (size_t j = 0; j < ref.size(); ++j) diff += std::memcmp(&ref[j], &got[j], 4) != 0;
      std::printf("N %u %s: %zu of %zu words differ from the control%s\n", N, arms[i].variant.c_str(), diff, ref.size(),
                  diff ? "  ** NOT BITWISE **" : "");
      all_ok = all_ok && diff == 0;
    }
    for (int r = 0; r < 3; ++r) for (const Arm& a : arms) run(a);
    std::vector<std::vector<double>> t(arms.size());
    for (int r = 0; r < rounds; ++r)
      for (size_t j = 0; j < arms.size(); ++j) {
        const size_t i = (j + size_t(r)) % arms.size();
        t[i].push_back(run(arms[i]));
      }
    for (size_t i = 0; i < arms.size(); ++i)
      std::printf("TIME {\"N\":%u,\"arm\":\"%s\",\"ms\":%.4f,\"vs_control\":%.3f}\n", N, arms[i].variant.c_str(),
                  median(t[i]), median(t[i]) / median(t[0]));
  }
  std::printf("BITWISE %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
