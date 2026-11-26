// PERF-124 step 2: time cpu::expert_forward on Flash-Next's expert shape (hidden 2560, inter 640, int4 g64, ~2.76 MB an
// expert), cold in RAM: 64 distinct experts (~177 MB, over the L3), and a 192 MB read between repetitions. For each
// M in {1, 4, 8} and thread count: one expert, then N experts back to back (N = 1, 8, 16, 64), microseconds an expert and
// the weight bytes streamed per second. Usage: cpu_expert_bench [--isa avx2|avx512] [--reps R] [--pin: one thread per idle core] [--warm] [--m M] [--threads T]
// CPU only; run under the box's CPU lock: flock /tmp/b70-cpu-model.lock build/tools/probe/cpu_expert_bench
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>

#include "cpu/expert_int4.h"
#include "cpu/cpu_pick.h"

namespace {
constexpr uint32_t kHidden = 2560, kInter = 640, kPool = 64;

cpu::Int4Matrix random_matrix(uint32_t K, uint32_t N, std::mt19937_64& rng) {
  cpu::Int4Matrix m;
  m.K = K;
  m.N = N;
  m.q.resize(size_t(N) * (K / 64) * 32);
  m.s.resize(size_t(N) * (K / 64));
  for (size_t i = 0; i + 8 <= m.q.size(); i += 8) { uint64_t r = rng(); std::memcpy(&m.q[i], &r, 8); }   // any byte is two valid codes
  for (auto& s : m.s) s = uint16_t(0x2000 + (rng() & 0x3FF));   // f16 in [2^-9, 2^-8)
  return m;
}
double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
  cpu::Isa isa = cpu::Isa::Avx2;
  int reps = 10;
  bool pin = false, warm = false;
  uint32_t only_m = 0;
  unsigned only_t = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--isa" && i + 1 < argc) isa = std::string(argv[++i]) == "avx512" ? cpu::Isa::Avx512 : cpu::Isa::Avx2;
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--m" && i + 1 < argc) only_m = std::atoi(argv[++i]);
    else if (a == "--threads" && i + 1 < argc) only_t = std::atoi(argv[++i]);
    else if (a == "--pin") pin = true;
    else if (a == "--warm") warm = true;   // no flush, one expert (L2 / L3 resident): the compute ceiling
  }
  if (isa == cpu::Isa::Avx512 && !cpu::cpu_has_avx512()) { std::printf("SKIP: no AVX-512 on this CPU\n"); return 0; }

  const std::vector<int> cpus = pin ? cpu::pick_cpus() : std::vector<int>{};
  if (pin) {
    std::printf("pinned cpus, idle cores first:");
    for (size_t i = 0; i < 24 && i < cpus.size(); ++i) std::printf(" %d", cpus[i]);
    std::printf("\n");
  }
  std::mt19937_64 rng(1);
  std::vector<cpu::ExpertInt4> experts(kPool);
  for (auto& e : experts) {
    e.gate_up = random_matrix(kHidden, 2 * kInter, rng);
    e.down = random_matrix(kInter, kHidden, rng);
  }
  const size_t eb = experts[0].bytes();
  std::vector<uint64_t> flush(24u << 20, 1);   // 192 MB
  std::vector<uint16_t> x(size_t(8) * kHidden);
  for (auto& v : x) v = uint16_t(0x3F00 + (rng() & 0x7F));   // bf16 ~ 0.5..1
  std::vector<float> out(size_t(8) * kHidden);
  cpu::ExpertScratch sc;
  volatile uint64_t sink = 0;

  std::printf("isa %s, expert %.3f MB, %d reps, %s\n", cpu::isa_name(isa), eb / 1e6, reps,
              warm ? "warm (one expert, no flush)" : "cold (192 MB read before each rep)");
  std::printf("%-3s %-7s %-4s %12s %8s\n", "M", "threads", "N", "us/expert", "GB/s");
  for (uint32_t M : {1u, 4u, 8u})
    for (unsigned threads : {1u, 8u, 22u, 44u}) {
      if ((only_m && M != only_m) || (only_t && threads != only_t)) continue;
      cpu::Pool pool(threads, pin ? cpus : std::vector<int>{});
      for (int w = 0; w < 5; ++w) cpu::expert_forward(experts[w], x.data(), M, out.data(), sc, &pool, isa);   // start the workers
      for (unsigned N : {1u, 8u, 16u, 64u}) {
        std::vector<double> t;
        for (int r = -1; r < reps; ++r) {   // r = -1: warm-up
          uint64_t s = 0;
          if (!warm) for (uint64_t v : flush) s += v;
          sink = sink + s;
          const double t0 = now_us();
          for (unsigned k = 0; k < N; ++k)
            cpu::expert_forward(experts[warm ? 0 : (size_t(r + 1) * N + k) % kPool], x.data(), M, out.data(), sc, &pool, isa);
          const double dt = now_us() - t0;
          if (r >= 0) t.push_back(dt / N);
        }
        std::sort(t.begin(), t.end());
        const double med = t[t.size() / 2];
        std::printf("%-3u %-7u %-4u %12.1f %8.2f\n", M, threads, N, med, eb / med / 1e3);
        std::fflush(stdout);
      }
    }
  return 0;
}
