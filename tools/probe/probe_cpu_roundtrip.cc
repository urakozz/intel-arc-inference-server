// PERF-124 step 3: what handing a MoE layer's routed rows to the CPU and taking the result back costs, per layer.
// One regular command list of L layers, replayed; per layer the GPU copies the rows (M x 2560 bf16 + 128 B routing) to
// pinned host memory, signals a host-visible event (list.barrier_signal), runs a stand-in for its own VRAM experts
// (rt_spin, D us), waits on a second event the host signals (list.wait_event), and copies the M x 2560 bf16 result back.
// The host thread polls the first event, runs K cpu::expert_forward on experts from a 64-expert pool (cold in RAM),
// combines them with weight 1/K to bf16 and host-signals the second. Baseline ("none"): the same copies and spin, no
// hand-off. Per layer = replay time / L; "exposed" = hand-off minus baseline.
// A GPU kernel polling a host-USM flag (rt_set / rt_wait) was tried first and dropped: the kernel never saw the host's
// store (a system-scope atomic load rewrote the line with a stale value, an uncached load saw nothing).
// Usage (card 0, CPU job beside): tools/gpu_lock.sh card 0 -- flock /tmp/b70-cpu-model.lock build/tools/probe/probe_cpu_roundtrip
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "common/bf16.h"
#include "cpu/expert_int4.h"
#include "cpu/cpu_pick.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "l0/sync_event.h"
#include "timer.h"

namespace {
constexpr uint32_t kHidden = 2560, kInter = 640, kPool = 64, kL = 8, kMaxM = 8, kRoute = 128;
enum class Mode { None, Event };

cpu::Int4Matrix random_matrix(uint32_t K, uint32_t N, std::mt19937_64& rng) {
  cpu::Int4Matrix m;
  m.K = K;
  m.N = N;
  m.q.resize(size_t(N) * (K / 64) * 32);
  m.s.resize(size_t(N) * (K / 64));
  for (size_t i = 0; i + 8 <= m.q.size(); i += 8) { uint64_t r = rng(); std::memcpy(&m.q[i], &r, 8); }
  for (auto& s : m.s) s = uint16_t(0x2000 + (rng() & 0x3FF));
  return m;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Shared {
  l0::SyncEventPool* ev = nullptr;
  uint8_t* rows[kL];                   // host USM: the GPU's copy of the rows (+ routing)
  uint16_t* res[kL];                   // host USM: the result the GPU copies back
  std::atomic<uint64_t> epoch{0};
  std::atomic<bool> quit{false};
  std::atomic<uint32_t> M{1}, K{0};
  std::atomic<int> mode{0};
};

void handoff_thread(Shared* sh, const std::vector<cpu::ExpertInt4>* experts, unsigned threads, std::vector<int> cpus) {
  cpu::Pool pool(threads, threads > 1 ? cpus : std::vector<int>{});
  cpu::ExpertScratch sc;
  std::vector<float> tmp(size_t(kMaxM) * kHidden), acc(size_t(kMaxM) * kHidden);
  uint64_t seen = 0, next_expert = 0;
  const cpu::Isa isa = cpu::Isa::Avx2;
  for (;;) {
    unsigned spins = 0;
    while (sh->epoch.load(std::memory_order_acquire) == seen && !sh->quit.load(std::memory_order_relaxed)) {
      if (++spins > 2000) std::this_thread::yield(); else _mm_pause();
    }
    if (sh->quit.load()) return;
    seen = sh->epoch.load(std::memory_order_acquire);
    const uint32_t M = sh->M.load(), K = sh->K.load();
    const Mode mode = Mode(sh->mode.load());
    if (mode == Mode::None) continue;
    for (uint32_t i = 0; i < kL; ++i) {
      const double w0 = now_s();
      while (!sh->ev->signalled(2 * i)) {
        if (now_s() - w0 > 5) { std::printf("host: layer %u never arrived\n", i); std::exit(3); }
        _mm_pause();
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint16_t* x = reinterpret_cast<const uint16_t*>(sh->rows[i]);
      std::fill(acc.begin(), acc.begin() + size_t(M) * kHidden, 0.f);
      for (uint32_t k = 0; k < K; ++k) {
        cpu::expert_forward((*experts)[next_expert++ % kPool], x, M, tmp.data(), sc, &pool, isa);
        for (size_t j = 0; j < size_t(M) * kHidden; ++j) acc[j] += tmp[j] / float(K);
      }
      for (size_t j = 0; j < size_t(M) * kHidden; ++j) sh->res[i][j] = K ? common::f32_to_bf16(acc[j]) : x[j];
      std::atomic_thread_fence(std::memory_order_release);
      sh->ev->host_signal(2 * i + 1);
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  unsigned threads = 22;
  int reps = 100;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = unsigned(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
  }
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("rt_probe"));
  std::printf("module loaded\n");
  l0::Kernel k_spin = mod.kernel("rt_spin");
  k_spin.group_size(16);

  const size_t rows_bytes = size_t(kMaxM) * kHidden * 2 + kRoute;
  l0::Mem dev_rows(ctx, l0::MemKind::Device, rows_bytes), dev_res(ctx, l0::MemKind::Device, rows_bytes);
  l0::Mem spin_out(ctx, l0::MemKind::Device, 64);
  l0::Mem host(ctx, l0::MemKind::Host, size_t(kL) * 2 * rows_bytes);
  l0::SyncEventPool evpool(ctx, {&ctx}, 2 * kL);
  {
    l0::CmdList im = l0::CmdList::immediate(ctx);
    std::vector<uint16_t> init(rows_bytes / 2);
    std::mt19937 r0(3);
    for (auto& v : init) v = uint16_t(0x3F00 + (r0() & 0x7F));
    im.copy(dev_rows.ptr(), init.data(), rows_bytes);
  }
  std::printf("buffers ready\n");
  Shared sh;
  sh.ev = &evpool;
  uint8_t* hp = host.as<uint8_t>();
  std::memset(hp, 0, host.size());
  for (uint32_t i = 0; i < kL; ++i) {
    sh.rows[i] = hp + size_t(i) * 2 * rows_bytes;
    sh.res[i] = reinterpret_cast<uint16_t*>(hp + size_t(i) * 2 * rows_bytes + rows_bytes);
  }

  std::mt19937_64 rng(1);
  std::vector<cpu::ExpertInt4> experts(kPool);
  for (auto& e : experts) { e.gate_up = random_matrix(kHidden, 2 * kInter, rng); e.down = random_matrix(kInter, kHidden, rng); }
  const std::vector<int> cpus = cpu::pick_cpus();
  std::printf("experts ready\n");
  std::thread th(handoff_thread, &sh, &experts, threads, cpus);

  auto time_list = [&](l0::CmdList& list, Mode mode, uint32_t M, uint32_t K, int n) {
    std::vector<double> us;
    sh.M = M; sh.K = K; sh.mode = int(mode);
    for (int r = 0; r < n + 10; ++r) {
      if (mode == Mode::Event) for (uint32_t e = 0; e < 2 * kL; ++e) evpool.host_reset(e);
      sh.epoch.fetch_add(1, std::memory_order_release);
      Timer t; t.start();
      q.execute(list, &fence);
      fence.wait();
      if (r >= 10) us.push_back(t.ms() * 1e3);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
  };
  auto build = [&](Mode mode, uint32_t M, uint32_t spin_iters) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    const size_t bo = size_t(M) * kHidden * 2 + kRoute, bi = size_t(M) * kHidden * 2;
    for (uint32_t i = 0; i < kL; ++i) {
      list.copy(sh.rows[i], dev_rows.ptr(), bo);
      if (mode == Mode::Event) list.barrier_signal(evpool.handle(2 * i));
      if (spin_iters) { k_spin.arg_ptr(0, spin_out.ptr()); k_spin.arg<uint32_t>(1, spin_iters); list.launch(k_spin, 1); }
      if (mode == Mode::Event) list.wait_event(evpool.handle(2 * i + 1));
      list.copy(dev_res.ptr(), sh.res[i], bi);
    }
    list.close();
    return list;
  };

  // The spin kernel's cost a step (one work-item, a dependent chain).
  double us_per_iter;
  {
    auto t = [&](uint32_t n) { l0::CmdList l = l0::CmdList::regular(ctx); k_spin.arg_ptr(0, spin_out.ptr()); k_spin.arg<uint32_t>(1, n); l.launch(k_spin, 1); l.close(); return time_list(l, Mode::None, 1, 0, 30); };
    us_per_iter = (t(400000) - t(100000)) / 300000.0;
  }
  std::printf("calibrated\n");
  std::printf("threads %u, L %u layers a replay, spin %.4f us a step; us per layer (median of %d replays)\n", threads, kL, us_per_iter, reps);
  std::printf("%-3s %-6s %-5s %-4s %9s %9s\n", "M", "mode", "D_us", "K", "us/layer", "exposed");
  for (uint32_t M : {1u, 8u})
    for (double D : {0.0, 150.0}) {
      const uint32_t it = uint32_t(D / us_per_iter);
      l0::CmdList base = build(Mode::None, M, it), le = build(Mode::Event, M, it);
      const double b = time_list(base, Mode::None, M, 0, reps) / kL;
      std::printf("%-3u %-6s %-5.0f %-4s %9.1f %9s\n", M, "none", D, "-", b, "-");
      for (uint32_t K : {0u, 1u, 2u, 4u, 7u}) {
        const double e = time_list(le, Mode::Event, M, K, reps) / kL;
        std::printf("%-3u %-6s %-5.0f %-4u %9.1f %9.1f\n", M, "event", D, K, e, e - b);
        std::fflush(stdout);
      }
    }
  sh.quit = true;
  th.join();
  return 0;
}
