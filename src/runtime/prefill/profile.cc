#include "runtime/prefill/profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace runtime::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kN = uint32_t(Phase::kCount);

const char* const kName[kN] = {
    "norm",       "dequant",     "gemm",        "slab_dequant", "slab_gemm", "i8_quant",
    "i8_requant", "i8_gemm",     "ab_proj",     "gdn_seed",
    "gdn_conv",   "gdn_l2norm",  "gdn_gate",    "gdn_A",       "gdn_solve", "gdn_wu",
    "gdn_A2",     "gdn_scan",    "gdn_head",    "silu",        "attn_prep", "attn_QK^T",
    "attn_softmax", "attn_PV",   "attn_gate",   "attn_flash", "head",
    "moe_route",  "moe_sort",    "moe_gather",  "moe_weights", "moe_gemm",  "moe_combine",
    "setup"};

struct Acc {
  double ms[kN] = {};
  size_t waits[kN] = {};
  Context::LaunchMetrics launch[kN] = {};
  // B70_PREFILL_PROFILE=2: (phase, entry) -> (launches, device ms).
  std::map<std::pair<uint32_t, std::string>, std::pair<size_t, double>> trace;
};
Acc g_acc;
// Plan 13b: two engines (--parallel replica) may prefill at once; the accumulators are guarded
// (only when B70_PREFILL_PROFILE=1 - the unprofiled path never takes it).
std::mutex g_acc_mu;

}  // namespace

bool profile_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PREFILL_PROFILE");
    return v != nullptr && (std::strcmp(v, "1") == 0 || std::strcmp(v, "2") == 0);
  }();
  return on;
}

void timed_wait(Context& cx, Phase p) {
  if (!profile_enabled()) {
    cx.wait();
    return;
  }
  const Clock::time_point t0 = Clock::now();
  cx.wait();
  const uint32_t i = uint32_t(p);
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::lock_guard<std::mutex> lock(g_acc_mu);
  g_acc.ms[i] += ms;
  ++g_acc.waits[i];
  const auto metrics = cx.take_launch_metrics();
  g_acc.launch[i].gpu_ms += metrics.gpu_ms;
  g_acc.launch[i].host_submit_ms += metrics.host_submit_ms;
  g_acc.launch[i].launches += metrics.launches;
  for (const auto& [name, gms] : metrics.each) {
    auto& t = g_acc.trace[{i, name}];
    ++t.first;
    t.second += gms;
  }
}

void profile_wait(Context& cx, Phase p) {
  if (!profile_enabled()) return;
  timed_wait(cx, p);
}

void profile_reset() {
  std::lock_guard<std::mutex> lock(g_acc_mu);
  g_acc = Acc{};
}

void profile_report(const char* label, double wall_ms) {
  if (!profile_enabled()) return;
  std::lock_guard<std::mutex> lock(g_acc_mu);
  double total = 0;
  size_t waits = 0;
  Context::LaunchMetrics launch;
  for (uint32_t i = 0; i < kN; ++i) {
    total += g_acc.ms[i];
    waits += g_acc.waits[i];
    launch.gpu_ms += g_acc.launch[i].gpu_ms;
    launch.host_submit_ms += g_acc.launch[i].host_submit_ms;
    launch.launches += g_acc.launch[i].launches;
  }
  std::fprintf(stderr,
               "\nprefill phase attribution -- %s (INSTRUMENTED: timestamp events and phase waits)\n"
               "  phase          wait_ms   L0_gpu_ms  L0_submit_ms   waits  L0_launches\n",
               label);
  for (uint32_t i = 0; i < kN; ++i)
    std::fprintf(stderr, "  %-13s %8.1f   %9.1f   %11.1f  %6zu  %11zu\n", kName[i],
                 g_acc.ms[i], g_acc.launch[i].gpu_ms, g_acc.launch[i].host_submit_ms,
                 g_acc.waits[i], g_acc.launch[i].launches);
  std::fprintf(stderr, "  %-13s %8.1f   %9.1f   %11.1f  %6zu  %11zu\n", "TOTAL",
               total, launch.gpu_ms, launch.host_submit_ms, waits, launch.launches);
  if (wall_ms > 0.0)
    std::fprintf(stderr, "  instrumented prefill call wall: %.1f ms (includes setup and host work)\n",
                 wall_ms);
  if (!g_acc.trace.empty()) {
    std::fprintf(stderr, "  per-launch trace (B70_PREFILL_PROFILE=2): phase, entry, launches, L0_gpu_ms\n");
    for (const auto& [key, t] : g_acc.trace)
      std::fprintf(stderr, "    %-13s %-36s %8zu %10.2f\n", kName[key.first], key.second.c_str(), t.first,
                   t.second);
  }
  std::fprintf(stderr,
               "  wait_ms is residual host synchronization time, not phase execution time.\n"
               "  L0_gpu_ms is kernel timestamp duration.\n"
               "  L0_submit_ms covers Context::launch, including event creation on first use.\n"
               "  GPU and host columns overlap: do not add them. Timestamp reads and other\n"
               "  host work are included only in call wall. Compare an independent unprofiled\n"
               "  run to measure instrumentation overhead. This table is diagnostic only.\n");
}

}  // namespace runtime::prefill
