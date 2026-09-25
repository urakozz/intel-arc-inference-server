#include "runtime/prefill/profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace runtime::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kN = uint32_t(Phase::kCount);

const char* const kName[kN] = {
    "norm",       "dequant",     "gemm",        "slab_dequant", "slab_gemm", "i8_quant",
    "i8_requant", "i8_gemm",     "ab_proj",     "gdn_seed",
    "gdn_conv",   "gdn_l2norm",  "gdn_gate",    "gdn_A",       "gdn_solve", "gdn_wu",
    "gdn_A2",     "gdn_scan",    "gdn_head",    "silu",        "attn_prep", "attn_QK^T",
    "attn_softmax", "attn_PV",   "attn_gate",   "attn_flash", "head"};

struct Acc {
  double ms[kN] = {};
  size_t waits[kN] = {};
  Context::LaunchMetrics launch[kN] = {};
};
Acc g_acc;

}  // namespace

bool profile_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PREFILL_PROFILE");
    return v != nullptr && std::strcmp(v, "1") == 0;
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
  g_acc.ms[i] += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  ++g_acc.waits[i];
  const auto metrics = cx.take_launch_metrics();
  g_acc.launch[i].gpu_ms += metrics.gpu_ms;
  g_acc.launch[i].host_submit_ms += metrics.host_submit_ms;
  g_acc.launch[i].launches += metrics.launches;
}

void profile_wait(Context& cx, Phase p) {
  if (!profile_enabled()) return;
  timed_wait(cx, p);
}

void profile_reset() { g_acc = Acc{}; }

void profile_report(const char* label, double wall_ms) {
  if (!profile_enabled()) return;
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
  std::fprintf(stderr,
               "  wait_ms is residual host synchronization time, not phase execution time.\n"
               "  L0_gpu_ms is kernel timestamp duration; SYCL kernels are not timestamped.\n"
               "  L0_submit_ms covers Context::launch, including event creation on first use.\n"
               "  GPU and host columns overlap: do not add them. Timestamp reads and other\n"
               "  host work are included only in call wall. Compare an independent unprofiled\n"
               "  run to measure instrumentation overhead. This table is diagnostic only.\n");
}

}  // namespace runtime::prefill
