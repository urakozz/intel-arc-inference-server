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
    "norm",       "dequant",     "gemm",        "ab_proj",     "gdn_seed",  "gdn_conv",
    "gdn_l2norm", "gdn_gate",    "gdn_A",       "gdn_solve",   "gdn_wu",    "gdn_A2",
    "gdn_scan",   "gdn_head",    "silu",        "attn_prep",   "attn_QK^T", "attn_softmax",
    "attn_PV",    "attn_gate",   "head"};

struct Acc {
  double ms[kN] = {};
  size_t waits[kN] = {};
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
}

void profile_wait(Context& cx, Phase p) {
  if (!profile_enabled()) return;
  timed_wait(cx, p);
}

void profile_reset() { g_acc = Acc{}; }

void profile_report(const char* label, double plain_ms) {
  if (!profile_enabled()) return;
  double total = 0;
  size_t waits = 0;
  for (uint32_t i = 0; i < kN; ++i) {
    total += g_acc.ms[i];
    waits += g_acc.waits[i];
  }
  std::fprintf(stderr,
               "\nprefill phase attribution -- %s (INSTRUMENTED: extra waits close each\n"
               "L0-only phase, so this total is an upper bound on the plain walk's)\n"
               "  phase           ms      share   waits    ms/wait\n",
               label);
  for (uint32_t i = 0; i < kN; ++i)
    std::fprintf(stderr, "  %-13s %8.1f   %5.1f%%  %6zu   %8.4f\n", kName[i], g_acc.ms[i],
                 total > 0 ? 100.0 * g_acc.ms[i] / total : 0.0, g_acc.waits[i],
                 g_acc.waits[i] ? g_acc.ms[i] / double(g_acc.waits[i]) : 0.0);
  std::fprintf(stderr, "  %-13s %8.1f   100.0%%  %6zu\n", "TOTAL", total, waits);
  if (plain_ms > 0.0)
    std::fprintf(stderr,
                 "  plain (unprofiled) run of the same shape: %.1f ms -- the instrument costs"
                 " %+.1f ms (%+.1f%%)\n",
                 plain_ms, total - plain_ms, 100.0 * (total - plain_ms) / plain_ms);
}

}  // namespace runtime::prefill
