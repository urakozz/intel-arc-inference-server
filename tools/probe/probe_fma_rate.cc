// probe_fma_rate - what fp32 VECTOR fma rate does this B70 sustain? (open-work 3m)
//
// Record: docs/probes/probe-gdn-prefill-2026-11-10.md. The GDN prefill kernels other than the
// split scan are plain fp32 vector code (fma chains, no DPAS), so their compute bound is this
// rate, not the 183.45 TFLOP/s bf16 DPAS peak. The clock-derived figure is 256 EUs x 16 lanes x
// 2 FLOP x 2.8 GHz = 22.94 TFLOP/s; this measures it, as probe_dpas_rate measures the DPAS one.
//
// Plain Level Zero, probe_dpas_rate's protocol: per grid, calibrate the trip count to ~40 ms,
// one discarded warm-up, then 8 replays of 2 launches each, first 3 discarded, median of the
// last 5. The output's checksum is printed (every accumulator converges to ~1, so the mean
// stored sum is ~16: a kernel that computed nothing would not print it).
//
// Device: run under tools/gpu_lock.sh (card N or box); `l0::Context(0)` takes the first device
// of the (possibly masked) enumeration.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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

constexpr int kReplays = 8;
constexpr int kDropped = 3;
constexpr int kEnqueues = 2;
constexpr double kTargetMs = 40.0;
constexpr uint32_t kCalibIters = 256;
constexpr uint32_t kWgSize = 256;
constexpr uint32_t kFmaPerIter = 64;   // UNROLL(4) * NACC(16) in the .cl
constexpr uint32_t kSrcWords = 4096;
constexpr uint32_t kGrids[] = {512, 1024, 2048, 4096, 8192};
constexpr uint32_t kMaxGroups = 8192;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main() {
  try {
    l0::Context ctx(0);
    l0::Queue q(ctx);
    l0::Fence fence(q);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    const char* mask = std::getenv("ZE_AFFINITY_MASK");

    std::printf("# probe_fma_rate: sustained fp32 vector fma rate, one B70\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", mask ? mask : "(unset)");
    std::printf("# L0 device: %s | EUs %u | max core clock %u MHz\n", ctx.name().c_str(),
                ctx.eu_count(), ctx.props().coreClockRate);
    const double derived = double(ctx.eu_count()) * 16 * 2 * ctx.props().coreClockRate * 1e6;
    std::printf("# clock-derived fp32 vector peak: EUs x 16 lanes x 2 FLOP x clock = %.2f TFLOP/s\n",
                derived / 1e12);

    l0::Mem src(ctx, l0::MemKind::Device, kSrcWords * sizeof(float));
    {
      std::vector<float> h(kSrcWords);
      for (uint32_t i = 0; i < kSrcWords; ++i) h[i] = 0.5f + float(i % 97) / 97.0f;
      imm.copy(src.ptr(), h.data(), h.size() * sizeof(float));
    }
    const size_t n_out = size_t(kMaxGroups) * kWgSize;
    l0::Mem dst(ctx, l0::MemKind::Device, n_out * sizeof(float));

    l0::Module mod(ctx, kernels::path("pfr_f32"));
    l0::Kernel k = mod.kernel("fma_rate");

    std::printf("\n| grid (WGs) | trip count | ms/launch | TFLOP/s | of derived |\n|---:|---:|---:|---:|---:|\n");
    double best = 0.0;
    for (uint32_t groups : kGrids) {
      auto timed = [&](uint32_t iters, int enqueues) {
        k.arg_ptr(0, src.ptr());
        k.arg_ptr(1, dst.ptr());
        k.arg(2, iters);
        k.group_size(kWgSize);
        l0::CmdList list = l0::CmdList::regular(ctx);
        for (int e = 0; e < enqueues; ++e) list.launch(k, groups);
        list.close();
        Timer t;
        t.start();
        q.execute(list, &fence);
        fence.wait();
        return t.ms() / enqueues;
      };
      const double calib = timed(kCalibIters, 1);
      const uint64_t want = uint64_t(double(kCalibIters) * kTargetMs / std::max(calib, 1e-3));
      const uint32_t iters = uint32_t(std::min<uint64_t>(std::max<uint64_t>(want, 1024), 4000000));
      timed(iters, kEnqueues);
      std::vector<double> s;
      for (int r = 0; r < kReplays; ++r) {
        const double ms = timed(iters, kEnqueues);
        if (r >= kDropped) s.push_back(ms);
      }
      const double ms = median(std::move(s));
      const double flop = double(groups) * kWgSize * double(iters) * kFmaPerIter * 2.0;
      const double rate = flop / (ms * 1e-3);
      best = std::max(best, rate);
      std::printf("| %u | %u | %.3f | %.2f | %.1f %% |\n", groups, iters, ms, rate / 1e12,
                  100.0 * rate / derived);
      std::fflush(stdout);
    }
    std::vector<float> h(size_t(kGrids[0]) * kWgSize);
    imm.copy(h.data(), dst.ptr(), h.size() * sizeof(float));
    double sum = 0.0;
    for (float v : h) sum += v;
    std::printf("\nfp32 vector fma peak: **%.2f TFLOP/s** (%.1f %% of the clock-derived %.2f)\n",
                best / 1e12, 100.0 * best / derived, derived / 1e12);
    std::printf("checksum: mean stored sum %.4f over %zu work-items (expect ~16)\n",
                sum / double(h.size()), h.size());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_fma_rate FAILED: %s\n", e.what());
    return 1;
  }
}
