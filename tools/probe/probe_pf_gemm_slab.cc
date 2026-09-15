// P-B: slab-interleaved dequant + P-A's `pf_gemm_bf16_slab` on ONE Level Zero
// in-order immediate list, with ZERO host waits between the dequant and the
// GEMM of each slab -- both are now raw L0 launches (A24's blocker, a
// cross-runtime L0<->SYCL host wait, does not exist for this pair). Runs
// only because P-A measured 156.53 TFLOP/s >= the brief's 128 bar.
//
// Pre-registration: docs/probe-prefill-vllm-parity-2026-09-14.md addendum
// §A3.2, committed before this file was built or run. Baseline and battery
// follow `probe_dequant_slab.cc`'s protocol and naming (C1/C2/C3/C4) with
// ONE change: the GEMM everywhere is `pf_gemm_bf16`/`pf_gemm_bf16_slab`
// (P-A's own kernel, raw L0), not sycl-tla's `gemm_bf16` -- the whole point
// of P-A existing is that the GEMM no longer needs the SYCL queue at all.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

// gate‖up -- the shape A24's Probe A used, and the one 2048 divides exactly.
constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;  // PrefillScratch::kC
constexpr size_t kScratchBytes = size_t(kK) * kN * sizeof(uint16_t);
constexpr int kReplays = 8;
constexpr int kDropped = 3;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class F>
double replay(F&& body) {
  body();  // discarded warm-up
  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    samples.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  samples.erase(samples.begin(), samples.begin() + kDropped);
  return median(std::move(samples));
}

struct Slab {
  uint32_t n0, ns;
};
std::vector<Slab> plan_slabs(uint32_t N, uint32_t ns) {
  std::vector<Slab> out;
  for (uint32_t n0 = 0; n0 < N;) {
    const uint32_t w = std::min(ns, N - n0);
    out.push_back({n0, w});
    n0 += w;
  }
  return out;
}

// zeEventQueryKernelTimestamp directly on the public handle -- l0::Event
// exposes only a self-duration (event.h), which is not what a CROSS-kernel
// gap needs; this is the same call `tests/l0/event_test.cc` makes.
uint64_t raw_end(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelEnd & mask;
}
uint64_t raw_start(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelStart & mask;
}

}  // namespace

int main() {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  std::printf("# P-B: slab dequant + pf_gemm_bf16_slab, ONE L0 list, zero host waits\n");
  std::printf("# L0 device: %s\n", ctx.name().c_str());
  std::printf("# shape: gate|up K=%u N=%u layout 0 [K][N]; GEMM at M=%u\n", kK, kN, kM);

  const common::Int4Gptq w0 = common::Int4Gptq::random(kK, kN, 42);
  l0::Mem dq(ctx, l0::MemKind::Device, w0.qweight.size() * sizeof(uint32_t));
  l0::Mem ds(ctx, l0::MemKind::Device, w0.scales.size() * sizeof(uint16_t));
  l0::Mem full(ctx, l0::MemKind::Device, kScratchBytes);
  const uint32_t kMaxNs = 2048;
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(kK) * kMaxNs * sizeof(uint16_t));
  const size_t a_bytes = size_t(kM) * kK * sizeof(uint16_t);
  const size_t c_bytes = size_t(kM) * kN * sizeof(float);
  l0::Mem da(ctx, l0::MemKind::Device, a_bytes);
  l0::Mem dc_full(ctx, l0::MemKind::Device, c_bytes);
  l0::Mem dc_lever(ctx, l0::MemKind::Device, c_bytes);
  {
    std::vector<uint16_t> a(size_t(kM) * kK);
    uint32_t s = 0x9e3779b9u;
    for (uint16_t& v : a) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      v = uint16_t((s & 0x807fu) | 0x3e00u);
    }
    l0::CmdList up = l0::CmdList::immediate(ctx);
    up.copy(da.ptr(), a.data(), a_bytes);
    up.copy(dq.ptr(), w0.qweight.data(), w0.qweight.size() * sizeof(uint32_t));
    up.copy(ds.ptr(), w0.scales.data(), w0.scales.size() * sizeof(uint16_t));
  }

  l0::Module prod_mod(ctx, kernels::path("pf_dequant_tile_K5120_N34816_L0_T0"));
  l0::Kernel prod = prod_mod.kernel("pf_dequant_tile");
  l0::Module pfmod(ctx, kernels::path("pf_gemm_bf16"));
  l0::Kernel pf_full = pfmod.kernel("pf_gemm_bf16");
  l0::Kernel pf_slab = pfmod.kernel("pf_gemm_bf16_slab");

  const uint32_t kWidths[] = {1024, 2048};
  constexpr size_t kNumWidths = sizeof kWidths / sizeof *kWidths;
  std::vector<std::unique_ptr<l0::Module>> slab_mods;
  std::vector<std::unique_ptr<l0::Kernel>> slab_ks;
  for (uint32_t ns : kWidths) {
    slab_mods.push_back(
        std::make_unique<l0::Module>(ctx, kernels::path("pds_K5120_N34816_NS" + std::to_string(ns))));
    slab_ks.push_back(std::make_unique<l0::Kernel>(slab_mods.back()->kernel("pf_dequant_slab")));
  }
  auto kernel_for = [&](uint32_t ns) -> l0::Kernel& {
    for (size_t i = 0; i < kNumWidths; ++i)
      if (kWidths[i] == ns) return *slab_ks[i];
    std::printf("no pds binary for NS=%u\n", ns);
    std::exit(4);
  };

  auto full_dequant = [&] {
    cx.launch(prod, kN / 16, kK / 64, 1, {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(full.ptr())});
  };
  auto slab_dequant = [&](const Slab& sl, void* dst, l0::Event* signal = nullptr) {
    const uint32_t n0 = sl.n0;
    cx.launch(kernel_for(sl.ns), sl.ns / 16, kK / 64, 1,
              {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(dst), arg_val(n0)}, signal);
  };
  auto gemm_full = [&] {
    cx.launch(pf_full, kM / 256, kN / 256, 1,
              {PtrArg(da.ptr()), PtrArg(full.ptr()), PtrArg(dc_full.ptr()), arg_val(kM), arg_val(kK),
               arg_val(kN)});
  };
  // B = the compact slab, ldb = sl.ns (implicit, pf_gemm_bf16_slab's own N
  // argument). C = the matching column block of the full [M][N] output,
  // ldc = kN (passed explicitly, decoupled from the grid's N extent).
  auto gemm_slab = [&](const Slab& sl, const uint16_t* B, float* C_at_n0,
                       l0::Event* signal = nullptr) {
    cx.launch(pf_slab, kM / 256, sl.ns / 256, 1,
              {PtrArg(da.ptr()), PtrArg(B), PtrArg(C_at_n0), arg_val(kM), arg_val(kK), arg_val(sl.ns),
               arg_val(kN)},
              signal);
  };

  // --- controls, THIS probe's own harness -----------------------------------
  const double t_dq_full = replay([&] { full_dequant(); cx.wait(); });
  const double t_gemm_full = replay([&] { gemm_full(); cx.wait(); });
  const double t_twopass = replay([&] {
    full_dequant();
    cx.wait();
    gemm_full();
    cx.wait();
  });
  const double gflop = 2.0 * double(kM) * kK * kN / 1e9;
  std::printf("\n## controls (measured, this probe's own harness)\n");
  std::printf("| control | ms | rate |\n|---|---:|---:|\n");
  std::printf("| C1 full dequant (pf_dequant_tile) | %.3f | - |\n", t_dq_full);
  std::printf("| C2 pf_gemm_bf16 full width | %.3f | %.2f TFLOP/s |\n", t_gemm_full, gflop / t_gemm_full);
  std::printf("| C3 two-pass, host wait between (C1 then C2) | %.3f | sum model %.3f, %+.2f%% |\n",
              t_twopass, t_dq_full + t_gemm_full, 100.0 * (t_twopass / (t_dq_full + t_gemm_full) - 1.0));

  // --- correctness: slab GEMM vs full GEMM over the SAME columns ------------
  bool correctness_ok = true;
  {
    full_dequant();
    cx.wait();
    gemm_full();
    cx.wait();
    const Slab sl{0, 2048};
    slab_dequant(sl, slab.ptr());
    cx.wait();
    gemm_slab(sl, slab.as<uint16_t>(), dc_lever.as<float>());
    cx.wait();
    // Compare dc_lever's [kM][0..2048) window against dc_full's SAME window
    // (both pitched at kN=34816 fp32 elements per row).
    l0::Mem h1(ctx, l0::MemKind::Host, size_t(kM) * sl.ns * sizeof(float));
    l0::Mem h2(ctx, l0::MemKind::Host, size_t(kM) * sl.ns * sizeof(float));
    for (uint32_t r = 0; r < kM; ++r) {
      l0::CmdList c = l0::CmdList::immediate(ctx);
      c.copy(static_cast<char*>(h1.ptr()) + size_t(r) * sl.ns * 4,
             static_cast<const char*>(dc_full.ptr()) + size_t(r) * kN * 4, sl.ns * 4);
      c.copy(static_cast<char*>(h2.ptr()) + size_t(r) * sl.ns * 4,
             static_cast<const char*>(dc_lever.ptr()) + size_t(r) * kN * 4, sl.ns * 4);
    }
    correctness_ok = std::memcmp(h1.ptr(), h2.ptr(), h1.size()) == 0;
    std::printf("\nslab GEMM (pf_gemm_bf16_slab, Ns=2048, n0=0) vs full GEMM (pf_gemm_bf16), "
                "%zu output elements: %s\n",
                size_t(kM) * sl.ns, correctness_ok ? "**bitwise identical**" : "**DIFFER**");
  }

  // --- device-timestamp handoff verification: ONE representative pair -------
  // A24's killer was a 22.35 us HOST wait between an L0 dequant and a SYCL
  // GEMM. Both kernels here are L0 launches on the SAME in-order list; the
  // timed region contains zero zeCommandListHostSynchronize/queue::wait calls
  // between them (by construction: no cx.wait() call appears between the two
  // launch() calls below). The device-side gap between the two events proves
  // it is not merely "no code calls wait", but that no execution gap opened.
  {
    const l0::TimerCalib calib = l0::TimerCalib::query(ctx);
    const uint64_t mask = calib.mask();
    l0::EventPool pool(ctx, 2);
    l0::Event e_dq(pool, 0), e_gemm(pool, 1);
    const Slab sl{0, 2048};
    slab_dequant(sl, slab.ptr(), &e_dq);
    gemm_slab(sl, slab.as<uint16_t>(), dc_lever.as<float>(), &e_gemm);
    cx.wait();
    const uint64_t dq_end = raw_end(e_dq, mask);
    const uint64_t gemm_start = raw_start(e_gemm, mask);
    const double gap_us = double((gemm_start - dq_end) & mask) / calib.cycles_per_us;
    std::printf("\nhandoff (device timestamps, Ns=2048): dequant end -> GEMM start = **%.3f us** "
                "(A24's host-wait handoff was 22.35 us; pre-registered pass: <= 5 us)\n",
                gap_us);
  }

  // --- the per-Ns battery: C4, the lever, ZERO host waits within a pair ----
  std::printf("\n## the lever (measured, iterate grade - RECORD if preflight was clean)\n");
  std::printf("| Ns | slabs | C4 zero-wait interleaved ms | saving vs C3 (this harness) ms | "
              "chunk-equiv (x C1 baseline) |\n");
  std::printf("|---:|---:|---:|---:|---:|\n");
  for (uint32_t ns : {1024u, 2048u}) {
    const std::vector<Slab> sl = plan_slabs(kN, ns);
    const double c4 = replay([&] {
      for (const Slab& s : sl)
        // NO cx.wait() between these two launches: both are raw L0 appends
        // to the same in-order immediate list, so the driver orders them
        // with no host round trip at all.
        {
          slab_dequant(s, slab.ptr());
          gemm_slab(s, slab.as<uint16_t>(), dc_lever.as<float>() + s.n0);
        }
      cx.wait();  // one trailing wait for the whole battery
    });
    const double saving = t_twopass - c4;
    std::printf("| %u | %zu | %.3f | **%+.3f** | %+.1f%% of C1 |\n", ns, sl.size(), c4, saving,
                100.0 * saving / t_dq_full);
  }

  std::printf("\ncorrectness: %s\n", correctness_ok ? "PASS" : "**FAIL**");
  return correctness_ok ? 0 : 1;
}
