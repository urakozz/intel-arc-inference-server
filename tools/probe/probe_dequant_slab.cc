// Can the 210 ms dequant be AVOIDED rather than hidden? (spec 2, the last
// unpriced lever on the bf16 scratch.)
//
// P3 measured the whole-chunk dequant at 210.116 ms and identified the
// mechanism as DRAM write-allocate: `r + 2w` reads 506-529 GB/s on a 590 GB/s
// part while `r + w` reads only 283-296. The overlap probe then measured that
// the write cannot be HIDDEN (recovery 0.11-0.14 against a 0.3 bar), and PROBE
// B measured that the B70 has exactly one compute queue, which closed that
// route. This probe asks the remaining question: if the dequant writes an
// N-slab small enough to sit in the 24 MB L2 and the GEMM reads it back
// immediately, does the write ever reach DRAM at all?
//
// The named risk is that Xe2's L2 need not retain a kernel's writes for the
// next kernel. Control C6 measures that directly and needs no GEMM to do it.
//
// Pre-registration: docs/probe-dequant-slab-2026-09-05.md (dequant overhead
// 210 -> <= 70 ms/chunk-equivalent; accept at >= 90 ms saved, priced at 30-90,
// dead below 30), committed before this file.
//
// 2026-10-27: the sycl-tla GEMM is gone from the engine (operator ruling), and with it this
// probe's GEMM arm - C2, C3, C4, C5a/b, R1 and the verdict table that priced the lever
// against it. What stays is the dequant side: C1, C1', the slab kernel's bitwise check and
// C6 (the L2-retention question, which needs no GEMM). The verdict is the recorded one.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"

namespace {

// gate‖up -- the largest prefill linear, the shape P3, P2 and the overlap probe
// all measured, and the one N that 2048 divides exactly (34816 = 2048 x 17).
constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr size_t kScratchBytes = size_t(kK) * kN * sizeof(uint16_t);  // 356,515,840
constexpr int kReplays = 8;
constexpr int kDropped = 3;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// 8 replays, first 3 discarded, median of the last 5 -- P2's, P3's and the
// overlap probe's protocol, unchanged.
template <class F>
double replay(F&& body) {
  body();  // discarded warm-up
  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const auto begin = std::chrono::steady_clock::now();
    body();
    samples.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count());
  }
  samples.erase(samples.begin(), samples.begin() + kDropped);
  return median(std::move(samples));
}

struct Slab {
  uint32_t n0, ns;
};

// N split into slabs of `ns`, with whatever remainder N leaves. 34816 is
// 17 x 2048 and 34 x 1024 exactly; at ns = 4096 it is 8 x 4096 + 1 x 2048, and
// that ragged last slab is kept rather than hidden -- it is what every matrix
// whose N is not a multiple of the slab width would look like in production.
std::vector<Slab> plan_slabs(uint32_t N, uint32_t ns) {
  std::vector<Slab> out;
  for (uint32_t n0 = 0; n0 < N;) {
    const uint32_t w = std::min(ns, N - n0);
    out.push_back({n0, w});
    n0 += w;
  }
  return out;
}

}  // namespace

int main() {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  std::printf("# P-slab: does an L2-sized dequant slab keep the write out of DRAM? "
              "(measured, iterate-grade)\n");
  std::printf("# L0 device: %s\n", ctx.name().c_str());
  std::printf("# shape: gate|up K=%u N=%u layout 0 [K][N]; full scratch %zu B\n", kK, kN,
              kScratchBytes);
  std::printf("# %d replays, discard first %d, median of last %d; one discarded warm-up.\n",
              kReplays, kDropped, kReplays - kDropped);

  // --- allocations ----------------------------------------------------------
  const common::Int4Gptq w0 = common::Int4Gptq::random(kK, kN, 42);
  l0::Mem dq(ctx, l0::MemKind::Device, w0.qweight.size() * sizeof(uint32_t));
  l0::Mem ds(ctx, l0::MemKind::Device, w0.scales.size() * sizeof(uint16_t));
  l0::Mem full(ctx, l0::MemKind::Device, kScratchBytes);
  // One slab buffer, sized for the widest cell. The pre-registered lever uses
  // ONE slab, not two: a second would double the working set to 42 MB and break
  // the very L2 residency the lever is about.
  const uint32_t kMaxNs = 4096;
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(kK) * kMaxNs * sizeof(uint16_t));
  {
    l0::CmdList up = l0::CmdList::immediate(ctx);
    up.copy(dq.ptr(), w0.qweight.data(), w0.qweight.size() * sizeof(uint32_t));
    up.copy(ds.ptr(), w0.scales.data(), w0.scales.size() * sizeof(uint16_t));
  }

  // The production dequant, and the probe's slab kernel at every width used.
  l0::Module prod_mod(ctx, kernels::path("pf_dequant_tile_K" + std::to_string(kK) + "_N" +
                                         std::to_string(kN) + "_L0_T0"));
  l0::Kernel prod = prod_mod.kernel("pf_dequant_tile");
  // l0::Module is neither copyable nor movable, so these are held by pointer
  // rather than by value in a vector that would have to move them.
  const uint32_t kWidths[] = {1024, 2048, 4096, kN};
  constexpr size_t kNumWidths = sizeof kWidths / sizeof *kWidths;
  std::vector<std::unique_ptr<l0::Module>> slab_mods;
  std::vector<std::unique_ptr<l0::Kernel>> slab_ks;
  for (uint32_t ns : kWidths) {
    slab_mods.push_back(std::make_unique<l0::Module>(
        ctx, kernels::path("pds_K" + std::to_string(kK) + "_N" + std::to_string(kN) + "_NS" +
                           std::to_string(ns))));
    slab_ks.push_back(
        std::make_unique<l0::Kernel>(slab_mods.back()->kernel("pf_dequant_slab")));
  }
  auto kernel_for = [&](uint32_t ns) -> l0::Kernel& {
    for (size_t i = 0; i < kNumWidths; ++i)
      if (kWidths[i] == ns) return *slab_ks[i];
    std::printf("no pds binary for NS=%u -- the CMake width list and this probe disagree\n", ns);
    std::exit(4);
  };

  // --- the two primitives ---------------------------------------------------
  auto full_dequant = [&] {
    void* w = dq.ptr(); void* s = ds.ptr(); void* out = full.ptr();
    cx.launch(prod, kN / 16, kK / 64, 1,
              {{&w, sizeof w}, {&s, sizeof s}, {&out, sizeof out}});
  };
  // One slab of `ns` columns starting at n0, into `dst` with row pitch `ns`.
  auto slab_dequant = [&](const Slab& sl, void* dst) {
    void* w = dq.ptr(); void* s = ds.ptr();
    const uint32_t n0 = sl.n0;
    cx.launch(kernel_for(sl.ns), sl.ns / 16, kK / 64, 1,
              {{&w, sizeof w}, {&s, sizeof s}, {&dst, sizeof dst}, {&n0, sizeof n0}});
  };
  // --- controls C1 / C1' ----------------------------------------------------
  const double t_dq_full = replay([&] { full_dequant(); cx.wait(); });
  const double t_dq_full_slabk = replay([&] {
    slab_dequant({0, kN}, full.ptr());
    cx.wait();
  });
  const size_t read_bytes = w0.bytes();
  auto gbps = [](size_t bytes, double ms) { return double(bytes) / (ms * 1e6); };

  std::printf("\n## controls (measured, iterate-grade)\n");
  std::printf("| control | ms | rate | reference |\n|---|---:|---:|---|\n");
  std::printf("| C1 full dequant, production pf_dequant_tile | %.3f | %.1f GB/s r+w, %.1f r+2w | "
              "P3 measured 1.539 ms / 293.1 / 524.7 |\n",
              t_dq_full, gbps(read_bytes + kScratchBytes, t_dq_full),
              gbps(read_bytes + 2 * kScratchBytes, t_dq_full));
  std::printf("| C1' full dequant, THIS probe's slab kernel at NS=N | %.3f | %.1f GB/s r+w | "
              "equivalence control vs C1: %+.2f%% |\n",
              t_dq_full_slabk, gbps(read_bytes + kScratchBytes, t_dq_full_slabk),
              100.0 * (t_dq_full_slabk / t_dq_full - 1.0));

  // --- correctness: the slab kernel must agree with the production one -------
  // Bitwise, over two slabs of the pre-registered width. A lever that changes a
  // bit is not a lever.
  {
    const uint32_t ns = 2048;
    full_dequant();
    cx.wait();
    bool ok = true;
    size_t mismatches = 0;
    const size_t slab_bytes = size_t(kK) * ns * sizeof(uint16_t);
    l0::Mem h_slab(ctx, l0::MemKind::Host, slab_bytes);
    l0::Mem h_ref(ctx, l0::MemKind::Host, kScratchBytes);
    l0::CmdList::immediate(ctx).copy(h_ref.ptr(), full.ptr(), kScratchBytes);
    const uint16_t* ref = h_ref.as<uint16_t>();
    for (uint32_t s : {0u, 8u, 16u}) {
      const Slab sl{s * ns, ns};
      slab_dequant(sl, slab.ptr());
      cx.wait();
      l0::CmdList::immediate(ctx).copy(h_slab.ptr(), slab.ptr(), slab_bytes);
      const uint16_t* got = h_slab.as<uint16_t>();
      for (uint32_t k = 0; k < kK; ++k)
        for (uint32_t j = 0; j < ns; ++j)
          if (got[size_t(k) * ns + j] != ref[size_t(k) * kN + sl.n0 + j]) { ++mismatches; ok = false; }
    }
    std::printf("\nslab kernel vs production kernel, slabs 0/8/16 at NS=2048, "
                "%zu weights compared: %s (%zu mismatches)\n",
                size_t(3) * kK * ns, ok ? "**bitwise identical**" : "**DIFFER**", mismatches);
    if (!ok) return 1;
  }

  // --- the per-Ns battery: C6 -------------------------------------------------
  std::printf("\n## the slab battery (measured, iterate-grade)\n");
  std::printf("| Ns | slabs | slab B | C6 dequant-only ms | C6 GB/s r+w | C6 r+2w |\n");
  std::printf("|---:|---:|---:|---:|---:|---:|\n");
  for (uint32_t ns : {1024u, 2048u, 4096u}) {
    const std::vector<Slab> sl = plan_slabs(kN, ns);
    // C6 -- the named risk, measured with NO consumer. Every slab of the source
    // is read exactly once (94.70 MB, cold), and every slab is written into the
    // SAME buffer. Nominal traffic is identical to C1's; if the L2 absorbs the
    // writes this is far faster, and if it does not this is C1.
    const double c6 = replay([&] {
      for (const Slab& s : sl) slab_dequant(s, slab.ptr());
      cx.wait();
    });
    const size_t slab_bytes = size_t(kK) * ns * sizeof(uint16_t);
    std::printf("| %u | %zu | %zu | %.3f | %.1f | %.1f |\n", ns, sl.size(), slab_bytes, c6,
                gbps(read_bytes + kScratchBytes, c6), gbps(read_bytes + 2 * kScratchBytes, c6));
  }
  return 0;
}
