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
#include "runtime/prefill/gemm.h"

namespace {

// gate‖up -- the largest prefill linear, the shape P3, P2 and the overlap probe
// all measured, and the one N that 2048 divides exactly (34816 = 2048 x 17).
constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;  // PrefillScratch::kC (ruling A13)
constexpr size_t kScratchBytes = size_t(kK) * kN * sizeof(uint16_t);  // 356,515,840
constexpr int kReplays = 8;
constexpr int kDropped = 3;
// P3's measured whole-chunk dequant, the quantity this lever is trying to move.
constexpr double kChunkDequantMs = 210.116;

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

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
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
  std::printf("# shape: gate|up K=%u N=%u layout 0 [K][N]; GEMM at M=%u; full scratch %zu B\n",
              kK, kN, kM, kScratchBytes);
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
  const size_t a_bytes = size_t(kM) * kK * sizeof(uint16_t);
  const size_t c_bytes = size_t(kM) * kN * sizeof(float);
  l0::Mem da(ctx, l0::MemKind::Device, a_bytes);
  l0::Mem dc(ctx, l0::MemKind::Device, c_bytes);
  {
    std::vector<uint16_t> a(size_t(kM) * kK);
    uint32_t s = 0x9e3779b9u;
    for (uint16_t& v : a) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      v = uint16_t((s & 0x807fu) | 0x3e00u);  // exponent 124: |x| in [0.125, 0.25)
    }
    l0::CmdList up = l0::CmdList::immediate(ctx);
    up.copy(da.ptr(), a.data(), a_bytes);
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

  // --- the four primitives --------------------------------------------------
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
  auto gemm_full = [&] {
    runtime::prefill::gemm_bf16(cx, {kM, kK, kN}, da.as<uint16_t>(), full.as<uint16_t>(),
                                dc.as<float>());
  };
  // B = the compact slab (row pitch ldb), C = the matching column block of the
  // full [M][N] output (row pitch kN). That is what the lever's GEMM does.
  auto gemm_slab = [&](const Slab& sl, const uint16_t* B, size_t ldb) {
    runtime::prefill::gemm_bf16_batched(
        cx, {kM, kK, sl.ns, 1, kK, ldb, kN, 0, 0, 0}, da.as<uint16_t>(), B,
        dc.as<float>() + sl.n0);
  };

  // --- controls C1 / C1' / C2 / C3 -----------------------------------------
  const double t_dq_full = replay([&] { full_dequant(); cx.wait(); });
  const double t_dq_full_slabk = replay([&] {
    slab_dequant({0, kN}, full.ptr());
    cx.wait();
  });
  const double t_gemm_full = replay([&] { gemm_full(); cx.wait(); });
  const double t_twopass = replay([&] {
    full_dequant();
    cx.wait();
    gemm_full();
    cx.wait();
  });
  const size_t read_bytes = w0.bytes();
  auto gbps = [](size_t bytes, double ms) { return double(bytes) / (ms * 1e6); };
  const double gflop = 2.0 * double(kM) * kK * kN / 1e9;

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
  std::printf("| C2 gemm_bf16 full width, 1088 WGs | %.3f | %.2f TFLOP/s | P2 measured 150.19, "
              "overlap probe 131.08-131.47 |\n", t_gemm_full, gflop / t_gemm_full);
  std::printf("| C3 two-pass total (C1 then C2) | %.3f | | sum model %.3f, %+.2f%% |\n",
              t_twopass, t_dq_full + t_gemm_full,
              100.0 * (t_twopass / (t_dq_full + t_gemm_full) - 1.0));

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

  // --- the per-Ns battery ---------------------------------------------------
  std::printf("\n## the slab battery (measured, iterate-grade)\n");
  std::printf("| Ns | slabs | slab B | C6 dequant-only ms | C6 GB/s r+w | C6 r+2w | "
              "C5a slab-GEMM ms | TFLOP/s | C5b with a wait per slab | C4 interleaved ms |\n");
  std::printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  struct Row { uint32_t ns; double c6, c5a, c5b, c4; size_t slabs; };
  std::vector<Row> rows;
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
    // C5a -- the slab GEMMs alone, from an already-filled compact slab, one
    // trailing wait (they are in-order on one queue, so none is needed between
    // them). This is the +10% occupancy check AND the guaranteed-warm case.
    slab_dequant(sl[0], slab.ptr());
    cx.wait();
    const double c5a = replay([&] {
      for (const Slab& s : sl) gemm_slab(s, slab.as<uint16_t>(), s.ns);
      cx.wait();
    });
    // C5b -- the same with a host wait after each, so (C5b - C5a) / slabs is a
    // MEASURED handoff cost rather than P1's 8.569 us carried over.
    const double c5b = replay([&] {
      for (const Slab& s : sl) {
        gemm_slab(s, slab.as<uint16_t>(), s.ns);
        cx.wait();
      }
    });
    // C4 -- the lever. One slab buffer, so the GEMM must finish before the next
    // dequant overwrites it: two waits per slab, and that cost is the lever's.
    const double c4 = replay([&] {
      for (const Slab& s : sl) {
        slab_dequant(s, slab.ptr());
        cx.wait();
        gemm_slab(s, slab.as<uint16_t>(), s.ns);
        cx.wait();
      }
    });
    const size_t slab_bytes = size_t(kK) * ns * sizeof(uint16_t);
    std::printf("| %u | %zu | %zu | %.3f | %.1f | %.1f | %.3f | %.2f | %.3f | %.3f |\n", ns,
                sl.size(), slab_bytes, c6, gbps(read_bytes + kScratchBytes, c6),
                gbps(read_bytes + 2 * kScratchBytes, c6), c5a, gflop / c5a, c5b, c4);
    rows.push_back({ns, c6, c5a, c5b, c4, sl.size()});
  }

  // --- R1: is an L2-resident B operand worth anything at all? ---------------
  // The literal control the brief asks for: the SAME slab GEMM twice
  // back-to-back, the first behind a full-width dequant whose 356 MB of writes
  // have just evicted the L2, the second guaranteed warm.
  {
    const Slab sl{0, 2048};
    slab_dequant(sl, slab.ptr());
    cx.wait();
    std::vector<double> first, second;
    for (int r = 0; r < kReplays + 1; ++r) {
      full_dequant();  // 356 MB of writes: the L2 is not the slab's any more
      cx.wait();
      auto t = std::chrono::steady_clock::now();
      gemm_slab(sl, slab.as<uint16_t>(), sl.ns);
      cx.wait();
      const double a = ms_since(t);
      t = std::chrono::steady_clock::now();
      gemm_slab(sl, slab.as<uint16_t>(), sl.ns);
      cx.wait();
      const double b = ms_since(t);
      if (r > kDropped) { first.push_back(a); second.push_back(b); }
    }
    const double f = median(first), s = median(second);
    std::printf("\nR1 (Ns=2048, one slab, L2 evicted by a full-width dequant between replays): "
                "first %.4f ms, second (guaranteed warm) %.4f ms, %+.2f%%\n",
                f, s, 100.0 * (s / f - 1.0));
  }

  // --- the verdict ----------------------------------------------------------
  std::printf("\n## verdict (pre-registered: overhead 210 -> <= 70 ms/chunk-equivalent; "
              "accept >= 90 ms saved, priced 30-90, dead < 30)\n");
  std::printf("| Ns | C4 - C2 = slab overhead ms | / C1 = ratio | x 210.116 = chunk-equiv ms | "
              "saved ms | C5a vs C2 | measured wait us |\n");
  std::printf("|---:|---:|---:|---:|---:|---:|---:|\n");
  for (const Row& r : rows) {
    const double overhead = r.c4 - t_gemm_full;
    const double ratio = overhead / t_dq_full;
    const double chunk_equiv = ratio * kChunkDequantMs;
    std::printf("| %u | %.3f | %.3f | %.1f | **%+.1f** | %+.1f%% | %.1f |\n", r.ns, overhead,
                ratio, chunk_equiv, kChunkDequantMs - chunk_equiv,
                100.0 * (r.c5a / t_gemm_full - 1.0), 1000.0 * (r.c5b - r.c5a) / double(r.slabs));
  }
  std::printf("\nchunk-equivalent uses THIS probe's own C1 and C2, so any harness-level GEMM "
              "difference cancels out of the ratio (derived from measured).\n");
  std::printf("every slab-GEMM slow-down is charged to the lever: it lands inside C4.\n");
  return 0;
}
