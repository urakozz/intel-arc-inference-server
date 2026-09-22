// S1 probe - fused int4 dequant in pf_gemm.
// Pre-registration: docs/superpowers/specs/2026-09-22-fused-dequant-gemm-probe-design.md,
// committed at 107e132 before this file was written or run. Program context:
// 2026-09-22-prefill-parity-program-design.md §3, stage S1.
//
// ONE shape (design §4): gate‖up, K = 5120, N = 34816, M = 2048, layout 0 --
// the largest linear and 1/3 of all linear FLOPs. Three measurements in one
// process, same buffers, same device, same harness:
//
//   control  pf_dequant_slab then pf_gemm_T0 per 1024-column slab, exactly what
//            runtime::prefill::linear_l0 does today (34 slabs, 68 launches).
//   V1       pfd_gemm_v1, registers-only fused dequant, ONE launch, no slab.
//   V2       pfd_gemm_v2, SLM staged and double buffered, ONE launch.
//
// Equality is a memcmp of the whole [M][N] fp32 output against the control's,
// not a tolerance: design §2 fixes the dequant expression, so a fused kernel
// that is faster but not bitwise equal is a FAILED probe, not a trade-off.
//
// Timing is L0 kernel timestamps (not the host clock): best of 5 after one
// discarded warm-up. For the control both the SUM of the 68 kernel durations
// (the quantity docs/prefill-parity-2026-09-20.md attributed as
// slab_dequant + slab_gemm, and therefore the one the pre-registered 122.9
// TFLOP/s break-even is derived from) and the first-start-to-last-end SPAN are
// reported; for a single-launch variant the two coincide.
//
// PROBE-ONLY: no production file is edited by this probe. The control runs the
// production binaries themselves (pf_dequant_slab_K5120_N34816_L0, pf_gemm_T0).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/int4.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
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

constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;             // PrefillScratch::kC, the prefill chunk width
constexpr uint32_t kNs = 1024;            // kernels::kPfSlabWidth
constexpr uint32_t kTile = 256;           // pf_gemm's WG_M / WG_N
constexpr uint32_t kSlabs = kN / kNs;     // 34
constexpr int kReplays = 5;               // best of 5 after one warm-up (design §4)
constexpr size_t kStageBytes = 64ul << 20;

// 730.1 GFLOP per call, derived: 2 x 2048 x 5120 x 34816.
constexpr double kGflop = 2.0 * double(kM) * double(kK) * double(kN) / 1e9;

// Restated from design §3, before any measurement.
constexpr double kBreakEven = 122.9;   // TFLOP/s; below this the probe is a failure
constexpr double kTarget = 140.0;      // TFLOP/s; at or above this, write the adoption spec

const char* env_or_unset(const char* name) {
  const char* v = std::getenv(name);
  return v ? v : "(unset)";
}

float bits_to_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint32_t xorshift(uint32_t& v) {
  v ^= v << 13;
  v ^= v >> 17;
  v ^= v << 5;
  return v;
}

// Exponent fixed at 124: magnitude in [0.125, 0.25), the generator every
// prefill GEMM probe here uses, so no K = 5120 dot product leaves fp32 range.
uint16_t random_bf16(uint32_t& s) { return uint16_t((xorshift(s) & 0x807fu) | 0x3e00u); }

uint64_t raw_start(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelStart & mask;
}
uint64_t raw_end(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelEnd & mask;
}

struct Timed {
  double sum_ms = 0.0;     // Σ kernel durations - the docs' slab_dequant + slab_gemm quantity
  double span_ms = 0.0;    // first kernelStart .. last kernelEnd on the in-order list
  double even_ms = 0.0;    // control only: Σ over the dequant launches
  double odd_ms = 0.0;     // control only: Σ over the GEMM launches
  double tflops() const { return kGflop / sum_ms; }
};

// Best (minimum Σ) of kReplays after one discarded warm-up. `run()` appends its
// launches signalling ev[0..n-1] on the in-order list, waits, and returns n.
template <class F>
Timed best_of(F run, std::vector<l0::Event>& ev, const l0::TimerCalib& calib) {
  const uint64_t mask = calib.mask();
  Timed best;
  bool have = false;
  for (int r = -1; r < kReplays; ++r) {
    for (l0::Event& e : ev) e.reset();
    const size_t n = run();
    Timed cur;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double us =
          double((raw_end(ev[i], mask) - raw_start(ev[i], mask)) & mask) / calib.cycles_per_us;
      sum += us;
      if (i % 2 == 0) cur.even_ms += us; else cur.odd_ms += us;
    }
    cur.sum_ms = sum / 1000.0;
    cur.even_ms /= 1000.0;
    cur.odd_ms /= 1000.0;
    // The list is in-order, so the span is the first kernel's start to the last
    // kernel's end.
    cur.span_ms = double((raw_end(ev[n - 1], mask) - raw_start(ev[0], mask)) & mask) /
                  calib.cycles_per_us / 1000.0;
    if (r < 0) continue;   // discarded warm-up
    if (!have || cur.sum_ms < best.sum_ms) {
      best = cur;
      have = true;
    }
  }
  return best;
}

struct Diff {
  bool equal = true;
  size_t index = 0;
  uint32_t a = 0, b = 0;
};

// memcmp over M x N fp32, staged 64 MB at a time. On a difference the first
// differing 32-bit word is reported as raw bits, so a sign-of-zero difference
// (0x00000000 vs 0x80000000, which == would hide) is visible.
Diff compare_c(l0::Context& ctx, const void* x, const void* y, size_t bytes) {
  const size_t chunk = std::min(kStageBytes, bytes);
  l0::Mem hx(ctx, l0::MemKind::Host, chunk);
  l0::Mem hy(ctx, l0::MemKind::Host, chunk);
  for (size_t off = 0; off < bytes; off += chunk) {
    const size_t now = std::min(chunk, bytes - off);
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(hx.ptr(), static_cast<const char*>(x) + off, now);
    c.copy(hy.ptr(), static_cast<const char*>(y) + off, now);
    if (std::memcmp(hx.ptr(), hy.ptr(), now) == 0) continue;
    const uint32_t* px = hx.as<uint32_t>();
    const uint32_t* py = hy.as<uint32_t>();
    for (size_t i = 0; i < now / 4; ++i)
      if (px[i] != py[i]) {
        Diff d;
        d.equal = false;
        d.index = off / 4 + i;
        d.a = px[i];
        d.b = py[i];
        return d;
      }
  }
  return Diff{};
}

void row(const char* label, const Timed& t) {
  std::printf("| %s | %.3f | %.3f | **%.2f** |\n", label, t.sum_ms, t.span_ms, t.tflops());
  std::fflush(stdout);
}

void diff_row(const char* label, const Diff& d) {
  if (d.equal) {
    std::printf("| %s | **bitwise identical** | - | - | - |\n", label);
    return;
  }
  std::printf("| %s | **DIFFERS** | %zu (m=%zu, n=%zu) | 0x%08X (%.9g) | 0x%08X (%.9g) |\n", label,
              d.index, d.index / kN, d.index % kN, d.a, double(bits_to_f32(d.a)), d.b,
              double(bits_to_f32(d.b)));
}

}  // namespace

int main() {
  try {
    std::printf("# S1 probe: fused int4 dequant in pf_gemm (design 2026-09-22 §4)\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    l0::Context ctx(0);
    std::printf("# L0 device: %s\n", ctx.name().c_str());
    std::printf("# shape: gate||up K=%u N=%u M=%u layout 0; %.1f GFLOP per call "
                "(derived: 2 x %u x %u x %u)\n",
                kK, kN, kM, kGflop, kM, kK, kN);
    std::printf("# pre-registered break-even %.1f TFLOP/s; target %.1f TFLOP/s (design §3)\n",
                kBreakEven, kTarget);
    std::printf("# timing: L0 kernel timestamps, best of %d after one discarded warm-up\n",
                kReplays);

    runtime::prefill::Context cx(ctx);

    // --- inputs -------------------------------------------------------------
    const common::Int4Gptq w = common::Int4Gptq::random(kK, kN, 42);
    const size_t q_bytes = w.qweight.size() * sizeof(uint32_t);
    const size_t s_bytes = w.scales.size() * sizeof(uint16_t);
    const size_t a_bytes = size_t(kM) * kK * sizeof(uint16_t);
    const size_t c_bytes = size_t(kM) * kN * sizeof(float);
    l0::Mem dq(ctx, l0::MemKind::Device, q_bytes);
    l0::Mem ds(ctx, l0::MemKind::Device, s_bytes);
    l0::Mem da(ctx, l0::MemKind::Device, a_bytes);
    l0::Mem slab(ctx, l0::MemKind::Device, size_t(kK) * kNs * sizeof(uint16_t));
    l0::Mem c_ctl(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_v1(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_v2(ctx, l0::MemKind::Device, c_bytes);
    {
      std::vector<uint16_t> a(size_t(kM) * kK);
      uint32_t seed = 0x9e3779b9u;
      for (uint16_t& v : a) v = random_bf16(seed);
      l0::CmdList up = l0::CmdList::immediate(ctx);
      up.copy(da.ptr(), a.data(), a_bytes);
      up.copy(dq.ptr(), w.qweight.data(), q_bytes);
      up.copy(ds.ptr(), w.scales.data(), s_bytes);
    }

    // --- kernels ------------------------------------------------------------
    l0::Module m_dq(ctx, kernels::path(kernels::pf_dequant_slab_variant(kK, kN, 0)));
    l0::Kernel k_dq = m_dq.kernel("pf_dequant_slab");
    l0::Module m_gemm(ctx, kernels::path(kernels::pf_gemm_variant(false)));
    l0::Kernel k_gemm = m_gemm.kernel("pf_gemm");
    l0::Module m_v1(ctx, kernels::path("pfd_gemm_v1"));
    l0::Kernel k_v1 = m_v1.kernel("pfd_gemm");
    l0::Module m_v2(ctx, kernels::path("pfd_gemm_v2"));
    l0::Kernel k_v2 = m_v2.kernel("pfd_gemm");

    l0::EventPool pool(ctx, 2 * kSlabs);
    std::vector<l0::Event> ev;
    ev.reserve(2 * kSlabs);
    for (uint32_t i = 0; i < 2 * kSlabs; ++i) ev.emplace_back(pool, i);
    const l0::TimerCalib calib = pool.calib();

    // --- the control: linear_l0's two-pass slab walk, launch for launch ------
    const uint32_t lda = kK, ldc = kN;
    const uint64_t zero = 0;
    auto run_control = [&]() -> size_t {
      float* out = c_ctl.as<float>();
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        cx.launch(k_dq, kNs / 16, kK / 64, 1,
                  {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0)},
                  &ev[2 * i]);
        // B = the slab [K][1024] (ldb 1024); C = c_ctl from column n0 (ldc N).
        cx.launch(k_gemm, kM / kTile, kNs / kTile, 1,
                  {PtrArg(da.ptr()), PtrArg(slab.ptr()), PtrArg(out + n0), arg_val(kM),
                   arg_val(kK), arg_val(kNs), arg_val(lda), arg_val(kNs), arg_val(ldc),
                   arg_val(zero), arg_val(zero), arg_val(zero)},
                  &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };
    auto fused = [&](l0::Kernel* k, void* c) {
      return [&, k, c]() -> size_t {
        cx.launch(*k, kM / kTile, kN / kTile, 1,
                  {PtrArg(da.ptr()), PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(c), arg_val(kM),
                   arg_val(kK), arg_val(kN), arg_val(lda), arg_val(ldc)},
                  &ev[0]);
        cx.wait();
        return size_t(1);
      };
    };

    // --- equality first: a fast wrong kernel is a failed probe ---------------
    run_control();
    fused(&k_v1, c_v1.ptr())();
    fused(&k_v2, c_v2.ptr())();
    const Diff d1 = compare_c(ctx, c_ctl.ptr(), c_v1.ptr(), c_bytes);
    const Diff d2 = compare_c(ctx, c_ctl.ptr(), c_v2.ptr(), c_bytes);

    std::printf("\n## equality - memcmp over %zu fp32 outputs (M x N)\n\n", size_t(kM) * kN);
    std::printf("| variant | verdict | first differing index | control | variant |\n");
    std::printf("|---|---|---:|---|---|\n");
    diff_row("V1 registers-only", d1);
    diff_row("V2 SLM staged", d2);
    std::fflush(stdout);

    // --- rates ---------------------------------------------------------------
    const Timed t_ctl = best_of(run_control, ev, calib);
    const Timed t_v1 = best_of(fused(&k_v1, c_v1.ptr()), ev, calib);
    const Timed t_v2 = best_of(fused(&k_v2, c_v2.ptr()), ev, calib);

    std::printf("\n## rate (measured, L0 kernel timestamps, best of %d)\n\n", kReplays);
    std::printf("| variant | sum kernel ms | span ms | TFLOP/s |\n|---|---:|---:|---:|\n");
    row("control (34 slabs, 68 launches)", t_ctl);
    row("V1 registers-only (1 launch)", t_v1);
    row("V2 SLM staged (1 launch)", t_v2);
    std::printf("\ncontrol split: pf_dequant_slab %.3f ms, pf_gemm_T0 %.3f ms (34 launches "
                "each)\n",
                t_ctl.even_ms, t_ctl.odd_ms);

    // --- verdict against §7's decision rule ----------------------------------
    const bool v1_faster = t_v1.tflops() >= t_v2.tflops();
    const double best_fused = v1_faster ? t_v1.tflops() : t_v2.tflops();
    const bool best_equal = v1_faster ? d1.equal : d2.equal;
    std::printf("\nfaster fused variant: %s at %.2f TFLOP/s; bitwise %s\n", v1_faster ? "V1" : "V2",
                best_fused, best_equal ? "identical" : "**DIFFERENT**");
    if (!d1.equal || !d2.equal || best_fused < kBreakEven)
      std::printf("§7 branch: **REJECTED** (below the %.1f TFLOP/s break-even, or a memcmp "
                  "difference). The slab path stands.\n",
                  kBreakEven);
    else if (best_fused >= kTarget)
      std::printf("§7 branch: **ADOPT** - %.2f >= %.1f TFLOP/s with bitwise equality.\n",
                  best_fused, kTarget);
    else
      std::printf("§7 branch: **OPERATOR RULING** - %.2f TFLOP/s is between %.1f and %.1f.\n",
                  best_fused, kBreakEven, kTarget);
    return (d1.equal && d2.equal) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_fused_dequant_gemm FAILED: %s\n", e.what());
    return 2;
  }
}
