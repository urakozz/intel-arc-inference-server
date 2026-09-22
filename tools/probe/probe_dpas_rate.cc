// probe_dpas_rate - what rate does this B70's XMX/DPAS actually sustain for
// each data type the compiler exposes?
//
// Record: docs/probe-dpas-rates-2026-09-22.md. This is a DIAGNOSTIC rate probe,
// not a benchmark series row: it answers "is a W4A8 / W4A4 / FP4 kernel worth
// designing at all" before anyone designs one.
//
// It drives `tools/probe/probe_dpas_rate.cl`, one binary per data type, each a
// straight chain of 64 independent register-resident `matrix_mad` calls per
// loop iteration. See the .cl header for why the loop cannot be deleted; the
// assembly dump quoted in the record is the proof.
//
// Plain g++ / Level Zero, exactly like probe_bw: Queue + Fence + a closed
// regular command list, timed with the wall clock around execute+wait. No
// SYCL, no prefill runtime, no production file touched.
//
// Protocol (the project standard, as probe_pf_gemm): per cell, calibrate the
// trip count to ~40 ms, one discarded warm-up, then 8 replays of `kEnqueues`
// launches each, first 3 discarded, median of the last 5.
//
// Device: pass ZE_AFFINITY_MASK=0. `l0::Context(0)` takes the first device of
// the (possibly masked) enumeration.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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
constexpr double kTargetMs = 40.0;   // per enqueue, set by the calibration pass
constexpr uint32_t kCalibIters = 256;
constexpr uint32_t kWgSize = 256;    // 16 sub-groups of 16 per work-group
constexpr uint32_t kDpasPerIter = 64;  // UNROLL(8) * NACC(8) in the .cl
constexpr uint32_t kMadM = 8;        // dpas repeat count
constexpr uint32_t kMadN = 16;       // sub-group width
constexpr uint32_t kSrcWords = 16384;
constexpr uint32_t kGrids[] = {64, 128, 256, 512};
constexpr uint32_t kMaxGroups = 512;
// E8M0 scale bytes for the scaled forms: 127 == 2^0. Unused by every type that
// builds on this device (all scaled builtins ICE the backend), passed anyway so
// the kernel signature is one signature.
constexpr uint32_t kScaleBits = 127u | (127u << 8);

struct Variant {
  const char* tag;       // ocloc binary name
  const char* name;      // the builtin family, as the record's table spells it
  uint32_t k;            // the builtin's k-suffix
  const char* unit;      // "FLOP" or "IOP" -- what 2*M*N*K counts
};

// Only the types whose binaries this device's backend actually produces are
// listed. The ones it refuses (FP4 `e2m1_*`: "FP4 Dpas instruction is not
// supported on this device!"; every `*_scaled_matrix_mad_*`: IGC internal
// compiler error) have no binary and no row -- the record carries the verbatim
// ocloc output instead.
constexpr Variant kVariants[] = {
    {"pdr_bf16", "bf16_bf16_k16", 16, "FLOP"},
    {"pdr_f16", "f16_f16_k16", 16, "FLOP"},
    {"pdr_i8", "i8_i8_k32", 32, "IOP"},
    {"pdr_i4i8", "i4_i8_k32", 32, "IOP"},
    {"pdr_i8i4", "i8_i4_k32", 32, "IOP"},
    {"pdr_i4", "i4_i4_k64", 64, "IOP"},
    {"pdr_u4", "u4_u4_k64", 64, "IOP"},
    {"pdr_i2", "i2_i2_k64", 64, "IOP"},
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

const char* env_or_unset(const char* name) {
  const char* v = std::getenv(name);
  return v ? v : "(unset)";
}

struct Cell {
  uint32_t groups = 0;
  uint32_t iters = 0;
  double ms = 0.0;        // per enqueue, median
  double dpas = 0.0;      // dpas instructions issued per enqueue, device-wide
  double ops_per_s = 0.0;
};

// One timed cell: `groups` work-groups, trip count calibrated to ~kTargetMs.
Cell run_cell(l0::Context& ctx, l0::Queue& q, l0::Fence& fence, l0::Kernel& k, void* src,
              void* dst, uint32_t groups) {
  auto timed = [&](uint32_t iters, int enqueues) {
    k.arg_ptr(0, src);
    k.arg_ptr(1, dst);
    k.arg(2, iters);
    k.arg(3, kScaleBits);
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

  // calibration: one short launch, scaled to the target, then a warm-up.
  const double calib_ms = timed(kCalibIters, 1);
  double scale = kTargetMs / std::max(calib_ms, 1e-3);
  uint64_t want = uint64_t(double(kCalibIters) * scale);
  const uint32_t iters = uint32_t(std::min<uint64_t>(std::max<uint64_t>(want, 1024), 4000000));
  timed(iters, kEnqueues);  // discarded warm-up

  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const double ms = timed(iters, kEnqueues);
    if (r >= kDropped) samples.push_back(ms);
  }
  Cell c;
  c.groups = groups;
  c.iters = iters;
  c.ms = median(std::move(samples));
  const double subgroups = double(groups) * (kWgSize / 16);
  c.dpas = subgroups * double(iters) * kDpasPerIter;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    l0::Context ctx(0);
    l0::Queue q(ctx);
    l0::Fence fence(q);
    l0::CmdList imm = l0::CmdList::immediate(ctx);

    std::printf("# probe_dpas_rate: sustained DPAS issue rate per data type, one B70\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    std::printf("# L0 device: %s | EUs %u | %u Xe-cores | max core clock %u MHz\n",
                ctx.name().c_str(), ctx.eu_count(),
                ctx.props().numSlices * ctx.props().numSubslicesPerSlice,
                ctx.props().coreClockRate);
    std::printf("# ops per mad = 2 * M * N * K with M=%u (dpas repeat count), N=%u "
                "(sub-group width), K = the builtin's k-suffix.\n", kMadM, kMadN);
    std::printf("# work-group %u (16 sub-groups); %u dpas per loop iteration; "
                "%d replays, discard %d, median of %d; %d launches per replay; "
                "one discarded warm-up.\n",
                kWgSize, kDpasPerIter, kReplays, kDropped, kReplays - kDropped, kEnqueues);
    const bool verbose = argc > 1 && std::strcmp(argv[1], "--sweep") == 0;

    // Operand source. 0x3c3c3c3c reads as a normal, near-unit value in every
    // format under test (bf16 ~0.0115, fp16 ~1.06, int8 60, int4 3/-4, int2
    // 0/-1) -- no denormals, no NaN, nothing that could make one type's
    // arithmetic special. Varied in the low bits so no two lanes are identical.
    l0::Mem src(ctx, l0::MemKind::Device, kSrcWords * sizeof(uint32_t));
    {
      std::vector<uint32_t> h(kSrcWords);
      for (uint32_t i = 0; i < kSrcWords; ++i) h[i] = 0x3c3c3c3cu + (i & 3u);
      imm.copy(src.ptr(), h.data(), h.size() * sizeof(uint32_t));
    }
    l0::Mem dst(ctx, l0::MemKind::Device, size_t(kMaxGroups) * kWgSize * 32);

    std::printf("\n| type | K | best grid (WGs) | trip count | ms/launch | dpas issued/launch |"
                " %s/s (T) | ratio to bf16 |\n", "ops");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|\n");

    double bf16_rate = 0.0;
    std::vector<std::string> missing;
    for (const Variant& v : kVariants) {
      l0::Module* mod = nullptr;
      try {
        mod = new l0::Module(ctx, kernels::path(v.tag));
      } catch (const std::exception& e) {
        missing.push_back(std::string(v.name) + " (" + v.tag + ": " + e.what() + ")");
        continue;
      }
      l0::Kernel k = mod->kernel("dpas_rate");
      const double ops_per_mad = 2.0 * kMadM * kMadN * double(v.k);

      Cell best;
      for (uint32_t g : kGrids) {
        const Cell c = run_cell(ctx, q, fence, k, src.ptr(), dst.ptr(), g);
        const double rate = c.dpas * ops_per_mad / (c.ms * 1e-3);
        if (verbose)
          std::printf("#   %-22s grid %4u  iters %8u  %8.3f ms  %.2f T%s/s\n", v.name, g,
                      c.iters, c.ms, rate / 1e12, v.unit);
        if (rate > best.ops_per_s) {
          best = c;
          best.ops_per_s = rate;
        }
      }
      if (std::strcmp(v.tag, "pdr_bf16") == 0) bf16_rate = best.ops_per_s;
      char ratio[32];
      if (bf16_rate > 0.0)
        std::snprintf(ratio, sizeof(ratio), "%.3fx", best.ops_per_s / bf16_rate);
      else
        std::snprintf(ratio, sizeof(ratio), "n/a");
      std::printf("| `%s` | %u | %u | %u | %.3f | %.4g | **%.2f T%s/s** | %s |\n", v.name, v.k,
                  best.groups, best.iters, best.ms, best.dpas, best.ops_per_s / 1e12, v.unit,
                  ratio);
      std::fflush(stdout);
      delete mod;
    }
    for (const std::string& m : missing) std::printf("\nNO BINARY: %s\n", m.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_dpas_rate FAILED: %s\n", e.what());
    return 1;
  }
}
