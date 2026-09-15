// P-A: our own OpenCL C bf16 DPAS GEMM (`pf_gemm_bf16.cl`) on the Level Zero
// list, timed and checked against `gemm_bf16` (sycl-tla, the C2 control) on
// identical inputs, same harness -- so any harness-level difference cancels.
//
// Pre-registration: docs/probe-prefill-vllm-parity-2026-09-14.md addendum
// §A3.1 / §A4, committed before this file was built or run. Protocol: 8
// replays, first 3 discarded, median of the last 5, 4 enqueues/replay, one
// discarded warm-up -- the project standard every prefill probe uses.
//
// This probe is a plain g++ translation unit (no SYCL of its own): the OpenCL
// kernel is launched through the raw L0 immediate list
// (`runtime::prefill::Context::launch`), and the control is the pre-built
// `gemm_bf16` entry point behind `runtime::prefill::gemm.h`, which is
// SYCL-free at this boundary (implemented in the icpx-linked
// libb70_prefill.so). Needs B70_PREFILL_ENABLED, exactly like
// probe_dequant_slab.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr int kReplays = 8;
constexpr int kDropped = 3;
constexpr int kEnqueues = 4;
constexpr size_t kSamples = 4096;
constexpr size_t kStageBytes = 64ul << 20;
constexpr double kUlp = 5.9604644775390625e-08;  // 2^-24
constexpr double kBar = 64.0;

const char* env_or_unset(const char* name) {
  const char* v = std::getenv(name);
  return v ? v : "(unset)";
}

uint32_t xorshift(uint32_t& v) {
  v ^= v << 13;
  v ^= v >> 17;
  v ^= v << 5;
  return v;
}

// Exponent fixed at 124: magnitude in [0.125, 0.25), so no dot product over
// K = 17408 reaches the fp32 range. Same generator P2/A24/A28's probes use.
uint16_t random_bf16(uint32_t& s) { return uint16_t((xorshift(s) & 0x807fu) | 0x3e00u); }

std::vector<uint16_t> random_bf16s(size_t n, uint32_t seed) {
  std::vector<uint16_t> v(n);
  for (uint16_t& x : v) x = random_bf16(seed);
  return v;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Shape {
  uint32_t K, N;
  const char* name;
};
constexpr Shape kShapes[] = {
    {5120, 34816, "gate‖up"},
    {17408, 5120, "down"},
};
constexpr uint32_t kGateUpMs[] = {1024, 2048, 4096};

bool bitwise_equal(l0::Context& ctx, const void* a, const void* b, size_t bytes) {
  const size_t chunk = std::min(kStageBytes, bytes);
  l0::Mem ha(ctx, l0::MemKind::Host, chunk);
  l0::Mem hb(ctx, l0::MemKind::Host, chunk);
  for (size_t off = 0; off < bytes; off += chunk) {
    const size_t now = std::min(chunk, bytes - off);
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(ha.ptr(), static_cast<const char*>(a) + off, now);
    c.copy(hb.ptr(), static_cast<const char*>(b) + off, now);
    if (std::memcmp(ha.ptr(), hb.ptr(), now) != 0) return false;
  }
  return true;
}

// gemm_batched_test.cc's own bar, transcribed verbatim (the addendum's
// pre-registered fallback): S = sum_k |A[m][k]*B[k][n]| in double;
// |C_dev - C_ref| <= 64 * 2^-24 * S + 2^-24.
struct Ratio {
  double max_abs_err = 0.0;
  double max_ratio = 0.0;
  bool within_bar = true;
};

Ratio sampled_reference_check(l0::Context& ctx, const void* c_dev, const std::vector<uint16_t>& a,
                              const std::vector<uint16_t>& b, uint32_t M, uint32_t K, uint32_t N) {
  struct Sample {
    size_t index;
    uint32_t m, n;
  };
  std::vector<Sample> samples;
  samples.reserve(kSamples);
  uint32_t seed = 0x7f4a7c15u;
  for (size_t i = 0; i < kSamples; ++i) {
    const size_t idx = size_t(xorshift(seed)) % (size_t(M) * N);
    samples.push_back({idx, uint32_t(idx / N), uint32_t(idx % N)});
  }
  std::sort(samples.begin(), samples.end(),
            [](const Sample& x, const Sample& y) { return x.index < y.index; });

  Ratio r;
  const size_t bytes = size_t(M) * N * sizeof(float);
  const size_t chunk = std::min(kStageBytes, bytes);
  l0::Mem stage(ctx, l0::MemKind::Host, chunk);
  size_t next = 0;
  for (size_t off = 0; off < bytes && next < samples.size(); off += chunk) {
    const size_t now = std::min(chunk, bytes - off);
    l0::CmdList::immediate(ctx).copy(stage.ptr(), static_cast<const char*>(c_dev) + off, now);
    const size_t first = off / sizeof(float), end = first + now / sizeof(float);
    const float* got = stage.as<float>();
    while (next < samples.size() && samples[next].index < end) {
      const Sample& s = samples[next++];
      double ref = 0.0, mag = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        const double av = common::bf16_to_f32(a[size_t(s.m) * K + k]);
        const double bv = common::bf16_to_f32(b[size_t(k) * N + s.n]);
        ref += av * bv;
        mag += std::fabs(av * bv);
      }
      const double err = std::fabs(double(got[s.index - first]) - ref);
      r.max_abs_err = std::max(r.max_abs_err, err);
      r.max_ratio = std::max(r.max_ratio, err / (kUlp * mag + kUlp));
      if (err > kBar * kUlp * mag + kUlp) r.within_bar = false;
    }
  }
  return r;
}

struct Timed {
  double ms = 0.0;
  double tflops = 0.0;
};

Timed time_pf(runtime::prefill::Context& cx, l0::Kernel& k, uint32_t M, uint32_t K, uint32_t N,
             const void* A, const void* B, void* C) {
  const uint32_t gx = M / 256, gy = N / 256;
  auto once = [&] {
    for (int e = 0; e < kEnqueues; ++e)
      cx.launch(k, gx, gy, 1,
                {PtrArg(A), PtrArg(B), PtrArg(C), arg_val(M), arg_val(K), arg_val(N)});
    cx.wait();
  };
  once();  // discarded warm-up
  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    if (r >= kDropped)
      samples.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count() /
          kEnqueues);
  }
  const double ms = median(std::move(samples));
  return {ms, 2.0 * double(M) * K * N / (ms * 1e9)};
}

Timed time_c2(runtime::prefill::Context& cx, uint32_t M, uint32_t K, uint32_t N,
             const uint16_t* A, const uint16_t* B, float* C) {
  auto once = [&] {
    for (int e = 0; e < kEnqueues; ++e) runtime::prefill::gemm_bf16(cx, {M, K, N}, A, B, C);
    cx.wait();
  };
  once();
  std::vector<double> samples;
  for (int r = 0; r < kReplays; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    if (r >= kDropped)
      samples.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count() /
          kEnqueues);
  }
  const double ms = median(std::move(samples));
  return {ms, 2.0 * double(M) * K * N / (ms * 1e9)};
}

struct CellResult {
  const char* shape;
  uint32_t M;
  Timed pf, c2;
  bool bitwise_vs_c2 = false;
  bool determinism = false;
  Ratio ratio;
};

CellResult run_cell(l0::Context& l0ctx, runtime::prefill::Context& cx, l0::Kernel& kpf,
                    const Shape& shape, uint32_t M) {
  const size_t a_n = size_t(M) * shape.K;
  const size_t b_n = size_t(shape.K) * shape.N;
  const size_t c_bytes = size_t(M) * shape.N * sizeof(float);
  std::vector<uint16_t> a = random_bf16s(a_n, 0x9e3779b9u ^ M ^ shape.K ^ shape.N);
  std::vector<uint16_t> b = random_bf16s(b_n, 0x85ebca6bu ^ M ^ shape.K ^ shape.N);

  l0::Mem da(l0ctx, l0::MemKind::Device, a_n * sizeof(uint16_t));
  l0::Mem db(l0ctx, l0::MemKind::Device, b_n * sizeof(uint16_t));
  l0::Mem c_pf1(l0ctx, l0::MemKind::Device, c_bytes);
  l0::Mem c_pf2(l0ctx, l0::MemKind::Device, c_bytes);
  l0::Mem c_ref(l0ctx, l0::MemKind::Device, c_bytes);
  {
    l0::CmdList up = l0::CmdList::immediate(l0ctx);
    up.copy(da.ptr(), a.data(), a_n * sizeof(uint16_t));
    up.copy(db.ptr(), b.data(), b_n * sizeof(uint16_t));
  }

  CellResult res;
  res.shape = shape.name;
  res.M = M;

  // --- determinism: two pf runs, bitwise ---
  {
    const uint32_t gx = M / 256, gy = shape.N / 256;
    cx.launch(kpf, gx, gy, 1,
              {PtrArg(da.ptr()), PtrArg(db.ptr()), PtrArg(c_pf1.ptr()), arg_val(M),
               arg_val(shape.K), arg_val(shape.N)});
    cx.launch(kpf, gx, gy, 1,
              {PtrArg(da.ptr()), PtrArg(db.ptr()), PtrArg(c_pf2.ptr()), arg_val(M),
               arg_val(shape.K), arg_val(shape.N)});
    cx.wait();
    res.determinism = bitwise_equal(l0ctx, c_pf1.ptr(), c_pf2.ptr(), c_bytes);
  }

  // --- correctness: pf vs C2 (gemm_bf16), same inputs ---
  runtime::prefill::gemm_bf16(cx, {M, shape.K, shape.N}, da.as<uint16_t>(), db.as<uint16_t>(),
                              c_ref.as<float>());
  cx.wait();
  res.bitwise_vs_c2 = bitwise_equal(l0ctx, c_pf1.ptr(), c_ref.ptr(), c_bytes);
  if (!res.bitwise_vs_c2)
    res.ratio = sampled_reference_check(l0ctx, c_pf1.ptr(), a, b, M, shape.K, shape.N);

  // --- timing: pf and the C2 control, same harness ---
  res.pf = time_pf(cx, kpf, M, shape.K, shape.N, da.ptr(), db.ptr(), c_pf1.ptr());
  res.c2 = time_c2(cx, M, shape.K, shape.N, da.as<uint16_t>(), db.as<uint16_t>(), c_ref.as<float>());
  return res;
}

}  // namespace

int main() {
  try {
    std::printf("# P-A: pf_gemm_bf16 (our OpenCL C DPAS GEMM) vs gemm_bf16 (sycl-tla, C2 control)\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    std::printf("# %d replays, discard first %d, median of last %d; %d enqueues/replay; "
                "one discarded warm-up.\n",
                kReplays, kDropped, kReplays - kDropped, kEnqueues);
    l0::Context l0ctx(0);
    std::printf("# L0 device: %s\n", l0ctx.name().c_str());
    runtime::prefill::Context cx(l0ctx);
    l0::Module mod(l0ctx, kernels::path("pf_gemm_bf16"));
    l0::Kernel kpf = mod.kernel("pf_gemm_bf16");

    std::printf("\n| shape | K×N | M | pf ms | pf TFLOP/s | C2 ms | C2 TFLOP/s | pf/C2 | "
                "bitwise vs C2 | max ratio (bar %.0f) | determinism |\n", kBar);
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|\n");
    bool all_ok = true;
    for (const Shape& shape : kShapes) {
      std::vector<uint32_t> ms = {2048};
      if (std::string(shape.name) == "gate‖up") ms.assign(std::begin(kGateUpMs), std::end(kGateUpMs));
      for (uint32_t M : ms) {
        const CellResult r = run_cell(l0ctx, cx, kpf, shape, M);
        all_ok = all_ok && r.determinism && (r.bitwise_vs_c2 || r.ratio.within_bar);
        std::printf("| %s | %u×%u | %u | %.3f | %.2f | %.3f | %.2f | %.4f | %s | %.3f | %s |\n",
                    shape.name, shape.K, shape.N, M, r.pf.ms, r.pf.tflops, r.c2.ms, r.c2.tflops,
                    r.pf.tflops / r.c2.tflops,
                    r.bitwise_vs_c2 ? "**bitwise identical**" : "differs", r.ratio.max_ratio,
                    r.determinism ? "bitwise" : "**FAILED**");
        std::fflush(stdout);
      }
    }
    std::printf("\nverdict inputs: correctness %s; determinism %s.\n",
                all_ok ? "PASS" : "**FAIL**", all_ok ? "PASS" : "**FAIL**");
    return all_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_pf_gemm FAILED: %s\n", e.what());
    return 1;
  }
}
