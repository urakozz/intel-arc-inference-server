// Probe the OpenCL C int4 -> bf16 expansion used before a prefill GEMM.
// It deliberately times both scratch geometries: [K][N] is the settled native
// GEMM input (ldb = N); [N][K] is a store-geometry control only.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {
constexpr size_t kScratchBytes = size_t(5120) * 34816 * sizeof(uint16_t);
constexpr int kTimedLaunches = 40;

struct Shape {
  const char* name;
  uint32_t K;
  uint32_t N;
};

// The final row is a loader-layout throughput control. It is not prefill
// scratch: the 356,515,840-byte runtime scratch intentionally cannot hold it.
const Shape kShapes[] = {
    {"qkv‖z", 5120, 16384}, {"out/o_proj", 6144, 5120},
    {"gate‖up", 5120, 34816}, {"down", 17408, 5120},
    {"q‖k‖v", 5120, 14336}, {"lm_head int4 (probe-only)", 5120, 248320},
};

struct Result {
  double us_per_launch = 0;
  size_t read_bytes = 0;
  size_t write_bytes = 0;
  bool bit_exact = true;
  size_t sampled = 0;
};

std::string variant_name(uint32_t K, uint32_t N, uint32_t layout, uint32_t transposed) {
  return "pf_dequant_tile_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_L" + std::to_string(layout) + "_T" + std::to_string(transposed);
}

double time_list(l0::Queue& queue, l0::Fence& fence, l0::CmdList& list, int launches) {
  std::vector<double> samples;
  for (int replay = 0; replay < 8; ++replay) {
    const auto begin = std::chrono::steady_clock::now();
    queue.execute(list, &fence);
    fence.wait();
    const double total =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    if (replay >= 3) samples.push_back(total / launches);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// One full output readback makes a deterministic 1% stride sample cheap and
// checks the device result independently of the bandwidth timing list.
bool sample_exact(const common::Int4Gptq& weights, const std::vector<uint16_t>& got,
                  uint32_t K, uint32_t N, uint32_t layout, uint32_t transposed,
                  size_t* sampled) {
  for (size_t index = 0; index < got.size(); index += 100) {
    const uint32_t k = transposed ? uint32_t(index % K) : uint32_t(index / N);
    const uint32_t n = transposed ? uint32_t(index / K) : uint32_t(index % N);
    const uint16_t expected = common::f32_to_bf16(weights.at(k, n));
    ++*sampled;
    if (got[index] == expected) continue;
    const uint32_t word = weights.qweight[size_t(k / 8) * N + n];
    const uint16_t scale = weights.scales[size_t(k / 64) * N + n];
    std::fprintf(stderr,
                 "pf_dequant_tile L%u T%u mismatch at k=%u n=%u: "
                 "got=0x%04X ref=0x%04X (word=0x%08X nibble q=%u "
                 "scale=0x%04X=%.9g)\n",
                 layout, transposed, k, n, got[index], expected, word,
                 (word >> (4 * (k % 8))) & 0xFu, scale,
                 double(common::f16_to_f32(scale)));
    return false;
  }
  return true;
}

Result run_case(l0::Context& ctx, l0::Queue& queue, l0::Fence& fence,
                l0::Mem& scratch, const common::Int4Gptq& weights,
                const Shape& shape, uint32_t layout, uint32_t transposed,
                bool check_sample) {
  Result result;
  result.read_bytes = weights.bytes();
  result.write_bytes = size_t(shape.K) * shape.N * sizeof(uint16_t);
  const int copies =
      std::max(2, int((72ull << 20) / result.read_bytes) + 1);

  const std::vector<uint32_t> tiled = layout ? weights.tiled() : std::vector<uint32_t>{};
  const uint32_t* source = layout ? tiled.data() : weights.qweight.data();
  const size_t weight_bytes = layout ? tiled.size() * sizeof(uint32_t)
                                     : weights.qweight.size() * sizeof(uint32_t);
  l0::CmdList immediate = l0::CmdList::immediate(ctx);
  std::vector<l0::Mem> weight_copies;
  std::vector<l0::Mem> scale_copies;
  weight_copies.reserve(copies);
  scale_copies.reserve(copies);
  for (int copy = 0; copy < copies; ++copy) {
    weight_copies.emplace_back(ctx, l0::MemKind::Device, weight_bytes);
    immediate.copy(weight_copies.back().ptr(), source, weight_bytes);
    if (!layout) {
      scale_copies.emplace_back(ctx, l0::MemKind::Device,
                                 weights.scales.size() * sizeof(uint16_t));
      immediate.copy(scale_copies.back().ptr(), weights.scales.data(),
                     weights.scales.size() * sizeof(uint16_t));
    }
  }

  // lm_head int4 is a probe-only sixth loader-layout shape. Production prefill
  // never binds it, and its 2.54 GB result cannot fit the fixed 356 MB runtime
  // scratch; all actual prefill rows use the single scratch above.
  std::unique_ptr<l0::Mem> oversize_output;
  void* output = scratch.ptr();
  if (result.write_bytes > scratch.size()) {
    oversize_output =
        std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, result.write_bytes);
    output = oversize_output->ptr();
  }

  l0::Module module(ctx, kernels::path(variant_name(shape.K, shape.N, layout, transposed)));
  l0::Kernel kernel = module.kernel("pf_dequant_tile");
  kernel.group_size(16);
  auto bind = [&](int copy) {
    kernel.arg_ptr(0, weight_copies[copy].ptr());
    kernel.arg_ptr(1, layout ? nullptr : scale_copies[copy].ptr());
    kernel.arg_ptr(2, output);
  };

  if (check_sample) {
    l0::CmdList correctness = l0::CmdList::regular(ctx);
    bind(0);
    correctness.launch(kernel, shape.N / 16, shape.K / 64);
    correctness.close();
    queue.execute(correctness, &fence);
    fence.wait();
    std::vector<uint16_t> got(size_t(shape.K) * shape.N);
    immediate.copy(got.data(), output, result.write_bytes);
    result.bit_exact =
        sample_exact(weights, got, shape.K, shape.N, layout, transposed, &result.sampled);
  }

  l0::CmdList timed = l0::CmdList::regular(ctx);
  for (int launch = 0; launch < kTimedLaunches; ++launch) {
    bind(launch % copies);
    timed.launch(kernel, shape.N / 16, shape.K / 64);
  }
  timed.close();
  result.us_per_launch = time_list(queue, fence, timed, kTimedLaunches);
  return result;
}

double gbps(size_t bytes, double us) {
  return double(bytes) / (us * 1e3);
}
}  // namespace

int main() {
  l0::Context ctx(0);
  l0::Queue queue(ctx);
  l0::Fence fence(queue);
  l0::Mem scratch(ctx, l0::MemKind::Device, kScratchBytes);
  bool all_exact = true;
  double first_gbps = 0;

  std::printf("# P3 pf_dequant_tile (iterate-grade; 8 replays, discard 3, median)\n");
  std::printf("# L0 device: %s; reusable prefill scratch: %zu bytes\n", ctx.name().c_str(),
              kScratchBytes);
  std::printf("# NB = max(2, floor(72 MiB / source_bytes) + 1); %d launches/list.\n",
              kTimedLaunches);
  std::printf("| shape | K×N | L | orient | µs | read MB | write MB | GB/s (r+w) | "
              "GB/s (r+2w) | bit-exact |\n");
  std::printf("|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n");

  for (size_t shape_index = 0; shape_index < sizeof(kShapes) / sizeof(kShapes[0]); ++shape_index) {
    const Shape& shape = kShapes[shape_index];
    const common::Int4Gptq weights =
        common::Int4Gptq::random(shape.K, shape.N, 42 + uint32_t(shape_index));
    for (uint32_t layout = 0; layout < 2; ++layout)
      for (uint32_t transposed = 0; transposed < 2; ++transposed) {
        if (shape_index == 0 && layout == 0 && transposed == 0) {
          const Result warm =
              run_case(ctx, queue, fence, scratch, weights, shape, layout, transposed, false);
          std::printf("# ramp control (discarded): %s L0 [K][N] %.0f GB/s (r+w)\n",
                      shape.name, gbps(warm.read_bytes + warm.write_bytes, warm.us_per_launch));
        }
        const Result result =
            run_case(ctx, queue, fence, scratch, weights, shape, layout, transposed, true);
        const double streaming = gbps(result.read_bytes + result.write_bytes, result.us_per_launch);
        const double write_allocate =
            gbps(result.read_bytes + 2 * result.write_bytes, result.us_per_launch);
        if (shape_index == 0 && layout == 0 && transposed == 0) first_gbps = streaming;
        all_exact = all_exact && result.bit_exact;
        std::printf("| %s | %u×%u | %u | %s | %.1f | %.2f | %.2f | %.1f | %.1f | %s "
                    "(1%%, %zu cells) |\n",
                    shape.name, shape.K, shape.N, layout, transposed ? "[N][K]" : "[K][N]",
                    result.us_per_launch, double(result.read_bytes) / 1e6,
                    double(result.write_bytes) / 1e6, streaming, write_allocate,
                    result.bit_exact ? "yes" : "**NO**", result.sampled);
        std::fflush(stdout);
      }
  }

  // The first recorded cell is the drift control's single repeat, mirroring
  // probe_gemv's first-shape end-of-battery check.
  const Shape& first = kShapes[0];
  const common::Int4Gptq first_weights = common::Int4Gptq::random(first.K, first.N, 42);
  run_case(ctx, queue, fence, scratch, first_weights, first, 0, 0, false);
  const Result drift = run_case(ctx, queue, fence, scratch, first_weights, first, 0, 0, true);
  const double final_gbps = gbps(drift.read_bytes + drift.write_bytes, drift.us_per_launch);
  std::printf("\n## drift control (iterate-grade)\n");
  std::printf("| cell | GB/s at battery start | GB/s at battery end | Δ%% |\n"
              "|---|---:|---:|---:|\n");
  std::printf("| %s L0 [K][N] | %.1f | %.1f | %+.2f%% |\n", first.name, first_gbps,
              final_gbps, 100.0 * (final_gbps / first_gbps - 1.0));
  std::printf("\nbit-exact sampled check: %s\n",
              all_exact ? "every row passed" : "**A ROW FAILED - see mismatch above**");
  return all_exact ? 0 : 1;
}
