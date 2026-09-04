// P2: stock sycl-tla bf16 GEMM at every production prefill shape.
//
// This instantiates the queue example's BMG configuration verbatim. It is a
// probe, not the eventual prefill GEMM wrapper: its job is to establish the
// achieved rate, required B layout, and determinism before S1 is designed.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

// cute's host/device printing headers map printf to SYCL's device printf.
// This probe is host code; preserve std::printf for its markdown output.
#undef printf

#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"
#include "tla_pin.h"

using namespace cute;

namespace {

constexpr int kReplays = 8;
constexpr int kDroppedReplays = 3;
constexpr int kTimedEnqueues = 4;
constexpr size_t kSamples = 4096;
constexpr size_t kCompareChunkBytes = 64ul << 20;

struct ProbeShape {
  uint32_t K;
  uint32_t N;
  const char* name;
};

constexpr ProbeShape kShapes[] = {
    {5120, 16384, "qkv‖z"},       {6144, 5120, "out/o_proj"},
    {5120, 34816, "gate‖up"},     {17408, 5120, "down"},
    {5120, 14336, "q‖k‖v"},       {5120, 248320, "lm_head"},
};
constexpr uint32_t kMs[] = {512, 1024, 2048, 4096};

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

const char* env_or_unset(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "(unset)";
}

uint32_t xorshift(uint32_t& value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

// A raw bf16 word whose exponent is fixed at 124: its magnitude is in
// [0.125, 0.25), so dot products stay well-scaled and never generate NaN/Inf.
uint16_t random_bf16(uint32_t& state) {
  return uint16_t((xorshift(state) & 0x807fu) | 0x3e00u);
}

std::vector<uint16_t> random_bf16s(size_t count, uint32_t seed) {
  std::vector<uint16_t> values(count);
  for (uint16_t& value : values) value = random_bf16(seed);
  return values;
}

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = bfloat16_t;
using ElementInputB = bfloat16_t;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Exactly examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp:345-420 at
// the content-verified pin. `void` copies make MainloopXeL1Staged select the
// XE_LOAD_2D / XE_LOAD_2D_VNNI operations itself.
using GmemTiledCopyA = void;
using GmemTiledCopyB = void;
using TileShape = cute::Shape<_256, _256, _32>;
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<TileShape>,
    Layout<cute::Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
constexpr int PipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy, TileShape, void, void,
    cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallbacks, void, void>;
static_assert(std::is_void_v<typename CollectiveEpilogue::ElementC>,
              "P2 is D = alpha*A*B: the epilogue must not read C");
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy, TileShape, ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma, GmemTiledCopyA, void,
    void, cute::identity, GmemTiledCopyB, void, void, cute::identity>;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                         CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
using Arguments = typename Gemm::GemmKernel::Arguments;

struct PreparedGemm {
  Gemm op;
  size_t workspace = 0;
  cutlass::Status can_implement = cutlass::Status::kErrorInternal;
};

Arguments make_arguments(uint32_t m, uint32_t k, uint32_t n, uint16_t* a, uint16_t* b,
                         float* c, const cutlass::KernelHardwareInfo& hw_info) {
  const ProblemShape problem{int(m), int(n), int(k), 1};
  const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(int(m), int(k), 1));
  // RowMajor B has a rank-3 CuTe stride over (N, K, L), not (K, N, L).
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(int(n), int(k), 1));
  const auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(int(m), int(n), 1));
  const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(int(m), int(n), 1));
  return {cutlass::gemm::GemmUniversalMode::kGemm, problem,
          {reinterpret_cast<ElementInputA*>(a), stride_a, reinterpret_cast<ElementInputB*>(b), stride_b},
          {{1.f, 0.f}, c, stride_c, c, stride_d}, hw_info};
}

uint64_t dim3_count(const dim3& dims) {
  return uint64_t(dims.x) * uint64_t(dims.y) * uint64_t(dims.z);
}

// This is the exact host-side path GemmUniversalAdapter::run() follows to
// choose the SYCL nd_range.  Keep it in the probe: the diagnostic must report
// the instantiated kernel's grid, not reconstruct one from the tile shape.
void print_launch_geometry(const ProbeShape& shape, uint32_t m,
                           const cutlass::KernelHardwareInfo& hw_info) {
  const Arguments args = make_arguments(m, shape.K, shape.N, nullptr, nullptr, nullptr, hw_info);
  const dim3 grid = Gemm::get_grid_shape(args);
  const dim3 block = GemmKernel::get_block_shape();
  const uint32_t expected_m = (m + 255) / 256;
  const uint32_t expected_n = (shape.N + 255) / 256;
  std::printf("| %s | %u | %u×%u = %llu | %u×%u×%u = %llu | %u×%u×%u = %llu |\n",
              shape.name, m, expected_m, expected_n,
              static_cast<unsigned long long>(uint64_t(expected_m) * expected_n), grid.x, grid.y,
              grid.z, static_cast<unsigned long long>(dim3_count(grid)), block.x, block.y, block.z,
              static_cast<unsigned long long>(dim3_count(block)));
}

void print_device_and_launch_diagnostic(l0::Context& l0ctx, runtime::prefill::Context& cx,
                                        const cutlass::KernelHardwareInfo& hw_info) {
  sycl::queue& queue = runtime::prefill::sycl_queue(cx);
  const sycl::device dev = queue.get_device();
  const uint32_t sycl_slices =
      dev.get_info<sycl::ext::intel::info::device::gpu_slices>();
  const uint32_t sycl_subslices =
      dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  const uint32_t sycl_eus_per_subslice =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
  const uint32_t sycl_eus =
      dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
  const uint32_t sycl_max_compute_units =
      dev.get_info<sycl::info::device::max_compute_units>();
  const ze_device_properties_t& l0 = l0ctx.props();
  const ze_device_compute_properties_t& l0_compute = l0ctx.compute();

  std::printf("# grid diagnostic (reported, not inferred); ZE_AFFINITY_MASK=%s\n",
              env_or_unset("ZE_AFFINITY_MASK"));
  std::printf("# SYCL device: %s; max_compute_units=%u; slices=%u; subslices/slice=%u; "
              "EUs/subslice=%u; EUs=%u; scheduler query sm_count=%d\n",
              dev.get_info<sycl::info::device::name>().c_str(), sycl_max_compute_units, sycl_slices,
              sycl_subslices, sycl_eus_per_subslice, sycl_eus, hw_info.sm_count);
  std::printf("# Level Zero device: slices=%u; subslices/slice=%u; EUs/subslice=%u; "
              "EUs=%u; threads/EU=%u; max_total_group_size=%u; max_group_size=%ux%ux%u\n",
              l0.numSlices, l0.numSubslicesPerSlice, l0.numEUsPerSubslice, l0ctx.eu_count(),
              l0.numThreadsPerEU, l0_compute.maxTotalGroupSize, l0_compute.maxGroupSizeX,
              l0_compute.maxGroupSizeY, l0_compute.maxGroupSizeZ);
  std::printf("# Level Zero exposes no max_compute_units field; its reported Xe-core analogue is "
              "slices*subslices/slice = %u. The scheduler query is SYCL slices*subslices/slice.\n",
              l0.numSlices * l0.numSubslicesPerSlice);
  std::printf("# expected data-parallel grid is ceil(M/256)×ceil(N/256). GemmUniversal grid and "
              "local size below are its exact get_grid_shape()/get_block_shape() launch values.\n\n");
  std::printf("| shape | M | expected M×N work-groups | GemmUniversal grid x×y×z | local x×y×z |\n");
  std::printf("|---|---:|---:|---:|---:|\n");
  for (const ProbeShape& shape : kShapes)
    for (uint32_t m : kMs) print_launch_geometry(shape, m, hw_info);
  std::printf("\n");
}

PreparedGemm prepare(uint32_t m, uint32_t k, uint32_t n, uint16_t* a, uint16_t* b, float* c,
                     const cutlass::KernelHardwareInfo& hw_info, sycl::queue& queue) {
  PreparedGemm prepared;
  const Arguments args = make_arguments(m, k, n, a, b, c, hw_info);
  prepared.workspace = Gemm::get_workspace_size(args);
  prepared.can_implement = prepared.op.can_implement(args);
  if (prepared.workspace != 0)
    fail("Gemm::get_workspace_size returned " + std::to_string(prepared.workspace));
  if (prepared.can_implement != cutlass::Status::kSuccess)
    return prepared;
  const cutlass::Status initialized = prepared.op.initialize(args, nullptr, &queue);
  if (initialized != cutlass::Status::kSuccess)
    fail("Gemm::initialize failed with status " + std::to_string(int(initialized)));
  return prepared;
}

void run_once(PreparedGemm& prepared, sycl::queue& queue) {
  const cutlass::Status status = prepared.op.run(&queue);
  if (status != cutlass::Status::kSuccess)
    fail("Gemm::run failed with status " + std::to_string(int(status)));
  queue.wait_and_throw();
}

double time_replays(PreparedGemm& prepared, sycl::queue& queue) {
  std::array<double, kReplays - kDroppedReplays> samples{};
  for (int replay = 0; replay < kReplays; ++replay) {
    const auto begin = std::chrono::steady_clock::now();
    for (int enqueue = 0; enqueue < kTimedEnqueues; ++enqueue) {
      const cutlass::Status status = prepared.op.run(&queue);
      if (status != cutlass::Status::kSuccess)
        fail("timed Gemm::run failed with status " + std::to_string(int(status)));
    }
    queue.wait_and_throw();
    if (replay >= kDroppedReplays) {
      samples[static_cast<size_t>(replay - kDroppedReplays)] =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() /
          kTimedEnqueues;
    }
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct LayoutVerdict {
  bool b_is_kn = false;
  std::string summary;
};

LayoutVerdict settle_b_layout(l0::Context& l0ctx, runtime::prefill::Context& cx,
                              const cutlass::KernelHardwareInfo& hw_info) {
  constexpr uint32_t d = 16;
  std::vector<uint16_t> a(size_t(d) * d, common::f32_to_bf16(0.f));
  std::vector<uint16_t> b(size_t(d) * d);
  for (uint32_t i = 0; i < d; ++i) {
    a[size_t(i) * d + i] = common::f32_to_bf16(1.f);
    for (uint32_t j = 0; j < d; ++j)
      b[size_t(i) * d + j] = common::f32_to_bf16(float(i * d + j));
  }
  l0::Mem device_a(l0ctx, l0::MemKind::Device, a.size() * sizeof(uint16_t));
  l0::Mem device_b(l0ctx, l0::MemKind::Device, b.size() * sizeof(uint16_t));
  l0::Mem device_c(l0ctx, l0::MemKind::Device, size_t(d) * d * sizeof(float));
  l0::Mem host_c(l0ctx, l0::MemKind::Host, size_t(d) * d * sizeof(float));
  l0::CmdList upload = l0::CmdList::immediate(l0ctx);
  upload.copy(device_a.ptr(), a.data(), a.size() * sizeof(uint16_t));
  upload.copy(device_b.ptr(), b.data(), b.size() * sizeof(uint16_t));

  sycl::queue& queue = runtime::prefill::sycl_queue(cx);
  PreparedGemm gemm = prepare(d, d, d, device_a.as<uint16_t>(), device_b.as<uint16_t>(),
                              device_c.as<float>(), hw_info, queue);
  if (gemm.can_implement != cutlass::Status::kSuccess)
    fail("16x16x16 layout GEMM cannot implement: status " + std::to_string(int(gemm.can_implement)));
  run_once(gemm, queue);
  l0::CmdList download = l0::CmdList::immediate(l0ctx);
  download.copy(host_c.ptr(), device_c.ptr(), size_t(d) * d * sizeof(float));

  bool matches_kn = true;
  bool matches_nk = true;
  const float* got = host_c.as<float>();
  for (uint32_t i = 0; i < d; ++i) {
    for (uint32_t j = 0; j < d; ++j) {
      const float kn = common::bf16_to_f32(b[size_t(i) * d + j]);
      const float nk = common::bf16_to_f32(b[size_t(j) * d + i]);
      matches_kn = matches_kn && std::memcmp(&got[size_t(i) * d + j], &kn, sizeof(float)) == 0;
      matches_nk = matches_nk && std::memcmp(&got[size_t(i) * d + j], &nk, sizeof(float)) == 0;
    }
  }
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(int(d), int(d), 1));
  std::printf("# 16x16x16 B-layout test, stride_B = ");
  cute::print(stride_b);
  std::printf("\n");
  std::printf("# B reference [K][N]: %s; [N][K]: %s\n", matches_kn ? "match" : "mismatch",
              matches_nk ? "match" : "mismatch");
  if (matches_kn == matches_nk)
    fail("16x16x16 layout test did not select exactly one B orientation");
  return {matches_kn, matches_kn ? "[K][N] row-major" : "[N][K] row-major"};
}

std::vector<uint16_t> make_b(uint32_t k, uint32_t n, bool b_is_kn, uint32_t seed) {
  std::vector<uint16_t> values(size_t(k) * n);
  for (uint32_t ki = 0; ki < k; ++ki)
    for (uint32_t ni = 0; ni < n; ++ni) {
      const uint16_t value = random_bf16(seed);
      values[b_is_kn ? size_t(ki) * n + ni : size_t(ni) * k + ki] = value;
    }
  return values;
}

float b_value(const std::vector<uint16_t>& b, uint32_t k, uint32_t n, bool b_is_kn, uint32_t ki,
              uint32_t ni) {
  return common::bf16_to_f32(b[b_is_kn ? size_t(ki) * n + ni : size_t(ni) * k + ki]);
}

struct Verification {
  double max_abs_err = 0.0;
  double tolerance = 0.0;
  bool bitwise = false;
};

Verification verify_outputs(l0::Context& l0ctx, const l0::Mem& first, const l0::Mem& second,
                            const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                            uint32_t m, uint32_t k, uint32_t n, bool b_is_kn) {
  const size_t elements = size_t(m) * n;
  const size_t bytes = elements * sizeof(float);
  const size_t chunk_bytes = std::min(kCompareChunkBytes, bytes);
  l0::Mem first_host(l0ctx, l0::MemKind::Host, chunk_bytes);
  l0::Mem second_host(l0ctx, l0::MemKind::Host, chunk_bytes);
  std::vector<size_t> samples;
  samples.reserve(kSamples);
  uint32_t seed = 0x7f4a7c15u;
  for (size_t i = 0; i < kSamples; ++i) samples.push_back(size_t(xorshift(seed)) % elements);
  std::sort(samples.begin(), samples.end());

  bool bitwise = true;
  double max_abs_err = 0.0;
  double max_abs_ref = 0.0;
  size_t sample = 0;
  for (size_t offset = 0; offset < bytes; offset += chunk_bytes) {
    const size_t now = std::min(chunk_bytes, bytes - offset);
    l0::CmdList copy = l0::CmdList::immediate(l0ctx);
    copy.copy(first_host.ptr(), static_cast<const char*>(first.ptr()) + offset, now);
    copy.copy(second_host.ptr(), static_cast<const char*>(second.ptr()) + offset, now);
    bitwise = bitwise && std::memcmp(first_host.ptr(), second_host.ptr(), now) == 0;

    const size_t first_element = offset / sizeof(float);
    const size_t end_element = first_element + now / sizeof(float);
    const float* got = first_host.as<float>();
    while (sample < samples.size() && samples[sample] < end_element) {
      const size_t index = samples[sample++];
      const uint32_t mi = uint32_t(index / n);
      const uint32_t ni = uint32_t(index % n);
      double reference = 0.0;
      for (uint32_t ki = 0; ki < k; ++ki)
        reference += double(common::bf16_to_f32(a[size_t(mi) * k + ki])) *
                     double(b_value(b, k, n, b_is_kn, ki, ni));
      max_abs_ref = std::max(max_abs_ref, std::fabs(reference));
      max_abs_err = std::max(max_abs_err, std::fabs(double(got[index - first_element]) - reference));
    }
  }
  const double tolerance = 5e-3 * max_abs_ref + 1e-4;
  return {max_abs_err, tolerance, bitwise};
}

struct Result {
  double ms = 0.0;
  double tflops = 0.0;
  size_t workspace = 0;
  int can_implement = 0;
  Verification verification;
};

Result run_case(l0::Context& l0ctx, runtime::prefill::Context& cx, const ProbeShape& shape, uint32_t m,
                bool b_is_kn, const cutlass::KernelHardwareInfo& hw_info, bool recorded) {
  const size_t a_elements = size_t(m) * shape.K;
  const size_t b_elements = size_t(shape.K) * shape.N;
  const size_t output_bytes = size_t(m) * shape.N * sizeof(float);
  std::vector<uint16_t> a = random_bf16s(a_elements, 0x9e3779b9u ^ m ^ shape.K ^ shape.N);
  std::vector<uint16_t> b = make_b(shape.K, shape.N, b_is_kn, 0x85ebca6bu ^ m ^ shape.K ^ shape.N);
  l0::Mem device_a(l0ctx, l0::MemKind::Device, a_elements * sizeof(uint16_t));
  l0::Mem device_b(l0ctx, l0::MemKind::Device, b_elements * sizeof(uint16_t));
  l0::Mem first(l0ctx, l0::MemKind::Device, output_bytes);
  l0::Mem second(l0ctx, l0::MemKind::Device, output_bytes);
  l0::CmdList upload = l0::CmdList::immediate(l0ctx);
  upload.copy(device_a.ptr(), a.data(), a_elements * sizeof(uint16_t));
  upload.copy(device_b.ptr(), b.data(), b_elements * sizeof(uint16_t));

  sycl::queue& queue = runtime::prefill::sycl_queue(cx);
  PreparedGemm first_run = prepare(m, shape.K, shape.N, device_a.as<uint16_t>(), device_b.as<uint16_t>(),
                                   first.as<float>(), hw_info, queue);
  if (first_run.can_implement != cutlass::Status::kSuccess)
    return {0.0, 0.0, first_run.workspace, int(first_run.can_implement), {}};
  run_once(first_run, queue);
  PreparedGemm second_run = prepare(m, shape.K, shape.N, device_a.as<uint16_t>(), device_b.as<uint16_t>(),
                                    second.as<float>(), hw_info, queue);
  if (second_run.can_implement != cutlass::Status::kSuccess)
    fail("second output unexpectedly cannot implement");
  run_once(second_run, queue);
  Verification verification = verify_outputs(l0ctx, first, second, a, b, m, shape.K, shape.N, b_is_kn);
  if (verification.max_abs_err > verification.tolerance)
    fail(std::string("sampled CPU reference check failed for ") + shape.name + " M=" + std::to_string(m));

  const double ms = recorded ? time_replays(first_run, queue) : 0.0;
  const double flops = 2.0 * double(m) * shape.K * shape.N;
  return {ms, recorded ? flops / (ms * 1e9) : 0.0, first_run.workspace, int(first_run.can_implement),
          verification};
}

void print_header() {
  std::printf("sycl-tla sha %s (pin %s)\n", B70_SYCL_TLA_SHA, B70_SYCL_TLA_PIN);
  std::printf("IGC env: SYCL_PROGRAM_COMPILE_OPTIONS=%s | IGC_VISAOptions=%s | "
              "IGC_VectorAliasBBThreshold=%s | IGC_ExtraOCLOptions=%s\n",
              env_or_unset("SYCL_PROGRAM_COMPILE_OPTIONS"), env_or_unset("IGC_VISAOptions"),
              env_or_unset("IGC_VectorAliasBBThreshold"), env_or_unset("IGC_ExtraOCLOptions"));
  std::printf("# Configuration: MainloopXeL1Staged<2>; TileShape 256x256x32; "
              "TiledMma XE_DPAS_TT<8,float,bf16>, subgroup layout 8x4x1 (32 subgroups); "
              "TileScheduler omitted (data-parallel PersistentScheduler).\n");
  std::printf("# Correctness is structural (layout/dtype/stride/dropped-k-tile), not a numerics bar: "
              "4096 fixed xorshift cells vs host double, tol = 5e-3*max|ref| + 1e-4.\n");
  std::printf("# Timing: %d replays, discard first %d, median of last %d; %d enqueues/replay; "
              "q.wait_and_throw closes every replay.\n", kReplays, kDroppedReplays,
              kReplays - kDroppedReplays, kTimedEnqueues);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool grid_only = argc == 2 && std::strcmp(argv[1], "--grid-only") == 0;
    const bool dump_one = argc == 2 && std::strcmp(argv[1], "--dump-one") == 0;
    if (argc > 2 || (argc == 2 && !grid_only && !dump_one))
      fail("usage: probe_prefill_gemm [--grid-only|--dump-one]");
    print_header();
    l0::Context l0ctx(0);
    std::printf("L0 device: %s\n", l0ctx.name().c_str());
    runtime::prefill::Context cx(l0ctx);
    cutlass::KernelHardwareInfo hw_info{};
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    print_device_and_launch_diagnostic(l0ctx, cx, hw_info);
    if (grid_only) return 0;
    const LayoutVerdict layout = settle_b_layout(l0ctx, cx, hw_info);
    std::printf("# B layout verdict: %s\n", layout.summary.c_str());
    if (dump_one) {
      // This is also the bounded execution path for an IGC shader-dump capture.
      std::printf("# --dump-one: the exact GemmUniversal instantiation ran only the 16x16x16 "
                  "layout control; no timed measurement was taken.\n");
      return 0;
    }
    std::printf("# workspace rule: every cell must report 0 bytes; no split-K atomics are instantiated.\n\n");
    std::printf("| shape | K×N | M | can_implement | ms | TFLOP/s | %% of 90 | max abs err | tol | bitwise |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");

    std::array<std::array<double, sizeof(kMs) / sizeof(kMs[0])>, sizeof(kShapes) / sizeof(kShapes[0])> rates{};
    bool all_identical = true;
    for (size_t shape_index = 0; shape_index < sizeof(kShapes) / sizeof(kShapes[0]); ++shape_index) {
      const ProbeShape& shape = kShapes[shape_index];
      const Result warm = run_case(l0ctx, cx, shape, kMs[0], layout.b_is_kn, hw_info, true);
      std::printf("# ramp control (discarded): %s M=%u, can_implement=%d, workspace=%zu B, %.3f ms, "
                  "%.2f TFLOP/s\n", shape.name, kMs[0], warm.can_implement, warm.workspace,
                  warm.ms, warm.tflops);
      for (size_t m_index = 0; m_index < sizeof(kMs) / sizeof(kMs[0]); ++m_index) {
        const uint32_t m = kMs[m_index];
        const Result result = run_case(l0ctx, cx, shape, m, layout.b_is_kn, hw_info, true);
        rates[shape_index][m_index] = result.tflops;
        all_identical = all_identical && result.verification.bitwise && result.workspace == 0 &&
                        result.can_implement == int(cutlass::Status::kSuccess);
        std::printf("| %s | %u×%u | %u | %d | %.3f | %.2f | %.1f%% | %.6g | %.6g | %s |\n",
                    shape.name, shape.K, shape.N, m, result.can_implement, result.ms, result.tflops,
                    result.tflops / 90.0 * 100.0, result.verification.max_abs_err,
                    result.verification.tolerance, result.verification.bitwise ? "identical" : "**DIFFER**");
        std::fflush(stdout);
      }
    }
    std::printf("\n## TFLOP/s matrix (measured, iterate-grade)\n\n");
    std::printf("| shape | M=512 | M=1024 | M=2048 | M=4096 |\n|---|---:|---:|---:|---:|\n");
    for (size_t shape_index = 0; shape_index < sizeof(kShapes) / sizeof(kShapes[0]); ++shape_index)
      std::printf("| %s | %.2f | %.2f | %.2f | %.2f |\n", kShapes[shape_index].name,
                  rates[shape_index][0], rates[shape_index][1], rates[shape_index][2],
                  rates[shape_index][3]);
    std::printf("\nDeterminism: %s; split-K atomics: none (data-parallel PersistentScheduler); "
                "workspace: 0 bytes for every cell.\n",
                all_identical ? "two runs bitwise equal for every cell" : "**FAILED - see row**");
    return all_identical ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_prefill_gemm FAILED: %s\n", e.what());
    return 1;
  }
}
