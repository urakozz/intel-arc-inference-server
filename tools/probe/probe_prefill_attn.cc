// P4 - compile-only first fork for sycl-tla FMHA at head_dim 256.
//
// This intentionally has no timing or correctness path yet. Its sole job is to
// instantiate the generic configuration named in the pre-registration before
// the probe grows measurement machinery. It reads no engine buffers and does
// not change any shipped path.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cutlass/util/packed_stride.hpp"
#include "flash_attention/fmha_configuration.hpp"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"
#include "tla_pin.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// CuTe maps printf to SYCL device printf; this probe prints its host verdict.
#undef printf

namespace {

using FMHA256Config = cutlass::flash_attention::FMHAConfigGenWithTileShape<
    cutlass::flash_attention::FMHAMode::Prefill,
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    float,
    /* Causal = */ true,
    /* VarLen = */ true,
    /* CachedKV = */ false,
    /* PagedKV = */ false,
    /* Persistent = */ false,
    /* BlockScale = */ false,
    /* WgTileQ = */ 128,
    /* WgTileK = */ 32,
    /* WgTileV = */ 64,
    /* SgTileQ = */ 16,
    /* SgTileK = */ 32,
    /* HeadDimQK = */ 256,
    /* HeadDimV = */ 256>::type;

using FMHA256Kernel = typename FMHA256Config::FMHAKernel;

using FMHA256V8Config = cutlass::flash_attention::FMHAConfigGenWithTileShape<
    cutlass::flash_attention::FMHAMode::Prefill,
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    float,
    /* Causal = */ true,
    /* VarLen = */ true,
    /* CachedKV = */ false,
    /* PagedKV = */ false,
    /* Persistent = */ false,
    /* BlockScale = */ false,
    /* WgTileQ = */ 256,
    /* WgTileK = */ 32,
    /* WgTileV = */ 32,
    /* SgTileQ = */ 16,
    /* SgTileK = */ 32,
    /* HeadDimQK = */ 256,
    /* HeadDimV = */ 256>::type;

using FMHA256V8Kernel = typename FMHA256V8Config::FMHAKernel;

static_assert(FMHA256Config::Causal);
static_assert(!FMHA256Config::Persistent);
static_assert(sizeof(FMHA256Kernel::Params) > 0);

// This is the runner's launch path copied without ExampleRunner: that helper
// pulls oneMKL random-fill headers, an optional benchmark dependency absent from
// the system oneAPI install. The production kernel needs only this launcher.
template <class Kernel>
void instantiate_device_launcher(typename Kernel::Params params) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const auto block = Kernel::get_block_shape();
  const auto grid = Kernel::get_grid_shape(params);
  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);
  const compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
  const compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
  const compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
  compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel, false>(policy, params);
}

uint32_t xorshift(uint32_t& value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

template <class Kernel>
double run_case(uint32_t kC) {
  constexpr int kDepth = 4096;
  constexpr int kQHeads = 24;
  constexpr int kKvHeads = 4;
  constexpr int kHeadDim = 256;

  l0::Context l0ctx(0);
  runtime::prefill::Context cx(l0ctx);
  compat::set_default_queue(runtime::prefill::sycl_queue(cx));
  cutlass::KernelHardwareInfo hw_info{};
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  l0::Mem q(l0ctx, l0::MemKind::Device, size_t(kC) * kQHeads * kHeadDim * sizeof(uint16_t));
  l0::Mem k(l0ctx, l0::MemKind::Device, size_t(kDepth) * kKvHeads * kHeadDim * sizeof(uint16_t));
  l0::Mem v(l0ctx, l0::MemKind::Device, size_t(kDepth) * kKvHeads * kHeadDim * sizeof(uint16_t));
  l0::Mem o(l0ctx, l0::MemKind::Device, size_t(kC) * kQHeads * kHeadDim * sizeof(uint16_t));
  l0::Mem cumulative(l0ctx, l0::MemKind::Shared, 6 * sizeof(int));
  auto* lengths = cumulative.as<int>();
  lengths[0] = 0;
  lengths[1] = kC;
  lengths[2] = 0;
  lengths[3] = kDepth;
  lengths[4] = 0;
  lengths[5] = 0;

  std::vector<uint16_t> host_kv(size_t(kDepth) * kKvHeads * kHeadDim);
  std::vector<uint16_t> host_q(size_t(kC) * kQHeads * kHeadDim);
  uint32_t state = 0x9e3779b9u;
  for (uint16_t& element : host_q) element = uint16_t(0x3c00u | (xorshift(state) & 0x00ffu));
  for (uint16_t& element : host_kv) element = uint16_t(0x3c00u | (xorshift(state) & 0x00ffu));
  l0::CmdList upload = l0::CmdList::immediate(l0ctx);
  upload.copy(q.ptr(), host_q.data(), host_q.size() * sizeof(uint16_t));
  upload.copy(k.ptr(), host_kv.data(), host_kv.size() * sizeof(uint16_t));
  upload.copy(v.ptr(), host_kv.data(), host_kv.size() * sizeof(uint16_t));

  using VariableLength = cutlass::fmha::collective::VariableLength;
  typename Kernel::ProblemShape shape{
      1, kQHeads, kKvHeads,
      VariableLength{static_cast<int>(kC), lengths}, VariableLength{kDepth, lengths + 2},
      VariableLength{0, lengths + 4}, kHeadDim, kHeadDim};
  typename Kernel::Arguments arguments{
      {shape,
       q.as<cutlass::bfloat16_t>(), cute::make_stride(kQHeads * kHeadDim, cute::_1{}, kHeadDim, 0),
       k.as<cutlass::bfloat16_t>(), cute::make_stride(kKvHeads * kHeadDim, cute::_1{}, kHeadDim, 0),
       v.as<cutlass::bfloat16_t>(), cute::make_stride(cute::_1{}, kKvHeads * kHeadDim, kHeadDim, 0),
       o.as<cutlass::bfloat16_t>(), cute::make_stride(kQHeads * kHeadDim, cute::_1{}, kHeadDim, 0),
       nullptr, {}, nullptr, {}, nullptr, {}, 1.0f, 1.0f, 1.0f, 32, nullptr, {}, nullptr, {}},
      {0.0625f, nullptr, 0, nullptr}, {}, hw_info};
  if (!Kernel::can_implement(arguments))
    throw std::runtime_error("FMHA rejected the prescribed in-place C/depth arguments");
  const auto params = Kernel::to_underlying_arguments(arguments, nullptr);
  instantiate_device_launcher<Kernel>(params);
  runtime::prefill::sycl_queue(cx).wait_and_throw();
  std::array<double, 8> samples{};
  for (double& sample : samples) {
    const auto started = std::chrono::steady_clock::now();
    instantiate_device_launcher<Kernel>(params);
    runtime::prefill::sycl_queue(cx).wait_and_throw();
    sample = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
  }
  std::sort(samples.begin() + 3, samples.end());
  return samples[5];
}

}  // namespace

int main() {
  try {
    std::printf("sycl-tla sha %s (pin %s)\n", B70_SYCL_TLA_SHA, B70_SYCL_TLA_PIN);
    std::printf("# P4 uses Context's in-order queue after compat::set_default_queue; host wall time, 8 replays, first 3 dropped.\n");
    std::printf("| config | VTiles | C | depth | us/launch | ms x 16 layers |\n|---|---:|---:|---:|---:|---:|\n");
    for (const uint32_t c : {1024u, 2048u, 4096u}) {
      const double v4_us = run_case<FMHA256Kernel>(c);
      std::printf("| 128x32x64 | 4 | %u | 4096 | %.3f | %.3f |\n", c, v4_us, v4_us * 16.0 / 1000.0);
      const double v8_us = run_case<FMHA256V8Kernel>(c);
      std::printf("| 256x32x32 | 8 | %u | 4096 | %.3f | %.3f |\n", c, v8_us, v8_us * 16.0 / 1000.0);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "probe_prefill_attn FAILED: %s\n", error.what());
    return 1;
  }
}
