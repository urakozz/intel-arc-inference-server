#pragma once

// The ONE sycl-tla instantiation the prefill path uses, as two alias chains
// that differ in a single line: `XeGemm` takes B row-major `[K][N]`,
// `XeGemmT` takes B row-major `[N][K]` (ColumnMajor from the GEMM's view).
//
// The configuration is Intel's, copied from
// examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp at the pinned revision
// and already exercised by P2 (tools/probe/probe_prefill_gemm.cc): work-group
// tile <256,256,32>, XE_DPAS_TT<8, float, bfloat16_t> (8x16x16, fp32
// accumulate), 8x4x1 = 32 subgroups, MainloopXeL1Staged<2>, all operands
// row-major, IntelXeGeneric epilogue. Spec 2 §3.2: this is inherited, not
// written. The CUTLASS headers arrive through -isystem so this file's own code
// still answers to -Werror.
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <cute/tensor.hpp>

// cute's printing headers map printf onto SYCL's device printf; host code in
// the including TU must keep std::printf.
#undef printf

namespace runtime::prefill::xe {

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = cute::bfloat16_t;
using ElementInputB = cute::bfloat16_t;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;
using TiledMma = typename cutlass::gemm::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_DPAS_TT<8, float, cute::bfloat16_t>>, cute::Layout<TileShape>,
    cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>,
                 cute::Stride<cute::_4, cute::_1, cute::_0>>>::TiledMMA;

constexpr int kPipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<kPipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(cute::tile_shape(TiledMma()))>;

// ElementC = void, deliberately, where Intel's example passes
// ElementAccumulator: `is_source_supported = !is_void_v<ElementC>`
// (xe_epilogue.hpp), so `void` deletes the epilogue's C load entirely. This
// GEMM never accumulates into C -- alpha = 1, beta = 0, D is written outright.
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy, TileShape, void, void, cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallbacks, void, void>;
static_assert(cute::is_void_v<typename CollectiveEpilogue::ElementC>,
              "prefill GEMM is D = A*B: the epilogue must not read C");

// The determinism decision, spelled out rather than defaulted (spec §6.4).
// PersistentScheduler is the ONLY scheduler xe_gemm.hpp accepts (its
// static_assert says so by name) and it is pure data-parallel: the grid is
// ceil(M/256) x ceil(N/256) x L, one work-group per output tile, the whole K
// loop inside that work-group. There is no cross-work-group reduction and no
// atomic on this path. Naming it here rather than passing `void` is what stops
// a future sycl-tla bump from changing the default under us. Stream-K
// (KernelXeCooperative + StreamKScheduler) is what spec §6.4 forbids; it is not
// reachable from these types.
template <class LayoutB_>
struct Chain {
  using LayoutB = LayoutB_;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma,
      void, void, void, cute::identity,    // A: auto copy atom -> XE_LOAD_2D
      void, void, void, cute::identity>;   // B: auto copy atom -> VNNI or transpose
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;  // (M, K, L)
  using StrideB = typename GemmKernel::StrideB;  // (N, K, L)  <-- N first
  using StrideC = typename GemmKernel::StrideC;  // (M, N, L)
};

using XeGemm = Chain<cutlass::layout::RowMajor>;
// B as [N][K] row-major -- what a K/V cache row already is. Whether this
// instantiates at all is plan 6d Task 1 Step 3's question; if it does not, the
// build fails here and the fallback is a host-side transpose kernel.
using XeGemmT = Chain<cutlass::layout::ColumnMajor>;

}  // namespace runtime::prefill::xe
