// The batched prefill GEMM (ruling A14/A15, plan 6d-composed Task 1) and, as a
// forward to it at L = 1, the plain `gemm_bf16` (plan 6c Task 1). One
// instantiated configuration, one determinism argument, one place a sycl-tla
// bump can change behaviour.
//
// The only additions over Intel's 00_bmg_gemm configuration are (a) L in the
// problem shape and the third stride mode, (b) runtime row pitches, (c) a
// ColumnMajor-B chain for `transB`.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "sycl/xe_gemm_config.h"

#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"
#include "runtime/prefill/gemm.h"

// Plan 6d Task 1 Step 3 settles whether the ColumnMajor-B chain instantiates at
// the pin. It does; if a bump ever breaks it, define this to 0 and the fallback
// is a host-side `pf_k_transpose` (0.23 ms/chunk, 33,554,432 B), NOT a silent
// change of layout.
#ifndef B70_PREFILL_TRANSB
#define B70_PREFILL_TRANSB 1
#endif

namespace runtime::prefill {
namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::gemm_bf16_batched: " + what);
}

// The Xe 2D block copies move 128 bits at a time -- 8 bf16 or 4 fp32 -- and the
// base of a block load must be 64-byte aligned. Checking it here turns
// can_implement's opaque kErrorInvalidProblem into a sentence that names the
// operand.
void check_operand(const void* p, size_t ld, const char* who) {
  require(p != nullptr, std::string(who) + " is null");
  require(reinterpret_cast<uintptr_t>(p) % 64 == 0,
          std::string(who) + " base is not 64-byte aligned");
  require(ld % 8 == 0, std::string(who) + " row pitch " + std::to_string(ld) +
                           " is not a multiple of 8 elements");
}

template <class G>
typename G::GemmKernel::Arguments make_args(const GemmBatch& b, const uint16_t* A,
                                            const uint16_t* B, float* C,
                                            typename G::StrideB sB) {
  const int M = int(b.M), K = int(b.K), N = int(b.N), L = int(b.L);
  const typename G::StrideA sA =
      cute::make_stride(int64_t(b.lda), cute::Int<1>{}, int64_t(b.strideA));
  const typename G::StrideC sC =
      cute::make_stride(int64_t(b.ldc), cute::Int<1>{}, int64_t(b.strideC));
  // sm_count is unused by the static Xe scheduler, so a default-constructed
  // hw_info avoids compat::get_device() -- which would consult the compat
  // device manager rather than the L0 context this engine owns.
  cutlass::KernelHardwareInfo hw_info{};
  return typename G::GemmKernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {reinterpret_cast<const cute::bfloat16_t*>(A), sA,
       reinterpret_cast<const cute::bfloat16_t*>(B), sB},
      // alpha = 1, beta = 0, ptr_C = nullptr: with ElementC = void the epilogue
      // never reads a source, so D = A*B outright.
      {{1.0f, 0.0f}, nullptr, sC, C, sC},
      hw_info};
}

template <class G>
void launch(Context& cx, const GemmBatch& b, const uint16_t* A, const uint16_t* B, float* C,
            typename G::StrideB sB) {
  const auto args = make_args<G>(b, A, B, C, sB);
  require(G::Gemm::can_implement(args) == cutlass::Status::kSuccess,
          "sycl-tla cannot implement M=" + std::to_string(b.M) + " K=" + std::to_string(b.K) +
              " N=" + std::to_string(b.N) + " L=" + std::to_string(b.L));
  // get_workspace_size returns 0 for this kernel at the pin; if a bump ever
  // changes that, the wrapper must grow an allocation rather than pass nullptr.
  require(G::Gemm::get_workspace_size(args) == 0,
          "this sycl-tla configuration now wants a workspace; the wrapper passes none");
  // cudaStream_t IS sycl::queue* under CUTLASS_ENABLE_SYCL, and the adapter
  // uses it instead of compat::get_default_queue() whenever it is non-null.
  // That one line is the whole L0<->SYCL interop story for the GEMM: our queue,
  // built from the engine's own ze_context/ze_device, IS the launch queue.
  sycl::queue& q = runtime::prefill::sycl_queue(cx);
  typename G::Gemm op;
  require(op.initialize(args, nullptr, &q) == cutlass::Status::kSuccess,
          "GemmUniversalAdapter::initialize failed");
  require(op.run(&q) == cutlass::Status::kSuccess, "GemmUniversalAdapter::run failed");
  // Asynchronous by contract: the caller syncs (Context::wait()).
}

// B is [K][N] row-major with row pitch ldb. StrideB is over (N, K, L), so it is
// (1, ldb, strideB) -- N first. That mode order is interfaces.md's "easiest bug
// here", and the non-packed-pitch case in gemm_batched_test is what catches it.
xe::XeGemm::StrideB stride_b_rowmajor(const GemmBatch& b) {
  return cute::make_stride(cute::Int<1>{}, int64_t(b.ldb), int64_t(b.strideB));
}

}  // namespace

bool gemm_bf16_supports_transb() { return B70_PREFILL_TRANSB != 0; }

void gemm_bf16_batched(Context& cx, GemmBatch b, const uint16_t* A, const uint16_t* B, float* C,
                       bool transB) {
  require(b.M && b.K && b.N && b.L, "M, K, N and L must all be non-zero");
  require(b.K % 8 == 0, "K must be a multiple of 8 bf16 elements, got " + std::to_string(b.K));
  require(b.N % 8 == 0, "N must be a multiple of 8 bf16 elements, got " + std::to_string(b.N));
  check_operand(A, b.lda, "A");
  check_operand(B, b.ldb, "B");
  check_operand(C, b.ldc, "C");
  if (transB) {
#if B70_PREFILL_TRANSB
    // B is [N][K] row-major with row pitch ldb -> ColumnMajor from the GEMM's
    // view. StrideB over (N, K, L) is (ldb, 1, strideB).
    launch<xe::XeGemmT>(cx, b, A, B, C,
                        cute::make_stride(int64_t(b.ldb), cute::Int<1>{}, int64_t(b.strideB)));
#else
    require(false, "this build has no ColumnMajor-B chain (B70_PREFILL_TRANSB=0)");
#endif
  } else {
    launch<xe::XeGemm>(cx, b, A, B, C, stride_b_rowmajor(b));
  }
}

void gemm_bf16(Context& cx, GemmDims d, const uint16_t* A, const uint16_t* B, float* C) {
  gemm_bf16_batched(cx, GemmBatch{d.M, d.K, d.N, 1, d.K, d.N, d.N, 0, 0, 0}, A, B, C, false);
}

void gemm_bf16_batched_grid(GemmBatch b, bool transB, uint32_t out_xyz[3]) {
  // The grid the instantiated kernel actually launches, reported rather than
  // reconstructed from the tile shape. Exists for the probe/test rows that must
  // print a measured rate beside its work-group count; production never calls
  // it.
  const auto store = [&](const auto& g) {
    out_xyz[0] = g.x;
    out_xyz[1] = g.y;
    out_xyz[2] = g.z;
  };
  out_xyz[0] = out_xyz[1] = out_xyz[2] = 0;
  if (transB) {
#if B70_PREFILL_TRANSB
    store(xe::XeGemmT::Gemm::get_grid_shape(
        make_args<xe::XeGemmT>(b, nullptr, nullptr, nullptr,
                               cute::make_stride(int64_t(b.ldb), cute::Int<1>{},
                                                 int64_t(b.strideB)))));
#endif
  } else {
    store(xe::XeGemm::Gemm::get_grid_shape(
        make_args<xe::XeGemm>(b, nullptr, nullptr, nullptr, stride_b_rowmajor(b))));
  }
}

}  // namespace runtime::prefill
