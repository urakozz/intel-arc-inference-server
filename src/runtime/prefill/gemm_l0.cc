#include "runtime/prefill/gemm_l0.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"

namespace runtime::prefill {
namespace {
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::gemm_l0: " + what);
}
bool aligned64(const void* p, size_t elem_bytes, size_t stride_elems, uint32_t L) {
  const auto base = reinterpret_cast<uintptr_t>(p);
  if (base % 64 != 0) return false;
  return L <= 1 || (stride_elems * elem_bytes) % 64 == 0;
}
}  // namespace

void gemm_l0(Context& cx, KernelCache& kc, const GemmBatch& b, const uint16_t* A,
             const uint16_t* B, float* C, bool transB) {
  require(b.M > 0 && b.N > 0 && b.L > 0, "empty problem");
  require(b.K >= 32 && b.K % 32 == 0, "K = " + std::to_string(b.K) + " is not a multiple of 32");
  const uint32_t M = pad256(b.M);
  const uint32_t N = transB ? pad256(b.N) : b.N;
  require(N % kPfGemmTile == 0,
          "N = " + std::to_string(b.N) + " is not a multiple of 256 (untransposed B is a slab "
          "or P·V's 256 columns; the caller pads everything else)");
  require(b.lda >= b.K, "lda < K");
  require(transB ? b.ldb >= b.K : b.ldb >= N, "ldb does not cover B");
  require(b.ldc >= N, "ldc < N");
  require((b.lda * 2) % 16 == 0 && (b.ldb * 2) % 16 == 0 && (b.ldc * 4) % 16 == 0,
          "a pitch is not a multiple of 16 bytes");
  require(aligned64(A, 2, b.strideA, b.L), "A (or a batch of it) is not 64-byte aligned");
  require(aligned64(B, 2, b.strideB, b.L), "B (or a batch of it) is not 64-byte aligned");
  require(aligned64(C, 4, b.strideC, b.L), "C (or a batch of it) is not 64-byte aligned");

  l0::Kernel& k = kc(kernels::pf_gemm_variant(transB), "pf_gemm");
  const uint32_t lda = uint32_t(b.lda), ldb = uint32_t(b.ldb), ldc = uint32_t(b.ldc);
  const uint64_t sA = b.strideA, sB = b.strideB, sC = b.strideC;
  cx.launch(k, M / kPfGemmTile, N / kPfGemmTile, b.L,
            {PtrArg(A), PtrArg(B), PtrArg(C), arg_val(M), arg_val(b.K), arg_val(N),
             arg_val(lda), arg_val(ldb), arg_val(ldc), arg_val(sA), arg_val(sB), arg_val(sC)});
}

}  // namespace runtime::prefill
