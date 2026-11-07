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
// pf_gemm_small over the rows pad(M, 8 NA) covers: same arguments as pf_gemm's minus the batch
// strides, `ldc` carrying ldx under the SiLU epilogue.
void launch_small(Context& cx, KernelCache& kc, const PfSmallTile& t, bool silu, uint32_t M,
                  uint32_t K, uint32_t N, const uint16_t* A, const uint16_t* B, void* C, uint32_t lda,
                  uint32_t ldb, uint32_t ldc) {
  require(N % t.cols() == 0, "N = " + std::to_string(N) + " is not a multiple of the small tile's " +
                                 std::to_string(t.cols()) + " columns");
  const uint32_t Mt = (M + t.rows() - 1) / t.rows() * t.rows();
  l0::Kernel& k = kc(kernels::pf_gemm_small_variant(t.na, t.nb, t.sn, silu), "pf_gemm_small");
  cx.launch(k, Mt / t.rows(), N / t.cols(), 1,
            {PtrArg(A), PtrArg(B), PtrArg(C), arg_val(Mt), arg_val(K), arg_val(N), arg_val(lda),
             arg_val(ldb), arg_val(ldc)});
}
}  // namespace

const PfSmallTile* pf_small_tile(uint32_t M) {
  // probe_small_chunk, one chunk's five Qwen3.8 linears (ms, 2026-11-08, card 1): pf_gemm ~580
  // at every M up to 1024. a1b2s1 90 at 16 rows; a2b2s1 102 / 114 / 147 at 32 / 64 / 128;
  // a4b2s1 196 at 256; a4b4s4 286 / 530 at 512 / 1024 (pf_gemm 587 / 605); above
  // kPfSmallMaxRows pf_gemm (its 256 x 256 tile fills the 32 cores).
  static constexpr PfSmallTile k16{1, 2, 1}, k128{2, 2, 1}, k256{4, 2, 1}, kBig{4, 4, 4};
  if (M == 0 || M > kPfSmallMaxRows) return nullptr;
  if (M <= 16) return &k16;
  if (M <= 128) return &k128;
  if (M <= 256) return &k256;
  return &kBig;
}

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

  const uint32_t lda = uint32_t(b.lda), ldb = uint32_t(b.ldb), ldc = uint32_t(b.ldc);
  if (const PfSmallTile* t = !transB && b.L == 1 ? pf_small_tile(b.M) : nullptr) {
    launch_small(cx, kc, *t, /*silu=*/false, b.M, b.K, N, A, B, C, lda, ldb, ldc);
    return;
  }
  l0::Kernel& k = kc(kernels::pf_gemm_variant(transB), "pf_gemm");
  const uint64_t sA = b.strideA, sB = b.strideB, sC = b.strideC;
  cx.launch(k, M / kPfGemmTile, N / kPfGemmTile, b.L,
            {PtrArg(A), PtrArg(B), PtrArg(C), arg_val(M), arg_val(b.K), arg_val(N),
             arg_val(lda), arg_val(ldb), arg_val(ldc), arg_val(sA), arg_val(sB), arg_val(sC)});
}

void gemm_l0_silu(Context& cx, KernelCache& kc, const GemmBatch& b, const uint16_t* A,
                  const uint16_t* B, uint16_t* X, uint32_t ldx) {
  require(b.M > 0 && b.N > 0, "empty problem");
  require(b.L == 1, "the fused-SiLU epilogue is one gate||up slab: L must be 1");
  require(b.K >= 32 && b.K % 32 == 0, "K = " + std::to_string(b.K) + " is not a multiple of 32");
  const uint32_t M = pad256(b.M);
  const uint32_t N = uint32_t(b.N);
  require(N % kPfGemmTile == 0, "N = " + std::to_string(N) + " is not a multiple of 256");
  require(b.lda >= b.K, "lda < K");
  require(b.ldb >= N, "ldb does not cover B");
  // Two columns of C collapse into one of x, so the slab covers N/2 x columns
  // and the row pitch must reach past them; the caller passes the WHOLE row.
  require(ldx >= N / 2, "ldx = " + std::to_string(ldx) + " does not cover the slab's N/2 columns");
  require((b.lda * 2) % 16 == 0 && (b.ldb * 2) % 16 == 0 && (size_t(ldx) * 2) % 16 == 0,
          "a pitch is not a multiple of 16 bytes");
  require(aligned64(A, 2, b.strideA, b.L), "A is not 64-byte aligned");
  require(aligned64(B, 2, b.strideB, b.L), "B is not 64-byte aligned");
  require(aligned64(X, 2, 0, 1), "X is not 64-byte aligned");

  const uint32_t lda = uint32_t(b.lda), ldb = uint32_t(b.ldb);
  if (const PfSmallTile* t = pf_small_tile(b.M)) {
    launch_small(cx, kc, *t, /*silu=*/true, b.M, b.K, N, A, B, X, lda, ldb, ldx);
    return;
  }
  l0::Kernel& k = kc(kernels::pf_gemm_silu_variant(), "pf_gemm");
  const uint64_t sA = b.strideA, sB = b.strideB, sC = 0;
  cx.launch(k, M / kPfGemmTile, N / kPfGemmTile, 1,
            {PtrArg(A), PtrArg(B), PtrArg(X), arg_val(M), arg_val(b.K), arg_val(N),
             arg_val(lda), arg_val(ldb), arg_val(ldx), arg_val(sA), arg_val(sB), arg_val(sC)});
}

}  // namespace runtime::prefill
