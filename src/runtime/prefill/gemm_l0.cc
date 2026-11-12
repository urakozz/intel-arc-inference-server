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
  cx.launch(k, N / t.cols(), Mt / t.rows(), 1,   // columns first (pf_gemm_small.cl)
            {PtrArg(A), PtrArg(B), PtrArg(C), arg_val(Mt), arg_val(K), arg_val(N), arg_val(lda),
             arg_val(ldb), arg_val(ldc)});
}
}  // namespace

const PfSmallTile* pf_small_tile(uint32_t M) {
  // probe_small_chunk on the engine's walk (each slab's dequant, then its GEMM), one chunk's five
  // Qwen3.8 linears with their dequants (ms, 2026-11-08, card 1): pf_gemm 680-770 at every M up
  // to 1024 (the dequants ~100 of it); A2_B2_S4 205-228 up to 224 rows, 239 at 256, 333 at 512,
  // 407 at 768, 460 at 896; A4_B2_S4 483 at 1024 (A2_B2_S4 513); above 1024 no tile beats pf_gemm
  // (A4_B4_S4 792-830 against 805-823 at 1280-2048).
  static constexpr PfSmallTile kTo896{2, 2, 4}, kTo1024{4, 2, 4};
  if (M == 0 || M > kPfSmallMaxRows) return nullptr;
  return M <= 896 ? &kTo896 : &kTo1024;
}

const PfSmallTile* pf_q4_tile(uint32_t M, bool silu) {
  // probe_fused_small against the slab walk (2026-11-08, card 1; interleaved medians, the probe
  // doc has every tile): the plain linears take A4_B1_S1 to 32 rows and A8_B1_S2 to 384 (a 64-row
  // tile a sub-group: the conversion repeats once per 64 rows; past 384 out_proj and qkv||z lose),
  // gate||up's SiLU form A1_B2_S4 to 8 rows and A4_B2_S1 to 192 (at 224 it is 4 % ahead, at 256
  // behind). One chunk's five linears: 51 ms at 1-8 rows, 56 at 16, 59 at 32, 76 at 64, 112 at
  // 128, 156 at 192, 247 at 384 (gate||up on the slab), against the slab walk's 236-297.
  static constexpr PfSmallTile kPlainTo32{4, 1, 1}, kPlain{8, 1, 2}, kSiluTo8{1, 2, 4}, kSilu{4, 2, 1};
  // Open-work 3d (2026-11-09, whole box): paired A8_B1_S4 sub-groups exchange gate bf16 in
  // SLM, then up folds SiLU into the output, all in one launch. Every row count 193-320 is
  // bitwise and 16.03-26.81% below the interleaved slab control; 321 misses the fixed 5% bar.
  // The earlier two-pass form gained in isolation but regressed on the real engine.
  static constexpr PfSmallTile kSiluSlm{8, 1, 4};
  if (M == 0) return nullptr;
  if (silu) {
    if (M > kPfQ4SiluMaxRows) return nullptr;
    if (M > 192) return &kSiluSlm;
    return M <= 8 ? &kSiluTo8 : &kSilu;
  }
  if (M > kPfQ4MaxRows) return nullptr;
  return M <= 32 ? &kPlainTo32 : &kPlain;
}

void gemm_q4(Context& cx, KernelCache& kc, const PfSmallTile& t, uint32_t layout, bool silu, uint32_t M,
             uint32_t K, uint32_t N, const uint16_t* A, uint32_t lda, const void* qw, const void* scales,
             void* C, uint32_t ldc) {
  require(M > 0 && N > 0, "empty problem");
  require(layout == 0 || layout == 1, "layout " + std::to_string(layout) + " is neither 0 nor 1");
  require(layout == 1 || scales != nullptr, "a layout-0 weight needs its scales");
  require(K % 64 == 0, "K = " + std::to_string(K) + " is not a multiple of the group size 64");
  const bool slm = silu && t.nb == 1;
  require(N % t.cols() == 0, "N = " + std::to_string(N) + " is not a multiple of the q4 tile's " +
                                 std::to_string(t.cols()) + " columns");
  require(!silu || t.nb % 2 == 0 || slm, "the SiLU epilogue needs paired atoms or NB1 sub-groups");
  require(!slm || (layout == 0 && t.na == 8 && (t.sn == 2 || t.sn == 4)),
          "the SLM SiLU epilogue needs A8_B1_S2/S4 and layout 0");
  require(lda >= K && ldc >= (silu ? N / 2 : N), "a leading dimension does not cover its matrix");
  require((lda * 2) % 16 == 0 && (size_t(ldc) * (silu ? 2 : 4)) % 16 == 0 && (N * 4) % 16 == 0,
          "a pitch is not a multiple of 16 bytes");
  require(aligned64(A, 2, 0, 1) && aligned64(qw, 4, 0, 1) && aligned64(C, 4, 0, 1) &&
              (scales == nullptr || aligned64(scales, 2, 0, 1)),
          "a base is not 64-byte aligned");
  const uint32_t Mt = (M + t.rows() - 1) / t.rows() * t.rows();
  const std::string variant = slm ? kernels::pf_gemm_q4_slm_variant(t.na, t.sn)
                                  : kernels::pf_gemm_q4_variant(t.na, t.nb, t.sn, layout, silu);
  l0::Kernel& k = kc(variant, "pf_gemm_q4");
  cx.launch(k, N / t.cols(), Mt / t.rows(), 1,   // columns first (pf_gemm_q4.cl)
            {PtrArg(A), PtrArg(qw), PtrArg(scales), PtrArg(C), arg_val(Mt), arg_val(K), arg_val(N),
             arg_val(lda), arg_val(ldc)});
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

void tiled_gemm(Context& cx, KernelCache& kc, const void* w, uint32_t N, uint32_t K, const uint16_t* A, uint32_t C,
                uint16_t* slab, float* out) {
  require(K % 8 == 0, "K = " + std::to_string(K) + " is not whole 8-k groups");
  const uint32_t P = kernels::pf_route_slab_pitch(N);
  cx.launch(kc(kernels::pf_route_slab_variant(K, N), "pf_route_slab"), P / 256, K / 8, 1,
            {PtrArg(w), PtrArg(slab)});
  gemm_l0(cx, kc, GemmBatch{C, K, P, 1, K, P, P, 0, 0, 0}, A, slab, out, /*transB=*/false);
}

}  // namespace runtime::prefill
