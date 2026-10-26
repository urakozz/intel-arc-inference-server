#pragma once

#include <cstddef>
#include <cstdint>

// The prefill GEMM's problem description (spec 2 §3.2, ruling A15). `gemm_l0` (gemm_l0.h)
// runs it on the Level Zero list.
namespace runtime::prefill {

// Ruling A15. `lda`/`ldb`/`ldc` are ROW PITCHES in elements; `strideA`/`strideB`
// /`strideC` are element strides between batch entries. A `strideB` of 0 is
// legal and is how GQA shares one kv-head across its six q-heads.
// C[l][M][N] fp32 = A[l][M][K] bf16 · B[l][K][N] bf16 for l in [0, L); with `transB`, B[l] is
// [N][K] row-major with row pitch `ldb` - the K/V cache's own geometry, so QKᵀ reads it in place.
struct GemmBatch {
  uint32_t M, K, N, L;
  size_t lda, ldb, ldc;
  size_t strideA, strideB, strideC;
};

}  // namespace runtime::prefill
