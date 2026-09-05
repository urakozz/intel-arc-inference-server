#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime/prefill/context.h"

// The prefill GEMM (spec 2 §3.2, interfaces.md stream S1). Like
// runtime/prefill/context.h this header is SYCL-free: the declarations below
// are implemented in the icpx half of the build (src/sycl/gemm_batched.cc) and
// called from g++ translation units. Nothing here names a cutlass or sycl type.
namespace runtime::prefill {

struct GemmDims {
  uint32_t M, K, N;
};

// C[M][N] fp32 = A[M][K] bf16 · B[K][N] bf16, packed pitches (lda = K,
// ldb = ldc = N). Deterministic: no split-K, no atomics. Throws on unsupported
// dims. M is a runtime argument. A forward to gemm_bf16_batched at L = 1, so
// there is ONE instantiated code path and ONE determinism argument.
void gemm_bf16(Context& cx, GemmDims d, const uint16_t* A, const uint16_t* B, float* C);

// Ruling A15. `lda`/`ldb`/`ldc` are ROW PITCHES in elements; `strideA`/`strideB`
// /`strideC` are element strides between batch entries. A `strideB` of 0 is
// legal and is how GQA shares one kv-head across its six q-heads.
struct GemmBatch {
  uint32_t M, K, N, L;
  size_t lda, ldb, ldc;
  size_t strideA, strideB, strideC;
};

// C[l][M][N] fp32 = A[l][M][K] bf16 · B[l][K][N] bf16 for l in [0, L).
// With `transB`, B[l] is [N][K] row-major with row pitch `ldb` instead - the
// K/V cache's own geometry, so QKᵀ reads it in place. Same determinism
// contract as gemm_bf16: the batch is a third GRID dimension, one work-group
// per output tile, the whole K loop inside it.
void gemm_bf16_batched(Context& cx, GemmBatch b, const uint16_t* A, const uint16_t* B,
                       float* C, bool transB = false);

// True iff the ColumnMajor-B chain instantiated in this build, i.e. whether
// `transB = true` is a legal argument above (plan 6d Task 1 Step 3). A test
// that would otherwise silently skip its transB case asks this instead.
bool gemm_bf16_supports_transb();

// The grid the instantiated kernel launches for `b`, from the kernel's own
// get_grid_shape -- reported, not reconstructed from the tile shape. This
// exists for the probe/test rows that must print a rate beside its work-group
// count (plan 6d Task 1 Step 6 case 8); the production path never calls it.
void gemm_bf16_batched_grid(GemmBatch b, bool transB, uint32_t out_xyz[3]);

}  // namespace runtime::prefill
