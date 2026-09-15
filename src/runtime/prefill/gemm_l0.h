#pragma once
#include <cstdint>

#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"     // GemmBatch - the same problem description sycl-tla takes
#include "runtime/prefill/kernels.h"

namespace runtime::prefill {

inline constexpr uint32_t kPfGemmTile = 256;
inline uint32_t pad256(uint32_t v) { return (v + kPfGemmTile - 1) / kPfGemmTile * kPfGemmTile; }

// Launch pf_gemm (src/kernels/prefill/pf_gemm.cl) on cx's list for the LOGICAL problem `b`.
// M is padded to 256 here; so is N when transB (the untransposed shapes, a 1024-column slab
// and P·V's N = 256, are already multiples). The padded rows and columns are computed and
// never read -- the CALLER guarantees they lie inside its buffers (spec 2.1 §3.1, §3.4).
// Throws on any 2D block-IO rule the kernel header states: K % 32, 64-byte bases (per batch),
// 16-byte pitches, leading dimensions that cover the matrix.
void gemm_l0(Context& cx, KernelCache& kc, const GemmBatch& b, const uint16_t* A,
             const uint16_t* B, float* C, bool transB);

}  // namespace runtime::prefill
