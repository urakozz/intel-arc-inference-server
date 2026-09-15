#pragma once
#include <cstdint>

#include "runtime/prefill/gemm.h"
#include "runtime/prefill/sycl_side.h"

// The sycl-tla GEMM as libb70_prefill.so exports it (spec 2.1 §3.5). It takes the SyclSide
// rather than the Context so the shared library references no symbol of the g++ host archive
// (a .so cannot resolve those from the executable). gemm.h's Context-taking gemm_bf16 /
// gemm_bf16_batched are wrappers in backend_sycl.cc.
namespace runtime::prefill {
void gemm_bf16_batched_on(SyclSide* side, GemmBatch b, const uint16_t* A, const uint16_t* B,
                          float* C, bool transB);
}  // namespace runtime::prefill
