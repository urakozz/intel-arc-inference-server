// tla_smoke.cc -- compile-only: does the pinned sycl-tla parse under our
// flags? Nothing calls this. See src/sycl/CMakeLists.txt.
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include "tla_pin.h"

namespace {
int tla_smoke_rank() { return int(cute::rank(cute::Shape<cute::_1, cute::_1>{})); }
}  // namespace

const char* b70_tla_sha() { return B70_SYCL_TLA_SHA; }
int b70_tla_smoke() { return tla_smoke_rank(); }
