#include "runtime/prefill/backend.h"

#include <stdexcept>
#include <string>

#include "runtime/prefill/gemm_sycl.h"

// The SYCL component is NOT linked (cmake/prefill.cmake: icpx unusable or -DB70_PREFILL=OFF).
// The L0 backend is the only one, and every sycl-tla entry point says so by name instead of
// failing to link (spec 2.1 §3.5).
namespace runtime::prefill {
namespace {
[[noreturn]] void absent(const char* what) {
  throw std::runtime_error(std::string("runtime::prefill: ") + what +
                           " needs the SYCL component, and this build has none (cmake/prefill.cmake);"
                           " use the l0 prefill backend");
}
}  // namespace

PrefillBackend default_prefill_backend() { return PrefillBackend::L0Int8; }   // spec 5, 2026-09-24
bool sycl_available() { return false; }

SyclSide* sycl_side_create(ze_context_handle_t, ze_device_handle_t) { absent("the SYCL side"); }
void sycl_side_destroy(SyclSide*) {}
void sycl_side_wait(SyclSide*) { absent("the SYCL side"); }
void* sycl_side_queue(SyclSide*) { absent("the SYCL side"); }
void* sycl_side_context(SyclSide*) { absent("the SYCL side"); }

bool gemm_bf16_supports_transb() { return false; }
void gemm_bf16_batched_grid(GemmBatch, bool, uint32_t[3]) { absent("gemm_bf16_batched_grid"); }
void gemm_bf16_batched(Context&, GemmBatch, const uint16_t*, const uint16_t*, float*, bool) {
  absent("gemm_bf16_batched");
}
void gemm_bf16(Context&, GemmDims, const uint16_t*, const uint16_t*, float*) { absent("gemm_bf16"); }
void gemm_bf16_batched_on(SyclSide*, GemmBatch, const uint16_t*, const uint16_t*, float*, bool) {
  absent("gemm_bf16_batched_on");
}
void linear_sycl(Context&, KernelCache&, PrefillScratch&, const loader::DeviceWeight&,
                 const uint16_t*, uint32_t) {
  absent("the sycl-tla prefill backend");
}
void attn_qk_sycl(Context&, const GemmBatch&, const uint16_t*, const uint16_t*, float*) {
  absent("the sycl-tla prefill backend");
}
void attn_pv_sycl(Context&, const GemmBatch&, const uint16_t*, const uint16_t*, float*) {
  absent("the sycl-tla prefill backend");
}

}  // namespace runtime::prefill
