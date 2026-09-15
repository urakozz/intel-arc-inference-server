#pragma once
#include <cstdint>

#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

// The sycl-tla side of spec 2.1's two seams, and the build's facts about it. Exactly one of
// backend_sycl.cc (SYCL component on) and backend_sycl_absent.cc (off) is compiled into
// b70_prefill_host (src/runtime/prefill/CMakeLists.txt) -- no #ifdef in the walk.
namespace runtime::prefill {

PrefillBackend default_prefill_backend();   // what Engine::prefill runs when nothing was set
bool sycl_available();                       // is libb70_prefill.so linked?

// pf_linear's sycl-tla body, spec 2's exactly: dequant into the scratch, host wait, sycl-tla
// GEMM into `partials`, host wait. The absent build's version throws.
void linear_sycl(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::DeviceWeight& w,
                 const uint16_t* x, uint32_t M);
// attention's two batched GEMMs through sycl-tla. qk includes the SYCL->L0 wait that follows
// it (Phase::kAttnQk); pv does not (attn.cc waits once after the loop, Phase::kAttnPv).
void attn_qk_sycl(Context& cx, const GemmBatch& b, const uint16_t* q, const uint16_t* k, float* S);
void attn_pv_sycl(Context& cx, const GemmBatch& b, const uint16_t* P, const uint16_t* v, float* O);

}  // namespace runtime::prefill
