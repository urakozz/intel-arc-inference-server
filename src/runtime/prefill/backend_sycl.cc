#include "runtime/prefill/backend.h"

#include <stdexcept>
#include <string>

#include "runtime/prefill/dequant.h"
#include "runtime/prefill/gemm_sycl.h"
#include "runtime/prefill/profile.h"

// The SYCL component is linked, and the two seams run spec 2's bodies verbatim; the default
// backend is no longer sycl-tla (spec 2.1 §3.5's flip, recorded below).
namespace runtime::prefill {
namespace {
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::linear_sycl: " + what);
}
}  // namespace

// Spec 2.1's closing flip (2026-09-18): the L0 backend is the default in every build;
// sycl-tla stays selectable as the reference (--pp-backend sycl-tla).
PrefillBackend default_prefill_backend() { return PrefillBackend::L0; }
bool sycl_available() { return true; }

// gemm.h's public entry points, now wrappers over the .so's SyclSide-taking one.
void gemm_bf16_batched(Context& cx, GemmBatch b, const uint16_t* A, const uint16_t* B, float* C,
                       bool transB) {
  gemm_bf16_batched_on(&cx.sycl(), b, A, B, C, transB);
}
void gemm_bf16(Context& cx, GemmDims d, const uint16_t* A, const uint16_t* B, float* C) {
  gemm_bf16_batched(cx, GemmBatch{d.M, d.K, d.N, 1, d.K, d.N, d.N, 0, 0, 0}, A, B, C, false);
}

// step.cc's pf_linear before spec 2.1, moved here unchanged (rulings A23/A24).
void linear_sycl(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::DeviceWeight& w,
                 const uint16_t* x, uint32_t M) {
  const model::GemvShape& sh = w.shape;
  require(w.kind == model::WeightKind::Int4,
          "the prefill walk's linears are all int4; this one is not");
  l0::Mem& dq = s.dequant_buffer();
  require(dq.size() >= size_t(sh.K) * sh.N * 2,
          "the bf16 dequant scratch is smaller than K x N for this linear");
  require(s.partials.size() >= size_t(M) * sh.N * 4,
          "`partials` is smaller than the [M][N] fp32 this linear writes");
  dequant_to_bf16(cx, kc, w, dq.as<uint16_t>());
  timed_wait(cx, Phase::kDequant);        // L0 -> SYCL
  gemm_bf16(cx, GemmDims{M, sh.K, sh.N}, x, dq.as<uint16_t>(), s.partials.as<float>());
  timed_wait(cx, Phase::kGemm);           // SYCL -> L0
}

void attn_qk_sycl(Context& cx, const GemmBatch& b, const uint16_t* q, const uint16_t* k, float* S) {
  gemm_bf16_batched(cx, b, q, k, S, /*transB=*/true);
  timed_wait(cx, Phase::kAttnQk);         // SYCL -> L0: one compute queue (A24)
}
void attn_pv_sycl(Context& cx, const GemmBatch& b, const uint16_t* P, const uint16_t* v, float* O) {
  gemm_bf16_batched(cx, b, P, v, O, /*transB=*/false);
}

}  // namespace runtime::prefill
