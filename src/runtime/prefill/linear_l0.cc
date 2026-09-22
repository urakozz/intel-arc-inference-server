#include "runtime/prefill/linear_l0.h"

#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

namespace runtime::prefill {
namespace {
constexpr uint32_t kNs = kernels::kPfSlabWidth;
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::linear_l0: " + what);
}
}  // namespace

size_t linear_l0_launches(const model::GemvShape& sh) { return 2 * (size_t(sh.N) / kNs); }

void linear_l0(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::DeviceWeight& w,
               const uint16_t* x, uint32_t M) {
  const model::GemvShape& sh = w.shape;
  require(w.kind == model::WeightKind::Int4, "the prefill walk's linears are all int4; this one is not");
  require(sh.N % kNs == 0, "N = " + std::to_string(sh.N) + " is not a whole number of 1024-column slabs");
  require(sh.K % 64 == 0, "K = " + std::to_string(sh.K) + " is not a multiple of the group size 64");
  require(sh.layout == 0 || sh.layout == 1, "layout " + std::to_string(sh.layout) + " is neither 0 nor 1");
  require(sh.layout != 0 || w.scales != nullptr, "layout 0 weight has no independent scales allocation");
  require(M > 0 && pad256(M) <= PrefillScratch::kC,
          "M = " + std::to_string(M) + " padded to 256 exceeds the kC-row scratch");
  l0::Mem& slab = s.slab_buffer();
  require(slab.size() >= size_t(sh.K) * kNs * 2, "the slab buffer is smaller than [K][1024] bf16");
  require(s.partials.size() >= size_t(pad256(M)) * sh.N * 4,
          "`partials` is smaller than the [pad256(M)][N] fp32 this linear writes");

  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(sh.K, sh.N, sh.layout), "pf_dequant_slab");
  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;
  float* out = s.partials.as<float>();
  for (uint32_t n0 = 0; n0 < sh.N; n0 += kNs) {
    cx.launch(dq, kNs / 16, sh.K / 64, 1,
              {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
    profile_wait(cx, Phase::kSlabDequant);   // only when B70_PREFILL_PROFILE=1
    // B = the slab [K][1024] (ldb 1024); C = partials from column n0 (ldc N). gemm_l0 pads M.
    const GemmBatch b{M, sh.K, kNs, 1, sh.K, kNs, sh.N, 0, 0, 0};
    gemm_l0(cx, kc, b, x, slab.as<uint16_t>(), out + n0, /*transB=*/false);
    profile_wait(cx, Phase::kSlabGemm);      // splits the old `linear_l0` row in two
  }
}

}  // namespace runtime::prefill
