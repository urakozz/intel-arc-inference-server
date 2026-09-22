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

// Everything both walks below require of the weight and of `M`. The slab
// buffer is checked here too; only the OUTPUT rectangle differs between them.
l0::Kernel& check_and_dequant_kernel(KernelCache& kc, PrefillScratch& s,
                                     const loader::DeviceWeight& w, uint32_t M) {
  const model::GemvShape& sh = w.shape;
  require(w.kind == model::WeightKind::Int4, "the prefill walk's linears are all int4; this one is not");
  require(sh.N % kNs == 0, "N = " + std::to_string(sh.N) + " is not a whole number of 1024-column slabs");
  require(sh.K % 64 == 0, "K = " + std::to_string(sh.K) + " is not a multiple of the group size 64");
  require(sh.layout == 0 || sh.layout == 1, "layout " + std::to_string(sh.layout) + " is neither 0 nor 1");
  require(sh.layout != 0 || w.scales != nullptr, "layout 0 weight has no independent scales allocation");
  require(M > 0 && pad256(M) <= PrefillScratch::kC,
          "M = " + std::to_string(M) + " padded to 256 exceeds the kC-row scratch");
  require(s.slab_buffer().size() >= size_t(sh.K) * kNs * 2,
          "the slab buffer is smaller than [K][1024] bf16");
  return kc(kernels::pf_dequant_slab_variant(sh.K, sh.N, sh.layout), "pf_dequant_slab");
}
}  // namespace

size_t linear_l0_launches(const model::GemvShape& sh) { return 2 * (size_t(sh.N) / kNs); }

void linear_l0(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::DeviceWeight& w,
               const uint16_t* x, uint32_t M) {
  const model::GemvShape& sh = w.shape;
  l0::Kernel& dq = check_and_dequant_kernel(kc, s, w, M);
  require(s.partials.size() >= size_t(pad256(M)) * sh.N * 4,
          "`partials` is smaller than the [pad256(M)][N] fp32 this linear writes");

  l0::Mem& slab = s.slab_buffer();
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

void linear_l0_silu(Context& cx, KernelCache& kc, PrefillScratch& s,
                    const loader::DeviceWeight& w, const uint16_t* x, uint32_t M, uint16_t* out,
                    uint32_t ldx) {
  const model::GemvShape& sh = w.shape;
  l0::Kernel& dq = check_and_dequant_kernel(kc, s, w, M);
  // The interleave is `gflat = (k/16).32 + k%16`: the linear's N columns are
  // N/2 gate||up PAIRS, so the x row this writes is exactly N/2 wide and the
  // pitch must be it. An `ldx` that is not N/2 would mean a caller whose x is
  // not this linear's output, which is a different kernel's contract.
  require(size_t(ldx) * 2 == size_t(sh.N),
          "ldx = " + std::to_string(ldx) + " is not N/2 for N = " + std::to_string(sh.N));
  // Every slab base is a multiple of 1024 and hence of 32, which is what makes
  // the kernel's "atom b is gate iff b is even" true of the global column too.
  static_assert(kNs % 32 == 0, "the slab width must keep the gate||up 32-column phase");

  l0::Mem& slab = s.slab_buffer();
  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;
  for (uint32_t n0 = 0; n0 < sh.N; n0 += kNs) {
    cx.launch(dq, kNs / 16, sh.K / 64, 1,
              {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
    profile_wait(cx, Phase::kSlabDequant);
    // No fp32 C: the slab's 1024 interleaved columns become x columns
    // [n0/2, n0/2 + 512) of every row. `b.ldc` is unused and left at 0.
    const GemmBatch b{M, sh.K, kNs, 1, sh.K, kNs, 0, 0, 0, 0};
    gemm_l0_silu(cx, kc, b, x, slab.as<uint16_t>(), out + n0 / 2, ldx);
    profile_wait(cx, Phase::kSlabGemm);
  }
}

}  // namespace runtime::prefill
