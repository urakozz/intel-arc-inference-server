#include "runtime/prefill/dequant.h"

#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"

namespace runtime::prefill {
namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::dequant_to_bf16: " + what);
}

}  // namespace

void dequant_to_bf16(Context& cx, KernelCache& kc, const loader::DeviceWeight& w,
                     uint16_t* scratch) {
  const model::GemvShape& sh = w.shape;
  require(w.kind == model::WeightKind::Int4,
          "this weight is not int4 -- a bf16 linear feeds gemm_bf16 directly");
  // The kernel's own geometry: SG = 16 lanes across N, GROUP = 64 along K.
  require(sh.N % 16 == 0, "N = " + std::to_string(sh.N) + " is not a multiple of the 16-lane"
                          " subgroup width");
  require(sh.K % 64 == 0,
          "K = " + std::to_string(sh.K) + " is not a multiple of the group size 64");
  require(sh.layout == 0 || sh.layout == 1,
          "layout " + std::to_string(sh.layout) + " is neither 0 nor 1");
  require(sh.layout != 0 || w.scales != nullptr,
          "layout 0 weight has no independent scales allocation");
  require(scratch != nullptr, "the bf16 scratch is null");

  // Layout 1 carries its scales inline in the tile, and the kernel never
  // dereferences argument 1 in that build; the harness passes null there too
  // (tests/prefill/dequant_harness.h), so this matches the tested binding.
  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;
  l0::Kernel& k = kc(kernels::pf_dequant_variant(sh.K, sh.N, sh.layout), "pf_dequant_tile");
  cx.launch(k, sh.N / 16, sh.K / 64, 1,
            {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(scratch)});
}

}  // namespace runtime::prefill
