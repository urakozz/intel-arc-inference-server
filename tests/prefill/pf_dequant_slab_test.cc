// pf_dequant_slab vs pf_dequant_tile, BITWISE, every slab of every production int4 shape at
// the layout the loader ships it in (spec 2.1 §2 bar 1).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "pf_harness.h"

namespace {
constexpr uint32_t kNS = kernels::kPfSlabWidth;

void shape_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t layout) {
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 7 + K + N);
  const std::vector<uint32_t> words = layout == 0 ? w.qweight : w.tiled();
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, words);
  l0::Mem ds = pf_harness::upload(d.ctx, d.imm, w.scales);   // unread by layout 1
  l0::Mem full(d.ctx, l0::MemKind::Device, size_t(K) * N * 2);
  l0::Mem slab(d.ctx, l0::MemKind::Device, size_t(K) * kNS * 2);
  l0::Module mfull(d.ctx, kernels::path(kernels::pf_dequant_variant(K, N, layout)));
  l0::Kernel kfull = mfull.kernel("pf_dequant_tile");
  kfull.group_size(16);
  kfull.arg_ptr(0, dw.ptr());
  kfull.arg_ptr(1, layout == 0 ? ds.ptr() : nullptr);
  kfull.arg_ptr(2, full.ptr());
  d.run(kfull, N / 16, K / 64);
  std::vector<uint16_t> ref(size_t(K) * N);
  pf_harness::download(d.imm, ref, full);

  l0::Module mslab(d.ctx, kernels::path(kernels::pf_dequant_slab_variant(K, N, layout)));
  l0::Kernel kslab = mslab.kernel("pf_dequant_slab");
  kslab.group_size(16);
  kslab.arg_ptr(0, dw.ptr());
  kslab.arg_ptr(1, layout == 0 ? ds.ptr() : nullptr);
  kslab.arg_ptr(2, slab.ptr());
  std::vector<uint16_t> got(size_t(K) * kNS);
  for (uint32_t n0 = 0; n0 < N; n0 += kNS) {
    kslab.arg(3, n0);
    d.run(kslab, kNS / 16, K / 64);
    pf_harness::download(d.imm, got, slab);
    for (uint32_t k = 0; k < K; ++k)
      if (std::memcmp(&got[size_t(k) * kNS], &ref[size_t(k) * N + n0], kNS * 2) != 0) {
        std::fprintf(stderr, "K=%u N=%u L%u slab n0=%u: row k=%u differs\n", K, N, layout, n0, k);
        CHECK(false);
      }
  }
  std::printf("pf_dequant_slab K=%u N=%u L%u: %u slabs bit-exact vs pf_dequant_tile\n", K, N,
              layout, N / kNS);
}
}  // namespace

int main() {
  pf_harness::Dev d;
  shape_case(d, 5120, 16384, 1);
  shape_case(d, 6144, 5120, 0);
  shape_case(d, 5120, 34816, 0);
  shape_case(d, 17408, 5120, 0);
  shape_case(d, 5120, 14336, 0);
  shape_case(d, 5120, 38912, 0);   // spec 14: Agnes gate'||up' (38 slabs)
  shape_case(d, 19456, 5120, 0);   // spec 14: Agnes down' (K 19456)
  std::puts("pf_dequant_slab_test OK");
  return 0;
}
