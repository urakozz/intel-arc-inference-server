// Phase 2 Task 2 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): pf_gdn_front.cl - the GDN
// front's conv1d + SiLU and q / k l2norm in one launch - against the two-launch walk it replaces in Ornith's
// gdn_chunk (pf_gdn_seed, pf_gdn_conv, pf_gdn_l2norm vs pf_gdn_seed, pf_gdn_front), every byte of xb and of the
// conv ring, at Ornith's heads (16 k / 32 v), C = 2048 / 37 / 1 at pos 0 and a pos past the ring's wrap.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "pf_harness.h"

namespace {
using pf_harness::Dev;
using pf_harness::download;
using pf_harness::random_bf16;
using pf_harness::random_f32;
using pf_harness::upload;

constexpr uint32_t kHk = 16, kHv = 32, kConv = (2 * kHk + kHv) * 128, kQkvz = kConv + kHv * 128, kRing = 16;

struct Out {
  std::vector<uint16_t> xb, ring;
};

Out run(Dev& d, bool front, uint32_t pos, uint32_t C, const std::vector<float>& qkvz, const std::vector<float>& w,
        const std::vector<uint16_t>& ring0) {
  l0::Mem qb = upload(d.ctx, d.imm, qkvz), wb = upload(d.ctx, d.imm, w), rb = upload(d.ctx, d.imm, ring0);
  l0::Mem sb(d.ctx, l0::MemKind::Device, size_t(3) * kConv * 2), xb(d.ctx, l0::MemKind::Device, size_t(C) * kConv * 2);
  l0::Module cm(d.ctx, kernels::path(kernels::pf_gdn_conv_variant(kHk, kHv)));
  l0::Module fm(d.ctx, kernels::path(kernels::pf_gdn_front_variant(kHk, kHv)));
  l0::Kernel seed = cm.kernel("pf_gdn_seed"), conv = cm.kernel("pf_gdn_conv"), l2 = cm.kernel("pf_gdn_l2norm"),
             fr = fm.kernel("pf_gdn_front");
  seed.group_size(256);
  seed.arg_ptr(0, rb.ptr());
  seed.arg_ptr(1, sb.ptr());
  seed.arg<uint32_t>(2, pos);
  d.run(seed, kConv / 256, 1);
  l0::Kernel& k = front ? fr : conv;
  k.group_size(256);
  k.arg_ptr(0, qb.ptr());
  k.arg_ptr(1, sb.ptr());
  k.arg_ptr(2, wb.ptr());
  k.arg_ptr(3, xb.ptr());
  k.arg_ptr(4, rb.ptr());
  k.arg<uint32_t>(5, pos);
  k.arg<uint32_t>(6, C);
  d.run(k, kConv / 256, front ? (C + 31) / 32 : (C + 127) / 128);
  if (!front) {
    l2.group_size(128);
    l2.arg_ptr(0, xb.ptr());
    l2.arg<uint32_t>(1, C);
    d.run(l2, 2 * kHk, C);
  }
  Out o;
  o.xb.resize(size_t(C) * kConv);
  o.ring.resize(ring0.size());
  download(d.imm, o.xb, xb);
  download(d.imm, o.ring, rb);
  return o;
}
}  // namespace

int main() {
  Dev d;
  const std::vector<float> w = random_f32(size_t(kConv) * 4, 3, 0.0f, 0.5f);
  const std::vector<uint16_t> ring0 = random_bf16(size_t(kRing) * kConv, 7, -2.0f, 2.0f);
  for (uint32_t pos : {0u, 4093u})
    for (uint32_t C : {2048u, 37u, 1u}) {
      const std::vector<float> qkvz = random_f32(size_t(C) * kQkvz, 100 + C + pos, 0.0f, 1.0f);
      const Out a = run(d, false, pos, C, qkvz, w, ring0), b = run(d, true, pos, C, qkvz, w, ring0);
      size_t dx = 0, dr = 0;
      for (size_t i = 0; i < a.xb.size(); ++i) dx += a.xb[i] != b.xb[i];
      for (size_t i = 0; i < a.ring.size(); ++i) dr += a.ring[i] != b.ring[i];
      std::printf("pf_gdn_front GK16V32 pos %u C %u: xb %zu of %zu, ring %zu differing words (front vs conv + l2norm)\n",
                  pos, C, dx, a.xb.size(), dr);
      CHECK_EQ(dx, size_t{0});
      CHECK_EQ(dr, size_t{0});
    }
  std::puts("gdn_front_test OK");
  return 0;
}
