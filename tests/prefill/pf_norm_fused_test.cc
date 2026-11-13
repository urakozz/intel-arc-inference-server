// Phase 2 Task 5 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): pf_norm_fused.cl's
// pf_res_norm - the residual fold and the RMSNorm in one launch - against the two-launch walk it replaces
// (pf_res_fold + pf_norm_finish), every byte of the residual and of x, at Ornith's hidden 2048, S_PREV 0
// and 1, M = 2048 (a chunk) and M = 37 (a ragged tail), on random rows of a residual stream's magnitudes.
#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr uint32_t kK = 2048, kG = 20;

struct Out {
  std::vector<uint16_t> resid, x;
};

Out two_launch(Dev& d, uint32_t M, uint32_t SP, const std::vector<float>& partials,
               const std::vector<uint16_t>& resid, const std::vector<float>& w) {
  l0::Mem pb = upload(d.ctx, d.imm, partials), rb = upload(d.ctx, d.imm, resid), wb = upload(d.ctx, d.imm, w);
  l0::Mem sb(d.ctx, l0::MemKind::Device, size_t(kG) * M * 4), xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 2);
  l0::Module fm(d.ctx, kernels::path(kernels::pf_res_fold_variant(kK, SP, kG)));
  l0::Module nm(d.ctx, kernels::path(kernels::pf_norm_finish_variant(kK, kG, kG)));
  l0::Kernel fold = fm.kernel("pf_res_fold"), fin = nm.kernel("pf_norm_finish");
  fold.group_size(256);
  fin.group_size(256);
  fold.arg_ptr(0, pb.ptr());
  fold.arg_ptr(1, rb.ptr());
  fold.arg_ptr(2, sb.ptr());
  fold.arg<uint32_t>(3, M);
  fin.arg_ptr(0, sb.ptr());
  fin.arg_ptr(1, rb.ptr());
  fin.arg_ptr(2, wb.ptr());
  fin.arg_ptr(3, xb.ptr());
  fin.arg<uint32_t>(4, M);
  d.run2(fold, kG, M, fin, kG, M);
  Out o;
  o.resid.resize(size_t(M) * kK);
  o.x.resize(o.resid.size());
  download(d.imm, o.resid, rb);
  download(d.imm, o.x, xb);
  return o;
}

Out fused(Dev& d, uint32_t M, uint32_t SP, const std::vector<float>& partials, const std::vector<uint16_t>& resid,
          const std::vector<float>& w) {
  l0::Mem pb = upload(d.ctx, d.imm, partials), rb = upload(d.ctx, d.imm, resid), wb = upload(d.ctx, d.imm, w);
  l0::Mem xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 2);
  l0::Module m(d.ctx, kernels::path(kernels::pf_res_norm_variant(kK, SP, kG)));
  l0::Kernel k = m.kernel("pf_res_norm");
  k.group_size(256);
  k.arg_ptr(0, pb.ptr());
  k.arg_ptr(1, rb.ptr());
  k.arg_ptr(2, wb.ptr());
  k.arg_ptr(3, xb.ptr());
  k.arg<uint32_t>(4, M);
  d.run(k, 1, M);
  Out o;
  o.resid.resize(size_t(M) * kK);
  o.x.resize(o.resid.size());
  download(d.imm, o.resid, rb);
  download(d.imm, o.x, xb);
  return o;
}

void check_case(Dev& d, uint32_t M, uint32_t SP) {
  const std::vector<float> partials = random_f32(size_t(M) * kK, 11 + M + SP, 0.0f, 0.5f);
  const std::vector<uint16_t> resid = random_bf16(size_t(M) * kK, 23 + M + SP, -4.0f, 4.0f);
  const std::vector<float> w = random_f32(kK, 5, 1.0f, 0.2f);
  const Out a = two_launch(d, M, SP, partials, resid, w), b = fused(d, M, SP, partials, resid, w);
  size_t dr = 0, dx = 0;
  for (size_t i = 0; i < a.resid.size(); ++i) {
    dr += a.resid[i] != b.resid[i];
    dx += a.x[i] != b.x[i];
  }
  std::printf("pf_res_norm K%u SP%u M=%u: resid %zu, x %zu differing words of %zu (fused vs fold + finish)\n", kK, SP,
              M, dr, dx, a.x.size());
  CHECK_EQ(dr, size_t{0});
  CHECK_EQ(dx, size_t{0});
}
}  // namespace

int main() {
  Dev d;
  for (uint32_t SP : {0u, 1u})
    for (uint32_t M : {2048u, 37u}) check_case(d, M, SP);
  std::puts("pf_norm_fused_test OK");
  return 0;
}
