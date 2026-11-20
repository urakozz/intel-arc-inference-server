// Phase 2 Task 5: pf_norm_fused.cl's
// pf_res_norm - the residual fold and the RMSNorm in one launch - against the two-launch walk it replaces
// (pf_res_fold + pf_norm_finish), every byte of the residual and of x, at the hidden size given on the command line
// (Ornith 2048, Qwen3.8 5120), S_PREV 0 and 1, M = 2048 (a chunk) and M = 37 (a ragged tail), on random rows of a
// residual stream's magnitudes.
//
// PERF-85, with `mtp` after the hidden size (the MTP head's prompt-pass forms, built with the head): the ZERO_RESID
// form (_Z) against pf_res_fold _Z + pf_norm_finish, and pf_mtp_front against the five launches it replaces -
// pf_embed_gather, then pf_res_fold SP0 + pf_norm_finish at X_STRIDE 2K over the embedding rows and over the hidden
// rows (into x's two halves) - every byte of x, and of the residual for _Z.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr uint32_t kG = 20;
uint32_t kK = 2048;   // the hidden size, from argv

struct Out {
  std::vector<uint16_t> resid, x;
};

Out two_launch(Dev& d, uint32_t M, uint32_t SP, const std::vector<float>& partials,
               const std::vector<uint16_t>& resid, const std::vector<float>& w, bool zero = false) {
  l0::Mem pb = upload(d.ctx, d.imm, partials), rb = upload(d.ctx, d.imm, resid), wb = upload(d.ctx, d.imm, w);
  l0::Mem sb(d.ctx, l0::MemKind::Device, size_t(kG) * M * 4), xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 2);
  l0::Module fm(d.ctx, kernels::path(zero ? kernels::pf_res_fold_zero_variant(kK, kG)
                                           : kernels::pf_res_fold_variant(kK, SP, kG)));
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
          const std::vector<float>& w, bool zero = false) {
  l0::Mem pb = upload(d.ctx, d.imm, partials), rb = upload(d.ctx, d.imm, resid), wb = upload(d.ctx, d.imm, w);
  l0::Mem xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 2);
  l0::Module m(d.ctx, kernels::path(zero ? kernels::pf_res_norm_zero_variant(kK, kG)
                                          : kernels::pf_res_norm_variant(kK, SP, kG)));
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

void check_case(Dev& d, uint32_t M, uint32_t SP, bool zero = false) {
  const std::vector<float> partials = random_f32(size_t(M) * kK, 11 + M + SP, 0.0f, 0.5f);
  const std::vector<uint16_t> resid = random_bf16(size_t(M) * kK, 23 + M + SP, -4.0f, 4.0f);
  const std::vector<float> w = random_f32(kK, 5, 1.0f, 0.2f);
  const Out a = two_launch(d, M, SP, partials, resid, w, zero), b = fused(d, M, SP, partials, resid, w, zero);
  size_t dr = 0, dx = 0;
  for (size_t i = 0; i < a.resid.size(); ++i) {
    dr += a.resid[i] != b.resid[i];
    dx += a.x[i] != b.x[i];
  }
  std::printf("pf_res_norm K%u SP%u%s M=%u: resid %zu, x %zu differing words of %zu (fused vs fold + finish)\n", kK,
              SP, zero ? " _Z" : "", M, dr, dx, a.x.size());
  CHECK_EQ(dr, size_t{0});
  CHECK_EQ(dx, size_t{0});
}
// PERF-85: the head's front, x [M][2K] - the five launches against pf_mtp_front.
std::vector<uint16_t> front_five(Dev& d, uint32_t M, const l0::Mem& ids, const l0::Mem& embed, const l0::Mem& hid,
                                 const l0::Mem& we, const l0::Mem& wh) {
  l0::Mem rb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 2), sb(d.ctx, l0::MemKind::Device, size_t(kG) * M * 4);
  l0::Mem xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 4);
  l0::Module gm(d.ctx, kernels::path(kernels::pf_embed_gather_variant(kK)));
  l0::Module fm(d.ctx, kernels::path(kernels::pf_res_fold_variant(kK, 0, kG)));
  l0::Module nm(d.ctx, kernels::path(kernels::pf_norm_finish_strided_variant(kK, kG, kG, 2 * kK)));
  l0::Kernel gather = gm.kernel("pf_embed_gather");
  gather.group_size(256);
  gather.arg_ptr(0, ids.ptr());
  gather.arg_ptr(1, embed.ptr());
  gather.arg_ptr(2, rb.ptr());
  gather.arg<uint32_t>(3, M);
  d.run(gather, 1, M);
  auto norm = [&](const void* rows, const void* w, void* out) {
    l0::Kernel fold = fm.kernel("pf_res_fold"), fin = nm.kernel("pf_norm_finish");
    fold.group_size(256);
    fin.group_size(256);
    fold.arg_ptr(0, nullptr);
    fold.arg_ptr(1, rows);
    fold.arg_ptr(2, sb.ptr());
    fold.arg<uint32_t>(3, M);
    fin.arg_ptr(0, sb.ptr());
    fin.arg_ptr(1, rows);
    fin.arg_ptr(2, w);
    fin.arg_ptr(3, out);
    fin.arg<uint32_t>(4, M);
    d.run2(fold, kG, M, fin, kG, M);
  };
  norm(rb.ptr(), we.ptr(), xb.ptr());
  norm(hid.ptr(), wh.ptr(), static_cast<uint16_t*>(xb.ptr()) + kK);
  std::vector<uint16_t> x(size_t(M) * kK * 2);
  download(d.imm, x, xb);
  return x;
}

std::vector<uint16_t> front_one(Dev& d, uint32_t M, const l0::Mem& ids, const l0::Mem& embed, const l0::Mem& hid,
                                const l0::Mem& we, const l0::Mem& wh) {
  l0::Mem xb(d.ctx, l0::MemKind::Device, size_t(M) * kK * 4);
  l0::Module m(d.ctx, kernels::path(kernels::pf_mtp_front_variant(kK, kG)));
  l0::Kernel k = m.kernel("pf_mtp_front");
  k.group_size(256);
  k.arg_ptr(0, ids.ptr());
  k.arg_ptr(1, embed.ptr());
  k.arg_ptr(2, hid.ptr());
  k.arg_ptr(3, we.ptr());
  k.arg_ptr(4, wh.ptr());
  k.arg_ptr(5, xb.ptr());
  k.arg<uint32_t>(6, M);
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(k, 1, M, 2);
  list.close();
  d.q.execute(list, &d.fence);
  d.fence.wait();
  std::vector<uint16_t> x(size_t(M) * kK * 2);
  download(d.imm, x, xb);
  return x;
}

void check_front(Dev& d, uint32_t M) {
  constexpr uint32_t kRows = 97;   // a small embedding table: the ids index it, the gather's VOCAB only bounds them
  std::vector<uint32_t> ids(M);
  for (uint32_t i = 0; i < M; ++i) ids[i] = (i * 7919u + 13u) % kRows;
  l0::Mem ib = upload(d.ctx, d.imm, ids);
  l0::Mem eb = upload(d.ctx, d.imm, random_bf16(size_t(kRows) * kK, 31 + M, -0.1f, 0.1f));
  l0::Mem hb = upload(d.ctx, d.imm, random_bf16(size_t(M) * kK, 37 + M, -4.0f, 4.0f));
  l0::Mem web = upload(d.ctx, d.imm, random_f32(kK, 41, 1.0f, 0.2f));
  l0::Mem whb = upload(d.ctx, d.imm, random_f32(kK, 43, 1.0f, 0.2f));
  const std::vector<uint16_t> a = front_five(d, M, ib, eb, hb, web, whb), b = front_one(d, M, ib, eb, hb, web, whb);
  size_t dx = 0;
  for (size_t i = 0; i < a.size(); ++i) dx += a[i] != b[i];
  std::printf("pf_mtp_front K%u M=%u: x %zu differing words of %zu (one launch vs gather + 2 x (fold + finish))\n", kK,
              M, dx, a.size());
  CHECK_EQ(dx, size_t{0});
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) kK = uint32_t(std::atoi(argv[1]));
  const bool mtp = argc > 2 && std::strcmp(argv[2], "mtp") == 0;
  Dev d;
  for (uint32_t SP : {0u, 1u})
    for (uint32_t M : {2048u, 37u}) check_case(d, M, SP);
  if (mtp)
    for (uint32_t M : {4096u, 37u}) {
      check_case(d, M, 1, /*zero=*/true);
      check_front(d, M);
    }
  std::puts("pf_norm_fused_test OK");
  return 0;
}
