// PERF-85: pf_attn_prep.cl's KV_ONLY build (grid (kv-heads, C) - the MTP head's prompt pass, which reads the cached
// K / V and never attn_q) against the full grid (q-heads + kv-heads, C) of the same source: every word of kv_k and
// kv_v, on random qkv partials at the heads given on the command line (Qwen3.8 24 4, Ornith 16 2, Flash-Next 24 2),
// C = 4096 and 37 rows at pos 0 and 5000.
//
// usage: pf_attn_prep_kvo_test <q-heads> <kv-heads>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "pf_harness.h"

namespace {
using pf_harness::Dev;
using pf_harness::download;
using pf_harness::random_f32;
using pf_harness::upload;

constexpr uint32_t kHD = 256, kCtrlWords = 32, kRotHalf = 32;
uint32_t kQ = 24, kKv = 4;

std::vector<uint16_t> run(Dev& d, bool kv_only, uint32_t pos, uint32_t C, const l0::Mem& ctrl, const l0::Mem& p,
                          const l0::Mem& small, const l0::Mem& rope) {
  const size_t n = size_t(pos + C) * kKv * kHD;
  l0::Mem q(d.ctx, l0::MemKind::Device, size_t(C) * kQ * kHD * 2);
  l0::Mem k = upload(d.ctx, d.imm, std::vector<uint16_t>(n, 0)), v = upload(d.ctx, d.imm, std::vector<uint16_t>(n, 0));
  const std::string name = kernels::pf_attn_prep_q16_variant(kQ, kKv);
  l0::Module m(d.ctx, kernels::path(kv_only ? kernels::kv_only(name) : name));
  l0::Kernel kn = m.kernel("pf_attn_prep");
  kn.group_size(256);
  kn.arg_ptr(0, ctrl.ptr());
  kn.arg_ptr(1, p.ptr());
  kn.arg_ptr(2, small.ptr());
  kn.arg_ptr(3, rope.ptr());
  kn.arg_ptr(4, q.ptr());
  kn.arg_ptr(5, nullptr);
  kn.arg_ptr(6, k.ptr());
  kn.arg_ptr(7, v.ptr());
  d.run(kn, kv_only ? kKv : kQ + kKv, C);
  std::vector<uint16_t> out(2 * n);
  std::vector<uint16_t> kk(n), vv(n);
  download(d.imm, kk, k);
  download(d.imm, vv, v);
  std::copy(kk.begin(), kk.end(), out.begin());
  std::copy(vv.begin(), vv.end(), out.begin() + long(n));
  return out;
}

void check_case(Dev& d, uint32_t pos, uint32_t C) {
  const uint32_t qkv_n = 2 * kQ * kHD + 2 * kKv * kHD;
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, kCtrlWords * sizeof(uint32_t));
  uint32_t* c = ctrl.as<uint32_t>();
  for (uint32_t i = 0; i < kCtrlWords; ++i) c[i] = 0;
  c[kernels::ctrl_index::kPos] = pos;
  c[kernels::ctrl_index::kNActive] = C;
  l0::Mem p = upload(d.ctx, d.imm, random_f32(size_t(C) * qkv_n, 7 + pos + C, 0.f, 1.f));
  l0::Mem small = upload(d.ctx, d.imm, random_f32(2 * kHD, 11, 1.0f, 0.03f));
  l0::Mem rope = upload(d.ctx, d.imm, random_f32(size_t(pos + C) * 2 * kRotHalf, 13, 0.f, 0.7f));
  const std::vector<uint16_t> a = run(d, false, pos, C, ctrl, p, small, rope), b = run(d, true, pos, C, ctrl, p, small, rope);
  size_t diff = 0;
  for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
  std::printf("pf_attn_prep_q16 Q%uKV%u pos %u C=%u: kv_k||kv_v %zu differing words of %zu (_KVO vs the full grid)\n",
              kQ, kKv, pos, C, diff, a.size());
  CHECK_EQ(diff, size_t{0});
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    kQ = uint32_t(std::atoi(argv[1]));
    kKv = uint32_t(std::atoi(argv[2]));
  }
  Dev d;
  for (uint32_t pos : {0u, 5000u})
    for (uint32_t C : {4096u, 37u}) check_case(d, pos, C);
  std::puts("pf_attn_prep_kvo_test OK");
  return 0;
}
