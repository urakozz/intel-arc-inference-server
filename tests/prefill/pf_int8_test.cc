// pf_int8 kernels vs CPU mirrors of their arithmetic, BIT-EXACT (spec 5 bar A1).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "common/repack.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "pf_harness.h"
#include "runtime/prefill/int8_signs.h"

namespace {
int8_t sat_rte(float v) {
  const float r = std::nearbyint(v);
  return int8_t(std::fmax(-128.0f, std::fmin(127.0f, r)));
}

// Canonical Sylvester FWHT per 1024-block, stages ascending, unnormalised.
void fwht_blocks(std::vector<float>& r) {
  for (size_t b = 0; b < r.size(); b += 1024)
    for (uint32_t h = 1; h < 1024; h <<= 1)
      for (uint32_t i = 0; i < 1024; i += 2 * h)
        for (uint32_t j = i; j < i + h; ++j) {
          const float a = r[b + j], c = r[b + j + h];
          r[b + j] = a + c;
          r[b + j + h] = a - c;
        }
}

void quant_case(pf_harness::Dev& d, uint32_t K, uint32_t rows) {
  const std::vector<uint16_t> x = pf_harness::random_bf16(size_t(rows) * K, 11 + K, -4.0f, 4.0f);
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  l0::Mem dx = pf_harness::upload(d.ctx, d.imm, x);
  l0::Mem ds = pf_harness::upload(d.ctx, d.imm, sgn);
  l0::Mem dq(d.ctx, l0::MemKind::Device, size_t(rows) * K);
  l0::Mem dsc(d.ctx, l0::MemKind::Device, size_t(rows) * 4);
  l0::Module mod(d.ctx, kernels::path(kernels::pf_quant_had_variant(K)));
  l0::Kernel k = mod.kernel("pf_quant_had");
  k.group_size(16 * (K / 1024));
  k.arg_ptr(0, dx.ptr());
  k.arg_ptr(1, ds.ptr());
  k.arg_ptr(2, dq.ptr());
  k.arg_ptr(3, dsc.ptr());
  k.arg(4, K);   // ldx
  d.run(k, rows, 1);
  std::vector<int8_t> q(size_t(rows) * K);
  std::vector<float> sc(rows);
  pf_harness::download(d.imm, q, dq);
  pf_harness::download(d.imm, sc, dsc);
  size_t bad = 0;
  std::vector<float> r(K);
  for (uint32_t m = 0; m < rows; ++m) {
    for (uint32_t kk = 0; kk < K; ++kk) r[kk] = common::bf16_to_f32(x[size_t(m) * K + kk]) * sgn[kk];
    fwht_blocks(r);
    float amax = 0.0f;
    for (float& v : r) { v *= 1.0f / 32.0f; amax = std::fmax(amax, std::fabs(v)); }
    const float scale = amax / 127.0f;
    const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
    bad += (scale != sc[m]);
    for (uint32_t kk = 0; kk < K; ++kk) bad += (sat_rte(r[kk] * iv) != q[size_t(m) * K + kk]);
  }
  std::printf("pf_quant_had K=%u rows=%u: %zu mismatches\n", K, rows, bad);
  CHECK(bad == 0);
}
}  // namespace

int main() {
  pf_harness::Dev d;
  for (uint32_t K : {5120u, 6144u, 17408u}) quant_case(d, K, 64);
  std::printf("pf_int8_test: PASS\n");
  return 0;
}
