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

// W R_K per column on the CPU: dequant (q - 8) * s, signs, FWHT per 1024-block, * 1/32.
std::vector<float> rotated_column(const common::Int4Gptq& w, uint32_t n, const std::vector<float>& sgn) {
  std::vector<float> col(w.K);
  for (uint32_t k = 0; k < w.K; ++k) {
    const uint32_t word = w.qweight[size_t(k / 8) * w.N + n];
    const float s = common::f16_to_f32(w.scales[size_t(k / 64) * w.N + n]);
    col[k] = (float(int((word >> (4 * (k % 8))) & 0xFu) - 8) * s) * sgn[k];
  }
  fwht_blocks(col);
  for (float& v : col) v *= 1.0f / 32.0f;
  return col;
}

void requant_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t layout) {
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 23 + K + N);
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  std::vector<float> inv(N);
  std::vector<std::vector<float>> cols(N);
  for (uint32_t n = 0; n < N; ++n) {
    cols[n] = rotated_column(w, n, sgn);
    float amax = 0.0f;
    for (float v : cols[n]) amax = std::fmax(amax, std::fabs(v));
    inv[n] = 1.0f / (amax > 0.0f ? amax / 127.0f : 1.0f);
  }
  const std::vector<uint32_t> words = layout == 0 ? w.qweight : w.tiled();
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, words);
  l0::Mem dsc = pf_harness::upload(d.ctx, d.imm, w.scales);
  l0::Mem dbits = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_sign_bits(K));
  l0::Mem dinv = pf_harness::upload(d.ctx, d.imm, inv);
  l0::Mem dout(d.ctx, l0::MemKind::Device, size_t(K) * N);
  l0::Module mod(d.ctx, kernels::path(kernels::pf_requant_rot_variant(layout)));
  l0::Kernel k = mod.kernel("pf_requant_rot");
  k.group_size(256);
  k.arg_ptr(0, dw.ptr());
  k.arg_ptr(1, layout == 0 ? dsc.ptr() : nullptr);
  k.arg_ptr(2, dbits.ptr());
  k.arg_ptr(3, dinv.ptr());
  k.arg_ptr(4, dout.ptr());
  k.arg(5, 0u);   // n0
  k.arg(6, N);    // Nfull
  k.arg(7, K);
  k.arg(8, N);    // ldo
  d.run(k, N / 16, K / 1024);
  std::vector<uint32_t> got(size_t(K / 4) * N);
  pf_harness::download(d.imm, got, dout);
  size_t bad = 0;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t kk = 0; kk < K; ++kk) {
      const int8_t want = sat_rte(cols[n][kk] * inv[n]);
      const int8_t have = int8_t((got[size_t(kk / 4) * N + n] >> (8 * (kk % 4))) & 0xFFu);
      bad += (want != have);
    }
  std::printf("pf_requant_rot K=%u N=%u L%u: %zu mismatches\n", K, N, layout, bad);
  CHECK(bad == 0);
}

void colmax_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t layout) {
  common::Int4Gptq w = common::Int4Gptq::random(K, N, 31 + K + N);
  for (uint32_t r = 0; r < K / 8; ++r) {
    w.qweight[size_t(r) * N + 5] = 0x88888888u;   // q - 8 == 0 everywhere: an all-zero column
    w.qweight[size_t(r) * N + 7] = 0x00000000u;   // q - 8 == -8 everywhere: negative-heavy
  }
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  const std::vector<uint32_t> words = layout == 0 ? w.qweight : w.tiled();
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, words);
  l0::Mem dsc = pf_harness::upload(d.ctx, d.imm, w.scales);
  l0::Mem dbits = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_sign_bits(K));
  l0::Mem dmax = pf_harness::upload(d.ctx, d.imm, std::vector<uint32_t>(N, 0u));
  l0::Module mod(d.ctx, kernels::path(kernels::pf_requant_rot_variant(layout)));
  l0::Kernel k = mod.kernel("pf_colmax_rot");
  k.group_size(256);
  k.arg_ptr(0, dw.ptr());
  k.arg_ptr(1, layout == 0 ? dsc.ptr() : nullptr);
  k.arg_ptr(2, dbits.ptr());
  k.arg_ptr(3, dmax.ptr());
  k.arg(4, 0u);
  k.arg(5, N);
  k.arg(6, K);
  d.run(k, N / 16, K / 1024);
  std::vector<uint32_t> got(N);
  pf_harness::download(d.imm, got, dmax);
  size_t bad = 0;
  for (uint32_t n = 0; n < N; ++n) {
    const std::vector<float> col = rotated_column(w, n, sgn);
    float amax = 0.0f;
    for (float v : col) amax = std::fmax(amax, std::fabs(v));
    uint32_t want;
    std::memcpy(&want, &amax, 4);
    bad += (want != got[n]);
  }
  CHECK(got[5] == 0u);
  std::printf("pf_colmax_rot K=%u N=%u L%u: %zu mismatches (zero column max %u)\n", K, N, layout, bad,
              got[5]);
  CHECK(bad == 0);
}
}  // namespace

int main() {
  pf_harness::Dev d;
  for (uint32_t K : {5120u, 6144u, 17408u}) quant_case(d, K, 64);
  requant_case(d, 5120, 1024, 0);
  requant_case(d, 5120, 1024, 1);
  requant_case(d, 6144, 1024, 0);
  requant_case(d, 17408, 1024, 0);
  colmax_case(d, 5120, 1024, 0);
  colmax_case(d, 5120, 1024, 1);
  colmax_case(d, 17408, 1024, 0);
  std::printf("pf_int8_test: PASS\n");
  return 0;
}
