// pf_int8 kernels vs CPU mirrors of their arithmetic, BIT-EXACT (spec 5 bar A1).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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

uint16_t rne_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t(u >> 16);
}
float bf16f(uint16_t h) { return common::bf16_to_f32(h); }
float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }

// One pf_gemm_i8 launch: grid (M / 256, N / 128), WG 512. `C` is fp32 [M][ldc],
// or for the SiLU build bf16 x [M][ldc] (ldc = ldx).
void run_gemm(pf_harness::Dev& d, bool silu, const l0::Mem& xq, const l0::Mem& xs,
              const l0::Mem& w8, const l0::Mem& ws, const l0::Mem& C, uint32_t M, uint32_t K,
              uint32_t N, uint32_t ldc) {
  l0::Module mod(d.ctx, kernels::path(kernels::pf_gemm_i8_variant(silu)));
  l0::Kernel k = mod.kernel("pf_gemm_i8");
  k.group_size(512);
  k.arg_ptr(0, xq.ptr());
  k.arg_ptr(1, xs.ptr());
  k.arg_ptr(2, w8.ptr());
  k.arg_ptr(3, ws.ptr());
  k.arg_ptr(4, C.ptr());
  k.arg(5, M);
  k.arg(6, K);
  k.arg(7, N);
  k.arg(8, K / 2);   // ldxq, in int8 pairs
  k.arg(9, N);       // ldb
  k.arg(10, ldc);
  d.run(k, M / 256, N / 128);
}

// The exact int32 accumulators: acc[m][n] = sum_k x[m][k] w[k][n], w read out of
// the VNNI-4 dwords (byte b of dword [k/4][n] is k = 4 (k/4) + b).
std::vector<int32_t> acc_ref(const std::vector<int8_t>& x, const std::vector<uint32_t>& w8,
                             uint32_t M, uint32_t K, uint32_t N) {
  std::vector<int8_t> wt(size_t(N) * K);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      wt[size_t(n) * K + k] = int8_t((w8[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu);
  std::vector<int32_t> acc(size_t(M) * N);
  for (uint32_t m = 0; m < M; ++m) {
    const int8_t* xr = x.data() + size_t(m) * K;
    for (uint32_t n = 0; n < N; ++n) {
      const int8_t* wr = wt.data() + size_t(n) * K;
      int32_t s = 0;
      for (uint32_t k = 0; k < K; ++k) s += int32_t(xr[k]) * int32_t(wr[k]);
      acc[size_t(m) * N + n] = s;
    }
  }
  return acc;
}

struct GemmOperands {
  std::vector<int8_t> x;
  std::vector<uint32_t> w8;
  std::vector<float> xs, ws;
};

GemmOperands random_operands(uint32_t K, uint32_t N, uint32_t M, uint32_t seed) {
  std::mt19937 rng(seed);
  GemmOperands o;
  o.x.resize(size_t(M) * K);
  for (int8_t& v : o.x) v = int8_t(uint8_t(rng() & 0xFFu));
  o.w8.resize(size_t(K / 4) * N);
  for (uint32_t& v : o.w8) v = rng();
  std::uniform_real_distribution<float> sd(0.001f, 0.05f);
  o.xs.resize(M);
  for (float& v : o.xs) v = sd(rng);
  o.ws.resize(N);
  for (float& v : o.ws) v = sd(rng);
  return o;
}

void gemm_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t M) {
  const GemmOperands o = random_operands(K, N, M, 41 + K + N + M);
  l0::Mem dx = pf_harness::upload(d.ctx, d.imm, o.x);
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, o.w8);
  l0::Mem dxs = pf_harness::upload(d.ctx, d.imm, o.xs);
  l0::Mem dws = pf_harness::upload(d.ctx, d.imm, o.ws);
  l0::Mem dc(d.ctx, l0::MemKind::Device, size_t(M) * N * 4);
  run_gemm(d, false, dx, dxs, dw, dws, dc, M, K, N, N);
  std::vector<float> got(size_t(M) * N);
  pf_harness::download(d.imm, got, dc);
  const std::vector<int32_t> acc = acc_ref(o.x, o.w8, M, K, N);
  size_t bad = 0;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      const float want = float(acc[size_t(m) * N + n]) * (o.ws[n] * o.xs[m]);
      bad += std::memcmp(&want, &got[size_t(m) * N + n], 4) != 0;
    }
  std::printf("pf_gemm_i8 K=%u N=%u M=%u: %zu mismatches\n", K, N, M, bad);
  CHECK(bad == 0);
}

void silu_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t M) {
  const GemmOperands o = random_operands(K, N, M, 43 + K + N + M);
  l0::Mem dx = pf_harness::upload(d.ctx, d.imm, o.x);
  l0::Mem dw = pf_harness::upload(d.ctx, d.imm, o.w8);
  l0::Mem dxs = pf_harness::upload(d.ctx, d.imm, o.xs);
  l0::Mem dws = pf_harness::upload(d.ctx, d.imm, o.ws);
  const uint32_t ldx = N / 2;
  l0::Mem dout(d.ctx, l0::MemKind::Device, size_t(M) * ldx * 2);
  run_gemm(d, true, dx, dxs, dw, dws, dout, M, K, N, ldx);
  std::vector<uint16_t> got(size_t(M) * ldx);
  pf_harness::download(d.imm, got, dout);
  const std::vector<int32_t> acc = acc_ref(o.x, o.w8, M, K, N);
  size_t bad = 0;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      if ((n / 16) % 2 != 0) continue;   // gate columns; up is n + 16
      const float g = float(acc[size_t(m) * N + n]) * (o.ws[n] * o.xs[m]);
      const float u = float(acc[size_t(m) * N + n + 16]) * (o.ws[n + 16] * o.xs[m]);
      const uint16_t want = rne_bf16(bf16f(rne_bf16(silu_f32(bf16f(rne_bf16(g))))) * bf16f(rne_bf16(u)));
      bad += (want != got[size_t(m) * ldx + (n / 32) * 16 + n % 16]);
    }
  std::printf("pf_gemm_i8_SILU K=%u N=%u M=%u: %zu mismatches\n", K, N, M, bad);
  CHECK(bad == 0);
}

// The whole h8 path on one layout-0 weight: pf_quant_had, pf_colmax_rot (host
// finish), pf_requant_rot, pf_gemm_i8. Rows [real, M) of x are padding.
std::vector<float> h8_run(pf_harness::Dev& d, const common::Int4Gptq& w,
                          const std::vector<uint16_t>& x, uint32_t M) {
  const uint32_t K = w.K, N = w.N;
  l0::Mem dx = pf_harness::upload(d.ctx, d.imm, x);
  l0::Mem dsg = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_signs(K));
  l0::Mem dbits = pf_harness::upload(d.ctx, d.imm, runtime::prefill::int8_sign_bits(K));
  l0::Mem dqw = pf_harness::upload(d.ctx, d.imm, w.qweight);
  l0::Mem dsc = pf_harness::upload(d.ctx, d.imm, w.scales);
  l0::Mem dxq(d.ctx, l0::MemKind::Device, size_t(M) * K);
  l0::Mem dxs(d.ctx, l0::MemKind::Device, size_t(M) * 4);
  {
    l0::Module mod(d.ctx, kernels::path(kernels::pf_quant_had_variant(K)));
    l0::Kernel k = mod.kernel("pf_quant_had");
    k.group_size(16 * (K / 1024));
    k.arg_ptr(0, dx.ptr());
    k.arg_ptr(1, dsg.ptr());
    k.arg_ptr(2, dxq.ptr());
    k.arg_ptr(3, dxs.ptr());
    k.arg(4, K);
    d.run(k, M, 1);
  }
  l0::Module rmod(d.ctx, kernels::path(kernels::pf_requant_rot_variant(0)));
  l0::Mem dmax = pf_harness::upload(d.ctx, d.imm, std::vector<uint32_t>(N, 0u));
  {
    l0::Kernel k = rmod.kernel("pf_colmax_rot");
    k.group_size(256);
    k.arg_ptr(0, dqw.ptr());
    k.arg_ptr(1, dsc.ptr());
    k.arg_ptr(2, dbits.ptr());
    k.arg_ptr(3, dmax.ptr());
    k.arg(4, 0u);
    k.arg(5, N);
    k.arg(6, K);
    d.run(k, N / 16, K / 1024);
  }
  std::vector<uint32_t> mx(N);
  pf_harness::download(d.imm, mx, dmax);
  std::vector<float> ws(N), inv(N);
  for (uint32_t n = 0; n < N; ++n) {
    float m;
    std::memcpy(&m, &mx[n], 4);
    ws[n] = m > 0.0f ? m / 127.0f : 1.0f;
    inv[n] = 1.0f / ws[n];
  }
  l0::Mem dws = pf_harness::upload(d.ctx, d.imm, ws);
  l0::Mem dinv = pf_harness::upload(d.ctx, d.imm, inv);
  l0::Mem dw8(d.ctx, l0::MemKind::Device, size_t(K) * N);
  {
    l0::Kernel k = rmod.kernel("pf_requant_rot");
    k.group_size(256);
    k.arg_ptr(0, dqw.ptr());
    k.arg_ptr(1, dsc.ptr());
    k.arg_ptr(2, dbits.ptr());
    k.arg_ptr(3, dinv.ptr());
    k.arg_ptr(4, dw8.ptr());
    k.arg(5, 0u);
    k.arg(6, N);
    k.arg(7, K);
    k.arg(8, N);
    d.run(k, N / 16, K / 1024);
  }
  l0::Mem dc(d.ctx, l0::MemKind::Device, size_t(M) * N * 4);
  run_gemm(d, false, dxq, dxs, dw8, dws, dc, M, K, N, N);
  std::vector<float> c(size_t(M) * N);
  pf_harness::download(d.imm, c, dc);
  return c;
}

void cross_case(pf_harness::Dev& d, uint32_t K, uint32_t N, uint32_t M, uint32_t real) {
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 53 + K + N);
  std::vector<uint16_t> x = pf_harness::random_bf16(size_t(M) * K, 59 + K, -1.0f, 1.0f);
  std::vector<uint16_t> xz = x;
  for (size_t i = size_t(real) * K; i < x.size(); ++i) {
    x[i] = 0x7FC0u;   // NaN bf16: stale padding at its worst
    xz[i] = 0u;
  }
  const std::vector<float> got = h8_run(d, w, x, M);
  const std::vector<float> got_z = h8_run(d, w, xz, M);
  // fp32 x W^T from the dequantised int4, accumulated in double.
  std::vector<float> wt(size_t(N) * K);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) wt[size_t(n) * K + k] = w.at(k, n);
  std::vector<float> xf(size_t(real) * K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = common::bf16_to_f32(x[i]);
  double num = 0.0, den = 0.0;
  size_t nan = 0, leak = 0;
  for (uint32_t m = 0; m < real; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0.0;
      const float* xr = xf.data() + size_t(m) * K;
      const float* wr = wt.data() + size_t(n) * K;
      for (uint32_t k = 0; k < K; ++k) ref += double(xr[k]) * double(wr[k]);
      const float g = got[size_t(m) * N + n];
      nan += std::isnan(g) ? 1 : 0;
      leak += std::memcmp(&g, &got_z[size_t(m) * N + n], 4) != 0;
      num += (double(g) - ref) * (double(g) - ref);
      den += ref * ref;
    }
  const double rel = std::sqrt(num / den);
  std::printf("h8 cross K=%u N=%u M=%u (rows %u..%u padding): rel L2 %.4f %%, %zu NaN, %zu words "
              "differ NaN vs zero padding\n",
              K, N, M, real, M - 1, 100.0 * rel, nan, leak);
  CHECK(nan == 0);
  CHECK(leak == 0);
  CHECK(rel <= 0.03);
  std::printf("h8 cross: PASS, padding isolated\n");
}
}  // namespace

int main() {
  pf_harness::Dev d;
  for (uint32_t K : {5120u, 6144u, 17408u, 19456u}) quant_case(d, K, 64);   // 19456: spec 14
  requant_case(d, 5120, 1024, 0);
  requant_case(d, 5120, 1024, 1);
  requant_case(d, 6144, 1024, 0);
  requant_case(d, 17408, 1024, 0);
  colmax_case(d, 5120, 1024, 0);
  colmax_case(d, 5120, 1024, 1);
  colmax_case(d, 17408, 1024, 0);
  requant_case(d, 19456, 1024, 0);   // spec 14: Agnes down' (19 Hadamard blocks)
  colmax_case(d, 19456, 1024, 0);
  gemm_case(d, 19456, 1024, 512);
  gemm_case(d, 5120, 1024, 256);
  gemm_case(d, 17408, 1024, 512);
  silu_case(d, 5120, 1024, 256);
  cross_case(d, 5120, 1024, 256, 200);
  std::printf("pf_int8_test: PASS\n");
  return 0;
}
