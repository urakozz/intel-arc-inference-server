// gemm_sm_ref_test - spec 28's host reference (tests/kernels/gemm_sm_ref.h), host only:
//   1. the layout-1 view reads exactly the layout-0 view's weights and scales (every element);
//   2. the fp64 reference against gemv.cl's fp32 twin within spec 28 N2's bound, both
//      layouts, M = 1 / 3 / 8 / 16 and S = 1 / 2 / 4 (random weights, random bf16 rows);
//   3. the int8 head's view: the per-column scale applied once, the twin within N2;
//   4. the N2 checker refuses an output moved by more than its bound, and passes an exact one;
//   5. the i8 quantiser: |x - q s| <= s / 2 per element, and q in [-127, 127], max |q| = 127.
#include <cstdio>
#include <random>
#include <vector>
#include "check.h"
#include "common/int4.h"
#include "gemm_sm_ref.h"
#include "gemv_ref.h"

using gemm_sm::WeightView;

int main() {
  // 1 + 2: int4, both layouts.
  for (uint32_t K : {256u, 384u}) {
    const uint32_t N = 64;
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 7 + K);
    std::vector<uint32_t> tiles = w.tiled();
    WeightView l0{K, N, 0, w.qweight.data(), w.scales.data()};
    WeightView l1{K, N, 1, tiles.data()};
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n) {
        CHECK_EQ(l0.q(k, n), l1.q(k, n));
        CHECK(l0.scale(k, n) == l1.scale(k, n));
        CHECK(double(w.at(k, n)) == l0.w(k, n));
      }
    for (uint32_t M : {1u, 3u, 8u, 16u}) {
      std::vector<float> xf = gemm_sm::bf16_rows(random_bf16(size_t(M) * K, 100 + M));
      std::vector<double> r, b;
      gemm_sm::ref(l0, xf, M, r, b);
      std::vector<double> r1, b1;
      gemm_sm::ref(l1, xf, M, r1, b1);
      CHECK(r == r1 && b == b1);
      for (uint32_t S : {1u, 2u, 4u}) {
        if ((K / 64) % S) continue;
        for (const WeightView* v : {&l0, &l1}) {
          gemm_sm::N2 e = gemm_sm::n2(gemm_sm::twin_gemv(*v, xf, M, S), r, b, M, N);
          std::printf("int4 L%u K=%u M=%u S=%u: max ratio %.3g median %.3g min cos %.10f\n", v->kind,
                      K, M, S, e.max_ratio, e.median_ratio, e.min_cos);
          CHECK(e.pass());
        }
      }
    }
  }
  // 3: the int8 head, tiled [n_tile][k16][16][16], fp32 column scale.
  {
    const uint32_t K = 256, N = 48;
    std::mt19937 rng(5);
    std::vector<int8_t> q8(size_t(K) * N);
    std::vector<float> cs(N);
    std::vector<double> dense(size_t(K) * N);
    for (uint32_t n = 0; n < N; ++n) cs[n] = 0.001f + 0.0001f * float(n);
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n) {
        const int v = int(rng() % 255) - 127;
        q8[(size_t(n / 16) * (K / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16] = int8_t(v);
        dense[size_t(k) * N + n] = double(v) * double(cs[n]);
      }
    WeightView h{K, N, 2, nullptr, nullptr, q8.data(), cs.data()};
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n) CHECK(h.w(k, n) == dense[size_t(k) * N + n]);
    for (uint32_t M : {1u, 8u, 16u}) {
      std::vector<float> xf = gemm_sm::bf16_rows(random_bf16(size_t(M) * K, 300 + M));
      std::vector<double> r, b;
      gemm_sm::ref(h, xf, M, r, b);
      gemm_sm::N2 e = gemm_sm::n2(gemm_sm::twin_gemv_i8w(h, xf, M), r, b, M, N);
      std::printf("int8 head M=%u: max ratio %.3g min cos %.10f\n", M, e.max_ratio, e.min_cos);
      CHECK(e.pass());
    }
  }
  // 4: the checker itself.
  {
    std::vector<double> r = {1.0, -2.0, 0.0}, b = {65536.0, 65536.0, 0.0};
    std::vector<float> exact = {1.f, -2.f, 0.f};
    CHECK(gemm_sm::n2(exact, r, b, 1, 3).pass());
    std::vector<float> at = {2.f, -2.f, 0.f};   // err 1 = exactly the bound 2^-16 x 65536
    CHECK(gemm_sm::n2(at, r, b, 1, 3).over == 0);
    std::vector<float> over = {2.5f, -2.f, 0.f};
    CHECK(gemm_sm::n2(over, r, b, 1, 3).over == 1);
    std::vector<float> zero_row = {1.f, -2.f, 1e-30f};   // bound 0 must be exactly 0
    CHECK(gemm_sm::n2(zero_row, r, b, 1, 3).over == 1);
    // a NaN or infinite output never passes (an unwritten register, a broken dequant)
    std::vector<float> nan_out = {std::nanf(""), -2.f, 0.f};
    CHECK(!gemm_sm::n2(nan_out, r, b, 1, 3).pass());
    std::vector<float> inf_out = {1.f, -INFINITY, 0.f};
    CHECK(!gemm_sm::n2(inf_out, r, b, 1, 3).pass());
  }
  // 5: the i8 quantiser.
  {
    const uint32_t M = 4, K = 512;
    std::vector<float> xf = gemm_sm::bf16_rows(random_bf16(size_t(M) * K, 77, -3.f, 3.f));
    for (uint32_t k = 0; k < K; ++k) xf[3 * K + k] = 0.f;   // an all-zero row
    std::vector<int8_t> q;
    std::vector<float> s;
    gemm_sm::quant_rows_i8(xf, M, K, q, s);
    CHECK(s[3] == 0.f);
    for (uint32_t m = 0; m < 3; ++m) {
      int amax = 0;
      for (uint32_t k = 0; k < K; ++k) {
        const int v = q[size_t(m) * K + k];
        CHECK(v >= -127 && v <= 127);
        amax = std::max(amax, v < 0 ? -v : v);
        CHECK(std::fabs(xf[size_t(m) * K + k] - float(v) * s[m]) <= 0.5f * s[m] * (1 + 1e-6f));
      }
      CHECK_EQ(amax, 127);
    }
  }
  std::puts("gemm_sm_ref_test OK");
  return 0;
}
