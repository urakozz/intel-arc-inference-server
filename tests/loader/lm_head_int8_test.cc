// Spec 9 §3 (plan 9b Task 1): the int8 `lm_head`, quantised on the host at load.
// Host-only, on synthetic rows: the per-row scale (max|row| / 127, fp32), round to
// nearest even, the clamp to [-127, 127], the round-trip bound plan 9a measured on
// the real head, the tiled layout gemv_i8w reads, and that the thread count does not
// change a byte. PERF-111: at an output head's width the tiles are k-major (while
// loader::kInt8HeadKMajor holds) - today's n-major tiles reordered by common::tiles_k_major,
// byte for byte; every other width keeps n-major.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/lm_head_int8.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

namespace {
std::vector<uint16_t> random_rows(uint32_t N, uint32_t K, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.f, 0.02f);
  std::vector<uint16_t> w(size_t(N) * K);
  for (size_t i = 0; i < w.size(); ++i) w[i] = common::f32_to_bf16(d(g));
  // A few rows with one large outlier (the case the per-row scale exists for), one
  // all-zero row, and one row whose maximum is negative.
  for (uint32_t k = 0; k < K; ++k) w[size_t(5) * K + k] = 0;
  w[size_t(3) * K + 17] = common::f32_to_bf16(0.9f);
  w[size_t(7) * K + K - 1] = common::f32_to_bf16(-1.5f);
  return w;
}
}  // namespace

int main() {
  // 1. Scale, rounding and clamp on hand-made rows. max 127 gives s = 1 exactly, so
  //    w / s is w and the ties are exact: 2.5 -> 2, 3.5 -> 4, -2.5 -> -2 (RNE).
  {
    const uint32_t K = 16;
    std::vector<uint16_t> w(K, 0);
    const float vals[] = {127.f, 2.5f, 3.5f, -2.5f, -127.f, 0.5f, 1.5f, -0.5f};
    for (uint32_t k = 0; k < 8; ++k) w[k] = common::f32_to_bf16(vals[k]);
    std::vector<int8_t> q(K);
    const float s = loader::quantise_row_int8(w.data(), K, q.data());
    CHECK_EQ(s, 1.0f);
    const int want[] = {127, 2, 4, -2, -127, 0, 2, 0};
    for (uint32_t k = 0; k < 8; ++k) CHECK_EQ(int(q[k]), want[k]);
    for (uint32_t k = 8; k < K; ++k) CHECK_EQ(int(q[k]), 0);
  }
  // A zero row: scale 0 and q 0, never a division by zero.
  {
    std::vector<uint16_t> w(64, 0);
    std::vector<int8_t> q(64, 1);
    CHECK_EQ(loader::quantise_row_int8(w.data(), 64, q.data()), 0.0f);
    for (int8_t v : q) CHECK_EQ(int(v), 0);
  }

  // 2. Synthetic rows through the tiled path: every element against its own row's
  //    quantisation at the tiled index, the scale layout [N], the round-trip bound
  //    |w - q s| <= s (0.5 + 127 2^-23), and the clamp (the row max maps to +-127).
  const uint32_t N = 64, K = 256;
  const std::vector<uint16_t> w = random_rows(N, K, 1234);
  std::vector<int8_t> tiled(size_t(N) * K);
  std::vector<float> scales(N);
  loader::quantise_int8_tiled(w.data(), K, N, tiled.data(), scales.data(), 1);
  size_t bound_fail = 0, layout_fail = 0;
  for (uint32_t n = 0; n < N; ++n) {
    std::vector<int8_t> row(K);
    const float s = loader::quantise_row_int8(w.data() + size_t(n) * K, K, row.data());
    CHECK_EQ(scales[n], s);
    float amax = 0.f;
    for (uint32_t k = 0; k < K; ++k)
      amax = std::fmax(amax, std::fabs(common::bf16_to_f32(w[size_t(n) * K + k])));
    CHECK_EQ(s, amax / 127.0f);
    int qmax = 0;
    for (uint32_t k = 0; k < K; ++k) {
      const int8_t v = tiled[loader::int8_tiled_index(K, N, k, n)];
      layout_fail += v != row[k];
      CHECK(v >= -127 && v <= 127);
      qmax = std::max(qmax, std::abs(int(v)));
      const double wf = common::bf16_to_f32(w[size_t(n) * K + k]);
      bound_fail += std::fabs(wf - double(v) * s) > double(s) * (0.5 + 127.0 * std::ldexp(1.0, -23));
    }
    CHECK_EQ(qmax, n == 5 ? 0 : 127);
  }
  CHECK_EQ(layout_fail, size_t(0));
  CHECK_EQ(bound_fail, size_t(0));
  // The tiled index is a bijection onto [0, N K): [n/16][k/16][k%16][n%16] at a width no head has.
  {
    CHECK(!loader::int8_k_major(N));
    std::vector<uint8_t> seen(size_t(N) * K, 0);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) ++seen[loader::int8_tiled_index(K, N, k, n)];
    for (uint8_t c : seen) CHECK_EQ(int(c), 1);
    CHECK_EQ(loader::int8_tiled_index(K, N, 1, 0), size_t(16));
    CHECK_EQ(loader::int8_tiled_index(K, N, 16, 0), size_t(256));
    CHECK_EQ(loader::int8_tiled_index(K, N, 0, 16), size_t(K) * 16);
  }

  // 2b. PERF-111: at a head width (the TP half, 124160; K 128, one K block) the quantised tiles are today's n-major tiles
  //     reordered k-major in blocks of kInt8HeadKBlock steps by common::tiles_k_major, byte for byte, the scales
  //     unchanged; the index is a bijection; 7 threads write the same bytes.
  {
    const uint32_t HN = 124160, HK = 128;
    CHECK_EQ(loader::int8_k_major(HN), loader::kInt8HeadKMajor);
    const std::vector<uint16_t> hw = random_rows(HN, HK, 77);
    std::vector<int8_t> nm(size_t(HN) * HK), want(nm.size()), got(nm.size()), got7(nm.size());
    std::vector<float> s_want(HN), s_got(HN), s_got7(HN);
    for (uint32_t n = 0; n < HN; ++n) {
      std::vector<int8_t> row(HK);
      s_want[n] = loader::quantise_row_int8(hw.data() + size_t(n) * HK, HK, row.data());
      for (uint32_t k = 0; k < HK; ++k)
        nm[(size_t(n / 16) * (HK / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16] = row[k];
    }
    if (loader::kInt8HeadKMajor)
      common::tiles_k_major(nm.data(), HN / 16, HK / 16, 256, want.data(), loader::kInt8HeadKBlock);
    else want = nm;
    loader::quantise_int8_tiled(hw.data(), HK, HN, got.data(), s_got.data(), 1);
    loader::quantise_int8_tiled(hw.data(), HK, HN, got7.data(), s_got7.data(), 7);
    CHECK(got == want);
    CHECK(got7 == want);
    CHECK(s_got == s_want && s_got7 == s_want);
    std::vector<uint8_t> seen(size_t(HN) * HK, 0);
    for (uint32_t n = 0; n < HN; ++n)
      for (uint32_t k = 0; k < HK; ++k) ++seen[loader::int8_tiled_index(HK, HN, k, n)];
    for (uint8_t c : seen) CHECK_EQ(int(c), 1);
    if (loader::kInt8HeadKMajor) {
      const size_t B = loader::kInt8HeadKBlock;
      CHECK_EQ(loader::int8_tiled_index(HK, HN, 16, 0), size_t(B > 1 ? 256 : size_t(HN / 16) * 256));
      CHECK_EQ(loader::int8_tiled_index(HK, HN, 16 * uint32_t(B), 0), size_t(HN / 16) * B * 256);
      CHECK_EQ(loader::int8_tiled_index(HK, HN, 0, 16), B * 256);
    }
    for (uint32_t w : loader::kInt8HeadWidths) CHECK(w % 64 == 0);   // gemv_i8w's grid: N / 64 work-groups
  }

  // 3. Threads change nothing.
  {
    std::vector<int8_t> t7(size_t(N) * K);
    std::vector<float> s7(N);
    loader::quantise_int8_tiled(w.data(), K, N, t7.data(), s7.data(), 7);
    CHECK(std::memcmp(t7.data(), tiled.data(), t7.size()) == 0);
    CHECK(std::memcmp(s7.data(), scales.data(), N * 4) == 0);
  }

  // 4. The form names and the model row.
  {
    loader::LmHeadForm f = loader::LmHeadForm::Int8;
    CHECK(loader::parse_lm_head_form("bf16", f) && f == loader::LmHeadForm::Checkpoint);
    CHECK(loader::parse_lm_head_form("int8", f) && f == loader::LmHeadForm::Int8);
    CHECK(!loader::parse_lm_head_form("int4", f));
    const model::FusedLinear& r = model::qwen38().lm_head(model::WeightKind::Int8);
    CHECK(r.kind == model::WeightKind::Int8);
    CHECK_EQ(r.shape.K, uint32_t(5120));
    CHECK_EQ(r.shape.N, uint32_t(248320));
    CHECK_EQ(r.shape.S, uint32_t(1));
    CHECK_EQ(loader::lm_head_int8_bytes(r.shape.K, r.shape.N), size_t(1271398400) + 993280);
  }
  std::printf("lm_head_int8_test OK\n");
  return 0;
}
