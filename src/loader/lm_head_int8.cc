#include "loader/lm_head_int8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/bf16.h"

namespace loader {

bool parse_lm_head_form(const std::string& s, LmHeadForm& out) {
  if (s == "bf16") { out = LmHeadForm::Checkpoint; return true; }
  if (s == "int8") { out = LmHeadForm::Int8; return true; }
  return false;
}
const char* lm_head_form_name(LmHeadForm f) { return f == LmHeadForm::Int8 ? "int8" : "bf16"; }

float quantise_row_int8(const uint16_t* w, uint32_t K, int8_t* q) {
  float amax = 0.f;
  for (uint32_t k = 0; k < K; ++k) amax = std::fmax(amax, std::fabs(common::bf16_to_f32(w[k])));
  const float s = amax / 127.0f;
  if (s == 0.f) {
    std::fill(q, q + K, int8_t(0));
    return 0.f;
  }
  for (uint32_t k = 0; k < K; ++k) {
    // nearbyint under the default rounding mode is round-to-nearest-even.
    const float r = std::nearbyint(common::bf16_to_f32(w[k]) / s);
    q[k] = int8_t(std::clamp(r, -127.f, 127.f));
  }
  return s;
}

void quantise_int8_tiled(const uint16_t* w, uint32_t K, uint32_t N, int8_t* q_tiled,
                         float* scales, unsigned threads) {
  if (K % 16 != 0 || N % 16 != 0)
    throw std::runtime_error("quantise_int8_tiled: K and N must be multiples of 16");
  if (int8_k_major(N) && K % (16 * kInt8HeadKBlock) != 0)
    throw std::runtime_error("quantise_int8_tiled: an output head's K must be a multiple of " +
                             std::to_string(16 * kInt8HeadKBlock));
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t tiles = N / 16;
  threads = std::min<unsigned>(threads, tiles);
  // Whole 16-row tiles per thread: a tile's 256-byte steps are whole blocks (contiguous
  // n-major, in blocks k-major), so no two threads write the same cache line.
  auto work = [&](uint32_t t0, uint32_t t1) {
    std::vector<int8_t> row(K);
    for (uint32_t t = t0; t < t1; ++t)
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t n = t * 16 + j;
        scales[n] = quantise_row_int8(w + size_t(n) * K, K, row.data());
        for (uint32_t k0 = 0; k0 < K; k0 += 16) {
          int8_t* dst = q_tiled + int8_tiled_index(K, N, k0, n);
          for (uint32_t k = 0; k < 16; ++k) dst[size_t(k) * 16] = row[k0 + k];
        }
      }
  };
  std::vector<std::thread> pool;
  const uint32_t per = (tiles + threads - 1) / threads;
  for (unsigned i = 0; i < threads; ++i) {
    const uint32_t t0 = i * per, t1 = std::min(tiles, t0 + per);
    if (t0 >= t1) break;
    pool.emplace_back(work, t0, t1);
  }
  for (std::thread& th : pool) th.join();
}

namespace {
// A non-negative finite float to IEEE half, round to nearest even, subnormals kept (common::f32_to_f16 flushes them,
// and a group of small weights - max|W| under 127 x 2^-14 - would lose its scale).
uint16_t f16_rne(float f) {
  if (f < 0x1p-14f) return uint16_t(std::nearbyint(f * 0x1p24f));   // exact scaling; 1024 is the least normal's bits
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
  if (exp >= 31) return uint16_t(0x7C00u);
  uint32_t half = (uint32_t(exp) << 10) | ((u & 0x7FFFFFu) >> 13);
  const uint32_t rem = u & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return uint16_t(half);
}
}  // namespace

void quantise_int8_g32_tiled(const uint16_t* w, uint32_t K, uint32_t N, int8_t* q_tiled, uint16_t* scales_f16,
                             unsigned threads) {
  if (K % 32 != 0 || N % 16 != 0)
    throw std::runtime_error("quantise_int8_g32_tiled: K must be a multiple of 32 and N of 16");
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t tiles = N / 16;
  threads = std::min<unsigned>(threads, tiles);
  auto work = [&](uint32_t t0, uint32_t t1) {
    for (uint32_t t = t0; t < t1; ++t)
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t n = t * 16 + j;
        const uint16_t* row = w + size_t(n) * K;
        int8_t* dst = q_tiled + size_t(t) * K * 16 + j;
        for (uint32_t g = 0; g < K / 32; ++g) {
          float amax = 0.f;
          for (uint32_t k = 32 * g; k < 32 * g + 32; ++k)
            amax = std::fmax(amax, std::fabs(common::bf16_to_f32(row[k])));
          const uint16_t sh = f16_rne(amax / 127.0f);
          const float s = common::f16_to_f32(sh);
          scales_f16[size_t(g) * N + n] = sh;
          for (uint32_t k = 32 * g; k < 32 * g + 32; ++k) {
            // nearbyint under the default rounding mode is round-to-nearest-even.
            const float r = s == 0.f ? 0.f : std::nearbyint(common::bf16_to_f32(row[k]) / s);
            dst[size_t(k) * 16] = int8_t(std::clamp(r, -127.f, 127.f));
          }
        }
      }
  };
  std::vector<std::thread> pool;
  const uint32_t per = (tiles + threads - 1) / threads;
  for (unsigned i = 0; i < threads; ++i) {
    const uint32_t t0 = i * per, t1 = std::min(tiles, t0 + per);
    if (t0 >= t1) break;
    pool.emplace_back(work, t0, t1);
  }
  for (std::thread& th : pool) th.join();
}

}  // namespace loader
