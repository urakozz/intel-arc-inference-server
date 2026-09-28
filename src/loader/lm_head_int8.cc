#include "loader/lm_head_int8.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
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
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t tiles = N / 16;
  threads = std::min<unsigned>(threads, tiles);
  // Whole 16-row tiles per thread: a tile's bytes are one contiguous block, so no
  // two threads write the same cache line.
  auto work = [&](uint32_t t0, uint32_t t1) {
    std::vector<int8_t> row(K);
    for (uint32_t t = t0; t < t1; ++t)
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t n = t * 16 + j;
        scales[n] = quantise_row_int8(w + size_t(n) * K, K, row.data());
        int8_t* dst = q_tiled + size_t(t) * K * 16 + j;
        for (uint32_t k = 0; k < K; ++k) dst[size_t(k) * 16] = row[k];
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
