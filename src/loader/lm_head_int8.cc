#include "loader/lm_head_int8.h"

#include <algorithm>
#include <cmath>
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

namespace {
// Both tiled layouts (int8 [N/16][K/16][16 k][16 n], bf16 [N/16][K/8][8 k][16 n]) put
// element (k, n) at (n / 16) * 16K + 16k + n % 16, counted in elements: a 16-row tile is
// K x 16 contiguous elements, row n's k-th element at stride 16. So one gather serves
// both. One output tile per thread step: lane l of compact tile t is row ids[16t + l].
template <class T>
void gather_tiled_rows(const char* what, const T* src_tiled, const float* scales, uint32_t K,
                       uint32_t N, const uint32_t* ids, uint32_t n, T* out_tiled,
                       float* out_scales, unsigned threads) {
  if (K % 16 != 0 || N % 16 != 0 || n % 16 != 0)
    throw std::runtime_error(std::string(what) + ": K, N and n must be multiples of 16");
  for (uint32_t j = 0; j < n; ++j)
    if (ids[j] >= N)
      throw std::runtime_error(std::string(what) + ": id " + std::to_string(ids[j]) +
                               " is outside the head's " + std::to_string(N) + " rows");
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t tiles = n / 16;
  const size_t tile_elems = size_t(K) * 16;
  threads = std::max(1u, std::min<unsigned>(threads, tiles));
  auto work = [&](uint32_t t0, uint32_t t1) {
    for (uint32_t t = t0; t < t1; ++t) {
      T* dst = out_tiled + size_t(t) * tile_elems;
      for (uint32_t l = 0; l < 16; ++l) {
        const uint32_t id = ids[t * 16 + l];
        if (out_scales) out_scales[t * 16 + l] = scales[id];
        const T* src = src_tiled + size_t(id / 16) * tile_elems + id % 16;
        for (uint32_t k = 0; k < K; ++k) dst[size_t(k) * 16 + l] = src[size_t(k) * 16];
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
}  // namespace

void gather_int8_tiled_rows(const int8_t* q_tiled, const float* scales, uint32_t K, uint32_t N,
                            const uint32_t* ids, uint32_t n, int8_t* out_tiled,
                            float* out_scales, unsigned threads) {
  gather_tiled_rows("gather_int8_tiled_rows", q_tiled, scales, K, N, ids, n, out_tiled, out_scales,
                    threads);
}

void gather_bf16_tiled_rows(const uint16_t* w_tiled, uint32_t K, uint32_t N, const uint32_t* ids,
                            uint32_t n, uint16_t* out_tiled, unsigned threads) {
  gather_tiled_rows<uint16_t>("gather_bf16_tiled_rows", w_tiled, nullptr, K, N, ids, n, out_tiled,
                              nullptr, threads);
}

}  // namespace loader
