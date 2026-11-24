#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "common/bf16.h"

namespace inkling2_ref {

constexpr size_t kLanes = 128;

template <class F>
float sum(size_t n, F value) {
  std::array<float, kLanes> partial{};
  for (size_t lane = 0; lane < kLanes; ++lane)
    for (size_t i = lane; i < n; i += kLanes) partial[lane] += value(i);
  for (size_t s = kLanes / 2; s > 0; s >>= 1)
    for (size_t i = 0; i < s; ++i) partial[i] += partial[i + s];
  return partial[0];
}

inline float sqnorm(std::span<const float> x) {
  return sum(x.size(), [&](size_t i) { return x[i] * x[i]; });
}

inline std::vector<float> depth_mix(std::span<const float> x, std::span<const float> q, float eps) {
  const size_t H = q.size(), n = x.size() / H;
  std::vector<float> scales(n), scores(n), result(H);
  for (size_t j = 0; j < n; ++j) {
    const auto row = x.subspan(j * H, H);
    scales[j] = 1.f / std::sqrt(sqnorm(row) / float(H) + eps);
    scores[j] = sum(H, [&](size_t i) { return q[i] * row[i]; }) * scales[j] / std::sqrt(float(H));
  }
  const float high = *std::max_element(scores.begin(), scores.end());
  float total = 0.f;
  for (float& score : scores) {
    score = std::exp(score - high);
    total += score;
  }
  for (size_t j = 0; j < n; ++j) scales[j] *= scores[j] / total;
  for (size_t i = 0; i < H; ++i)
    for (size_t j = 0; j < n; ++j) result[i] += scales[j] * x[j * H + i];
  return result;
}

inline float controller(float x, float message, float forget, float gain) {
  return (1.f - std::tanh(forget)) * x + std::tanh(gain) * message;
}

inline float convolve(std::span<const float> input, std::span<const float> history, uint32_t width,
                      uint32_t row, uint32_t channel, std::span<const float> taps,
                      std::span<const float> projected = {}, uint32_t head = 0) {
  float acc = 0.f;
  for (int a = 0; a < 4; ++a) {
    const int t = projected.empty() ? int(row) + a - 3 : int(row) - a;
    const float x = t < 0 ? history[(t + 3) * width + channel] : input[t * width + channel];
    const float w = projected.empty() ? taps[channel * 4 + a] : taps[a * width + channel] + projected[head * 4 + a];
    acc += x * w;
  }
  return (input[row * width + channel] + acc) / (projected.empty() ? std::sqrt(5.f) : 3.f);
}

inline void commit_conv(std::span<const float> pending, std::span<float> history, size_t width, size_t accepted) {
  for (size_t j = 0; j < width; ++j) {
    std::array<float, 3> next{};
    for (size_t i = 0; i < next.size(); ++i) {
      const size_t row = i + accepted;
      next[i] = row < 3 ? history[row * width + j] : pending[(row - 3) * width + j];
    }
    for (size_t i = 0; i < next.size(); ++i) history[i * width + j] = next[i];
  }
}

inline void head_scale(std::span<float> x, float clip, float eps, bool normalise, float tau = 1.f) {
  const float sq = sqnorm(x);
  const float scale = normalise ? 1.f / std::sqrt(sq / float(x.size()) + eps)
                                : std::min(1.f, clip / (std::sqrt(sq) + 1e-6f));
  for (float& v : x) v = v * scale * tau;
}

inline float relative_bias(std::span<const float> query, std::span<const float> table,
                            uint32_t extent, uint32_t distance) {
  if (distance >= extent) return 0.f;
  float bias = 0.f;
  for (size_t i = 0; i < query.size(); ++i) bias = std::fma(query[i], table[i * extent + distance], bias);
  return bias;
}

inline uint16_t gated_value(float acc, float denominator, float gate) {
  return common::f32_to_bf16((acc / denominator) / (1.f + std::exp(-gate)));
}

}  // namespace inkling2_ref
