#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q4judge {

inline bool finite_reference(const float* values, size_t count) {
  return std::all_of(values, values + count, [](float x) { return std::isfinite(x); });
}

// Compare the same number of layer-position decisions and teacher-forced token rows on both paths.
struct RelativeCounts {
  uint64_t prefill_rejected = 0, decode_rejected = 0;
  uint32_t prefill_accepted = 0, decode_accepted = 0;
  bool routes_ok() const {
    // prefill <= floor(1.2 * decode), without overflow or a floating-point boundary.
    return prefill_rejected <= decode_rejected || prefill_rejected - decode_rejected <= decode_rejected / 5;
  }
  bool tokens_ok() const { return prefill_accepted >= decode_accepted; }
  bool passes() const { return routes_ok() && tokens_ok(); }
};

// Distances, rather than cosines: an exact decode reference gives no allowance beyond numerical noise.
inline bool within_twice(double prefill, double decode) {
  return std::isfinite(prefill) && std::isfinite(decode) && prefill >= 0 && decode >= 0 &&
         prefill <= 2 * decode + 1e-9;
}
inline double ratio(double prefill, double decode) {
  return decode > 0 ? prefill / decode : prefill > 0 ? INFINITY : 1.0;
}
struct Cosine {
  double ab = 0, aa = 0, bb = 0;
  void add(double a, double b) {
    ab += a * b;
    aa += a * a;
    bb += b * b;
  }
  double distance() const {
    if (!std::isfinite(ab) || !std::isfinite(aa) || !std::isfinite(bb))
      return std::numeric_limits<double>::quiet_NaN();
    if (aa == 0 && bb == 0) return 0;
    if (aa == 0 || bb == 0) return 1;
    return std::clamp(1 - ab / std::sqrt(aa * bb), 0.0, 2.0);
  }
};
// Unordered top-k sets: the fraction of reference members missing from a row (half its symmetric difference).
inline double set_distance(const uint32_t* a, const uint32_t* reference, size_t k) {
  size_t missing = 0;
  for (size_t i = 0; i < k; ++i) missing += std::find(a, a + k, reference[i]) == a + k;
  return k ? double(missing) / k : 0;
}
inline bool valid_set(const uint32_t* ids, size_t k, uint32_t limit) {
  for (size_t i = 0; i < k; ++i)
    if (ids[i] >= limit || std::find(ids, ids + i, ids[i]) != ids + i) return false;
  return true;
}

}  // namespace q4judge
