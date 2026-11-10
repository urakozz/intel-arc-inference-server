#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bolt_judge {
inline bool near_cut(float gap, float cut) {
  int e = 0;
  std::frexp(std::fabs(cut), &e);
  return std::isfinite(gap) && std::isfinite(cut) && gap >= 0 && double(gap) <= std::ldexp(1.0, e - 8);
}
template <class A, class B>
bool same_set(const A* a, const B* b, uint32_t k) {
  for (uint32_t i = 0; i < k; ++i)
    if (std::find(b, b + k, int32_t(a[i])) == b + k) return false;
  return true;
}
struct Counts {
  uint64_t pairs = 0, engine = 0, golden = 0, near = 0;
  void add(const uint32_t* e, const int32_t* b, const int32_t* f, uint32_t k, float gap, float cut) {
    ++pairs;
    engine += !same_set(e, f, k);
    golden += !same_set(b, f, k);
    near += near_cut(gap, cut);
  }
  bool ok() const { return engine <= golden; }
};
inline bool state_ok(double engine, double baseline) {
  return std::isfinite(engine) && std::isfinite(baseline) && engine >= baseline;
}
struct Cosines {
  double ee = 0, bb = 0, ff = 0, ef = 0, bf = 0;
  void add(double e, double b, double f) {
    ee += e * e;
    bb += b * b;
    ff += f * f;
    ef += e * f;
    bf += b * f;
  }
  static double cosine(double dot, double a, double b) {
    return a == 0 && b == 0 ? 1.0 : dot / std::sqrt(a * b + 1e-300);
  }
  double engine() const { return cosine(ef, ee, ff); }
  double baseline() const { return cosine(bf, bb, ff); }
  bool ok() const { return state_ok(engine(), baseline()); }
};
}  // namespace bolt_judge
