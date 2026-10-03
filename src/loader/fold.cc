#include "loader/fold.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace loader {
namespace {

constexpr uint32_t kGroup = 64;

void check(const PackedInt4View& v, const char* what) {
  if (v.K == 0 || v.N == 0 || v.K % kGroup != 0 || !v.qweight || !v.scales)
    throw std::runtime_error(std::string("loader::fold: ") + what + " is not a packed g64 linear (K=" +
                             std::to_string(v.K) + ", N=" + std::to_string(v.N) + ")");
}

}  // namespace

PackedInt4 fold_n(const PackedInt4View& a, const PackedInt4View& b) {
  check(a, "fold_n's first part");
  check(b, "fold_n's second part");
  if (a.K != b.K)
    throw std::runtime_error("loader::fold_n: K " + std::to_string(a.K) + " != " +
                             std::to_string(b.K) + " - a column join needs one input width");
  PackedInt4 out;
  out.K = a.K;
  out.N = a.N + b.N;
  out.qweight.resize(size_t(out.K / 8) * out.N);
  out.scales.resize(size_t(out.K / kGroup) * out.N);
  // Row r of the result is row r of a, then row r of b - for both arrays.
  for (uint32_t r = 0; r < out.K / 8; ++r) {
    uint32_t* dst = out.qweight.data() + size_t(r) * out.N;
    std::copy_n(a.qweight + size_t(r) * a.N, a.N, dst);
    std::copy_n(b.qweight + size_t(r) * b.N, b.N, dst + a.N);
  }
  for (uint32_t g = 0; g < out.K / kGroup; ++g) {
    uint16_t* dst = out.scales.data() + size_t(g) * out.N;
    std::copy_n(a.scales + size_t(g) * a.N, a.N, dst);
    std::copy_n(b.scales + size_t(g) * b.N, b.N, dst + a.N);
  }
  return out;
}

PackedInt4 fold_k(const PackedInt4View& a, const PackedInt4View& b) {
  check(a, "fold_k's first part");   // a.K % 64 == 0: the join is a group boundary
  check(b, "fold_k's second part");
  if (a.N != b.N)
    throw std::runtime_error("loader::fold_k: N " + std::to_string(a.N) + " != " +
                             std::to_string(b.N) + " - a row join needs one output width");
  PackedInt4 out;
  out.K = a.K + b.K;
  out.N = a.N;
  const size_t qa = size_t(a.K / 8) * a.N, qb = size_t(b.K / 8) * b.N;
  const size_t sa = size_t(a.K / kGroup) * a.N, sb = size_t(b.K / kGroup) * b.N;
  out.qweight.resize(qa + qb);
  out.scales.resize(sa + sb);
  std::copy_n(a.qweight, qa, out.qweight.data());
  std::copy_n(b.qweight, qb, out.qweight.data() + qa);   // at qweight row a.K / 8
  std::copy_n(a.scales, sa, out.scales.data());
  std::copy_n(b.scales, sb, out.scales.data() + sa);     // at scales row a.K / 64
  return out;
}

}  // namespace loader
