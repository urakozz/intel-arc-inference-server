#include "loader/ab.h"

#include <stdexcept>

#include "common/bf16.h"

namespace loader {

const model::FusedLinear& ab_row_for(const model::ModelDesc& d, WKind in_proj_a_kind) {
  return in_proj_a_kind == WKind::Int4 ? d.ab(model::WeightKind::Int4)
                                       : d.linear(model::LinearId::AB);
}

std::vector<common::ColSource> int4_row_cols(const model::FusedLinear& fl,
                                             const std::vector<LinearSrc>& srcs, Int4Pad& pad,
                                             const std::string& id) {
  const model::GemvShape& sh = fl.shape;
  std::vector<common::Part> parts;
  uint32_t n_sum = 0;
  for (const LinearSrc& s : srcs) {
    if (s.kind != WKind::Int4) throw std::runtime_error(id + ": part '" + s.name + "' is not int4");
    parts.push_back({s.qweight, s.scales, s.N});
    n_sum += s.N;
  }
  std::vector<common::ColSource> cols;
  if (fl.fuse == model::Fuse::Interleave16) {
    if (parts.size() != 2 || parts[0].N != parts[1].N || parts[0].N % 16 != 0)
      throw std::runtime_error(id + ": interleave16 needs two parts of equal, 16-divisible N");
    if (n_sum != sh.N)
      throw std::runtime_error(id + ": an interleaved int4 row is never padded (" +
                               std::to_string(n_sum) + " -> " + std::to_string(sh.N) + ")");
    cols = common::cols_interleave16(parts[0], parts[1]);
  } else {
    if (n_sum > sh.N)
      throw std::runtime_error(id + ": parts sum to N=" + std::to_string(n_sum) +
                               ", more than the row's " + std::to_string(sh.N));
    if (n_sum != sh.N) {
      // The zero columns (see ab.h): a part of their own, appended after the real ones.
      pad.N = sh.N - n_sum;
      pad.qweight.assign(size_t(sh.K / 8) * pad.N, kInt4ZeroWord);
      pad.scales.assign(size_t(sh.K / 64) * pad.N, 0);
      parts.push_back({pad.qweight.data(), pad.scales.data(), pad.N});
    }
    cols = common::cols_concat(parts);
  }
  if (cols.size() != sh.N)
    throw std::runtime_error(id + ": column map has " + std::to_string(cols.size()) +
                             " entries, need " + std::to_string(sh.N));
  return cols;
}

void dequant_int4_bf16_tiled(const model::GemvShape& sh, const std::vector<common::ColSource>& cols,
                             uint16_t* out) {
  if (cols.size() != sh.N || sh.K % 64 != 0 || sh.N % 16 != 0)
    throw std::logic_error("dequant_int4_bf16_tiled: " + std::to_string(cols.size()) +
                           " columns for a K=" + std::to_string(sh.K) + " N=" +
                           std::to_string(sh.N) + " row");
  std::vector<uint16_t> rows(size_t(sh.N) * sh.K);
  for (uint32_t n = 0; n < sh.N; ++n) {
    const common::ColSource& c = cols[n];
    uint16_t* row = rows.data() + size_t(n) * sh.K;
    for (uint32_t k = 0; k < sh.K; ++k) {
      const uint32_t word = c.qweight[size_t(k / 8) * c.n_part + c.n];
      const int q = int((word >> (4 * (k % 8))) & 0xFu);
      const float s = common::f16_to_f32(c.scales[size_t(k / 64) * c.n_part + c.n]);
      // pf_dequant_slab.cl: rne_bf16((float)qm8 * scale) - exact product, one RNE.
      row[k] = common::f32_to_bf16(float(q - 8) * s);
    }
  }
  common::repack_bf16_tiled(rows.data(), sh.K, sh.N, out);
}

}  // namespace loader
