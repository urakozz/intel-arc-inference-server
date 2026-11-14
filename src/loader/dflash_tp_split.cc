#include "loader/dflash_tp_split.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/dflash_dims.h"

namespace loader {
namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("loader::dflash_tp_split: " + what);
}

// A form's tile: its k rows, its elements and an element's bytes (dflash_repack.h DFlashHostLinear).
struct Tile {
  uint32_t kt, elems, bytes;
};
Tile tile_of(const DFlashHostLinear& h) {
  switch (h.kind) {
    case model::WeightKind::Bf16: return {8, 128, 2};
    case model::WeightKind::Int8: return {16, 256, 1};
    case model::WeightKind::Int4: return {64, 136, 4};
    default: break;
  }
  require(false, "a linear that is not bf16, int8 or int4 g64");
  return {};
}
const uint8_t* data_of(const DFlashHostLinear& h) {
  return h.kind == model::WeightKind::Bf16   ? reinterpret_cast<const uint8_t*>(h.bf16.data())
         : h.kind == model::WeightKind::Int8 ? reinterpret_cast<const uint8_t*>(h.i8.data())
                                             : reinterpret_cast<const uint8_t*>(h.int4.data());
}
DFlashHostLinear empty_like(const DFlashHostLinear& h, uint32_t K, uint32_t N) {
  require(h.gs.empty(), "an NVFP4 output scale (the LiLiCorr drafter's) has no TP form");
  DFlashHostLinear o;
  o.kind = h.kind;
  o.K = K;
  o.N = N;
  const Tile t = tile_of(h);
  const size_t n = size_t(N / 16) * (K / t.kt) * t.elems;
  if (h.kind == model::WeightKind::Bf16) o.bf16.resize(n);
  if (h.kind == model::WeightKind::Int8) o.i8.resize(n);
  if (h.kind == model::WeightKind::Int4) o.int4.resize(n);
  o.slices = h.kind == model::WeightKind::Int4 ? kernels::dflash::int4_slices(K, N) : 1;
  return o;
}
uint8_t* mut_data(DFlashHostLinear& h) { return const_cast<uint8_t*>(data_of(h)); }

}  // namespace

DFlashHostLinear dflash_take_outputs(const DFlashHostLinear& h, const std::vector<std::array<uint32_t, 2>>& ranges) {
  uint32_t N = 0;
  for (const auto& r : ranges) {
    require(r[0] < r[1] && r[1] <= h.N && r[0] % 16 == 0 && r[1] % 16 == 0,
            "output rows [" + std::to_string(r[0]) + ", " + std::to_string(r[1]) + ") of " + std::to_string(h.N) +
                " are not whole 16-row tiles");
    N += r[1] - r[0];
  }
  DFlashHostLinear o = empty_like(h, h.K, N);
  const Tile t = tile_of(h);
  const size_t row_tile = size_t(h.K / t.kt) * t.elems * t.bytes;   // one 16-row tile's bytes
  uint8_t* dst = mut_data(o);
  for (const auto& r : ranges) {
    std::memcpy(dst, data_of(h) + size_t(r[0] / 16) * row_tile, size_t(r[1] - r[0]) / 16 * row_tile);
    dst += size_t(r[1] - r[0]) / 16 * row_tile;
    if (h.kind == model::WeightKind::Int8) o.scales.insert(o.scales.end(), h.scales.begin() + r[0], h.scales.begin() + r[1]);
  }
  return o;
}

DFlashHostLinear dflash_take_inputs(const DFlashHostLinear& h, uint32_t k0, uint32_t k1) {
  const Tile t = tile_of(h);
  require(k0 < k1 && k1 <= h.K && k0 % t.kt == 0 && k1 % t.kt == 0,
          "input rows [" + std::to_string(k0) + ", " + std::to_string(k1) + ") of " + std::to_string(h.K) +
              " are not whole " + std::to_string(t.kt) + "-row k-tiles");
  DFlashHostLinear o = empty_like(h, k1 - k0, h.N);
  const size_t tile = size_t(t.elems) * t.bytes, src_row = size_t(h.K / t.kt) * tile, dst_row = size_t(o.K / t.kt) * tile;
  for (uint32_t n = 0; n < h.N / 16; ++n)
    std::memcpy(mut_data(o) + n * dst_row, data_of(h) + n * src_row + size_t(k0 / t.kt) * tile, dst_row);
  o.scales = h.scales;   // int8: a row's scale is the whole row's, so the two partial sums add to the one-card sum
  return o;
}

DFlashHostLinear dflash_tp_layer_linear(const model::DFlashDesc& d, DFlashLinear which, uint32_t rank,
                                        const DFlashHostLinear& whole) {
  require(rank < 2, "rank " + std::to_string(rank));
  const uint32_t qn = d.q_n() / 2, kvn = d.kv_n() / 2, in = d.inter / 2;
  switch (which) {
    case DFlashLinear::Qkv:
      return dflash_take_outputs(whole, {{rank * qn, (rank + 1) * qn},
                                         {d.q_n() + rank * kvn, d.q_n() + (rank + 1) * kvn},
                                         {d.q_n() + d.kv_n() + rank * kvn, d.q_n() + d.kv_n() + (rank + 1) * kvn}});
    case DFlashLinear::GateUp:
      return dflash_take_outputs(whole, {{rank * in, (rank + 1) * in}, {d.inter + rank * in, d.inter + (rank + 1) * in}});
    case DFlashLinear::O: return dflash_take_inputs(whole, rank * qn, (rank + 1) * qn);
    case DFlashLinear::Down: return dflash_take_inputs(whole, rank * in, (rank + 1) * in);
    case DFlashLinear::AttnKproj:
    case DFlashLinear::MlpKproj: return whole;
  }
  return whole;
}

}  // namespace loader
