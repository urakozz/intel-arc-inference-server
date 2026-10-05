#pragma once
// CPU reference for spec 15d's prefill MoE block - src/kernels/prefill/pf_moe.cl and
// pf_moe_gemm.cl - every function with a twin in a kernel; the two must be edited
// together. Built on tests/kernels/moe_ref.h (spec 15c's decode reference), which it
// includes and does not change: the route row, the shape, rne / f32 and the decode
// combine are moe_ref's, so "prefill and decode combine by one rule" is a comparison
// between two functions of this directory (pf_moe_ref_test.cc), not a claim.
//
//   tmax()          the padded tile count of a C-row chunk (runtime::moe_prefill_tiles's
//                   formula, which src/runtime/buffer_sizes.cc owns for the host)
//   sort()          pf_moe_sort: per-expert counts, the TM-padded tile table, the sorted
//                   rows (expert-major, ascending token within an expert, the shared expert
//                   last with its C tokens in order), pair_row, the header
//   gather()        pf_moe_gather / _i8: row r of the sorted A operand
//   dequant_block() pf_moe_dequant_*: one layout-1 int4 block as bf16 [K][N]
//   gemm_rows()     the grouped GEMM's arithmetic for ONE row against one expert block,
//                   fp32 in ascending k (the device's DPAS order differs: a tolerance)
//   grouped()       pf_moe_gemm's tile walk over the tile table: f(block, row) for every
//                   row of every live tile in [b0, b1)
//   combine_token() pf_moe_combine for one token and column: moe_down's epilogue over the
//                   sorted rows pair_row names
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "kernels/prefill/pf_kernels.h"
#include "moe_ref.h"

namespace pf_moe_ref {

using moe_ref::f32;
using moe_ref::rne;
using moe_ref::Shape;
namespace k = kernels::pf_moe;
constexpr uint32_t kTm = k::kTileM;
constexpr uint32_t kNone = k::kNone;

// Σ_e ceil(c_e / TM) <= (C x top_k + experts x (TM - 1)) / TM, plus the shared expert's
// ceil(C / TM) tiles.
inline uint32_t tmax(const Shape& s, uint32_t C) {
  return (C * s.top_k + s.experts * (kTm - 1)) / kTm + (C + kTm - 1) / kTm;
}

struct Sorted {
  std::vector<uint32_t> hdr;        // k::hdr_words(experts)
  std::vector<uint32_t> tiles;      // [tmax][2]: (block, first row); padding (kNone, 0)
  std::vector<uint32_t> row_tok;    // [tmax x TM]: the token of each sorted row, kNone for
                                    // padding; rows past hdr[kHdrRows] are kNone here and
                                    // untouched on the device
  std::vector<uint32_t> pair_row;   // [C][top_k]: the sorted row of (token, slot)
  uint32_t tiles_used() const { return hdr[k::kHdrTiles]; }
  uint32_t rows_used() const { return hdr[k::kHdrRows]; }
  uint32_t shared_row() const { return hdr[k::kHdrSharedRow]; }
};

// pf_moe_sort over the route rows [C][moe_ref::kWords] (ids at kIds; ids past the last
// expert clamp to it, as the kernel's byte staging does).
inline Sorted sort(const uint32_t* route, uint32_t C, const Shape& s) {
  const uint32_t E = s.experts, K = s.top_k, T = tmax(s, C);
  std::vector<uint32_t> id(size_t(C) * K);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j)
      id[size_t(t) * K + j] = std::min(route[size_t(t) * moe_ref::kWords + moe_ref::kIds + j], E - 1);
  std::vector<uint32_t> cnt(E, 0), toff(E + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < E; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ts = toff[E], st = (C + kTm - 1) / kTm, ntiles = ts + st, rs = ts * kTm;
  Sorted o;
  o.hdr.assign(k::hdr_words(E), 0);
  o.tiles.assign(size_t(T) * 2, 0);
  o.row_tok.assign(size_t(T) * kTm, kNone);
  o.pair_row.assign(size_t(C) * K, kNone);
  for (uint32_t t = 0; t < T; ++t) o.tiles[2 * t] = kNone;
  for (uint32_t e = 0; e < E; ++e) {
    const uint32_t t0 = toff[e], r0 = t0 * kTm;
    for (uint32_t i = 0; i < toff[e + 1] - t0; ++i) {
      o.tiles[2 * (t0 + i)] = e;
      o.tiles[2 * (t0 + i) + 1] = r0 + i * kTm;
    }
    uint32_t p = 0;
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < K; ++j)
        if (id[size_t(t) * K + j] == e) {
          o.row_tok[r0 + p] = t;
          o.pair_row[size_t(t) * K + j] = r0 + p;
          ++p;
        }
  }
  for (uint32_t i = 0; i < st; ++i) {
    o.tiles[2 * (ts + i)] = s.shared_block();
    o.tiles[2 * (ts + i) + 1] = rs + i * kTm;
  }
  for (uint32_t i = 0; i < st * kTm; ++i) o.row_tok[rs + i] = i < C ? i : kNone;
  o.hdr[k::kHdrTiles] = ntiles;
  o.hdr[k::kHdrSharedRow] = rs;
  o.hdr[k::kHdrRows] = ntiles * kTm;
  o.hdr[k::kHdrC] = C;
  for (uint32_t e = 0; e < E; ++e) o.hdr[k::kHdrCount + e] = cnt[e];
  o.hdr[k::kHdrCount + E] = C;
  return o;
}

// Row r of the sorted A operand: x row row_tok[r], or zeros for a padding row. `width`
// elements of T per row (bf16 hidden, or int8 hidden bytes).
template <class T>
std::vector<T> gather(const Sorted& st, const T* x, uint32_t width) {
  std::vector<T> g(size_t(st.rows_used()) * width, T(0));
  for (uint32_t r = 0; r < st.rows_used(); ++r)
    if (st.row_tok[r] != kNone)
      std::copy(x + size_t(st.row_tok[r]) * width, x + size_t(st.row_tok[r] + 1) * width,
                g.begin() + size_t(r) * width);
  return g;
}

// One layout-1 int4 block (K x N, loader/moe_layout.h) as bf16 [K][N]: pf_dequant_slab's
// rne((q - 8) x scale).
inline std::vector<uint16_t> dequant_block(const uint32_t* blk, uint32_t K, uint32_t N) {
  std::vector<uint16_t> out(size_t(K) * N);
  const uint32_t G = K / 64;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t* tile = blk + (size_t(n / 16) * G + g) * 136;
      const uint32_t sw = tile[128 + (n % 16) / 2];
      const float scale = common::f16_to_f32(uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16));
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t word = tile[j * 16 + n % 16];
        for (uint32_t i = 0; i < 8; ++i) {
          const int q = int((word >> (4 * i)) & 0xFu) - 8;
          out[size_t(g * 64 + j * 8 + i) * N + n] = rne(float(q) * scale);
        }
      }
    }
  return out;
}

// One row a [K] (bf16) against a bf16 [K][N] block: fp32 sums in ascending k (each
// bf16 x bf16 product is exact in fp32).
inline std::vector<float> gemm_rows(const uint16_t* a, const uint16_t* b, uint32_t K, uint32_t N) {
  std::vector<float> o(N, 0.0f);
  for (uint32_t kk = 0; kk < K; ++kk) {
    const float av = f32(a[kk]);
    if (av == 0.0f) continue;
    for (uint32_t n = 0; n < N; ++n) o[n] += av * f32(b[size_t(kk) * N + n]);
  }
  return o;
}

// gate||up's SiLU epilogue over one row's N = 2 x inter interleaved sums (moe_ref::silu_up
// per intermediate column): bf16 [inter].
inline std::vector<uint16_t> silu_row(const std::vector<float>& gu, uint32_t inter) {
  std::vector<uint16_t> h(inter);
  for (uint32_t i = 0; i < inter; ++i)
    h[i] = moe_ref::silu_up(gu[moe_ref::gate_col(i)], gu[moe_ref::up_col(i)]);
  return h;
}

// pf_moe_gemm's walk: every row of every live tile whose block is in [b0, b1), in tile
// order, f(block, row). The device runs the same (tile, row) set, each work-group one tile.
template <class F>
void grouped(const Sorted& st, uint32_t b0, uint32_t b1, F f) {
  const uint32_t T = uint32_t(st.tiles.size() / 2);
  for (uint32_t t = 0; t < T; ++t) {
    const uint32_t b = st.tiles[2 * t];
    if (b == kNone || b < b0 || b >= b1) continue;
    for (uint32_t i = 0; i < kTm; ++i) f(b, st.tiles[2 * t + 1] + i);
  }
}

// pf_moe_combine for token t, column n: y the bf16 [rows][hidden] down outputs (rne'd),
// `route` the token's route row, resid the column's residual in.
inline uint16_t combine_token(const Sorted& st, const uint32_t* route_row, const uint16_t* y,
                              uint32_t hidden, const Shape& s, uint32_t t, uint32_t n,
                              uint16_t resid) {
  float sum = 0.0f;
  for (uint32_t j = 0; j < s.top_k; ++j) {
    float w;
    std::memcpy(&w, &route_row[moe_ref::kWeights + j], 4);
    const uint16_t d_b = y[size_t(st.pair_row[size_t(t) * s.top_k + j]) * hidden + n];
    sum += f32(rne(f32(d_b) * w));
  }
  float sg;
  std::memcpy(&sg, &route_row[moe_ref::kSharedGate], 4);
  const uint16_t r_b = rne(sum);
  const uint16_t ds_b = y[size_t(st.shared_row() + t) * hidden + n];
  const uint16_t sh_b = rne(f32(ds_b) * sg);
  const uint16_t o_b = rne(f32(r_b) + f32(sh_b));
  return rne(f32(resid) + f32(o_b));
}

}  // namespace pf_moe_ref
