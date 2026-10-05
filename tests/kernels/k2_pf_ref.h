#pragma once
// CPU references for spec 18c's K2-Horizon prefill kernels (src/kernels/k2/k2_pf_moe.cl,
// k2_pf_linear.cl, k2_pf_attn.cl) - each function with a twin on the device; edit the two
// together. Built on tests/kernels/k2_ref.h (18b's decode references) and pf_moe_ref.h (spec
// 15d's grouped-GEMM references), which it includes and does not change - so "prefill and
// decode combine by one rule" is a comparison between two functions of this directory
// (k2_pf_ref_test.cc), not a claim.
//
//   tmax()           the padded tile count (runtime::k2::pf_tiles is the host's home)
//   sort()           k2_pf_sort: counts, the TM-padded tile table, the sorted rows (expert-major,
//                    ascending token within an expert, the shared expert last iff `shared`),
//                    pair_row, the header
//   gather()         k2_pf_gather: row r of the sorted A operand
//   dequant_slab()   k2_pf_dequant_slab: a layout-0 linear's [K][ns] bf16 slab from column n0,
//                    the columns past N zero
//   grouped()        pf_moe_gemm's tile walk (pf_moe_ref::grouped over this file's Sorted)
//   moe_combine()    k2_pf_moe_combine for one (token, column): k2_moe_down's epilogue
//   mova_combine()   k2_pf_mova_combine: k2_mova_value's epilogue
//   attention()      one (row, head)'s causal attention in fp64 (the K1 reference)
//   attention_eager() the reference's rounding points in fp32 (k2_pf_attn.cl EAGER's model)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"
#include "kernels/pf_moe_ref.h"

namespace k2_pf_ref {

using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;
constexpr uint32_t kTm = 32;
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kHdrTiles = 0, kHdrSharedRow = 1, kHdrRows = 2, kHdrC = 3, kHdrCount = 4;
inline uint32_t hdr_words(uint32_t experts) { return (4 + experts + 1 + 15) / 16 * 16; }

inline uint32_t tmax(uint32_t E, uint32_t K, bool shared, uint32_t C) {
  return (C * K + E * (kTm - 1)) / kTm + (shared ? (C + kTm - 1) / kTm : 0);
}

struct Sorted {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  uint32_t rows_used() const { return hdr[kHdrRows]; }
  uint32_t shared_row() const { return hdr[kHdrSharedRow]; }
};

// k2_pf_sort over route rows [C][k2_ref::kWords] (ids at kIds; clamped to E - 1 as the
// kernel's byte staging does). The shared expert is block E.
inline Sorted sort(const uint32_t* route, uint32_t C, uint32_t E, uint32_t K, bool shared) {
  const uint32_t T = tmax(E, K, shared, C);
  std::vector<uint32_t> id(size_t(C) * K);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j)
      id[size_t(t) * K + j] = std::min(route[size_t(t) * k2_ref::kWords + k2_ref::kIds + j], E - 1);
  std::vector<uint32_t> cnt(E, 0), toff(E + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < E; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ts = toff[E], st = shared ? (C + kTm - 1) / kTm : 0, ntiles = ts + st, rs = ts * kTm;
  Sorted o;
  o.hdr.assign(hdr_words(E), 0);
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
    o.tiles[2 * (ts + i)] = E;
    o.tiles[2 * (ts + i) + 1] = rs + i * kTm;
  }
  for (uint32_t i = 0; i < st * kTm; ++i) o.row_tok[rs + i] = i < C ? i : kNone;
  o.hdr[kHdrTiles] = ntiles;
  o.hdr[kHdrSharedRow] = shared ? rs : kNone;
  o.hdr[kHdrRows] = ntiles * kTm;
  o.hdr[kHdrC] = C;
  for (uint32_t e = 0; e < E; ++e) o.hdr[kHdrCount + e] = cnt[e];
  o.hdr[kHdrCount + E] = shared ? C : 0;
  return o;
}

// Row r of the sorted A operand: x row row_tok[r] (bf16, `width` per row), zeros for padding.
inline std::vector<uint16_t> gather(const Sorted& st, const uint16_t* x, uint32_t width) {
  std::vector<uint16_t> g(size_t(st.rows_used()) * width, 0);
  for (uint32_t r = 0; r < st.rows_used(); ++r)
    if (st.row_tok[r] != kNone)
      std::copy(x + size_t(st.row_tok[r]) * width, x + size_t(st.row_tok[r] + 1) * width,
                g.begin() + size_t(r) * width);
  return g;
}

// pf_moe_gemm's walk over this Sorted: every row of every live tile with a block in [b0, b1).
template <class F>
void grouped(const Sorted& st, uint32_t b0, uint32_t b1, F f) {
  const uint32_t T = uint32_t(st.tiles.size() / 2);
  for (uint32_t t = 0; t < T; ++t) {
    const uint32_t b = st.tiles[2 * t];
    if (b == kNone || b < b0 || b >= b1) continue;
    for (uint32_t i = 0; i < kTm; ++i) f(b, st.tiles[2 * t + 1] + i);
  }
}

// k2_pf_dequant_slab: columns [n0, n0 + ns) of a layout-0 K x N int4 linear (words
// [K/8][N] u32, scales [K/64][N] f16 bits) as bf16 [K][ns]; columns >= N are zero.
inline std::vector<uint16_t> dequant_slab(const uint32_t* words, const uint16_t* scales, uint32_t K,
                                          uint32_t N, uint32_t n0, uint32_t ns) {
  std::vector<uint16_t> out(size_t(K) * ns, 0);
  for (uint32_t c = 0; c < ns; ++c) {
    const uint32_t n = n0 + c;
    if (n >= N) continue;
    for (uint32_t g = 0; g < K / 64; ++g) {
      const float scale = common::f16_to_f32(scales[size_t(g) * N + n]);
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t word = words[size_t(g * 8 + j) * N + n];
        for (uint32_t i = 0; i < 8; ++i) {
          const int q = int((word >> (4 * i)) & 0xFu) - 8;
          out[size_t(g * 64 + j * 8 + i) * ns + c] = rne(float(q) * scale);
        }
      }
    }
  }
  return out;
}

// k2_pf_moe_combine for token t, column n: `route_row` the token's k2_route row, y the
// rne'd down outputs per sorted row (pitch `hidden`), resid the column's residual in.
inline uint16_t moe_combine(const Sorted& st, const uint32_t* route_row, const uint16_t* y,
                            uint32_t hidden, uint32_t K, uint32_t t, uint32_t n, uint16_t resid) {
  float acc = 0.0f;
  for (uint32_t j = 0; j < K; ++j) {
    float w;
    std::memcpy(&w, &route_row[k2_ref::kWeights + j], 4);
    const uint16_t d_b = y[size_t(st.pair_row[size_t(t) * K + j]) * hidden + n];
    const uint16_t t_b = rne(f32(d_b) * w);
    acc = rf(acc + f32(t_b));
  }
  const uint16_t sh_b = y[size_t(st.shared_row() + t) * hidden + n];
  const uint16_t o_b = rne(acc + f32(sh_b));
  return rne(f32(resid) + f32(o_b));
}

// k2_pf_mova_combine for token t, column n: y the rne'd value-expert outputs (pitch vn).
inline uint16_t mova_combine(const Sorted& st, const uint32_t* route_row, const uint16_t* y,
                             uint32_t vn, uint32_t K, uint32_t t, uint32_t n) {
  float acc = 0.0f;
  for (uint32_t j = 0; j < K; ++j) {
    float w;
    std::memcpy(&w, &route_row[k2_ref::kWeights + j], 4);
    const uint16_t v_b = y[size_t(st.pair_row[size_t(t) * K + j]) * vn + n];
    const uint16_t a_b = rne(k2_ref::silu_f32(f32(v_b)));
    const uint16_t t_b = rne(f32(a_b) * w);
    acc = rf(acc + f32(t_b));
  }
  return rne(acc);
}

// --- the attention ------------------------------------------------------------------------
// q bf16-valued fp32 [hd] of the row at absolute position p, keys 0..p of kv head j in a
// [len][kvh][hd] bf16 cache. fp64 throughout (the K1 reference, spec 6's).
inline std::vector<double> attention(const float* q, const uint16_t* kc, const uint16_t* vc,
                                     uint32_t p, uint32_t j, uint32_t kvh, uint32_t hd) {
  return k2_ref::attention(q, kc, vc, p + 1, j, kvh, hd);
}

// The reference's rounding points (tools/oracle/k2_ref.py attention(): s_b = rne(q.k),
// s = rne(s_b x 1/sqrt(hd)), p = rne(softmax(s)) with the softmax in fp32, o = rne(Σ p v)) in
// fp32 with ascending sums: what k2_pf_attn.cl EAGER moves the rounding points to. Not the
// device's order (DPAS, tiles): a tolerance reference with the reference's ROUNDING.
inline std::vector<float> attention_eager(const float* q, const uint16_t* kc, const uint16_t* vc,
                                          uint32_t p, uint32_t j, uint32_t kvh, uint32_t hd) {
  const float scale = float(1.0 / std::sqrt(double(hd)));
  std::vector<float> s(p + 1);
  float mx = -INFINITY;
  for (uint32_t t = 0; t <= p; ++t) {
    float a = 0.0f;
    for (uint32_t d = 0; d < hd; ++d) a += q[d] * f32(kc[(size_t(t) * kvh + j) * hd + d]);
    s[t] = rf(rf(a) * scale);
    mx = std::max(mx, s[t]);
  }
  float z = 0.0f;
  for (uint32_t t = 0; t <= p; ++t) z += std::exp(s[t] - mx);
  std::vector<float> o(hd, 0.0f);
  for (uint32_t t = 0; t <= p; ++t) {
    const float pb = rf(std::exp(s[t] - mx) / z);
    for (uint32_t d = 0; d < hd; ++d) o[d] += pb * f32(vc[(size_t(t) * kvh + j) * hd + d]);
  }
  for (float& v : o) v = rf(v);
  return o;
}

}  // namespace k2_pf_ref
