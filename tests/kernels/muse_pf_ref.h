#pragma once
// CPU references for spec 23d's Muse-Glimmer prefill attention (src/kernels/muse/mg_pf_attn.cl and the gate in
// mg_pf_prep.cl) - each function with a twin on the device; edit the two together. Built on 23c's
// tests/kernels/muse_ref.h (museref: key_lo, ring_row, eager_block, attention_eager, gate_product), which it includes
// and does not change - so "prefill attends as decode does" is a comparison between two functions of this directory
// (tests/kernels/muse_pf_ref_test.cc), not a claim.
//
// The layouts are the prefill walk's (runtime/muse/muse_prefill.cc):
//   q       bf16 [rows][32][128]       row t of the chunk at c0 sits at absolute position p = c0 + t (mg_attn_prep's
//                                      attn_q holds the same bf16 values widened to fp32)
//   kc, vc  bf16 [slots][2][128]       SLIDING: the 4096-slot ring (key k at slot k & 4095); FULL: the growing NoPE
//                                      cache (key k at row k). The chunk's own rows were written before the attention
//                                      (mg_attn_prep runs over the whole chunk first).
//   out     [rows][32][128]
//
//   visible()          the absolute key range [lo, hi] of the query at `pos`: (pos - 2048, pos] sliding (2048 keys
//                      including the query, MU:93-102), [0, pos] full (causal, no window, no rotary)
//   first_tile()       the first 64-key tile of the 8-row group whose first row sits at p0: lo(p0) & ~63 sliding
//                      (an ABSOLUTE multiple of 64 - Kolibri's rule, kol_pf_attn.cl), 0 full
//   flash_row_fp64()   mg_pf_flash_attn's walk (EAGER 0) in fp64: per 8-row group, 64-key tiles from first_tile up to
//                      the group's last row, the keys read through the ring / the cache, the online softmax with
//                      m_safe = (m == -INF ? 0 : m) - a tile holding none of the row's keys leaves (m, l, o) EXACTLY as
//                      they were, even as the row's first tile. Within a tile the sums run key-ascending; across tiles
//                      the tile order: the sum order is the tile order.
//   flash_window_fp64() every (row, head) of a chunk
//   flash_window_sim() the same walk in the kernel's fp32 arithmetic (score = fp32 dot x 128^-0.5, natural exp, P
//                      rounded to bf16 for P·V, O / l rounded once) but for the dot's and P·V's sum orders (DPAS on
//                      the card): flash's tolerance twin - fp64 at cosine 0.99999 (spec 23 §4.2 MG2)
//   direct_fp64()      the plain softmax over visible() - what the walk must equal (1e-12)
//   eager_window()     mg_pf_flash_attn's EAGER build (B70_MUSE_ATTN=eager): the reference's bf16 chain row by row,
//                      in the kernel's own structure - 128-key tiles, the 8 softmax lanes (i % 8) summed tile after
//                      tile, P·V in eager_block(L)-key blocks crossing tile boundaries - and NO gate (the gate is the
//                      next launch). It must equal museref::attention_eager_head bitwise: rne(o) of decode's eager row
//   gate_window()      mg_pf_attn_gate: out = rne(rne(o) x bf16(sigmoid(g))) in place, decode's mg_attn_reduce chain
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "kernels/muse_ref.h"

namespace musepf {

using museref::bf;
using museref::rf;
using museref::rne;
constexpr uint32_t kHd = museref::kHd, kQH = museref::kQHeads, kKVH = museref::kKvHeads, kGqa = kQH / kKVH,
                   kRing = museref::kRing, kWindow = museref::kWindow;
constexpr uint32_t kKt = 64, kRpw = 8, kC = 2048;   // mg_pf_flash_attn: key tile, rows per sub-group; the chunk
constexpr uint32_t kEagerTile = 128;                 // the eager build's key tile (its work-group: 128 lanes)

// [lo, hi] of the query at `pos` (absolute key positions).
inline std::pair<uint32_t, uint32_t> visible(uint32_t pos, bool sliding) { return {museref::key_lo(pos, sliding), pos}; }
inline uint32_t first_tile(uint32_t p0, bool sliding) { return sliding ? museref::key_lo(p0, true) & ~(kKt - 1) : 0u; }
inline size_t key_at(uint32_t k, bool sliding, uint32_t j) {
  return (size_t(museref::ring_row(k, sliding)) * kKVH + j) * kHd;
}

// One (row t, head h) of the chunk at c0 (C rows) by the walk, in fp64. `start` overrides the first tile (a test:
// starting past a wholly masked tile must change nothing); ~0u = first_tile of the row's group.
inline std::vector<double> flash_row_fp64(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0,
                                          uint32_t C, uint32_t t, bool sliding, uint32_t h, uint32_t start = ~0u) {
  const uint32_t j = h / kGqa, r0 = t / kRpw * kRpw, p = c0 + t, lo = visible(p, sliding).first;
  const uint32_t last = std::min(c0 + C, c0 + r0 + kRpw);
  const uint16_t* qr = q + (size_t(t) * kQH + h) * kHd;
  double m = -INFINITY, l = 0.0;
  std::vector<double> o(kHd, 0.0);
  const double scale = 1.0 / std::sqrt(double(kHd));
  for (uint32_t t0 = start == ~0u ? first_tile(c0 + r0, sliding) : start; t0 < last; t0 += kKt) {
    double s[kKt];
    double tmax = -INFINITY;
    for (uint32_t i = 0; i < kKt; ++i) {
      const uint32_t k = t0 + i;
      if (k < lo || k > p) {
        s[i] = -INFINITY;
        continue;
      }
      double a = 0;
      const size_t row = key_at(k, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) a += double(bf(qr[d])) * bf(kc[row + d]);
      s[i] = a * scale;
      tmax = std::max(tmax, s[i]);
    }
    const double mnew = std::max(m, tmax), msafe = mnew == -INFINITY ? 0.0 : mnew;
    const double corr = std::exp(m - msafe);
    double psum = 0;
    for (uint32_t d = 0; d < kHd; ++d) o[d] *= corr;
    for (uint32_t i = 0; i < kKt; ++i) {
      const double pi = std::exp(s[i] - msafe);
      if (pi == 0.0) continue;
      psum += pi;
      const size_t row = key_at(t0 + i, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) o[d] += pi * bf(vc[row + d]);
    }
    l = l * corr + psum;
    m = mnew;
  }
  for (double& v : o) v /= l;
  return o;
}
inline void flash_window_fp64(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t rows,
                              bool sliding, double* out) {
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t h = 0; h < kQH; ++h) {
      const std::vector<double> r = flash_row_fp64(q, kc, vc, c0, rows, t, sliding, h);
      std::copy(r.begin(), r.end(), out + (size_t(t) * kQH + h) * kHd);
    }
}

// The kernel's fp32 arithmetic for one (row, head): bf16 out.
inline std::vector<uint16_t> flash_row_sim(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0,
                                           uint32_t C, uint32_t t, bool sliding, uint32_t h) {
  const uint32_t j = h / kGqa, r0 = t / kRpw * kRpw, p = c0 + t, lo = visible(p, sliding).first;
  const uint32_t last = std::min(c0 + C, c0 + r0 + kRpw);
  const uint16_t* qr = q + (size_t(t) * kQH + h) * kHd;
  float m = -INFINITY, l = 0.0f;
  std::vector<float> o(kHd, 0.0f);
  for (uint32_t t0 = first_tile(c0 + r0, sliding); t0 < last; t0 += kKt) {
    float s[kKt];
    float tmax = -INFINITY;
    for (uint32_t i = 0; i < kKt; ++i) {
      const uint32_t k = t0 + i;
      if (k < lo || k > p) {
        s[i] = -INFINITY;
        continue;
      }
      float a = 0;
      const size_t row = key_at(k, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) a = std::fma(bf(qr[d]), bf(kc[row + d]), a);
      s[i] = a * museref::kAttnScale;
      tmax = std::max(tmax, s[i]);
    }
    const float mnew = std::max(m, tmax), msafe = mnew == -INFINITY ? 0.0f : mnew;
    const float corr = std::exp(m - msafe);
    float psum = 0;
    for (uint32_t d = 0; d < kHd; ++d) o[d] *= corr;
    for (uint32_t i = 0; i < kKt; ++i) {
      const float pi = std::exp(s[i] - msafe);
      psum += pi;
      const float pb = rf(pi);   // P rounded to bf16: the P·V DPAS operand
      if (pb == 0.0f) continue;
      const size_t row = key_at(t0 + i, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) o[d] = std::fma(pb, bf(vc[row + d]), o[d]);
    }
    l = l * corr + psum;
    m = mnew;
  }
  std::vector<uint16_t> out(kHd);
  for (uint32_t d = 0; d < kHd; ++d) out[d] = rne(o[d] / l);
  return out;
}
inline void flash_window_sim(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t rows,
                             bool sliding, uint16_t* out) {
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t h = 0; h < kQH; ++h) {
      const std::vector<uint16_t> r = flash_row_sim(q, kc, vc, c0, rows, t, sliding, h);
      std::copy(r.begin(), r.end(), out + (size_t(t) * kQH + h) * kHd);
    }
}

// The plain softmax over visible(p), fp64, keys read from per-POSITION arrays (not the ring: this is what proves the
// ring addressing). kpos / vpos: bf16 [positions][2][128].
inline std::vector<double> direct_fp64(const uint16_t* qrow, const uint16_t* kpos, const uint16_t* vpos, uint32_t p,
                                       bool sliding, uint32_t h) {
  const uint32_t j = h / kGqa;
  const auto [lo, hi] = visible(p, sliding);
  std::vector<double> s(hi - lo + 1);
  double mx = -INFINITY;
  for (uint32_t k = lo; k <= hi; ++k) {
    double a = 0;
    for (uint32_t d = 0; d < kHd; ++d) a += double(bf(qrow[d])) * bf(kpos[(size_t(k) * kKVH + j) * kHd + d]);
    s[k - lo] = a / std::sqrt(double(kHd));
    mx = std::max(mx, s[k - lo]);
  }
  double z = 0;
  for (double& v : s) z += (v = std::exp(v - mx));
  std::vector<double> o(kHd, 0.0);
  for (uint32_t k = lo; k <= hi; ++k)
    for (uint32_t d = 0; d < kHd; ++d) o[d] += s[k - lo] / z * bf(vpos[(size_t(k) * kKVH + j) * kHd + d]);
  return o;
}

// The EAGER build's structure for one (row, head): rne(o), no gate. Passes over 128-key tiles from lo:
//   1  s_i = museref's score (one fma chain, d ascending; rf(rf(dot) x scale)) - the row's max
//   2  e_i = exp_torch(s_i - max); lane[i % 8] += e_i, key-ascending inside a tile, tiles ascending (a tile starts at
//      a multiple of 128 from lo, so tile index i' % 8 == i % 8); sum = torch's 8-lane fold; inv = 1 / sum
//   3  p_i = rf(e_i x inv); o = Σ over eager_block(L)-key blocks from i = 0 of (an fma chain from 0), blocks added
//      ascending - a block boundary may fall inside a tile
inline std::vector<uint16_t> eager_row(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t p,
                                       bool sliding, uint32_t h) {
  const uint32_t j = h / kGqa;
  const uint32_t lo = visible(p, sliding).first, L = p - lo + 1, blk = museref::eager_block(L);
  float qf[kHd];
  for (uint32_t d = 0; d < kHd; ++d) qf[d] = bf(q[size_t(h) * kHd + d]);
  auto score = [&](uint32_t i) { return k2_ref::eager_score(qf, kc + key_at(lo + i, sliding, j), kHd); };
  float mx = -INFINITY;
  for (uint32_t i0 = 0; i0 < L; i0 += kEagerTile)
    for (uint32_t i = i0; i < std::min(L, i0 + kEagerTile); ++i) mx = std::max(mx, score(i));
  float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i0 = 0; i0 < L; i0 += kEagerTile) {
    float e[kEagerTile];
    const uint32_t n = std::min(kEagerTile, L - i0);
    for (uint32_t i = 0; i < n; ++i) e[i] = museref::exp_torch(score(i0 + i) - mx);
    for (uint32_t c = 0; c < 8; ++c)
      for (uint32_t i = c; i < n; i += 8) lane[c] += e[i];
  }
  const float sum = ((lane[0] + lane[4]) + (lane[2] + lane[6])) + ((lane[1] + lane[5]) + (lane[3] + lane[7]));
  const float inv = 1.0f / sum;
  std::vector<float> o(kHd, 0.0f), acc(kHd, 0.0f);
  for (uint32_t i0 = 0; i0 < L; i0 += kEagerTile) {
    float pt[kEagerTile];
    const uint32_t n = std::min(kEagerTile, L - i0);
    for (uint32_t i = 0; i < n; ++i) pt[i] = rf(museref::exp_torch(score(i0 + i) - mx) * inv);
    for (uint32_t d = 0; d < kHd; ++d)
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t ii = i0 + i;
        if (ii != 0 && ii % blk == 0) {
          o[d] += acc[d];
          acc[d] = 0.0f;
        }
        acc[d] = std::fma(pt[i], bf(vc[key_at(lo + ii, sliding, j) + d]), acc[d]);
      }
  }
  std::vector<uint16_t> out(kHd);
  for (uint32_t d = 0; d < kHd; ++d) out[d] = rne(o[d] + acc[d]);
  return out;
}
inline void eager_window(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t rows,
                         bool sliding, uint16_t* out) {
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t h = 0; h < kQH; ++h) {
      const std::vector<uint16_t> r = eager_row(q + size_t(t) * kQH * kHd, kc, vc, c0 + t, sliding, h);
      std::copy(r.begin(), r.end(), out + (size_t(t) * kQH + h) * kHd);
    }
}

// mg_pf_attn_gate over `rows` rows in place: attn [rows][4096] bf16, gate [rows][4096] = bf16(sigmoid(g)).
inline void gate_window(uint16_t* attn, const uint16_t* gate, uint32_t rows) {
  for (size_t i = 0; i < size_t(rows) * kQH * kHd; ++i) attn[i] = museref::gate_product(attn[i], gate[i]);
}

}  // namespace musepf
