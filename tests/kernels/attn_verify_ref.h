#pragma once
// attn_verify_ref.h - spec 28c's host reference for the verify step's attention at M rows
// (plan 28c Task 1): what `attn_verify_dpas` + `attn_reduce_v2` (or `_kv8`) must compute, in fp64,
// from the inputs today's v2 pair reads:
//
//   q     fp32 [M][24][256]   attn_q as attn_prep leaves it (bf16 KV), or rotated as attn_prep_kv8
//                             leaves it (int8 KV) - the values the kernel reads, used as is
//   gate  fp32 [M][24][256]   attn_gate
//   K, V  bf16 [keys][4][256] the cache rows, or int8 + fp16 scales (kv8_ref::Cache, the writer's
//                             bitwise host twin src/common/kv8.h), dequantised exactly
//
// Row m of a launch at `pos` attends to keys [0, pos + m] with scale 1/16 (q head h reads kv head
// h / 6). The output is the pre-gate attention in fp64 (`Result::o`; under int8 un-rotated by
// kv8_ref's fp64 R^T first) and the gated bf16 row exactly as attn_reduce_v2 forms it from an fp32
// o: rne(f32(rne(o)) sigmoid_f32(g)) (`Result::out`, [M][6144]).
//
// The block view (`row_blocks`) is v2's blocking, which attn_verify_dpas writes its partials in:
// row m's key count L = pos + m + 1 walks blocks of ppw(L) = max(64, roundup64(ceil(L / 32))) keys,
// block b = [b ppw, min((b + 1) ppw, L)), nb = (pos + m) / ppw(L) + 1 <= 32 of them (attn_v2.cl's
// v2_ppw and attn_reduce_v2's nb, TGT = 32). A block's partial means what v2's does: mx = the max of
// its scaled scores (natural log domain), sm = sum exp(s - mx), acc = sum exp(s - mx) v - V in the
// cache's basis (rotated under int8: the reduce un-rotates). `merge` folds blocks in ascending order
// with attn_reduce_v2's formulas (in fp64), giving the cache-basis output back.
//
// Grades: this is the N1 bar's reference (spec 28c §4: cosine >= 0.99999 per (q head, row)); the
// kernels' `exp` and summation orders are not the host's, so they are held to it by cosine.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/bf16.h"
#include "kernels/kv8_ref.h"

namespace attn_verify_ref {

constexpr uint32_t kQH = 24, kKvH = 4, kGqa = 6, kHD = 256;
constexpr uint32_t kTgt = 32;      // runtime::DecodeScratch::kAttnV2Blocks, attn_v2.cl's TGT
constexpr uint32_t kPart = 258;    // {mx, sm, acc[256]} per (q head, block, row)
constexpr uint32_t kOutN = kQH * kHD;
constexpr double kScale = 1.0 / 16.0;

struct Range { uint32_t lo, hi; };

// attn_v2.cl's v2_ppw: positions per work-group for a row attending to `len` keys.
inline uint32_t ppw(uint32_t len) {
  uint32_t p = (len + kTgt - 1) / kTgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}
// attn_reduce_v2's block count for row m of a launch at pos.
inline uint32_t nblocks(uint32_t pos, uint32_t m) { return (pos + m) / ppw(pos + m + 1) + 1; }
// Row m's v2 blocks: [b s, min((b + 1) s, L)), b < nb.
inline std::vector<Range> blocks(uint32_t pos, uint32_t m) {
  const uint32_t L = pos + m + 1, s = ppw(L), nb = nblocks(pos, m);
  std::vector<Range> r;
  for (uint32_t b = 0; b < nb; ++b) r.push_back({b * s, std::min((b + 1) * s, L)});
  return r;
}

// The cache: bf16 rows, or kv8_ref's int8 rows + scales. Values in the cache's basis.
struct Kv {
  const uint16_t* k16 = nullptr;
  const uint16_t* v16 = nullptr;
  const kv8_ref::Cache* c8 = nullptr;
  static Kv bf16(const uint16_t* k, const uint16_t* v) { return Kv{k, v, nullptr}; }
  static Kv int8(const kv8_ref::Cache& c) { return Kv{nullptr, nullptr, &c}; }
  bool kv8() const { return c8 != nullptr; }
  double k(uint32_t p, uint32_t j, uint32_t i) const {
    const size_t row = size_t(p) * kKvH + j;
    return c8 ? double(c8->kd(row, i)) : double(common::bf16_to_f32(k16[row * kHD + i]));
  }
  double v(uint32_t p, uint32_t j, uint32_t i) const {
    const size_t row = size_t(p) * kKvH + j;
    return c8 ? double(c8->vd(row, i)) : double(common::bf16_to_f32(v16[row * kHD + i]));
  }
};

struct Block { double mx, sm; double acc[kHD]; };

inline double score(const float* q, const Kv& kv, uint32_t j, uint32_t p) {
  double a = 0;
  for (uint32_t i = 0; i < kHD; ++i) a += double(q[i]) * kv.k(p, j, i);
  return a * kScale;
}

// One block's partial over keys [lo, hi) (hi > lo).
inline Block block(const float* q, const Kv& kv, uint32_t j, uint32_t lo, uint32_t hi) {
  Block b;
  std::vector<double> sc(hi - lo);
  b.mx = -std::numeric_limits<double>::infinity();
  for (uint32_t p = lo; p < hi; ++p) {
    sc[p - lo] = score(q, kv, j, p);
    b.mx = std::max(b.mx, sc[p - lo]);
  }
  b.sm = 0;
  for (uint32_t i = 0; i < kHD; ++i) b.acc[i] = 0;
  for (uint32_t p = lo; p < hi; ++p) {
    const double w = std::exp(sc[p - lo] - b.mx);
    b.sm += w;
    for (uint32_t i = 0; i < kHD; ++i) b.acc[i] += w * kv.v(p, j, i);
  }
  return b;
}

// Row m's v2 blocks (the partials attn_verify_dpas writes for (q head, block, m)).
inline std::vector<Block> row_blocks(const float* q, const Kv& kv, uint32_t j, uint32_t pos, uint32_t m) {
  std::vector<Block> r;
  for (const Range& g : blocks(pos, m)) r.push_back(block(q, kv, j, g.lo, g.hi));
  return r;
}

// attn_reduce_v2's merge (ascending blocks), fp64: the cache-basis output acc / sm.
inline void merge(const std::vector<Block>& bl, double* o) {
  double mx = -std::numeric_limits<double>::infinity(), sm = 0, acc[kHD] = {};
  for (const Block& b : bl) {
    const double nmx = std::max(mx, b.mx);
    const double a = std::exp(mx - nmx), bs = std::exp(b.mx - nmx);
    sm = sm * a + b.sm * bs;
    for (uint32_t i = 0; i < kHD; ++i) acc[i] = acc[i] * a + b.acc[i] * bs;
    mx = nmx;
  }
  for (uint32_t i = 0; i < kHD; ++i) o[i] = acc[i] / sm;
}
// log sum exp of the merged blocks' scores: mx + log(sm), the row's softmax normaliser.
inline double merged_lse(const std::vector<Block>& bl) {
  double mx = -std::numeric_limits<double>::infinity();
  for (const Block& b : bl) mx = std::max(mx, b.mx);
  double sm = 0;
  for (const Block& b : bl) sm += b.sm * std::exp(b.mx - mx);
  return mx + std::log(sm);
}

// The whole attention over keys [0, n), in the cache's basis (rotated under int8).
inline void attend_rot(const float* q, const Kv& kv, uint32_t j, uint32_t n, double* o) {
  const Block b = block(q, kv, j, 0, n);
  for (uint32_t i = 0; i < kHD; ++i) o[i] = b.acc[i] / b.sm;
}
inline double lse(const float* q, const Kv& kv, uint32_t j, uint32_t n) {
  const Block b = block(q, kv, j, 0, n);
  return b.mx + std::log(b.sm);
}

// Every row of a launch: the pre-gate output (un-rotated under int8) [n_act][24][256] and the
// gated bf16 rows [n_act][6144].
struct Result {
  std::vector<double> o;
  std::vector<uint16_t> out;
};
inline Result run(uint32_t pos, uint32_t n_act, const float* q, const float* gate, const Kv& kv) {
  Result r;
  r.o.assign(size_t(n_act) * kQH * kHD, 0.0);
  r.out.assign(size_t(n_act) * kOutN, 0);
  for (uint32_t m = 0; m < n_act; ++m)
    for (uint32_t h = 0; h < kQH; ++h) {
      const size_t base = (size_t(m) * kQH + h) * kHD;
      double rot[kHD];
      attend_rot(q + base, kv, h / kGqa, pos + m + 1, rot);
      if (kv.kv8())
        kv8_ref::unrotate64(rot, &r.o[base]);
      else
        for (uint32_t i = 0; i < kHD; ++i) r.o[base + i] = rot[i];
      for (uint32_t i = 0; i < kHD; ++i)
        r.out[size_t(m) * kOutN + size_t(h) * kHD + i] = kv8_ref::gated(r.o[base + i], gate[base + i]);
    }
  return r;
}

// Cosine of two rows of 256 (1 for two zero rows, 0 for one zero row).
inline double cos_q(const double* a, const double* b) {
  double xy = 0, xx = 0, yy = 0;
  for (uint32_t i = 0; i < kHD; ++i) {
    xy += a[i] * b[i];
    xx += a[i] * a[i];
    yy += b[i] * b[i];
  }
  if (xx == 0 || yy == 0) return xx == yy ? 1.0 : 0.0;
  return xy / std::sqrt(xx * yy);
}
// A kernel's fp32 row against the reference's fp64 row.
inline double cos_q(const float* a, const double* b) {
  double t[kHD];
  for (uint32_t i = 0; i < kHD; ++i) t[i] = a[i];
  return cos_q(t, b);
}
// The kernel's gated bf16 row (its 256 dims of one head) against the reference's pre-gate fp64 row,
// both through the gate: the reference's own gated row is `Result::out`.
inline double cos_bf16(const uint16_t* a, const uint16_t* b) { return kv8_ref::cos_bf16(a, b); }

}  // namespace attn_verify_ref
