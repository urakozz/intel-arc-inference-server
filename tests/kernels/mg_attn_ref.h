#pragma once
// CPU twin of the full (NoPE) decode attention Nemotron 3.5 Bolt binds by name (Decision 15): src/kernels/bolt/mg_attn.cl
// (flash) and mg_attn_eager.cl (the reference's bf16 chain, one launch) at Q 32 / KV 2 / head_dim 128, TGT 64. The kernels
// came with spec 23c's Muse-Glimmer-30B (dropped 2026-10-27; their names and command lines are kept: they are Bolt's
// binaries); this is the full-layer half of that spec's twin (tests/kernels/muse_ref.h), line for line - no window, no
// ring, no gate (Bolt's ones gate is exact).
//
//   attention_eager()  the reference's bf16 eager chain (k2_ref's score / softmax) over keys [0, pos]; P·V in
//                      eager_block-key blocks (the device's)
//   attention_fp64()   the same in fp64 (flash's tolerance reference)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"   // eager_score, eager_softmax (torch's fp32 softmax)

namespace mg_attn_ref {

constexpr uint32_t kHd = 128, kQHeads = 32, kKvHeads = 2, kAttnTgt = 64;

inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float bf(uint16_t h) { return common::bf16_to_f32(h); }

// v2's ppw over the step's key span (the eager kernel's P·V block).
inline uint32_t eager_block(uint32_t span, uint32_t tgt = kAttnTgt) {
  uint32_t p = (span + tgt - 1) / tgt;
  p = (p + 63u) & ~63u;
  return std::max(64u, p);
}

// q [32][128] (bf16), caches [rows][2][128] read at row p.
struct EagerHead {
  std::vector<float> s, p, o;   // scores and probabilities (bf16 values) over keys [0, pos], o's fp32 sums
};
inline EagerHead attention_eager_head(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, uint32_t h) {
  const uint32_t L = pos + 1, j = h / (kQHeads / kKvHeads);
  const uint32_t blk = eager_block(L);
  float qf[kHd];
  for (uint32_t d = 0; d < kHd; ++d) qf[d] = bf(q[size_t(h) * kHd + d]);
  EagerHead r;
  r.s.resize(L);
  r.p.resize(L);
  r.o.assign(kHd, 0.0f);
  for (uint32_t i = 0; i < L; ++i) r.s[i] = k2_ref::eager_score(qf, kc + (size_t(i) * kKvHeads + j) * kHd, kHd);
  k2_ref::eager_softmax(r.s.data(), L, r.p.data());
  for (uint32_t d = 0; d < kHd; ++d) {
    float o = 0.0f;
    for (uint32_t b0 = 0; b0 < L; b0 += blk) {
      float acc = 0.0f;
      for (uint32_t i = b0; i < std::min(L, b0 + blk); ++i)
        acc = std::fma(r.p[i], bf(vc[(size_t(i) * kKvHeads + j) * kHd + d]), acc);
      o += acc;
    }
    r.o[d] = o;
  }
  return r;
}
// Every head: out [32][128] = rne(o).
inline void attention_eager(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, uint16_t* out) {
  for (uint32_t h = 0; h < kQHeads; ++h) {
    const EagerHead e = attention_eager_head(q, kc, vc, pos, h);
    for (uint32_t d = 0; d < kHd; ++d) out[size_t(h) * kHd + d] = rne(e.o[d]);
  }
}
inline void attention_fp64(const uint16_t* q, const uint16_t* kc, const uint16_t* vc, uint32_t pos, double* out) {
  const uint32_t L = pos + 1;
  std::vector<double> sc(L);
  for (uint32_t h = 0; h < kQHeads; ++h) {
    const uint32_t j = h / (kQHeads / kKvHeads);
    double mx = -INFINITY;
    for (uint32_t i = 0; i < L; ++i) {
      const size_t rw = (size_t(i) * kKvHeads + j) * kHd;
      double a = 0;
      for (uint32_t d = 0; d < kHd; ++d) a += double(bf(q[size_t(h) * kHd + d])) * bf(kc[rw + d]);
      sc[i] = a / std::sqrt(double(kHd));
      mx = std::max(mx, sc[i]);
    }
    double z = 0;
    for (double& s : sc) z += (s = std::exp(s - mx));
    for (uint32_t d = 0; d < kHd; ++d) {
      double o = 0;
      for (uint32_t i = 0; i < L; ++i) o += sc[i] / z * bf(vc[(size_t(i) * kKvHeads + j) * kHd + d]);
      out[size_t(h) * kHd + d] = o;
    }
  }
}

}  // namespace mg_attn_ref
