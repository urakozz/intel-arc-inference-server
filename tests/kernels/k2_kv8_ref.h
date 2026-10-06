#pragma once
// CPU references for src/kernels/k2/k2_kv8.cl (spec 18e Task 1: K2-Horizon's int8 KV cache,
// rotkv at head_dim 128; src/common/kv8.h namespace hd128 is the scheme).
//
//   prep()         k2_attn_prep_kv8: k2_ref::attn_prep's chain (bit for bit - the kernel
//                  transcribes it), then q -> rotate_q (fp32), K = f32(RoPE out) and V =
//                  f32(v) -> rotate_kv -> int8 + fp16 scale at row pos + m. V is the fused row's
//                  (dense) or the STAGED routed mix (MoVA: k2_ref::mova_value's bf16 output, the
//                  combine done - Review Focus 2). Bitwise: every op is one fp32 rounding in a
//                  fixed order.
//   attend()       a reader's attention in fp64 over the dequantised cache (scale 1/sqrt(128)),
//                  the output un-rotated in fp64 (y R^T / sqrt(2)): the tolerance reference for
//                  the flash readers (decode and prefill), whose exp and orders are their own.
//   eager_head()   k2_attn_eager_*_kv8's chain, the kernel's orders: the score's fma chain on
//                  the exact dequantised operand, rounded twice; k2_ref::eager_softmax (torch's);
//                  P·V blocks of fma chains added ascending; then hd128::unrotate in fp32 and
//                  k2_ref::attn_gate. Bitwise against the kernel.
//   flash_unrotate()  k2_pf_flash_attn_kv8's epilogue FWHT as the kernel decomposes it (lane
//                  stages 1..8 by xor partner, then register stages 16..64), checked against
//                  hd128::unrotate bit for bit (k2_kv8_ref_test).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/k2_ref.h"

namespace k2_kv8_ref {

namespace kv8 = common::kv8;
namespace h128 = common::kv8::hd128;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;
constexpr uint32_t kHD = 128;

// One layer's int8 cache: rows [max_len][kvh][128], scales [max_len][kvh].
struct Cache {
  uint32_t max_len = 0, kvh = 0;
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
  Cache(uint32_t L, uint32_t kv_heads)
      : max_len(L), kvh(kv_heads), k(size_t(L) * kv_heads * kHD), v(size_t(L) * kv_heads * kHD),
        ks(size_t(L) * kv_heads), vs(size_t(L) * kv_heads) {}
  size_t row(uint32_t p, uint32_t j) const { return size_t(p) * kvh + j; }
  float kd(size_t r, uint32_t i) const { return kv8::dequant(k[r * kHD + i], ks[r]); }
  float vd(size_t r, uint32_t i) const { return kv8::dequant(v[r * kHD + i], vs[r]); }
  // Encode one (position, kv head) K or V row from its bf16 values, as the writer does.
  void put_k(size_t r, const uint16_t* kb) {
    float x[kHD];
    for (uint32_t i = 0; i < kHD; ++i) x[i] = f32(kb[i]);
    ks[r] = h128::encode(x, &k[r * kHD]);
  }
  void put_v(size_t r, const uint16_t* vb) {
    float x[kHD];
    for (uint32_t i = 0; i < kHD; ++i) x[i] = f32(vb[i]);
    vs[r] = h128::encode(x, &v[r * kHD]);
  }
};

// k2_attn_prep_kv8 for token m at pos + m: returns q (rotated, fp32) and the gate; writes K and
// V into `c`. `v_rows`: the staged bf16 [kvh x 128] row of token m (MoVA), or null for a dense
// layer (V from the fused row's v columns, as k2_ref::attn_prep reads them).
struct Prep {
  std::vector<float> q, gate;   // [qh][128]: q rotated (fp32), gate as k2_attn_prep's
};
inline Prep prep(const float* partials, uint32_t M, uint32_t m, uint32_t n, uint32_t S, const float* cs,
                 uint32_t qh, uint32_t kvh, uint32_t pos, const uint16_t* v_rows, Cache& c) {
  const bool dense = v_rows == nullptr;
  const k2_ref::Prep b = k2_ref::attn_prep(partials, M, m, n, S, cs, qh, kvh, kHD, dense);
  Prep p;
  p.q.resize(size_t(qh) * kHD);
  p.gate = b.gate;
  for (uint32_t h = 0; h < qh; ++h) h128::rotate_q(&b.q[size_t(h) * kHD], &p.q[size_t(h) * kHD]);
  for (uint32_t j = 0; j < kvh; ++j) {
    const size_t r = c.row(pos + m, j);
    c.put_k(r, &b.k[size_t(j) * kHD]);
    c.put_v(r, dense ? &b.v[size_t(j) * kHD] : v_rows + size_t(j) * kHD);
  }
  return p;
}

// y R^T / sqrt(2) in fp64: FWHT(s (.) y) / 16, the FWHT in double.
inline void unrotate64(const double* y, double* x) {
  for (uint32_t j = 0; j < kHD; ++j) x[j] = h128::sign_neg(j) ? -y[j] : y[j];
  for (uint32_t h = 1; h < kHD; h <<= 1)
    for (uint32_t i = 0; i < kHD; ++i)
      if ((i & h) == 0) {
        const double a = x[i], b = x[i + h];
        x[i] = a + b;
        x[i + h] = a - b;
      }
  for (uint32_t i = 0; i < kHD; ++i) x[i] /= 16.0;
}

// One (q head, query) over keys [0, n) of kv head j: the un-rotated fp64 output (pre-gate),
// and the rotated one when `o_rot` is given.
inline void attend(const float* q_rot, const Cache& c, uint32_t j, uint32_t n, double* o,
                   double* o_rot = nullptr) {
  std::vector<double> sc(n);
  double mx = -INFINITY;
  for (uint32_t p = 0; p < n; ++p) {
    const size_t r = c.row(p, j);
    double a = 0;
    for (uint32_t i = 0; i < kHD; ++i) a += double(q_rot[i]) * double(c.kd(r, i));
    sc[p] = a / std::sqrt(128.0);
    mx = std::max(mx, sc[p]);
  }
  double sum = 0, orot[kHD] = {};
  for (uint32_t p = 0; p < n; ++p) {
    const double w = std::exp(sc[p] - mx);
    sum += w;
    const size_t r = c.row(p, j);
    for (uint32_t i = 0; i < kHD; ++i) orot[i] += w * double(c.vd(r, i));
  }
  for (uint32_t i = 0; i < kHD; ++i) orot[i] /= sum;
  if (o_rot)
    for (uint32_t i = 0; i < kHD; ++i) o_rot[i] = orot[i];
  unrotate64(orot, o);
}

// k2_attn_eager_*_kv8 over keys [0, len) of kv head j: s and p as k2_ref::EagerHead, o_rot the
// fp32 P·V sums (blocks of `blk` keys), o = hd128::unrotate(o_rot) (fp32: what the reduce gates).
struct EagerHead {
  std::vector<float> s, p, o_rot, o;
};
inline EagerHead eager_head(const float* q_rot, const Cache& c, uint32_t len, uint32_t j, uint32_t blk) {
  EagerHead r;
  r.s.resize(len);
  r.p.resize(len);
  r.o_rot.resize(kHD);
  r.o.resize(kHD);
  for (uint32_t pp = 0; pp < len; ++pp) {
    const size_t row = c.row(pp, j);
    const float sk = kv8::f16f(c.ks[row]);
    float a = 0.0f;
    for (uint32_t d = 0; d < kHD; ++d) a = std::fma(q_rot[d], float(c.k[row * kHD + d]) * sk, a);
    r.s[pp] = rf(rf(a) * k2_ref::kAttnScale);
  }
  k2_ref::eager_softmax(r.s.data(), len, r.p.data());
  for (uint32_t d = 0; d < kHD; ++d) {
    float o = 0.0f;
    for (uint32_t b0 = 0; b0 < len; b0 += blk) {
      float acc = 0.0f;
      for (uint32_t pp = b0; pp < std::min(len, b0 + blk); ++pp) acc = std::fma(r.p[pp], c.vd(c.row(pp, j), d), acc);
      o += acc;
    }
    r.o_rot[d] = o;
  }
  h128::unrotate(r.o_rot.data(), r.o.data());
  return r;
}

// The flash epilogue's un-rotation as k2_pf_flash_attn_kv8 decomposes it: w[dd][lane] holds dim
// 16 dd + lane; flip; stages h = 1, 2, 4, 8 with the xor partner (lower lane a + c, upper c' - v
// = a - c); stages 16, 32, 64 across dd; times 1/16.
inline void flash_unrotate(const float* y, float* out) {
  float w[8][16];
  for (uint32_t dd = 0; dd < 8; ++dd)
    for (uint32_t l = 0; l < 16; ++l) w[dd][l] = kv8::flip(y[16 * dd + l], h128::sign_neg(16 * dd + l));
  for (uint32_t h = 1; h < 16; h <<= 1) {
    float nw[8][16];
    for (uint32_t dd = 0; dd < 8; ++dd)
      for (uint32_t l = 0; l < 16; ++l) {
        const float v = w[dd][l], o = w[dd][l ^ h];
        nw[dd][l] = (l & h) ? (o - v) : (v + o);
      }
    std::memcpy(w, nw, sizeof w);
  }
  for (uint32_t hb = 1; hb < 8; hb <<= 1)
    for (uint32_t dd = 0; dd < 8; ++dd)
      if ((dd & hb) == 0)
        for (uint32_t l = 0; l < 16; ++l) {
          const float a = w[dd][l], c = w[dd | hb][l];
          w[dd][l] = a + c;
          w[dd | hb][l] = a - c;
        }
  for (uint32_t dd = 0; dd < 8; ++dd)
    for (uint32_t l = 0; l < 16; ++l) out[16 * dd + l] = w[dd][l] * h128::kQScale;
}

inline double cos64(const double* a, const double* b, uint32_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (uint32_t i = 0; i < n; ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

}  // namespace k2_kv8_ref
