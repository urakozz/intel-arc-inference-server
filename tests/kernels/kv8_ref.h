#pragma once
// CPU references for src/kernels/kv8.cl (spec 12b, the int8 KV cache, rotkv).
//
//   prep    attn_prep_kv8: tests/kernels/attn_ref.h's attn_prep chain (bit for bit - the
//           kernel transcribes it), then src/common/kv8.h on its outputs: q rotated (fp32,
//           or rne_bf16 at PF = 1), K = rne_bf16(roped k) and V = rne_bf16(v) rotated and
//           quantised. Bitwise: every op of the chain is a single fp32 rounding in a fixed
//           order (the FWHT stages, the 1/16, the correctly rounded divide, rint).
//   attend  the attention a reader computes from that cache, in fp64: scores q_rot . deq(K)
//           / 16 over keys [0, n), softmax, o_rot = sum p deq(V), o = unrotate(o_rot) (fp64
//           R^T), then attn.cl's gated chain rne(f32(rne(o)) sigmoid(gate)). The readers
//           (decode v2, flash) are held to it by cosine, as their bf16 twins are held to
//           theirs: their `exp` and summation orders are not the host's.
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/attn_ref.h"

namespace kv8_ref {

namespace kv8 = common::kv8;
constexpr uint32_t kHD = 256, kQH = 24, kKvH = 4, kGqa = 6;

// One FA layer's int8 cache on the host: rows [max_len][4][256], scales [max_len][4].
struct Cache {
  uint32_t max_len = 0;
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
  explicit Cache(uint32_t L)
      : max_len(L), k(size_t(L) * kKvH * kHD), v(size_t(L) * kKvH * kHD),
        ks(size_t(L) * kKvH), vs(size_t(L) * kKvH) {}
  float kd(size_t row, uint32_t i) const { return kv8::dequant(k[row * kHD + i], ks[row]); }
  float vd(size_t row, uint32_t i) const { return kv8::dequant(v[row * kHD + i], vs[row]); }
  // Encode one (position, kv head) row from its bf16 values, as the writer does.
  void put(size_t row, const uint16_t* kb, const uint16_t* vb) {
    float x[kHD];
    for (uint32_t i = 0; i < kHD; ++i) x[i] = common::bf16_to_f32(kb[i]);
    ks[row] = kv8::encode(x, &k[row * kHD]);
    for (uint32_t i = 0; i < kHD; ++i) x[i] = common::bf16_to_f32(vb[i]);
    vs[row] = kv8::encode(x, &v[row * kHD]);
  }
};

// attn_prep_kv8 at (pos, n_act) over S-slice partials ([S][M][14336], attn_ref's layout; the
// S1 / PF builds are driven with slice 1 = +0.0f, which is exact - pf_harness's trick).
// attn_q: fp32 [M][24][256] rotated; attn_q16: the PF build's bf16 of it; gate as attn_prep.
inline void prep(uint32_t pos, uint32_t n_act, uint32_t M, const float* partials,
                 const float* fa_small, const float* rope, uint32_t max_len, float* attn_q,
                 uint16_t* attn_q16, float* attn_gate, Cache& c) {
  std::vector<float> q(size_t(M) * kQH * kHD);
  std::vector<uint16_t> kb(size_t(max_len) * kKvH * kHD), vb(size_t(max_len) * kKvH * kHD);
  attn_ref::prep(pos, n_act, M, partials, fa_small, rope, q.data(), attn_gate, kb.data(), vb.data());
  for (uint32_t m = 0; m < n_act; ++m) {
    for (uint32_t h = 0; h < kQH; ++h) {
      float y[kHD];
      kv8::rotate(&q[(size_t(m) * kQH + h) * kHD], y);
      for (uint32_t i = 0; i < kHD; ++i) {
        if (attn_q) attn_q[(size_t(m) * kQH + h) * kHD + i] = y[i];
        if (attn_q16) attn_q16[(size_t(m) * kQH + h) * kHD + i] = common::f32_to_bf16(y[i]);
      }
    }
    for (uint32_t j = 0; j < kKvH; ++j) {
      const size_t row = size_t(pos + m) * kKvH + j;
      c.put(row, &kb[row * kHD], &vb[row * kHD]);
    }
  }
}

// R^T in fp64: x = FWHT(s (.) y) / 16, the FWHT in double.
inline void unrotate64(const double* y, double* x) {
  for (uint32_t j = 0; j < kHD; ++j) x[j] = kv8::sign_neg(j) ? -y[j] : y[j];
  for (uint32_t h = 1; h < kHD; h <<= 1)
    for (uint32_t i = 0; i < kHD; ++i)
      if ((i & h) == 0) {
        const double a = x[i], b = x[i + h];
        x[i] = a + b;
        x[i + h] = a - b;
      }
  for (uint32_t i = 0; i < kHD; ++i) x[i] /= 16.0;
}

// One (q head, query) of a reader: q_rot [256] (the values the kernel reads), keys [0, n)
// of kv head j. Writes the un-rotated fp64 output (pre-gate) to `o`, and the rotated one
// (what pf_flash_attn_kv8 leaves in pf_o) to `o_rot` when non-null.
inline void attend(const float* q_rot, const Cache& c, uint32_t j, uint32_t n, double* o,
                   double* o_rot = nullptr) {
  std::vector<double> sc(n);
  double mx = -INFINITY;
  for (uint32_t p = 0; p < n; ++p) {
    const size_t row = size_t(p) * kKvH + j;
    double a = 0;
    for (uint32_t i = 0; i < kHD; ++i) a += double(q_rot[i]) * double(c.kd(row, i));
    sc[p] = a / 16.0;
    mx = std::max(mx, sc[p]);
  }
  double sum = 0, orot[kHD] = {};
  for (uint32_t p = 0; p < n; ++p) {
    const double w = std::exp(sc[p] - mx);
    sum += w;
    const size_t row = size_t(p) * kKvH + j;
    for (uint32_t i = 0; i < kHD; ++i) orot[i] += w * double(c.vd(row, i));
  }
  for (uint32_t i = 0; i < kHD; ++i) orot[i] /= sum;
  if (o_rot)
    for (uint32_t i = 0; i < kHD; ++i) o_rot[i] = orot[i];
  unrotate64(orot, o);
}

// attn.cl's gated chain on an fp64 output: rne(f32(rne(o)) sigmoid_f32(gate)).
inline uint16_t gated(double o, float gate) {
  return attn_ref::rne(attn_ref::f32(attn_ref::rne(float(o))) * attn_ref::sigmoid_f32(gate));
}

// Cosine of two bf16 rows of 256.
inline double cos_bf16(const uint16_t* a, const uint16_t* b) {
  double xy = 0, xx = 0, yy = 0;
  for (uint32_t i = 0; i < kHD; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    xy += x * y;
    xx += x * x;
    yy += y * y;
  }
  return xx > 0 && yy > 0 ? xy / std::sqrt(xx * yy) : (xx == yy ? 1.0 : 0.0);
}

}  // namespace kv8_ref
