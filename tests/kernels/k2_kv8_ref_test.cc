// k2_kv8_ref_test - spec 18e Task 1: the host references of K2-Horizon's int8 KV cache
// (tests/kernels/k2_kv8_ref.h over src/common/kv8.h hd128), no device. Host only (the Mac runs it).
//
//   1. the flash epilogue's un-rotation as k2_pf_flash_attn_kv8 decomposes it (xor-lane stages,
//      then register stages) is hd128::unrotate bit for bit;
//   2. the writer (Review Focus 1's scheme at 128): K2's prep chain then rotate_q / rotate_kv +
//      int8 - q_rot . deq(K) against the bf16 q . k within the int8 error, the gate untouched,
//      unrotate(deq(V)) against the bf16 V;
//   3. Review Focus 2, MoVA writes the KV: the staged row is k2_ref::mova_value's routed mix
//      (the ascending-id combine done) and the cache's V row decodes back to THAT mix - not to
//      any single expert's output;
//   4. the readers' fp64 attention over the int8 cache against bf16 KV at K2's shape (32 q / 8 kv
//      heads, GQA 4) at depths 6 / 300 / 3000 with K outlier channels: rotkv per head cosine
//      >= 0.999 (PROPOSED kernel-level bar), and closer than per token without the rotation;
//   5. the eager int8 chain (k2_attn_eager_*_kv8's orders) against the fp64 reference, gated.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/k2_kv8_ref.h"
#include "kernels/k2_ref.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"

namespace {

namespace kr = k2_kv8_ref;
namespace kv8 = common::kv8;
namespace h128 = common::kv8::hd128;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(rng);
  return v;
}

void check_flash_unrotate() {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  size_t n = 0;
  for (int t = 0; t < 2000; ++t) {
    float y[128], a[128], b[128];
    const float mag = std::pow(10.0f, -6.0f + 10.0f * float(t) / 2000.0f);
    for (float& v : y) v = nd(rng) * mag * (t % 7 == 3 ? 100.0f : 1.0f);
    h128::unrotate(y, a);
    kr::flash_unrotate(y, b);
    CHECK(std::memcmp(a, b, sizeof a) == 0);
    n += 128;
  }
  std::printf("1. the flash epilogue's lane/register FWHT == hd128::unrotate: %zu values bitwise\n", n);
}

void check_writer(const model::K2Desc& m, const std::vector<float>& rope) {
  const uint32_t pos = 211, S = 2;
  for (int dense = 1; dense >= 0; --dense) {
    const uint32_t N = dense ? m.attn_dense_n() : m.attn_sparse_n();
    std::vector<float> part = random_f32(size_t(S) * N, -1.f, 1.f, 10 + dense);
    // Outlier channels on K (dims 9, 73 of every kv head), as real K has.
    for (uint32_t s = 0; s < S; ++s)
      for (uint32_t j = 0; j < m.kv_heads; ++j)
        for (uint32_t i : {9u, 73u}) part[size_t(s) * N + m.q_n() + j * 128 + i] *= 20.0f;
    const float* cs = rope.data() + size_t(pos) * 128;
    std::vector<uint16_t> staged;
    if (!dense) staged.resize(m.kv_n());
    for (uint32_t i = 0; i < staged.size(); ++i) staged[i] = rne(std::sin(float(i) * 0.37f) * 0.5f);
    kr::Cache c(pos + 1, m.kv_heads);
    const kr::Prep p = kr::prep(part.data(), 1, 0, N, S, cs, m.q_heads, m.kv_heads, pos,
                                dense ? nullptr : staged.data(), c);
    const k2_ref::Prep b = k2_ref::attn_prep(part.data(), 1, 0, N, S, cs, m.q_heads, m.kv_heads, 128, dense);
    CHECK(p.gate == b.gate);
    double worst_dot = 0, worst_v = 1;
    for (uint32_t h = 0; h < m.q_heads; ++h) {
      const uint32_t j = h / m.gqa();
      const size_t r = c.row(pos, j);
      double d0 = 0, d1 = 0, nq = 0, nk = 0;
      for (uint32_t i = 0; i < 128; ++i) {
        d0 += double(b.q[size_t(h) * 128 + i]) * f32(b.k[size_t(j) * 128 + i]);
        d1 += double(p.q[size_t(h) * 128 + i]) * c.kd(r, i);
        nq += double(b.q[size_t(h) * 128 + i]) * b.q[size_t(h) * 128 + i];
        nk += double(f32(b.k[size_t(j) * 128 + i])) * f32(b.k[size_t(j) * 128 + i]);
      }
      worst_dot = std::max(worst_dot, std::fabs(d0 - d1) / std::sqrt(nq * nk));
    }
    for (uint32_t j = 0; j < m.kv_heads; ++j) {
      const size_t r = c.row(pos, j);
      double y[128], x[128], want[128];
      for (uint32_t i = 0; i < 128; ++i) {
        y[i] = c.vd(r, i);
        want[i] = dense ? f32(b.v[size_t(j) * 128 + i]) : f32(staged[size_t(j) * 128 + i]);
      }
      kr::unrotate64(y, x);
      worst_v = std::min(worst_v, kr::cos64(x, want, 128));
    }
    std::printf("2. writer (%s): |q_rot.deq(K) - q.k| / (|q||k|) <= %.2e over 32 heads; "
                "unrotate(deq(V)) vs V cos >= %.7f; gate bitwise\n", dense ? "dense" : "MoVA, staged V",
                worst_dot, worst_v);
    CHECK(worst_dot < 1e-2);
    CHECK(worst_v > 0.9995);   // int8 per row: ~1 - (step / sqrt(12))^2 / 2, a structured row lower
  }
}

std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) | (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}

void check_mova_writes_kv(const model::K2Desc& m) {
  const uint32_t H = 512, N = m.kv_n();   // a short hidden: the semantics, not the shape
  const size_t vblk = size_t(N / 16) * (H / 64) * 136;
  const std::vector<uint32_t> vw = random_blocks(vblk * 8, 40);
  std::vector<uint16_t> x(H);
  for (uint32_t i = 0; i < H; ++i) x[i] = rne(std::cos(float(i) * 0.11f) * 0.1f);
  k2_ref::Route r;
  const uint32_t ids[4] = {1, 2, 5, 7};
  const float ws[4] = {0.625f, 0.75f, 0.5f, 0.625f};
  for (uint32_t j = 0; j < 4; ++j) {
    r.ids[j] = ids[j];
    r.w[j] = ws[j];
  }
  const std::vector<uint16_t> mix = k2_ref::mova_value(r, x.data(), vw.data(), H, N, 4, 4);
  kr::Cache c(1, m.kv_heads);
  for (uint32_t j = 0; j < m.kv_heads; ++j) c.put_v(c.row(0, j), mix.data() + size_t(j) * 128);
  double worst_mix = 1, best_single = -1;
  for (uint32_t j = 0; j < m.kv_heads; ++j) {
    double y[128], got[128], want[128];
    for (uint32_t i = 0; i < 128; ++i) {
      y[i] = c.vd(c.row(0, j), i);
      want[i] = f32(mix[size_t(j) * 128 + i]);
    }
    kr::unrotate64(y, got);
    worst_mix = std::min(worst_mix, kr::cos64(got, want, 128));
    for (uint32_t e = 0; e < 4; ++e) {   // one expert alone, weighted: what a pre-combine
      k2_ref::Route one;                 // quantisation point would see
      one.ids[0] = ids[e];
      one.w[0] = ws[e];
      const std::vector<uint16_t> single = k2_ref::mova_value(one, x.data(), vw.data(), H, N, 1, 4);
      double s[128];
      for (uint32_t i = 0; i < 128; ++i) s[i] = f32(single[size_t(j) * 128 + i]);
      best_single = std::max(best_single, kr::cos64(got, s, 128));
    }
  }
  std::printf("3. MoVA writes the KV: the cache's V decodes to the routed mix (cos >= %.7f per head); "
              "to any one weighted expert at most %.4f\n", worst_mix, best_single);
  CHECK(worst_mix > 0.9999);
  CHECK(best_single < 0.99);
}

void check_attention(const model::K2Desc& m) {
  const uint32_t ML = 3000, QH = m.q_heads, KVH = m.kv_heads;
  std::mt19937 rng(50);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> K(size_t(ML) * KVH * 128), V(K.size());
  for (size_t i = 0; i < K.size(); ++i) {
    const uint32_t d = uint32_t(i % 128);
    K[i] = rne(nd(rng) * (d == 9 || d == 73 ? 12.0f : 1.0f));
    V[i] = rne(nd(rng));
  }
  std::vector<float> q(size_t(QH) * 128), g(size_t(QH) * 128);
  for (float& e : q) e = rf(nd(rng) * 0.6f);
  for (uint32_t i = 0; i < q.size(); ++i) g[i] = rf(i % 3 == 0 ? 30.0f : nd(rng));
  kr::Cache c(ML, KVH), pt(ML, KVH);   // rotkv, and per token without the rotation
  for (uint32_t p = 0; p < ML; ++p)
    for (uint32_t j = 0; j < KVH; ++j) {
      const size_t r = c.row(p, j);
      c.put_k(r, &K[r * 128]);
      c.put_v(r, &V[r * 128]);
      float x[128];
      for (uint32_t i = 0; i < 128; ++i) x[i] = f32(K[r * 128 + i]);
      pt.ks[r] = h128::quantise(x, &pt.k[r * 128]);
      for (uint32_t i = 0; i < 128; ++i) x[i] = f32(V[r * 128 + i]);
      pt.vs[r] = h128::quantise(x, &pt.v[r * 128]);
    }
  for (uint32_t len : {6u, 300u, 3000u}) {
    double worst_rot = 1, worst_pt = 1, worst_eager = 1;
    double eager_rel = 0;
    for (uint32_t h = 0; h < QH; ++h) {
      const uint32_t j = h / m.gqa();
      const float* qh = q.data() + size_t(h) * 128;
      const std::vector<double> ref = k2_ref::attention(qh, K.data(), V.data(), len, j, KVH, 128);
      float qr[128];
      h128::rotate_q(qh, qr);
      double o[128], o_pt[128];
      kr::attend(qr, c, j, len, o);
      // per token, no rotation: q as is, deq as is, no un-rotation
      {
        std::vector<double> sc(len);
        double mx = -INFINITY, sum = 0, acc[128] = {};
        for (uint32_t p = 0; p < len; ++p) {
          double a = 0;
          for (uint32_t i = 0; i < 128; ++i) a += double(qh[i]) * pt.kd(pt.row(p, j), i);
          sc[p] = a / std::sqrt(128.0);
          mx = std::max(mx, sc[p]);
        }
        for (uint32_t p = 0; p < len; ++p) {
          const double w = std::exp(sc[p] - mx);
          sum += w;
          for (uint32_t i = 0; i < 128; ++i) acc[i] += w * pt.vd(pt.row(p, j), i);
        }
        for (uint32_t i = 0; i < 128; ++i) o_pt[i] = acc[i] / sum;
      }
      worst_rot = std::min(worst_rot, kr::cos64(o, ref.data(), 128));
      worst_pt = std::min(worst_pt, kr::cos64(o_pt, ref.data(), 128));
      // The eager int8 chain, gated, against the fp64 output gated.
      const kr::EagerHead e = kr::eager_head(qr, c, len, j, k2_ref::eager_block(len - 1, 1, 32));
      double eo[128];
      for (uint32_t d = 0; d < 128; ++d) eo[d] = e.o[d];
      worst_eager = std::min(worst_eager, kr::cos64(eo, ref.data(), 128));
      // Gated, the eager chain against the fp64 int8 output: the worst |difference| relative to
      // the head's largest gated value (ulps mislead near zero, where a sign flip is ~2^15 keys).
      double dmax = 0, vmax = 0;
      for (uint32_t d = 0; d < 128; ++d) {
        const double a = f32(k2_ref::attn_gate(e.o[d], g[size_t(h) * 128 + d]));
        const double b = f32(k2_ref::attn_gate(float(o[d]), g[size_t(h) * 128 + d]));
        dmax = std::max(dmax, std::fabs(a - b));
        vmax = std::max(vmax, std::fabs(b));
      }
      eager_rel = std::max(eager_rel, vmax > 0 ? dmax / vmax : 0.0);
    }
    std::printf("4/5. %4u keys, K outliers: per-head cos vs bf16 KV (fp64) - rotkv >= %.7f, per token "
                "(no rotation) >= %.7f; eager int8 chain >= %.7f (gated vs the fp64 int8 output: max |d| "
                "%.2e of the head's largest)\n", len, worst_rot, worst_pt, worst_eager, eager_rel);
    CHECK(worst_rot >= 0.999);
    CHECK(worst_rot > worst_pt);
    CHECK(worst_eager >= 0.999);
  }
}

}  // namespace

int main() {
  const model::K2Desc& m = model::k2();
  CHECK_EQ(m.head_dim, 128u);
  const std::vector<float> rope = loader::k2_rope_table(m, 256);
  check_flash_unrotate();
  check_writer(m, rope);
  check_mova_writes_kv(m);
  check_attention(m);
  std::puts("k2_kv8_ref_test OK");
  return 0;
}
