// attn_verify_ref_test - spec 28c's host reference (tests/kernels/attn_verify_ref.h), host only:
//   1. blocks(): v2's per-row stride ppw(L) and block count nb at the stride's steps, the blocks
//      tiling [0, L) contiguously with at most TGT of them;
//   2. bf16 KV: the fp64 reference against attn_ref.h's v1 chain (decode + reduce, the fp32 kernel
//      twin) at M = 1..4, synthetic KV at 512 / 2049 / 4096 keys: cosine >= 0.9999999 per
//      (q head, row) on the pre-gate output, its bf16 rounding equal or 1 ulp apart (counted), the
//      gated rows within 2 ulp (the sigmoid can take a 1-ulp pre-gate step a binade down; counted);
//   3. int8 KV: the reference against kv8_ref.h's attend (fp64, the same cache) - the same values;
//   4. the block view merged in attn_reduce_v2's order (fp64) gives the full attention back;
//   5. cos_q: 1 on equal rows, < 1 on different ones, 1 on two zero rows.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "attn_verify_ref.h"
#include "check.h"
#include "kernels/attn_ref.h"
#include "kernels/kv8_ref.h"

namespace avr = attn_verify_ref;

namespace {

// Synthetic inputs: fp32 q [M][24][256] ~ N(0, 1), gates ~ U(-3, 3) rounded to bf16, bf16 K / V
// [max_len][4][256] ~ N(0, 1) (score variance ~1 after the 1/16 scale).
struct Synth {
  uint32_t max_len, M;
  std::vector<float> q, gate;
  std::vector<uint16_t> k, v;
  Synth(uint32_t L, uint32_t m, uint32_t seed) : max_len(L), M(m) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> gd(-3.f, 3.f);
    q.resize(size_t(M) * avr::kQH * avr::kHD);
    gate.resize(q.size());
    for (auto& x : q) x = nd(rng);
    for (auto& x : gate) x = common::bf16_to_f32(common::f32_to_bf16(gd(rng)));
    k.resize(size_t(L) * avr::kKvH * avr::kHD);
    v.resize(k.size());
    for (auto& x : k) x = common::f32_to_bf16(nd(rng));
    for (auto& x : v) x = common::f32_to_bf16(nd(rng));
  }
};

}  // namespace

int main() {
  // 1. the blocking.
  struct Want { uint32_t L, ppw; };
  for (Want w : {Want{1, 64}, Want{64, 64}, Want{2048, 64}, Want{2049, 128}, Want{4096, 128},
                 Want{32768, 1024}, Want{130824, 4096}}) {
    CHECK_EQ(avr::ppw(w.L), w.ppw);
    const uint32_t pos = w.L - 1;                        // row 0 of a launch at pos has L keys
    const std::vector<avr::Range> b = avr::blocks(pos, 0);
    CHECK_EQ(uint32_t(b.size()), (w.L - 1) / w.ppw + 1);   // attn_reduce_v2's nb
    CHECK(b.size() <= avr::kTgt);
    CHECK_EQ(avr::nblocks(pos, 0), uint32_t(b.size()));
    uint32_t next = 0;
    for (size_t i = 0; i < b.size(); ++i) {
      CHECK_EQ(b[i].lo, uint32_t(i) * w.ppw);
      CHECK_EQ(b[i].lo, next);
      CHECK(b[i].hi > b[i].lo && b[i].hi - b[i].lo <= w.ppw);
      next = b[i].hi;
    }
    CHECK_EQ(next, w.L);
    std::printf("blocks L=%u: ppw %u, nb %zu\n", w.L, w.ppw, b.size());
  }
  // A launch's row m has its own stride: pos 2047, row 1 has 2049 keys -> ppw 128.
  CHECK_EQ(avr::blocks(2047, 0).size(), size_t(32));
  CHECK_EQ(avr::blocks(2047, 1).size(), size_t(17));

  // 2. bf16 KV against the v1 chain.
  size_t rows = 0, exact = 0, one_ulp = 0, gated_diff = 0;
  int gated_max = 0;
  double worst = 1.0;
  for (uint32_t keys : {512u, 2049u, 4096u}) {
    for (uint32_t M = 1; M <= 4; ++M) {
      const uint32_t max_len = ((keys + M + 63) / 64) * 64;
      Synth s(max_len, M, 1000 + keys + M);
      const uint32_t pos = keys - 1;                     // row 0 attends to `keys` keys
      std::vector<float> part(size_t(avr::kQH) * (max_len / 64) * M * avr::kPart);
      std::vector<uint16_t> out_v1(size_t(M) * avr::kOutN);
      attn_ref::decode(pos, M, M, max_len, s.q.data(), s.k.data(), s.v.data(), part.data());
      attn_ref::reduce(pos, M, M, max_len, part.data(), s.gate.data(), out_v1.data());
      const avr::Kv kv = avr::Kv::bf16(s.k.data(), s.v.data());
      avr::Result r = avr::run(pos, M, s.q.data(), s.gate.data(), kv);
      // the v1 chain's pre-gate output, recomputed from its partials in its merge order
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t h = 0; h < avr::kQH; ++h) {
          const uint32_t nb = (pos + m) / 64 + 1;
          double mx = -INFINITY, sm = 0, acc[avr::kHD] = {};
          for (uint32_t b = 0; b < nb; ++b) {
            const float* p = &part[((size_t(h) * (max_len / 64) + b) * M + m) * avr::kPart];
            const double nmx = std::fmax(mx, double(p[0]));
            const double a = std::exp(mx - nmx), bs = std::exp(double(p[0]) - nmx);
            sm = sm * a + p[1] * bs;
            for (uint32_t d = 0; d < avr::kHD; ++d) acc[d] = acc[d] * a + p[2 + d] * bs;
            mx = nmx;
          }
          double o1[avr::kHD];
          for (uint32_t d = 0; d < avr::kHD; ++d) o1[d] = acc[d] / sm;
          const double c = avr::cos_q(o1, &r.o[(size_t(m) * avr::kQH + h) * avr::kHD]);
          worst = std::fmin(worst, c);
          CHECK(c >= 0.9999999);
          for (uint32_t d = 0; d < avr::kHD; ++d) {
            // the pre-gate bf16 (the chain's first rounding): equal or 1 ulp; the gated row after
            // it may move 2 ulp when the sigmoid takes it a binade down - counted, not barred.
            const uint16_t b1 = common::f32_to_bf16(float(o1[d]));
            const uint16_t br =
                common::f32_to_bf16(float(r.o[(size_t(m) * avr::kQH + h) * avr::kHD + d]));
            const int du = std::abs(int(int16_t(b1)) - int(int16_t(br)));
            ++rows;
            if (du == 0) ++exact;
            else if (du == 1) ++one_ulp;
            else {
              std::fprintf(stderr, "keys %u M %u m %u h %u d %u: v1 %04x ref %04x\n", keys, M, m, h,
                           d, b1, br);
              CHECK(du <= 1);
            }
            const size_t i = size_t(m) * avr::kOutN + size_t(h) * avr::kHD + d;
            const int dg = std::abs(int(int16_t(out_v1[i])) - int(int16_t(r.out[i])));
            gated_max = std::max(gated_max, dg);
            if (dg) ++gated_diff;
          }
        }
    }
  }
  std::printf("bf16: worst cosine vs the v1 chain %.10f; pre-gate bf16 elements %zu: %zu equal, %zu one "
              "ulp; gated: %zu differ, max %d ulp\n",
              worst, rows, exact, one_ulp, gated_diff, gated_max);
  CHECK(gated_max <= 2);

  // 3. int8 KV against kv8_ref::attend, and 4. the block view merged back.
  {
    const uint32_t keys = 2049, M = 3, max_len = 2112;
    Synth s(max_len, M, 77);
    kv8_ref::Cache c(max_len);
    for (size_t row = 0; row < size_t(max_len) * avr::kKvH; ++row)
      c.put(row, &s.k[row * avr::kHD], &s.v[row * avr::kHD]);
    const uint32_t pos = keys - 1;
    const avr::Kv kv = avr::Kv::int8(c);
    avr::Result r = avr::run(pos, M, s.q.data(), s.gate.data(), kv);
    double w8 = 1.0, wb = 1.0, lse = 0.0;
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t h = 0; h < avr::kQH; ++h) {
        const float* q = &s.q[(size_t(m) * avr::kQH + h) * avr::kHD];
        double o[avr::kHD];
        kv8_ref::attend(q, c, h / avr::kGqa, pos + m + 1, o);
        const double* ro = &r.o[(size_t(m) * avr::kQH + h) * avr::kHD];
        w8 = std::fmin(w8, avr::cos_q(o, ro));
        for (uint32_t d = 0; d < avr::kHD; ++d) CHECK_NEAR(o[d], ro[d], 1e-9);
        // 4. the v2 blocks of this row, merged as attn_reduce_v2 does, give the rotated output
        std::vector<avr::Block> bl = avr::row_blocks(q, kv, h / avr::kGqa, pos, m);
        CHECK_EQ(uint32_t(bl.size()), avr::nblocks(pos, m));
        double rot[avr::kHD], full[avr::kHD];
        avr::merge(bl, rot);
        avr::attend_rot(q, kv, h / avr::kGqa, pos + m + 1, full);
        wb = std::fmin(wb, avr::cos_q(rot, full));
        for (uint32_t d = 0; d < avr::kHD; ++d) CHECK_NEAR(rot[d], full[d], 1e-9);
        lse = std::fmax(lse, std::fabs(avr::merged_lse(bl) - avr::lse(q, kv, h / avr::kGqa, pos + m + 1)));
      }
    CHECK(lse < 1e-9);
    std::printf("int8: cosine vs kv8_ref::attend %.12f; blocks merged vs whole %.12f (lse %.2g)\n", w8,
                wb, lse);
  }

  // 5. cos_q.
  {
    double a[avr::kHD], b[avr::kHD], z[avr::kHD] = {};
    for (uint32_t d = 0; d < avr::kHD; ++d) { a[d] = std::sin(d + 1.0); b[d] = a[d]; }
    CHECK(avr::cos_q(a, b) > 0.999999999999);
    b[7] += 0.5;
    CHECK(avr::cos_q(a, b) < 0.9999);
    CHECK(avr::cos_q(z, z) == 1.0);
    CHECK(avr::cos_q(a, z) == 0.0);
  }
  std::printf("attn_verify_ref_test: PASS\n");
  return 0;
}
