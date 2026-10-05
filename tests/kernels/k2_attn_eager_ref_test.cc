// k2_attn_eager_ref_test - K2-Horizon's eager decode attention (B70_K2_ATTN=eager, spec 18
// §10): tests/kernels/k2_ref.h's eager chain against the reference itself, and against the
// flash path it is the alternative to. Host only (the Mac runs it).
//
//   (a) torch, bit for bit. tests/kernels/k2_attn_eager_fixture.h is tools/oracle/k2_ref.py
//       attention() - transformers' bf16 eager path - run by tools/oracle/k2_attn_eager_fixture.py
//       in the oracle container on hash inputs both sides draw. Each stage is fed the FIXTURE's
//       input to it, so a failure names the stage:
//         scores   eager_score from q, k
//         softmax  eager_softmax from the fixture's scores (Sleef's exp, reduce_all's sum order)
//         P·V      eager_pv from the fixture's probabilities, one chain and 16-key blocks
//         chain    attention_eager from q, k, v
//       All four bitwise. `coarse` has exact dots (no accumulation order can show in its
//       scores); the fine cases' dots and every P·V are inexact, and match because no fp32 sum
//       landed within its order noise of a bf16 boundary - k2_ref.h's eager section says
//       which parts are bitwise by construction and which by measurement.
//   (b) eager and flash differ after the bf16 rounding: two keys whose fp32 scores differ but
//       round to one bf16 score get equal weights from eager and unequal ones from flash, three
//       bf16 ulps apart in the output; and on the fixture's inputs flash's arithmetic (taken
//       in fp64: fp32 scores and probabilities, one rounding at the end) lands on another bf16
//       output than torch in many places, where eager lands on none.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/k2_attn_eager_fixture.h"
#include "kernels/k2_ref.h"

namespace {

namespace fx = k2_attn_eager_fixture;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

std::vector<uint16_t> unhex(const char* h) {
  std::vector<uint16_t> v;
  for (; h[0]; h += 4) v.push_back(uint16_t(std::strtoul(std::string(h, 4).c_str(), nullptr, 16)));
  return v;
}
int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

struct Tally {
  size_t n = 0, diff = 0;
  int worst = 0;
  void add(uint16_t got, uint16_t want) {
    ++n;
    if (got != want) {
      ++diff;
      worst = std::max(worst, ulps(got, want));
    }
  }
};

struct Case {
  const char* name;
  bool coarse;
  unsigned seed, T, P;
  const char *s, *p, *o;
};

void run_case(const Case& c) {
  const unsigned H = fx::kHeads, KV = fx::kKvHeads, HD = fx::kHd, rep = H / KV;
  const std::vector<uint16_t> fs = unhex(c.s), fp = unhex(c.p), fo = unhex(c.o);
  CHECK_EQ(fs.size(), size_t(H) * c.T * c.P);
  CHECK_EQ(fp.size(), fs.size());
  CHECK_EQ(fo.size(), size_t(c.T) * H * HD);
  // The inputs, as the fixture drew them: q [H][T][hd], k / v [KV][P][hd] -> the engine's
  // cache layout [P][KV][hd].
  std::vector<float> q(size_t(H) * c.T * HD);
  for (size_t i = 0; i < q.size(); ++i) q[i] = k2_ref::eager_fixture_val(c.coarse, 0, uint32_t(i), c.seed);
  std::vector<uint16_t> kc(size_t(c.P) * KV * HD), vc(kc.size());
  for (unsigned j = 0; j < KV; ++j)
    for (unsigned p = 0; p < c.P; ++p)
      for (unsigned d = 0; d < HD; ++d) {
        const uint32_t i = (j * c.P + p) * HD + d;
        kc[(size_t(p) * KV + j) * HD + d] = rne(k2_ref::eager_fixture_val(c.coarse, 1, i, c.seed));
        vc[(size_t(p) * KV + j) * HD + d] = rne(k2_ref::eager_fixture_val(c.coarse, 2, i, c.seed));
      }

  Tally ts, tp, to, tchain, tdev, tflash;
  size_t masked_bad = 0;
  for (unsigned h = 0; h < H; ++h)
    for (unsigned t = 0; t < c.T; ++t) {
      const unsigned pos = c.P - c.T + t, len = pos + 1, j = h / rep;
      const float* qh = q.data() + (size_t(h) * c.T + t) * HD;
      const size_t row = (size_t(h) * c.T + t) * c.P;
      // Masked entries: -inf scores, exact-zero probabilities.
      for (unsigned p = len; p < c.P; ++p) masked_bad += fs[row + p] != 0xFF80 || fp[row + p] != 0;
      // scores from q, k
      for (unsigned p = 0; p < len; ++p)
        ts.add(rne(k2_ref::eager_score(qh, kc.data() + (size_t(p) * KV + j) * HD, HD)), fs[row + p]);
      // softmax from the fixture's scores
      std::vector<float> s(len), pr(len);
      for (unsigned p = 0; p < len; ++p) s[p] = f32(fs[row + p]);
      k2_ref::eager_softmax(s.data(), len, pr.data());
      for (unsigned p = 0; p < len; ++p) tp.add(rne(pr[p]), fp[row + p]);
      // P·V from the fixture's probabilities (one chain), and at the device's blocking
      std::vector<float> fpr(len);
      for (unsigned p = 0; p < len; ++p) fpr[p] = f32(fp[row + p]);
      const size_t orow = (size_t(t) * H + h) * HD;
      for (unsigned d = 0; d < HD; ++d) {
        to.add(rne(k2_ref::eager_pv(fpr.data(), vc.data(), len, j, KV, HD, d, len)), fo[orow + d]);
        tdev.add(rne(k2_ref::eager_pv(fpr.data(), vc.data(), len, j, KV, HD, d, 16)), fo[orow + d]);
      }
      // the whole chain from q, k, v
      const k2_ref::EagerHead e = k2_ref::attention_eager(qh, kc.data(), vc.data(), len, j, KV, HD, len);
      for (unsigned d = 0; d < HD; ++d) tchain.add(rne(e.o[d]), fo[orow + d]);
      // (b) flash's arithmetic (fp32 scores and probabilities, one rounding at the end), taken
      // in fp64: where its bf16 output is not the reference's.
      const std::vector<double> fl = k2_ref::attention(qh, kc.data(), vc.data(), len, j, KV, HD);
      for (unsigned d = 0; d < HD; ++d) tflash.add(rne(float(fl[d])), fo[orow + d]);
    }
  std::printf("  %-7s %u row(s) x %u heads, %u keys: differing bf16 values - scores %zu/%zu, "
              "softmax %zu/%zu, P.V %zu/%zu (16-key blocks %zu), chain %zu/%zu; flash's arithmetic "
              "(fp64) %zu/%zu\n",
              c.name, c.T, H, c.P, ts.diff, ts.n, tp.diff, tp.n, to.diff, to.n, tdev.diff, tchain.diff,
              tchain.n, tflash.diff, tflash.n);
  CHECK_EQ(masked_bad, size_t(0));
  // (a) every stage bit for bit on this fixture (measured 2026-10-05, torch 2.14.1 AVX2: 0 of
  // 5,456 scores, 5,456 probabilities and 8,192 outputs over the three cases differ; flash's
  // arithmetic: 5,742 of the outputs). The softmax is torch's own and must stay exact on any
  // fixture; the scores of `coarse` too (exact dots). A fixture regenerated on another torch
  // could show
  // one-ulp ORDER flips in the fine cases' scores and P·V (k2_ref.h's eager section) - read
  // them before loosening anything here.
  CHECK_EQ(tp.diff, size_t(0));
  CHECK_EQ(ts.diff, size_t(0));
  CHECK_EQ(to.diff, size_t(0));
  CHECK_EQ(tdev.diff, size_t(0));
  CHECK_EQ(tchain.diff, size_t(0));
  // (b) flash's arithmetic is not the reference's: it lands elsewhere.
  CHECK(tflash.diff > 0);
}

// (b) by construction: dots 113.5 and 114 (bf16 values; q = e_0) scale to fp32 scores 10.0321
// and 10.0763, which both round to the bf16 score 10.0625. Eager weighs the two keys 1/2 each
// - o = 0.5 exactly with V rows 0 and 1 - while flash's fp32 scores weigh key 1 by
// e^0.0442 / (1 + e^0.0442) = 0.511: bf16 0.51171875, three bf16 ulps away.
void flash_vs_eager_two_keys() {
  const unsigned HD = 128;
  std::vector<float> q(HD, 0.0f);
  std::vector<uint16_t> k(2 * HD, 0), v(2 * HD, 0);
  q[0] = 1.0f;
  k[0] = rne(113.5f);
  k[HD] = rne(114.0f);
  CHECK(f32(k[0]) == 113.5f && f32(k[HD]) == 114.0f);
  for (unsigned d = 0; d < HD; ++d) v[HD + d] = rne(1.0f);   // key 0's V = 0, key 1's V = 1
  const k2_ref::EagerHead e = k2_ref::attention_eager(q.data(), k.data(), v.data(), 2, 0, 1, HD, 2);
  const std::vector<double> flash = k2_ref::attention(q.data(), k.data(), v.data(), 2, 0, 1, HD);
  std::printf("  two keys: eager scores %g %g -> p %g %g -> o %g; flash (fp64) o %.6f -> bf16 %g\n",
              e.s[0], e.s[1], e.p[0], e.p[1], f32(rne(e.o[0])), flash[0], f32(rne(float(flash[0]))));
  CHECK(e.s[0] == 10.0625f && e.s[1] == 10.0625f);
  CHECK(e.p[0] == 0.5f && e.p[1] == 0.5f);
  for (unsigned d = 0; d < HD; ++d) CHECK(rne(e.o[d]) == rne(0.5f));
  CHECK(rne(float(flash[0])) == rne(0.51171875f));
  CHECK_EQ(ulps(rne(e.o[0]), rne(float(flash[0]))), 3);
}

}  // namespace

int main() {
  std::printf("k2_attn_eager_ref_test: k2_ref.h's eager chain against torch (fixture) and flash\n");
  const Case cases[] = {
      {"coarse", true, fx::kCoarseSeed, fx::kCoarseT, fx::kCoarseP, fx::kCoarseS, fx::kCoarsePr, fx::kCoarseO},
      {"decode", false, fx::kDecodeSeed, fx::kDecodeT, fx::kDecodeP, fx::kDecodeS, fx::kDecodePr, fx::kDecodeO},
      {"prompt", false, fx::kPromptSeed, fx::kPromptT, fx::kPromptP, fx::kPromptS, fx::kPromptPr, fx::kPromptO},
  };
  for (const Case& c : cases) run_case(c);
  flash_vs_eager_two_keys();
  std::printf("k2_attn_eager_ref_test OK\n");
  return 0;
}
