// Spec 23c Task 1: tests/kernels/muse_ref.h (the Muse-Glimmer decode kernels' host twin) against transformers
// 5.19.0's own muse_glimmer modules (tests/kernels/muse_fixture.h, tools/oracle/muse_fixture.py) and against the
// semantics' properties (plan 23c Review Focus 1-5). Host only (the RoPE table is runtime::muse::rope_table, the
// engine's own host builder).
//
//   (a) the normed embedding: coarse row bitwise, fine row within 1 ulp (Σx²'s order), counted
//   (b) the centered norms: coarse rows bitwise at eps 1e-5 (w = 0, random) and 1e-8; fine within 1 ulp; Kolibri's
//       form (x̂ rounded first, then the weight) and a plain-w form both differ from torch (Review Focus 1)
//   (c) the final norm: plain w with pow - coarse bitwise, fine within 1 ulp; the (1 + w) form differs
//   (d) post_add: the sandwich's add - coarse bitwise, fine within 1 ulp; its Σ² is prep_res_fold's tree over the
//       new residual; a zero sub-block leaves the residual unchanged
//   (e) q / k prep: NoPE q and k bitwise; RoPE'd at 0 .. 131071 bitwise; the table's cos / sin rows bitwise; q's
//       two roundings (a one-rounding x 3.87 differs: Review Focus 2); NoPE rows are position-free (Focus 5)
//   (f) the gate product and (g) SiLU x up bitwise
//   (h) eager attention: scores and probabilities bitwise, outputs within 1 ulp (P·V's fp32 order), counted;
//       fp64 within cosine 0.9999 (the bf16 chain at 2048+ keys; flash holds 0.99999); the gated output rne(rne(o) x rne(sigmoid(g))); the window's edge at
//       2046 / 2047 / 2048 / 4095 / 4096 / 4097 / 9000 and the ring's rows (Review Focus 3); ring = linear cache
//   (i) the softcap: the fp32 form within 2 bf16 ulps of the reference's four roundings, monotone
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "kernels/muse_fixture.h"
#include "kernels/muse_ref.h"
#include "runtime/muse/muse_rope.h"

namespace {

namespace fx = muse_fixture;
namespace mr = museref;
using mr::bf;
using mr::rf;
using mr::rne;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

std::vector<uint16_t> row(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step, float amp, uint32_t base = 0) {
  std::vector<uint16_t> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = rne(mr::fixture_val(coarse, t, base + i, seed, step, amp));
  return v;
}

// Bitwise, or (bar > 0) within `bar` ulps with the count printed.
void expect(const char* what, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want, int bar) {
  CHECK_EQ(got.size(), want.size());
  size_t diff = 0;
  int worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int u = ulps16(got[i], want[i]);
    worst = u > worst ? u : worst;
    diff += u != 0;
  }
  std::printf("  %-44s %zu values, %zu differ (worst %d ulp, bar %d)\n", what, got.size(), diff, worst, bar);
  if (worst > bar) std::fprintf(stderr, "FAIL: %s\n", what);
  CHECK(worst <= bar);
}
bool differs(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { return a != b; }

const uint32_t H = mr::kHidden, HD = mr::kHd;

void test_embed() {
  const std::vector<uint16_t> c = row(true, 0, H, 1, 0.125f, 1.f), f = row(false, 0, H, 2, 0.125f, 4.f);
  std::vector<uint16_t> oc(H), of(H);
  mr::embed_norm(c.data(), H, oc.data());
  mr::embed_norm(f.data(), H, of.data());
  expect("embed norm, coarse row", oc, mr::hex16(fx::kEmbedCoarse), 0);
  expect("embed norm, fine row", of, mr::hex16(fx::kEmbedFine), 1);
  // the residual it writes and its per-chunk Σ² (what layer 0's mg_norm reads)
  std::vector<uint16_t> r(H);
  std::vector<float> ss(mr::kG), ss2(mr::kG);
  mr::embed_norm_sums(c.data(), H, r.data(), ss.data());
  CHECK(r == oc);
  mr::chunk_sumsq(r.data(), H, mr::kG, ss2.data());
  CHECK(ss == ss2);
}

void test_centered() {
  const std::vector<uint16_t> xc = row(true, 0, H, 1, 0.125f, 1.f), xf = row(false, 0, H, 2, 0.125f, 4.f);
  std::vector<float> wr(H), w0(H, 0.f);
  for (uint32_t k = 0; k < H; ++k) wr[k] = rf(mr::fixture_val(false, 1, k, 1, 0.125f, 0.5f));
  std::vector<uint16_t> o(H);
  mr::centered_norm(xc.data(), w0.data(), mr::kEps, H, o.data());
  expect("centered norm eps 1e-5, w = 0, coarse", o, mr::hex16(fx::kCenteredCoarseZero), 0);
  mr::centered_norm(xc.data(), wr.data(), mr::kEps, H, o.data());
  expect("centered norm eps 1e-5, random w, coarse", o, mr::hex16(fx::kCenteredCoarseRand), 0);
  mr::centered_norm(xc.data(), wr.data(), mr::kPostEps, H, o.data());
  expect("centered norm eps 1e-8, random w, coarse", o, mr::hex16(fx::kCenteredCoarseRandPost), 0);
  CHECK(differs(mr::hex16(fx::kCenteredCoarseRand), mr::hex16(fx::kCenteredCoarseRandPost)));   // eps is seen
  mr::centered_norm(xf.data(), wr.data(), mr::kEps, H, o.data());
  expect("centered norm eps 1e-5, random w, fine", o, mr::hex16(fx::kCenteredFineRand), 1);
  // Review Focus 1: Kolibri's chain (x̂ rounded to bf16, then x (1 + w)) and a plain-w chain are both wrong here.
  std::vector<uint16_t> kol(H), plain(H);
  mr::centered_norm_kolibri_form(xf.data(), wr.data(), mr::kEps, H, kol.data());
  mr::final_norm_eps(xf.data(), wr.data(), mr::kEps, H, plain.data());
  CHECK(differs(kol, mr::hex16(fx::kCenteredFineRand)));
  CHECK(differs(plain, mr::hex16(fx::kCenteredFineRand)));
  std::printf("  Kolibri's two-rounding form and a plain-w form both differ from torch (Review Focus 1)\n");
}

void test_final() {
  const std::vector<uint16_t> xc = row(true, 0, H, 1, 0.125f, 1.f), xf = row(false, 0, H, 2, 0.125f, 4.f);
  std::vector<float> w(H);
  for (uint32_t k = 0; k < H; ++k) w[k] = rf(1.0f + mr::fixture_val(false, 2, k, 1, 0.125f, 0.5f));
  std::vector<uint16_t> o(H);
  mr::final_norm(xc.data(), w.data(), H, o.data());
  expect("final norm (plain w, pow), coarse", o, mr::hex16(fx::kFinalCoarse), 0);
  mr::final_norm(xf.data(), w.data(), H, o.data());
  expect("final norm (plain w, pow), fine", o, mr::hex16(fx::kFinalFine), 1);
  mr::centered_norm(xc.data(), w.data(), mr::kEps, H, o.data());
  CHECK(differs(o, mr::hex16(fx::kFinalCoarse)));
}

void test_post_add() {
  const std::vector<uint16_t> xc = row(true, 0, H, 1, 0.125f, 1.f), xf = row(false, 0, H, 2, 0.125f, 4.f);
  const std::vector<uint16_t> r0 = row(false, 3, H, 4, 0.125f, 2.f);
  std::vector<float> wr(H);
  for (uint32_t k = 0; k < H; ++k) wr[k] = rf(mr::fixture_val(false, 1, k, 1, 0.125f, 0.5f));
  std::vector<uint16_t> r = r0;
  mr::post_add(r.data(), xc.data(), wr.data(), H);
  expect("post_add (eps 1e-8, one add), coarse sub", r, mr::hex16(fx::kPostAddCoarse), 0);
  r = r0;
  mr::post_add(r.data(), xf.data(), wr.data(), H);
  expect("post_add (eps 1e-8, one add), fine sub", r, mr::hex16(fx::kPostAddFine), 1);
  // the device form: the sub-block's chunk sums in, the new residual's chunk sums out
  std::vector<float> sa(mr::kG), so(mr::kG), want(mr::kG);
  mr::chunk_sumsq(xc.data(), H, mr::kG, sa.data());
  std::vector<uint16_t> r2 = r0, r3 = r0;
  mr::post_add_sums(sa.data(), mr::kG, xc.data(), wr.data(), r2.data(), so.data(), H);
  mr::post_add(r3.data(), xc.data(), wr.data(), H);
  CHECK(r2 == r3);
  mr::chunk_sumsq(r2.data(), H, mr::kG, want.data());
  CHECK(so == want);
  // a zero sub-block contributes exact zeros (eps 1e-8 keeps rstd finite)
  std::vector<uint16_t> z(H, 0), r4 = r0;
  mr::post_add(r4.data(), z.data(), wr.data(), H);
  CHECK(r4 == r0);
}

void test_qk_prep() {
  const std::vector<uint16_t> q = row(true, 4, mr::kQN, 5, 0.25f, 1.f), k = row(true, 5, mr::kKvN, 5, 0.25f, 1.f);
  std::vector<uint16_t> qo(mr::kQN), ko(mr::kKvN);
  const uint32_t maxp = fx::kRopePos[fx::kRopeN - 1] + 1;
  const std::vector<float> tab = runtime::muse::rope_table(HD, fx::kRopeTheta, maxp);
  // the table's rows against the rotary module's bf16 cos / sin
  std::vector<uint16_t> cs, sn;
  for (uint32_t i = 0; i < fx::kRopeN; ++i)
    for (uint32_t d = 0; d < HD / 2; ++d) {
      cs.push_back(rne(tab[(size_t(fx::kRopePos[i]) * 2 + 0) * (HD / 2) + d]));
      sn.push_back(rne(tab[(size_t(fx::kRopePos[i]) * 2 + 1) * (HD / 2) + d]));
    }
  expect("RoPE table cos rows (theta 5e5)", cs, mr::hex16(fx::kRopeCos), 0);
  expect("RoPE table sin rows (theta 5e5)", sn, mr::hex16(fx::kRopeSin), 0);
  mr::qk_prep(q.data(), k.data(), 5, false, tab.data(), qo.data(), ko.data());
  expect("q NoPE (norm, x 3.87, two roundings)", qo, mr::hex16(fx::kQNope), 0);
  expect("k NoPE (norm)", ko, mr::hex16(fx::kKNope), 0);
  // Review Focus 5: a NoPE row does not depend on its position.
  std::vector<uint16_t> qo2(mr::kQN), ko2(mr::kKvN);
  mr::qk_prep(q.data(), k.data(), 9000, false, tab.data(), qo2.data(), ko2.data());
  CHECK(qo2 == qo && ko2 == ko);
  // Review Focus 2: q x 3.87 with ONE rounding (bf16(x * rstd * 3.87)) is not torch's.
  std::vector<uint16_t> one(mr::kQN);
  mr::q_prep_one_rounding(q.data(), one.data());
  CHECK(differs(one, mr::hex16(fx::kQNope)));
  std::vector<uint16_t> gq, gk;
  for (uint32_t i = 0; i < fx::kRopeN; ++i) {
    mr::qk_prep(q.data(), k.data(), fx::kRopePos[i], true, tab.data(), qo.data(), ko.data());
    gq.insert(gq.end(), qo.begin(), qo.begin() + HD);                                 // q head 0
    gq.insert(gq.end(), qo.begin() + size_t(mr::kQHeads - 1) * HD, qo.end());          // q head 31
    gk.insert(gk.end(), ko.begin(), ko.end());                                        // k heads 0, 1
  }
  expect("q RoPE'd at 0 .. 131071 (heads 0, 31)", gq, mr::hex16(fx::kQRoped), 0);
  expect("k RoPE'd at 0 .. 131071 (heads 0, 1)", gk, mr::hex16(fx::kKRoped), 0);
}

void test_gate_silu() {
  const std::vector<uint16_t> a = row(false, 6, mr::kQN, 7, 0.125f, 2.f), g = row(false, 7, mr::kQN, 7, 0.125f, 8.f);
  std::vector<uint16_t> o(mr::kQN);
  for (uint32_t i = 0; i < mr::kQN; ++i) o[i] = mr::gate_product(a[i], mr::gate_bf16(g[i]));
  expect("gate attn x rne(sigmoid(g))", o, mr::hex16(fx::kGate), 0);
  const std::vector<uint16_t> sg = row(false, 8, fx::kInterSmall, 9, 0.125f, 8.f),
                              su = row(false, 9, fx::kInterSmall, 9, 0.125f, 2.f);
  std::vector<uint16_t> so(fx::kInterSmall);
  mr::silu_mul(sg.data(), su.data(), fx::kInterSmall, so.data());
  expect("silu(g) x u", so, mr::hex16(fx::kSiluMul), 0);
}

// The caches of the fixture's attention cases: key p's k / v at index (p x 2 + j) x 128 + d, coarse 1/8.
struct Cache {
  std::vector<uint16_t> k, v;
};
Cache cache(uint32_t lo, uint32_t hi, bool ring, uint32_t rows) {
  Cache c;
  c.k.assign(size_t(rows) * mr::kKvN, 0);
  c.v.assign(size_t(rows) * mr::kKvN, 0);
  for (uint32_t p = lo; p <= hi; ++p) {
    const size_t r = ring ? (p & (mr::kRing - 1)) : p;
    for (uint32_t i = 0; i < mr::kKvN; ++i) {
      c.k[r * mr::kKvN + i] = rne(mr::fixture_val(true, 21, p * mr::kKvN + i, 9, 0.125f, 1.f));
      c.v[r * mr::kKvN + i] = rne(mr::fixture_val(true, 22, p * mr::kKvN + i, 9, 0.125f, 1.f));
    }
  }
  return c;
}

void test_attention() {
  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const fx::AttnCase& c = fx::kAttn[ci];
    const bool sliding = c.sliding != 0;
    const std::vector<uint16_t> q = row(true, 20, mr::kQN, ci, 0.5f, 1.f);
    const uint32_t rows = sliding ? mr::kRing : c.query + 1;
    const Cache kv = cache(c.lo, c.query, sliding, rows);
    CHECK_EQ(mr::key_lo(c.query, sliding), c.query + 1 - c.visible);
    const mr::EagerHead e0 = mr::attention_eager_head(q.data(), kv.k.data(), kv.v.data(), c.query, sliding, 0);
    std::vector<uint16_t> s, p;
    for (float x : e0.s) s.push_back(rne(x));
    for (float x : e0.p) p.push_back(rne(x));
    expect("eager scores, head 0", s, mr::hex16(c.s), 0);
    expect("eager probabilities, head 0", p, mr::hex16(c.p), 0);
    std::vector<uint16_t> o(mr::kQN);
    mr::attention_eager(q.data(), kv.k.data(), kv.v.data(), c.query, sliding, mr::kRing, nullptr, o.data());
    expect("eager output, every head (P.V order)", o, mr::hex16(c.o), 1);
    std::vector<double> f(mr::kQN);
    mr::attention_fp64(q.data(), kv.k.data(), kv.v.data(), c.query, sliding, mr::kRing, nullptr, f.data());
    double dot = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < mr::kQN; ++i) {
      dot += f[i] * bf(o[i]);
      na += f[i] * f[i];
      nb += double(bf(o[i])) * bf(o[i]);
    }
    const double cosv = dot / std::sqrt(na * nb);
    std::printf("  case %u (query %u, %s): fp64 vs eager cosine %.8f\n", ci, c.query, sliding ? "sliding" : "full", cosv);
    CHECK(cosv >= 0.9999);   // eager rounds scores and probabilities to bf16 (flash: 0.99999, muse_run)
    // the gate applied after the attention (Review Focus 4's product): rne(rne(o) x g_bf16)
    const std::vector<uint16_t> g = row(false, 30, mr::kQN, ci, 0.125f, 4.f);
    std::vector<uint16_t> gb(mr::kQN), og(mr::kQN);
    for (uint32_t i = 0; i < mr::kQN; ++i) gb[i] = mr::gate_bf16(g[i]);
    mr::attention_eager(q.data(), kv.k.data(), kv.v.data(), c.query, sliding, mr::kRing, gb.data(), og.data());
    for (uint32_t i = 0; i < mr::kQN; ++i) CHECK_EQ(og[i], mr::gate_product(o[i], gb[i]));
  }
  // Review Focus 3: the window's edge and the ring's rows.
  for (uint32_t pos : {0u, 1u, 2046u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 9000u}) {
    const uint32_t lo = mr::key_lo(pos, true);
    CHECK_EQ(lo, pos >= 2047 ? pos - 2047 : 0u);
    CHECK_EQ(pos - lo + 1, pos >= 2047 ? 2048u : pos + 1);
    CHECK_EQ(mr::key_lo(pos, false), 0u);
    CHECK_EQ(mr::ring_row(pos, true), pos & 4095u);
    CHECK_EQ(mr::ring_row(pos, false), pos);
  }
  // ring addressing == a linear cache, bitwise, across the ring's wrap (query 9000: keys 6953..9000)
  const std::vector<uint16_t> q = row(true, 20, mr::kQN, 7, 0.5f, 1.f);
  for (uint32_t pos : {2046u, 2047u, 2048u, 4095u, 4096u, 4097u, 9000u}) {
    const uint32_t lo = mr::key_lo(pos, true);
    const Cache ring = cache(lo, pos, true, mr::kRing), lin = cache(lo, pos, false, pos + 1);
    std::vector<uint16_t> a(mr::kQN), b(mr::kQN);
    mr::attention_eager(q.data(), ring.k.data(), ring.v.data(), pos, true, mr::kRing, nullptr, a.data());
    mr::attention_eager(q.data(), lin.k.data(), lin.v.data(), pos, true, 0, nullptr, b.data());
    CHECK(a == b);
  }
  std::printf("  the window [p - 2047, p] at 0 .. 9000, ring rows p & 4095, ring == linear bitwise (Review Focus 3)\n");
}

void test_softcap() {
  const std::vector<uint16_t> z = row(false, 10, 64, 11, 0.125f, 64.f);
  std::vector<float> zf(64), c(64);
  for (uint32_t i = 0; i < 64; ++i) zf[i] = bf(z[i]);
  mr::softcap_row(zf.data(), 64, c.data());
  std::vector<uint16_t> cb(64);
  for (uint32_t i = 0; i < 64; ++i) cb[i] = rne(c[i]);
  expect("softcap fp32 vs the four bf16 roundings", cb, mr::hex16(fx::kSoftcap), 2);
  for (uint32_t i = 0; i < 64; ++i)
    for (uint32_t j = 0; j < 64; ++j)
      if (zf[i] < zf[j]) CHECK(c[i] <= c[j]);
}

}  // namespace

int main() {
  std::printf("muse_ref_test: tests/kernels/muse_ref.h against transformers 5.19.0 (muse_fixture.h)\n");
  test_embed();
  test_centered();
  test_final();
  test_post_add();
  test_qk_prep();
  test_gate_silu();
  test_attention();
  test_softcap();
  std::printf("muse_ref_test: OK\n");
  return 0;
}
