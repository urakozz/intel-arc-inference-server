// Spec 26b Task 3 (host): the LiLiCorr host twin (tests/kernels/lilicorr_ref.h) against tools/oracle/dflash_ref.py's
// LiLiCorr form - the reference the D1 gate is measured against - on its own outputs (tests/loader/
// lilicorr_dflash_fixture.safetensors, tools/oracle/dflash_fixture.py --lilicorr: 26a's tiny drafter at narrow widths over
// a synthetic target), the drafter read by the loader's host half from the fixture's own checkpoint tensors.
//
//   1. NVFP4: the loader's decode of three linears equals the reference's nvfp4_decode (W' bitwise, g bitwise).
//   2. The projection (Reference numerics): fc (+ g) -> hidden_norm -> per layer k (k_norm, RoPE) and v at every position
//      0..39 against context_kv's planes; the anchor = the projected row (Review Focus 5).
//   3. draft() at every case (no context, the window's edge for rows 0 and 7, the ring's wrap, K 7 / 3 / 1, the mask id as
//      the anchor, sampled walks with the slot-keyed seed): the 8 final-norm rows, the 8 candidates (exact), their raw logits
//      and log-probs, the anchor, the lattice (|diff| printed - fp32 order differences), the walk (chosen / ids) and its
//      realized scores. Gates: rows cosine >= 0.999999, candidates and walks equal, the lattice and the anchor within 1e-5
//      (the lattice's |values| reach 8: logit_scale x a cosine), the logits and log-probs within 2e-6 x the largest |logit|.
//   4. Review Focus on the host: the twin with the sinks zeroed equals the reference's `no_sink` mutation and the twin fed
//      the anchor row of p - 2 its `anchor_p2` mutation (each far from the true lattice); the sink's term by itself (equal
//      scores: out = Σ v / (n + exp(sink - m))); the window per row; top-8's tie / NaN / -inf / huge-value rules and a
//      multi-chunk logsumexp (fixed chunk order, no NaN); K = 1..7 prefixes of one walk; the walk's slot keying.
//   5. Engine numerics (bf16 GEMV inputs, the bf16 ring, the device forms) for the three drafter formats with the bf16 and
//      the int8 head: candidate sets and walks against the reference (reported: the tiny random head's logits are flat, so
//      bf16 inputs reorder near-ties), rows cosine >= 0.9999 exact / 0.999 int8 / 0.9 int4 g64 (a 4-bit format
//      re-quantised into another) and the lattice cosine >= 0.999 where the candidates agree;
//      the real drafter's D1 is the box's (row 64).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/lilicorr_fixture_io.h"
#include "kernels/lilicorr_ref.h"

namespace {

std::string g_fixture = "tests/loader/lilicorr_dflash_fixture.safetensors";

float maxabs(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::fmax(m, std::fabs(x));
  return m;
}
double maxdiff(const float* a, const float* b, size_t n) {
  double d = 0;
  for (size_t i = 0; i < n; ++i) d = std::fmax(d, std::fabs(double(a[i]) - double(b[i])));
  return d;
}

struct Arm {
  const char* name;
  llfix::Form f;
  llfix::HeadForm hf;
  double rows_bar;   // the final-norm rows' cosine bar on these tiny random weights
};

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2) g_fixture = argv[1];
  const dffix::Fixture fx(g_fixture);
  const std::string dir = llfix::write_checkpoint(fx, "ref");
  const loader::LiliCheckpoint ck(dir);
  const model::DFlashDesc& d = ck.desc();
  CHECK(d.lilicorr() && d.hidden == 128 && d.lil_hidden == 64 && d.window[0] == 16);
  const uint32_t PB = 8;   // two context blocks over the 16-slot ring: the blocks' merge is exercised

  // --- 1. NVFP4 ----------------------------------------------------------------------------------------------------------
  for (const char* base : {"fc", "layers.1.mlp.down_proj", "lilicorr.layers.0.attn.out_proj"}) {
    const loader::LiliDecoded dc = ck.decoded(base);
    CHECK(dc.nvfp4 && fx.bf16(std::string("lil.nvfp4.") + base + ".wp") == dc.wp);
    CHECK(fx.f32(std::string("lil.nvfp4.") + base + ".g")[0] == dc.g);
  }
  std::printf("1. NVFP4: fc, layers.1.mlp.down_proj, lilicorr.layers.0.attn.out_proj decoded = nvfp4_decode (W', g bitwise)\n");

  const llref::Weights wr = llfix::weights(fx, ck, llfix::Form::Reference, llfix::HeadForm::Reference, PB);
  const llref::Dims& D = wr.d;
  const std::vector<float> rope = dfref::rope_table(D.theta, D.HD, 64);

  // --- 2. the projection -------------------------------------------------------------------------------------------------
  {
    const std::vector<uint16_t> taps = fx.bf16("taps");
    const std::vector<float> want_fc = fx.f32("lil.ctx.fc");
    double worst = 0, worst_a = 0;
    for (uint32_t pos = 0; pos < 40; ++pos) {
      dfref::Ring r = dfref::make_ring(D);
      const std::vector<float> a = llref::project(wr, dfref::Numerics::Reference, &taps[size_t(pos) * D.fck()], 1, pos, rope, r);
      worst_a = std::fmax(worst_a, maxdiff(a.data(), &want_fc[size_t(pos) * D.H], D.H));
      const uint32_t slot = pos % D.ring;
      for (uint32_t l = 0; l < D.layers; ++l) {
        const std::vector<float> wk = fx.f32("lil.ctx.k" + std::to_string(l)), wv = fx.f32("lil.ctx.v" + std::to_string(l));
        const float mk = maxabs(wk), mv = maxabs(wv);
        for (uint32_t i = 0; i < D.kvn(); ++i) {
          const size_t at = (size_t(l) * D.ring + slot) * D.kvn() + i;
          worst = std::fmax(worst, std::fmax(std::fabs(r.k[at] - wk[size_t(pos) * D.kvn() + i]) / mk,
                                             std::fabs(r.v[at] - wv[size_t(pos) * D.kvn() + i]) / mv));
        }
      }
    }
    std::printf("2. projection: 40 positions x %u layers, k / v worst |diff| / max %.2e; the anchor rows' max |diff| %.2e\n",
                D.layers, worst, worst_a);
    CHECK(worst <= 2e-5 && worst_a <= 1e-5);
  }

  // --- 3. draft() against the reference -----------------------------------------------------------------------------------
  const std::vector<dffix::Case> cs = dffix::cases(fx);
  double worst_lat = 0, worst_lp = 0;
  for (size_t ci = 0; ci < cs.size(); ++ci) {
    const dffix::Case& c = cs[ci];
    const std::string pre = "lil.case" + std::to_string(ci) + ".";
    const llfix::RingAt ra = llfix::ring_at(fx, wr, dfref::Numerics::Reference, rope, c.p);
    const llref::DraftOut o = llref::draft(wr, dfref::Numerics::Reference, ra.ring, rope, ra.anchor, ra.valid, c.anchor,
                                           c.p, c.K, c.T, c.seed);
    const std::vector<float> hidden = fx.f32(pre + "hidden"), lp = fx.f32(pre + "lp"), lg = fx.f32(pre + "logit"),
                             lat = fx.f32(pre + "lattice"), anc = fx.f32(pre + "anchor"), sc = fx.f32(pre + "scores");
    const std::vector<int32_t> cand = fx.i32(pre + "cand"), chosen = fx.i32(pre + "chosen"), ids = fx.i32(pre + "ids");
    double hc = 1;
    for (uint32_t r = 0; r < D.M; ++r)
      hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * D.H], &hidden[size_t(r) * D.H], D.H));
    bool cands = true, walk = true;
    for (size_t i = 0; i < cand.size(); ++i) cands &= o.cand[i] == uint32_t(cand[i]);
    for (uint32_t s = 0; s < c.K; ++s) walk &= o.walk.chosen[s] == uint32_t(chosen[s]) && o.walk.ids[s] == uint32_t(ids[s]);
    const double dlat = maxdiff(o.lattice.data(), lat.data(), lat.size()), dlp = maxdiff(o.lp.data(), lp.data(), lp.size()),
                 dlg = maxdiff(o.logit.data(), lg.data(), lg.size()), danc = maxdiff(ra.anchor.data(), anc.data(), D.H),
                 dsc = maxdiff(o.walk.scores.data(), sc.data(), sc.size());
    std::printf("  case %zu (p %u, K %u, T %.1f): rows cos %.9f, cand %s, logit %.1e, lp %.1e, anchor %.1e, lattice %.2e "
                "(max |v| %.2f), walk %s, scores %.1e\n",
                ci, c.p, c.K, double(c.T), hc, cands ? "=" : "DIFF", dlg, dlp, danc, dlat, double(maxabs(lat)),
                walk ? "=" : "DIFF", dsc);
    CHECK(hc >= 0.999999 && cands && walk);
    const double ml = double(maxabs(lg));   // the logits' scale: fp32 order differences are relative to it
    CHECK(dlp <= 2e-6 * ml && dlat <= 1e-5 && danc <= 1e-5 && dlg <= 2e-6 * ml);
    CHECK(ra.valid == (c.p > 0));
    worst_lat = std::fmax(worst_lat, dlat);
    worst_lp = std::fmax(worst_lp, dlp);
  }
  std::printf("3. draft: %zu cases - candidates and walks equal, the lattice within %.2e, the log-probs within %.2e\n",
              cs.size(), worst_lat, worst_lp);

  // --- 4. Review Focus on the host ----------------------------------------------------------------------------------------
  {
    const dffix::Case& c = cs[3];
    const std::vector<float> truth = fx.f32("lil.case3.lattice");
    llref::Weights ns = wr;
    for (llref::Layer& L : ns.layers) std::fill(L.sink.begin(), L.sink.end(), -INFINITY);   // exp(-inf - m) = 0: no sink
    llfix::RingAt ra = llfix::ring_at(fx, ns, dfref::Numerics::Reference, rope, c.p);
    const llref::DraftOut a = llref::draft(ns, dfref::Numerics::Reference, ra.ring, rope, ra.anchor, ra.valid, c.anchor, c.p, 7);
    const std::vector<float> m1 = fx.f32("lil.mut.no_sink.lattice");
    const double d1 = maxdiff(a.lattice.data(), m1.data(), m1.size()), t1 = maxdiff(truth.data(), m1.data(), m1.size());
    CHECK(d1 <= 1e-5 && t1 > 1e-2);
    // the anchor at p - 2
    const llfix::RingAt r2 = llfix::ring_at(fx, wr, dfref::Numerics::Reference, rope, c.p - 1);
    const llfix::RingAt rp = llfix::ring_at(fx, wr, dfref::Numerics::Reference, rope, c.p);
    const llref::DraftOut b = llref::draft(wr, dfref::Numerics::Reference, rp.ring, rope, r2.anchor, true, c.anchor, c.p, 7);
    const std::vector<float> m2 = fx.f32("lil.mut.anchor_p2.lattice");
    const double d2 = maxdiff(b.lattice.data(), m2.data(), m2.size()), t2 = maxdiff(truth.data(), m2.data(), m2.size());
    CHECK(d2 <= 1e-5 && t2 > 1e-2);
    std::printf("4a. no sink: the twin = the reference's mutation within %.1e (the true lattice %.2f away); the anchor at "
                "p - 2: within %.1e (%.2f away)\n", d1, t1, d2, t2);
    // the sink's term alone: q . k = 0 everywhere -> every score 0, m = 0, out = Σ v / (n + exp(sink))
    {
      dfref::AttnDims ad{2, 1, 4, 3, 8, 8, 4, 0.5f};
      std::vector<float> q(3 * 2 * 4, 0.0f), kb(3 * 4, 0.0f), vb(3 * 4, 0.0f), rk(8 * 4, 0.0f), rv(8 * 4, 0.0f), out(3 * 8);
      const uint32_t p = 20;   // the ring holds 12..19; row 0 sees >= 13 (7 keys), row 2 sees >= 15 (5)
      for (uint32_t pos = 12; pos < 20; ++pos)
        for (uint32_t i = 0; i < 4; ++i) rv[(pos % 8) * 4 + i] = float(pos);
      const float sink[2] = {0.75f, -1.25f};
      llref::attention_sink(ad, q.data(), kb.data(), vb.data(), rk.data(), rv.data(), p, 2, sink, out.data());
      auto want = [](uint32_t lo, float s) {
        double v = 0;
        for (uint32_t x = lo; x < 20; ++x) v += x;
        return v / (double(20 - lo) + 3.0 + std::exp(double(s)));   // + the block's 3 keys (values 0)
      };
      CHECK_NEAR(out[0], want(13, 0.75f), 1e-5);
      CHECK_NEAR(out[4], want(13, -1.25f), 1e-5);
      CHECK_NEAR(out[2 * 8], want(15, 0.75f), 1e-5);
      CHECK(out[3 * 8 - 1] == out[2 * 8 + 4]);
    }
    // top-8 + the logsumexp: ties to the lower id, NaN never, a -inf chunk, a huge value, several chunks in order
    {
      const uint32_t V = 3 * llref::kTopkChunk + 100;
      std::vector<float> lg(V);
      for (uint32_t i = 0; i < V; ++i) lg[i] = std::sin(float(i) * 0.37f) * 4.0f;
      for (uint32_t i = llref::kTopkChunk; i < 2 * llref::kTopkChunk; ++i) lg[i] = -INFINITY;   // a chunk of -inf
      lg[5] = 9.0f;
      lg[7] = 9.0f;                    // a tie: 5 before 7
      lg[11] = NAN;                    // never a candidate, not in the sum
      const llref::TopK t = llref::topk_lse(lg.data(), V, 8);
      CHECK(t.id[0] == 5 && t.id[1] == 7 && t.v[0] == 9.0f);
      double s = 0, M = 9.0;
      for (uint32_t i = 0; i < V; ++i)
        if (!std::isnan(lg[i]) && lg[i] != -INFINITY) s += std::exp(double(lg[i]) - M);
      CHECK_NEAR(t.lse, M + std::log(s), 1e-5);
      for (float x : t.lp) CHECK(std::isfinite(x));
      std::vector<float> hg(V, 0.0f);
      hg[3] = 3.0e38f;                 // huge, finite: lse = that value, its lp 0, the rest -huge, no NaN
      const llref::TopK u = llref::topk_lse(hg.data(), V, 8);
      CHECK(u.id[0] == 3 && u.lp[0] == 0.0f && std::isfinite(u.lse));
      for (float x : u.lp) CHECK(!std::isnan(x));
    }
    // K = 1..7 are prefixes of one walk (one block); the slot keying moves a sampled walk
    {
      const dffix::Case& s = cs[8];
      const llfix::RingAt rs = llfix::ring_at(fx, wr, dfref::Numerics::Reference, rope, s.p);
      const llref::DraftOut full = llref::draft(wr, dfref::Numerics::Reference, rs.ring, rope, rs.anchor, true, s.anchor, s.p, 7);
      for (uint32_t K = 1; K < 7; ++K) {
        const llref::Walk w = llref::walk(full.lattice, full.cand.data(), 8, s.p, K, 0.0f, 0);
        for (uint32_t j = 0; j < K; ++j) CHECK(w.ids[j] == full.walk.ids[j]);
      }
      uint32_t moved = 0;
      for (uint64_t seed = 1; seed < 64; ++seed) {
        const llref::Walk a1 = llref::walk(full.lattice, full.cand.data(), 8, s.p, 7, 1.0f, seed << 32);
        const llref::Walk a2 = llref::walk(full.lattice, full.cand.data(), 8, s.p, 7, 1.0f, (seed << 32) ^ (1ull << 32));
        moved += a1.ids != a2.ids;
      }
      CHECK(moved > 0);
      std::printf("4b. the sink's term alone; top-8 ties / NaN / -inf chunk / huge value / 3 chunks; K 1..6 prefixes of "
                  "the K 7 walk; the slot keying moves %u of 63 sampled walks\n", moved);
    }
  }

  // --- 5. Engine numerics -------------------------------------------------------------------------------------------------
  {
    // int4 g64 re-quantises a 4-bit format into another (groups of 64 over four NVFP4 blocks of 16, each its own scale):
    // 26a measured ~11 % weight error on the real drafter; the tiny drafter's codes are uniform over e2m1, worse.
    const Arm arms[] = {{"exact + bf16 head", llfix::Form::Exact, llfix::HeadForm::Bf16, 0.9999},
                        {"exact + int8 head", llfix::Form::Exact, llfix::HeadForm::Int8, 0.9999},
                        {"int8 + int8 head", llfix::Form::Int8, llfix::HeadForm::Int8, 0.999},
                        {"int4 g64 + int8 head", llfix::Form::Int4, llfix::HeadForm::Int8, 0.9}};
    for (const Arm& arm : arms) {
      const llref::Weights we = llfix::weights(fx, ck, arm.f, arm.hf, PB);
      double hc = 1, lc = 1;
      int sets = 0, walks = 0, n = 0, nsame = 0;
      for (size_t ci = 0; ci < cs.size(); ++ci) {
        const dffix::Case& c = cs[ci];
        if (c.T != 0.0f) continue;
        const std::string pre = "lil.case" + std::to_string(ci) + ".";
        const llfix::RingAt ra = llfix::ring_at(fx, we, dfref::Numerics::Engine, rope, c.p);
        const llref::DraftOut o = llref::draft(we, dfref::Numerics::Engine, ra.ring, rope, ra.anchor, ra.valid, c.anchor,
                                               c.p, c.K);
        const std::vector<float> hidden = fx.f32(pre + "hidden"), lat = fx.f32(pre + "lattice");
        const std::vector<int32_t> cand = fx.i32(pre + "cand"), ids = fx.i32(pre + "ids");
        for (uint32_t r = 0; r < D.M; ++r)
          hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * D.H], &hidden[size_t(r) * D.H], D.H));
        bool same = true;   // the lattice of other candidates is another lattice: compared where the candidates agree
        for (size_t i = 0; i < cand.size(); ++i) same &= o.cand[i] == uint32_t(cand[i]);
        if (same) {
          lc = std::fmin(lc, dfref::cosine(o.lattice.data(), lat.data(), lat.size()));
          ++nsame;
        }
        bool st = true, wk = true;
        for (uint32_t s = 0; s < D.S; ++s) {
          std::vector<uint32_t> a(o.cand.begin() + s * 8, o.cand.begin() + (s + 1) * 8), b(8);
          for (uint32_t i = 0; i < 8; ++i) b[i] = uint32_t(cand[s * 8 + i]);
          std::sort(a.begin(), a.end());
          std::sort(b.begin(), b.end());
          st &= a == b;
        }
        for (uint32_t s = 0; s < c.K; ++s) wk &= o.walk.ids[s] == uint32_t(ids[s]);
        sets += st;
        walks += wk;
        ++n;
      }
      std::printf("5. Engine numerics, %s: rows cos >= %.6f, candidate sets %d / %d, the lattice (%d cases with the "
                  "reference's candidates) cos >= %.6f, walks %d / %d\n",
                  arm.name, hc, sets, n, nsame, lc, walks, n);
      CHECK(hc >= arm.rows_bar && lc >= 0.999);   // lc stays 1 when no case kept the reference's candidates (reported)
    }
  }
  std::filesystem::remove_all(dir);
  std::printf("lilicorr_ref_test OK\n");
  return 0;
}
