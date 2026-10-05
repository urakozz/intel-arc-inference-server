// pf_moe_run - src/kernels/prefill/pf_moe.cl (spec 15d: the prefill MoE block's sort,
// gathers, expert dequant and combine) on the Mac's OpenCL GPU against
// tests/kernels/pf_moe_ref.h (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   pf_moe_run
//
// Everything here is integer bookkeeping, copies or a fixed chain of roundings with no
// `exp`, so the bar is exact equality:
//   1. pf_moe_sort at Ornith's shape (256 experts, top-8, TM 32, KC 2048) on random routes
//      at C = 1, 37, 2048, on a skewed one (every token to experts 0..7) and on the
//      tile bound's adversary: the header, the WHOLE padded tile table, pair_row and every
//      used row's token, against the host walk;
//   2. pf_moe_gather and pf_moe_gather_i8 on the C = 37 sort: every used row (padding rows
//      zero) and the gathered scales;
//   3. at a small shape (16 experts, top-4, hidden 256, inter 64): pf_moe_dequant_gu / _dn
//      over a batch of blocks with an empty expert (skipped: its slice keeps the buffer's
//      zeros) against the host dequant, and pf_moe_combine against the host combine.
// pf_moe_gemm.cl (DPAS, 2D block I/O) cannot run here; the box's pf_moe_test runs it.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "kernels/pf_moe_ref.h"

namespace {

using Words = std::vector<uint32_t>;

std::vector<std::string> defines(const moe_ref::Shape& s, uint32_t kc) {
  return {"EXPERTS=" + std::to_string(s.experts), "TOP_K=" + std::to_string(s.top_k),
          "HIDDEN=" + std::to_string(s.hidden), "INTER=" + std::to_string(s.inter),
          "TM=" + std::to_string(pf_moe_ref::kTm), "KC=" + std::to_string(kc)};
}

Words random_routes(uint32_t C, const moe_ref::Shape& s, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> all(s.experts);
  std::iota(all.begin(), all.end(), 0u);
  Words r(size_t(C) * moe_ref::kWords, 0);
  for (uint32_t t = 0; t < C; ++t) {
    std::shuffle(all.begin(), all.end(), rng);
    for (uint32_t k = 0; k < s.top_k; ++k) r[size_t(t) * moe_ref::kWords + k] = all[k];
  }
  return r;
}

int g_bad = 0;
void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("  MISMATCH: %s\n", what);
    ++g_bad;
  }
}

struct DevSort {
  Words hdr, tiles, row_tok, pair_row;
};

DevSort run_sort(clrun::Device& dev, clrun::Program& prog, const Words& route, uint32_t C,
                 const moe_ref::Shape& s) {
  const uint32_t T = pf_moe_ref::tmax(s, C);
  clrun::Buffer br(dev, route), bh(dev, kernels::pf_moe::hdr_words(s.experts) * 4),
      bt(dev, size_t(T) * 2 * 4), brt(dev, size_t(T) * pf_moe_ref::kTm * 4),
      bp(dev, size_t(C) * s.top_k * 4);
  prog.run("pf_moe_sort", {s.experts}, {s.experts}, br, bh, bt, brt, bp, C, T);
  return {bh.read<uint32_t>(), bt.read<uint32_t>(), brt.read<uint32_t>(), bp.read<uint32_t>()};
}

void check_sort(clrun::Device& dev, clrun::Program& prog, const Words& route, uint32_t C,
                const moe_ref::Shape& s, const char* what) {
  const pf_moe_ref::Sorted want = pf_moe_ref::sort(route.data(), C, s);
  const DevSort got = run_sort(dev, prog, route, C, s);
  expect(got.hdr == want.hdr, "pf_moe_sort header");
  expect(got.tiles == want.tiles, "pf_moe_sort tile table (padded to tmax)");
  expect(got.pair_row == want.pair_row, "pf_moe_sort pair_row");
  expect(std::equal(want.row_tok.begin(), want.row_tok.begin() + want.rows_used(), got.row_tok.begin()),
         "pf_moe_sort row_tok (used rows)");
  std::printf("  sort %-22s C %4u: %3u tiles of %3u, %5u rows - %s\n", what, C, want.tiles_used(),
              pf_moe_ref::tmax(s, C), want.rows_used(), g_bad ? "see above" : "exact");
}

}  // namespace

int main() {
  try {
    clrun::Device dev;
    std::printf("pf_moe_run on %s\n", dev.name().c_str());
    // ---- 1, 2: Ornith's shape --------------------------------------------------------
    const moe_ref::Shape o = moe_ref::ornith_shape();
    clrun::Program po(dev, "src/kernels/prefill/pf_moe.cl", defines(o, 2048));
    for (uint32_t C : {1u, 37u, 2048u}) check_sort(dev, po, random_routes(C, o, C), C, o, "random");
    {
      Words r(size_t(2048) * moe_ref::kWords, 0);
      for (uint32_t t = 0; t < 2048; ++t)
        for (uint32_t k = 0; k < o.top_k; ++k) r[size_t(t) * moe_ref::kWords + k] = 7 - k;
      check_sort(dev, po, r, 2048, o, "all to experts 0..7");
      // the tile bound's adversary: experts 8.. one row each, the rest on 0..7
      uint32_t single = 8;
      for (uint32_t t = 0; t < 2048; ++t) {
        uint32_t* row = &r[size_t(t) * moe_ref::kWords];
        for (uint32_t k = 0; k < o.top_k; ++k) row[k] = k;
        if (single < o.experts) row[0] = single++;
      }
      check_sort(dev, po, r, 2048, o, "one-row experts");
    }
    {
      const uint32_t C = 37;
      const Words route = random_routes(C, o, 99);
      const pf_moe_ref::Sorted st = pf_moe_ref::sort(route.data(), C, o);
      const DevSort ds = run_sort(dev, po, route, C, o);
      clrun::Buffer bh(dev, ds.hdr), brt(dev, ds.row_tok);
      std::mt19937 rng(5);
      std::vector<uint16_t> x(size_t(C) * o.hidden);
      for (auto& v : x) v = uint16_t(rng());
      std::vector<uint8_t> xq(size_t(C) * o.hidden);
      for (auto& v : xq) v = uint8_t(rng());
      std::vector<float> xs(C);
      for (auto& v : xs) v = float(rng() % 1000) / 7.0f;
      const uint32_t rows = st.rows_used();
      clrun::Buffer bx(dev, x), bxg(dev, size_t(rows) * o.hidden * 2), bxq(dev, xq), bxs(dev, xs),
          bxg8(dev, size_t(rows) * o.hidden), bxsg(dev, size_t(rows) * 4);
      po.run("pf_moe_gather", {size_t(rows) * 64}, {64}, bx, bh, brt, bxg);
      po.run("pf_moe_gather_i8", {size_t(rows) * 64}, {64}, bxq, bxs, bh, brt, bxg8, bxsg);
      expect(bxg.read<uint16_t>() == pf_moe_ref::gather(st, x.data(), o.hidden), "pf_moe_gather");
      expect(bxg8.read<uint8_t>() == pf_moe_ref::gather(st, xq.data(), o.hidden), "pf_moe_gather_i8 rows");
      std::vector<float> want_s(rows, 0.0f);
      for (uint32_t r = 0; r < rows; ++r)
        if (st.row_tok[r] != pf_moe_ref::kNone) want_s[r] = xs[st.row_tok[r]];
      expect(bxsg.read<float>() == want_s, "pf_moe_gather_i8 scales");
      std::printf("  gather bf16 / int8 C %u: %u rows - %s\n", C, rows, g_bad ? "see above" : "exact");
    }
    // ---- 3: a small shape: dequant and combine ----------------------------------------
    const moe_ref::Shape s{16, 4, 256, 64, 32};
    clrun::Program ps(dev, "src/kernels/prefill/pf_moe.cl", defines(s, 256));
    {
      const uint32_t C = 50;
      Words route = random_routes(C, s, 3);
      for (uint32_t t = 0; t < C; ++t)   // expert 5 gets no row: its blocks are skipped
        for (uint32_t k = 0; k < s.top_k; ++k)
          if (route[size_t(t) * moe_ref::kWords + k] == 5) route[size_t(t) * moe_ref::kWords + k] = 15 - k;
      // (distinctness can break; the sort does not need it, the combine reads pair_row)
      const pf_moe_ref::Sorted st = pf_moe_ref::sort(route.data(), C, s);
      std::mt19937 rng(4);
      std::uniform_real_distribution<float> wd(0.0f, 0.3f), sd(0.01f, 0.05f), yd(-4.0f, 4.0f);
      for (uint32_t t = 0; t < C; ++t) {
        for (uint32_t k = 0; k < s.top_k; ++k) {
          const float w = moe_ref::rf(wd(rng));
          std::memcpy(&route[size_t(t) * moe_ref::kWords + moe_ref::kWeights + k], &w, 4);
        }
        const float sg = moe_ref::rf(wd(rng) + 0.5f);
        std::memcpy(&route[size_t(t) * moe_ref::kWords + moe_ref::kSharedGate], &sg, 4);
      }
      // dequant: all 17 blocks of a layer in one batch (b0 = 0), then blocks [8, 17) (b0 = 8)
      auto random_blocks = [&](size_t words) {
        Words v(words);
        for (size_t t = 0; t < words / 136; ++t) {
          for (int i = 0; i < 128; ++i) v[t * 136 + i] = uint32_t(rng());
          for (int i = 0; i < 8; ++i)
            v[t * 136 + 128 + i] = uint32_t(common::f32_to_f16(sd(rng))) |
                                   (uint32_t(common::f32_to_f16(sd(rng))) << 16);
        }
        return v;
      };
      const uint32_t blocks = s.experts + 1;
      const Words gu = random_blocks(s.gate_up_words() * blocks), dn = random_blocks(s.down_words() * blocks);
      clrun::Buffer bh(dev, st.hdr), bgu(dev, gu), bdn(dev, dn);
      for (uint32_t b0 : {0u, 8u}) {
        const uint32_t nb = blocks - b0;
        clrun::Buffer og(dev, size_t(nb) * s.hidden * 2 * s.inter * 2), od(dev, size_t(nb) * s.inter * s.hidden * 2);
        ps.run("pf_moe_dequant_gu", {2 * s.inter, s.hidden / 64, nb}, {16, 1, 1}, bgu, bh, og, b0);
        ps.run("pf_moe_dequant_dn", {s.hidden, s.inter / 64, nb}, {16, 1, 1}, bdn, bh, od, b0);
        const std::vector<uint16_t> g = og.read<uint16_t>(), d = od.read<uint16_t>();
        for (uint32_t bl = 0; bl < nb; ++bl) {
          const uint32_t b = b0 + bl;
          const bool skip = b < s.experts && st.hdr[kernels::pf_moe::kHdrCount + b] == 0;
          std::vector<uint16_t> wg = pf_moe_ref::dequant_block(gu.data() + b * s.gate_up_words(), s.hidden, 2 * s.inter);
          std::vector<uint16_t> wdn = pf_moe_ref::dequant_block(dn.data() + b * s.down_words(), s.inter, s.hidden);
          if (skip) {
            std::fill(wg.begin(), wg.end(), uint16_t(0));
            std::fill(wdn.begin(), wdn.end(), uint16_t(0));
          }
          expect(std::equal(wg.begin(), wg.end(), g.begin() + size_t(bl) * wg.size()), "pf_moe_dequant_gu");
          expect(std::equal(wdn.begin(), wdn.end(), d.begin() + size_t(bl) * wdn.size()), "pf_moe_dequant_dn");
        }
      }
      expect(st.hdr[kernels::pf_moe::kHdrCount + 5] == 0, "the empty expert of the dequant case");
      std::printf("  dequant gate||up / down, batches from block 0 and 8, empty expert 5 skipped - %s\n",
                  g_bad ? "see above" : "exact");
      // combine
      std::vector<uint16_t> y(size_t(st.rows_used()) * s.hidden), resid(size_t(C) * s.hidden);
      for (auto& v : y) v = moe_ref::rne(yd(rng));
      for (auto& v : resid) v = moe_ref::rne(yd(rng));
      std::vector<uint16_t> want(resid.size());
      for (uint32_t t = 0; t < C; ++t)
        for (uint32_t n = 0; n < s.hidden; ++n)
          want[size_t(t) * s.hidden + n] = pf_moe_ref::combine_token(
              st, &route[size_t(t) * moe_ref::kWords], y.data(), s.hidden, s, t, n,
              resid[size_t(t) * s.hidden + n]);
      clrun::Buffer brt(dev, route), bp(dev, st.pair_row), by(dev, y), bres(dev, resid);
      ps.run("pf_moe_combine", {s.hidden, C}, {256, 1}, brt, bh, bp, by, bres, C);
      expect(bres.read<uint16_t>() == want, "pf_moe_combine");
      std::printf("  combine C %u x %u columns - %s\n", C, s.hidden, g_bad ? "see above" : "exact");
    }
    std::printf("pf_moe on %s: %s\n", dev.name().c_str(),
                g_bad ? "DISAGREES with the host reference (indicative: read the kernel before the box)"
                      : "agrees with the host reference (indicative only; the B70 is the real test)");
    return g_bad ? 1 : 0;
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "pf_moe_run: %s\n", e.what());
    return 1;
  }
}
