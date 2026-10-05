// Spec 18c: the K2 prefill references (tests/kernels/k2_pf_ref.h) against independent
// formulas and against 18b's decode references (k2_ref.h) - host only, so the Mac runs it.
//
//   1. the tile bound: k2_pf_ref::tmax == runtime::k2::pf_tiles (the host's home of it) and
//      holds on its adversary (every expert one row: the most padding a chunk can have);
//   2. the sort, with routing TIES: route rows from k2_ref::route on coarse logits and biases
//      (the checkpoint's biases are 8-28 in steps up to 0.125, so exact selection ties are
//      expected - docs/probe-k2-2026-10-05.md) at C 1 / 37 / 300 / 2048, MoE (100 + shared,
//      top-8) and MoVA (64, top-4, no shared): a tie goes to the LOWER id, every (token, slot)
//      lands once in a tile of its expert, an expert's rows ascend by token, padding is NONE,
//      the shared rows are the tokens in order, the header counts; and for the MoE shape the
//      whole result equals spec 15d's pf_moe_ref::sort of the same rows;
//   3. grouped-GEMM row independence (Review Focus 2): the grouped walk over a chunk and over
//      the same chunk REVERSED gives every (token, slot) the same output (a row's output is a
//      function of its own A row and its expert's block, wherever its tile puts it);
//   4. prefill combines == decode's (Review Focus 1's rule): k2_pf_ref::moe_combine over sorted
//      rows holding rne(split_dot) equals k2_ref::moe_down bit for bit, and mova_combine
//      equals k2_ref::mova_value - the ascending-id bf16 chain is ONE rule on both paths;
//   5. the slab dequant: a linear whose N is not whole slabs (pf_slab_width's tail) - the real
//      columns the int4 formula, the tail's padding exact zeros; K2's real walk widths;
//   6. the eager attention reference against fp64: the reference's rounding points cost about
//      bf16 precision (cosine >= 0.999), never a different function.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/k2_ref.h"
#include "kernels/pf_moe_ref.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_sizes.h"

namespace {

using k2_pf_ref::kNone;
using k2_pf_ref::kTm;
using k2_ref::f32;
using k2_ref::rne;

// A k2_route row (k2_moe.cl R_*) from the reference's Route.
std::vector<uint32_t> route_row(const k2_ref::Route& r, uint32_t K) {
  std::vector<uint32_t> row(k2_ref::kWords, 0);
  for (uint32_t j = 0; j < K; ++j) {
    row[k2_ref::kIds + j] = r.ids[j];
    std::memcpy(&row[k2_ref::kWeights + j], &r.w[j], 4);
    std::memcpy(&row[k2_ref::kSel + j], &r.sel[j], 4);
  }
  std::memcpy(&row[k2_ref::kNext], &r.next, 4);
  std::memcpy(&row[k2_ref::kSum], &r.sum, 4);
  return row;
}

// C tokens routed by k2_ref::route over logits and biases on coarse grids (ties likely).
std::vector<uint32_t> tied_routes(uint32_t C, uint32_t E, uint32_t K, uint32_t seed, uint32_t* ties) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> lq(-8, 8), bq(0, 7);
  std::vector<float> bias(E);
  for (float& b : bias) b = 8.0f + 0.125f * float(bq(rng));   // 8 values: many equal biases
  std::vector<uint32_t> rows(size_t(C) * k2_ref::kWords);
  std::vector<float> lg(E);
  for (uint32_t t = 0; t < C; ++t) {
    for (float& l : lg) l = 0.25f * float(lq(rng));           // 17 values: many equal logits
    const k2_ref::Route r = k2_ref::route(lg.data(), 1, 0, E, 0, 1, bias.data(), E, K, 2.5f);
    // The tie rule: within the top-K, an expert whose sel equals the first one NOT taken has
    // a lower id than it (k2_ref::route's rank: ties to the lower id).
    for (uint32_t j = 0; j < K; ++j) {
      CHECK(j == 0 || r.ids[j] > r.ids[j - 1]);   // ascending, distinct
      CHECK(r.sel[j] >= r.next);
      if (r.sel[j] == r.next) {
        CHECK(r.ids[j] < r.rank_ids[K]);
        ++*ties;
      }
    }
    const std::vector<uint32_t> row = route_row(r, K);
    std::copy(row.begin(), row.end(), rows.begin() + size_t(t) * k2_ref::kWords);
  }
  return rows;
}

// Every invariant of the sorted layout (the kernel header's "no pair is lost" argument).
void check_sorted(const k2_pf_ref::Sorted& st, const uint32_t* route, uint32_t C, uint32_t E,
                  uint32_t K, bool shared) {
  const uint32_t T = k2_pf_ref::tmax(E, K, shared, C);
  CHECK(st.hdr[k2_pf_ref::kHdrTiles] <= T);
  CHECK_EQ(st.hdr[k2_pf_ref::kHdrRows], st.hdr[k2_pf_ref::kHdrTiles] * kTm);
  CHECK_EQ(st.hdr[k2_pf_ref::kHdrC], C);
  CHECK_EQ(st.hdr[k2_pf_ref::kHdrCount + E], shared ? C : 0u);
  std::vector<uint32_t> cnt(E, 0), seen(size_t(T) * kTm, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j) {
      const uint32_t e = route[size_t(t) * k2_ref::kWords + j];
      const uint32_t r = st.pair_row[size_t(t) * K + j];
      CHECK(r < st.rows_used());
      CHECK_EQ(st.row_tok[r], t);
      CHECK_EQ(st.tiles[2 * (r / kTm)], e);   // the row's tile is its expert's
      CHECK_EQ(seen[r], 0u);
      seen[r] = 1;
      ++cnt[e];
    }
  for (uint32_t e = 0; e < E; ++e) CHECK_EQ(st.hdr[k2_pf_ref::kHdrCount + e], cnt[e]);
  // Rows ascend by token inside an expert; padding rows are NONE and only at an expert's end.
  for (uint32_t tile = 0; tile < st.hdr[k2_pf_ref::kHdrTiles]; ++tile) {
    const uint32_t b = st.tiles[2 * tile], r0 = st.tiles[2 * tile + 1];
    CHECK_EQ(r0, tile * kTm);
    for (uint32_t i = 1; i < kTm; ++i) {
      const uint32_t a = st.row_tok[r0 + i - 1], c = st.row_tok[r0 + i];
      if (c != kNone) CHECK(a != kNone && (b == E ? c == a + 1 : c > a));
    }
  }
  for (uint32_t tile = st.hdr[k2_pf_ref::kHdrTiles]; tile < T; ++tile) CHECK_EQ(st.tiles[2 * tile], kNone);
  if (shared) {
    for (uint32_t t = 0; t < C; ++t) CHECK_EQ(st.row_tok[st.shared_row() + t], t);
  } else {
    CHECK_EQ(st.shared_row(), kNone);
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
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}

}  // namespace

int main() {
  const model::K2Desc& d = model::k2();

  // ---- 1. the tile bound -------------------------------------------------------------------
  for (uint32_t C : {1u, 37u, 300u, 2048u}) {
    CHECK_EQ(k2_pf_ref::tmax(d.experts, d.top_k, true, C), runtime::k2::pf_moe_tiles(d, C));
    CHECK_EQ(k2_pf_ref::tmax(d.value_experts, d.value_top_k, false, C), runtime::k2::pf_mova_tiles(d, C));
  }
  CHECK_EQ(runtime::k2::pf_moe_tiles(d, 2048), 672u);
  CHECK_EQ(runtime::k2::pf_mova_tiles(d, 2048), 318u);
  {   // the adversary: C x K = E distinct experts with one row each (16 tokens x top-4 = 64)
    const uint32_t E = 64, K = 4, C = 16;
    std::vector<uint32_t> rows(size_t(C) * k2_ref::kWords, 0);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < K; ++j) rows[size_t(t) * k2_ref::kWords + j] = t * K + j;
    const k2_pf_ref::Sorted st = k2_pf_ref::sort(rows.data(), C, E, K, false);
    check_sorted(st, rows.data(), C, E, K, false);
    CHECK_EQ(st.hdr[k2_pf_ref::kHdrTiles], 64u);   // every expert one tile of 1 row + 31 padding
    CHECK(64u <= k2_pf_ref::tmax(E, K, false, C));
  }

  // ---- 2. the sort, with ties --------------------------------------------------------------
  uint32_t ties = 0;
  for (uint32_t C : {1u, 37u, 300u, 2048u}) {
    const std::vector<uint32_t> moe = tied_routes(C, d.experts, d.top_k, 100 + C, &ties);
    const k2_pf_ref::Sorted sm = k2_pf_ref::sort(moe.data(), C, d.experts, d.top_k, true);
    check_sorted(sm, moe.data(), C, d.experts, d.top_k, true);
    // Spec 15d's walk on the same rows (pf_moe_ref::sort reads ids at the same words).
    const moe_ref::Shape ms{d.experts, d.top_k, d.hidden, d.moe_inter, d.router_n()};
    const pf_moe_ref::Sorted ref = pf_moe_ref::sort(moe.data(), C, ms);
    CHECK(sm.tiles == ref.tiles);
    CHECK(sm.row_tok == ref.row_tok);
    CHECK(sm.pair_row == ref.pair_row);
    CHECK(sm.hdr == ref.hdr);
    const std::vector<uint32_t> mova = tied_routes(C, d.value_experts, d.value_top_k, 200 + C, &ties);
    const k2_pf_ref::Sorted sv = k2_pf_ref::sort(mova.data(), C, d.value_experts, d.value_top_k, false);
    check_sorted(sv, mova.data(), C, d.value_experts, d.value_top_k, false);
  }
  CHECK(ties > 0);   // the coarse grids did produce ties at the cut
  std::printf("sort: MoE and MoVA at C 1 / 37 / 300 / 2048 hold every invariant; MoE == spec 15d's "
              "walk; %u slot(s) taken on an exact tie at the cut, each the lower id\n", ties);

  // ---- 3. grouped-GEMM row independence ----------------------------------------------------
  {
    const uint32_t E = 8, K = 2, H = 128, N = 64, C = 50;
    const std::vector<uint32_t> blocks = random_blocks(size_t(E + 1) * (N / 16) * (H / 64) * 136, 7);
    const size_t bw = size_t(N / 16) * (H / 64) * 136;
    const std::vector<uint16_t> x = random_bf16(size_t(C) * H, -1.f, 1.f, 8);
    std::vector<uint32_t> fwd = tied_routes(C, E, K, 9, &ties), rev(fwd.size());
    std::vector<uint16_t> xr(x.size());
    for (uint32_t t = 0; t < C; ++t) {
      std::copy(fwd.begin() + size_t(t) * 32, fwd.begin() + size_t(t + 1) * 32, rev.begin() + size_t(C - 1 - t) * 32);
      std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(C - 1 - t) * H);
    }
    auto run = [&](const std::vector<uint32_t>& rt, const std::vector<uint16_t>& xs) {
      const k2_pf_ref::Sorted st = k2_pf_ref::sort(rt.data(), C, E, K, true);
      const std::vector<uint16_t> g = k2_pf_ref::gather(st, xs.data(), H);
      std::vector<uint16_t> y(size_t(st.rows_used()) * N, 0);
      k2_pf_ref::grouped(st, 0, E + 1, [&](uint32_t b, uint32_t r) {
        const std::vector<uint16_t> bb = pf_moe_ref::dequant_block(blocks.data() + b * bw, H, N);
        const std::vector<float> o = pf_moe_ref::gemm_rows(g.data() + size_t(r) * H, bb.data(), H, N);
        for (uint32_t n = 0; n < N; ++n) y[size_t(r) * N + n] = rne(o[n]);
      });
      // Per (token, slot) and per token's shared row: the output row.
      std::vector<uint16_t> out(size_t(C) * (K + 1) * N);
      for (uint32_t t = 0; t < C; ++t)
        for (uint32_t j = 0; j <= K; ++j) {
          const uint32_t r = j < K ? st.pair_row[size_t(t) * K + j] : st.shared_row() + t;
          std::copy(y.begin() + size_t(r) * N, y.begin() + size_t(r + 1) * N,
                    out.begin() + (size_t(t) * (K + 1) + j) * N);
        }
      return out;
    };
    const std::vector<uint16_t> a = run(fwd, x), b = run(rev, xr);
    for (uint32_t t = 0; t < C; ++t)
      CHECK(std::equal(a.begin() + size_t(t) * (K + 1) * N, a.begin() + size_t(t + 1) * (K + 1) * N,
                       b.begin() + size_t(C - 1 - t) * (K + 1) * N));
    std::printf("row independence: %u tokens forward and reversed - every (token, slot) row identical\n", C);
  }

  // ---- 4. the combines are decode's chains ---------------------------------------------------
  {
    const uint32_t E = 12, K = 4, H = 256, I = 128, C = 9, ks = 2;
    const moe_ref::Shape ms{E, K, H, I, 16};
    const std::vector<uint32_t> dn = random_blocks(ms.down_words() * (E + 1), 11);
    const std::vector<uint32_t> rt = tied_routes(C, E, K, 12, &ties);
    const k2_pf_ref::Sorted st = k2_pf_ref::sort(rt.data(), C, E, K, true);
    std::vector<uint16_t> y(size_t(st.rows_used()) * H, 0);
    std::vector<std::vector<uint16_t>> h(C);
    const std::vector<uint16_t> resid = random_bf16(size_t(C) * H, -2.f, 2.f, 13);
    for (uint32_t t = 0; t < C; ++t) {
      h[t] = random_bf16(size_t(K + 1) * I, -0.5f, 0.5f, 14 + t);
      for (uint32_t j = 0; j <= K; ++j) {
        const uint32_t e = j < K ? rt[size_t(t) * 32 + j] : E;
        const uint32_t r = j < K ? st.pair_row[size_t(t) * K + j] : st.shared_row() + t;
        for (uint32_t n = 0; n < H; ++n)
          y[size_t(r) * H + n] = rne(moe_ref::split_dot(dn.data() + e * ms.down_words(), I, n,
                                                        h[t].data() + size_t(j) * I, ks));
      }
    }
    for (uint32_t t = 0; t < C; ++t) {
      k2_ref::Route r;
      for (uint32_t j = 0; j < K; ++j) {
        r.ids[j] = rt[size_t(t) * 32 + j];
        std::memcpy(&r.w[j], &rt[size_t(t) * 32 + 8 + j], 4);
      }
      const std::vector<uint16_t> want = k2_ref::moe_down(r, h[t], dn.data(), ms, resid.data() + size_t(t) * H, ks);
      for (uint32_t n = 0; n < H; ++n)
        CHECK_EQ(k2_pf_ref::moe_combine(st, rt.data() + size_t(t) * 32, y.data(), H, K, t, n,
                                        resid[size_t(t) * H + n]),
                 want[n]);
    }
    // MoVA: y = rne(split_dot) of each selected value expert, the combine vs mova_value.
    const uint32_t EV = 16, KV = 4, D = 256, VN = 64;
    const size_t vblk = size_t(VN / 16) * (D / 64) * 136;
    const std::vector<uint32_t> vw = random_blocks(vblk * EV, 15);
    const std::vector<uint32_t> vr = tied_routes(C, EV, KV, 16, &ties);
    const k2_pf_ref::Sorted sv = k2_pf_ref::sort(vr.data(), C, EV, KV, false);
    const std::vector<uint16_t> xs = random_bf16(size_t(C) * D, -0.2f, 0.2f, 17);
    std::vector<uint16_t> yv(size_t(sv.rows_used()) * VN, 0);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < KV; ++j) {
        const uint32_t r = sv.pair_row[size_t(t) * KV + j];
        for (uint32_t n = 0; n < VN; ++n)
          yv[size_t(r) * VN + n] =
              rne(moe_ref::split_dot(vw.data() + vr[size_t(t) * 32 + j] * vblk, D, n, xs.data() + size_t(t) * D, ks));
      }
    for (uint32_t t = 0; t < C; ++t) {
      k2_ref::Route r;
      for (uint32_t j = 0; j < KV; ++j) {
        r.ids[j] = vr[size_t(t) * 32 + j];
        std::memcpy(&r.w[j], &vr[size_t(t) * 32 + 8 + j], 4);
      }
      const std::vector<uint16_t> want = k2_ref::mova_value(r, xs.data() + size_t(t) * D, vw.data(), D, VN, KV, ks);
      for (uint32_t n = 0; n < VN; ++n)
        CHECK_EQ(k2_pf_ref::mova_combine(sv, vr.data() + size_t(t) * 32, yv.data(), VN, KV, t, n), want[n]);
    }
    std::printf("combines: the MoE (ascending id + shared + residual) and MoVA (SiLU, ascending id) "
                "chains over sorted rows are decode's, bit for bit\n");
  }

  // ---- 5. the slab dequant -----------------------------------------------------------------
  {
    const uint32_t K = 128, N = 336;   // not whole slabs: the walk is one tail of pad256(336) = 512
    CHECK_EQ(runtime::k2::pf_slabs(N), 1u);
    CHECK_EQ(runtime::k2::pf_slab_width(N, 0), 512u);
    std::mt19937 rng(21);
    std::vector<uint32_t> words(size_t(K / 8) * N);
    for (uint32_t& w : words) w = rng();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    std::uniform_real_distribution<float> sd(0.01f, 0.05f);
    for (uint16_t& s : sc) s = common::f32_to_f16(sd(rng));
    const std::vector<uint16_t> slab = k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, 0, 512);
    for (uint32_t c = 0; c < 512; ++c)
      for (uint32_t k = 0; k < K; ++k) {
        const uint16_t got = slab[size_t(k) * 512 + c];
        if (c >= N) {
          CHECK_EQ(got, uint16_t(0));
          continue;
        }
        const uint32_t word = words[size_t(k / 8) * N + c];
        const int q = int((word >> (4 * (k % 8))) & 0xFu) - 8;
        CHECK_EQ(got, rne(float(q) * common::f16_to_f32(sc[size_t(k / 64) * N + c])));
      }
    // K2's walk widths: 10 / 10 / 3 / 12 / 3 slabs, the MoVA row's tail 256 at 9216, pitch 9472.
    CHECK_EQ(runtime::k2::pf_slabs(d.attn_dense_n()), 10u);
    CHECK_EQ(runtime::k2::pf_slabs(d.attn_sparse_n()), 10u);
    CHECK_EQ(runtime::k2::pf_slab_width(d.attn_sparse_n(), 9216), 256u);
    CHECK_EQ(runtime::k2::pf_ld(d.attn_sparse_n()), 9472u);
    CHECK_EQ(runtime::k2::pf_slab_width(d.hidden, 2048), 512u);
    std::printf("slab dequant: whole columns the int4 formula, the tail's padding zeros; K2's walk widths\n");
  }

  // ---- 6. the eager attention reference against fp64 ---------------------------------------
  {
    const uint32_t hd = 128, kvh = 2, len = 300;
    const std::vector<uint16_t> kc = random_bf16(size_t(len) * kvh * hd, -1.f, 1.f, 31);
    const std::vector<uint16_t> vc = random_bf16(size_t(len) * kvh * hd, -1.f, 1.f, 32);
    std::vector<float> q(hd);
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> qd(-1.f, 1.f);
    double worst = 1.0;
    for (uint32_t p : {0u, 1u, 63u, 64u, 299u}) {
      for (float& v : q) v = k2_ref::rf(qd(rng));
      const std::vector<double> o = k2_pf_ref::attention(q.data(), kc.data(), vc.data(), p, 1, kvh, hd);
      const std::vector<float> e = k2_pf_ref::attention_eager(q.data(), kc.data(), vc.data(), p, 1, kvh, hd);
      double ab = 0, aa = 0, bb = 0;
      for (uint32_t i = 0; i < hd; ++i) {
        ab += o[i] * e[i];
        aa += o[i] * o[i];
        bb += double(e[i]) * e[i];
      }
      worst = std::min(worst, ab / std::sqrt(aa * bb));
    }
    CHECK(worst >= 0.999);
    std::printf("eager attention reference vs fp64: worst cosine %.6f (the bf16 rounding points)\n", worst);
  }

  std::printf("k2_pf_ref_test OK\n");
  return 0;
}
