// k2_run - spec 18b's portable K2-Horizon kernels on the Mac's OpenCL GPU against
// tests/kernels/k2_ref.h (INDICATIVE ONLY: clrun.h says what this can and cannot show).
//
//   k2_run        no arguments; K2's real shapes (hidden 2560, 100 + 1 experts x 768,
//                 64 value experts x 1024, 32 q / 8 kv heads x 128)
//
// What runs: src/kernels/k2/k2_prep.cl (k2_norm_finish after prep.cl's prep_res_fold,
// k2_silu_mul, both k2_attn_prep builds) and src/kernels/k2/k2_moe.cl (k2_route - the MoE
// router with its 28 padded lanes and MoVA's over two slices of the fused row -,
// k2_moe_gate_up, k2_moe_down, k2_mova_value), with the CMake block's defines except ONE:
// k2_moe_down runs at DN_KS 1 here (144 lanes) because the Mac's GPUs cap a work-group at
// 256 and the B70 build's DN_KS 2 is 288. k2_attn.cl is not run: its decode is shuffles,
// sub-group reductions and 2D block reads, which the emulation leaves undeclared on purpose.
// k2_attn_eager.cl (B70_K2_ATTN=eager, spec 18 §10.1) is plain OpenCL C and does run: its
// scores bit-exact, its probabilities and outputs within 1 ulp (Apple's compiler is not told
// -cl-fp32-correctly-rounded-divide-sqrt; measured on the UHD 630: 0 differences anywhere).
//
// Bars: the exp-free chains (the norm, the RoPE prep) bit-exact; the rest within 2 bf16
// ulps (OpenCL's exp, the GEMVs' fma contraction), the route ids exact.
//
// Spec 18c adds the portable PREFILL kernels (prefill_checks, against tests/kernels/
// k2_pf_ref.h): k2_pf_linear.cl's slab dequant (a whole slab and the MoVA row's zero-padded
// tail), the grouped norm and the attention prep at M = kPfC over a chunk (pf_prep.cl's
// pf_res_fold + k2_prep.cl), MoVA's k2_route on grid (1, C) at pitch 9472, and k2_pf_moe.cl's
// sort (routes with exact ties at the cut), gather, dequant and both combines (over host-made
// y: the grouped GEMMs pf_moe_gemm and the flash attention k2_pf_attn are DPAS, box only).
// Bars: the sort, gather, dequant, norm and prep exact; the MoE combine exact (no exp); the
// MoVA combine within 2 ulps (SiLU's exp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "clrun.h"
#include "common/bf16.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/k2_ref.h"
#include "kernels/pf_moe_ref.h"
#include "kernels/prep_ref.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"

namespace {

using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}
std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(rng);
  return v;
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

int failures = 0;
void report_bool(const char* what, bool ok);
void report(const char* what, int worst, int bar, size_t n);

// Spec 18c: the prefill kernels that are portable (no DPAS, no 2D block I/O, no cross-lane
// reduction): the slab dequant, the grouped norm and the attention prep at M = kPfC over a
// chunk, the MoVA router at pitch 9472 on grid (1, C), and k2_pf_moe.cl's sort / gather /
// dequant / combine for both routers - against tests/kernels/k2_pf_ref.h. The grouped GEMMs
// (pf_moe_gemm) and the flash attention (k2_pf_attn) are DPAS kernels: box only.
void prefill_checks(clrun::Device& dev, const model::K2Desc& m) {
  const uint32_t H = m.hidden, kC = 2048;
  {   // the slab dequant, a whole slab and the 256-column tail of the MoVA row (N 9280)
    const uint32_t K = H, N = m.attn_sparse_n();
    std::mt19937 rng(70);
    std::vector<uint32_t> words(size_t(K / 8) * N);
    for (uint32_t& w : words) w = rng();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    std::uniform_real_distribution<float> sd(0.01f, 0.05f);
    for (uint16_t& s : sc) s = common::f32_to_f16(sd(rng));
    clrun::Program p(dev, "src/kernels/k2/k2_pf_linear.cl", {"K=2560", "N=9280"});
    clrun::Buffer wb(dev, words), sb(dev, sc), out(dev, size_t(K) * 1024 * 2);
    bool ok = true;
    for (uint32_t n0 : {0u, 9216u}) {
      const uint32_t ns = n0 ? 256 : 1024;
      p.run("k2_pf_dequant_slab", {size_t(ns), size_t(K / 64)}, {16, 1}, wb, sb, out, n0, ns);
      const std::vector<uint16_t> got = out.read<uint16_t>();
      const std::vector<uint16_t> want = k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, ns);
      ok = ok && std::equal(want.begin(), want.end(), got.begin());
    }
    report_bool("prefill slab dequant (whole + the zero tail)", ok);
  }
  {   // the grouped norm over a chunk: pf_res_fold (SP1, m_count kPfC) + k2_norm_finish (M kPfC)
    const uint32_t C = 5, G = 20;
    std::vector<uint16_t> resid = random_bf16(size_t(kC) * H, -1.f, 1.f, 71);
    const std::vector<float> part = random_f32(size_t(kC) * H, -0.5f, 0.5f, 72);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
    std::vector<uint16_t> r_ref(resid.begin(), resid.begin() + size_t(C) * H), x_ref(size_t(C) * H);
    std::vector<float> ss_ref(size_t(G) * C);
    prep_ref::res_fold(part.data(), r_ref.data(), ss_ref.data(), C, H, 1, G);
    k2_ref::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), C, H, G, 2);
    clrun::Program fold(dev, "src/kernels/prefill/pf_prep.cl", {"K=2560", "S_PREV=1", "FOLD_G=20"});
    clrun::Program norm(dev, "src/kernels/k2/k2_prep.cl",
                        {"M=2048", "K=2560", "NORM_G=20", "NORM_WGS=20", "NORM_GROUPS=2"});
    clrun::Buffer pb(dev, part), rb(dev, resid), wb(dev, w), ss(dev, size_t(G) * kC * 4), xb(dev, size_t(kC) * H * 2);
    fold.run("pf_res_fold", {20 * 256, C}, {256, 1}, pb, rb, ss, kC);
    norm.run("k2_norm_finish", {20 * 256, C}, {256, 1}, ss, rb, wb, xb);
    const std::vector<uint16_t> r = rb.read<uint16_t>(), x = xb.read<uint16_t>();
    report_bool("prefill grouped norm (5 rows, resid + x bit-exact)",
                std::equal(r_ref.begin(), r_ref.end(), r.begin()) && std::equal(x_ref.begin(), x_ref.end(), x.begin()));
  }
  {   // the attention prep at M kPfC, S 1, both builds, a 3-row chunk at pos 37
    const std::vector<float> rope = loader::k2_rope_table(m, 64);
    const uint32_t C = 3, pos = 37;
    for (int dense = 1; dense >= 0; --dense) {
      const uint32_t ld = dense ? 10240 : 9472;
      const std::vector<float> part = random_f32(size_t(kC) * ld, -1.f, 1.f, 73 + dense);
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = C;
      clrun::Program p(dev, "src/kernels/k2/k2_prep.cl",
                       {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18", "M=2048",
                        "QKV_S=1", "Q_HEADS=32", "KV_HEADS=8", "HD=128", "QKV_N=" + std::to_string(ld),
                        std::string("V_FROM_PARTIALS=") + (dense ? "1" : "0")});
      clrun::Buffer cb(dev, ctrl), pb(dev, part), rb(dev, rope), aq(dev, size_t(kC) * m.q_n() * 4),
          ag(dev, size_t(kC) * m.q_n() * 4), kk(dev, size_t(64) * m.kv_n() * 2), kv(dev, size_t(64) * m.kv_n() * 2);
      p.run("k2_attn_prep", {size_t(m.q_heads + m.kv_heads) * 128, C}, {128, 1}, cb, pb, rb, aq, ag, kk, kv);
      const std::vector<float> q = aq.read<float>(), g = ag.read<float>();
      const std::vector<uint16_t> kr = kk.read<uint16_t>(), vr = kv.read<uint16_t>();
      bool ok = true;
      for (uint32_t t = 0; t < C; ++t) {
        const k2_ref::Prep want = k2_ref::attn_prep(part.data(), kC, t, ld, 1, rope.data() + size_t(pos + t) * 128,
                                                    m.q_heads, m.kv_heads, 128, dense);
        ok = ok && std::equal(want.q.begin(), want.q.end(), q.begin() + size_t(t) * m.q_n()) &&
             std::equal(want.gate.begin(), want.gate.end(), g.begin() + size_t(t) * m.q_n()) &&
             std::equal(want.k.begin(), want.k.end(), kr.begin() + size_t(pos + t) * m.kv_n());
        if (dense) ok = ok && std::equal(want.v.begin(), want.v.end(), vr.begin() + size_t(pos + t) * m.kv_n());
      }
      report_bool(dense ? "prefill attn prep, dense (3 rows bit-exact)" : "prefill attn prep, MoVA (3 rows bit-exact)", ok);
    }
  }
  // The routers' rows for the sort: MoVA's k2_route on grid (1, C) at pitch 9472 (ids exact);
  // the MoE's from k2_ref::route on coarse logits (ties at the cut), as the device would.
  const uint32_t C = 37;
  std::vector<uint32_t> mova_rows(size_t(C) * 32), moe_rows(size_t(C) * 32, 0);
  {
    const std::vector<float> part = random_f32(size_t(kC) * 9472, -3.f, 3.f, 75);
    const std::vector<float> bias = random_f32(64, -0.05f, 0.05f, 76);
    clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                     {"M=1", "ROUTE_E=64", "ROUTE_K=4", "ROUTE_WG=64", "ROUTE_LN=9472", "ROUTE_LOFF=9216",
                      "ROUTE_LS=1", "ROUTE_SCALE=2.5f"});
    clrun::Buffer pb(dev, part), bb(dev, bias), rt(dev, size_t(C) * 32 * 4);
    p.run("k2_route", {64, C}, {64, 1}, pb, bb, rt);
    mova_rows = rt.read<uint32_t>();
    bool ok = true;
    for (uint32_t t = 0; t < C; ++t) {
      const k2_ref::Route want = k2_ref::route(part.data(), kC, t, 9472, 9216, 1, bias.data(), 64, 4, 2.5f);
      for (uint32_t j = 0; j < 4; ++j) ok = ok && mova_rows[size_t(t) * 32 + j] == want.ids[j];
    }
    report_bool("prefill MoVA route, 37 rows on grid (1, C)", ok);
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> lq(-8, 8), bq(0, 7);
    std::vector<float> mb(100), lg(100);
    for (float& b : mb) b = 8.0f + 0.125f * float(bq(rng));
    for (uint32_t t = 0; t < C; ++t) {
      for (float& l : lg) l = 0.25f * float(lq(rng));
      const k2_ref::Route r = k2_ref::route(lg.data(), 1, 0, 100, 0, 1, mb.data(), 100, 8, 2.5f);
      for (uint32_t j = 0; j < 8; ++j) {
        moe_rows[size_t(t) * 32 + j] = r.ids[j];
        std::memcpy(&moe_rows[size_t(t) * 32 + 8 + j], &r.w[j], 4);
      }
    }
  }
  // k2_pf_moe.cl, both families: sort / gather / dequant exact; the combines over host-made y.
  for (int fam = 0; fam < 2; ++fam) {   // 0 MoVA, 1 MoE
    const bool moe = fam == 1;
    const uint32_t E = moe ? 100 : 64, K = moe ? 8 : 4, WG = moe ? 128 : 64, N = moe ? 2 * 768 : 1024;
    const std::vector<uint32_t>& rows = moe ? moe_rows : mova_rows;
    std::vector<std::string> defs = {"PF_E=" + std::to_string(E), "PF_K=" + std::to_string(K),
                                     "PF_WG=" + std::to_string(WG), std::string("PF_SHARED=") + (moe ? "1" : "0"),
                                     "PF_D=2560", "TM=32", "KC=2048"};
    defs.push_back(moe ? "PF_INTER=768" : "PF_VN=1024");
    clrun::Program p(dev, "src/kernels/k2/k2_pf_moe.cl", defs);
    const uint32_t T = k2_pf_ref::tmax(E, K, moe, C), R = T * 32;
    const k2_pf_ref::Sorted st = k2_pf_ref::sort(rows.data(), C, E, K, moe);
    clrun::Buffer rt(dev, rows), hdr(dev, size_t(k2_pf_ref::hdr_words(E)) * 4), tiles(dev, size_t(T) * 8),
        row_tok(dev, size_t(R) * 4), pair_row(dev, size_t(C) * K * 4);
    p.run("k2_pf_sort", {WG}, {WG}, rt, hdr, tiles, row_tok, pair_row, C, T);
    const std::vector<uint32_t> h = hdr.read<uint32_t>(), ti = tiles.read<uint32_t>(), rtk = row_tok.read<uint32_t>(),
                                pr = pair_row.read<uint32_t>();
    const bool sort_ok = h == st.hdr && ti == st.tiles && pr == st.pair_row &&
                         std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), rtk.begin());
    report_bool(moe ? "prefill MoE sort (37 rows, ties, + shared)" : "prefill MoVA sort (37 rows, no shared)", sort_ok);
    const std::vector<uint16_t> x = random_bf16(size_t(C) * H, -1.f, 1.f, 80 + fam);
    clrun::Buffer xb(dev, x), xg(dev, size_t(R) * H * 2);
    p.run("k2_pf_gather", {size_t(R) * 64}, {64}, xb, hdr, row_tok, xg);
    const std::vector<uint16_t> g = xg.read<uint16_t>(), gw = k2_pf_ref::gather(st, x.data(), H);
    report_bool(moe ? "prefill MoE gather" : "prefill MoVA gather", std::equal(gw.begin(), gw.end(), g.begin()));
    // dequant: the first busy block and its neighbour (a block with no row is skipped)
    const size_t blk = size_t(N / 16) * (H / 64) * 136;
    const std::vector<uint32_t> wts = random_blocks(blk * 4, 82 + fam);
    uint32_t busy = 0;
    while (st.hdr[k2_pf_ref::kHdrCount + busy] == 0) ++busy;
    std::vector<uint32_t> all(blk * (busy + 2), 0);
    std::copy(wts.begin(), wts.begin() + blk * 2, all.begin() + blk * busy);
    clrun::Buffer wb(dev, all), out(dev, size_t(2) * H * N * 2);
    p.run(moe ? "k2_pf_dequant_gu" : "k2_pf_dequant_v", {size_t(N), size_t(H / 64), 2}, {16, 1, 1}, wb, hdr, out, busy);
    const std::vector<uint16_t> dq = out.read<uint16_t>();
    const std::vector<uint16_t> want0 = pf_moe_ref::dequant_block(all.data() + blk * busy, H, N);
    report_bool(moe ? "prefill MoE gate||up dequant (a busy block)" : "prefill MoVA value dequant (a busy block)",
                std::equal(want0.begin(), want0.end(), dq.begin()));
    // the combine over host-made y rows (rne'd values as the grouped GEMM leaves them)
    const uint32_t yw = moe ? H : 1024;
    const std::vector<uint16_t> y = random_bf16(size_t(R) * yw, -1.f, 1.f, 84 + fam);
    clrun::Buffer yb(dev, y);
    if (moe) {
      const std::vector<uint16_t> resid0 = random_bf16(size_t(C) * H, -2.f, 2.f, 86);
      clrun::Buffer rb(dev, resid0);
      p.run("k2_pf_moe_combine", {size_t(H), C}, {256, 1}, rt, hdr, pair_row, yb, rb, C);
      const std::vector<uint16_t> got = rb.read<uint16_t>();
      int worst = 0;
      for (uint32_t t = 0; t < C; ++t)
        for (uint32_t n = 0; n < H; ++n)
          worst = std::max(worst, ulps(got[size_t(t) * H + n],
                                       k2_pf_ref::moe_combine(st, rows.data() + size_t(t) * 32, y.data(), H, K, t, n,
                                                              resid0[size_t(t) * H + n])));
      report("prefill MoE combine (ascending id + shared)", worst, 0, size_t(C) * H);
    } else {
      const uint32_t pos = 3;
      clrun::Buffer kv(dev, size_t(pos + C) * 1024 * 2);
      p.run("k2_pf_mova_combine", {1024, C}, {256, 1}, rt, pair_row, yb, kv, pos, C);
      const std::vector<uint16_t> got = kv.read<uint16_t>();
      int worst = 0;
      for (uint32_t t = 0; t < C; ++t)
        for (uint32_t n = 0; n < 1024; ++n)
          worst = std::max(worst, ulps(got[size_t(pos + t) * 1024 + n],
                                       k2_pf_ref::mova_combine(st, rows.data() + size_t(t) * 32, y.data(), 1024, K, t, n)));
      report("prefill MoVA combine into V[pos + t]", worst, 2, size_t(C) * 1024);
    }
  }
}

void report(const char* what, int worst, int bar, size_t n) {
  const bool ok = worst <= bar;
  std::printf("  %-44s %zu values, worst %d bf16 ulps (bar %d) %s\n", what, n, worst, bar,
              ok ? "agrees" : "DISAGREES");
  if (!ok) ++failures;
}
void report_bool(const char* what, bool ok) {
  std::printf("  %-44s %s\n", what, ok ? "agrees" : "DISAGREES");
  if (!ok) ++failures;
}

}  // namespace

int main() {
  const model::K2Desc& m = model::k2();
  const uint32_t H = m.hidden;
  try {
    clrun::Device dev;
    std::printf("k2_run on %s (INDICATIVE: Mac OpenCL, emulated sub-groups)\n", dev.name().c_str());

    // --- the grouped norm: prep_res_fold (SP4) + k2_norm_finish ------------------------------
    {
      const uint32_t G = 20, S = 4;
      std::vector<uint16_t> resid = random_bf16(H, -1.f, 1.f, 1);
      for (uint32_t k = 0; k < H / 2; ++k) resid[k] = rne(f32(resid[k]) * 0.01f);
      const std::vector<float> part = random_f32(size_t(S) * H, -0.5f, 0.5f, 2);
      std::vector<float> w(H);
      for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
      std::vector<uint16_t> r_ref = resid, x_ref(H);
      std::vector<float> ss_ref(G);
      prep_ref::res_fold(part.data(), r_ref.data(), ss_ref.data(), 1, H, S, G);
      k2_ref::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), 1, H, G, 2);
      clrun::Program fold(dev, "src/kernels/prep.cl", {"M=1", "K=2560", "S_PREV=4", "FOLD_G=20"});
      clrun::Program norm(dev, "src/kernels/k2/k2_prep.cl",
                          {"M=1", "K=2560", "NORM_G=20", "NORM_WGS=20", "NORM_GROUPS=2"});
      clrun::Buffer pb(dev, part), rb(dev, resid), wb(dev, w), ss(dev, G * 4), xb(dev, H * 2);
      fold.run("prep_res_fold", {20 * 256, 1}, {256, 1}, pb, rb, ss);
      norm.run("k2_norm_finish", {20 * 256, 1}, {256, 1}, ss, rb, wb, xb);
      const std::vector<uint16_t> r = rb.read<uint16_t>(), x = xb.read<uint16_t>();
      int worst = 0;
      for (uint32_t k = 0; k < H; ++k) worst = std::max(worst, std::max(ulps(r[k], r_ref[k]), ulps(x[k], x_ref[k])));
      report("grouped norm (resid, x)", worst, 0, 2 * H);
    }
    // --- SiLU x up ---------------------------------------------------------------------------
    {
      const uint32_t I = 6144, S = 4;
      const std::vector<float> part = random_f32(size_t(S) * 2 * I, -1.f, 1.f, 3);
      std::vector<uint16_t> x_ref(I);
      k2_ref::silu_mul(part.data(), x_ref.data(), 1, I, S);
      clrun::Program p(dev, "src/kernels/k2/k2_prep.cl", {"M=1", "SILU_I=6144", "SILU_S=4"});
      clrun::Buffer pb(dev, part), xb(dev, I * 2);
      p.run("k2_silu_mul", {2 * 256, 1}, {256, 1}, pb, xb);
      const std::vector<uint16_t> x = xb.read<uint16_t>();
      int worst = 0;
      for (uint32_t k = 0; k < I; ++k) worst = std::max(worst, ulps(x[k], x_ref[k]));
      report("silu x up (I 6144, S 4)", worst, 2, I);
    }
    // --- the attention prep, both builds -------------------------------------------------------
    const std::vector<float> rope = loader::k2_rope_table(m, 64);
    for (int dense = 1; dense >= 0; --dense) {
      const uint32_t N = dense ? m.attn_dense_n() : m.attn_sparse_n(), S = 2, pos = 37;
      const std::vector<float> part = random_f32(size_t(S) * N, -1.f, 1.f, 10 + dense);
      const k2_ref::Prep want = k2_ref::attn_prep(part.data(), 1, 0, N, S, rope.data() + size_t(pos) * 128,
                                                  m.q_heads, m.kv_heads, 128, dense);
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = pos;
      ctrl[1] = 1;
      clrun::Program p(dev, "src/kernels/k2/k2_prep.cl",
                       {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                        "M=1", "QKV_S=2", "Q_HEADS=32", "KV_HEADS=8", "HD=128",
                        "QKV_N=" + std::to_string(N), std::string("V_FROM_PARTIALS=") + (dense ? "1" : "0")});
      clrun::Buffer cb(dev, ctrl), pb(dev, part), rb(dev, rope), aq(dev, m.q_n() * 4), ag(dev, m.q_n() * 4);
      clrun::Buffer kk(dev, size_t(64) * m.kv_n() * 2), kv(dev, size_t(64) * m.kv_n() * 2);
      p.run("k2_attn_prep", {size_t(m.q_heads + m.kv_heads) * 128, 1}, {128, 1}, cb, pb, rb, aq, ag, kk, kv);
      const std::vector<float> q = aq.read<float>(), g = ag.read<float>();
      const std::vector<uint16_t> kr = kk.read<uint16_t>(), vr = kv.read<uint16_t>();
      bool ok = q == want.q && g == want.gate &&
                std::equal(want.k.begin(), want.k.end(), kr.begin() + size_t(pos) * m.kv_n());
      if (dense) ok = ok && std::equal(want.v.begin(), want.v.end(), vr.begin() + size_t(pos) * m.kv_n());
      report_bool(dense ? "attn prep, dense (q, gate, k, v bit-exact)" : "attn prep, MoVA (q, gate, k bit-exact)", ok);
    }
    // --- the routers -----------------------------------------------------------------------------
    {
      const std::vector<float> bias = random_f32(128, -0.05f, 0.05f, 20);
      std::vector<float> lg = random_f32(128, -3.f, 3.f, 21);
      for (uint32_t e = 100; e < 128; ++e) lg[e] = 0.0f;   // the GEMV's zero rows
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "ROUTE_E=100", "ROUTE_K=8", "ROUTE_WG=128", "ROUTE_LN=128",
                        "ROUTE_LOFF=0", "ROUTE_LS=1", "ROUTE_SCALE=2.5f"});
      clrun::Buffer lb(dev, lg), bb(dev, bias), rt(dev, 32 * 4);
      p.run("k2_route", {128, 1}, {128, 1}, lb, bb, rt);
      const k2_ref::Route want = k2_ref::route(lg.data(), 1, 0, 128, 0, 1, bias.data(), 100, 8, 2.5f);
      const std::vector<uint32_t> row = rt.read<uint32_t>();
      bool ok = true;
      int worst = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        ok = ok && row[j] == want.ids[j];
        float w;
        std::memcpy(&w, &row[8 + j], 4);
        worst = std::max(worst, ulps(rne(w), rne(want.w[j])));
      }
      report_bool("MoE route: ids (ascending)", ok);
      report("MoE route: weights", worst, 1, 8);
      // The padded lanes at +80, the real scores far below: never selected.
      for (uint32_t e = 0; e < 100; ++e) lg[e] = -30.0f - float(e % 5);
      for (uint32_t e = 100; e < 128; ++e) lg[e] = 80.0f;
      std::vector<float> nb(128, -0.9f);
      lb.write(lg.data(), lg.size() * 4);
      bb.write(nb.data(), nb.size() * 4);
      p.run("k2_route", {128, 1}, {128, 1}, lb, bb, rt);
      const std::vector<uint32_t> r2 = rt.read<uint32_t>();
      bool pad_ok = true;
      for (uint32_t j = 0; j < 8; ++j) pad_ok = pad_ok && r2[j] < 100;
      report_bool("MoE route: the 28 padded lanes never selected", pad_ok);
    }
    {
      const uint32_t N = 9280, S = 2;
      const std::vector<float> part = random_f32(size_t(S) * N, -2.f, 2.f, 22);
      const std::vector<float> bias = random_f32(64, -0.05f, 0.05f, 23);
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "ROUTE_E=64", "ROUTE_K=4", "ROUTE_WG=64", "ROUTE_LN=9280",
                        "ROUTE_LOFF=9216", "ROUTE_LS=2", "ROUTE_SCALE=2.5f"});
      clrun::Buffer pb(dev, part), bb(dev, bias), rt(dev, 32 * 4);
      p.run("k2_route", {64, 1}, {64, 1}, pb, bb, rt);
      const k2_ref::Route want = k2_ref::route(part.data(), 1, 0, N, 9216, S, bias.data(), 64, 4, 2.5f);
      const std::vector<uint32_t> row = rt.read<uint32_t>();
      bool ok = true;
      for (uint32_t j = 0; j < 4; ++j) ok = ok && row[j] == want.ids[j];
      report_bool("MoVA route over two slices of the fused row", ok);
    }
    // --- the MoE block (DN_KS 1 here, see the header) ---------------------------------------------
    {
      const moe_ref::Shape ms = k2_ref::moe_shape(100, 8, H, 768);
      const std::vector<uint32_t> gu = random_blocks(ms.gate_up_words() * 101, 30);
      const std::vector<uint32_t> dn = random_blocks(ms.down_words() * 101, 31);
      k2_ref::Route forced;
      const uint32_t ids[8] = {0, 3, 17, 42, 63, 77, 98, 99};
      const float ws[8] = {0.5f, 0.375f, 0.3125f, 0.25f, 0.4375f, 0.1875f, 0.25f, 0.1875f};
      std::vector<uint32_t> row(32, 0);
      for (uint32_t j = 0; j < 8; ++j) {
        forced.ids[j] = ids[j];
        forced.w[j] = ws[j];
        row[j] = ids[j];
        std::memcpy(&row[8 + j], &ws[j], 4);
      }
      const std::vector<uint16_t> xs = random_bf16(H, -0.1f, 0.1f, 32);
      const std::vector<uint16_t> resid0 = random_bf16(H, -2.f, 2.f, 33);
      clrun::Program p(dev, "src/kernels/k2/k2_moe.cl",
                       {"M=1", "MOE_E=100", "MOE_K=8", "HIDDEN=2560", "INTER=768", "UP_KS=4", "DN_KS=1"});
      clrun::Buffer rb(dev, row), xb(dev, xs), gb(dev, gu), db(dev, dn), hb(dev, size_t(9) * 768 * 2), sb(dev, resid0);
      p.run("k2_moe_gate_up", {size_t(9) * (2 * 768 / 16 / 4) * 256, 1}, {256, 1}, rb, xb, gb, hb);
      const std::vector<uint16_t> hd = hb.read<uint16_t>();
      const std::vector<uint16_t> hr = moe_ref::gate_up(k2_ref::as_moe(forced, 8), xs.data(), gu.data(), ms, 4);
      int worst = 0;
      for (size_t i = 0; i < hd.size(); ++i) worst = std::max(worst, ulps(hd[i], hr[i]));
      report("MoE gate||up (8 slots + shared block 100)", worst, 2, hd.size());
      p.run("k2_moe_down", {size_t(H / 16) * 144, 1}, {144, 1}, rb, hb, db, sb);
      const std::vector<uint16_t> rd = sb.read<uint16_t>();
      const std::vector<uint16_t> rr = k2_ref::moe_down(forced, hd, dn.data(), ms, resid0.data(), 1);
      worst = 0;
      for (uint32_t n = 0; n < H; ++n) worst = std::max(worst, ulps(rd[n], rr[n]));
      report("MoE down + ascending-id combine + residual", worst, 2, H);

      // --- MoVA's value experts ------------------------------------------------------------------
      const uint32_t N = 1024;
      const size_t vblk = size_t(N / 16) * (H / 64) * 136;
      const std::vector<uint32_t> vw = random_blocks(vblk * 64, 40);
      k2_ref::Route vr;
      const uint32_t vids[4] = {2, 31, 32, 63};
      const float vws[4] = {0.625f, 0.75f, 0.5f, 0.625f};
      std::vector<uint32_t> vrow(32, 0);
      for (uint32_t j = 0; j < 4; ++j) {
        vr.ids[j] = vids[j];
        vr.w[j] = vws[j];
        vrow[j] = vids[j];
        std::memcpy(&vrow[8 + j], &vws[j], 4);
      }
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[0] = 5;
      ctrl[1] = 1;
      clrun::Program pv(dev, "src/kernels/k2/k2_moe.cl",
                        {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                         "M=1", "MOVA_E=64", "MOVA_K=4", "MOVA_D=2560", "MOVA_N=1024", "MOVA_KS=4"});
      clrun::Buffer cb(dev, ctrl), vrb(dev, vrow), vwb(dev, vw), kv(dev, size_t(8) * N * 2);
      pv.run("k2_mova_value", {size_t(N / 16) * 256, 1}, {256, 1}, cb, vrb, xb, vwb, kv);
      const std::vector<uint16_t> v = kv.read<uint16_t>();
      const std::vector<uint16_t> want = k2_ref::mova_value(vr, xs.data(), vw.data(), H, N, 4, 4);
      worst = 0;
      for (uint32_t n = 0; n < N; ++n) worst = std::max(worst, ulps(v[size_t(5) * N + n], want[n]));
      report("MoVA value experts into V[pos 5]", worst, 2, N);
    }
    prefill_checks(dev, m);   // spec 18c
    // --- eager attention (B70_K2_ATTN=eager): k2_attn_eager.cl's four kernels --------------------
    {
      const uint32_t ML = 1024, QH = m.q_heads, KVH = m.kv_heads, TGT = 32, QN = m.q_n();
      const std::vector<uint16_t> K = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 60);
      const std::vector<uint16_t> V = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 61);
      std::vector<float> q(QN), g(QN);
      {
        const std::vector<uint16_t> qb = random_bf16(QN, -4.f, 4.f, 62);
        for (uint32_t i = 0; i < QN; ++i) q[i] = f32(qb[i]);
        for (uint32_t i = 0; i < QN; ++i) g[i] = rf(32.0f + float(i % 9));   // softplus = g (no exp)
      }
      std::vector<uint32_t> ctrl(32, 0);
      ctrl[1] = 1;
      clrun::Program p(dev, "src/kernels/k2/k2_attn_eager.cl",
                       {"CTRL_POS=0", "CTRL_NACT=1", "CTRL_CUR=2", "CTRL_OUT=10", "CTRL_DEBUG=18",
                        "M=1", "TGT=32", "Q_HEADS=32", "KV_HEADS=8"});
      clrun::Buffer cb(dev, ctrl), qb(dev, q), gb(dev, g), kb(dev, K), vb(dev, V);
      clrun::Buffer sb(dev, size_t(QH) * ML * 4), part(dev, size_t(QH) * TGT * 130 * 4), ob(dev, size_t(QN) * 2);
      for (uint32_t pos : {5u, 299u, 1023u}) {
        ctrl[0] = pos;
        cb.write(ctrl.data(), ctrl.size() * 4);
        const uint32_t len = pos + 1, blk = k2_ref::eager_block(pos, 1, TGT);
        p.run("k2_attn_eager_score", {size_t(KVH) * 128, TGT}, {128, 1}, cb, qb, kb, sb, ML);
        const std::vector<float> s = sb.read<float>();
        p.run("k2_attn_eager_softmax", {size_t(QH) * 128, 1}, {128, 1}, cb, sb, ML);
        const std::vector<float> pr = sb.read<float>();
        p.run("k2_attn_eager_pv", {size_t(KVH) * 128, TGT}, {128, 1}, cb, sb, vb, part, ML);
        p.run("k2_attn_eager_reduce", {size_t(QH) * 128, 1}, {128, 1}, cb, part, gb, ob);
        const std::vector<uint16_t> o = ob.read<uint16_t>();
        size_t s_bad = 0, p_bad = 0, o_bad = 0, o_own = 0;
        int p_worst = 0, o_worst = 0;
        for (uint32_t h = 0; h < QH; ++h) {
          const uint32_t j = h / (QH / KVH);
          const k2_ref::EagerHead e = k2_ref::attention_eager(q.data() + size_t(h) * 128, K.data(), V.data(),
                                                              len, j, KVH, 128, blk);
          const float* srow = s.data() + size_t(h) * ML;
          const float* prow = pr.data() + size_t(h) * ML;
          for (uint32_t i = 0; i < len; ++i) {
            s_bad += srow[i] != e.s[i];
            if (prow[i] != e.p[i]) {
              ++p_bad;
              p_worst = std::max(p_worst, ulps(rne(prow[i]), rne(e.p[i])));
            }
          }
          // The output from the DEVICE's probabilities (the P·V and gate stages alone) and from
          // the host's (the whole chain).
          const std::vector<float> dev_p(prow, prow + len);
          for (uint32_t dd = 0; dd < 128; ++dd) {
            const uint16_t got = o[size_t(h) * 128 + dd];
            const float gate = g[size_t(h) * 128 + dd];
            const uint16_t own = k2_ref::attn_gate(k2_ref::eager_pv(dev_p.data(), V.data(), len, j, KVH, 128, dd, blk), gate);
            const uint16_t want = k2_ref::attn_gate(e.o[dd], gate);
            o_own += got != own;
            if (got != want) {
              ++o_bad;
              o_worst = std::max(o_worst, ulps(got, want));
            }
          }
        }
        std::printf("  eager attention at %4u keys: scores %zu, probabilities %zu (worst %d ulps), "
                    "output %zu (worst %d) of the host's differ; output vs host P.V on the device's p: %zu\n",
                    len, s_bad, p_bad, p_worst, o_bad, o_worst, o_own);
        report_bool("eager scores bit-exact", s_bad == 0);
        report("eager probabilities (Mac: 1/x not CR)", p_worst, 1, size_t(QH) * len);
        report_bool("eager P.V + gate on the device's p bit-exact", o_own == 0);
        report("eager output", o_worst, 1, QN);
      }
    }
  } catch (const clrun::Error& e) {
    std::fprintf(stderr, "k2_run: %s\n", e.what());
    return 1;
  }
  if (failures) {
    std::printf("DISAGREES with the host reference in %d check(s) (indicative: read the kernel "
                "before the box)\n", failures);
    return 1;
  }
  std::printf("agrees with the host reference (indicative only; the B70 is the real test)\n");
  return 0;
}
