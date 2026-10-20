// Spec 24c Task 3 (F2): LFM2.5-24B-A2B's decode kernels on the card, at the real shapes, against
// tests/kernels/lfm2moe_ref.h (and, where the fixture has them, torch's own values). Needs a B70, no checkpoint
// (synthetic weights). Every binary here is one a decode list binds (kernels::lfm2moe names), plus the M = 4 conv
// build (the M-row form 24d and a verify list use; test-only).
//
//   1. lfm_res_norm _NF and _S1 on 8 work-groups: h_out and x bitwise, against the fixture's torch rows too
//      (Review Focus 3: every work-group's Σ is the same tree - a chunk written by one equals the host's)
//   2. lfm_conv (Review Focus 1): the fixture's 18 positions bitwise from a garbage ring, as M = 1 launches and as
//      M = 1 / 2 / 3 / 4 rows of one launch (the M4 build), the ring holding Bx at pos % 16
//   3. lfm_router + lfm_route (Review Focus 2): the fixture's 64 rows (the router bitwise the host's order; the ids
//      exact; w / sel / sum bitwise but for sigmoid's exp, printed); the crafted rows (a tie at the 4th / 5th to the
//      lower id, all-equal logits, a bias that reorders, one dominant expert)
//   4. lfm_moe_gate_up / lfm_moe_down over the fixture's int4 experts: each slot alone (the others' weights zero)
//      and the whole 4-slot combine + the fold, within 2 bf16 ulps (exp); a replay bitwise
//   5. lfm_attn_prep (S 1) at 0, 1, 4095, 131071 bitwise; lfm_silu_mul within 2 ulps; lfm_embed_gather bitwise;
//      the argmax at 128000 / 125017 (Review Focus 5: a crafted 127999 never wins; ties to the lower id)
//   6. eager attention (lfm_attn_eager.cl) against attention_eager at 6 / 300 / 3000 / 4097 keys, outputs within 1
//      ulp; flash (lfm_attn.cl) against the fp64 form at cosine >= 0.99999 (spec 6 K1) at depths 1, 2, 300, 4096,
//      131072
//   7. the fused arms (Task 5): lfm_res_norm _R against _S1 + lfm_router; lfm_moe _R against lfm_route +
//      lfm_moe_gate_up; lfm_conv_in on the conv-tile in_proj against gemv (S 1) + lfm_conv - bitwise
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/kernels.h"
#include "kernels/lfm2moe_fixture.h"
#include "kernels/lfm2moe_kernels.h"
#include "kernels/lfm2moe_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/lfm2moe_repack.h"
#include "model/lfm2moe.h"
#include "runtime/control.h"

namespace {

namespace kl = kernels::lfm2moe;
namespace lr = lfmref;
namespace fx = lfm2moe_fixture;
using lr::f32;
using lr::rf;
using lr::rne;

static_assert(kl::route::kWords == lr::kWords && kl::route::kIds == lr::kIds && kl::route::kW == lr::kW &&
                  kl::route::kSel == lr::kSel && kl::route::kSum == lr::kSum,
              "the LFM2 route row: lfm_moe.cl R_*, kernels::lfm2moe::route and lfmref disagree");
static_assert(kl::kUpKs == lr::kUpKs && kl::kDnKs == lr::kDnKs, "the MoE K splits: host and twin disagree");

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  }
  template <class T>
  l0::Mem up(const std::vector<T>& v) { return upload(v.data(), v.size() * sizeof(T)); }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, n * sizeof(T));
    return v;
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};
struct Ctrl {
  l0::Mem m;
  explicit Ctrl(Dev& d) : m(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)) {
    std::memset(m.ptr(), 0, sizeof(runtime::Control));
  }
  runtime::Control* c() { return m.as<runtime::Control>(); }
  void set(uint32_t pos, uint32_t n) {
    c()->pos = pos;
    c()->n_active = n;
  }
};

std::vector<float> vals(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step = 0.125f, float amp = 1.0f,
                        uint32_t base = 0) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = lr::fixture_val(coarse, t, base + i, seed, step, amp);
  return v;
}
std::vector<uint16_t> bf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = rne(v[i]);
  return o;
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}

constexpr uint32_t H = kl::kHidden, HD = kl::kHd, QH = kl::kQHeads, KVH = kl::kKvHeads, QN = QH * HD, KVN = KVH * HD;

// --- 1. the norm ----------------------------------------------------------------------------------------------
void test_norm(Dev& d) {
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  l0::Mem wb = d.up(w);
  const std::vector<uint16_t> fx_x = lr::hex16(fx::kNormX), fx_h = lr::hex16(fx::kNormH);
  {
    l0::Module m(d.ctx, kernels::path(kl::res_norm_variant(1, 0, false)));
    for (uint32_t r : {0u, 1u, 2u, 3u}) {
      const std::vector<std::vector<float>> rows = {vals(true, 0, H, 10, 0.125f), vals(false, 0, H, 11, 0.125f, 4.0f),
                                                    vals(true, 0, H, 12, 128.0f), vals(true, 0, H, 13, 0x1p-20f)};
      const std::vector<uint16_t> h = bf(rows[r]);
      l0::Mem hb = d.up(h), sb = d.zeros(64), ob = d.zeros(H * 2), xb = d.zeros(H * 2);
      l0::Kernel k = m.kernel("lfm_res_norm");
      k.group_size(kl::kNormWg);
      k.arg_ptr(0, hb.ptr());
      k.arg_ptr(1, sb.ptr());
      k.arg_ptr(2, wb.ptr());
      k.arg_ptr(3, ob.ptr());
      k.arg_ptr(4, xb.ptr());
      d.run(k, kl::kNormWgs, 1);
      const std::vector<uint16_t> x = d.read<uint16_t>(xb, H), want = lr::res_norm(h.data(), nullptr, 0, 1, w.data()).x;
      CHECK(x == want);
      CHECK(std::equal(x.begin(), x.end(), fx_x.begin() + size_t(r) * H));   // torch's own, on these rows
    }
  }
  {
    l0::Module m(d.ctx, kernels::path(kl::res_norm_variant(1, 1, false)));
    for (uint32_t r = 4; r < 8; ++r) {
      const std::vector<uint16_t> h = bf(vals(false, 0, H, 14 + (r - 4), 0.125f, 2.0f));
      // one slice holding s0 + s1 of the fixture's two (exact): the fold's input
      const std::vector<float> s0 = vals(true, 2, H, 20 + r, 0.125f), s1 = vals(true, 3, H, 20 + r, 0.125f);
      std::vector<float> sl(H);
      for (uint32_t k = 0; k < H; ++k) sl[k] = s0[k] + s1[k];
      l0::Mem hb = d.up(h), sb = d.up(sl), ob = d.zeros(H * 2), xb = d.zeros(H * 2);
      l0::Kernel k = m.kernel("lfm_res_norm");
      k.group_size(kl::kNormWg);
      k.arg_ptr(0, hb.ptr());
      k.arg_ptr(1, sb.ptr());
      k.arg_ptr(2, wb.ptr());
      k.arg_ptr(3, ob.ptr());
      k.arg_ptr(4, xb.ptr());
      d.run(k, kl::kNormWgs, 1);
      const lr::NormOut want = lr::res_norm(h.data(), sl.data(), 1, 1, w.data());
      CHECK(d.read<uint16_t>(ob, H) == want.h_out);
      CHECK(d.read<uint16_t>(xb, H) == want.x);
      CHECK(std::equal(want.h_out.begin(), want.h_out.end(), fx_h.begin() + size_t(r - 4) * H));
      CHECK(d.read<uint16_t>(hb, H) == h);   // h_in untouched: the ping-pong's other buffer
    }
  }
  std::printf("  1. lfm_res_norm _NF / _S1: x and h_out bitwise (8 work-groups, one Σ tree), torch's rows too\n");
}

// --- 2. the conv ring -------------------------------------------------------------------------------------------
void test_conv(Dev& d) {
  const uint32_t T = fx::kConvT, N = 3 * H;
  const std::vector<uint16_t> B = bf(vals(false, 20, T * H, 5, 0.125f, 2.0f)), C = bf(vals(false, 21, T * H, 5, 0.125f, 2.0f)),
                              X = bf(vals(false, 22, T * H, 5, 0.125f, 2.0f));
  const std::vector<float> taps = vals(false, 23, H * 3, 5, 0.125f, 0.5f);
  const std::vector<uint16_t> want_y = lr::hex16(fx::kConvY), want_bx = lr::hex16(fx::kConvBx);
  l0::Mem tb = d.up(taps);
  for (uint32_t Mb : {1u, 4u}) {
    l0::Module m(d.ctx, kernels::path("lfm_conv_M" + std::to_string(Mb) + "_K2048_S1"));
    std::vector<uint16_t> ring(size_t(kl::kConvRing) * H);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage
    l0::Mem rb = d.up(ring), yb = d.zeros(size_t(Mb) * H * 2);
    Ctrl ctl(d);
    const std::vector<uint32_t> steps = Mb == 1 ? std::vector<uint32_t>{1} : std::vector<uint32_t>{1, 2, 3, 4};
    uint32_t p = 0;
    for (size_t i = 0; p < T; ++i) {
      const uint32_t n = std::min(steps[i % steps.size()], T - p);
      std::vector<float> s(size_t(Mb) * N, 0.0f);   // the M-row slice: rows >= n never read
      for (uint32_t r = 0; r < n; ++r)
        for (uint32_t c = 0; c < H; ++c) {
          s[size_t(r) * N + c] = f32(B[size_t(p + r) * H + c]);
          s[size_t(r) * N + H + c] = f32(C[size_t(p + r) * H + c]);
          s[size_t(r) * N + 2 * H + c] = f32(X[size_t(p + r) * H + c]);
        }
      l0::Mem sb = d.up(s);
      ctl.set(p, n);
      l0::Kernel k = m.kernel("lfm_conv");
      k.group_size(kl::kConvWg);
      k.arg_ptr(0, ctl.m.ptr());
      k.arg_ptr(1, sb.ptr());
      k.arg_ptr(2, tb.ptr());
      k.arg_ptr(3, rb.ptr());
      k.arg_ptr(4, yb.ptr());
      d.run(k, H / kl::kConvWg, 1);
      const std::vector<uint16_t> y = d.read<uint16_t>(yb, size_t(n) * H);
      CHECK(std::equal(y.begin(), y.end(), want_y.begin() + size_t(p) * H));
      const std::vector<uint16_t> rg = d.read<uint16_t>(rb, size_t(kl::kConvRing) * H);
      for (uint32_t r = 0; r < n; ++r)
        CHECK(std::equal(rg.begin() + size_t((p + r) % kl::kConvRing) * H, rg.begin() + size_t((p + r) % kl::kConvRing + 1) * H,
                         want_bx.begin() + size_t(p + r) * H));
      p += n;
    }
  }
  std::printf("  2. lfm_conv: positions 0 .. 17 bitwise torch's, M = 1 and M = 1 / 2 / 3 / 4 rows a launch, the ring "
              "holds Bx (garbage before position 0 never read)\n");
}

// --- 3. the router and the route -------------------------------------------------------------------------------
std::vector<uint32_t> run_route(Dev& d, l0::Module& m, const std::vector<float>& logits, const std::vector<float>& bias) {
  l0::Mem lb = d.up(logits), bb = d.up(bias), rt = d.zeros(kl::route::kWords * 4);
  l0::Kernel k = m.kernel("lfm_route");
  k.group_size(kl::kRouteWg);
  k.arg_ptr(0, lb.ptr());
  k.arg_ptr(1, bb.ptr());
  k.arg_ptr(2, rt.ptr());
  d.run(k, 1, 1);
  return d.read<uint32_t>(rt, kl::route::kWords);
}
void test_route(Dev& d) {
  const uint32_t E = kl::kExperts;
  const std::vector<uint16_t> x = bf(vals(false, 30, 64 * H, 6)), gw = bf(vals(false, 31, E * H, 6, 0.125f, 0.05f));
  std::vector<uint16_t> tiles(size_t(H) * E);
  common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
  std::vector<float> bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = (float(lr::fixture_hash(32, e, 6) >> 8) * 0x1p-23f - 1.0f) * 0.05f;
  l0::Module rm(d.ctx, kernels::path(kl::router_variant(1))), tm(d.ctx, kernels::path(kl::route_variant(1)));
  l0::Mem tb = d.up(tiles), xb = d.up(x), lb = d.zeros(size_t(64) * E * 4);
  l0::Kernel k = rm.kernel("lfm_router");
  k.group_size(kl::kRouterWg);
  k.arg_ptr(0, xb.ptr());
  k.arg_ptr(1, tb.ptr());
  k.arg_ptr(2, lb.ptr());
  d.run(k, E / 16, 64);
  const std::vector<float> lg = d.read<float>(lb, size_t(64) * E);
  size_t other = 0;
  for (uint32_t r = 0; r < 64; ++r) {
    const std::vector<float> want = lr::router(x.data() + size_t(r) * H, tiles.data());
    for (uint32_t e = 0; e < E; ++e) CHECK_EQ(lr::as_u32(lg[size_t(r) * E + e]), lr::as_u32(want[e]));
    const std::vector<float> row(lg.begin() + size_t(r) * E, lg.begin() + size_t(r + 1) * E);
    const std::vector<uint32_t> got = run_route(d, tm, row, bias);
    const lr::Route rt = lr::route(row.data(), bias.data());
    for (uint32_t j = 0; j < 4; ++j) CHECK_EQ(got[kl::route::kIds + j], rt.ids[j]);
    for (uint32_t w = 4; w < kl::route::kWords; ++w) other += got[w] != rt.row[w];
  }
  std::printf("  3. lfm_router bitwise the host order on 64 rows; lfm_route ids exact, %zu other words off (sigmoid's "
              "exp)\n", other);
  const std::vector<uint32_t> cl = lr::hex32(fx::kCraftLogits), cb = lr::hex32(fx::kCraftBias);
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> l(E), b(E);
    for (uint32_t e = 0; e < E; ++e) {
      l[e] = lr::as_f32(cl[size_t(r) * E + e]);
      b[e] = lr::as_f32(cb[size_t(r) * E + e]);
    }
    const std::vector<uint32_t> got = run_route(d, tm, l, b);
    const lr::Route rt = lr::route(l.data(), b.data());
    for (uint32_t w = 0; w < kl::route::kWords; ++w) CHECK_EQ(got[w], rt.row[w]);
  }
  std::printf("     crafted rows (tie at the 4th / 5th -> the lower id, all-equal, bias reorders, dominant) bitwise\n");
}

// --- 4. the MoE block -------------------------------------------------------------------------------------------
void test_moe(Dev& d) {
  const uint32_t I = kl::kInter;
  std::vector<uint32_t> used;
  for (const auto& row : fx::kMoeIds)
    for (uint32_t e : row)
      if (std::find(used.begin(), used.end(), e) == used.end()) used.push_back(e);
  std::sort(used.begin(), used.end());
  // The real allocation's addressing: 64 blocks, the fixture's experts at their own ids, random elsewhere.
  std::vector<uint32_t> gu(size_t(kl::kExperts) * lr::gate_up_block_words()), dn(size_t(kl::kExperts) * lr::down_block_words());
  std::mt19937 rng(5);
  for (uint32_t& w : gu) w = rng();
  for (uint32_t& w : dn) w = rng();
  for (uint32_t e : used) {
    const lr::Gptq g = lr::expert_gptq(e, 0, H, I), up = lr::expert_gptq(e, 1, H, I), dd = lr::expert_gptq(e, 2, I, H);
    common::repack_int4_layout1_cols(
        H, 2 * I, common::cols_interleave16({g.qweight.data(), g.scales.data(), I}, {up.qweight.data(), up.scales.data(), I}),
        gu.data() + size_t(e) * lr::gate_up_block_words());
    common::repack_int4_layout1(dd.qweight.data(), dd.scales.data(), I, H, dn.data() + size_t(e) * lr::down_block_words());
  }
  l0::Module m(d.ctx, kernels::path(kl::moe_variant(1, false)));
  l0::Mem gb = d.up(gu), db = d.up(dn);
  const std::vector<uint32_t> wbits = lr::hex32(fx::kMoeW);
  int worst = 0;
  for (uint32_t r = 0; r < 4; ++r)
    for (int alone = -1; alone < 4; ++alone) {   // -1: all four slots; j: slot j alone (the others' weights 0)
      lr::Route rt;
      std::vector<uint32_t> row(kl::route::kWords, 0);
      for (uint32_t k = 0; k < 4; ++k) {
        rt.ids[k] = fx::kMoeIds[r][k];
        rt.w[k] = (alone < 0 || int(k) == alone) ? lr::as_f32(wbits[r * 4 + k]) : 0.0f;
        row[kl::route::kIds + k] = rt.ids[k];
        row[kl::route::kW + k] = lr::as_u32(rt.w[k]);
      }
      const std::vector<uint16_t> x = bf(vals(true, 40, H, 7, 0.125f, 1.0f, r * H)), h0 = random_bf16(H, -1.f, 1.f, 9 + r);
      l0::Mem rb = d.up(row), xb = d.up(x), hb = d.zeros(size_t(4) * I * 2), ob = d.up(h0);
      l0::Kernel g = m.kernel("lfm_moe_gate_up");
      g.group_size(kl::moe_gate_up_wg());
      g.arg_ptr(0, rb.ptr());
      g.arg_ptr(1, xb.ptr());
      g.arg_ptr(2, gb.ptr());
      g.arg_ptr(3, hb.ptr());
      d.run(g, kl::moe_gate_up_groups(), 1);
      l0::Kernel dk = m.kernel("lfm_moe_down");
      dk.group_size(kl::moe_down_wg());
      dk.arg_ptr(0, rb.ptr());
      dk.arg_ptr(1, hb.ptr());
      dk.arg_ptr(2, db.ptr());
      dk.arg_ptr(3, ob.ptr());
      d.run(dk, H / 16, 1);
      const lr::MoeOut want = lr::moe(rt, x.data(), gu.data(), dn.data(), h0.data());
      const std::vector<uint16_t> h = d.read<uint16_t>(hb, size_t(4) * I), o = d.read<uint16_t>(ob, H);
      for (size_t i = 0; i < h.size(); ++i) worst = std::max(worst, ulps(h[i], want.h[i]));
      for (uint32_t n = 0; n < H; ++n) worst = std::max(worst, ulps(o[n], want.h_res[n]));
      if (alone < 0 && r == 0) {   // a replay: bitwise
        d.imm.copy(ob.ptr(), h0.data(), H * 2);
        d.run(g, kl::moe_gate_up_groups(), 1);
        d.run(dk, H / 16, 1);
        CHECK(d.read<uint16_t>(ob, H) == o);
      }
    }
  CHECK(worst <= 2);
  std::printf("  4. lfm_moe_gate_up / lfm_moe_down: each slot alone and all four (rank order, the fold) within %d bf16 "
              "ulps (bar 2: exp); replay bitwise\n", worst);
}

// --- 5. prep, silu, gather, argmax ---------------------------------------------------------------------------
void test_small(Dev& d) {
  {   // the attention prep (S 1)
    const std::vector<float> table = loader::lfm_rope_table(model::lfm2moe(), 131072);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> sl(kl::kQkvN), qkn(2 * HD);
    for (float& v : sl) v = u(rng);
    for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
    l0::Module m(d.ctx, kernels::path(kl::attn_prep_variant(1, 1)));
    l0::Mem rb = d.up(table), sb = d.up(sl), qb = d.up(qkn), aq = d.zeros(QN * 4);
    l0::Mem kb = d.zeros(size_t(131072) * KVN * 2), vb = d.zeros(size_t(131072) * KVN * 2);
    Ctrl ctl(d);
    for (uint32_t pos : {0u, 1u, 4095u, 131071u}) {
      ctl.set(pos, 1);
      l0::Kernel k = m.kernel("lfm_attn_prep");
      k.group_size(kl::kPrepWg);
      k.arg_ptr(0, ctl.m.ptr());
      k.arg_ptr(1, sb.ptr());
      k.arg_ptr(2, qb.ptr());
      k.arg_ptr(3, rb.ptr());
      k.arg_ptr(4, aq.ptr());
      k.arg_ptr(5, kb.ptr());
      k.arg_ptr(6, vb.ptr());
      d.run(k, QH + 2 * KVH, 1);
      const lr::Prep want = lr::attn_prep(sl.data(), 1, 1, 0, qkn.data(), table.data() + size_t(pos) * HD);
      CHECK(d.read<float>(aq, QN) == want.q);
      CHECK(d.read<uint16_t>(kb, KVN, size_t(pos) * KVN * 2) == want.k);
      CHECK(d.read<uint16_t>(vb, KVN, size_t(pos) * KVN * 2) == want.v);
    }
  }
  {   // SiLU x up at 11776 (S 1)
    const uint32_t I = kl::kDenseInter;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-3.f, 3.f);
    std::vector<float> sl(size_t(2) * I);
    for (float& v : sl) v = u(rng);
    l0::Module m(d.ctx, kernels::path(kl::silu_mul_variant(1, 1)));
    l0::Mem sb = d.up(sl), ob = d.zeros(size_t(I) * 2);
    l0::Kernel k = m.kernel("lfm_silu_mul");
    k.group_size(kl::kSiluWg);
    k.arg_ptr(0, sb.ptr());
    k.arg_ptr(1, ob.ptr());
    d.run(k, (I + kl::kSiluWg - 1) / kl::kSiluWg, 1);
    const std::vector<uint16_t> got = d.read<uint16_t>(ob, I), want = lr::silu_mul(sl.data(), 1, 1);
    int worst = 0;
    for (uint32_t i = 0; i < I; ++i) worst = std::max(worst, ulps(got[i], want[i]));
    CHECK(worst <= 2);
  }
  {   // the gather out of the tiles
    std::vector<uint16_t> tiles(size_t(H) * kl::kVocab);
    std::mt19937 rng(13);
    for (uint16_t& t : tiles) t = uint16_t(rng());
    l0::Module m(d.ctx, kernels::path(kl::embed_variant(1)));
    l0::Mem tb = d.up(tiles), hb = d.zeros(H * 2);
    Ctrl ctl(d);
    for (uint32_t id : {0u, 7u, 125016u, 127999u}) {
      ctl.set(0, 1);
      ctl.c()->cur_token[0] = id;
      l0::Kernel k = m.kernel("lfm_embed_gather");
      k.group_size(kl::kEmbedWg);
      k.arg_ptr(0, ctl.m.ptr());
      k.arg_ptr(1, tb.ptr());
      k.arg_ptr(2, hb.ptr());
      d.run(k, 1, 1);
      CHECK(d.read<uint16_t>(hb, H) == lr::embed_row(tiles.data(), id));
    }
  }
  {   // the argmax: the fixture's logits (125016 wins over a planted 127999), then a tie at the top
    std::vector<float> lg = vals(false, 90, kl::kVocab, 4);
    lg[125016] = 50.0f;
    lg[127999] = 100.0f;
    l0::Module m1(d.ctx, kernels::path(kl::argmax1_variant(1))), m2(d.ctx, kernels::path(kl::argmax2_variant()));
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) lg[100] = lg[200] = 60.0f;
      l0::Mem lb = d.up(lg), pb = d.zeros(size_t(kl::argmax_groups()) * 2 * 4);
      Ctrl ctl(d);
      ctl.set(5, 1);
      l0::Kernel a = m1.kernel("argmax_stage1");
      a.group_size(kl::kArgmaxWg);
      a.arg_ptr(0, lb.ptr());
      a.arg_ptr(1, pb.ptr());
      d.run(a, kl::argmax_groups(), 1);
      l0::Kernel b = m2.kernel("argmax_stage2");
      b.group_size(kl::kArgmaxWg);
      b.arg_ptr(0, ctl.m.ptr());
      b.arg_ptr(1, pb.ptr());
      d.run(b, 1, 1);
      CHECK_EQ(ctl.c()->out_token[0], pass == 0 ? fx::kArgmax : 100u);
      CHECK_EQ(ctl.c()->out_token[0], lr::argmax(lg.data()));
      CHECK_EQ(ctl.c()->pos, 6u);
    }
  }
  std::printf("  5. lfm_attn_prep bitwise at 0, 1, 4095, 131071; lfm_silu_mul within 2 ulps; the gather bitwise; the "
              "argmax never past 125016 (a planted 127999), ties to the lower id\n");
}

// --- 6. attention -----------------------------------------------------------------------------------------------
void test_attention(Dev& d) {
  const uint32_t TGT = kl::kAttnTgt;
  l0::Module me(d.ctx, kernels::path(kl::attn_variant(1, true))), mf(d.ctx, kernels::path(kl::attn_variant(1, false)));
  for (uint32_t L : {1u, 2u, 6u, 300u, 3000u, 4096u, 4097u, 131072u}) {
    const uint32_t pos = L - 1, rows = L;
    const std::vector<uint16_t> K = random_bf16(size_t(rows) * KVN, -1.f, 1.f, 50 + L), V = random_bf16(size_t(rows) * KVN, -1.f, 1.f, 60 + L);
    const std::vector<uint16_t> qb16 = random_bf16(QN, -4.f, 4.f, 70 + L);
    std::vector<float> q(QN);
    for (uint32_t i = 0; i < QN; ++i) q[i] = f32(qb16[i]);
    l0::Mem kb = d.up(K), vb = d.up(V), qb = d.up(q), part = d.zeros(size_t(QH) * TGT * kl::kAttnPart * 4), ob = d.zeros(QN * 2);
    Ctrl ctl(d);
    ctl.set(pos, 1);
    // flash
    {
      l0::Kernel k = mf.kernel("lfm_attn_decode");
      k.group_size(kl::kAttnWg);
      k.arg_ptr(0, ctl.m.ptr());
      k.arg_ptr(1, qb.ptr());
      k.arg_ptr(2, kb.ptr());
      k.arg_ptr(3, vb.ptr());
      k.arg_ptr(4, part.ptr());
      d.run(k, KVH, TGT);
      l0::Kernel r = mf.kernel("lfm_attn_reduce");
      r.group_size(kl::kAttnWg);
      r.arg_ptr(0, ctl.m.ptr());
      r.arg_ptr(1, part.ptr());
      r.arg_ptr(2, ob.ptr());
      d.run(r, QH, 1);
      const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
      double worst = 1.0;
      for (uint32_t h = 0; h < QH; ++h) {
        const std::vector<double> want = lr::attention(q.data() + size_t(h) * HD, K.data(), V.data(), L, h / 4);
        double dot = 0, na = 0, nb = 0;
        for (uint32_t dd = 0; dd < HD; ++dd) {
          const double a = f32(o[size_t(h) * HD + dd]);
          dot += a * want[dd];
          na += a * a;
          nb += want[dd] * want[dd];
        }
        worst = std::min(worst, dot / std::sqrt(na * nb + 1e-300));
      }
      std::printf("  6. flash at %u keys: worst head cosine %.7f against fp64 (bar 0.99999)\n", L, worst);
      CHECK(worst >= 0.99999);
    }
    if (L > 4097) continue;   // eager's score row at 131072 is the box's speed stage, not this test
    {
      const uint32_t stride = L;
      l0::Mem sb = d.zeros(size_t(QH) * stride * 4);
      {
        l0::Kernel k = me.kernel("lfm_attn_eager_score");
        k.group_size(kl::kAttnWg);
        k.arg_ptr(0, ctl.m.ptr());
        k.arg_ptr(1, qb.ptr());
        k.arg_ptr(2, kb.ptr());
        k.arg_ptr(3, sb.ptr());
        k.arg<uint32_t>(4, stride);
        d.run(k, KVH, TGT);
      }
      {
        l0::Kernel k = me.kernel("lfm_attn_eager_softmax");
        k.group_size(kl::kAttnWg);
        k.arg_ptr(0, ctl.m.ptr());
        k.arg_ptr(1, sb.ptr());
        k.arg<uint32_t>(2, stride);
        d.run(k, QH, 1);
      }
      {
        l0::Kernel k = me.kernel("lfm_attn_eager_pv");
        k.group_size(kl::kAttnWg);
        k.arg_ptr(0, ctl.m.ptr());
        k.arg_ptr(1, sb.ptr());
        k.arg_ptr(2, vb.ptr());
        k.arg_ptr(3, part.ptr());
        k.arg<uint32_t>(4, stride);
        d.run(k, KVH, TGT);
      }
      {
        l0::Kernel k = me.kernel("lfm_attn_eager_reduce");
        k.group_size(kl::kAttnWg);
        k.arg_ptr(0, ctl.m.ptr());
        k.arg_ptr(1, part.ptr());
        k.arg_ptr(2, ob.ptr());
        d.run(k, QH, 1);
      }
      const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
      int worst = 0;
      for (uint32_t h = 0; h < QH; ++h) {
        const lr::EagerHead e = lr::attention_eager(q.data() + size_t(h) * HD, K.data(), V.data(), L, h / 4, lr::eager_block(L));
        for (uint32_t dd = 0; dd < HD; ++dd) worst = std::max(worst, ulps(o[size_t(h) * HD + dd], rne(e.o[dd])));
      }
      std::printf("     eager at %u keys: outputs within %d ulp of attention_eager (bar 1)\n", L, worst);
      CHECK(worst <= 1);
    }
  }
}

// --- 7. the fused arms (Task 5, Review Focus 4): each arm's kernel against the as-built pair, bitwise ------------------
std::vector<uint32_t> random_words(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> v(n);
  for (uint32_t& w : v) w = rng();
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
void test_arms(Dev& d) {
  const uint32_t E = kl::kExperts;
  const model::Lfm2MoeDesc& desc = model::lfm2moe();
  {   // router: lfm_res_norm _S1 _R against lfm_res_norm _S1 + lfm_router
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> sl(H), w(H);
    for (float& v : sl) v = u(rng);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.5f + float(k % 9) * 0.1f);
    const std::vector<uint16_t> h = random_bf16(H, -2.f, 2.f, 22), rw = random_bf16(size_t(H) * E, -0.05f, 0.05f, 23);
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(rw.data(), H, E, tiles.data());
    l0::Mem hb = d.up(h), sb = d.up(sl), wb = d.up(w), tb = d.up(tiles);
    l0::Mem o1 = d.zeros(H * 2), x1 = d.zeros(H * 2), o2 = d.zeros(H * 2), x2 = d.zeros(H * 2);
    l0::Mem l1 = d.zeros(E * 4), l2 = d.zeros(E * 4);
    l0::Module mn(d.ctx, kernels::path(kl::res_norm_variant(1, 1, false))), mr(d.ctx, kernels::path(kl::router_variant(1))),
        mf(d.ctx, kernels::path(kl::res_norm_variant(1, 1, true)));
    l0::Kernel n = mn.kernel("lfm_res_norm");
    n.group_size(kl::kNormWg);
    n.arg_ptr(0, hb.ptr());
    n.arg_ptr(1, sb.ptr());
    n.arg_ptr(2, wb.ptr());
    n.arg_ptr(3, o1.ptr());
    n.arg_ptr(4, x1.ptr());
    d.run(n, kl::kNormWgs, 1);
    l0::Kernel r = mr.kernel("lfm_router");
    r.group_size(kl::kRouterWg);
    r.arg_ptr(0, x1.ptr());
    r.arg_ptr(1, tb.ptr());
    r.arg_ptr(2, l1.ptr());
    d.run(r, E / 16, 1);
    l0::Kernel f = mf.kernel("lfm_res_norm");
    f.group_size(kl::kNormWg);
    f.arg_ptr(0, hb.ptr());
    f.arg_ptr(1, sb.ptr());
    f.arg_ptr(2, wb.ptr());
    f.arg_ptr(3, o2.ptr());
    f.arg_ptr(4, x2.ptr());
    f.arg_ptr(5, tb.ptr());
    f.arg_ptr(6, l2.ptr());
    d.run(f, kl::kNormWgs + kl::kNormRouterWgs, 1);
    CHECK(d.read<uint16_t>(o1, H) == d.read<uint16_t>(o2, H));
    CHECK(d.read<uint16_t>(x1, H) == d.read<uint16_t>(x2, H));
    CHECK(d.read<uint32_t>(l1, E) == d.read<uint32_t>(l2, E));
    std::printf("  7. arm router: lfm_res_norm _S1 _R's h, x and 64 logits bitwise _S1 + lfm_router\n");
  }
  {   // route: lfm_moe _R's gate||up (the route recomputed in every work-group) against lfm_route + lfm_moe_gate_up
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> u(-3.f, 3.f), ub(-0.05f, 0.05f);
    std::vector<float> lg(E), bias(E);
    for (float& v : lg) v = u(rng);
    for (float& v : bias) v = ub(rng);
    const std::vector<uint32_t> gu = random_blocks(size_t(E) * lfmref::gate_up_block_words(), 32);
    const std::vector<uint16_t> x = random_bf16(H, -0.5f, 0.5f, 33);
    l0::Mem lb = d.up(lg), bb = d.up(bias), gb = d.up(gu), xb = d.up(x);
    l0::Mem r1 = d.zeros(kl::route::kWords * 4), r2 = d.zeros(kl::route::kWords * 4);
    l0::Mem h1 = d.zeros(size_t(4) * kl::kInter * 2), h2 = d.zeros(size_t(4) * kl::kInter * 2);
    l0::Module mt(d.ctx, kernels::path(kl::route_variant(1))), mm(d.ctx, kernels::path(kl::moe_variant(1, false))),
        mf(d.ctx, kernels::path(kl::moe_variant(1, true)));
    l0::Kernel t = mt.kernel("lfm_route");
    t.group_size(kl::kRouteWg);
    t.arg_ptr(0, lb.ptr());
    t.arg_ptr(1, bb.ptr());
    t.arg_ptr(2, r1.ptr());
    d.run(t, 1, 1);
    l0::Kernel g = mm.kernel("lfm_moe_gate_up");
    g.group_size(kl::moe_gate_up_wg());
    g.arg_ptr(0, r1.ptr());
    g.arg_ptr(1, xb.ptr());
    g.arg_ptr(2, gb.ptr());
    g.arg_ptr(3, h1.ptr());
    d.run(g, kl::moe_gate_up_groups(), 1);
    l0::Kernel f = mf.kernel("lfm_moe_gate_up");
    f.group_size(kl::moe_gate_up_wg());
    f.arg_ptr(0, lb.ptr());
    f.arg_ptr(1, bb.ptr());
    f.arg_ptr(2, r2.ptr());
    f.arg_ptr(3, xb.ptr());
    f.arg_ptr(4, gb.ptr());
    f.arg_ptr(5, h2.ptr());
    d.run(f, kl::moe_gate_up_groups(), 1);
    CHECK(d.read<uint32_t>(r1, kl::route::kWords) == d.read<uint32_t>(r2, kl::route::kWords));
    CHECK(d.read<uint16_t>(h1, size_t(4) * kl::kInter) == d.read<uint16_t>(h2, size_t(4) * kl::kInter));
    std::printf("  7. arm route: lfm_moe _R's route row and gate||up bitwise lfm_route + lfm_moe_gate_up\n");
  }
  {   // conv: lfm_conv_in on the conv-tile in_proj against gemv (S 1) + lfm_conv, positions 0 .. 5 from a garbage ring
    model::GemvShape s{};
    for (const model::LfmLinear& ln : desc.linears(0))
      if (ln.id == model::LfmLinearId::InProj) s = ln.shape;
    CHECK(s.S == 1);
    const uint32_t K = H, N = 3 * H;
    const std::vector<uint32_t> qw = random_words(size_t(K / 8) * N, 41);
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (uint16_t& v : sc) v = common::f32_to_f16(sd(rng));
    const loader::HostLinear plain = loader::repack_in_proj(qw.data(), sc.data(), K, N, false, s.layout),
                             tiled = loader::repack_in_proj(qw.data(), sc.data(), K, N, true, s.layout);
    const std::vector<float> taps = [] {
      std::vector<float> t(size_t(H) * 3);
      std::mt19937 r(43);
      std::uniform_real_distribution<float> u(-0.5f, 0.5f);
      for (float& v : t) v = rf(u(r));
      return t;
    }();
    std::vector<uint16_t> ring0(size_t(kl::kConvRing) * H);
    for (size_t i = 0; i < ring0.size(); ++i) ring0[i] = uint16_t(0x3F80 + (i * 7919) % 4000);
    l0::Mem wp = d.up(plain.words), wt = d.up(tiled.words);
    std::unique_ptr<l0::Mem> sp, st;
    if (s.layout == 0) {
      sp = std::make_unique<l0::Mem>(d.upload(plain.scales.data(), plain.scales.size() * 2));
      st = std::make_unique<l0::Mem>(d.upload(tiled.scales.data(), tiled.scales.size() * 2));
    }
    l0::Mem tb = d.up(taps), rp = d.up(ring0), rt = d.up(ring0), slices = d.zeros(size_t(N) * 4);
    l0::Mem yp = d.zeros(H * 2), yt = d.zeros(H * 2);
    l0::Module mg(d.ctx, kernels::path(kernels::gemv_variant(1, K, N, 1, s.layout))),
        mc(d.ctx, kernels::path(kl::conv_variant(1, 1))), mi(d.ctx, kernels::path(kl::conv_in_variant(1, s.layout)));
    Ctrl ctl(d);
    for (uint32_t pos = 0; pos < 6; ++pos) {
      const std::vector<uint16_t> x = random_bf16(H, -1.f, 1.f, 50 + pos);
      l0::Mem xb = d.up(x);
      ctl.set(pos, 1);
      l0::Kernel g = mg.kernel("gemv");
      g.group_size(64);
      g.arg_ptr(0, wp.ptr());
      g.arg_ptr(1, s.layout == 0 ? sp->ptr() : wp.ptr());
      g.arg_ptr(2, xb.ptr());
      g.arg_ptr(3, slices.ptr());
      d.run(g, N / 64, 1);
      l0::Kernel c = mc.kernel("lfm_conv");
      c.group_size(kl::kConvWg);
      c.arg_ptr(0, ctl.m.ptr());
      c.arg_ptr(1, slices.ptr());
      c.arg_ptr(2, tb.ptr());
      c.arg_ptr(3, rp.ptr());
      c.arg_ptr(4, yp.ptr());
      d.run(c, H / kl::kConvWg, 1);
      l0::Kernel i = mi.kernel("lfm_conv_in");
      i.group_size(kl::kConvInWg);
      i.arg_ptr(0, ctl.m.ptr());
      i.arg_ptr(1, wt.ptr());
      i.arg_ptr(2, s.layout == 0 ? st->ptr() : wt.ptr());
      i.arg_ptr(3, xb.ptr());
      i.arg_ptr(4, tb.ptr());
      i.arg_ptr(5, rt.ptr());
      i.arg_ptr(6, yt.ptr());
      d.run(i, H / 16, 1);
      CHECK(d.read<uint16_t>(yp, H) == d.read<uint16_t>(yt, H));
      CHECK(d.read<uint16_t>(rp, size_t(kl::kConvRing) * H) == d.read<uint16_t>(rt, size_t(kl::kConvRing) * H));
    }
    std::printf("  7. arm conv: lfm_conv_in (conv-tile in_proj, layout %u) bitwise gemv S1 + lfm_conv, y and the ring, "
                "positions 0 .. 5\n", s.layout);
  }
}

}  // namespace

int main() {
  try {
    Dev d;
    std::printf("lfm2moe_kernels_test on %s\n", d.ctx.name().c_str());
    test_norm(d);
    test_conv(d);
    test_route(d);
    test_moe(d);
    test_small(d);
    test_attention(d);
    test_arms(d);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lfm2moe_kernels_test: %s\n", e.what());
    return 1;
  }
  std::printf("lfm2moe_kernels_test OK\n");
  return 0;
}
