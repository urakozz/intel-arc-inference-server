// Spec 20c Task 4 (K1): Kolibri-1's kernels on the card, at the real shapes, against
// tests/kernels/kolibri_ref.h. Needs a B70, no checkpoint (synthetic weights). Every binary here is
// one the decode list binds (kernels::kolibri names).
//
//   1. the norm: prep_res_fold SP0 + kol_norm_finish - resid and x bitwise; the sandwich:
//      prep_res_fold SP4 _Z over o_proj's slices + kol_post_add - a, resid and both Σ² bitwise
//      (no exp anywhere; sqrt / divide correctly rounded)
//   2. kol_attn_prep S2, sliding at 37 and 4097 (ring row 1) and full at 0 and 100000 - q, k, v
//      bitwise; a full layer's k at 0 equals its k at 100000 (NoPE, Review Focus 4)
//   3. the router: gemv_bf16 2560 -> 512 (rows 384.. zero) + kol_route - ids exact, weights within 2
//      ulp; crafted rows: a tie at the cut to the lower id, every real logit -1e30 (ids 0..5), the
//      padded slots' logits +80 never selected (Review Focus 3)
//   4. kol_moe_gate_up / kol_moe_down with a forced route row (ids 0 .. 383 spread, the bf16 shared
//      slot), within 2 bf16 ulps; then the shared slot alone (routed weights zero) and the routed
//      slots alone (shared weights zero); bitwise replay
//   5. eager attention (kol_attn_eager.cl, both builds) bitwise against attention_eager: sliding at
//      5, 512, 513, 4103 (a wrapped ring), 9000; full at 30000
//   6. flash (kol_attn.cl) against the fp64 form within spec 10's bar (cosine 0.99999): sliding at
//      1, 513, 4103, full at 1 and 30000; and sliding at 9000 bitwise sliding at 808 (the same ring
//      rows: the ring is only addressing)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"
#include "kernels/kolibri_ref.h"
#include "kernels/prep_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/kolibri1_repack.h"
#include "model/kolibri1.h"
#include "runtime/control.h"

namespace {

namespace kk = kernels::kolibri;
namespace kr = kolibri_ref;
using kr::f32;
using kr::rf;
using kr::rne;

static_assert(kk::route::kWords == kr::kWords && kk::route::kIds == kr::kIds &&
                  kk::route::kWeights == kr::kWeights && kk::route::kSel == kr::kSel &&
                  kk::route::kNext == kr::kNext && kk::route::kLogitMin == kr::kLogitMin,
              "the Kolibri route row: kol_moe.cl R_*, kernels::kolibri::route and kolibri_ref disagree");

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
std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(rng);
  return v;
}

const uint32_t H = kk::kHidden, HD = kk::kHd, QH = kk::kQHeads, KVH = kk::kKvHeads, QN = QH * HD, KVN = KVH * HD;

// One route launch over `logits` [512], bias [512]: the row back.
std::vector<uint32_t> run_route(Dev& d, l0::Module& m, const std::vector<float>& logits, const std::vector<float>& bias) {
  l0::Mem lb = d.upload(logits.data(), logits.size() * 4), bb = d.upload(bias.data(), bias.size() * 4);
  l0::Mem rt = d.zeros(kr::kWords * 4);
  l0::Kernel k = m.kernel("kol_route");
  k.group_size(kk::kRouteWg);
  k.arg_ptr(0, lb.ptr());
  k.arg_ptr(1, bb.ptr());
  k.arg_ptr(2, rt.ptr());
  d.run(k, 1, 1);
  return d.read<uint32_t>(rt, kr::kWords);
}

}  // namespace

int main() {
  Dev d;
  const model::Kolibri1Desc& m = model::kolibri1();
  const uint32_t G = kk::kNormG;

  // ---- 1. the norm and the sandwich -------------------------------------------------------------
  {
    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 1);
    std::vector<float> w(H), w2(H);
    for (uint32_t k = 0; k < H; ++k) {
      w[k] = rf(0.3f + float(k % 11) * 0.1f);
      w2[k] = rf(1.5f + float(k % 7) * 0.2f);
    }
    std::vector<uint16_t> r_ref = resid, x_ref(H);
    std::vector<float> ss_ref(G);
    prep_ref::res_fold(nullptr, r_ref.data(), ss_ref.data(), 1, H, 0, G);
    kr::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), 1);
    l0::Mem rb = d.upload(resid.data(), H * 2), wb = d.upload(w.data(), H * 4), ss = d.zeros(G * 4), xb = d.zeros(H * 2);
    l0::Module fm(d.ctx, kernels::path(kk::fold_variant(1)));
    l0::Kernel kf = fm.kernel("prep_res_fold");
    kf.group_size(256);
    kf.arg_ptr(0, rb.ptr());   // S_PREV 0 reads no partials
    kf.arg_ptr(1, rb.ptr());
    kf.arg_ptr(2, ss.ptr());
    d.run(kf, G, 1);
    l0::Module nm(d.ctx, kernels::path(kk::norm_variant(1)));
    l0::Kernel kn = nm.kernel("kol_norm_finish");
    kn.group_size(kk::kNormWg);
    kn.arg_ptr(0, ss.ptr());
    kn.arg_ptr(1, rb.ptr());
    kn.arg_ptr(2, wb.ptr());
    kn.arg_ptr(3, xb.ptr());
    d.run(kn, kk::kNormW, 1);
    CHECK(d.read<uint16_t>(rb, H) == r_ref);
    CHECK(d.read<uint16_t>(xb, H) == x_ref);
    CHECK(d.read<float>(ss, G) == ss_ref);
    // the sandwich: o_proj's 4 slices -> a (ZERO_RESID) and Σa², then the post norm + add
    const std::vector<float> part = random_f32(size_t(4) * H, -0.5f, 0.5f, 2);
    std::vector<uint16_t> a_ref(H, 0), r2_ref = r_ref;
    std::vector<float> sa_ref(G), so_ref(G);
    for (uint32_t k = 0; k < H; ++k) {
      float acc = 0.f;
      for (uint32_t s = 0; s < 4; ++s) acc += part[size_t(s) * H + k];
      a_ref[k] = rne(acc);
    }
    {
      std::vector<uint16_t> tmp = a_ref;
      prep_ref::res_fold(nullptr, tmp.data(), sa_ref.data(), 1, H, 0, G);
    }
    kr::post_add(sa_ref.data(), a_ref.data(), w2.data(), r2_ref.data(), so_ref.data(), 1);
    l0::Mem pb = d.upload(part.data(), part.size() * 4), ab = d.zeros(H * 2), sa = d.zeros(G * 4), so = d.zeros(G * 4);
    l0::Mem w2b = d.upload(w2.data(), H * 4);
    l0::Module zm(d.ctx, kernels::path(kk::fold_zero_variant(1, 4)));
    l0::Kernel kz = zm.kernel("prep_res_fold");
    kz.group_size(256);
    kz.arg_ptr(0, pb.ptr());
    kz.arg_ptr(1, ab.ptr());
    kz.arg_ptr(2, sa.ptr());
    d.run(kz, G, 1);
    l0::Module am(d.ctx, kernels::path(kk::post_add_variant(1)));
    l0::Kernel ka = am.kernel("kol_post_add");
    ka.group_size(256);
    ka.arg_ptr(0, sa.ptr());
    ka.arg_ptr(1, ab.ptr());
    ka.arg_ptr(2, w2b.ptr());
    ka.arg_ptr(3, rb.ptr());
    ka.arg_ptr(4, so.ptr());
    d.run(ka, G, 1);
    CHECK(d.read<uint16_t>(ab, H) == a_ref);
    CHECK(d.read<float>(sa, G) == sa_ref);
    CHECK(d.read<uint16_t>(rb, H) == r2_ref);
    CHECK(d.read<float>(so, G) == so_ref);
    std::puts("  norm + sandwich: resid, x, a, both Σ² bitwise");
  }

  // ---- 2. the attention prep -------------------------------------------------------------------
  {
    const uint32_t S = m.qkv_s, ML = 100001;
    const std::vector<float> rope = loader::kol_rope_table(m, ML);
    l0::Mem ropeb = d.upload(rope.data(), rope.size() * 4);
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    runtime::Control* c = ctrl.as<runtime::Control>();
    std::memset(c, 0, sizeof(*c));
    c->n_active = 1;
    const std::vector<float> part = random_f32(size_t(S) * kk::kQkvN, -1.f, 1.f, 10);
    std::vector<float> qkn(2 * HD);
    for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
    l0::Mem pb = d.upload(part.data(), part.size() * 4), qb = d.upload(qkn.data(), qkn.size() * 4);
    l0::Mem aq = d.zeros(size_t(QN) * 4);
    l0::Mem ring_k = d.zeros(size_t(kk::kRing) * KVN * 2), ring_v = d.zeros(size_t(kk::kRing) * KVN * 2);
    l0::Mem full_k = d.zeros(size_t(ML) * KVN * 2), full_v = d.zeros(size_t(ML) * KVN * 2);
    std::vector<uint16_t> k_at0;
    for (int sliding = 1; sliding >= 0; --sliding)
      for (uint32_t pos : sliding ? std::vector<uint32_t>{37, 4097} : std::vector<uint32_t>{0, 100000}) {
        c->pos = pos;
        l0::Module pm(d.ctx, kernels::path(kk::attn_prep_variant(1, S, sliding)));
        l0::Kernel kp = pm.kernel("kol_attn_prep");
        kp.group_size(kk::kPrepWg);
        kp.arg_ptr(0, ctrl.ptr());
        kp.arg_ptr(1, pb.ptr());
        kp.arg_ptr(2, qb.ptr());
        kp.arg_ptr(3, ropeb.ptr());
        kp.arg_ptr(4, aq.ptr());
        kp.arg_ptr(5, sliding ? ring_k.ptr() : full_k.ptr());
        kp.arg_ptr(6, sliding ? ring_v.ptr() : full_v.ptr());
        d.run(kp, QH + 2 * KVH, 1);
        const kr::Prep want = kr::attn_prep(part.data(), 1, 0, S, qkn.data(), sliding ? rope.data() + size_t(pos) * HD : nullptr);
        const size_t row = kr::ring_row(pos, sliding);
        CHECK(d.read<float>(aq, QN) == want.q);
        const std::vector<uint16_t> k = d.read<uint16_t>(sliding ? ring_k : full_k, KVN, row * KVN * 2);
        CHECK(k == want.k);
        CHECK(d.read<uint16_t>(sliding ? ring_v : full_v, KVN, row * KVN * 2) == want.v);
        if (!sliding && pos == 0) k_at0 = k;
        if (!sliding && pos == 100000) CHECK(k == k_at0);   // NoPE: position-free
      }
    std::puts("  attn prep: q, k, v bitwise (sliding at 37 and 4097 -> ring row 1; full at 0 and 100000, k equal)");
  }

  // ---- 3. the router -----------------------------------------------------------------------------
  l0::Module rm(d.ctx, kernels::path(kk::route_variant(1)));
  {
    // the GEMV: router rows [384][2560] bf16, zero rows to 512, tiled
    const std::vector<uint16_t> rows384 = random_bf16(size_t(kk::kExperts) * H, -0.06f, 0.06f, 20);
    std::vector<uint16_t> rows(size_t(kk::kRouterN) * H, 0), tiles(rows.size());
    std::copy(rows384.begin(), rows384.end(), rows.begin());
    common::repack_bf16_tiled(rows.data(), H, kk::kRouterN, tiles.data());
    const std::vector<uint16_t> x = random_bf16(H, -2.f, 2.f, 21);
    l0::Mem wb = d.upload(tiles.data(), tiles.size() * 2), xb = d.upload(x.data(), H * 2), lb = d.zeros(kk::kRouterN * 4);
    l0::Module gm(d.ctx, kernels::path(kk::router_variant()));
    l0::Kernel kg = gm.kernel("gemv_bf16");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(kk::kRouterN);
    kg.group_size(t.cols * t.ksplit);
    kg.arg_ptr(0, wb.ptr());
    kg.arg_ptr(1, xb.ptr());
    kg.arg_ptr(2, lb.ptr());
    d.run(kg, kk::kRouterN / t.cols);
    const std::vector<float> logits = d.read<float>(lb, kk::kRouterN);
    for (uint32_t e = kk::kExperts; e < kk::kRouterN; ++e) CHECK(logits[e] == 0.0f);
    std::vector<float> bias = random_f32(kk::kRouterN, -0.5f, 0.5f, 22);
    for (uint32_t e = kk::kExperts; e < kk::kRouterN; ++e) bias[e] = 0.0f;
    const std::vector<uint32_t> row = run_route(d, rm, logits, bias);
    const kr::Route want = kr::route(logits.data(), bias.data());
    int worst = 0;
    for (uint32_t j = 0; j < kr::kTopK; ++j) {
      CHECK(row[kr::kIds + j] == want.ids[j]);
      worst = std::max(worst, int(std::abs(int64_t(row[kr::kWeights + j]) - int64_t(kr::as_u32(want.w[j])))));
      CHECK(kr::as_f32(row[kr::kSel + j]) == want.sel[j]);
    }
    CHECK(worst <= 2);
    CHECK(kr::as_f32(row[kr::kNext]) == want.next);
    // crafted: the tie at the cut (77 and 300 for the sixth place: 77 goes in)
    std::vector<float> lg(kk::kRouterN, -4.0f), bz(kk::kRouterN, 0.0f);
    for (uint32_t e : {5u, 50u, 100u, 150u, 200u}) lg[e] = 2.0f;
    lg[77] = 1.0f;
    lg[300] = 0.5f;
    bz[300] = 0.5f;
    for (uint32_t e = kk::kExperts; e < kk::kRouterN; ++e) lg[e] = 80.0f;   // the padded slots: never ranked
    std::vector<uint32_t> r2 = run_route(d, rm, lg, bz);
    const uint32_t tie[6] = {5, 50, 77, 100, 150, 200};
    for (uint32_t j = 0; j < 6; ++j) CHECK(r2[j] == tie[j]);
    // every real logit -1e30, zero bias: ids 0..5, weights 0; the padded slots at +80 never selected
    std::fill(lg.begin(), lg.begin() + kk::kExperts, -1e30f);
    std::fill(bz.begin(), bz.end(), 0.0f);
    r2 = run_route(d, rm, lg, bz);
    for (uint32_t j = 0; j < 6; ++j) CHECK(r2[j] == j && kr::as_f32(r2[kr::kWeights + j]) == 0.0f);
    std::printf("  route: ids exact (GEMV-fed, a tie at the cut, all -1e30, padded +80), weights within %d ulp\n", worst);
  }

  // ---- 4. the MoE block --------------------------------------------------------------------------
  {
    const size_t gub = kr::gate_up_block_words(), dnb = kr::down_block_words();
    const std::vector<uint32_t> gu = random_blocks(gub * kk::kExperts, 30), dn = random_blocks(dnb * kk::kExperts, 31);
    const std::vector<uint16_t> sh_gu_rows = random_bf16(size_t(2) * kk::kInter * H, -0.03f, 0.03f, 32);
    const std::vector<uint16_t> sh_dn_rows = random_bf16(size_t(H) * kk::kInter, -0.05f, 0.05f, 33);
    std::vector<uint16_t> sh_gu(sh_gu_rows.size()), sh_dn(sh_dn_rows.size());
    common::repack_bf16_tiled(sh_gu_rows.data(), H, 2 * kk::kInter, sh_gu.data());
    common::repack_bf16_tiled(sh_dn_rows.data(), kk::kInter, H, sh_dn.data());
    const std::vector<uint16_t> x = random_bf16(H, -0.5f, 0.5f, 34);
    const uint32_t ids[6] = {0, 17, 128, 255, 300, 383};
    const float ws[6] = {0.73f, 0.51f, 0.95f, 0.12f, 0.66f, 0.88f};
    std::vector<uint32_t> row(kr::kWords, 0);
    for (uint32_t j = 0; j < 6; ++j) {
      row[kr::kIds + j] = ids[j];
      row[kr::kWeights + j] = kr::as_u32(ws[j]);
    }
    std::vector<uint32_t> zero_gu(gu.size(), 0), zero_dn(dn.size(), 0);
    std::vector<uint16_t> zero_sgu(sh_gu.size(), 0), zero_sdn(sh_dn.size(), 0);
    l0::Module mm(d.ctx, kernels::path(kk::moe_variant(1)));
    const auto run_moe = [&](const std::vector<uint32_t>& g, const std::vector<uint32_t>& dw, const std::vector<uint16_t>& sg,
                             const std::vector<uint16_t>& sd, std::vector<uint16_t>& h_out) {
      l0::Mem rb = d.upload(row.data(), row.size() * 4), xb = d.upload(x.data(), H * 2);
      l0::Mem gb = d.upload(g.data(), g.size() * 4), db = d.upload(dw.data(), dw.size() * 4);
      l0::Mem sgb = d.upload(sg.data(), sg.size() * 2), sdb = d.upload(sd.data(), sd.size() * 2);
      l0::Mem hb = d.zeros(size_t(kk::kSlots) * kk::kInter * 2), ob = d.zeros(H * 2);
      l0::Kernel k1 = mm.kernel("kol_moe_gate_up");
      k1.group_size(kk::moe_gate_up_wg());
      k1.arg_ptr(0, rb.ptr());
      k1.arg_ptr(1, xb.ptr());
      k1.arg_ptr(2, gb.ptr());
      k1.arg_ptr(3, sgb.ptr());
      k1.arg_ptr(4, hb.ptr());
      d.run(k1, kk::moe_gate_up_groups(), 1);
      l0::Kernel k2 = mm.kernel("kol_moe_down");
      k2.group_size(kk::moe_down_wg());
      k2.arg_ptr(0, rb.ptr());
      k2.arg_ptr(1, hb.ptr());
      k2.arg_ptr(2, db.ptr());
      k2.arg_ptr(3, sdb.ptr());
      k2.arg_ptr(4, ob.ptr());
      d.run(k2, H / 16, 1);
      h_out = d.read<uint16_t>(hb, size_t(kk::kSlots) * kk::kInter);
      return d.read<uint16_t>(ob, H);
    };
    const auto check = [&](const char* what, const std::vector<uint32_t>& g, const std::vector<uint32_t>& dw,
                           const std::vector<uint16_t>& sg, const std::vector<uint16_t>& sd) {
      std::vector<uint16_t> h;
      const std::vector<uint16_t> mo = run_moe(g, dw, sg, sd, h);
      const std::vector<uint16_t> h_ref = kr::gate_up(ids, x.data(), g.data(), sg.data(), kk::kUpKs);
      int wh = 0, wo = 0;
      for (size_t i = 0; i < h.size(); ++i) wh = std::max(wh, ulps(h[i], h_ref[i]));
      // the down + combine from the DEVICE's h (the gate_up's exp apart)
      const std::vector<uint16_t> mo_ref = kr::down(ids, ws, h, dw.data(), sd.data(), kk::kDnKs);
      for (uint32_t n = 0; n < H; ++n) wo = std::max(wo, ulps(mo[n], mo_ref[n]));
      CHECK(wh <= 2 && wo <= 2);
      std::printf("  moe %-28s gate||up worst %d ulp, down + combine worst %d ulp\n", what, wh, wo);
      return mo;
    };
    const std::vector<uint16_t> mo1 = check("(6 routed + shared)", gu, dn, sh_gu, sh_dn);
    check("(the shared slot alone)", zero_gu, zero_dn, sh_gu, sh_dn);
    check("(the routed slots alone)", gu, dn, zero_sgu, zero_sdn);
    std::vector<uint16_t> h2;
    CHECK(run_moe(gu, dn, sh_gu, sh_dn, h2) == mo1);   // replay: bitwise
  }

  // ---- 5 / 6. attention ----------------------------------------------------------------------------
  {
    const uint32_t ML = 30001;
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    runtime::Control* c = ctrl.as<runtime::Control>();
    std::memset(c, 0, sizeof(*c));
    c->n_active = 1;
    // caches: the ring (4096 rows) and a full layer's (ML rows), random bf16
    const std::vector<uint16_t> rk = random_bf16(size_t(kk::kRing) * KVN, -1.f, 1.f, 40);
    const std::vector<uint16_t> rv = random_bf16(size_t(kk::kRing) * KVN, -1.f, 1.f, 41);
    const std::vector<uint16_t> fk = random_bf16(size_t(ML) * KVN, -1.f, 1.f, 42);
    const std::vector<uint16_t> fv = random_bf16(size_t(ML) * KVN, -1.f, 1.f, 43);
    std::vector<float> q(QN);
    {
      const std::vector<uint16_t> qb = random_bf16(QN, -4.f, 4.f, 44);
      for (uint32_t i = 0; i < QN; ++i) q[i] = f32(qb[i]);
    }
    l0::Mem rkb = d.upload(rk.data(), rk.size() * 2), rvb = d.upload(rv.data(), rv.size() * 2);
    l0::Mem fkb = d.upload(fk.data(), fk.size() * 2), fvb = d.upload(fv.data(), fv.size() * 2);
    l0::Mem qb = d.upload(q.data(), q.size() * 4);
    l0::Mem sb = d.zeros(size_t(QH) * ML * 4), part = d.zeros(size_t(QH) * kk::kAttnTgt * kk::kAttnPart * 4);
    l0::Mem ob = d.zeros(size_t(QN) * 2);
    // eager, bitwise
    for (int sliding = 1; sliding >= 0; --sliding) {
      l0::Module em(d.ctx, kernels::path(kk::attn_eager_variant(1, sliding)));
      for (uint32_t pos : sliding ? std::vector<uint32_t>{5, 512, 513, 4103, 9000} : std::vector<uint32_t>{30000}) {
        c->pos = pos;
        void* kk_ = sliding ? rkb.ptr() : fkb.ptr();
        void* vv_ = sliding ? rvb.ptr() : fvb.ptr();
        const uint32_t stride = ML;
        l0::Kernel s = em.kernel("kol_attn_eager_score");
        s.group_size(kk::kAttnWg);
        s.arg_ptr(0, ctrl.ptr());
        s.arg_ptr(1, qb.ptr());
        s.arg_ptr(2, kk_);
        s.arg_ptr(3, sb.ptr());
        s.arg<uint32_t>(4, stride);
        d.run(s, KVH, kk::kAttnTgt);
        l0::Kernel sm = em.kernel("kol_attn_eager_softmax");
        sm.group_size(kk::kAttnWg);
        sm.arg_ptr(0, ctrl.ptr());
        sm.arg_ptr(1, sb.ptr());
        sm.arg<uint32_t>(2, stride);
        d.run(sm, QH, 1);
        l0::Kernel pv = em.kernel("kol_attn_eager_pv");
        pv.group_size(kk::kAttnWg);
        pv.arg_ptr(0, ctrl.ptr());
        pv.arg_ptr(1, sb.ptr());
        pv.arg_ptr(2, vv_);
        pv.arg_ptr(3, part.ptr());
        pv.arg<uint32_t>(4, stride);
        d.run(pv, KVH, kk::kAttnTgt);
        l0::Kernel rd = em.kernel("kol_attn_eager_reduce");
        rd.group_size(kk::kAttnWg);
        rd.arg_ptr(0, ctrl.ptr());
        rd.arg_ptr(1, part.ptr());
        rd.arg_ptr(2, ob.ptr());
        d.run(rd, QH, 1);
        const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
        const uint32_t lo = kr::key_lo(pos, sliding);
        const uint32_t blk = kr::eager_block(pos + 1 - lo);
        size_t bad = 0;
        for (uint32_t h = 0; h < QH; ++h) {
          const kr::EagerHead e = kr::attention_eager(q.data() + size_t(h) * HD, sliding ? rk.data() : fk.data(),
                                                      sliding ? rv.data() : fv.data(), lo, pos, sliding, h / (QH / KVH), blk);
          for (uint32_t dd = 0; dd < HD; ++dd) bad += o[size_t(h) * HD + dd] != rne(e.o[dd]);
        }
        std::printf("  eager %s at %u: %zu of %u outputs differ from attention_eager\n", sliding ? "sliding" : "full", pos,
                    bad, QN);
        CHECK(bad == 0);
      }
    }
    // flash, against fp64
    std::vector<uint16_t> o9000;
    for (int sliding = 1; sliding >= 0; --sliding) {
      l0::Module fm(d.ctx, kernels::path(kk::attn_variant(1, sliding)));
      for (uint32_t pos : sliding ? std::vector<uint32_t>{0, 512, 4102, 9000, 808} : std::vector<uint32_t>{0, 29999}) {
        c->pos = pos;
        l0::Kernel dk = fm.kernel("kol_attn_decode");
        dk.group_size(kk::kAttnWg);
        dk.arg_ptr(0, ctrl.ptr());
        dk.arg_ptr(1, qb.ptr());
        dk.arg_ptr(2, sliding ? rkb.ptr() : fkb.ptr());
        dk.arg_ptr(3, sliding ? rvb.ptr() : fvb.ptr());
        dk.arg_ptr(4, part.ptr());
        d.run(dk, KVH, kk::kAttnTgt);
        l0::Kernel rk2 = fm.kernel("kol_attn_reduce");
        rk2.group_size(kk::kAttnWg);
        rk2.arg_ptr(0, ctrl.ptr());
        rk2.arg_ptr(1, part.ptr());
        rk2.arg_ptr(2, ob.ptr());
        d.run(rk2, QH, 1);
        const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
        if (sliding && pos == 9000) o9000 = o;
        if (sliding && pos == 808) CHECK(o == o9000);   // the same ring rows, the same blocks
        double worst = 1.0;
        for (uint32_t h = 0; h < QH; ++h) {
          const std::vector<double> want = kr::attention(q.data() + size_t(h) * HD, sliding ? rk.data() : fk.data(),
                                                         sliding ? rv.data() : fv.data(), kr::key_lo(pos, sliding), pos,
                                                         sliding, h / (QH / KVH));
          double dot = 0, na = 0, nb = 0;
          for (uint32_t dd = 0; dd < HD; ++dd) {
            const double g = f32(o[size_t(h) * HD + dd]);
            dot += g * want[dd];
            na += g * g;
            nb += want[dd] * want[dd];
          }
          worst = std::min(worst, dot / std::sqrt(na * nb));
        }
        std::printf("  flash %s at %u (%u keys): worst cosine %.7f\n", sliding ? "sliding" : "full", pos,
                    pos + 1 - kr::key_lo(pos, sliding), worst);
        CHECK(worst >= 0.99999);
      }
    }
  }
  std::puts("kolibri_kernels_test OK");
  return 0;
}
