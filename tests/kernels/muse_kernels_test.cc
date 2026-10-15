// Spec 23c (MG2 on the card): Muse-Glimmer-30B's decode kernels at the real shapes against tests/kernels/muse_ref.h
// (the host twin, bitwise to transformers 5.19.0 - muse_ref_test). Needs a B70, no checkpoint (synthetic inputs).
// Every binary here is one the decode list binds (kernels::muse names).
//
//   1. mg_embed (the real 202048-row table: rows 7 and 202047, the image / video placeholders as row 0) - resid and
//      its 26 chunk sums bitwise; mg_norm (centered) and mg_final_norm (plain w) bitwise; the sandwich:
//      prep_res_fold SP4 _Z at G 26 over a sub-block's slices + mg_post_add - the row, resid and both Σ² bitwise
//      (no exp anywhere; sqrt / divide correctly rounded)
//   2. mg_attn_prep S2, sliding at 37 and 4097 (ring row 1) and full at 0 and 100000 - q, gate, k, v bitwise; a full
//      layer's q and k at 0 equal those at 100000 (NoPE, Review Focus 5)
//   3. mg_silu_mul S8 over the concatenated gate||up - bitwise (exp_torch on both sides)
//   4. eager attention (mg_attn_eager.cl, both builds) bitwise against attention_eager with the gate: sliding at 5,
//      2046, 2047, 2048, 4095, 4096, 4097, 9000 (Review Focus 3's rows); full at 3000 and 30000
//   5. flash (mg_attn.cl) against the fp64 form within spec 10's bar (cosine 0.99999): the same rows, full at 20000;
//      sliding at 9000 bitwise sliding at 4904 (9000 - 4096: the same ring rows - the ring is only addressing)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "kernels/muse_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/muse/muse_rope.h"

namespace {

namespace km = kernels::muse;
namespace mr = museref;
using mr::bf;
using mr::rf;
using mr::rne;

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

const uint32_t H = km::kHidden, HD = km::kHd, QH = km::kQHeads, KVH = km::kKvHeads, QN = km::kQN, KVN = km::kKvN,
               G = km::kNormG;

double worst_cosine(const std::vector<uint16_t>& o, const std::vector<double>& f) {
  double worst = 1.0;
  for (uint32_t h = 0; h < QH; ++h) {
    double dot = 0, na = 0, nb = 0;
    for (uint32_t d = 0; d < HD; ++d) {
      const double a = bf(o[size_t(h) * HD + d]), b = f[size_t(h) * HD + d];
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    worst = std::min(worst, dot / std::sqrt(na * nb));
  }
  return worst;
}

}  // namespace

int main() {
  Dev d;
  static_assert(km::kNormG == mr::kG && km::kAttnTgt == mr::kAttnTgt && km::kRing == mr::kRing, "twin / names");
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  runtime::Control* c = ctrl.as<runtime::Control>();
  std::memset(c, 0, sizeof(*c));
  c->n_active = 1;

  // ---- 1. the embedding, the norms and the sandwich -----------------------------------------------
  {
    l0::Mem table = d.zeros(size_t(km::kVocab) * H * 2);
    const std::vector<uint16_t> r0 = random_bf16(H, -3.f, 3.f, 1), r7 = random_bf16(H, -0.02f, 0.02f, 2),
                                rl = random_bf16(H, -1.f, 1.f, 3);
    d.imm.copy(table.ptr(), r0.data(), H * 2);
    d.imm.copy(static_cast<uint8_t*>(table.ptr()) + size_t(7) * H * 2, r7.data(), H * 2);
    d.imm.copy(static_cast<uint8_t*>(table.ptr()) + size_t(km::kVocab - 1) * H * 2, rl.data(), H * 2);
    l0::Module em(d.ctx, kernels::path(km::embed_variant(1)));
    l0::Mem rb = d.zeros(H * 2), sb = d.zeros(G * 4);
    for (uint32_t id : {7u, km::kVocab - 1, km::kImageToken, km::kVideoToken}) {
      c->cur_token[0] = id;
      l0::Kernel k = em.kernel("mg_embed");
      k.group_size(km::kRowWg);
      k.arg_ptr(0, ctrl.ptr());
      k.arg_ptr(1, table.ptr());
      k.arg_ptr(2, rb.ptr());
      k.arg_ptr(3, sb.ptr());
      d.run(k, 1, 1);
      const std::vector<uint16_t>& src = id == 7 ? r7 : (id == km::kVocab - 1 ? rl : r0);
      std::vector<uint16_t> want(H);
      std::vector<float> ss(G);
      mr::embed_norm_sums(src.data(), H, want.data(), ss.data());
      CHECK(d.read<uint16_t>(rb, H) == want);
      CHECK(d.read<float>(sb, G) == ss);
    }
    std::printf("  mg_embed: rows 7, 202047 and the placeholders (row 0) bitwise, their 26 chunk sums too\n");

    const std::vector<uint16_t> resid = random_bf16(H, -2.f, 2.f, 4);
    std::vector<float> w(H), w2(H);
    for (uint32_t k = 0; k < H; ++k) {
      w[k] = rf(-0.4f + float(k % 11) * 0.08f);
      w2[k] = rf(0.5f + float(k % 7) * 0.2f);
    }
    std::vector<float> ss(G);
    mr::chunk_sumsq(resid.data(), H, G, ss.data());
    l0::Mem rr = d.upload(resid.data(), H * 2), ssb = d.upload(ss.data(), G * 4), wb = d.upload(w.data(), H * 4),
            xb = d.zeros(H * 2), fb = d.zeros(H * 2);
    l0::Module nm(d.ctx, kernels::path(km::norm_variant(1)));
    for (const char* entry : {"mg_norm", "mg_final_norm"}) {
      l0::Kernel k = nm.kernel(entry);
      k.group_size(km::kRowWg);
      k.arg_ptr(0, ssb.ptr());
      k.arg_ptr(1, rr.ptr());
      k.arg_ptr(2, wb.ptr());
      k.arg_ptr(3, std::string(entry) == "mg_norm" ? xb.ptr() : fb.ptr());
      d.run(k, km::kNormW, 1);
    }
    std::vector<uint16_t> x_ref(H), f_ref(H);
    mr::centered_norm_from(ss.data(), G, resid.data(), w.data(), mr::kEps, H, x_ref.data());
    mr::final_norm_from(ss.data(), G, resid.data(), w.data(), H, f_ref.data());
    CHECK(d.read<uint16_t>(xb, H) == x_ref);
    CHECK(d.read<uint16_t>(fb, H) == f_ref);
    // the sandwich: a sub-block's 4 slices -> its row (ZERO_RESID) and Σ², then the post norm + add
    const std::vector<float> part = random_f32(size_t(4) * H, -0.5f, 0.5f, 5);
    std::vector<uint16_t> a_ref(H), r2_ref = resid;
    for (uint32_t k = 0; k < H; ++k) a_ref[k] = mr::partials_col(part.data(), 4, 1, 0, H, k);
    std::vector<float> sa_ref(G), so_ref(G);
    mr::chunk_sumsq(a_ref.data(), H, G, sa_ref.data());
    mr::post_add_sums(sa_ref.data(), G, a_ref.data(), w2.data(), r2_ref.data(), so_ref.data(), H);
    l0::Mem pb = d.upload(part.data(), part.size() * 4), ab = d.zeros(H * 2), sa = d.zeros(G * 4), so = d.zeros(G * 4);
    l0::Mem w2b = d.upload(w2.data(), H * 4);
    l0::Module zm(d.ctx, kernels::path(km::fold_zero_variant(1, km::kOprojS)));
    l0::Kernel kz = zm.kernel("prep_res_fold");
    kz.group_size(256);
    kz.arg_ptr(0, pb.ptr());
    kz.arg_ptr(1, ab.ptr());
    kz.arg_ptr(2, sa.ptr());
    d.run(kz, G, 1);
    l0::Module am(d.ctx, kernels::path(km::post_add_variant(1)));
    l0::Kernel ka = am.kernel("mg_post_add");
    ka.group_size(km::kRowWg);
    ka.arg_ptr(0, sa.ptr());
    ka.arg_ptr(1, ab.ptr());
    ka.arg_ptr(2, w2b.ptr());
    ka.arg_ptr(3, rr.ptr());
    ka.arg_ptr(4, so.ptr());
    d.run(ka, G, 1);
    CHECK(d.read<uint16_t>(ab, H) == a_ref);
    CHECK(d.read<float>(sa, G) == sa_ref);
    CHECK(d.read<uint16_t>(rr, H) == r2_ref);
    CHECK(d.read<float>(so, G) == so_ref);
    std::printf("  mg_norm, mg_final_norm, prep_res_fold _Z + mg_post_add: bitwise\n");
  }

  // ---- 2. the attention prep ------------------------------------------------------------------------
  {
    const std::vector<float> rope = runtime::muse::rope_table(HD, 5e5f, 100001);
    l0::Mem rb = d.upload(rope.data(), rope.size() * 4);
    const std::vector<float> part = random_f32(size_t(2) * km::kQgkvN, -1.f, 1.f, 10);
    l0::Mem pb = d.upload(part.data(), part.size() * 4);
    std::vector<uint16_t> q(QN), g(QN), k(KVN), v(KVN);
    for (uint32_t i = 0; i < QN; ++i) {
      q[i] = mr::partials_col(part.data(), 2, 1, 0, km::kQgkvN, i);
      g[i] = mr::partials_col(part.data(), 2, 1, 0, km::kQgkvN, mr::kOffGate + i);
    }
    for (uint32_t i = 0; i < KVN; ++i) {
      k[i] = mr::partials_col(part.data(), 2, 1, 0, km::kQgkvN, mr::kOffK + i);
      v[i] = mr::partials_col(part.data(), 2, 1, 0, km::kQgkvN, mr::kOffV + i);
    }
    std::vector<float> q_full0;
    std::vector<uint16_t> k_full0;
    for (int sliding = 1; sliding >= 0; --sliding) {
      const uint32_t rows = sliding ? km::kRing : 100001;
      l0::Module m(d.ctx, kernels::path(km::attn_prep_variant(1, km::kQgkvS, sliding)));
      l0::Mem aq = d.zeros(QN * 4), gb = d.zeros(QN * 2), kb = d.zeros(size_t(rows) * KVN * 2),
              vb = d.zeros(size_t(rows) * KVN * 2);
      for (uint32_t pos : sliding ? std::vector<uint32_t>{37, 4097} : std::vector<uint32_t>{0, 100000}) {
        c->pos = pos;
        l0::Kernel kk = m.kernel("mg_attn_prep");
        kk.group_size(km::kPrepWg);
        kk.arg_ptr(0, ctrl.ptr());
        kk.arg_ptr(1, pb.ptr());
        kk.arg_ptr(2, rb.ptr());
        kk.arg_ptr(3, aq.ptr());
        kk.arg_ptr(4, gb.ptr());
        kk.arg_ptr(5, kb.ptr());
        kk.arg_ptr(6, vb.ptr());
        d.run(kk, km::kPrepGroups, 1);
        std::vector<uint16_t> qo(QN), ko(KVN);
        mr::qk_prep(q.data(), k.data(), pos, sliding != 0, rope.data(), qo.data(), ko.data());
        const size_t row = mr::ring_row(pos, sliding != 0);
        const std::vector<float> aqv = d.read<float>(aq, QN);
        for (uint32_t i = 0; i < QN; ++i) CHECK(aqv[i] == bf(qo[i]));
        CHECK(d.read<uint16_t>(kb, KVN, row * KVN * 2) == ko);
        CHECK(d.read<uint16_t>(vb, KVN, row * KVN * 2) == v);
        const std::vector<uint16_t> gd = d.read<uint16_t>(gb, QN);
        for (uint32_t i = 0; i < QN; ++i) CHECK(gd[i] == mr::gate_bf16(g[i]));
        if (!sliding && pos == 0) {
          q_full0 = aqv;
          k_full0 = ko;
        }
        if (!sliding && pos == 100000) CHECK(aqv == q_full0 && ko == k_full0);   // NoPE: position-free
        std::printf("  mg_attn_prep %s at %u: q, gate, k (row %zu), v bitwise\n", sliding ? "sliding" : "full", pos, row);
      }
    }
  }

  // ---- 3. SiLU x up ---------------------------------------------------------------------------------
  {
    const uint32_t I = km::kInter;
    const std::vector<float> part = random_f32(size_t(km::kGateUpS) * 2 * I, -1.f, 1.f, 20);
    l0::Mem pb = d.upload(part.data(), part.size() * 4), hb = d.zeros(size_t(I) * 2);
    l0::Module m(d.ctx, kernels::path(km::silu_variant(1, km::kGateUpS)));
    l0::Kernel k = m.kernel("mg_silu_mul");
    k.group_size(km::kRowWg);
    k.arg_ptr(0, pb.ptr());
    k.arg_ptr(1, hb.ptr());
    d.run(k, km::kSiluGroups, 1);
    std::vector<uint16_t> g(I), u(I), want(I);
    for (uint32_t i = 0; i < I; ++i) {
      g[i] = mr::partials_col(part.data(), km::kGateUpS, 1, 0, 2 * I, i);
      u[i] = mr::partials_col(part.data(), km::kGateUpS, 1, 0, 2 * I, I + i);
    }
    mr::silu_mul(g.data(), u.data(), I, want.data());
    CHECK(d.read<uint16_t>(hb, I) == want);
    std::printf("  mg_silu_mul: bitwise\n");
  }

  // ---- 4 / 5. attention -----------------------------------------------------------------------------
  {
    const uint32_t ML = 30001;
    std::mt19937 rng(40);
    std::uniform_real_distribution<float> u(-1.f, 1.f), uq(-4.f, 4.f);
    std::vector<uint16_t> rk(size_t(km::kRing) * KVN), rv(rk.size()), fk(size_t(ML) * KVN), fv(fk.size()), q(QN), g(QN);
    for (auto* v : {&rk, &rv, &fk, &fv})
      for (uint16_t& e : *v) e = rne(u(rng));
    for (uint16_t& e : q) e = rne(uq(rng));
    for (uint16_t& e : g) e = mr::gate_bf16(rne(uq(rng)));
    std::vector<float> qf(QN);
    for (uint32_t i = 0; i < QN; ++i) qf[i] = bf(q[i]);
    l0::Mem rkb = d.upload(rk.data(), rk.size() * 2), rvb = d.upload(rv.data(), rv.size() * 2);
    l0::Mem fkb = d.upload(fk.data(), fk.size() * 2), fvb = d.upload(fv.data(), fv.size() * 2);
    l0::Mem qb = d.upload(qf.data(), qf.size() * 4), gb = d.upload(g.data(), g.size() * 2);
    l0::Mem sb = d.zeros(size_t(QH) * ML * 4), part = d.zeros(size_t(QH) * km::kAttnTgt * km::kAttnPart * 4);
    l0::Mem ob = d.zeros(size_t(QN) * 2);
    const std::vector<uint32_t> srows = {5, 2046, 2047, 2048, 4095, 4096, 4097, 4904, 9000};
    for (int sliding = 1; sliding >= 0; --sliding) {
      l0::Module em(d.ctx, kernels::path(km::attn_eager_variant(1, sliding)));
      for (uint32_t pos : sliding ? srows : std::vector<uint32_t>{3000, 30000}) {
        c->pos = pos;
        l0::Kernel k = em.kernel("mg_attn_eager");
        k.group_size(km::kAttnWg);
        k.arg_ptr(0, ctrl.ptr());
        k.arg_ptr(1, qb.ptr());
        k.arg_ptr(2, sliding ? rkb.ptr() : fkb.ptr());
        k.arg_ptr(3, sliding ? rvb.ptr() : fvb.ptr());
        k.arg_ptr(4, gb.ptr());
        k.arg_ptr(5, sb.ptr());
        k.arg_ptr(6, ob.ptr());
        k.arg<uint32_t>(7, ML);
        d.run(k, QH, 1);
        std::vector<uint16_t> want(QN);
        mr::attention_eager(q.data(), sliding ? rk.data() : fk.data(), sliding ? rv.data() : fv.data(), pos, sliding != 0,
                            km::kRing, g.data(), want.data());
        const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
        size_t bad = 0;
        for (uint32_t i = 0; i < QN; ++i) bad += o[i] != want[i];
        std::printf("  eager %s at %u: %zu of %u outputs differ from attention_eager\n", sliding ? "sliding" : "full", pos,
                    bad, QN);
        CHECK(bad == 0);
      }
    }
    std::vector<uint16_t> o4904;
    for (int sliding = 1; sliding >= 0; --sliding) {
      l0::Module fm(d.ctx, kernels::path(km::attn_variant(1, sliding)));
      for (uint32_t pos : sliding ? srows : std::vector<uint32_t>{0, 3000, 20000, 30000}) {
        c->pos = pos;
        l0::Kernel dk = fm.kernel("mg_attn_decode");
        dk.group_size(km::kAttnWg);
        dk.arg_ptr(0, ctrl.ptr());
        dk.arg_ptr(1, qb.ptr());
        dk.arg_ptr(2, sliding ? rkb.ptr() : fkb.ptr());
        dk.arg_ptr(3, sliding ? rvb.ptr() : fvb.ptr());
        dk.arg_ptr(4, part.ptr());
        d.run(dk, KVH, km::kAttnTgt);
        l0::Kernel rk2 = fm.kernel("mg_attn_reduce");
        rk2.group_size(km::kAttnWg);
        rk2.arg_ptr(0, ctrl.ptr());
        rk2.arg_ptr(1, part.ptr());
        rk2.arg_ptr(2, gb.ptr());
        rk2.arg_ptr(3, ob.ptr());
        d.run(rk2, QH, 1);
        const std::vector<uint16_t> o = d.read<uint16_t>(ob, QN);
        if (sliding && pos == 4904) o4904 = o;
        if (sliding && pos == 9000) CHECK(o == o4904);   // the same ring rows, the same blocks
        std::vector<double> f(QN);
        mr::attention_fp64(q.data(), sliding ? rk.data() : fk.data(), sliding ? rv.data() : fv.data(), pos, sliding != 0,
                           km::kRing, g.data(), f.data());
        const double worst = worst_cosine(o, f);
        std::printf("  flash %s at %u (%u keys): worst cosine %.7f\n", sliding ? "sliding" : "full", pos,
                    pos + 1 - mr::key_lo(pos, sliding != 0), worst);
        CHECK(worst >= 0.99999);
      }
    }
  }
  std::puts("muse_kernels_test OK");
  return 0;
}
