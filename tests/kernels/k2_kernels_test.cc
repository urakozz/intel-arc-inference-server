// Spec 18b Task 2 (K1): K2-Horizon's kernels on the card, at K2's real shapes, against
// tests/kernels/k2_ref.h. Needs a B70, no checkpoint (synthetic weights). Every binary
// here is the one the decode list binds (kernels::k2 names).
//
//   1. the grouped norm: prep_res_fold (SP4) + k2_norm_finish - resid and x bit-exact
//      (no exp anywhere; sqrt / divide are correctly rounded);
//   2. k2_silu_mul at I 6144 / S 4 - within 1 bf16 ulp (exp);
//   3. k2_attn_prep, both builds (dense with v, MoVA without) at pos 37 - q, gate, k, v
//      bit-exact (the RoPE chain's products are exact in fp32);
//   4. the routers: the MoE router GEMV (gemv_bf16 2560 -> 128) + k2_route; k2_route on
//      crafted logits - exact ties to the lower id, and the 28 PADDED lanes at +80 against
//      real scores far below (never selected: Review Focus 4); MoVA's k2_route over two
//      split-K slices of the fused row from column 9216;
//   5. k2_moe_gate_up / k2_moe_down with a forced route row (ascending ids incl. 0 and 99,
//      the shared expert block 100) - every slot against ITS expert, the ascending-id bf16
//      combine and the residual within 2 bf16 ulps;
//   6. k2_mova_value - the 4 value experts by id, SiLU, the combine, into the V cache at pos;
//   7. k2_attn_decode / k2_attn_reduce at depths 6, 300 and 3000 (1, 5 and 24 blocks) against
//      an fp64 softmax, with gates on both sides of softplus's threshold (Review Focus 2);
//   8. bitwise replay of the MoE pair and the attention pair;
//   9. the EAGER attention (B70_K2_ATTN=eager, k2_attn_eager.cl's four kernels) at depths 6,
//      300, 3000 and 4096 against k2_ref.h attention_eager, BITWISE at every stage: the score
//      row, the probabilities (exp_torch is the same lines on both sides; 1 / sum is correctly
//      rounded), the output through gates above softplus's threshold (no exp on that path);
//      below it within 1 ulp (softplus's OpenCL exp / log1p).
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
#include "kernels/k2_kernels.h"
#include "kernels/k2_ref.h"
#include "kernels/kernels.h"
#include "kernels/prep_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"

namespace {

namespace kk = kernels::k2;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;

static_assert(kk::route::kWords == k2_ref::kWords && kk::route::kIds == k2_ref::kIds &&
                  kk::route::kWeights == k2_ref::kWeights && kk::route::kSel == k2_ref::kSel &&
                  kk::route::kNext == k2_ref::kNext && kk::route::kSum == k2_ref::kSum,
              "the K2 route row: k2_moe.cl R_*, kernels::k2::route and k2_ref disagree");

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

// The route row the kernels read, from a reference Route.
std::vector<uint32_t> row_of(const k2_ref::Route& r) {
  std::vector<uint32_t> row(k2_ref::kWords, 0);
  for (uint32_t j = 0; j < 8; ++j) {
    row[k2_ref::kIds + j] = r.ids[j];
    std::memcpy(&row[k2_ref::kWeights + j], &r.w[j], 4);
  }
  return row;
}

// ids exactly the reference's unless the k-th / (k+1)-th selection scores are within the
// host / device `exp` slack; weights within 1 bf16 ulp of the same expert's.
void check_route(const std::vector<uint32_t>& row, const k2_ref::Route& want, uint32_t K,
                 const char* what) {
  float kth = -INFINITY;
  for (uint32_t j = 0; j < K; ++j) kth = std::max(kth, -want.sel[j]);
  kth = -kth;   // the smallest selected sel
  const bool near_tie = kth != want.next && kth - want.next <= 2e-6f * std::fabs(kth);
  if (near_tie) std::printf("  %s: near-tie at the top-%u boundary\n", what, K);
  uint32_t same = 0;
  for (uint32_t j = 0; j < K; ++j) {
    const uint32_t id = row[k2_ref::kIds + j];
    const uint32_t* at = std::find(want.ids, want.ids + K, id);
    if (at == want.ids + K) continue;
    ++same;
    float w;
    std::memcpy(&w, &row[k2_ref::kWeights + j], 4);
    CHECK(ulps(rne(w), rne(want.w[at - want.ids])) <= 1);
    if (j) CHECK(row[k2_ref::kIds + j] > row[k2_ref::kIds + j - 1]);   // ascending id
  }
  if (same != K && !(near_tie && same == K - 1)) {
    std::fprintf(stderr, "%s: %u of %u expert ids agree with the reference\n", what, same, K);
    std::exit(1);
  }
}

}  // namespace

int main() {
  Dev d;
  const model::K2Desc& m = model::k2();
  const uint32_t H = m.hidden;

  // ---- 1. the grouped norm ---------------------------------------------------------------
  {
    const uint32_t G = kk::kNormG, S = 4;
    std::vector<uint16_t> resid = random_bf16(H, -1.f, 1.f, 1);
    for (uint32_t k = 0; k < H / 2; ++k) resid[k] = rne(f32(resid[k]) * 0.01f);   // group 0 small
    const std::vector<float> part = random_f32(size_t(S) * H, -0.5f, 0.5f, 2);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
    std::vector<uint16_t> r_ref = resid, x_ref(H);
    std::vector<float> ss_ref(G);
    prep_ref::res_fold(part.data(), r_ref.data(), ss_ref.data(), 1, H, S, G);
    k2_ref::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), 1, H, G, 2);
    l0::Mem rb = d.upload(resid.data(), H * 2), pb = d.upload(part.data(), part.size() * 4);
    l0::Mem wb = d.upload(w.data(), H * 4), ss = d.zeros(G * 4), xb = d.zeros(H * 2);
    l0::Module fold(d.ctx, kernels::path(kernels::prep_res_fold_variant(1, H, S, G)));
    l0::Kernel kf = fold.kernel("prep_res_fold");
    kf.group_size(256);
    kf.arg_ptr(0, pb.ptr());
    kf.arg_ptr(1, rb.ptr());
    kf.arg_ptr(2, ss.ptr());
    d.run(kf, G, 1);
    l0::Module nm(d.ctx, kernels::path(kk::norm_variant(1, H, G, kk::kNormW, 2)));
    l0::Kernel kn = nm.kernel("k2_norm_finish");
    kn.group_size(kk::kNormWg);
    kn.arg_ptr(0, ss.ptr());
    kn.arg_ptr(1, rb.ptr());
    kn.arg_ptr(2, wb.ptr());
    kn.arg_ptr(3, xb.ptr());
    d.run(kn, kk::kNormW, 1);
    CHECK(d.read<uint16_t>(rb, H) == r_ref);
    CHECK(d.read<uint16_t>(xb, H) == x_ref);
    std::puts("  grouped norm: resid and x bit-exact (2 groups of 1280, plain w)");
  }

  // ---- 2. SiLU x up ----------------------------------------------------------------------
  {
    const uint32_t I = m.dense_inter, S = m.gate_up_s;
    const std::vector<float> part = random_f32(size_t(S) * 2 * I, -1.f, 1.f, 3);
    std::vector<uint16_t> x_ref(I);
    k2_ref::silu_mul(part.data(), x_ref.data(), 1, I, S);
    l0::Mem pb = d.upload(part.data(), part.size() * 4), xb = d.zeros(I * 2);
    l0::Module sm(d.ctx, kernels::path(kk::silu_variant(1, I, S)));
    l0::Kernel ks = sm.kernel("k2_silu_mul");
    ks.group_size(kk::kSiluWg);
    ks.arg_ptr(0, pb.ptr());
    ks.arg_ptr(1, xb.ptr());
    d.run(ks, (I + kk::kSiluChunk - 1) / kk::kSiluChunk, 1);
    const std::vector<uint16_t> x = d.read<uint16_t>(xb, I);
    int worst = 0;
    for (uint32_t k = 0; k < I; ++k) worst = std::max(worst, ulps(x[k], x_ref[k]));
    CHECK(worst <= 1);
    std::printf("  silu x up: worst %d bf16 ulp\n", worst);
  }

  // ---- 3. the attention prep, both builds ------------------------------------------------
  const uint32_t ML = 4096;   // the test's KV length
  const std::vector<float> rope = loader::k2_rope_table(m, ML);
  l0::Mem ropeb = d.upload(rope.data(), rope.size() * 4);
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  runtime::Control* c = ctrl.as<runtime::Control>();
  std::memset(c, 0, sizeof(*c));
  c->n_active = 1;
  l0::Mem kvk = d.zeros(size_t(ML) * m.kv_n() * 2), kvv = d.zeros(size_t(ML) * m.kv_n() * 2);
  l0::Mem aq = d.zeros(size_t(m.q_n()) * 4), ag = d.zeros(size_t(m.q_n()) * 4);
  for (int dense = 1; dense >= 0; --dense) {
    const uint32_t N = dense ? m.attn_dense_n() : m.attn_sparse_n(), S = m.attn_s;
    const std::vector<float> part = random_f32(size_t(S) * N, -1.f, 1.f, 10 + dense);
    c->pos = 37;
    const k2_ref::Prep want =
        k2_ref::attn_prep(part.data(), 1, 0, N, S, rope.data() + size_t(37) * 2 * 64, m.q_heads,
                          m.kv_heads, m.head_dim, dense);
    l0::Mem pb = d.upload(part.data(), part.size() * 4);
    l0::Module pm(d.ctx, kernels::path(kk::attn_prep_variant(1, N, S, m.q_heads, m.kv_heads, dense)));
    l0::Kernel kp = pm.kernel("k2_attn_prep");
    kp.group_size(m.head_dim);
    kp.arg_ptr(0, ctrl.ptr());
    kp.arg_ptr(1, pb.ptr());
    kp.arg_ptr(2, ropeb.ptr());
    kp.arg_ptr(3, aq.ptr());
    kp.arg_ptr(4, ag.ptr());
    kp.arg_ptr(5, kvk.ptr());
    kp.arg_ptr(6, kvv.ptr());
    d.imm.fill(kvv.ptr(), 0u, kvv.size());
    d.run(kp, m.q_heads + m.kv_heads, 1);
    CHECK(d.read<float>(aq, m.q_n()) == want.q);
    CHECK(d.read<float>(ag, m.q_n()) == want.gate);
    CHECK(d.read<uint16_t>(kvk, m.kv_n(), size_t(37) * m.kv_n() * 2) == want.k);
    const std::vector<uint16_t> v = d.read<uint16_t>(kvv, m.kv_n(), size_t(37) * m.kv_n() * 2);
    if (dense) CHECK(v == want.v);
    else CHECK(std::all_of(v.begin(), v.end(), [](uint16_t e) { return e == 0; }));
  }
  std::puts("  attn prep: q, gate, k (and the dense build's v) bit-exact at pos 37; the MoVA "
            "build leaves v to k2_mova_value");

  // ---- 4. the routers ----------------------------------------------------------------------
  const std::vector<float> bias_moe = random_f32(m.router_n(), -0.05f, 0.05f, 20);
  const std::vector<float> bias_mova = random_f32(m.value_experts, -0.05f, 0.05f, 21);
  l0::Mem bmoe = d.upload(bias_moe.data(), bias_moe.size() * 4);
  l0::Mem bmova = d.upload(bias_mova.data(), bias_mova.size() * 4);
  l0::Mem routeb = d.zeros(k2_ref::kWords * 4);
  const std::string rv_moe = kk::route_variant(1, m.experts, m.top_k, m.router_n(), 0, 1);
  l0::Module rmod(d.ctx, kernels::path(rv_moe));
  l0::Kernel kr = rmod.kernel("k2_route");
  kr.group_size(model::K2Desc::route_wg(m.experts));
  {
    // The router GEMV over 100 real rows + 28 zero rows, then the route of its output.
    const std::vector<uint16_t> x = random_bf16(H, -1.f, 1.f, 22);
    std::vector<uint16_t> rows = random_bf16(size_t(m.router_n()) * H, -0.05f, 0.05f, 23);
    std::fill(rows.begin() + size_t(m.experts) * H, rows.end(), uint16_t(0));
    std::vector<uint16_t> tiled(rows.size());
    common::repack_bf16_tiled(rows.data(), H, m.router_n(), tiled.data());
    l0::Mem wr = d.upload(tiled.data(), tiled.size() * 2), xb = d.upload(x.data(), H * 2);
    l0::Mem lg = d.zeros(m.router_n() * 4);
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(m.router_n());
    l0::Module gm(d.ctx, kernels::path(kernels::gemv_bf16_variant(1, H, m.router_n(), t)));
    l0::Kernel kg = gm.kernel("gemv_bf16");
    kg.group_size(t.cols * t.ksplit);
    kg.arg_ptr(0, wr.ptr());
    kg.arg_ptr(1, xb.ptr());
    kg.arg_ptr(2, lg.ptr());
    d.run(kg, m.router_n() / t.cols);
    const std::vector<float> lgh = d.read<float>(lg, m.router_n());
    for (uint32_t n = m.experts; n < m.router_n(); ++n) CHECK(lgh[n] == 0.0f);
    kr.arg_ptr(0, lg.ptr());
    kr.arg_ptr(1, bmoe.ptr());
    kr.arg_ptr(2, routeb.ptr());
    d.run(kr, 1, 1);
    check_route(d.read<uint32_t>(routeb, k2_ref::kWords),
                k2_ref::route(lgh.data(), 1, 0, m.router_n(), 0, 1, bias_moe.data(), m.experts,
                              m.top_k, m.router_scale),
                m.top_k, "MoE route of the device logits");
  }
  {
    // Exact ties go to the lower id; the padded lanes at +80 are never taken.
    std::vector<float> tie(m.router_n(), -30.0f);
    for (uint32_t e = m.experts; e < m.router_n(); ++e) tie[e] = 80.0f;
    for (uint32_t e : {90u, 5u, 61u, 33u, 99u, 12u, 0u, 47u, 70u}) tie[e] = 1.0f;   // 9 tied
    std::vector<float> zb(m.router_n(), 0.0f);
    l0::Mem tl = d.upload(tie.data(), tie.size() * 4), zbb = d.upload(zb.data(), zb.size() * 4);
    kr.arg_ptr(0, tl.ptr());
    kr.arg_ptr(1, zbb.ptr());
    d.run(kr, 1, 1);
    const std::vector<uint32_t> row = d.read<uint32_t>(routeb, k2_ref::kWords);
    const uint32_t want[8] = {0, 5, 12, 33, 47, 61, 70, 90};   // the 8 lowest of the 9, ascending
    for (uint32_t j = 0; j < 8; ++j) CHECK_EQ(row[k2_ref::kIds + j], want[j]);
    float w0, nx, s7;
    std::memcpy(&w0, &row[k2_ref::kWeights], 4);
    std::memcpy(&nx, &row[k2_ref::kNext], 4);
    std::memcpy(&s7, &row[k2_ref::kSel + 7], 4);
    CHECK(w0 == rf(2.5f / 8.0f));
    CHECK(nx == s7);   // expert 99: tied with the taken ones, the first not taken
    // All real scores far below: still only real experts.
    for (uint32_t e = 0; e < m.experts; ++e) tie[e] = -30.0f - float(e % 5);
    std::vector<float> nb(m.router_n(), -0.9f);
    d.imm.copy(tl.ptr(), tie.data(), tie.size() * 4);
    d.imm.copy(zbb.ptr(), nb.data(), nb.size() * 4);
    d.run(kr, 1, 1);
    const std::vector<uint32_t> row2 = d.read<uint32_t>(routeb, k2_ref::kWords);
    for (uint32_t j = 0; j < 8; ++j) CHECK(row2[k2_ref::kIds + j] < m.experts);
    std::puts("  MoE route: device logits as the reference; ties to the lower id, ascending; the "
              "28 padded lanes never selected");
  }
  {
    // MoVA's router: two slices of the fused row, columns 9216..9279.
    const uint32_t N = m.attn_sparse_n(), S = m.attn_s;
    const std::vector<float> part = random_f32(size_t(S) * N, -2.f, 2.f, 24);
    l0::Mem pb = d.upload(part.data(), part.size() * 4);
    l0::Module vm(d.ctx, kernels::path(kk::route_variant(1, m.value_experts, m.value_top_k, N, m.v_off(), S)));
    l0::Kernel kv = vm.kernel("k2_route");
    kv.group_size(model::K2Desc::route_wg(m.value_experts));
    kv.arg_ptr(0, pb.ptr());
    kv.arg_ptr(1, bmova.ptr());
    kv.arg_ptr(2, routeb.ptr());
    d.run(kv, 1, 1);
    check_route(d.read<uint32_t>(routeb, k2_ref::kWords),
                k2_ref::route(part.data(), 1, 0, N, m.v_off(), S, bias_mova.data(), m.value_experts,
                              m.value_top_k, m.router_scale),
                m.value_top_k, "MoVA route of the fused row");
    std::puts("  MoVA route: the v_router columns' two slices, top-4 ascending");
  }

  // ---- 5. the MoE block --------------------------------------------------------------------
  const moe_ref::Shape ms = k2_ref::moe_shape(m.experts, m.top_k, H, m.moe_inter);
  const std::vector<uint32_t> gu = random_blocks(ms.gate_up_words() * m.moe_blocks(), 30);
  const std::vector<uint32_t> dn = random_blocks(ms.down_words() * m.moe_blocks(), 31);
  l0::Mem gub = d.upload(gu.data(), gu.size() * 4), dnb = d.upload(dn.data(), dn.size() * 4);
  k2_ref::Route forced;
  const uint32_t ids[8] = {0, 3, 17, 42, 63, 77, 98, 99};
  const float ws[8] = {0.5f, 0.375f, 0.3125f, 0.25f, 0.4375f, 0.1875f, 0.25f, 0.1875f};
  for (uint32_t j = 0; j < 8; ++j) {
    forced.ids[j] = ids[j];
    forced.w[j] = ws[j];
  }
  l0::Mem rfb = d.upload(row_of(forced).data(), k2_ref::kWords * 4);
  const std::vector<uint16_t> xs = random_bf16(H, -0.1f, 0.1f, 32);
  l0::Mem xsb = d.upload(xs.data(), H * 2), hb = d.zeros(size_t(ms.slots()) * m.moe_inter * 2);
  l0::Module mm(d.ctx, kernels::path(kk::moe_variant(1, m.experts, m.top_k, H, m.moe_inter)));
  l0::Kernel ku = mm.kernel("k2_moe_gate_up");
  ku.group_size(kk::moe_gate_up_wg());
  ku.arg_ptr(0, rfb.ptr());
  ku.arg_ptr(1, xsb.ptr());
  ku.arg_ptr(2, gub.ptr());
  ku.arg_ptr(3, hb.ptr());
  const uint32_t up_groups = kk::moe_gate_up_groups(m.top_k, m.moe_inter);
  d.run(ku, up_groups, 1);
  const std::vector<uint16_t> hd = d.read<uint16_t>(hb, size_t(ms.slots()) * m.moe_inter);
  const std::vector<uint16_t> hr = moe_ref::gate_up(k2_ref::as_moe(forced, m.top_k), xs.data(),
                                                    gu.data(), ms, kk::kUpKs);
  for (uint32_t slot = 0; slot < ms.slots(); ++slot) {
    int worst = 0;
    for (uint32_t i = 0; i < m.moe_inter; ++i)
      worst = std::max(worst, ulps(hd[size_t(slot) * m.moe_inter + i], hr[size_t(slot) * m.moe_inter + i]));
    if (worst > 2) {
      std::fprintf(stderr, "k2_moe_gate_up slot %u (block %u): %d bf16 ulps off its expert\n", slot,
                   slot < 8 ? ids[slot] : m.shared_block(), worst);
      return 1;
    }
  }
  const std::vector<uint16_t> resid0 = random_bf16(H, -2.f, 2.f, 33);
  l0::Mem rb = d.upload(resid0.data(), H * 2);
  l0::Kernel kd = mm.kernel("k2_moe_down");
  kd.group_size(kk::moe_down_wg(m.top_k));
  kd.arg_ptr(0, rfb.ptr());
  kd.arg_ptr(1, hb.ptr());
  kd.arg_ptr(2, dnb.ptr());
  kd.arg_ptr(3, rb.ptr());
  d.run(kd, H / 16, 1);
  const std::vector<uint16_t> rd = d.read<uint16_t>(rb, H);
  const std::vector<uint16_t> rr = k2_ref::moe_down(forced, hd, dn.data(), ms, resid0.data(), kk::kDnKs);
  {
    int worst = 0;
    size_t exact = 0;
    for (uint32_t n = 0; n < H; ++n) {
      worst = std::max(worst, ulps(rd[n], rr[n]));
      exact += rd[n] == rr[n];
    }
    std::printf("  MoE: every gate||up slot within 2 ulps of its expert (shared = block 100); down + "
                "ascending-id combine + residual %zu / %u exact, worst %d ulps\n", exact, H, worst);
    CHECK(worst <= 2);
  }

  // ---- 6. MoVA's value experts -------------------------------------------------------------
  {
    const uint32_t N = m.kv_n();
    const size_t vblk = size_t(N / 16) * (H / 64) * 136;
    const std::vector<uint32_t> vw = random_blocks(vblk * m.value_experts, 40);
    l0::Mem vwb = d.upload(vw.data(), vw.size() * 4);
    k2_ref::Route vr;
    const uint32_t vids[4] = {2, 31, 32, 63};
    const float vws[4] = {0.625f, 0.75f, 0.5f, 0.625f};
    for (uint32_t j = 0; j < 4; ++j) {
      vr.ids[j] = vids[j];
      vr.w[j] = vws[j];
    }
    l0::Mem vrb = d.upload(row_of(vr).data(), k2_ref::kWords * 4);
    c->pos = 211;
    d.imm.fill(kvv.ptr(), 0u, kvv.size());
    l0::Module vm(d.ctx, kernels::path(kk::mova_variant(1, m.value_experts, m.value_top_k, H, N)));
    l0::Kernel kv = vm.kernel("k2_mova_value");
    kv.group_size(kk::mova_wg(m.value_top_k));
    kv.arg_ptr(0, ctrl.ptr());
    kv.arg_ptr(1, vrb.ptr());
    kv.arg_ptr(2, xsb.ptr());
    kv.arg_ptr(3, vwb.ptr());
    kv.arg_ptr(4, kvv.ptr());
    d.run(kv, N / 16, 1);
    const std::vector<uint16_t> got = d.read<uint16_t>(kvv, N, size_t(211) * N * 2);
    const std::vector<uint16_t> want = k2_ref::mova_value(vr, xs.data(), vw.data(), H, N, 4, kk::kMovaKs);
    int worst = 0;
    for (uint32_t n = 0; n < N; ++n) worst = std::max(worst, ulps(got[n], want[n]));
    CHECK(worst <= 2);
    CHECK(d.read<uint16_t>(kvv, N, size_t(210) * N * 2) == std::vector<uint16_t>(N, 0));
    std::printf("  MoVA value: 4 experts by id into V[pos 211], worst %d ulps; row 210 untouched\n", worst);
  }

  // ---- 7. decode attention + the softplus gate --------------------------------------------
  l0::Module am(d.ctx, kernels::path(kk::attn_variant(1, kk::kAttnTgt, m.q_heads, m.kv_heads)));
  l0::Kernel kad = am.kernel("k2_attn_decode");
  l0::Kernel kar = am.kernel("k2_attn_reduce");
  kad.group_size(kk::kAttnWg);
  kar.group_size(kk::kAttnWg);
  l0::Mem part = d.zeros(size_t(m.q_heads) * kk::kAttnTgt * kk::kAttnPart * 4);
  l0::Mem out = d.zeros(size_t(m.q_n()) * 2);
  {
    const std::vector<uint16_t> K = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 50);
    const std::vector<uint16_t> V = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 51);
    d.imm.copy(kvk.ptr(), K.data(), K.size() * 2);
    d.imm.copy(kvv.ptr(), V.data(), V.size() * 2);
    std::vector<float> q(m.q_n()), g(m.q_n());
    std::mt19937 rng(52);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    for (float& e : q) e = rf(u(rng));
    for (uint32_t i = 0; i < m.q_n(); ++i)   // gates on both sides of the threshold (28.85)
      g[i] = rf(i % 5 == 0 ? 40.0f + float(i % 7) : (i % 5 == 1 ? 28.0f : u(rng) * 4.f));
    d.imm.copy(aq.ptr(), q.data(), q.size() * 4);
    d.imm.copy(ag.ptr(), g.data(), g.size() * 4);
    kad.arg_ptr(0, ctrl.ptr());
    kad.arg_ptr(1, aq.ptr());
    kad.arg_ptr(2, kvk.ptr());
    kad.arg_ptr(3, kvv.ptr());
    kad.arg_ptr(4, part.ptr());
    kar.arg_ptr(0, ctrl.ptr());
    kar.arg_ptr(1, part.ptr());
    kar.arg_ptr(2, ag.ptr());
    kar.arg_ptr(3, out.ptr());
    for (uint32_t pos : {5u, 299u, 2999u}) {
      c->pos = pos;
      d.run(kad, m.kv_heads, kk::kAttnTgt);
      d.run(kar, m.q_heads, 1);
      const std::vector<uint16_t> o = d.read<uint16_t>(out, m.q_n());
      int worst = 0;
      for (uint32_t h = 0; h < m.q_heads; ++h) {
        const std::vector<double> ref = k2_ref::attention(q.data() + size_t(h) * 128, K.data(), V.data(),
                                                          pos + 1, h / m.gqa(), m.kv_heads, 128);
        for (uint32_t dd = 0; dd < 128; ++dd) {
          const uint16_t want = k2_ref::attn_gate(float(ref[dd]), g[size_t(h) * 128 + dd]);
          worst = std::max(worst, ulps(o[size_t(h) * 128 + dd], want));
        }
      }
      std::printf("  attention at %u keys: worst %d bf16 ulps against the fp64 softmax x softplus\n",
                  pos + 1, worst);
      CHECK(worst <= 2);
    }
  }

  // ---- 8. bitwise replay ------------------------------------------------------------------
  {
    const std::vector<uint16_t> o_first = d.read<uint16_t>(out, m.q_n());
    for (int rep = 0; rep < 2; ++rep) {
      d.imm.copy(rb.ptr(), resid0.data(), H * 2);
      l0::CmdList list = l0::CmdList::regular(d.ctx);
      list.launch(ku, up_groups, 1);
      list.launch(kd, H / 16, 1);
      list.launch(kad, m.kv_heads, kk::kAttnTgt);
      list.launch(kar, m.q_heads, 1);
      list.close();
      d.q.execute(list, &d.f);
      d.f.wait();
      CHECK(d.read<uint16_t>(hb, hd.size()) == hd);
      CHECK(d.read<uint16_t>(rb, H) == rd);
      CHECK(d.read<uint16_t>(out, m.q_n()) == o_first);
    }
    std::puts("  replay: the MoE pair and the attention pair bitwise identical");
  }
  // ---- 9. eager attention (spec 18 §10) ---------------------------------------------------
  {
    l0::Module em(d.ctx, kernels::path(kk::attn_eager_variant(1, kk::kAttnTgt, m.q_heads, m.kv_heads)));
    l0::Kernel es = em.kernel("k2_attn_eager_score");
    l0::Kernel ex = em.kernel("k2_attn_eager_softmax");
    l0::Kernel ev = em.kernel("k2_attn_eager_pv");
    l0::Kernel er = em.kernel("k2_attn_eager_reduce");
    for (l0::Kernel* k : {&es, &ex, &ev, &er}) k->group_size(kk::kAttnWg);
    const std::vector<uint16_t> K = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 70);
    const std::vector<uint16_t> V = random_bf16(size_t(ML) * m.kv_n(), -1.f, 1.f, 71);
    d.imm.copy(kvk.ptr(), K.data(), K.size() * 2);
    d.imm.copy(kvv.ptr(), V.data(), V.size() * 2);
    std::vector<float> q(m.q_n()), g(m.q_n());
    std::mt19937 rng(72);
    std::uniform_real_distribution<float> u(-4.f, 4.f);   // scores up to ~10: a peaked softmax
    for (float& e : q) e = rf(u(rng));
    for (uint32_t i = 0; i < m.q_n(); ++i)   // half above the threshold (exact path), half below
      g[i] = rf((i / 128) % 2 == 0 ? 30.0f + float(i % 11) : u(rng));
    d.imm.copy(aq.ptr(), q.data(), q.size() * 4);
    d.imm.copy(ag.ptr(), g.data(), g.size() * 4);
    l0::Mem sc = d.zeros(size_t(m.q_heads) * ML * 4);   // [M = 1][q_heads][ML]
    es.arg_ptr(0, ctrl.ptr());
    es.arg_ptr(1, aq.ptr());
    es.arg_ptr(2, kvk.ptr());
    es.arg_ptr(3, sc.ptr());
    es.arg<uint32_t>(4, ML);
    ex.arg_ptr(0, ctrl.ptr());
    ex.arg_ptr(1, sc.ptr());
    ex.arg<uint32_t>(2, ML);
    ev.arg_ptr(0, ctrl.ptr());
    ev.arg_ptr(1, sc.ptr());
    ev.arg_ptr(2, kvv.ptr());
    ev.arg_ptr(3, part.ptr());
    ev.arg<uint32_t>(4, ML);
    er.arg_ptr(0, ctrl.ptr());
    er.arg_ptr(1, part.ptr());
    er.arg_ptr(2, ag.ptr());
    er.arg_ptr(3, out.ptr());
    for (uint32_t pos : {5u, 299u, 2999u, ML - 1}) {
      c->pos = pos;
      const uint32_t len = pos + 1, blk = k2_ref::eager_block(pos, 1, kk::kAttnTgt);
      d.run(es, m.kv_heads, kk::kAttnTgt);
      const std::vector<float> s = d.read<float>(sc, size_t(m.q_heads) * ML);
      d.run(ex, m.q_heads, 1);
      const std::vector<float> pr = d.read<float>(sc, size_t(m.q_heads) * ML);
      d.run(ev, m.kv_heads, kk::kAttnTgt);
      d.run(er, m.q_heads, 1);
      const std::vector<uint16_t> o = d.read<uint16_t>(out, m.q_n());
      size_t s_bad = 0, p_bad = 0, o_exact_bad = 0, o_sp = 0;
      int sp_worst = 0;
      for (uint32_t h = 0; h < m.q_heads; ++h) {
        const k2_ref::EagerHead e = k2_ref::attention_eager(q.data() + size_t(h) * 128, K.data(), V.data(),
                                                            len, h / m.gqa(), m.kv_heads, 128, blk);
        for (uint32_t i = 0; i < len; ++i) {
          s_bad += s[size_t(h) * ML + i] != e.s[i];
          p_bad += pr[size_t(h) * ML + i] != e.p[i];
        }
        for (uint32_t dd = 0; dd < 128; ++dd) {
          const float gate = g[size_t(h) * 128 + dd];
          const uint16_t got = o[size_t(h) * 128 + dd], want = k2_ref::attn_gate(e.o[dd], gate);
          if (gate * k2_ref::kSpBeta > k2_ref::kSpThreshold) {
            o_exact_bad += got != want;
          } else {
            o_sp += got != want;
            sp_worst = std::max(sp_worst, ulps(got, want));
          }
        }
      }
      std::printf("  eager attention at %u keys (%u-key blocks): scores %zu, probabilities %zu, outputs "
                  "(gate past the threshold) %zu differ from k2_ref; below it %zu within %d ulp\n",
                  len, blk, s_bad, p_bad, o_exact_bad, o_sp, sp_worst);
      CHECK_EQ(s_bad, size_t(0));
      CHECK_EQ(p_bad, size_t(0));
      CHECK_EQ(o_exact_bad, size_t(0));
      CHECK(sp_worst <= 1);
    }
  }
  std::puts("k2_kernels_test OK");
  return 0;
}
