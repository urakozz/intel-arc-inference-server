// Spec 15c: src/kernels/moe.cl on the card, at Ornith's shape (256 experts, top-8,
// hidden 2048, expert 512), against tests/kernels/moe_ref.h. Needs a B70; no checkpoint.
//
//   1. the router || shared-gate GEMV (gemv_bf16 at N 272, {16, 16}) and moe_route on
//      its device logits: expert ids exactly the reference's (ties to the lower id),
//      weights / gate within a bf16 ulp, probabilities within 1e-5 relative;
//   2. moe_route on crafted logits with exact ties at Ornith's 256 experts: the lower
//      ids are taken, in id order, and the first expert not taken is the tie's partner;
//   3. moe_gate_up with a FORCED route row - ids {255, 0, 17, 128, 1, 200, 64, 3} and
//      the shared expert as slot 8 - every slot's h against ITS expert's reference: each
//      expert's weights are distinct, so a wrong id read or a wrong block stride shows as
//      one slot wrong, named (spec 15c Review Focus 2);
//   4. moe_down on the device h: the weighted sum in fixed slot order, the shared
//      expert by its gate, the residual fold, against the reference (<= 2 bf16 ulps);
//   5. bitwise determinism: the gate_up + down pair replayed from the same inputs gives
//      the same bits.
// The GEMV-fed values carry a tolerance (OpenCL may contract the GEMV's multiply-adds,
// and `exp` is 3 ulp); everything after them is exact in the reference (moe_ref_test).
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
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "moe_ref.h"

namespace {

static_assert(kernels::moe_route::kWords == moe_ref::kWords && kernels::moe_route::kIds == moe_ref::kIds &&
                  kernels::moe_route::kWeights == moe_ref::kWeights &&
                  kernels::moe_route::kSharedGate == moe_ref::kSharedGate &&
                  kernels::moe_route::kProbs == moe_ref::kProbs &&
                  kernels::moe_route::kProb9 == moe_ref::kProb9,
              "the route row layout: moe.cl R_*, kernels::moe_route and moe_ref disagree");

using moe_ref::f32;
using moe_ref::rne;

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
  std::vector<T> read(const l0::Mem& m, size_t n) {
    std::vector<T> v(n);
    imm.copy(v.data(), m.ptr(), n * sizeof(T));
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

// Random layout-1 expert blocks, generated as tiles: any nibble word is a valid weight,
// and each tile's 16 f16 scales are set to [0.01, 0.05] (a real checkpoint's range).
std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) |
                      (uint32_t(common::f32_to_f16(sd(rng))) << 16);
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

const moe_ref::Shape kS = moe_ref::ornith_shape();

// ids exactly the reference's - except at the top-k boundary when the reference's k-th
// and (k+1)-th probabilities are within the host / device `exp` slack (a near-tie: R2's
// tolerance case, spec 15 §5), where only the first top_k - 1 are compared.
void check_route(const std::vector<uint32_t>& row, const moe_ref::Route& want, const char* what) {
  const float pk = want.p[kS.top_k - 1];
  const bool near_tie = pk != want.p9 && pk - want.p9 <= 1e-5f * want.p9;
  if (near_tie)
    std::printf("  %s: near-tie at the top-%u boundary, its slot not compared\n", what, kS.top_k);
  for (uint32_t k = 0; k < kS.top_k; ++k) {
    if (near_tie && k == kS.top_k - 1) continue;
    if (row[moe_ref::kIds + k] != want.ids[k]) {
      std::fprintf(stderr, "%s: slot %u expert %u, reference %u\n", what, k, row[moe_ref::kIds + k],
                   want.ids[k]);
      std::exit(1);
    }
    float w, p;
    std::memcpy(&w, &row[moe_ref::kWeights + k], 4);
    std::memcpy(&p, &row[moe_ref::kProbs + k], 4);
    CHECK(ulps(rne(w), rne(want.w[k])) <= 1);
    CHECK_NEAR(p, want.p[k], 1e-5 * want.p[k] + 1e-12);
  }
  float sg;
  std::memcpy(&sg, &row[moe_ref::kSharedGate], 4);
  CHECK(ulps(rne(sg), rne(want.sg)) <= 1);
}

}  // namespace

int main() {
  Dev d;
  const std::string moe_v = kernels::moe_variant(1, kS.experts, kS.top_k, kS.hidden, kS.inter);
  CHECK_EQ(moe_v, std::string("moe_M1_E256_T8_D2048_I512"));
  l0::Module moe(d.ctx, kernels::path(moe_v));

  // ---- 1. the router GEMV + moe_route ------------------------------------------
  const std::vector<uint16_t> x = random_bf16(kS.hidden, -1.0f, 1.0f, 1);
  std::vector<uint16_t> rows = random_bf16(size_t(kS.router_n) * kS.hidden, -0.05f, 0.05f, 2);
  std::fill(rows.begin() + size_t(kS.experts + 1) * kS.hidden, rows.end(), uint16_t(0));
  std::vector<uint16_t> tiled(rows.size());
  common::repack_bf16_tiled(rows.data(), kS.hidden, kS.router_n, tiled.data());
  l0::Mem wr = d.upload(tiled.data(), tiled.size() * 2), xb = d.upload(x.data(), x.size() * 2);
  l0::Mem logits(d.ctx, l0::MemKind::Device, size_t(kS.router_n) * 4);
  {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(kS.router_n);
    CHECK(t.cols == 16 && t.ksplit == 16);
    l0::Module gm(d.ctx, kernels::path(kernels::gemv_bf16_variant(1, kS.hidden, kS.router_n, t)));
    l0::Kernel k = gm.kernel("gemv_bf16");
    k.group_size(t.cols * t.ksplit);
    k.arg_ptr(0, wr.ptr());
    k.arg_ptr(1, xb.ptr());
    k.arg_ptr(2, logits.ptr());
    d.run(k, kS.router_n / t.cols);
  }
  const std::vector<float> lg = d.read<float>(logits, kS.router_n);
  for (uint32_t n = 0; n < kS.router_n; ++n) {
    double ref = 0, mag = 0;
    for (uint32_t k = 0; k < kS.hidden; ++k) {
      ref += double(f32(rows[size_t(n) * kS.hidden + k])) * f32(x[k]);
      mag += std::fabs(double(f32(rows[size_t(n) * kS.hidden + k])) * f32(x[k]));
    }
    CHECK_NEAR(lg[n], ref, 1e-5 * mag + 1e-7);
  }
  l0::Mem route(d.ctx, l0::MemKind::Device, moe_ref::kWords * 4);
  l0::Kernel kr = moe.kernel("moe_route");
  kr.group_size(kS.experts);
  kr.arg_ptr(0, logits.ptr());
  kr.arg_ptr(1, route.ptr());
  d.run(kr, 1, 1);
  check_route(d.read<uint32_t>(route, moe_ref::kWords), moe_ref::route(lg.data(), kS),
              "route of the device logits");
  std::puts("  router GEMV + moe_route: ids exact, weights / gate within 1 bf16 ulp");

  // ---- 2. exact ties at 256 experts ----------------------------------------------
  {
    std::vector<float> tie(kS.router_n, 0.0f);
    for (uint32_t e = 0; e < kS.experts; ++e) tie[e] = -2.0f + float(e % 64) / 64.0f;   // repeats
    tie[200] = tie[7] = 4.0f;    // slot 0, 1: 7 then 200
    tie[31] = 3.0f;
    tie[250] = tie[100] = tie[3] = tie[66] = 2.5f;   // slots 3..6: 3, 66, 100, 250
    tie[129] = tie[12] = 2.0f;   // slot 7 is 12; 129 is the first expert not taken
    tie[kS.experts] = -1.0f;
    l0::Mem tl = d.upload(tie.data(), tie.size() * 4);
    kr.arg_ptr(0, tl.ptr());
    d.run(kr, 1, 1);
    const std::vector<uint32_t> row = d.read<uint32_t>(route, moe_ref::kWords);
    const uint32_t want[8] = {7, 200, 31, 3, 66, 100, 250, 12};
    for (uint32_t k = 0; k < 8; ++k) CHECK_EQ(row[moe_ref::kIds + k], want[k]);
    float p9, p8;
    std::memcpy(&p9, &row[moe_ref::kProb9], 4);
    std::memcpy(&p8, &row[moe_ref::kProbs + 7], 4);
    CHECK(p9 == p8);   // expert 129: tied with slot 7, not taken
    check_route(row, moe_ref::route(tie.data(), kS), "route of tied logits");
    std::puts("  moe_route ties: lower ids taken, in id order");
  }

  // ---- 3. moe_gate_up on a forced route row --------------------------------------
  const std::vector<uint32_t> gu = random_blocks(kS.gate_up_words() * (kS.experts + 1), 3);
  const std::vector<uint32_t> dn = random_blocks(kS.down_words() * (kS.experts + 1), 4);
  l0::Mem gub = d.upload(gu.data(), gu.size() * 4), dnb = d.upload(dn.data(), dn.size() * 4);
  moe_ref::Route forced;
  const uint32_t ids[8] = {255, 0, 17, 128, 1, 200, 64, 3};
  const float ws[8] = {0.25f, 0.1875f, 0.15625f, 0.125f, 0.09375f, 0.078125f, 0.0625f, 0.046875f};
  std::vector<uint32_t> row(moe_ref::kWords, 0);
  for (uint32_t k = 0; k < 8; ++k) {
    forced.ids[k] = ids[k];
    forced.w[k] = ws[k];
    row[moe_ref::kIds + k] = ids[k];
    std::memcpy(&row[moe_ref::kWeights + k], &ws[k], 4);
  }
  forced.sg = f32(rne(0.6f));
  std::memcpy(&row[moe_ref::kSharedGate], &forced.sg, 4);
  l0::Mem rf = d.upload(row.data(), row.size() * 4);
  l0::Mem h(d.ctx, l0::MemKind::Device, size_t(kS.slots()) * kS.inter * 2);
  const std::vector<uint16_t> xs = random_bf16(kS.hidden, -0.1f, 0.1f, 5);
  l0::Mem xsb = d.upload(xs.data(), xs.size() * 2);
  l0::Kernel ku = moe.kernel("moe_gate_up");
  ku.group_size(kernels::moe_gate_up_wg());
  ku.arg_ptr(0, rf.ptr());
  ku.arg_ptr(1, xsb.ptr());
  ku.arg_ptr(2, gub.ptr());
  ku.arg_ptr(3, h.ptr());
  const uint32_t up_groups = kernels::moe_gate_up_groups(kS.top_k, kS.inter);
  d.run(ku, up_groups, 1);
  const std::vector<uint16_t> hd = d.read<uint16_t>(h, size_t(kS.slots()) * kS.inter);
  const std::vector<uint16_t> hr =
      moe_ref::gate_up(forced, xs.data(), gu.data(), kS, kernels::kMoeUpKs);
  for (uint32_t slot = 0; slot < kS.slots(); ++slot) {
    int worst = 0;
    for (uint32_t i = 0; i < kS.inter; ++i)
      worst = std::max(worst, ulps(hd[size_t(slot) * kS.inter + i], hr[size_t(slot) * kS.inter + i]));
    if (worst > 2) {
      std::fprintf(stderr, "moe_gate_up slot %u (expert %u): %d bf16 ulps off its expert's reference"
                   " - wrong id or block stride?\n", slot, slot < 8 ? ids[slot] : kS.experts, worst);
      std::exit(1);
    }
  }
  std::puts("  moe_gate_up: every slot within 2 bf16 ulps of ITS expert (shared = slot 8)");

  // ---- 4. moe_down -----------------------------------------------------------------
  const std::vector<uint16_t> resid0 = random_bf16(kS.hidden, -2.0f, 2.0f, 6);
  l0::Mem rb = d.upload(resid0.data(), resid0.size() * 2);
  l0::Kernel kd = moe.kernel("moe_down");
  kd.group_size(kernels::moe_down_wg(kS.top_k));
  kd.arg_ptr(0, rf.ptr());
  kd.arg_ptr(1, h.ptr());
  kd.arg_ptr(2, dnb.ptr());
  kd.arg_ptr(3, rb.ptr());
  d.run(kd, kS.hidden / 16, 1);
  const std::vector<uint16_t> rd = d.read<uint16_t>(rb, kS.hidden);
  const std::vector<uint16_t> rr =
      moe_ref::down(forced, hd, dn.data(), kS, resid0.data(), kernels::kMoeDnKs);
  int worst = 0;
  size_t exact = 0;
  for (uint32_t n = 0; n < kS.hidden; ++n) {
    worst = std::max(worst, ulps(rd[n], rr[n]));
    exact += rd[n] == rr[n];
  }
  std::printf("  moe_down: %zu / %u columns exact, worst %d bf16 ulps\n", exact, kS.hidden, worst);
  CHECK(worst <= 2);

  // ---- 5. bitwise replay -------------------------------------------------------------
  for (int rep = 0; rep < 2; ++rep) {
    d.imm.copy(rb.ptr(), resid0.data(), resid0.size() * 2);
    l0::CmdList list = l0::CmdList::regular(d.ctx);
    list.launch(ku, up_groups, 1);
    list.launch(kd, kS.hidden / 16, 1);
    list.close();
    d.q.execute(list, &d.f);
    d.f.wait();
    CHECK(d.read<uint16_t>(h, hd.size()) == hd);
    CHECK(d.read<uint16_t>(rb, kS.hidden) == rd);
  }
  std::puts("moe_test OK");
  return 0;
}
