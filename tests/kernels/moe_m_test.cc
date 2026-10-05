// Spec 15e: src/kernels/moe.cl at M = 2..4 rows (the MTP verify lists' MoE block, spec 15e
// Review Focus 3) against M = 1 on each row alone, on the card. Needs a B70; no checkpoint.
//
// moe_route / moe_gate_up / moe_down read and write every buffer at the row's own group id
// (get_group_id(1)) and no line of moe.cl depends on M but the grid, so an M-row launch must
// give each row exactly the bits an M = 1 launch of that row gives - routing per row, the
// experts each row chose, the weighted sum, the residual fold. Bitwise here, because spec
// 8's M2 (verify rows bitwise equal to decode) rests on it:
//
//   1. M rows with distinct logits (one row with exact ties at the top-8 boundary, one with
//      every row's x different): the route rows, h and the folded resid of row m of
//      moe_M<M> == moe_M1 run on row m alone, for M = 2, 3, 4;
//   2. the rows reversed inside the launch: each row's bits are unchanged (a row's result
//      does not depend on its neighbours or its position);
//   3. the MoE scratch's own rows (kM = 8 per layer slot, runtime::moe_scratch_layout) are
//      where capture binds an M-row launch: the strides used here are its.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
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

const moe_ref::Shape kS = moe_ref::ornith_shape();

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
};

std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {   // moe_test's
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

// One block run of `rows` rows through the binary for M = rows: logits [rows][router_n]
// fp32 -> route, x [rows][hidden] -> h, resid [rows][hidden] folded in place.
struct Out {
  std::vector<uint32_t> route;
  std::vector<uint16_t> h, resid;
};
Out run(Dev& d, uint32_t M, const std::vector<float>& logits, const std::vector<uint16_t>& x,
        const std::vector<uint16_t>& resid, const l0::Mem& gu, const l0::Mem& dn) {
  l0::Module mod(d.ctx, kernels::path(kernels::moe_variant(M, kS.experts, kS.top_k, kS.hidden,
                                                            kS.inter)));
  l0::Mem lg = d.upload(logits.data(), logits.size() * 4);
  l0::Mem xb = d.upload(x.data(), x.size() * 2);
  l0::Mem rb = d.upload(resid.data(), resid.size() * 2);
  l0::Mem route(d.ctx, l0::MemKind::Device, size_t(M) * moe_ref::kWords * 4);
  l0::Mem h(d.ctx, l0::MemKind::Device, size_t(M) * kS.slots() * kS.inter * 2);
  l0::Kernel kr = mod.kernel("moe_route");
  kr.group_size(kS.experts);
  kr.arg_ptr(0, lg.ptr());
  kr.arg_ptr(1, route.ptr());
  l0::Kernel ku = mod.kernel("moe_gate_up");
  ku.group_size(kernels::moe_gate_up_wg());
  ku.arg_ptr(0, route.ptr());
  ku.arg_ptr(1, xb.ptr());
  ku.arg_ptr(2, gu.ptr());
  ku.arg_ptr(3, h.ptr());
  l0::Kernel kd = mod.kernel("moe_down");
  kd.group_size(kernels::moe_down_wg(kS.top_k));
  kd.arg_ptr(0, route.ptr());
  kd.arg_ptr(1, h.ptr());
  kd.arg_ptr(2, dn.ptr());
  kd.arg_ptr(3, rb.ptr());
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(kr, 1, M);   // the capture's grids (capture.cc moe_block)
  list.launch(ku, kernels::moe_gate_up_groups(kS.top_k, kS.inter), M);
  list.launch(kd, kS.hidden / 16, M);
  list.close();
  d.q.execute(list, &d.f);
  d.f.wait();
  return {d.read<uint32_t>(route, size_t(M) * moe_ref::kWords),
          d.read<uint16_t>(h, size_t(M) * kS.slots() * kS.inter), d.read<uint16_t>(rb, resid.size())};
}

template <class T>
std::vector<T> row_of(const std::vector<T>& v, size_t width, uint32_t m) {
  return std::vector<T>(v.begin() + m * width, v.begin() + (m + 1) * width);
}

}  // namespace

int main() {
  Dev d;
  const std::vector<uint32_t> gu = random_blocks(kS.gate_up_words() * (kS.experts + 1), 3);
  const std::vector<uint32_t> dn = random_blocks(kS.down_words() * (kS.experts + 1), 4);
  l0::Mem gub = d.upload(gu.data(), gu.size() * 4), dnb = d.upload(dn.data(), dn.size() * 4);
  std::mt19937 rng(1505);
  std::uniform_real_distribution<float> ld(-3.f, 3.f), xd(-0.1f, 0.1f), rd(-2.f, 2.f);
  const uint32_t kMax = 4, LW = kS.router_n, RW = moe_ref::kWords, HW = kS.slots() * kS.inter;
  std::vector<float> logits(size_t(kMax) * LW);
  for (float& v : logits) v = ld(rng);
  // Row 1: exact ties across the top-8 boundary (the lower ids taken, per row).
  for (uint32_t e : {9u, 40u, 41u, 77u, 130u, 131u, 200u, 201u, 250u}) logits[size_t(1) * LW + e] = 5.f;
  std::vector<uint16_t> x(size_t(kMax) * kS.hidden), resid(size_t(kMax) * kS.hidden);
  for (uint16_t& v : x) v = common::f32_to_bf16(xd(rng));
  for (uint16_t& v : resid) v = common::f32_to_bf16(rd(rng));

  // The M = 1 reference of every row, alone.
  std::vector<Out> one;
  for (uint32_t m = 0; m < kMax; ++m)
    one.push_back(run(d, 1, row_of(logits, LW, m), row_of(x, kS.hidden, m),
                      row_of(resid, kS.hidden, m), gub, dnb));
  for (uint32_t M = 2; M <= kMax; ++M) {
    for (bool reversed : {false, true}) {
      std::vector<float> lg;
      std::vector<uint16_t> xs, rs;
      std::vector<uint32_t> order;
      for (uint32_t i = 0; i < M; ++i) order.push_back(reversed ? M - 1 - i : i);
      for (uint32_t m : order) {
        const auto a = row_of(logits, LW, m);
        const auto b = row_of(x, kS.hidden, m);
        const auto c = row_of(resid, kS.hidden, m);
        lg.insert(lg.end(), a.begin(), a.end());
        xs.insert(xs.end(), b.begin(), b.end());
        rs.insert(rs.end(), c.begin(), c.end());
      }
      const Out got = run(d, M, lg, xs, rs, gub, dnb);
      for (uint32_t i = 0; i < M; ++i) {
        const uint32_t m = order[i];
        const bool ok = row_of(got.route, RW, i) == one[m].route && row_of(got.h, HW, i) == one[m].h &&
                        row_of(got.resid, kS.hidden, i) == one[m].resid;
        if (!ok) {
          std::fprintf(stderr, "moe_M%u%s row %u (input row %u) is not moe_M1's bits on that row\n",
                       M, reversed ? " reversed" : "", i, m);
          return 1;
        }
      }
    }
    std::printf("  moe_M%u: every row bitwise moe_M1's, in order and reversed\n", M);
  }
  // Row 1's tie: the lower ids, in id order (moe_route's rank rule), per row at every M.
  const uint32_t want[8] = {9, 40, 41, 77, 130, 131, 200, 201};
  for (uint32_t k = 0; k < 8; ++k) CHECK_EQ(one[1].route[moe_ref::kIds + k], want[k]);
  std::puts("moe_m_test OK");
  return 0;
}
