// Spec 8 §3.4 (plan 8b Task 4): gdn_step's SPEC_SLOTS build - the MTP verify step's
// per-row GDN state.
//
// One synthetic GDN layer. The reference is the shipped M = 1 kernel run M times in a
// row (row m at pos + m, carrying the ring and the state); the candidate is
// gdn_step_slots_M<M> once over the M rows, reading the state from slot `live` and
// writing the state after row m into slot (live + m) % 4. Each written slot must equal
// the reference state after that row (cosine >= 0.99999, max abs recorded; the two do
// the same per-row arithmetic, so bitwise equality is expected and reported), gdn_o
// and the conv ring must match, the slots not written must be untouched, and a second
// run must be bitwise identical. Cases: M = 1..4, live 0 and 3 (the slot index wraps),
// pos 14 (the conv ring's position index wraps inside the rows at M >= 3).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "gdn_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {
using gdn_ref::kAbStride;
using gdn_ref::kChunks;
using gdn_ref::kConvRows;
using gdn_ref::kDim;
using gdn_ref::kHeads;
using gdn_ref::kQkvzN;
using gdn_ref::kRing;

constexpr size_t kSmallFloats = 164480 / 4;
constexpr size_t kStateElems = size_t(kHeads) * kDim * kDim;       // one layer
constexpr size_t kSlotStride = 48 * kStateElems;                    // SPEC_SLOT_STRIDE
constexpr size_t kRingElems = size_t(kRing) * kConvRows;
constexpr uint32_t kSlots = 4;

std::vector<float> normal(size_t n, uint32_t seed, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.f, sigma);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

void run(Dev& d, l0::Kernel& k, uint32_t gx, uint32_t gy) {
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(k, gx, gy);
  list.close();
  d.q.execute(list, &d.fence);
  d.fence.wait();
}

double cosine(const float* a, const float* b, size_t n, double* max_abs) {
  double ab = 0, aa = 0, bb = 0;
  *max_abs = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
    *max_abs = std::fmax(*max_abs, std::fabs(double(a[i]) - b[i]));
  }
  return ab / std::sqrt(aa * bb);
}

void run_case(Dev& d, l0::Mem& spec, uint32_t M, uint32_t live, uint32_t pos, uint32_t seed) {
  std::vector<float> qkvz = normal(size_t(M) * kQkvzN, seed + 1, 1.f);
  std::vector<float> ab = normal(size_t(M) * kAbStride, seed + 2, 1.f);
  std::vector<float> small = normal(kSmallFloats, seed + 3, 0.3f);
  std::mt19937 rng(seed + 4);
  std::uniform_real_distribution<float> alog(-4.0f, 0.5f), dtb(-1.0f, 1.0f);
  for (uint32_t h = 0; h < kHeads; ++h) {
    small[gdn_ref::kNegAOff + h] = -std::exp(alog(rng));
    small[gdn_ref::kDtBiasOff + h] = dtb(rng);
  }
  std::vector<uint16_t> ring0(kRingElems);
  for (size_t i = 0; i < ring0.size(); ++i)
    ring0[i] = common::f32_to_bf16(float(int(i * 7919 % 4001) - 2000) / 1000.f);
  const std::vector<float> state0 = normal(kStateElems, seed + 6, 0.5f);

  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  l0::Mem qb(d.ctx, l0::MemKind::Device, qkvz.size() * 4), abb(d.ctx, l0::MemKind::Device, ab.size() * 4);
  l0::Mem sb(d.ctx, l0::MemKind::Device, small.size() * 4);
  l0::Mem rb(d.ctx, l0::MemKind::Device, kRingElems * 2), st(d.ctx, l0::MemKind::Device, kStateElems * 4);
  l0::Mem ob(d.ctx, l0::MemKind::Device, size_t(M) * kHeads * kDim * 4);
  d.imm.copy(sb.ptr(), small.data(), small.size() * 4);
  runtime::Control* c = ctrl.as<runtime::Control>();

  // Reference: the shipped M = 1 kernel, row by row.
  l0::Module m1(d.ctx, kernels::path(kernels::gdn_step_variant(1)));
  l0::Kernel k1(m1, "gdn_step");
  k1.group_size(256);
  d.imm.copy(rb.ptr(), ring0.data(), kRingElems * 2);
  d.imm.copy(st.ptr(), state0.data(), kStateElems * 4);
  std::vector<std::vector<float>> ref_state(M, std::vector<float>(kStateElems));
  std::vector<float> ref_o(size_t(M) * kHeads * kDim);
  for (uint32_t m = 0; m < M; ++m) {
    *c = runtime::Control{};
    c->pos = pos + m;
    c->n_active = 1;
    d.imm.copy(qb.ptr(), qkvz.data() + size_t(m) * kQkvzN, size_t(kQkvzN) * 4);
    d.imm.copy(abb.ptr(), ab.data() + size_t(m) * kAbStride, size_t(kAbStride) * 4);
    k1.arg_ptr(0, ctrl.ptr());
    k1.arg_ptr(1, qb.ptr());
    k1.arg_ptr(2, abb.ptr());
    k1.arg_ptr(3, sb.ptr());
    k1.arg_ptr(4, rb.ptr());
    k1.arg_ptr(5, st.ptr());
    k1.arg_ptr(6, ob.ptr());
    run(d, k1, kHeads, kChunks);
    d.imm.copy(ref_state[m].data(), st.ptr(), kStateElems * 4);
    d.imm.copy(ref_o.data() + size_t(m) * kHeads * kDim, ob.ptr(), size_t(kHeads) * kDim * 4);
  }
  std::vector<uint16_t> ref_ring(kRingElems);
  d.imm.copy(ref_ring.data(), rb.ptr(), kRingElems * 2);

  // Candidate: the slots build over all M rows, state in slot `live`.
  l0::Module ms(d.ctx, kernels::path(kernels::gdn_step_slots_variant(M, 48)));
  l0::Kernel ks(ms, "gdn_step");
  ks.group_size(256);
  auto slot = [&](uint32_t s) -> float* {
    return s == 0 ? st.as<float>() : spec.as<float>() + size_t(s - 1) * kSlotStride;
  };
  std::vector<std::vector<float>> got(2, std::vector<float>(size_t(kSlots) * kStateElems));
  std::vector<float> got_o(ref_o.size());
  std::vector<uint16_t> got_ring(kRingElems);
  const std::vector<float> sentinel(kStateElems, 12345.0f);
  for (int rep = 0; rep < 2; ++rep) {
    for (uint32_t s = 0; s < kSlots; ++s)
      d.imm.copy(slot(s), s == live ? state0.data() : sentinel.data(), kStateElems * 4);
    d.imm.copy(rb.ptr(), ring0.data(), kRingElems * 2);
    d.imm.copy(qb.ptr(), qkvz.data(), qkvz.size() * 4);
    d.imm.copy(abb.ptr(), ab.data(), ab.size() * 4);
    *c = runtime::Control{};
    c->pos = pos;
    c->n_active = M;
    c->gdn_live = live;
    ks.arg_ptr(0, ctrl.ptr());
    ks.arg_ptr(1, qb.ptr());
    ks.arg_ptr(2, abb.ptr());
    ks.arg_ptr(3, sb.ptr());
    ks.arg_ptr(4, rb.ptr());
    ks.arg_ptr(5, st.ptr());
    ks.arg_ptr(6, ob.ptr());
    ks.arg_ptr(7, spec.ptr());
    run(d, ks, kHeads, kChunks);
    for (uint32_t s = 0; s < kSlots; ++s)
      d.imm.copy(got[rep].data() + size_t(s) * kStateElems, slot(s), kStateElems * 4);
    if (rep == 0) {
      d.imm.copy(got_o.data(), ob.ptr(), got_o.size() * 4);
      d.imm.copy(got_ring.data(), rb.ptr(), kRingElems * 2);
    }
  }
  CHECK(std::memcmp(got[0].data(), got[1].data(), got[0].size() * 4) == 0);
  CHECK(std::memcmp(got_ring.data(), ref_ring.data(), kRingElems * 2) == 0);
  double oma = 0;
  const double oc = cosine(got_o.data(), ref_o.data(), ref_o.size(), &oma);
  CHECK(oc >= 0.99999);
  for (uint32_t m = 0; m < M; ++m) {
    const float* g = got[0].data() + size_t((live + m) % kSlots) * kStateElems;
    double ma = 0;
    const double cs = cosine(g, ref_state[m].data(), kStateElems, &ma);
    const bool bits = std::memcmp(g, ref_state[m].data(), kStateElems * 4) == 0;
    std::printf("  M=%u live=%u pos=%u row %u -> slot %u: cos %.9f max abs %.3e%s\n", M, live, pos,
                m, (live + m) % kSlots, cs, ma, bits ? " (bitwise)" : "");
    CHECK(cs >= 0.99999);
  }
  // Slots the rows did not reach keep the sentinel.
  for (uint32_t s = 0; s < kSlots; ++s) {
    bool written = false;
    for (uint32_t m = 0; m < M; ++m) written |= (live + m) % kSlots == s;
    if (!written)
      CHECK(std::memcmp(got[0].data() + size_t(s) * kStateElems, sentinel.data(),
                        kStateElems * 4) == 0);
  }
  std::printf("gdn_step_slots M=%u live=%u: gdn_o cos %.9f max abs %.3e, ring bitwise\n", M, live,
              oc, oma);
}
}  // namespace

int main() {
  Dev d;
  l0::Mem spec(d.ctx, l0::MemKind::Device, size_t(kSlots - 1) * kSlotStride * 4);
  uint32_t seed = 100;
  for (uint32_t M = 1; M <= 4; ++M)
    for (uint32_t live : {0u, 3u}) run_case(d, spec, M, live, 14, seed += 10);
  std::puts("gdn_step_slots_test OK");
  return 0;
}
