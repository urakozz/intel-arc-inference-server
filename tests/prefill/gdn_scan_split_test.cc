// Opt-in D3 arithmetic discriminator.  These fixtures invoke only the scan
// entry point: their expected values are hand-derived scalar recurrences, not
// the scan implementation or the chunk reference.  Removing either low DPAS
// chain changes a checked value by orders of magnitude.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "gdn_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "prefill/pf_harness.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {
using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;
namespace G = gdn_ref;

constexpr uint32_t kHeads = G::kHeads;
constexpr uint32_t kDim = G::kDim;
constexpr uint32_t kConvRows = G::kConvRows;
constexpr uint32_t kState = kHeads * kDim * kDim;

struct Dev {
  l0::Context ctx{0};
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

struct Input {
  uint32_t C;
  std::vector<uint16_t> xb, w, u;
  std::vector<float> a2, gc, state;

  explicit Input(uint32_t count)
      : C(count), xb(size_t(C) * kConvRows, 0), w(size_t(C) * kHeads * kDim, 0),
        u(size_t(C) * kHeads * kDim, 0),
        a2(size_t((C + 63) / 64) * kHeads * 64 * 64, 0), gc(size_t(C) * kHeads, 0),
        state(kState, 0) {}
};

struct Output {
  std::vector<float> state, o;
};

Output run(Dev& d, const Input& in) {
  l0::Mem xb = pf_harness::upload(d.ctx, d.imm, in.xb);
  l0::Mem w = pf_harness::upload(d.ctx, d.imm, in.w);
  l0::Mem u = pf_harness::upload(d.ctx, d.imm, in.u);
  l0::Mem a2 = pf_harness::upload(d.ctx, d.imm, in.a2);
  l0::Mem gc = pf_harness::upload(d.ctx, d.imm, in.gc);
  l0::Mem state = pf_harness::upload(d.ctx, d.imm, in.state);
  l0::Mem o(d.ctx, l0::MemKind::Device, size_t(in.C) * kHeads * kDim * sizeof(float));
  // A zero-output fixture must prove stores happened; zero initialization would
  // let an unwritten output pass it.
  d.imm.fill(o.ptr(), 0x7fc00000u, o.size());
  d.cx.launch(d.kc(kernels::pf_gdn_scan_variant(), "pf_gdn_scan_dpas_split"), kHeads, 4, 1,
              {PtrArg(xb.ptr()), PtrArg(w.ptr()), PtrArg(u.ptr()), PtrArg(a2.ptr()),
               PtrArg(gc.ptr()), PtrArg(state.ptr()), PtrArg(o.ptr()), arg_val(in.C)});
  d.cx.wait();
  Output out{std::vector<float>(kState), std::vector<float>(size_t(in.C) * kHeads * kDim)};
  d.imm.copy(out.state.data(), state.ptr(), state.size());
  d.imm.copy(out.o.data(), o.ptr(), o.size());
  return out;
}

size_t cv(uint32_t m, uint32_t h, uint32_t x) {
  return (size_t(m) * kHeads + h) * kDim + x;
}
size_t sk(uint32_t h, uint32_t k, uint32_t x) {
  return (size_t(h) * kDim + k) * kDim + x;
}
size_t a2i(uint32_t h, uint32_t i, uint32_t j) { return (size_t(h) * 64 + i) * 64 + j; }

void finite(const std::vector<float>& v) {
  for (float x : v) CHECK(std::isfinite(x));
}

void check_split_s_residual(Dev& d) {
  // Scalar recurrence: S=1+2^-8, W=U=Q=1, K=0, g=0, A2=22.625.
  // Therefore vn=-2^-8 and o=(1+2^-8)/sqrt(128)-22.625*2^-8 = 0.0003547...
  // A high-only S chain rounds S to one and instead produces about 0.0883883.
  Input in(1);
  in.state[sk(0, 0, 0)] = 1.0f + 0x1p-8f;
  in.w[cv(0, 0, 0)] = G::rne(1.0f);
  in.u[cv(0, 0, 0)] = G::rne(1.0f);
  in.xb[G::kQOff] = G::rne(1.0f);
  in.a2[a2i(0, 0, 0)] = 22.625f;  // BF16 exact.
  const Output a = run(d, in);
  const Output b = run(d, in);
  const float want = 0.0003547f;
  CHECK(std::fabs(a.o[cv(0, 0, 0)] - want) < 5e-6f);
  CHECK(std::memcmp(a.o.data(), b.o.data(), a.o.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(a.state.data(), b.state.data(), a.state.size() * sizeof(float)) == 0);
  finite(a.state);
  finite(a.o);
}

void check_split_d_residual(Dev& d) {
  // Scalar update: initial S=0, U0=1, K0=1, gc0=0, gc1=-2^-10.
  // At the sub-chunk end D0=exp(-2^-10), so the retained FP32 state is that
  // value.  A high-only D chain rounds it to one.
  Input in(2);
  in.u[cv(0, 0, 0)] = G::rne(1.0f);
  in.xb[G::kKOff] = G::rne(1.0f);
  for (uint32_t h = 0; h < kHeads; ++h) in.gc[size_t(1) * kHeads + h] = -0x1p-10f;
  const Output a = run(d, in);
  const float want = std::exp(-0x1p-10f);
  CHECK(std::fabs(a.state[sk(0, 0, 0)] - want) < 5e-6f);
  finite(a.state);
  finite(a.o);
}

void check_zero_ragged(Dev& d) {
  // C=100 forces a live 64+36 scan; a zero initial state/input must remain
  // finite zero, exercising the clamped final sub-chunk without an oracle that
  // mirrors the GPU's split arithmetic.
  Input in(100);
  const Output a = run(d, in);
  const Output b = run(d, in);
  finite(a.state);
  finite(a.o);
  for (float x : a.state) CHECK_EQ(x, 0.0f);
  for (float x : a.o) CHECK_EQ(x, 0.0f);
  CHECK(std::memcmp(a.state.data(), b.state.data(), a.state.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(a.o.data(), b.o.data(), a.o.size() * sizeof(float)) == 0);
}
}  // namespace

int main() {
  Dev d;
  check_split_s_residual(d);
  check_split_d_residual(d);
  check_zero_ragged(d);
  std::puts("gdn_scan_split_test OK");
  return 0;
}
