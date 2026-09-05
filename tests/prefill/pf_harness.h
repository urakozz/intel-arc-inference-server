#pragma once
// Shared plumbing for the runtime-`M` prefill kernel tests. Everything here is
// lifted verbatim from `tests/kernels/prep_test.cc` -- the bf16 ulp metric, the
// upload/download pair and the one-launch `Dev` -- so that the three prefill
// tests grade their outputs on exactly the scale the decode tests do, and a
// number printed by `pf_prep_test` means what the same number means in
// `prep_test`.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace pf_harness {

inline std::vector<float> random_f32(size_t n, uint32_t seed, float mean, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(mean, sigma);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

inline std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}

// bf16 words ordered as integers: the distance between two keys is the number
// of representable bf16 values between them (one "ulp" of the format).
inline int32_t bf16_key(uint16_t v) {
  return (v & 0x8000u) ? -int32_t(v & 0x7FFFu) : int32_t(v);
}

struct Cmp {
  uint32_t max_ulp = 0;
  size_t exact = 0, n = 0, worst = 0;
};

inline Cmp compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  Cmp c;
  c.n = ref.size();
  CHECK_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] == ref[i]) { ++c.exact; continue; }
    int32_t d = bf16_key(got[i]) - bf16_key(ref[i]);
    uint32_t u = uint32_t(d < 0 ? -d : d);
    if (u > c.max_ulp) { c.max_ulp = u; c.worst = i; }
  }
  return c;
}

inline void require(const Cmp& c, uint32_t tol, const char* what) {
  if (c.max_ulp > tol) {
    std::fprintf(stderr, "%s: max %u bf16 ulp > tol %u at index %zu (%zu/%zu exact)\n", what,
                 c.max_ulp, tol, c.worst, c.exact, c.n);
  }
  CHECK(c.max_ulp <= tol);
}

template <class T>
l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const std::vector<T>& v) {
  l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
  imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
  return m;
}

template <class T>
void download(l0::CmdList& imm, std::vector<T>& v, const l0::Mem& m) {
  imm.copy(v.data(), m.ptr(), v.size() * sizeof(T));
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // One launch of one kernel, recorded in a regular list and executed once.
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }

  // Two launches in ONE in-order list -- the shape runtime::build appends the
  // two-stage prep pair in: stage B must see stage A's `sumsq` and its
  // rewritten `resid`, and the list's in-order flag is the only thing that
  // makes it so.
  void run2(l0::Kernel& a, uint32_t agx, uint32_t agy, l0::Kernel& b, uint32_t bgx,
            uint32_t bgy) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(a, agx, agy);
    list.launch(b, bgx, bgy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
};

// The `+0.0f` padding trick these tests use to drive a decode binary compiled
// at S > 1 from a single-slice rectangle: slices 1.. are filled with `+0.0f`,
// so `Sigma_s` reduces to slice 0 EXACTLY. It is exact only while no value in
// slice 0 is `-0.0f` -- `+0.0f + -0.0f == +0.0f`, which would flip a sign of
// zero and therefore a bf16 word. Every caller asserts that first, through
// this function, rather than trusting its random fill.
inline void require_no_negative_zero(const std::vector<float>& v, const char* what) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i] == 0.0f && std::signbit(v[i])) {
      std::fprintf(stderr, "%s: -0.0f at index %zu -- the +0.0f slice padding is not exact\n",
                   what, i);
      CHECK(false);
    }
  }
}

}  // namespace pf_harness
