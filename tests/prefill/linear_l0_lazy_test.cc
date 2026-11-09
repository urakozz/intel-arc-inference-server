// 3g: real fused calls must skip unused slab/module setup, retain input validation,
// and still enter the original slab walk on a later larger call. No checkpoint.
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "common/int4.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "pf_harness.h"
#include "runtime/buffers.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/linear_l0.h"

namespace {
using namespace runtime::prefill;
constexpr uint32_t kK = 256, kN = 2048, kCap = 2048, kNs = 1024;
constexpr size_t kSlabBytes = 35651584;  // Qwen3.8's widest K=17408, not this fixture's K.
bool corrupt = false;

loader::DeviceWeight weight(l0::Context& ctx, l0::CmdList& imm, uint32_t layout) {
  const auto g = common::Int4Gptq::random(kK, kN, 830);
  const auto words = layout == 0 ? g.qweight : g.tiled();
  loader::DeviceWeight w{l0::Mem(ctx, l0::MemKind::Device, words.size() * 4), nullptr,
                         model::GemvShape{kK, kN, 1, layout}, model::WeightKind::Int4};
  imm.copy(w.mem.ptr(), words.data(), words.size() * 4);
  if (layout == 0) {
    w.scales = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, g.scales.size() * 2);
    imm.copy(w.scales->ptr(), g.scales.data(), g.scales.size() * 2);
  }
  return w;
}

struct Fixture {
  l0::Context ctx{0};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  Context cx{ctx};
  KernelCache kc{ctx}, reference{ctx};
  runtime::PrefillScratch s{ctx, 4096, model::qwen38(), kCap};
  loader::DeviceWeight w;
  l0::Mem input{ctx, l0::MemKind::Device, size_t(kCap) * kK * 2};
  l0::Mem slab{ctx, l0::MemKind::Device, size_t(kK) * kNs * 2};
  bool silu;

  Fixture(uint32_t layout, bool gated) : w(weight(ctx, imm, layout)), silu(gated) {
    const auto x = pf_harness::random_bf16(size_t(kCap) * kK, 831, -1.0f, 1.0f);
    imm.copy(input.ptr(), x.data(), x.size() * 2);
  }
  void* out() { return silu ? s.x.ptr() : s.partials.ptr(); }
  size_t bytes(uint32_t M) const { return size_t(M) * kN * (silu ? 1 : 4); }
  void run(uint32_t M) {
    imm.fill(out(), 0xA5A5A5A5u, bytes(pad256(M)));
    if (silu)
      linear_l0_silu(cx, kc, s, w, input.as<uint16_t>(), M, s.x.as<uint16_t>(), kN / 2);
    else
      linear_l0(cx, kc, s, w, input.as<uint16_t>(), M);
    cx.wait();
  }
  void bitwise(uint32_t M) {
    std::vector<uint8_t> got(bytes(M)), want(bytes(M));
    imm.copy(got.data(), out(), got.size());
    imm.fill(out(), 0x5A5A5A5Au, bytes(pad256(M)));
    auto& dq = reference(kernels::pf_dequant_slab_variant(kK, kN, w.shape.layout), "pf_dequant_slab");
    const void* scales = w.scales ? w.scales->ptr() : nullptr;
    for (uint32_t n0 = 0; n0 < kN; n0 += kNs) {
      cx.launch(dq, kNs / 16, kK / 64, 1,
                {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
      const GemmBatch b{M, kK, kNs, 1, kK, kNs, kN, 0, 0, 0};
      if (silu)
        gemm_l0_silu(cx, reference, b, input.as<uint16_t>(), slab.as<uint16_t>(),
                     s.x.as<uint16_t>() + n0 / 2, kN / 2);
      else
        gemm_l0(cx, reference, b, input.as<uint16_t>(), slab.as<uint16_t>(),
                s.partials.as<float>() + n0, false);
    }
    cx.wait();
    imm.copy(want.data(), out(), want.size());
    if (corrupt) got[0] ^= 1;
    CHECK(std::memcmp(got.data(), want.data(), got.size()) == 0);
    std::printf("  %s L%u M=%u: all %zu output bytes bitwise\n",
                silu ? "SiLU" : "plain", w.shape.layout, M, got.size());
  }
};

void group(uint32_t layout, bool silu) {
  Fixture f(layout, silu);
  struct Case { uint32_t rows; size_t modules; };
  const std::vector<Case> cases = silu ? std::vector<Case>{{8, 1}, {9, 2}, {192, 2}, {193, 3}, {320, 3}}
                                     : std::vector<Case>{{1, 1}, {32, 1}, {33, 2}, {384, 2}};
  for (const auto& c : cases) {
    const size_t launches = f.cx.launches();
    f.run(c.rows);
    CHECK_EQ(f.s.lazy_bytes(), size_t(0));
    CHECK_EQ(f.kc.modules(), c.modules);
    CHECK_EQ(f.cx.launches() - launches, size_t(1));
    f.bitwise(c.rows);
  }
  const uint32_t larger = silu ? 321 : 385;
  f.run(larger);
  CHECK_EQ(f.s.lazy_bytes(), kSlabBytes);
  void* slab = f.s.slab_buffer().ptr();
  const size_t modules = f.kc.modules();
  f.bitwise(larger);
  for (uint32_t M : {cases.front().rows, larger}) {
    f.run(M);
    CHECK_EQ(f.s.lazy_bytes(), kSlabBytes);
    CHECK(f.s.slab_buffer().ptr() == slab);
    CHECK_EQ(f.kc.modules(), modules);
    f.bitwise(M);
  }
  std::puts("  short -> slab -> short -> slab: allocation and module reuse PASS");
}

void invalid_inputs() {
  Fixture f(0, false);
  auto rejects = [&](const char* name, const std::function<void()>& call) {
    bool rejected = false;
    try { call(); } catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    CHECK_EQ(f.s.lazy_bytes(), size_t(0));
    CHECK_EQ(f.kc.modules(), size_t(0));
    CHECK_EQ(f.cx.launches(), size_t(0));
    std::printf("  rejects %s before allocation/module/launch\n", name);
  };
  auto plain = [&](uint32_t M) {
    linear_l0(f.cx, f.kc, f.s, f.w, f.input.as<uint16_t>(), M);
  };
  f.w.kind = model::WeightKind::Bf16;
  rejects("weight kind", [&] { plain(1); });
  f.w.kind = model::WeightKind::Int4;
  for (const auto& bad : std::vector<model::GemvShape>{{kK, 2000, 1, 0}, {255, kN, 1, 0}, {kK, kN, 1, 2}}) {
    f.w.shape = bad;
    rejects("weight shape/layout", [&] { plain(1); });
  }
  f.w.shape = {kK, kN, 1, 0};
  auto scales = std::move(f.w.scales);
  rejects("missing scales", [&] { plain(1); });
  f.w.scales = std::move(scales);
  rejects("zero rows", [&] { plain(0); });
  rejects("rows beyond scratch cap", [&] {
    linear_l0(f.cx, f.kc, f.s, f.w, f.input.as<uint16_t>(), 2049);
  });
  l0::Mem small(f.ctx, l0::MemKind::Device, 64);
  f.s.partials.swap(small);
  rejects("undersized partials", [&] { plain(1); });
  f.s.partials.swap(small);
  rejects("SiLU output pitch", [&] {
    linear_l0_silu(f.cx, f.kc, f.s, f.w, f.input.as<uint16_t>(), 193, f.s.x.as<uint16_t>(), 1008);
  });
}
}  // namespace

int main(int argc, char** argv) {
  corrupt = argc == 2 && std::strcmp(argv[1], "--negative") == 0;
  CHECK(argc == 1 || corrupt);
  group(0, false);
  group(1, false);
  group(0, true);
  invalid_inputs();
  std::puts("linear_l0_lazy_test PASS");
}
