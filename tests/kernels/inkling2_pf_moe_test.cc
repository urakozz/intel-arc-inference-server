#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "kernels/inkling2_kernels.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "loader/expert_fp.h"

namespace {
namespace ink = kernels::inkling2;
using D = ink::D;
using common::ExpertFpFormat;
constexpr uint32_t rows = 64;

loader::ExpertFpMatrix matrix(ExpertFpFormat f, uint32_t K, uint32_t N) {
  const bool fp4 = f == ExpertFpFormat::Nvfp4;
  const auto bits = common::expert_fp_code_bits(f);
  const auto tile_words = common::expert_fp_tile_words(f);
  loader::ExpertFpMatrix w{f, K, N, std::vector<uint32_t>(size_t(N / 16) * (K / 64) * tile_words + fp4)};
  // Two nonzero columns in distinct tiles, with different block scales.
  // The dot can be evaluated exactly in either DPAS or scalar order.
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t k : {0u, 65u}) {
      auto* tile = w.words.data() + (size_t(n / 16) * (K / 64) + k / 64) * tile_words;
      const auto lane = n % 16;
      const auto code = fp4 ? 1u + n % 15u : (n & 1u ? 0xa0u : 0x20u) + n % 16u;
      const auto scale = fp4 ? (k == 0 ? 0x38u : 0x30u) : (k == 0 ? 128u : 126u);
      tile[lane] |= code << ((k % 64) * bits);
      tile[32 * bits + lane / 4] |= scale << (8 * (lane % 4));
    }
  }
  if (fp4) w.words.back() = std::bit_cast<uint32_t>(.375f);
  return w;
}

float dot(const loader::ExpertFpMatrix& w, uint32_t n, const uint16_t* x) {
  // DPAS starts from positive zero, including when all products are -0.
  float sum = 0.f;
  for (uint32_t k : {0u, 65u})
    sum += common::bf16_to_f32(w.block_bf16(n, k)) * common::bf16_to_f32(x[k]);
  return sum * w.outer_scale();
}
}  // namespace

int main() {
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate(ctx);
  const auto upload = [&](const auto& v) {
    l0::Mem mem(ctx, l0::MemKind::Device, v.size() * sizeof(v[0]));
    list.copy(mem.ptr(), v.data(), mem.size());
    return mem;
  };
  std::vector<uint16_t> x(rows * D::hidden), h(rows * D::inter);
  for (uint32_t r = 0; r < rows; ++r) {
    x[r * D::hidden] = common::f32_to_bf16(float(r % 13 + 1));
    x[r * D::hidden + 65] = common::f32_to_bf16(float(r % 7 + 1) / 16.f);
  }
  std::vector<float> y(rows * D::hidden);
  auto input = upload(x), hidden = upload(h), output = upload(y);
  const auto items = upload(std::array<uint32_t, 8>{0, 0, 32, 0, 0, 32, 32, 0});
  const auto gemm = [&](const std::string& name, const void* a, const l0::Mem& w, void* out) {
    l0::Module module(ctx, kernels::path(name));
    auto k = module.kernel("pf_moe_q4gemm");
    k.group_size(ink::kPrefillExpertGroup);
    k.arg_ptr(0, items.ptr());
    k.arg_ptr(1, a);
    k.arg_ptr(2, w.ptr());
    k.arg_ptr(3, nullptr);
    k.arg_ptr(4, out);
    list.launch(k, D::hidden / 128, 2);
  };
  for (auto format : {ExpertFpFormat::Nvfp4, ExpertFpFormat::Mxfp8}) {
    const auto gu = matrix(format, D::hidden, 2 * D::inter);
    const auto dn = matrix(format, D::inter, D::hidden);
    auto wgu = upload(gu.words), wdn = upload(dn.words);
    gemm(ink::prefill_expert(format, true), input.ptr(), wgu, output.ptr());
    {
      l0::Module module(ctx, kernels::path(std::string(ink::kPrefillMoe)));
      auto k = module.kernel("ink_prefill_swiglu");
      k.group_size(96);
      k.arg_ptr(0, output.ptr());
      k.arg_ptr(1, hidden.ptr());
      list.launch(k, D::inter / (4 * 96), rows);
    }
    gemm(ink::prefill_expert(format, false), hidden.ptr(), wdn, output.ptr());
    list.copy(h.data(), hidden.ptr(), hidden.size());
    list.copy(y.data(), output.ptr(), output.size());
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t i = 0; i < D::inter; ++i) {
        const uint32_t n = i / 16 * 32 + i % 16;
        const float g = dot(gu, n, x.data() + r * D::hidden);
        const float u = dot(gu, n + 16, x.data() + r * D::hidden);
        const float capped_g = std::tanh(g / D::gate_cap) * D::gate_cap;
        const float capped_u = std::tanh(u / D::up_cap) * D::up_cap;
        const auto expected = common::f32_to_bf16(capped_g * (1.f / (1.f + std::exp(-g))) * capped_u);
        const auto actual = h[r * D::inter + i];
        if (actual != expected)
          std::fprintf(stderr, "format %u row %u column %u g %.9g u %.9g actual %04x expected %04x\n",
                       unsigned(format), r, i, g, u, unsigned(actual), unsigned(expected));
        CHECK_EQ(actual, expected);
      }
      for (uint32_t i = 0; i < D::hidden; ++i)
        CHECK_EQ(std::bit_cast<uint32_t>(y[r * D::hidden + i]),
                 std::bit_cast<uint32_t>(dot(dn, i, h.data() + r * D::inter)));
    }
  }

  std::array<float, 3 * D::experts> logits{};
  std::array<float, D::experts> bias{};
  for (uint32_t i = 0; i < logits.size(); ++i) logits[i] = float(int(i * 13 % 51) - 25) / 8.f;
  for (uint32_t i = 0; i < bias.size(); ++i) bias[i] = float(i % 9) / 32.f;
  auto ld = upload(logits), bd = upload(bias);
  std::array<std::array<uint32_t, 32>, 3> route{};
  auto rd = upload(route);
  l0::Module module(ctx, kernels::path(ink::route()));
  auto k = module.kernel("kol_route");
  k.group_size(128);
  k.arg_ptr(0, ld.ptr());
  k.arg_ptr(1, bd.ptr());
  k.arg_ptr(2, rd.ptr());
  list.launch(k, 1, 3);
  list.copy(route.data(), rd.ptr(), rd.size());
  k.arg_ptr(0, static_cast<float*>(ld.ptr()) + D::experts);
  list.launch(k, 1, 1);
  std::array<uint32_t, 32> single{};
  list.copy(single.data(), rd.ptr(), sizeof(single));
  for (uint32_t i = 0; i < 2 * D::top_k; ++i) CHECK_EQ(route[1][i], single[i]);
  std::puts("Inkling prompt expert GEMMs match the decoded-weight twin and routing matches decode");
}
