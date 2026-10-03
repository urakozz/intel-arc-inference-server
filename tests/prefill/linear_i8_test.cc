// runtime::prefill::linear_i8 / linear_i8_silu (spec 5 T1-T2, plan 5a Task 5):
// the h8 slab walk against fp32 x W^T at the cross test's 3 % bar, layout 1
// bitwise equal to layout 0 on the same weights, and the SiLU form bitwise
// equal to the PF_SILU_ROW chain run on the host over linear_i8's own fp32
// output. M = 300 is not a multiple of 256, and rows [300, 512) of x hold NaN
// bf16, so the padded rows are exercised too.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "pf_harness.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/kernels.h"

namespace {
using runtime::prefill::pad256;
constexpr uint32_t kK = 5120, kN = 2048, kM = 300;

uint16_t rne_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t(u >> 16);
}
float bf16f(uint16_t h) { return common::bf16_to_f32(h); }
float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }

struct Dev {
  l0::Context ctx{0};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
  runtime::PrefillScratch s{ctx, 256, model::qwen38()};
  runtime::prefill::Int8State q{ctx, 17408};
};

loader::DeviceWeight make_weight(Dev& d, const common::Int4Gptq& g, uint32_t layout) {
  const std::vector<uint32_t> words = layout == 0 ? g.qweight : g.tiled();
  loader::DeviceWeight w{l0::Mem(d.ctx, l0::MemKind::Device, words.size() * 4), nullptr,
                         model::GemvShape{g.K, g.N, 1, layout}, model::WeightKind::Int4};
  d.imm.copy(w.mem.ptr(), words.data(), words.size() * 4);
  if (layout == 0) {
    w.scales = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, g.scales.size() * 2);
    d.imm.copy(w.scales->ptr(), g.scales.data(), g.scales.size() * 2);
  }
  return w;
}

std::vector<float> partials_rows(Dev& d, uint32_t M, uint32_t N) {
  std::vector<float> p(size_t(M) * N);
  d.imm.copy(p.data(), d.s.partials.ptr(), p.size() * 4);
  return p;
}

double rel_l2(const std::vector<float>& got, const std::vector<double>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(!std::isnan(got[i]));
    num += (double(got[i]) - ref[i]) * (double(got[i]) - ref[i]);
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / den);
}

// fp32 x W^T from the dequantised int4 weights, accumulated in double.
std::vector<double> oracle(const std::vector<uint16_t>& x, const common::Int4Gptq& g, uint32_t M) {
  std::vector<float> wt(size_t(g.N) * g.K);
  for (uint32_t k = 0; k < g.K; ++k)
    for (uint32_t n = 0; n < g.N; ++n) wt[size_t(n) * g.K + k] = g.at(k, n);
  std::vector<float> xf(size_t(M) * g.K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = common::bf16_to_f32(x[i]);
  std::vector<double> y(size_t(M) * g.N);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < g.N; ++n) {
      double acc = 0.0;
      const float* xr = xf.data() + size_t(m) * g.K;
      const float* wr = wt.data() + size_t(n) * g.K;
      for (uint32_t k = 0; k < g.K; ++k) acc += double(xr[k]) * double(wr[k]);
      y[size_t(m) * g.N + n] = acc;
    }
  return y;
}
}  // namespace

int main() {
  using runtime::prefill::linear_i8;
  using runtime::prefill::linear_i8_launches;
  using runtime::prefill::linear_i8_silu;
  Dev d;
  CHECK(linear_i8_launches(model::GemvShape{5120, 2048, 1, 0}) == 5);

  // x [pad256(M)][K]: rows [0, M) random, rows [M, pad256(M)) NaN bf16.
  const uint32_t Mp = pad256(kM);
  std::vector<uint16_t> x(size_t(Mp) * kK, 0x7FC0u);
  {
    const std::vector<uint16_t> r = pf_harness::random_bf16(size_t(kM) * kK, 67, -1.0f, 1.0f);
    std::memcpy(x.data(), r.data(), r.size() * 2);
  }
  l0::Mem dx(d.ctx, l0::MemKind::Device, x.size() * 2);
  d.imm.copy(dx.ptr(), x.data(), x.size() * 2);

  const common::Int4Gptq g = common::Int4Gptq::random(kK, kN, 71);
  const std::vector<double> ref = oracle(x, g, kM);
  const loader::DeviceWeight w0 = make_weight(d, g, 0);
  const loader::DeviceWeight w1 = make_weight(d, g, 1);
  d.q.scales(d.cx, d.kc, w0);   // plan 5b's order: scales before the first chunk
  d.q.scales(d.cx, d.kc, w1);

  d.cx.reset_launches();
  linear_i8(d.cx, d.kc, d.s, d.q, w0, dx.as<uint16_t>(), kM);
  d.cx.wait();
  CHECK(d.cx.launches() == linear_i8_launches(w0.shape));
  const std::vector<float> y0 = partials_rows(d, kM, kN);
  const double r0 = rel_l2(y0, ref);
  std::printf("linear_i8 K=%u N=%u M=%u L0: rel L2 %.4f %%\n", kK, kN, kM, 100.0 * r0);
  CHECK(r0 <= 0.03);

  linear_i8(d.cx, d.kc, d.s, d.q, w1, dx.as<uint16_t>(), kM);
  d.cx.wait();
  const std::vector<float> y1 = partials_rows(d, kM, kN);
  const double r1 = rel_l2(y1, ref);
  size_t diff = 0;
  for (size_t i = 0; i < y0.size(); ++i) diff += std::memcmp(&y0[i], &y1[i], 4) != 0;
  std::printf("linear_i8 K=%u N=%u M=%u L1: rel L2 %.4f %%, %zu words differ from L0\n", kK, kN, kM,
              100.0 * r1, diff);
  CHECK(diff == 0);
  std::printf("layout 1 == layout 0 bitwise\n");

  // gate||up-shaped: the same N = 2048 read as 16-column gate/up interleave.
  const common::Int4Gptq gu = common::Int4Gptq::random(kK, kN, 73);
  const loader::DeviceWeight wg = make_weight(d, gu, 0);
  d.q.scales(d.cx, d.kc, wg);
  linear_i8(d.cx, d.kc, d.s, d.q, wg, dx.as<uint16_t>(), kM);
  d.cx.wait();
  const std::vector<float> yg = partials_rows(d, kM, kN);
  const uint32_t ldx = kN / 2;
  l0::Mem dout(d.ctx, l0::MemKind::Device, size_t(Mp) * ldx * 2);
  d.cx.reset_launches();
  linear_i8_silu(d.cx, d.kc, d.s, d.q, wg, dx.as<uint16_t>(), kM, dout.as<uint16_t>(), ldx);
  d.cx.wait();
  CHECK(d.cx.launches() == linear_i8_launches(wg.shape));
  std::vector<uint16_t> got(size_t(kM) * ldx);
  d.imm.copy(got.data(), dout.ptr(), got.size() * 2);
  size_t bad = 0;
  for (uint32_t m = 0; m < kM; ++m)
    for (uint32_t n = 0; n < kN; ++n) {
      if ((n / 16) % 2 != 0) continue;   // gate columns; up is n + 16
      const float gv = yg[size_t(m) * kN + n], uv = yg[size_t(m) * kN + n + 16];
      const uint16_t want =
          rne_bf16(bf16f(rne_bf16(silu_f32(bf16f(rne_bf16(gv))))) * bf16f(rne_bf16(uv)));
      bad += (want != got[size_t(m) * ldx + (n / 32) * 16 + n % 16]);
    }
  std::printf("linear_i8_silu K=%u N=%u M=%u: %zu mismatches against the host chain over "
              "linear_i8's fp32\n",
              kK, kN, kM, bad);
  CHECK(bad == 0);
  std::printf("Int8State bytes %zu\n", d.q.bytes());
  std::printf("linear_i8_test: PASS\n");
  return 0;
}
