// Native fp16 widening must preserve every finite scale's original fp32 bits.
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "prefill/pf_harness.h"

int main() {
  std::vector<uint16_t> scales;
  std::vector<float> expected;
  for (uint32_t h = 0; h < 65536; ++h) {
    if ((h & 0x7C00u) == 0x7C00u) continue;   // the quantiser never stores inf / NaN
    scales.push_back(uint16_t(h));
    expected.push_back(common::kv8::f16f(uint16_t(h)));
  }
  pf_harness::Dev d;
  auto input = pf_harness::upload(d.ctx, d.imm, scales);
  l0::Mem output(d.ctx, l0::MemKind::Device, (scales.size() + 16) * sizeof(float));
  d.imm.fill(output.ptr(), 0x3F123456u, output.size());
  l0::Module module(d.ctx, kernels::path("kv8_scale_test"));
  auto kernel = module.kernel("kv8_scale_test");
  kernel.group_size(256);
  kernel.arg_ptr(0, input.ptr());
  kernel.arg_ptr(1, output.ptr());
  kernel.arg(2, uint32_t(scales.size()));
  d.run(kernel, (uint32_t(scales.size()) + 255u) / 256u, 1);
  std::vector<float> got(scales.size() + 16);
  pf_harness::download(d.imm, got, output);
  CHECK(std::memcmp(got.data(), expected.data(), expected.size() * sizeof(float)) == 0);
  for (size_t i = scales.size(); i < got.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &got[i], sizeof(bits));
    CHECK_EQ(bits, 0x3F123456u);
  }
  std::printf("kv8_scale_test: %zu finite fp16 codes bitwise identical, tail guards intact\n", scales.size());
}
