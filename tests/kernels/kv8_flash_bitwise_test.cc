// Item 3e: the production flash against its frozen int8 control, every output byte.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "l0/event.h"
#include "prefill/pf_harness.h"

namespace {

constexpr uint32_t kHD = 256, kQH = 24, kKVH = 4;

void compare(pf_harness::Dev& d, l0::Kernel& fast, l0::Kernel& control,
             l0::Kernel& gate, uint32_t pos, uint32_t C, bool time, uint32_t head_groups,
             l0::Kernel* widen) {
  const bool raw = widen != nullptr;
  const uint32_t depth = pos + C, rows = (C + 255u) & ~255u;
  std::mt19937 rng(pos + C);
  std::vector<int8_t> k(size_t(depth) * kKVH * kHD), v(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = int8_t(int(rng() % 255) - 127);
    v[i] = int8_t(int(rng() % 255) - 127);
  }
  std::vector<uint16_t> ks(size_t(depth) * kKVH), vs(ks.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    ks[i] = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
    vs[i] = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
    if (i % 19 == 0) ks[i] = vs[i] = 0;
    if (i % 23 == 0) ks[i] = vs[i] = 1;   // smallest subnormal fp16 scale
  }
  const auto q = pf_harness::random_bf16(size_t(C) * kQH * kHD, pos, -2.0f, 2.0f);
  const auto part = pf_harness::random_f32(size_t(C) * 14336, pos + 1, 0.0f, 1.5f);
  auto dk = pf_harness::upload(d.ctx, d.imm, k);
  auto dv = pf_harness::upload(d.ctx, d.imm, v);
  std::unique_ptr<l0::Mem> dkraw, dvraw;
  if (raw) {
    std::vector<uint16_t> kr(k.size() + 16, 0xA55A), vr(v.size() + 16, 0xA55A);
    for (size_t i = 0; i < k.size(); ++i) {
      kr[i] = common::f32_to_bf16(float(k[i]));
      vr[i] = common::f32_to_bf16(float(v[i]));
    }
    dkraw = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, kr.size() * 2);
    dvraw = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, vr.size() * 2);
    d.imm.fill(dkraw->ptr(), 0xA55AA55Au, dkraw->size());
    d.imm.fill(dvraw->ptr(), 0xA55AA55Au, dvraw->size());
    widen->arg_ptr(0, dk.ptr());
    widen->arg_ptr(1, dv.ptr());
    widen->arg_ptr(2, dkraw->ptr());
    widen->arg_ptr(3, dvraw->ptr());
    widen->arg(4, uint32_t(k.size()));
    d.run(*widen, (uint32_t(k.size()) + 4095u) / 4096u, 1);
    std::vector<uint16_t> expanded(kr.size());
    pf_harness::download(d.imm, expanded, *dkraw);
    CHECK(std::memcmp(expanded.data(), kr.data(), dkraw->size()) == 0);
    pf_harness::download(d.imm, expanded, *dvraw);
    CHECK(std::memcmp(expanded.data(), vr.data(), dvraw->size()) == 0);
  }
  auto dks = pf_harness::upload(d.ctx, d.imm, ks);
  auto dvs = pf_harness::upload(d.ctx, d.imm, vs);
  auto dq = pf_harness::upload(d.ctx, d.imm, q);
  auto dp = pf_harness::upload(d.ctx, d.imm, part);
  const size_t count = size_t(kQH) * rows * kHD;
  l0::Mem o(d.ctx, l0::MemKind::Device, count * sizeof(float));
  l0::Mem out(d.ctx, l0::MemKind::Device, q.size() * sizeof(uint16_t));
  std::vector<float> got(count), ref(count);
  std::vector<uint16_t> gated(q.size()), gated_ref(q.size());
  const uint32_t stride = rows * kHD;
  gate.arg_ptr(0, o.ptr());
  gate.arg_ptr(1, dp.ptr());
  gate.arg_ptr(2, out.ptr());
  gate.arg(3, stride);
  auto run = [&](l0::Kernel& kernel, std::vector<float>& result,
                 std::vector<uint16_t>& gated_result) {
    d.imm.fill(o.ptr(), 0x3F123456u, o.size());
    kernel.arg_ptr(0, dq.ptr());
    kernel.arg_ptr(1, &kernel == &fast && raw ? dkraw->ptr() : dk.ptr());
    kernel.arg_ptr(2, dks.ptr());
    kernel.arg_ptr(3, &kernel == &fast && raw ? dvraw->ptr() : dv.ptr());
    kernel.arg_ptr(4, dvs.ptr());
    kernel.arg_ptr(5, o.ptr());
    kernel.arg(6, pos);
    kernel.arg(7, C);
    kernel.arg(8, rows);
    auto list = l0::CmdList::regular(d.ctx);
    list.launch(kernel, (C + 7u) / 8u, kKVH, &kernel == &fast ? head_groups : 1);
    list.launch(gate, kQH, C);
    list.close();
    d.q.execute(list, &d.fence);
    d.fence.wait();
    pf_harness::download(d.imm, result, o);
    pf_harness::download(d.imm, gated_result, out);
  };
  run(control, ref, gated_ref);
  run(fast, got, gated);
  const bool same = std::memcmp(got.data(), ref.data(), o.size()) == 0;
  const bool gate_same = std::memcmp(gated.data(), gated_ref.data(), out.size()) == 0;
  size_t bad = 0;
  for (size_t i = 0; i < count; ++i) {
    bad += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
    CHECK(std::isfinite(got[i]));
  }
  std::printf("pos %6u C %4u: %zu fp32 words, %zu differ; %zu gated words, bitwise %d\n",
              pos, C, count, bad, gated.size(), gate_same);
  CHECK(same);
  CHECK(gate_same);
  if (time) {
    l0::EventPool pool(d.ctx, 1);
    l0::Event event(pool, 0);
    auto list = l0::CmdList::regular(d.ctx);
    list.launch(fast, (C + 7u) / 8u, kKVH, head_groups, &event);
    list.close();
    for (uint32_t i = 0; i < 4; ++i) {
      event.reset();
      d.q.execute(list, &d.fence);
      d.fence.wait();
      if (i != 0)
        std::printf("  time pos %6u C %4u run %u: %.3f us\n", pos, C, i, event.duration_us());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc <= 4);
  bool time = false, raw = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--time") time = true;
    else if (std::string(argv[i]) == "--raw") raw = true;
    else CHECK(false);
  }
  pf_harness::Dev d;
  l0::Module fm(d.ctx, argc > 1 ? argv[1] : kernels::path("pf_flash_attn_kv8"));
  l0::Module cm(d.ctx, kernels::path("kv8_flash_control"));
  l0::Module gm(d.ctx, kernels::path("pf_attn_gate_kv8"));
  auto fast = fm.kernel("pf_flash_attn_kv8");
  auto control = cm.kernel("pf_flash_attn_kv8");
  auto gate = gm.kernel("pf_attn_gate_kv8");
  std::unique_ptr<l0::Kernel> widen;
  if (raw) {
    widen = std::make_unique<l0::Kernel>(fm.kernel("kv8_widen"));
    widen->group_size(256);
  }
  ze_kernel_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  ZE_CHECK(zeKernelGetProperties(fast.handle(), &props));
  // Probe binaries keep RPW=8 and vary how many of the six heads share a group.
  const uint32_t hpw = props.requiredGroupSizeX / 16;
  CHECK(hpw > 0 && props.requiredGroupSizeX == hpw * 16 && 6 % hpw == 0);
  fast.group_size(props.requiredGroupSizeX);
  std::printf("kernel: HPW %u, private %u B, spill %u B\n", hpw, props.privateMemSize, props.spillMemSize);
  control.group_size(96);
  gate.group_size(256);
  for (const auto& shape : {std::pair{0u, 1u}, {0u, 9u}, {0u, 64u}, {777u, 300u},
                            {2000u, 2048u}, {13209u, 2048u}, {30963u, 37u},
                            {64991u, 9u}, {127743u, 257u}, {28952u, 2048u},
                            {62952u, 2048u}, {125952u, 2048u}, {0u, 4096u}})
    compare(d, fast, control, gate, shape.first, shape.second, time, 6 / hpw, widen.get());
  std::puts("kv8_flash_bitwise_test: PASS");
}
