// Item 3e: the production flash against its frozen int8 control, every output byte.
// 3h step 0 (--chunk-time): short chunks deep in a session, the int8 in-place flash versus
// the widening pair (kv8_widen over pos + C rows, then the raw flash) and the frozen control.
#include <algorithm>
#include <chrono>
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
  // 3j: --raw widens into ONE buffer laid out as the engine's PrefillScratch::kv8_raw_buffer - K's rows, then V's
  // at k + max_len x 4 x 256 (attn_chunk_kv8) - at a max_len past the rows used, so the V offset is this test's to
  // catch: K and V exact, the gap between them and the tail untouched, and the flash below reads both from there.
  std::unique_ptr<l0::Mem> draw;
  uint16_t *kraw = nullptr, *vraw = nullptr;
  if (raw) {
    const uint32_t max_len = ((depth + 255u) & ~255u) + 256u;   // > depth: a nonzero gap before V
    const size_t plane = size_t(max_len) * kKVH * kHD;
    std::vector<uint16_t> want(2 * plane + 16, 0xA55A);
    for (size_t i = 0; i < k.size(); ++i) {
      want[i] = common::f32_to_bf16(float(k[i]));
      want[plane + i] = common::f32_to_bf16(float(v[i]));
    }
    draw = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, want.size() * 2);
    d.imm.fill(draw->ptr(), 0xA55AA55Au, draw->size());
    kraw = draw->as<uint16_t>();
    vraw = kraw + plane;
    widen->arg_ptr(0, dk.ptr());
    widen->arg_ptr(1, dv.ptr());
    widen->arg_ptr(2, kraw);
    widen->arg_ptr(3, vraw);
    widen->arg(4, uint32_t(k.size()));
    d.run(*widen, (uint32_t(k.size()) + 4095u) / 4096u, 1);
    std::vector<uint16_t> expanded(want.size());
    pf_harness::download(d.imm, expanded, *draw);
    CHECK(std::memcmp(expanded.data(), want.data(), draw->size()) == 0);
    size_t v_nonzero = 0;
    for (size_t i = 0; i < k.size(); ++i) v_nonzero += expanded[plane + i] != 0;
    CHECK(v_nonzero > 0);
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
    kernel.arg_ptr(1, &kernel == &fast && raw ? static_cast<void*>(kraw) : dk.ptr());
    kernel.arg_ptr(2, dks.ptr());
    kernel.arg_ptr(3, &kernel == &fast && raw ? static_cast<void*>(vraw) : dv.ptr());
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

// 3h step 0: one FA layer's attention for a C-row chunk at `pos`, as each path issues it,
// sixteen times in one list (Qwen3.8's FA layers). Arms interleave in rotating order after
// a warm-up; the medians and the paired ratios against the widening pair are printed.
void chunk_time(pf_harness::Dev& d, l0::Kernel& inplace, l0::Kernel& raw, l0::Kernel& widen,
                l0::Kernel& control, uint32_t pos, const std::vector<uint32_t>& chunks) {
  using Clock = std::chrono::steady_clock;
  const uint32_t cmax = *std::max_element(chunks.begin(), chunks.end());
  const uint32_t depth_max = pos + cmax, max_len = ((depth_max + 255u) & ~255u) + 256u;
  std::mt19937 rng(pos);
  std::vector<int8_t> k(size_t(depth_max) * kKVH * kHD), v(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = int8_t(int(rng() % 255) - 127);
    v[i] = int8_t(int(rng() % 255) - 127);
  }
  std::vector<uint16_t> ks(size_t(depth_max) * kKVH), vs(ks.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    ks[i] = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
    vs[i] = common::kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
  }
  const auto q = pf_harness::random_bf16(size_t(cmax) * kQH * kHD, pos, -2.0f, 2.0f);
  auto dk = pf_harness::upload(d.ctx, d.imm, k);
  auto dv = pf_harness::upload(d.ctx, d.imm, v);
  auto dks = pf_harness::upload(d.ctx, d.imm, ks);
  auto dvs = pf_harness::upload(d.ctx, d.imm, vs);
  auto dq = pf_harness::upload(d.ctx, d.imm, q);
  const size_t plane = size_t(max_len) * kKVH * kHD;
  l0::Mem draw(d.ctx, l0::MemKind::Device, 2 * plane * sizeof(uint16_t));
  uint16_t* kraw = draw.as<uint16_t>();
  uint16_t* vraw = kraw + plane;
  const uint32_t rows_max = (cmax + 255u) & ~255u;
  l0::Mem o(d.ctx, l0::MemKind::Device, size_t(kQH) * rows_max * kHD * sizeof(float));
  enum { kControl, kInplace, kWiden, kArms };
  const char* names[kArms] = {"control", "inplace", "widen+raw"};
  for (const uint32_t C : chunks) {
    const uint32_t depth = pos + C, rows = (C + 255u) & ~255u;
    const uint32_t count = depth * kKVH * kHD;
    auto bind = [&](l0::Kernel& kernel, bool is_raw) {
      kernel.arg_ptr(0, dq.ptr());
      kernel.arg_ptr(1, is_raw ? static_cast<void*>(kraw) : dk.ptr());
      kernel.arg_ptr(2, dks.ptr());
      kernel.arg_ptr(3, is_raw ? static_cast<void*>(vraw) : dv.ptr());
      kernel.arg_ptr(4, dvs.ptr());
      kernel.arg_ptr(5, o.ptr());
      kernel.arg(6, pos);
      kernel.arg(7, C);
      kernel.arg(8, rows);
    };
    bind(control, false);
    bind(inplace, false);
    bind(raw, true);
    widen.arg_ptr(0, dk.ptr());
    widen.arg_ptr(1, dv.ptr());
    widen.arg_ptr(2, kraw);
    widen.arg_ptr(3, vraw);
    widen.arg(4, count);
    std::vector<l0::CmdList> lists;
    for (int a = 0; a < kArms; ++a) {
      lists.push_back(l0::CmdList::regular(d.ctx));
      for (uint32_t layer = 0; layer < 16; ++layer) {
        if (a == kWiden) lists.back().launch(widen, (count + 4095u) / 4096u, 1, 1);
        lists.back().launch(a == kControl ? control : a == kInplace ? inplace : raw, (C + 7u) / 8u, kKVH, 1);
      }
      lists.back().close();
    }
    // The three paths' pf_o, bitwise, on this shape.
    std::vector<float> ref(size_t(kQH) * rows * kHD), got(ref.size());
    for (int a = 0; a < kArms; ++a) {
      d.imm.fill(o.ptr(), 0x3F123456u, o.size());
      d.q.execute(lists[a], &d.fence);
      d.fence.wait();
      pf_harness::download(d.imm, a == kControl ? ref : got, o);
      if (a != kControl) CHECK(std::memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) == 0);
    }
    auto once = [&](int a) {
      const auto t0 = Clock::now();
      d.q.execute(lists[a], &d.fence);
      d.fence.wait();
      return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    for (int w = 0; w < 3; ++w)
      for (int a = 0; a < kArms; ++a) once(a);
    std::vector<double> ms[kArms], vs_widen[kArms];
    for (uint32_t round = 0; round < 9; ++round) {
      double t[kArms];
      for (int i = 0; i < kArms; ++i) {
        const int a = int((round + i) % kArms);
        t[a] = once(a);
      }
      for (int a = 0; a < kArms; ++a) {
        ms[a].push_back(t[a]);
        vs_widen[a].push_back(t[a] / t[kWiden]);
      }
    }
    for (int a = 0; a < kArms; ++a) {
      std::sort(ms[a].begin(), ms[a].end());
      std::sort(vs_widen[a].begin(), vs_widen[a].end());
      std::printf("chunk-time pos %u C %4u %-9s 16 layers %9.3f ms  vs widen+raw %.4f [%.4f, %.4f]\n", pos, C,
                  names[a], ms[a][4], vs_widen[a][4], vs_widen[a].front(), vs_widen[a].back());
    }
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--chunk-time") {
    // --chunk-time [pos]: the production in-place and raw binaries against the control.
    const uint32_t pos = argc > 2 ? uint32_t(std::stoul(argv[2])) : 128000u;
    pf_harness::Dev d;
    l0::Module im(d.ctx, kernels::path("pf_flash_attn_kv8"));
    l0::Module rm(d.ctx, kernels::path("pf_flash_attn_kv8_raw"));
    l0::Module cm(d.ctx, kernels::path("kv8_flash_control"));
    auto inplace = im.kernel("pf_flash_attn_kv8");
    auto raw = rm.kernel("pf_flash_attn_kv8");
    auto widen = rm.kernel("kv8_widen");
    auto control = cm.kernel("pf_flash_attn_kv8");
    inplace.group_size(96);
    raw.group_size(96);
    control.group_size(96);
    widen.group_size(256);
    chunk_time(d, inplace, raw, widen, control, pos,
               {1, 8, 16, 32, 64, 128, 300, 512, 1024, 2048});
    std::puts("kv8_flash_bitwise_test --chunk-time: done");
    return 0;
  }
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
