// 3f: full partial/output words against the frozen reader, including batched tails.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "prefill/pf_harness.h"
#include "runtime/batch_rows.h"
#include "runtime/control.h"

namespace {
constexpr uint32_t kCapacity = 160000, kRow = 6144, kHeads = 24, kBlocks = 32;
using Clock = std::chrono::steady_clock;
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
struct Inputs {
  l0::Mem k, v, ks, vs, q, gate, control, rows;
  Inputs(pf_harness::Dev& d)
      : k(d.ctx, l0::MemKind::Device, size_t(kCapacity) * 1024),
        v(d.ctx, l0::MemKind::Device, k.size()),
        ks(d.ctx, l0::MemKind::Device, size_t(kCapacity) * 4 * 2),
        vs(d.ctx, l0::MemKind::Device, ks.size()),
        q(d.ctx, l0::MemKind::Device, size_t(8) * kRow * 4),
        gate(d.ctx, l0::MemKind::Device, q.size()),
        control(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)),
        rows(d.ctx, l0::MemKind::Shared, sizeof(runtime::BatchRowTable)) {
    std::vector<int8_t> data(k.size());
    for (size_t i = 0; i < data.size(); ++i) data[i] = int8_t((i * 37 + i / 1024) % 255 - 127);
    d.imm.copy(k.ptr(), data.data(), data.size());
    for (size_t i = 0; i < data.size(); ++i) data[i] = int8_t((i * 13 + i / 256) % 255 - 127);
    d.imm.copy(v.ptr(), data.data(), data.size());
    std::vector<uint16_t> scales(size_t(kCapacity) * 4);
    const uint16_t special[] = {0, 0x8000, 1, 0x3FF, 0x400, 0x7BFF, 0x2000, 0x2400};
    for (size_t i = 0; i < scales.size(); ++i)
      scales[i] = i % 257 == 0 ? special[(i / 257) % 8] : uint16_t(0x2000 + i % 512);
    d.imm.copy(ks.ptr(), scales.data(), ks.size());
    std::reverse(scales.begin(), scales.end());
    d.imm.copy(vs.ptr(), scales.data(), vs.size());
    auto queries = pf_harness::random_f32(8 * kRow, 301, 0.0f, 0.03f);
    auto gates = pf_harness::random_f32(8 * kRow, 302, 0.0f, 1.0f);
    d.imm.copy(q.ptr(), queries.data(), q.size());
    d.imm.copy(gate.ptr(), gates.data(), gate.size());
  }
  void set(uint32_t M, uint32_t pos, uint32_t active) {
    auto* c = control.as<runtime::Control>();
    *c = runtime::Control{};
    c->pos = pos;
    c->n_active = active;
    auto* table = rows.as<runtime::BatchRowTable>();
    *table = runtime::BatchRowTable{};
    for (uint32_t r = 0; r < M; ++r) {
      const bool second = r >= 5;
      table->row[r].pos = second ? 4096 + r - 5 : pos + r;
      table->row[r].kv_base = second ? 150080 : 0;
      table->row[r].slot = second ? 1 : 0;
    }
  }
};
struct Step {
  l0::Module module;
  l0::Kernel dec, reduce;
  l0::Mem partial, out;
  l0::CmdList list;
  Step(pf_harness::Dev& d, Inputs& in, uint32_t kind, uint32_t M)
      : module(d.ctx, kernels::path("kv8_decode_probe" + std::to_string(kind) + "_M" + std::to_string(M))),
        dec(module.kernel("attn_decode_v2_kv8")), reduce(module.kernel("attn_reduce_v2_kv8")),
        partial(d.ctx, l0::MemKind::Device, (size_t(kHeads) * kBlocks * M * 258 + 16) * 4),
        out(d.ctx, l0::MemKind::Device, (size_t(M) * kRow + 16) * 2), list(l0::CmdList::regular(d.ctx)) {
    dec.group_size(256);
    reduce.group_size(256);
    dec.arg_ptr(0, in.control.ptr()); dec.arg_ptr(1, in.q.ptr());
    dec.arg_ptr(2, in.k.ptr()); dec.arg_ptr(3, in.ks.ptr());
    dec.arg_ptr(4, in.v.ptr()); dec.arg_ptr(5, in.vs.ptr()); dec.arg_ptr(6, partial.ptr());
    reduce.arg_ptr(0, in.control.ptr()); reduce.arg_ptr(1, partial.ptr());
    reduce.arg_ptr(2, in.gate.ptr()); reduce.arg_ptr(3, out.ptr());
    if (M == 8) { dec.arg_ptr(7, in.rows.ptr()); reduce.arg_ptr(4, in.rows.ptr()); }
    list.launch(dec, 4, kBlocks);
    list.launch(reduce, kHeads, M);
    list.close();
  }
  void clear(pf_harness::Dev& d) {
    d.imm.fill(partial.ptr(), 0x7FAFF00Du, partial.size());
    d.imm.fill(out.ptr(), 0xA55AA55Au, out.size());
  }
  void run(pf_harness::Dev& d) { d.q.execute(list, &d.fence); d.fence.wait(); }
};
size_t compare(pf_harness::Dev& d, Step& a, Step& b) {
  size_t bad = 0;
  for (const auto pair : {std::pair<l0::Mem*, l0::Mem*>{&a.partial, &b.partial}, {&a.out, &b.out}}) {
    std::vector<uint32_t> lhs(pair.first->size() / 4), rhs(lhs.size());
    d.imm.copy(lhs.data(), pair.first->ptr(), pair.first->size());
    d.imm.copy(rhs.data(), pair.second->ptr(), pair.second->size());
    for (size_t i = 0; i < lhs.size(); ++i) bad += lhs[i] != rhs[i];
    for (size_t i = lhs.size() - (pair.first == &a.partial ? 16 : 8); i < lhs.size(); ++i) {
      const uint32_t guard = pair.first == &a.partial ? 0x7FAFF00Du : 0xA55AA55Au;
      CHECK_EQ(lhs[i], guard); CHECK_EQ(rhs[i], guard);
    }
  }
  return bad;
}
void check_scales(pf_harness::Dev& d, uint32_t kind) {
  std::vector<uint16_t> codes;
  std::vector<float> expected;
  for (uint32_t h = 0; h < 65536; ++h) {
    if ((h & 0x7C00u) == 0x7C00u) continue;
    codes.push_back(uint16_t(h));
    expected.push_back(common::kv8::f16f(uint16_t(h)));
  }
  auto input = pf_harness::upload(d.ctx, d.imm, codes);
  l0::Mem out(d.ctx, l0::MemKind::Device, (codes.size() + 16) * 4);
  d.imm.fill(out.ptr(), 0x3F123456u, out.size());
  l0::Module module(d.ctx, kernels::path("kv8_decode_probe" + std::to_string(kind) + "_M1"));
  auto kernel = module.kernel("kv8_decode_scale_probe");
  kernel.group_size(256);
  kernel.arg_ptr(0, input.ptr()); kernel.arg_ptr(1, out.ptr()); kernel.arg(2, uint32_t(codes.size()));
  d.run(kernel, (uint32_t(codes.size()) + 255) / 256, 1);
  std::vector<float> got(codes.size() + 16);
  d.imm.copy(got.data(), out.ptr(), out.size());
  CHECK(std::memcmp(got.data(), expected.data(), expected.size() * 4) == 0);
  for (size_t i = codes.size(); i < got.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &got[i], 4);
    CHECK_EQ(bits, 0x3F123456u);
  }
  std::printf("scale widening: %zu finite codes bitwise, guards intact\n", codes.size());
}
}  // namespace

int main(int argc, char** argv) {
  bool time = false, negative = false, require_gain = false;
  uint32_t kind = 2;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--time") time = true;
    else if (arg == "--negative") negative = true;
    else if (arg == "--require-gain") require_gain = true;
    else if (arg == "--native") kind = 1;
    else { std::fprintf(stderr, "unknown probe argument %s\n", argv[i]); return 2; }
  }
  pf_harness::Dev d;
  check_scales(d, kind);
  Inputs input(d);
  size_t cases = 0;
  for (uint32_t M : {1u, 2u, 3u, 4u, 8u}) {
    Step before(d, input, 0, M), candidate(d, input, kind, M);
    for (uint32_t pos : {0u, 1u, 15u, 16u, 63u, 64u, 2047u, 2048u, 4095u, 4096u,
                         32768u, 81920u, 150000u, 150015u}) {
      for (uint32_t active : {M, M > 1 ? M - 1 : 1}) {
        input.set(M, pos, active);
        before.clear(d); candidate.clear(d);
        before.run(d);
        if (negative) {
          const int8_t changed = 0;
          d.imm.fill(input.v.ptr(), uint32_t(changed), input.v.size());
        }
        candidate.run(d);
        const size_t bad = compare(d, before, candidate);
        std::printf("bitwise M=%u active=%u pos=%u kind=%u bad_words=%zu\n", M, active, pos, kind, bad);
        CHECK_EQ(bad, size_t(0));
        ++cases;
      }
    }
    if (time && (M == 1 || M == 4)) {
      for (uint32_t depth : {4096u, 32768u, 81920u, 150000u}) {
        input.set(M, depth, M);
        std::vector<double> old_ms, new_ms, ratios;
        for (unsigned warm = 0; warm < 8; ++warm) { before.run(d); candidate.run(d); }
        for (unsigned round = 0; round < 9; ++round) {
          double values[2] = {};
          for (unsigned order = 0; order < 2; ++order) {
            const unsigned arm = (round + order) % 2;
            Step& step = arm ? candidate : before;
            const auto start = Clock::now();
            for (unsigned rep = 0; rep < 16; ++rep) step.run(d);
            values[arm] = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / 16;
          }
          old_ms.push_back(values[0]); new_ms.push_back(values[1]); ratios.push_back(values[0] / values[1]);
        }
        const double ratio = median(ratios);
        std::printf("timing M=%u depth=%u kind=%u old_ms=%.6f new_ms=%.6f paired_speedup=%.6f range=[%.6f,%.6f]\n",
            M, depth, kind, median(old_ms), median(new_ms), ratio,
            *std::min_element(ratios.begin(), ratios.end()), *std::max_element(ratios.begin(), ratios.end()));
        std::fflush(stdout);
        if (require_gain && M == 1 && depth >= 81920) CHECK(ratio >= 1.05);
      }
    }
  }
  std::printf("kv8_decode_bitwise_test: %zu cases PASS, guards intact\n", cases);
  return 0;
}
