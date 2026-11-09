// 3l: the production attn_verify_dpas[_kv8] (src/kernels/attn_verify_dpas.cl) against its frozen copy at main 865da900
// (tests/kernels/attn_verify_control.cl, the same levers and options), every partial word (mx, sm, acc[256]) and every
// word of v2's reduce over them, bf16 KV at M = 2..8 (the built lists) and int8 KV at M = 2..4. Depths through the
// block-stride changes and the long tails, all rows and one inactive, two input fills (patterned scales with -0,
// subnormals and the largest finite code; then every finite fp16 code cycled over the positions).
//   attn_verify_bitwise_test [--negative] [--cand <prefix>] [--max-m <M>]
//     --negative: V is zeroed between the two runs - the comparison must fail;
//     --cand: the candidate is <prefix>M<M>[_KV8] (a kernel-dir-relative probe build) instead of production's.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <level_zero/ze_api.h>

#include "check.h"
#include "kernels/kernels.h"
#include "prefill/pf_harness.h"
#include "runtime/control.h"

namespace {
constexpr uint32_t kCapacity = 160000, kRow = 6144, kHeads = 24, kBlocks = 32;
struct Inputs {
  l0::Mem k8, v8, ks, vs, k16, v16, q, gate, control;
  Inputs(pf_harness::Dev& d)
      : k8(d.ctx, l0::MemKind::Device, size_t(kCapacity) * 1024),
        v8(d.ctx, l0::MemKind::Device, k8.size()),
        ks(d.ctx, l0::MemKind::Device, size_t(kCapacity) * 4 * 2),
        vs(d.ctx, l0::MemKind::Device, ks.size()),
        k16(d.ctx, l0::MemKind::Device, size_t(kCapacity) * 1024 * 2),
        v16(d.ctx, l0::MemKind::Device, k16.size()),
        q(d.ctx, l0::MemKind::Device, size_t(8) * kRow * 4),
        gate(d.ctx, l0::MemKind::Device, q.size()),
        control(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)) {
    std::vector<int8_t> data(k8.size());
    for (size_t i = 0; i < data.size(); ++i) data[i] = int8_t((i * 37 + i / 1024) % 255 - 127);
    d.imm.copy(k8.ptr(), data.data(), data.size());
    // bf16 rows: sign and mantissa from a pattern, exponents 2^-3 .. 2^4 (finite, varied).
    std::vector<uint16_t> h(k16.size() / 2);
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = uint16_t(((i * 2654435761u) >> 7 & 0x807Fu) | (0x3E00u + ((i * 40503u) >> 5) % 8 * 0x80u));
    d.imm.copy(k16.ptr(), h.data(), k16.size());
    fill_v(d);
    patterned_scales(d);
    auto queries = pf_harness::random_f32(8 * kRow, 301, 0.0f, 0.03f);
    auto gates = pf_harness::random_f32(8 * kRow, 302, 0.0f, 1.0f);
    d.imm.copy(q.ptr(), queries.data(), q.size());
    d.imm.copy(gate.ptr(), gates.data(), gate.size());
  }
  void fill_v(pf_harness::Dev& d) {
    std::vector<int8_t> data(v8.size());
    for (size_t i = 0; i < data.size(); ++i) data[i] = int8_t((i * 13 + i / 256) % 255 - 127);
    d.imm.copy(v8.ptr(), data.data(), data.size());
    std::vector<uint16_t> h(v16.size() / 2);
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = uint16_t(((i * 2246822519u) >> 9 & 0x807Fu) | (0x3E00u + ((i * 31337u) >> 4) % 8 * 0x80u));
    d.imm.copy(v16.ptr(), h.data(), v16.size());
  }
  // kv8_decode_bitwise_test's fill: specials (0, -0, subnormals, the largest finite) among ordinary codes.
  void patterned_scales(pf_harness::Dev& d) {
    std::vector<uint16_t> scales(size_t(kCapacity) * 4);
    const uint16_t special[] = {0, 0x8000, 1, 0x3FF, 0x400, 0x7BFF, 0x2000, 0x2400};
    for (size_t i = 0; i < scales.size(); ++i)
      scales[i] = i % 257 == 0 ? special[(i / 257) % 8] : uint16_t(0x2000 + i % 512);
    d.imm.copy(ks.ptr(), scales.data(), ks.size());
    std::reverse(scales.begin(), scales.end());
    d.imm.copy(vs.ptr(), scales.data(), vs.size());
  }
  // Every finite fp16 code (63,488, signed zeros and subnormals included) cycled over the (position, kv head) slots.
  void every_scale(pf_harness::Dev& d) {
    std::vector<uint16_t> codes;
    for (uint32_t c = 0; c < 65536; ++c)
      if ((c & 0x7C00u) != 0x7C00u) codes.push_back(uint16_t(c));
    std::vector<uint16_t> scales(size_t(kCapacity) * 4);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = codes[(i * 7919u) % codes.size()];
    d.imm.copy(ks.ptr(), scales.data(), ks.size());
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = codes[(i * 104729u + 11u) % codes.size()];
    d.imm.copy(vs.ptr(), scales.data(), vs.size());
  }
  void set(uint32_t pos, uint32_t active) {
    auto* c = control.as<runtime::Control>();
    *c = runtime::Control{};
    c->pos = pos;
    c->n_active = active;
  }
};
struct Step {
  l0::Module module, red_module;
  l0::Kernel dec, reduce;
  l0::Mem partial, out;
  l0::CmdList list;
  Step(pf_harness::Dev& d, Inputs& in, bool control, uint32_t M, bool kv8, const std::string& cand = "")
      : module(d.ctx, kernels::path(control ? "attn_verify_control_M" + std::to_string(M) + (kv8 ? "_KV8" : "")
                                   : !cand.empty() ? cand + "M" + std::to_string(M) + (kv8 ? "_KV8" : "")
                                                   : kernels::attn_verify_dpas_variant(M, kBlocks, kv8))),
        red_module(d.ctx, kernels::path(kv8 ? kernels::attn_v2_kv8_variant(M, kBlocks) : kernels::attn_v2_variant(M, kBlocks))),
        dec(module.kernel(kv8 ? "attn_verify_dpas_kv8" : "attn_verify_dpas")),
        reduce(red_module.kernel(kv8 ? "attn_reduce_v2_kv8" : "attn_reduce_v2")),
        partial(d.ctx, l0::MemKind::Device, (size_t(kHeads) * kBlocks * M * 258 + 16) * 4),
        out(d.ctx, l0::MemKind::Device, (size_t(M) * kRow + 16) * 2), list(l0::CmdList::regular(d.ctx)) {
    ze_kernel_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    CHECK(zeKernelGetProperties(dec.handle(), &p) == ZE_RESULT_SUCCESS && p.requiredGroupSizeX >= 16);
    dec.group_size(p.requiredGroupSizeX);
    reduce.group_size(256);
    uint32_t a = 0;
    dec.arg_ptr(a++, in.control.ptr()); dec.arg_ptr(a++, in.q.ptr());
    if (kv8) {
      dec.arg_ptr(a++, in.k8.ptr()); dec.arg_ptr(a++, in.ks.ptr());
      dec.arg_ptr(a++, in.v8.ptr()); dec.arg_ptr(a++, in.vs.ptr());
    } else {
      dec.arg_ptr(a++, in.k16.ptr()); dec.arg_ptr(a++, in.v16.ptr());
    }
    dec.arg_ptr(a, partial.ptr());
    reduce.arg_ptr(0, in.control.ptr()); reduce.arg_ptr(1, partial.ptr());
    reduce.arg_ptr(2, in.gate.ptr()); reduce.arg_ptr(3, out.ptr());
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
// Mismatched words over partials and outputs (the guard words past each buffer's end must be untouched); `finite`
// counts the output words that are finite bf16, so a case that only compares NaN to NaN is visible.
size_t compare(pf_harness::Dev& d, Step& a, Step& b, size_t* finite) {
  size_t bad = 0;
  for (const auto pair : {std::pair<l0::Mem*, l0::Mem*>{&a.partial, &b.partial}, {&a.out, &b.out}}) {
    std::vector<uint32_t> lhs(pair.first->size() / 4), rhs(lhs.size());
    d.imm.copy(lhs.data(), pair.first->ptr(), pair.first->size());
    d.imm.copy(rhs.data(), pair.second->ptr(), pair.second->size());
    for (size_t i = 0; i < lhs.size(); ++i) bad += lhs[i] != rhs[i];
    const bool part = pair.first == &a.partial;
    for (size_t i = lhs.size() - (part ? 16 : 8); i < lhs.size(); ++i) {
      const uint32_t guard = part ? 0x7FAFF00Du : 0xA55AA55Au;
      CHECK_EQ(lhs[i], guard); CHECK_EQ(rhs[i], guard);
    }
    if (!part)
      for (size_t i = 0; i + 8 < lhs.size(); ++i)
        for (uint32_t w : {lhs[i] & 0xFFFFu, lhs[i] >> 16})
          if ((w & 0x7F80u) != 0x7F80u) ++*finite;
  }
  return bad;
}
}  // namespace

int main(int argc, char** argv) {
  bool negative = false;
  std::string cand;
  uint32_t max_m = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--negative") negative = true;
    else if (arg == "--cand" && i + 1 < argc) cand = argv[++i];
    else if (arg == "--max-m" && i + 1 < argc) max_m = uint32_t(std::stoul(argv[++i]));
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  pf_harness::Dev d;
  Inputs input(d);
  std::vector<std::pair<uint32_t, bool>> shapes;
  for (uint32_t m = 2; m <= 4; ++m) shapes.push_back({m, true});
#ifdef B70_VERIFY_TEST_M8
  for (uint32_t m = 2; m <= 8; ++m) shapes.push_back({m, false});
#else
  for (uint32_t m = 2; m <= 4; ++m) shapes.push_back({m, false});
#endif
  size_t cases = 0, total_bad = 0;
  for (unsigned fill = 0; fill < 2; ++fill) {
    if (fill == 1) input.every_scale(d);
    for (const auto& [M, kv8] : shapes) {
      if (fill == 1 && !kv8) continue;   // the scales are the int8 form's
      if (M > max_m) continue;
      Step before(d, input, true, M, kv8), candidate(d, input, false, M, kv8, cand);
      for (uint32_t pos : {0u, 1u, 15u, 31u, 32u, 63u, 64u, 2047u, 2048u, 2049u, 4093u, 4095u, 4096u, 11583u, 11585u,
                           32768u, 81920u, 149999u, 150000u, 150015u}) {
        if (fill == 1 && pos < 4096u) continue;
        for (uint32_t active : {M, M - 1}) {
          input.set(pos, active);
          before.clear(d); candidate.clear(d);
          before.run(d);
          if (negative) d.imm.fill(kv8 ? input.v8.ptr() : input.v16.ptr(), 0u, kv8 ? input.v8.size() : input.v16.size());
          candidate.run(d);
          size_t finite = 0;
          const size_t bad = compare(d, before, candidate, &finite);
          std::printf("bitwise %s M=%u active=%u pos=%u fill=%u bad_words=%zu finite_out=%zu\n", kv8 ? "int8" : "bf16",
                      M, active, pos, fill, bad, finite);
          std::fflush(stdout);
          CHECK(finite > 0);
          total_bad += bad;
          if (!negative) CHECK_EQ(bad, size_t(0));
          ++cases;
          if (negative) input.fill_v(d);   // V back for the next case
        }
      }
    }
  }
  if (negative) {
    CHECK(total_bad > 0);
    std::printf("attn_verify_bitwise_test --negative: %zu cases, %zu mismatched words (expected > 0)\n", cases, total_bad);
    return 0;
  }
  std::printf("attn_verify_bitwise_test: %zu cases PASS, guards intact\n", cases);
  return 0;
}
