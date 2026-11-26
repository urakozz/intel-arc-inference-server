// Plan 41h (one card): Inkling's drafter, its full layer's attention at depth - df_attn_part from the scalar FULL
// binary against the DPAS one (df_attn_dpas.cl FULL_TARGETS), each merged by the scalar binary's df_attn_reduce, over
// random bf16 keys and values and random f32 queries and block rows (8 rows, K 7). Per depth: each arm's median of
// kRounds interleaved timings of the part + reduce pair, then the outputs compared (the largest bf16 difference and
// the cosine over every live row). Each partition count T after the depths adds a DPAS arm of T partitions a kv head
// (the probe builds _T<T> at 32 / 64 / 256, the reduce from the scalar binary of the same T; the production binaries
// are kDraftPartitions'); every arm is compared with the scalar.
//   probe_ink_draft_attn <depths,...> [T ...]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/inkling2_kernels.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/inkling2/inkling2_sizes.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kM = 8, kQ = 24, kKv = 8, kHd = 128, kRounds = 7;

void seed(l0::CmdList& imm, l0::Context& ctx, void* dst, size_t bytes, uint32_t salt, uint32_t mask, uint32_t set) {
  l0::Mem host(ctx, l0::MemKind::Host, bytes);
  auto* w = static_cast<uint32_t*>(host.ptr());
  uint32_t s = 0x9E3779B9u ^ salt;
  for (size_t i = 0; i < bytes / 4; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    w[i] = (s & mask) | set;
  }
  imm.copy(dst, host.ptr(), bytes);
}

float bf16(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe_ink_draft_attn <depths,...> [T ...]\n");
    return 2;
  }
  std::vector<uint32_t> depths;
  std::stringstream ss(argv[1]);
  for (std::string t; std::getline(ss, t, ',');) depths.push_back(uint32_t(std::stoul(t)));
  const uint32_t capacity = *std::max_element(depths.begin(), depths.end());
  l0::Context ctx(0);
  l0::Queue queue(ctx);
  auto imm = l0::CmdList::immediate(ctx);
  const size_t kv_bytes = size_t(capacity) * kKv * kHd * 2;
  l0::Mem k(ctx, l0::MemKind::Device, kv_bytes), v(ctx, l0::MemKind::Device, kv_bytes);
  // bf16 of modest size (an exponent near 1), f32 likewise for q and the block's K / V.
  for (size_t off = 0; off < kv_bytes; off += 64ul << 20) {
    const size_t n = std::min<size_t>(64ul << 20, kv_bytes - off);
    seed(imm, ctx, static_cast<char*>(k.ptr()) + off, n, 1 + uint32_t(off >> 20), 0x803F803Fu, 0x3D803D80u);
    seed(imm, ctx, static_cast<char*>(v.ptr()) + off, n, 7 + uint32_t(off >> 20), 0x807F807Fu, 0x3F003F00u);
  }
  l0::Mem q(ctx, l0::MemKind::Device, size_t(kM) * kQ * kHd * 4), kb(ctx, l0::MemKind::Device, size_t(kM) * kKv * kHd * 4),
      vb(ctx, l0::MemKind::Device, size_t(kM) * kKv * kHd * 4);
  seed(imm, ctx, q.ptr(), q.size(), 11, 0x807FFFFFu, 0x3E000000u);
  seed(imm, ctx, kb.ptr(), kb.size(), 12, 0x807FFFFFu, 0x3E000000u);
  seed(imm, ctx, vb.ptr(), vb.size(), 13, 0x807FFFFFu, 0x3F000000u);
  l0::Mem ctrl_mem(ctx, l0::MemKind::Shared, sizeof(runtime::Control)),
      dctl_mem(ctx, l0::MemKind::Shared, sizeof(runtime::dflash::DFlashCtl));
  std::memset(ctrl_mem.ptr(), 0, ctrl_mem.size());
  std::memset(dctl_mem.ptr(), 0, dctl_mem.size());
  auto* ctrl = static_cast<runtime::Control*>(ctrl_mem.ptr());
  static_cast<runtime::dflash::DFlashCtl*>(dctl_mem.ptr())->k = kM - 1;

  namespace ik = kernels::inkling2;
  const float scale = 1.f / std::sqrt(float(kHd));
  struct Arm {
    std::string name;
    uint32_t tgt, group;
    std::unique_ptr<l0::Module> part, reduce;
    std::unique_ptr<l0::Mem> parts, out;
  };
  std::vector<Arm> arms;
  const auto add = [&](const std::string& part, const std::string& reduce, uint32_t tgt, uint32_t group) {
    auto& a = arms.emplace_back();
    a.name = part;
    a.tgt = tgt;
    a.group = group;
    a.part = std::make_unique<l0::Module>(ctx, kernels::path(part));
    a.reduce = std::make_unique<l0::Module>(ctx, kernels::path(reduce));
    a.parts = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(kQ) * tgt * kM * (kHd + 2) * 4);
    a.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(kM) * kQ * kHd * 2);
  };
  const std::string scalar(ik::kDraftGlobal), dpas(ik::kDraftGlobalDpas);
  add(scalar, scalar, runtime::inkling2::kDraftPartitions, 128);
  add(dpas, scalar, runtime::inkling2::kDraftPartitions, 16 * kQ / kKv);
  for (int i = 2; i < argc; ++i) {
    const auto t = std::string(argv[i]);
    add(dpas + "_T" + t, scalar + "_T" + t, uint32_t(std::stoul(t)), 16 * kQ / kKv);
  }
  const auto run = [&](Arm& arm) {
    auto list = l0::CmdList::regular(ctx);
    auto p = arm.part->kernel("df_attn_part");
    p.group_size(arm.group);
    uint32_t i = 0;
    p.arg_ptr(i++, ctrl_mem.ptr());
    p.arg_ptr(i++, dctl_mem.ptr());
    p.arg_ptr(i++, q.ptr());
    p.arg_ptr(i++, k.ptr());
    p.arg_ptr(i++, v.ptr());
    p.arg(i++, scale);
    p.arg_ptr(i++, arm.parts->ptr());
    list.launch(p, kKv, arm.tgt, 1);
    auto r = arm.reduce->kernel("df_attn_reduce");
    r.group_size(128);
    i = 0;
    for (const void* x : {ctrl_mem.ptr(), dctl_mem.ptr(), arm.parts->ptr(), q.ptr(), kb.ptr(), vb.ptr()}) r.arg_ptr(i++, x);
    r.arg(i++, scale);
    r.arg_ptr(i++, arm.out->ptr());
    list.launch(r, kQ, kM, 1);
    list.close();
    const auto start = Clock::now();
    queue.execute(list);
    queue.synchronize();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  for (const uint32_t depth : depths) {
    ctrl->pos = depth;
    std::vector<std::vector<double>> ms(arms.size());
    for (auto& arm : arms) run(arm);   // warm-up
    for (uint32_t round = 0; round < kRounds; ++round)
      for (size_t a = 0; a < arms.size(); ++a) ms[a].push_back(run(arms[a]));
    std::vector<std::vector<uint16_t>> out(arms.size());
    for (size_t a = 0; a < arms.size(); ++a) {
      out[a].resize(size_t(kM) * kQ * kHd);
      imm.copy(out[a].data(), arms[a].out->ptr(), out[a].size() * 2);
      std::sort(ms[a].begin(), ms[a].end());
    }
    for (size_t a = 0; a < arms.size(); ++a) {
      double dot = 0, n0 = 0, n1 = 0, worst = 0;
      for (size_t i = 0; i < out[0].size(); ++i) {
        const double x = bf16(out[0][i]), y = bf16(out[a][i]);
        dot += x * y;
        n0 += x * x;
        n1 += y * y;
        worst = std::max(worst, std::fabs(x - y));
      }
      std::printf("depth %u: %s %.3f ms (%.2fx the scalar); largest difference %.3g, cosine %.7f\n", depth,
                  arms[a].name.c_str(), ms[a][kRounds / 2], ms[0][kRounds / 2] / ms[a][kRounds / 2], worst,
                  dot / std::sqrt(n0 * n1));
    }
  }
}
