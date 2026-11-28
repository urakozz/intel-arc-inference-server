// Plan 41h Task 2 (one card): Inkling's full-visit decode / verify attention at depth, kernels only - the
// production kol_attn pair beside probe builds of it (tools/probe/CMakeLists.txt). One arm is one list of `visits`
// launch pairs (decode + reduce), each visit on its OWN cache over incompressible random data, timed as interleaved
// rounds after a warm-up; the medians, the bytes the visits' caches hold over the time, and each arm's ratio to the
// first are printed. After each depth every arm runs once more and its outputs are compared word for word with arm
// 0's (0 bad words: bitwise the first arm).
//   probe_ink_attn_depth <M> <visits> <depths,...> <binary> [<binary> ...]
// The binary's name gives its form: _KV8 the int8 cache, _T<n>_ its key blocks (the partial table's TGT).
//   probe_ink_attn_depth pf<rows> <visits> <depths,...> <binary> [<binary> ...]
// Task 3: the prompt flash (kol_pf_flash_attn, a full visit's PAIR form) over a <rows>-row turn at each depth; a
// *_SPLIT binary walks the keys in runtime::inkling2::attention_spans' spans, a *_VERIFY one in 32 spans from the
// Control block (the deep verify's form), then kol_pf_flash_merge.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/inkling2/inkling2_prefill.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kQ = 24, kKv = 8, kHd = 128, kRel = 16, kRelBias = 1024, kRounds = 7;

[[noreturn]] void fail(const std::string& what) {
  std::fprintf(stderr, "probe_ink_attn_depth: %s\n", what.c_str());
  std::exit(2);
}

// probe_kv8_decode's incompressible fill: one xorshift chunk copied at shifting offsets.
void seed(l0::Context& ctx, l0::CmdList& imm, void* dst, size_t bytes, uint32_t salt, uint32_t mask = ~0u,
          uint32_t set = 0) {
  const size_t chunk = 32ul << 20;
  l0::Mem host(ctx, l0::MemKind::Host, chunk + 256);
  auto* w = static_cast<uint32_t*>(host.ptr());
  uint32_t s = 0x9E3779B9u ^ salt;
  for (size_t i = 0; i < (chunk + 256) / 4; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    w[i] = (s & mask) | set;
  }
  uint32_t k = 0;
  for (size_t off = 0; off < bytes; off += chunk, ++k)
    imm.copy(static_cast<char*>(dst) + off, static_cast<char*>(host.ptr()) + (k * 68u) % 256u,
             std::min(chunk, bytes - off));
}

struct Arm {
  std::string binary;
  bool kv8 = false, split = false, verify = false;
  uint32_t tgt = 32;
  std::unique_ptr<l0::Module> module;
  std::unique_ptr<l0::Mem> part, out;
  std::unique_ptr<l0::CmdList> list;
  std::vector<l0::Kernel> kernels;
  std::vector<double> ms;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) fail("usage: probe_ink_attn_depth <M> <visits> <depths,...> <binary> [<binary> ...]");
  const std::string mode = argv[1];
  const bool pf = mode.rfind("pf", 0) == 0;
  const uint32_t M = uint32_t(std::stoul(pf ? mode.substr(2) : mode)), visits = uint32_t(std::stoul(argv[2]));
  if (M < 1 || (!pf && M > 8) || visits < 1) fail("M is 1..8 (pf: any rows), visits >= 1");
  std::vector<uint32_t> depths;
  {
    std::stringstream ss(argv[3]);
    for (std::string t; std::getline(ss, t, ',');) depths.push_back(uint32_t(std::stoul(t)));
  }
  const uint32_t capacity = (*std::max_element(depths.begin(), depths.end()) + M + 255u) & ~255u;
  l0::Context ctx(0);
  l0::Queue queue(ctx);
  l0::Fence fence(queue);
  auto imm = l0::CmdList::immediate(ctx);

  std::vector<Arm> arms(argc - 4);
  bool any8 = false, any16 = false;
  for (size_t a = 0; a < arms.size(); ++a) {
    Arm& arm = arms[a];
    arm.binary = argv[4 + a];
    arm.kv8 = arm.binary.find("_KV8") != std::string::npos;   // not the head shape's Q24KV8
    arm.split = arm.binary.ends_with("_SPLIT");
    arm.verify = arm.binary.ends_with("_VERIFY");
    const auto t = arm.binary.find("_T");
    if (!pf && t == std::string::npos) fail(arm.binary + ": no _T<n>_ in the name");
    if (!pf) arm.tgt = uint32_t(std::stoul(arm.binary.substr(t + 2)));
    if (pf) arm.tgt = arm.verify ? 32 : runtime::inkling2::attention_spans(M, UINT32_MAX);   // the partials' spans
    (arm.kv8 ? any8 : any16) = true;
    arm.module = std::make_unique<l0::Module>(ctx, kernels::path(arm.binary));
    arm.part = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(kQ) * arm.tgt * M * (kHd + 2) * 4);
    arm.out = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(M) * kQ * kHd * 2);
  }
  // Each visit's own cache at `capacity` positions: int8 codes + fp16 scales, or bf16 rows.
  const size_t rows = size_t(capacity) * kKv * kHd;
  std::vector<std::unique_ptr<l0::Mem>> k8, v8, ks, vs, k16, v16;
  for (uint32_t v = 0; v < visits; ++v) {
    if (any8) {
      for (auto* vec : {&k8, &v8}) {
        vec->push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, rows));
        seed(ctx, imm, vec->back()->ptr(), rows, 4 * v + (vec == &k8));
      }
      for (auto* vec : {&ks, &vs}) {   // finite positive fp16 scales in the writer's range
        vec->push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(capacity) * kKv * 2));
        seed(ctx, imm, vec->back()->ptr(), vec->back()->size(), 4 * v + 2 + (vec == &ks), 0x03FF03FFu, 0x1C001C00u);
      }
    }
    if (any16)
      for (auto* vec : {&k16, &v16}) {   // bf16: random sign / mantissa, exponent near 1
        vec->push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, rows * 2));
        seed(ctx, imm, vec->back()->ptr(), rows * 2, 100 + 2 * v + (vec == &k16), 0x807F807Fu, 0x3C003C00u);
      }
  }
  l0::Mem q(ctx, l0::MemKind::Device, size_t(M) * kQ * kHd * 4), relq(ctx, l0::MemKind::Device, size_t(M) * kQ * kRel * 4),
      table(ctx, l0::MemKind::Device, size_t(kRel) * kRelBias * 4), sinks(ctx, l0::MemKind::Device, size_t(M) * 64 * 4),
      gate(ctx, l0::MemKind::Device, size_t(M) * kQ * kHd * 4);
  // f32 inputs of modest size: sign, a small exponent, random mantissa.
  for (l0::Mem* m : {&q, &relq, &table, &sinks, &gate}) seed(ctx, imm, m->ptr(), m->size(), 7, 0x807FFFFFu, 0x3D000000u);
  l0::Mem ctrl_mem(ctx, l0::MemKind::Host, sizeof(runtime::Control));
  auto* ctrl = static_cast<runtime::Control*>(ctrl_mem.ptr());
  std::memset(ctrl, 0, sizeof(*ctrl));
  ctrl->n_active = M;

  const auto record = [&](Arm& arm, uint32_t depth) {   // one list: every visit's decode then reduce (pf: flash, merge)
    arm.kernels.clear();
    arm.list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx));
    for (uint32_t v = 0; v < visits && pf; ++v) {
      const uint32_t spans = arm.verify ? 32 : arm.split ? runtime::inkling2::attention_spans(M, depth) : 1;
      auto& f = arm.kernels.emplace_back(arm.module->kernel("kol_pf_flash_attn"));
      f.group_size(16 * 2 * 3);
      uint32_t i = 0;
      f.arg_ptr(i++, q.ptr());
      f.arg_ptr(i++, (arm.kv8 ? k8 : k16)[v]->ptr());
      f.arg_ptr(i++, (arm.kv8 ? v8 : v16)[v]->ptr());
      if (arm.kv8) {
        f.arg_ptr(i++, ks[v]->ptr());
        f.arg_ptr(i++, vs[v]->ptr());
      }
      f.arg_ptr(i++, arm.out->ptr());
      if (arm.verify) {
        f.arg_ptr(i++, ctrl);
      } else {
        f.arg(i++, depth);
        f.arg(i++, M);
      }
      f.arg_ptr(i++, relq.ptr());
      f.arg_ptr(i++, table.ptr());
      f.arg_ptr(i++, sinks.ptr());
      f.arg_ptr(i++, gate.ptr());
      if (arm.split) f.arg(i++, depth / spans / 64 * 64);
      if (arm.split || arm.verify) f.arg_ptr(i++, arm.part->ptr());
      arm.list->launch(f, (M + 15) / 16, kKv, spans);
      if (arm.split || arm.verify) {
        auto& mg = arm.kernels.emplace_back(arm.module->kernel("kol_pf_flash_merge"));
        mg.group_size(kHd);
        mg.arg_ptr(0, arm.part->ptr());
        mg.arg_ptr(1, arm.out->ptr());
        if (arm.verify) mg.arg_ptr(2, ctrl);
        else mg.arg(2, M);
        mg.arg(3, spans);
        mg.arg_ptr(4, sinks.ptr());
        mg.arg_ptr(5, gate.ptr());
        arm.list->launch(mg, kQ, M);
      }
    }
    for (uint32_t v = 0; v < visits && !pf; ++v) {
      auto& dec = arm.kernels.emplace_back(arm.module->kernel("kol_attn_decode"));
      dec.group_size(128);
      uint32_t i = 0;
      dec.arg_ptr(i++, ctrl);
      dec.arg_ptr(i++, q.ptr());
      dec.arg_ptr(i++, (arm.kv8 ? k8 : k16)[v]->ptr());
      dec.arg_ptr(i++, (arm.kv8 ? v8 : v16)[v]->ptr());
      if (arm.kv8) {
        dec.arg_ptr(i++, ks[v]->ptr());
        dec.arg_ptr(i++, vs[v]->ptr());
      }
      dec.arg_ptr(i++, arm.part->ptr());
      dec.arg_ptr(i++, relq.ptr());
      dec.arg_ptr(i++, table.ptr());
      arm.list->launch(dec, kKv, arm.tgt);
      auto& red = arm.kernels.emplace_back(arm.module->kernel("kol_attn_reduce"));
      red.group_size(128);
      red.arg_ptr(0, ctrl);
      red.arg_ptr(1, arm.part->ptr());
      red.arg_ptr(2, arm.out->ptr());
      red.arg_ptr(3, sinks.ptr());
      red.arg_ptr(4, gate.ptr());
      arm.list->launch(red, kQ, M);
    }
    arm.list->close();
  };
  const auto replay = [&](Arm& arm) {
    const auto t0 = Clock::now();
    queue.execute(*arm.list, &fence);
    fence.wait();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  for (const uint32_t depth : depths) {
    ctrl->pos = depth;
    for (auto& arm : arms) {
      record(arm, depth);
      arm.ms.clear();
      for (uint32_t w = 0; w < 2; ++w) replay(arm);
    }
    for (uint32_t r = 0; r < kRounds; ++r)
      for (size_t a = 0; a < arms.size(); ++a) {
        auto& arm = arms[(a + r) % arms.size()];
        arm.ms.push_back(replay(arm));
      }
    std::vector<std::vector<uint16_t>> outs;
    for (auto& arm : arms) {
      imm.fill(arm.out->ptr(), 0u, arm.out->size());
      replay(arm);
      outs.emplace_back(arm.out->size() / 2);
      imm.copy(outs.back().data(), arm.out->ptr(), arm.out->size());
    }
    double base = 0;
    for (size_t a = 0; a < arms.size(); ++a) {
      auto& arm = arms[a];
      std::sort(arm.ms.begin(), arm.ms.end());
      const double med = arm.ms[arm.ms.size() / 2];
      if (a == 0) base = med;
      size_t bad = 0;
      for (size_t i = 0; i < outs[a].size(); ++i) bad += outs[a][i] != outs[0][i];
      const double bytes = double(depth + M) * visits * kKv * (arm.kv8 ? 2 * kHd + 4 : 4 * kHd);
      std::printf("depth %u M %u %u visits  %-60s %8.3f ms  %6.1f GB/s  %.3fx  bad %zu\n", depth, M, visits,
                  arm.binary.c_str(), med, bytes / med / 1e6, med / base, bad);
    }
  }
  return 0;
}
