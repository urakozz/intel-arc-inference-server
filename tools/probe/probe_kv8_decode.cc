// probe_kv8_decode (3h): plain decode attention at depth, one card, kernels only - the
// production bf16 and int8 v2 pairs beside probe binaries (tools/probe/kv8_decode_ablate.cl,
// or any binary with v2's arguments). One arm is one list of 16 FA layers' launches,
// each layer on its OWN cache (as the engine reads them), over incompressible random data
// (the card compresses constant memory; docs/01-hardware.md). Arms are timed as
// interleaved rounds in rotating order after a warm-up; the medians and the ratios paired
// by round against the first arm are printed.
//
//   probe_kv8_decode <M> <depths,...> <arm> [<arm> ...]
//     arm = <binary>[+<reduce binary>][@dec|@red]
//                                 int8 when the binary's name holds "KV8" or "kv8", else bf16;
//                                 @dec: the decode launches only, @red: the reduces only;
//                                 3l: `+<v2 binary>` takes the reduce from that binary, and a decode
//                                 binary named *verify_dpas* / *vdpas* binds attn_verify_dpas[_kv8] at its
//                                 own reqd_work_group_size (the verify list's pair)
//
// Depth is Control::pos (row 0 reads pos + 1 keys).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
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
#include <level_zero/ze_api.h>
#include "runtime/control.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kLayers = 16, kBlocks = 32, kRow = 6144, kHeads = 24, kKvRow = 1024;

void fail(const std::string& what) {
  std::fprintf(stderr, "probe_kv8_decode: %s\n", what.c_str());
  std::exit(2);
}

// Incompressible bytes: one xorshift-filled host chunk copied over the buffer at offsets that
// shift by an odd count per copy, so no two cache lines of a layer repeat at the same phase.
void seed(l0::Context& ctx, l0::CmdList& imm, void* dst, size_t bytes, uint32_t salt) {
  const size_t chunk = 32ul << 20;
  std::vector<uint32_t> h(chunk / 4 + 64);
  uint32_t s = 0x9E3779B9u ^ salt;
  for (auto& w : h) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    w = s;
  }
  l0::Mem host(ctx, l0::MemKind::Host, chunk + 256);
  std::memcpy(host.ptr(), h.data(), chunk + 256);
  uint32_t k = 0;
  for (size_t off = 0; off < bytes; off += chunk, ++k)
    imm.copy(static_cast<char*>(dst) + off, static_cast<char*>(host.ptr()) + (k * 68u) % 256u,
             std::min(chunk, bytes - off));
}

struct Arm {
  std::string spec, binary, red_binary;
  bool kv8 = false, dec = true, red = true, dpas = false;
  std::unique_ptr<l0::Module> module, red_module;
  std::unique_ptr<l0::CmdList> list;
  std::vector<l0::Kernel> kernels;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) fail("usage: probe_kv8_decode <M> <depths,...> <arm> [<arm> ...]");
  const uint32_t M = uint32_t(std::stoul(argv[1]));
  if (M < 1 || M > 4) fail("M is 1..4");
  std::vector<uint32_t> depths;
  {
    std::stringstream ss(argv[2]);
    for (std::string t; std::getline(ss, t, ',');) depths.push_back(uint32_t(std::stoul(t)));
  }
  const uint32_t capacity = ((*std::max_element(depths.begin(), depths.end()) + M + 255u) & ~255u);
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  std::vector<Arm> arms(argc - 3);
  bool any8 = false, any16 = false;
  for (size_t a = 0; a < arms.size(); ++a) {
    Arm& arm = arms[a];
    arm.spec = argv[3 + a];
    arm.binary = arm.spec.substr(0, arm.spec.find('@'));
    if (arm.binary.find('+') != std::string::npos) {
      arm.red_binary = arm.binary.substr(arm.binary.find('+') + 1);
      arm.binary = arm.binary.substr(0, arm.binary.find('+'));
    }
    arm.dpas = arm.binary.find("verify_dpas") != std::string::npos || arm.binary.find("vdpas") != std::string::npos;
    if (arm.dpas && arm.red_binary.empty()) fail(arm.spec + ": a DPAS arm needs +<v2 binary> for its reduce");
    const std::string part = arm.spec.find('@') == std::string::npos ? "" : arm.spec.substr(arm.spec.find('@') + 1);
    if (part == "dec") arm.red = false;
    else if (part == "red") arm.dec = false;
    else if (!part.empty()) fail("unknown part @" + part);
    arm.kv8 = arm.binary.find("KV8") != std::string::npos || arm.binary.find("kv8") != std::string::npos;
    (arm.kv8 ? any8 : any16) = true;
    arm.module = std::make_unique<l0::Module>(ctx, kernels::path(arm.binary));
    if (!arm.red_binary.empty()) arm.red_module = std::make_unique<l0::Module>(ctx, kernels::path(arm.red_binary));
  }
  // Per layer: its own K / V (and scales) at `capacity` positions.
  const size_t kv8_bytes = size_t(capacity) * kKvRow, kv16_bytes = kv8_bytes * 2, sc_bytes = size_t(capacity) * 4 * 2;
  std::vector<std::unique_ptr<l0::Mem>> k8, v8, ks, vs, k16, v16;
  for (uint32_t l = 0; l < kLayers; ++l) {
    if (any8) {
      k8.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kv8_bytes));
      v8.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kv8_bytes));
      ks.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sc_bytes));
      vs.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sc_bytes));
      seed(ctx, imm, k8.back()->ptr(), kv8_bytes, 4 * l + 1);
      seed(ctx, imm, v8.back()->ptr(), kv8_bytes, 4 * l + 2);
      // Scales: finite positive fp16 in the writer's range (amax / 127 of rotated rows).
      std::vector<uint16_t> sc(size_t(capacity) * 4);
      std::mt19937 rng(l);
      for (auto& h : sc) h = uint16_t(0x1C00 + rng() % 0x0C00);
      imm.copy(ks.back()->ptr(), sc.data(), sc_bytes);
      for (auto& h : sc) h = uint16_t(0x1C00 + rng() % 0x0C00);
      imm.copy(vs.back()->ptr(), sc.data(), sc_bytes);
    }
    if (any16) {
      k16.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kv16_bytes));
      v16.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kv16_bytes));
      // bf16 rows: random sign / mantissa, exponent held near 1 so every value is finite.
      const size_t words = 16u << 20;
      std::vector<uint16_t> h(words + 8);
      std::mt19937 rng(100 + l);
      for (auto& w : h) w = uint16_t((rng() & 0x807F) | (0x3C00 + (rng() % 8) * 0x80));
      for (l0::Mem* m : {k16.back().get(), v16.back().get()})
        for (size_t off = 0; off < kv16_bytes; off += words * 2)
          imm.copy(static_cast<char*>(m->ptr()) + off, h.data() + (off / 2 / words) % 7,
                   std::min(words * 2, kv16_bytes - off));
    }
  }
  l0::Mem control(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  l0::Mem attn_q(ctx, l0::MemKind::Device, size_t(M) * kRow * 4), gate(ctx, l0::MemKind::Device, attn_q.size());
  l0::Mem part(ctx, l0::MemKind::Device, size_t(kHeads) * kBlocks * M * 258 * 4);
  l0::Mem out(ctx, l0::MemKind::Device, size_t(M) * kRow * 2);
  {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.03f);
    std::vector<float> h(size_t(M) * kRow);
    for (auto& x : h) x = nd(rng);
    imm.copy(attn_q.ptr(), h.data(), attn_q.size());
    for (auto& x : h) x = nd(rng) * 30.0f;
    imm.copy(gate.ptr(), h.data(), gate.size());
  }
  for (Arm& arm : arms) {
    arm.kernels.reserve(2 * kLayers);
    arm.list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx));
    const std::string dn = arm.dpas ? (arm.kv8 ? "attn_verify_dpas_kv8" : "attn_verify_dpas")
                                    : (arm.kv8 ? "attn_decode_v2_kv8" : "attn_decode_v2");
    const std::string rn = arm.kv8 ? "attn_reduce_v2_kv8" : "attn_reduce_v2";
    for (uint32_t l = 0; l < kLayers; ++l) {
      if (arm.dec) {
        arm.kernels.push_back(arm.module->kernel(dn.c_str()));
        l0::Kernel& k = arm.kernels.back();
        if (arm.dpas) {
          ze_kernel_properties_t p{};
          p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
          if (zeKernelGetProperties(k.handle(), &p) != ZE_RESULT_SUCCESS || p.requiredGroupSizeX < 16)
            fail(arm.binary + ": no reqd_work_group_size");
          k.group_size(p.requiredGroupSizeX);
        } else {
          k.group_size(256);
        }
        k.arg_ptr(0, control.ptr());
        k.arg_ptr(1, attn_q.ptr());
        if (arm.kv8) {
          k.arg_ptr(2, k8[l]->ptr()); k.arg_ptr(3, ks[l]->ptr());
          k.arg_ptr(4, v8[l]->ptr()); k.arg_ptr(5, vs[l]->ptr()); k.arg_ptr(6, part.ptr());
        } else {
          k.arg_ptr(2, k16[l]->ptr()); k.arg_ptr(3, v16[l]->ptr()); k.arg_ptr(4, part.ptr());
        }
        arm.list->launch(k, 4, kBlocks);
      }
      if (arm.red) {
        arm.kernels.push_back((arm.red_module ? arm.red_module : arm.module)->kernel(rn.c_str()));
        l0::Kernel& k = arm.kernels.back();
        k.group_size(256);
        k.arg_ptr(0, control.ptr()); k.arg_ptr(1, part.ptr());
        k.arg_ptr(2, gate.ptr()); k.arg_ptr(3, out.ptr());
        arm.list->launch(k, kHeads, M);
      }
    }
    arm.list->close();
  }
  auto once = [&](Arm& arm) {
    const auto t0 = Clock::now();
    q.execute(*arm.list, &fence);
    fence.wait();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  std::printf("probe_kv8_decode: M %u capacity %u, %u layers per list, %zu arms\n", M, capacity, kLayers, arms.size());
  for (const uint32_t depth : depths) {
    auto* c = control.as<runtime::Control>();
    *c = runtime::Control{};
    c->pos = depth;
    c->n_active = M;
    for (int w = 0; w < 6; ++w)
      for (Arm& arm : arms) once(arm);
    const size_t n = arms.size();
    std::vector<std::vector<double>> ms(n), ratio(n);
    for (uint32_t round = 0; round < 11; ++round) {
      std::vector<double> t(n);
      for (size_t i = 0; i < n; ++i) {
        const size_t a = (round + i) % n;
        t[a] = once(arms[a]);
      }
      for (size_t a = 0; a < n; ++a) {
        ms[a].push_back(t[a]);
        ratio[a].push_back(t[a] / t[0]);
      }
    }
    for (size_t a = 0; a < n; ++a) {
      std::sort(ms[a].begin(), ms[a].end());
      std::sort(ratio[a].begin(), ratio[a].end());
      const double bytes = double(depth + M) * kKvRow * 2 * (arms[a].kv8 ? 1 : 2) * kLayers +
                           (arms[a].kv8 ? double(depth + M) * 4 * 2 * 2 * kLayers : 0.0);
      std::printf("depth %6u M %u %-34s %8.3f ms  %6.1f GB/s  x arm0 %.4f [%.4f, %.4f]\n", depth, M,
                  arms[a].spec.c_str(), ms[a][5], arms[a].dec ? bytes / ms[a][5] / 1e6 : 0.0, ratio[a][5],
                  ratio[a].front(), ratio[a].back());
    }
    std::fflush(stdout);
  }
  return 0;
}
