// probe_kv_paged_attn - plan 19v3b Task 0: the decode attention's paged twin against the contiguous kernel, in ONE
// process (interleaved pairs after a warm-up), so the host's load and the clock state are the same for both.
//
//   probe_kv_paged_attn [--depths 4096,32768,131072] [--calls 32] [--rounds 15] [--kv bf16|int8]
//
// One FA layer's cache of random values at each depth (bf16, or int8 rows + fp16 scales), the identity page table, a
// Control at pos = depth - 1 (the plain step, M = 1). Per round, each arm is ONE list of `calls` launches of
// attn_decode_v2[_kv8] (the kernel the paging changed; the reduce is the same binary's, unchanged): the contiguous
// binary over the cache, then the `_PG2048` binary over the cache and the table, the arms' order alternating by round.
// Printed per depth: the median us a launch of each arm and paged / contiguous, and whether the two arms' partials
// (attn_part) are bitwise equal.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kKvH = 4, kHD = 256, kQH = 24, kTgt = 32, kPart = 258, kWG = 256;

std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  for (size_t i = 0; i < s.size();) {
    const size_t j = std::min(s.find(',', i), s.size());
    v.push_back(uint32_t(std::stoul(s.substr(i, j - i))));
    i = j + 1;
  }
  return v;
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<uint32_t> depths = {4096, 32768, 131072};
  uint32_t calls = 32, rounds = 15;
  bool kv8 = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const std::string v = i + 1 < argc ? argv[i + 1] : "";
    if (a == "--depths") depths = parse_list(v), ++i;
    else if (a == "--calls") calls = uint32_t(std::stoul(v)), ++i;
    else if (a == "--rounds") rounds = uint32_t(std::stoul(v)), ++i;
    else if (a == "--kv") kv8 = v == "int8", ++i;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const std::string v = kv8 ? kernels::attn_v2_kv8_variant(1, kTgt) : kernels::attn_v2_variant(1, kTgt);
  l0::Module mc(ctx, kernels::path(v)), mp(ctx, kernels::path(kernels::kv_paged(v)));
  const char* entry = kv8 ? "attn_decode_v2_kv8" : "attn_decode_v2";
  std::printf("probe_kv_paged_attn: %s, M = 1, %u launches a list, %u rounds (median), arms alternating\n",
              kv8 ? "int8" : "bf16", calls, rounds);
  std::printf("| depth | contiguous us | paged us | paged / contiguous | partials |\n");
  std::mt19937 rng(7);
  for (uint32_t depth : depths) {
    const uint32_t len = runtime::kv_pages(depth) * runtime::kKvPage;
    const size_t rows = size_t(len) * kKvH * kHD * (kv8 ? 1 : 2), scales = kv8 ? size_t(len) * kKvH * 2 : 0;
    l0::Mem k(ctx, l0::MemKind::Device, rows + scales), vv(ctx, l0::MemKind::Device, rows + scales);
    l0::Mem aq(ctx, l0::MemKind::Device, size_t(kQH) * kHD * 4);
    l0::Mem pa(ctx, l0::MemKind::Device, size_t(kQH) * kTgt * kPart * 4), pb(ctx, l0::MemKind::Device, pa.size());
    l0::Mem ctrl(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    l0::Mem pages(ctx, l0::MemKind::Device, size_t(len / runtime::kKvPage) * 4);
    {   // random K / V (bf16 in [-1, 1), or int8 rows with fp16 scales of 1/64), q in [-1, 1)
      std::uniform_int_distribution<uint32_t> u(0, 0xFFFFFFFFu);
      std::vector<uint8_t> h(size_t(64) << 20);
      for (l0::Mem* m : {&k, &vv})
        for (size_t off = 0; off < m->size(); off += h.size()) {
          const size_t n = std::min(h.size(), m->size() - off);
          for (size_t i = 0; i < n; i += 2) {
            const bool scale = kv8 && off + i >= rows;
            const uint16_t x = scale ? uint16_t(0x2400)   // fp16 1/64
                                     : kv8 ? uint16_t(u(rng) & 0x7F7F)
                                           : uint16_t((0x3F00 + (u(rng) & 0xFF)) | ((u(rng) & 1) << 15));
            std::memcpy(&h[i], &x, 2);
          }
          imm.copy(static_cast<uint8_t*>(m->ptr()) + off, h.data(), n);
        }
      std::vector<float> qh(size_t(kQH) * kHD);
      for (float& x : qh) x = float(int(u(rng) % 2001) - 1000) / 1000.0f;
      imm.copy(aq.ptr(), qh.data(), qh.size() * 4);
      std::vector<uint32_t> id(len / runtime::kKvPage);
      for (uint32_t p = 0; p < id.size(); ++p) id[p] = p;
      imm.copy(pages.ptr(), id.data(), id.size() * 4);
      imm.fill(pa.ptr(), 0u, pa.size());
      imm.fill(pb.ptr(), 0u, pb.size());
      auto* c = ctrl.as<runtime::Control>();
      *c = runtime::Control{};
      c->pos = depth - 1;
      c->n_active = 1;
    }
    auto list_of = [&](bool paged) {
      l0::Kernel kn(paged ? mp : mc, entry);
      kn.group_size(kWG);
      uint32_t i = 0;
      kn.arg_ptr(i++, ctrl.ptr());
      kn.arg_ptr(i++, aq.ptr());
      kn.arg_ptr(i++, k.ptr());
      if (kv8) kn.arg_ptr(i++, static_cast<uint8_t*>(k.ptr()) + rows);
      kn.arg_ptr(i++, vv.ptr());
      if (kv8) kn.arg_ptr(i++, static_cast<uint8_t*>(vv.ptr()) + rows);
      kn.arg_ptr(i++, (paged ? pb : pa).ptr());
      if (paged) kn.arg_ptr(i, pages.ptr());
      l0::CmdList l = l0::CmdList::regular(ctx);
      for (uint32_t c = 0; c < calls; ++c) l.launch(kn, kKvH, kTgt);
      l.close();
      return std::make_pair(std::move(kn), std::move(l));
    };
    auto a = list_of(false), b = list_of(true);
    auto time = [&](l0::CmdList& l) {
      const Clock::time_point t0 = Clock::now();
      q.execute(l, &fence);
      fence.wait();
      return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / calls;
    };
    for (int w = 0; w < 3; ++w) time(a.second), time(b.second);   // warm-up
    std::vector<double> ta, tb;
    for (uint32_t r = 0; r < rounds; ++r) {
      if (r % 2) {
        tb.push_back(time(b.second));
        ta.push_back(time(a.second));
      } else {
        ta.push_back(time(a.second));
        tb.push_back(time(b.second));
      }
    }
    std::vector<uint8_t> ha(pa.size()), hb(pb.size());
    imm.copy(ha.data(), pa.ptr(), ha.size());
    imm.copy(hb.data(), pb.ptr(), hb.size());
    const double ma = median(ta), mb = median(tb);
    std::printf("| %u | %.1f | %.1f | %.4f | %s |\n", depth, ma, mb, mb / ma,
                std::memcmp(ha.data(), hb.data(), ha.size()) == 0 ? "bitwise" : "DIFFER");
  }
  return 0;
}
