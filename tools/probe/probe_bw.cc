// probe_bw: device-to-device read bandwidth with a plain L0 launch.
// Method (doc 01): 5 warm-ups, 20 timed iterations, buffers 2 GB and 8 GB.
// Timing wraps execute+fence; at >3 ms per kernel the ~20 us submit cost is <1%.
//
// The buffer is seeded with pseudo-random bytes, NOT a constant fill. The B70
// compresses device-local memory losslessly, so a buffer of identical 32-bit
// words costs almost no DRAM traffic and the probe reports ~1022 GB/s -- above
// the 608 GB/s theoretical peak. Random data is incompressible within a cache
// line, which is what model weights look like. See docs/01-hardware.md.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

namespace {
// Fills `bytes` of device memory with incompressible data by staging one
// xorshift-filled host chunk and copying it repeatedly. Repetition at chunk
// granularity is harmless: compression works per cache line.
void seed_incompressible(l0::Context& ctx, l0::CmdList& imm, void* dst, size_t bytes) {
  const size_t chunk = 64ul << 20;
  l0::Mem host(ctx, l0::MemKind::Host, chunk);
  uint32_t* h = host.as<uint32_t>();
  uint32_t s = 0x9E3779B9u;
  for (size_t i = 0; i < chunk / 4; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    h[i] = s;
  }
  for (size_t off = 0; off < bytes; off += chunk)
    imm.copy(static_cast<char*>(dst) + off, h, std::min(chunk, bytes - off));
}
}  // namespace

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("bw_sum"));
  l0::Kernel k = mod.kernel("bw_sum");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem out(ctx, l0::MemKind::Device, 1 << 20);

  const uint32_t wg = 256;
  const uint32_t groups = ctx.eu_count() * 8 * 16 / wg * 4;  // 4x oversubscribed threads
  std::printf("| buffer | GB/s (median of 20) | min | max |\n|---|---|---|---|\n");
  for (size_t gb : {2ul, 8ul}) {
    const size_t bytes = gb << 30;
    l0::Mem buf(ctx, l0::MemKind::Device, bytes);
    seed_incompressible(ctx, imm, buf.ptr(), bytes);
    const uint64_t n_vec = bytes / 16;
    k.arg_ptr(0, buf.ptr());
    k.arg(1, n_vec);
    k.arg_ptr(2, out.ptr());
    k.group_size(wg);
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, groups);
    list.close();
    std::vector<double> gbps;
    for (int it = 0; it < 25; ++it) {
      Timer t; t.start();
      q.execute(list, &fence);
      fence.wait();
      double ms = t.ms();
      if (it >= 5) gbps.push_back(double(bytes) / (ms * 1e6));
    }
    std::sort(gbps.begin(), gbps.end());
    std::printf("| %zu GB | %.0f | %.0f | %.0f |\n", gb, gbps[gbps.size() / 2], gbps.front(), gbps.back());
  }
  return 0;
}
