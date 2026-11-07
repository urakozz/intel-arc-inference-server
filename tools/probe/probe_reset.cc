// Engine::reset()'s cost (branch cheap-reset, 2026-11-07): the persistent fills reset()
// does - PersistentBuffers::zero and MtpBuffers::zero on Qwen3.8's descriptor - at a small
// max_len and at --max-len auto's, timed as interleaved pairs after a warm-up (one arm
// right after the other, so both see the same clock state). Also: what one fill of the
// KV allocation costs on each command-queue group the device has (which engine is fast),
// and the per-buffer sizes. No weights are loaded.
//   probe_reset [--small N] [--big N] [--pairs N]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "timer.h"

namespace {
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double gb(size_t b) { return double(b) / 1e9; }
}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  uint32_t small = 16384, big = 201984;
  int pairs = 7;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--small")) small = uint32_t(std::atoi(argv[i + 1]));
    if (!std::strcmp(argv[i], "--big")) big = uint32_t(std::atoi(argv[i + 1]));
    if (!std::strcmp(argv[i], "--pairs")) pairs = std::atoi(argv[i + 1]);
  }
  l0::Context ctx(0);
  const model::ModelDesc& d = model::qwen38();
  const runtime::KvCache kv = runtime::KvCache::Bf16;

  runtime::PersistentBuffers ps(ctx, small, d, kv), pb(ctx, big, d, kv);
  runtime::MtpBuffers ms(ctx, small, d, 0, kv), mb(ctx, big, d, 0, kv);
  std::printf("max_len %u / %u, bf16 KV\n", small, big);
  std::printf("  persistent: control %zu  gdn_state %.3f GB  conv_ring %.3f GB  kv_k %.3f / %.3f GB  kv_v same\n",
              ps.control.size(), gb(ps.gdn_state.size()), gb(ps.conv_ring.size()), gb(ps.kv_k.size()),
              gb(pb.kv_k.size()));
  std::printf("  mtp: gdn_spec %.3f GB  kv_k %.3f / %.3f GB  hh %zu  dh %zu  logits %zu\n", gb(ms.gdn_spec.size()),
              gb(ms.kv_k.size()), gb(mb.kv_k.size()), ms.hh.size(), ms.dh.size(), ms.logits.size());
  std::printf("  reset() bytes: %.3f GB / %.3f GB\n", gb(ps.bytes() + ms.bytes()), gb(pb.bytes() + mb.bytes()));

  // The engine's list: CmdList::immediate(ctx) - ordinal 0.
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto arm = [&](runtime::PersistentBuffers& p, runtime::MtpBuffers& m) {
    Timer t;
    t.start();
    p.zero(imm);
    m.zero(imm);
    return t.ms();
  };
  for (int w = 0; w < 2; ++w) {
    arm(ps, ms);
    arm(pb, mb);
  }
  std::vector<double> a, b;
  for (int i = 0; i < pairs; ++i) {
    a.push_back(arm(ps, ms));
    b.push_back(arm(pb, mb));
  }
  std::printf("reset fills, ordinal 0 (the engine's): max_len %u %.2f ms, max_len %u %.2f ms (median of %d pairs)\n",
              small, median(a), big, median(b), pairs);
  std::printf("  -> %.1f GB/s on the big arm\n", gb(pb.bytes() + mb.bytes()) / (median(b) / 1e3));

  // Which engine is behind each ordinal, and one fill of the big kv_k on each.
  uint32_t ng = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(ctx.device(), &ng, nullptr));
  std::vector<ze_command_queue_group_properties_t> gp(ng);
  for (auto& g : gp) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(ctx.device(), &ng, gp.data()));
  for (uint32_t o = 0; o < ng; ++o) {
    const bool compute = gp[o].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    const bool copy = gp[o].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    std::printf("ordinal %u: %s%s, %u queues, max fill pattern %zu B\n", o, compute ? "compute " : "",
                copy ? "copy" : "", gp[o].numQueues, gp[o].maxMemoryFillPatternSize);
    l0::CmdList l = l0::CmdList::immediate(ctx, o);
    std::vector<double> t;
    try {
      for (int i = 0; i < pairs + 2; ++i) {
        Timer tm;
        tm.start();
        l.fill(pb.kv_k.ptr(), 0u, pb.kv_k.size());
        if (i >= 2) t.push_back(tm.ms());
      }
    } catch (const std::exception& e) {
      std::printf("  fill refused: %s\n", e.what());
      continue;
    }
    std::printf("  fill kv_k %.3f GB: %.2f ms, %.1f GB/s\n", gb(pb.kv_k.size()), median(t),
                gb(pb.kv_k.size()) / (median(t) / 1e3));
  }
  return 0;
}
