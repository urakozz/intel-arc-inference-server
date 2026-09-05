// T6 - direct M=2048 device-timestamp measurements for the largest widened
// small kernels. These probe-only variants retain prep.cl's production source
// and geometry; no runtime binding or tile parameter changes here.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {

constexpr uint32_t kM = 2048;
constexpr uint32_t kReplays = 8;
constexpr uint32_t kDropped = 3;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;
constexpr uint32_t kGateUp = 34816;
constexpr uint32_t kGatedOut = 6144;
constexpr uint32_t kFoldGroups = 20;

uint32_t xorshift(uint32_t& value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

// Incompressible, finite values avoid B70's lossless-compression timing trap.
void fill_f32(l0::CmdList& upload, void* dst, size_t bytes, uint32_t seed) {
  constexpr size_t kChunkWords = (16u << 20) / sizeof(uint32_t);
  std::vector<uint32_t> host(kChunkWords);
  for (size_t offset = 0; offset < bytes;) {
    for (uint32_t& word : host) word = (xorshift(seed) & 0x807fffffu) | 0x3e000000u;
    const size_t n = std::min(bytes - offset, host.size() * sizeof(uint32_t));
    upload.copy(static_cast<std::byte*>(dst) + offset, host.data(), n);
    offset += n;
  }
}

void fill_bf16(l0::CmdList& upload, void* dst, size_t bytes, uint32_t seed) {
  constexpr size_t kChunkWords = (16u << 20) / sizeof(uint16_t);
  std::vector<uint16_t> host(kChunkWords);
  for (size_t offset = 0; offset < bytes;) {
    for (uint16_t& word : host) word = uint16_t((xorshift(seed) & 0x807fu) | 0x3e00u);
    const size_t n = std::min(bytes - offset, host.size() * sizeof(uint16_t));
    upload.copy(static_cast<std::byte*>(dst) + offset, host.data(), n);
    offset += n;
  }
}

enum class Kind { ResFold, NormFinish, Silu, GatedHead };

struct Case {
  const char* binary;
  const char* entry;
  Kind kind;
  uint32_t wg;
  uint32_t gx;
  uint32_t calls_per_chunk;
};

double time_case(l0::Context& ctx, l0::Queue& queue, l0::Fence& fence, l0::Event& event,
                 const Case& c, l0::Mem& partials, l0::Mem& resid, l0::Mem& sumsq,
                 l0::Mem& norm_w, l0::Mem& x, l0::Mem& gdn_o, l0::Mem& gated_w) {
  l0::Module module(ctx, kernels::path(c.binary));
  l0::Kernel kernel = module.kernel(c.entry);
  kernel.group_size(c.wg);
  switch (c.kind) {
    case Kind::ResFold:
      kernel.arg_ptr(0, partials.ptr());
      kernel.arg_ptr(1, resid.ptr());
      kernel.arg_ptr(2, sumsq.ptr());
      break;
    case Kind::NormFinish:
      kernel.arg_ptr(0, sumsq.ptr());
      kernel.arg_ptr(1, resid.ptr());
      kernel.arg_ptr(2, norm_w.ptr());
      kernel.arg_ptr(3, x.ptr());
      break;
    case Kind::Silu:
      kernel.arg_ptr(0, partials.ptr());
      kernel.arg_ptr(1, x.ptr());
      break;
    case Kind::GatedHead:
      kernel.arg_ptr(0, partials.ptr());
      kernel.arg_ptr(1, gdn_o.ptr());
      kernel.arg_ptr(2, gated_w.ptr());
      kernel.arg_ptr(3, x.ptr());
      break;
  }

  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(kernel, c.gx, kM, 1, &event);
  list.close();
  std::array<double, kReplays> samples{};
  for (double& sample : samples) {
    event.reset();
    queue.execute(list, &fence);
    fence.wait();
    sample = event.duration_us();
  }
  std::sort(samples.begin() + kDropped, samples.end());
  return samples[kDropped + (kReplays - kDropped) / 2];
}

}  // namespace

int main() {
  try {
    l0::Context ctx(0);
    l0::Queue queue(ctx);
    l0::Fence fence(queue);
    l0::CmdList upload = l0::CmdList::immediate(ctx);

    // The largest reader is gate||up's eight fp32 split-K partial planes.
    l0::Mem partials(ctx, l0::MemKind::Device, size_t(8) * kM * kGateUp * sizeof(float));
    l0::Mem resid(ctx, l0::MemKind::Device, size_t(kM) * kHidden * sizeof(uint16_t));
    l0::Mem sumsq(ctx, l0::MemKind::Device, size_t(kFoldGroups) * kM * sizeof(float));
    l0::Mem norm_w(ctx, l0::MemKind::Device, size_t(kHidden) * sizeof(float));
    l0::Mem x(ctx, l0::MemKind::Device, size_t(kM) * kIntermediate * sizeof(uint16_t));
    l0::Mem gdn_o(ctx, l0::MemKind::Device, size_t(kM) * kGatedOut * sizeof(float));
    l0::Mem gated_w(ctx, l0::MemKind::Device, size_t(kGatedOut) * sizeof(uint16_t));
    fill_f32(upload, partials.ptr(), partials.size(), 0x9e3779b9u);
    fill_bf16(upload, resid.ptr(), resid.size(), 0x85ebca6bu);
    fill_f32(upload, norm_w.ptr(), norm_w.size(), 0xc2b2ae35u);
    fill_f32(upload, gdn_o.ptr(), gdn_o.size(), 0x27d4eb2fu);
    fill_bf16(upload, gated_w.ptr(), gated_w.size(), 0x165667b1u);

    l0::EventPool pool(ctx, 1);
    l0::Event event(pool, 0);
    const Case cases[] = {
        {"pfs_res_fold_sp4_M2048", "prep_res_fold", Kind::ResFold, 256, kFoldGroups, 129},
        {"pfs_norm_finish_M2048", "prep_norm_finish", Kind::NormFinish, 256, kFoldGroups, 129},
        {"pfs_silu_mul_M2048", "prep_silu_mul", Kind::Silu, 256, 5, 64},
        {"pfs_gated_head_M2048", "prep_gated_head", Kind::GatedHead, 128, 48, 48},
    };

    std::printf("# T6 direct widened-small-kernel measurement: device timestamps, M=2048, %u replays, first %u dropped; inputs are finite incompressible random data.\n", kReplays, kDropped);
    std::printf("# These probe-only M=2048 binaries retain prep.cl's production source and launch geometry; no runtime path loads them.\n");
    std::printf("| kernel | grid work-groups | us/launch | calls/chunk | ms/chunk |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    for (const Case& c : cases) {
      const double us = time_case(ctx, queue, fence, event, c, partials, resid, sumsq, norm_w, x, gdn_o,
                                  gated_w);
      std::printf("| %s | %u x %u = %u | %.3f | %u | %.3f |\n", c.entry, c.gx, kM, c.gx * kM,
                  us, c.calls_per_chunk, us * c.calls_per_chunk / 1000.0);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "probe_prefill_small FAILED: %s\n", error.what());
    return 1;
  }
}
