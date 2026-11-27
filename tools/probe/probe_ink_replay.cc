// Inkling one card: what a replayed decode list spends outside its kernels.
//   probe_ink_replay SNAPSHOT [depth=4096] [reps=200]
// After a `depth`-id prompt (synthetic ids), drafter on, bf16 cache. For the plain step's list (append's) and the
// 8-row verify's list (verify's), each replayed alone at one position, medians over `reps`:
//   host      execute + synchronize, wall clock
//   device    the list's span on the card: one execution of [stamp, list, stamp], the stamps' own pair subtracted
//             (two one-kernel lists with a timestamp event each; nothing inside the list is marked)
//   noop      the same span for a list of as many empty kernels: the launch floor any replay pays a kernel
// host - device is the submission and the wake-up; noop / launches is the per-kernel cost a recorded list leaves.
// Also prints the driver's extensions (mutable command lists, immutable lists) and its API version.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernels/kernels.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "runtime/inkling2/inkling2_engine.h"

namespace runtime::inkling2 {
struct ReplayProbeAccess {
  static l0::CmdList& plain(Engine& e) { return *e.plain_; }
  static l0::CmdList& target(Engine& e, uint32_t rows) { return *e.target_[0][0][rows - 1]; }
  static l0::Queue& queue(Engine& e) { return e.queue_; }
  static size_t kernels(const Engine& e) { return e.kernels_.size(); }
};
}  // namespace runtime::inkling2

namespace {
namespace ink = runtime::inkling2;
using A = ink::ReplayProbeAccess;
using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Stamps {
  l0::EventPool pool;
  l0::Event first, last;
  l0::Module module;
  l0::Kernel kernel;
  l0::CmdList a, b;
  l0::Fence fence;
  Stamps(l0::Context& ctx, l0::Queue& q, l0::Mem& out)
      : pool(ctx, 2), first(pool, 0), last(pool, 1), module(ctx, kernels::path("noop")), kernel(module, "noop"),
        a(l0::CmdList::regular(ctx)), b(l0::CmdList::regular(ctx)), fence(q) {
    kernel.arg_ptr(0, out.ptr());
    kernel.group_size(16);
    a.launch(kernel, 1, 1, 1, &first);
    a.close();
    b.launch(kernel, 1, 1, 1, &last);
    b.close();
  }
  // The card's time from the first stamp's start to the last stamp's end, `list` (or nothing) between them.
  double span_us(l0::Queue& q, l0::CmdList* list) {
    first.reset();
    last.reset();
    std::vector<ze_command_list_handle_t> lists{a.handle()};
    if (list) lists.push_back(list->handle());
    lists.push_back(b.handle());
    q.execute(uint32_t(lists.size()), lists.data(), &fence);
    fence.wait();
    return last.elapsed_us(first);
  }
};

void report(const char* what, size_t launches, l0::Queue& q, l0::CmdList& list, l0::CmdList& noops, Stamps& s,
            uint32_t reps) {
  std::vector<double> host, device, empty, floor;
  for (uint32_t i = 0; i < reps + 10; ++i) {
    const auto t0 = Clock::now();
    q.execute(list);
    q.synchronize();
    const double h = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    const double d = s.span_us(q, &list), e = s.span_us(q, nullptr), n = s.span_us(q, &noops);
    if (i < 10) continue;
    host.push_back(h), device.push_back(d), empty.push_back(e), floor.push_back(n);
  }
  const double h = median(host), e = median(empty), d = median(device) - e, n = median(floor) - e;
  std::printf("%-12s %5zu launches: host %8.1f us, device %8.1f us, host - device %6.1f us (%.2f %%); "
              "%zu empty kernels %6.1f us = %.3f us a launch (%.2f %% of the device span)\n",
              what, launches, h, d, h - d, 100 * (h - d) / h, launches, n, n / launches, 100 * n / d);
}

std::vector<uint32_t> prompt(uint32_t n) {
  std::vector<uint32_t> ids(n);
  for (uint32_t i = 0; i < n; ++i) ids[i] = 1000 + (i * 7919u) % 100000u;
  return ids;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe_ink_replay SNAPSHOT [depth] [reps]\n");
    return 2;
  }
  const uint32_t depth = argc > 2 ? uint32_t(std::atoi(argv[2])) : 4096, reps = argc > 3 ? uint32_t(std::atoi(argv[3])) : 200;
  const uint32_t max_len = (depth + 64 + 4096) / 4096 * 4096, chunk = 2048;
  l0::Context ctx(0);
  {
    ze_api_version_t version{};
    zeDriverGetApiVersion(ctx.driver(), &version);
    uint32_t count = 0;
    zeDriverGetExtensionProperties(ctx.driver(), &count, nullptr);
    std::vector<ze_driver_extension_properties_t> ext(count);
    zeDriverGetExtensionProperties(ctx.driver(), &count, ext.data());
    std::printf("driver API %u.%u; extensions:", ZE_MAJOR_VERSION(version), ZE_MINOR_VERSION(version));
    for (const auto& e : ext)
      if (std::strstr(e.name, "command_list") || std::strstr(e.name, "graph") || std::strstr(e.name, "immutable"))
        std::printf(" %s %u.%u", e.name, ZE_MAJOR_VERSION(e.version), ZE_MINOR_VERSION(e.version));
    std::printf("\n");
  }
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  ink::Engine engine(ctx, checkpoint, max_len, true, chunk);
  uint32_t next = engine.prefill(prompt(depth), chunk);
  size_t before = A::kernels(engine);
  for (uint32_t i = 0; i < 4; ++i) next = engine.append(next);
  const size_t plain = A::kernels(engine) - before;
  std::vector<uint32_t> rows(8, next);
  before = A::kernels(engine);
  engine.verify(rows);
  const size_t verify = A::kernels(engine) - before;
  engine.commit(1);

  auto& q = A::queue(engine);
  l0::Mem out(ctx, l0::MemKind::Device, 4096);
  Stamps stamps(ctx, q, out);
  l0::Module module(ctx, kernels::path("noop"));
  l0::Kernel noop(module, "noop");
  noop.arg_ptr(0, out.ptr());
  noop.group_size(16);
  const auto noops = [&](size_t n) {
    auto list = l0::CmdList::regular(ctx);
    for (size_t i = 0; i < n; ++i) list.launch(noop, 1);
    list.close();
    return list;
  };
  auto plain_noops = noops(plain), verify_noops = noops(verify);
  std::printf("depth %u, %u replays a row (medians)\n", depth, reps);
  report("plain step", plain, q, A::plain(engine), plain_noops, stamps, reps);
  report("8-row verify", verify, q, A::target(engine, 8), verify_noops, stamps, reps);
  return 0;
}
