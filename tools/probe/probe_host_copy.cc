// Spec 7 P0: pinned host memory on the T5810 - the largest l0 Host allocation the
// driver gives, and device<->host copy bandwidth on an immediate list for the
// shapes the prefix cache moves: one KV block (128 MiB), one state snapshot
// (gdn_state 144 MiB + conv_ring 15 MiB, two copies), and a KV position range as
// it really lies (16 layers x K and V, one contiguous [n][4][256] bf16 slice each).
//   probe_host_copy [--max-gb N]
// Host RAM is shared with CPU oracle runs: allocations above MemAvailable - 16 GiB
// are not attempted; the cap is printed. Median of 5 after 2 warm-ups.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "timer.h"

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t kMiB = 1024 * 1024;

uint64_t mem_available_bytes() {
  std::ifstream in("/proc/meminfo");
  std::string key;
  uint64_t kb = 0;
  std::string unit;
  while (in >> key >> kb >> unit) {
    if (key == "MemAvailable:") return kb * 1024;
  }
  return 0;
}

template <class F>
double median_ms(F&& f, int warm = 2, int reps = 5) {
  for (int i = 0; i < warm; ++i) f();
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    Timer timer;
    timer.start();
    f();
    t.push_back(timer.ms());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

double gbps(size_t bytes, double ms) { return bytes / (ms * 1e6); }

}  // namespace

int main(int argc, char** argv) {
  double max_gb = 48;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--max-gb" && i + 1 < argc) max_gb = std::atof(argv[++i]);
  }
  l0::Context ctx(0);
  l0::CmdList list = l0::CmdList::immediate(ctx);
  const uint64_t avail = mem_available_bytes();
  const double cap_gib = avail / kGiB - 16.0;
  std::printf("# probe_host_copy: device %s, MemAvailable %.1f GiB, cap %.1f GiB\n",
              ctx.name().c_str(), avail / kGiB, cap_gib);

  // 1. The largest pinned host allocation, touched page by page and read by the device.
  std::printf("\n| host alloc GiB | alloc ms | touch ms | device reads its tail | result |\n");
  std::printf("|---|---|---|---|---|\n");
  double largest = 0;
  {
    l0::Mem probe_dev(ctx, l0::MemKind::Device, 128 * kMiB);
    for (double gib : {4.0, 8.0, 16.0, 24.0, 32.0, 48.0}) {
      if (gib > max_gb) break;
      if (gib > cap_gib) {
        std::printf("| %.0f | - | - | - | not attempted (over cap %.1f) |\n", gib, cap_gib);
        break;
      }
      const size_t bytes = static_cast<size_t>(gib * kGiB);
      try {
        Timer t;
        t.start();
        l0::Mem host(ctx, l0::MemKind::Host, bytes, 4096);
        const double alloc_ms = t.ms();
        t.start();
        auto* p = host.as<unsigned char>();
        for (size_t off = 0; off < bytes; off += 4096) p[off] = static_cast<unsigned char>(off >> 12);
        const double touch_ms = t.ms();
        list.copy(probe_dev.ptr(), p + bytes - 128 * kMiB, 128 * kMiB);
        unsigned char back = 0;
        list.copy(&back, static_cast<unsigned char*>(probe_dev.ptr()) + 128 * kMiB - 4096, 1);
        const bool ok = back == p[bytes - 4096];
        std::printf("| %.0f | %.0f | %.0f | %s | %s |\n", gib, alloc_ms, touch_ms, ok ? "yes" : "NO",
                    ok ? "ok" : "readback mismatch");
        if (!ok) break;
        largest = gib;
      } catch (const std::exception& e) {
        std::printf("| %.0f | - | - | - | FAILED: %s |\n", gib, e.what());
        break;
      }
    }
  }
  std::printf("\nlargest pinned host allocation: %.0f GiB\n", largest);

  // 2. Bandwidth, one block and one snapshot.
  const size_t kBlock = 128 * kMiB;
  const size_t kGdn = 144 * kMiB, kConv = 15 * kMiB;
  l0::Mem dev(ctx, l0::MemKind::Device, kGdn + kConv);
  l0::Mem host(ctx, l0::MemKind::Host, kGdn + kConv, 4096);
  std::memset(host.ptr(), 1, host.size());
  list.fill(dev.ptr(), 0x3f803f80u, dev.size());
  auto* h = host.as<unsigned char>();
  auto* d = dev.as<unsigned char>();
  std::printf("\n| shape | bytes | D2H ms | D2H GB/s | H2D ms | H2D GB/s |\n|---|---|---|---|---|---|\n");
  {
    const double d2h = median_ms([&] { list.copy(h, d, kBlock); });
    const double h2d = median_ms([&] { list.copy(d, h, kBlock); });
    std::printf("| KV block (1 copy) | %zu | %.2f | %.2f | %.2f | %.2f |\n", kBlock, d2h,
                gbps(kBlock, d2h), h2d, gbps(kBlock, h2d));
  }
  {
    const size_t bytes = kGdn + kConv;
    const double d2h = median_ms([&] {
      list.copy(h, d, kGdn);
      list.copy(h + kGdn, d + kGdn, kConv);
    });
    const double h2d = median_ms([&] {
      list.copy(d, h, kGdn);
      list.copy(d + kGdn, h + kGdn, kConv);
    });
    std::printf("| state snapshot (2 copies) | %zu | %.2f | %.2f | %.2f | %.2f |\n", bytes, d2h,
                gbps(bytes, d2h), h2d, gbps(bytes, h2d));
  }

  // 3. A KV position range: 16 FA layers x (K, V), one [n][4][256] bf16 slice each.
  const size_t kRow = 4 * 256 * 2;
  const size_t kMaxN = 60000;
  std::vector<std::unique_ptr<l0::Mem>> layers;
  for (int i = 0; i < 32; ++i) {
    layers.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kMaxN * kRow));
    list.fill(layers.back()->ptr(), 0x3f803f80u, kMaxN * kRow);
  }
  l0::Mem range_host(ctx, l0::MemKind::Host, 32 * kMaxN * kRow, 4096);
  std::memset(range_host.ptr(), 1, range_host.size());
  auto* rh = range_host.as<unsigned char>();
  for (size_t n : {size_t(2048), kMaxN}) {
    const size_t slice = n * kRow;
    const size_t bytes = 32 * slice;
    const double d2h = median_ms([&] {
      for (int i = 0; i < 32; ++i) list.copy(rh + i * slice, layers[i]->ptr(), slice);
    });
    const double h2d = median_ms([&] {
      for (int i = 0; i < 32; ++i) list.copy(layers[i]->ptr(), rh + i * slice, slice);
    });
    std::printf("| KV range n=%zu (32 copies) | %zu | %.2f | %.2f | %.2f | %.2f |\n", n, bytes, d2h,
                gbps(bytes, d2h), h2d, gbps(bytes, h2d));
  }
  return 0;
}
