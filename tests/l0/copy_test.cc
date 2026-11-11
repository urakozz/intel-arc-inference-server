// Round-trips 1 MiB host -> device -> host through the immediate list.
//
// B70_COPY_STRESS=<seconds> (2026-11-11, the snapshot-copy hangs: 100 % CPU inside the xe VM_BIND ioctl under
// l0::CmdList::copy, kernel 7.4.0-rc2 / runtime 26.40.40196.15): afterwards, for that long, the copy pattern of a
// prefix snapshot - 1..64 MiB each way, from a freshly malloc'd buffer (a new host-pointer bind each time) and from
// pinned host memory - with a watchdog: a copy that takes over 20 s prints COPY_HANG and exits 2 instead of hanging.
// B70_COPY_STRESS_PINNED_GB=<n>: the pinned buffer is n GiB, like the server's prefix cache (16-32 GiB), and each pinned
// copy walks to the next 64 MiB of it.
// The device is ZE_AFFINITY_MASK's (run under tools/gpu_lock.sh card N).
//
// B70_COPY_PINNED_GB=<n> (run under tools/gpu_lock.sh box): instead, the prefix cache's first use on both cards - ONE
// context holding cards 0 and 1 (the server's), n GiB of l0::MemKind::Host memory written by the CPU, then a 1 MiB copy
// out of the end of every allocation on card 0 and then on card 1. B70_COPY_PINNED_PIECE_GB=<p>: the n GiB as p GiB
// allocations (default one). B70_COPY_PINNED_FILL_GB=<f>: f GiB of device memory allocated and written on each card
// first. B70_COPY_PINNED_RESIDENT=1: zeContextMakeMemoryResident on card 0 then card 1 before the copies. A step that
// takes over B70_COPY_PINNED_WATCHDOG seconds (default 120) prints COPY_HANG and exits 2.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/error.h"
#include "l0/memory.h"

namespace {

using Clock = std::chrono::steady_clock;

int stress(l0::Context& ctx, l0::CmdList& imm, double seconds) {
  constexpr size_t kMax = size_t(64) << 20;
  l0::Mem dev(ctx, l0::MemKind::Device, kMax);
  const char* gb = std::getenv("B70_COPY_STRESS_PINNED_GB");
  const size_t pinned_bytes = gb ? size_t(std::atof(gb) * double(size_t(1) << 30)) / kMax * kMax : kMax;
  l0::Mem pinned(ctx, l0::MemKind::Host, pinned_bytes);
  std::printf("copy stress: pinned buffer %.1f GiB\n", pinned_bytes / double(size_t(1) << 30));
  // B70_COPY_STRESS_FILL_GB=<n>: n GiB of device memory allocated and written first, the card as full as a server's
  std::vector<l0::Mem> fill;
  if (const char* f = std::getenv("B70_COPY_STRESS_FILL_GB")) {
    for (int g = 0; g < std::atoi(f); ++g) {
      fill.emplace_back(ctx, l0::MemKind::Device, size_t(1) << 30);
      imm.fill(fill.back().ptr(), 0x5A5A5A5Au, size_t(1) << 30);
    }
    std::printf("copy stress: %zu GiB of device memory filled\n", fill.size());
  }
  std::atomic<int64_t> started{0};   // ns since t0 of the copy in flight, -1 between copies
  std::atomic<uint64_t> iter{0};
  std::atomic<bool> done{false};
  const Clock::time_point t0 = Clock::now();
  const auto ns = [&] { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count(); };
  started = -1;
  std::thread watchdog([&] {
    while (!done) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const int64_t s = started;
      if (s >= 0 && ns() - s > 20'000'000'000) {
        std::printf("COPY_HANG: copy %llu has not returned after 20 s (%.0f s into the run)\n",
                    static_cast<unsigned long long>(iter.load()), ns() / 1e9);
        std::fflush(stdout);
        std::_Exit(2);
      }
    }
  });
  uint64_t bytes = 0;
  double last = 0;
  for (uint64_t i = 0; ns() < seconds * 1e9; ++i) {
    const size_t n = size_t(1) << (20 + i % 7);   // 1, 2, 4 .. 64 MiB
    const bool fresh = i % 2 == 0;
    std::vector<uint8_t> heap;
    uint8_t* host = pinned.as<uint8_t>() + (i / 2 * kMax) % pinned_bytes;
    if (fresh) {
      heap.resize(n);
      host = heap.data();
    }
    host[0] = static_cast<uint8_t>(i);
    host[n - 1] = static_cast<uint8_t>(i * 3);
    iter = i;
    started = ns();
    imm.copy(dev.ptr(), host, n);
    host[0] = host[n - 1] = 0;
    imm.copy(host, dev.ptr(), n);
    started = -1;
    CHECK_EQ(host[0], static_cast<uint8_t>(i));
    CHECK_EQ(host[n - 1], static_cast<uint8_t>(i * 3));
    bytes += 2 * n;
    if (ns() / 1e9 - last >= 10) {
      last = ns() / 1e9;
      std::printf("copy stress: %.0f s, %llu round trips, %.1f GB\n", last, static_cast<unsigned long long>(i + 1),
                  bytes / 1e9);
      std::fflush(stdout);
    }
  }
  done = true;
  watchdog.join();
  std::printf("copy stress OK: %llu round trips, %.1f GB in %.0f s\n", static_cast<unsigned long long>(iter + 1),
              bytes / 1e9, ns() / 1e9);
  return 0;
}

int pinned_probe(size_t gb) {
  const char* piece_env = std::getenv("B70_COPY_PINNED_PIECE_GB");
  const char* wd_env = std::getenv("B70_COPY_PINNED_WATCHDOG");
  const size_t piece = (piece_env ? std::strtoull(piece_env, nullptr, 10) : gb) << 30;
  const int watchdog_s = wd_env ? std::atoi(wd_env) : 120;
  CHECK(piece > 0);
  CHECK(l0::Context::gpu_count() >= 2);
  l0::Context c0(0);
  l0::Context c1(c0, 1);

  const Clock::time_point t0 = Clock::now();
  const auto secs = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  std::atomic<double> started{-1};   // seconds since t0 of the copy in flight, -1 between copies
  std::thread([&] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const double s = started;
      if (s >= 0 && secs() - s > watchdog_s) {
        std::printf("COPY_HANG: a first copy has not returned after %d s\n", watchdog_s);
        std::fflush(stdout);
        std::_Exit(2);
      }
    }
  }).detach();

  std::vector<l0::Mem> fill;
  if (const char* f = std::getenv("B70_COPY_PINNED_FILL_GB")) {
    for (l0::Context* c : {&c0, &c1}) {
      l0::CmdList imm = l0::CmdList::immediate(*c);
      for (int g = 0; g < std::atoi(f); ++g) {
        fill.emplace_back(*c, l0::MemKind::Device, size_t(1) << 30);
        imm.fill(fill.back().ptr(), 0x5A5A5A5Au, size_t(1) << 30);
      }
    }
    std::printf("pinned probe: %zu GiB of device memory filled over both cards\n", fill.size());
  }

  std::vector<std::unique_ptr<l0::Mem>> pieces;
  for (size_t left = gb << 30; left > 0;) {
    const size_t n = left < piece ? left : piece;
    pieces.push_back(std::make_unique<l0::Mem>(c0, l0::MemKind::Host, n, size_t(64) << 10));
    std::memset(pieces.back()->ptr(), 0x5A, n);
    left -= n;
  }
  std::printf("pinned probe: %zu GiB as %zu allocation(s), allocated and written in %.1f s\n", gb, pieces.size(),
              secs());
  std::fflush(stdout);

  if (std::getenv("B70_COPY_PINNED_RESIDENT")) {
    for (l0::Context* c : {&c0, &c1}) {
      for (size_t i = 0; i < pieces.size(); ++i) {
        started = secs();
        ZE_CHECK(zeContextMakeMemoryResident(c0.handle(), c->device(), pieces[i]->ptr(), pieces[i]->size()));
        std::printf("pinned probe: card %d allocation %zu made resident in %.3f s\n", c == &c0 ? 0 : 1, i,
                    secs() - started);
        std::fflush(stdout);
        started = -1;
      }
    }
  }

  const size_t n = 1u << 20;
  for (l0::Context* c : {&c0, &c1}) {
    l0::Mem dev(*c, l0::MemKind::Device, n);
    l0::CmdList imm = l0::CmdList::immediate(*c);
    std::vector<uint8_t> back(n);
    for (size_t i = 0; i < pieces.size(); ++i) {
      started = secs();
      imm.copy(dev.ptr(), pieces[i]->as<uint8_t>() + pieces[i]->size() - n, n);
      imm.copy(back.data(), dev.ptr(), n);
      std::printf("pinned probe: card %d allocation %zu first copy %.3f s\n", c == &c0 ? 0 : 1, i, secs() - started);
      std::fflush(stdout);
      started = -1;
      CHECK_EQ(back[0], uint8_t(0x5A));
      CHECK_EQ(back[n - 1], uint8_t(0x5A));
    }
  }
  std::puts("copy_test pinned probe OK");
  return 0;
}

}  // namespace

int main() {
  if (const char* gb = std::getenv("B70_COPY_PINNED_GB")) return pinned_probe(std::strtoull(gb, nullptr, 10));
  l0::Context ctx(0);
  std::printf("device: %s, EUs: %u\n", ctx.name().c_str(), ctx.eu_count());
  CHECK(ctx.eu_count() > 0);

  const size_t n = 1u << 20;
  std::vector<uint8_t> src(n), dst(n, 0);
  for (size_t i = 0; i < n; ++i) src[i] = static_cast<uint8_t>(i * 7 + 3);

  l0::Mem dev(ctx, l0::MemKind::Device, n);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(dev.ptr(), src.data(), n);
  imm.copy(dst.data(), dev.ptr(), n);
  CHECK_EQ(std::memcmp(src.data(), dst.data(), n), 0);

  l0::Mem shared(ctx, l0::MemKind::Shared, 64);
  imm.fill(shared.ptr(), 0xA5A5A5A5u, 64);
  CHECK_EQ(shared.as<uint32_t>()[0], 0xA5A5A5A5u);
  CHECK_EQ(shared.as<uint32_t>()[15], 0xA5A5A5A5u);
  std::puts("copy_test OK");
  if (const char* s = std::getenv("B70_COPY_STRESS")) return stress(ctx, imm, std::atof(s));
  return 0;
}
