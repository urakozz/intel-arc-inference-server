// l0::MemKind::HostLocked: host memory locked against swap when RLIMIT_MEMLOCK covers it.
//  1. Why it is not mlock() on a zeMemAllocHost range: the call succeeds and locks nothing (VmLck unchanged) - the
//     driver's mapping is VM_PFNMAP|VM_IO, which mlock skips, and the xe/TTM shrinker swaps those pages regardless.
//  2. Under a covering limit (4 MiB fits the 8 MiB distro default), a HostLocked allocation is locked (VmLck grows by
//     its size) and works like Host memory on every card of the one context: a copy out of it, and a kernel reading
//     it through a pointer table (indirect access, the PLE gather's way).
//  3. Under a limit that does not cover the plan, HostLocked is plain Host memory and nothing is locked.
// Both cards when two are visible (nothing else on them), else card 0.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <vector>
#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"

namespace {
size_t vm_lck_kb() {
  std::ifstream f("/proc/self/status");
  for (std::string line; std::getline(f, line);)
    if (line.rfind("VmLck:", 0) == 0) return std::stoul(line.substr(6));
  return 0;
}
}  // namespace

int main() {
  constexpr size_t kBytes = size_t(4) << 20;
  const uint32_t cards = l0::Context::gpu_count() >= 2 ? 2 : 1;
  l0::Context c0(0);
  std::unique_ptr<l0::Context> c1;
  if (cards == 2) c1 = std::make_unique<l0::Context>(c0, 1);

  // 1. mlock() on the driver's host allocation: accepted, nothing locked.
  {
    l0::Mem plain(c0, l0::MemKind::Host, kBytes);
    const size_t before = vm_lck_kb();
    CHECK_EQ(mlock(plain.ptr(), kBytes), 0);
    std::printf("mlock on a zeMemAllocHost range: VmLck %zu -> %zu kB\n", before, vm_lck_kb());
    CHECK_EQ(vm_lck_kb(), before);
    munlock(plain.ptr(), kBytes);
  }

  // 2. A covering limit: locked, and read by every card.
  CHECK(l0::lock_host_memory(kBytes));
  const size_t before = vm_lck_kb();
  l0::Mem locked(c0, l0::MemKind::HostLocked, kBytes, size_t(2) << 20);
  std::printf("HostLocked %zu MiB: locked %d, VmLck %zu -> %zu kB\n", kBytes >> 20, int(locked.locked()), before,
              vm_lck_kb());
  CHECK(locked.locked());
  CHECK_EQ(reinterpret_cast<uintptr_t>(locked.ptr()) % (size_t(2) << 20), uintptr_t(0));
  CHECK(vm_lck_kb() >= before + (kBytes >> 10));
  CHECK_EQ(l0::host_locked_bytes(), kBytes);
  auto* w = locked.as<uint32_t>();
  for (size_t i = 0; i < kBytes / 4; ++i) w[i] = uint32_t(i * 2654435761u);

  for (uint32_t card = 0; card < cards; ++card) {
    l0::Context& c = card == 0 ? c0 : *c1;
    l0::CmdList imm = l0::CmdList::immediate(c);
    l0::Mem dev(c, l0::MemKind::Device, kBytes);
    std::vector<uint32_t> back(kBytes / 4);
    imm.copy(dev.ptr(), locked.ptr(), kBytes);
    imm.copy(back.data(), dev.ptr(), kBytes);
    CHECK_EQ(std::memcmp(back.data(), w, kBytes), 0);

    l0::Module mod(c, kernels::path("host_read"));
    l0::Kernel k = mod.kernel("host_read");
    l0::Mem table(c, l0::MemKind::Shared, 64);
    l0::Mem out(c, l0::MemKind::Shared, 16 * 4);
    table.as<uint64_t>()[0] = reinterpret_cast<uint64_t>(locked.ptr());
    k.arg_ptr(0, table.ptr());
    k.arg_ptr(1, out.ptr());
    k.indirect_host();
    k.group_size(16);
    imm.launch(k, 1);
    for (uint32_t i = 0; i < 16; ++i) CHECK_EQ(out.as<uint32_t>()[i], w[i * 1024]);
    std::printf("card %u: copy and indirect kernel read of the locked memory OK\n", card);
  }

  // 3. A limit that does not cover the plan: plain Host memory, as before.
  if (l0::memlock_limit() != SIZE_MAX) {
    CHECK(!l0::lock_host_memory(l0::memlock_limit() + 1));
    l0::Mem plain(c0, l0::MemKind::HostLocked, kBytes);
    CHECK(!plain.locked());
    CHECK_EQ(l0::host_unlocked_bytes(), kBytes);
    CHECK_EQ(l0::host_locked_bytes(), kBytes);
  }
  std::puts("host_lock_test OK");
  return 0;
}
