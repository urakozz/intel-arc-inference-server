#include "l0/memory.h"
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include "l0/error.h"

namespace l0 {
namespace {
std::atomic<size_t> g_locked{0}, g_unlocked{0};
std::atomic<bool> g_lock{false};

// HostLocked: `bytes` (rounded up to pages) of anonymous memory at `align`, mlock()ed, or null when mlock() is refused.
void* locked_pages(size_t bytes, size_t align, size_t& mapped) {
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  if (align < page) align = page;
  mapped = (bytes + page - 1) / page * page;
  // Over-map by `align` and trim both ends: mmap gives only page alignment.
  void* raw = mmap(nullptr, mapped + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) return nullptr;
  auto* lo = static_cast<uint8_t*>(raw);
  auto* p = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(lo) + align - 1) / align * align);
  if (p > lo) munmap(lo, size_t(p - lo));
  munmap(p + mapped, size_t(lo + align - p));
  madvise(p, mapped, MADV_HUGEPAGE);   // 2 MiB pages where the kernel has them: fewer to lock and for the device to map
  if (mlock(p, mapped) != 0) {
    munmap(p, mapped);
    return nullptr;
  }
  return p;
}
}  // namespace

size_t memlock_limit() {
  rlimit r{};
  if (getrlimit(RLIMIT_MEMLOCK, &r) != 0) return 0;
  return r.rlim_cur == RLIM_INFINITY ? SIZE_MAX : size_t(r.rlim_cur);
}

bool lock_host_memory(size_t planned) {
  rlimit r{};   // the soft limit up to the hard one first: any process may
  if (getrlimit(RLIMIT_MEMLOCK, &r) == 0 && r.rlim_cur != r.rlim_max) {
    r.rlim_cur = r.rlim_max;
    setrlimit(RLIMIT_MEMLOCK, &r);
  }
  const size_t limit = memlock_limit();
  g_lock = limit == SIZE_MAX || (planned <= limit && g_locked + planned <= limit);
  return g_lock;
}

size_t host_locked_bytes() { return g_locked; }
size_t host_unlocked_bytes() { return g_unlocked; }

Mem::Mem(Context& ctx, MemKind kind, size_t bytes, size_t align)
    : ctx_(&ctx), kind_(kind), bytes_(bytes) {
  ze_device_mem_alloc_desc_t dd{};
  dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t hd{};
  hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  // Allocations above the device's maxMemAllocSize need the relaxed-limits
  // extension; the 8 GB bandwidth probe and the 2.5 GB lm_head both do. It goes
  // on whichever descriptor the alloc call takes, so chain it onto both: Host
  // allocations only see hd, Device only dd, Shared reads it once from either.
  ze_relaxed_allocation_limits_exp_desc_t relaxed{};
  relaxed.stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
  relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
  if (bytes > ctx.props().maxMemAllocSize) {
    dd.pNext = &relaxed;
    hd.pNext = &relaxed;
  }

  switch (kind) {
    case MemKind::Device:
      ZE_CHECK(zeMemAllocDevice(ctx.handle(), &dd, bytes, align, ctx.device(), &ptr_));
      break;
    case MemKind::Host:
      ZE_CHECK(zeMemAllocHost(ctx.handle(), &hd, bytes, align, &ptr_));
      break;
    case MemKind::Shared:
      ZE_CHECK(zeMemAllocShared(ctx.handle(), &dd, &hd, bytes, align, ctx.device(), &ptr_));
      break;
    case MemKind::HostLocked: {
      size_t mapped = 0;
      sys_ = g_lock ? locked_pages(bytes, align, mapped) : nullptr;
      if (sys_) {
        ze_external_memmap_sysmem_ext_desc_t ext{};
        ext.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC;
        ext.pNext = hd.pNext;
        ext.pSystemMemory = sys_;
        ext.size = mapped;
        hd.pNext = &ext;
        if (zeMemAllocHost(ctx.handle(), &hd, mapped, align, &ptr_) == ZE_RESULT_SUCCESS) {
          sys_bytes_ = mapped;
          g_locked += bytes_;
          break;
        }
        // The runtime would not map it: plain Host memory, as when the limit refuses the lock.
        hd.pNext = ext.pNext;
        munmap(sys_, mapped);
        sys_ = nullptr;
      }
      ZE_CHECK(zeMemAllocHost(ctx.handle(), &hd, bytes, align, &ptr_));
      g_unlocked += bytes_;
      break;
    }
  }
}

Mem::~Mem() {
  if (!ptr_) return;
  zeMemFree(ctx_->handle(), ptr_);
  if (kind_ != MemKind::HostLocked) return;
  if (sys_) {
    munmap(sys_, sys_bytes_);
    g_locked -= bytes_;
  } else {
    g_unlocked -= bytes_;
  }
}

Mem::Mem(Mem&& o) noexcept
    : ctx_(o.ctx_), kind_(o.kind_), bytes_(o.bytes_), ptr_(o.ptr_), sys_(o.sys_), sys_bytes_(o.sys_bytes_) {
  o.ptr_ = nullptr;
  o.sys_ = nullptr;
}
}  // namespace l0
