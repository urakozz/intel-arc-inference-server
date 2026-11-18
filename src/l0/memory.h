#pragma once
#include <cstddef>
#include <utility>
#include "l0/context.h"

namespace l0 {
enum class MemKind { Device, Host, Shared, HostLocked };

// One allocation. Device memory is not host-accessible; Host memory is
// host-resident and device-visible; Shared migrates (used for the control block).
// HostLocked is Host memory that cannot be swapped out, when RLIMIT_MEMLOCK covers it:
// mlock() on a zeMemAllocHost range does nothing (the driver's mapping is VM_PFNMAP|VM_IO,
// which mlock skips) and the xe/TTM shrinker backs such pages up to swap under memory
// pressure. So it is our own anonymous memory, mlock()ed, mapped for the devices with
// zeMemAllocHost's external-system-memory extension (ze_external_memmap_sysmem_ext_desc_t).
// When the limit does not cover it (lock_host_memory) it is a plain Host allocation, as before.
class Mem {
 public:
  Mem(Context& ctx, MemKind kind, size_t bytes, size_t align = 64);
  ~Mem();
  Mem(Mem&& o) noexcept;
  Mem& operator=(Mem&&) = delete;
  Mem(const Mem&) = delete;
  // Exchanges the two allocations. Spelled out rather than a move-assignment, which
  // stays deleted: a captured command list bakes ptr(), so replacing an allocation in
  // place is only ever correct before any capture, and a call site should say so
  // (loader::set_max_len, the one user, re-tables RoPE before the Engine exists).
  void swap(Mem& o) noexcept {
    std::swap(ctx_, o.ctx_);
    std::swap(kind_, o.kind_);
    std::swap(bytes_, o.bytes_);
    std::swap(ptr_, o.ptr_);
    std::swap(sys_, o.sys_);
    std::swap(sys_bytes_, o.sys_bytes_);
  }

  void* ptr() const { return ptr_; }
  size_t size() const { return bytes_; }
  MemKind kind() const { return kind_; }
  bool locked() const { return sys_ != nullptr; }
  template <class T> T* as() const { return static_cast<T*>(ptr_); }

 private:
  Context* ctx_;
  MemKind kind_;
  size_t bytes_;
  void* ptr_ = nullptr;
  void* sys_ = nullptr;     // HostLocked's own mapping (null: a driver allocation)
  size_t sys_bytes_ = 0;
};

// From this call on, HostLocked allocations are locked against swap when RLIMIT_MEMLOCK (the
// soft limit raised to the hard one first) covers `planned` more bytes than are locked already,
// and plain Host memory when it does not; returns which. Before the first call they are plain.
bool lock_host_memory(size_t planned);
// RLIMIT_MEMLOCK's soft limit in bytes, SIZE_MAX when unlimited.
size_t memlock_limit();
// The HostLocked bytes this process holds locked against swap, and those held as plain Host
// memory because the limit did not cover them (b70-serve's startup line).
size_t host_locked_bytes();
size_t host_unlocked_bytes();
}  // namespace l0
