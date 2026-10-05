#pragma once
#include <cstddef>
#include <utility>
#include "l0/context.h"

namespace l0 {
enum class MemKind { Device, Host, Shared };

// One allocation. Device memory is not host-accessible; Host memory is
// host-resident and device-visible; Shared migrates (used for the control block).
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
  }

  void* ptr() const { return ptr_; }
  size_t size() const { return bytes_; }
  MemKind kind() const { return kind_; }
  template <class T> T* as() const { return static_cast<T*>(ptr_); }

 private:
  Context* ctx_;
  MemKind kind_;
  size_t bytes_;
  void* ptr_ = nullptr;
};
}  // namespace l0
