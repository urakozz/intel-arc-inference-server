#include "l0/memory.h"
#include "l0/error.h"

namespace l0 {
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
  }
}

Mem::~Mem() {
  if (ptr_) zeMemFree(ctx_->handle(), ptr_);
}

Mem::Mem(Mem&& o) noexcept : ctx_(o.ctx_), kind_(o.kind_), bytes_(o.bytes_), ptr_(o.ptr_) {
  o.ptr_ = nullptr;
}
}  // namespace l0
