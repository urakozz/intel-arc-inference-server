#include "runtime/prefill/context.h"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/event.h"
#include "runtime/prefill/context_sycl.h"

namespace runtime::prefill {
namespace {

constexpr sycl::backend kL0 = sycl::backend::ext_oneapi_level_zero;

void zc(ze_result_t r, const char* what) {
  if (r == ZE_RESULT_SUCCESS) return;
  char b[24];
  std::snprintf(b, sizeof b, "0x%X", unsigned(r));
  throw std::runtime_error(std::string("prefill::Context: ") + what + " failed: " + b);
}

}  // namespace

struct Context::Impl {
  ze_context_handle_t ctx;
  ze_device_handle_t dev;
  ze_command_list_handle_t list;
  sycl::device sdev;
  sycl::context sctx;
  sycl::queue sq;
};

Context::Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev) {
  // The L0 side: one IMMEDIATE command list on ordinal 0 -- the same shape
  // l0::CmdList::immediate builds (src/l0/cmdlist.cc:7-15) but ASYNCHRONOUS,
  // because prefill's ordering is by wait()/events and a synchronous append
  // would serialise the host on every kernel.
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.ordinal = 0;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t list = nullptr;
  zc(zeCommandListCreateImmediate(ze_ctx, ze_dev, &qd, &list),
     "zeCommandListCreateImmediate");

  try {
    // The SYCL side: device and context by INTEROP over the same handles, so
    // the engine's zeMemAllocDevice pointers are valid USM in SYCL kernels
    // (that is the whole requirement; P1 proves it). The QUEUE is a plain
    // in-order sycl::queue ON that context, NOT an interop queue: the
    // extension's queue interop wants a ze_command_queue we would then own and
    // drain twice, and it is the CONTEXT that makes the pointers shared.
    // ownership::keep everywhere -- l0::Context still owns the handles and
    // still destroys them. The 2026.1 aggregate order is NativeHandle,
    // DeviceList, Ownership (recorded in docs/probe-interop-2026-09-04.md).
    sycl::device sdev = sycl::make_device<kL0>(ze_dev);
    sycl::context sctx = sycl::make_context<kL0>(
        sycl::backend_input_t<kL0, sycl::context>{
            ze_ctx, std::vector<sycl::device>{sdev},
            sycl::ext::oneapi::level_zero::ownership::keep});
    p_ = new Impl{ze_ctx, ze_dev, list, sdev, sctx,
                  sycl::queue(sctx, sdev, sycl::property::queue::in_order{})};
  } catch (...) {
    zeCommandListDestroy(list);
    throw;
  }
}

Context::~Context() {
  if (p_) {
    zeCommandListDestroy(p_->list);
    delete p_;
  }
}

void Context::wait() {
  p_->sq.wait_and_throw();
  zc(zeCommandListHostSynchronize(p_->list, UINT64_MAX), "zeCommandListHostSynchronize");
}

void Context::launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                     std::initializer_list<KernelArg> args, l0::Event* signal) {
  uint32_t i = 0;
  for (const KernelArg& a : args)
    zc(zeKernelSetArgumentValue(k, i++, a.size, a.ptr), "zeKernelSetArgumentValue");
  // The group size comes from the kernel's own reqd_work_group_size. Every
  // prefill OpenCL C kernel declares one, as every decode kernel does, so this
  // is one fewer argument a caller can get wrong; a kernel without one is
  // rejected by name rather than launched at a guessed width. This is how
  // interfaces.md's group-size-free `launch` signature is honoured.
  ze_kernel_properties_t kp{};
  kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  zc(zeKernelGetProperties(k, &kp), "zeKernelGetProperties");
  if (kp.requiredGroupSizeX == 0)
    throw std::runtime_error("prefill::Context::launch: kernel has no reqd_work_group_size");
  zc(zeKernelSetGroupSize(k, kp.requiredGroupSizeX, kp.requiredGroupSizeY,
                          kp.requiredGroupSizeZ), "zeKernelSetGroupSize");
  ze_group_count_t g{gx, gy, gz};
  zc(zeCommandListAppendLaunchKernel(p_->list, k, &g,
                                     signal ? signal->handle() : nullptr, 0, nullptr),
     "zeCommandListAppendLaunchKernel");
}

void* Context::sycl_queue_raw() const { return &p_->sq; }
void* Context::sycl_context_raw() const { return &p_->sctx; }
ze_context_handle_t Context::ze_context() const { return p_->ctx; }
ze_device_handle_t Context::ze_device() const { return p_->dev; }
ze_command_list_handle_t Context::l0_list() const { return p_->list; }

}  // namespace runtime::prefill
