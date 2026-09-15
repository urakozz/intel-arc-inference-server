#include "runtime/prefill/context.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "l0/event.h"
#include "runtime/prefill/sycl_side.h"

// The prefill Context's Level Zero half (spec 2.1 §3.5). Before spec 2.1 this file was
// src/sycl/context.cc and built a SYCL queue in its constructor; the queue is now a SyclSide
// created on the first sycl() call, so a chunk on the L0 backend never touches SYCL and this
// translation unit compiles with g++ in every build.
namespace runtime::prefill {
namespace {
void zc(ze_result_t r, const char* what) {
  if (r == ZE_RESULT_SUCCESS) return;
  char b[24];
  std::snprintf(b, sizeof b, "0x%X", unsigned(r));
  throw std::runtime_error(std::string("prefill::Context: ") + what + " failed: " + b);
}
std::string kernel_name(ze_kernel_handle_t k) {
  size_t n = 0;
  if (zeKernelGetName(k, &n, nullptr) != ZE_RESULT_SUCCESS || n == 0) return "<unnamed kernel>";
  std::string s(n, '\0');
  if (zeKernelGetName(k, &n, s.data()) != ZE_RESULT_SUCCESS) return "<unnamed kernel>";
  if (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}
}  // namespace

struct Context::Impl {
  ze_context_handle_t ctx;
  ze_device_handle_t dev;
  ze_command_list_handle_t list;
  SyclSide* sycl = nullptr;   // lazily created; owned
  size_t launches = 0;
};

Context::Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev) {
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.ordinal = 0;
  qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t list = nullptr;
  zc(zeCommandListCreateImmediate(ze_ctx, ze_dev, &qd, &list), "zeCommandListCreateImmediate");
  p_ = new Impl{ze_ctx, ze_dev, list, nullptr, 0};
}

Context::~Context() {
  if (!p_) return;
  if (p_->sycl) sycl_side_destroy(p_->sycl);
  zeCommandListDestroy(p_->list);
  delete p_;
}

void Context::wait() {
  if (p_->sycl) sycl_side_wait(p_->sycl);
  zc(zeCommandListHostSynchronize(p_->list, UINT64_MAX), "zeCommandListHostSynchronize");
}

SyclSide& Context::sycl() {
  if (!p_->sycl) p_->sycl = sycl_side_create(p_->ctx, p_->dev);
  return *p_->sycl;
}
bool Context::has_sycl() const { return p_->sycl != nullptr; }

void Context::launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                     std::initializer_list<KernelArg> args, l0::Event* signal) {
  uint32_t i = 0;
  for (const KernelArg& a : args) {
    const ze_result_t r = zeKernelSetArgumentValue(k, i, a.size, a.ptr);
    if (r != ZE_RESULT_SUCCESS) {
      char b[24];
      std::snprintf(b, sizeof b, "0x%X", unsigned(r));
      throw std::runtime_error("prefill::Context::launch: " + kernel_name(k) + " argument " +
                               std::to_string(i) + " (" + std::to_string(a.size) +
                               " bytes) rejected: " + b);
    }
    ++i;
  }
  // The group size comes from the kernel's own reqd_work_group_size (every prefill kernel
  // declares one); a kernel without one is rejected by name, never launched at a guess.
  ze_kernel_properties_t kp{};
  kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  zc(zeKernelGetProperties(k, &kp), "zeKernelGetProperties");
  if (kp.requiredGroupSizeX == 0)
    throw std::runtime_error("prefill::Context::launch: " + kernel_name(k) +
                             " has no reqd_work_group_size");
  zc(zeKernelSetGroupSize(k, kp.requiredGroupSizeX, kp.requiredGroupSizeY, kp.requiredGroupSizeZ),
     "zeKernelSetGroupSize");
  ze_group_count_t g{gx, gy, gz};
  zc(zeCommandListAppendLaunchKernel(p_->list, k, &g, signal ? signal->handle() : nullptr, 0,
                                     nullptr),
     "zeCommandListAppendLaunchKernel");
  ++p_->launches;
}

ze_context_handle_t Context::ze_context() const { return p_->ctx; }
ze_device_handle_t Context::ze_device() const { return p_->dev; }
ze_command_list_handle_t Context::l0_list() const { return p_->list; }
size_t Context::launches() const { return p_->launches; }
void Context::reset_launches() { p_->launches = 0; }

}  // namespace runtime::prefill
