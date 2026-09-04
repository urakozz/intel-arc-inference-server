#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "l0/context.h"
#include "l0/kernel.h"

namespace l0 {
class Event;
}

namespace runtime::prefill {

// This header is SYCL-free because g++ translation units include it. Including
// l0/context.h and l0/kernel.h costs no link dependency: the only members it
// touches (Context::handle(), Context::device(), Kernel::handle()) are inline,
// so libb70_prefill.so links no project archive. sycl_queue_raw() and
// sycl_context_raw() have exactly two legitimate callers: the typed inline
// accessors in context_sycl.h.
struct KernelArg {
  const void* ptr;
  size_t size;
};

class Context {
 public:
  Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev);
  explicit Context(l0::Context& c) : Context(c.handle(), c.device()) {}
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  void wait();
  void launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<KernelArg> args, l0::Event* signal = nullptr);
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<KernelArg> args, l0::Event* signal = nullptr) {
    launch(k.handle(), gx, gy, gz, args, signal);
  }

  void* sycl_queue_raw() const;
  void* sycl_context_raw() const;
  ze_context_handle_t ze_context() const;
  ze_device_handle_t ze_device() const;
  ze_command_list_handle_t l0_list() const;

 private:
  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace runtime::prefill
