#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <functional>
#include <memory>

#include "l0/context.h"
#include "l0/kernel.h"
#include "runtime/prefill/sycl_side.h"

namespace l0 {
class Event;
}

namespace runtime::prefill {

// `Context` is host-side (g++): the Level Zero immediate list, `launch`, `wait` and the
// launch counter. Its SYCL half is `SyclSide` (sycl_side.h), built by icpx into
// libb70_prefill.so and reached only through `sycl()`, which creates it on first use (spec
// 2.1 §3.5). `libb70_prefill.so` links no project archive and names no `Context` symbol --
// it is a leaf the g++ host archive calls into through `gemm_sycl.h`'s SyclSide-taking
// entry points, never the other way around. Including l0/context.h and l0/kernel.h here
// costs no link dependency: the only members they touch (Context::handle(),
// Context::device(), Kernel::handle()) are inline.
struct KernelArg {
  const void* ptr;
  size_t size;
};

// **The pointer-argument trap, stated here because it is the one way this API
// can be silently wrong.** `zeKernelSetArgumentValue` copies `size` bytes FROM
// `ptr`, so a pointer argument must pass the *address of a pointer variable*,
// not the pointer. A helper that took `const void* p` by value and returned
// `{&p, sizeof p}` would return the address of its own dead parameter. `PtrArg`
// therefore holds the pointer VALUE in a temporary the caller creates, and
// `&value_` stays valid for the whole `launch(...)` call - every temporary in a
// full-expression lives until that expression ends, which is after `launch`
// returns. Call sites read like the kernel's signature:
//
//     cx.launch(k, gx, gy, 1, {PtrArg(a), PtrArg(b), arg_val(m_count)});
struct PtrArg {
  const void* value_;
  explicit PtrArg(const void* p) : value_(p) {}
  operator KernelArg() const { return KernelArg{&value_, sizeof(const void*)}; }
};

// A by-value scalar argument (uint32_t `M`, `pos`, `c_count`, …). Same
// lifetime rule: `v` is either a live variable or a temporary of the enclosing
// full-expression, and `launch` copies out of it before returning.
template <class T>
inline KernelArg arg_val(const T& v) {
  return KernelArg{&v, sizeof(T)};
}

class Context {
 public:
  struct LaunchMetrics {
    double gpu_ms = 0;
    double host_submit_ms = 0;
    size_t launches = 0;
  };
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

  // Spec 2.1 §3.5: the SYCL side, built on first use. The L0 backend never calls sycl(), so a
  // pure-L0 session never links a queue to the device -- has_sycl() is how the equivalence test
  // proves that. wait() synchronises the SYCL queue first (when it exists) and then the list,
  // spec 2's order.
  SyclSide& sycl();
  bool has_sycl() const;
  ze_context_handle_t ze_context() const;
  ze_device_handle_t ze_device() const;
  ze_command_list_handle_t l0_list() const;

  // Spec 16c (the two-card prefill pipeline, runtime/prefill/pipeline_prefill.cc): appended
  // on the immediate list in order with the launches; never inside a recording.
  //   copy       zeCommandListAppendMemoryCopy (the hand-off's rows, a block-end shadow)
  //   signal     a barrier that signals `e` once everything appended before it has completed
  //   wait_event everything appended after it waits until `e` is signalled
  //   timestamp  the device's global timestamp, written to `dst` when the list reaches it
  void copy(void* dst, const void* src, size_t bytes);
  void signal(ze_event_handle_t e);
  void wait_event(ze_event_handle_t e);
  void timestamp(uint64_t* dst);
  // wait() with a bound: true once everything appended has completed, false when
  // `timeout_ns` passed first (the list keeps running). L0 only: a session with a SYCL side
  // has no bounded wait and throws.
  bool wait_for(uint64_t timeout_ns);
  // Two of timestamp()'s values as milliseconds (the device's timer frequency and its
  // timestamp width, read once).
  double timestamp_ms(uint64_t begin, uint64_t end) const;

  // Launches appended to the L0 list since construction (or the last
  // reset_launches()). The prefill step's anatomy report is launch arithmetic,
  // and `gdn_chunk`'s ten-launches-per-GDN-layer contract is asserted through
  // this counter rather than restated in prose.
  size_t launches() const;
  void reset_launches();

  // Diagnostic only: timestamp L0 launches and measure host launch submission.
  // GPU execution and host submission overlap; these times are not additive.
  // Enabled by B70_PREFILL_PROFILE=1, or explicitly by tests. Changing mode
  // drains the list. take_launch_metrics() drains timestamp results and clears
  // the counters after the caller has waited; it does not synchronize itself.
  void set_profiling(bool enabled);
  LaunchMetrics take_launch_metrics();

  // Opt-in pure-L0 recording. Capture resolves pointer/scalar arguments but
  // executes nothing; replay reads current buffer contents and synchronizes
  // before returning. Kernels and referenced allocations must outlive the
  // recording. Waits, SYCL, profiling and nested captures are rejected.
  class Recording {
   public:
    ~Recording();
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;
   private:
    friend class Context;
    explicit Recording(const Context* owner) : owner_(owner) {}
    const Context* owner_;
    ze_command_list_handle_t list_ = nullptr;
    size_t launches_ = 0;
  };
  std::unique_ptr<Recording> capture(const std::function<void()>& encode);
  void replay(const Recording& recording);

 private:
  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace runtime::prefill
