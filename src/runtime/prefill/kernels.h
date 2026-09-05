#pragma once
#include <map>
#include <memory>
#include <string>

#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/module.h"

namespace runtime::prefill {

// The prefill path's module/kernel cache - plan 6b Task 2's `kernel(variant,
// entry)`, living **beside** `Context` rather than on it.
//
// **Why it is not a member of `Context`.** `Context`'s methods are compiled by
// icpx into `libb70_prefill.so`, and that .so deliberately links no project
// archive (`context.h`'s header comment; `cmake/prefill.cmake` states the
// reason - a g++ link of icpx objects silently drops the registered device
// images, so the SYCL objects must be linked by icpx, and nothing in the decode
// build may therefore be dragged in as -fPIC). `l0::Module` and `l0::Kernel`
// are `b70_l0` symbols with out-of-line constructors. A `Context::kernel()`
// returning `l0::Kernel&` would pull that archive into the .so and void the
// split. This class is plain C++17, compiled by g++, and hands `Context` a
// `ze_kernel_handle_t` through `l0::Kernel::handle()`, which is inline.
//
// Semantics are `runtime::Capture::kernel()`'s (`capture.cc:246-267`): one
// `l0::Module` per variant and one `l0::Kernel` per (variant, entry), both
// cached for the cache's lifetime, so a chunk that launches the same ten
// kernels for 48 GDN layers loads ten binaries once.
//
// **No work-group size argument.** `Context::launch` reads the kernel's own
// `reqd_work_group_size` (every prefill `.cl` declares one, as every decode one
// does) and rejects a kernel without it by name. Passing the width again here
// would be a second home for a number the binary already bakes - exactly the
// failure mode `DecodeScratch::kAttnBlock`'s comment describes.
class KernelCache {
 public:
  explicit KernelCache(l0::Context& ctx) : ctx_(ctx) {}

  // `variant` is the ocloc target name (`kernels::pf_gdn_conv_variant()`),
  // `entry` the `__kernel` symbol inside it. Throws, naming both, if the
  // binary or the entry point is absent.
  l0::Kernel& get(const std::string& variant, const char* entry);
  l0::Kernel& operator()(const std::string& variant, const char* entry) {
    return get(variant, entry);
  }

  size_t modules() const { return modules_.size(); }
  size_t kernels() const { return kernels_.size(); }
  l0::Context& l0() const { return ctx_; }

 private:
  l0::Context& ctx_;
  std::map<std::string, std::unique_ptr<l0::Module>> modules_;
  std::map<std::string, std::unique_ptr<l0::Kernel>> kernels_;
};

}  // namespace runtime::prefill
