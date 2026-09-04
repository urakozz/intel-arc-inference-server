// The prefill execution context: one ze_context, two queues (spec 2 §3.6).
// Needs a B70 and the `noop` binary and nothing else -- no checkpoint, so no
// ctest label. It is also the ABI check for the two-compiler build: this
// translation unit is g++-compiled and calls into an icpx-linked .so across a
// std::initializer_list and a std::runtime_error.
#include <cstdint>
#include <cstdio>

#include "check.h"
#include "kernels/kernels.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"

int main() {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  CHECK(cx.ze_context() == ctx.handle());
  CHECK(cx.ze_device() == ctx.device());
  CHECK(cx.l0_list() != nullptr);
  CHECK(cx.sycl_queue_raw() != nullptr);
  CHECK(cx.sycl_context_raw() != nullptr);
  cx.wait();
  l0::Module mod(ctx, kernels::path("noop"));
  l0::Kernel k = mod.kernel("noop");
  l0::Mem out(ctx, l0::MemKind::Shared, 64);
  out.as<uint32_t>()[0] = 0;
  void* out_ptr = out.ptr();
  cx.launch(k, 1, 1, 1, {{&out_ptr, sizeof out_ptr}});
  cx.wait();
  CHECK_EQ(out.as<uint32_t>()[0], 42u);
  std::printf("context_test OK\n");
  return 0;
}
