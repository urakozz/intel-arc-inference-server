// Does Level Zero capture kernel arguments at zeCommandListAppendLaunchKernel
// time, or does it resolve them at execute time from the kernel's last-bound
// state? The whole timed harness (tests/kernels/gemv_harness.h, probe_gemv)
// depends on the first: it records N launches of one Kernel object, rebinding
// the weight pointer between appends so consecutive launches hit different
// buffers and miss L2. Every existing test binds identical copies, so none of
// them can tell the two behaviours apart.
//
// This one can. Two distinct buffers, one list, one launch each, different
// pointers bound between the appends. Capture-at-append => both get 42.
// Use-last-bound => both launches write B and A stays 0.
#include <cstdint>
#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("noop"));
  l0::Kernel k = mod.kernel("noop");

  l0::Mem a(ctx, l0::MemKind::Shared, 64);
  l0::Mem b(ctx, l0::MemKind::Shared, 64);
  k.group_size(16);

  l0::CmdList list = l0::CmdList::regular(ctx);
  k.arg_ptr(0, a.ptr());
  list.launch(k, 1);
  k.arg_ptr(0, b.ptr());
  list.launch(k, 1);
  list.close();

  // Replayed twice: the argument capture must survive re-execution of the same
  // closed list, which is what the decode runtime replays per token.
  for (int replay = 0; replay < 2; ++replay) {
    a.as<uint32_t>()[0] = 0;
    b.as<uint32_t>()[0] = 0;
    q.execute(list, &fence);
    fence.wait();
    CHECK_EQ(a.as<uint32_t>()[0], 42u);  // 0 here == arguments resolved at execute time
    CHECK_EQ(b.as<uint32_t>()[0], 42u);
  }
  std::puts("arg_capture_test OK");
  return 0;
}
