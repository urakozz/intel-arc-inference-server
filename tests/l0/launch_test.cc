// Loads noop.bin, records one launch in a regular list, executes it twice
// (replay), and checks the write happened. This is the smallest replay.
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

  l0::Mem out(ctx, l0::MemKind::Shared, 64);
  out.as<uint32_t>()[0] = 0;
  k.arg_ptr(0, out.ptr());
  k.group_size(16);

  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(k, 1);
  list.close();

  for (int replay = 0; replay < 2; ++replay) {
    out.as<uint32_t>()[0] = 0;
    q.execute(list, &fence);
    fence.wait();
    CHECK_EQ(out.as<uint32_t>()[0], 42u);
  }
  std::puts("launch_test OK");
  return 0;
}
