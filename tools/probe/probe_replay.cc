// probe_replay: fixed cost per kernel inside a replayed regular command list.
// One in-order list with N launches, closed once, executed 1000x with a fence.
// Reports us per replay and us per kernel for N in {1, 250, 700}, for noop
// and for ctrl_read (which reads a shared-memory dword, like every decode
// kernel will). Decision rule is in spec 1 Section 4.1.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

static double median_us(l0::Queue& q, l0::Fence& f, l0::CmdList& list, int reps) {
  std::vector<double> us;
  for (int i = 0; i < reps + 20; ++i) {
    Timer t; t.start();
    q.execute(list, &f);
    f.wait();
    if (i >= 20) us.push_back(t.ms() * 1e3);
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Mem out(ctx, l0::MemKind::Device, 4096);
  l0::Mem ctrl(ctx, l0::MemKind::Shared, 64);
  ctrl.as<uint32_t>()[0] = 7;

  struct Case { const char* bin; const char* fn; bool has_ctrl; };
  const Case cases[] = {{"noop", "noop", false}, {"ctrl_read", "ctrl_read", true}};
  std::printf("| kernel | N | us/replay | us/kernel |\n|---|---|---|---|\n");
  {
    l0::CmdList empty = l0::CmdList::regular(ctx);
    empty.close();
    std::printf("| (empty list) | 0 | %.1f | - |\n", median_us(q, fence, empty, 1000));
  }
  for (const Case& c : cases) {
    l0::Module mod(ctx, kernels::path(c.bin));
    l0::Kernel k = mod.kernel(c.fn);
    if (c.has_ctrl) { k.arg_ptr(0, ctrl.ptr()); k.arg_ptr(1, out.ptr()); }
    else { k.arg_ptr(0, out.ptr()); }
    k.group_size(16);
    for (int n : {1, 250, 700}) {
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int i = 0; i < n; ++i) list.launch(k, 1);
      list.close();
      double us = median_us(q, fence, list, 1000);
      std::printf("| %s | %d | %.1f | %.2f |\n", c.fn, n, us, us / n);
    }
  }
  return 0;
}
