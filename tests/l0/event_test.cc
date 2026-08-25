// Kernel-timestamp events on a replayed regular list: the device timer
// calibration, one launch's duration, the same events reused across replays,
// and two launches in one in-order list proving non-overlap. This is the
// measurement layer plan 4's per-kernel profiler is built on, tested here on
// the smallest possible list (noop) so a failure is the event machinery and
// not the decode step.
#include <cstdint>
#include <cstdio>
#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {
// The wall-clock pair straight from the driver, masked to the device's valid
// bits. Absolute values mean nothing across a step (the counter wraps in
// seconds at 32 bits) - only wrap-safe differences do.
struct Raw {
  uint64_t start;
  uint64_t end;
};
Raw raw_global(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  CHECK_EQ(zeEventQueryKernelTimestamp(e.handle(), &ts), ZE_RESULT_SUCCESS);
  return Raw{ts.global.kernelStart & mask, ts.global.kernelEnd & mask};
}
}  // namespace

int main() {
  l0::Context ctx(0);

  // 1. Calibration. `cycles_per_us` divides tick counts into a time, so it has
  //    to be positive; `valid_bits` is 32 on BMG and the mask is only sane in
  //    a range that covers every part we might run on.
  const l0::TimerCalib calib = l0::TimerCalib::query(ctx);
  std::printf("timer: %.6f cycles/us, %u valid bits\n", calib.cycles_per_us, calib.valid_bits);
  CHECK(calib.cycles_per_us > 0.0);
  CHECK(calib.valid_bits >= 24 && calib.valid_bits <= 64);
  const uint64_t mask = calib.mask();

  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("noop"));

  l0::EventPool pool(ctx, 4);
  CHECK_EQ(pool.capacity(), 4u);

  // Two kernels writing to two buffers, so "both ran" is checked separately.
  l0::Mem out_a(ctx, l0::MemKind::Shared, 64);
  l0::Mem out_b(ctx, l0::MemKind::Shared, 64);
  l0::Kernel ka = mod.kernel("noop");
  l0::Kernel kb = mod.kernel("noop");
  ka.arg_ptr(0, out_a.ptr());
  ka.group_size(16);
  kb.arg_ptr(0, out_b.ptr());
  kb.group_size(16);

  l0::Event ea(pool, 0);
  l0::Event eb(pool, 1);

  // 2. A REGULAR list - the shape the decode step replays - with a signal
  //    event on each launch, executed through Queue + Fence.
  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(ka, 1, 1, 1, &ea);
  list.launch(kb, 1, 1, 1, &eb);
  list.close();

  out_a.as<uint32_t>()[0] = 0;
  out_b.as<uint32_t>()[0] = 0;
  q.execute(list, &fence);
  fence.wait();
  CHECK_EQ(out_a.as<uint32_t>()[0], 42u);
  CHECK_EQ(out_b.as<uint32_t>()[0], 42u);

  const double da = ea.duration_us();
  const double db = eb.duration_us();
  std::printf("noop a: %.3f us, b: %.3f us\n", da, db);
  CHECK(da > 0.0 && da < 1e6);
  CHECK(db > 0.0 && db < 1e6);

  // 4. The list is in-order, so b's global start is at or after a's global
  //    end. Asserted on the raw query, and wrap-safely: the forward distance
  //    of two samples microseconds apart is tiny, while an overlap of even one
  //    tick underflows the mask and lands near the top of the counter's range.
  //    So "small forward gap" IS start_b >= end_a, and stays true across a wrap.
  const Raw ra = raw_global(ea, mask);
  const Raw rb = raw_global(eb, mask);
  const double gap_us = static_cast<double>((rb.start - ra.end) & mask) / calib.cycles_per_us;
  std::printf("a->b gap: %.3f us\n", gap_us);
  CHECK(gap_us < 1e6);

  // 3. Reset and replay: the same events serve every replay, which is the
  //    property the profiled decode step needs (it replays one captured list
  //    for every token).
  ea.reset();
  eb.reset();
  out_a.as<uint32_t>()[0] = 0;
  out_b.as<uint32_t>()[0] = 0;
  q.execute(list, &fence);
  fence.wait();
  CHECK_EQ(out_a.as<uint32_t>()[0], 42u);
  CHECK_EQ(out_b.as<uint32_t>()[0], 42u);
  const double da2 = ea.duration_us();
  const double db2 = eb.duration_us();
  std::printf("replay a: %.3f us, b: %.3f us\n", da2, db2);
  CHECK(da2 > 0.0 && da2 < 1e6);
  CHECK(db2 > 0.0 && db2 < 1e6);

  std::puts("event_test OK");
  return 0;
}
