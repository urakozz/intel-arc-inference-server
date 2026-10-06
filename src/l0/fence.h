#pragma once
#include "l0/queue.h"

namespace l0 {
// Host-side completion signal for one queue submission. wait() blocks, then
// resets, so the same fence serves every replay.
class Fence {
 public:
  explicit Fence(Queue& q);
  ~Fence();
  Fence(const Fence&) = delete;
  Fence& operator=(const Fence&) = delete;
  ze_fence_handle_t handle() const { return f_; }
  void wait();
  // Spec 16b: wait at most `timeout_ns`. True (and the fence reset, as wait() does) when
  // the submission completed; false when it had not - the fence is left as it is, so the
  // caller can drain the queue and wait again. A pipeline-parallel step never waits
  // without a bound (spec 16 §4 P4).
  bool wait_for(uint64_t timeout_ns);

 private:
  ze_fence_handle_t f_ = nullptr;
};
}  // namespace l0
