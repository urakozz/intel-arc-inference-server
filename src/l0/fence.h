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

 private:
  ze_fence_handle_t f_ = nullptr;
};
}  // namespace l0
