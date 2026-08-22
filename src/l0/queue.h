#pragma once
#include "l0/context.h"

namespace l0 {
class CmdList;
class Fence;

// One asynchronous compute queue on the device's compute ordinal.
class Queue {
 public:
  explicit Queue(Context& ctx, uint32_t ordinal = 0);
  ~Queue();
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  ze_command_queue_handle_t handle() const { return q_; }
  Context& context() const { return *ctx_; }
  // Submits a closed regular command list. fence may be null.
  void execute(CmdList& list, Fence* fence = nullptr);
  void synchronize();

 private:
  Context* ctx_;
  ze_command_queue_handle_t q_ = nullptr;
};
}  // namespace l0
