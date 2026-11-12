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
  // Submits closed regular lists in order with one queue execution.
  void execute(uint32_t count, ze_command_list_handle_t* lists, Fence* fence = nullptr);
  uint64_t submissions() const { return submissions_; }
  void synchronize();

 private:
  Context* ctx_;
  ze_command_queue_handle_t q_ = nullptr;
  uint64_t submissions_ = 0;
};
}  // namespace l0
