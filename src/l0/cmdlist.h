#pragma once
#include <cstddef>
#include <cstdint>
#include "l0/context.h"

namespace l0 {
class Kernel;
class Event;

// Two flavours. immediate(): synchronous - every append executes and completes
// before returning; used for uploads and tests. regular(): in-order, recorded
// once, closed, executed many times by Queue::execute - the decode list.
class CmdList {
 public:
  static CmdList immediate(Context& ctx, uint32_t ordinal = 0);
  static CmdList regular(Context& ctx, uint32_t ordinal = 0);
  ~CmdList();
  CmdList(CmdList&& o) noexcept;
  CmdList(const CmdList&) = delete;
  CmdList& operator=(const CmdList&) = delete;

  void copy(void* dst, const void* src, size_t bytes);
  void fill(void* dst, uint32_t pattern, size_t bytes);
  // Appends a launch with gx*gy*gz work-groups; the kernel's group size must
  // already be set (Kernel::group_size). `signal`, when given, is signalled by
  // the device at this kernel's completion and carries its timestamps; it
  // observes only - nothing in the list waits on it, so a profiled list runs
  // the same commands in the same order as an unprofiled one.
  void launch(Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
              Event* signal = nullptr);
  // Spec 16b (pipeline parallel), the cross-device hand-off on regular lists:
  // barrier_signal appends a barrier that signals `e` once every earlier command of this
  // list has completed; wait_event makes every later command of this list wait until `e`
  // is signalled. `e` is a l0::SyncEvent's handle (an event in a pool both devices see).
  void barrier_signal(ze_event_handle_t e);
  void wait_event(ze_event_handle_t e);
  void close();
  void reset();
  bool is_immediate() const { return immediate_; }
  ze_command_list_handle_t handle() const { return l_; }

 private:
  CmdList(ze_command_list_handle_t l, bool immediate) : l_(l), immediate_(immediate) {}
  ze_command_list_handle_t l_ = nullptr;
  bool immediate_;
};
}  // namespace l0
