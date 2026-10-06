#pragma once
#include <cstdint>
#include <vector>
#include "l0/context.h"

namespace l0 {
// Spec 16b (pipeline parallel, `--pipeline-handoff copy`): one event that a list on one
// device signals and a list on another device waits on. Its own pool of one slot, created
// for every device in `devices` (all views of one context - l0::Context's view
// constructor), HOST_VISIBLE so the host can reset it before a step, query it, and signal
// it to release a waiting list after a timeout. Signal and wait scopes are HOST: the event
// crosses devices, so its signal must be visible beyond the signalling device.
//
// Unlike l0::Event (a kernel-timestamp observer nothing waits on), this event IS waited
// on: it orders device 1's list after device 0's hand-off. The host resets it before each
// step; nothing on either list resets it, so a replayed pair of lists needs no device-side
// reset and no counter.
class SyncEvent {
 public:
  SyncEvent(Context& ctx, const std::vector<const Context*>& devices);
  ~SyncEvent();
  SyncEvent(const SyncEvent&) = delete;
  SyncEvent& operator=(const SyncEvent&) = delete;

  void host_reset();         // zeEventHostReset
  void host_signal();        // zeEventHostSignal: releases a list waiting on it
  bool signalled() const;    // zeEventQueryStatus == SUCCESS
  // Spec 16c: zeEventHostSynchronize with a bound - true once signalled, false when
  // `timeout_ns` passed first (the event is left as it is). The prefill pipeline's host
  // waits on a device's per-chunk done event this way, never without a bound.
  bool host_wait(uint64_t timeout_ns) const;
  ze_event_handle_t handle() const { return e_; }

 private:
  ze_event_pool_handle_t pool_ = nullptr;
  ze_event_handle_t e_ = nullptr;
};
}  // namespace l0
