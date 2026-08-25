#pragma once
#include <cstdint>
#include "l0/context.h"

namespace l0 {
// Device timer calibration, queried once per context.
// Uses ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 explicitly so timerResolution
// is pinned to cycles-per-second semantics. That is not a formality: on this
// box the default stype reports 52 (NANOSECONDS per tick) where 1_2 reports
// 19200000 (cycles/sec) - reading the first as the second scales every
// duration by 369231x. kernelTimestampValidBits masks the counter before any
// subtraction; this driver reports 64 bits, so nothing wraps in practice, but
// a 32-bit part wraps in seconds and the mask costs nothing. Either way the
// rule stands: durations only, never absolute times across a step.
struct TimerCalib {
  double cycles_per_us;
  uint32_t valid_bits;
  static TimerCalib query(Context& ctx);
  // The low `valid_bits` set: what a raw timestamp must be AND-ed with before
  // it is compared or subtracted. The bits above are not counter state.
  uint64_t mask() const {
    return valid_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << valid_bits) - 1);
  }
};

// Pool of host-visible kernel-timestamp events. One pool, N slots.
class EventPool {
 public:
  EventPool(Context& ctx, uint32_t capacity);  // HOST_VISIBLE | KERNEL_TIMESTAMP
  ~EventPool();
  EventPool(EventPool&&) noexcept;
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;
  ze_event_pool_handle_t handle() const { return p_; }
  uint32_t capacity() const { return cap_; }
  // Queried once here so that N events cost one zeDeviceGetProperties, not N.
  const TimerCalib& calib() const { return calib_; }

 private:
  ze_event_pool_handle_t p_ = nullptr;
  uint32_t cap_ = 0;
  TimerCalib calib_{};
};

// One timestamp event. Device signals it at kernel completion; host queries.
class Event {
 public:
  Event(EventPool& pool, uint32_t index);
  ~Event();
  Event(Event&&) noexcept;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  // Global (wall) kernel timestamps in device cycles, wrap-masked delta.
  // Returns microseconds. Query AFTER the fence that covers the launch:
  // an unsignalled event makes the driver return NOT_READY, which throws.
  double duration_us() const;
  void reset();  // zeEventHostReset
  ze_event_handle_t handle() const { return e_; }

 private:
  ze_event_handle_t e_ = nullptr;
  // A copy, not a reference to the pool: events live in a vector alongside a
  // pool that may itself be moved, and two scalars are cheaper than the
  // indirection anyway.
  TimerCalib calib_{};
};
}  // namespace l0
