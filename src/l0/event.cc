#include "l0/event.h"
#include <string>
#include "l0/error.h"

namespace l0 {
TimerCalib TimerCalib::query(Context& ctx) {
  // NOT ctx.props(): that is filled with stype ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
  // where `timerResolution` is deprecated since L0 1.17 and means nanoseconds
  // per tick on some drivers and cycles/sec on others. The 1_2 stype pins it
  // to cycles/sec, which is the only reading under which `kernelStart` /
  // `kernelEnd` tick counts divide out to a time.
  ze_device_properties_t p{};
  p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
  ZE_CHECK(zeDeviceGetProperties(ctx.device(), &p));
  if (p.timerResolution == 0)
    throw std::runtime_error("device reports timerResolution 0: no timestamp calibration");
  // 0 would make mask() 0 and every duration 0; > 64 cannot be a uint64 mask.
  if (p.kernelTimestampValidBits == 0 || p.kernelTimestampValidBits > 64)
    throw std::runtime_error("device reports kernelTimestampValidBits " +
                             std::to_string(p.kernelTimestampValidBits) + " (expected 1..64)");
  TimerCalib c{};
  c.cycles_per_us = static_cast<double>(p.timerResolution) / 1e6;
  c.valid_bits = p.kernelTimestampValidBits;
  return c;
}

EventPool::EventPool(Context& ctx, uint32_t capacity)
    : cap_(capacity), calib_(TimerCalib::query(ctx)) {
  if (capacity == 0) throw std::runtime_error("EventPool capacity must be > 0");
  ze_event_pool_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  // HOST_VISIBLE so the host can read the slots; KERNEL_TIMESTAMP so the slots
  // carry the start/end pair zeEventQueryKernelTimestamp needs at all.
  d.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  d.count = capacity;
  ze_device_handle_t dev = ctx.device();
  ZE_CHECK(zeEventPoolCreate(ctx.handle(), &d, 1, &dev, &p_));
}
EventPool::~EventPool() {
  if (p_) zeEventPoolDestroy(p_);
}
EventPool::EventPool(EventPool&& o) noexcept : p_(o.p_), cap_(o.cap_), calib_(o.calib_) {
  o.p_ = nullptr;
  o.cap_ = 0;
}

Event::Event(EventPool& pool, uint32_t index) : calib_(pool.calib()) {
  if (index >= pool.capacity())
    throw std::runtime_error("event index " + std::to_string(index) + " past pool capacity " +
                             std::to_string(pool.capacity()));
  ze_event_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  d.index = index;
  // Flush to host scope on signal so the timestamp the host reads after the
  // fence is the one the device wrote. The flush lands after `kernelEnd`, so
  // it inflates the gap between kernels, never a measured duration.
  d.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  d.wait = 0;  // nothing in the list waits on these; they only observe.
  ZE_CHECK(zeEventCreate(pool.handle(), &d, &e_));
}
Event::~Event() {
  if (e_) zeEventDestroy(e_);
}
Event::Event(Event&& o) noexcept : e_(o.e_), calib_(o.calib_) { o.e_ = nullptr; }

double Event::duration_us() const {
  ze_kernel_timestamp_result_t ts{};
  ZE_CHECK(zeEventQueryKernelTimestamp(e_, &ts));
  // Mask BEFORE subtracting: the bits above `valid_bits` are not counter
  // state. Masking the difference too makes a wrap between start and end come
  // out as the short forward distance rather than a ~2^64 nonsense number.
  // (This driver reports 64 valid bits, so both masks are identity here - the
  // code is written for the narrow-counter part it will meet elsewhere.)
  const uint64_t m = calib_.mask();
  const uint64_t start = ts.global.kernelStart & m;
  const uint64_t end = ts.global.kernelEnd & m;
  return static_cast<double>((end - start) & m) / calib_.cycles_per_us;
}

void Event::reset() { ZE_CHECK(zeEventHostReset(e_)); }
}  // namespace l0
