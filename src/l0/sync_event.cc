#include "l0/sync_event.h"
#include <stdexcept>
#include "l0/error.h"

namespace l0 {
SyncEvent::SyncEvent(Context& ctx, const std::vector<const Context*>& devices) {
  if (devices.empty()) throw std::runtime_error("l0::SyncEvent: no device to create the pool for");
  std::vector<ze_device_handle_t> devs;
  for (const Context* c : devices) {
    if (c->handle() != ctx.handle())
      throw std::runtime_error("l0::SyncEvent: every device must be a view of one context");
    devs.push_back(c->device());
  }
  ze_event_pool_desc_t pd{};
  pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  pd.count = 1;
  ZE_CHECK(zeEventPoolCreate(ctx.handle(), &pd, static_cast<uint32_t>(devs.size()), devs.data(),
                             &pool_));
  ze_event_desc_t ed{};
  ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ed.index = 0;
  ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  const ze_result_t r = zeEventCreate(pool_, &ed, &e_);
  if (r != ZE_RESULT_SUCCESS) {
    zeEventPoolDestroy(pool_);
    pool_ = nullptr;
    ZE_CHECK(r);
  }
}

SyncEvent::~SyncEvent() {
  if (e_) zeEventDestroy(e_);
  if (pool_) zeEventPoolDestroy(pool_);
}

void SyncEvent::host_reset() { ZE_CHECK(zeEventHostReset(e_)); }
void SyncEvent::host_signal() { ZE_CHECK(zeEventHostSignal(e_)); }
bool SyncEvent::signalled() const {
  const ze_result_t r = zeEventQueryStatus(e_);
  if (r == ZE_RESULT_NOT_READY) return false;
  ZE_CHECK(r);
  return true;
}
bool SyncEvent::host_wait(uint64_t timeout_ns) const {
  const ze_result_t r = zeEventHostSynchronize(e_, timeout_ns);
  if (r == ZE_RESULT_NOT_READY) return false;
  ZE_CHECK(r);
  return true;
}
}  // namespace l0
