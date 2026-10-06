#include "l0/fence.h"
#include "l0/error.h"

namespace l0 {
Fence::Fence(Queue& q) {
  ze_fence_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
  ZE_CHECK(zeFenceCreate(q.handle(), &d, &f_));
}
Fence::~Fence() {
  if (f_) zeFenceDestroy(f_);
}
bool Fence::wait_for(uint64_t timeout_ns) {
  const ze_result_t r = zeFenceHostSynchronize(f_, timeout_ns);
  if (r == ZE_RESULT_NOT_READY) return false;
  ZE_CHECK(r);
  ZE_CHECK(zeFenceReset(f_));
  return true;
}

void Fence::wait() {
  ZE_CHECK(zeFenceHostSynchronize(f_, UINT64_MAX));
  ZE_CHECK(zeFenceReset(f_));
}
}  // namespace l0
