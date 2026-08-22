#include "l0/queue.h"
#include "l0/cmdlist.h"
#include "l0/error.h"
#include "l0/fence.h"

namespace l0 {
Queue::Queue(Context& ctx, uint32_t ordinal) : ctx_(&ctx) {
  ze_command_queue_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  d.ordinal = ordinal;
  d.index = 0;
  d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ZE_CHECK(zeCommandQueueCreate(ctx.handle(), ctx.device(), &d, &q_));
}
Queue::~Queue() {
  if (q_) zeCommandQueueDestroy(q_);
}
void Queue::execute(CmdList& list, Fence* fence) {
  ze_command_list_handle_t h = list.handle();
  ZE_CHECK(zeCommandQueueExecuteCommandLists(q_, 1, &h, fence ? fence->handle() : nullptr));
}
void Queue::synchronize() { ZE_CHECK(zeCommandQueueSynchronize(q_, UINT64_MAX)); }
}  // namespace l0
