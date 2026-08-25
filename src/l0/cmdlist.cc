#include "l0/cmdlist.h"
#include "l0/error.h"
#include "l0/event.h"
#include "l0/kernel.h"

namespace l0 {
CmdList CmdList::immediate(Context& ctx, uint32_t ordinal) {
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.ordinal = ordinal;
  qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t l = nullptr;
  ZE_CHECK(zeCommandListCreateImmediate(ctx.handle(), ctx.device(), &qd, &l));
  return CmdList(l, true);
}

CmdList CmdList::regular(Context& ctx, uint32_t ordinal) {
  ze_command_list_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  d.commandQueueGroupOrdinal = ordinal;
  d.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
  ze_command_list_handle_t l = nullptr;
  ZE_CHECK(zeCommandListCreate(ctx.handle(), ctx.device(), &d, &l));
  return CmdList(l, false);
}

CmdList::~CmdList() {
  if (l_) zeCommandListDestroy(l_);
}
CmdList::CmdList(CmdList&& o) noexcept : l_(o.l_), immediate_(o.immediate_) { o.l_ = nullptr; }

void CmdList::copy(void* dst, const void* src, size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryCopy(l_, dst, src, bytes, nullptr, 0, nullptr));
}
void CmdList::fill(void* dst, uint32_t pattern, size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryFill(l_, dst, &pattern, sizeof pattern, bytes, nullptr, 0, nullptr));
}
void CmdList::launch(Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz, Event* signal) {
  ze_group_count_t gc{gx, gy, gz};
  ZE_CHECK(zeCommandListAppendLaunchKernel(l_, k.handle(), &gc,
                                           signal ? signal->handle() : nullptr, 0, nullptr));
}
void CmdList::close() { ZE_CHECK(zeCommandListClose(l_)); }
void CmdList::reset() { ZE_CHECK(zeCommandListReset(l_)); }
}  // namespace l0
