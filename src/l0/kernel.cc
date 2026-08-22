#include "l0/kernel.h"
#include "l0/module.h"

namespace l0 {
Kernel::Kernel(Module& m, const char* name) {
  ze_kernel_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  d.pKernelName = name;
  ZE_CHECK(zeKernelCreate(m.handle(), &d, &k_));
}
Kernel::~Kernel() {
  if (k_) zeKernelDestroy(k_);
}
}  // namespace l0
