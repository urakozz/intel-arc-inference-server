#include "l0/context.h"
#include <vector>
#include "l0/error.h"

namespace l0 {
Context::Context(uint32_t device_index) {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t n_drivers = 0;
  ZE_CHECK(zeDriverGet(&n_drivers, nullptr));
  if (n_drivers == 0) throw std::runtime_error("no Level Zero driver");
  std::vector<ze_driver_handle_t> drivers(n_drivers);
  ZE_CHECK(zeDriverGet(&n_drivers, drivers.data()));
  driver_ = drivers[0];

  uint32_t n_dev = 0;
  ZE_CHECK(zeDeviceGet(driver_, &n_dev, nullptr));
  std::vector<ze_device_handle_t> devs(n_dev);
  ZE_CHECK(zeDeviceGet(driver_, &n_dev, devs.data()));
  std::vector<ze_device_handle_t> gpus;
  for (auto d : devs) {
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(d, &p));
    if (p.type == ZE_DEVICE_TYPE_GPU) gpus.push_back(d);
  }
  if (device_index >= gpus.size())
    throw std::runtime_error("GPU index " + std::to_string(device_index) + " out of range (" +
                             std::to_string(gpus.size()) + " GPUs)");
  dev_ = gpus[device_index];
  props_.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZE_CHECK(zeDeviceGetProperties(dev_, &props_));
  compute_.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ZE_CHECK(zeDeviceGetComputeProperties(dev_, &compute_));

  ze_context_desc_t cd{};
  cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ZE_CHECK(zeContextCreate(driver_, &cd, &ctx_));
}

Context::~Context() {
  if (ctx_) zeContextDestroy(ctx_);
}
}  // namespace l0
