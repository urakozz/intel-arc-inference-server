#include "l0/context.h"
#include <cstddef>
#include <stdexcept>
#include <string>
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
  if (ctx_ && owns_) zeContextDestroy(ctx_);
}

namespace {
// The GPUs of `driver`, in enumeration order - the same filter the primary constructor
// applies, for the spec 16b additions below.
std::vector<ze_device_handle_t> gpus_of(ze_driver_handle_t driver) {
  uint32_t n_dev = 0;
  ZE_CHECK(zeDeviceGet(driver, &n_dev, nullptr));
  std::vector<ze_device_handle_t> devs(n_dev);
  ZE_CHECK(zeDeviceGet(driver, &n_dev, devs.data()));
  std::vector<ze_device_handle_t> gpus;
  for (auto d : devs) {
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(d, &p));
    if (p.type == ZE_DEVICE_TYPE_GPU) gpus.push_back(d);
  }
  return gpus;
}
}  // namespace

Context::Context(const Context& primary, uint32_t device_index)
    : driver_(primary.driver_), ctx_(primary.ctx_), owns_(false) {
  const std::vector<ze_device_handle_t> gpus = gpus_of(driver_);
  if (device_index >= gpus.size())
    throw std::runtime_error("GPU index " + std::to_string(device_index) + " out of range (" +
                             std::to_string(gpus.size()) + " GPUs)");
  dev_ = gpus[device_index];
  props_.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZE_CHECK(zeDeviceGetProperties(dev_, &props_));
  compute_.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ZE_CHECK(zeDeviceGetComputeProperties(dev_, &compute_));
}

uint32_t Context::gpu_count() {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t n_drivers = 0;
  ZE_CHECK(zeDriverGet(&n_drivers, nullptr));
  if (n_drivers == 0) return 0;
  std::vector<ze_driver_handle_t> drivers(n_drivers);
  ZE_CHECK(zeDriverGet(&n_drivers, drivers.data()));
  return static_cast<uint32_t>(gpus_of(drivers[0]).size());
}

bool Context::can_access_peer(const Context& other) const {
  ze_bool_t ok = 0;
  ZE_CHECK(zeDeviceCanAccessPeer(dev_, other.dev_, &ok));
  return ok != 0;
}

namespace {
// Moved from Engine::memory_line() (spec 6), unchanged: every memory the device
// reports, summed - one HBM/GDDR region on the B70 (32.530 GB).
size_t device_memory(ze_device_handle_t dev) {
  uint32_t n = 0;
  zeDeviceGetMemoryProperties(dev, &n, nullptr);
  std::vector<ze_device_memory_properties_t> props(n);
  for (auto& p : props) p.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  if (n) zeDeviceGetMemoryProperties(dev, &n, props.data());
  size_t total = 0;
  for (const auto& p : props) total += p.totalSize;
  return total;
}

std::vector<ze_device_handle_t> visible_gpus() {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t n_drivers = 0;
  ZE_CHECK(zeDriverGet(&n_drivers, nullptr));
  if (n_drivers == 0) return {};
  std::vector<ze_driver_handle_t> drivers(n_drivers);
  ZE_CHECK(zeDriverGet(&n_drivers, drivers.data()));
  return gpus_of(drivers[0]);
}
}  // namespace

size_t Context::memory_bytes() const { return device_memory(dev_); }

std::vector<GpuInfo> Context::gpu_infos() {
  std::vector<GpuInfo> out;
  for (ze_device_handle_t d : visible_gpus()) {
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(d, &p));
    GpuInfo g;
    g.vendor_id = p.vendorId;
    g.device_id = p.deviceId;
    g.name = p.name;
    g.xe_cores = p.numSlices * p.numSubslicesPerSlice;
    g.memory_bytes = device_memory(d);
    out.push_back(std::move(g));
  }
  return out;
}

bool Context::peer_access(uint32_t a, uint32_t b) {
  const std::vector<ze_device_handle_t> gpus = visible_gpus();
  if (a >= gpus.size() || b >= gpus.size())
    throw std::runtime_error("GPU index out of range (" + std::to_string(gpus.size()) + " GPUs)");
  ze_bool_t ok = 0;
  ZE_CHECK(zeDeviceCanAccessPeer(gpus[a], gpus[b], &ok));
  return ok != 0;
}
}  // namespace l0
