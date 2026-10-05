#include "l0/context.h"
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include "l0/error.h"

namespace l0 {
namespace {
constexpr char kSelectorVar[] = "ONEAPI_DEVICE_SELECTOR";
constexpr char kPrefix[] = "level_zero:";
constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;

// One message for every rejection: the operator sees what they typed and the
// only two things that work. Silently falling back to device 0 would run the
// model on a card they did not ask for.
[[noreturn]] void reject(const std::string& value) {
  throw std::runtime_error(std::string(kSelectorVar) + "=\"" + value +
                           "\" is not supported by the Level Zero decode path: it accepts "
                           "\"level_zero:N\" (one GPU index) or \"level_zero:*\" (device 0). "
                           "Pass --device N to choose another card.");
}
}  // namespace

uint32_t Context::device_index_from_env() {
  const char* raw = std::getenv(kSelectorVar);
  if (raw == nullptr || raw[0] == '\0') return 0;  // unset, or exported empty
  const std::string v(raw);
  if (v.compare(0, kPrefixLen, kPrefix) != 0) reject(v);
  const std::string index = v.substr(kPrefixLen);
  if (index == "*") return 0;
  if (index.empty()) reject(v);
  // Digits only: this is what rejects `0,1` and `0;opencl:*` (lists), signs,
  // ranges and everything else the real selector grammar allows and we do not.
  uint64_t n = 0;
  for (const char c : index) {
    if (c < '0' || c > '9') reject(v);
    n = n * 10 + static_cast<uint64_t>(c - '0');
    if (n >= kFromEnv) reject(v);  // no uint32 index, and never kFromEnv itself
  }
  return static_cast<uint32_t>(n);
}

Context::Context(uint32_t device_index) {
  // Before zeInit: a bad selector is a configuration error and should say so
  // on a machine with no driver at all.
  if (device_index == kFromEnv) device_index = device_index_from_env();
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

size_t Context::memory_bytes() const {
  // Moved from Engine::memory_line() (spec 6), unchanged: every memory the device
  // reports, summed - one HBM/GDDR region on the B70 (32.530 GB).
  uint32_t n = 0;
  zeDeviceGetMemoryProperties(dev_, &n, nullptr);
  std::vector<ze_device_memory_properties_t> props(n);
  for (auto& p : props) p.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  if (n) zeDeviceGetMemoryProperties(dev_, &n, props.data());
  size_t total = 0;
  for (const auto& p : props) total += p.totalSize;
  return total;
}
}  // namespace l0
