#include "runtime/prefill/context.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/event.h"
#include "runtime/prefill/sycl_side.h"

// The prefill Context's Level Zero half (spec 2.1 §3.5). Before spec 2.1 this file was
// src/sycl/context.cc and built a SYCL queue in its constructor; the queue is now a SyclSide
// created on the first sycl() call, so a chunk on the L0 backend never touches SYCL and this
// translation unit compiles with g++ in every build.
namespace runtime::prefill {
namespace {
void zc(ze_result_t r, const char* what) {
  if (r == ZE_RESULT_SUCCESS) return;
  char b[24];
  std::snprintf(b, sizeof b, "0x%X", unsigned(r));
  throw std::runtime_error(std::string("prefill::Context: ") + what + " failed: " + b);
}
std::string kernel_name(ze_kernel_handle_t k) {
  size_t n = 0;
  if (zeKernelGetName(k, &n, nullptr) != ZE_RESULT_SUCCESS || n == 0) return "<unnamed kernel>";
  std::string s(n, '\0');
  if (zeKernelGetName(k, &n, s.data()) != ZE_RESULT_SUCCESS) return "<unnamed kernel>";
  if (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

// Events are reused only after the list has drained. Pools grow in blocks so
// diagnostics also cover callers with thousands of launches between waits.
struct LaunchProfiler {
  static constexpr uint32_t kPoolSize = 256;
  ze_context_handle_t ctx;
  ze_device_handle_t dev;
  std::vector<ze_event_pool_handle_t> pools;
  std::vector<ze_event_handle_t> events;
  size_t pending = 0;
  double cycles_per_ms;
  uint64_t mask;
  Context::LaunchMetrics metrics;

  LaunchProfiler(ze_context_handle_t c, ze_device_handle_t d) : ctx(c), dev(d) {
    ze_device_properties_t prop{};
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zc(zeDeviceGetProperties(dev, &prop), "profile device properties");
    if (!prop.timerResolution || !prop.kernelTimestampValidBits ||
        prop.kernelTimestampValidBits > 64)
      throw std::runtime_error("prefill profile: invalid device timer calibration");
    cycles_per_ms = double(prop.timerResolution) / 1000.0;
    mask = prop.kernelTimestampValidBits == 64 ? ~uint64_t{0}
        : (uint64_t{1} << prop.kernelTimestampValidBits) - 1;
  }
  ~LaunchProfiler() {
    for (auto e : events) zeEventDestroy(e);
    for (auto pool : pools) zeEventPoolDestroy(pool);
  }
  ze_event_handle_t next() {
    if (pending == events.size()) {
      if (events.size() % kPoolSize == 0) {
        ze_event_pool_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        desc.count = kPoolSize;
        ze_event_pool_handle_t pool = nullptr;
        zc(zeEventPoolCreate(ctx, &desc, 1, &dev, &pool), "profile event pool");
        pools.push_back(pool);
      }
      ze_event_desc_t desc{};
      desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
      desc.index = uint32_t(events.size() % kPoolSize);
      desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      ze_event_handle_t event = nullptr;
      zc(zeEventCreate(pools.back(), &desc, &event), "profile event");
      events.push_back(event);
    }
    return events[pending];
  }
  void collect() {
    for (size_t i = 0; i < pending; ++i) {
      ze_kernel_timestamp_result_t ts{};
      zc(zeEventQueryKernelTimestamp(events[i], &ts), "profile timestamp");
      const uint64_t start = ts.global.kernelStart & mask;
      const uint64_t end = ts.global.kernelEnd & mask;
      metrics.gpu_ms += double((end - start) & mask) / cycles_per_ms;
      zc(zeEventHostReset(events[i]), "profile event reset");
    }
    pending = 0;
  }
};
}  // namespace

struct Context::Impl {
  ze_context_handle_t ctx;
  ze_device_handle_t dev;
  ze_command_list_handle_t list;
  SyclSide* sycl = nullptr;   // lazily created; owned
  size_t launches = 0;
  std::unique_ptr<LaunchProfiler> profiler;
  Recording* recording = nullptr;  // non-owning, only during capture callback
  ze_command_queue_handle_t replay_queue = nullptr;
};

Context::Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev) {
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.ordinal = 0;
  qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t list = nullptr;
  zc(zeCommandListCreateImmediate(ze_ctx, ze_dev, &qd, &list), "zeCommandListCreateImmediate");
  p_ = new Impl{ze_ctx, ze_dev, list, nullptr, 0, nullptr};
  const char* profile = std::getenv("B70_PREFILL_PROFILE");
  if (profile && std::strcmp(profile, "1") == 0) {
    try {
      p_->profiler = std::make_unique<LaunchProfiler>(ze_ctx, ze_dev);
    } catch (...) {
      zeCommandListDestroy(list);
      delete p_;
      throw;
    }
  }
}

Context::~Context() {
  if (!p_) return;
  // Keep timestamp events alive until any pending launches have finished.
  zeCommandListHostSynchronize(p_->list, UINT64_MAX);
  if (p_->replay_queue) {
    zeCommandQueueSynchronize(p_->replay_queue, UINT64_MAX);
    zeCommandQueueDestroy(p_->replay_queue);
  }
  if (p_->sycl) sycl_side_destroy(p_->sycl);
  zeCommandListDestroy(p_->list);
  delete p_;
}

void Context::wait() {
  if (p_->recording) throw std::runtime_error("prefill capture: wait inside recording");
  if (p_->sycl) sycl_side_wait(p_->sycl);
  zc(zeCommandListHostSynchronize(p_->list, UINT64_MAX), "zeCommandListHostSynchronize");
}

SyclSide& Context::sycl() {
  if (p_->recording) throw std::runtime_error("prefill capture: SYCL inside recording");
  if (!p_->sycl) p_->sycl = sycl_side_create(p_->ctx, p_->dev);
  return *p_->sycl;
}
bool Context::has_sycl() const { return p_->sycl != nullptr; }

void Context::launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                     std::initializer_list<KernelArg> args, l0::Event* signal) {
  using Clock = std::chrono::steady_clock;
  const auto begin = p_->profiler ? Clock::now() : Clock::time_point{};
  if (p_->profiler && signal)
    throw std::runtime_error("prefill profile: explicit signal event conflicts with profiling");
  if (p_->recording && signal)
    throw std::runtime_error("prefill capture: explicit signal event is unsupported");
  uint32_t i = 0;
  for (const KernelArg& a : args) {
    const ze_result_t r = zeKernelSetArgumentValue(k, i, a.size, a.ptr);
    if (r != ZE_RESULT_SUCCESS) {
      char b[24];
      std::snprintf(b, sizeof b, "0x%X", unsigned(r));
      throw std::runtime_error("prefill::Context::launch: " + kernel_name(k) + " argument " +
                               std::to_string(i) + " (" + std::to_string(a.size) +
                               " bytes) rejected: " + b);
    }
    ++i;
  }
  // The group size comes from the kernel's own reqd_work_group_size (every prefill kernel
  // declares one); a kernel without one is rejected by name, never launched at a guess.
  ze_kernel_properties_t kp{};
  kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  zc(zeKernelGetProperties(k, &kp), "zeKernelGetProperties");
  if (kp.requiredGroupSizeX == 0)
    throw std::runtime_error("prefill::Context::launch: " + kernel_name(k) +
                             " has no reqd_work_group_size");
  zc(zeKernelSetGroupSize(k, kp.requiredGroupSizeX, kp.requiredGroupSizeY, kp.requiredGroupSizeZ),
     "zeKernelSetGroupSize");
  ze_group_count_t g{gx, gy, gz};
  ze_event_handle_t event = p_->profiler ? p_->profiler->next()
                                       : signal ? signal->handle() : nullptr;
  const auto target = p_->recording ? p_->recording->list_ : p_->list;
  zc(zeCommandListAppendLaunchKernel(target, k, &g, event, 0,
                                     nullptr),
     "zeCommandListAppendLaunchKernel");
  if (p_->profiler) {
    ++p_->profiler->pending;
    ++p_->profiler->metrics.launches;
    p_->profiler->metrics.host_submit_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }
  if (p_->recording) ++p_->recording->launches_;
  else ++p_->launches;
}

ze_context_handle_t Context::ze_context() const { return p_->ctx; }
ze_device_handle_t Context::ze_device() const { return p_->dev; }
ze_command_list_handle_t Context::l0_list() const { return p_->list; }
size_t Context::launches() const { return p_->launches; }
void Context::reset_launches() { p_->launches = 0; }

void Context::set_profiling(bool enabled) {
  wait();
  if (enabled && !p_->profiler)
    p_->profiler = std::make_unique<LaunchProfiler>(p_->ctx, p_->dev);
  if (!enabled) p_->profiler.reset();
}

Context::LaunchMetrics Context::take_launch_metrics() {
  if (!p_->profiler) return {};
  p_->profiler->collect();
  const auto metrics = p_->profiler->metrics;
  p_->profiler->metrics = {};
  return metrics;
}

Context::Recording::~Recording() {
  if (list_) zeCommandListDestroy(list_);
}

std::unique_ptr<Context::Recording> Context::capture(const std::function<void()>& encode) {
  if (p_->recording || p_->profiler || p_->sycl)
    throw std::runtime_error("prefill capture: requires an unprofiled, non-recording L0 context");
  wait();
  auto recording = std::unique_ptr<Recording>(new Recording(this));
  ze_command_list_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  desc.commandQueueGroupOrdinal = 0;
  desc.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
  zc(zeCommandListCreate(p_->ctx, p_->dev, &desc, &recording->list_), "capture list create");
  p_->recording = recording.get();
  try {
    encode();
    zc(zeCommandListClose(recording->list_), "capture list close");
  } catch (...) {
    p_->recording = nullptr;
    throw;
  }
  p_->recording = nullptr;
  return recording;
}

void Context::replay(const Recording& recording) {
  if (recording.owner_ != this || p_->recording || p_->profiler || p_->sycl)
    throw std::runtime_error("prefill replay: requires its owning, unprofiled L0 context");
  wait();
  if (!p_->replay_queue) {
    ze_command_queue_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    desc.ordinal = 0;
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    zc(zeCommandQueueCreate(p_->ctx, p_->dev, &desc, &p_->replay_queue), "replay queue create");
  }
  auto list = recording.list_;
  zc(zeCommandQueueExecuteCommandLists(p_->replay_queue, 1, &list, nullptr), "replay execute");
  zc(zeCommandQueueSynchronize(p_->replay_queue, UINT64_MAX), "replay synchronize");
  p_->launches += recording.launches_;
}

}  // namespace runtime::prefill
