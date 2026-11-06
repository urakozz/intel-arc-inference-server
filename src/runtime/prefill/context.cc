#include "runtime/prefill/context.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/copy_chunks.h"
#include "l0/event.h"

// The prefill Context: the Level Zero immediate list every prefill launch goes on (spec 2.1
// §3.5).
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
  size_t launches = 0;
  std::unique_ptr<LaunchProfiler> profiler;
  // Spec 16c: timestamp()'s clock, read on first use (0 = not yet).
  mutable double ts_hz = 0;
  mutable uint64_t ts_mask = 0;
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
  p_ = new Impl{ze_ctx, ze_dev, list, 0, nullptr};
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
  zeCommandListDestroy(p_->list);
  delete p_;
}

void Context::wait() {
  zc(zeCommandListHostSynchronize(p_->list, UINT64_MAX), "zeCommandListHostSynchronize");
}

void Context::launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                     std::initializer_list<KernelArg> args, l0::Event* signal) {
  using Clock = std::chrono::steady_clock;
  const auto begin = p_->profiler ? Clock::now() : Clock::time_point{};
  if (p_->profiler && signal)
    throw std::runtime_error("prefill profile: explicit signal event conflicts with profiling");
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
  zc(zeCommandListAppendLaunchKernel(p_->list, k, &g, event, 0,
                                     nullptr),
     "zeCommandListAppendLaunchKernel");
  if (p_->profiler) {
    ++p_->profiler->pending;
    ++p_->profiler->metrics.launches;
    p_->profiler->metrics.host_submit_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  }
  ++p_->launches;
}

void Context::copy(void* dst, const void* src, size_t bytes) {
  if (bytes <= common::kMaxCopyChunk) {
    zc(zeCommandListAppendMemoryCopy(p_->list, dst, src, bytes, nullptr, 0, nullptr),
       "zeCommandListAppendMemoryCopy");
    return;
  }
  // Fix round A: the l0::CmdList::copy rule (common/copy_chunks.h) - in-order pieces of at most 256 MiB.
  common::for_each_copy_chunk(bytes, common::kMaxCopyChunk, [&](size_t off, size_t n) {
    zc(zeCommandListAppendMemoryCopy(p_->list, static_cast<uint8_t*>(dst) + off,
                                     static_cast<const uint8_t*>(src) + off, n, nullptr, 0, nullptr),
       "zeCommandListAppendMemoryCopy");
  });
}

void Context::copy_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width,
                      size_t rows) {
  if (rows == 0) return;
  if (width > dst_pitch || width > src_pitch || dst_pitch > UINT32_MAX || src_pitch > UINT32_MAX ||
      rows > UINT32_MAX)
    throw std::runtime_error("prefill copy_2d: a row is wider than its pitch, or a pitch / count exceeds 32 bits");
  const ze_copy_region_t region{0, 0, 0, uint32_t(width), uint32_t(rows), 0};   // depth 0: a 2D copy
  zc(zeCommandListAppendMemoryCopyRegion(p_->list, dst, &region, uint32_t(dst_pitch), 0, src, &region,
                                         uint32_t(src_pitch), 0, nullptr, 0, nullptr),
     "zeCommandListAppendMemoryCopyRegion");
}

void Context::signal(ze_event_handle_t e) {
  zc(zeCommandListAppendBarrier(p_->list, e, 0, nullptr), "zeCommandListAppendBarrier (signal)");
}

void Context::wait_event(ze_event_handle_t e) {
  zc(zeCommandListAppendWaitOnEvents(p_->list, 1, &e), "zeCommandListAppendWaitOnEvents");
}

void Context::timestamp(uint64_t* dst) {
  zc(zeCommandListAppendWriteGlobalTimestamp(p_->list, dst, nullptr, 0, nullptr),
     "zeCommandListAppendWriteGlobalTimestamp");
}

bool Context::wait_for(uint64_t timeout_ns) {
  const ze_result_t r = zeCommandListHostSynchronize(p_->list, timeout_ns);
  if (r == ZE_RESULT_NOT_READY) return false;
  zc(r, "zeCommandListHostSynchronize (bounded)");
  return true;
}

double Context::timestamp_ms(uint64_t begin, uint64_t end) const {
  if (p_->ts_hz == 0) {
    ze_device_properties_t prop{};
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;   // timerResolution in cycles/s
    zc(zeDeviceGetProperties(p_->dev, &prop), "timestamp device properties");
    if (!prop.timerResolution || !prop.timestampValidBits || prop.timestampValidBits > 64)
      throw std::runtime_error("prefill::Context: invalid device timer calibration");
    p_->ts_mask = prop.timestampValidBits == 64 ? ~uint64_t{0}
                                                : (uint64_t{1} << prop.timestampValidBits) - 1;
    p_->ts_hz = double(prop.timerResolution);
  }
  return double((end - begin) & p_->ts_mask) * 1000.0 / p_->ts_hz;
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

}  // namespace runtime::prefill
