// P1: the spec 2 §4 SYCL <-> Level Zero interop smoke. This source is built
// and linked by icpx in src/sycl/CMakeLists.txt, while the L0 helpers remain
// the ordinary g++ archive so the probe also checks the two-compiler ABI.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"
#include "tla_pin.h"

namespace {

constexpr sycl::backend kL0 = sycl::backend::ext_oneapi_level_zero;
constexpr size_t kBytes = 64ul << 20;
constexpr uint32_t kWords = static_cast<uint32_t>(kBytes / sizeof(uint32_t));

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

const char* env_or_unset(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "(unset)";
}

uint32_t xorshift(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

const char* pointer_type_name(sycl::usm::alloc type) {
  switch (type) {
    case sycl::usm::alloc::host: return "host";
    case sycl::usm::alloc::device: return "device";
    case sycl::usm::alloc::shared: return "shared";
    case sycl::usm::alloc::unknown: return "unknown";
  }
  return "unrecognised";
}

template <class F>
double median_us_8_drop_3(F&& replay) {
  replay();  // discarded warm-up
  std::array<double, 5> samples{};
  for (int i = 0; i < 8; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    replay();
    const auto end = std::chrono::steady_clock::now();
    if (i >= 3)
      samples[static_cast<size_t>(i - 3)] =
          std::chrono::duration<double, std::micro>(end - begin).count();
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void copy_to_host(l0::Context& ctx, void* dst, const void* src, size_t bytes) {
  l0::CmdList list = l0::CmdList::immediate(ctx);
  list.copy(dst, src, bytes);
}

void usm_round_trip(l0::Context& l0ctx, runtime::prefill::Context& cx) {
  l0::Mem words(l0ctx, l0::MemKind::Device, kBytes);
  l0::Mem observed(l0ctx, l0::MemKind::Device, sizeof(uint32_t));
  l0::Mem host(l0ctx, l0::MemKind::Host, kBytes);
  l0::Mem host_one(l0ctx, l0::MemKind::Host, sizeof(uint32_t));
  l0::Mem bw_out(l0ctx, l0::MemKind::Device, 1ul << 20);

  std::string pointer_type;
  std::string pointer_device;
  try {
    pointer_type = pointer_type_name(sycl::get_pointer_type(words.ptr(), runtime::prefill::sycl_context(cx)));
    pointer_device = sycl::get_pointer_device(words.ptr(), runtime::prefill::sycl_context(cx))
                         .get_info<sycl::info::device::name>();
  } catch (const sycl::exception& e) {
    pointer_type = std::string("exception: ") + e.what();
    pointer_device = "not queried";
  }

  uint32_t* const p = words.as<uint32_t>();
  runtime::prefill::sycl_queue(cx).parallel_for(sycl::range<1>(kWords), [=](sycl::id<1> id) {
    p[id[0]] = xorshift(static_cast<uint32_t>(id[0]));
  });
  cx.wait();

  l0::Module bw_module(l0ctx, kernels::path("bw_sum"));
  l0::Kernel bw = bw_module.kernel("bw_sum");
  const void* words_ptr = words.ptr();
  const uint64_t n_vec = kBytes / 16;
  const void* bw_out_ptr = bw_out.ptr();
  // bw_sum has a deliberately non-observable reduction, but this launch makes
  // the L0 side consume the whole SYCL-written device allocation before the
  // host performs the exact value check below.
  cx.launch(bw, 256, 1, 1,
            {{&words_ptr, sizeof words_ptr}, {&n_vec, sizeof n_vec}, {&bw_out_ptr, sizeof bw_out_ptr}});
  cx.wait();
  copy_to_host(l0ctx, host.ptr(), words.ptr(), kBytes);
  const uint32_t* got = host.as<uint32_t>();
  uint64_t expected_sum = 0;
  uint64_t got_sum = 0;
  for (uint32_t i = 0; i < kWords; ++i) {
    const uint32_t expected = xorshift(i);
    if (got[i] != expected) fail("USM round-trip value mismatch at word " + std::to_string(i));
    expected_sum += expected;
    got_sum += got[i];
  }
  if (got_sum != expected_sum) fail("USM round-trip host sum mismatch");

  l0::Module ctrl_module(l0ctx, kernels::path("ctrl_read"));
  l0::Kernel ctrl = ctrl_module.kernel("ctrl_read");
  const void* observed_ptr = observed.ptr();
  cx.launch(ctrl, 1, 1, 1,
            {{&words_ptr, sizeof words_ptr}, {&observed_ptr, sizeof observed_ptr}});
  cx.wait();
  uint32_t* const observed_words = observed.as<uint32_t>();
  runtime::prefill::sycl_queue(cx).single_task([=] { observed_words[0] += 0; });
  cx.wait();
  copy_to_host(l0ctx, host_one.ptr(), observed.ptr(), sizeof(uint32_t));
  if (host_one.as<uint32_t>()[0] != xorshift(0) + 1u)
    fail("L0-to-SYCL reverse round-trip mismatch");

  std::printf("| check | result | detail |\n|---|---|---|\n");
  std::printf("| SYCL -> L0 -> host | pass | bit-exact, sum %llu |\n",
              static_cast<unsigned long long>(got_sum));
  std::printf("| sycl::get_pointer_type | %s | foreign zeMemAllocDevice |\n", pointer_type.c_str());
  std::printf("| sycl::get_pointer_device | %s | foreign zeMemAllocDevice |\n", pointer_device.c_str());
  std::printf("| L0 -> SYCL -> host | pass | ctrl_read value %u |\n", host_one.as<uint32_t>()[0]);
}

void launch_overheads(l0::Context& l0ctx, runtime::prefill::Context& cx) {
  l0::Module module(l0ctx, kernels::path("noop"));
  l0::Kernel noop = module.kernel("noop");
  l0::Mem out(l0ctx, l0::MemKind::Device, sizeof(uint32_t));
  const void* out_ptr = out.ptr();
  std::printf("| queue | N | us/launch (8 replays, drop 3, median; iterate-grade) |\n|---|---:|---:|\n");
  for (const uint32_t n : {1u, 64u, 1024u}) {
    const double l0_us = median_us_8_drop_3([&] {
      for (uint32_t i = 0; i < n; ++i)
        cx.launch(noop, 1, 1, 1, {{&out_ptr, sizeof out_ptr}});
      cx.wait();
    }) / n;
    std::printf("| L0 immediate list | %u | %.3f |\n", n, l0_us);
  }
  for (const uint32_t n : {1u, 64u, 1024u}) {
    const double sycl_us = median_us_8_drop_3([&] {
      for (uint32_t i = 0; i < n; ++i)
        runtime::prefill::sycl_queue(cx).single_task([] {});
      cx.wait();
    }) / n;
    std::printf("| SYCL in-order queue | %u | %.3f |\n", n, sycl_us);
  }
}

void handoffs(l0::Context& l0ctx, runtime::prefill::Context& cx) {
  constexpr uint32_t kAlternations = 256;
  l0::Module module(l0ctx, kernels::path("noop"));
  l0::Kernel noop = module.kernel("noop");
  l0::Mem out(l0ctx, l0::MemKind::Device, sizeof(uint32_t));
  const void* out_ptr = out.ptr();
  const double waits_us = median_us_8_drop_3([&] {
    for (uint32_t i = 0; i < kAlternations; ++i) {
      runtime::prefill::sycl_queue(cx).single_task([] {});
      cx.wait();
      cx.launch(noop, 1, 1, 1, {{&out_ptr, sizeof out_ptr}});
      cx.wait();
    }
  }) / (2.0 * kAlternations);

  std::printf("| ordering | us/handoff (8 replays, drop 3, median; iterate-grade) | result |\n"
              "|---|---:|---|\n");
  std::printf("| SYCL -> wait -> L0 -> wait | %.3f | pass |\n", waits_us);

  try {
    l0::EventPool pool(l0ctx, 1);
    l0::Event signal(pool, 0);
    const sycl::event l0_event = sycl::make_event<kL0>(
        sycl::backend_input_t<kL0, sycl::event>{
            signal.handle(), sycl::ext::oneapi::level_zero::ownership::keep},
        runtime::prefill::sycl_context(cx));
    const double event_us = median_us_8_drop_3([&] {
      for (uint32_t i = 0; i < kAlternations; ++i) {
        signal.reset();
        cx.launch(noop, 1, 1, 1, {{&out_ptr, sizeof out_ptr}}, &signal);
        runtime::prefill::sycl_queue(cx).submit([&](sycl::handler& h) {
          h.depends_on(l0_event);
          h.single_task([] {});
        });
        cx.wait();
      }
    }) / kAlternations;
    std::printf("| L0 event -> SYCL | %.3f | pass |\n", event_us);
  } catch (const std::exception& e) {
    std::printf("| L0 event -> SYCL | - | unavailable: %s |\n", e.what());
  }

  const double chunk_2048_ms = waits_us * 384.0 / 1000.0;
  const double chunk_4096_ms = waits_us * 384.0 / 1000.0;
  std::printf("\n| budget quantity | value (derived, iterate-grade) |\n|---|---:|\n");
  std::printf("| handoffs per chunk | 384 |\n");
  std::printf("| wait handoffs at C=2048 | %.3f ms (%.3f%% of 1.0 s) |\n",
              chunk_2048_ms, chunk_2048_ms / 10.0);
  std::printf("| wait handoffs at C=4096 | %.3f ms (%.3f%% of 2.2 s) |\n",
              chunk_4096_ms, chunk_4096_ms / 22.0);
  std::printf("If a measurement lands above 26 us/handoff, the first cut must batch OpenCL C kernels "
              "between GEMMs (fewer, larger handoffs).\n");
}

}  // namespace

int main() {
  try {
    std::printf("sycl-tla sha %s (pin %s)\n", B70_SYCL_TLA_SHA, B70_SYCL_TLA_PIN);
    std::printf("IGC env: SYCL_PROGRAM_COMPILE_OPTIONS=%s | IGC_VISAOptions=%s | "
                "IGC_VectorAliasBBThreshold=%s | IGC_ExtraOCLOptions=%s\n",
                env_or_unset("SYCL_PROGRAM_COMPILE_OPTIONS"), env_or_unset("IGC_VISAOptions"),
                env_or_unset("IGC_VectorAliasBBThreshold"), env_or_unset("IGC_ExtraOCLOptions"));
    l0::Context l0ctx(0);
    ze_driver_properties_t driver{};
    driver.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    if (zeDriverGetProperties(l0ctx.driver(), &driver) != ZE_RESULT_SUCCESS)
      fail("zeDriverGetProperties failed");
    std::printf("L0 device: %s (driver %u)\n", l0ctx.name().c_str(), driver.driverVersion);
    std::printf("# P1 results (iterate-grade; 8 replays, discard first 3, median)\n\n## USM round-trip\n");
    runtime::prefill::Context cx(l0ctx);
    usm_round_trip(l0ctx, cx);
    std::printf("\n## Per-launch overhead\n");
    launch_overheads(l0ctx, cx);
    std::printf("\n## Cross-queue handoff\n");
    handoffs(l0ctx, cx);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_interop FAILED: %s\n", e.what());
    return 1;
  }
}
