#pragma once
#include <level_zero/ze_api.h>

// The SYCL half of a prefill Context (spec 2.1 §3.5): a sycl::device, sycl::context and
// in-order sycl::queue made over the SAME Level Zero context the list uses. Opaque here so
// the g++ half never includes <sycl/sycl.hpp>. Built by src/sycl/sycl_side.cc (icpx,
// libb70_prefill.so) and created ONLY when Context::sycl() is first called -- which only the
// sycl-tla backend does. A build without the component defines sycl_side_create to throw
// (backend_sycl_absent.cc).
namespace runtime::prefill {
struct SyclSide;
SyclSide* sycl_side_create(ze_context_handle_t ctx, ze_device_handle_t dev);
void sycl_side_destroy(SyclSide* s);
void sycl_side_wait(SyclSide* s);       // queue.wait_and_throw()
void* sycl_side_queue(SyclSide* s);     // sycl::queue*
void* sycl_side_context(SyclSide* s);   // sycl::context*
}  // namespace runtime::prefill
