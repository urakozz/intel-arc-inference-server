#pragma once

#include <sycl/sycl.hpp>

#include "runtime/prefill/context.h"

namespace runtime::prefill {

inline sycl::queue& sycl_queue(Context& cx) {
  return *static_cast<sycl::queue*>(cx.sycl_queue_raw());
}

inline sycl::context& sycl_context(Context& cx) {
  return *static_cast<sycl::context*>(cx.sycl_context_raw());
}

}  // namespace runtime::prefill
