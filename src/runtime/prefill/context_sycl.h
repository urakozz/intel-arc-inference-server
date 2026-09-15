#pragma once
#include <sycl/sycl.hpp>

#include "runtime/prefill/context.h"
#include "runtime/prefill/sycl_side.h"

namespace runtime::prefill {
inline sycl::queue& sycl_queue(Context& cx) {
  return *static_cast<sycl::queue*>(sycl_side_queue(&cx.sycl()));
}
inline sycl::context& sycl_context(Context& cx) {
  return *static_cast<sycl::context*>(sycl_side_context(&cx.sycl()));
}
inline sycl::queue& sycl_queue(SyclSide& s) { return *static_cast<sycl::queue*>(sycl_side_queue(&s)); }
}  // namespace runtime::prefill
