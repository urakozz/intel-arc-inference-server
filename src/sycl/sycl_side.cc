#include "runtime/prefill/sycl_side.h"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <vector>

// The SYCL half of a prefill Context (spec 2.1 §3.5) -- what src/sycl/context.cc's
// constructor built unconditionally before this stage, now built on demand.
namespace runtime::prefill {

struct SyclSide {
  sycl::device dev;
  sycl::context ctx;
  sycl::queue q;
};

SyclSide* sycl_side_create(ze_context_handle_t zctx, ze_device_handle_t zdev) {
  constexpr sycl::backend kL0 = sycl::backend::ext_oneapi_level_zero;
  sycl::device dev = sycl::make_device<kL0>(zdev);
  sycl::context ctx = sycl::make_context<kL0>(sycl::backend_input_t<kL0, sycl::context>{
      zctx, std::vector<sycl::device>{dev}, sycl::ext::oneapi::level_zero::ownership::keep});
  return new SyclSide{dev, ctx, sycl::queue(ctx, dev, sycl::property::queue::in_order{})};
}
void sycl_side_destroy(SyclSide* s) { delete s; }
void sycl_side_wait(SyclSide* s) { s->q.wait_and_throw(); }
void* sycl_side_queue(SyclSide* s) { return &s->q; }
void* sycl_side_context(SyclSide* s) { return &s->ctx; }

}  // namespace runtime::prefill
