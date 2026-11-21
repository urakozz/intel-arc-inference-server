// Spec 39 Task 1: this file must NOT compile (tests/CMakeLists.txt builds it as an excluded target and passes when the
// build prints runtime/gpu_state.h's refusal). GPU_STATE_FAIL_UNANNOTATED: a sizes member added without an annotation.
// GPU_STATE_FAIL_UNPLANNED: a buffer added without a sizes member - PERF-116's first idx_q4 / idx_keys4, which sat
// outside the plan and the snapshot.
#include <cstdint>
#include <memory>

#include "runtime/gpu_state.h"

namespace gs = runtime::gpu_state;

struct FakeMem {
  size_t size() const { return 0; }
};
enum class T : uint32_t { A };

struct Sizes {
  [[=gs::kv, =gs::zero_on_reset, =gs::Snapshot<T>{T::A}]] size_t idx_keys = 0;
  [[=gs::scratch]] size_t idx_q = 0;
#ifdef GPU_STATE_FAIL_UNANNOTATED
  size_t idx_q4 = 0;
#endif
};
struct Bufs {
  FakeMem idx_keys, idx_q;
#if defined(GPU_STATE_FAIL_UNANNOTATED) || defined(GPU_STATE_FAIL_UNPLANNED)
  std::unique_ptr<FakeMem> idx_q4;
#endif
};
static_assert(gs::sizes_problem<Sizes, T>().empty(), gs::sizes_problem<Sizes, T>());
static_assert(gs::buffers_problem<FakeMem, Bufs, Sizes>().empty(), gs::buffers_problem<FakeMem, Bufs, Sizes>());
