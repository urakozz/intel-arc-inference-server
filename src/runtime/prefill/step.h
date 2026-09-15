#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

namespace runtime::prefill {

// ONE chunk of at most `PrefillScratch::kC` positions, at absolute position
// `pos`. The caller has already uploaded the chunk's ids into `s.ids` and set
// `Control::{pos, n_active}` (after a `wait()`, because the L0 list is
// asynchronous). This function walks all 64 layers.
//
// **It does NOT return without waiting.** Plan 6b Task 10 wrote `step_chunk` as
// pure appending, which was true while every op was an L0 launch. It is not
// true now: rulings A23/A24 put the int4 linears on the two-pass
// `dequant_to_bf16` (L0) -> `gemm_bf16` (SYCL) path with a host `wait()`
// between them, and A24 measured that the B70 has ONE compute queue, two
// software queues, and no device-side cross-runtime dependency to order them
// with. So the walk synchronises at every L0<->SYCL boundary, and the caller
// gets a partially-drained list. That is a deviation from the plan's prose,
// forced by a ruling that post-dates it, and it is the reason the walk's cost
// carries a handoff term at all (656 waits per chunk at C > 0; see docs/15).
//
// The four persistent `l0::Mem&` are passed rather than reached through an
// `Engine`, so this function holds no engine reference and is unit-testable.
void step_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C,
                l0::Mem& gdn_state_mem, l0::Mem& conv_ring_mem, l0::Mem& kv_k_mem,
                l0::Mem& kv_v_mem, PrefillBackend backend);

// The tail only the LAST chunk runs: the final norm on the last row, `lm_head`
// at M = 1 through the EXISTING decode binary, and the two argmax stages --
// the `Control` handoff. `last_row` is the row of the last chunk that holds the
// final position.
void step_head(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
               void* ctrl, uint32_t last_row);

// The launch arithmetic, derived from the walk itself rather than restated:
// what `Context::launches()` advances by. SYCL GEMMs are NOT on the L0 list and
// therefore not counted here; `step_chunk_gemms()` reports those separately.
size_t step_chunk_launches(PrefillBackend b);   // per chunk, independent of C
size_t step_chunk_gemms(PrefillBackend b);      // SYCL GEMM calls per chunk (0 on L0)
size_t step_chunk_waits(PrefillBackend b);      // host L0<->SYCL handoffs per chunk (0 on L0)
inline constexpr size_t kStepHeadLaunches = 5;   // 2 norm + lm_head + 2 argmax

}  // namespace runtime::prefill
