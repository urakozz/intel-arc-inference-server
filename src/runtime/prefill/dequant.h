#pragma once
#include <cstdint>

#include "loader/loader.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace runtime::prefill {

// Dequantise ONE int4 linear's weights -- whatever tile layout the loader chose
// for that shape (layout 0: `qweight [K/8][N]` u32 + an independent f16
// `scales [K/64][N]`; layout 1: 136-u32 tiles with the f16 scale inline) --
// into the bf16 scratch `[K][N]` row-major, `ldb = N`. That is exactly
// `gemm_bf16`'s B operand (interfaces.md, stream S1).
//
// **This is THE GEMM path, by ruling A23 and A24, and it is two passes with a
// host `wait()` between them.** Every alternative was measured and is dead:
// the int4 mixed-input mainloop runs the same shape at 42.21 TFLOP/s against
// the bf16 scratch's 150.19 (A23); the dequant/GEMM overlap recovers 0.127 of
// the dequant against a 0.3 bar because the B70 exposes exactly one compute
// queue (A24, Probe B); the L2-resident slab is -70.1 ms/chunk because each
// slab needs a 22.35 us host handoff at the L0 -> SYCL boundary (A24, Probe A).
// So the caller launches this, calls `Context::wait()`, and then calls
// `gemm_bf16`. Do not interleave, slab or double-buffer them.
//
// One launch, grid `(N / 16, K / 64)`: one 16-lane subgroup owns one
// (N tile, K group), so each lane dequantises the 64 values of one column.
// `w.kind` must be `Int4`; a bf16 weight has nothing to expand and throws.
void dequant_to_bf16(Context& cx, KernelCache& kc, const loader::DeviceWeight& w,
                     uint16_t* scratch);

// What one `dequant_to_bf16` call appends to the L0 list, for Task 14's launch
// arithmetic. Asserted through `Context::launches()`, not restated in prose.
inline constexpr size_t kDequantLaunches = 1;

}  // namespace runtime::prefill
