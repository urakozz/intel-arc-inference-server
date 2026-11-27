// tap_mean: a DFlash 2 drafter's tap row (plan 21-QFN-v3e Task 1) - after the combine that materialises layer l's
// output H, the mean of H's HC hyper-connection streams goes to tap row l: an fp32 sum, then one rounding to bf16
// (the drafter's authors' capture: hidden_states.unflatten(-1, (HC, H)).mean(-2) on bf16, one rounding).
//
//   in   bf16 [rows][HC * H]   the materialised H, stream k at the row's offset k * H
//   out  bf16 [rows][H]        the tap rows (the caller's buffer, tap row l at its offset)
//
// One work-item a hidden column of one row, grid (H, rows) - the row count is a launch argument, so one binary
// serves the decode, verify and prefill lists. The mean's division by HC is exact in fp32 (HC a power of two), so
// the fp32 sum then one bf16 rounding is the kernel's arithmetic. Not wired into any list yet (Task 2).
#include "bf16.h"

#ifndef HC
#error "tap_mean: HC (the stream count) is the variant's"
#endif
#ifndef H
#error "tap_mean: H (the hidden width) is the variant's"
#endif

__kernel void tap_mean(__global const ushort* restrict in, __global ushort* restrict out, uint rows) {
  const uint c = get_global_id(0);
  const uint r = get_global_id(1);
  if (c >= H || r >= rows) return;
  const __global const ushort* row = in + (size_t)r * HC * H + c;
  float sum = 0.0f;
  for (uint k = 0; k < HC; ++k) sum += bf16f(row[(size_t)k * H]);
  out[(size_t)r * H + c] = rne_bf16(sum / HC);
}
