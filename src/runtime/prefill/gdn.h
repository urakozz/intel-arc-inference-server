#pragma once
#include <cstdint>

#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace runtime::prefill {

// The WY-representation chunked gated delta rule for ONE GDN layer over `C` new
// positions, intra-chunk 64. Reads and writes the SAME `gdn_state` and
// `conv_ring` buffers `gdn_step` uses, in place and in decode's layouts, so
// decode continues from position `pos + C` with no translation.
//
//   qkvz_partials  fp32 [C][16384]   the layer's qkv‖z GEMV output at **S = 1**.
//                                    Rounded to bf16 INSIDE, at the single point
//                                    gdn_step.cl:253 rounds (ruling A6/R4) -
//                                    which is what spec §6.3's self-consistency
//                                    bar rests on.
//   ab_out         fp32 [C][128]     decode's ab_out layout: a at [0,48),
//                                    b at [48,96), zero-padded to 128. The bf16
//                                    round happens inside, at gdn_step.cl:289-290.
//   gdn_state      fp32 [48][128][128]  THIS layer's slice, k-major
//   conv_ring      bf16 [16][10240]     THIS layer's slice, slot-major
//   small          the layer's GDN block base: conv weights at 0, negA at
//                  NEGA_OFF, dt_bias at DTBIAS_OFF, the gated norm at
//                  loader::kGdnOffGatedNorm
//   y              bf16 [C][6144]    the out_proj input. Ruling R3:
//                                    `gdn_chunk` owns the WHOLE GDN mixer, so
//                                    its tenth and last launch is
//                                    `pf_gated_head` and `y` is post-gate.
//
// `layer` is not a parameter: the caller passes that layer's pointers, which is
// how `capture.cc:492-493` binds them (ruling A6).
//
// **Two parameters interfaces.md's declaration does not have, both mechanical
// and both recorded rather than silent:**
//   * `PrefillScratch& s` - the nine kernels' scratch. Ruling A18 names this
//     deviation explicitly ("the same deviation 6b took for `gdn_chunk`") when
//     granting it to `attn_chunk`. The tensor contract above is unchanged.
//   * `KernelCache& kc` - plan 6b Task 2 put binary loading on `Context` as
//     `Context::kernel(variant, entry, wg)`. It cannot live there: `Context`'s
//     methods are compiled by icpx into libb70_prefill.so, which deliberately
//     links no project archive, and `l0::Module`/`l0::Kernel` are `b70_l0`
//     symbols. The cache is the same object, relocated to the g++ side; see
//     `kernels.h`.
//
// Ten launches per call, in the order `gdn.cc` lists them. `C` must be in
// (0, PrefillScratch::kC]; anything else throws.
void gdn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
               const float* qkvz_partials, const float* ab_out, float* gdn_state,
               uint16_t* conv_ring, const void* small, uint16_t* y);

// What one `gdn_chunk` call appends to the L0 list. Asserted through
// `Context::launches()` rather than restated in prose; Task 14's launch
// arithmetic reads it.
inline constexpr size_t kGdnChunkLaunches = 10;

}  // namespace runtime::prefill
