#pragma once
// CPU references for the prefill path's runtime-`M`, S = 1 small kernels
// (src/kernels/prefill/pf_*.cl).
//
// **Most of this file is a redirection, not a reimplementation.** The decode
// references in `tests/kernels/prep_ref.h` and `tests/kernels/attn_ref.h` are
// already parametric in `M` -- `prep_ref::res_fold`, `::norm_finish` and
// `::gated_head` take it as an argument, and so do `attn_ref::prep` and its
// two neighbours -- and the prefill kernels are the decode kernels with `M`
// moved from a `-D` to an argument. Writing a second copy of those chains here
// would create exactly the failure mode the references exist to prevent: two
// texts that must be edited together and are not. So the only thing this header
// adds is what genuinely differs, which is the split-K width:
//
//   * `silu_mul_s1` -- `prep_ref::silu_mul` folds `kSiluS` = 8 slices, because
//     that is decode's `gate||up`. Ruling R1 makes the prefill path fold ONE,
//     and the fold is inside the chain (before the linear's bf16 rounding), so
//     it cannot be expressed by passing a different `M`.
//   * `embed_gather` -- decode's has no CPU reference at all (its test drives
//     the device twice with two control-block ids); a gather does not need
//     one, but the runtime-`M` version wants a row-independence bar.
//
// `prep_ref::gated_head` already folds `kGatedS` = 1 -- qkv||z runs unsplit in
// decode too -- so it is the prefill reference unchanged, and `attn_ref::prep`
// is driven with a two-slice partials buffer whose second slice is `+0.0f`
// (see pf_attn_test.cc, which asserts no input is `-0.0f` first: `+0.0f +
// -0.0f == +0.0f` would flip a sign of zero and therefore a bf16 word).
#include <cstddef>
#include <cstdint>

#include "common/bf16.h"
#include "prep_ref.h"

namespace pf_ref {

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }

// pf_silu_mul: gate||up (interleaved in 16-column blocks) -> silu(gate)*up,
// folding ONE fp32 slice. prep_ref::silu_mul's chain with `kSiluS` = 1:
//   gflat = (k/16)*32 + k%16 ;  uflat = gflat + 16
//   g_b = rne(partials[m][gflat]) ;  u_b likewise
//   s_b = rne(silu_f32(f32(g_b)))
//   x_out[m][k] = rne(f32(s_b) * f32(u_b))
inline void silu_mul_s1(const float* partials, uint16_t* x_out, uint32_t M) {
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < prep_ref::kSiluN; ++k) {
      const size_t gflat = size_t(k / 16) * 32 + (k % 16), uflat = gflat + 16;
      const size_t base = size_t(m) * prep_ref::kSiluFusedN;
      const uint16_t g_b = rne(partials[base + gflat]), u_b = rne(partials[base + uflat]);
      const uint16_t s_b = rne(prep_ref::silu_f32(f32(g_b)));
      x_out[size_t(m) * prep_ref::kSiluN + k] = rne(f32(s_b) * f32(u_b));
    }
}

// pf_embed_gather: resid[m][*] = embed[ids[m]][*], copied verbatim as bf16.
// An id >= vocab writes nothing (the kernel returns; the host validates ids).
inline void embed_gather(const uint32_t* ids, const uint16_t* embed, uint16_t* resid, uint32_t M,
                         uint32_t hidden, uint32_t vocab) {
  for (uint32_t m = 0; m < M; ++m) {
    const uint32_t row = ids[m];
    if (row >= vocab) continue;
    for (uint32_t k = 0; k < hidden; ++k)
      resid[size_t(m) * hidden + k] = embed[size_t(row) * hidden + k];
  }
}

}  // namespace pf_ref
