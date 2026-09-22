#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace runtime::prefill {

// One int4 linear on the L0 backend (spec 2.1 §3.3), probe P-B's slab walk in production: for
// n0 = 0, 1024, ... < N, pf_dequant_slab writes columns [n0, n0 + 1024) into the slab buffer
// and pf_gemm multiplies x [M][K] by it into partials[:, n0 .. n0 + 1024), both on cx's
// in-order list with NO host wait -- between the two, between slabs, or before the consumer.
// M is padded to 256 by gemm_l0; rows [M, pad256(M)) of x are read and of partials written,
// and both buffers have kC = 2048 rows, so every padded row exists.
void linear_l0(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::DeviceWeight& w,
               const uint16_t* x, uint32_t M);

// Parity program S2(a): the SAME walk for gate||up, with `pf_silu_mul` fused
// into each slab GEMM's epilogue. `out` is the bf16 x it writes -- [pad256(M)]
// [N/2] at row pitch `ldx` -- and `partials` is not touched at all, which is the
// whole saving: the fp32 [M][34816] rectangle is neither written nor read back.
//
// **`out` must not alias `x`.** The unfused pair could overlap them because the
// GEMM had fully retired before `pf_silu_mul` ran; one kernel that reads row m
// of `x` at pitch K and writes row m of `out` at pitch N/2 cannot. `step.cc`
// gives the MLP's normed activations their own buffer for exactly this reason.
void linear_l0_silu(Context& cx, KernelCache& kc, PrefillScratch& s,
                    const loader::DeviceWeight& w, const uint16_t* x, uint32_t M, uint16_t* out,
                    uint32_t ldx);

// Launches one linear appends: a dequant and a GEMM per slab. The same for both
// forms above -- the fusion removes a CONSUMER launch, not a slab.
size_t linear_l0_launches(const model::GemvShape& sh);

}  // namespace runtime::prefill
