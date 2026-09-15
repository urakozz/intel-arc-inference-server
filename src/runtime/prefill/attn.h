#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

namespace runtime::prefill {

// The COMPOSED prefill attention of ruling A14, at L1: two inherited GEMMs and
// one bandwidth-bound softmax of ours, per kv group.
//
//   S = Q K^T   gemm_bf16_batched(L = 6, transB, strideB = 0)
//   P = softmax(S)                                pf_softmax_causal
//   O = P V     gemm_bf16_batched(L = 6, strideB = 0)
//   out = O . sigmoid(gate)                       pf_attn_gate
//
// **Label, everywhere this is timed: "L1 functional, untuned."** Plan 6d owns
// the tuned version and its 85-100 ms/chunk pre-registration. What this file
// does NOT do, deliberately: no head tiling beyond one kv group (`Lh` = 6,
// ruling A15's recorded design fact), no depth-adaptive `Lh`, no fusion of the
// softmax's three passes, no reuse of `S` between the two GEMMs.
//
// **GQA is a kv-head loop with the six sharing q-heads folded into the batch**
// (A15). One kv group per `gemm_bf16_batched` call, `strideB = 0` so the one
// kv-head's cache rows are read by all six without a copy, `transB` for QK^T so
// the `[pos][4][256]` cache IS the operand at `ldb = 1024`.
//
// **These functions SYNCHRONISE.** `attn_chunk` alternates SYCL GEMMs with an
// L0 softmax, and the B70 has one compute queue with two software queues and no
// device-side cross-runtime dependency (ruling A24, Probe B), so the boundary
// is `Context::wait()` -- the same boundary the dequant/GEMM pair uses. The
// count is 2 per kv group plus one at the end: 9 waits per FA layer.
namespace attn {
// One GQA group: the six q-heads that share kv-head j. Ruling A15's `Lh`.
inline constexpr uint32_t kGroup = PrefillScratch::kSHeads;   // 6
inline constexpr uint32_t kKvHeads = 4;
inline constexpr uint32_t kQHeads = 24;
inline constexpr uint32_t kHeadDim = 256;
inline constexpr uint32_t kQkvN = 14336;   // q||gate (12288) || k (1024) || v (1024)
inline constexpr uint32_t kOutN = 6144;    // 24 x 256, the o_proj input row
}  // namespace attn

// (1) q/k RMSNorm, partial RoPE, and the chunk's K/V written into the cache at
//     absolute positions [pos, pos + C). Reads `pos` and `n_active` out of the
//     shared `Control` block, which the caller has already set (and waited for).
//     Writes `s.pf_q` -- bf16 [C][24][256], ruling A9 -- and the two caches.
//     ONE launch (`pf_attn_prep_q16`, grid (28, C)).
void attn_prep_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                     const float* qkv_partials, const float* fa_small, const float* rope,
                     uint16_t* kv_k, uint16_t* kv_v);

// (2) Causal attention of the C queries over the cache rows [0, pos + C).
//     Writes `s.pf_o` -- fp32 [24][C][256], per-head stride `C * 256`, which is
//     what `attn_gate_chunk` and the tests read it at. Four L0 launches (one
//     softmax per kv group) and eight GEMMs; synchronises, see above.
void attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                PrefillBackend backend);

// (3) The output gate (ruling A16): `out[m][h*256+d] =
//     rne_bf16(f32(rne_bf16(o)) . sigmoid(f32(rne_bf16(gate))))`, with the gate
//     read straight out of `qkv_partials`. ONE launch, grid (24, C).
void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C,
                     const float* qkv_partials, uint16_t* out);

// The L0 launches each appends -- the SYCL GEMMs are not on the L0 list and do
// not move `Context::launches()`. Task 14's launch arithmetic reads these.
inline constexpr size_t kAttnPrepLaunches = 1;
inline constexpr size_t kAttnChunkLaunches = attn::kKvHeads;   // one softmax per kv group
inline constexpr size_t kAttnGateLaunches = 1;

}  // namespace runtime::prefill
