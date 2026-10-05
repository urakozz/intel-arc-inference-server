#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
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
// **On the sycl-tla backend these functions SYNCHRONISE.** `attn_chunk`
// alternates SYCL GEMMs with an L0 softmax, and the B70 has one compute queue
// with two software queues and no device-side cross-runtime dependency (ruling
// A24, Probe B), so the boundary is `Context::wait()` -- the same boundary the
// dequant/GEMM pair uses. The count is 2 per kv group plus one at the end: 9
// waits per FA layer.
//
// **On the L0 backend they do not** (spec 2.1 S3): both GEMMs are `pf_gemm` on
// the same in-order immediate list as the softmax, so the whole group is
// `blocks + 2` launches (parity-program S3 blocks QK^T by rows) with no host
// wait between them and none after the loop. The timed
// path pays zero waits per FA layer; `profile_wait` still inserts the
// diagnostic ones when B70_PREFILL_PROFILE=1.
// Spec 15b: the head counts are the model descriptor's (PrefillScratch::desc()):
// q-heads (24 on Qwen3.8), kv-heads (4), and one GQA group - the q-heads that
// share kv-head j, ruling A15's `Lh` - is PrefillScratch::s_heads() (6). The
// head dim is shared.
namespace attn {
inline constexpr uint32_t kHeadDim = model::Qwen35::kFaHeadDim;   // 256
}  // namespace attn

// Spec 2.1 §3.4: on the L0 backend every attention buffer's per-head slot is strided by the
// 256-padded row count, never by C -- a batched pf_gemm launch computes every head's padded
// rows at once, and a C-row stride would let head l's padding overwrite head l+1's rows.
inline uint32_t attn_rows(uint32_t C, PrefillBackend b) {
  return is_l0(b) ? pad256(C) : C;
}

// Spec S3 (`docs/superpowers/specs/2026-09-22-prefill-parity-program-design.md` §3): on the
// L0 backend each kv group's QK^T is issued in row blocks of `kPfGemmTile`, so block b
// computes only the columns causality can reach (`pad256(pos + r1)`) instead of the whole
// padded depth. P·V is NOT blocked -- attn.cc records the occupancy measurement that
// rejected it. sycl-tla is untouched and keeps its one GEMM per group, hence 1 here.
inline uint32_t attn_row_blocks(uint32_t C, PrefillBackend b) {
  return is_l0(b) ? attn_rows(C, b) / kPfGemmTile : 1;
}

// Spec 6 (plan 6b): which prefill attention `attn_chunk` runs on the L0 backends.
//   Flash    - `pf_flash_attn` (pfa_KT64_R8_H6 + exp2, spec 6c), ONE launch per FA layer for
//              all four kv groups, no score scratch. The default.
//   Composed - the QK^T / softmax / PV path below (ruling A14), kept as the correctness
//              reference, the role sycl-tla plays for the GEMM. `pf_s` / `pf_p` exist only
//              once it has run (PrefillScratch::pf_s_buffer()).
// `B70_PREFILL_ATTN=composed` selects Composed; unset, `flash` or anything else is Flash.
// sycl-tla always runs Composed whatever this says.
enum class AttnMode { Flash, Composed };
// The process-wide mode: B70_PREFILL_ATTN, read once at first use, unless a test has set it.
AttnMode attn_mode();
// Test-only override (Review Focus: one engine prefills in both modes). A replay recording
// is keyed by the mode it was captured in (engine_prefill.cc), so switching is safe.
void set_attn_mode_for_test(AttnMode m);
const char* attn_mode_name(AttnMode m);

// (1) q/k RMSNorm, partial RoPE, and the chunk's K/V written into the cache at
//     absolute positions [pos, pos + C). Reads `pos` and `n_active` out of the
//     shared `Control` block, which the caller has already set (and waited for).
//     Writes `s.pf_q` -- bf16 [C][24][256], ruling A9 -- and the two caches.
//     ONE launch (`pf_attn_prep_q16`, grid (28, C)).
void attn_prep_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                     const float* qkv_partials, const float* fa_small, const float* rope,
                     uint16_t* kv_k, uint16_t* kv_v);

// (2) Causal attention of the C queries over the cache rows [0, pos + C).
//     Writes `s.pf_o` -- fp32 [24][rows][256], per-head stride `rows * 256`
//     with `rows = attn_rows(C, backend)`, which is what `attn_gate_chunk` and
//     the tests read it at. On L0 in Flash mode (the default, spec 6): ONE
//     `pf_flash_attn` launch, grid (ceil(C / 8), 4, 1), writing the same pf_o
//     layout, so `attn_gate_chunk` is unchanged. Otherwise the composed path. On sycl-tla: four L0 launches (one softmax per kv
//     group) and eight GEMMs; synchronises, see above. On L0, per kv group: one
//     pf_gemm per QK^T ROW BLOCK (spec S3), one softmax, one P·V, and no wait --
//     `attn_chunk_launches(C, backend)` below, the one term of the prefill
//     launch arithmetic that depends on C.
void attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                PrefillBackend backend);

// (3) The output gate (ruling A16): `out[m][h*256+d] =
//     rne_bf16(f32(rne_bf16(o)) . sigmoid(f32(rne_bf16(gate))))`, with the gate
//     read straight out of `qkv_partials`. ONE launch, grid (24, C). `rows` is
//     `pf_o`'s per-head slot stride in rows -- `attn_rows(C, backend)`, the same
//     value `attn_chunk` laid the buffer out with.
void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, uint32_t rows,
                     const float* qkv_partials, uint16_t* out);

// The L0 launches each appends -- the SYCL GEMMs are not on the L0 list and do
// not move `Context::launches()`. Task 14's launch arithmetic reads these.
inline constexpr size_t kAttnPrepLaunches = 1;
// sycl-tla: one softmax per kv group (4 on Qwen3.8).
inline size_t attn_chunk_launches_sycl(const model::ModelDesc& d) { return d.fa_kv_heads; }
inline constexpr size_t kAttnGateLaunches = 1;

// What `attn_chunk` appends to the L0 list. sycl-tla: `attn_chunk_launches_sycl`, its two GEMMs
// per group being SYCL calls off this counter. L0 (spec S3): per kv group one pf_gemm per
// QK^T row block, one softmax and one P·V -- so this, and only this, makes the chunk's
// launch count a function of C. At C <= 256 there is one block and the count is what it was
// before S3.
//
// Spec 6: in Flash mode (L0 backends only) the whole thing is ONE `pf_flash_attn` launch and
// the count stops depending on C. `attn_chunk_launches_composed` is the composed count
// whatever the mode, for the arithmetic that states the difference.
inline size_t attn_chunk_launches_composed(const model::ModelDesc& d, uint32_t C,
                                           PrefillBackend b) {
  return is_l0(b) ? size_t(d.fa_kv_heads) * (2 + size_t(attn_row_blocks(C, b)))
                  : attn_chunk_launches_sycl(d);
}
inline size_t attn_chunk_launches(const model::ModelDesc& d, uint32_t C, PrefillBackend b,
                                  AttnMode mode) {
  return is_l0(b) && mode == AttnMode::Flash ? 1 : attn_chunk_launches_composed(d, C, b);
}
inline size_t attn_chunk_launches(const model::ModelDesc& d, uint32_t C, PrefillBackend b) {
  return attn_chunk_launches(d, C, b, attn_mode());
}

}  // namespace runtime::prefill
