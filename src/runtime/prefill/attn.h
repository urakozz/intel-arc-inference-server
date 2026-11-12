#pragma once
#include <cstddef>
#include <cstdint>

#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

namespace runtime::prefill {

// The COMPOSED prefill attention of ruling A14, at L1: two GEMMs and one
// bandwidth-bound softmax, per kv group.
//
//   S = Q K^T   gemm_l0(L = 6, transB, strideB = 0)
//   P = softmax(S)                                pf_softmax_causal
//   O = P V     gemm_l0(L = 6, strideB = 0)
//   out = O . sigmoid(gate)                       pf_attn_gate
//
// **Label, everywhere this is timed: "L1 functional, untuned."** Plan 6d owns
// the tuned version and its 85-100 ms/chunk pre-registration. What this file
// does NOT do, deliberately: no head tiling beyond one kv group (`Lh` = 6,
// ruling A15's recorded design fact), no depth-adaptive `Lh`, no fusion of the
// softmax's three passes, no reuse of `S` between the two GEMMs.
//
// **GQA is a kv-head loop with the six sharing q-heads folded into the batch**
// (A15). One kv group per batched GEMM, `strideB = 0` so the one
// kv-head's cache rows are read by all six without a copy, `transB` for QK^T so
// the `[pos][4][256]` cache IS the operand at `ldb = 1024`.
//
// **They do not synchronise** (spec 2.1 S3): both GEMMs are `pf_gemm` on
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

// Spec 2.1 §3.4: every attention buffer's per-head slot is strided by the 256-padded row
// count, never by C -- a batched pf_gemm launch computes every head's padded rows at once,
// and a C-row stride would let head l's padding overwrite head l+1's rows. The same on both
// backends (the argument is kept for the call sites that name it).
inline uint32_t attn_rows(uint32_t C, PrefillBackend) { return pad256(C); }

// Spec S3 (`docs/superpowers/specs/2026-09-22-prefill-parity-program-design.md` §3): each kv
// group's QK^T is issued in row blocks of `kPfGemmTile`, so block b computes only the columns
// causality can reach (`pad256(pos + r1)`) instead of the whole padded depth. P·V is NOT
// blocked -- attn.cc records the occupancy measurement that rejected it.
inline uint32_t attn_row_blocks(uint32_t C, PrefillBackend b) { return attn_rows(C, b) / kPfGemmTile; }

// Spec 6 (plan 6b): which prefill attention `attn_chunk` runs.
//   Flash    - `pf_flash_attn` (pfa_KT64_R8_H6 + exp2, spec 6c), ONE launch per FA layer for
//              all four kv groups, no score scratch. The default.
//   Composed - the QK^T / softmax / PV path below (ruling A14), kept as the correctness
//              reference the tests grade flash against (attn_chunk_test, flash_long_test,
//              prefill_smoke_test, probe_flash_attn). `pf_s` / `pf_p` exist only once it has
//              run (PrefillScratch::pf_s_buffer()). Since the zero-config cleanup (2026-11-06)
//              nothing but set_attn_mode_for_test reaches it (B70_PREFILL_ATTN is gone).
enum class AttnMode { Flash, Composed };
// The process-wide mode: Flash, unless a test has set it.
AttnMode attn_mode();
// Test-only override (Review Focus: one engine prefills in both modes). A replay recording
// is keyed by the mode it was captured in (engine_prefill.cc), so switching is safe.
void set_attn_mode_for_test(AttnMode m);
const char* attn_mode_name(AttnMode m);

// 3l (2): the split-K flash for a short chunk deep in the cache. `flash_split_span` is the rule:
// 0 runs today's one-pass launch (bitwise as before); otherwise the key range [0, pos) is cut
// into flash_split_count(pos, span) spans of `span` keys (a multiple of the 64-key tile), one
// work-group each per 8 rows and kv head, folded by pf_flash_merge (attn.cc has the measurement
// behind the rule). Built at Qwen3.8's heads (24 / 4) and a tensor-parallel card's (12 / 2); any
// other model keeps today's launch. Auto is the engine's; Off / On are a test seam (the timing
// pairs and the f32 judge's two runs): On splits whenever a split is possible (pos >= 64).
enum class FlashSplit { Auto, Off, On };
FlashSplit flash_split_mode();
void set_flash_split_for_test(FlashSplit m);
uint32_t flash_split_span(const model::ModelDesc& d, uint32_t pos, uint32_t C);
inline uint32_t flash_split_count(uint32_t pos, uint32_t span) { return span ? (pos + span - 1) / span : 1; }
// int8 KV: a split chunk of few rows reads the int8 rows in place (no kv8_widen launch); a larger one
// widens and runs the raw split.
bool kv8_split_in_place(const model::ModelDesc& d, uint32_t pos, uint32_t C);

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
//     the tests read it at. In Flash mode (the default, spec 6): ONE
//     `pf_flash_attn` launch, grid (ceil(C / 8), 4, 1), writing the same pf_o
//     layout, so `attn_gate_chunk` is unchanged. Otherwise the composed path, per kv group: one
//     pf_gemm per QK^T ROW BLOCK (spec S3), one softmax, one P·V, and no wait --
//     `attn_chunk_launches(C, backend)` below, the one term of the prefill
//     launch arithmetic that depends on C.
//     MoE prefill phase 3: with `gate_src` / `gate_out` given and attn_gate_fused(), the flash launch also does
//     (3) below in its epilogue and writes `gate_out` instead of pf_o; it returns whether it did.
bool attn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                PrefillBackend backend, const float* gate_src = nullptr, uint16_t* gate_out = nullptr);

// (3) The output gate (ruling A16): `out[m][h*256+d] =
//     rne_bf16(f32(rne_bf16(o)) . sigmoid(f32(rne_bf16(gate))))`, with the gate
//     read straight out of `qkv_partials`. ONE launch, grid (24, C). `rows` is
//     `pf_o`'s per-head slot stride in rows -- `attn_rows(C, backend)`, the same
//     value `attn_chunk` laid the buffer out with.
void attn_gate_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, uint32_t rows,
                     const float* qkv_partials, uint16_t* out);

// Spec 12b (`--kv-cache int8`): the same three steps over the int8 KV cache, one launch
// each, from src/kernels/kv8.cl. `kv` is one FA layer's int8 rows and fp16 scales
// (KvLayout::layer). The writer is pf_attn_prep_q16's chain then rotkv's rotation and
// quantiser (pf_q holds the ROTATED q); the attention is pf_flash_attn's over int8 + scales
// (its output still rotated, in pf_o); the gate un-rotates each head first. There is no
// int8 form of the composed path: `require_kv8_path` refuses it, so an int8 cache is never
// read as bf16.
void require_kv8_path(PrefillBackend backend);
void attn_prep_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C, void* ctrl,
                         const float* qkv_partials, const float* fa_small, const float* rope,
                         const KvLayer& kv);
void attn_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
                    const uint16_t* q, const KvLayer& kv, PrefillBackend backend);
void attn_gate_chunk_kv8(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t C,
                         uint32_t rows, const float* qkv_partials, uint16_t* out);

// The L0 launches each appends. Task 14's launch arithmetic reads these.
inline constexpr size_t kAttnPrepLaunches = 1;
inline constexpr size_t kAttnGateLaunches = 1;

// What `attn_chunk` appends to the L0 list (spec S3): per kv group one pf_gemm per
// QK^T row block, one softmax and one P·V -- so this, and only this, makes the chunk's
// launch count a function of C. At C <= 256 there is one block and the count is what it was
// before S3.
//
// Spec 6: in Flash mode the whole thing is ONE `pf_flash_attn` launch and
// the count stops depending on C. `attn_chunk_launches_composed` is the composed count
// whatever the mode, for the arithmetic that states the difference.
inline size_t attn_chunk_launches_composed(const model::ModelDesc& d, uint32_t C,
                                           PrefillBackend b) {
  return size_t(d.fa_kv_heads) * (2 + size_t(attn_row_blocks(C, b)));
}
// 3l (2): a split chunk (flash_split_span at `pos`) adds the merge.
inline size_t attn_chunk_launches(const model::ModelDesc& d, uint32_t C, PrefillBackend b,
                                  AttnMode mode, uint32_t pos = 0) {
  return mode == AttnMode::Flash ? 1 + (flash_split_span(d, pos, C) ? 1 : 0) : attn_chunk_launches_composed(d, C, b);
}
inline size_t attn_chunk_launches(const model::ModelDesc& d, uint32_t C, PrefillBackend b, uint32_t pos = 0) {
  return attn_chunk_launches(d, C, b, attn_mode(), pos);
}
// MoE prefill phase 3: Ornith's 16 / 2 heads gate inside the 32-row flash launch on a bf16 cache (the prefill walk), so
// the walk's gate launch is 0 there; the int8 cache keeps pf_attn_gate_kv8.
inline bool attn_gate_fused(const model::ModelDesc& d, uint32_t C, uint32_t pos, KvCache kv) {
  return kv == KvCache::Bf16 && attn_mode() == AttnMode::Flash && d.fa_q_heads == 16 && d.fa_kv_heads == 2 &&
         flash_split_span(d, pos, C) == 0;
}
inline size_t attn_gate_launches(const model::ModelDesc& d, uint32_t C, uint32_t pos, KvCache kv) {
  return attn_gate_fused(d, C, pos, kv) ? 0 : kAttnGateLaunches;
}

}  // namespace runtime::prefill
