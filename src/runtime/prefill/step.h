#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

namespace runtime::prefill {

class Int8State;   // runtime/prefill/int8.h, spec 5's h8 path

// ONE chunk of at most `PrefillScratch::kC` positions, at absolute position
// `pos`. The caller has already uploaded the chunk's ids into `s.ids` and set
// `Control::{pos, n_active}` (after a `wait()`, because the L0 list is
// asynchronous). This function walks all 64 layers, appending to the L0 list
// (pure appending: every op is an L0 launch, spec 2.1).
//
// The four persistent `l0::Mem&` are passed rather than reached through an
// `Engine`, so this function holds no engine reference and is unit-testable.
//
// `q` (spec 5) is the int8 path's state and must be non-null iff `backend` is
// L0Int8. Every int4 linear's column scales must already be in it (`Engine::prefill`
// builds them before the first chunk): `Int8State::scales` on a new weight waits on
// the host, which a recorded chunk must never contain.
//
// `kv` (spec 12b) is kv_k_mem / kv_v_mem's layout (PersistentBuffers::kv_lay): at int8 the
// FA layers write and read the int8 rows and scales through attn_*_kv8 (one launch each,
// as the bf16 flash path), on the flash attention only.
//
// `taps` (spec 19c, a drafter - runtime/tap_drafter.h): after the input fold of layer s (`resid` = layer s - 1's
// output) the chunk's rows r0 .. C - 1 of every tap whose site is s go to `dst` laid out [row - r0][tap][hidden]
// bf16 (one 2D copy per tap: Context::copy_2d), i.e. the prefill's taps of the positions the drafter's ring will
// read. Null (the default) is the walk above, unchanged. A tap of the LAST layer's output is not built here (its
// fold is step_head's, last row only): Engine::prefill refuses such a drafter.
struct PrefillTaps {
  void* dst = nullptr;
  const std::vector<std::vector<uint32_t>>* sites = nullptr;   // runtime::tap_sites(desc, layers)
  uint32_t n_taps = 0;
  uint32_t r0 = 0;
};
void step_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C,
                l0::Mem& gdn_state_mem, l0::Mem& conv_ring_mem, l0::Mem& kv_k_mem,
                l0::Mem& kv_v_mem, const KvLayout& kv, PrefillBackend backend, Int8State* q,
                const PrefillTaps* taps = nullptr);

// Spec 16c (pipeline parallel prefill across two B70s): step_chunk restricted to one pipeline
// stage's layers [st.first, st.last), on the device `cx` / `s` / `m` belong to - `m` a stage
// model (runtime/pipeline_place.h: its layer_small / moe hold the stage's layers, indexed by
// layer - first) and the four state buffers the stage's own (PersistentBuffers' stage
// constructor; `kv` laid out for st.fa layers). For every layer the same launches, arguments
// and order as step_chunk, so the two stages compute exactly what one card does (P1):
//
//   stage 0  pf_embed_gather of `ids` (C ids; the caller's buffer, so two chunks' ids can be
//            in flight), layers [0, s), then layer s's pf_res_fold - the cut. Device 0 hands
//            off `s.resid` (C rows) and `s.norm_sumsq`.
//   stage 1  `ids` null: layer s from its pf_norm_finish over the resid and norm sums the
//            caller copied into `s` from the hand-off, then layers (s, L). step_head after
//            the last chunk, as on one card.
//
// L0 backends and the flash attention only; `ctrl` is any Control block holding the chunk's
// pos / n_active (the pipeline keeps one per chunk in flight). Throws, by name, for anything
// else.
//
// `taps` (spec 19d, a drafter across the split): step_chunk's, for the sites this stage walks - each device writes
// its own taps' columns of the one staging ([row - r0][tap][hidden], on the drafter's device: device 0's copies go
// into the other card's memory). Null is the walk above, unchanged.
void step_stage(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C, const void* ids,
                l0::Mem& gdn_state_mem, l0::Mem& conv_ring_mem, l0::Mem& kv_k_mem,
                l0::Mem& kv_v_mem, const KvLayout& kv, PrefillBackend backend, Int8State* q,
                const PpStage& st, const PrefillTaps* taps = nullptr);

// The tail only the LAST chunk runs: the final norm on the last row, `lm_head`
// at M = 1 through the EXISTING decode binary, and the two argmax stages --
// the `Control` handoff. `last_row` is the row of the last chunk that holds the
// final position.
//
// `normed_row` (spec 8): non-null when the final norm of the last row has already been
// written there (step_mtp_kv normalises every row); the two norm launches are skipped
// and lm_head reads it. Null is the walk above, unchanged.
void step_head(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
               void* ctrl, uint32_t last_row, const void* normed_row = nullptr);

// Spec 8 §3.2 (plan 8b Task 3): after `step_chunk`, the MTP head's K/V for the chunk.
// First the main model's final norm on EVERY row r into `hid` row 1 + r (bf16
// [kC + 1][5120]; row 0 is h_{pos-1}, the caller's copy of MtpBuffers::hh row 0).
// Then the head over the pairs (hid[r], ids[r]) = (h_{pos-1+r}, x[pos+r]) at positions
// pos - 1 + r, r < C - the previous chunk's last position is filled here, the chunk's
// own last position by the next chunk or the first draft/verify. At pos = 0 there is
// no h_{-1}: rows 1..C-1 only, at positions 0..C-2. Only the K/V are computed (fc,
// input_layernorm, the k||v slabs of q||k||v, attn_prep), which is all a later draft
// reads. `hctl` must hold pos = max(pos, 1) - 1 and n_active = the row count (the
// caller sets both, like Control before step_chunk). L0 backends only (pf_gemm).
// `kv` is the head's one layer (MtpBuffers::kv_lay.layer(...)): bf16, or (spec 12b) the
// int8 rows and scales, written by attn_prep_chunk_kv8.
void step_mtp_kv(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m,
                 void* hctl, uint32_t pos, uint32_t C, uint16_t* hid, const KvLayer& kv);
// Rows step_mtp_kv runs the head over, and its L0 launches (for the arithmetic).
inline uint32_t mtp_kv_rows(uint32_t pos, uint32_t C) { return pos == 0 ? C - 1 : C; }
size_t step_mtp_kv_launches(uint32_t pos, uint32_t C);

// The launch arithmetic, derived from the walk itself rather than restated:
// what `Context::launches()` advances by. `C` is the chunk width: since the parity
// program's S3 the composed attention issues its QK^T GEMM per 256-row block, so a chunk's
// launch count depends on it there (8625 at C <= 256, 9073 at C = 2048).
// Spec 6: the count follows `attn_mode()`; in Flash mode (the default) attention is one
// launch per FA layer and the count no longer depends on C (8449; step.cc).
// Spec 14: per model - the layer counts and the gate||up slab count are the descriptor's.
size_t step_chunk_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C);
inline constexpr size_t kStepHeadLaunches = 5;   // 2 norm + lm_head + 2 argmax
// Spec 16c: one layer's launches, and a stage's: its
// layers, the embed on stage 0 and the cut's fold, less the fold stage 1 resumes after - so
// the two stages add up to step_chunk_launches (the pipeline asserts it).
size_t step_layer_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C,
                           model::LayerKind kind);
size_t step_stage_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C,
                           const PpStage& st);

}  // namespace runtime::prefill
