#pragma once
#include <cstddef>
#include <cstdint>
#include "model/model_desc.h"

// The byte size of every runtime allocation that a model and a max_len decide, as
// device-free arithmetic (the max_len auto amendment of spec 6, §10). There is ONE
// source for each size: the buffer constructors in buffers.cc and int8.cc allocate
// exactly what these functions return, and runtime/memory_plan.h adds them up before
// anything is allocated. A planner that kept its own copy of the formulas would agree
// with the allocations until the first buffer change, and then plan a context that
// does not fit - so it does not keep one.
//
// The constants the sizes are built from live here for the same reason: the buffer
// classes in buffers.h inherit them from the `*Dims` bases below, so
// `DecodeScratch::kAttnBlock` and `DecodeScratch::sizes()` are still spelled the way
// every call site has always spelled them, and this header needs no Level Zero.
namespace runtime {

// Spec 10 (plan 10b): which decode-attention pair every capture binds - attn.cl's
// attn_decode + attn_reduce (v1) or attn_v2.cl's (v2). `B70_DECODE_ATTN=v1|v2`, read
// at each call (so a test can set it between engines); unset or empty means the
// default; any other value throws. Both pairs share attn_prep, the KV layout, attn_q /
// attn_gate / attn_part / attn_out - but not attn_part's SIZE (DecodeScratchDims), which
// is why the choice lives here, beside the sizes, rather than in capture.h.
enum class DecodeAttn { V1, V2 };
inline constexpr DecodeAttn kDefaultDecodeAttn = DecodeAttn::V2;   // spec 10 gates, 2026-09-28
DecodeAttn decode_attn();
const char* decode_attn_name(DecodeAttn a);

// --- PersistentBuffers ---------------------------------------------------------------

struct PersistentSizes {
  size_t control, gdn_state, conv_ring, kv_k, kv_v;
  size_t total() const { return control + gdn_state + conv_ring + kv_k + kv_v; }
};

struct PersistentDims {
  static constexpr uint32_t kConvRing = 16;   // ring depth >= M + 3 (spec §9.4)
  static PersistentSizes sizes(uint32_t max_len, const model::ModelDesc& desc);
};

// --- DecodeScratch -------------------------------------------------------------------

struct DecodeScratchSizes {
  size_t resid, x, partials, ab_out, norm_sumsq, gdn_o, attn_q, attn_gate, attn_part,
      attn_out, logits, argmax_part;
  size_t total() const {
    return resid + x + partials + ab_out + norm_sumsq + gdn_o + attn_q + attn_gate + attn_part +
           attn_out + logits + argmax_part;
  }
};

struct DecodeScratchDims {
  static constexpr uint32_t kM = 8;
  // KV positions per `attn_decode` work-group - spec 1.5's lever L5 took it
  // from 256 to 64. It is the ONE home for the number on the host: `attn_part`
  // below is strided by `max_len / kAttnBlock` blocks, capture.cc launches
  // `attn_decode` with exactly that many, and it is the `_B64` half of both
  // compiled variant names (`kernels::attn_decode_variant`), so a value here
  // that no compiled binary matches throws at capture instead of striding
  // `attn_part` at one size while the kernel writes it at another.
  //
  // **One home on the host, three in the tree - and the only thing binding them
  // needs a GPU.** The same 64 also lives in `src/kernels/CMakeLists.txt`
  // (`ATTN_BLOCK`, which compiles it into the kernel and into the binary's
  // name) and in `tests/kernels/attn_ref.h` (`attn_ref::kBlock`, which the
  // reference walks blocks with). Nothing checks the three against each other
  // at compile time: the guard is the missing-binary throw described above, and
  // that throw fires at CAPTURE - it needs a device, a loaded model and a
  // built kernel set, so a CPU-only build or a host-side review sees a
  // disagreement not at all. `kNormGroups` below has exactly the same shape
  // (here, CMake's `FOLD_G`/`NORM_G`, and `prep.cl`'s own `#define FOLD_G 20`
  // fallback). A device-free cross-check - a static_assert or a generated
  // header - is recorded as spec-1.6 work and is deliberately NOT added here.
  //
  // **64 is measured, not derived - but it is not a knee either.** docs/15 §2's
  // `F + fill·P` fit predicted 214.5 µs/launch at a 128-position block; the
  // retile measured 296.684. So the block size was swept in situ at depth 4096
  // instead - `attn_decode` µs/launch, then the attn family's ms/token:
  //
  //     B256 369.988 / 6.058   B128 296.684 / 4.931
  //     B64  224.046 / 3.839   B32  203.837 / 3.769
  //
  // B32's family total is **0.070 ms/token better than B64**, so there is no
  // crossover to point at: the step-Σ difference that once looked like one
  // (+19.5 µs) is inside the +0.283% drift the untouched launches showed
  // between those two runs. 64 is chosen because the marginal gain has
  // collapsed (−1.127, −1.092, −0.070 ms/token for the three halvings, the last
  // about a third of one run's drift), because `attn_reduce` is on a steep ramp
  // (80 → 127 → 196 → 450 µs/step), and because B32 would double `attn_part`
  // again - 50.7 → 101.4 MB, per-step scratch 77.6 → 128.3 MB - to buy that
  // 0.070 ms. docs/12 `attn` → Measured carries the arithmetic.
  static constexpr uint32_t kAttnBlock = 64;
  // Spec 10 (plan 10b): decode attention v2's work-groups per kv head, TGT in
  // src/kernels/attn_v2.cl and `_T32` in its variant name. v2 walks a row of L keys in
  // blocks of max(64, roundup64(ceil(L / 32))) positions, so no row has more than 32
  // partials: v2 needs [24][32][M][258] floats whatever max_len is, so `attn_part` is
  // sized by the pair the engine binds (`sizes(..., attn)`): v1's
  // [24][max_len / kAttnBlock][M][258] only under B70_DECODE_ATTN=v1 (buffers_test pins
  // both).
  static constexpr uint32_t kAttnV2Blocks = 32;
  // **prep_res_norm's two-stage grid** (spec 1.5 lever L1). Stage A
  // (`prep_res_fold`) runs this many work-groups over the hidden row and writes
  // one fp32 sum-of-squares each; stage B (`prep_norm_finish`) folds exactly
  // these, in ascending index order, into the rms. It is the ONE home for the
  // number: `norm_sumsq` below is sized from it, src/runtime/capture.cc
  // launches both grids from it AND puts it in both variant names
  // (`kernels::prep_res_fold_variant`), so a change here that no compiled
  // binary matches throws at capture instead of reducing the wrong count.
  //
  // 20 at kHidden = 5120 is one element per lane in a 256-lane work-group:
  // 320 subgroups against the single-work-group kernel's 16, which is the axis
  // docs/15 §L2 measured as the one that pays.
  static constexpr uint32_t kNormGroups = 20;

  // `attn`: the decode-attention pair the engine will bind; by default the one
  // `B70_DECODE_ATTN` selects, which is what capture binds, so a plan and an allocation
  // made in the same process agree.
  static DecodeScratchSizes sizes(uint32_t max_len, const model::ModelDesc& desc,
                                  DecodeAttn attn = decode_attn());
};

// --- PrefillScratch ------------------------------------------------------------------

// The eager members (what `PrefillScratch::bytes()` sums) and the four lazy ones, each
// at the size its accessor allocates on first use. Which lazy ones a session builds is
// a property of the backend and the attention path, not of this struct: see
// `prefill_lazy_bytes` in runtime/memory_plan.h.
struct PrefillScratchSizes {
  size_t ids, resid, x, partials, ab_out, norm_sumsq, gdn_o, mixer_out, logits, argmax_part,
      gdn_xb, gdn_seed, gdn_g, gdn_beta, gdn_A, gdn_A2, gdn_w, gdn_u, pf_q, pf_attn, pf_o,
      pf_rowsum;
  size_t dequant, slab, pf_s, pf_p;   // lazy
  size_t eager() const {
    return ids + resid + x + partials + ab_out + norm_sumsq + gdn_o + mixer_out + logits +
           argmax_part + gdn_xb + gdn_seed + gdn_g + gdn_beta + gdn_A + gdn_A2 + gdn_w + gdn_u +
           pf_q + pf_attn + pf_o + pf_rowsum;
  }
};

struct PrefillScratchDims {
  static constexpr uint32_t kC = 2048;               // ruling A13 (was 4096)
  static constexpr uint32_t kGdnChunk = 64;          // the FLA intra-chunk size
  static constexpr uint32_t kNormGroups = DecodeScratchDims::kNormGroups;

  static PrefillScratchSizes sizes(uint32_t max_len, const model::ModelDesc& desc);
};

// --- MtpBuffers (spec 8) -------------------------------------------------------------

struct MtpSizes {
  size_t hctl, gdn_spec, kv_k, kv_v, hh, dh, logits;
  size_t total() const { return hctl + gdn_spec + kv_k + kv_v + hh + dh + logits; }
};

struct MtpDims {
  static constexpr uint32_t kSlots = 4;   // M <= 4: K <= 3 drafts + the pending token
  static constexpr uint32_t kMaxK = kSlots - 1;
  static MtpSizes sizes(uint32_t max_len, const model::ModelDesc& desc);
};

// Spec 8 (plan 8b Task 3): `Engine::prefill`'s MTP hidden rows, bf16 [kC + 1][hidden],
// allocated on the first prefill of an engine that carries the head.
size_t mtp_prefill_hidden_bytes(const model::ModelDesc& desc);

// --- prefill::Int8State (spec 5, the h8 path) ----------------------------------------

// The three scratch buffers Int8State's constructor allocates for its `max_k` (the
// MLP intermediate: down's K), and - separately, because they are built per weight
// on first use - the rotated column scales: ws and 1/ws, fp32 [N] each, for every
// int4 linear of every layer (lm_head excluded: it is never an h8 linear). Both are
// what `Int8State::bytes()` counts. The per-K sign tables (`signs_f32`, `sign_bits`:
// K x 4 + K / 8 bytes for each of the 3-4 distinct K, ~0.1 MB) are NOT in bytes() and
// so are not here either; they are inside the reserve (memory_plan.h).
struct Int8ScratchSizes {
  size_t xq, xs, w8;
  size_t total() const { return xq + xs + w8; }
};
Int8ScratchSizes int8_scratch_sizes(uint32_t max_k);
size_t int8_scale_bytes(const model::ModelDesc& desc);

}  // namespace runtime
