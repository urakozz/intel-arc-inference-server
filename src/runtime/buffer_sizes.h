#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
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

// --- the KV cache's form (spec 12b) --------------------------------------------------

// `--kv-cache bf16|int8`. bf16 is today's cache, bit for bit. int8 is the operator's
// `rotkv` scheme (spec 12 §8; src/common/kv8.h, src/kernels/kv8.cl): every K and V row
// rotated by the same 256-point Hadamard and stored as int8 with one fp16 scale per
// (position, kv head). `B70_KV_CACHE=bf16|int8` sets the default (read at each call,
// so a test can set it between engines; unset or empty is bf16; anything else throws), which is how the gate tests run the same binaries
// over the int8 cache; the CLIs pass their flag explicitly.
enum class KvCache { Bf16, Int8 };
inline constexpr KvCache kDefaultKvCache = KvCache::Bf16;   // until spec 12's gates pass
KvCache default_kv_cache();
KvCache parse_kv_cache(const char* v);   // "bf16" | "int8"; throws std::runtime_error
const char* kv_cache_name(KvCache kv);

// One FA layer's cache, either form: the K and V rows, and at int8 their scales (null at
// bf16). Pointers only - device-free.
struct KvLayer {
  void* k = nullptr;
  void* v = nullptr;
  void* ks = nullptr;   // int8: fp16 bits [max_len][kv-heads]
  void* vs = nullptr;
  bool int8() const { return ks != nullptr; }
};

// The layout of ONE K (or V) allocation of `layers` FA layers - the main cache
// (PersistentBuffers::kv_k, fa_layers) or the MTP head's (MtpBuffers::kv_k, 1):
//
//   rows    [layers][max_len][kv_heads][256]   bf16 (2 B) or int8 (1 B)
//   scales  [layers][max_len][kv_heads]        fp16, int8 only, after EVERY layer's rows
//
// At bf16 the scale region is empty and the rows are exactly today's allocation, so a
// layer's rows start where they always did. At int8 a position of one layer is
// kv_heads x (256 + 2) B: 1032 B on Qwen3.8 against 2048.
struct KvLayout {
  KvCache form = KvCache::Bf16;
  uint32_t max_len = 0, kv_heads = 0, head_dim = 0, layers = 0;

  size_t row_bytes() const {   // one position of one layer's K (or V) rows
    return size_t(kv_heads) * head_dim * (form == KvCache::Int8 ? 1 : 2);
  }
  size_t scale_row_bytes() const { return form == KvCache::Int8 ? size_t(kv_heads) * 2 : 0; }
  size_t pos_bytes() const { return row_bytes() + scale_row_bytes(); }
  size_t layer_rows() const { return size_t(max_len) * row_bytes(); }
  size_t layer_scales() const { return size_t(max_len) * scale_row_bytes(); }
  size_t rows_offset(uint32_t layer) const { return size_t(layer) * layer_rows(); }
  size_t scales_offset(uint32_t layer) const {
    return size_t(layers) * layer_rows() + size_t(layer) * layer_scales();
  }
  size_t bytes() const { return size_t(layers) * (layer_rows() + layer_scales()); }
  // Layer `layer`'s pointers inside the K and V allocations at `k_base` / `v_base`.
  KvLayer layer(void* k_base, void* v_base, uint32_t layer) const;
  // Spec 13c: a layer's pointers moved to position `base` - the rows by base rows, the scales by base scale rows - so
  // a kernel that indexes positions from the pointers it is given reads and writes [base, base + n) (a sequence
  // slot's KV range). base 0 is `l` itself.
  KvLayer at_base(KvLayer l, uint32_t base) const;
};
KvLayout kv_layout(uint32_t max_len, const model::ModelDesc& desc, uint32_t layers, KvCache kv);

// --- PersistentBuffers ---------------------------------------------------------------

struct PersistentSizes {
  size_t control, gdn_state, conv_ring, kv_k, kv_v;
  size_t total() const { return control + gdn_state + conv_ring + kv_k + kv_v; }
};

struct PersistentDims {
  static constexpr uint32_t kConvRing = 16;   // ring depth >= M + 3 (spec §9.4)
  // kv_k / kv_v are kv_layout(max_len, desc, fa_layers, kv).bytes() each (spec 12b).
  static PersistentSizes sizes(uint32_t max_len, const model::ModelDesc& desc,
                               KvCache kv = default_kv_cache());
  // Spec 16b (pipeline parallel): one stage's persistent state - the same formulas over
  // `gdn_layers` GDN and `fa_layers` FA layers instead of the model's. At the descriptor's
  // own counts it equals sizes() (tests/runtime/pipeline_plan_test.cc pins that).
  static PersistentSizes stage_sizes(uint32_t max_len, const model::ModelDesc& desc, KvCache kv,
                                     uint32_t gdn_layers, uint32_t fa_layers);
};

// --- DecodeScratch -------------------------------------------------------------------

struct DecodeScratchSizes {
  size_t resid, x, partials, ab_out, norm_sumsq, gdn_o, attn_q, attn_gate, attn_part,
      attn_out, logits, argmax_part;
  size_t moe;   // spec 15c: MoeScratchLayout::total, 0 on a dense model (no allocation)
  size_t total() const {
    return resid + x + partials + ab_out + norm_sumsq + gdn_o + attn_q + attn_gate + attn_part +
           attn_out + logits + argmax_part + moe;
  }
};

// Spec 15c: a mixture-of-experts model's decode scratch - ONE allocation
// (DecodeScratch::moe), three regions, sized here and nowhere else:
//
//   logits  fp32 [layer slots][kM][router_n]   the router || shared-gate GEMV's output
//   route   u32  [layer slots][kM][kMoeRouteWords]   moe_route's rows (ids, weights, gate)
//   h       bf16 [kM][top_k + 1][expert_intermediate]   SiLU(gate) x up of every slot
//
// The first two are PER LAYER - a layer's MoE block writes its own slice - so after a
// step every layer's routing is still on the card for R2's router gate to read back
// (tests/golden/ornith_decode_test.cc) without a debug list; ~0.4 MB on Ornith. The
// layer slots are layers + 1: the MTP head's MoE layer (spec 15e) gets the last one,
// so no kernel or buffer assumes a layer index below `layers`.
inline constexpr uint32_t kMoeRouteWords = 32;   // = kernels::moe_route::kWords (capture.cc)
struct MoeScratchLayout {
  uint32_t layer_slots = 0;
  size_t logits_layer = 0, route_layer = 0;   // bytes per layer slot
  size_t logits_off = 0, route_off = 0, h_off = 0, h = 0;
  size_t total = 0;
  size_t logits_at(uint32_t layer) const { return logits_off + size_t(layer) * logits_layer; }
  size_t route_at(uint32_t layer) const { return route_off + size_t(layer) * route_layer; }
};
MoeScratchLayout moe_scratch_layout(const model::ModelDesc& desc);   // all zero when dense

struct DecodeScratchDims {
  static constexpr uint32_t kM = 8;
  // KV positions per `attn_decode` work-group - spec 1.5's lever L5 took it
  // from 256 to 64. Since the zero-config cleanup (2026-11-06) decode attention v1
  // (attn.cl's attn_decode / attn_reduce) is no engine's: it survives only as the
  // test-only reference attn_v2_test grades v2 against (and attn_test grades it), at
  // max_len 4096 / 131072, and this is its block size. The text below is its history. It is the ONE home for the number on the host: `attn_part`
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
  // **64 is measured, not derived - but it is not a knee either.** docs/15 part 2's
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
  // 0.070 ms. docs/12 "Decode attention" → Measured carries the arithmetic.
  static constexpr uint32_t kAttnBlock = 64;
  // Spec 10 (plan 10b): decode attention v2's work-groups per kv head, TGT in
  // src/kernels/attn_v2.cl and `_T32` in its variant name. v2 walks a row of L keys in
  // blocks of max(64, roundup64(ceil(L / 32))) positions, so no row has more than 32
  // partials: v2 needs [24][32][M][258] floats whatever max_len is, which is how
  // `attn_part` is sized.
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
  // docs/15, "What the measurement says that nothing else did" measured as the one that pays.
  static constexpr uint32_t kNormGroups = 20;

  static DecodeScratchSizes sizes(uint32_t max_len, const model::ModelDesc& desc);
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
  size_t slab, pf_s, pf_p, kv8_raw;   // lazy; kv8_raw is exact, unscaled bf16 K/V for one layer
  size_t moe = 0;   // spec 15d: MoePrefillLayout::total, eager on a MoE model, 0 when dense
  size_t eager() const {
    return ids + resid + x + partials + ab_out + norm_sumsq + gdn_o + mixer_out + logits +
           argmax_part + gdn_xb + gdn_seed + gdn_g + gdn_beta + gdn_A + gdn_A2 + gdn_w + gdn_u +
           pf_q + pf_attn + pf_o + pf_rowsum + moe;
  }
};

// Spec 15d: a mixture-of-experts model's PREFILL scratch - one allocation
// (PrefillScratch::moe), regions at these offsets, sized here and nowhere else
// (src/kernels/prefill/pf_moe.cl, pf_moe_sort_mw.cl, pf_moe_q4gemm.cl and pf_moe_w4a8.cl say what each holds):
//
// (kC here is the session's chunk cap, 2048 or 4096.)
//
//   logits   fp32 [kC][router_n]            the router || shared-gate GEMV over the chunk
//   route    u32  [layers][kC][32]          moe_route's rows, PER LAYER: after a prefill every
//                                           layer's routing is still there for R2 and the
//                                           prefill-vs-decode routing check to read back
//   hdr      u32  [pf_moe::hdr_words]       pf_moe_sort's header (counts, tiles used, Rs)
//   tiles    u32  [tmax][2]                 the padded tile table: (weight block, first row)
//   row_tok  u32  [rows]                    the token of every sorted row (NONE: padding)
//   pair_row u32  [kC][top_k]               the sorted row of every (token, slot)
//   h        bf16 [rows][expert_intermediate]   SiLU(gate) x up of every sorted row
//   xg       bf16 [rows][hidden]            the gathered activations (l0-int8: int8 [rows][hidden]
//                                           in its first half), then - gate||up having consumed
//                                           them - the down GEMM's rne'd output y
//   cnt, items, xsg_tok, xsg               the sort's block counts and work items, W4A8's group scales
//   (no weight batch since plan 2026-11-12: the GEMMs read the int4 blocks in place)
//
// rows = TM x tmax(kC), tmax(C) = floor((C x top_k + experts x (TM - 1)) / TM) + ceil(C / TM):
// every expert's rows padded to whole TM-row tiles, the shared expert's C rows last
// (moe_prefill_tiles). Ornith at kC 2048: 824 tiles, 26,368 rows; ~0.15 GB in all.
struct MoePrefillLayout {
  uint32_t tmax = 0, rows = 0;   // at the chunk cap
  size_t logits_off = 0, route_off = 0, route_layer = 0, hdr_off = 0, tiles_off = 0,
         row_tok_off = 0, pair_row_off = 0, h_off = 0, xg_off = 0,
         cnt_off = 0,            // pf_moe_sort_mw's per-block counts, u32 [ceil(C / 64)][experts]
         items_off = 0,          // its work items, u32 [ibound][4] (the fused GEMMs' grid)
         xsg_tok_off = 0,        // W4A8 (l0-int8): the chunk's per-token 64-k group scales, fp32 [C][hidden / 64]
         xsg_off = 0,            // and the sorted rows', fp32 [rows][hidden / 64]
         // Phase 3 Task 2 (l0): the router || shared gate's and the a||b's dense GEMM output, INSIDE xg (no bytes
         // of its own): fp32 [pad256(kC)][gemm_pitch]. The router's rows are consumed by moe_route before the
         // gather writes xg, the a||b's by pf_gdn_gate before the layer's MoE block.
         gemm_out_off = 0;
  uint32_t gemm_pitch = 0;       // the wider of the two GEMMs' padded N (pf_route_slab_pitch)
  uint32_t ibound = 0;           // at the chunk cap
  size_t total = 0;
  size_t route_at(uint32_t layer) const { return route_off + size_t(layer) * route_layer; }
};
// At the session's chunk cap (plan 2026-11-12 Task 6: 2048 or 4096); all zero when dense.
MoePrefillLayout moe_prefill_layout(const model::ModelDesc& desc, uint32_t chunk_cap = 2048);
// tmax(C): the tile table's fixed length for a C-row chunk - the grouped GEMMs' grid.
uint32_t moe_prefill_tiles(const model::ModelDesc& desc, uint32_t C);
// Phase 3 Task 2: the l0 backend's persistent pf_gemm B slabs of a MoE model's layers [first, last) - every layer's
// router || shared gate, bf16 [hidden][pf_route_slab_pitch(router_n)], and every GDN layer's a||b, bf16
// [hidden][pf_route_slab_pitch(128)] (PrefillScratch::gemm_slab; Ornith 40 x 2 MiB + 30 x 1 MiB). 0 when dense.
size_t route_slab_bytes(const model::ModelDesc& desc, uint32_t first, uint32_t last);
// The work items' bound at C (pf_moe_sort_mw.cl, MI = kernels::pf_moe::kItemM rows an item): the fused grouped
// GEMMs' grid, a function of C alone.
uint32_t moe_prefill_items(const model::ModelDesc& desc, uint32_t C);

struct PrefillScratchDims {
  // Plan 5d: the chunk's row count is a SESSION value, the chunk cap - kCDefault (ruling
  // A13's 2048) or kCMax (4096) - and every per-chunk buffer is sized by it. There is no
  // single kC any more: a site that bounds a chunk asks the scratch it writes
  // (PrefillScratch::chunk_cap(), Int8State::chunk_cap()). A MoE model takes either since plan
  // 2026-11-12 (the multi-work-group sort has no chunk bound of its own).
  static constexpr uint32_t kCDefault = 2048;        // ruling A13 (was 4096)
  static constexpr uint32_t kCMax = 4096;            // plan 5d
  static constexpr uint32_t kGdnChunk = 64;          // the FLA intra-chunk size
  static constexpr uint32_t kNormGroups = DecodeScratchDims::kNormGroups;
  static_assert(kCDefault % kGdnChunk == 0 && kCMax % kGdnChunk == 0,
                "a chunk cap must be a whole number of 64-position GDN chunks");

  // Throws std::invalid_argument for a cap check_prefill_chunk_cap refuses.
  static PrefillScratchSizes sizes(uint32_t max_len, const model::ModelDesc& desc, uint32_t chunk_cap);
};

// Plan 5d: the chunk caps a model may run. Throws std::invalid_argument unless `chunk_cap`
// is kCDefault or kCMax (a MoE model takes either since plan 2026-11-12 Task 6).
void check_prefill_chunk_cap(const model::ModelDesc& desc, uint32_t chunk_cap);
// The cap a `--prefill-chunk C` width needs: 0 for C = 0 (nothing asked), kCDefault for
// C <= 2048, kCMax for C <= 4096; above 4096, or a cap the model refuses, throws
// std::invalid_argument with the reason.
uint32_t prefill_chunk_cap_for(uint32_t chunk, const model::ModelDesc& desc);
// The cap an engine uses when nobody set one (Engine / PipelineEngine::set_prefill_chunk_cap):
// kCDefault. Only a test moves it, for its whole process (testing:: below, through the
// `--chunk-cap N` argument of tests/chunk_cap_arg.h) - how the gate tests run the same
// binaries at 4096. No CLI flag, server option or environment variable reaches it (operator,
// 2026-11-03: no new engine env vars): the CLIs choose the cap themselves (cli/max_len.h).
uint32_t default_prefill_chunk_cap();
namespace testing {
// Test-only seam: kCDefault or kCMax, else std::invalid_argument. Never called by src/.
void set_default_prefill_chunk_cap(uint32_t cap);
}  // namespace testing

// --- MtpBuffers (spec 8) -------------------------------------------------------------

struct MtpSizes {
  size_t hctl, gdn_spec, kv_k, kv_v, hh, dh, logits;
  size_t dv_logits;   // spec 8 §11: fp32 [|V'|], 0 without a draft vocabulary
  size_t total() const { return hctl + gdn_spec + kv_k + kv_v + hh + dh + logits + dv_logits; }
};

struct MtpDims {
  static constexpr uint32_t kSlots = 4;   // M <= 4: K <= 3 drafts + the pending token
  static constexpr uint32_t kMaxK = kSlots - 1;
  // Spec 19a (plan 19a Task 4): the most GDN state slots a buffer may carry - a verify
  // list at M rows needs M slots, and DecodeScratchDims::kM = 8 bounds M.
  static constexpr uint32_t kMaxSlots = 8;
  // `draft_vocab` = |V'| (spec 8 §11, `--draft-vocab`), 0 when the full head drafts.
  // `kv`: the head's KV follows the main cache's form (spec 12b), one layer.
  // `slots` (spec 19a): the GDN state slots, kSlots for every engine; only
  // tools/probe/probe_mtp_steps' M = 5..8 lists ask for more (gdn_spec grows by one
  // gdn_state per slot). Must be in [kSlots, kMaxSlots].
  // `seqs` (spec 13c): the engine's sequence slots (SlotDims below; 1 = today's engine, every
  // size unchanged). Each further sequence adds its kSlots GDN states to gdn_spec (after the
  // first sequence's kSlots - 1: sequence q's state k is GDN index q x kSlots + k, index 0 being
  // PersistentBuffers::gdn_state and index i > 0 gdn_spec's state i - 1 - the row lists' GDN
  // index) and its hh rows (hh_rows).
  static MtpSizes sizes(uint32_t max_len, const model::ModelDesc& desc, uint32_t draft_vocab = 0,
                        KvCache kv = default_kv_cache(), uint32_t slots = kSlots, uint32_t seqs = 1);
  // Spec 17d: a tensor-parallel card's - the GDN state slots at the card's shapes (`card`, model::tp_card_desc), the
  // head's KV layer at the head's own (`head`: the one-card descriptor - the MTP head is whole on each card). No draft
  // vocabulary, one sequence, kSlots.
  static MtpSizes tp_sizes(uint32_t max_len, const model::ModelDesc& card, const model::ModelDesc& head, KvCache kv);
  // Spec 13c: MtpBuffers::hh's rows. One sequence: kM + 1 (row 0 = h_{pos-1}, rows 1..M the
  // verify rows' hidden). `seqs` > 1: each sequence's kM + 1 rows (sequence q's from row
  // q x (kM + 1)), then the row lists' saved hidden rows (kM: the rows' post-final-norm `x`),
  // the head's input rows (kM, gathered from the two regions before by embed_gather) and the
  // row draft list's chain rows (kM: runtime::build_draft_rows' dh).
  static uint32_t hh_rows(uint32_t seqs);
};

// --- spec 13c: the sequence slots of one card ----------------------------------------------
//
// An engine with `seqs` > 1 sequence slots (spec 13 §11.2: the memory an asked --max-len leaves
// becomes 32768-position slots) holds ONE KV allocation at the slots' summed positions (slot q is
// a position range of it, from its KV base), and per slot beyond the first a GDN state group, a
// conv ring and control blocks. What PersistentBuffers / MtpBuffers do not already hold,
// SlotBuffers (buffers.h) allocates, sized here:
//
//   states    without MTP: the fp32 GDN states of slots 1..seqs-1, one whole-model state each
//             (with MTP they are MtpBuffers::gdn_spec's: MtpDims::sizes' `seqs`)
//   rings     the bf16 conv rings of slots 1..seqs-1, one whole-model ring each
//   controls  shared memory: slots 1..seqs-1's Control (and with MTP their head's), the row
//             lists' Control, head Control and gather Control, the row draft list's head Control,
//             and their three row tables
struct SlotSizes {
  size_t states = 0, rings = 0, controls = 0;
  size_t total() const { return states + rings + controls; }
};
struct SlotDims {
  static constexpr uint32_t kMaxSeqs = 4;         // slot 0 and at most 3 short slots (spec 13 §11.2)
  static constexpr uint32_t kShortLen = 32768;    // a short slot's positions (none shorter)
  // All zero at seqs = 1. Throws std::invalid_argument outside [1, kMaxSeqs].
  static SlotSizes sizes(const model::ModelDesc& desc, uint32_t seqs, bool mtp);
  // Byte offsets inside SlotBuffers::controls (a Control is 128 B, a row table 256 B).
  static size_t control_at(uint32_t seqs, uint32_t seq);        // slot seq's Control, 1 <= seq < seqs
  static size_t head_control_at(uint32_t seqs, uint32_t seq);   // its head's Control (MTP)
  static size_t rows_control_at(uint32_t seqs);       // the row lists' Control
  static size_t rows_head_control_at(uint32_t seqs);  // the row lists' head Control (MTP)
  static size_t gather_control_at(uint32_t seqs);     // the head's input gather (MTP)
  static size_t draft_control_at(uint32_t seqs);      // the row draft list's head Control (MTP)
  static size_t rows_table_at(uint32_t seqs);         // runtime::BatchRowTable, the main rows
  static size_t head_table_at(uint32_t seqs);         // the head's rows (MTP)
  static size_t draft_table_at(uint32_t seqs);        // the row draft list's head rows (MTP)
};
// --- spec 19c: a tap drafter's verify state (DFlash) --------------------------------------
//
// With a drafter (runtime/tap_drafter.h) the Engine's verify lists run up to M = 8 rows (K <= 7 drafts + the
// pending id), so the GDN state needs 8 slots: slot 0 is PersistentBuffers::gdn_state, gdn_spec_bytes() the
// other 7 (one gdn_state each, slot-major - MtpBuffers::gdn_spec's layout at 8 slots). Prefill stages the taps
// of a chunk's rows inside the drafter's window: bf16 [chunk_cap][taps][hidden] (prefill_taps_bytes), allocated on
// the first prefill of an engine with a drafter.
struct TapVerifyDims {
  static constexpr uint32_t kSlots = 8;
  static size_t gdn_spec_bytes(const model::ModelDesc& desc);
  static size_t prefill_taps_bytes(const model::ModelDesc& desc, uint32_t n_taps, uint32_t chunk_cap);
};
// Where the verify list copies each tap: for every site s in [0, layers] the indices t into `layers` whose
// tapped layer is s - 1 - copied after the input fold of layer s (layer s - 1's output), or for s == layers
// after the final norm's fold (the last layer's output). Throws std::invalid_argument unless `layers` is
// non-empty, strictly ascending and every entry < desc.layers. Host only (tests/runtime/dflash_engine_host_test).
std::vector<std::vector<uint32_t>> tap_sites(const model::ModelDesc& desc, const std::vector<uint32_t>& layers);

// The draft vocabulary's compact head and id table (spec 8 §11) are the LOADER's
// allocations, sized by loader::draft_vocab_bytes (loader/draft_vocab.h, header-only
// so this archive needs no loader); memory_plan.h adds them as their own term.

// Spec 8 (plan 8b Task 3): `Engine::prefill`'s MTP hidden rows, bf16 [chunk_cap + 1][hidden],
// allocated on the first prefill of an engine that carries the head.
size_t mtp_prefill_hidden_bytes(const model::ModelDesc& desc, uint32_t chunk_cap);

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
Int8ScratchSizes int8_scratch_sizes(uint32_t max_k, uint32_t chunk_cap);   // xq / xs: chunk_cap rows
size_t int8_scale_bytes(const model::ModelDesc& desc);
// Spec 16c: the same for layers [first, last) only - one pipeline stage's linears, which is
// what that device's Int8State holds (int8_scale_bytes(desc) is the whole range).
size_t int8_scale_bytes(const model::ModelDesc& desc, uint32_t first, uint32_t last);
// The `max_k` Engine::prepare_prefill builds Int8State with: the widest int4 K the h8
// walk runs - the MLP intermediate (down's K) on a dense model, as since spec 14; on a
// MoE model (spec 15d) the widest dense linear's K, out_proj / o_proj's 4096 on Ornith
// (the experts' gate||up K is the hidden size; their down is not an h8 linear).
uint32_t prefill_int8_max_k(const model::ModelDesc& desc);
inline uint32_t int8_padded_k(uint32_t K) { return (K + 1023) / 1024 * 1024; }

}  // namespace runtime
