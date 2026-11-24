#pragma once
#include <string>

#include "kernels/shape_suffix.h"

// Spec 13a (plan 13a Task 2) / 13c: the row builds' device binaries by name - the host half of src/kernels/
// CMakeLists.txt's "rows of different sequences" block (built by default since plan 13c), in its own header so
// kernels.h is not edited (dflash_kernels.h's arrangement). Define-free, so the host test (tests/runtime/batch_rows_test.cc) includes it without a kernel dir.
//
// A `_ROWS` binary is its production twin with -DBATCH_ROWS: the per-sequence quantities - the position, the KV base,
// the GDN state read and written, the conv ring - come from the row table (runtime::BatchRowTable, the extra last
// argument) instead of Control::pos / gdn_live and the layer slices. Everything else in the binary is the twin's
// text (the BATCH_ROWS blocks of gdn_step.cl, attn.cl, attn_v2.cl and kv8.cl).
namespace kernels {

// The row table's words as the kernels index it: the BROW_DEFINES line of src/kernels/CMakeLists.txt. The host
// struct is runtime::BatchRow (runtime/batch_rows.h); src/runtime/capture.cc static_asserts the
// two together and tests/runtime/batch_rows_test.cc repeats the values.
namespace batch_row {
inline constexpr unsigned kWords = 8;    // BROW_WORDS: one row's stride
inline constexpr unsigned kPos = 0;      // BROW_POS
inline constexpr unsigned kKvBase = 1;   // BROW_KVB
inline constexpr unsigned kSrc = 2;      // BROW_SRC
inline constexpr unsigned kDst = 3;      // BROW_DST
inline constexpr unsigned kSlot = 4;     // BROW_SLOT
inline constexpr unsigned kPend = 5;     // BROW_PEND (plan 40a): the run's sequence's pending count | parity << 8
}  // namespace batch_row

// gdn_step's row build: SPEC_SLOT_STRIDE (one whole-model GDN state) and RING_SLOT_STRIDE (one whole-model conv
// ring) are baked, so the GDN layer count is in the name as in gdn_step_slots_variant (48 keeps none).
inline std::string gdn_step_rows_variant(unsigned M, unsigned gdn_layers, unsigned gdn_k = kRefGdnKHeads,
                                         unsigned gdn_v = kRefGdnVHeads) {
  return "gdn_step_rows_M" + std::to_string(M) + (gdn_layers == 48 ? std::string() : "_G" + std::to_string(gdn_layers)) +
         gdn_suffix(gdn_k, gdn_v);
}
// Plan 40a Task 2: the DEFERRED row build (BROW_PEND, PEND_SEQ - the shapes kernels::gdn_defer_built names; PERF-120:
// Ornith's `_G30_GK16V32` too).
inline std::string gdn_step_rows_defer_variant(unsigned M, unsigned gdn_layers = 48, unsigned gdn_k = kRefGdnKHeads,
                                               unsigned gdn_v = kRefGdnVHeads) {
  return "gdn_step_rows_defer_M" + std::to_string(M) +
         (gdn_layers == 48 ? std::string() : "_G" + std::to_string(gdn_layers)) + gdn_suffix(gdn_k, gdn_v);
}
// Plan 40a Task 3: the 16- and 32-row lists' delta-net step - the M 8 deferred build over chunks of whole runs
// (gdn_step.cl, WIDE_ROWS 32), a chunk a sequence a launch at most (grid dimension 2).
inline std::string gdn_step_rows_wide_variant(unsigned gdn_layers = 48, unsigned gdn_k = kRefGdnKHeads,
                                              unsigned gdn_v = kRefGdnVHeads) {
  return "gdn_step_rows_defer_M8_W32" + (gdn_layers == 48 ? std::string() : "_G" + std::to_string(gdn_layers)) +
         gdn_suffix(gdn_k, gdn_v);
}
// Plan 40a Task 3: the 16- and 32-row lists' attention - attn_verify_dpas.cl's M 8 line (BATCH_ROWS: a run a
// work-group's grid dimension 2) writing the partials of attn_v2_rows_variant(8, ...)'s groups for its reduce. Paged
// only (kv_paged's _PG suffix after it).
inline std::string attn_verify_dpas_rows_variant(unsigned T, bool kv8, unsigned q = kRefFaQHeads,
                                                 unsigned kv = kRefFaKvHeads) {
  return "attn_verify_dpas_M8_T" + std::to_string(T) + fa_suffix(q, kv) + (kv8 ? "_KV8" : "") + "_ROWS";
}
// PERF-120: the 16- and 32-row lists' MoE block reading each routed expert once for its rows (moe.cl, MOE_UNION):
// moe_variant's name with `_U` - moe_route, moe_union_plan, moe_gate_up_union and moe_down_union over (the plan's
// most items x their n-tile groups), moe_finish over (hidden / 16, M). The down's n-tiles a work-group (moe.cl's NTW),
// the pairs an item (UR), the plan's items at most (MAX_ITEMS) and its words (moe.cl's plan layout).
inline constexpr unsigned kMoeUnionTiles = 8, kMoeUnionPairs = 8;
inline constexpr unsigned moe_union_items(unsigned M, unsigned top_k) {
  return M * top_k + (M + kMoeUnionPairs - 1) / kMoeUnionPairs;
}
inline constexpr unsigned moe_union_plan_words(unsigned M, unsigned top_k) {
  return 1 + 2 * moe_union_items(M, top_k) + M * (top_k + 1);
}
// attn.cl's attn_prep (`s1`: the MTP head's QKV_S = 1 build) and kv8.cl's writer (`kv8`), row builds.
inline std::string attn_prep_rows_variant(unsigned M, bool s1, bool kv8, unsigned q = kRefFaQHeads,
                                          unsigned kv = kRefFaKvHeads) {
  return "attn_prep_M" + std::to_string(M) + fa_suffix(q, kv) + (s1 ? "_S1" : "") + (kv8 ? "_KV8" : "") + "_ROWS";
}
// attn_v2.cl's pair (or kv8.cl's KV8_DEC twin), row builds: groups are a slot's rows of one stride.
inline std::string attn_v2_rows_variant(unsigned M, unsigned T, bool kv8, unsigned q = kRefFaQHeads,
                                        unsigned kv = kRefFaKvHeads) {
  return "attn_v2_M" + std::to_string(M) + "_T" + std::to_string(T) + fa_suffix(q, kv) + (kv8 ? "_KV8" : "") + "_ROWS";
}

}  // namespace kernels
