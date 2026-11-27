#pragma once
#include <array>
#include "loader/inkling2_loader.h"
#include "runtime/gpu_state.h"
#include "runtime/memory_plan.h"

namespace runtime::inkling2 {

inline constexpr uint32_t kRows = 8, kRing = 4096, kDraftRing = 2048, kPartitions = 32;
// ink_glue.cl INK_GLUE_SPLIT: a row's work-groups (one sub-group each), its WG lanes, the depth bank's states.
inline constexpr uint32_t kGlueSplit = 8, kGlueLanes = 128, kGlueBank = 5;
// The drafter's full layer: its key spans a kv head (df_attn FULL_TARGETS; 32 / 64 / 128 / 256 spans: 13.9 / 7.1 / 5.0 /
// 4.7 ms at 128k keys, 54.6 / 27.8 / 19.1 / 19.2 at 512k - probe_ink_draft_attn).
inline constexpr uint32_t kDraftPartitions = 128;
// PERF-123's candidate head (head_cand.cl): C candidates a row from the 2-bit pass, NC = 8 C columns rescored by the
// int8 head; df_candidate's 4096-logit chunks.
struct StepIo {
  uint32_t pos, n_active, cur_token[8];   // Control's first words, in its order
  uint32_t k, proj_pos, proj_n;           // DFlashCtl's first words
  uint32_t out_token[8], drafts[8];       // copied back: the predictions, cur_token (drafts from [1])
  uint32_t accepted;                      // copied back: DFlashCtl::proj_n after the round's ink_accept
};
inline constexpr uint32_t kCandidates = 64, kCandidateColumns = 512, kCandidateChunk = 4096;
inline constexpr size_t kCandidateHeadBytes =
    size_t(model::Inkling2Desc::head_rows / 16) * (model::Inkling2Desc::hidden / 64) * 72 * sizeof(uint32_t);
inline constexpr size_t kCandidateChunks = (model::Inkling2Desc::head_rows + kCandidateChunk - 1) / kCandidateChunk;
inline constexpr size_t kCandidateParts = size_t(kRows) * kCandidateChunks * kCandidates * 8,
                        kCandidateIds = size_t(kRows) * kCandidates * 4,
                        kCandidateWeights = size_t(kCandidateColumns) * model::Inkling2Desc::hidden,
                        kCandidateScales = size_t(kCandidateColumns) * 4,
                        kCandidateScores = size_t(kRows) * kCandidateColumns * 4;
// A prompt chunk's rows (the experts' sort fills its 256-row items best at 4096); attention takes a chunk in parts
// of kPrefillAttentionRows, which the window ring holds with their 1,023-key window.
inline constexpr uint32_t kPrefillRows = 4096, kPrefillAttentionRows = 2048, kPrefillExpertTile = 32,
                          kPrefillSlab = 1024, kDraftInt4Slab = 256;
// `--parallel pp`: the visits one card runs, [first, end) of the walk's 48, and whether the drafter on the last card
// reads taps this card's visits write (they cross with the hand-off). A card holds the layers its visits read, the
// caches and convolution histories of its visits (slots counted from the stage's first), the embedding when it runs
// visit 0 and the head, the final norm and the drafter when it runs the last. One card: every visit.
struct Stage {
  uint32_t first = 0, end = model::Inkling2Desc::visit_count;
  bool taps = false;
  bool whole() const { return first == 0 && end == model::Inkling2Desc::visit_count; }
  bool head() const { return end == model::Inkling2Desc::visit_count; }
};
// `--parallel tp` (plan 41j): this card's part, `rank` of `size`: half of every visit's heads (12 q / 4 kv) - their
// caches, rings and k / v convolution histories - and half of every expert's width (576); the drafter's layers split
// the same way (12 q / 4 kv heads and their caches, 4,608 of the MLP's 9,216 columns), its head and Markov fold by
// rows (100,032 of the 200,064); the rest of the walk, the head and the Markov recurrence whole on both cards. One
// card: rank 0 of 1.
struct Tp {
  uint32_t rank = 0, size = 1;
  constexpr bool on() const { return size > 1; }
  constexpr uint32_t draft_q_heads() const { return model::Inkling2Desc::q_heads / size; }
  constexpr uint32_t draft_kv_heads() const { return model::Inkling2Desc::kv_heads / size; }
  constexpr uint32_t draft_q_n() const { return draft_q_heads() * model::Inkling2Desc::head_dim; }
  constexpr uint32_t draft_kv_n() const { return draft_kv_heads() * model::Inkling2Desc::head_dim; }
  constexpr uint32_t draft_inter() const { return model::Inkling2Desc::draft_inter / size; }
  constexpr uint32_t q_heads() const { return model::Inkling2Desc::q_heads / size; }
  constexpr uint32_t kv_heads() const { return model::Inkling2Desc::kv_heads / size; }
  constexpr uint32_t q_n() const { return q_heads() * model::Inkling2Desc::head_dim; }
  constexpr uint32_t kv_n() const { return kv_heads() * model::Inkling2Desc::head_dim; }
  constexpr uint32_t rel_n() const { return q_heads() * model::Inkling2Desc::relative_dim; }
  constexpr uint32_t inter() const { return model::Inkling2Desc::inter / size; }
};
inline constexpr uint32_t kTpFoldG = 9;   // the all-reduce's chunks a row (tp.cl FOLD_G, kernels::inkling2::kTpFoldG)
// Owner ruling 2026-11-25: the walk splits at visit 24 - card 0 the prelude, pass 1 and pass 2's layers 6-11 (layers
// 0-17), card 1 pass 2's layers 12-17, pass 3 and the coda (layers 6-23); four full-attention visits each.
inline constexpr uint32_t kPpSplit = 24;
// What one stage holds: the full and window slots of its visits (from `full_base` / `window_base` on the whole
// walk's numbering), its layers, its weight bytes.
struct StageShare {
  uint32_t full_base = 0, full = 0, window_base = 0, window = 0;
  std::array<bool, model::Inkling2Desc::layers> layers{};
  size_t layer_bytes = 0;
  uint32_t visits() const { return full + window; }
};
StageShare stage_share(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, const Stage& stage)
    pre(stage.first < stage.end && stage.end <= model::Inkling2Desc::visit_count);
// The two cards' stages at kPpSplit; `draft`: card 0's taps 0 and 1 cross to the drafter on card 1.
std::array<Stage, 2> pp_stages(bool draft);
// The rows live at the split (in the walk's order): the stage input, the two finished blocks (the embedding's and
// the prelude's), the previous pass's layer-12 input and output - f32 each - then, with the drafter, the rows' four
// bf16 tap slots (taps 0 and 1 written; the last card writes 2 and 3).
inline constexpr size_t handoff_row_bytes(bool taps) {
  return 5 * size_t(model::Inkling2Desc::hidden) * sizeof(float) +
         (taps ? 4 * size_t(model::Inkling2Desc::hidden) * sizeof(uint16_t) : 0);
}

enum class SnapTensor { FullK, FullV, FullKScale, FullVScale, RingK, RingV, ConvK, ConvV, ConvAttn, ConvMlp,
                        DraftFullK, DraftFullV, DraftRingK, DraftRingV, DraftTap };
using Snap = gs::Snapshot<SnapTensor>;

struct WeightSizes {
  [[=gs::constant]] size_t model = 0;
  [[=gs::constant]] size_t layers = 0;
  [[=gs::constant]] size_t draft = 0;
  // The int8 prompt pass: its rotated int8 copies of the q / gate / k / v / o linears (written once at load).
  [[=gs::constant]] size_t prompt_int8 = 0;
  // The head's 2-bit copy (gemm_sm's layout 3, built from the int8 head at load): the greedy steps' candidate pass.
  [[=gs::constant]] size_t candidate_head = 0;
};
struct PersistentSizes {
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("restoring the sequence position reconstructs control")]]
  size_t control = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullK}]] size_t full_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullV}]] size_t full_v = 0;
  // `--kv-cache int8` (PERF-102): full_k / full_v hold int8 rows, these their fp16 scales a (position, kv head).
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullKScale}]] size_t full_k_scale = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::FullVScale}]] size_t full_v_scale = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::RingK}]] size_t ring_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::RingV}]] size_t ring_v = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvK}]] size_t conv_k = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvV}]] size_t conv_v = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvAttn}]] size_t conv_attn = 0;
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::ConvMlp}]] size_t conv_mlp = 0;
};
struct ScratchSizes {
  [[=gs::scratch]] size_t residual = 0;
  [[=gs::scratch]] size_t block_bank = 0;
  [[=gs::scratch]] size_t messages = 0;
  [[=gs::scratch]] size_t controller = 0;
  [[=gs::scratch]] size_t inputs = 0;
  [[=gs::scratch]] size_t projections = 0;
  [[=gs::scratch]] size_t full_taps = 0;
  [[=gs::scratch]] size_t attention_parts = 0;
  [[=gs::scratch]] size_t attn_output = 0;
  [[=gs::scratch]] size_t routes = 0;
  [[=gs::scratch]] size_t expert_union = 0;
  [[=gs::scratch]] size_t expert_hidden = 0;
  [[=gs::scratch]] size_t expert_output = 0;
  [[=gs::scratch]] size_t pending_conv = 0;
  [[=gs::scratch]] size_t logits = 0;
  [[=gs::scratch]] size_t argmax_parts = 0;
  [[=gs::scratch]] size_t draft_taps = 0;
  // The candidate head's working set: the top-C passes' parts, the candidates, their gathered int8 columns and scales,
  // and their exact scores (kCandidate* below).
  [[=gs::scratch]] size_t head_candidates = 0;
  // The split mixes' partial sums (ink_glue.cl INK_GLUE_SPLIT): a (stream, row)'s scale sums, then its norm sums.
  [[=gs::scratch]] size_t glue_parts = 0;
  // The plain and drafted steps' host side (host memory, StepIo): their inputs copied into Control and the draft
  // control at the list's head, their outputs copied back at its end, so the host never touches those shared pages
  // in a step (a host write there migrates the page: ~0.2-0.28 ms a step).
  [[=gs::scratch]] size_t step_io = 0;
};
struct DraftSizes {
  [[=gs::constant]] size_t rope = 0;
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("each draft resets the recurrence from its anchor")]]
  size_t draft_control = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftFullK}]] size_t draft_full_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftFullV}]] size_t draft_full_v = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftRingK}]] size_t draft_ring_k = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{SnapTensor::DraftRingV}]] size_t draft_ring_v = 0;
  // The last committed position's four target taps: the Markov head's context for the next draft.
  [[=gs::state, =gs::zero_on_reset, =Snap{SnapTensor::DraftTap}]] size_t draft_tap = 0;
  [[=gs::scratch]] size_t block_hidden = 0;
  [[=gs::scratch]] size_t block_projections = 0;
  [[=gs::scratch]] size_t draft_parts = 0;
  [[=gs::scratch]] size_t context_mix = 0;
  [[=gs::scratch]] size_t markov = 0;
  [[=gs::scratch]] size_t draft_logits = 0;
};
struct PrefillSizes {
  [[=gs::scratch]] size_t ids = 0;
  [[=gs::scratch]] size_t residual = 0;
  [[=gs::scratch]] size_t block_bank = 0;
  [[=gs::scratch]] size_t messages = 0;
  [[=gs::scratch]] size_t controller = 0;
  [[=gs::scratch]] size_t inputs = 0;
  [[=gs::scratch]] size_t projections = 0;
  [[=gs::scratch]] size_t conv = 0;
  [[=gs::scratch]] size_t full_taps = 0;
  [[=gs::scratch]] size_t attention = 0;
  [[=gs::scratch]] size_t split_parts = 0;   // PERF-12: a short chunk's attention partials, a key span each
  [[=gs::scratch]] size_t linear_slab = 0;
  [[=gs::scratch]] size_t linear_tail_input = 0;
  [[=gs::scratch]] size_t linear_tail_output = 0;
  [[=gs::scratch]] size_t route_logits = 0;
  [[=gs::scratch]] size_t routes = 0;
  [[=gs::scratch]] size_t expert_counts = 0;
  [[=gs::scratch]] size_t expert_header = 0;
  [[=gs::scratch]] size_t expert_rows = 0;
  [[=gs::scratch]] size_t expert_pairs = 0;
  [[=gs::scratch]] size_t expert_items = 0;
  [[=gs::scratch]] size_t expert_inputs = 0;
  [[=gs::scratch]] size_t expert_hidden = 0;
  [[=gs::scratch]] size_t expert_output = 0;
  [[=gs::scratch]] size_t draft_taps = 0;
  [[=gs::scratch]] size_t draft_projection = 0;
  [[=gs::scratch]] size_t prelude_trace = 0;
  // The int8 prompt pass: a linear's input rows rotated and quantised (int8 [rows][K]), their scales.
  [[=gs::scratch]] size_t int8_rows = 0;
  [[=gs::scratch]] size_t int8_scales = 0;
  // `--parallel tp`: the chunk's part of an all-reduce (fp32 [rows][hidden]) and the landing of the other card's,
  // two slots (an all-reduce uses slot n % 2), written by that card's copy engine.
  [[=gs::scratch]] size_t tp_part = 0;
  [[=gs::scratch]] size_t tp_landing = 0;
};
static_assert(gs::sizes_problem<WeightSizes, SnapTensor>().empty(), gs::sizes_problem<WeightSizes, SnapTensor>());
static_assert(gs::sizes_problem<PersistentSizes, SnapTensor>().empty(), gs::sizes_problem<PersistentSizes, SnapTensor>());
static_assert(gs::sizes_problem<ScratchSizes, SnapTensor>().empty(), gs::sizes_problem<ScratchSizes, SnapTensor>());
static_assert(gs::sizes_problem<DraftSizes, SnapTensor>().empty(), gs::sizes_problem<DraftSizes, SnapTensor>());
static_assert(gs::sizes_problem<PrefillSizes, SnapTensor>().empty(), gs::sizes_problem<PrefillSizes, SnapTensor>());
static_assert(gs::snapshot_problem<SnapTensor, PersistentSizes, DraftSizes>().empty());

// `--parallel pp`: the hand-off's buffers. The verify width's landing (handoff_row_bytes x kRows) and the prompt
// chunk's on the card that receives; the prompt chunk's outbox (its rows copied out of the scratch, so the next chunk
// may start while the copy engine sends them) on the card that sends.
struct HandoffSizes {
  [[=gs::scratch]] size_t landing = 0;
  [[=gs::scratch]] size_t prefill_landing = 0;
  [[=gs::scratch]] size_t prefill_outbox = 0;
  // The prefix cache under pp: the card's snapshot state (Engine::state_runs) at two mid-prompt block ends, copied on
  // the card right after their chunks, so the host stores them while the next chunks run. Under tp two images of the
  // state buffers (Engine::shadow_image), each visit's slots copied right after the visit (the next chunk in flight
  // rewrites them a visit later).
  [[=gs::scratch]] size_t snapshot_shadow = 0;
};
static_assert(gs::sizes_problem<HandoffSizes, SnapTensor>().empty(), gs::sizes_problem<HandoffSizes, SnapTensor>());

// `--parallel tp`: the all-reduce's buffers (tp.cl, runtime/tp_protocol.h's layout at hidden 2304, one slice): the
// landing the other card's tp_send writes, this card's send and sum counters, the first failure's state words (shared,
// read by the host after the fence), and the part this card sends (fp32 [kRows][hidden]).
struct TpSizes {
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("a landing slot is read only in the step that filled it")]]
  size_t tp_landing = 0;
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("both cards' counters advance together; a restore keeps them")]]
  size_t tp_send_ctr = 0;
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("both cards' counters advance together; a restore keeps them")]]
  size_t tp_sum_ctr = 0;
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("a step's first failure, read after its fence")]]
  size_t tp_state = 0;
  [[=gs::scratch]] size_t tp_part = 0;
};
static_assert(gs::sizes_problem<TpSizes, SnapTensor>().empty(), gs::sizes_problem<TpSizes, SnapTensor>());

struct Sizes {
  WeightSizes weights;
  PersistentSizes persistent;
  ScratchSizes scratch;
  DraftSizes draft;
  PrefillSizes prefill;
  PrefillSizes prefill_b;   // `--parallel tp`: the second chunk in flight's scratch
  HandoffSizes handoff;
  TpSizes tp;
  MemoryComponents memory() const;
};

// kv8: the full visits' cache in int8 (`--kv-cache int8`); the rings and the drafter's caches stay bf16.
// int8_pass: the int8 prompt pass's rotated linears and its quantised rows.
// stage: one card's share under `--parallel pp` (draft then means the drafter is on this card: the last stage).
// tp: a `--parallel tp` card (its layout tp_weights'); pp's stage and tp are not combined.
Sizes sizes(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, uint32_t max_len, bool draft,
            uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false, const Stage& stage = {},
            const Tp& tp = {})
    pre(!tp.on() || stage.whole());
uint32_t max_len_that_fits(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout,
                           size_t capacity, size_t reserve, bool draft, uint32_t prefill_rows = 0, bool kv8 = false,
                           bool int8_pass = false);
// `--parallel pp`: each card's sizes (pp_stages(draft); the drafter on card 1), and the longest context both cards hold.
std::array<Sizes, 2> pp_sizes(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, uint32_t max_len,
                              bool draft, uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false);
// `--parallel tp`: each card's sizes (card r's layout tp_weights(d, layout, r)), and the longest context both hold.
std::array<Sizes, 2> tp_sizes(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, uint32_t max_len,
                              bool draft, uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false);
uint32_t tp_max_len_that_fits(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout,
                              size_t capacity, size_t reserve, bool draft, uint32_t prefill_rows = 0, bool kv8 = false,
                              bool int8_pass = false);
uint32_t pp_max_len_that_fits(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout,
                              size_t capacity, size_t reserve, bool draft, uint32_t prefill_rows = 0, bool kv8 = false,
                              bool int8_pass = false);
// `--parallel pp`'s prompt-pass rows for a context of max_len: kPrefillRows where both cards hold it at that width
// (PERF-36), else 2,048 (the context is kept, the chunk narrows); 0 when neither fits.
uint32_t pp_prefill_rows(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& layout, size_t capacity,
                         size_t reserve, uint32_t max_len, bool draft, bool kv8 = false, bool int8_pass = false);

// The int8 prompt pass's linears, (name under the layer's prefix, K, N), and one layer's rotated bytes: an int8
// [K / 4][Np] word array, then Np fp32 column scales, a linear after another; Np = N rounded up to the int8 GEMM's
// 256-column work-group (wr's 384), the columns past N zero. The sinks' projection (N 64, 24 live) stays bf16: at
// the int8 GEMM's 256 columns it was slower (4k 1.5 -> 3.6 ms).
struct Int8Linear {
  const char* name;
  uint32_t K, N;
};
// At a `--parallel tp` card's widths (its rows of q / gate / k / v / relative, its K of o_proj).
inline constexpr std::array<Int8Linear, 6> int8_prompt_linears(const Tp& tp = {}) {
  return {{{"attn.wq_du.weight", model::Inkling2Desc::hidden, tp.q_n()},
           {"attn.gate_attn.weight", model::Inkling2Desc::hidden, tp.q_n()},
           {"attn.wk_dv.weight", model::Inkling2Desc::hidden, tp.kv_n()},
           {"attn.wv_dv.weight", model::Inkling2Desc::hidden, tp.kv_n()},
           {"attn.wo_ud.weight", tp.q_n(), model::Inkling2Desc::hidden},
           {"attn.wr_du.weight", model::Inkling2Desc::hidden, tp.rel_n()}}};
}
inline constexpr uint32_t int8_columns(const Int8Linear& w) { return (w.N + 255) / 256 * 256; }
inline constexpr size_t int8_linear_bytes(const Int8Linear& w) {
  return size_t(w.K) * int8_columns(w) + size_t(int8_columns(w)) * 4;
}
inline constexpr size_t int8_layer_bytes(const Tp& tp = {}) {
  size_t b = 0;
  for (const auto& w : int8_prompt_linears(tp)) b += int8_linear_bytes(w);
  return b;
}

}  // namespace runtime::inkling2
