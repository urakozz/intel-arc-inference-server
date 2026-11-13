#include "runtime/capture.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/tp_kernels.h"
#include "loader/loader.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/gemv_kind.h"
#include "runtime/verify_attn.h"
#include "kernels/batch_kernels.h"
#include "runtime/batch_rows.h"

// capture.cc - the 774-kernel decode step, bound once (645 until spec 1.5's
// lever L1 split every `prep_res_norm` site into two launches).
//
// This is the file where every kernel of plan 3 meets every buffer of Task 1
// and every weight of plan 2, in the one order spec §9.1 fixes. It is a
// straight walk with no conditionals on runtime state: what varies per token
// lives in `runtime::Control` (shared memory, read at execution) and nowhere
// else, which is what makes a captured list replayable at all (doc 04,
// "Execution model").
//
// **Every binding site names the kernel source it is honouring.** The argument
// order of a `__kernel` is not checked by anything at compile time - a swapped
// pair is a silently wrong token 64 layers later - so the rule here is that a
// reader can put this file and the `.cl` side by side and check a launch in
// isolation. The grids and work-group sizes are the same: they come from the
// kernel's own contract, cited, never from arithmetic invented here.
//
// **The dataflow, in one paragraph.** `resid` (bf16 [M][5120]) is the residual
// stream and only the `prep_res_fold` / `prep_norm_finish` pair advances it:
// stage A folds the previous GEMV's split-K `partials` into it, stage B writes
// the normalised row to the shared bf16 scratch `x`. Every int4 GEMV reads `x`
// (or, for o_proj, `attn_out`) and writes `partials`; every prep reads
// `partials` and writes `x`. So a layer is
// a chain of alternating writes to two buffers, serialised by the list's
// in-order flag - no aliasing hazard needs an event, and no scratch value is
// read in a step before that step has written it. The persistent state
// (`gdn_state`, `conv_ring`, `kv_k`, `kv_v`, `control`) is what survives the
// token boundary, and each layer gets its own slice of it.
namespace runtime {
namespace {

using model::LinearId;
using model::Qwen35;

// The control block as the six ctrl-reading kernels index it - a flat `uint`
// array. `kernels::ctrl_index` is the host mirror of the CTRL_DEFINES line in
// src/kernels/CMakeLists.txt; `runtime::Control` is the struct. This unit binds
// `b.control` into embed_gather, gdn_step, attn_prep, attn_decode, attn_reduce
// and argmax_stage2, so it pins the two together one more time (argmax_test
// asserts the same pair from the test side). A mismatch here is a silently
// wrong token, not a crash.
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4, "CTRL_POS moved");
static_assert(offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4, "CTRL_NACT moved");
static_assert(offsetof(Control, cur_token) == 8 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4,
              "CTRL_CUR moved");
static_assert(offsetof(Control, out_token) == 40 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "CTRL_OUT moved");
static_assert(offsetof(Control, debug_flag) == kernels::ctrl_index::kDebugFlag * 4,
              "CTRL_DEBUG moved");
// Spec 8: CTRL_LIVE=19 in src/kernels/CMakeLists.txt's GDN_SLOT_DEFINES.
static_assert(offsetof(Control, gdn_live) == kernels::ctrl_index::kGdnLive * 4, "CTRL_LIVE moved");
// Open-work 13: CTRL_DSTAT=28 in src/kernels/CMakeLists.txt's draft_prob lines.
static_assert(offsetof(Control, draft_stat) == kernels::ctrl_index::kDraftStat * 4, "CTRL_DSTAT moved");
// Spec 15c: the MoE route row - buffer_sizes.h sizes it, moe.cl (RW) writes it.
static_assert(kMoeRouteWords == kernels::moe_route::kWords, "the MoE route row's width moved");
// Spec 13a / 13c: the row table as the BATCH_ROWS builds index it (BROW_DEFINES, src/kernels/CMakeLists.txt) - the
// host mirror kernels::batch_row against runtime::BatchRow.
static_assert(kBatchRowWords == kernels::batch_row::kWords, "BROW_WORDS moved");
static_assert(offsetof(BatchRow, pos) == kernels::batch_row::kPos * 4, "BROW_POS moved");
static_assert(offsetof(BatchRow, kv_base) == kernels::batch_row::kKvBase * 4, "BROW_KVB moved");
static_assert(offsetof(BatchRow, src) == kernels::batch_row::kSrc * 4, "BROW_SRC moved");
static_assert(offsetof(BatchRow, dst) == kernels::batch_row::kDst * 4, "BROW_DST moved");
static_assert(offsetof(BatchRow, slot) == kernels::batch_row::kSlot * 4, "BROW_SLOT moved");

// The captured list's M is `build`'s `M` argument (default 1, what ships). The
// default build compiles M = 1 only (spec §9: the M loop exists in every
// kernel, M = 1 is what ships), while `DecodeBuffers::kM` = 8 is the
// *allocation* capacity. Every kernel indexes its buffers with its own compiled
// M, so the argument and the variant names below move together. M = 2..4 at
// max_len 16384 exist under the CMake option B70_MTP (on by default); any other M names a binary that does not exist and throws at
// capture. The caller sets `Control::n_active` = M and `cur_token[0..M)`.

constexpr size_t kBf16 = 2, kFp32 = 4;

// --- work-group sizes, one per kernel, from each kernel's own contract -------
constexpr uint32_t kWgEmbed = 256;    // embed_gather.cl WG_EMBED  (Task 3)
constexpr uint32_t kWgGemv = 64;      // gemv.cl WG_N = 16x4 (plan 1 §9.2); gemv_bf16.cl's
                                      // WG_N is its COLS_PER_WG, per shape (see gemv_bf16 below)
// prep.cl's WG_RES (the single-work-group `prep_res_norm`) has no constant
// here any more: spec 1.5's lever L1 replaced that binding with the pair below,
// and the kernel is compiled and tested but never captured.
constexpr uint32_t kWgResFold = 256;  // prep.cl WG_FOLD           (spec 1.5 L1)
constexpr uint32_t kWgNormFinish = 256;  // prep.cl WG_NORM        (spec 1.5 L1)
constexpr uint32_t kWgSilu = 256;     // prep.cl WG_SILU           (Task 2)
constexpr uint32_t kWgGated = 128;    // prep.cl WG_GATED          (Task 2)
constexpr uint32_t kWgGdn = 256;      // gdn_step.cl WG_GDN        (Task 4)
constexpr uint32_t kWgAttn = 256;     // attn.cl WG_PREP/WG_DEC/WG_RED (Task 5)
constexpr uint32_t kWgArgmax = 256;   // argmax.cl WG_ARGMAX       (Task 3)

// --- grid divisors that are a kernel's constant, not a model dimension ------
// gemv: a work-group is 4 subgroups of 16 lanes, one column each. (gemv_bf16's
// grid is per shape and comes from kernels::gemv_bf16_tiling - see the binding
// below.)
constexpr uint32_t kGemvColsPerWg = 64;
// prep_silu_mul: SILU_CHUNK outputs per work-group (prep.cl).
constexpr uint32_t kSiluChunk = 4096;
// argmax_stage1: CHUNK logits per work-group (argmax.cl).
constexpr uint32_t kArgmaxChunk = 1024;
// gdn_step: CHUNK_V = 32 state columns per work-group, so 128/32 chunks of a
// head (gdn_step.cl, "Grid and the tile mapping").
constexpr uint32_t kGdnStateChunks = Qwen35::kGdnHeadDim / 32;

// --- per-layer slices of the persistent state -------------------------------
// The strides that runtime/buffers.cc sized these allocations with are
// re-derived per walk (Capture::gdn_state_stride / conv_ring_stride, kv_stride_)
// from the same descriptor rather than shared, and `check_sizes()` below asserts
// stride x layers == allocation size - so a divergence is a throw at capture,
// not a wrong KV slot at token 300. The layer COUNTS are the model descriptor's
// (spec 14: 48 / 16 on Qwen3.8, 30 / 10 on Ornith); so are the widths (spec 15b).

uint8_t* at(const l0::Mem& m, size_t byte_off) {
  return static_cast<uint8_t*>(m.ptr()) + byte_off;
}

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::build: " + what);
}

// The walk itself. One instance per `build` call; it owns the CapturedStep
// under construction and hands it back closed.
// What a walk captures (spec 8, plan 8b). `Plain` is the decode list, unchanged.
// `Verify` is the same list at M rows with the per-row GDN slot variant, followed by
// the MTP head's KV fill of those M rows. `Draft` is the MTP head alone at M = 1.
// `Stage` (spec 16b) is the plain list restricted to one pipeline stage's layers, with the
// hand-off appended or prepended (capture.h, build_stage). Spec 16d: `StageVerify` is the
// verify list restricted the same way (the head's KV fill on the last stage), `StageDraft`
// the draft list on the last stage's device (capture.h, build_stage_verify / _draft).
// Spec 19c: `VerifyTaps` is a tap drafter's verify list (capture.h, build_verify_taps) - the decode
// list at M rows with the per-row GDN slots over the drafter's slots, and the taps copied after the
// input fold of every tapped layer's successor; no MTP head anywhere. Spec 19d: `StageVerifyTaps` is that list
// restricted to one pipeline stage (capture.h, build_stage_verify_taps).
// Spec 13a / 13c: `Rows` is build_rows' list (capture.h) - the plain walk with the per-sequence sites bound to their
// row builds, and the MTP head's KV fill over the rows when it has the head.
// Spec 13c: `DraftRows` is build_draft_rows' list - the draft walk at M rows of different sequences.
// Spec 17b: `Tp` is build_tp's list (capture.h) - the plain walk at a TP card's shapes with the crossings.
// Spec 17d: `TpVerify` is build_tp_verify's - the TP walk at M rows with the per-row GDN slots, then the MTP head's KV
// fill at the head's own (one-card) shapes; `TpDraft` is build_tp_draft's - the head at M = 1 whose lm_head is the
// card's vocabulary half and whose argmax is tp.cl's (the host merges the two cards' winners).
enum class Mode { Plain, Verify, Draft, Stage, StageVerify, StageDraft, VerifyTaps, StageVerifyTaps, Rows, DraftRows, Tp,
                  TpVerify, TpDraft };

class Capture {
 public:
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, l0::Mem* tap,
          ProfileEvents* prof, uint32_t cap_m, const MtpBuffers* mtp = nullptr,
          Mode mode = Mode::Plain, uint32_t draft_i = 0)
      : ctx_(ctx), m_(m), b_(b), tap_(tap), prof_(prof), kCapM(cap_m), mtp_(mtp), mode_(mode),
        draft_i_(draft_i), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}
  // Spec 16b: one pipeline stage's walk. `b` and `m` hold only the stage's layers.
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const PpStage& st,
          const StageLink& link)
      : Capture(ctx, m, b, nullptr, nullptr, 1, nullptr, Mode::Stage) {
    stage_ = &st;
    link_ = &link;
    small_base_ = st.first;
    n_layers_ = st.layers();
    n_gdn_ = st.gdn;
    n_fa_ = st.fa;
  }
  // Spec 16d: one stage's MTP verify list at M rows (`link` set, `gdn_spec` the stage's
  // verify slots, `mtp` the head's buffers on the last stage only) or the last stage's
  // draft list `draft_i` (no link; the head's buffers).
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const PpStage& st,
          const StageLink* link, uint32_t cap_m, const MtpBuffers* mtp, const l0::Mem* gdn_spec,
          Mode mode, uint32_t draft_i)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, mtp, mode, draft_i) {
    stage_ = &st;
    link_ = link;
    small_base_ = st.first;
    n_layers_ = st.layers();
    n_gdn_ = st.gdn;
    n_fa_ = st.fa;
    gdn_spec_ = gdn_spec;
  }

  // Spec 19c: a tap drafter's verify list at M rows.
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const TapVerify& tv,
          uint32_t cap_m)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, nullptr, Mode::VerifyTaps) {
    tv_ = &tv;
  }
  // Spec 19d: one pipeline stage's tap verify list at M rows (`tv.gdn_spec` the stage's slots).
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const PpStage& st,
          const StageLink& link, const TapVerify& tv, uint32_t cap_m)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, nullptr, Mode::StageVerifyTaps) {
    stage_ = &st;
    link_ = &link;
    small_base_ = st.first;
    n_layers_ = st.layers();
    n_gdn_ = st.gdn;
    n_fa_ = st.fa;
    tv_ = &tv;
  }
  // Spec 13a / 13c: build_rows' list at M rows (the head's buffers from the binding, or none). Its Control and head
  // Control are the binding's: the slot binding below, with every state pointer slot 0's (index 0, ring 0).
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const BatchRowsBind& br, uint32_t cap_m)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, br.mtp, Mode::Rows) {
    br_ = &br;
    sb_.control = br.control;
    sb_.hctl = br.hctl;
  }
  // Spec 17b: one TP card's decode list (M = 1).
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, l0::Mem* tap, const TpLink& link)
      : Capture(ctx, m, b, tap, nullptr, 1, nullptr, Mode::Tp) {
    tp_ = &link;
  }
  // Spec 17d: one TP card's verify list at M rows (TpVerify) or its draft list `draft_i` (TpDraft, M = 1).
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
          const TpLink& link, uint32_t cap_m, Mode mode, uint32_t draft_i)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, &mtp, mode, draft_i) {
    tp_ = &link;
  }
  // Spec 13c: build_draft_rows' list at M rows.
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
          const DraftRowsBind& dr, uint32_t cap_m)
      : Capture(ctx, m, b, nullptr, nullptr, cap_m, &mtp, Mode::DraftRows) {
    dr_ = &dr;
    sb_.hctl = dr.hctl;
  }

  // Spec 28b: the test / probe override of runtime::gemv_kind (runtime/gemv_kind.h).
  // Spec 28c: the test / probe override of the verify-attention rule (runtime/verify_attn.h).
  Capture& options(const CaptureOptions& o) {
    opt_ = o;
    if (mode_ == Mode::VerifyTaps && kCapM == o.profile_m) prof_ = o.profile;
    return *this;
  }
  // Spec 13c: the sequence slot the list binds (capture.h, SlotBind).
  Capture& slot(const SlotBind& sb) {
    sb_ = sb;
    return *this;
  }
  // Open-work 13: a draft list followed by its probability's two launches (draft_prob).
  Capture& prob(bool on) {
    prob_ = on;
    return *this;
  }

  CapturedStep run() {
    // One layer's K (or V) rows; at int8 (spec 12b) the scales follow every layer's rows
    // in the same allocation (KvLayout), and `kv8_` selects kv8.cl's binaries throughout.
    kv_stride_ = size_t(b_.max_len) * d_.fa_kv_heads * Qwen35::kFaHeadDim *
                 (kv8_ ? 1 : kBf16);
    check_sizes();
    if (mode_ == Mode::Stage || mode_ == Mode::StageVerify || mode_ == Mode::StageVerifyTaps) return stage_run();
    if (mode_ == Mode::VerifyTaps)
      check_taps();
    else if (mode_ == Mode::Rows)
      check_rows();
    else if (mode_ == Mode::Tp)
      check_tp();
    else if (mode_ == Mode::TpVerify || mode_ == Mode::TpDraft) {   // spec 17d
      check_tp();
      check_mtp();
      check_tp_head();
    } else if (mode_ != Mode::Plain)
      check_mtp();
    if (mode_ == Mode::StageDraft)
      require(stage_ != nullptr && stage_->has_head(d_),
              "a stage's draft list runs on the device that holds the head");
    if (mode_ == Mode::Draft || mode_ == Mode::StageDraft || mode_ == Mode::DraftRows || mode_ == Mode::TpDraft) {
      layer_ = head_layer();
      draft();
      if (prob_) draft_prob();
      // Spec 15e: 23 launches with a dense head, 24 with a MoE head (draft_launches). Spec 17d: the head's own
      // descriptor under TP (tp.cl's two argmax stages replace argmax.cl's one for one). Open-work 13: 2 more with
      // the probability.
      const size_t want = draft_launches(hd()) + (prob_ ? 2 : 0);
      require(step_.kernel_count == want, "the draft list has " + std::to_string(step_.kernel_count) +
                                              " launches, not the " + std::to_string(want) + " draft_launches() gives " +
                                              hd().name);
      require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
      require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
      step_.list.close();
      return std::move(step_);
    }
    // A ProfileEvents handed to a second build must describe that build, not
    // both. (The profiler's own test builds twice against one set of buffers.)
    //
    // **The hazard that creates, named so it is not rediscovered.** This clear
    // destroys the `l0::Event` objects, and `launch()` below has already copied
    // their handles into an EARLIER profiled list's commands - a captured list
    // holds handles, not the vector, and has no way to learn they are gone. So
    // one `ProfileEvents` may back exactly ONE live profiled list: reusing one
    // across two profiled builds leaves the first list signalling destroyed
    // handles on its next replay. Both callers that build twice against one set
    // of buffers (src/cli/b70_decode.cc's --profile and
    // tests/runtime/profile_capture_test.cc) build the PLAIN list first and the
    // profiled one second, so neither hits it; that ordering is the pattern to
    // copy, not an accident of how they were written.
    if (prof_) prof_->events.clear();

    // `layer_` is what the labels' "L<n>" prefix reads; the six launches
    // outside the layer loop belong to the token boundary and get "--".
    embed_gather();
    uint32_t gdn = 0, fa = 0;   // the two per-kind index maps: GDN 0..47, FA 0..15
    for (const model::LayerDesc& layer : d_.layer_descs()) {
      layer_ = static_cast<int>(layer.index);
      if (layer.kind == model::LayerKind::GDN)
        gdn_layer(layer.index, gdn++);
      else
        fa_layer(layer.index, fa++);
      tap(layer.index);
    }
    layer_ = kBoundary;
    head();
    if (mode_ == Mode::Verify || mode_ == Mode::TpVerify) {   // spec 17d: TP's tail is the same, at the head's shapes
      // The verify rows' post-final-norm hidden (`x`, what lm_head read) into hh rows
      // 1..M; row 0 is h_{pos-1}, left by the previous commit or prefill. Then the
      // head's KV fill over hh rows 0..M-1 (MtpBuffers).
      const size_t row = size_t(d_.hidden) * kBf16;
      step_.list.copy(hh() + row, b_.x.ptr(), row * kCapM);
      layer_ = head_layer();
      head_kv_fill(hh());
    }
    if (mode_ == Mode::Rows && mtp_) {
      // Spec 13c: the rows' hidden saved (each slot's commit takes its row's from there), then every row's head input
      // gathered - its slot's h_{pos-1} at a run's first row, the saved row before it otherwise (capture.h,
      // build_rows) - and the head's KV fill over them.
      step_.list.copy(br_->saved, b_.x.ptr(), size_t(d_.hidden) * kBf16 * kCapM);
      layer_ = head_layer();
      {
        // embed_gather(gctl, table, hin) - embed_gather.cl over the table of hidden rows, grid (1, M), WG 256.
        l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM, d_.hidden), "embed_gather", kWgEmbed);
        k.arg_ptr(0, br_->gctl);
        k.arg_ptr(1, br_->table);
        k.arg_ptr(2, br_->hin);
        launch(k, 1, kCapM);
      }
      head_kv_fill(br_->hin);
    }
    require(mode_ != Mode::Tp || step_.kernel_count == tp_launches(d_),
            "the TP list has " + std::to_string(step_.kernel_count) + " launches, not tp_launches()'s " +
                std::to_string(tp_launches(d_)));
    require(mode_ != Mode::TpVerify || step_.kernel_count == tp_verify_launches(d_),
            "the TP verify list has " + std::to_string(step_.kernel_count) + " launches, not tp_verify_launches()'s " +
                std::to_string(tp_verify_launches(d_)));
    require(mode_ != Mode::Rows || step_.kernel_count == rows_launches(d_, mtp_ != nullptr),
            "the row list has " + std::to_string(step_.kernel_count) + " launches, not rows_launches()'s " +
                std::to_string(rows_launches(d_, mtp_ != nullptr)));

    require(gdn == d_.gdn_layers && fa == d_.fa_layers,
            "layer kind counts are not the descriptor's " + std::to_string(d_.gdn_layers) +
                " GDN / " + std::to_string(d_.fa_layers) + " FA");
    // The plain list's length is a property of the model (decode_launches): 774 on
    // Qwen3.8, 526 on Ornith (spec 15c). A walk that grew or lost a launch throws here.
    require(mode_ != Mode::Plain || step_.kernel_count == decode_launches(d_),
            "the decode list has " + std::to_string(step_.kernel_count) + " launches, not the " +
                std::to_string(decode_launches(d_)) + " decode_launches() gives " + d_.name);
    // Spec 15e: the verify list is the decode list at M rows plus the head's KV fill
    // (verify_launches): Qwen3.8 784, Ornith 536, at every M.
    require(mode_ != Mode::Verify || step_.kernel_count == verify_launches(d_),
            "the verify list has " + std::to_string(step_.kernel_count) + " launches, not the " +
                std::to_string(verify_launches(d_)) + " verify_launches() gives " + d_.name);
    // Spec 19c: a tap drafter's verify list launches exactly the decode list's kernels at M rows
    // (the taps are copies, which no count includes).
    require(mode_ != Mode::VerifyTaps || step_.kernel_count == decode_launches(d_),
            "the tap verify list has " + std::to_string(step_.kernel_count) + " launches, not the " +
                std::to_string(decode_launches(d_)) + " decode_launches() gives " + d_.name);
    // Every Kernel this walk made was launched exactly once, so kernels[i] is
    // what launch i ran - which is the property that makes the vector a usable
    // record of the list rather than a lifetime bag.
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    // The same property for the two index-parallel records Task 3 reads:
    // labels[i] names launch i, and (profiled) events[i] times it.
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    require(!prof_ || prof_->events.size() == step_.kernel_count,
            "a launch got no profiling event");
    step_.list.close();
    return std::move(step_);
  }

 private:
  // Preconditions this file can check before a single command is appended.
  void check_sizes() {
    // The grid comes from b.max_len and the RoPE table from the loader's: one number, or the
    // KV cache is read at the wrong stride.
    require(m_.max_len == b_.max_len, "model max_len " + std::to_string(m_.max_len) +
                                          " != buffers max_len " + std::to_string(b_.max_len));
    // Spec 10: decode attention v2 bakes no MAXLEN (one binary per M, `_T<kAttnV2Blocks>`); spec 12b: the
    // int8 cache is read by its kv8 twin. (v1, which baked MAXLEN, was removed in the zero-config cleanup.)
    std::vector<std::string> attn_bins =
        kv8_ ? std::vector<std::string>{kernels::attn_v2_kv8_variant(kCapM, DecodeBuffers::kAttnV2Blocks, d_.fa_q_heads, d_.fa_kv_heads)}
             : std::vector<std::string>{kernels::attn_v2_variant(kCapM, DecodeBuffers::kAttnV2Blocks, d_.fa_q_heads, d_.fa_kv_heads)};
    // Spec 13a: a row list binds the `_ROWS` pair (attn_pair_rows), not the production one - which at M = 5..8 over
    // int8 KV nothing builds (B70_KV8's decode block stops at M = 4).
    if (mode_ == Mode::Rows)
      attn_bins = {kernels::attn_v2_rows_variant(kCapM, DecodeBuffers::kAttnV2Blocks, kv8_, d_.fa_q_heads, d_.fa_kv_heads)};
    // attn_part is sized by DecodeScratchDims::sizes: v2's fixed 32 blocks per row.
    require(b_.attn_part.size() >= DecodeScratchDims::sizes(b_.max_len, d_).attn_part,
            std::string("attn_part (") + std::to_string(b_.attn_part.size()) + " B) is smaller than decode attention "
                "v2's partials");
    // Spec 28c: an M >= 2 list of v2 binds attn_verify_dpas + the v2 binary's reduce where the measured rule
    // (runtime/verify_attn.h) says so; M = 1 and the row builds never switch.
    {
      const bool rows = mode_ == Mode::Rows;
      vattn_ = !rows
                   ? verify_attn_kind(kCapM, vattn_keys_, kv8_, d_.fa_q_heads, d_.fa_kv_heads, opt_.force_attn)
                   : VerifyAttn::V2;
      if (vattn_ == VerifyAttn::Dpas)
        attn_bins.push_back(kernels::attn_verify_dpas_variant(kCapM, DecodeBuffers::kAttnV2Blocks, kv8_,
                                                              d_.fa_q_heads, d_.fa_kv_heads));
    }
    for (const std::string& v : attn_bins)
      require(std::ifstream(kernels::path(v)).good(),
              "no decode attention is compiled: " + v + " is missing (" + kernels::path(v) + ")");

    // n_gdn_ / n_fa_ are the descriptor's counts, or (spec 16b) a stage's own.
    const std::string gdn_n = std::to_string(n_gdn_), fa_n = std::to_string(n_fa_);
    require(b_.gdn_state.size() == gdn_state_stride() * n_gdn_,
            "gdn_state is not " + gdn_n + " slices");
    require(b_.conv_ring.size() == conv_ring_stride() * n_gdn_,
            "conv_ring is not " + gdn_n + " slices");
    // The layout the buffers were allocated with must be this walk's: its rows stride, its
    // layer count, and (spec 12b) the scales after them at int8.
    const KvLayout& kl = b_.kv_lay;
    require(kl.max_len == b_.max_len && kl.kv_heads == d_.fa_kv_heads &&
                kl.head_dim == Qwen35::kFaHeadDim && kl.layers == n_fa_ &&
                kl.layer_rows() == kv_stride_,
            "the KV layout (" + std::string(kv_cache_name(kl.form)) + ", " +
                std::to_string(kl.layers) + " layers) is not this model's " + fa_n + " FA layers");
    require(b_.kv_k.size() == kl.bytes(), "kv_k is not " + fa_n + " slices");
    require(b_.kv_v.size() == kl.bytes(), "kv_v is not " + fa_n + " slices");
    if (tap_)
      require(tap_->size() >= size_t(d_.layers) * kCapM * d_.hidden * kBf16,
              "debug_resid is smaller than [layers][M][hidden] bf16");
    // The profiling precondition that IS knowable before the walk. The other
    // one - that the walk fits the pool - is not: `kernel_count` is 0 here and
    // only the walk itself produces it, so that bound is checked per launch in
    // `launch()`, which throws on the launch that would overrun the pool.
    if (prof_)
      require(prof_->pool.capacity() >= ProfileEvents::kProfileCapacity,
              "profile event pool capacity " + std::to_string(prof_->pool.capacity()) +
                  " is below ProfileEvents::kProfileCapacity " +
                  std::to_string(ProfileEvents::kProfileCapacity));

    // Four `.cl` constants mirror three of the model table's split-K counts,
    // and nothing else connects the two. A GEMV's `partials` are fp32
    // `[S][M][N]`; at S = 1 that collapses to `[M][N]` and a token's row base is
    // `m·N`, which is what `gdn_step`, `attn_prep` and `prep_gated_head` assume.
    // `prep_silu_mul` is the other side of the same coin: it *does* loop the
    // slices, and it loops exactly SILU_S = 8 of them. `attn_prep` likewise
    // folds QKV_S = 2 before applying the linear's bf16 rounding.
    //
    // Retuning a row in `model/qwen35.cc` changes which GEMV variant is bound
    // but not the baked constant, and the mismatch is silent - the consumer
    // reads slice 0 of N (or sums four slices of a two-slice tensor) and the
    // token is wrong 64 layers later with no error anywhere. The kernels
    // `#error` on their own constants; only this side can see the table. So the
    // pairing is asserted here, at capture, before a single command is
    // appended. If one of these fires, the fix is the kernel (loop `s`), not
    // the guard.
    require(d_.shape(LinearId::QkvZ).S == 1,
            "qkv||z is no longer S=1, but gdn_step.cl (QKVZ_S) and prep.cl "
            "(GATED_S) bake S=1 into their partials indexing");
    require(d_.shape(LinearId::GateUp).S == 8,
            "gate||up is no longer S=8, but prep.cl (SILU_S) bakes an 8-slice sum");
    require(d_.shape(LinearId::Qkv).S == 2,
            "qkv is no longer S=2, but attn.cl (QKV_S) bakes a 2-slice sum into every "
            "partials load");
    // Spec 15c: a MoE model's block - its weights (one MoeLayer per layer), its scratch
    // (runtime::moe_scratch_layout), and the attention it is built for. Spec 15e: the
    // verify and draft lists exist for it too (check_mtp holds the head to its form).
    if (d_.is_moe()) {
      require(m_.moe.size() == n_layers_,
              "the loaded model has " + std::to_string(m_.moe.size()) + " MoE layers, not " +
                  std::to_string(n_layers_));
      require(b_.moe != nullptr && b_.moe->size() >= moe_scratch_layout(d_).total,
              "the decode scratch has no MoE region of moe_scratch_layout()'s size");
    }
  }

  // Spec 8: what the MTP walks need of the head and its buffers.
  void check_mtp() {
    require(mtp_ != nullptr && m_.mtp != nullptr, "an MTP list needs the loaded head and MtpBuffers");
    // Spec 15e: the head's FFN in the descriptor's form - a dense head's two bf16 linears,
    // or a MoE head's layer (its route row is the MoE scratch's last layer slot).
    require(hd().mtp_head_moe() ? m_.mtp->moe != nullptr && !m_.mtp->gate_up
                                : m_.mtp->gate_up != nullptr && m_.mtp->down != nullptr && !m_.mtp->moe,
            std::string("the loaded MTP head's FFN is not ") +
                (hd().mtp_head_moe() ? "the MoE layer" : "the dense MLP") + " " + hd().name + " has");
    require(!hd().mtp_head_moe() || moe_scratch_layout(d_).layer_slots == d_.layers + 1,
            "the MoE scratch has no layer slot for the MTP head's MoE layer");
    require(mtp_->max_len == b_.max_len, "MtpBuffers max_len != buffers max_len");
    // Spec 19a: a buffer with more slots (probe_mtp_steps' M = 5..8 lists) takes up to
    // `slots` rows; every engine's buffers have kSlots.
    require(kCapM <= mtp_->slots, "an MTP verify list has at most MtpBuffers::slots rows");
    require(mtp_->slots == MtpBuffers::kSlots || mode_ == Mode::Verify,
            "only a single-card verify list binds a GDN slot count other than kSlots");
    require((mode_ != Mode::Draft && mode_ != Mode::StageDraft) ||
                (kCapM == 1 && draft_i_ < MtpBuffers::kMaxK),
            "a draft list is M = 1 with draft index < kMaxK");
    // Spec 13c: a row draft list - its table, Control, rows and logits; a dense head, the main cache's form.
    require(mode_ != Mode::DraftRows ||
                (dr_ != nullptr && dr_->hctl && dr_->head_rows && dr_->dh && dr_->logits && !d_.mtp_head_moe() &&
                 !m_.draft_vocab && stage_ == nullptr),
            "a row draft list needs its head Control, row table, hidden rows and logits rows (a dense head, no draft "
            "vocabulary, one card)");
    // Spec 13c: an engine with sequence slots holds every further sequence's kSlots states after them.
    require(mtp_->gdn_spec.size() ==
                gdn_state_stride() * d_.gdn_layers * ((mtp_->slots - 1) + size_t(mtp_->slots) * (mtp_->seqs - 1)),
            "gdn_spec is not (slots - 1) x " + std::to_string(d_.gdn_layers) + " slices (+ slots per further sequence)");
    require(mtp_->kv_lay.form == b_.kv_lay.form && mtp_->kv_lay.layers == 1 &&
                mtp_->kv_lay.layer_rows() == head_kv_stride() && mtp_->kv_k.size() == mtp_->kv_lay.bytes() &&
                mtp_->kv_v.size() == mtp_->kv_lay.bytes(),
            "the head's KV is not one [max_len][kv-heads][256] layer in the main cache's form");
    require(mtp_->hh.size() >= size_t(kCapM + 1) * d_.hidden * kBf16, "hh is too small");
    // Spec 8 §11: the buffers were sized for the model's draft vocabulary, or for none.
    const uint32_t nv = m_.draft_vocab ? m_.draft_vocab->size() : 0;
    require(mtp_->draft_vocab == nv && bool(mtp_->dv_logits) == (nv != 0) &&
                (!mtp_->dv_logits || mtp_->dv_logits->size() >= size_t(nv) * kFp32),
            "MtpBuffers' draft vocabulary (" + std::to_string(mtp_->draft_vocab) +
                ") is not the loaded model's (" + std::to_string(nv) + ")");
  }

  // One launch site: load (or reuse) the variant's device binary, make a fresh
  // Kernel for it and set its group size. A Kernel per site is not required -
  // arguments are captured at append, so one object could serve many launches -
  // but it keeps the list auditable (kernels[i] is what launch i ran) for the
  // cost of 774 handles.
  // Decode attention over one KV cache: attn_v2.cl's pair (spec 10, plan 10b; kv8.cl's twin
  // over an int8 cache, spec 12b). `ctrl` is the list's
  // control block (the head's `hctl` in the draft list); both read q from attn_q, write
  // partials into attn_part and the gated bf16 rows into attn_out.
  void attn_pair(void* ctrl, const KvLayer& kv) { attn_pair(ctrl, kv, d_); }
  // `ad`: the descriptor whose heads the cache holds - the walk's, or (spec 17d) the MTP head's own under TP.
  void attn_pair(void* ctrl, const KvLayer& kv, const model::ModelDesc& ad) {
    // Spec 13a: the main layers' pair over the row table (a row list's head runs none); spec 13c: a row draft list's
    // head over its own table.
    if (mode_ == Mode::Rows || mode_ == Mode::DraftRows) {
      attn_pair_rows(ctrl, kv);
      return;
    }
    require(&ad == &d_ || vattn_ == VerifyAttn::V2, "the head's attention runs v2 (its lists are M = 1)");
    void* kk = kv.k;
    void* vv = kv.v;
    if (kv8_) {
      // Spec 12b: kv8.cl's attn_decode_v2_kv8 / attn_reduce_v2_kv8 - v2's grid, partials and
      // stride, reading int8 K / V and their fp16 scales; the reduce un-rotates each head's
      // output before the gate.
      const std::string v = kernels::attn_v2_kv8_variant(kCapM, DecodeBuffers::kAttnV2Blocks, ad.fa_q_heads, ad.fa_kv_heads);
      {
        // Spec 28c: attn_verify_dpas_kv8 takes attn_decode_v2_kv8's arguments and writes its partials.
        l0::Kernel& k = vattn_ == VerifyAttn::Dpas ? dpas_kernel() : kernel(v, "attn_decode_v2_kv8", kWgAttn);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.attn_q.ptr());
        k.arg_ptr(2, kv.k);
        k.arg_ptr(3, kv.ks);
        k.arg_ptr(4, kv.v);
        k.arg_ptr(5, kv.vs);
        k.arg_ptr(6, b_.attn_part.ptr());
        launch(k, ad.fa_kv_heads, DecodeBuffers::kAttnV2Blocks);
      }
      {
        l0::Kernel& k = kernel(v, "attn_reduce_v2_kv8", kWgAttn);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.attn_part.ptr());
        k.arg_ptr(2, b_.attn_gate.ptr());
        k.arg_ptr(3, b_.attn_out.ptr());
        launch(k, ad.fa_q_heads, kCapM);
      }
      return;
    }
    // attn_v2.cl: grid (4 kv-heads, kAttnV2Blocks), WG 256; the stride is derived per
    // row from Control::pos on the device, so one list serves every depth.
    const std::string v = kernels::attn_v2_variant(kCapM, DecodeBuffers::kAttnV2Blocks, ad.fa_q_heads, ad.fa_kv_heads);
    {
      // Spec 28c: attn_verify_dpas takes attn_decode_v2's arguments and writes its partials.
      l0::Kernel& k = vattn_ == VerifyAttn::Dpas ? dpas_kernel() : kernel(v, "attn_decode_v2", kWgAttn);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kk);
      k.arg_ptr(3, vv);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, ad.fa_kv_heads, DecodeBuffers::kAttnV2Blocks);
    }
    // attn_reduce_v2(ctrl, attn_part, attn_gate, attn_out), grid (24 q-heads, M).
    {
      l0::Kernel& k = kernel(v, "attn_reduce_v2", kWgAttn);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, ad.fa_q_heads, kCapM);
    }
  }

  // --- spec 13a: the row builds (build_rows, capture.h) ---------------------------------------------------------
  //
  // Every site below is its production twin's binding with the `_ROWS` binary (kernels/batch_kernels.h) and the row
  // table as the last argument; gdn_step also takes the two pools (the layer's slice of each, at the whole-model
  // strides the binary bakes - its SPEC_SLOT_STRIDE / RING_SLOT_STRIDE).

  // What a row list needs, before a command is appended (the pools' sizes are the caller's: the table names the
  // indices, and runtime::check_rows / the engine hold them to the slots it has).
  void check_rows() {
    require(br_ != nullptr && br_->rows != nullptr && br_->control != nullptr,
            "a row list needs its row table (runtime::BatchRowTable) and its Control");
    require(stage_ == nullptr && tap_ == nullptr && prof_ == nullptr,
            "a row list is single-card, with no residual tap and no profiling events");
    if (!mtp_) return;
    // The head's KV fill: its buffers in the main cache's form and length, its table and Control, the gather.
    require(m_.mtp != nullptr, "a row list with the head needs the loaded MTP head");
    require(mtp_->max_len == b_.max_len && mtp_->kv_lay.form == b_.kv_lay.form && mtp_->kv_lay.layers == 1 &&
                mtp_->kv_lay.layer_rows() == kv_stride_,
            "the head's KV is not one layer of the main cache's form and length");
    require(br_->head_rows != nullptr && br_->hctl != nullptr && br_->gctl != nullptr && br_->table != nullptr &&
                br_->saved != nullptr && br_->hin != nullptr,
            "a row list with the head needs the head's row table and Control, and the gather's Control, table, saved "
            "rows and input rows");
  }

  // gdn_step's row build: (ctrl, partials, ab_out, gdn_small, conv_ring, state, gdn_o, state_pool, ring_pool, rows).
  void gdn_step_rows(uint32_t layer, uint32_t g) {
    l0::Kernel& k = kernel(kernels::gdn_step_rows_variant(kCapM, d_.gdn_layers, d_.gdn_k_heads, d_.gdn_v_heads),
                           "gdn_step", kWgGdn);
    k.arg_ptr(0, ctrl());
    k.arg_ptr(1, b_.partials.ptr());
    k.arg_ptr(2, b_.ab_out.ptr());
    k.arg_ptr(3, small_of(layer).gdn.ptr());
    k.arg_ptr(4, ring0() + size_t(g) * conv_ring_stride());
    k.arg_ptr(5, state0() + size_t(g) * gdn_state_stride());
    k.arg_ptr(6, b_.gdn_o.ptr());
    // A pool that is not there is never indexed (the table names index 0 / slot 0 only): bind slot 0's slice.
    k.arg_ptr(7, (br_->state_pool ? static_cast<const uint8_t*>(br_->state_pool) : state0()) +
                     size_t(g) * gdn_state_stride());
    k.arg_ptr(8, (br_->ring_pool ? static_cast<const uint8_t*>(br_->ring_pool) : ring0()) +
                     size_t(g) * conv_ring_stride());
    k.arg_ptr(9, br_->rows);
    launch(k, d_.gdn_v_heads, kGdnStateChunks);
  }

  // attn_prep's (or attn_prep_kv8's) row build over one KV layer; `s1` the MTP head's QKV_S 1 build.
  void attn_prep_rows(void* ctrl, const void* fa_small, const KvLayer& kv, bool s1, void* rows) {
    l0::Kernel& k = kernel(kernels::attn_prep_rows_variant(kCapM, s1, kv8_, d_.fa_q_heads, d_.fa_kv_heads),
                           kv8_ ? "attn_prep_kv8" : "attn_prep", kWgAttn);
    k.arg_ptr(0, ctrl);
    k.arg_ptr(1, b_.partials.ptr());
    k.arg_ptr(2, fa_small);
    k.arg_ptr(3, m_.rope.ptr());
    k.arg_ptr(4, b_.attn_q.ptr());
    k.arg_ptr(5, b_.attn_gate.ptr());
    k.arg_ptr(6, kv.k);
    k.arg_ptr(7, kv.v);
    uint32_t i = 8;
    if (kv8_) {
      k.arg_ptr(i++, kv.ks);
      k.arg_ptr(i++, kv.vs);
    }
    k.arg_ptr(i, rows);
    launch(k, d_.fa_q_heads + d_.fa_kv_heads, kCapM);
  }

  // Decode attention v2's row build (bf16 or its kv8 twin): the layer's whole cache, each group at its slot's base.
  void attn_pair_rows(void* ctrl, const KvLayer& kv) {
    const std::string v =
        kernels::attn_v2_rows_variant(kCapM, DecodeBuffers::kAttnV2Blocks, kv8_, d_.fa_q_heads, d_.fa_kv_heads);
    {
      l0::Kernel& k = kernel(v, kv8_ ? "attn_decode_v2_kv8" : "attn_decode_v2", kWgAttn);
      uint32_t i = 0;
      k.arg_ptr(i++, ctrl);
      k.arg_ptr(i++, b_.attn_q.ptr());
      k.arg_ptr(i++, kv.k);
      if (kv8_) k.arg_ptr(i++, kv.ks);
      k.arg_ptr(i++, kv.v);
      if (kv8_) k.arg_ptr(i++, kv.vs);
      k.arg_ptr(i++, b_.attn_part.ptr());
      k.arg_ptr(i, rows_table());
      launch(k, d_.fa_kv_heads, DecodeBuffers::kAttnV2Blocks);
    }
    {
      l0::Kernel& k = kernel(v, kv8_ ? "attn_reduce_v2_kv8" : "attn_reduce_v2", kWgAttn);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, b_.attn_out.ptr());
      k.arg_ptr(4, rows_table());
      launch(k, d_.fa_q_heads, kCapM);
    }
  }

  // Spec 28c: the DPAS verify attention's decode kernel. Its work-group is the binary's own
  // reqd_work_group_size (the CMake lever line decides it: 16 x row blocks x dim halves x key slices).
  l0::Kernel& dpas_kernel() {
    l0::Kernel& k = kernel(kernels::attn_verify_dpas_variant(kCapM, DecodeBuffers::kAttnV2Blocks, kv8_,
                                                             d_.fa_q_heads, d_.fa_kv_heads),
                           kv8_ ? "attn_verify_dpas_kv8" : "attn_verify_dpas", 0);
    ze_kernel_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    require(zeKernelGetProperties(k.handle(), &p) == ZE_RESULT_SUCCESS && p.requiredGroupSizeX >= 16,
            "attn_verify_dpas: no reqd_work_group_size in the binary");
    k.group_size(p.requiredGroupSizeX);
    return k;
  }

  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules
               .emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant)))
               .first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    l0::Kernel& k = *step_.kernels.back();
    if (wg != 0) k.group_size(wg);   // 0 (spec 28c's dpas_kernel): the caller sets the binary's reqd size
    // The two halves of the launch's label. Held here rather than passed to
    // `launch()` because this is where they are already known and it keeps
    // every binding site's `launch(k, grid…)` line unchanged. `launch()`
    // consumes both and clears **`pending_entry_` only** - that one field is
    // the whole interlock, and its emptiness is what the `require` at the top
    // of `launch()` tests, so an unpaired kernel()/launch() throws instead of
    // silently repeating the previous site's name. `pending_variant_` keeps its
    // last value on purpose-of-omission rather than by design: it is never read
    // without `pending_entry_` having just been set beside it.
    pending_entry_ = entry;
    pending_variant_ = variant;
    return k;
  }

  // THE launch site. Every binding function above funnels through here, so the
  // three index-parallel records - the list's commands, `labels`, and (when
  // profiling) `events` - are appended together or not at all. Tap copies do
  // not come through here: they are commands, not kernels, and get neither a
  // label nor an event, exactly as they are excluded from `kernel_count`.
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    require(!pending_entry_.empty(), "launch() without a preceding kernel()");
    l0::Event* signal = nullptr;
    if (prof_) {
      // Checked before the event is made, so the throw names the launch that
      // would have overrun the pool rather than coming out of zeEventCreate.
      require(step_.kernel_count < ProfileEvents::kProfileCapacity,
              "the decode walk has more than " +
                  std::to_string(ProfileEvents::kProfileCapacity) +
                  " launches: raise ProfileEvents::kProfileCapacity");
      prof_->events.emplace_back(prof_->pool, static_cast<uint32_t>(step_.kernel_count));
      // The vector reserved kProfileCapacity, so this reference stays valid -
      // and it would not matter if it did not: `launch` copies the handle into
      // the command list here and never looks at the Event object again.
      signal = &prof_->events.back();
    }
    step_.list.launch(k, gx, gy, 1, signal);
    step_.labels.push_back((layer_ == kBoundary ? std::string("--")
                                                : "L" + std::to_string(layer_)) +
                           " " + pending_entry_ + " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  // The per-layer residual tap (capture.h). A device-to-device copy on the
  // regular list - `l0::CmdList::copy` is `zeCommandListAppendMemoryCopy`, an
  // append on either list flavour - so it costs a command, not a kernel.
  void tap(uint32_t layer) {
    if (!tap_) return;
    const size_t bytes = size_t(kCapM) * d_.hidden * kBf16;
    step_.list.copy(at(*tap_, size_t(layer) * bytes), b_.resid.ptr(), bytes);
  }

  // CaptureOptions::attn_dump (tests / probes only): FA layer f's attention inputs and output, copied after its pair.
  void attn_dump(uint32_t f) {
    if (opt_.attn_dump == nullptr) return;
    require(size_t(d_.fa_q_heads) * Qwen35::kFaHeadDim == kAttnDumpRow && kCapM <= 8,
            "CaptureOptions::attn_dump: Qwen3.8's 24 x 256 q heads at M <= 8 only");
    uint8_t* base = static_cast<uint8_t*>(opt_.attn_dump) + size_t(f) * kAttnDumpLayerBytes;
    const size_t f32 = size_t(kCapM) * kAttnDumpRow * 4;
    step_.list.copy(base, b_.attn_q.ptr(), f32);
    step_.list.copy(base + 8 * kAttnDumpRow * 4, b_.attn_gate.ptr(), f32);
    step_.list.copy(base + 8 * kAttnDumpRow * 8, b_.attn_out.ptr(), size_t(kCapM) * kAttnDumpRow * 2);
  }

  // --- the kernels ----------------------------------------------------------

  // embed_gather(ctrl, embed, resid) - src/kernels/embed_gather.cl (Task 3),
  // grid (1, M), WG 256. Reads the id from `ctrl` at execution time, which is
  // the whole reason the list can be captured once.
  void embed_gather() {
    l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM, d_.hidden), "embed_gather", kWgEmbed);
    k.arg_ptr(0, ctrl());
    k.arg_ptr(1, m_.embed.ptr());
    k.arg_ptr(2, b_.resid.ptr());
    launch(k, 1, kCapM);
  }

  // The residual add + RMSNorm site - **TWO launches since spec 1.5's lever
  // L1**, src/kernels/prep.cl:
  //
  //   prep_res_fold(partials, resid, sumsq)        grid (kNormGroups, M), WG 256
  //   prep_norm_finish(sumsq, resid, norm_w, x)    grid (kNormGroups, M), WG 256
  //
  // `s_prev` is the split-K width of the GEMV whose partials stage A folds into
  // the residual stream; the SP0 variant folds nothing and does not read
  // `partials` at all (layer 0, whose residual stream is embed_gather's
  // output). `norm_w` is fp32 `1 + w` (loader/small_layout.h - the RMSNorm
  // family is fp32 on device) and only stage B reads it.
  //
  // **This is the one binding site in the file that appends two commands**, so
  // it is the one that moved the launch count: 129 sites x 2 = 258 launches
  // where there were 129, and the walk is 645 + 129 = **774** (the constant
  // pinned by tests/runtime/replay_determinism_test.cc and
  // tests/runtime/profile_capture_test.cc, and the count docs/04 and docs/12
  // quote). `ProfileEvents::kProfileCapacity` is 1024 and did not have to move.
  //
  // Why two: RMSNorm's mean is over the whole row, so ONE kernel's reduction
  // domain is one work-group - 16 subgroups on one Xe-core, measured at 22.3 µs
  // and 17.0 GB/s in situ. The split makes it 320 subgroups on both passes at
  // the cost of a launch (0.73 µs derived, docs/15).
  //
  // The per-element fold chain is unchanged, so `resid` out of one launch is
  // bit-identical for the same inputs; the global Σx² tree is NOT, so `x` can
  // move in the last bf16 ulp and everything downstream of it moves with it.
  // The golden gate's per-layer tap diagnostics were **measured to move in both
  // directions** across this lever while the token ids stayed 96/96 exact
  // (docs/14). That is why this lever's acceptance ran the gate before as well
  // as after - the gate is the arbiter, not the bit-identity of `resid`
  // (prep.cl states the tree, docs/12 "Rounding discipline" the contract).
  //
  // `b_.norm_sumsq` is a single 640-byte allocation shared by all 129 sites:
  // the list is in-order, so a site's stage B has consumed its own stage A's
  // sums before the next site's stage A overwrites them.
  void res_norm(uint32_t s_prev, const void* norm_w) {
    const uint32_t g = DecodeBuffers::kNormGroups;
    // Spec 16b: the first site of a later pipeline stage resumes after its fold, which the
    // previous device ran (cut_fold) and handed off with the residual. Never set on the
    // single-card walks.
    if (resume_) {
      resume_ = false;
    } else if (tp_main() && s_prev != 0) {
      tp_fold(s_prev);   // spec 17b: both cards' parts of the row-parallel linear, after the peer's arrives
    } else {
      l0::Kernel& k =
          kernel(kernels::prep_res_fold_variant(kCapM, d_.hidden, s_prev, g),
                 "prep_res_fold", kWgResFold);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.resid.ptr());
      k.arg_ptr(2, b_.norm_sumsq.ptr());
      launch(k, g, kCapM);
    }
    {
      l0::Kernel& k =
          kernel(kernels::prep_norm_finish_variant(kCapM, d_.hidden, g, g),
                 "prep_norm_finish", kWgNormFinish);
      k.arg_ptr(0, b_.norm_sumsq.ptr());
      k.arg_ptr(1, b_.resid.ptr());
      k.arg_ptr(2, norm_w);
      k.arg_ptr(3, b_.x.ptr());
      launch(k, g, kCapM);
    }
  }

  // gemv(w, scales, x, out) - src/kernels/gemv.cl (plan 1 §9.2), grid (N/64, S),
  // WG 64. Layout 0 reads GPTQ-native qweight and f16 scales from two distinct
  // allocations owned by DeviceWeight. Layout 1 carries scales inline and does
  // not dereference the scales argument, so it retains the historical binding
  // of w.mem there. The requires below make either ownership mistake loud.
  //
  // **The shape and the kind come from the LOADED weight, not from the model
  // table** (spec 1.6 §5.1). For seven of the eight rows the two are the same
  // object - `loader::load_linear` copies the table row's `GemvShape` and
  // `WeightKind` into the `DeviceWeight` it returns. The eighth is `lm_head`,
  // whose kind is a property of the checkpoint (model/qwen35.h): the loader
  // classifies it by content and stores the row it actually packed, so reading
  // the shape from the device weight is what makes one capture serve both
  // checkpoints without a conditional anywhere in the walk.
  //
  // `out` is `partials` for every per-layer GEMV - its consumer is a `prep`
  // that folds the S slices. `lm_head` passes `logits` instead; see `head()`.
  // Returns the split-K slices it wrote (gemv's S; 1 for gemm_sm's 2x8 work-group on gate||up - spec 28b).
  uint32_t gemv(uint32_t layer, LinearId id, const void* x, const l0::Mem& out) {
    const loader::DeviceWeight& w = m_.linears.at({layer, id});
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4, "gemv bound to a bf16 weight");
    require(s.N % kGemvColsPerWg == 0, "gemv N is not a multiple of 64");
    require((s.layout == 0) == bool(w.scales),
            s.layout == 0 ? "layout-0 GEMV has no independent scales allocation"
                          : "layout-1 GEMV unexpectedly owns a separate scales allocation");
    // `gemv.cl` writes `out[(s*M + m)*N + n]`, so the output allocation must
    // hold S*M*N floats. Nothing else checks this: `partials` is sized as the
    // max-S x max-N rectangle over the table's int4 rows (buffers.cc), which
    // stops covering a GEMV the moment one is bound to a different allocation.
    require(out.size() >= size_t(s.S) * kCapM * s.N * sizeof(float),
            "the GEMV's output allocation is smaller than its [S][M][N] fp32 result");
    // Spec 28b: at M >= 2 (runtime::gemv_kind) a tuned shape binds gemm_sm (gemm_sm.cl) - gemv's four
    // arguments in gemv's order, gemv's [S][M][N] output at the shape's S (the consumer's fold unchanged),
    // except gate||up under the 2x8 work-group: one slice, and mlp() binds the SiLU's _S1 build. Only the
    // variant, the entry, the work-group and the grid differ.
    const kernels::SmallMTuning* t = small_m(s.K, s.N, s.S, s.layout);
    if (t)
      require(kernels::gemm_sm_out_s(*t) == s.S || id == LinearId::GateUp,
              "gemm_sm writes another split-K S than the consumer folds (only gate||up's SiLU has an S 1 build)");
    l0::Kernel& k = t ? kernel(kernels::gemm_sm_variant(kCapM, *t), kernels::gemm_sm_entry(*t),
                               kernels::gemm_sm_wg_size(*t))
                      : kernel(kernels::gemv_variant(kCapM, s.K, s.N, s.S, s.layout), "gemv", kWgGemv);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, s.layout == 0 ? w.scales->ptr() : w.mem.ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, out.ptr());
    if (t)
      launch(k, s.N / kernels::gemm_sm_cols(*t), kernels::gemm_sm_grid_y(*t));
    else
      launch(k, s.N / kGemvColsPerWg, s.S);
    if (tp_main() && layer != loader::kTopLevel && loader::tp_row_parallel(id))   // spec 17b
      tp_send(t ? kernels::gemm_sm_out_s(*t) : s.S);
    return t ? kernels::gemm_sm_out_s(*t) : s.S;
  }

  // --- spec 17b: the TP crossings (capture.h, build_tp; src/kernels/tp.cl) ----------------------------------------
  // tp_send(partials, peer_land, send_ctr, own) after a row-parallel linear of `s` slices, grid (kNormGroups, M).
  void tp_send(uint32_t s) {
    require(s == tp_s_, "a row-parallel linear writes " + std::to_string(s) + " slices, the crossing sends " +
                            std::to_string(tp_s_));
    l0::Kernel& k = kernel(tp_cross(), "tp_send", kWgResFold);
    k.arg_ptr(0, b_.partials.ptr());
    k.arg_ptr(1, tp_->peer_land);
    k.arg_ptr(2, tp_->send_ctr);
    k.arg_ptr(3, tp_->folded ? tp_->own : b_.partials.ptr());
    launch(k, DecodeBuffers::kNormGroups, kCapM);
  }
  // tp_res_fold(mine, land, resid, sumsq, fold_ctr, state, spin_limit) in place of prep_res_fold.
  void tp_fold(uint32_t s_prev) {
    require(s_prev == tp_s_, "a TP fold of " + std::to_string(s_prev) + " slices, the crossing's are " +
                                 std::to_string(tp_s_));
    l0::Kernel& k = kernel(tp_cross(), "tp_res_fold", kWgResFold);
    k.arg_ptr(0, tp_->folded ? tp_->own : b_.partials.ptr());
    k.arg_ptr(1, tp_->land);
    k.arg_ptr(2, b_.resid.ptr());
    k.arg_ptr(3, b_.norm_sumsq.ptr());
    k.arg_ptr(4, tp_->fold_ctr);
    k.arg_ptr(5, tp_->state);
    k.arg(6, tp_->spin_limit);
    launch(k, DecodeBuffers::kNormGroups, kCapM);
  }
  std::string tp_cross() const {
    return kernels::tp_cross_variant(kCapM, d_.hidden, DecodeBuffers::kNormGroups, tp_s_, tp_->folded, tp_->rank);
  }
  void check_tp() {
    require(tp_ != nullptr && tp_->peer_land && tp_->land && tp_->send_ctr && tp_->fold_ctr && tp_->state &&
                (!tp_->folded || tp_->own) && tp_->spin_limit != 0 && tp_->rank < model::kTpSize,
            "a TP list needs both landing buffers, the counters, the state words, the spin bound (and the folded "
            "row for the folded form)");
    require(prof_ == nullptr && (mode_ == Mode::TpVerify ? kCapM <= MtpBuffers::kSlots : kCapM == 1),
            "a TP list is M = 1 (a verify list M <= 4) with no profiling events (spec 17b / 17d)");
    tp_s_ = d_.shape(LinearId::Down).S;
    require(d_.shape(LinearId::OutProj).S == tp_s_ && d_.shape(LinearId::OProj).S == tp_s_,
            "the row-parallel linears of a TP card share one S");
    require(d_.ffn_fold_s() == tp_s_, "the FFN's fold is not the crossing's");
  }
  // Spec 17d: the MTP head runs at its own (one-card) shapes in a TP card's decode scratch, which is sized for kM rows
  // at the card's - so it holds the head's rows at M <= 4 (its 24 q heads are two cards' 12) and its draft's
  // intermediate at M = 1. Checked, not assumed; the head's lm_head is the card's int8 vocabulary half.
  void check_tp_head() {
    const model::ModelDesc& h = hd();
    require(tp_->head != nullptr && h.hidden == d_.hidden && !h.mtp_head_moe() && !m_.draft_vocab,
            "a TP card's head lists need the head's descriptor at the card's hidden (a dense head, no draft "
            "vocabulary)");
    const size_t M = kCapM, qv = size_t(h.fa_value_dim());
    require(b_.attn_q.size() >= M * qv * kFp32 && b_.attn_gate.size() >= M * qv * kFp32 &&
                b_.attn_out.size() >= M * qv * kBf16 &&
                b_.attn_part.size() >= size_t(h.fa_q_heads) * DecodeBuffers::kAttnV2Blocks * M * 258 * kFp32 &&
                b_.x.size() >= M * 2 * h.hidden * kBf16 &&
                b_.partials.size() >= M * size_t(m_.mtp->qkv.shape.N) * kFp32,
            "the card's decode scratch does not hold the MTP head's " + std::to_string(kCapM) + " rows");
    if (mode_ == Mode::TpDraft)
      require(b_.x.size() >= size_t(m_.mtp->down->shape.K) * kBf16 &&
                  b_.partials.size() >= size_t(m_.mtp->gate_up->shape.N) * kFp32,
              "the card's decode scratch does not hold the MTP head's intermediate");
    require(m_.linears.at({loader::kTopLevel, LinearId::LmHead}).kind == model::WeightKind::Int8,
            "a TP card's draft head is the int8 vocabulary half (loader::load_tp, LmHeadForm::Int8)");
  }
  // The MTP head's descriptor: the model's own, or under TP (spec 17d) the one-card descriptor the head was loaded
  // at (TpLink::head) - the card's is half the heads.
  const model::ModelDesc& hd() const { return tp_ && tp_->head ? *tp_->head : d_; }
  size_t head_kv_stride() const {
    return size_t(b_.max_len) * hd().fa_kv_heads * Qwen35::kFaHeadDim * (kv8_ ? 1 : kBf16);
  }
  bool tp_main() const { return mode_ == Mode::Tp || mode_ == Mode::TpVerify; }
  uint32_t gemv(uint32_t layer, LinearId id, const void* x) { return gemv(layer, id, x, b_.partials); }

  // Spec 28b: the gemm_sm row this list binds for a shape, or null (gemv / gemv_i8w): the rule
  // runtime::linear_kind(M, options) - Gemv at M = 1 unless forced - and the shape's kSmallMTuning row.
  const kernels::SmallMTuning* small_m(uint32_t K, uint32_t N, uint32_t S, uint32_t layout) const {
    return linear_kind(kCapM, opt_) == kernels::LinearKind::SmallM ? kernels::small_m_tuning(K, N, S, layout)
                                                                    : nullptr;
  }

  // gemv_bf16(w, x, out) - src/kernels/gemv_bf16.cl (plan 1 §9.2), grid
  // (N/COLS_PER_WG), WG COLS_PER_WG. No split-K: it writes its fp32 output
  // directly, not into `partials`.
  //
  // **The tiling is the kernel's, and it is per shape.** Unlike every other
  // binding in this file the grid divisor is not a constant: it comes from
  // `kernels::gemv_bf16_tiling(N)`, the same function
  // tests/kernels/kernel_table_test.cc uses to name the binary that must exist.
  // `lm_head` (N = 248320) takes `{64, 1}` and is untouched - 3880 work-groups,
  // 98.5% of measured bandwidth in situ (docs/15), same binary as before the
  // knob existed. `a‖b` (N = 128) takes `{16, 16}`, spec 1.5's lever L2: 8
  // work-groups of 16 K-slice subgroups, 128 hardware threads against 8. The
  // launch COUNT is unchanged either way - one launch per site, so the
  // 774-kernel walk and every ripple rule that counts it are unaffected - but
  // the split DOES reorder the summation (gemv_bf16.cl names the tree), which
  // is why this lever's acceptance runs the golden gate.
  void gemv_bf16(uint32_t layer, LinearId id, const void* x, const l0::Mem& out) {
    const loader::DeviceWeight& w = m_.linears.at({layer, id});
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Bf16, "gemv_bf16 bound to an int4 weight");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(s.N);
    require(s.N % t.cols == 0, "gemv_bf16 N is not a multiple of its work-group width");
    require(out.size() >= size_t(kCapM) * s.N * sizeof(float),
            "the bf16 GEMV's output allocation is smaller than its [M][N] fp32 result");
    // The work-group is one lane per column per K slice - the kernel's
    // reqd_work_group_size(COLS_PER_WG / 16 * KSPLIT subgroups of 16).
    l0::Kernel& k =
        kernel(kernels::gemv_bf16_variant(kCapM, s.K, s.N, t), "gemv_bf16", t.cols * t.ksplit);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, x);
    k.arg_ptr(2, out.ptr());
    launch(k, s.N / t.cols);
  }

  // gemv_i8w(w, scales, x, out) - src/kernels/gemv_i8w.cl (spec 9 §3), grid N/64, WG 64.
  // The int8 `lm_head` only: int8 rows tiled [N/16][K/16][16][16] in `w.mem` and one
  // fp32 scale per row in `w.scales`. Writes the fp32 [M][N] result straight at `out`.
  void gemv_i8w(const loader::DeviceWeight& w, const void* x, const void* out) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int8, "gemv_i8w bound to a non-int8 weight");
    require(bool(w.scales), "the int8 lm_head has no fp32 row-scale allocation");
    require(s.N % kernels::kGemvI8wCols == 0, "gemv_i8w N is not a multiple of 64");
    // Spec 28b: at M >= 2 a tuned shape binds gemm_sm_i8w - gemv_i8w's arguments and [M][N] output.
    const kernels::SmallMTuning* t = small_m(s.K, s.N, 1, 2);
    l0::Kernel& k = t ? kernel(kernels::gemm_sm_variant(kCapM, *t), kernels::gemm_sm_entry(*t),
                               kernels::gemm_sm_wg_size(*t))
                      : kernel(kernels::gemv_i8w_variant(kCapM, s.K, s.N), "gemv_i8w", kernels::kGemvI8wCols);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, w.scales->ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, out);
    if (t)
      launch(k, s.N / kernels::gemm_sm_cols(*t), kernels::gemm_sm_grid_y(*t));
    else
      launch(k, s.N / kernels::kGemvI8wCols);
  }

  // The MLP half, identical in both layer kinds: post-norm folding the mixer's
  // partials, gate‖up, SiLU·mul, down. `mixer_s` is the split-K width of the
  // GEMV that produced those partials (out_proj / o_proj, both S = 4).
  void mlp(uint32_t layer, uint32_t mixer_s) {
    res_norm(mixer_s, at(small_of(layer).norms, sl_.norms_off_post));
    const uint32_t gu_s = gemv(layer, LinearId::GateUp, b_.x.ptr());
    // prep_silu_mul(partials, x_out) - prep.cl (Task 2), grid (I/4096, M) = (5, M)
    // at I = 17408 (ragged last chunk 1024), WG 256.
    // Spec 28b: SILU_S is baked - gate||up's S, or the _S1 build after gemm_sm's 2x8 work-group.
    {
      l0::Kernel& k = kernel(kernels::gemm_sm_silu_variant(kCapM, d_.intermediate, gu_s,
                                                           d_.shape(LinearId::GateUp).S),
                             "prep_silu_mul", kWgSilu);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.x.ptr());
      launch(k, (d_.intermediate + kSiluChunk - 1) / kSiluChunk, kCapM);
    }
    gemv(layer, LinearId::Down, b_.x.ptr());
  }

  // The layer's feed-forward half: the dense MLP, or (spec 15c) the MoE block.
  void ffn(uint32_t layer, uint32_t mixer_s) {
    if (d_.is_moe())
      moe(layer, mixer_s);
    else
      mlp(layer, mixer_s);
  }

  // Spec 15c: the mixture-of-experts block (spec 15 §4.1-4.2, §9) - the post norm
  // pair and FOUR launches, the expert ids produced and consumed on the device:
  //
  //   gemv_bf16(router, x, logits_l)                gemv_bf16.cl, router || shared gate,
  //                                                 N = router_n (272), {16, 16}
  //   moe_route(logits_l, route_l)                  moe.cl, grid (1, M), WG experts
  //   moe_gate_up(route_l, x, gate_up, h)           moe.cl, grid (slots x 2I/64, M), WG 256
  //   moe_down(route_l, h, down, resid)             moe.cl, grid (hidden / 16, M), WG 288
  //
  // `logits_l` / `route_l` are this layer's slices of the MoE scratch
  // (moe_scratch_layout), so every layer's routing survives the step for R2's gate;
  // `h` is one region every layer reuses (in-order list). moe_down folds the block's
  // output into `resid` itself, so the next fold is SP0 (ModelDesc::ffn_fold_s). The
  // shared expert is slot top_k, its weights the last block of `gate_up` / `down`.
  void moe(uint32_t layer, uint32_t mixer_s) {
    res_norm(mixer_s, at(small_of(layer).norms, sl_.norms_off_post));
    moe_block(m_.moe.at(layer - small_base_), layer);
  }

  // The block's four launches on `x` (normed) into `resid`, for the weights `w` and the
  // MoE scratch's layer slot `slot` - a main layer's index, or (spec 15e) `layers` for
  // the MTP head's MoE layer, which runs exactly these binaries (moe_scratch_layout's
  // last slot; its route row survives the draft for the M1 diagnostics).
  void moe_block(const loader::MoeLayer& w, uint32_t slot) {
    const model::MoeDesc& md = d_.moe;
    const MoeScratchLayout ml = moe_scratch_layout(d_);
    require(slot < ml.layer_slots, "the MoE scratch has no layer slot " + std::to_string(slot));
    void* lg = at(*b_.moe, ml.logits_at(slot));
    void* rt = at(*b_.moe, ml.route_at(slot));
    void* h = at(*b_.moe, ml.h_off);
    require(w.router.shape.N == md.router_n() && w.router.shape.K == d_.hidden,
            "the MoE router weight is not [router_n][hidden]");
    head_gemv(w.router, b_.x.ptr(), lg);   // gemv_bf16; the [M][router_n] fp32 logits
    const std::string v =
        kernels::moe_variant(kCapM, md.experts, md.top_k, d_.hidden, md.expert_intermediate);
    {
      l0::Kernel& k = kernel(v, "moe_route", md.experts);
      k.arg_ptr(0, lg);
      k.arg_ptr(1, rt);
      launch(k, 1, kCapM);
    }
    {
      l0::Kernel& k = kernel(v, "moe_gate_up", kernels::moe_gate_up_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, w.gate_up.ptr());
      k.arg_ptr(3, h);
      launch(k, kernels::moe_gate_up_groups(md.top_k, md.expert_intermediate), kCapM);
    }
    {
      l0::Kernel& k = kernel(v, "moe_down", kernels::moe_down_wg(md.top_k));
      k.arg_ptr(0, rt);
      k.arg_ptr(1, h);
      k.arg_ptr(2, w.down.ptr());
      k.arg_ptr(3, b_.resid.ptr());
      launch(k, d_.hidden / 16, kCapM);
    }
  }

  // A gated-delta-net layer (48 of 64 on Qwen3.8): 10 kernels.
  void gdn_layer(uint32_t layer, uint32_t g) {
    // Layer 0 leads the whole step, so there are no previous partials to fold
    // and `resid` is exactly embed_gather's output: the SP0 variant. Every
    // other layer folds the previous layer's `down` (S = 4).
    res_norm(layer == 0 ? 0u : d_.ffn_fold_s(),
             at(small_of(layer).norms, sl_.norms_off_input));
    tap_copy(layer);   // spec 19c: layer - 1's output, when a drafter taps it
    gemv(layer, LinearId::QkvZ, b_.x.ptr());
    // a||b in the checkpoint's form (spec 15 §13, ModelDesc::ab): bf16 through gemv_bf16
    // ({16, 16}), or - when the checkpoint quantised in_proj_a / in_proj_b, as the published
    // Ornith int4 export does - int4 g64 through gemv.cl at S 1. Either writes fp32 [M][128]
    // straight at ab_out (gemv.cl's out[(s*M + m)*N + n] at s = 0), the layout gdn_step
    // reads at AB_STRIDE; one launch either way, so every launch count is unchanged.
    if (m_.linears.at({layer, LinearId::AB}).kind == model::WeightKind::Int4) {
      require(m_.linears.at({layer, LinearId::AB}).shape.S == 1,
              "an int4 a||b must be an S = 1 GEMV: gdn_step reads ab_out, not partials");
      gemv(layer, LinearId::AB, b_.x.ptr(), b_.ab_out);
    } else {
      gemv_bf16(layer, LinearId::AB, b_.x.ptr(), b_.ab_out);
    }
    // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o)
    // - src/kernels/gdn_step.cl (Task 4), grid (48 v-heads, 4 state-column
    // chunks), WG 256. `conv_ring` and `state` are this GDN layer's slices;
    // `gdn_small` is the layer's GDN block base as a `const float*` (the
    // kernel finds conv/negA/dt_bias inside it at loader/small_layout.h's
    // offsets, passed to ocloc as NEGA_OFF/DTBIAS_OFF).
    if (mode_ == Mode::Rows) gdn_step_rows(layer, g); else   // spec 13a: the BATCH_ROWS build
    {
      // Spec 8: the verify list binds the SPEC_SLOTS build, whose eighth argument is
      // this layer's slice of slot 1 (MtpBuffers::gdn_spec is slot-major, so slot s
      // is that plus (s - 1) whole slots - gdn_step.cl, SPEC_SLOTS). Spec 16d: a stage's
      // verify list binds its own slots' allocation (spec_mem(), pp_gdn_spec_bytes) at its
      // own slice g - the same binary, the same whole-model slot stride.
      const bool slots = mode_ == Mode::Verify || mode_ == Mode::StageVerify || mode_ == Mode::VerifyTaps ||
                         mode_ == Mode::StageVerifyTaps || mode_ == Mode::TpVerify;   // spec 17d: TP's too
      // Spec 19a: the slot count is the buffers' (kSlots on every engine: the historical
      // names); a stage's verify list keeps kSlots (its spec_mem() is pp_gdn_spec_bytes').
      // Spec 19c: a tap drafter's verify list binds its own TapVerifyDims::kSlots (8).
      const uint32_t n_slots = mode_ == Mode::Verify || mode_ == Mode::TpVerify               ? mtp_->slots
                               : mode_ == Mode::VerifyTaps || mode_ == Mode::StageVerifyTaps ? tv_->slots
                                                                                              : MtpBuffers::kSlots;
      l0::Kernel& k = kernel(slots ? kernels::gdn_step_slots_variant(kCapM, d_.gdn_layers, d_.gdn_k_heads,
                                                                     d_.gdn_v_heads, n_slots)
                                   : kernels::gdn_step_variant(kCapM, d_.gdn_k_heads, d_.gdn_v_heads),
                             "gdn_step", kWgGdn);
      k.arg_ptr(0, ctrl());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.ab_out.ptr());
      k.arg_ptr(3, small_of(layer).gdn.ptr());
      k.arg_ptr(4, ring0() + size_t(g) * conv_ring_stride());
      k.arg_ptr(5, state0() + size_t(g) * gdn_state_stride());
      k.arg_ptr(6, b_.gdn_o.ptr());
      if (slots) k.arg_ptr(7, spec0() + size_t(g) * gdn_state_stride());
      launch(k, d_.gdn_v_heads, kGdnStateChunks);
    }
    // prep_gated_head(qkvz_partials, gdn_o, gated_w, x_out) - prep.cl (Task 2),
    // grid (48 v-heads, M), WG 128. `gated_w` is the GDN block's RMSNormGated
    // weight: plain `w`, and bf16 - the kernel takes it as `const ushort*`, so
    // this is the one small-tensor binding that is not fp32.
    {
      l0::Kernel& k = kernel(kernels::prep_gated_head_variant(kCapM, d_.gdn_k_heads, d_.gdn_v_heads), "prep_gated_head", kWgGated);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.gdn_o.ptr());
      k.arg_ptr(2, at(small_of(layer).gdn, sl_.gdn_off_gated_norm));
      k.arg_ptr(3, b_.x.ptr());
      launch(k, d_.gdn_v_heads, kCapM);
    }
    gemv(layer, LinearId::OutProj, b_.x.ptr());
    ffn(layer, d_.shape(LinearId::OutProj).S);
  }

  // A full-attention layer (16 of 64 on Qwen3.8): 10 kernels.
  void fa_layer(uint32_t layer, uint32_t f) {
    res_norm(d_.ffn_fold_s(), at(small_of(layer).norms, sl_.norms_off_input));
    tap_copy(layer);   // spec 19c: layer - 1's output, when a drafter taps it
    gemv(layer, LinearId::Qkv, b_.x.ptr());
    // This FA layer's KV cache slices - bf16 [max_len][4][256] each, indexed by
    // absolute position inside the kernels; at int8 (spec 12b) the int8 rows and their
    // fp16 scales [max_len][4] (KvLayout).
    // Spec 13c: at the bound slot's KV base (0: the allocation's first position).
    const KvLayer kv = b_.kv_lay.at_base(b_.kv_lay.layer(b_.kv_k.ptr(), b_.kv_v.ptr(), f), sb_.kv_base);
    void* kk = kv.k;
    void* vv = kv.v;
    // attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v)
    // - src/kernels/attn.cl (Task 5), grid (24 q-heads + 4 kv-heads, M), WG 256.
    // `fa_small` is the layer's FA block (q_norm ‖ k_norm, fp32 1+w) - the same
    // `SmallTensors::gdn` allocation the GDN layers use for their own block.
    if (mode_ == Mode::Rows)   // spec 13a: the row build over the main rows
      attn_prep_rows(ctrl(), small_of(layer).gdn.ptr(), kv, false, br_->rows);
    else if (kv8_) {
      // Spec 12b: kv8.cl's attn_prep_kv8 - attn_prep's chain, then q rotated (attn_q stays
      // fp32) and K, V rotated and quantised into the int8 rows + scales. Two more args.
      l0::Kernel& k = kernel(kernels::attn_prep_kv8_variant(kCapM, d_.fa_q_heads, d_.fa_kv_heads), "attn_prep_kv8", kWgAttn);
      k.arg_ptr(0, ctrl());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, small_of(layer).gdn.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kv.k);
      k.arg_ptr(7, kv.v);
      k.arg_ptr(8, kv.ks);
      k.arg_ptr(9, kv.vs);
      launch(k, d_.fa_q_heads + d_.fa_kv_heads, kCapM);
    } else {
      l0::Kernel& k = kernel(kernels::attn_prep_variant(kCapM, d_.fa_q_heads, d_.fa_kv_heads), "attn_prep", kWgAttn);
      k.arg_ptr(0, ctrl());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, small_of(layer).gdn.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kk);
      k.arg_ptr(7, vv);
      launch(k, d_.fa_q_heads + d_.fa_kv_heads, kCapM);
    }
    // Decode attention (attn_pair): v2's fixed grid of (4 kv-heads, kAttnV2Blocks) work-groups, the
    // stride derived per row from Control::pos on the device.
    attn_pair(ctrl(), kv);
    attn_dump(f);
    // o_proj is the one GEMV whose activations are not the shared `x` scratch:
    // attn_reduce writes bf16 [M][6144] into `attn_out`, which is exactly this
    // GEMV's K.
    gemv(layer, LinearId::OProj, b_.attn_out.ptr());
    ffn(layer, d_.shape(LinearId::OProj).S);
  }

  // The token boundary: fold layer 63's MLP into the residual stream under the
  // final norm, project to logits, sample.
  // **`lm_head` is the one launch in the walk whose kernel depends on the
  // checkpoint** (spec 1.6 §5.1). The published checkpoint ships it bf16 and it
  // is a `gemv_bf16`; a checkpoint quantised with `--quant_lm_head` ships it
  // int4 g64 and it is a `gemv`. The loader has already decided which
  // (`Qwen35::lm_head`, by content), so this reads the decision off the loaded
  // weight rather than re-deriving it.
  //
  // **Both write straight into `logits`, and the int4 path needs no kernel
  // change to do it.** `gemv_bf16` has always written its fp32 output
  // directly. `gemv` writes `out[(s*M + m)*N + n]` - at `S = 1` that is exactly
  // `[M][N]`, the layout `argmax_stage1` reads, so binding `logits` as the
  // `out` argument makes the split-K accumulator's degenerate case *be* the
  // logits row. No `partials` slice is involved, no rebinding of `argmax`, no
  // direct-out variant of `gemv.cl`: the S = 1 that the shape needs for
  // occupancy reasons is the same S = 1 that makes this legal.
  //
  // **The sizes, exactly.** At the shipped `kCapM = 1` the launch
  // writes `S*kCapM*N*4` = 1*1*248320*4 = **993 280 B**. `logits` is allocated
  // for `DecodeBuffers::kM = 8` tokens in flight - 8*248320*4 = **7 946 240 B**
  // - so there is **8x headroom, not an exact fit**. The `require` in `gemv()`
  // checks the figure the launch actually writes against the allocation, which
  // is a real bound (it would catch a genuinely undersized output) and not a
  // tight one. When M > 1 is turned on the two converge: at `kCapM = kM` the
  // launch fills the allocation exactly, and `logits`' [M][N] layout is
  // already what `out[(s*M + m)*N + n]` produces at S = 1 for every m.
  //
  // The launch COUNT is identical either way (one launch, 774 total). The
  // MODULE count is too, though not trivially: `gemv_bf16_M1_K5120_N248320`
  // stops being opened and `gemv_M1_K5120_N248320_S1_L1` starts, one for one,
  // so `CapturedStep::modules.size()` is 19 on both checkpoints.
  void head() {
    res_norm(d_.ffn_fold_s(), m_.final_norm.ptr());
    tap_copy(d_.layers);   // spec 19c: the last layer's output, when a drafter taps it
    // Spec 9: or an int8 head quantised at load (`--lm-head int8`), a `gemv_i8w`.
    const loader::DeviceWeight& lm = m_.linears.at({loader::kTopLevel, LinearId::LmHead});
    if (lm.kind == model::WeightKind::Int4) {
      gemv(loader::kTopLevel, LinearId::LmHead, b_.x.ptr(), b_.logits);
    } else if (lm.kind == model::WeightKind::Int8) {
      require(b_.logits.size() >= size_t(kCapM) * lm.shape.N * sizeof(float),
              "logits is smaller than the int8 head's [M][N] fp32 result");
      gemv_i8w(lm, b_.x.ptr(), b_.logits.ptr());
    } else {
      gemv_bf16(loader::kTopLevel, LinearId::LmHead, b_.x.ptr(), b_.logits);
    }
    // argmax_stage1(logits, part) - src/kernels/argmax.cl (Task 3), grid
    // (kVocab/1024 = 243, M), WG 256.
    if (tp_main()) {   // spec 17b: the card's vocabulary half, the global id and its value into Control (17d: M rows)
      const uint32_t rows = lm.shape.N;
      const std::string v = kernels::tp_argmax_variant(kCapM, rows, tp_->vocab_used, tp_->rank * rows);
      {
        l0::Kernel& k = kernel(v, "tp_argmax_stage1", kWgArgmax);
        k.arg_ptr(0, b_.logits.ptr());
        k.arg_ptr(1, b_.argmax_part.ptr());
        launch(k, (rows + kArgmaxChunk - 1) / kArgmaxChunk, kCapM);
      }
      {
        l0::Kernel& k = kernel(v, "tp_argmax_stage2", kWgArgmax);
        k.arg_ptr(0, ctrl());
        k.arg_ptr(1, b_.argmax_part.ptr());
        launch(k, 1, 1);
      }
      return;
    }
    {
      l0::Kernel& k = kernel(kernels::argmax_stage1_variant(kCapM, d_.vocab_used), "argmax_stage1",
                             kWgArgmax);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, (Qwen35::kVocab + kArgmaxChunk - 1) / kArgmaxChunk, kCapM);
    }
    // argmax_stage2(ctrl, part) - argmax.cl (Task 3), grid (1, 1), WG 256. The
    // only writer of `pos` and `cur_token` inside the list, and therefore the
    // last kernel of the step.
    {
      l0::Kernel& k = kernel(kernels::argmax_stage2_variant(), "argmax_stage2", kWgArgmax);
      k.arg_ptr(0, ctrl());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  // --- spec 16b: one pipeline stage --------------------------------------------
  //
  // capture.h (build_stage) has the contract. The walk is run()'s with three differences:
  // only layers [first, last) are walked, with stage-local GDN / FA slice indices (the
  // stage's buffers hold its own layers only); a stage without the embedding starts with the
  // hand-off in and resumes layer `first` after its fold; a stage without the head ends
  // with layer `last`'s fold and the hand-off out.
  CapturedStep stage_run() {
    const PpStage& st = *stage_;
    require(link_ != nullptr, "a stage walk needs its link");
    require(tap_ == nullptr && prof_ == nullptr,
            "a pipeline stage captures no residual tap and no profiling events");
    require(st.last <= d_.layers && st.first < st.last &&
                m_.layer_small.size() == st.layers(),
            "the stage model holds " + std::to_string(m_.layer_small.size()) +
                " layers' small tensors, not layers [" + std::to_string(st.first) + ", " +
                std::to_string(st.last) + ")");
    require(st.has_embed() || st.has_head(d_), "spec 16b builds two stages: one holds the "
                                               "embedding, the other the head");
    const bool verify = mode_ == Mode::StageVerify;
    if (verify) check_stage_verify();
    if (mode_ == Mode::StageVerifyTaps) check_stage_taps();   // spec 19d
    if (st.has_embed()) {
      embed_gather();
    } else {
      handoff_in();
      resume_ = true;   // layer `first`'s fold ran on the previous device
    }
    uint32_t gdn = 0, fa = 0;   // this stage's slices: its own GDN / FA layers from 0
    for (const model::LayerDesc& layer : d_.layer_descs()) {
      if (layer.index < st.first || layer.index >= st.last) continue;
      layer_ = static_cast<int>(layer.index);
      if (layer.kind == model::LayerKind::GDN)
        gdn_layer(layer.index, gdn++);
      else
        fa_layer(layer.index, fa++);
    }
    require(!resume_, "the stage's first layer did not consume the hand-off");
    require(gdn == st.gdn && fa == st.fa,
            "the stage's layer kind counts are not its " + std::to_string(st.gdn) + " GDN / " +
                std::to_string(st.fa) + " FA");
    layer_ = kBoundary;
    if (st.has_head(d_)) {
      head();
      if (verify) {
        // Spec 16d: build_verify's tail on the head's device - the M rows' post-final-norm
        // hidden into hh rows 1..M, then the head's KV fill over hh rows 0..M-1.
        const size_t row = size_t(d_.hidden) * kBf16;
        step_.list.copy(hh() + row, b_.x.ptr(), row * kCapM);
        layer_ = head_layer();
        head_kv_fill(hh());
      }
    } else {
      layer_ = static_cast<int>(st.last);
      cut_fold();
      layer_ = kBoundary;
      handoff_out();
    }
    // The compute launches are the single-card list's, cut in two (pp_stage_launches); the
    // peer hand-off adds its one kernel per side.
    const size_t want = (verify ? pp_stage_verify_launches(d_, st) : pp_stage_launches(d_, st)) +
                        (link_->mode == PpHandoff::Peer ? 1 : 0);
    require(step_.kernel_count == want,
            "the stage list has " + std::to_string(step_.kernel_count) + " launches, not " +
                std::to_string(want));
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

  // The fold half of layer `last`'s leading res_norm - the launch the single-card walk
  // makes there, with that site's split-K width (every layer after 0 folds the previous
  // FFN's partials: ffn_fold_s). It reads `partials` and `resid` and writes `resid` and
  // `norm_sumsq`, the two things the hand-off moves.
  void cut_fold() {
    const uint32_t g = DecodeBuffers::kNormGroups;
    l0::Kernel& k = kernel(kernels::prep_res_fold_variant(kCapM, d_.hidden, d_.ffn_fold_s(), g),
                           "prep_res_fold", kWgResFold);
    k.arg_ptr(0, b_.partials.ptr());
    k.arg_ptr(1, b_.resid.ptr());
    k.arg_ptr(2, b_.norm_sumsq.ptr());
    launch(k, g, kCapM);
  }

  // What crosses: the list's M residual rows and the whole norm_sumsq allocation (its
  // [groups][M] layout is the kernels'; copying all of it needs no knowledge of it).
  size_t handoff_resid_bytes() const { return size_t(kCapM) * d_.hidden * kBf16; }
  size_t handoff_sumsq_bytes() const { return b_.norm_sumsq.size(); }
  void check_link() const {
    require(link_->land_resid && link_->land_sumsq && link_->resid_cap >= handoff_resid_bytes() &&
                link_->sumsq_cap == handoff_sumsq_bytes(),
            "the landing buffer does not hold " + std::to_string(handoff_resid_bytes()) +
                " B of rows and exactly " + std::to_string(handoff_sumsq_bytes()) +
                " B of norm sums (pp_send stamps the word after them)");
    require(handoff_resid_bytes() % 4 == 0 && handoff_sumsq_bytes() % 4 == 0,
            "the hand-off moves whole 32-bit words");
    if (link_->mode == PpHandoff::Copy)
      require(link_->event != nullptr, "the copy hand-off needs its cross-device event");
  }

  void handoff_out() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.copy(link_->land_resid, b_.resid.ptr(), handoff_resid_bytes());
      step_.list.copy(link_->land_sumsq, b_.norm_sumsq.ptr(), handoff_sumsq_bytes());
      step_.list.barrier_signal(link_->event);
      return;
    }
    require(link_->flag && link_->send_seq, "the peer hand-off needs its flag and counter");
    // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words) -
    // pp_handoff.cl, grid (1), WG 256.
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_send", kernels::kPpHandoffWg);
    k.arg_ptr(0, b_.resid.ptr());
    k.arg_ptr(1, b_.norm_sumsq.ptr());
    k.arg_ptr(2, link_->land_resid);
    k.arg_ptr(3, link_->land_sumsq);
    k.arg_ptr(4, link_->flag);
    k.arg_ptr(5, link_->send_seq);
    k.arg<uint32_t>(6, static_cast<uint32_t>(handoff_resid_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(handoff_sumsq_bytes() / 4));
    launch(k, 1);
  }

  void handoff_in() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.wait_event(link_->event);
      step_.list.copy(b_.resid.ptr(), link_->land_resid, handoff_resid_bytes());
      step_.list.copy(b_.norm_sumsq.ptr(), link_->land_sumsq, handoff_sumsq_bytes());
      return;
    }
    require(link_->flag && link_->recv_state && link_->spin_limit > 0,
            "the peer hand-off needs its flag, its state words and a spin bound");
    // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words,
    // spin_limit) - pp_handoff.cl, grid (1), WG 256.
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_recv", kernels::kPpHandoffWg);
    k.arg_ptr(0, link_->land_resid);
    k.arg_ptr(1, link_->land_sumsq);
    k.arg_ptr(2, link_->flag);
    k.arg_ptr(3, link_->recv_state);
    k.arg_ptr(4, b_.resid.ptr());
    k.arg_ptr(5, b_.norm_sumsq.ptr());
    k.arg<uint32_t>(6, static_cast<uint32_t>(handoff_resid_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(handoff_sumsq_bytes() / 4));
    k.arg<uint32_t>(8, link_->spin_limit);
    launch(k, 1);
  }

  // Spec 16d: what a stage's verify list needs - its verify slots, sized for its own GDN
  // layers at the whole model's slot stride (pp_gdn_spec_bytes), the M rows in the slots,
  // and on the head's device everything build_verify needs of the head (check_mtp).
  void check_stage_verify() {
    require(gdn_spec_ != nullptr, "a stage's verify list needs its verify slots");
    require(kCapM <= MtpBuffers::kSlots, "an MTP verify list has at most kSlots rows");
    require(gdn_spec_->size() >= size_t(MtpBuffers::kSlots - 2) * gdn_state_stride() * d_.gdn_layers +
                                     size_t(n_gdn_) * gdn_state_stride(),
            "the stage's verify slots hold less than (kSlots - 2) whole slots and its " +
                std::to_string(n_gdn_) + " GDN layers of the last (pp_gdn_spec_bytes)");
    if (stage_->has_head(d_)) check_mtp();
  }
  const l0::Mem& spec_mem() const {
    if (tv_) return *tv_->gdn_spec;   // spec 19c: the drafter's verify slots
    return gdn_spec_ ? *gdn_spec_ : mtp_->gdn_spec;
  }
  // Spec 13c: the bound slot's persistent state (capture.h, SlotBind) - each the buffers' own when not bound.
  void* ctrl() const { return sb_.control ? sb_.control : b_.control.ptr(); }
  uint8_t* state0() const { return static_cast<uint8_t*>(sb_.gdn_state ? sb_.gdn_state : b_.gdn_state.ptr()); }
  uint8_t* ring0() const { return static_cast<uint8_t*>(sb_.conv_ring ? sb_.conv_ring : b_.conv_ring.ptr()); }
  uint8_t* spec0() const { return static_cast<uint8_t*>(sb_.gdn_spec && !tv_ && !gdn_spec_ ? sb_.gdn_spec : spec_mem().ptr()); }
  void* hctl() const { return sb_.hctl ? sb_.hctl : mtp_->hctl.ptr(); }
  // The row table decode attention's row build reads: a row list's main rows, a row draft list's head rows.
  void* rows_table() const { return mode_ == Mode::DraftRows ? dr_->head_rows : br_->rows; }
  uint8_t* hh() const { return static_cast<uint8_t*>(sb_.hh ? sb_.hh : mtp_->hh.ptr()); }

  // Spec 19c: what a tap drafter's verify list needs - its slots (one whole gdn_state each, slot 0
  // being gdn_state itself), the tap buffer at [kM][taps][hidden] and the tapped layers.
  void check_taps() {
    require(tv_ != nullptr && tv_->gdn_spec != nullptr && tv_->taps != nullptr,
            "a tap verify list needs its GDN slots and its tap buffer");
    require(stage_ == nullptr && tap_ == nullptr,
            "a tap verify list is single-card, with no residual tap");
    require(tv_->slots == TapVerifyDims::kSlots,
            "a tap verify list binds gdn_step_slots_M<M>_N" + std::to_string(TapVerifyDims::kSlots) +
                ": " + std::to_string(tv_->slots) + " slots are not built");
    require(kCapM <= tv_->slots, "a tap verify list has at most its slot count of rows");
    require(tv_->gdn_spec->size() == gdn_state_stride() * d_.gdn_layers * (tv_->slots - 1),
            "the tap verify slots are not (slots - 1) x " + std::to_string(d_.gdn_layers) +
                " GDN slices");
    sites_ = tap_sites(d_, tv_->layers);   // throws on an empty, unordered or out-of-range list
    require(tv_->taps->size() >= size_t(DecodeBuffers::kM) * tv_->layers.size() * d_.hidden * kBf16,
            "the tap buffer is smaller than [kM][taps][hidden] bf16");
  }

  // Spec 19c: after the input fold of layer `site` (or, at site == layers, the final norm's fold),
  // `resid` holds the output of layer site - 1; the taps of that layer go to every row's slot of the
  // tap buffer, one device copy per row (5 x M <= 40 on Qwen3.8's drafter; copies are commands, not
  // launches, exactly as the residual tap above).
  void tap_copy(uint32_t site) {
    if (mode_ != Mode::VerifyTaps && mode_ != Mode::StageVerifyTaps) return;
    const size_t row = size_t(d_.hidden) * kBf16, n = tv_->layers.size();
    for (uint32_t t : sites_.at(site))
      for (uint32_t m = 0; m < kCapM; ++m)
        step_.list.copy(at(*tv_->taps, (size_t(m) * n + t) * row), at(b_.resid, size_t(m) * row), row);
  }

  // Spec 19d: a stage's tap verify list - its slots (pp_gdn_spec_bytes(d, stage gdn, 8): the whole model's slot
  // stride, the stage's slice), the taps (every row's taps of THIS stage's sites, into the drafter's buffer on the
  // last device - a device-0 stage copies them across, ordered before its hand-off), the copy hand-off only (the
  // peer hand-off's flag does not order the copy engine's earlier writes into the other device).
  void check_stage_taps() {
    require(tv_ != nullptr && tv_->gdn_spec != nullptr && tv_->taps != nullptr,
            "a stage's tap verify list needs its GDN slots and the drafter's tap buffer");
    require(link_->mode == PpHandoff::Copy,
            "a stage's tap verify list runs on the copy hand-off only (--pipeline-handoff copy)");
    require(tv_->slots == TapVerifyDims::kSlots && kCapM <= tv_->slots,
            "a stage's tap verify list binds gdn_step_slots_M<M>_N" + std::to_string(TapVerifyDims::kSlots));
    require(tv_->gdn_spec->size() == pp_gdn_spec_bytes(d_, n_gdn_, tv_->slots),
            "the stage's tap verify slots are not pp_gdn_spec_bytes(" + std::to_string(n_gdn_) + " GDN layers, " +
                std::to_string(tv_->slots) + " slots)");
    sites_ = tap_sites(d_, tv_->layers);
    require(tv_->taps->size() >= size_t(DecodeBuffers::kM) * tv_->layers.size() * d_.hidden * kBf16,
            "the tap buffer is smaller than [kM][taps][hidden] bf16");
  }

  // A layer's small blocks: m_.layer_small is indexed by layer, or (spec 16b) by layer -
  // first in a stage model.
  const loader::SmallTensors& small_of(uint32_t layer) const {
    return m_.layer_small[layer - small_base_];
  }

  // --- spec 8: the MTP head (plan 8b) ---------------------------------------
  //
  // The head is one full-attention layer over fc(cat(pre_fc_norm_embedding(embed(x)),
  // pre_fc_norm_hidden(h))) (docs/probes/probe-mtp-2026-09-27.md §1). It runs on its OWN
  // control block `hctl` (MtpBuffers): embed_gather reads the input ids from
  // hctl.cur_token, the attention trio reads the head's position from hctl.pos, and in
  // the draft list argmax_stage2 writes the draft into hctl.cur_token and advances
  // hctl.pos - the chain, with every main-model kernel unchanged.
  //
  // Every head linear is bf16 and writes its [M][N] fp32 output straight into
  // `partials` (gemv_bf16 has no split-K), so every consumer is an S = 1 build: the
  // `_S1` attn_prep and prep_silu_mul, and the SP1 folds.

  // A fold + norm pair on explicit buffers (res_norm() is the main walk's, on b_).
  void fold_norm(const std::string& fold_v, const std::string& finish_v, const void* partials,
                 void* resid, const void* norm_w, void* x_out) {
    const uint32_t g = DecodeBuffers::kNormGroups;
    {
      l0::Kernel& k = kernel(fold_v, "prep_res_fold", kWgResFold);
      k.arg_ptr(0, partials);
      k.arg_ptr(1, resid);
      k.arg_ptr(2, b_.norm_sumsq.ptr());
      launch(k, g, kCapM);
    }
    {
      l0::Kernel& k = kernel(finish_v, "prep_norm_finish", kWgNormFinish);
      k.arg_ptr(0, b_.norm_sumsq.ptr());
      k.arg_ptr(1, resid);
      k.arg_ptr(2, norm_w);
      k.arg_ptr(3, x_out);
      launch(k, g, kCapM);
    }
  }

  void head_gemv(const loader::DeviceWeight& w, const void* x, void* out) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Bf16, "the MTP head's linears are bf16");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(s.N);
    l0::Kernel& k =
        kernel(kernels::gemv_bf16_variant(kCapM, s.K, s.N, t), "gemv_bf16", t.cols * t.ksplit);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, x);
    k.arg_ptr(2, out);
    launch(k, s.N / t.cols);
  }

  // The head from its input to its K/V at M rows: embed(hctl.cur_token[m]) and hidden
  // row m of `hidden` ([M][5120] bf16) -> the two pre-fc norms into one [M][10240] row
  // of `x` -> fc -> the residual stream (ZERO_RESID fold) -> input_layernorm -> q||k||v
  // -> attn_prep, which writes K/V at hctl.pos + m of the head's own cache. Leaves the
  // queries in attn_q / attn_gate and the head's residual in `resid`.
  void head_front(const void* hidden) {
    const loader::MtpHead& h = *m_.mtp;
    const model::ModelDesc& hd_ = hd();   // spec 17d: the head's own shapes under TP
    const uint32_t G = DecodeBuffers::kNormGroups, H = hd_.hidden, X = 2 * H;
    {
      l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM, hd_.hidden), "embed_gather", kWgEmbed);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, m_.embed.ptr());
      k.arg_ptr(2, b_.resid.ptr());
      launch(k, 1, kCapM);
    }
    const std::string fold0 = kernels::prep_res_fold_variant(kCapM, H, 0, G);
    const std::string cat = kernels::prep_norm_finish_strided_variant(kCapM, H, G, G, X);
    fold_norm(fold0, cat, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormPreE)),
              b_.x.ptr());
    fold_norm(fold0, cat, b_.partials.ptr(), const_cast<void*>(hidden),
              at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormPreH)), at(b_.x, size_t(H) * kBf16));
    head_gemv(h.fc, b_.x.ptr(), b_.partials.ptr());
    fold_norm(kernels::prep_res_fold_zero_variant(kCapM, H, G),
              kernels::prep_norm_finish_variant(kCapM, H, G, G), b_.partials.ptr(),
              b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormInput)), b_.x.ptr());
    head_gemv(h.qkv, b_.x.ptr(), b_.partials.ptr());
    const KvLayer kv = head_kv();
    if (mode_ == Mode::Rows || mode_ == Mode::DraftRows)   // spec 13a / 13c: the head's rows over its own table
      attn_prep_rows(hctl(), h.fa.ptr(), kv, true, mode_ == Mode::Rows ? br_->head_rows : dr_->head_rows);
    else if (kv8_) {   // spec 12b: the head's KV in the main cache's form
      l0::Kernel& k = kernel(kernels::attn_prep_s1_kv8_variant(kCapM, hd_.fa_q_heads, hd_.fa_kv_heads), "attn_prep_kv8", kWgAttn);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, h.fa.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kv.k);
      k.arg_ptr(7, kv.v);
      k.arg_ptr(8, kv.ks);
      k.arg_ptr(9, kv.vs);
      launch(k, hd_.fa_q_heads + hd_.fa_kv_heads, kCapM);
    } else {
      l0::Kernel& k = kernel(kernels::attn_prep_s1_variant(kCapM, hd_.fa_q_heads, hd_.fa_kv_heads), "attn_prep", kWgAttn);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, h.fa.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kv.k);   // layer 0 of the head's cache (at the bound slot's base)
      k.arg_ptr(7, kv.v);
      launch(k, hd_.fa_q_heads + hd_.fa_kv_heads, kCapM);
    }
  }

  // The MTP head's one KV layer (MtpBuffers::kv_lay), either form.
  KvLayer head_kv() const {
    return mtp_->kv_lay.at_base(mtp_->kv_lay.layer(mtp_->kv_k.ptr(), mtp_->kv_v.ptr(), 0), sb_.kv_base);
  }

  // The verify list's tail: only the head's K/V of the M rows are needed (the next
  // drafts attend over them); its output hidden and logits are not, so the walk stops
  // at attn_prep. 10 launches.
  void head_kv_fill(const void* hidden) { head_front(hidden); }

  // The draft list: the whole head at M = 1 on (hctl.cur_token[0], dh) at hctl.pos,
  // its post-mtp.norm hidden back into dh (the chain's next `h`), its logits into
  // MtpBuffers::logits row draft_i_, the argmax into hctl. 23 launches and one copy -
  // with a draft vocabulary too (spec 8 §11: the compact GEMV and the two dv_argmax
  // stages replace lm_head and the two argmax stages one for one; mtp_gpu_test's gate
  // checks 23 on whichever head it loaded, mtp_gpu_dv128k_test with V'). Spec 15e: a MoE
  // head's FFN is the four-launch MoE block in place of gate||up, SiLU, down - 24.
  void draft() {
    const loader::MtpHead& h = *m_.mtp;
    const model::ModelDesc& hd_ = hd();   // spec 17d: the head's own shapes under TP
    const uint32_t G = DecodeBuffers::kNormGroups, H = hd_.hidden;
    void* const dh = mode_ == Mode::DraftRows ? dr_->dh : mtp_->dh.ptr();   // spec 13c: the rows' chain hidden
    head_front(dh);
    attn_pair(hctl(), head_kv(), hd_);
    head_gemv(h.o, b_.attn_out.ptr(), b_.partials.ptr());
    const std::string fold1 = kernels::prep_res_fold_variant(kCapM, H, 1, G);
    const std::string fin = kernels::prep_norm_finish_variant(kCapM, H, G, G);
    fold_norm(fold1, fin, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormPost)),
              b_.x.ptr());
    if (h.moe) {
      // Spec 15e: a MoE head's FFN is the main layers' block (moe.cl at M = 1) on the
      // head's weights and the MoE scratch's last layer slot. moe_down folds the block
      // into the head's residual itself, so the final norm folds nothing (SP0) - 24
      // launches in all, one more than the dense head's 23.
      moe_block(*h.moe, d_.layers);
      fold_norm(kernels::prep_res_fold_variant(kCapM, H, 0, G), fin, b_.partials.ptr(),
                b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormFinal)),
                b_.x.ptr());
    } else {
      head_gemv(*h.gate_up, b_.x.ptr(), b_.partials.ptr());
      {
        // The HEAD's intermediate (its down linear's K): 17408 on Qwen3.8.
        const uint32_t head_i = h.down->shape.K;
        l0::Kernel& k =
            kernel(kernels::prep_silu_mul_s1_variant(kCapM, head_i), "prep_silu_mul", kWgSilu);
        k.arg_ptr(0, b_.partials.ptr());
        k.arg_ptr(1, b_.x.ptr());
        launch(k, (head_i + kSiluChunk - 1) / kSiluChunk, kCapM);
      }
      head_gemv(*h.down, b_.x.ptr(), b_.partials.ptr());
      fold_norm(fold1, fin, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(hd_, loader::kMtpNormFinal)),
                b_.x.ptr());
    }
    step_.list.copy(dh, b_.x.ptr(), size_t(H) * kBf16 * kCapM);
    float* logits = mode_ == Mode::DraftRows ? static_cast<float*>(dr_->logits)
                                             : mtp_->logits.as<float>() + size_t(draft_i_) * Qwen35::kVocab;
    if (m_.draft_vocab) {
      draft_vocab_head(logits);
      return;
    }
    const loader::DeviceWeight& lm = m_.linears.at({loader::kTopLevel, LinearId::LmHead});
    if (mode_ == Mode::TpDraft) {
      // Spec 17d: the card's int8 vocabulary half and tp.cl's argmax over it into hctl (out_token, out_value,
      // cur_token, pos + 1) - the host merges the two cards' winners before the next draft list (TpEngine::draft).
      gemv_i8w(lm, b_.x.ptr(), logits);
      const uint32_t rows = lm.shape.N;
      const std::string v = kernels::tp_argmax_variant(kCapM, rows, tp_->vocab_used, tp_->rank * rows);
      {
        l0::Kernel& k = kernel(v, "tp_argmax_stage1", kWgArgmax);
        k.arg_ptr(0, logits);
        k.arg_ptr(1, b_.argmax_part.ptr());
        launch(k, (rows + kArgmaxChunk - 1) / kArgmaxChunk, kCapM);
      }
      {
        l0::Kernel& k = kernel(v, "tp_argmax_stage2", kWgArgmax);
        k.arg_ptr(0, hctl());
        k.arg_ptr(1, b_.argmax_part.ptr());
        launch(k, 1, 1);
      }
      return;
    }
    if (lm.kind == model::WeightKind::Int4) {
      const model::GemvShape& s = lm.shape;
      l0::Kernel& k =
          kernel(kernels::gemv_variant(kCapM, s.K, s.N, s.S, s.layout), "gemv", kWgGemv);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, s.layout == 0 ? lm.scales->ptr() : lm.mem.ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, logits);
      launch(k, s.N / kGemvColsPerWg, s.S);
    } else if (lm.kind == model::WeightKind::Int8) {
      gemv_i8w(lm, b_.x.ptr(), logits);   // spec 9: the draft reads the same int8 head
    } else {
      head_gemv(lm, b_.x.ptr(), logits);
    }
    {
      l0::Kernel& k = kernel(kernels::argmax_stage1_variant(kCapM, d_.vocab_used), "argmax_stage1",
                             kWgArgmax);
      k.arg_ptr(0, logits);
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, (Qwen35::kVocab + kArgmaxChunk - 1) / kArgmaxChunk, kCapM);
    }
    {
      l0::Kernel& k = kernel(kernels::argmax_stage2_variant(), "argmax_stage2", kWgArgmax);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  // Open-work 13: the draft's probability, after the draft list's argmax - draft_prob.cl over the logits row this
  // list wrote (a TP card's vocabulary half, or the whole row: under a draft vocabulary -inf outside V'), its softmax
  // max and normaliser into hctl.draft_stat. argmax_part is free again once the argmax's stage 2 has read it.
  void draft_prob() {
    require(mode_ == Mode::Draft || mode_ == Mode::TpDraft, "the draft probability is a draft list's (one card or TP)");
    float* logits = mtp_->logits.as<float>() + size_t(draft_i_) * Qwen35::kVocab;
    const bool tp = mode_ == Mode::TpDraft;
    const uint32_t rows = tp ? m_.linears.at({loader::kTopLevel, LinearId::LmHead}).shape.N : Qwen35::kVocab;
    const std::string v = kernels::draft_prob_variant(rows, tp ? tp_->vocab_used : d_.vocab_used);
    {
      l0::Kernel& k = kernel(v, "draft_prob_stage1", kWgArgmax);
      k.arg_ptr(0, logits);
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, (rows + kArgmaxChunk - 1) / kArgmaxChunk, 1);
    }
    {
      l0::Kernel& k = kernel(v, "draft_prob_stage2", kWgArgmax);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  // Spec 8 §11: the draft's head over the reduced vocabulary V' (loader::DraftVocab) in
  // place of the full one - three launches for three:
  //
  //   gemv_i8w(w', s', x, dv_logits)                  gemv_i8w.cl at N = |V'|, grid |V'|/64
  //     or gemv_bf16(w', x, dv_logits)                gemv_bf16.cl at N = |V'|, {64, 1}
  //   dv_argmax_stage1(dv_logits, ids, part, row_i)   draft_vocab.cl, grid |V'|/1024, WG 256
  //   dv_argmax_stage2(hctl, part, ids)               draft_vocab.cl, grid 1, WG 256
  //
  // The GEMV is the head's own form (`--lm-head`): int8 reads |V'| x 5120 + 4|V'| B of
  // scales instead of 248320 x 5120 - 168 / 336 / 671 MB at 32k / 64k / 128k against
  // 1.27 GB; bf16 twice that against 2.54 GB (derived). The bf16 tiling is
  // kernels::gemv_bf16_tiling(|V'|) = lm_head's {64, 1} at all three sizes (>= 512
  // work-groups), so column j runs the full head's code for column ids[j]. Stage 1 also
  // scatters the compact logits into `row_i` (MtpBuffers::logits row draft_i_) at V''s ids, the rest of
  // which MtpBuffers::zero() holds at -inf; stage 2 maps the winning compact index through
  // `ids` into hctl.out_token[0] and cur_token[0] and advances hctl.pos, as argmax_stage2
  // does for the full head. `part` is the decode scratch's argmax_part ([kM][243][2] fp32;
  // |V'|/1024 <= 128 pairs used), free here as in the full-head draft.
  void draft_vocab_head(float* row_i) {
    const loader::DraftVocab& dv = *m_.draft_vocab;
    const uint32_t nv = dv.size();
    require(nv % kernels::kDraftVocabChunk == 0 && nv / kernels::kDraftVocabChunk <= kWgArgmax,
            "the draft vocabulary's size " + std::to_string(nv) +
                " is not a multiple of 1024 up to 262144 (draft_vocab.cl's grid)");
    require(dv.head.shape.N == nv && dv.head.shape.K == d_.hidden && dv.ids.size() >= size_t(nv) * 4,
            "the draft vocabulary's head is not [|V'|][hidden] with a |V'| id table");
    require(b_.argmax_part.size() >= size_t(nv / kernels::kDraftVocabChunk) * 2 * kFp32,
            "argmax_part is smaller than the draft vocabulary's stage-1 pairs");
    if (dv.head.kind == model::WeightKind::Int8) {
      gemv_i8w(dv.head, b_.x.ptr(), mtp_->dv_logits->ptr());
    } else {
      require(kernels::gemv_bf16_tiling(nv).ksplit == 1 &&
                  kernels::gemv_bf16_tiling(nv).cols == kernels::kGemvBf16Cols,
              "the bf16 draft head's tiling is not lm_head's {64, 1}");
      head_gemv(dv.head, b_.x.ptr(), mtp_->dv_logits->ptr());   // requires kind Bf16
    }
    const std::string v = kernels::dv_argmax_variant(nv);
    {
      l0::Kernel& k = kernel(v, "dv_argmax_stage1", kWgArgmax);
      k.arg_ptr(0, mtp_->dv_logits->ptr());
      k.arg_ptr(1, dv.ids.ptr());
      k.arg_ptr(2, b_.argmax_part.ptr());
      k.arg_ptr(3, row_i);
      launch(k, nv / kernels::kDraftVocabChunk);
    }
    {
      l0::Kernel& k = kernel(v, "dv_argmax_stage2", kWgArgmax);
      k.arg_ptr(0, hctl());
      k.arg_ptr(1, b_.argmax_part.ptr());
      k.arg_ptr(2, dv.ids.ptr());
      launch(k, 1, 1);
    }
  }

  // The layer a launch belongs to, for its label. `kBoundary` is the six
  // launches outside the layer loop (embed_gather, the final norm's
  // prep_res_fold + prep_norm_finish, lm_head and the two argmax stages).
  // It was five until spec 1.5's lever L1 split that norm.
  static constexpr int kBoundary = -1;
  // Spec 8: the MTP head's launches are labelled "L<layers>" - the layer after the
  // last ("L64" on Qwen3.8).
  int head_layer() const { return static_cast<int>(d_.layers); }
  // One GDN layer's recurrent state, fp32 [v-heads][128][128] (Qwen3.8: 48 heads,
  // 3,145,728 B), and its conv ring, bf16 [kConvRing][conv dim] (Qwen3.8 10240).
  size_t gdn_state_stride() const {
    return size_t(d_.gdn_v_heads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * kFp32;
  }
  size_t conv_ring_stride() const {
    return size_t(DecodeBuffers::kConvRing) * d_.gdn_conv_dim() * kBf16;
  }

  l0::Context& ctx_;
  const loader::LoadedModel& m_;
  // Spec 14: the model this list is for (layer counts, intermediate, GEMV rows).
  const model::ModelDesc& d_ = *m_.desc;
  const loader::SmallLayout sl_ = d_.small_layout();   // the small blocks' offsets (spec 15b)
  DecodeBuffers& b_;
  l0::Mem* tap_;
  ProfileEvents* prof_;
  // The list's M (rows in flight). Named like the constant it replaced so the
  // binding sites read unchanged; set once, before the walk.
  const uint32_t kCapM;
  const MtpBuffers* mtp_;
  const Mode mode_;
  const uint32_t draft_i_;
  // Spec 12b: the buffers' KV form decides every KV writer and reader this walk binds.
  const bool kv8_ = b_.kv_lay.form == KvCache::Int8;
  // Spec 28b: which linear kernel the walk binds at kCapM (runtime/gemv_kind.h); Auto unless a test forces it.
  // Spec 28c: `opt_.force_attn` likewise for the verify attention (runtime/verify_attn.h); the attention this walk
  // binds is decided in check_sizes by the rule at row 0's capture-time key count `vattn_keys_`.
  CaptureOptions opt_{};
  uint32_t vattn_keys_ = 1;
  VerifyAttn vattn_ = VerifyAttn::V2;
  CapturedStep step_;
  size_t kv_stride_ = 0;
  int layer_ = kBoundary;
  std::string pending_entry_, pending_variant_;
  // Spec 16b: a stage walk's range and link (null otherwise), the layer m_.layer_small[0]
  // and m_.moe[0] hold, the layer counts the buffers and the model were built for (the
  // descriptor's on every single-card walk), and the one-shot "skip this fold" flag.
  const PpStage* stage_ = nullptr;
  const StageLink* link_ = nullptr;
  uint32_t small_base_ = 0;
  uint32_t n_layers_ = d_.layers, n_gdn_ = d_.gdn_layers, n_fa_ = d_.fa_layers;
  bool resume_ = false;
  // Spec 16d: a stage's verify slots (null: MtpBuffers::gdn_spec, the single card's).
  const l0::Mem* gdn_spec_ = nullptr;
  // Spec 19c: a tap drafter's verify list - its slots, taps and copy sites (null / empty otherwise).
  const TapVerify* tv_ = nullptr;
  const TpLink* tp_ = nullptr;   // spec 17b
  uint32_t tp_s_ = 0;            // the crossing's slices (the row-parallel linears' S)
  std::vector<std::vector<uint32_t>> sites_;
  const BatchRowsBind* br_ = nullptr;   // spec 13a / 13c: build_rows' table, pools and head bindings
  SlotBind sb_{};                        // spec 13c: the bound slot (capture.h; all null: the buffers' own)
  const DraftRowsBind* dr_ = nullptr;    // spec 13c: build_draft_rows' table, Control, rows and logits
  bool prob_ = false;                    // open-work 13: the draft list ends with draft_prob
};

}  // namespace

size_t tp_launches(const model::ModelDesc& d) { return decode_launches(d) + 2 * size_t(d.layers); }
size_t tp_verify_launches(const model::ModelDesc& d) { return tp_launches(d) + 10; }

CapturedStep build_tp(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const TpLink& link,
                      l0::Mem* debug_resid) {
  return Capture(ctx, m, b, debug_resid, link).run();
}

CapturedStep build_tp_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                             const TpLink& link, uint32_t M, const CaptureOptions& opt) {
  if (M == 0 || M > mtp.slots || M > MtpBuffers::kSlots)
    throw std::runtime_error("runtime::build_tp_verify: M " + std::to_string(M) + " is outside [1, " +
                             std::to_string(MtpBuffers::kSlots) + "]");
  return Capture(ctx, m, b, mtp, link, M, Mode::TpVerify, 0).options(opt).run();
}

CapturedStep build_tp_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                            const TpLink& link, uint32_t i, bool prob) {
  if (i >= MtpBuffers::kMaxK)
    throw std::runtime_error("runtime::build_tp_draft: draft index " + std::to_string(i) + " is outside [0, " +
                             std::to_string(MtpBuffers::kMaxK) + ")");
  return Capture(ctx, m, b, mtp, link, 1, Mode::TpDraft, i).prob(prob).run();
}

size_t decode_launches(const model::ModelDesc& d) {
  return 1 + size_t(d.layers) * (d.is_moe() ? 13 : 12) + 5;
}

// Spec 15e: the head's KV fill (head_front: embed, the two pre-fc norm pairs, fc, the
// zero-residual fold pair, q||k||v, attn_prep) after the decode list at M rows.
size_t verify_launches(const model::ModelDesc& d) { return decode_launches(d) + 10; }

// Spec 13c: the row list - the decode list, and with the head its KV fill plus the gather (capture.h, build_rows).
size_t rows_launches(const model::ModelDesc& d, bool head) { return head ? verify_launches(d) + 1 : decode_launches(d); }

// head_front 10, the attention pair 2, o 1, the post norm pair 2, the FFN (gate||up,
// SiLU, down: 3; a MoE head's router, route, gate||up, down: 4), the final norm pair 2,
// lm_head and the two argmax stages 3.
size_t draft_launches(const model::ModelDesc& d) {
  return 10 + 2 + 1 + 2 + (d.mtp_head_moe() ? 4 : 3) + 2 + 3;
}

ProfileEvents::ProfileEvents(l0::Context& ctx) : pool(ctx, kProfileCapacity) {
  // Reserve rather than let `build` grow it: 1024 events is 24 KB of host
  // vector and the reservation keeps every reference `build` hands to the
  // command list stable for the whole walk.
  events.reserve(kProfileCapacity);
}

CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid, ProfileEvents* prof, uint32_t M, const CaptureOptions& opt,
                   const SlotBind& slot) {
  if (M == 0 || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build: M " + std::to_string(M) + " is outside [1, " +
                             std::to_string(DecodeBuffers::kM) + "]");
  return Capture(ctx, m, b, debug_resid, prof, M).options(opt).slot(slot).run();
}

CapturedStep build_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                          const MtpBuffers& mtp, uint32_t M, ProfileEvents* prof, const CaptureOptions& opt,
                          const SlotBind& slot) {
  // Spec 19a: up to the buffers' slot count (kSlots = 4 on every engine; 8 for
  // probe_mtp_steps' M = 5..8 lists), never past DecodeBuffers::kM.
  if (M == 0 || M > mtp.slots || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build_verify: M " + std::to_string(M) +
                             " is outside [1, " + std::to_string(mtp.slots) +
                             "] (MtpBuffers::slots)");
  return Capture(ctx, m, b, nullptr, prof, M, &mtp, Mode::Verify).options(opt).slot(slot).run();
}

CapturedStep build_verify_taps(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                               const TapVerify& tv, uint32_t M, const CaptureOptions& opt) {
  if (M == 0 || M > tv.slots || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build_verify_taps: M " + std::to_string(M) +
                             " is outside [1, " + std::to_string(tv.slots) + "] (TapVerify::slots)");
  return Capture(ctx, m, b, tv, M).options(opt).run();
}

CapturedStep build_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const MtpBuffers& mtp, uint32_t i, const SlotBind& slot, bool prob) {
  return Capture(ctx, m, b, nullptr, nullptr, 1, &mtp, Mode::Draft, i).slot(slot).prob(prob).run();
}

CapturedStep build_stage_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                                const PpStage& st, const StageLink& link, uint32_t M,
                                const MtpBuffers* mtp, const l0::Mem& gdn_spec, const CaptureOptions& opt) {
  if (M == 0 || M > MtpBuffers::kSlots)
    throw std::runtime_error("runtime::build_stage_verify: M " + std::to_string(M) +
                             " is outside [1, " + std::to_string(MtpBuffers::kSlots) + "]");
  const model::ModelDesc& d = *m.desc;
  if (st.has_head(d) != (mtp != nullptr))
    throw std::invalid_argument("runtime::build_stage_verify: the head's buffers belong to the "
                                "stage that holds the head, and only to it");
  const std::array<PpStage, kPpDevices> both = pp_stages(d, st.first == 0 ? st.last : st.first);
  if (pp_stage_verify_launches(d, both[0]) + pp_stage_verify_launches(d, both[1]) != verify_launches(d))
    throw std::logic_error("runtime::build_stage_verify: the stages' launches do not add up to " +
                           std::to_string(verify_launches(d)));
  return Capture(ctx, m, b, st, &link, M, mtp, &gdn_spec, Mode::StageVerify, 0).options(opt).run();
}

CapturedStep build_stage_verify_taps(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                                     const PpStage& st, const StageLink& link, const TapVerify& tv, uint32_t M,
                                     const CaptureOptions& opt) {
  if (M == 0 || M > tv.slots || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build_stage_verify_taps: M " + std::to_string(M) + " is outside [1, " +
                             std::to_string(tv.slots) + "] (TapVerify::slots)");
  const model::ModelDesc& d = *m.desc;
  const std::array<PpStage, kPpDevices> both = pp_stages(d, st.first == 0 ? st.last : st.first);
  if (pp_stage_launches(d, both[0]) + pp_stage_launches(d, both[1]) != decode_launches(d))
    throw std::logic_error("runtime::build_stage_verify_taps: the stages' launches do not add up to " +
                           std::to_string(decode_launches(d)));
  return Capture(ctx, m, b, st, link, tv, M).options(opt).run();
}

CapturedStep build_draft_rows(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const MtpBuffers& mtp,
                              const DraftRowsBind& dr, uint32_t M) {
  if (M < 2 || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build_draft_rows: M " + std::to_string(M) + " is outside [2, " +
                             std::to_string(DecodeBuffers::kM) + "] (one row is build_draft's list)");
  return Capture(ctx, m, b, mtp, dr, M).run();
}

CapturedStep build_rows(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, const BatchRowsBind& br,
                        uint32_t M) {
  if (M == 0 || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build_rows: M " + std::to_string(M) + " is outside [1, " +
                             std::to_string(DecodeBuffers::kM) + "]");
  return Capture(ctx, m, b, br, M).run();
}

CapturedStep build_stage_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                               const PpStage& st, const MtpBuffers& mtp, uint32_t i) {
  return Capture(ctx, m, b, st, nullptr, 1, &mtp, nullptr, Mode::StageDraft, i).run();
}

CapturedStep build_stage(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const PpStage& st, const StageLink& link) {
  // The two stages' compute launches are the single-card list's, cut in two.
  const model::ModelDesc& d = *m.desc;
  if (st.last == d.layers || st.first == 0) {
    const std::array<PpStage, kPpDevices> both =
        pp_stages(d, st.first == 0 ? st.last : st.first);
    if (pp_stage_launches(d, both[0]) + pp_stage_launches(d, both[1]) != decode_launches(d))
      throw std::logic_error("runtime::build_stage: the stages' launches do not add up to " +
                             std::to_string(decode_launches(d)));
  }
  return Capture(ctx, m, b, st, link).run();
}

}  // namespace runtime
