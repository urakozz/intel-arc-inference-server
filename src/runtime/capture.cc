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
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/control.h"

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
// Spec 15c: the MoE route row - buffer_sizes.h sizes it, moe.cl (RW) writes it.
static_assert(kMoeRouteWords == kernels::moe_route::kWords, "the MoE route row's width moved");

// The captured list's M is `build`'s `M` argument (default 1, what ships). The
// default build compiles M = 1 only (spec §9: the M loop exists in every
// kernel, M = 1 is what ships), while `DecodeBuffers::kM` = 8 is the
// *allocation* capacity. Every kernel indexes its buffers with its own compiled
// M, so the argument and the variant names below move together. M = 2..4 at
// max_len 16384 exist only under the CMake option B70_DECODE_EXTRA_M (spec 8a's
// probe); any other M names a binary that does not exist and throws at
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
// (spec 14: 48 / 16 on Qwen3.8, 54 / 18 on Agnes); so are the widths (spec 15b).

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
enum class Mode { Plain, Verify, Draft };

class Capture {
 public:
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, l0::Mem* tap,
          ProfileEvents* prof, uint32_t cap_m, const MtpBuffers* mtp = nullptr,
          Mode mode = Mode::Plain, uint32_t draft_i = 0)
      : ctx_(ctx), m_(m), b_(b), tap_(tap), prof_(prof), kCapM(cap_m), mtp_(mtp), mode_(mode),
        draft_i_(draft_i), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    kv_stride_ = size_t(b_.max_len) * d_.fa_kv_heads * Qwen35::kFaHeadDim * kBf16;
    check_sizes();
    if (mode_ != Mode::Plain) check_mtp();
    if (mode_ == Mode::Draft) {
      layer_ = head_layer();
      draft();
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
    if (mode_ == Mode::Verify) {
      // The verify rows' post-final-norm hidden (`x`, what lm_head read) into hh rows
      // 1..M; row 0 is h_{pos-1}, left by the previous commit or prefill. Then the
      // head's KV fill over hh rows 0..M-1 (MtpBuffers).
      const size_t row = size_t(d_.hidden) * kBf16;
      step_.list.copy(at(mtp_->hh, row), b_.x.ptr(), row * kCapM);
      layer_ = head_layer();
      head_kv_fill(mtp_->hh.ptr());
    }

    require(gdn == d_.gdn_layers && fa == d_.fa_layers,
            "layer kind counts are not the descriptor's " + std::to_string(d_.gdn_layers) +
                " GDN / " + std::to_string(d_.fa_layers) + " FA");
    // The plain list's length is a property of the model (decode_launches): 774 on
    // Qwen3.8, 526 on Ornith (spec 15c). A walk that grew or lost a launch throws here.
    require(mode_ != Mode::Plain || step_.kernel_count == decode_launches(d_),
            "the decode list has " + std::to_string(step_.kernel_count) + " launches, not the " +
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
    // The attention variants bake MAXLEN (attn_part's stride) and ATTN_BLOCK
    // (its block count); the grid comes from b.max_len and kAttnBlock, and the
    // RoPE table from the loader's. All of them are one number or the KV cache
    // is read at the wrong stride. The block size needs no `require` here - it
    // is in the variant NAME (`..._B64`), so a host/device disagreement is a
    // missing binary at `kernel()` rather than a wrong stride at replay.
    require(m_.max_len == b_.max_len, "model max_len " + std::to_string(m_.max_len) +
                                          " != buffers max_len " + std::to_string(b_.max_len));
    require(b_.max_len % DecodeBuffers::kAttnBlock == 0,
            "max_len must be a multiple of DecodeBuffers::kAttnBlock (" +
                std::to_string(DecodeBuffers::kAttnBlock) + ") - attn_decode's block grid");
    // Spec 6: MAXLEN is baked into both attention binaries, and only some max_lens are
    // compiled (src/kernels/CMakeLists.txt: 4096, 16384, 32768 and 131072 at M = 1). Name the missing one
    // here, before a single command is appended, rather than as a bare path later.
    // Spec 10: v2 bakes no MAXLEN (one binary per M, `_T<kAttnV2Blocks>`).
    const std::vector<std::string> attn_bins =
        attn_ == DecodeAttn::V2
            ? std::vector<std::string>{kernels::attn_v2_variant(kCapM, DecodeBuffers::kAttnV2Blocks, d_.fa_q_heads, d_.fa_kv_heads)}
            : std::vector<std::string>{
                  kernels::attn_decode_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock, d_.fa_q_heads, d_.fa_kv_heads),
                  kernels::attn_reduce_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock, d_.fa_q_heads, d_.fa_kv_heads)};
    // attn_part is sized for one pair (DecodeScratchDims::sizes): v2's is a fixed 32 blocks
    // per row, v1's every block of the cache. Binding v1 over buffers sized for v2 would
    // write past the allocation, so refuse it here rather than corrupt memory on token 1.
    require(b_.attn_part.size() >= DecodeScratchDims::sizes(b_.max_len, d_, attn_).attn_part,
            std::string("attn_part (") + std::to_string(b_.attn_part.size()) +
                " B) was sized for the other decode attention pair than " +
                decode_attn_name(attn_) + " - set B70_DECODE_ATTN before the buffers are built");
    for (const std::string& v : attn_bins)
      require(std::ifstream(kernels::path(v)).good(),
              "no decode attention is compiled for max_len " + std::to_string(b_.max_len) +
                  ": " + v + " is missing (" + kernels::path(v) +
                  "); the compiled max_lens are listed in src/kernels/CMakeLists.txt");

    const std::string gdn_n = std::to_string(d_.gdn_layers), fa_n = std::to_string(d_.fa_layers);
    require(b_.gdn_state.size() == gdn_state_stride() * d_.gdn_layers,
            "gdn_state is not " + gdn_n + " slices");
    require(b_.conv_ring.size() == conv_ring_stride() * d_.gdn_layers,
            "conv_ring is not " + gdn_n + " slices");
    require(b_.kv_k.size() == kv_stride_ * d_.fa_layers, "kv_k is not " + fa_n + " slices");
    require(b_.kv_v.size() == kv_stride_ * d_.fa_layers, "kv_v is not " + fa_n + " slices");
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
    // (runtime::moe_scratch_layout), and the lists that do not exist for it yet.
    if (d_.is_moe()) {
      require(m_.moe.size() == d_.layers,
              "the loaded model has " + std::to_string(m_.moe.size()) + " MoE layers, not " +
                  std::to_string(d_.layers));
      require(b_.moe != nullptr && b_.moe->size() >= moe_scratch_layout(d_).total,
              "the decode scratch has no MoE region of moe_scratch_layout()'s size");
      require(mode_ == Mode::Plain, "MTP on a mixture-of-experts model is spec 15e");
      require(attn_ == DecodeAttn::V2,
              "decode attention v1 (B70_DECODE_ATTN=v1) is not built at " + d_.name + "'s heads");
    }
  }

  // Spec 8: what the MTP walks need of the head and its buffers.
  void check_mtp() {
    require(mtp_ != nullptr && m_.mtp != nullptr, "an MTP list needs the loaded head and MtpBuffers");
    require(mtp_->max_len == b_.max_len, "MtpBuffers max_len != buffers max_len");
    require(kCapM <= MtpBuffers::kSlots, "an MTP verify list has at most kSlots rows");
    require(mode_ != Mode::Draft || (kCapM == 1 && draft_i_ < MtpBuffers::kMaxK),
            "a draft list is M = 1 with draft index < kMaxK");
    require(mtp_->gdn_spec.size() == gdn_state_stride() * d_.gdn_layers * (MtpBuffers::kSlots - 1),
            "gdn_spec is not (kSlots - 1) x " + std::to_string(d_.gdn_layers) + " slices");
    require(mtp_->kv_k.size() == kv_stride_ && mtp_->kv_v.size() == kv_stride_,
            "the head's KV is not one [max_len][kv-heads][256] layer");
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
  // Decode attention over one KV cache: attn.cl's attn_decode + attn_reduce (v1) or
  // attn_v2.cl's pair (spec 10, plan 10b), whichever `attn_` names. `ctrl` is the list's
  // control block (the head's `hctl` in the draft list); both read q from attn_q, write
  // partials into attn_part and the gated bf16 rows into attn_out.
  void attn_pair(void* ctrl, void* kk, void* vv) {
    if (attn_ == DecodeAttn::V2) {
      // attn_v2.cl: grid (4 kv-heads, kAttnV2Blocks), WG 256; the stride is derived per
      // row from Control::pos on the device, so one list serves every depth.
      const std::string v = kernels::attn_v2_variant(kCapM, DecodeBuffers::kAttnV2Blocks, d_.fa_q_heads, d_.fa_kv_heads);
      {
        l0::Kernel& k = kernel(v, "attn_decode_v2", kWgAttn);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.attn_q.ptr());
        k.arg_ptr(2, kk);
        k.arg_ptr(3, vv);
        k.arg_ptr(4, b_.attn_part.ptr());
        launch(k, d_.fa_kv_heads, DecodeBuffers::kAttnV2Blocks);
      }
      // attn_reduce_v2(ctrl, attn_part, attn_gate, attn_out), grid (24 q-heads, M).
      {
        l0::Kernel& k = kernel(v, "attn_reduce_v2", kWgAttn);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.attn_part.ptr());
        k.arg_ptr(2, b_.attn_gate.ptr());
        k.arg_ptr(3, b_.attn_out.ptr());
        launch(k, d_.fa_q_heads, kCapM);
      }
      return;
    }
    // attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part) - attn.cl (Task 5), grid
    // (4 kv-heads, max_len/kAttnBlock blocks), WG 256. The grid spans the whole
    // cache because a captured list cannot resize; the work-groups past the
    // context early-out on their first instruction (the grid's history and the
    // idle work-groups' measured cost: docs/15 "Spec 1.6 §5.2", docs/12 `attn`).
    {
      l0::Kernel& k = kernel(
          kernels::attn_decode_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock, d_.fa_q_heads, d_.fa_kv_heads),
          "attn_decode", kWgAttn);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kk);
      k.arg_ptr(3, vv);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, d_.fa_kv_heads, b_.max_len / DecodeBuffers::kAttnBlock);
    }
    // attn_reduce(ctrl, attn_part, attn_gate, attn_out) - attn.cl (Task 5),
    // grid (24 q-heads, M), WG 256.
    {
      l0::Kernel& k =
          kernel(kernels::attn_reduce_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock, d_.fa_q_heads, d_.fa_kv_heads),
                 "attn_reduce", kWgAttn);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, d_.fa_q_heads, kCapM);
    }
  }

  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules
               .emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant)))
               .first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    l0::Kernel& k = *step_.kernels.back();
    k.group_size(wg);
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

  // --- the kernels ----------------------------------------------------------

  // embed_gather(ctrl, embed, resid) - src/kernels/embed_gather.cl (Task 3),
  // grid (1, M), WG 256. Reads the id from `ctrl` at execution time, which is
  // the whole reason the list can be captured once.
  void embed_gather() {
    l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM, d_.hidden), "embed_gather", kWgEmbed);
    k.arg_ptr(0, b_.control.ptr());
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
  // (prep.cl states the tree, docs/12 the contract).
  //
  // `b_.norm_sumsq` is a single 640-byte allocation shared by all 129 sites:
  // the list is in-order, so a site's stage B has consumed its own stage A's
  // sums before the next site's stage A overwrites them.
  void res_norm(uint32_t s_prev, const void* norm_w) {
    const uint32_t g = DecodeBuffers::kNormGroups;
    {
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
  void gemv(uint32_t layer, LinearId id, const void* x, const l0::Mem& out) {
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
    l0::Kernel& k =
        kernel(kernels::gemv_variant(kCapM, s.K, s.N, s.S, s.layout), "gemv", kWgGemv);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, s.layout == 0 ? w.scales->ptr() : w.mem.ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, out.ptr());
    launch(k, s.N / kGemvColsPerWg, s.S);
  }
  void gemv(uint32_t layer, LinearId id, const void* x) { gemv(layer, id, x, b_.partials); }

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
    l0::Kernel& k =
        kernel(kernels::gemv_i8w_variant(kCapM, s.K, s.N), "gemv_i8w", kernels::kGemvI8wCols);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, w.scales->ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, out);
    launch(k, s.N / kernels::kGemvI8wCols);
  }

  // The MLP half, identical in both layer kinds: post-norm folding the mixer's
  // partials, gate‖up, SiLU·mul, down. `mixer_s` is the split-K width of the
  // GEMV that produced those partials (out_proj / o_proj, both S = 4).
  void mlp(uint32_t layer, uint32_t mixer_s) {
    res_norm(mixer_s, at(m_.layer_small[layer].norms, sl_.norms_off_post));
    gemv(layer, LinearId::GateUp, b_.x.ptr());
    // prep_silu_mul(partials, x_out) - prep.cl (Task 2), grid (I/4096, M) = (5, M)
    // at both I = 17408 (ragged last chunk 1024) and Agnes's 19456 (3072), WG 256.
    {
      l0::Kernel& k = kernel(kernels::prep_silu_mul_variant(kCapM, d_.intermediate),
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
    res_norm(mixer_s, at(m_.layer_small[layer].norms, sl_.norms_off_post));
    const loader::MoeLayer& w = m_.moe.at(layer);
    const model::MoeDesc& md = d_.moe;
    const MoeScratchLayout ml = moe_scratch_layout(d_);
    void* lg = at(*b_.moe, ml.logits_at(layer));
    void* rt = at(*b_.moe, ml.route_at(layer));
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

  // A gated-delta-net layer (48 of 64 on Qwen3.8, 54 of 72 on Agnes): 10 kernels.
  void gdn_layer(uint32_t layer, uint32_t g) {
    // Layer 0 leads the whole step, so there are no previous partials to fold
    // and `resid` is exactly embed_gather's output: the SP0 variant. Every
    // other layer folds the previous layer's `down` (S = 4).
    res_norm(layer == 0 ? 0u : d_.ffn_fold_s(),
             at(m_.layer_small[layer].norms, sl_.norms_off_input));
    gemv(layer, LinearId::QkvZ, b_.x.ptr());
    gemv_bf16(layer, LinearId::AB, b_.x.ptr(), b_.ab_out);
    // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o)
    // - src/kernels/gdn_step.cl (Task 4), grid (48 v-heads, 4 state-column
    // chunks), WG 256. `conv_ring` and `state` are this GDN layer's slices;
    // `gdn_small` is the layer's GDN block base as a `const float*` (the
    // kernel finds conv/negA/dt_bias inside it at loader/small_layout.h's
    // offsets, passed to ocloc as NEGA_OFF/DTBIAS_OFF).
    {
      // Spec 8: the verify list binds the SPEC_SLOTS build, whose eighth argument is
      // this layer's slice of slot 1 (MtpBuffers::gdn_spec is slot-major, so slot s
      // is that plus (s - 1) whole slots - gdn_step.cl, SPEC_SLOTS).
      const bool slots = mode_ == Mode::Verify;
      l0::Kernel& k = kernel(slots ? kernels::gdn_step_slots_variant(kCapM, d_.gdn_layers, d_.gdn_k_heads, d_.gdn_v_heads)
                                   : kernels::gdn_step_variant(kCapM, d_.gdn_k_heads, d_.gdn_v_heads),
                             "gdn_step", kWgGdn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.ab_out.ptr());
      k.arg_ptr(3, m_.layer_small[layer].gdn.ptr());
      k.arg_ptr(4, at(b_.conv_ring, size_t(g) * conv_ring_stride()));
      k.arg_ptr(5, at(b_.gdn_state, size_t(g) * gdn_state_stride()));
      k.arg_ptr(6, b_.gdn_o.ptr());
      if (slots) k.arg_ptr(7, at(mtp_->gdn_spec, size_t(g) * gdn_state_stride()));
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
      k.arg_ptr(2, at(m_.layer_small[layer].gdn, sl_.gdn_off_gated_norm));
      k.arg_ptr(3, b_.x.ptr());
      launch(k, d_.gdn_v_heads, kCapM);
    }
    gemv(layer, LinearId::OutProj, b_.x.ptr());
    ffn(layer, d_.shape(LinearId::OutProj).S);
  }

  // A full-attention layer (16 of 64 on Qwen3.8, 18 of 72 on Agnes): 10 kernels.
  void fa_layer(uint32_t layer, uint32_t f) {
    res_norm(d_.ffn_fold_s(), at(m_.layer_small[layer].norms, sl_.norms_off_input));
    gemv(layer, LinearId::Qkv, b_.x.ptr());
    // This FA layer's KV cache slices - bf16 [max_len][4][256] each, indexed by
    // absolute position inside the kernels.
    void* kk = at(b_.kv_k, size_t(f) * kv_stride_);
    void* vv = at(b_.kv_v, size_t(f) * kv_stride_);
    // attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v)
    // - src/kernels/attn.cl (Task 5), grid (24 q-heads + 4 kv-heads, M), WG 256.
    // `fa_small` is the layer's FA block (q_norm ‖ k_norm, fp32 1+w) - the same
    // `SmallTensors::gdn` allocation the GDN layers use for their own block.
    {
      l0::Kernel& k = kernel(kernels::attn_prep_variant(kCapM, d_.fa_q_heads, d_.fa_kv_heads), "attn_prep", kWgAttn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, m_.layer_small[layer].gdn.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kk);
      k.arg_ptr(7, vv);
      launch(k, d_.fa_q_heads + d_.fa_kv_heads, kCapM);
    }
    // attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part) - attn.cl (Task 5), grid
    // (4 kv-heads, max_len/kAttnBlock blocks), WG 256. The grid spans the whole
    // cache because a captured list cannot resize; the work-groups past the
    // context early-out on their first instruction. Since spec 1.5's lever L5
    // the block is 64 positions, so this grid is (4, 256) = 1024 work-groups at
    // max_len 16384, of which 260 are live at depth 4096 - nearly 4x the 68
    // before. docs/15 §2 measured work-group count as very nearly free but put
    // ~264 work-groups "past the point any measurement here reaches"; the
    // measurement now reaches it. **"Nearly free" is WITHDRAWN for this grid
    // (2026-08-25, tools/probe/probe_attn; docs/15 "Spec 1.6 §5.2"):** that was
    // measured at ATTN_BLOCK 256 over 20 -> 68 work-groups, a low-occupancy
    // regime, and at the shipped block the launch is LINEAR in live
    // work-groups (68 -> 1004 = 14.8x for 11.8x the time). What survives, and
    // is what this comment actually needs, is the sentence below: the IDLE
    // work-groups are free -- now measured per kernel at 1.63 ns each. The idle work-groups grow with the grid: at
    // depth 4096 there are 1024 - 260 = 764 of them per launch, against
    // 256 - 68 = 188 before, so the retile added 576. **Their cost is not a
    // term to add**: the 224.046 us/launch below was measured WITH all 764 of
    // them present, so whatever they cost is already inside it. For a bound
    // anyway, the two available transplants disagree 47x -- doc 07 #12's own
    // 14.97 ns/work-group puts 576 at 8.6 us/launch (3.8% of 224 us, 0.38% of
    // the step across 16 launches), docs/15 §2's in-situ 0.32 ns/work-group puts
    // it at 0.18 us -- and doc 07 #12 keeps the larger one so the bound is not
    // flattered. What is bought is a quartered per-work-group serial walk
    // (369.988 -> 224.046 us/launch, measured, grid growth included).
    attn_pair(b_.control.ptr(), kk, vv);
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
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  // --- spec 8: the MTP head (plan 8b) ---------------------------------------
  //
  // The head is one full-attention layer over fc(cat(pre_fc_norm_embedding(embed(x)),
  // pre_fc_norm_hidden(h))) (docs/probe-mtp-2026-09-27.md §1). It runs on its OWN
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
    const uint32_t G = DecodeBuffers::kNormGroups, H = d_.hidden, X = 2 * H;
    {
      l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM, d_.hidden), "embed_gather", kWgEmbed);
      k.arg_ptr(0, mtp_->hctl.ptr());
      k.arg_ptr(1, m_.embed.ptr());
      k.arg_ptr(2, b_.resid.ptr());
      launch(k, 1, kCapM);
    }
    const std::string fold0 = kernels::prep_res_fold_variant(kCapM, H, 0, G);
    const std::string cat = kernels::prep_norm_finish_strided_variant(kCapM, H, G, G, X);
    fold_norm(fold0, cat, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(d_, loader::kMtpNormPreE)),
              b_.x.ptr());
    fold_norm(fold0, cat, b_.partials.ptr(), const_cast<void*>(hidden),
              at(h.norms, loader::mtp_norm_off(d_, loader::kMtpNormPreH)), at(b_.x, size_t(H) * kBf16));
    head_gemv(h.fc, b_.x.ptr(), b_.partials.ptr());
    fold_norm(kernels::prep_res_fold_zero_variant(kCapM, H, G),
              kernels::prep_norm_finish_variant(kCapM, H, G, G), b_.partials.ptr(),
              b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(d_, loader::kMtpNormInput)), b_.x.ptr());
    head_gemv(h.qkv, b_.x.ptr(), b_.partials.ptr());
    {
      l0::Kernel& k = kernel(kernels::attn_prep_s1_variant(kCapM, d_.fa_q_heads, d_.fa_kv_heads), "attn_prep", kWgAttn);
      k.arg_ptr(0, mtp_->hctl.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, h.fa.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, mtp_->kv_k.ptr());
      k.arg_ptr(7, mtp_->kv_v.ptr());
      launch(k, d_.fa_q_heads + d_.fa_kv_heads, kCapM);
    }
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
  // checks 23 on whichever head it loaded, mtp_gpu_dv128k_test with V').
  void draft() {
    const loader::MtpHead& h = *m_.mtp;
    const uint32_t G = DecodeBuffers::kNormGroups, H = d_.hidden;
    head_front(mtp_->dh.ptr());
    attn_pair(mtp_->hctl.ptr(), mtp_->kv_k.ptr(), mtp_->kv_v.ptr());
    head_gemv(h.o, b_.attn_out.ptr(), b_.partials.ptr());
    const std::string fold1 = kernels::prep_res_fold_variant(kCapM, H, 1, G);
    const std::string fin = kernels::prep_norm_finish_variant(kCapM, H, G, G);
    fold_norm(fold1, fin, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(d_, loader::kMtpNormPost)),
              b_.x.ptr());
    head_gemv(h.gate_up, b_.x.ptr(), b_.partials.ptr());
    {
      // The HEAD's intermediate (its down linear's K): 17408 on both supported
      // checkpoints - Agnes's MTP block has no parallel FFN.
      const uint32_t head_i = h.down.shape.K;
      l0::Kernel& k =
          kernel(kernels::prep_silu_mul_s1_variant(kCapM, head_i), "prep_silu_mul", kWgSilu);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.x.ptr());
      launch(k, (head_i + kSiluChunk - 1) / kSiluChunk, kCapM);
    }
    head_gemv(h.down, b_.x.ptr(), b_.partials.ptr());
    fold_norm(fold1, fin, b_.partials.ptr(), b_.resid.ptr(), at(h.norms, loader::mtp_norm_off(d_, loader::kMtpNormFinal)),
              b_.x.ptr());
    step_.list.copy(mtp_->dh.ptr(), b_.x.ptr(), size_t(H) * kBf16);
    float* logits = mtp_->logits.as<float>() + size_t(draft_i_) * Qwen35::kVocab;
    if (m_.draft_vocab) {
      draft_vocab_head(logits);
      return;
    }
    const loader::DeviceWeight& lm = m_.linears.at({loader::kTopLevel, LinearId::LmHead});
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
      k.arg_ptr(0, mtp_->hctl.ptr());
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
      k.arg_ptr(0, mtp_->hctl.ptr());
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
  // last ("L64" on Qwen3.8, "L72" on Agnes).
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
  // Spec 10: the decode-attention pair, read from B70_DECODE_ATTN once per build.
  const DecodeAttn attn_ = decode_attn();
  CapturedStep step_;
  size_t kv_stride_ = 0;
  int layer_ = kBoundary;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

size_t decode_launches(const model::ModelDesc& d) {
  return 1 + size_t(d.layers) * (d.is_moe() ? 13 : 12) + 5;
}

ProfileEvents::ProfileEvents(l0::Context& ctx) : pool(ctx, kProfileCapacity) {
  // Reserve rather than let `build` grow it: 1024 events is 24 KB of host
  // vector and the reservation keeps every reference `build` hands to the
  // command list stable for the whole walk.
  events.reserve(kProfileCapacity);
}

CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid, ProfileEvents* prof, uint32_t M) {
  if (M == 0 || M > DecodeBuffers::kM)
    throw std::runtime_error("runtime::build: M " + std::to_string(M) + " is outside [1, " +
                             std::to_string(DecodeBuffers::kM) + "]");
  return Capture(ctx, m, b, debug_resid, prof, M).run();
}

CapturedStep build_verify(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                          const MtpBuffers& mtp, uint32_t M) {
  if (M == 0 || M > MtpBuffers::kSlots)
    throw std::runtime_error("runtime::build_verify: M " + std::to_string(M) +
                             " is outside [1, " + std::to_string(MtpBuffers::kSlots) + "]");
  return Capture(ctx, m, b, nullptr, nullptr, M, &mtp, Mode::Verify).run();
}

CapturedStep build_draft(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                         const MtpBuffers& mtp, uint32_t i) {
  return Capture(ctx, m, b, nullptr, nullptr, 1, &mtp, Mode::Draft, i).run();
}

}  // namespace runtime
