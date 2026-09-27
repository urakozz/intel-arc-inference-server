#include "runtime/capture.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

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
// The strides that runtime/buffers.cc sized these allocations with. They are
// re-derived here from the same model constants rather than shared, and
// `check_sizes()` below asserts stride x layers == allocation size - so a
// divergence is a throw at capture, not a wrong KV slot at token 300.
constexpr uint32_t kFaLayers = Qwen35::kLayers / 4;            // is_fa(l) == (l % 4 == 3)
constexpr uint32_t kGdnLayers = Qwen35::kLayers - kFaLayers;   // 48
constexpr uint32_t kConvDim =
    (2 * Qwen35::kGdnKHeads + Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim;  // 10240
constexpr size_t kGdnStateStride =
    size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * kFp32;
constexpr size_t kConvRingStride = size_t(DecodeBuffers::kConvRing) * kConvDim * kBf16;

uint8_t* at(const l0::Mem& m, size_t byte_off) {
  return static_cast<uint8_t*>(m.ptr()) + byte_off;
}

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::build: " + what);
}

// The walk itself. One instance per `build` call; it owns the CapturedStep
// under construction and hands it back closed.
class Capture {
 public:
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, l0::Mem* tap,
          ProfileEvents* prof, uint32_t cap_m)
      : ctx_(ctx), m_(m), b_(b), tap_(tap), prof_(prof), kCapM(cap_m),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    kv_stride_ = size_t(b_.max_len) * Qwen35::kFaKvHeads * Qwen35::kFaHeadDim * kBf16;
    check_sizes();
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
    for (const model::LayerDesc& layer : Qwen35::layers()) {
      layer_ = static_cast<int>(layer.index);
      if (layer.kind == model::LayerKind::GDN)
        gdn_layer(layer.index, gdn++);
      else
        fa_layer(layer.index, fa++);
      tap(layer.index);
    }
    layer_ = kBoundary;
    head();

    require(gdn == kGdnLayers && fa == kFaLayers, "layer kind counts are not 48 GDN / 16 FA");
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
    for (const std::string& v :
         {kernels::attn_decode_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock),
          kernels::attn_reduce_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock)})
      require(std::ifstream(kernels::path(v)).good(),
              "no decode attention is compiled for max_len " + std::to_string(b_.max_len) +
                  ": " + v + " is missing (" + kernels::path(v) +
                  "); the compiled max_lens are listed in src/kernels/CMakeLists.txt");

    require(b_.gdn_state.size() == kGdnStateStride * kGdnLayers, "gdn_state is not 48 slices");
    require(b_.conv_ring.size() == kConvRingStride * kGdnLayers, "conv_ring is not 48 slices");
    require(b_.kv_k.size() == kv_stride_ * kFaLayers, "kv_k is not 16 slices");
    require(b_.kv_v.size() == kv_stride_ * kFaLayers, "kv_v is not 16 slices");
    if (tap_)
      require(tap_->size() >= size_t(Qwen35::kLayers) * kCapM * Qwen35::kHidden * kBf16,
              "debug_resid is smaller than [64][M][5120] bf16");
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
    require(Qwen35::shape(LinearId::QkvZ).S == 1,
            "qkv||z is no longer S=1, but gdn_step.cl (QKVZ_S) and prep.cl "
            "(GATED_S) bake S=1 into their partials indexing");
    require(Qwen35::shape(LinearId::GateUp).S == 8,
            "gate||up is no longer S=8, but prep.cl (SILU_S) bakes an 8-slice sum");
    require(Qwen35::shape(LinearId::Qkv).S == 2,
            "qkv is no longer S=2, but attn.cl (QKV_S) bakes a 2-slice sum into every "
            "partials load");
  }

  // One launch site: load (or reuse) the variant's device binary, make a fresh
  // Kernel for it and set its group size. A Kernel per site is not required -
  // arguments are captured at append, so one object could serve many launches -
  // but it keeps the list auditable (kernels[i] is what launch i ran) for the
  // cost of 774 handles.
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
    const size_t bytes = size_t(kCapM) * Qwen35::kHidden * kBf16;
    step_.list.copy(at(*tap_, size_t(layer) * bytes), b_.resid.ptr(), bytes);
  }

  // --- the kernels ----------------------------------------------------------

  // embed_gather(ctrl, embed, resid) - src/kernels/embed_gather.cl (Task 3),
  // grid (1, M), WG 256. Reads the id from `ctrl` at execution time, which is
  // the whole reason the list can be captured once.
  void embed_gather() {
    l0::Kernel& k = kernel(kernels::embed_gather_variant(kCapM), "embed_gather", kWgEmbed);
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
          kernel(kernels::prep_res_fold_variant(kCapM, Qwen35::kHidden, s_prev, g),
                 "prep_res_fold", kWgResFold);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.resid.ptr());
      k.arg_ptr(2, b_.norm_sumsq.ptr());
      launch(k, g, kCapM);
    }
    {
      l0::Kernel& k =
          kernel(kernels::prep_norm_finish_variant(kCapM, Qwen35::kHidden, g, g),
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

  // The MLP half, identical in both layer kinds: post-norm folding the mixer's
  // partials, gate‖up, SiLU·mul, down. `mixer_s` is the split-K width of the
  // GEMV that produced those partials (out_proj / o_proj, both S = 4).
  void mlp(uint32_t layer, uint32_t mixer_s) {
    res_norm(mixer_s, at(m_.layer_small[layer].norms, loader::kNormsOffPost));
    gemv(layer, LinearId::GateUp, b_.x.ptr());
    // prep_silu_mul(partials, x_out) - prep.cl (Task 2), grid (N/4096, M) = (5, M),
    // WG 256; the last work-group covers the ragged 1024.
    {
      l0::Kernel& k = kernel(kernels::prep_silu_mul_variant(kCapM), "prep_silu_mul", kWgSilu);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.x.ptr());
      launch(k, (Qwen35::kIntermediate + kSiluChunk - 1) / kSiluChunk, kCapM);
    }
    gemv(layer, LinearId::Down, b_.x.ptr());
  }

  // A gated-delta-net layer (48 of 64): 10 kernels.
  void gdn_layer(uint32_t layer, uint32_t g) {
    // Layer 0 leads the whole step, so there are no previous partials to fold
    // and `resid` is exactly embed_gather's output: the SP0 variant. Every
    // other layer folds the previous layer's `down` (S = 4).
    res_norm(layer == 0 ? 0u : Qwen35::shape(LinearId::Down).S,
             at(m_.layer_small[layer].norms, loader::kNormsOffInput));
    gemv(layer, LinearId::QkvZ, b_.x.ptr());
    gemv_bf16(layer, LinearId::AB, b_.x.ptr(), b_.ab_out);
    // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o)
    // - src/kernels/gdn_step.cl (Task 4), grid (48 v-heads, 4 state-column
    // chunks), WG 256. `conv_ring` and `state` are this GDN layer's slices;
    // `gdn_small` is the layer's GDN block base as a `const float*` (the
    // kernel finds conv/negA/dt_bias inside it at loader/small_layout.h's
    // offsets, passed to ocloc as NEGA_OFF/DTBIAS_OFF).
    {
      l0::Kernel& k = kernel(kernels::gdn_step_variant(kCapM), "gdn_step", kWgGdn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.ab_out.ptr());
      k.arg_ptr(3, m_.layer_small[layer].gdn.ptr());
      k.arg_ptr(4, at(b_.conv_ring, size_t(g) * kConvRingStride));
      k.arg_ptr(5, at(b_.gdn_state, size_t(g) * kGdnStateStride));
      k.arg_ptr(6, b_.gdn_o.ptr());
      launch(k, Qwen35::kGdnVHeads, kGdnStateChunks);
    }
    // prep_gated_head(qkvz_partials, gdn_o, gated_w, x_out) - prep.cl (Task 2),
    // grid (48 v-heads, M), WG 128. `gated_w` is the GDN block's RMSNormGated
    // weight: plain `w`, and bf16 - the kernel takes it as `const ushort*`, so
    // this is the one small-tensor binding that is not fp32.
    {
      l0::Kernel& k = kernel(kernels::prep_gated_head_variant(kCapM), "prep_gated_head", kWgGated);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.gdn_o.ptr());
      k.arg_ptr(2, at(m_.layer_small[layer].gdn, loader::kGdnOffGatedNorm));
      k.arg_ptr(3, b_.x.ptr());
      launch(k, Qwen35::kGdnVHeads, kCapM);
    }
    gemv(layer, LinearId::OutProj, b_.x.ptr());
    mlp(layer, Qwen35::shape(LinearId::OutProj).S);
  }

  // A full-attention layer (16 of 64): 10 kernels.
  void fa_layer(uint32_t layer, uint32_t f) {
    res_norm(Qwen35::shape(LinearId::Down).S,
             at(m_.layer_small[layer].norms, loader::kNormsOffInput));
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
      l0::Kernel& k = kernel(kernels::attn_prep_variant(kCapM), "attn_prep", kWgAttn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, m_.layer_small[layer].gdn.ptr());
      k.arg_ptr(3, m_.rope.ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kk);
      k.arg_ptr(7, vv);
      launch(k, Qwen35::kFaQHeads + Qwen35::kFaKvHeads, kCapM);
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
    {
      l0::Kernel& k = kernel(
          kernels::attn_decode_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock),
          "attn_decode", kWgAttn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kk);
      k.arg_ptr(3, vv);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, Qwen35::kFaKvHeads, b_.max_len / DecodeBuffers::kAttnBlock);
    }
    // attn_reduce(ctrl, attn_part, attn_gate, attn_out) - attn.cl (Task 5),
    // grid (24 q-heads, M), WG 256.
    {
      l0::Kernel& k =
          kernel(kernels::attn_reduce_variant(kCapM, b_.max_len, DecodeBuffers::kAttnBlock),
                 "attn_reduce", kWgAttn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, Qwen35::kFaQHeads, kCapM);
    }
    // o_proj is the one GEMV whose activations are not the shared `x` scratch:
    // attn_reduce writes bf16 [M][6144] into `attn_out`, which is exactly this
    // GEMV's K.
    gemv(layer, LinearId::OProj, b_.attn_out.ptr());
    mlp(layer, Qwen35::shape(LinearId::OProj).S);
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
    res_norm(Qwen35::shape(LinearId::Down).S, m_.final_norm.ptr());
    if (m_.linears.at({loader::kTopLevel, LinearId::LmHead}).kind == model::WeightKind::Int4)
      gemv(loader::kTopLevel, LinearId::LmHead, b_.x.ptr(), b_.logits);
    else
      gemv_bf16(loader::kTopLevel, LinearId::LmHead, b_.x.ptr(), b_.logits);
    // argmax_stage1(logits, part) - src/kernels/argmax.cl (Task 3), grid
    // (kVocab/1024 = 243, M), WG 256.
    {
      l0::Kernel& k = kernel(kernels::argmax_stage1_variant(kCapM), "argmax_stage1", kWgArgmax);
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

  // The layer a launch belongs to, for its label. `kBoundary` is the six
  // launches outside the layer loop (embed_gather, the final norm's
  // prep_res_fold + prep_norm_finish, lm_head and the two argmax stages).
  // It was five until spec 1.5's lever L1 split that norm.
  static constexpr int kBoundary = -1;

  l0::Context& ctx_;
  const loader::LoadedModel& m_;
  DecodeBuffers& b_;
  l0::Mem* tap_;
  ProfileEvents* prof_;
  // The list's M (rows in flight). Named like the constant it replaced so the
  // binding sites read unchanged; set once, before the walk.
  const uint32_t kCapM;
  CapturedStep step_;
  size_t kv_stride_ = 0;
  int layer_ = kBoundary;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

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

}  // namespace runtime
