#include "runtime/capture.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/kernels.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/control.h"

// capture.cc - the 645-kernel decode step, bound once.
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
// stream and only `prep_res_norm` advances it: it folds the previous GEMV's
// split-K `partials` into it, then writes the normalised row to the shared
// bf16 scratch `x`. Every int4 GEMV reads `x` (or, for o_proj, `attn_out`) and
// writes `partials`; every prep reads `partials` and writes `x`. So a layer is
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

// The captured list's M. src/kernels/CMakeLists.txt compiles M = 1 only in this
// plan (spec §9: the M loop exists in every kernel, M = 1 is what ships), while
// `DecodeBuffers::kM` = 8 is the *allocation* capacity. Every kernel indexes
// its buffers with its own compiled M, so this constant and the variant names
// below must move together - and the M = 2 attention variants deliberately do
// not exist at max_len 16384 (src/kernels/CMakeLists.txt), so raising this is
// the MTP task's job, not a one-line edit.
constexpr uint32_t kCapM = 1;

constexpr size_t kBf16 = 2, kFp32 = 4;

// --- work-group sizes, one per kernel, from each kernel's own contract -------
constexpr uint32_t kWgEmbed = 256;    // embed_gather.cl WG_EMBED  (Task 3)
constexpr uint32_t kWgGemv = 64;      // gemv.cl / gemv_bf16.cl WG_N = 16x4 (plan 1 §9.2)
constexpr uint32_t kWgResNorm = 256;  // prep.cl WG_RES            (Task 2)
constexpr uint32_t kWgSilu = 256;     // prep.cl WG_SILU           (Task 2)
constexpr uint32_t kWgGated = 128;    // prep.cl WG_GATED          (Task 2)
constexpr uint32_t kWgGdn = 256;      // gdn_step.cl WG_GDN        (Task 4)
constexpr uint32_t kWgAttn = 256;     // attn.cl WG_PREP/WG_DEC/WG_RED (Task 5)
constexpr uint32_t kWgArgmax = 256;   // argmax.cl WG_ARGMAX       (Task 3)

// --- grid divisors that are a kernel's constant, not a model dimension ------
// gemv/gemv_bf16: a work-group is 4 subgroups of 16 lanes, one column each.
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
  Capture(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b, l0::Mem* tap)
      : ctx_(ctx), m_(m), b_(b), tap_(tap), step_{l0::CmdList::regular(ctx), 0, {}, {}} {}

  CapturedStep run() {
    kv_stride_ = size_t(b_.max_len) * Qwen35::kFaKvHeads * Qwen35::kFaHeadDim * kBf16;
    check_sizes();

    embed_gather();
    uint32_t gdn = 0, fa = 0;   // the two per-kind index maps: GDN 0..47, FA 0..15
    for (const model::LayerDesc& layer : Qwen35::layers()) {
      if (layer.kind == model::LayerKind::GDN)
        gdn_layer(layer.index, gdn++);
      else
        fa_layer(layer.index, fa++);
      tap(layer.index);
    }
    head();

    require(gdn == kGdnLayers && fa == kFaLayers, "layer kind counts are not 48 GDN / 16 FA");
    // Every Kernel this walk made was launched exactly once, so kernels[i] is
    // what launch i ran - which is the property that makes the vector a usable
    // record of the list rather than a lifetime bag.
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    step_.list.close();
    return std::move(step_);
  }

 private:
  // Preconditions this file can check before a single command is appended.
  void check_sizes() {
    // The attention variants bake MAXLEN (attn_part's stride); the grid comes
    // from b.max_len, and the RoPE table from the loader's. All three are one
    // number or the KV cache is read at the wrong stride.
    require(m_.max_len == b_.max_len, "model max_len " + std::to_string(m_.max_len) +
                                          " != buffers max_len " + std::to_string(b_.max_len));
    require(b_.max_len % DecodeBuffers::kAttnBlock == 0,
            "max_len must be a multiple of 256 (attn_decode's block grid)");
    require(b_.gdn_state.size() == kGdnStateStride * kGdnLayers, "gdn_state is not 48 slices");
    require(b_.conv_ring.size() == kConvRingStride * kGdnLayers, "conv_ring is not 48 slices");
    require(b_.kv_k.size() == kv_stride_ * kFaLayers, "kv_k is not 16 slices");
    require(b_.kv_v.size() == kv_stride_ * kFaLayers, "kv_v is not 16 slices");
    if (tap_)
      require(tap_->size() >= size_t(Qwen35::kLayers) * kCapM * Qwen35::kHidden * kBf16,
              "debug_resid is smaller than [64][M][5120] bf16");

    // Four `.cl` constants mirror three of the model table's split-K counts,
    // and nothing else connects the two. A GEMV's `partials` are fp32
    // `[S][M][N]`; at S = 1 that collapses to `[M][N]` and a token's row base is
    // `m·N`, which is what `gdn_step`, `attn_prep` and `prep_gated_head` assume.
    // `prep_silu_mul` is the other side of the same coin: it *does* loop the
    // slices, and it loops exactly SILU_S = 4 of them.
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
    require(Qwen35::shape(LinearId::GateUp).S == 4,
            "gate||up is no longer S=4, but prep.cl (SILU_S) bakes a 4-slice sum");
    require(Qwen35::shape(LinearId::Qkv).S == 1,
            "qkv is no longer S=1, but attn.cl (QKV_S) bakes S=1 into every "
            "partials load");
  }

  // One launch site: load (or reuse) the variant's device binary, make a fresh
  // Kernel for it and set its group size. A Kernel per site is not required -
  // arguments are captured at append, so one object could serve many launches -
  // but it keeps the list auditable (kernels[i] is what launch i ran) for the
  // cost of 645 handles.
  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules
               .emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant)))
               .first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    l0::Kernel& k = *step_.kernels.back();
    k.group_size(wg);
    return k;
  }
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    step_.list.launch(k, gx, gy);
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

  // prep_res_norm(partials, resid, norm_w, x_out) - src/kernels/prep.cl
  // (Task 2), grid (1, M), WG 256. `s_prev` is the split-K width of the GEMV
  // whose partials this call folds into the residual stream; the SP0 variant
  // folds nothing and does not read `partials` at all (layer 0, whose residual
  // stream is embed_gather's output). `norm_w` is fp32 `1 + w`
  // (loader/small_layout.h - the RMSNorm family is fp32 on device).
  void res_norm(uint32_t s_prev, const void* norm_w) {
    l0::Kernel& k = kernel(kernels::prep_res_norm_variant(kCapM, Qwen35::kHidden, s_prev),
                           "prep_res_norm", kWgResNorm);
    k.arg_ptr(0, b_.partials.ptr());
    k.arg_ptr(1, b_.resid.ptr());
    k.arg_ptr(2, norm_w);
    k.arg_ptr(3, b_.x.ptr());
    launch(k, 1, kCapM);
  }

  // gemv(w, scales, x, out) - src/kernels/gemv.cl (plan 1 §9.2), grid (N/64, S),
  // WG 64. Every int4 row of the model table is layout 1, whose tiled buffer
  // carries its f16 scales inline (`loader::load_linear` throws on any other
  // layout), so the `scales` argument is unread - bind the same allocation,
  // which is the plan-1 convention and keeps the argument count honest. (Both
  // parameters are `restrict`-qualified; the aliasing is unobservable because
  // `scales` is never dereferenced under LAYOUT 1.)
  void gemv(uint32_t layer, LinearId id, const void* x) {
    const model::FusedLinear& fl = Qwen35::linear(id);
    const model::GemvShape& s = fl.shape;
    require(fl.kind == model::WeightKind::Int4, "gemv bound to a bf16 table row");
    require(s.N % kGemvColsPerWg == 0, "gemv N is not a multiple of 64");
    const loader::DeviceWeight& w = m_.linears.at({layer, id});
    l0::Kernel& k =
        kernel(kernels::gemv_variant(kCapM, s.K, s.N, s.S, s.layout), "gemv", kWgGemv);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, w.mem.ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, b_.partials.ptr());
    launch(k, s.N / kGemvColsPerWg, s.S);
  }

  // gemv_bf16(w, x, out) - src/kernels/gemv_bf16.cl (plan 1 §9.2), grid (N/64),
  // WG 64. No split-K: it writes its fp32 output directly, not into `partials`.
  void gemv_bf16(uint32_t layer, LinearId id, const void* x, const l0::Mem& out) {
    const model::FusedLinear& fl = Qwen35::linear(id);
    const model::GemvShape& s = fl.shape;
    require(fl.kind == model::WeightKind::Bf16, "gemv_bf16 bound to an int4 table row");
    require(s.N % kGemvColsPerWg == 0, "gemv_bf16 N is not a multiple of 64");
    const loader::DeviceWeight& w = m_.linears.at({layer, id});
    l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kCapM, s.K, s.N), "gemv_bf16", kWgGemv);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, x);
    k.arg_ptr(2, out.ptr());
    launch(k, s.N / kGemvColsPerWg);
  }

  // The MLP half, identical in both layer kinds: post-norm folding the mixer's
  // partials, gate‖up, SiLU·mul, down. `mixer_s` is the split-K width of the
  // GEMV that produced those partials (out_proj / o_proj, both S = 16).
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
    // other layer folds the previous layer's `down` (S = 16).
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
    // (4 kv-heads, max_len/256 blocks), WG 256. The grid spans the whole cache
    // because a captured list cannot resize; the work-groups past the context
    // early-out on their first instruction.
    {
      l0::Kernel& k =
          kernel(kernels::attn_decode_variant(kCapM, b_.max_len), "attn_decode", kWgAttn);
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
          kernel(kernels::attn_reduce_variant(kCapM, b_.max_len), "attn_reduce", kWgAttn);
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
  void head() {
    res_norm(Qwen35::shape(LinearId::Down).S, m_.final_norm.ptr());
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

  l0::Context& ctx_;
  const loader::LoadedModel& m_;
  DecodeBuffers& b_;
  l0::Mem* tap_;
  CapturedStep step_;
  size_t kv_stride_ = 0;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m, DecodeBuffers& b,
                   l0::Mem* debug_resid) {
  return Capture(ctx, m, b, debug_resid).run();
}

}  // namespace runtime
