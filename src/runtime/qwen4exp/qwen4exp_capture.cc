#include "runtime/qwen4exp/qwen4exp_capture.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "loader/qwen4exp_layout.h"
#include "runtime/control.h"

// qwen4exp_capture.cc - one device's Qwen3.8-Flash-Next decode list (qwen4exp_capture.h has the walk in one
// table). Every binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/qwen4exp/*.cl and the reused sources.
namespace runtime::qwen4exp {
namespace {

namespace kq = kernels::qwen4exp;
using kq::HcSrc;

static_assert(kRouteWords == kq::route::kWords, "the route row's width moved");
static_assert(kListMax == kq::kListMax && kListRow == kq::kListRow && kCountWord == kq::kCountWord,
              "the selection row moved (runtime/qwen4exp/qwen4exp_sizes.h vs kernels/qwen4exp_kernels.h)");
static_assert(kAttnTgt == kq::kAttnTgt && kAttnPart == kq::kAttnPart && kIdxTail == kq::kTailSlots &&
                  kPleRing == kq::kPleRing && kPleConsts == kq::kPleConsts,
              "the attention slices / tail ring / PLE rings moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the Qwen3.8-Flash-Next decode binaries are M = 1 only");

constexpr uint32_t kWgGdn = 256, kGdnChunks = 4, kWgGated = 128, kWgAttnPrep = 256;

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part, Qwen4ExpBuffers& b,
       const StageLink* link, l0::Mem* tap, const l0::Mem* injected)
      : ctx_(ctx), m_(m), part_(part), d_(m.desc), b_(b), link_(link), tap_(tap), injected_(injected),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    const uint32_t dev = part_.device;
    if (dev == 0) {
      embed();
      src_ = HcSrc::Embed;
    }
    if (link_ && dev == 1) {
      handoff_in();
      src_ = HcSrc::None;
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) {
      layer_ = int(l);
      layer(l);
    }
    layer_ = -1;
    if (link_ && dev == 0) {
      combine(HcSrc::Y, false, nullptr);   // _Y_NN: the materialised H crosses
      handoff_out();
    }
    if (dev + 1 == b_.placement.devices) head();
    const PpHandoff h = link_ ? link_->mode : kDefaultPpHandoff;
    const size_t want = device_launches(d_, b_.placement, dev, b_.attn, h, injected_ != nullptr);
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s decode list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::qwen4exp::device_launches gives (" +
                                            q4_attn_name(b_.attn) + " attention" + (injected_ ? ", injected" : "") + ")");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    // The kernels' shapes are Qwen3.8-Flash-Next's (kernels::qwen4exp); a descriptor may differ only in layers.
    require(d_.hidden == kq::kHidden && d_.hc == kq::kHc && d_.hc_n() == kq::kHcN && d_.hc_low == kq::kHcLow &&
                d_.hc_down_rows() == kq::kHcDownN && d_.q_heads == kq::kQHeads && d_.kv_heads == kq::kKvHeads &&
                d_.head_dim == kq::kHd && d_.qkvg_n() == kq::kQkvgN && d_.idx_n() == kq::kIdxN &&
                d_.idx_compress == kq::kBlock && d_.block_topk() == kq::kTopBlocks && d_.experts == kq::kExperts &&
                d_.top_k == kq::kTopK && d_.moe_inter == kq::kInter && d_.shared_inter == kq::kInter &&
                d_.router_n() == kq::kRouterN && d_.ple_heads == kq::kPleHeads && d_.ple_dim == kq::kPleDim &&
                d_.ple_eos == kq::kPleEos && d_.vocab == kq::kVocab && d_.vocab_used == kq::kVocabUsed &&
                d_.gdn_v_heads == kq::kGdnHeads && d_.gdn_head == kq::kGdnHd && d_.qkvz_n() == kq::kQkvzN &&
                d_.ple_dilation == 3 && d_.ple_conv_taps == 4,
            "the descriptor is not Qwen3.8-Flash-Next's shape (" + d_.name + "); the kernels bake it");
    require(&b_.desc == &d_ || b_.desc.layers == d_.layers, "the buffers were sized for another descriptor");
    require(part_.device == b_.device, "the part is device " + std::to_string(part_.device) + ", the buffers device " +
                                           std::to_string(b_.device));
    require(part_.first == b_.placement.first(b_.device) && part_.end == b_.placement.end(b_.device) &&
                part_.layers.size() == part_.end - part_.first,
            "the loaded part's layers are not the placement's");
    require(b_.placement.devices == 1 || link_ != nullptr, "two devices need the hand-off's binding");
    require(b_.placement.devices == 2 || link_ == nullptr, "a hand-off on one device");
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    if (part_.device + 1 == b_.placement.devices)
      require(part_.final_mixer && part_.lm_head, "the last device holds no final mixer / head");
    if (loader::q4_device_has_qsa(d_, b_.placement, part_.device, false))
      require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the RoPE table is shorter than max_len");
    if (b_.holds_ple())
      require(m_.ple_device == part_.device && m_.ple.ptrs && m_.ple.q.size() == d_.ple_heads,
              "the PLE layer's device holds no PLE pointer table");
    const PersistentSizes ps = persistent_sizes(d_, b_.placement, b_.device, b_.max_len);
    require(b_.kv.size() == ps.kv && b_.idx_keys.size() == ps.idx_keys && b_.idx_tail.size() == ps.idx_tail &&
                b_.gdn_state.size() == ps.gdn_state && b_.ple.size() == ps.ple,
            "the persistent groups are not persistent_sizes'");
    const ScratchSizes ss = scratch_sizes(d_, b_.max_len, b_.attn);
    require(b_.partials.size() == ss.partials && b_.routes.size() == ss.routes && b_.list.size() == ss.list &&
                b_.scores.size() == ss.scores && b_.attn_part.size() == ss.attn_part,
            "the decode scratch is not runtime::qwen4exp::scratch_sizes'");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][10240] bf16");
    if (injected_) require(injected_->size() >= injected_bytes(d_), "the injected selection rows are short");
    const bool int4 = d_.forms.dense == model::Q4Form::Int4;
    if (int4)
      require(d_.linear(model::Q4LinearId::GdnQkvz).shape.S == 1 && d_.qkvg_s == 2 && d_.gdn_out_s == 4 && d_.o_s == 4,
              "the int4 cells moved from S1 / S2 / S4 / S4 (the names spell them)");
    require((d_.forms.dense == model::Q4Form::Int4) == (d_.forms.shared == model::Q4Form::Int4),
            "the dense and shared forms differ (21b reads them together: ours int4, Intel's bf16)");
    const bool int8 = part_.lm_head ? part_.lm_head->kind == model::WeightKind::Int8 : false;
    std::vector<std::string> need =
        kq::decode_variants(d_, int8, b_.attn == Q4Attn::Eager, b_.placement.devices == 2,
                            m_.ple.scale == loader::Q4PleScale::Bf16);
    if (link_ && link_->mode == PpHandoff::Peer) need.push_back(kernels::pp_handoff_variant());
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON");
  }

  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules.emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant))).first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    l0::Kernel& k = *step_.kernels.back();
    k.group_size(wg);
    pending_entry_ = entry;
    pending_variant_ = variant;
    return k;
  }
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    require(!pending_entry_.empty(), "launch() without a preceding kernel()");
    step_.list.launch(k, gx, gy, gz, nullptr);
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ +
                           " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  // embed_gather(ctrl, embed, x) - embed_gather.cl at hidden 2560, vocab 248320, grid (1, M).
  void embed() {
    l0::Kernel& k = kernel(kq::embed_variant(kM), "embed_gather", kq::kEmbedWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, part_.embed->ptr());
    k.arg_ptr(2, b_.x.ptr());
    launch(k, 1, kM);
  }

  // q4_hc_combine_norm(ctrl, H, src, inj, norm_w, xn), grid (4, M): folds what is pending (src_), then (norm)
  // the grouped norm with `w`. The tap of the layer whose output this materialises follows it.
  void combine(HcSrc src, bool norm, const float* w) {
    const void* sp = b_.H.ptr();   // _X reads nothing
    if (src == HcSrc::Embed) sp = b_.x.ptr();
    if (src == HcSrc::Slices) sp = b_.partials.ptr();
    if (src == HcSrc::Y) sp = b_.y.ptr();
    l0::Kernel& k = kernel(kq::hc_combine_norm_variant(kM, src, slices_, norm), "q4_hc_combine_norm", kq::kHcWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, b_.H.ptr());
    k.arg_ptr(2, sp);
    k.arg_ptr(3, b_.inj.ptr());
    k.arg_ptr(4, w ? static_cast<const void*>(w) : b_.partials.ptr());   // _NN reads no weight
    k.arg_ptr(5, b_.xn.ptr());
    launch(k, d_.hc, kM);
    if (src == HcSrc::Y && pending_layer_ >= 0) {   // layer pending_layer_'s output is H now
      if (tap_)
        step_.list.copy(at(*tap_, size_t(pending_layer_) * kM * d_.hc_n() * 2), b_.H.ptr(), size_t(kM) * d_.hc_n() * 2);
      pending_layer_ = -1;
    }
  }
  // A gated residual: combine_norm (what is pending) -> xn; gemv_bf16 down||inject -> down_f32; q4_hc_up_mix ->
  // x (the block input) and, with inject, inj.
  void hc(const l0::Mem& block, bool inject) {
    const loader::Q4HcOffsets o = loader::q4_hc_offsets(d_, inject);
    combine(src_, true, reinterpret_cast<const float*>(at(block, o.norm)));
    src_ = HcSrc::None;
    bf16_linear(at(block, o.down), d_.hc_n(), o.down_n, b_.xn.ptr(), b_.down_f32);
    l0::Kernel& k = kernel(kq::hc_up_mix_variant(kM, inject), "q4_hc_up_mix", kq::kUpMixWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, b_.down_f32.ptr());
    k.arg_ptr(2, at(block, o.up));
    k.arg_ptr(3, b_.xn.ptr());
    k.arg_ptr(4, b_.x.ptr());
    k.arg_ptr(5, b_.inj.ptr());
    launch(k, d_.hidden / 16, kM);
  }

  // gemv_bf16(w, x, out) over {K, N} tiles at `w`, grid N / cols.
  void bf16_linear(const void* w, uint32_t K, uint32_t N, const void* x, const l0::Mem& out) {
    require(out.size() >= size_t(kM) * N * 4, "a bf16 GEMV's output is smaller than [M][N]");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(N);
    l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kM, K, N, t), "gemv_bf16", t.cols * t.ksplit);
    k.arg_ptr(0, w);
    k.arg_ptr(1, x);
    k.arg_ptr(2, out.ptr());
    launch(k, N / t.cols);
  }
  // A dense linear into `out`: gemv(w, scales, x, out) grid (N / 64, S) for int4 layout 0, gemv_bf16 otherwise.
  // Returns the slices it wrote (S, 1 for bf16).
  uint32_t linear(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
    if (w.kind == model::WeightKind::Int4) {
      require(s.layout == 0 && w.scales, "a Qwen3.8-Flash-Next int4 linear is GPTQ layout 0 with its own scales");
      require(out.size() >= size_t(s.S) * kM * s.N * 4, "the GEMV's output is smaller than [S][M][N]");
      l0::Kernel& k = kernel(kernels::gemv_variant(kM, s.K, s.N, s.S, s.layout), "gemv", 64);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, w.scales->ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out.ptr());
      launch(k, s.N / 64, s.S);
      return s.S;
    }
    require(w.kind == model::WeightKind::Bf16, "gemv_bf16 bound to a non-bf16 weight");
    bf16_linear(w.mem.ptr(), s.K, s.N, x, out);
    return 1;
  }

  void layer(uint32_t l) {
    const loader::Q4Layer& L = part_.layer(l);
    require(L.hc_attn && L.hc_mlp && L.router && L.gate_up && L.down && L.shared,
            "layer " + std::to_string(l) + " is not fully loaded");
    if (l == d_.ple_layer) ple(L);
    hc(*L.hc_attn, true);
    if (d_.is_qsa(l))
      qsa(l, L);
    else
      gdn(l, L);
    hc(*L.hc_mlp, true);
    moe(l, L);
    pending_layer_ = int(l);
  }

  // The PLE layer's prologue (spec 21 §2.4: `H = H + ple(H)` before the attn side): materialise H (unless it
  // landed materialised), the gather from host USM, the key||value projections, the block.
  void ple(const loader::Q4Layer& L) {
    require(bool(L.ple), "the PLE layer holds no PLE block");
    if (src_ != HcSrc::None) {
      require(src_ == HcSrc::Y, "the PLE prologue follows a MoE block");
      combine(HcSrc::Y, false, nullptr);   // _Y_NN
    }
    const bool bf16s = m_.ple.scale == loader::Q4PleScale::Bf16;
    {
      // q4_ple_gather(ctrl, ids_ring, ptrs, consts, e, ids_out), grid (16 heads, M)
      l0::Kernel& k = kernel(kq::ple_gather_variant(kM, bf16s), "q4_ple_gather", kq::kPleWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.ple_ids_ring());
      k.arg_ptr(2, m_.ple.ptrs->ptr());
      k.arg_ptr(3, b_.ple_consts.ptr());
      k.arg_ptr(4, b_.ple_e.ptr());
      k.arg_ptr(5, b_.ple_ids.ptr());
      launch(k, d_.ple_heads, kM);
    }
    const loader::Q4PleOffsets o = loader::q4_ple_offsets(d_);
    bf16_linear(at(*L.ple, o.kv), d_.ple_e(), d_.ple_kv_n(), b_.ple_e.ptr(), b_.ple_kv);
    {
      // q4_ple_block(ctrl, H, kv, pw, conv_ring), grid (4 streams, M); pw at norm_key (norm_query, norm_conv, the
      // taps follow it: Q4PleOffsets)
      require(o.norm_query == o.norm_key + size_t(d_.hc_n()) * 4 && o.norm_conv == o.norm_query + size_t(d_.hc_n()) * 4 &&
                  o.conv == o.norm_conv + size_t(d_.hc_n()) * 4,
              "the PLE block's norms / taps are not contiguous after norm_key (q4_ple.cl indexes them so)");
      l0::Kernel& k = kernel(kq::ple_block_variant(kM), "q4_ple_block", kq::kPleBlockWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.H.ptr());
      k.arg_ptr(2, b_.ple_kv.ptr());
      k.arg_ptr(3, at(*L.ple, o.norm_key));
      k.arg_ptr(4, b_.ple_conv_ring());
      launch(k, d_.hc, kM);
    }
    src_ = HcSrc::None;
  }

  void gdn(uint32_t l, const loader::Q4Layer& L) {
    require(L.gdn_qkvz && L.gdn_ab && L.gdn_out && L.gdn_small, "GDN layer " + std::to_string(l) + " is not loaded");
    require(L.gdn_qkvz->shape.N == d_.qkvz_n() && L.gdn_qkvz->shape.S == 1, "qkv||z is not [16384] at S 1");
    linear(*L.gdn_qkvz, b_.x.ptr(), b_.partials);
    linear(*L.gdn_ab, b_.x.ptr(), b_.ab);
    {
      // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o), grid (48 heads, 4 chunks)
      l0::Kernel& k = kernel(kq::gdn_variant(kM), "gdn_step", kWgGdn);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.ab.ptr());
      k.arg_ptr(3, L.gdn_small->ptr());
      k.arg_ptr(4, b_.conv_ring_layer(l));
      k.arg_ptr(5, b_.gdn_state_layer(l));
      k.arg_ptr(6, b_.gdn_o.ptr());
      launch(k, d_.gdn_v_heads, kGdnChunks);
    }
    {
      // prep_gated_head(qkvz_partials, gdn_o, gated_w, x_out) - the sigmoid gate (_SIG), grid (48, M)
      l0::Kernel& k = kernel(kq::gated_head_sig_variant(kM), "prep_gated_head", kWgGated);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.gdn_o.ptr());
      k.arg_ptr(2, at(*L.gdn_small, loader::q4_gdn_small(d_).gdn_off_gated_norm));
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, d_.gdn_v_heads, kM);
    }
    slices_ = linear(*L.gdn_out, b_.attn_out.ptr(), b_.partials);
    src_ = HcSrc::Slices;
  }

  void qsa(uint32_t l, const loader::Q4Layer& L) {
    require(L.qsa_qkvg && L.qsa_idx && L.qsa_o && L.qsa_small, "QSA layer " + std::to_string(l) + " is not loaded");
    require(L.qsa_qkvg->shape.N == d_.qkvg_n() && L.qsa_idx->shape.N == d_.idx_n(), "QSA layer " + std::to_string(l) +
                                                                                       "'s projections");
    const uint32_t qi = d_.qsa_before(l);
    const uint32_t S = linear(*L.qsa_qkvg, b_.x.ptr(), b_.partials);
    linear(*L.qsa_idx, b_.x.ptr(), b_.idx_f32);
    void* kv_k = b_.k_layer(l);
    void* kv_v = b_.v_layer(l);
    {
      // attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v), grid (24 + 2, M)
      l0::Kernel& k = kernel(kq::attn_prep_q4_variant(kM, S), "attn_prep", kWgAttnPrep);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, L.qsa_small->ptr());
      k.arg_ptr(3, part_.rope->ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, kv_k);
      k.arg_ptr(7, kv_v);
      launch(k, d_.q_heads + d_.kv_heads, kM);
    }
    const std::string qv = kq::qsa_variant(kM);
    {
      // q4_qsa_prep(ctrl, idx_f32, small, rope, idx_q, tail, idx_keys), grid (5, M)
      l0::Kernel& k = kernel(qv, "q4_qsa_prep", kq::kPrepWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.idx_f32.ptr());
      k.arg_ptr(2, L.qsa_small->ptr());
      k.arg_ptr(3, part_.rope->ptr());
      k.arg_ptr(4, b_.idx_q.ptr());
      k.arg_ptr(5, b_.tail_layer(l));
      k.arg_ptr(6, b_.idx_keys_layer(l));
      launch(k, d_.idx_heads + 1, kM);
    }
    const uint32_t stride = b_.max_len / d_.idx_compress;
    void* listp = injected_ ? at(*injected_, list_at(qi)) : at(b_.list, list_at(qi));
    if (!injected_) {
      {
        // q4_qsa_score(ctrl, idx_q, idx_keys, scores, stride), grid (ceil(max_len / 4 / 256), M); work-groups past
        // the row's complete blocks exit on their first instruction
        l0::Kernel& k = kernel(qv, "q4_qsa_score", kq::kScoreWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.idx_q.ptr());
        k.arg_ptr(2, b_.idx_keys_layer(l));
        k.arg_ptr(3, b_.scores.ptr());
        k.arg<uint32_t>(4, stride);
        launch(k, (stride + kq::kScoreWg - 1) / kq::kScoreWg, kM);
      }
      {
        // q4_qsa_select(ctrl, scores, stride, list, diag), grid (1, M)
        l0::Kernel& k = kernel(qv, "q4_qsa_select", kq::kSelectWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.scores.ptr());
        k.arg<uint32_t>(2, stride);
        k.arg_ptr(3, listp);
        k.arg_ptr(4, at(b_.diag, diag_at(qi)));
        launch(k, 1, kM);
      }
    }
    if (b_.attn == Q4Attn::Eager) {
      // q4_qsa_attn_eager(ctrl, attn_q, attn_gate, kv_k, kv_v, list, attn_out), grid (24, M)
      l0::Kernel& k = kernel(kq::qsa_attn_variant(kM, true), "q4_qsa_attn_eager", kq::kEagerWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, kv_k);
      k.arg_ptr(4, kv_v);
      k.arg_ptr(5, listp);
      k.arg_ptr(6, b_.attn_out.ptr());
      launch(k, d_.q_heads, kM);
    } else {
      const std::string av = kq::qsa_attn_variant(kM, false);
      {
        // q4_qsa_attn(ctrl, attn_q, kv_k, kv_v, list, part), grid (2 kv heads, TGT, M)
        l0::Kernel& k = kernel(av, "q4_qsa_attn", kq::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_q.ptr());
        k.arg_ptr(2, kv_k);
        k.arg_ptr(3, kv_v);
        k.arg_ptr(4, listp);
        k.arg_ptr(5, b_.attn_part.ptr());
        launch(k, d_.kv_heads, kq::kAttnTgt, kM);
      }
      {
        // q4_qsa_reduce(ctrl, part, attn_gate, list, attn_out), grid (24, M)
        l0::Kernel& k = kernel(av, "q4_qsa_reduce", kq::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_part.ptr());
        k.arg_ptr(2, b_.attn_gate.ptr());
        k.arg_ptr(3, listp);
        k.arg_ptr(4, b_.attn_out.ptr());
        launch(k, d_.q_heads, kM);
      }
    }
    slices_ = linear(*L.qsa_o, b_.attn_out.ptr(), b_.partials);
    src_ = HcSrc::Slices;
  }

  void moe(uint32_t l, const loader::Q4Layer& L) {
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden, "the router is not [528][2560]");
    linear(*L.router, b_.x.ptr(), b_.logits_r);
    void* rt = at(b_.routes, route_at(l));
    {
      l0::Kernel& k = kernel(kq::route_variant(kM), "q4_route", kq::kRouteLanes);
      k.arg_ptr(0, b_.logits_r.ptr());
      k.arg_ptr(1, rt);
      launch(k, 1, kM);
    }
    const bool shb = d_.forms.shared == model::Q4Form::Bf16;
    const std::string mv = kq::moe_variant(kM, shb);
    {
      // q4_moe_gate_up(route, x, w_gu, w_sh_gu, h), grid (11 x 20, M)
      l0::Kernel& k = kernel(mv, "q4_moe_gate_up", kq::moe_gate_up_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, L.gate_up->ptr());
      k.arg_ptr(3, L.shared->ptr());
      k.arg_ptr(4, b_.moe_h.ptr());
      launch(k, kq::moe_gate_up_groups(), kM);
    }
    {
      // q4_moe_down(route, h, w_dn, w_sh_dn, y), grid (2560 / 16, M) - y is NOT folded: the next combine does
      l0::Kernel& k = kernel(mv, "q4_moe_down", kq::moe_down_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_h.ptr());
      k.arg_ptr(2, L.down->ptr());
      k.arg_ptr(3, at(*L.shared, L.shared_down_offset));
      k.arg_ptr(4, b_.y.ptr());
      launch(k, d_.hidden / 16, kM);
    }
    src_ = HcSrc::Y;
  }

  void head() {
    const loader::Q4HcOffsets o = loader::q4_hc_offsets(d_, false);
    require(src_ == HcSrc::Y, "the final mixer follows a MoE block");
    combine(HcSrc::Y, true, reinterpret_cast<const float*>(at(*part_.final_mixer, o.norm)));
    bf16_linear(at(*part_.final_mixer, o.down), d_.hc_n(), o.down_n, b_.xn.ptr(), b_.down_f32);
    {
      l0::Kernel& k = kernel(kq::hc_up_mix_variant(kM, false), "q4_hc_up_mix", kq::kUpMixWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.down_f32.ptr());
      k.arg_ptr(2, at(*part_.final_mixer, o.up));
      k.arg_ptr(3, b_.xn.ptr());
      k.arg_ptr(4, b_.x.ptr());
      k.arg_ptr(5, b_.inj.ptr());
      launch(k, d_.hidden / 16, kM);
    }
    const loader::DeviceWeight& lm = *part_.lm_head;
    require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden, "lm_head is not [vocab][hidden]");
    if (lm.kind == model::WeightKind::Int8) {
      require(bool(lm.scales), "the int8 lm_head has no row scales");
      l0::Kernel& k = kernel(kernels::gemv_i8w_variant(kM, lm.shape.K, lm.shape.N), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, b_.logits.ptr());
      launch(k, lm.shape.N / kernels::kGemvI8wCols);
    } else {
      linear(lm, b_.x.ptr(), b_.logits);
    }
    require(kq::argmax_groups() <= kq::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
    {
      l0::Kernel& k = kernel(kq::argmax1_variant(kM), "argmax_stage1", kq::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kq::argmax_groups(), kM);
    }
    {
      l0::Kernel& k = kernel(kq::argmax2_variant(), "argmax_stage2", kq::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  // --- the hand-off (spec 16b's; Kolibri's capture with H and no norm sums) -------------------------------------
  size_t h_bytes() const { return size_t(kM) * d_.hc_n() * 2; }
  void check_link() const {
    require(link_->land_resid && link_->resid_cap >= h_bytes() && link_->sumsq_cap == 0,
            "the landing buffer does not hold " + std::to_string(h_bytes()) + " B of H and no norm sums");
    if (link_->mode == PpHandoff::Copy) require(link_->event != nullptr, "the copy hand-off needs its cross-device event");
  }
  void handoff_out() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.copy(link_->land_resid, b_.H.ptr(), h_bytes());
      step_.list.barrier_signal(link_->event);
      return;
    }
    require(link_->flag && link_->send_seq && link_->land_sumsq, "the peer hand-off needs its flag, counter and stamp");
    // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words): 0 sum words - the stamp
    // lands at land_sumsq[0]
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_send", kernels::kPpHandoffWg);
    k.arg_ptr(0, b_.H.ptr());
    k.arg_ptr(1, b_.H.ptr());
    k.arg_ptr(2, link_->land_resid);
    k.arg_ptr(3, link_->land_sumsq);
    k.arg_ptr(4, link_->flag);
    k.arg_ptr(5, link_->send_seq);
    k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
    k.arg<uint32_t>(7, 0u);
    launch(k, 1);
  }
  void handoff_in() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.wait_event(link_->event);
      step_.list.copy(b_.H.ptr(), link_->land_resid, h_bytes());
      return;
    }
    require(link_->flag && link_->recv_state && link_->spin_limit > 0 && link_->land_sumsq,
            "the peer hand-off needs its flag, its state words and a spin bound");
    // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words, spin_limit)
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_recv", kernels::kPpHandoffWg);
    k.arg_ptr(0, link_->land_resid);
    k.arg_ptr(1, link_->land_sumsq);
    k.arg_ptr(2, link_->flag);
    k.arg_ptr(3, link_->recv_state);
    k.arg_ptr(4, b_.H.ptr());
    k.arg_ptr(5, b_.H.ptr());
    k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
    k.arg<uint32_t>(7, 0u);
    k.arg<uint32_t>(8, link_->spin_limit);
    launch(k, 1);
  }

  l0::Context& ctx_;
  const loader::Q4LoadedModel& m_;
  const loader::Q4DevicePart& part_;
  const model::Qwen4ExpDesc& d_;
  Qwen4ExpBuffers& b_;
  const StageLink* link_;
  l0::Mem* tap_;
  const l0::Mem* injected_;
  CapturedStep step_;
  HcSrc src_ = HcSrc::None;    // what the next combine folds
  uint32_t slices_ = 1;        // the pending mixer GEMV's split-K (HcSrc::Slices)
  int layer_ = -1, pending_layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part,
                   Qwen4ExpBuffers& b, const StageLink* link, l0::Mem* tap, const l0::Mem* injected) {
  return Walk(ctx, m, part, b, link, tap, injected).run();
}

}  // namespace runtime::qwen4exp
