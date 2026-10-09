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
// table) and, spec 21e, spec 8's verify and draft lists. Every binding site names the kernel it honours; argument
// orders are each __kernel's, in src/kernels/qwen4exp/*.cl and the reused sources.
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
static_assert(kVerifyRows == kq::kMtpVerifyRows, "the verify rows moved (sizes vs kernels)");
static_assert(offsetof(Control, gdn_live) == 19 * 4, "gdn_step's CTRL_LIVE (19) is not Control::gdn_live");

constexpr uint32_t kWgGdn = 256, kGdnChunks = 4, kWgGated = 128, kWgAttnPrep = 256;

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part, Qwen4ExpBuffers& b,
       const StageLink* link, l0::Mem* tap, const l0::Mem* injected, const ListSpec& spec, const MtpBinding* mtp)
      : ctx_(ctx), m_(m), part_(part), d_(m.desc), b_(b), link_(link), tap_(tap), injected_(injected), spec_(spec),
        mtp_(mtp), M_(spec.M), ctrl_(b.control.ptr()), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    const uint32_t dev = part_.device;
    const bool last = dev + 1 == b_.placement.devices;
    if (spec_.kind == ListKind::Draft) {
      draft_step();
    } else {
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
      if (last) head();
      if (last && spec_.kind == ListKind::Verify) head_kv_pass();
    }
    const PpHandoff h = link_ ? link_->mode : kDefaultPpHandoff;
    size_t want = 0;
    switch (spec_.kind) {
      case ListKind::Decode: want = device_launches(d_, b_.placement, dev, b_.attn, h, injected_ != nullptr); break;
      case ListKind::Verify: want = verify_device_launches(d_, b_.placement, dev, M_, b_.attn, h); break;
      case ListKind::Draft: want = draft_launches(b_.attn, spec_.select); break;
    }
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s " + kind_name() + " list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::qwen4exp gives (" + q4_attn_name(b_.attn) +
                                            " attention" + (injected_ ? ", injected" : "") + ")");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

 private:
  std::string kind_name() const {
    switch (spec_.kind) {
      case ListKind::Decode: return "decode";
      case ListKind::Verify: return "verify (M = " + std::to_string(M_) + ")";
      case ListKind::Draft: return "draft step " + std::to_string(spec_.draft);
    }
    return "?";
  }

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
    // A draft list runs on the last device alone (no hand-off); the decode and verify lists cross on two cards.
    if (spec_.kind != ListKind::Draft)
      require(b_.placement.devices == 1 || link_ != nullptr, "two devices need the hand-off's binding");
    require(b_.placement.devices == 2 || link_ == nullptr, "a hand-off on one device");
    require(spec_.kind != ListKind::Draft || link_ == nullptr, "a draft list with a hand-off");
    const bool last = part_.device + 1 == b_.placement.devices;
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    if (last) require(part_.final_mixer && part_.lm_head, "the last device holds no final mixer / head");
    if (loader::q4_device_has_qsa(d_, b_.placement, part_.device, false))
      require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the RoPE table is shorter than max_len");
    if (b_.holds_ple())
      require(m_.ple_device == part_.device && m_.ple.ptrs && m_.ple.q.size() == d_.ple_heads,
              "the PLE layer's device holds no PLE pointer table");
    const PersistentSizes ps = persistent_sizes(d_, b_.placement, b_.device, b_.max_len);
    require(b_.kv.size() == ps.kv && b_.idx_keys.size() == ps.idx_keys && b_.idx_tail.size() == ps.idx_tail &&
                b_.gdn_state.size() == ps.gdn_state && b_.ple.size() == ps.ple,
            "the persistent groups are not persistent_sizes'");
    const ScratchSizes ss = scratch_sizes(d_, b_.max_len, b_.attn, b_.rows);
    require(b_.partials.size() == ss.partials && b_.routes.size() == ss.routes && b_.list.size() == ss.list &&
                b_.scores.size() == ss.scores && b_.attn_part.size() == ss.attn_part && b_.logits.size() == ss.logits,
            "the decode scratch is not runtime::qwen4exp::scratch_sizes'");
    require(M_ >= 1 && M_ <= b_.rows, "a " + kind_name() + " list of " + std::to_string(M_) + " rows over buffers of " +
                                          std::to_string(b_.rows));
    if (tap_) require(spec_.kind == ListKind::Decode && tap_->size() >= tap_bytes(d_), "the tap is a decode list's, [layers][M][10240]");
    if (injected_)
      require(spec_.kind == ListKind::Decode && injected_->size() >= injected_bytes(d_), "the injected rows are a decode list's");
    const bool int4 = d_.forms.dense == model::Q4Form::Int4;
    if (int4)
      require(d_.linear(model::Q4LinearId::GdnQkvz).shape.S == 1 && d_.qkvg_s == 2 && d_.gdn_out_s == 4 && d_.o_s == 4,
              "the int4 cells moved from S1 / S2 / S4 / S4 (the names spell them)");
    require((d_.forms.dense == model::Q4Form::Int4) == (d_.forms.shared == model::Q4Form::Int4),
            "the dense and shared forms differ (21b reads them together: ours int4, Intel's bf16)");
    const bool int8 = part_.lm_head ? part_.lm_head->kind == model::WeightKind::Int8 : false;
    const bool bf16s = m_.ple.scale == loader::Q4PleScale::Bf16, eager = b_.attn == Q4Attn::Eager;
    std::vector<std::string> need;
    if (spec_.kind != ListKind::Decode) {
      require(mtp_ && mtp_->bufs && mtp_->embed, "a " + kind_name() + " list without the MTP head's binding");
      require(b_.rows == kVerifyRows && b_.gdn_spec, "the MTP lists need the buffers' verify rows and GDN slots");
      if (b_.holds_ple()) require(b_.ple_gated && b_.ple_gn, "the PLE layer's device holds no verify PLE rows");
      if (last) {
        require(part_.mtp && part_.mtp_fc && part_.mtp_mixer, "the last device holds no MTP head (load with mtp)");
        const loader::Q4Layer& H = *part_.mtp;
        require(H.hc_attn && H.hc_mlp && H.qsa_qkvg && H.qsa_idx && H.qsa_o && H.qsa_small && H.router && H.gate_up &&
                    H.down && H.shared,
                "the MTP head's layer is not fully loaded");
        require(H.qsa_qkvg->kind == model::WeightKind::Bf16 && H.qsa_o->kind == model::WeightKind::Bf16,
                "the MTP head's dense projections are not bf16 (loader::q4_mtp_desc)");
        require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the head's RoPE table is short");
        require(mtp_->bufs->kv.size() == mtp_sizes(d_, b_.max_len).kv && mtp_->bufs->max_len == b_.max_len,
                "the head's buffers are not mtp_sizes' at this max_len");
      }
    }
    if (spec_.kind == ListKind::Decode)
      need = kq::decode_variants(d_, int8, eager, b_.placement.devices == 2, bf16s);
    else if (spec_.kind == ListKind::Verify)
      need = kq::verify_variants(d_, M_, int8, eager, bf16s, mtp_->single);
    else
      need = kq::draft_variants(int8, eager, mtp_->single);
    if (link_ && link_->mode == PpHandoff::Peer) need.push_back(kernels::pp_handoff_variant());
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON" +
                  (spec_.kind == ListKind::Decode ? "" : " and B70_MTP=ON (spec 21e)"));
    if (spec_.kind == ListKind::Draft) require(last, "a draft list off the last device");
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
    const std::string where = layer_ == -2 ? std::string("MTP") : layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_);
    step_.labels.push_back(where + " " + pending_entry_ + " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  // embed_gather(ctrl, embed, x) - embed_gather.cl at hidden 2560, vocab 248320, grid (1, M).
  void embed() {
    l0::Kernel& k = kernel(kq::embed_variant(M_), "embed_gather", kq::kEmbedWg);
    k.arg_ptr(0, ctrl_);
    k.arg_ptr(1, part_.embed->ptr());
    k.arg_ptr(2, b_.x.ptr());
    launch(k, 1, M_);
  }

  // q4_hc_combine_norm(ctrl, H, src, inj, norm_w, xn), grid (4, M): folds what is pending (src_), then (norm)
  // the grouped norm with `w`. The tap of the layer whose output this materialises follows it.
  void combine(HcSrc src, bool norm, const float* w) {
    const void* sp = b_.H.ptr();   // _X reads nothing
    if (src == HcSrc::Embed) sp = b_.x.ptr();
    if (src == HcSrc::Slices) sp = b_.partials.ptr();
    if (src == HcSrc::Y) sp = b_.y.ptr();
    l0::Kernel& k = kernel(kq::hc_combine_norm_variant(M_, src, slices_, norm), "q4_hc_combine_norm", kq::kHcWg);
    k.arg_ptr(0, ctrl_);
    k.arg_ptr(1, b_.H.ptr());
    k.arg_ptr(2, sp);
    k.arg_ptr(3, b_.inj.ptr());
    k.arg_ptr(4, w ? static_cast<const void*>(w) : b_.partials.ptr());   // _NN reads no weight
    k.arg_ptr(5, b_.xn.ptr());
    launch(k, d_.hc, M_);
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
    bf16_linear(at(block, o.down), d_.hc_n(), o.down_n, b_.xn.ptr(), b_.down_f32.ptr(), b_.down_f32.size());
    l0::Kernel& k = kernel(kq::hc_up_mix_variant(M_, inject), "q4_hc_up_mix", kq::kUpMixWg);
    k.arg_ptr(0, ctrl_);
    k.arg_ptr(1, b_.down_f32.ptr());
    k.arg_ptr(2, at(block, o.up));
    k.arg_ptr(3, b_.xn.ptr());
    k.arg_ptr(4, b_.x.ptr());
    k.arg_ptr(5, b_.inj.ptr());
    launch(k, d_.hidden / 16, M_);
  }

  // gemv_bf16(w, x, out) over {K, N} tiles at `w`, M rows (`rows`: M_ unless given), grid N / cols.
  void bf16_linear(const void* w, uint32_t K, uint32_t N, const void* x, void* out, size_t out_bytes, uint32_t rows = 0) {
    const uint32_t R = rows ? rows : M_;
    require(out_bytes >= size_t(R) * N * 4, "a bf16 GEMV's output is smaller than [M][N]");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(N);
    l0::Kernel& k = kernel(kernels::gemv_bf16_variant(R, K, N, t), "gemv_bf16", t.cols * t.ksplit);
    k.arg_ptr(0, w);
    k.arg_ptr(1, x);
    k.arg_ptr(2, out);
    launch(k, N / t.cols);
  }
  // A dense linear into `out`: gemv(w, scales, x, out) grid (N / 64, S) for int4 layout 0, gemv_bf16 otherwise.
  // Returns the slices it wrote (S, 1 for bf16).
  uint32_t linear(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
    if (w.kind == model::WeightKind::Int4) {
      require(s.layout == 0 && w.scales, "a Qwen3.8-Flash-Next int4 linear is GPTQ layout 0 with its own scales");
      require(out.size() >= size_t(s.S) * M_ * s.N * 4, "the GEMV's output is smaller than [S][M][N]");
      l0::Kernel& k = kernel(kernels::gemv_variant(M_, s.K, s.N, s.S, s.layout), "gemv", 64);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, w.scales->ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out.ptr());
      launch(k, s.N / 64, s.S);
      return s.S;
    }
    require(w.kind == model::WeightKind::Bf16, "gemv_bf16 bound to a non-bf16 weight");
    bf16_linear(w.mem.ptr(), s.K, s.N, x, out.ptr(), out.size());
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
    moe(L, at(b_.routes, b_.route_off(l)), d_.forms.shared == model::Q4Form::Bf16);
    pending_layer_ = int(l);
  }

  // The PLE layer's prologue (spec 21 §2.4: `H = H + ple(H)` before the attn side): materialise H (unless it
  // landed materialised), the gather from host USM, the key||value projections, the block (M = 1) or - a verify at
  // M > 1 - q4_pf_ple's gate / conv / ring at C = 4 (q4_ple_block is M = 1 only: a row's p - 3 may be this launch's).
  void ple(const loader::Q4Layer& L) {
    require(bool(L.ple), "the PLE layer holds no PLE block");
    if (src_ != HcSrc::None) {
      require(src_ == HcSrc::Y, "the PLE prologue follows a MoE block");
      combine(HcSrc::Y, false, nullptr);   // _Y_NN
    }
    const bool bf16s = m_.ple.scale == loader::Q4PleScale::Bf16;
    {
      // q4_ple_gather(ctrl, ids_ring, ptrs, consts, e, ids_out), grid (16 heads, M)
      l0::Kernel& k = kernel(kq::ple_gather_variant(M_, bf16s), "q4_ple_gather", kq::kPleWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.ple_ids_ring());
      k.arg_ptr(2, m_.ple.ptrs->ptr());
      k.arg_ptr(3, b_.ple_consts.ptr());
      k.arg_ptr(4, b_.ple_e.ptr());
      k.arg_ptr(5, b_.ple_ids.ptr());
      launch(k, d_.ple_heads, M_);
    }
    const loader::Q4PleOffsets o = loader::q4_ple_offsets(d_);
    bf16_linear(at(*L.ple, o.kv), d_.ple_e(), d_.ple_kv_n(), b_.ple_e.ptr(), b_.ple_kv.ptr(), b_.ple_kv.size());
    require(o.norm_query == o.norm_key + size_t(d_.hc_n()) * 4 && o.norm_conv == o.norm_query + size_t(d_.hc_n()) * 4 &&
                o.conv == o.norm_conv + size_t(d_.hc_n()) * 4,
            "the PLE block's norms / taps are not contiguous after norm_key (q4_ple.cl / q4_pf_ple.cl index them so)");
    const void* pw = at(*L.ple, o.norm_key);
    if (M_ == 1) {
      // q4_ple_block(ctrl, H, kv, pw, conv_ring), grid (4 streams, M)
      l0::Kernel& k = kernel(kq::ple_block_variant(1), "q4_ple_block", kq::kPleBlockWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.H.ptr());
      k.arg_ptr(2, b_.ple_kv.ptr());
      k.arg_ptr(3, pw);
      k.arg_ptr(4, b_.ple_conv_ring());
      launch(k, d_.hc, 1);
    } else {
      require(b_.ple_gated && b_.ple_gn, "the verify's PLE rows are not allocated");
      const std::string v = kq::pf_ple_verify_variant();
      // q4_pf_ple_gate(ctrl, kv, H, pw, gated, gn), grid (4 streams, M) - the kv rows at 12800, gemv_bf16's [M][N]
      {
        l0::Kernel& k = kernel(v, "q4_pf_ple_gate", kq::kPfPleWg);
        k.arg_ptr(0, ctrl_);
        k.arg_ptr(1, b_.ple_kv.ptr());
        k.arg_ptr(2, b_.H.ptr());
        k.arg_ptr(3, pw);
        k.arg_ptr(4, b_.ple_gated->ptr());
        k.arg_ptr(5, b_.ple_gn->ptr());
        launch(k, d_.hc, M_);
      }
      // q4_pf_ple_conv(ctrl, gated, gn, ring, pw, H), grid (10240 / 256, M)
      {
        l0::Kernel& k = kernel(v, "q4_pf_ple_conv", kq::kPfPleWg);
        k.arg_ptr(0, ctrl_);
        k.arg_ptr(1, b_.ple_gated->ptr());
        k.arg_ptr(2, b_.ple_gn->ptr());
        k.arg_ptr(3, b_.ple_conv_ring());
        k.arg_ptr(4, pw);
        k.arg_ptr(5, b_.H.ptr());
        launch(k, d_.hc_n() / kq::kPfPleWg, M_);
      }
      // q4_pf_ple_ring(ctrl, gn, ids, ring, id_ring), grid (10240 / 256, 16): the rows' gn into the conv ring and their
      // ids - Control's cur_token[0..M) (the gather wrote the same ids already)
      {
        l0::Kernel& k = kernel(v, "q4_pf_ple_ring", kq::kPfPleWg);
        k.arg_ptr(0, ctrl_);
        k.arg_ptr(1, b_.ple_gn->ptr());
        k.arg_ptr(2, at(b_.control, offsetof(Control, cur_token)));
        k.arg_ptr(3, b_.ple_conv_ring());
        k.arg_ptr(4, b_.ple_ids_ring());
        launch(k, d_.hc_n() / kq::kPfPleWg, kPleRing);
      }
    }
    src_ = HcSrc::None;
  }

  void gdn(uint32_t l, const loader::Q4Layer& L) {
    require(L.gdn_qkvz && L.gdn_ab && L.gdn_out && L.gdn_small, "GDN layer " + std::to_string(l) + " is not loaded");
    require(L.gdn_qkvz->shape.N == d_.qkvz_n() && L.gdn_qkvz->shape.S == 1, "qkv||z is not [16384] at S 1");
    linear(*L.gdn_qkvz, b_.x.ptr(), b_.partials);
    linear(*L.gdn_ab, b_.x.ptr(), b_.ab);
    const bool slots = spec_.kind == ListKind::Verify;
    {
      // gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o [, state_spec]), grid (48 heads, 4
      // chunks); the verify's slots build reads the live slot and writes row r's state into slot (live + r) % 4
      l0::Kernel& k = kernel(slots ? kq::gdn_slots_variant(M_) : kq::gdn_variant(M_), "gdn_step", kWgGdn);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.ab.ptr());
      k.arg_ptr(3, L.gdn_small->ptr());
      k.arg_ptr(4, b_.conv_ring_layer(l));
      k.arg_ptr(5, b_.gdn_state_layer(l));
      k.arg_ptr(6, b_.gdn_o.ptr());
      if (slots) k.arg_ptr(7, b_.gdn_spec_layer(l));
      launch(k, d_.gdn_v_heads, kGdnChunks);
    }
    {
      // prep_gated_head(qkvz_partials, gdn_o, gated_w, x_out) - the sigmoid gate (_SIG), grid (48, M)
      l0::Kernel& k = kernel(kq::gated_head_sig_variant(M_), "prep_gated_head", kWgGated);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.gdn_o.ptr());
      k.arg_ptr(2, at(*L.gdn_small, loader::q4_gdn_small(d_).gdn_off_gated_norm));
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, d_.gdn_v_heads, M_);
    }
    slices_ = linear(*L.gdn_out, b_.attn_out.ptr(), b_.partials);
    src_ = HcSrc::Slices;
  }

  // --- a QSA layer's pieces (the main layers' and the MTP head's) --------------------------------------------------
  struct QsaTarget {   // where a QSA layer's state lives
    void* k;
    void* v;
    void* tail;
    void* keys;
  };
  // q||gate||k||v and the indexer GEMVs; returns q||gate||k||v's split-K.
  uint32_t qsa_proj(const loader::Q4Layer& L) {
    require(L.qsa_qkvg->shape.N == d_.qkvg_n() && L.qsa_idx->shape.N == d_.idx_n(), "a QSA layer's projections");
    const uint32_t S = linear(*L.qsa_qkvg, b_.x.ptr(), b_.partials);
    linear(*L.qsa_idx, b_.x.ptr(), b_.idx_f32);
    return S;
  }
  // attn_prep (q / k norms, partial RoPE, the KV write, the gate) and q4_qsa_prep (the indexer's query heads, the raw
  // key into the tail ring, a completing block's compressed key).
  void qsa_prep(const loader::Q4Layer& L, uint32_t S, const QsaTarget& t) {
    {
      // attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v), grid (24 + 2, M)
      l0::Kernel& k = kernel(kq::attn_prep_q4_variant(M_, S), "attn_prep", kWgAttnPrep);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, L.qsa_small->ptr());
      k.arg_ptr(3, part_.rope->ptr());
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, b_.attn_gate.ptr());
      k.arg_ptr(6, t.k);
      k.arg_ptr(7, t.v);
      launch(k, d_.q_heads + d_.kv_heads, M_);
    }
    {
      // q4_qsa_prep(ctrl, idx_f32, small, rope, idx_q, tail, idx_keys), grid (5, M)
      l0::Kernel& k = kernel(kq::qsa_variant(M_), "q4_qsa_prep", kq::kPrepWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.idx_f32.ptr());
      k.arg_ptr(2, L.qsa_small->ptr());
      k.arg_ptr(3, part_.rope->ptr());
      k.arg_ptr(4, b_.idx_q.ptr());
      k.arg_ptr(5, t.tail);
      k.arg_ptr(6, t.keys);
      launch(k, d_.idx_heads + 1, M_);
    }
  }
  // q4_qsa_score + q4_qsa_select: every row's own selection into `listp` / `diagp`.
  void qsa_select(const QsaTarget& t, void* listp, void* diagp) {
    const uint32_t stride = b_.max_len / d_.idx_compress;
    const std::string qv = kq::qsa_variant(M_);
    {
      // q4_qsa_score(ctrl, idx_q, idx_keys, scores, stride), grid (ceil(max_len / 4 / 256), M); work-groups past
      // the row's complete blocks exit on their first instruction
      l0::Kernel& k = kernel(qv, "q4_qsa_score", kq::kScoreWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.idx_q.ptr());
      k.arg_ptr(2, t.keys);
      k.arg_ptr(3, b_.scores.ptr());
      k.arg<uint32_t>(4, stride);
      launch(k, (stride + kq::kScoreWg - 1) / kq::kScoreWg, M_);
    }
    {
      // q4_qsa_select(ctrl, scores, stride, list, diag), grid (1, M)
      l0::Kernel& k = kernel(qv, "q4_qsa_select", kq::kSelectWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.scores.ptr());
      k.arg<uint32_t>(2, stride);
      k.arg_ptr(3, listp);
      k.arg_ptr(4, diagp);
      launch(k, 1, M_);
    }
  }
  // The attention over each row's list (flash: q4_qsa_attn + q4_qsa_reduce; eager: q4_qsa_attn_eager) -> attn_out.
  void qsa_attend(const QsaTarget& t, const void* listp) {
    if (b_.attn == Q4Attn::Eager) {
      // q4_qsa_attn_eager(ctrl, attn_q, attn_gate, kv_k, kv_v, list, attn_out), grid (24, M)
      l0::Kernel& k = kernel(kq::qsa_attn_variant(M_, true), "q4_qsa_attn_eager", kq::kEagerWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, t.k);
      k.arg_ptr(4, t.v);
      k.arg_ptr(5, listp);
      k.arg_ptr(6, b_.attn_out.ptr());
      launch(k, d_.q_heads, M_);
      return;
    }
    const std::string av = kq::qsa_attn_variant(M_, false);
    {
      // q4_qsa_attn(ctrl, attn_q, kv_k, kv_v, list, part), grid (2 kv heads, TGT, M)
      l0::Kernel& k = kernel(av, "q4_qsa_attn", kq::kAttnWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, t.k);
      k.arg_ptr(3, t.v);
      k.arg_ptr(4, listp);
      k.arg_ptr(5, b_.attn_part.ptr());
      launch(k, d_.kv_heads, kq::kAttnTgt, M_);
    }
    {
      // q4_qsa_reduce(ctrl, part, attn_gate, list, attn_out), grid (24, M)
      l0::Kernel& k = kernel(av, "q4_qsa_reduce", kq::kAttnWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_gate.ptr());
      k.arg_ptr(3, listp);
      k.arg_ptr(4, b_.attn_out.ptr());
      launch(k, d_.q_heads, M_);
    }
  }

  void qsa(uint32_t l, const loader::Q4Layer& L) {
    require(L.qsa_qkvg && L.qsa_idx && L.qsa_o && L.qsa_small, "QSA layer " + std::to_string(l) + " is not loaded");
    const uint32_t qi = d_.qsa_before(l);
    const QsaTarget t{b_.k_layer(l), b_.v_layer(l), b_.tail_layer(l), b_.idx_keys_layer(l)};
    const uint32_t S = qsa_proj(L);
    qsa_prep(L, S, t);
    void* listp = injected_ ? at(*injected_, list_at(qi)) : at(b_.list, b_.list_off(qi));
    if (!injected_) qsa_select(t, listp, at(b_.diag, b_.diag_off(qi)));
    qsa_attend(t, listp);
    slices_ = linear(*L.qsa_o, b_.attn_out.ptr(), b_.partials);
    src_ = HcSrc::Slices;
  }

  // The 11-slot MoE over route row `rt`: the router GEMV (+ the shared gate), q4_route, gate||up, down -> y.
  void moe(const loader::Q4Layer& L, void* rt, bool shb) {
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden, "the router is not [528][2560]");
    linear(*L.router, b_.x.ptr(), b_.logits_r);
    {
      l0::Kernel& k = kernel(kq::route_variant(M_), "q4_route", kq::kRouteLanes);
      k.arg_ptr(0, b_.logits_r.ptr());
      k.arg_ptr(1, rt);
      launch(k, 1, M_);
    }
    const std::string mv = kq::moe_variant(M_, shb);
    {
      // q4_moe_gate_up(route, x, w_gu, w_sh_gu, h), grid (11 x 20, M)
      l0::Kernel& k = kernel(mv, "q4_moe_gate_up", kq::moe_gate_up_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, L.gate_up->ptr());
      k.arg_ptr(3, L.shared->ptr());
      k.arg_ptr(4, b_.moe_h.ptr());
      launch(k, kq::moe_gate_up_groups(), M_);
    }
    {
      // q4_moe_down(route, h, w_dn, w_sh_dn, y), grid (2560 / 16, M) - y is NOT folded: the next combine does
      l0::Kernel& k = kernel(mv, "q4_moe_down", kq::moe_down_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_h.ptr());
      k.arg_ptr(2, L.down->ptr());
      k.arg_ptr(3, at(*L.shared, L.shared_down_offset));
      k.arg_ptr(4, b_.y.ptr());
      launch(k, d_.hidden / 16, M_);
    }
    src_ = HcSrc::Y;
  }

  // A final mixer (the main model's or the head's: no inject) over what is pending (_Y), then the head GEMV into
  // `logits` and the two argmax stages (into ctrl_).
  void mixer_and_head(const l0::Mem& mixer, void* logits) {
    const loader::Q4HcOffsets o = loader::q4_hc_offsets(d_, false);
    require(src_ == HcSrc::Y, "the final mixer follows a MoE block");
    combine(HcSrc::Y, true, reinterpret_cast<const float*>(at(mixer, o.norm)));
    bf16_linear(at(mixer, o.down), d_.hc_n(), o.down_n, b_.xn.ptr(), b_.down_f32.ptr(), b_.down_f32.size());
    {
      l0::Kernel& k = kernel(kq::hc_up_mix_variant(M_, false), "q4_hc_up_mix", kq::kUpMixWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.down_f32.ptr());
      k.arg_ptr(2, at(mixer, o.up));
      k.arg_ptr(3, b_.xn.ptr());
      k.arg_ptr(4, b_.x.ptr());
      k.arg_ptr(5, b_.inj.ptr());
      launch(k, d_.hidden / 16, M_);
    }
    const loader::DeviceWeight& lm = *part_.lm_head;
    require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden, "lm_head is not [vocab][hidden]");
    if (lm.kind == model::WeightKind::Int8) {
      require(bool(lm.scales), "the int8 lm_head has no row scales");
      l0::Kernel& k = kernel(kernels::gemv_i8w_variant(M_, lm.shape.K, lm.shape.N), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, logits);
      launch(k, lm.shape.N / kernels::kGemvI8wCols);
    } else {
      require(lm.kind == model::WeightKind::Bf16, "lm_head is int8 or bf16");
      bf16_linear(lm.mem.ptr(), lm.shape.K, lm.shape.N, b_.x.ptr(), logits, size_t(M_) * lm.shape.N * 4);
    }
    require(kq::argmax_groups() <= kq::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
    {
      l0::Kernel& k = kernel(kq::argmax1_variant(M_), "argmax_stage1", kq::kArgmaxWg);
      k.arg_ptr(0, logits);
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kq::argmax_groups(), M_);
    }
    {
      l0::Kernel& k = kernel(kq::argmax2_variant(), "argmax_stage2", kq::kArgmaxWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
  }

  void head() { mixer_and_head(*part_.final_mixer, b_.logits.ptr()); }

  // --- spec 21e: the MTP head ------------------------------------------------------------------------------------
  // The fusion over M_ rows of R (`R`: [M][10240] bf16) and the head Control's tokens: q4_mtp_norm, fc_embedding at
  // M, fc_hidden at M = 4 a row, q4_mtp_fuse -> b.H (the head layer's input, materialised).
  void mtp_fuse(const void* R) {
    const Qwen4ExpMtpBuffers& mb = *mtp_->bufs;
    const loader::Q4MtpFcOffsets o = loader::q4_mtp_fc_offsets(d_);
    const l0::Mem& fc = *part_.mtp_fc;
    const std::string v = kq::mtp_variant(M_, mtp_->single);
    {
      // q4_mtp_norm(ctrl, R, embed, w_e, w_h, xe, xh), grid (1 + 4, M)
      l0::Kernel& k = kernel(v, "q4_mtp_norm", kq::kMtpWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, R);
      k.arg_ptr(2, mtp_->embed);
      k.arg_ptr(3, at(fc, o.norm_embedding));
      k.arg_ptr(4, at(fc, o.norm_hidden));
      k.arg_ptr(5, mb.xe.ptr());
      k.arg_ptr(6, mb.xh.ptr());
      launch(k, 1 + d_.hc, M_);
    }
    bf16_linear(at(fc, o.fc_embedding), d_.hidden, d_.hidden, mb.xe.ptr(), mb.fe.ptr(), mb.fe.size());
    for (uint32_t r = 0; r < M_; ++r)   // a row's 4 streams: one GEMV at M = 4
      bf16_linear(at(fc, o.fc_hidden), d_.hidden, d_.hidden, at(mb.xh, size_t(r) * d_.hc_n() * 2),
                  at(mb.fh, size_t(r) * d_.hc * d_.hidden * 4), size_t(d_.hc) * d_.hidden * 4, d_.hc);
    {
      // q4_mtp_fuse(ctrl, fe, fh, H), grid (10240 / 256, M)
      l0::Kernel& k = kernel(v, "q4_mtp_fuse", kq::kMtpWg);
      k.arg_ptr(0, ctrl_);
      k.arg_ptr(1, mb.fe.ptr());
      k.arg_ptr(2, mb.fh.ptr());
      k.arg_ptr(3, b_.H.ptr());
      launch(k, d_.hc_n() / kq::kMtpWg, M_);
    }
    src_ = HcSrc::None;   // H is materialised: the head layer's first combine is _X
  }

  // The verify list's head pass (last device): the M rows' pre-mixer H into hh rows 1..M, then the head on (hh[r],
  // the head Control's cur_token[r]) at positions pos - 1 + r - the attn side up to its K / V and indexer keys only.
  void head_kv_pass() {
    Qwen4ExpMtpBuffers& mb = *mtp_->bufs;
    step_.list.copy(mb.hh_row(1), b_.H.ptr(), size_t(M_) * d_.hc_n() * 2);   // R_{pos .. pos + M - 1}
    layer_ = -2;
    ctrl_ = mb.hctl.ptr();
    const loader::Q4Layer& L = *part_.mtp;
    mtp_fuse(mb.hh_row(0));
    hc(*L.hc_attn, true);
    const uint32_t S = qsa_proj(L);
    require(S == 1, "the head's q||gate||k||v is bf16 (S 1)");
    qsa_prep(L, S, QsaTarget{mb.k(), mb.v(), mb.idx_tail.ptr(), mb.idx_keys.ptr()});
    ctrl_ = b_.control.ptr();
    layer_ = -1;
  }

  // A draft step (last device, M = 1, the head Control).
  void draft_step() {
    Qwen4ExpMtpBuffers& mb = *mtp_->bufs;
    require(M_ == 1 && spec_.draft < kMaxDraft, "a draft step is M = 1, index < kMaxDraft");
    layer_ = -2;
    ctrl_ = mb.hctl.ptr();
    const loader::Q4Layer& L = *part_.mtp;
    mtp_fuse(spec_.draft == 0 ? mb.hh_row(0) : b_.H.ptr());
    hc(*L.hc_attn, true);
    const uint32_t S = qsa_proj(L);
    require(S == 1, "the head's q||gate||k||v is bf16 (S 1)");
    const QsaTarget t{mb.k(), mb.v(), mb.idx_tail.ptr(), mb.idx_keys.ptr()};
    qsa_prep(L, S, t);
    if (spec_.select) qsa_select(t, mb.list.ptr(), mb.diag.ptr());
    qsa_attend(t, mb.list.ptr());   // steps after a selecting one attend its list as it is (decision 5)
    slices_ = linear(*L.qsa_o, b_.attn_out.ptr(), b_.partials);
    src_ = HcSrc::Slices;
    hc(*L.hc_mlp, true);
    moe(L, at(mb.routes, size_t(spec_.draft) * kRouteWords * 4), true);   // bf16 shared (q4_mtp_desc)
    mixer_and_head(*part_.mtp_mixer, at(mb.logits, size_t(spec_.draft) * d_.vocab * 4));
    ctrl_ = b_.control.ptr();
    layer_ = -1;
  }

  // --- the hand-off (spec 16b's; Kolibri's capture with H and no norm sums) -------------------------------------
  size_t h_bytes() const { return size_t(M_) * d_.hc_n() * 2; }
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
  const ListSpec spec_;
  const MtpBinding* mtp_;
  const uint32_t M_;
  void* ctrl_;                 // the Control the kernels read: the device's, or the head's for its launches
  CapturedStep step_;
  HcSrc src_ = HcSrc::None;    // what the next combine folds
  uint32_t slices_ = 1;        // the pending mixer GEMV's split-K (HcSrc::Slices)
  int layer_ = -1, pending_layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part,
                   Qwen4ExpBuffers& b, const StageLink* link, l0::Mem* tap, const l0::Mem* injected, const ListSpec& spec,
                   const MtpBinding* mtp) {
  return Walk(ctx, m, part, b, link, tap, injected, spec, mtp).run();
}

}  // namespace runtime::qwen4exp
