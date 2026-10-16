#include "runtime/muse/muse_capture.h"

#include <cstddef>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "runtime/control.h"

// muse_capture.cc - one device's Muse-Glimmer decode list (muse_capture.h has the walk in one table). Every binding
// site names the kernel it honours; argument orders are each __kernel's, in src/kernels/muse/*.cl and the reused
// sources.
namespace runtime::muse {
namespace {

namespace km = kernels::muse;

static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the Muse binaries are M = 1 only (23f adds 2..16)");
static_assert(kNormG == km::kNormG && kAttnTgt == km::kAttnTgt && kAttnPart == km::kAttnPart &&
                  kArgmaxChunk == km::kArgmaxChunk,
              "runtime::muse's scratch geometry and kernels::muse disagree");

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }
const float* fat(const l0::Mem& m, size_t floats) { return static_cast<const float*>(m.ptr()) + floats; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::muse::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b,
       const StageLink* link, l0::Mem* tap, uint32_t M = 1, const VerifyBind* verify = nullptr)
      : ctx_(ctx), part_(part), d_(d), b_(b), link_(link), tap_(tap), verify_(verify), M_(M),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    const uint32_t dev = part_.device;
    if (dev == 0) embed();
    if (link_ && dev == 1) handoff_in();
    for (uint32_t l = part_.first; l < part_.end; ++l) {
      layer_ = int(l);
      layer(l);
      if (tap_) step_.list.copy(at(*tap_, size_t(l) * M_ * d_.hidden * 2), b_.resid.ptr(), size_t(M_) * d_.hidden * 2);
      if (verify_ && verify_->taps) drafter_taps(l);
    }
    layer_ = -1;
    if (link_ && dev == 0) handoff_out();
    if (dev + 1 == b_.placement.devices) head();
    const PpHandoff h = link_ ? link_->mode : kDefaultPpHandoff;
    const size_t want = device_launches(d_, b_.placement, dev, b_.attn, h, M_);
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s decode list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::muse::device_launches gives (" +
                                            muse_attn_name(b_.attn) + " attention)");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    // The kernels' shapes are Muse-Glimmer-30B's (kernels::muse); a descriptor may differ only in its layer count.
    require(d_.hidden == km::kHidden && d_.q_heads == km::kQHeads && d_.kv_heads == km::kKvHeads &&
                d_.head_dim == km::kHd && d_.qgkv_rows() == km::kQgkvN && d_.inter == km::kInter &&
                d_.vocab == km::kVocab && d_.window == km::kWindow && model::MuseGlimmerDesc::kRing == km::kRing,
            "the descriptor is not Muse-Glimmer-30B's shape; the kernels bake it");
    require(d_.qgkv_s == km::kQgkvS && d_.oproj_s == km::kOprojS && d_.gate_up_s == km::kGateUpS &&
                d_.down_s == km::kDownS,
            "the int4 cells moved from kernels::muse's S2 / S4 / S8 / S4 (the PROVISIONAL picks the names spell)");
    require(b_.desc.layers == d_.layers, "the buffers were sized for another descriptor");
    require(part_.device == b_.device, "the part is device " + std::to_string(part_.device) + ", the buffers device " +
                                           std::to_string(b_.device));
    require(part_.first == b_.placement.first(b_.device) && part_.end == b_.placement.end(b_.device) &&
                part_.layers.size() == part_.end - part_.first,
            "the loaded part's layers are not the placement's");
    require(b_.placement.devices == 1 || link_ != nullptr, "two devices need the hand-off's binding");
    require(b_.placement.devices == 2 || link_ == nullptr, "a hand-off on one device");
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    if (part_.device + 1 == b_.placement.devices)
      require(part_.final_norm && part_.lm_head, "the last device holds no head");
    require(b_.kv_bytes() == device_kv_bytes(d_, b_.placement, b_.device, b_.max_len) &&
                b_.ring_bytes() == device_ring_bytes(d_, b_.placement, b_.device),
            "the KV groups are not runtime::muse's");
    require(b_.scratch_bytes() == decode_scratch_bytes(d_, b_.max_len, b_.rows, b_.attn),
            "the decode scratch is not runtime::muse::decode_scratch_bytes'");
    require(M_ >= 1 && M_ <= b_.rows, "a list of " + std::to_string(M_) + " rows over a scratch of " +
                                          std::to_string(b_.rows));
    require(bool(b_.attn_s) == (b_.attn == MuseAttn::Eager), "eager's score row exists exactly under eager");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][hidden] bf16");
    const bool int8 = part_.lm_head ? part_.lm_head->kind == model::WeightKind::Int8 : true;
    std::vector<std::string> need = verify_ ? km::verify_variants(M_, int8, b_.attn == MuseAttn::Eager)
                                            : km::decode_variants(int8, b_.attn == MuseAttn::Eager);
    if (verify_) {
      require(b_.placement.devices == 1 && link_ == nullptr, "a verify list is one card's (spec 23f)");
      require(verify_->ids && verify_->ids->size() >= M_ * 4 && verify_->out && verify_->out->size() >= M_ * 4,
              "the verify's ids / output buffers hold fewer than M ids");
      if (verify_->taps) {
        require(verify_->tap_layers && !verify_->tap_layers->empty(), "the drafter's taps without tap layers");
        for (uint32_t t : *verify_->tap_layers)
          require(t < d_.layers, "the drafter taps layer " + std::to_string(t) + " of a " + std::to_string(d_.layers) +
                                     "-layer target");
        require(verify_->taps->size() >= size_t(M_) * verify_->tap_layers->size() * d_.hidden * 2,
                "the drafter's tap buffer is smaller than [M][taps][hidden] bf16");
      }
    }
    if (link_ && link_->mode == PpHandoff::Peer) need.push_back(kernels::pp_handoff_variant());
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(), v + " is not compiled (" + kernels::path(v) + "): build with B70_MUSE=ON");
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
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    require(!pending_entry_.empty(), "launch() without a preceding kernel()");
    step_.list.launch(k, gx, gy, 1, nullptr);
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ +
                           " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  // mg_embed(ctrl, embed, resid, sumsq), grid (1, M): the residual row and layer 0's Σ². Spec 23f's verify:
  // mg_pf_embed(ids, embed, resid, sumsq) at M - the same chain over the verify's ids buffer.
  void embed() {
    l0::Kernel& k = verify_ ? kernel(km::verify_embed_variant(M_), "mg_pf_embed", km::kRowWg)
                            : kernel(km::embed_variant(M_), "mg_embed", km::kRowWg);
    k.arg_ptr(0, verify_ ? verify_->ids->ptr() : b_.control.ptr());
    k.arg_ptr(1, part_.embed->ptr());
    k.arg_ptr(2, b_.resid.ptr());
    k.arg_ptr(3, b_.sumsq_r.ptr());
    launch(k, 1, M_);
  }
  // mg_norm / mg_final_norm(sumsq, resid, w, x), grid (kNormW, M).
  void norm(const float* w, const char* entry = "mg_norm") {
    l0::Kernel& k = kernel(km::norm_variant(M_), entry, km::kRowWg);
    k.arg_ptr(0, b_.sumsq_r.ptr());
    k.arg_ptr(1, b_.resid.ptr());
    k.arg_ptr(2, w);
    k.arg_ptr(3, b_.x.ptr());
    launch(k, km::kNormW, M_);
  }
  // prep_res_fold ..._Z(partials, a, sumsq_a), grid (kNormG, M): a sub-block's S slices -> its own row and Σ².
  void fold_zero(uint32_t S) {
    l0::Kernel& k = kernel(km::fold_zero_variant(M_, S), "prep_res_fold", km::kRowWg);
    k.arg_ptr(0, b_.partials.ptr());
    k.arg_ptr(1, b_.a.ptr());
    k.arg_ptr(2, b_.sumsq_a.ptr());
    launch(k, km::kNormG, M_);
  }
  // mg_post_add(sumsq_sub, sub, w_post, resid, sumsq_out), grid (kNormG, M).
  void post_add(const float* w) {
    l0::Kernel& k = kernel(km::post_add_variant(M_), "mg_post_add", km::kRowWg);
    k.arg_ptr(0, b_.sumsq_a.ptr());
    k.arg_ptr(1, b_.a.ptr());
    k.arg_ptr(2, w);
    k.arg_ptr(3, b_.resid.ptr());
    k.arg_ptr(4, b_.sumsq_r.ptr());
    launch(k, km::kNormG, M_);
  }
  // gemv(w, scales, x, out), grid (N / 64, S) - an int4 layout-0 linear into `partials` [S][M][N].
  void linear(const loader::DeviceWeight& w, const void* x, const std::string& variant) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4 && s.layout == 0 && w.scales, "a Muse linear is int4 layout 0 with its scales");
    require(variant == kernels::gemv_variant(M_, s.K, s.N, s.S, s.layout),
            "the loaded weight's shape {" + std::to_string(s.K) + ", " + std::to_string(s.N) + ", S" + std::to_string(s.S) +
                "} is not the binary " + variant + "'s");
    require(b_.partials.size() >= size_t(s.S) * M_ * s.N * 4, "the GEMV's output is smaller than [S][M][N]");
    l0::Kernel& k = kernel(variant, "gemv", 64);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, w.scales->ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, b_.partials.ptr());
    launch(k, s.N / 64, s.S);
  }

  void layer(uint32_t l) {
    const loader::MuseLayer& L = part_.layer(l);
    const bool sliding = !d_.is_full(l);
    require(L.qgkv && L.oproj && L.gate_up && L.down && L.norms, "layer " + std::to_string(l) + " is not fully loaded");
    const l0::Mem& nw = *L.norms;
    // --- attention ----------------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_input()));
    linear(*L.qgkv, b_.x.ptr(), km::qgkv_variant(M_));
    void* kv_k = b_.k_layer(l);
    void* kv_v = b_.v_layer(l);
    {
      // mg_attn_prep(ctrl, partials, rope, attn_q, gate, kv_k, kv_v), grid (2 q + 2 kv heads, M)
      l0::Kernel& k = kernel(km::attn_prep_variant(M_, km::kQgkvS, sliding), "mg_attn_prep", km::kPrepWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, b_.rope.ptr());   // read only when sliding: the full layers are NoPE
      k.arg_ptr(3, b_.attn_q.ptr());
      k.arg_ptr(4, b_.gate.ptr());
      k.arg_ptr(5, kv_k);
      k.arg_ptr(6, kv_v);
      launch(k, km::kPrepGroups, M_);
    }
    if (b_.attn == MuseAttn::Eager)
      attention_eager(sliding, kv_k, kv_v);
    else
      attention_flash(sliding, kv_k, kv_v);
    linear(*L.oproj, b_.attn_out.ptr(), km::oproj_variant(M_));
    fold_zero(km::kOprojS);
    post_add(fat(nw, d_.norm_off_post_attn()));
    // --- the MLP --------------------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_pre_ff()));
    linear(*L.gate_up, b_.x.ptr(), km::gate_up_variant(M_));
    {
      // mg_silu_mul(partials, h), grid (ceil(I / 256), M)
      l0::Kernel& k = kernel(km::silu_variant(M_, km::kGateUpS), "mg_silu_mul", km::kRowWg);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.h.ptr());
      launch(k, km::kSiluGroups, M_);
    }
    linear(*L.down, b_.h.ptr(), km::down_variant(M_));
    fold_zero(km::kDownS);
    post_add(fat(nw, d_.norm_off_post_ff()));
  }

  // mg_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)   grid (kv heads, TGT)
  // mg_attn_reduce(ctrl, attn_part, gate, attn_out)       grid (q heads, M)
  void attention_flash(bool sliding, void* kv_k, void* kv_v) {
    const std::string v = km::attn_variant(M_, sliding);
    {
      l0::Kernel& k = kernel(v, "mg_attn_decode", km::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, kv_v);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, d_.kv_heads, km::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "mg_attn_reduce", km::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.gate.ptr());
      k.arg_ptr(3, b_.attn_out.ptr());
      launch(k, d_.q_heads, M_);
    }
  }
  // B70_MUSE_ATTN=eager: mg_attn_eager(ctrl, attn_q, kv_k, kv_v, gate, attn_s, attn_out, stride), grid (q heads, M).
  void attention_eager(bool sliding, void* kv_k, void* kv_v) {
    l0::Kernel& k = kernel(km::attn_eager_variant(M_, sliding), "mg_attn_eager", km::kAttnWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, b_.attn_q.ptr());
    k.arg_ptr(2, kv_k);
    k.arg_ptr(3, kv_v);
    k.arg_ptr(4, b_.gate.ptr());
    k.arg_ptr(5, b_.attn_s->ptr());
    k.arg_ptr(6, b_.attn_out.ptr());
    k.arg<uint32_t>(7, b_.max_len);
    launch(k, d_.q_heads, M_);
  }

  void head() {
    norm(static_cast<const float*>(part_.final_norm->ptr()), "mg_final_norm");
    const loader::DeviceWeight& lm = *part_.lm_head;
    require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden, "lm_head is not [vocab][hidden]");
    if (lm.kind == model::WeightKind::Int8) {
      require(bool(lm.scales), "the int8 lm_head has no row scales");
      l0::Kernel& k = kernel(verify_ ? km::verify_lm_head_variant(M_, true) : km::lm_head_variant(true), "gemv_i8w",
                             kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, b_.logits.ptr());
      launch(k, lm.shape.N / kernels::kGemvI8wCols);
    } else {
      require(lm.kind == model::WeightKind::Bf16, "lm_head is neither int8 nor bf16");
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
      l0::Kernel& k = kernel(verify_ ? km::verify_lm_head_variant(M_, false) : km::lm_head_variant(false), "gemv_bf16",
                             t.cols * t.ksplit);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, b_.logits.ptr());
      launch(k, lm.shape.N / t.cols);
    }
    if (verify_) {   // spec 23f: mg_vargmax - every row's id into the verify's output, Control::pos left alone
      {
        l0::Kernel& k = kernel(km::vargmax1_variant(M_), "mg_vargmax_stage1", km::kArgmaxWg);
        k.arg_ptr(0, b_.logits.ptr());
        k.arg_ptr(1, b_.argmax_part.ptr());
        launch(k, km::argmax_groups(), M_);
      }
      l0::Kernel& k = kernel(km::vargmax2_variant(), "mg_vargmax_stage2", km::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      k.arg_ptr(2, verify_->out->ptr());
      launch(k, 1, 1);
      require(km::argmax_groups() <= km::kArgmaxWg, "mg_vargmax_stage2 folds at most 256 groups");
      return;
    }
    {
      l0::Kernel& k = kernel(km::argmax1_variant(M_), "argmax_stage1", km::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, km::argmax_groups(), M_);
    }
    {
      l0::Kernel& k = kernel(km::argmax2_variant(), "argmax_stage2", km::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
    require(km::argmax_groups() <= km::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
  }

  // --- the hand-off (spec 16b's: capture.cc handoff_out / handoff_in; Kolibri's) --------------------------------
  size_t resid_bytes() const { return size_t(M_) * d_.hidden * 2; }
  size_t sumsq_bytes() const { return b_.sumsq_r.size(); }
  void check_link() const {
    require(link_->land_resid && link_->land_sumsq && link_->resid_cap >= resid_bytes() &&
                link_->sumsq_cap == sumsq_bytes(),
            "the landing buffer does not hold " + std::to_string(resid_bytes()) + " B of the row and exactly " +
                std::to_string(sumsq_bytes()) + " B of norm sums (pp_send stamps the word after them)");
    if (link_->mode == PpHandoff::Copy) require(link_->event != nullptr, "the copy hand-off needs its cross-device event");
  }
  void handoff_out() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.copy(link_->land_resid, b_.resid.ptr(), resid_bytes());
      step_.list.copy(link_->land_sumsq, b_.sumsq_r.ptr(), sumsq_bytes());
      step_.list.barrier_signal(link_->event);
      return;
    }
    require(link_->flag && link_->send_seq, "the peer hand-off needs its flag and counter");
    // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words), grid (1)
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_send", kernels::kPpHandoffWg);
    k.arg_ptr(0, b_.resid.ptr());
    k.arg_ptr(1, b_.sumsq_r.ptr());
    k.arg_ptr(2, link_->land_resid);
    k.arg_ptr(3, link_->land_sumsq);
    k.arg_ptr(4, link_->flag);
    k.arg_ptr(5, link_->send_seq);
    k.arg<uint32_t>(6, static_cast<uint32_t>(resid_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(sumsq_bytes() / 4));
    launch(k, 1);
  }
  void handoff_in() {
    check_link();
    if (link_->mode == PpHandoff::Copy) {
      step_.list.wait_event(link_->event);
      step_.list.copy(b_.resid.ptr(), link_->land_resid, resid_bytes());
      step_.list.copy(b_.sumsq_r.ptr(), link_->land_sumsq, sumsq_bytes());
      return;
    }
    require(link_->flag && link_->recv_state && link_->spin_limit > 0,
            "the peer hand-off needs its flag, its state words and a spin bound");
    // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words, spin_limit)
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_recv", kernels::kPpHandoffWg);
    k.arg_ptr(0, link_->land_resid);
    k.arg_ptr(1, link_->land_sumsq);
    k.arg_ptr(2, link_->flag);
    k.arg_ptr(3, link_->recv_state);
    k.arg_ptr(4, b_.resid.ptr());
    k.arg_ptr(5, b_.sumsq_r.ptr());
    k.arg<uint32_t>(6, static_cast<uint32_t>(resid_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(sumsq_bytes() / 4));
    k.arg<uint32_t>(8, link_->spin_limit);
    launch(k, 1);
  }

  // Spec 23f: layer l's OUTPUT (resid after its last mg_post_add) of every row into the drafter's taps
  // [row][tap][hidden] when l is a tap layer (target_layer_ids: 1, 13, 25, 37, 49).
  void drafter_taps(uint32_t l) {
    const std::vector<uint32_t>& tl = *verify_->tap_layers;
    const size_t row = size_t(d_.hidden) * 2;
    for (size_t t = 0; t < tl.size(); ++t) {
      if (tl[t] != l) continue;
      for (uint32_t r = 0; r < M_; ++r)
        step_.list.copy(at(*verify_->taps, (size_t(r) * tl.size() + t) * row), at(b_.resid, size_t(r) * row), row);
    }
  }

  l0::Context& ctx_;
  const loader::MuseDevicePart& part_;
  const model::MuseGlimmerDesc& d_;
  MuseBuffers& b_;
  const StageLink* link_;
  l0::Mem* tap_;
  const VerifyBind* verify_;
  uint32_t M_;
  CapturedStep step_;
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b,
                   const StageLink* link, l0::Mem* tap) {
  return Walk(ctx, part, d, b, link, tap).run();
}

CapturedStep build_verify(l0::Context& ctx, const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d,
                          MuseBuffers& b, uint32_t M, const VerifyBind& v) {
  if (M < 1 || M > kVerifyMaxRows)
    throw std::invalid_argument("runtime::muse::build_verify: M " + std::to_string(M) + " outside 1..16");
  return Walk(ctx, part, d, b, nullptr, nullptr, M, &v).run();
}

}  // namespace runtime::muse
