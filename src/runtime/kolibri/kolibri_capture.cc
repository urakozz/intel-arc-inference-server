#include "runtime/kolibri/kolibri_capture.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"
#include "loader/kolibri1_layout.h"
#include "runtime/control.h"

// kolibri_capture.cc - one device's Kolibri-1 decode list (kolibri_capture.h has the walk in one
// table). Every binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/kolibri/*.cl and the reused sources.
namespace runtime::kolibri {
namespace {

namespace kk = kernels::kolibri;

static_assert(kRouteWords == kk::route::kWords, "the Kolibri route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the Kolibri binaries are M = 1 only");

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }
const float* fat(const l0::Mem& m, size_t floats) { return static_cast<const float*>(m.ptr()) + floats; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::kolibri::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d, KolibriBuffers& b,
       const StageLink* link, l0::Mem* tap)
      : ctx_(ctx), part_(part), d_(d), b_(b), link_(link), tap_(tap), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    const uint32_t dev = part_.device;
    if (dev == 0) {
      embed();
      fold(nullptr, b_.resid, b_.sumsq_r, kk::fold_variant(kM), "embed's");
    }
    if (link_ && dev == 1) handoff_in();
    for (uint32_t l = part_.first; l < part_.end; ++l) {
      layer_ = int(l);
      layer(l);
      if (tap_) step_.list.copy(at(*tap_, size_t(l) * kM * d_.hidden * 2), b_.resid.ptr(), size_t(kM) * d_.hidden * 2);
    }
    layer_ = -1;
    if (link_ && dev == 0) handoff_out();
    if (dev + 1 == b_.placement.devices) head();
    const PpHandoff h = link_ ? link_->mode : kDefaultPpHandoff;
    const size_t want = device_launches(d_, b_.placement, dev, b_.attn, h);
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s decode list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::kolibri::device_launches gives (" +
                                            kol_attn_name(b_.attn) + " attention)");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    // The kernels' shapes are Kolibri-1's (kernels::kolibri); a descriptor may differ only in layers.
    require(d_.hidden == kk::kHidden && d_.q_heads == kk::kQHeads && d_.kv_heads == kk::kKvHeads &&
                d_.head_dim == kk::kHd && d_.experts == kk::kExperts && d_.top_k == kk::kTopK &&
                d_.router_n() == kk::kRouterN && d_.moe_inter == kk::kInter && d_.shared_inter == kk::kInter &&
                d_.vocab == kk::kVocab && d_.window == kk::kWindow && model::Kolibri1Desc::kRing == kk::kRing,
            "the descriptor is not Kolibri-1's shape (" + d_.name + "); the kernels bake it");
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
      require(part_.final_norm && part_.lm_head, "the last device holds no head");
    if (loader::kol_device_has_sliding(d_, b_.placement, part_.device))
      require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the RoPE table is shorter than max_len");
    const PersistentSizes ps = persistent_sizes(d_, b_.placement, b_.device, b_.max_len);
    require(b_.full_k.size() == ps.full_k && b_.ring_k.size() == ps.ring_k, "the KV groups are not persistent_sizes'");
    const ScratchSizes ss = scratch_sizes(d_);
    require(b_.partials.size() == ss.partials && b_.routes.size() == ss.routes && b_.attn_part.size() == ss.attn_part,
            "the decode scratch is not runtime::kolibri::scratch_sizes'");
    require(ss.sumsq_r == size_t(kk::kNormG) * kM * 4, "sumsq is not kNormG floats");
    require(ss.attn_part == size_t(d_.q_heads) * kk::kAttnTgt * kM * kk::kAttnPart * 4, "attn_part's geometry moved");
    require(bool(b_.attn_scores) == (b_.attn == KolAttn::Eager), "eager's score row exists exactly under eager");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][hidden] bf16");
    // The int4 arm's {S, layout} are the names' (decode_variants spells S2 / S4).
    if (d_.attn == model::KolAttnForm::Int4) require(d_.qkv_s == 2 && d_.oproj_s == 4, "the int4 cells moved from S2 / S4");
    const bool int8 = part_.lm_head ? part_.lm_head->kind == model::WeightKind::Int8 : false;
    std::vector<std::string> need = kk::decode_variants(d_.attn, int8, b_.attn == KolAttn::Eager);
    if (link_ && link_->mode == PpHandoff::Peer) need.push_back(kernels::pp_handoff_variant());
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_KOLIBRI=ON");
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

  // embed_gather(ctrl, embed, resid) - embed_gather.cl at Kolibri's hidden and vocabulary.
  void embed() {
    l0::Kernel& k = kernel(kk::embed_variant(kM), "embed_gather", kk::kEmbedWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, part_.embed->ptr());
    k.arg_ptr(2, b_.resid.ptr());
    launch(k, 1, kM);
  }

  // prep_res_fold(partials, row, sumsq) - prep.cl stage A, grid (G, M): SP0 re-reads `row` and reduces
  // its Σ²; the _Z builds write rne(Σ partials) into `row` first.
  void fold(const void* partials, const l0::Mem& row, const l0::Mem& sumsq, const std::string& variant, const char*) {
    l0::Kernel& k = kernel(variant, "prep_res_fold", kk::kNormWg);
    k.arg_ptr(0, partials ? partials : b_.partials.ptr());
    k.arg_ptr(1, row.ptr());
    k.arg_ptr(2, sumsq.ptr());
    launch(k, kk::kNormG, kM);
  }
  // kol_norm_finish(sumsq, resid, w, x), grid (kNormW, M).
  void norm(const float* w) {
    l0::Kernel& k = kernel(kk::norm_variant(kM), "kol_norm_finish", kk::kNormWg);
    k.arg_ptr(0, b_.sumsq_r.ptr());
    k.arg_ptr(1, b_.resid.ptr());
    k.arg_ptr(2, w);
    k.arg_ptr(3, b_.x.ptr());
    launch(k, kk::kNormW, kM);
  }
  // kol_post_add(sumsq_a, a, w, resid, sumsq_out), grid (kNormG, M).
  void post_add(const l0::Mem& a, const float* w) {
    l0::Kernel& k = kernel(kk::post_add_variant(kM), "kol_post_add", kk::kNormWg);
    k.arg_ptr(0, b_.sumsq_a.ptr());
    k.arg_ptr(1, a.ptr());
    k.arg_ptr(2, w);
    k.arg_ptr(3, b_.resid.ptr());
    k.arg_ptr(4, b_.sumsq_r.ptr());
    launch(k, kk::kNormG, kM);
  }
  // A dense linear into `out`: gemv(w, scales, x, out) grid (N / 64, S) for int4 layout 0,
  // gemv_bf16(w, x, out) grid (N / cols) for bf16.
  void linear(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
    if (w.kind == model::WeightKind::Int4) {
      require(s.layout == 0 && w.scales, "a Kolibri int4 linear is GPTQ layout 0 with its own scales");
      require(out.size() >= size_t(s.S) * kM * s.N * 4, "the GEMV's output is smaller than [S][M][N]");
      l0::Kernel& k = kernel(kernels::gemv_variant(kM, s.K, s.N, s.S, s.layout), "gemv", 64);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, w.scales->ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out.ptr());
      launch(k, s.N / 64, s.S);
      return;
    }
    require(w.kind == model::WeightKind::Bf16, "gemv_bf16 bound to a non-bf16 weight");
    require(out.size() >= size_t(kM) * s.N * 4, "the bf16 GEMV's output is smaller than [M][N]");
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(s.N);
    l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kM, s.K, s.N, t), "gemv_bf16", t.cols * t.ksplit);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, x);
    k.arg_ptr(2, out.ptr());
    launch(k, s.N / t.cols);
  }

  void layer(uint32_t l) {
    const loader::KolLayer& L = part_.layer(l);
    const bool sliding = d_.is_sliding(l);
    const l0::Mem& nw = *L.norms;
    require(L.qkv && L.oproj && L.norms && L.router && L.bias && L.gate_up && L.down && L.shared_gate_up && L.shared_down,
            "layer " + std::to_string(l) + " is not fully loaded");
    require(L.qkv->shape.N == d_.qkv_n() && L.oproj->shape.N == d_.hidden, "layer " + std::to_string(l) + "'s attention rows");
    // --- attention ---------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_input()));
    linear(*L.qkv, b_.x.ptr(), b_.partials);
    void* kv_k = b_.k_layer(l);
    void* kv_v = b_.v_layer(l);
    {
      // kol_attn_prep(ctrl, partials, qkn, rope, attn_q, kv_k, kv_v), grid (q + 2 kv heads, M)
      l0::Kernel& k = kernel(kk::attn_prep_variant(kM, L.qkv->shape.S, sliding), "kol_attn_prep", kk::kPrepWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, fat(nw, d_.norm_off_q()));
      k.arg_ptr(3, part_.rope ? part_.rope->ptr() : b_.partials.ptr());   // read only when sliding
      k.arg_ptr(4, b_.attn_q.ptr());
      k.arg_ptr(5, kv_k);
      k.arg_ptr(6, kv_v);
      launch(k, d_.q_heads + 2 * d_.kv_heads, kM);
    }
    if (b_.attn == KolAttn::Eager)
      attention_eager(sliding, kv_k, kv_v);
    else
      attention_flash(sliding, kv_k, kv_v);
    linear(*L.oproj, b_.attn_out.ptr(), b_.partials);
    fold(b_.partials.ptr(), b_.a, b_.sumsq_a, kk::fold_zero_variant(kM, L.oproj->shape.S), "o_proj's");
    post_add(b_.a, fat(nw, d_.norm_off_post_attn()));
    // --- the MoE block -------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_post_attention()));
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden, "the router is not [router_n][hidden]");
    linear(*L.router, b_.x.ptr(), b_.logits_r);
    void* rt = at(b_.routes, route_at(l));
    {
      l0::Kernel& k = kernel(kk::route_variant(kM), "kol_route", kk::kRouteWg);
      k.arg_ptr(0, b_.logits_r.ptr());
      k.arg_ptr(1, L.bias->ptr());
      k.arg_ptr(2, rt);
      launch(k, 1, kM);
    }
    const std::string mv = kk::moe_variant(kM);
    {
      l0::Kernel& k = kernel(mv, "kol_moe_gate_up", kk::moe_gate_up_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, L.gate_up->ptr());
      k.arg_ptr(3, L.shared_gate_up->ptr());
      k.arg_ptr(4, b_.moe_h.ptr());
      launch(k, kk::moe_gate_up_groups(), kM);
    }
    {
      l0::Kernel& k = kernel(mv, "kol_moe_down", kk::moe_down_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_h.ptr());
      k.arg_ptr(2, L.down->ptr());
      k.arg_ptr(3, L.shared_down->ptr());
      k.arg_ptr(4, b_.mo.ptr());
      launch(k, d_.hidden / 16, kM);
    }
    fold(nullptr, b_.mo, b_.sumsq_a, kk::fold_variant(kM), "the MoE's");
    post_add(b_.mo, fat(nw, d_.norm_off_post_ffn()));
  }

  // kol_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)   grid (kv heads, TGT)
  // kol_attn_reduce(ctrl, attn_part, attn_out)             grid (q heads, M)
  void attention_flash(bool sliding, void* kv_k, void* kv_v) {
    const std::string v = kk::attn_variant(kM, sliding);
    {
      l0::Kernel& k = kernel(v, "kol_attn_decode", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, kv_v);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, d_.kv_heads, kk::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "kol_attn_reduce", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_out.ptr());
      launch(k, d_.q_heads, kM);
    }
  }
  // B70_KOLIBRI_ATTN=eager: kol_attn_eager.cl's four kernels, the score row between.
  //   kol_attn_eager_score  (ctrl, attn_q, kv_k, attn_s, stride)      grid (kv heads, TGT)
  //   kol_attn_eager_softmax(ctrl, attn_s, stride)                    grid (q heads, M)
  //   kol_attn_eager_pv     (ctrl, attn_s, kv_v, attn_part, stride)   grid (kv heads, TGT)
  //   kol_attn_eager_reduce (ctrl, attn_part, attn_out)               grid (q heads, M)
  void attention_eager(bool sliding, void* kv_k, void* kv_v) {
    const std::string v = kk::attn_eager_variant(kM, sliding);
    void* sc = b_.attn_scores->ptr();
    const uint32_t stride = b_.max_len;
    {
      l0::Kernel& k = kernel(v, "kol_attn_eager_score", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, sc);
      k.arg<uint32_t>(4, stride);
      launch(k, d_.kv_heads, kk::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "kol_attn_eager_softmax", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, sc);
      k.arg<uint32_t>(2, stride);
      launch(k, d_.q_heads, kM);
    }
    {
      l0::Kernel& k = kernel(v, "kol_attn_eager_pv", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, sc);
      k.arg_ptr(2, kv_v);
      k.arg_ptr(3, b_.attn_part.ptr());
      k.arg<uint32_t>(4, stride);
      launch(k, d_.kv_heads, kk::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "kol_attn_eager_reduce", kk::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_out.ptr());
      launch(k, d_.q_heads, kM);
    }
  }

  void head() {
    norm(static_cast<const float*>(part_.final_norm->ptr()));
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
    {
      l0::Kernel& k = kernel(kk::argmax1_variant(kM), "argmax_stage1", kk::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kk::argmax_groups(), kM);
    }
    {
      l0::Kernel& k = kernel(kk::argmax2_variant(), "argmax_stage2", kk::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
    require(kk::argmax_groups() <= kk::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
  }

  // --- the hand-off (spec 16b's: capture.cc handoff_out / handoff_in) -----------------------------------
  size_t resid_bytes() const { return size_t(kM) * d_.hidden * 2; }
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

  l0::Context& ctx_;
  const loader::KolDevicePart& part_;
  const model::Kolibri1Desc& d_;
  KolibriBuffers& b_;
  const StageLink* link_;
  l0::Mem* tap_;
  CapturedStep step_;
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d,
                   KolibriBuffers& b, const StageLink* link, l0::Mem* tap) {
  return Walk(ctx, part, d, b, link, tap).run();
}

}  // namespace runtime::kolibri
