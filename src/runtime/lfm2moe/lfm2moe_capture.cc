#include "runtime/lfm2moe/lfm2moe_capture.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/lfm2moe_kernels.h"
#include "loader/lfm2moe_layout.h"
#include "runtime/control.h"

// lfm2moe_capture.cc - one device's LFM2.5-24B-A2B decode list (lfm2moe_capture.h has the walk in one table). Every
// binding site names the kernel it honours; argument orders are each __kernel's, in src/kernels/lfm2moe/*.cl and the
// reused sources.
namespace runtime::lfm2moe {
namespace {

namespace kl = kernels::lfm2moe;

static_assert(kRouteWords == kl::route::kWords, "the LFM2 route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the LFM2 decode binaries are M = 1");

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::lfm2moe::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b,
       LfmFuse fuse, const StageLink* link, l0::Mem* tap, ProfileEvents* prof)
      : ctx_(ctx), part_(part), d_(d), b_(b), fuse_(fuse), link_(link), tap_(tap), prof_(prof),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    if (prof_) prof_->events.clear();
    const uint32_t dev = part_.device;
    if (dev == 0) embed();
    if (link_ && dev == 1) handoff_in();
    for (uint32_t l = part_.first; l < part_.end; ++l) {
      layer_ = int(l);
      layer(l);
      // The tap holds layer l's whole output: a dense layer's down slices are folded by the NEXT norm, which then
      // writes the tap row (pending_tap_); every other layer's h is complete after its last launch.
      if (tap_) {
        if (pending_s_)
          pending_tap_ = int(l);
        else
          tap_row(l, cur_);
      }
    }
    layer_ = -1;
    if (link_ && dev == 0) handoff_out();
    if (dev + 1 == b_.placement.devices) head();
    const PpHandoff hm = link_ ? link_->mode : kDefaultPpHandoff;
    const size_t want = device_launches(d_, b_.placement, dev, b_.attn, fuse_, hm);
    require(step_.kernel_count == want, "device " + std::to_string(dev) + "'s decode list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::lfm2moe::device_launches gives (" +
                                            lfm_attn_name(b_.attn) + " attention, fuse " + lfm_fuse_name(fuse_) + ")");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    require(!prof_ || prof_->events.size() == step_.kernel_count, "a launch got no profiling event");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    // The kernels' shapes are LFM2.5-24B-A2B's (kernels::lfm2moe); a descriptor may differ only in layers.
    require(d_.hidden == kl::kHidden && d_.q_heads == kl::kQHeads && d_.kv_heads == kl::kKvHeads &&
                d_.head_dim == kl::kHd && d_.experts == kl::kExperts && d_.top_k == kl::kTopK &&
                d_.moe_inter == kl::kInter && d_.dense_inter == kl::kDenseInter && d_.vocab == kl::kVocab &&
                d_.vocab_used == kl::kVocabUsed && d_.conv_taps == 3 && loader::lfm::kConvRing == kl::kConvRing,
            "the descriptor is not LFM2.5-24B-A2B's shape (" + d_.name + "); the kernels bake it");
    require(b_.desc.layers == d_.layers, "the buffers were sized for another descriptor");
    require(part_.device == b_.device, "the part is device " + std::to_string(part_.device) + ", the buffers device " +
                                           std::to_string(b_.device));
    require(part_.first == b_.placement.first(b_.device) && part_.end == b_.placement.end(b_.device) &&
                part_.layers.size() == part_.end - part_.first,
            "the loaded part's layers are not the placement's");
    require(b_.placement.devices == 1 || link_ != nullptr, "two devices need the hand-off's binding");
    require(b_.placement.devices == 2 || link_ == nullptr, "a hand-off on one device");
    if (part_.device == 0) require(bool(part_.embed_tiles), "device 0 holds no embedding tiles");
    const bool last = part_.device + 1 == b_.placement.devices;
    if (last) require(part_.final_norm && (part_.head_int8 || part_.embed_tiles), "the last device holds no head");
    if (loader::lfm::device_has_attention(d_, b_.placement, part_.device))
      require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the RoPE table is shorter than max_len");
    const PersistentSizes ps = persistent_sizes(d_, b_.placement, b_.device, b_.max_len);
    require(b_.kv_k.size() == ps.kv_k && b_.conv_ring.size() == ps.conv_ring, "the persistent group is not persistent_sizes'");
    const ScratchSizes ss = scratch_sizes(d_, b_.max_len, b_.attn);
    require(b_.slices.size() == ss.slices && b_.routes.size() == ss.routes && b_.attn_part.size() == ss.attn_part &&
                b_.h0.size() == ss.h0 && b_.h1.size() == ss.h1,
            "the decode scratch is not runtime::lfm2moe::scratch_sizes'");
    require(ss.attn_part == size_t(d_.q_heads) * kl::kAttnTgt * kM * kl::kAttnPart * 4, "attn_part's geometry moved");
    require(bool(b_.attn_scores) == (b_.attn == LfmAttn::Eager), "eager's score row exists exactly under eager");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][hidden] bf16");
    if (fuse_.conv) require(conv_tiles_, "the arm conv needs in_proj in the conv-tile layout (load with conv_tiles)");
    else require(!conv_tiles_, "in_proj was loaded in the conv-tile layout: only the arm conv reads it");
    const bool int8 = last && part_.head_int8;
    kl::Arms arms;
    arms.route = fuse_.route;
    arms.router = fuse_.router;
    arms.conv = fuse_.conv;
    std::vector<std::string> need = kl::decode_variants(d_, int8, b_.attn == LfmAttn::Eager,
                                                        link_ && link_->mode == PpHandoff::Peer, arms);
    for (const std::string& v : need)   // a missing binary is named before a command is appended
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_LFM2MOE=ON");
  }

  l0::Mem& h(int i) { return i ? b_.h1 : b_.h0; }
  void tap_row(uint32_t l, int buf) {
    step_.list.copy(at(*tap_, size_t(l) * kM * d_.hidden * 2), h(buf).ptr(), size_t(kM) * d_.hidden * 2);
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
    l0::Event* signal = nullptr;
    if (prof_) {
      require(step_.kernel_count < ProfileEvents::kProfileCapacity,
              "the decode walk has more than " + std::to_string(ProfileEvents::kProfileCapacity) + " launches");
      prof_->events.emplace_back(prof_->pool, static_cast<uint32_t>(step_.kernel_count));
      signal = &prof_->events.back();
    }
    step_.list.launch(k, gx, gy, 1, signal);
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ +
                           " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  const float* small(uint32_t l, size_t off) const {
    return reinterpret_cast<const float*>(at(*part_.layer(l).small, off));
  }

  // lfm_embed_gather(ctrl, tiles, h) - the gathered embedding becomes the residual h[cur]; nothing pending.
  void embed() {
    l0::Kernel& k = kernel(kl::embed_variant(kM), "lfm_embed_gather", kl::kEmbedWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, part_.embed_tiles->ptr());
    k.arg_ptr(2, h(cur_).ptr());
    launch(k, 1, kM);
    pending_s_ = 0;
  }

  // lfm_res_norm(h_in, slices, w, h_out, x[, router tiles, logits_r]), grid (kNormWgs [+ 4], M): folds the pending
  // slices (h[cur] -> h[1 - cur], then cur flips) or only normalises (_NF).
  void norm(const float* w, const loader::DeviceWeight* router = nullptr) {
    const uint32_t S = pending_s_;
    const int in = cur_, out = S ? 1 - cur_ : cur_;
    l0::Kernel& k = kernel(kl::res_norm_variant(kM, S, router != nullptr), "lfm_res_norm", kl::kNormWg);
    k.arg_ptr(0, h(in).ptr());
    k.arg_ptr(1, b_.slices.ptr());
    k.arg_ptr(2, w);
    k.arg_ptr(3, h(out).ptr());
    k.arg_ptr(4, b_.x.ptr());
    if (router) {
      require(router->shape.K == d_.hidden && router->shape.N == d_.experts && router->kind == model::WeightKind::Bf16,
              "the router is not bf16 [experts][hidden] tiles");
      k.arg_ptr(5, router->mem.ptr());
      k.arg_ptr(6, b_.logits_r.ptr());
    }
    launch(k, kl::kNormWgs + (router ? kl::kNormRouterWgs : 0), kM);
    // Review Focus 3: a fold writes the buffer it did not read; the next reader reads the one written.
    require(!S || &h(out) != &h(in), "a fold site would rewrite the buffer its work-groups read");
    cur_ = out;
    if (S && tap_ && pending_tap_ >= 0) {   // the folded h IS the dense layer's output
      tap_row(uint32_t(pending_tap_), cur_);
      pending_tap_ = -1;
    }
    pending_s_ = 0;
    ++norm_sites_;
  }

  // An int4 GEMV into the split-K slices: gemv(w, scales, x, out), grid (N / 64, S), at the weight's own layout (the
  // descriptor's {S, layout}: layout 0 reads its separate scales, layout 1 the tiles' inline ones).
  void gemv(const loader::DeviceWeight& w, const void* x) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4, "an LFM2 linear is int4 (spec 24 §5)");
    require(s.layout == 1 || bool(w.scales), "a layout-0 linear has no scales allocation");
    require(b_.slices.size() >= size_t(s.S) * kM * s.N * 4, "the slices are smaller than [S][M][N]");
    l0::Kernel& k = kernel(kernels::gemv_variant(kM, s.K, s.N, s.S, s.layout), "gemv", 64);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, s.layout == 0 ? w.scales->ptr() : w.mem.ptr());   // layout 1: not read
    k.arg_ptr(2, x);
    k.arg_ptr(3, b_.slices.ptr());
    launch(k, s.N / 64, s.S);
  }

  void layer(uint32_t l) {
    const loader::LfmLayer& L = part_.layer(l);
    const loader::lfm::SmallBlock sb = loader::lfm::small_block(d_, l);
    require(L.small && L.small->size() == sb.bytes, "layer " + std::to_string(l) + "'s small block is not small_block's");
    const bool attn = d_.is_attention(l), dense = d_.is_dense(l);
    // --- the mixer ------------------------------------------------------------------------------------------
    norm(small(l, sb.operator_norm));
    if (attn) {
      require(L.qkv && L.o_proj && L.qkv->shape.N == d_.qkv_n() && L.o_proj->shape.N == d_.hidden,
              "layer " + std::to_string(l) + "'s attention linears");
      gemv(*L.qkv, b_.x.ptr());
      void* kv_k = b_.k_layer(l);
      void* kv_v = b_.v_layer(l);
      {   // lfm_attn_prep(ctrl, slices, qkn, rope, attn_q, kv_k, kv_v), grid (q + 2 kv heads, M)
        require(sb.k_norm == sb.q_norm + size_t(d_.head_dim) * 4, "q_norm / k_norm are not adjacent");
        l0::Kernel& k = kernel(kl::attn_prep_variant(kM, L.qkv->shape.S), "lfm_attn_prep", kl::kPrepWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.slices.ptr());
        k.arg_ptr(2, small(l, sb.q_norm));
        k.arg_ptr(3, part_.rope->ptr());
        k.arg_ptr(4, b_.attn_q.ptr());
        k.arg_ptr(5, kv_k);
        k.arg_ptr(6, kv_v);
        launch(k, d_.q_heads + 2 * d_.kv_heads, kM);
      }
      if (b_.attn == LfmAttn::Eager)
        attention_eager(kv_k, kv_v);
      else
        attention_flash(kv_k, kv_v);
      gemv(*L.o_proj, b_.attn_out.ptr());
      pending_s_ = L.o_proj->shape.S;
    } else {
      require(L.in_proj && L.out_proj && L.in_proj->shape.N == 3 * d_.hidden && L.out_proj->shape.N == d_.hidden,
              "layer " + std::to_string(l) + "'s conv linears");
      if (fuse_.conv) {
        conv_in(*L.in_proj, small(l, sb.conv_taps), b_.ring_layer(l));
      } else {
        gemv(*L.in_proj, b_.x.ptr());
        // lfm_conv(ctrl, slices, taps, ring, y), grid (hidden / 256)
        l0::Kernel& k = kernel(kl::conv_variant(kM, L.in_proj->shape.S), "lfm_conv", kl::kConvWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.slices.ptr());
        k.arg_ptr(2, small(l, sb.conv_taps));
        k.arg_ptr(3, b_.ring_layer(l));
        k.arg_ptr(4, b_.conv_y.ptr());
        launch(k, d_.hidden / kl::kConvWg, 1);
      }
      gemv(*L.out_proj, b_.conv_y.ptr());
      pending_s_ = L.out_proj->shape.S;
    }
    // --- the FFN --------------------------------------------------------------------------------------------
    if (dense) {
      norm(small(l, sb.ffn_norm));
      require(L.dense_gate_up && L.dense_down && L.dense_gate_up->shape.N == 2 * d_.dense_inter,
              "layer " + std::to_string(l) + "'s dense FFN");
      gemv(*L.dense_gate_up, b_.x.ptr());
      {   // lfm_silu_mul(slices, out), grid (ceil(I / 256), M)
        l0::Kernel& k = kernel(kl::silu_mul_variant(kM, L.dense_gate_up->shape.S), "lfm_silu_mul", kl::kSiluWg);
        k.arg_ptr(0, b_.slices.ptr());
        k.arg_ptr(1, b_.s_mlp.ptr());
        launch(k, (d_.dense_inter + kl::kSiluWg - 1) / kl::kSiluWg, kM);
      }
      gemv(*L.dense_down, b_.s_mlp.ptr());
      pending_s_ = L.dense_down->shape.S;   // the next norm folds it
      return;
    }
    require(L.router && L.experts_gate_up && L.experts_down, "layer " + std::to_string(l) + "'s MoE block");
    norm(small(l, sb.ffn_norm), fuse_.router ? L.router.get() : nullptr);
    void* rt = at(b_.routes, route_at(l));
    const float* bias = small(l, sb.expert_bias);
    if (!fuse_.router) {   // lfm_router(x, w, logits), grid (E / 16, M)
      l0::Kernel& k = kernel(kl::router_variant(kM), "lfm_router", kl::kRouterWg);
      k.arg_ptr(0, b_.x.ptr());
      k.arg_ptr(1, L.router->mem.ptr());
      k.arg_ptr(2, b_.logits_r.ptr());
      launch(k, d_.experts / 16, kM);
    }
    if (!fuse_.route) {   // lfm_route(logits, bias, route), grid (1, M)
      l0::Kernel& k = kernel(kl::route_variant(kM), "lfm_route", kl::kRouteWg);
      k.arg_ptr(0, b_.logits_r.ptr());
      k.arg_ptr(1, bias);
      k.arg_ptr(2, rt);
      launch(k, 1, kM);
    }
    const std::string mv = kl::moe_variant(kM, fuse_.route);
    {   // lfm_moe_gate_up([logits, bias,] route, x, w_gu, h), grid (4 x 48, M)
      l0::Kernel& k = kernel(mv, "lfm_moe_gate_up", kl::moe_gate_up_wg());
      uint32_t a = 0;
      if (fuse_.route) {
        k.arg_ptr(a++, b_.logits_r.ptr());
        k.arg_ptr(a++, bias);
      }
      k.arg_ptr(a++, rt);
      k.arg_ptr(a++, b_.x.ptr());
      k.arg_ptr(a++, L.experts_gate_up->ptr());
      k.arg_ptr(a++, b_.moe_h.ptr());
      launch(k, kl::moe_gate_up_groups(), kM);
    }
    {   // lfm_moe_down(route, h, w_dn, h_res), grid (hidden / 16, M) - folds y into h[cur], in place
      l0::Kernel& k = kernel(kl::moe_variant(kM, false), "lfm_moe_down", kl::moe_down_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_h.ptr());
      k.arg_ptr(2, L.experts_down->ptr());
      k.arg_ptr(3, h(cur_).ptr());
      launch(k, d_.hidden / 16, kM);
    }
    pending_s_ = 0;
  }

  // The arm conv: lfm_conv_in(ctrl, w, scales, x, taps, ring, y), grid (hidden / 16), WG 48 - in_proj's GEMV at S 1 on
  // the conv-tile column order ([B 16 | C 16 | x 16] a 16-channel tile) with the conv as its epilogue.
  void conv_in(const loader::DeviceWeight& w, const float* taps, void* ring) {
    const model::GemvShape& s = w.shape;
    require(s.S == 1, "the arm conv is bitwise the as-built pair only at split-K 1 (in_proj is S" + std::to_string(s.S) + ")");
    require(s.layout == 1 || bool(w.scales), "a layout-0 in_proj has no scales allocation");
    l0::Kernel& k = kernel(kl::conv_in_variant(kM, s.layout), "lfm_conv_in", kl::kConvInWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, w.mem.ptr());
    k.arg_ptr(2, s.layout == 0 ? w.scales->ptr() : w.mem.ptr());
    k.arg_ptr(3, b_.x.ptr());
    k.arg_ptr(4, taps);
    k.arg_ptr(5, ring);
    k.arg_ptr(6, b_.conv_y.ptr());
    launch(k, d_.hidden / 16, 1);
  }

  // lfm_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part) grid (kv heads, TGT); lfm_attn_reduce(ctrl, attn_part,
  // attn_out) grid (q heads, M).
  void attention_flash(void* kv_k, void* kv_v) {
    const std::string v = kl::attn_variant(kM, false);
    {
      l0::Kernel& k = kernel(v, "lfm_attn_decode", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, kv_v);
      k.arg_ptr(4, b_.attn_part.ptr());
      launch(k, d_.kv_heads, kl::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "lfm_attn_reduce", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_out.ptr());
      launch(k, d_.q_heads, kM);
    }
  }
  // B70_LFM_ATTN=eager: lfm_attn_eager.cl's four kernels, the score row between (stride max_len).
  void attention_eager(void* kv_k, void* kv_v) {
    const std::string v = kl::attn_variant(kM, true);
    void* sc = b_.attn_scores->ptr();
    const uint32_t stride = b_.max_len;
    {
      l0::Kernel& k = kernel(v, "lfm_attn_eager_score", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_q.ptr());
      k.arg_ptr(2, kv_k);
      k.arg_ptr(3, sc);
      k.arg<uint32_t>(4, stride);
      launch(k, d_.kv_heads, kl::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "lfm_attn_eager_softmax", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, sc);
      k.arg<uint32_t>(2, stride);
      launch(k, d_.q_heads, kM);
    }
    {
      l0::Kernel& k = kernel(v, "lfm_attn_eager_pv", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, sc);
      k.arg_ptr(2, kv_v);
      k.arg_ptr(3, b_.attn_part.ptr());
      k.arg<uint32_t>(4, stride);
      launch(k, d_.kv_heads, kl::kAttnTgt);
    }
    {
      l0::Kernel& k = kernel(v, "lfm_attn_eager_reduce", kl::kAttnWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.attn_part.ptr());
      k.arg_ptr(2, b_.attn_out.ptr());
      launch(k, d_.q_heads, kM);
    }
  }

  void head() {
    norm(static_cast<const float*>(part_.final_norm->ptr()));   // embedding_norm, after the last layer
    if (part_.head_int8) {   // gemv_i8w(w, scales, x, out), grid (N / 64)
      const loader::DeviceWeight& lm = *part_.head_int8;
      require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden && lm.scales, "the int8 head is not [vocab][hidden]");
      l0::Kernel& k = kernel(kl::head_variant(true), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, b_.logits.ptr());
      launch(k, d_.vocab / kernels::kGemvI8wCols);
    } else {   // gemv_bf16(w, x, out) over the embedding tiles (tied: the gather's own copy, or the replica)
      const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(d_.vocab);
      l0::Kernel& k = kernel(kl::head_variant(false), "gemv_bf16", t.cols * t.ksplit);
      k.arg_ptr(0, part_.embed_tiles->ptr());
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, b_.logits.ptr());
      launch(k, d_.vocab / t.cols);
    }
    {
      l0::Kernel& k = kernel(kl::argmax1_variant(kM), "argmax_stage1", kl::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kl::argmax_groups(), kM);
    }
    {
      l0::Kernel& k = kernel(kl::argmax2_variant(), "argmax_stage2", kl::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
    require(kl::argmax_groups() <= kl::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
  }

  // --- the hand-off (spec 16b's: kolibri_capture.cc handoff_out / handoff_in) -----------------------------------------
  size_t h_bytes() const { return size_t(kM) * d_.hidden * 2; }
  size_t cut_slices_bytes() const { return size_t(cut_pending_s(d_, b_.placement)) * kM * d_.hidden * 4; }
  void check_link() const {
    require(link_->land_resid && link_->resid_cap >= h_bytes() && link_->sumsq_cap == cut_slices_bytes() &&
                (cut_slices_bytes() == 0 || link_->land_sumsq),
            "the landing buffer does not hold " + std::to_string(h_bytes()) + " B of the row and exactly " +
                std::to_string(cut_slices_bytes()) + " B of pending slices");
    if (link_->mode == PpHandoff::Copy) require(link_->event != nullptr, "the copy hand-off needs its cross-device event");
  }
  void handoff_out() {
    check_link();
    require(pending_s_ == cut_pending_s(d_, b_.placement), "device 0's list ends with other slices pending than the cut's");
    if (link_->mode == PpHandoff::Copy) {
      step_.list.copy(link_->land_resid, h(cur_).ptr(), h_bytes());
      if (cut_slices_bytes()) step_.list.copy(link_->land_sumsq, b_.slices.ptr(), cut_slices_bytes());
      step_.list.barrier_signal(link_->event);
      return;
    }
    require(link_->flag && link_->send_seq, "the peer hand-off needs its flag and counter");
    // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words), grid (1)
    l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_send", kernels::kPpHandoffWg);
    k.arg_ptr(0, h(cur_).ptr());
    k.arg_ptr(1, b_.slices.ptr());
    k.arg_ptr(2, link_->land_resid);
    k.arg_ptr(3, link_->land_sumsq);
    k.arg_ptr(4, link_->flag);
    k.arg_ptr(5, link_->send_seq);
    k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
    k.arg<uint32_t>(7, static_cast<uint32_t>(cut_slices_bytes() / 4));
    launch(k, 1);
  }
  void handoff_in() {
    check_link();
    cur_ = 0;
    if (link_->mode == PpHandoff::Copy) {
      step_.list.wait_event(link_->event);
      step_.list.copy(h(0).ptr(), link_->land_resid, h_bytes());
      if (cut_slices_bytes()) step_.list.copy(b_.slices.ptr(), link_->land_sumsq, cut_slices_bytes());
    } else {
      require(link_->flag && link_->recv_state && link_->spin_limit > 0,
              "the peer hand-off needs its flag, its state words and a spin bound");
      // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words, spin_limit)
      l0::Kernel& k = kernel(kernels::pp_handoff_variant(), "pp_recv", kernels::kPpHandoffWg);
      k.arg_ptr(0, link_->land_resid);
      k.arg_ptr(1, link_->land_sumsq);
      k.arg_ptr(2, link_->flag);
      k.arg_ptr(3, link_->recv_state);
      k.arg_ptr(4, h(0).ptr());
      k.arg_ptr(5, b_.slices.ptr());
      k.arg<uint32_t>(6, static_cast<uint32_t>(h_bytes() / 4));
      k.arg<uint32_t>(7, static_cast<uint32_t>(cut_slices_bytes() / 4));
      k.arg<uint32_t>(8, link_->spin_limit);
      launch(k, 1);
    }
    pending_s_ = cut_pending_s(d_, b_.placement);   // the receiving norm folds them (_S) or only normalises (_NF)
    if (pending_s_ && tap_) pending_tap_ = int(b_.placement.split) - 1;   // device 0 could not tap the dense cut
  }

 public:
  bool conv_tiles_ = false;

 private:
  l0::Context& ctx_;
  const loader::LfmDevicePart& part_;
  const model::Lfm2MoeDesc& d_;
  Lfm2MoeBuffers& b_;
  LfmFuse fuse_;
  const StageLink* link_;
  l0::Mem* tap_;
  ProfileEvents* prof_;
  CapturedStep step_;
  int layer_ = -1;
  int cur_ = 0;               // the residual's current buffer: h0 or h1
  uint32_t pending_s_ = 0;    // split-K slices waiting in b_.slices for the next norm to fold (0: none)
  int pending_tap_ = -1;      // a dense layer whose tap row the next fold writes
  size_t norm_sites_ = 0;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

uint32_t cut_pending_s(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p) {
  if (p.devices < 2) return 0;
  const uint32_t last = p.split - 1;
  if (!d.is_dense(last)) return 0;
  for (const model::LfmLinear& ln : d.linears(last))
    if (ln.id == model::LfmLinearId::DenseDown) return ln.shape.S;
  return 0;
}

PpLandingLayout landing_layout(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p) {
  return pp_landing_layout(size_t(kM) * d.hidden * 2, size_t(cut_pending_s(d, p)) * kM * d.hidden * 4);
}

CapturedStep build(l0::Context& ctx, const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d,
                   Lfm2MoeBuffers& b, LfmFuse fuse, bool conv_tiles, const StageLink* link, l0::Mem* tap,
                   ProfileEvents* prof) {
  Walk w(ctx, part, d, b, fuse, link, tap, prof);
  w.conv_tiles_ = conv_tiles;   // what the loader uploaded (LfmLoaded::conv_tiles): the arm conv needs exactly that
  return w.run();
}

}  // namespace runtime::lfm2moe
