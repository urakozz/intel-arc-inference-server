#include "runtime/k2/k2_capture.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/k2_kernels.h"
#include "kernels/kernels.h"
#include "runtime/control.h"

// k2_capture.cc - K2-Horizon's decode list (k2_capture.h has the walk in one table). Every
// binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/k2/*.cl and the reused sources.
namespace runtime::k2 {
namespace {

namespace kk = kernels::k2;

static_assert(kRouteWords == kk::route::kWords, "the K2 route row's width moved");
// The control block as CTRL_DEFINES gives it to embed_gather / k2_attn_prep / k2_mova_value /
// k2_attn / argmax (capture.cc pins the same pair).
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4 &&
                  offsetof(Control, out_token) == kernels::ctrl_index::kOutToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kM == 1, "the K2 binaries are M = 1 only");

uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::k2::build: " + what);
}

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b, l0::Mem* tap)
      : ctx_(ctx), m_(m), d_(*m.desc), b_(b), tap_(tap),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    check();
    embed();
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      layer(l);
      if (tap_) step_.list.copy(at(*tap_, size_t(l) * kM * d_.hidden * 2), b_.resid.ptr(),
                                size_t(kM) * d_.hidden * 2);
    }
    layer_ = -1;
    head();
    // Review Focus 5: the list's length is a property of the model.
    require(step_.kernel_count == decode_launches(d_),
            "the decode list has " + std::to_string(step_.kernel_count) + " launches, not the " +
                std::to_string(decode_launches(d_)) + " runtime::k2::decode_launches gives");
    require(step_.kernels.size() == step_.kernel_count, "a Kernel was created but never launched");
    require(step_.labels.size() == step_.kernel_count, "a launch went unlabelled");
    step_.list.close();
    return std::move(step_);
  }

 private:
  void check() {
    require(m_.max_len == b_.max_len, "model max_len " + std::to_string(m_.max_len) +
                                          " != buffers max_len " + std::to_string(b_.max_len));
    require(&b_.desc == m_.desc, "the buffers were sized for another K2 descriptor");
    require(m_.layers.size() == d_.layers, "the loaded model has " + std::to_string(m_.layers.size()) +
                                               " layers, the descriptor " + std::to_string(d_.layers));
    require(m_.rope && m_.rope->size() >= d_.rope_table_bytes(b_.max_len),
            "the RoPE table is shorter than max_len");
    const PersistentSizes ps = persistent_sizes(d_, b_.max_len);
    require(b_.kv_k.size() == ps.kv_k && b_.kv_v.size() == ps.kv_v, "the KV cache is not layers x max_len");
    const ScratchSizes ss = scratch_sizes(d_);
    require(b_.partials.size() == ss.partials && b_.routes.size() == ss.routes &&
                b_.attn_part.size() == ss.attn_part && b_.logits.size() == ss.logits,
            "the decode scratch is not runtime::k2::scratch_sizes'");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][M][hidden] bf16");
    // The kernels' constants the sizes were computed with (k2_sizes.cc spells them).
    require(ss.norm_sumsq == size_t(kk::kNormG) * kM * 4, "norm_sumsq is not kNormG floats");
    require(ss.attn_part == size_t(d_.q_heads) * kk::kAttnTgt * kM * kk::kAttnPart * 4,
            "attn_part is not [q_heads][kAttnTgt][M][130]");
    require(d_.head_dim == kk::kAttnWg, "k2_attn.cl is written for head_dim 128");
    for (uint32_t l = 0; l < d_.layers; ++l) {
      const loader::K2Layer& L = m_.layers[l];
      require(L.linears.size() == d_.layer_linears(l).size() && L.norms,
              "layer " + std::to_string(l) + " is not the descriptor's");
      if (!d_.is_dense(l))
        require(L.route && L.router && L.value && L.gate_up && L.down,
                "MoVA/MoE layer " + std::to_string(l) + " lacks its router or expert blocks");
    }
    // A missing binary is named here, before a command is appended (capture.cc's rule).
    for (const std::string& v : kk::decode_variants(d_, m_.lm_head->kind == model::WeightKind::Int8))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_K2=ON");
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
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " +
                           pending_entry_ + " " + pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }

  // embed_gather(ctrl, embed, resid) - embed_gather.cl at K2's hidden and vocabulary.
  void embed() {
    l0::Kernel& k = kernel(kk::embed_variant(kM, d_.hidden, d_.vocab), "embed_gather", kk::kEmbedWg);
    k.arg_ptr(0, b_.control.ptr());
    k.arg_ptr(1, m_.embed->ptr());
    k.arg_ptr(2, b_.resid.ptr());
    launch(k, 1, kM);
  }

  // The grouped RMSNorm: prep.cl's prep_res_fold (stage A, unchanged; folds `s_prev` slices
  // of `partials` into resid) then k2_norm_finish (stage B: two groups, plain w).
  void fold_norm(uint32_t s_prev, const void* norm_w) {
    {
      l0::Kernel& k = kernel(kernels::prep_res_fold_variant(kM, d_.hidden, s_prev, kk::kNormG),
                             "prep_res_fold", kk::kNormWg);
      k.arg_ptr(0, b_.partials.ptr());
      k.arg_ptr(1, b_.resid.ptr());
      k.arg_ptr(2, b_.norm_sumsq.ptr());
      launch(k, kk::kNormG, kM);
    }
    {
      l0::Kernel& k = kernel(kk::norm_variant(kM, d_.hidden, kk::kNormG, kk::kNormW, d_.norm_groups),
                             "k2_norm_finish", kk::kNormWg);
      k.arg_ptr(0, b_.norm_sumsq.ptr());
      k.arg_ptr(1, b_.resid.ptr());
      k.arg_ptr(2, norm_w);
      k.arg_ptr(3, b_.x.ptr());
      launch(k, kk::kNormW, kM);
    }
  }

  // gemv(w, scales, x, out) - gemv.cl, layout 0, grid (N / 64, S), WG 64.
  void gemv(const loader::DeviceWeight& w, const void* x) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4 && s.layout == 0 && w.scales,
            "a K2 int4 linear is GPTQ layout 0 with its own scales");
    require(b_.partials.size() >= size_t(s.S) * kM * s.N * 4, "partials is smaller than [S][M][N]");
    l0::Kernel& k = kernel(kernels::gemv_variant(kM, s.K, s.N, s.S, s.layout), "gemv", 64);
    k.arg_ptr(0, w.mem.ptr());
    k.arg_ptr(1, w.scales->ptr());
    k.arg_ptr(2, x);
    k.arg_ptr(3, b_.partials.ptr());
    launch(k, s.N / 64, s.S);
  }
  // gemv_bf16(w, x, out) - gemv_bf16.cl at kernels::gemv_bf16_tiling(N).
  void gemv_bf16(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
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
    const loader::K2Layer& L = m_.layers[l];
    const bool dense = d_.is_dense(l);
    // The fold brings in what the previous layer left in `partials`: nothing at layer 0
    // (embed_gather wrote resid) and after a MoE layer (k2_moe_down folded itself); the
    // dense down's S slices after a dense layer.
    const uint32_t s_prev = l == 0 || !d_.is_dense(l - 1) ? 0 : d_.down_s;
    fold_norm(s_prev, at(*L.norms, d_.norms_off_input()));
    gemv(L.linears[0], b_.x.ptr());   // q || k || gate || (v | v_router)
    void* kvk = b_.kv_k_layer(l);
    void* kvv = b_.kv_v_layer(l);
    const uint32_t attn_n = L.linears[0].shape.N;
    if (!dense) {
      // MoVA: route over the v_router columns of the fused row, then the 4 value experts
      // straight into this layer's V cache at pos.
      void* rt = at(b_.routes, mova_route_at(l));
      {
        l0::Kernel& k = kernel(kk::route_variant(kM, d_.value_experts, d_.value_top_k, attn_n,
                                                 d_.v_off(), L.linears[0].shape.S),
                               "k2_route", model::K2Desc::route_wg(d_.value_experts));
        k.arg_ptr(0, b_.partials.ptr());
        k.arg_ptr(1, at(*L.route, d_.route_off_mova()));
        k.arg_ptr(2, rt);
        launch(k, 1, kM);
      }
      {
        l0::Kernel& k = kernel(kk::mova_variant(kM, d_.value_experts, d_.value_top_k, d_.hidden, d_.kv_n()),
                               "k2_mova_value", kk::mova_wg(d_.value_top_k));
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, rt);
        k.arg_ptr(2, b_.x.ptr());
        k.arg_ptr(3, L.value->ptr());
        k.arg_ptr(4, kvv);
        launch(k, d_.kv_n() / 16, kM);
      }
    }
    // k2_attn_prep(ctrl, partials, rope, attn_q, attn_gate, kv_k, kv_v), grid (q + kv heads, M).
    {
      l0::Kernel& k = kernel(kk::attn_prep_variant(kM, attn_n, L.linears[0].shape.S, d_.q_heads,
                                                   d_.kv_heads, dense),
                             "k2_attn_prep", d_.head_dim);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.partials.ptr());
      k.arg_ptr(2, m_.rope->ptr());
      k.arg_ptr(3, b_.attn_q.ptr());
      k.arg_ptr(4, b_.attn_gate.ptr());
      k.arg_ptr(5, kvk);
      k.arg_ptr(6, kvv);
      launch(k, d_.q_heads + d_.kv_heads, kM);
    }
    // k2_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part), grid (kv heads, TGT);
    // k2_attn_reduce(ctrl, attn_part, attn_gate, attn_out), grid (q heads, M).
    {
      const std::string v = kk::attn_variant(kM, kk::kAttnTgt, d_.q_heads, d_.kv_heads);
      {
        l0::Kernel& k = kernel(v, "k2_attn_decode", kk::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_q.ptr());
        k.arg_ptr(2, kvk);
        k.arg_ptr(3, kvv);
        k.arg_ptr(4, b_.attn_part.ptr());
        launch(k, d_.kv_heads, kk::kAttnTgt);
      }
      {
        l0::Kernel& k = kernel(v, "k2_attn_reduce", kk::kAttnWg);
        k.arg_ptr(0, b_.control.ptr());
        k.arg_ptr(1, b_.attn_part.ptr());
        k.arg_ptr(2, b_.attn_gate.ptr());
        k.arg_ptr(3, b_.attn_out.ptr());
        launch(k, d_.q_heads, kM);
      }
    }
    gemv(L.linears[1], b_.attn_out.ptr());   // o_proj
    fold_norm(L.linears[1].shape.S, at(*L.norms, d_.norms_off_post()));
    if (dense) {
      gemv(L.linears[2], b_.x.ptr());   // gate || up
      {
        const uint32_t I = d_.dense_inter;
        l0::Kernel& k = kernel(kk::silu_variant(kM, I, L.linears[2].shape.S), "k2_silu_mul", kk::kSiluWg);
        k.arg_ptr(0, b_.partials.ptr());
        k.arg_ptr(1, b_.x.ptr());
        launch(k, (I + kk::kSiluChunk - 1) / kk::kSiluChunk, kM);
      }
      gemv(L.linears[3], b_.x.ptr());   // down: its partials fold into the next layer
      require(L.linears[3].shape.S == d_.down_s, "the dense down's S is not the descriptor's");
      return;
    }
    // The MoE block: the router GEMV (128 rows, 28 zero), the route, gate||up of the 8 + the
    // shared expert, down + the ascending-id combine + the residual fold.
    void* rt = at(b_.routes, moe_route_at(l));
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden,
            "the MoE router is not [router_n][hidden]");
    gemv_bf16(*L.router, b_.x.ptr(), b_.router);
    {
      l0::Kernel& k = kernel(kk::route_variant(kM, d_.experts, d_.top_k, d_.router_n(), 0, 1),
                             "k2_route", model::K2Desc::route_wg(d_.experts));
      k.arg_ptr(0, b_.router.ptr());
      k.arg_ptr(1, at(*L.route, d_.route_off_moe()));
      k.arg_ptr(2, rt);
      launch(k, 1, kM);
    }
    const std::string mv = kk::moe_variant(kM, d_.experts, d_.top_k, d_.hidden, d_.moe_inter);
    {
      l0::Kernel& k = kernel(mv, "k2_moe_gate_up", kk::moe_gate_up_wg());
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.x.ptr());
      k.arg_ptr(2, L.gate_up->ptr());
      k.arg_ptr(3, b_.moe_h.ptr());
      launch(k, kk::moe_gate_up_groups(d_.top_k, d_.moe_inter), kM);
    }
    {
      l0::Kernel& k = kernel(mv, "k2_moe_down", kk::moe_down_wg(d_.top_k));
      k.arg_ptr(0, rt);
      k.arg_ptr(1, b_.moe_h.ptr());
      k.arg_ptr(2, L.down->ptr());
      k.arg_ptr(3, b_.resid.ptr());
      launch(k, d_.hidden / 16, kM);
    }
  }

  void head() {
    const uint32_t last = d_.layers - 1;
    fold_norm(d_.is_dense(last) ? d_.down_s : 0, m_.final_norm->ptr());
    const loader::DeviceWeight& lm = *m_.lm_head;
    require(lm.shape.N == d_.vocab && lm.shape.K == d_.hidden, "lm_head is not [vocab][hidden]");
    if (lm.kind == model::WeightKind::Int8) {
      require(bool(lm.scales), "the int8 lm_head has no row scales");
      l0::Kernel& k = kernel(kernels::gemv_i8w_variant(kM, lm.shape.K, lm.shape.N), "gemv_i8w",
                             kernels::kGemvI8wCols);
      k.arg_ptr(0, lm.mem.ptr());
      k.arg_ptr(1, lm.scales->ptr());
      k.arg_ptr(2, b_.x.ptr());
      k.arg_ptr(3, b_.logits.ptr());
      launch(k, lm.shape.N / kernels::kGemvI8wCols);
    } else {
      gemv_bf16(lm, b_.x.ptr(), b_.logits);
    }
    {
      l0::Kernel& k = kernel(kk::argmax1_variant(kM, d_.vocab, d_.vocab_used), "argmax_stage1",
                             kk::kArgmaxWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, kk::argmax_groups(d_.vocab), kM);
    }
    {
      l0::Kernel& k = kernel(kk::argmax2_variant(d_.vocab), "argmax_stage2", kk::kArgmaxWg);
      k.arg_ptr(0, b_.control.ptr());
      k.arg_ptr(1, b_.argmax_part.ptr());
      launch(k, 1, 1);
    }
    require(kk::argmax_groups(d_.vocab) <= kk::kArgmaxWg, "argmax_stage2 folds at most 256 groups");
  }

  l0::Context& ctx_;
  const loader::K2LoadedModel& m_;
  const model::K2Desc& d_;
  K2Buffers& b_;
  l0::Mem* tap_;
  CapturedStep step_;
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b, l0::Mem* tap) {
  return Walk(ctx, m, b, tap).run();
}

}  // namespace runtime::k2
