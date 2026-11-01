#include "runtime/dflash/lilicorr_drafter.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"
#include "loader/lilicorr_layout.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_sizes.h"

// lilicorr_drafter.cc - the LiLiCorr drafter's buffers, its two captured lists and LiliDrafter (lilicorr_drafter.h has the
// walks in order). Every binding site names the kernel it honours; argument orders are each __kernel's in
// src/kernels/dflash/{df_prep,ll_*}.cl and the reused gemv_i8w.cl / gemv_bf16.cl / gemv.cl.
namespace runtime::dflash {
namespace {

namespace kd = kernels::dflash;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash LiLiCorr: " + what);
}
const uint8_t* at(const void* p, size_t off) { return static_cast<const uint8_t*>(p) + off; }
l0::Mem dev(l0::Context& ctx, size_t bytes) { return l0::Mem(ctx, l0::MemKind::Device, bytes); }

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::LiliWeights& w, LiliBuffers& b)
      : ctx_(ctx), w_(w), d_(w.desc), b_(b), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep draft(const LiliTarget& t, const l0::Mem& control) {
    check();
    require(t.embed && t.head, "the target's embedding replica and head are not given");
    require(t.head->kind == model::WeightKind::Int8 || t.head->kind == model::WeightKind::Bf16,
            "the target's head is int4: no draft binaries for it - load the head int8 or bf16");
    require(t.head->shape.K == d_.hidden && t.head->shape.N == d_.vocab && b_.vocab == d_.vocab,
            "the target's head is not [" + std::to_string(d_.vocab) + "][" + std::to_string(d_.hidden) + "]");
    require(t.embed->size() >= size_t(d_.vocab) * d_.hidden * 2, "the embedding replica is not [vocab][hidden] bf16");
    require(control.size() >= sizeof(Control), "the control block is smaller than runtime::Control");
    for (const std::string& v : kd::lilicorr_draft_variants(fmt(), t.head->kind == model::WeightKind::Int8))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_LILICORR=ON");
    const void* ctrl = control.ptr();
    const uint32_t M = kd::kM, S = kd::kLiliSlots, KC = kd::kLiliTopK, H = d_.hidden, R = d_.lil_rows(),
                   Hl = d_.lil_hidden, V = d_.vocab;
    const kd::Shape& s = kd::kLiLiCorr;
    layer_ = -1;
    {   // df_embed(ctrl, embed, mask_row, in_scale, resid), grid (1, M)
      l0::Kernel& k = kernel(kd::prep_variant(s), "df_embed", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, t.embed->ptr());
      k.arg_ptr(2, w_.mask_row->ptr());
      k.arg(3, d_.in_scale);
      k.arg_ptr(4, b_.resid.ptr());
      launch(k, 1, M);
    }
    const size_t ring_layer = size_t(lili_ring_slots(d_)) * d_.kv_n() * 2;
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::LiliLayer& L = w_.layers[l];
      const float* norms = L.norms.as<float>();
      const void* attn_base = at(w_.conv_base->ptr(), loader::dflash_conv_base_off(d_, l, 0, 0) * 4);
      const void* mlp_base = at(w_.conv_base->ptr(), loader::dflash_conv_base_off(d_, l, 1, 0) * 4);
      norm(b_.resid.ptr(), norms);
      gemv(L.attn_kproj, b_.x.ptr(), b_.co.ptr());
      conv(true, b_.h.ptr(), attn_base, b_.xa.ptr());
      gemv(L.qkv, b_.xa.ptr(), b_.gout.ptr());
      {   // df_qkv_prep(ctrl, qkv, qn_kn, rope, q, kb, vb), grid (NH + 2 NKV, M), WG HD
        l0::Kernel& k = kernel(kd::prep_variant(s), "df_qkv_prep", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.gout.ptr());
        k.arg_ptr(2, at(w_.qk_norm->ptr(), loader::dflash_qk_norm_off(d_, l, 0) * 4));
        k.arg_ptr(3, b_.rope.ptr());
        k.arg_ptr(4, b_.q.ptr());
        k.arg_ptr(5, b_.kb.ptr());
        k.arg_ptr(6, b_.vb.ptr());
        launch(k, d_.q_heads + 2 * d_.kv_heads, M);
      }
      {   // ll_attn_part(ctrl, dctl, q, ring_k, ring_v, scale, part), grid (NKV x G / HPW, ring / PB), WG 128
        l0::Kernel& k = kernel(kd::ll_attn_variant(), "ll_attn_part", 128);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.q.ptr());
        k.arg_ptr(3, at(b_.ring_k.ptr(), l * ring_layer));
        k.arg_ptr(4, at(b_.ring_v.ptr(), l * ring_layer));
        k.arg(5, scale_);
        k.arg_ptr(6, b_.part.ptr());
        launch(k, d_.kv_heads * (d_.q_heads / d_.kv_heads / kd::kLiliHpw), lili_ring_slots(d_) / s.pb);
      }
      {   // ll_attn_reduce(ctrl, dctl, part, q, kb, vb, sink, scale, out), grid (NH, M), WG HD
        l0::Kernel& k = kernel(kd::ll_attn_variant(), "ll_attn_reduce", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.part.ptr());
        k.arg_ptr(3, b_.q.ptr());
        k.arg_ptr(4, b_.kb.ptr());
        k.arg_ptr(5, b_.vb.ptr());
        k.arg_ptr(6, at(w_.sinks->ptr(), loader::lili_sink_off(d_, l) * 4));
        k.arg(7, scale_);
        k.arg_ptr(8, b_.xa.ptr());
        launch(k, d_.q_heads, M);
      }
      gemv(L.o, b_.xa.ptr(), b_.gout.ptr());
      conv(false, b_.gout.ptr(), attn_base, b_.resid.ptr());
      norm(b_.resid.ptr(), norms + H);
      gemv(L.mlp_kproj, b_.x.ptr(), b_.co.ptr());
      conv(true, b_.h.ptr(), mlp_base, b_.xa.ptr());
      gemv(L.gate_up, b_.xa.ptr(), b_.gout.ptr());
      {   // df_silu_mul(gu, a), grid (ceil(I / 256), M)
        l0::Kernel& k = kernel(kd::prep_variant(s), "df_silu_mul", kd::kWg);
        k.arg_ptr(0, b_.gout.ptr());
        k.arg_ptr(1, b_.xa.ptr());
        launch(k, (d_.inter + kd::kWg - 1) / kd::kWg, M);
      }
      gemv(L.down, b_.xa.ptr(), b_.gout.ptr());
      conv(false, b_.gout.ptr(), mlp_base, b_.resid.ptr());
    }
    layer_ = -1;
    norm(b_.resid.ptr(), w_.final_norm->as<float>());
    {   // the target's head over rows 1..7 (gemv_i8w / gemv_bf16 at M 7)
      const loader::DeviceWeight& hw = *t.head;
      if (hw.kind == model::WeightKind::Int8) {
        l0::Kernel& k = kernel(kernels::gemv_i8w_variant(S, H, V), "gemv_i8w", kernels::kGemvI8wCols);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, hw.scales->ptr());
        k.arg_ptr(2, at(b_.x.ptr(), size_t(H) * 2));
        k.arg_ptr(3, b_.logits.ptr());
        launch(k, V / kernels::kGemvI8wCols);
      } else {
        const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(V);
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(S, H, V, tl), "gemv_bf16", tl.cols * tl.ksplit);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, at(b_.x.ptr(), size_t(H) * 2));
        k.arg_ptr(2, b_.logits.ptr());
        launch(k, V / tl.cols);
      }
    }
    {   // ll_topk_stage1(logits, part), grid (chunks, 7); ll_topk_stage2(part, cand, logit, lp), grid (7)
      l0::Kernel& k1 = kernel(kd::ll_topk_variant(), "ll_topk_stage1", kd::kWg);
      k1.arg_ptr(0, b_.logits.ptr());
      k1.arg_ptr(1, b_.tpart.ptr());
      launch(k1, (V + kd::kTopkChunk - 1) / kd::kTopkChunk, S);
      l0::Kernel& k2 = kernel(kd::ll_topk_variant(), "ll_topk_stage2", kd::kWg);
      k2.arg_ptr(0, b_.tpart.ptr());
      k2.arg_ptr(1, b_.cand.ptr());
      k2.arg_ptr(2, b_.logit.ptr());
      k2.arg_ptr(3, b_.lp.ptr());
      launch(k2, S);
    }
    // the LiLiCorr head (ll_lattice.cl)
    const LiliHeadScratch hs = lili_head_scratch(d_);
    const loader::LiliHeadSmall so = loader::lili_head_small(d_);
    auto sc = [&](size_t off) { return at(b_.head.ptr(), off * 4); };
    using LH = loader::LiliHead;
    {   // ll_gemm_gather(embed, cand, wt, gb, y, rows, K, N, vrows): token_proj over the candidates' rows
      const loader::LiliHeadWeight& W = w_.head_at(LH::TokenProj);
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_gemm_gather", kd::kLiliGemmWg);
      k.arg_ptr(0, t.embed->ptr());
      k.arg_ptr(1, b_.cand.ptr());
      k.arg_ptr(2, W.wt.ptr());
      k.arg_ptr(3, W.gb.ptr());
      k.arg_ptr(4, sc(hs.tok));
      k.arg(5, R);
      k.arg(6, W.K);
      k.arg(7, W.N);
      k.arg(8, V);
      launch(k, W.N / kd::kLiliGemmWg);
    }
    gemm(b_.h.ptr(), H, H, w_.head_at(LH::PassHidden), sc(hs.pass), S, 0, 0);   // the mask rows' final-normed hidden
    {   // ll_feat(lp, small, fn_off, f), grid (1), WG RMAX
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_feat", kd::kLiliRmax);
      k.arg_ptr(0, b_.lp.ptr());
      k.arg_ptr(1, w_.head_small->ptr());
      k.arg(2, uint32_t(so.fn_w));
      k.arg_ptr(3, sc(hs.feat));
      launch(k, 1);
    }
    gemm(sc(hs.feat), 0, 5, w_.head_at(LH::FeatUp), sc(hs.fu), R, 0, 1);
    gemm(sc(hs.fu), 0, Hl, w_.head_at(LH::FeatDown), sc(hs.fm), R, 0, 0);
    {   // ll_sum(tok, pas, fm, small, slot_off, rank_off, h), grid (ceil(Hl / 256), R)
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_sum", kd::kWg);
      k.arg_ptr(0, sc(hs.tok));
      k.arg_ptr(1, sc(hs.pass));
      k.arg_ptr(2, sc(hs.fm));
      k.arg_ptr(3, w_.head_small->ptr());
      k.arg(4, uint32_t(so.slot));
      k.arg(5, uint32_t(so.rank));
      k.arg_ptr(6, sc(hs.xh));
      launch(k, (Hl + kd::kWg - 1) / kd::kWg, R);
    }
    gemm(b_.anchor.ptr(), 0, H, w_.head_at(LH::ContextProj), sc(hs.a), 1, 0, 0);
    {   // ll_valid(dctl, a)
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_valid", kd::kWg);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, sc(hs.a));
      launch(k, (Hl + kd::kWg - 1) / kd::kWg);
    }
    for (uint32_t l = 0; l < d_.lil_layers; ++l) {
      rmsnorm(sc(hs.xh), so.attn_norm(l, Hl), sc(hs.xn), R);
      gemm(sc(hs.xn), 0, Hl, w_.head_layer(l, LH::InProj), sc(hs.qkv), R, 0, 0);
      {   // ll_lattn(qkv, small, rel_off, same_off, scale, o), grid (heads, R), WG RMAX
        l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_lattn", kd::kLiliRmax);
        k.arg_ptr(0, sc(hs.qkv));
        k.arg_ptr(1, w_.head_small->ptr());
        k.arg(2, uint32_t(so.rel));
        k.arg(3, uint32_t(so.same));
        k.arg(4, float(1.0 / std::sqrt(double(d_.lil_head_dim()))));
        k.arg_ptr(5, sc(hs.o));
        launch(k, d_.lil_heads, R);
      }
      gemm(sc(hs.o), 0, Hl, w_.head_layer(l, LH::OutProj), sc(hs.xh), R, 1, 0);
      rmsnorm(sc(hs.xh), so.mlp_norm(l, Hl), sc(hs.xn), R);
      gemm(sc(hs.xn), 0, Hl, w_.head_layer(l, LH::MlpUp), sc(hs.u), R, 0, 1);
      gemm(sc(hs.u), 0, d_.lil_mlp, w_.head_layer(l, LH::MlpDown), sc(hs.xh), R, 1, 0);
    }
    rmsnorm(sc(hs.xh), so.out_norm, sc(hs.xn), R);
    rmsnorm(sc(hs.a), so.anchor_norm, sc(hs.an), 1);
    {   // ll_factor(xn, an, wt, gb, fh), grid (Hl / GW)
      const loader::LiliHeadWeight& W = w_.head_at(LH::Factor);
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_factor", kd::kLiliGemmWg);
      k.arg_ptr(0, sc(hs.xn));
      k.arg_ptr(1, sc(hs.an));
      k.arg_ptr(2, W.wt.ptr());
      k.arg_ptr(3, W.gb.ptr());
      k.arg_ptr(4, sc(hs.fh));
      launch(k, Hl / kd::kLiliGemmWg);
    }
    gemm(sc(hs.fh), 0, Hl, w_.head_at(LH::OutIn), sc(hs.ov), R, 0, 0);
    l2norm(sc(hs.ov), 2, R, 2 * d_.factor_dim);
    gemm(sc(hs.an), 0, Hl, w_.head_at(LH::AnchorOut), sc(hs.ao), 1, 0, 0);
    l2norm(sc(hs.ao), 1, 1, d_.factor_dim);
    {   // ll_scores(ov, ao, F, lscale, lat), grid (S, KC), WG KC
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_scores", KC);
      k.arg_ptr(0, sc(hs.ov));
      k.arg_ptr(1, sc(hs.ao));
      k.arg(2, d_.factor_dim);
      k.arg(3, d_.lil_logit_scale);
      k.arg_ptr(4, sc(hs.lat));
      launch(k, S, KC);
    }
    {   // ll_select(ctrl, dctl, cand, lp, logit, lat, out), grid (1)
      l0::Kernel& k = kernel(kd::ll_select_variant(), "ll_select", kd::kLiliSelectWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.dctl.ptr());
      k.arg_ptr(2, b_.cand.ptr());
      k.arg_ptr(3, b_.lp.ptr());
      k.arg_ptr(4, b_.logit.ptr());
      k.arg_ptr(5, sc(hs.lat));
      k.arg_ptr(6, b_.out.ptr());
      launch(k, 1);
    }
    return finish(lili_draft_launches(d_), "draft");
  }

  CapturedStep project() {
    check();
    const uint32_t M = kd::kM;
    for (const std::string& v : {kd::prep_variant(kd::kLiLiCorr), kd::ll_lattice_variant(),
                                 lili_name(*w_.fc), lili_name(w_.layers[0].qkv, 2 * d_.kv_n())})
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_LILICORR=ON");
    layer_ = -1;
    gemv(*w_.fc, b_.taps.ptr(), b_.gout.ptr());
    norm(b_.gout.ptr(), w_.hidden_norm->as<float>());   // hidden_norm
    {   // ll_anchor(dctl, h, n, anchor): the last projected row's normed vector
      l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_anchor", kd::kWg);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, b_.h.ptr());
      k.arg(2, d_.hidden);
      k.arg_ptr(3, b_.anchor.ptr());
      launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg);
    }
    // per layer the k||v rows [q_n, qkv_n) of q||k||v (n-tile-major layouts: the weight from row q_n on)
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DeviceWeight& qkv = w_.layers[l].qkv.w;
      require(!w_.layers[l].qkv.gs, "an NVFP4 q||k||v is not built (the published one is BF16)");
      const size_t K = d_.hidden, rows = d_.q_n(), n = 2 * d_.kv_n();
      void* out = const_cast<uint8_t*>(at(b_.kv.ptr(), size_t(l) * M * n * 4));
      if (qkv.kind == model::WeightKind::Int4) {
        const void* sub = at(qkv.mem.ptr(), rows / 16 * (K / 64) * 136 * 4);
        l0::Kernel& k = kernel(kernels::gemv_variant(M, uint32_t(K), uint32_t(n), 1, 1), "gemv", 64);
        k.arg_ptr(0, sub);
        k.arg_ptr(1, sub);
        k.arg_ptr(2, b_.x.ptr());
        k.arg_ptr(3, out);
        launch(k, uint32_t(n) / 64, 1);
      } else if (qkv.kind == model::WeightKind::Int8) {
        l0::Kernel& k = kernel(kernels::gemv_i8w_variant(M, uint32_t(K), uint32_t(n)), "gemv_i8w", kernels::kGemvI8wCols);
        k.arg_ptr(0, at(qkv.mem.ptr(), rows * K));
        k.arg_ptr(1, at(qkv.scales->ptr(), rows * 4));
        k.arg_ptr(2, b_.x.ptr());
        k.arg_ptr(3, out);
        launch(k, uint32_t(n) / kernels::kGemvI8wCols);
      } else {
        const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(uint32_t(n));
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(M, uint32_t(K), uint32_t(n), tl), "gemv_bf16",
                               tl.cols * tl.ksplit);
        k.arg_ptr(0, at(qkv.mem.ptr(), rows * K * 2));
        k.arg_ptr(1, b_.x.ptr());
        k.arg_ptr(2, out);
        launch(k, uint32_t(n) / tl.cols);
      }
    }
    layer_ = -1;
    {   // df_ctx_store(dctl, kv, qk_norm, rope, ring_k, ring_v), grid (L x 2 NKV, M), WG HD
      l0::Kernel& k = kernel(kd::prep_variant(kd::kLiLiCorr), "df_ctx_store", d_.head_dim);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, b_.kv.ptr());
      k.arg_ptr(2, w_.qk_norm->ptr());
      k.arg_ptr(3, b_.rope.ptr());
      k.arg_ptr(4, b_.ring_k.ptr());
      k.arg_ptr(5, b_.ring_v.ptr());
      launch(k, d_.layers * 2 * d_.kv_heads, M);
    }
    return finish(lili_project_launches(d_), "projection");
  }

 private:
  unsigned fmt() const {
    return w_.format == loader::DraftFormat::Int4G64 ? 2u : w_.format == loader::DraftFormat::Int8 ? 1u : 0u;
  }
  // The kernels bake the published drafter's shapes (kernels::dflash::kLiLiCorr): model::check_lilicorr, again here.
  void check() {
    model::check_lilicorr(d_);
    require(w_.format != loader::DraftFormat::Bf16, "a LiLiCorr drafter's formats are exact / int8 / int4 g64");
    require(w_.target_hidden == kd::kLiLiCorr.H && d_.vocab == kd::kLiLiCorr.vocab,
            "the target is not Bolt's shape (hidden 4096, vocab 131072)");
    require(lili_ring_slots(d_) == kd::kLiLiCorr.ring && b_.desc.layers == d_.layers &&
                b_.target_hidden == w_.target_hidden && b_.ring_k.size() == b_.sz.ring_k,
            "the buffers were sized for another drafter");
    require(b_.max_len >= 1, "the buffers' max_len is 0");
    require(loader::lili_has_token_proj(d_), "lilicorr_hidden_size == hidden_size (no token_proj) is not built");
  }
  std::string lili_name(const loader::LiliLinear& L, uint32_t N = 0) const {
    const model::GemvShape& s = L.w.shape;
    return kd::lili_linear_variant(fmt(), bool(L.gs), kd::kM, s.K, N ? N : s.N);
  }

  // df_norm(src, w, h, x), grid (1, M)
  void norm(const void* src, const float* w) {
    l0::Kernel& k = kernel(kd::prep_variant(kd::kLiLiCorr), "df_norm", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, w);
    k.arg_ptr(2, b_.h.ptr());
    k.arg_ptr(3, b_.x.ptr());
    launch(k, 1, kd::kM);
  }
  // df_conv_prep(src, co, base, out) (side 0, bf16 out) or df_conv_add(src, co, base, resid) (side 1, into the residual)
  void conv(bool prep, const void* src, const void* base, const void* out) {
    l0::Kernel& k = kernel(kd::prep_variant(kd::kLiLiCorr), prep ? "df_conv_prep" : "df_conv_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.co.ptr());
    k.arg_ptr(2, base);
    k.arg_ptr(3, out);
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, kd::kM);
  }
  // A backbone linear at M 8: ll_gemv_bf16_g(w, x, gs, out) for an exact NVFP4 one, else gemv_bf16 / gemv_i8w / gemv.cl
  void gemv(const loader::LiliLinear& L, const void* x, const void* out) {
    const model::GemvShape& s = L.w.shape;
    const uint32_t M = kd::kM;
    if (L.w.kind == model::WeightKind::Int4) {
      require(s.layout == 1 && s.S == 1, "an int4 drafter linear is layout 1 at S 1");
      l0::Kernel& k = kernel(kernels::gemv_variant(M, s.K, s.N, 1, 1), "gemv", 64);
      k.arg_ptr(0, L.w.mem.ptr());
      k.arg_ptr(1, L.w.mem.ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out);
      launch(k, s.N / 64, 1);
    } else if (L.w.kind == model::WeightKind::Int8) {
      l0::Kernel& k = kernel(kernels::gemv_i8w_variant(M, s.K, s.N), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, L.w.mem.ptr());
      k.arg_ptr(1, L.w.scales->ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out);
      launch(k, s.N / kernels::kGemvI8wCols);
    } else {
      const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(s.N);
      if (L.gs) {
        l0::Kernel& k = kernel(kd::ll_gemv_g_variant(M, s.K, s.N), "ll_gemv_bf16_g", tl.cols * tl.ksplit);
        k.arg_ptr(0, L.w.mem.ptr());
        k.arg_ptr(1, x);
        k.arg_ptr(2, L.gs->ptr());
        k.arg_ptr(3, out);
        launch(k, s.N / tl.cols);
      } else {
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(M, s.K, s.N, tl), "gemv_bf16", tl.cols * tl.ksplit);
        k.arg_ptr(0, L.w.mem.ptr());
        k.arg_ptr(1, x);
        k.arg_ptr(2, out);
        launch(k, s.N / tl.cols);
      }
    }
  }
  // ll_gemm(x, xoff, ldx, wt, gb, y, rows, K, N, mode, act), grid (N / GW)
  void gemm(const void* x, uint32_t xoff, uint32_t ldx, const loader::LiliHeadWeight& W, const void* y, uint32_t rows,
            uint32_t mode, uint32_t act) {
    l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_gemm", kd::kLiliGemmWg);
    k.arg_ptr(0, x);
    k.arg(1, xoff);
    k.arg(2, ldx);
    k.arg_ptr(3, W.wt.ptr());
    k.arg_ptr(4, W.gb.ptr());
    k.arg_ptr(5, y);
    k.arg(6, rows);
    k.arg(7, W.K);
    k.arg(8, W.N);
    k.arg(9, mode);
    k.arg(10, act);
    launch(k, W.N / kd::kLiliGemmWg);
  }
  // ll_rmsnorm(x, small, w_off, y, n), grid (1, rows)
  void rmsnorm(const void* x, size_t w_off, const void* y, uint32_t rows) {
    l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_rmsnorm", kd::kWg);
    k.arg_ptr(0, x);
    k.arg_ptr(1, w_.head_small->ptr());
    k.arg(2, uint32_t(w_off));
    k.arg_ptr(3, y);
    k.arg(4, d_.lil_hidden);
    launch(k, 1, rows);
  }
  // ll_l2norm(x, n, ld, eps), grid (halves, rows)
  void l2norm(const void* x, uint32_t halves, uint32_t rows, uint32_t ld) {
    l0::Kernel& k = kernel(kd::ll_lattice_variant(), "ll_l2norm", kd::kWg);
    k.arg_ptr(0, x);
    k.arg(1, d_.factor_dim);
    k.arg(2, ld);
    k.arg(3, d_.vector_eps);
    launch(k, halves, rows);
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
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + pending_entry_ + " " +
                           pending_variant_);
    pending_entry_.clear();
    ++step_.kernel_count;
  }
  CapturedStep finish(size_t want, const char* what) {
    require(step_.kernel_count == want, std::string("the ") + what + " list has " +
                                            std::to_string(step_.kernel_count) + " launches, not the " +
                                            std::to_string(want) + " runtime::dflash's formula gives");
    require(step_.kernels.size() == step_.kernel_count && step_.labels.size() == step_.kernel_count,
            "a kernel went unlaunched or unlabelled");
    step_.list.close();
    return std::move(step_);
  }

  l0::Context& ctx_;
  const loader::LiliWeights& w_;
  const model::DFlashDesc& d_;
  LiliBuffers& b_;
  CapturedStep step_;
  const float scale_ = float(1.0 / std::sqrt(double(d_.head_dim)));
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

LiliBuffers::LiliBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t th, uint32_t v, uint32_t ml)
    : desc(d),
      target_hidden(th),
      vocab(v),
      max_len(ml),
      sz(lili_sizes(d, th, v, ml)),
      dctl(ctx, l0::MemKind::Shared, sz.dctl),
      out(ctx, l0::MemKind::Shared, sz.out),
      rope(dev(ctx, sz.rope)),
      ring_k(dev(ctx, sz.ring_k)),
      ring_v(dev(ctx, sz.ring_v)),
      anchor(dev(ctx, sz.anchor)),
      taps(dev(ctx, sz.taps)),
      resid(dev(ctx, sz.resid)),
      h(dev(ctx, sz.h)),
      x(dev(ctx, sz.x)),
      xa(dev(ctx, sz.xa)),
      gout(dev(ctx, sz.gout)),
      co(dev(ctx, sz.co)),
      q(dev(ctx, sz.q)),
      kb(dev(ctx, sz.kb)),
      vb(dev(ctx, sz.vb)),
      part(dev(ctx, sz.part)),
      kv(dev(ctx, sz.kv)),
      logits(dev(ctx, sz.logits)),
      tpart(dev(ctx, sz.tpart)),
      cand(dev(ctx, sz.cand)),
      logit(dev(ctx, sz.logit)),
      lp(dev(ctx, sz.lp)),
      head(dev(ctx, sz.head)) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

size_t LiliBuffers::bytes() const {
  size_t b = 0;
  for (const l0::Mem* m : {&dctl, &out, &rope, &ring_k, &ring_v, &anchor, &taps, &resid, &h, &x, &xa, &gout, &co, &q, &kb,
                           &vb, &part, &kv, &logits, &tpart, &cand, &logit, &lp, &head})
    b += m->size();
  return b;
}

void LiliBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&ring_k, &ring_v, &anchor, &taps, &resid, &h, &x, &xa, &gout, &co, &q, &kb, &vb, &part, &kv, &logits,
                     &tpart, &cand, &logit, &lp, &head})
    imm.fill(m->ptr(), 0u, m->size());
  std::memset(dctl.ptr(), 0, dctl.size());
  std::memset(out.ptr(), 0, out.size());
  const std::vector<float> t = rope_table(desc.theta, desc.head_dim, max_len + lili_rows(desc));
  imm.copy(rope.ptr(), t.data(), t.size() * 4);
}

CapturedStep build_lili_draft(l0::Context& ctx, const loader::LiliWeights& w, const LiliTarget& t, LiliBuffers& b,
                              const l0::Mem& control) {
  return Walk(ctx, w, b).draft(t, control);
}
CapturedStep build_lili_project(l0::Context& ctx, const loader::LiliWeights& w, LiliBuffers& b) {
  return Walk(ctx, w, b).project();
}

LiliDrafter::LiliDrafter(l0::Context& ctx, loader::LiliWeights w, uint32_t max_len, uint32_t max_k)
    : w_(std::move(w)),
      b_(ctx, w_.desc, w_.target_hidden, w_.desc.vocab, max_len),
      max_k_(max_k),
      project_(build_lili_project(ctx, w_, b_)) {
  require(w_.desc.lilicorr(), "the weights are not a LiLiCorr drafter's (model::DFlashForm::LiLiCorr)");
  require(max_k >= 1 && max_k <= w_.desc.max_k(), "max_k " + std::to_string(max_k) + " is outside 1..7 (the block of 8)");
  std::vector<uint8_t> norms(w_.hidden_norm->size() + w_.final_norm->size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(norms.data(), w_.hidden_norm->ptr(), w_.hidden_norm->size());
  imm.copy(norms.data() + w_.hidden_norm->size(), w_.final_norm->ptr(), w_.final_norm->size());
  key_ = lili_drafter_key(w_.desc, w_.format, norms.data(), norms.size());
}

void LiliDrafter::attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head, const l0::Mem& control) {
  require(draft_ == nullptr, "attach() twice");
  LiliTarget t;
  t.embed = &embed;
  t.head = &head;
  draft_ = std::make_unique<CapturedStep>(build_lili_draft(ctx, w_, t, b_, control));
}

const LiliDraftOut& LiliDrafter::draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k, float temperature,
                                       uint64_t seed) {
  require(draft_ != nullptr, "draft before attach");
  require(k >= 1 && k <= max_k_, "k " + std::to_string(k) + " is outside 1.." + std::to_string(max_k_));
  require(temperature >= 0.0f, "the temperature must be >= 0");
  LiliCtl& c = *b_.ctl();
  c.k = draft_list_k(w_.desc, k);   // 7: the whole block, every slot walked
  c.temperature = temperature;
  c.seed_lo = uint32_t(seed);
  c.seed_hi = uint32_t(seed >> 32);
  replay(draft_->list);
  const LiliDraftOut& out = *b_.readback();
  require(out.k == c.k, "the draft list walked " + std::to_string(out.k) + " slots, not " + std::to_string(c.k));
  return out;
}

l0::CmdList& LiliDrafter::projection(uint32_t pos, uint32_t n) {
  require(n >= 1 && n <= kernels::dflash::kM, "a projection takes 1..8 rows, not " + std::to_string(n));
  require(size_t(pos) + n <= size_t(b_.max_len) + kernels::dflash::kM,
          "positions " + std::to_string(pos) + " + " + std::to_string(n) + " are past the RoPE table");
  LiliCtl& c = *b_.ctl();
  c.proj_pos = pos;
  c.proj_n = n;
  c.anchor_valid = 1;   // after this replay the anchor buffer holds row pos + n - 1
  return project_.list;
}

void LiliDrafter::save_ring(l0::CmdList& imm, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  imm.copy(h, b_.ring_k.ptr(), b_.ring_k.size());
  imm.copy(h + b_.ring_k.size(), b_.ring_v.ptr(), b_.ring_v.size());
  imm.copy(h + b_.ring_k.size() + b_.ring_v.size(), b_.anchor.ptr(), b_.anchor.size());
}

void LiliDrafter::load_ring(l0::CmdList& imm, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  imm.copy(b_.ring_k.ptr(), h, b_.ring_k.size());
  imm.copy(b_.ring_v.ptr(), h + b_.ring_k.size(), b_.ring_v.size());
  imm.copy(b_.anchor.ptr(), h + b_.ring_k.size() + b_.ring_v.size(), b_.anchor.size());
  b_.ctl()->anchor_valid = 1;
}

}  // namespace runtime::dflash
