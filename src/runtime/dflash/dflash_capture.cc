#include "runtime/dflash/dflash_capture.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"
#include "loader/dflash_layout.h"
#include "loader/draft_vocab.h"
#include "runtime/control.h"

// dflash_capture.cc - the DFlash drafter's draft and projection lists (dflash_capture.h has both walks in order). Every
// binding site names the kernel it honours; argument orders are each __kernel's in src/kernels/dflash/df_*.cl and the
// reused gemv_i8w.cl / gemv_bf16.cl.
namespace runtime::dflash {
namespace {

namespace kd = kernels::dflash;

static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, cur_token) == kernels::ctrl_index::kCurToken * 4,
              "runtime::Control and CTRL_DEFINES disagree");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash capture: " + what);
}
const uint8_t* at(const void* p, size_t off) { return static_cast<const uint8_t*>(p) + off; }

class Walk {
 public:
  Walk(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b)
      : ctx_(ctx), w_(w), d_(w.desc), b_(b), s_(w.desc.muse() ? kd::kMuse : kd::kQwen38),
        step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep draft(const DFlashTarget& t, const l0::Mem& control) {
    check_common();
    require(t.embed && t.head, "the target's embedding and head are not given");
    require(t.head->kind == model::WeightKind::Int8 || t.head->kind == model::WeightKind::Bf16,
            "the target's head is int4 (--quant_lm_head): no draft binaries for it - load the head int8 or bf16");
    require(t.head->shape.K == d_.hidden, "the target's head is not [rows][hidden]");
    require(t.vocab_rows == d_.vocab && t.embed->size() >= size_t(t.vocab_rows) * d_.hidden * 2,
            "the target's embedding / vocabulary is not the drafter's (vocab_size " + std::to_string(d_.vocab) + ")");
    const bool compact = t.ids != nullptr;
    const uint32_t N = t.head->shape.N;
    if (compact) {
      bool known = false;
      for (uint32_t nv : kd::draft_vocab_sizes()) known |= nv == N;
      require(known && t.ids->size() >= size_t(N) * 4,
              "the draft vocabulary's head has " + std::to_string(N) + " rows: compiled for 32768 / 65536 / 131072");
    } else {
      require(N == d_.vocab, "the full head has " + std::to_string(N) + " rows, not vocab_size");
    }
    require(b_.head_rows == N, "the buffers were sized for a head of " + std::to_string(b_.head_rows) + " rows, not " +
                                   std::to_string(N));
    require(control.size() >= sizeof(Control), "the control block is smaller than runtime::Control");
    const bool int8 = w_.format == loader::DraftFormat::Int8;
    if (s_.muse) require(!compact, "a draft vocabulary is not built for Muse-Glimmer's drafter");
    const std::vector<std::string> need =
        s_.muse ? kd::muse_draft_variants(fmt(), t.head->kind == model::WeightKind::Int8)
                : kd::draft_variants(d_.v2(), int8, t.head->kind == model::WeightKind::Int8, compact ? N : 0);
    for (const std::string& v : need)
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with " +
                  (s_.muse ? "B70_MUSE_DFLASH=ON" : "B70_DFLASH=ON"));
    const void* ctrl = control.ptr();
    const uint32_t M = s_.M, R = s_.R(), H = d_.hidden;

    layer_ = -1;
    {   // df_embed(ctrl, embed, mask_row, in_scale, resid), grid (1, M)
      l0::Kernel& k = kernel(kd::prep_variant(s_), "df_embed", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, t.embed->ptr());
      k.arg_ptr(2, w_.mask_row->ptr());
      k.arg(3, d_.in_scale);
      k.arg_ptr(4, b_.resid.ptr());
      launch(k, 1, M);
    }
    const size_t ring_layer = size_t(ring_slots(d_, 0, b_.max_len)) * d_.kv_n() * 2;
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DFlashLayerWeights& L = w_.layers[l];
      const float* norms = L.norms.as<float>();
      const void* attn_base = d_.v2() ? at(w_.conv_base->ptr(), loader::dflash_conv_base_off(d_, l, 0, 0) * 4) : nullptr;
      const void* mlp_base = d_.v2() ? at(w_.conv_base->ptr(), loader::dflash_conv_base_off(d_, l, 1, 0) * 4) : nullptr;
      // attention sub-block
      norm(b_.resid.ptr(), norms);
      const void* qkv_in = b_.x.ptr();
      if (d_.v2()) {
        gemv(*L.attn_kproj, b_.x.ptr(), b_.co->ptr());
        conv(true, b_.h.ptr(), attn_base, b_.xa.ptr());
        qkv_in = b_.xa.ptr();
      }
      gemv(L.qkv, qkv_in, b_.gout.ptr());
      {   // df_qkv_prep(ctrl, qkv, qn_kn, rope, q, kb, vb), grid (NH + 2 NKV, M), WG HD
        l0::Kernel& k = kernel(kd::prep_variant(s_), "df_qkv_prep", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.gout.ptr());
        k.arg_ptr(2, at(w_.qk_norm->ptr(), loader::dflash_qk_norm_off(d_, l, 0) * 4));
        k.arg_ptr(3, b_.rope.ptr());
        k.arg_ptr(4, b_.q.ptr());
        k.arg_ptr(5, b_.kb.ptr());
        k.arg_ptr(6, b_.vb.ptr());
        launch(k, d_.q_heads + 2 * d_.kv_heads, M);
      }
      {   // df_attn_part(ctrl, dctl, q, ring_k, ring_v, scale, part), grid (NKV, ring / PB), WG 128
        l0::Kernel& k = kernel(kd::attn_variant(s_), "df_attn_part", kd::kAttnWg);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.q.ptr());
        k.arg_ptr(3, at(b_.ring_k.ptr(), l * ring_layer));
        k.arg_ptr(4, at(b_.ring_v.ptr(), l * ring_layer));
        k.arg(5, scale_);
        k.arg_ptr(6, b_.part.ptr());
        launch(k, d_.kv_heads, ring_slots(d_, l, b_.max_len) / s_.pb);
      }
      {   // df_attn_reduce(ctrl, dctl, part, q, kb, vb, scale, out), grid (NH, M), WG HD
        l0::Kernel& k = kernel(kd::attn_variant(s_), "df_attn_reduce", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.part.ptr());
        k.arg_ptr(3, b_.q.ptr());
        k.arg_ptr(4, b_.kb.ptr());
        k.arg_ptr(5, b_.vb.ptr());
        k.arg(6, scale_);
        k.arg_ptr(7, b_.xa.ptr());
        launch(k, d_.q_heads, M);
      }
      gemv(L.o, b_.xa.ptr(), b_.gout.ptr());
      fold(b_.gout.ptr(), attn_base);
      // MLP sub-block
      norm(b_.resid.ptr(), norms + H);
      const void* mlp_in = b_.x.ptr();
      if (d_.v2()) {
        gemv(*L.mlp_kproj, b_.x.ptr(), b_.co->ptr());
        conv(true, b_.h.ptr(), mlp_base, b_.xa.ptr());
        mlp_in = b_.xa.ptr();
      }
      gemv(L.gate_up, mlp_in, b_.gout.ptr());
      {   // df_silu_mul(gu, a), grid (ceil(I / 256), M)
        l0::Kernel& k = kernel(kd::prep_variant(s_), "df_silu_mul", kd::kWg);
        k.arg_ptr(0, b_.gout.ptr());
        k.arg_ptr(1, b_.xa.ptr());
        launch(k, (d_.inter + kd::kWg - 1) / kd::kWg, M);
      }
      gemv(L.down, b_.xa.ptr(), b_.gout.ptr());
      fold(b_.gout.ptr(), mlp_base);
    }
    layer_ = -1;
    norm(b_.resid.ptr(), w_.final_norm->as<float>());
    // the target's head over rows 1..R (the mask rows), at M R (7; Muse 15)
    {
      const loader::DeviceWeight& hw = *t.head;
      if (hw.kind == model::WeightKind::Int8) {
        l0::Kernel& k = kernel(kernels::gemv_i8w_variant(R, H, N), "gemv_i8w", kernels::kGemvI8wCols);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, hw.scales->ptr());
        k.arg_ptr(2, at(b_.x.ptr(), size_t(H) * 2));
        k.arg_ptr(3, b_.logits.ptr());
        launch(k, N / kernels::kGemvI8wCols);
      } else {
        const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(N);
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(R, H, N, tl), "gemv_bf16", tl.cols * tl.ksplit);
        k.arg_ptr(0, hw.mem.ptr());
        k.arg_ptr(1, at(b_.x.ptr(), size_t(H) * 2));
        k.arg_ptr(2, b_.logits.ptr());
        launch(k, N / tl.cols);
      }
    }
    const std::string tv = kd::topk_variant(N, compact ? N : t.vocab_used, compact, R);
    {   // df_topk_stage1(logits, part), grid (chunks, R)
      l0::Kernel& k = kernel(tv, "df_topk_stage1", kd::kWg);
      k.arg_ptr(0, b_.logits.ptr());
      k.arg_ptr(1, b_.tpart.ptr());
      launch(k, (N + kd::kTopkChunk - 1) / kd::kTopkChunk, R);
    }
    {   // df_topk_stage2(part, ids, out_mult, softcap, cand, unary), grid (R)
      l0::Kernel& k = kernel(tv, "df_topk_stage2", kd::kWg);
      k.arg_ptr(0, b_.tpart.ptr());
      k.arg_ptr(1, compact ? t.ids->ptr() : b_.cand.ptr());   // the full head's build never reads it
      k.arg(2, t.q_mult >= 0.0f ? t.q_mult : d_.out_mult);
      k.arg(3, t.q_softcap >= 0.0f ? t.q_softcap : d_.softcap);
      k.arg_ptr(4, b_.cand.ptr());
      k.arg_ptr(5, b_.unary.ptr());
      launch(k, R);
    }
    if (d_.v2()) gemv(*w_.sel_proj, b_.x.ptr(), b_.hid->ptr());
    if (s_.muse) {   // spec 23f: mg_df_select(ctrl, dctl, cand, unary, out, vids), grid (1), WG 16
      require(bool(b_.vids), "a Muse-form drafter's buffers hold the verify's ids (vids)");
      l0::Kernel& k = kernel(kd::select_variant(false, s_), "mg_df_select", kd::kTopK);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.dctl.ptr());
      k.arg_ptr(2, b_.cand.ptr());
      k.arg_ptr(3, b_.unary.ptr());
      k.arg_ptr(4, b_.out.ptr());
      k.arg_ptr(5, b_.vids->ptr());
      launch(k, 1);
      return finish(draft_launches(d_), "draft");
    }
    {   // df_select(ctrl, dctl, cand, unary, hid, pred, succ, out), grid (1), WG 256
      l0::Kernel& k = kernel(kd::select_variant(d_.v2()), "df_select", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, b_.dctl.ptr());
      k.arg_ptr(2, b_.cand.ptr());
      k.arg_ptr(3, b_.unary.ptr());
      // v1's build reads none of the selector's buffers: any valid pointer
      k.arg_ptr(4, d_.v2() ? b_.hid->ptr() : b_.unary.ptr());
      k.arg_ptr(5, d_.v2() ? w_.pred->ptr() : b_.unary.ptr());
      k.arg_ptr(6, d_.v2() ? w_.succ->ptr() : b_.unary.ptr());
      k.arg_ptr(7, b_.out.ptr());
      launch(k, 1);
    }
    return finish(draft_launches(d_), "draft");
  }

  CapturedStep project() {
    check_common();
    const uint32_t M = s_.M;
    for (const std::string& v : {kd::prep_variant(s_), kd::linear_variant_fmt(fmt(), M, d_.fc_k(w_.target_hidden), d_.hidden),
                                 kd::linear_variant_fmt(fmt(), M, d_.hidden, 2 * d_.kv_n())})
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with " +
                  (s_.muse ? "B70_MUSE_DFLASH=ON" : "B70_DFLASH=ON"));
    if (s_.muse)
      require(std::ifstream(kernels::path(kd::select_variant(false, s_))).good(),
              kd::select_variant(false, s_) + " is not compiled: build with B70_MUSE_DFLASH=ON");
    layer_ = -1;
    gemv(*w_.fc, b_.taps.ptr(), b_.gout.ptr());
    norm(b_.gout.ptr(), w_.hidden_norm->as<float>());   // hidden_norm (output_norm_enc)
    // per layer the k||v rows [q_n, qkv_n) of q||k||v: n-tile-major layouts, so they are the weight from row q_n on
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DeviceWeight& qkv = w_.layers[l].qkv;
      const size_t K = d_.hidden, rows = d_.q_n(), n = 2 * d_.kv_n();
      void* out = const_cast<uint8_t*>(at(b_.kv.ptr(), size_t(l) * M * n * 4));
      if (qkv.kind == model::WeightKind::Int4) {
        // spec 23f: layout 1 is n-tile-major - the k||v rows are the tiles from n-tile rows / 16 on
        require(qkv.shape.layout == 1 && qkv.shape.S == 1, "an int4 drafter linear is layout 1 at S 1");
        const void* sub = at(qkv.mem.ptr(), rows / 16 * (K / 64) * 136 * 4);
        l0::Kernel& k = kernel(kernels::gemv_variant(M, uint32_t(K), uint32_t(n), 1, 1), "gemv", 64);
        k.arg_ptr(0, sub);
        k.arg_ptr(1, sub);   // layout 1 reads its scales from the tiles
        k.arg_ptr(2, b_.x.ptr());
        k.arg_ptr(3, out);
        launch(k, uint32_t(n) / 64, 1);
      } else if (qkv.kind == model::WeightKind::Int8) {
        l0::Kernel& k = kernel(kernels::gemv_i8w_variant(M, uint32_t(K), uint32_t(n)), "gemv_i8w",
                               kernels::kGemvI8wCols);
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
      l0::Kernel& k = kernel(kd::prep_variant(s_), "df_ctx_store", d_.head_dim);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, b_.kv.ptr());
      k.arg_ptr(2, w_.qk_norm->ptr());
      k.arg_ptr(3, b_.rope.ptr());
      k.arg_ptr(4, b_.ring_k.ptr());
      k.arg_ptr(5, b_.ring_v.ptr());
      launch(k, d_.layers * 2 * d_.kv_heads, M);
    }
    return finish(project_launches(d_), "projection");
  }

 private:
  // The kernels' shapes are z-lab/Qwen3.8-27B-DFlash2's beside Qwen3.8 (kernels::dflash::kQwen38); another drafter's
  // config is refused here by name before a command is appended (its binaries are its stage's: Ornith's v1 drafter
  // has a full-attention layer and a 16-row block).
  // The descriptor's form decides the shape (Qwen3.8's drafter, or spec 23f's Muse-Glimmer assistant).
  unsigned fmt() const {
    return w_.format == loader::DraftFormat::Int4G64 ? 2u : w_.format == loader::DraftFormat::Int8 ? 1u : 0u;
  }
  void check_common() {
    if (s_.muse) {
      check_muse();
      return;
    }
    require(w_.format != loader::DraftFormat::Int4G64,
            "an int4 g64 drafter is spec 23f's (Muse-Glimmer); Qwen3.8's lists bind int8 or bf16 linears");
    const kd::Shape& s = kd::kQwen38;
    require(d_.hidden == s.H && d_.inter == s.I && d_.q_heads == s.NH && d_.kv_heads == s.NKV && d_.head_dim == s.HD &&
                d_.layers == 5 && d_.taps.size() == 5 && w_.target_hidden == s.H,
            "the drafter's shapes are not z-lab/Qwen3.8-27B-DFlash2's (the kernels bake them)");
    require(d_.block == kd::kM, "the drafter's block is " + std::to_string(d_.block) + " rows; the lists are 8");
    for (uint32_t l = 0; l < d_.layers; ++l) {
      require(d_.window[l] == s.win, "layer " + std::to_string(l) + " is not a 2048-window sliding layer (a full "
                                     "layer needs every position's context - its stage, not this list)");
      require(!d_.causal[l], "layer " + std::to_string(l) + " is causal: the lists are non-causal (is_causal false)");
    }
    require(d_.eps == 1e-6f, "rms_norm_eps is not 1e-6 (df_prep bakes it)");
    if (d_.v2())
      require(d_.conv_group == s.group && d_.conv_taps == s.ctaps && d_.sel_rank == s.rank &&
                  d_.sel_top_k == kd::kTopK && d_.vocab == s.vocab,
              "the convolutions / selector are not the published drafter's (group 16, 2 taps, rank 256, top 16)");
    require(b_.desc.layers == d_.layers && b_.target_hidden == w_.target_hidden && b_.ring_k.size() == b_.sz.ring_k,
            "the buffers were sized for another drafter");
    require(b_.max_len >= 1, "the buffers' max_len is 0");
  }
  // Spec 23f: meta-models/Muse-Glimmer-30B-assistant's shapes (kernels::dflash::kMuse) - refused by name otherwise
  // (the kernels bake them): hidden 6656, 32 / 8 heads x 128, FF 19968, 5 layers, 5 taps of a 6656-wide target, the
  // block of 16, every layer sliding at 2048 with the inclusive edge, non-causal, eps 1e-5.
  void check_muse() {
    const kd::Shape& s = kd::kMuse;
    require(d_.hidden == s.H && d_.inter == s.I && d_.q_heads == s.NH && d_.kv_heads == s.NKV && d_.head_dim == s.HD &&
                d_.layers == 5 && d_.taps.size() == 5 && w_.target_hidden == s.H && d_.vocab == s.vocab,
            "the drafter's shapes are not meta-models/Muse-Glimmer-30B-assistant's beside Muse-Glimmer-30B (the "
            "kernels bake them)");
    require(d_.block == s.M, "the drafter's block is " + std::to_string(d_.block) + " rows; Muse's lists are 16");
    for (uint32_t l = 0; l < d_.layers; ++l) {
      require(d_.window[l] == 2048 && d_.window_inclusive && ring_slots(d_, l, b_.max_len) == s.ring,
              "layer " + std::to_string(l) + " is not a 2048-window (|q - k| <= 2048) sliding layer");
      require(!d_.causal[l], "layer " + std::to_string(l) + " is causal: the assistant's attention is bidirectional");
    }
    require(d_.eps == 1e-5f && d_.norm_two_roundings, "rms_norm_eps is not 1e-5 or the norm is not the assistant's "
                                                      "(df_prep / mg_dflash bake them)");
    require(b_.desc.layers == d_.layers && b_.target_hidden == w_.target_hidden && b_.ring_k.size() == b_.sz.ring_k &&
                bool(b_.vids),
            "the buffers were sized for another drafter");
    require(b_.max_len >= 1, "the buffers' max_len is 0");
  }

  // df_norm(src, w, h, x), grid (1, M) - or, a Muse-form drafter, mg_df_norm (the same arguments: the two-rounding
  // RMSNorm, mg_dflash.cl).
  void norm(const void* src, const float* w) {
    l0::Kernel& k = s_.muse ? kernel(kd::select_variant(false, s_), "mg_df_norm", kd::kWg)
                            : kernel(kd::prep_variant(s_), "df_norm", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, w);
    k.arg_ptr(2, b_.h.ptr());
    k.arg_ptr(3, b_.x.ptr());
    launch(k, 1, s_.M);
  }
  // df_conv_prep(src, co, base, out) (prepare: out bf16) - only the prepare side goes through here.
  void conv(bool prep, const void* src, const void* base, const void* out) {
    l0::Kernel& k = kernel(kd::prep_variant(s_), prep ? "df_conv_prep" : "df_conv_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.co->ptr());
    k.arg_ptr(2, base);
    k.arg_ptr(3, out);
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, s_.M);
  }
  // The sub-block's output into the residual: df_conv_add with the finish side (v2) or df_add (v1).
  void fold(const void* src, const void* base) {
    if (d_.v2()) {
      conv(false, src, base, b_.resid.ptr());
      return;
    }
    l0::Kernel& k = kernel(kd::prep_variant(s_), "df_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.resid.ptr());
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, s_.M);
  }
  // A drafter linear at the block's M rows: gemv_i8w(w, scales, x, out) or gemv_bf16(w, x, out).
  void gemv(const loader::DeviceWeight& w, const void* x, const void* out) {
    const model::GemvShape& s = w.shape;
    if (w.kind == model::WeightKind::Int4) {   // spec 23f: gemv.cl layout 1, S 1 - [M][N] straight into `out`
      require(s.layout == 1 && s.S == 1, "an int4 drafter linear is layout 1 at S 1");
      l0::Kernel& k = kernel(kernels::gemv_variant(s_.M, s.K, s.N, 1, 1), "gemv", 64);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, w.mem.ptr());   // layout 1 reads its scales from the tiles
      k.arg_ptr(2, x);
      k.arg_ptr(3, out);
      launch(k, s.N / 64, 1);
      return;
    }
    if (w.kind == model::WeightKind::Int8) {
      l0::Kernel& k = kernel(kernels::gemv_i8w_variant(s_.M, s.K, s.N), "gemv_i8w", kernels::kGemvI8wCols);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, w.scales->ptr());
      k.arg_ptr(2, x);
      k.arg_ptr(3, out);
      launch(k, s.N / kernels::kGemvI8wCols);
    } else {
      require(w.kind == model::WeightKind::Bf16, "a drafter linear is neither int8 nor bf16");
      const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(s.N);
      l0::Kernel& k = kernel(kernels::gemv_bf16_variant(s_.M, s.K, s.N, tl), "gemv_bf16", tl.cols * tl.ksplit);
      k.arg_ptr(0, w.mem.ptr());
      k.arg_ptr(1, x);
      k.arg_ptr(2, out);
      launch(k, s.N / tl.cols);
    }
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
    step_.labels.push_back((layer_ < 0 ? std::string("--") : "D" + std::to_string(layer_)) + " " + pending_entry_ +
                           " " + pending_variant_);
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
  const loader::DFlashWeights& w_;
  const model::DFlashDesc& d_;
  DFlashBuffers& b_;
  const kd::Shape& s_;
  CapturedStep step_;
  const float scale_ = float(1.0 / std::sqrt(double(d_.head_dim)));
  int layer_ = -1;
  std::string pending_entry_, pending_variant_;
};

}  // namespace

DFlashTarget dflash_target(const loader::LoadedModel& m, bool draft_vocab) {
  DFlashTarget t;
  t.embed = &m.embed;
  const loader::DeviceWeight& head = m.linears.at({loader::kTopLevel, model::LinearId::LmHead});
  t.vocab_rows = head.shape.N;
  t.vocab_used = m.desc ? m.desc->vocab_used : head.shape.N;
  if (draft_vocab) {
    if (!m.draft_vocab) throw std::runtime_error("dflash_target: the model was loaded without a draft vocabulary");
    t.head = &m.draft_vocab->head;
    t.ids = &m.draft_vocab->ids;
  } else {
    t.head = &head;
  }
  return t;
}

CapturedStep build_dflash_draft(l0::Context& ctx, const loader::DFlashWeights& w, const DFlashTarget& t,
                                DFlashBuffers& b, const l0::Mem& control) {
  return Walk(ctx, w, b).draft(t, control);
}

CapturedStep build_dflash_project(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b) {
  return Walk(ctx, w, b).project();
}

}  // namespace runtime::dflash
