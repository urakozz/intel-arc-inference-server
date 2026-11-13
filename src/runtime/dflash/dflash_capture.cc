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
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm_l0.h"

// dflash_capture.cc - the DFlash drafter's draft and projection lists (dflash_capture.h has both walks in order). Every
// binding site names the kernel it honours; argument orders are each __kernel's in src/kernels/dflash/df_*.cl and the
// reused gemm_sm.cl (gemm_sm_i8w; gemv_i8w.cl under the test override) / gemv_bf16.cl.
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
  Walk(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b, const CaptureOptions& opt)
      : ctx_(ctx), w_(w), d_(w.desc), b_(b), opt_(opt), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {
    if (opt_.profile) opt_.profile->events.clear();
  }

  CapturedStep draft(const DFlashTarget& t, const l0::Mem& control) {
    check_common();
    require(t.embed && t.head, "the target's embedding and head are not given");
    require(t.head->kind == model::WeightKind::Int8 || t.head->kind == model::WeightKind::Bf16,
            "the target's head is int4 (--quant_lm_head): no draft binaries for it - load the head int8 or bf16");
    require(t.head->shape.K == d_.hidden, "the target's head is not [rows][hidden]");
    require(t.vocab_rows == d_.vocab && t.embed->size() >= size_t(t.vocab_rows) * d_.hidden * 2,
            "the target's embedding / vocabulary is not the drafter's (vocab_size " + std::to_string(d_.vocab) + ")");
    const bool compact = t.ids != nullptr;
    require(!w_.candidate_head || (!compact && t.head->kind == model::WeightKind::Int8),
            "the int4 candidate head requires a full int8 target head for exact rerank");
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
    const unsigned fmt = w_.format == loader::DraftFormat::Int4G64 ? 2u : w_.format == loader::DraftFormat::Int8 ? 1u : 0u;
    for (const std::string& v : kd::draft_variants(d_.v2(), fmt, t.head->kind == model::WeightKind::Int8,
                                                   compact ? N : 0, lkind(), hkind(), opt_.dflash_attn_heads, opt_.dflash_attn_rows, bool(w_.candidate_head), opt_.dflash_attn_parallel_exp,
                                                   opt_.dflash_attn_dpas, opt_.dflash_attn_ks))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_DFLASH=ON");
    const void* ctrl = control.ptr();
    const uint32_t M = kd::kM, H = d_.hidden;

    layer_ = -1;
    {   // df_embed(ctrl, embed, mask_row, in_scale, resid), grid (1, M)
      l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_embed", kd::kWg);
      k.arg_ptr(0, ctrl);
      k.arg_ptr(1, t.embed->ptr());
      k.arg_ptr(2, w_.mask_row->ptr());
      k.arg(3, d_.in_scale);
      k.arg_ptr(4, b_.resid.ptr());
      launch(k, 1, M);
    }
    const size_t ring_layer = size_t(d_.window[0]) * d_.kv_n() * 2;
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
        l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_qkv_prep", d_.head_dim);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.gout.ptr());
        k.arg_ptr(2, at(w_.qk_norm->ptr(), loader::dflash_qk_norm_off(d_, l, 0) * 4));
        k.arg_ptr(3, b_.rope.ptr());
        k.arg_ptr(4, b_.q.ptr());
        k.arg_ptr(5, b_.kb.ptr());
        k.arg_ptr(6, b_.vb.ptr());
        launch(k, d_.q_heads + 2 * d_.kv_heads, M);
      }
      {   // df_attn_part(ctrl, dctl, q, ring_k, ring_v, scale, part), grid (NKV, ring / PB), WG 128 (DPAS: 16 a q head)
        l0::Kernel& k = opt_.dflash_attn_dpas
                            ? kernel(kd::attn_dpas_variant(kd::kQwen38, opt_.dflash_attn_ks), "df_attn_part",
                                     16 * (d_.q_heads / d_.kv_heads) * opt_.dflash_attn_ks)
                            : kernel(kd::attn_variant(kd::kQwen38, opt_.dflash_attn_heads, opt_.dflash_attn_rows, opt_.dflash_attn_parallel_exp), "df_attn_part", kd::kAttnWg);
        k.arg_ptr(0, ctrl);
        k.arg_ptr(1, b_.dctl.ptr());
        k.arg_ptr(2, b_.q.ptr());
        k.arg_ptr(3, at(b_.ring_k.ptr(), l * ring_layer));
        k.arg_ptr(4, at(b_.ring_v.ptr(), l * ring_layer));
        k.arg(5, scale_);
        k.arg_ptr(6, b_.part.ptr());
        if (opt_.dflash_attn_dpas)
          launch(k, d_.kv_heads, d_.window[l] / kd::kAttnPb, 1);
        else
          launch(k, opt_.dflash_attn_heads ? d_.q_heads / opt_.dflash_attn_heads : d_.kv_heads,
                 d_.window[l] / kd::kAttnPb, opt_.dflash_attn_rows ? kd::kM / opt_.dflash_attn_rows : 1);
      }
      {   // df_attn_reduce(ctrl, dctl, part, q, kb, vb, scale, out), grid (NH, M), WG HD
        l0::Kernel& k = kernel(kd::attn_variant(kd::kQwen38, opt_.dflash_attn_heads, opt_.dflash_attn_rows, opt_.dflash_attn_parallel_exp), "df_attn_reduce", d_.head_dim);
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
        l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_silu_mul", kd::kWg);
        k.arg_ptr(0, b_.gout.ptr());
        k.arg_ptr(1, b_.xa.ptr());
        launch(k, (d_.inter + kd::kWg - 1) / kd::kWg, M);
      }
      gemv(L.down, b_.xa.ptr(), b_.gout.ptr());
      fold(b_.gout.ptr(), mlp_base);
    }
    layer_ = -1;
    norm(b_.resid.ptr(), w_.final_norm->as<float>());
    if (w_.candidate_head) {
      i4_linear(kd::kR, H, N, w_.candidate_head->mem.ptr(), at(b_.x.ptr(), size_t(H) * 2), b_.logits.ptr(),
                w_.candidate_bits == 2 ? 3u : 1u);
      {
        auto& k = kernel(kd::candidate_topk_variant(), "df_candidate_select1", kd::kWg);   // spec 19 v3d Task 7
        k.arg_ptr(0, b_.logits.ptr());
        k.arg_ptr(1, b_.tpart.ptr());
        launch(k, (N + kd::kTopkChunk - 1) / kd::kTopkChunk, kd::kR);
      }
      {
        auto& k = kernel(kd::candidate_topk_variant(), "df_candidate_stage2", kd::kWg);
        k.arg_ptr(0, b_.tpart.ptr());
        k.arg_ptr(1, b_.candidate->ptr());
        launch(k, kd::kR);
      }
      {
        auto& k = kernel(kd::rerank_variant(), "df_rerank", kd::kWg);
        k.arg_ptr(0, t.head->mem.ptr());
        k.arg_ptr(1, t.head->scales->ptr());
        k.arg_ptr(2, at(b_.x.ptr(), size_t(H) * 2));
        k.arg_ptr(3, b_.candidate->ptr());
        k.arg(4, d_.out_mult);
        k.arg(5, d_.softcap);
        k.arg_ptr(6, b_.cand.ptr());
        k.arg_ptr(7, b_.unary.ptr());
        launch(k, kd::kR);
      }
    } else {
      // the target's head over rows 1..7 (the mask rows), at M 7
      {
        const loader::DeviceWeight& hw = *t.head;
        if (hw.kind == model::WeightKind::Int8) {
          i8_linear(kd::kR, H, N, hw.mem.ptr(), hw.scales->ptr(), at(b_.x.ptr(), size_t(H) * 2), b_.logits.ptr());
        } else {
          const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(N);
          l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kd::kR, H, N, tl), "gemv_bf16", tl.cols * tl.ksplit);
          k.arg_ptr(0, hw.mem.ptr());
          k.arg_ptr(1, at(b_.x.ptr(), size_t(H) * 2));
          k.arg_ptr(2, b_.logits.ptr());
          launch(k, N / tl.cols);
        }
      }
      const std::string tv = kd::topk_variant(N, compact ? N : t.vocab_used, compact);
      {   // df_topk_stage1(logits, part), grid (chunks, R)
        l0::Kernel& k = kernel(tv, "df_topk_stage1", kd::kWg);
        k.arg_ptr(0, b_.logits.ptr());
        k.arg_ptr(1, b_.tpart.ptr());
        launch(k, (N + kd::kTopkChunk - 1) / kd::kTopkChunk, kd::kR);
      }
      {   // df_topk_stage2(part, ids, out_mult, softcap, cand, unary), grid (R)
        l0::Kernel& k = kernel(tv, "df_topk_stage2", kd::kWg);
        k.arg_ptr(0, b_.tpart.ptr());
        k.arg_ptr(1, compact ? t.ids->ptr() : b_.cand.ptr());   // the full head's build never reads it
        k.arg(2, d_.out_mult);
        k.arg(3, d_.softcap);
        k.arg_ptr(4, b_.cand.ptr());
        k.arg_ptr(5, b_.unary.ptr());
        launch(k, kd::kR);
      }
    }
    if (d_.v2()) gemv(*w_.sel_proj, b_.x.ptr(), b_.hid->ptr());
    {   // df_select(ctrl, dctl, cand, unary, hid, pred, succ, out), grid (1), WG 256
      l0::Kernel& k = kernel(kd::select_variant(d_.v2(), kd::kQwen38, w_.format == loader::DraftFormat::Int4G64), "df_select", kd::kWg);
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
    return finish(draft_launches(d_, bool(w_.candidate_head)), "draft");
  }

  CapturedStep project() {
    check_common();
    const unsigned fmt = w_.format == loader::DraftFormat::Int4G64 ? 2u : w_.format == loader::DraftFormat::Int8 ? 1u : 0u;
    for (const std::string& v : {kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                                 kd::linear_variant(fmt, kd::kM, d_.fc_k(w_.target_hidden), d_.hidden, lkind()),
                                 kd::linear_variant(fmt, kd::kM, d_.hidden, 2 * d_.kv_n(), lkind())})
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_DFLASH=ON");
    layer_ = -1;
    gemv(*w_.fc, b_.taps.ptr(), b_.gout.ptr());
    {   // df_norm(src, w, h, x) with hidden_norm
      l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             w_.format == loader::DraftFormat::Int4G64 ? "df_norm_fc" : "df_norm", kd::kWg);
      k.arg_ptr(0, b_.gout.ptr());
      k.arg_ptr(1, w_.hidden_norm->ptr());
      k.arg_ptr(2, b_.h.ptr());
      k.arg_ptr(3, b_.x.ptr());
      launch(k, 1, kd::kM);
    }
    // per layer the k||v rows [q_n, qkv_n) of q||k||v: n-tile-major layouts, so they are the weight from row q_n on
    for (uint32_t l = 0; l < d_.layers; ++l) {
      layer_ = int(l);
      const loader::DeviceWeight& qkv = w_.layers[l].qkv;
      const size_t K = d_.hidden, rows = d_.q_n(), n = 2 * d_.kv_n();
      const unsigned S = w_.format == loader::DraftFormat::Int4G64 ? kd::int4_slices(uint32_t(K), uint32_t(n)) : 1;
      void* out = const_cast<uint8_t*>(at(b_.kv.ptr(), size_t(l) * S * kd::kM * n * 4));
      if (qkv.kind == model::WeightKind::Int4) {
        i4_linear(kd::kM, uint32_t(K), uint32_t(n),
                  at(qkv.mem.ptr(), loader::dflash_linear_bytes(uint32_t(K), uint32_t(rows), w_.format)), b_.x.ptr(), out);
      } else if (qkv.kind == model::WeightKind::Int8) {
        // the int8 tiles are [N/16][K/16][16][16]: row q_n (a multiple of 16) starts tile q_n / 16
        i8_linear(kd::kM, uint32_t(K), uint32_t(n), at(qkv.mem.ptr(), rows * K), at(qkv.scales->ptr(), rows * 4),
                  b_.x.ptr(), out);
      } else {
        const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(uint32_t(n));
        l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kd::kM, uint32_t(K), uint32_t(n), tl), "gemv_bf16",
                               tl.cols * tl.ksplit);
        k.arg_ptr(0, at(qkv.mem.ptr(), rows * K * 2));
        k.arg_ptr(1, b_.x.ptr());
        k.arg_ptr(2, out);
        launch(k, uint32_t(n) / tl.cols);
      }
    }
    layer_ = -1;
    {   // df_ctx_store(dctl, kv, qk_norm, rope, ring_k, ring_v), grid (L x 2 NKV, M), WG HD
      l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_ctx_store", d_.head_dim);
      k.arg_ptr(0, b_.dctl.ptr());
      k.arg_ptr(1, b_.kv.ptr());
      k.arg_ptr(2, w_.qk_norm->ptr());
      k.arg_ptr(3, b_.rope.ptr());
      k.arg_ptr(4, b_.ring_k.ptr());
      k.arg_ptr(5, b_.ring_v.ptr());
      launch(k, d_.layers * 2 * d_.kv_heads, kd::kM);
    }
    return finish(project_launches(d_), "projection");
  }

 private:
  // The kernels' shapes are z-lab/Qwen3.8-27B-DFlash2's beside Qwen3.8 (kernels::dflash::kQwen38); another drafter's
  // config is refused here by name before a command is appended (its binaries are its stage's: Ornith's v1 drafter
  // has a full-attention layer and a 16-row block).
  void check_common() {
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

  void norm(const void* src, const float* w) {   // df_norm(src, w, h, x), grid (1, M)
    l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_norm", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, w);
    k.arg_ptr(2, b_.h.ptr());
    k.arg_ptr(3, b_.x.ptr());
    launch(k, 1, kd::kM);
  }
  // df_conv_prep(src, co, base, out) (prepare: out bf16) - only the prepare side goes through here.
  void conv(bool prep, const void* src, const void* base, const void* out) {
    l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                           prep ? "df_conv_prep" : "df_conv_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.co->ptr());
    k.arg_ptr(2, base);
    k.arg_ptr(3, out);
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, kd::kM);
  }
  // The sub-block's output into the residual: df_conv_add with the finish side (v2) or df_add (v1).
  void fold(const void* src, const void* base) {
    if (d_.v2()) {
      conv(false, src, base, b_.resid.ptr());
      return;
    }
    l0::Kernel& k = kernel(kd::prep_variant(kd::kQwen38, w_.format == loader::DraftFormat::Int4G64),
                             "df_add", kd::kWg);
    k.arg_ptr(0, src);
    k.arg_ptr(1, b_.resid.ptr());
    launch(k, (d_.hidden + kd::kWg - 1) / kd::kWg, kd::kM);
  }
  // The int8 linears' kernel (runtime/gemv_kind.h): the lists are M >= 2, so SmallM unless a test forces gemv.
  kernels::LinearKind lkind() const { return linear_kind(kd::kM, opt_); }
  kernels::LinearKind hkind() const {
    CaptureOptions head = opt_;
    if (head.force_dflash_head != LinearForce::Auto) head.force_linear = head.force_dflash_head;
    return linear_kind(kd::kR, head);
  }
  // An int8 linear at M rows (spec 28d): gemm_sm_i8w(w, scales, x, out) at the shape's kSmallMTuning row - the
  // in-work-group split-K, grid N / 32, WG 256 - or, forced, gemv_i8w(w, scales, x, out), grid N / 64, WG 64. The same
  // arguments, the same [M][N] fp32 output, scaled.
  void i8_linear(uint32_t M, uint32_t K, uint32_t N, const void* w, const void* scales, const void* x, const void* out) {
    const auto kind = M == kd::kR ? hkind() : lkind();
    const kernels::SmallMTuning* t =
        kind == kernels::LinearKind::SmallM ? kernels::small_m_tuning(K, N, 1, 2) : nullptr;
    require(kind == kernels::LinearKind::Gemv || t, "no gemm_sm_i8w row for the int8 linear " + std::to_string(K) +
                                                           " x " + std::to_string(N) + " (kernels::kSmallMTuning)");
    l0::Kernel& k = t ? kernel(kernels::gemm_sm_variant(M, *t), kernels::gemm_sm_entry(*t), kernels::gemm_sm_wg_size(*t))
                      : kernel(kernels::gemv_i8w_variant(M, K, N), "gemv_i8w", kernels::kGemvI8wCols);
    k.arg_ptr(0, w);
    k.arg_ptr(1, scales);
    k.arg_ptr(2, x);
    k.arg_ptr(3, out);
    if (t)
      launch(k, N / kernels::gemm_sm_cols(*t), kernels::gemm_sm_grid_y(*t));
    else
      launch(k, N / kernels::kGemvI8wCols);
  }
  // Int4 g64 layout 1, grid split-K, at the block's M rows; scales travel inside the weight tiles.
  // layout 3: spec 19 v3d's 2-bit candidate head (gemm_sm only, one slice).
  void i4_linear(uint32_t M, uint32_t K, uint32_t N, const void* w, const void* x, const void* out, unsigned layout = 1) {
    const unsigned S = layout == 3 ? 1 : kd::int4_slices(K, N);
    const auto kind = M == kd::kR ? hkind() : lkind();
    const kernels::SmallMTuning* t =
        kind == kernels::LinearKind::SmallM || layout == 3 ? kernels::small_m_tuning(K, N, S, layout) : nullptr;
    require(kind == kernels::LinearKind::Gemv || t, "no gemm_sm row for the int4 linear " + std::to_string(K) +
                                                       " x " + std::to_string(N) + " (kernels::kSmallMTuning)");
    l0::Kernel& k = t ? kernel(kernels::gemm_sm_variant(M, *t), kernels::gemm_sm_entry(*t), kernels::gemm_sm_wg_size(*t))
                      : kernel(kernels::gemv_variant(M, K, N, S, 1), "gemv", 64);
    k.arg_ptr(0, w);
    k.arg_ptr(1, w);
    k.arg_ptr(2, x);
    k.arg_ptr(3, out);
    if (t)
      launch(k, N / kernels::gemm_sm_cols(*t), kernels::gemm_sm_grid_y(*t));
    else
      launch(k, N / 64, S);
  }
  // A drafter linear at the block's M rows: a quantised linear above or gemv_bf16(w, x, out).
  void gemv(const loader::DeviceWeight& w, const void* x, const void* out) {
    const model::GemvShape& s = w.shape;
    if (w.kind == model::WeightKind::Int4) {
      i4_linear(kd::kM, s.K, s.N, w.mem.ptr(), x, out);
    } else if (w.kind == model::WeightKind::Int8) {
      i8_linear(kd::kM, s.K, s.N, w.mem.ptr(), w.scales->ptr(), x, out);
    } else {
      require(w.kind == model::WeightKind::Bf16, "a drafter linear is not int4, int8 or bf16");
      const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(s.N);
      l0::Kernel& k = kernel(kernels::gemv_bf16_variant(kd::kM, s.K, s.N, tl), "gemv_bf16", tl.cols * tl.ksplit);
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
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    require(!pending_entry_.empty(), "launch() without a preceding kernel()");
    l0::Event* event = nullptr;
    if (opt_.profile) {
      require(step_.kernel_count < opt_.profile->pool.capacity(), "the profile event pool is full");
      opt_.profile->events.emplace_back(opt_.profile->pool, uint32_t(step_.kernel_count));
      event = &opt_.profile->events.back();
    }
    step_.list.launch(k, gx, gy, gz, event);
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
  const CaptureOptions opt_;
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
                                DFlashBuffers& b, const l0::Mem& control, const CaptureOptions& opt) {
  return Walk(ctx, w, b, opt).draft(t, control);
}

CapturedStep build_dflash_project(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b,
                                  const CaptureOptions& opt) {
  return Walk(ctx, w, b, opt).project();
}

// project_wide: the projection list's walk (Walk::project above) with every GEMM on pf_gemm - per
// linear, slabs of its columns as bf16 B (df_w_slab_*), then the rows against each. fc's slab is 512 columns:
// [25600][512] fits the scratch's slab ([17408][1024] on Qwen3.8), a 1024-column one does not.
bool project_wide(prefill::Context& cx, prefill::KernelCache& kc, PrefillScratch& s, const loader::DFlashWeights& w,
                  DFlashBuffers& b, const void* taps, uint32_t pos, uint32_t n) {
  using prefill::PtrArg, prefill::arg_val;
  if (w.format == loader::DraftFormat::Int4G64) return false;
  const model::DFlashDesc& d = w.desc;
  const uint32_t H = d.hidden, KV = 2 * d.kv_n(), cap = s.chunk_cap(), scaled = w.format == loader::DraftFormat::Int8;
  require(n >= 1 && n <= cap, "a wide projection takes 1.." + std::to_string(cap) + " rows, not " + std::to_string(n));
  require(size_t(pos) + n <= b.max_len, "positions " + std::to_string(pos) + " + " + std::to_string(n) +
                                             " are past the RoPE table");
  // partials [0, cap x H) holds the chunk's last linear, which step_head folds: fc's rows go after it, k||v's after.
  float* const out = s.partials.as<float>() + size_t(cap) * H;
  float* const kv = out + size_t(cap) * H;
  l0::Mem& slab = s.slab_buffer();
  require(s.partials.size() >= (size_t(cap) * (2 * H + KV)) * 4 && s.x.size() >= size_t(cap) * H * 2 &&
              slab.size() >= size_t(w.fc->shape.K) * 512 * 2,
          "the prefill scratch is smaller than the wide projection's rows (not Qwen3.8's)");
  const std::string v = kernels::dflash::proj_wide_variant();
  l0::Kernel& un = kc(v, scaled ? "df_w_slab_i8" : "df_w_slab_bf16");
  // C [n][N] (pitch N) = A [n][K] x the N columns of `wt` (their first tile), ns columns a slab.
  const auto linear = [&](const void* wt, const uint16_t* A, uint32_t K, uint32_t N, uint32_t ns, float* C) {
    for (uint32_t n0 = 0; n0 < N; n0 += ns) {
      cx.launch(un, ns / 256, K / (scaled ? 16 : 8), 1,
                {PtrArg(wt), PtrArg(slab.ptr()), arg_val(n0), arg_val(K), arg_val(ns)});
      prefill::gemm_l0(cx, kc, prefill::GemmBatch{n, K, ns, 1, K, ns, N, 0, 0, 0}, A, slab.as<uint16_t>(), C + n0,
                       /*transB=*/false);
    }
  };
  linear(w.fc->mem.ptr(), static_cast<const uint16_t*>(taps), w.fc->shape.K, H, 512, out);
  const void* const fc_scales = scaled ? w.fc->scales->ptr() : out;   // unscaled: never read
  cx.launch(kc(v, "df_w_norm"), 1, n, 1,
            {PtrArg(out), PtrArg(fc_scales), PtrArg(w.hidden_norm->ptr()), PtrArg(s.x.ptr()), arg_val(scaled)});
  const size_t ring_layer = size_t(d.window[0]) * d.kv_n() * 2, q_n = d.q_n();
  for (uint32_t l = 0; l < d.layers; ++l) {
    // the k||v rows [q_n, qkv_n) of q||k||v: n-tile-major layouts, so they are the weight from row q_n on
    const loader::DeviceWeight& qkv = w.layers[l].qkv;
    const auto* wm = static_cast<const uint8_t*>(qkv.mem.ptr()) + q_n * H * (scaled ? 1 : 2);
    linear(wm, s.x.as<uint16_t>(), H, KV, 1024, kv);
    const void* const scales = scaled ? static_cast<const void*>(qkv.scales->as<float>() + q_n) : kv;
    cx.launch(kc(v, "df_w_ctx_store"), 2 * d.kv_heads, n, 1,
              {PtrArg(kv), PtrArg(scales), PtrArg(w.qk_norm->as<float>() + (size_t(l) * 2 + 1) * d.head_dim),
               PtrArg(b.rope.ptr()), PtrArg(static_cast<uint8_t*>(b.ring_k.ptr()) + l * ring_layer),
               PtrArg(static_cast<uint8_t*>(b.ring_v.ptr()) + l * ring_layer), arg_val(pos), arg_val(scaled)});
  }
  return true;
}

}  // namespace runtime::dflash
