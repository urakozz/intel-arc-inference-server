#include "runtime/qwen4exp/qwen4exp_prefill.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "loader/qwen4exp_layout.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"
#include "runtime/qwen4exp/qwen4exp_prefill_gdn.h"

// qwen4exp_prefill.cc - Qwen3.8-Flash-Next's prefill chunk on one device (qwen4exp_prefill.h has the walk in one
// table). Every binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/qwen4exp/{q4_pf_moe,q4_pf_attn,q4_pf_ple,q4_hc,q4_qsa,q4_ple,q4_moe}.cl and the reused sources
// (prefill/pf_embed.cl, pf_gemv_bf16.cl, pf_dequant_slab.cl, pf_attn_prep.cl, pf_attn.cl, pf_flash_attn.cl,
// pf_gemm.cl, pf_moe_gemm.cl, k2/k2_pf_linear.cl, kolibri/kol_pf_linear.cl).
namespace runtime::qwen4exp {
namespace {

namespace kq = kernels::qwen4exp;
using kq::HcSrc;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants qwen4exp_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == kq::kPfC && kPfTm == kq::kPfTm && kPfSlab == kq::kPfSlab && kPfBatchBytes == kq::kPfBatchBytes,
              "runtime/qwen4exp/qwen4exp_sizes.h and kernels/qwen4exp_kernels.h disagree on the prefill chunk");
static_assert(kq::kPfTm == kernels::pf_moe::kTileM, "the grouped GEMM's tile is pf_moe_gemm's TM");
static_assert(kq::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(kRouteWords == kq::route::kWords && kListRow == kq::kListRow && kCountWord == kq::kCountWord,
              "the route / selection rows moved");
static_assert(kPfDenseLast + 1 == kq::kTopBlocks * kq::kBlock + 3, "the dense range is the rows seeing <= 2051 positions");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::prefill_chunk: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
       const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t pos, uint32_t C, bool eager, const l0::Mem* injected,
       bool two_cards)
      : cx_(cx), kc_(kc), s_(s), m_(m), part_(part), d_(m.desc), b_(b), pos_(pos), C_(C), eager_(eager),
        injected_(injected), two_(two_cards), dense_(pf_dense_rows(pos, C)) {}

  void run() {
    check();
    if (part_.device == 0) {
      // pf_embed_gather(ids, embed, resid, m_count) - grid (1, C): the rows into x; layer 0's combine _E repeats them
      launch(kc_(kq::pf_embed_variant(), "pf_embed_gather"), 1, C_, 1,
             {PtrArg(s_.ids.ptr()), PtrArg(part_.embed->ptr()), PtrArg(s_.x.ptr()), arg_val(C_)});
      src_ = HcSrc::Embed;
    } else {
      src_ = HcSrc::None;   // device 1: H landed materialised
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) layer(l);
    if (two_ && part_.device == 0) combine(HcSrc::Y, false, nullptr);   // _Y_NN: the materialised H crosses
    const size_t want = prefill_device_launches(d_, b_.placement, part_.device, pos_, C_, injected_ != nullptr);
    require(n_ == want, "device " + std::to_string(part_.device) + "'s chunk at " + std::to_string(pos_) + " (" +
                            std::to_string(C_) + " rows) appended " + std::to_string(n_) + " launches, not the " +
                            std::to_string(want) + " runtime::qwen4exp::prefill_device_launches gives");
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len");
    require(d_.hidden == kq::kHidden && d_.hc_n() == kq::kHcN && d_.hc_down_rows() == kq::kHcDownN &&
                d_.q_heads == kq::kQHeads && d_.kv_heads == kq::kKvHeads && d_.head_dim == kq::kHd &&
                d_.qkvg_n() == kq::kQkvgN && d_.idx_n() == kq::kIdxN && d_.experts == kq::kExperts &&
                d_.top_k == kq::kTopK && d_.moe_inter == kq::kInter && d_.router_n() == kq::kRouterN &&
                d_.ple_kv_n() == kq::kPleKvN && d_.qkvz_n() == kq::kQkvzN && d_.gdn_v_heads == kq::kGdnHeads,
            "the descriptor is not Qwen3.8-Flash-Next's shape (" + d_.name + "); the kernels bake it");
    require(&s_.desc == &d_ || s_.desc.layers == d_.layers, "the prefill scratch was sized for another descriptor");
    require(s_.max_len == b_.max_len, "the prefill scratch's score rows were sized for another max_len");
    require(part_.device == b_.device && part_.first == b_.placement.first(b_.device) &&
                part_.end == b_.placement.end(b_.device),
            "the loaded part and the buffers are not the same device's");
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    if (b_.holds_ple()) require(m_.ple_device == part_.device && m_.ple.ptrs, "the PLE layer's device holds no pointer table");
    const PrefillSizes ps = prefill_sizes(d_, b_.max_len);
    require(s_.partials.size() == ps.partials && s_.xg.size() == ps.xg && s_.w.size() == ps.w &&
                s_.routes.size() == ps.routes && s_.lists.size() == ps.lists && s_.scores.size() == ps.scores &&
                s_.slab.size() == ps.slab && s_.gdn.size() == ps.gdn,
            "the prefill scratch is not runtime::qwen4exp::prefill_sizes'");
    require(ps.hdr == size_t(kq::pf_hdr::words()) * 4, "the sort's header width moved");
    if (injected_) require(injected_->size() >= pf_injected_bytes(d_), "the injected selection rows are short");
  }

  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz, std::initializer_list<prefill::KernelArg> args) {
    cx_.launch(k, gx, gy, gz, args);
    ++n_;
  }
  const void* ctrl() const { return b_.control.ptr(); }

  // One linear over the chunk: per slab the slab kernel then pf_gemm_T0 [C][K] x slab [K][ns] into out + n0 at pitch
  // ld - slabs of 1024 and one zero-padded tail (pf_slab_width).
  //   int4 (layout 0): pf_dequant_slab(w, scales, slab, n0)            whole slabs, grid (1024 / 16, K / 64)
  //                    k2_pf_dequant_slab(w, scales, slab, n0, ns)     with a tail, grid (ns / 16, K / 64)
  //   bf16 (tiled):    kol_pf_bf16_slab(w, slab, n0, ns)               grid (ns / 16, K / 8)
  void slabs(bool int4, const void* w, const void* scales, uint32_t K, uint32_t N, const void* x, void* out, uint32_t ld) {
    require(K % 64 == 0 && N % 16 == 0, "a linear is not whole k-groups / tiles");
    require(s_.slab.size() >= size_t(K) * pf_slab_width(N, 0) * 2, "the slab buffer is smaller than [K][slab]");
    require(ld >= pf_ld(N), "the output pitch does not hold the padded columns");
    const bool tail = kq::pf_int4_slab_tail(N);
    l0::Kernel& sk = int4 ? kc_(kq::pf_int4_slab_variant(K, N), tail ? "k2_pf_dequant_slab" : "pf_dequant_slab")
                          : kc_(kq::pf_bf16_slab_variant(K, N), "kol_pf_bf16_slab");
    for (uint32_t n0 = 0; n0 < N;) {
      const uint32_t ns = pf_slab_width(N, n0);
      if (int4 && tail)
        launch(sk, ns / 16, K / 64, 1, {PtrArg(w), PtrArg(scales), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      else if (int4)
        launch(sk, ns / 16, K / 64, 1, {PtrArg(w), PtrArg(scales), PtrArg(s_.slab.ptr()), arg_val(n0)});
      else
        launch(sk, ns / 16, K / 8, 1, {PtrArg(w), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, K, ns, 1, K, ns, ld, 0, 0, 0};
      prefill::gemm_l0(cx_, kc_, gb, static_cast<const uint16_t*>(x), s_.slab.as<uint16_t>(),
                       static_cast<float*>(out) + n0, false);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
      n0 += ns;
    }
  }
  void linear(const loader::DeviceWeight& w, const void* x, void* out, uint32_t ld) {
    const bool i4 = w.kind == model::WeightKind::Int4;
    require(i4 ? (w.shape.layout == 0 && bool(w.scales)) : w.kind == model::WeightKind::Bf16,
            "a Qwen3.8-Flash-Next dense linear is int4 GPTQ layout 0 with its scales, or bf16 tiled");
    require(i4 == (d_.forms.dense == model::Q4Form::Int4), "the linear's form is not the descriptor's dense arm");
    slabs(i4, w.mem.ptr(), i4 ? w.scales->ptr() : nullptr, w.shape.K, w.shape.N, x, out, ld);
  }

  // q4_hc_combine_norm(ctrl, H, src, inj, norm_w, xn) at M = kPfC, grid (4, C): folds what is pending, then (norm)
  // the grouped norm with `w`.
  void combine(HcSrc src, bool norm, const float* w) {
    const void* sp = s_.H.ptr();   // _X reads nothing
    if (src == HcSrc::Embed) sp = s_.x.ptr();
    if (src == HcSrc::Slices) sp = s_.partials.ptr();   // the mixer's linear at pitch 2560 (S 1)
    if (src == HcSrc::Y) sp = s_.y.ptr();
    launch(kc_(kq::pf_hc_combine_norm_variant(src, norm), "q4_hc_combine_norm"), d_.hc, C_, 1,
           {PtrArg(ctrl()), PtrArg(s_.H.ptr()), PtrArg(sp), PtrArg(s_.inj.ptr()),
            PtrArg(w ? static_cast<const void*>(w) : s_.partials.ptr()), PtrArg(s_.xn.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }
  // A gated residual: combine_norm -> xn; the down||inject slab + pf_gemm -> down_f32 (pitch 512); q4_hc_up_mix _D512
  // (ctrl, down_f32, up, xn, x, inj), grid (2560 / 16, C) -> x and inj.
  void hc(const l0::Mem& block) {
    const loader::Q4HcOffsets o = loader::q4_hc_offsets(d_, true);
    require(o.down_n == kq::kHcDownN, "the HC down||inject tiles are not {10240, 336}");
    combine(src_, true, reinterpret_cast<const float*>(at(block, o.norm)));
    src_ = HcSrc::None;
    slabs(false, at(block, o.down), nullptr, d_.hc_n(), o.down_n, s_.xn.ptr(), s_.down_f32.ptr(), kq::kPfHcDownLd);
    launch(kc_(kq::pf_hc_up_mix_variant(), "q4_hc_up_mix"), d_.hidden / 16, C_, 1,
           {PtrArg(ctrl()), PtrArg(s_.down_f32.ptr()), PtrArg(at(block, o.up)), PtrArg(s_.xn.ptr()), PtrArg(s_.x.ptr()),
            PtrArg(s_.inj.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  void layer(uint32_t l) {
    const loader::Q4Layer& L = part_.layer(l);
    require(L.hc_attn && L.hc_mlp && L.router && L.gate_up && L.down && L.shared,
            "layer " + std::to_string(l) + " is not fully loaded");
    if (l == d_.ple_layer) ple(L);
    hc(*L.hc_attn);
    if (d_.is_qsa(l))
      qsa(l, L);
    else
      gdn(l, L);
    hc(*L.hc_mlp);
    moe(l, L);
  }

  // The PLE layer's prologue: materialise H (unless it landed so), the gather from host USM over the chunk, the
  // key||value projections, then gate / conv / ring (q4_pf_ple.cl: a row's conv never reads a ring slot this chunk
  // overwrites). The gated / gn rows live in xg's first 2 x [kPfC][10240] (xg is free outside the MoE).
  void ple(const loader::Q4Layer& L) {
    require(bool(L.ple), "the PLE layer holds no PLE block");
    if (src_ != HcSrc::None) {
      require(src_ == HcSrc::Y, "the PLE prologue follows a MoE block");
      combine(HcSrc::Y, false, nullptr);   // _Y_NN
    }
    const bool bf16s = m_.ple.scale == loader::Q4PleScale::Bf16;
    // q4_ple_gather(ctrl, ids_ring, ptrs, consts, e, ids_out, ids) _PF, grid (16 heads, C): e into x
    launch(kc_(kq::pf_ple_gather_variant(bf16s), "q4_ple_gather"), d_.ple_heads, C_, 1,
           {PtrArg(ctrl()), PtrArg(b_.ple_ids_ring()), PtrArg(m_.ple.ptrs->ptr()), PtrArg(b_.ple_consts.ptr()),
            PtrArg(s_.x.ptr()), PtrArg(s_.ple_ids.ptr()), PtrArg(s_.ids.ptr())});
    const loader::Q4PleOffsets o = loader::q4_ple_offsets(d_);
    require(o.norm_query == o.norm_key + size_t(d_.hc_n()) * 4 && o.norm_conv == o.norm_query + size_t(d_.hc_n()) * 4 &&
                o.conv == o.norm_conv + size_t(d_.hc_n()) * 4,
            "the PLE block's norms / taps are not contiguous after norm_key (q4_pf_ple.cl indexes them so)");
    slabs(false, at(*L.ple, o.kv), nullptr, d_.ple_e(), d_.ple_kv_n(), s_.x.ptr(), s_.partials.ptr(), pf_ld(d_.ple_kv_n()));
    void* gated = s_.xg.ptr();
    void* gn = at(s_.xg, size_t(kPfC) * d_.hc_n() * 2);
    const void* pw = at(*L.ple, o.norm_key);
    const std::string v = kq::pf_ple_variant();
    // q4_pf_ple_gate(ctrl, kv, H, pw, gated, gn), grid (4 streams, C)
    launch(kc_(v, "q4_pf_ple_gate"), d_.hc, C_, 1,
           {PtrArg(ctrl()), PtrArg(s_.partials.ptr()), PtrArg(s_.H.ptr()), PtrArg(pw), PtrArg(gated), PtrArg(gn)});
    // q4_pf_ple_conv(ctrl, gated, gn, ring, pw, H), grid (10240 / 256, C)
    launch(kc_(v, "q4_pf_ple_conv"), d_.hc_n() / kq::kPfPleWg, C_, 1,
           {PtrArg(ctrl()), PtrArg(gated), PtrArg(gn), PtrArg(b_.ple_conv_ring()), PtrArg(pw), PtrArg(s_.H.ptr())});
    // q4_pf_ple_ring(ctrl, gn, ids, ring, id_ring), grid (10240 / 256, 16)
    launch(kc_(v, "q4_pf_ple_ring"), d_.hc_n() / kq::kPfPleWg, kPleRing, 1,
           {PtrArg(ctrl()), PtrArg(gn), PtrArg(s_.ids.ptr()), PtrArg(b_.ple_conv_ring()), PtrArg(b_.ple_ids_ring())});
    prefill::profile_wait(cx_, Phase::kNorm);
    src_ = HcSrc::None;
  }

  void gdn(uint32_t l, const loader::Q4Layer& L) {
    require(L.gdn_qkvz && L.gdn_ab && L.gdn_out && L.gdn_small, "GDN layer " + std::to_string(l) + " is not loaded");
    require(L.gdn_qkvz->shape.N == d_.qkvz_n(), "qkv||z is not [16384]");
    require(L.gdn_ab->kind == model::WeightKind::Bf16 && L.gdn_ab->shape.N == model::Qwen4ExpDesc::kAbPaddedN &&
                L.gdn_ab->shape.K == d_.hidden,
            "a||b is not bf16 {2560, 128} (pf_ab_proj's tiles)");
    linear(*L.gdn_qkvz, s_.x.ptr(), s_.partials.ptr(), pf_ld(d_.qkvz_n()));
    // pf_ab_proj(w, x, out, m_count) - grid (128 / 16, ceil(C / 8))
    launch(kc_(kq::pf_ab_variant(), "pf_ab_proj"), model::Qwen4ExpDesc::kAbPaddedN / 16, (C_ + 7) / 8, 1,
           {PtrArg(L.gdn_ab->mem.ptr()), PtrArg(s_.x.ptr()), PtrArg(s_.ab.ptr()), arg_val(C_)});
    const GdnScratchSizes g = gdn_scratch_sizes(kPfC);
    GdnChunkBuffers gb;
    gb.qkvz = s_.partials.as<float>();
    gb.ab = s_.ab.as<float>();
    gb.state = static_cast<float*>(b_.gdn_state_layer(l));
    gb.ring = static_cast<uint16_t*>(b_.conv_ring_layer(l));
    gb.small = L.gdn_small->ptr();
    gb.gated_norm_off = loader::q4_gdn_small(d_).gdn_off_gated_norm;
    gb.y = s_.attn_out.as<uint16_t>();
    size_t off = 0;
    const auto carve = [&](size_t bytes) {
      void* p = at(s_.gdn, off);
      off += bytes;
      return p;
    };
    gb.xb = carve(g.xb);
    gb.seed = carve(g.seed);
    gb.g = carve(g.g);
    gb.beta = carve(g.beta);
    gb.A = carve(g.A);
    gb.A2 = carve(g.A2);
    gb.w = carve(g.w);
    gb.u = carve(g.u);
    gb.o = carve(g.o);
    require(off == s_.gdn.size(), "the GDN scratch is not gdn_scratch_sizes'");
    gdn_chunk_q4(cx_, kc_, gb, pos_, C_);
    n_ += kGdnChunkLaunches;
    linear(*L.gdn_out, s_.attn_out.ptr(), s_.partials.ptr(), pf_ld(d_.hidden));
    src_ = HcSrc::Slices;
  }

  void qsa(uint32_t l, const loader::Q4Layer& L) {
    require(L.qsa_qkvg && L.qsa_idx && L.qsa_o && L.qsa_small, "QSA layer " + std::to_string(l) + " is not loaded");
    require(L.qsa_qkvg->shape.N == d_.qkvg_n() && L.qsa_idx->shape.N == d_.idx_n() &&
                L.qsa_idx->kind == model::WeightKind::Bf16,
            "QSA layer " + std::to_string(l) + "'s projections");
    require(bool(part_.rope), "a QSA layer on a device without the RoPE table");
    const uint32_t qi = d_.qsa_before(l);
    linear(*L.qsa_qkvg, s_.x.ptr(), s_.partials.ptr(), pf_ld(d_.qkvg_n()));
    slabs(false, L.qsa_idx->mem.ptr(), nullptr, d_.hidden, d_.idx_n(), s_.x.ptr(), s_.idx_f32.ptr(), kq::kPfIdxLd);
    void* kv_k = b_.k_layer(l);
    void* kv_v = b_.v_layer(l);
    // pf_attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v) q16 - grid (24 + 2, C); the gate
    // is read by pf_attn_gate straight from the partials (no attn_gate write)
    launch(kc_(kq::pf_attn_prep_q4_variant(), "pf_attn_prep"), d_.q_heads + d_.kv_heads, C_, 1,
           {PtrArg(ctrl()), PtrArg(s_.partials.ptr()), PtrArg(L.qsa_small->ptr()), PtrArg(part_.rope->ptr()),
            PtrArg(s_.q16.ptr()), PtrArg(nullptr), PtrArg(kv_k), PtrArg(kv_v)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    const std::string qv = kq::pf_qsa_variant();
    // q4_qsa_prep(ctrl, idx_f32, small, rope, idx_q, tail, idx_keys) _PF - grid (5, C)
    launch(kc_(qv, "q4_qsa_prep"), d_.idx_heads + 1, C_, 1,
           {PtrArg(ctrl()), PtrArg(s_.idx_f32.ptr()), PtrArg(L.qsa_small->ptr()), PtrArg(part_.rope->ptr()),
            PtrArg(s_.idx_q.ptr()), PtrArg(b_.tail_layer(l)), PtrArg(b_.idx_keys_layer(l))});
    // q4_qsa_ring(ctrl, idx_f32, tail) - grid (1, 8): the last min(8, C) raw keys, after every read of the ring
    launch(kc_(qv, "q4_qsa_ring"), 1, kIdxTail, 1, {PtrArg(ctrl()), PtrArg(s_.idx_f32.ptr()), PtrArg(b_.tail_layer(l))});
    const uint32_t sparse = C_ - dense_;
    const uint32_t stride = b_.max_len / d_.idx_compress;
    void* lists = injected_ ? at(*injected_, pf_list_at(qi)) : at(s_.lists, pf_list_at(qi));
    if (sparse > 0 && !injected_) {
      // q4_qsa_score(ctrl, idx_q, idx_keys, scores, stride) - grid (ceil(stride / 256), C); work-groups past a row's
      // complete blocks exit on their first instruction
      launch(kc_(qv, "q4_qsa_score"), (stride + kq::kScoreWg - 1) / kq::kScoreWg, C_, 1,
             {PtrArg(ctrl()), PtrArg(s_.idx_q.ptr()), PtrArg(b_.idx_keys_layer(l)), PtrArg(s_.scores.ptr()),
              arg_val(stride)});
      // q4_qsa_select(ctrl, scores, stride, list, diag) - grid (1, C): decode's rule per row
      launch(kc_(qv, "q4_qsa_select"), 1, C_, 1,
             {PtrArg(ctrl()), PtrArg(s_.scores.ptr()), arg_val(stride), PtrArg(lists), PtrArg(at(s_.diag, pf_diag_at(qi)))});
    }
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    if (dense_ > 0) {
      // pf_flash_attn(Q, Kc, Vc, O, pos, C, rows) _Q24KV2 - grid (ceil(dense / 8), 2 kv heads, GQA 12 / HPW 6): the
      // dense rows [0, dense) - causal over depth pos + dense
      launch(kc_(kq::pf_flash_q4_variant(), "pf_flash_attn"), (dense_ + kq::kPfFlashRpw - 1) / kq::kPfFlashRpw,
             d_.kv_heads, d_.q_heads / d_.kv_heads / kq::kPfFlashHpw,
             {PtrArg(s_.q16.ptr()), PtrArg(kv_k), PtrArg(kv_v), PtrArg(s_.o.ptr()), arg_val(pos_), arg_val(dense_),
              arg_val(kPfC)});
    }
    if (sparse > 0) {
      // q4_pf_sparse_attn(Q, Kc, Vc, lists, O, r0, rows) - grid (sparse rows, 2 kv heads), after the flash (whose last
      // 8-row group may write rows past `dense`)
      launch(kc_(kq::pf_sparse_attn_variant(eager_), "q4_pf_sparse_attn"), sparse, d_.kv_heads, 1,
             {PtrArg(s_.q16.ptr()), PtrArg(kv_k), PtrArg(kv_v), PtrArg(lists), PtrArg(s_.o.ptr()), arg_val(dense_),
              arg_val(kPfC)});
    }
    prefill::profile_wait(cx_, Phase::kAttnFlash);
    // pf_attn_gate(o, qkv_partials, out, stride_h) _Q24KV2 - grid (24, C): out = rne(rne(o) x sigmoid(gate))
    launch(kc_(kq::pf_gate_q4_variant(), "pf_attn_gate"), d_.q_heads, C_, 1,
           {PtrArg(s_.o.ptr()), PtrArg(s_.partials.ptr()), PtrArg(s_.attn_out.ptr()), arg_val(uint32_t(kPfC * d_.head_dim))});
    linear(*L.qsa_o, s_.attn_out.ptr(), s_.partials.ptr(), pf_ld(d_.hidden));
    src_ = HcSrc::Slices;
  }

  // One weight batch group: `dequant` (q4_pf_dequant_*(w, w_sh, hdr, out, b0, b1), grid (N / 16, K / 64, b1 - b0))
  // then pf_moe_gemm(tiles, A, -, B, -, out, b0, b1, 0) over the tile table, grid (tmax, N / 256).
  void grouped(const char* dequant, const void* blocks, const void* shared, uint32_t per, uint32_t K, uint32_t N, bool silu,
               const void* A, void* out, uint32_t tmax) {
    l0::Kernel& gemm = kc_(silu ? kq::pf_gemm_gu_variant() : kq::pf_gemm_dn_variant(), "pf_moe_gemm");
    l0::Kernel& dq = kc_(kq::pf_moe_variant(), dequant);
    const uint32_t B = d_.experts + 1;   // + the shared expert, block 512
    for (uint32_t b0 = 0; b0 < B; b0 += per) {
      const uint32_t b1 = std::min(B, b0 + per);
      launch(dq, N / 16, K / 64, b1 - b0,
             {PtrArg(blocks), PtrArg(shared), PtrArg(s_.hdr.ptr()), PtrArg(s_.w.ptr()), arg_val(b0), arg_val(b1)});
      prefill::profile_wait(cx_, Phase::kMoeWeights);
      launch(gemm, tmax, N / kernels::pf_moe::kGemmWgN, 1,
             {PtrArg(s_.tiles.ptr()), PtrArg(A), PtrArg(nullptr), PtrArg(s_.w.ptr()), PtrArg(nullptr), PtrArg(out),
              arg_val(b0), arg_val(b1), arg_val(0u)});
      prefill::profile_wait(cx_, Phase::kMoeGemm);
    }
  }

  void moe(uint32_t l, const loader::Q4Layer& L) {
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden && L.router->kind == model::WeightKind::Bf16,
            "the router is not bf16 [528][2560]");
    void* rt = at(s_.routes, pf_route_at(l));
    // pf_ab_proj(w, x, out, m_count) at {2560, 528} - grid (528 / 16, ceil(C / 8)): decode's {16, 16} tiling
    launch(kc_(kq::pf_router_variant(), "pf_ab_proj"), d_.router_n() / 16, (C_ + 7) / 8, 1,
           {PtrArg(L.router->mem.ptr()), PtrArg(s_.x.ptr()), PtrArg(s_.logits.ptr()), arg_val(C_)});
    // q4_route(logits, route) - decode's binary, row m of [C][528]
    launch(kc_(kq::route_variant(1), "q4_route"), 1, C_, 1, {PtrArg(s_.logits.ptr()), PtrArg(rt)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    const uint32_t tmax = pf_tiles(d_, C_);
    const std::string v = kq::pf_moe_variant();
    // q4_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax) - grid (1)
    launch(kc_(v, "q4_pf_sort"), 1, 1, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.tiles.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.pair_row.ptr()),
            arg_val(C_), arg_val(tmax)});
    prefill::profile_wait(cx_, Phase::kMoeSort);
    // q4_pf_gather(x, hdr, row_tok, xg) - grid (tmax x TM)
    launch(kc_(v, "q4_pf_gather"), tmax * kPfTm, 1, 1,
           {PtrArg(s_.x.ptr()), PtrArg(s_.hdr.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.xg.ptr())});
    prefill::profile_wait(cx_, Phase::kMoeGather);
    const bool shb = d_.forms.shared == model::Q4Form::Bf16;
    grouped(shb ? "q4_pf_dequant_gu_shb" : "q4_pf_dequant_gu", L.gate_up->ptr(), L.shared->ptr(), pf_batch_blocks_gu(d_),
            d_.hidden, 2 * d_.moe_inter, true, s_.xg.ptr(), s_.h.ptr(), tmax);
    // down: A = h [rows][640], y = rne(down) into xg (the gathered A is consumed)
    grouped(shb ? "q4_pf_dequant_dn_shb" : "q4_pf_dequant_dn", L.down->ptr(), at(*L.shared, L.shared_down_offset),
            pf_batch_blocks_dn(d_), d_.moe_inter, d_.hidden, false, s_.h.ptr(), s_.xg.ptr(), tmax);
    // q4_pf_moe_combine(route, hdr, pair_row, y, out, C) - grid (2560 / 256, C): the block's y (NOT folded into H)
    launch(kc_(v, "q4_pf_moe_combine"), d_.hidden / kq::kPfCombineWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.pair_row.ptr()), PtrArg(s_.xg.ptr()), PtrArg(s_.y.ptr()), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
    src_ = HcSrc::Y;
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  Qwen4ExpPrefillScratch& s_;
  const loader::Q4LoadedModel& m_;
  const loader::Q4DevicePart& part_;
  const model::Qwen4ExpDesc& d_;
  Qwen4ExpBuffers& b_;
  const uint32_t pos_, C_;
  const bool eager_;
  const l0::Mem* injected_;
  const bool two_;
  const uint32_t dense_;
  HcSrc src_ = HcSrc::None;
  size_t n_ = 0;
};

}  // namespace

Qwen4ExpPrefillScratch::Qwen4ExpPrefillScratch(l0::Context& ctx, const model::Qwen4ExpDesc& d, uint32_t max_len_)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d, max_len_).ids),
      H(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).H),
      xn(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).xn),
      x(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).x),
      down_f32(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).down_f32),
      inj(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).inj),
      partials(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).partials),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).slab),
      idx_f32(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).idx_f32),
      idx_q(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).idx_q),
      scores(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).scores),
      lists(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).lists),
      diag(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).diag),
      q16(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).q16),
      o(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).o),
      attn_out(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).attn_out),
      ab(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).ab),
      gdn(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).gdn),
      logits(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).logits),
      routes(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).routes),
      hdr(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).hdr),
      tiles(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).tiles),
      row_tok(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).row_tok),
      pair_row(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).pair_row),
      xg(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).xg),
      h(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).h),
      y(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).y),
      w(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).w),
      ple_ids(ctx, l0::MemKind::Device, prefill_sizes(d, max_len_).ple_ids),
      desc(d),
      max_len(max_len_) {}

size_t Qwen4ExpPrefillScratch::bytes() const {
  return ids.size() + H.size() + xn.size() + x.size() + down_f32.size() + inj.size() + partials.size() + slab.size() +
         idx_f32.size() + idx_q.size() + scores.size() + lists.size() + diag.size() + q16.size() + o.size() +
         attn_out.size() + ab.size() + gdn.size() + logits.size() + routes.size() + hdr.size() + tiles.size() +
         row_tok.size() + pair_row.size() + xg.size() + h.size() + y.size() + w.size() + ple_ids.size();
}

void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                   const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t pos, uint32_t C, bool eager,
                   const l0::Mem* injected, bool two_cards) {
  Walk(cx, kc, s, m, part, b, pos, C, eager, injected, two_cards).run();
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                  const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t last_row) {
  const model::Qwen4ExpDesc& d = m.desc;
  if (last_row >= kPfC) throw std::runtime_error("runtime::qwen4exp::prefill_head: last_row past kPfC");
  if (!part.final_mixer || !part.lm_head)
    throw std::runtime_error("runtime::qwen4exp::prefill_head: device " + std::to_string(part.device) + " holds no head");
  const loader::Q4HcOffsets o = loader::q4_hc_offsets(d, false);
  // decode's final mixer over the last row: combine _Y (M 1: the row's H, its pending y and inject weights) -> b.xn;
  // gemv_bf16 {10240, 320} -> b.down_f32; q4_hc_up_mix (M 1, no inject) -> b.x
  cx.launch(kc(kq::hc_combine_norm_variant(1, HcSrc::Y, 0, true), "q4_hc_combine_norm"), d.hc, 1, 1,
            {PtrArg(b.control.ptr()), PtrArg(at(s.H, size_t(last_row) * d.hc_n() * 2)),
             PtrArg(at(s.y, size_t(last_row) * d.hidden * 2)), PtrArg(at(s.inj, size_t(last_row) * d.hc * 4)),
             PtrArg(at(*part.final_mixer, o.norm)), PtrArg(b.xn.ptr())});
  {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(o.down_n);
    cx.launch(kc(kernels::gemv_bf16_variant(1, d.hc_n(), o.down_n, t), "gemv_bf16"), o.down_n / t.cols, 1, 1,
              {PtrArg(at(*part.final_mixer, o.down)), PtrArg(b.xn.ptr()), PtrArg(b.down_f32.ptr())});
  }
  cx.launch(kc(kq::hc_up_mix_variant(1, false), "q4_hc_up_mix"), d.hidden / 16, 1, 1,
            {PtrArg(b.control.ptr()), PtrArg(b.down_f32.ptr()), PtrArg(at(*part.final_mixer, o.up)), PtrArg(b.xn.ptr()),
             PtrArg(b.x.ptr()), PtrArg(b.inj.ptr())});
  const loader::DeviceWeight& lm = *part.lm_head;
  if (lm.kind == model::WeightKind::Int8) {
    cx.launch(kc(kernels::gemv_i8w_variant(1, lm.shape.K, lm.shape.N), "gemv_i8w"), lm.shape.N / kernels::kGemvI8wCols,
              1, 1, {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    cx.launch(kc(kernels::gemv_bf16_variant(1, lm.shape.K, lm.shape.N, t), "gemv_bf16"), lm.shape.N / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(kq::argmax1_variant(1), "argmax_stage1"), kq::argmax_groups(), 1, 1,
            {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(kq::argmax2_variant(), "argmax_stage2"), 1, 1, 1, {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::qwen4exp
