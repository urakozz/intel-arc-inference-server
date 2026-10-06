#include "runtime/kolibri/kolibri_prefill.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

// kolibri_prefill.cc - Kolibri-1's prefill chunk on one device (kolibri_prefill.h has the walk in one
// table). Every binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/kolibri/{kol_pf_moe,kol_pf_attn,kol_pf_linear,kol_prep,kol_moe}.cl and the reused sources
// (prefill/pf_embed.cl, pf_prep.cl, pf_gemv_bf16.cl, pf_gemm.cl, pf_moe_gemm.cl, k2/k2_pf_linear.cl).
namespace runtime::kolibri {
namespace {

namespace kk = kernels::kolibri;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants kolibri_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == kk::kPfC && kPfTm == kk::kPfTm && kPfSlab == kk::kPfSlab && kPfBatchBytes == kk::kPfBatchBytes,
              "runtime/kolibri/kolibri_sizes.h and kernels/kolibri_kernels.h disagree on the prefill chunk");
static_assert(kk::kPfTm == kernels::pf_moe::kTileM, "the grouped GEMM's tile is pf_moe_gemm's TM");
static_assert(kk::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(kRouteWords == kk::route::kWords, "the Kolibri route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(model::Kolibri1Desc::kRing >= kPfC + 513 - 1,
              "a chunk's own keys and the window before it must fit the ring without colliding");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::kolibri::prefill_chunk: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }
const float* fat(const l0::Mem& m, size_t floats) { return static_cast<const float*>(m.ptr()) + floats; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, KolibriPrefillScratch& s, const loader::KolDevicePart& part,
       const model::Kolibri1Desc& d, KolibriBuffers& b, uint32_t pos, uint32_t C, bool eager)
      : cx_(cx), kc_(kc), s_(s), part_(part), d_(d), b_(b), pos_(pos), C_(C), eager_(eager) {}

  void run() {
    check();
    if (part_.device == 0) {
      // pf_embed_gather(ids, embed, resid, m_count) - grid (1, C)
      launch(kc_(kk::pf_embed_variant(), "pf_embed_gather"), 1, C_, 1,
             {PtrArg(s_.ids.ptr()), PtrArg(part_.embed->ptr()), PtrArg(s_.resid.ptr()), arg_val(C_)});
      fold(nullptr, s_.resid, s_.sumsq_r, kk::pf_fold_variant());
      prefill::profile_wait(cx_, Phase::kNorm);
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) layer(l);
    const size_t want = prefill_device_launches(d_, b_.placement, part_.device);
    require(n_ == want, "device " + std::to_string(part_.device) + "'s chunk appended " + std::to_string(n_) +
                            " launches, not the " + std::to_string(want) +
                            " runtime::kolibri::prefill_device_launches gives");
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len");
    require(d_.hidden == kk::kHidden && d_.q_heads == kk::kQHeads && d_.kv_heads == kk::kKvHeads &&
                d_.head_dim == kk::kHd && d_.experts == kk::kExperts && d_.top_k == kk::kTopK &&
                d_.router_n() == kk::kRouterN && d_.moe_inter == kk::kInter && d_.shared_inter == kk::kInter &&
                d_.window == kk::kWindow && model::Kolibri1Desc::kRing == kk::kRing,
            "the descriptor is not Kolibri-1's shape (" + d_.name + "); the kernels bake it");
    require(&s_.desc == &d_ || s_.desc.layers == d_.layers, "the prefill scratch was sized for another descriptor");
    require(part_.device == b_.device && part_.first == b_.placement.first(b_.device) &&
                part_.end == b_.placement.end(b_.device),
            "the loaded part and the buffers are not the same device's");
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    const PrefillSizes ps = prefill_sizes(d_);
    require(s_.partials.size() == ps.partials && s_.xg.size() == ps.xg && s_.w.size() == ps.w &&
                s_.routes.size() == ps.routes && s_.attn_q.size() == ps.attn_q && s_.slab.size() == ps.slab,
            "the prefill scratch is not runtime::kolibri::prefill_sizes'");
    require(ps.hdr == size_t(kk::pf_hdr::words()) * 4, "the sort's header width moved");
  }

  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz, std::initializer_list<prefill::KernelArg> args) {
    cx_.launch(k, gx, gy, gz, args);
    ++n_;
  }

  // pf_res_fold(partials, row, sumsq, m_count) - pf_prep.cl stage A at runtime M, grid (G, C): SP0
  // re-reads `row` and reduces its Σ² ([G][kPfC]); _Z writes rne(partials) into `row` first.
  void fold(const void* partials, const l0::Mem& row, const l0::Mem& sumsq, const std::string& variant) {
    launch(kc_(variant, "pf_res_fold"), kk::kNormG, C_, 1,
           {PtrArg(partials ? partials : s_.partials.ptr()), PtrArg(row.ptr()), PtrArg(sumsq.ptr()), arg_val(kPfC)});
  }
  // kol_norm_finish(sumsq, resid, w, x) at M kPfC - grid (kNormW, C).
  void norm(const float* w) {
    launch(kc_(kk::pf_norm_variant(), "kol_norm_finish"), kk::kNormW, C_, 1,
           {PtrArg(s_.sumsq_r.ptr()), PtrArg(s_.resid.ptr()), PtrArg(w), PtrArg(s_.x.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }
  // kol_post_add(sumsq_a, a, w, resid, sumsq_out) at M kPfC - grid (kNormG, C).
  void post_add(const l0::Mem& a, const float* w) {
    launch(kc_(kk::pf_post_add_variant(), "kol_post_add"), kk::kNormG, C_, 1,
           {PtrArg(s_.sumsq_a.ptr()), PtrArg(a.ptr()), PtrArg(w), PtrArg(s_.resid.ptr()), PtrArg(s_.sumsq_r.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  // One attention linear: per slab the slab kernel then pf_gemm_T0 x [C][K] x slab [K][ns] into partials +
  // n0 at pitch pf_ld(N) - slabs of 1024 and one zero-padded tail.
  //   int4 (layout 0): k2_pf_dequant_slab(w, scales, slab, n0, ns)   grid (ns / 16, K / 64)
  //   bf16 (tiled):    kol_pf_bf16_slab(w, slab, n0, ns)             grid (ns / 16, K / 8)
  void linear(const loader::DeviceWeight& w, const void* x) {
    const model::GemvShape& sh = w.shape;
    const bool i4 = w.kind == model::WeightKind::Int4;
    require(i4 ? (sh.layout == 0 && bool(w.scales)) : w.kind == model::WeightKind::Bf16,
            "a Kolibri attention linear is int4 GPTQ layout 0 with its scales, or bf16 tiled");
    require((i4 ? d_.attn == model::KolAttnForm::Int4 : d_.attn == model::KolAttnForm::Bf16),
            "the linear's form is not the descriptor's attention arm");
    require(sh.K % 64 == 0 && sh.N % 16 == 0, "the linear is not whole k-groups / tiles");
    require(s_.slab.size() >= size_t(sh.K) * kPfSlab * 2, "the slab buffer is smaller than [K][1024]");
    const uint32_t ld = pf_ld(sh.N);
    require(s_.partials.size() >= size_t(kPfC) * ld * 4, "partials is smaller than [kPfC][pf_ld(N)]");
    l0::Kernel& sk = i4 ? kc_(kk::pf_int4_slab_variant(sh.K, sh.N), "k2_pf_dequant_slab")
                        : kc_(kk::pf_bf16_slab_variant(sh.K, sh.N), "kol_pf_bf16_slab");
    for (uint32_t n0 = 0; n0 < sh.N;) {
      const uint32_t ns = pf_slab_width(sh.N, n0);
      if (i4)
        launch(sk, ns / 16, sh.K / 64, 1,
               {PtrArg(w.mem.ptr()), PtrArg(w.scales->ptr()), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      else
        launch(sk, ns / 16, sh.K / 8, 1, {PtrArg(w.mem.ptr()), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, sh.K, ns, 1, sh.K, ns, ld, 0, 0, 0};
      prefill::gemm_l0(cx_, kc_, gb, static_cast<const uint16_t*>(x), s_.slab.as<uint16_t>(), s_.partials.as<float>() + n0,
                       false);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
      n0 += ns;
    }
  }

  // kol_attn_prep at M kPfC / S 1, then the flash attention over the ring or the full cache.
  void attention(uint32_t l, const loader::KolLayer& L) {
    const bool sliding = d_.is_sliding(l);
    void* kvk = b_.k_layer(l);
    void* kvv = b_.v_layer(l);
    require(!sliding || bool(part_.rope), "a sliding layer on a device without the RoPE table");
    // kol_attn_prep(ctrl, partials, qkn, rope, attn_q, kv_k, kv_v) - grid (q + 2 kv heads, C)
    launch(kc_(kk::pf_attn_prep_variant(sliding), "kol_attn_prep"), d_.q_heads + 2 * d_.kv_heads, C_, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.partials.ptr()), PtrArg(fat(*L.norms, d_.norm_off_q())),
            PtrArg(part_.rope ? part_.rope->ptr() : s_.partials.ptr()), PtrArg(s_.attn_q.ptr()), PtrArg(kvk),
            PtrArg(kvv)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    // kol_pf_flash_attn(q, kv_k, kv_v, out, pos, C) - grid (ceil(C / RPW), kv heads, GQA / HPW)
    launch(kc_(kk::pf_flash_variant(sliding, eager_), "kol_pf_flash_attn"), (C_ + kk::kPfRpw - 1) / kk::kPfRpw,
           d_.kv_heads, d_.gqa() / kk::kPfHpw,
           {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.attn_out.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
  }

  // One weight batch group: `dequant` (kol_pf_dequant_*(w, w_sh, hdr, out, b0, b1), grid (N / 16, K / 64,
  // b1 - b0)) then pf_moe_gemm(tiles, A, -, B, -, out, b0, b1, 0) over the tile table, grid (tmax, N / 256).
  void grouped(const char* dequant, const void* blocks, const void* shared, uint32_t per, uint32_t K, uint32_t N,
               bool silu, const void* A, void* out, uint32_t tmax) {
    l0::Kernel& gemm = kc_(silu ? kk::pf_gemm_gu_variant() : kk::pf_gemm_dn_variant(), "pf_moe_gemm");
    l0::Kernel& dq = kc_(kk::pf_moe_variant(), dequant);
    const uint32_t B = d_.experts + 1;   // + the shared expert, block 384
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

  // The MoE block: the router GEMV over the chunk (pf_gemv_bf16 at decode's {16, 16}), decode's kol_route
  // (grid (1, C)), sort, gather, gate||up (SiLU epilogue) and down grouped GEMMs, the combine into mo.
  void moe(uint32_t l, const loader::KolLayer& L) {
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden &&
                L.router->kind == model::WeightKind::Bf16,
            "the router is not bf16 [router_n][hidden]");
    void* rt = at(s_.routes, pf_route_at(l));
    // pf_ab_proj(w, x, out, m_count) - grid (router_n / 16, ceil(C / 8))
    launch(kc_(kk::pf_router_variant(), "pf_ab_proj"), d_.router_n() / 16, (C_ + 7) / 8, 1,
           {PtrArg(L.router->mem.ptr()), PtrArg(s_.x.ptr()), PtrArg(s_.logits.ptr()), arg_val(C_)});
    // kol_route(logits, bias, route) - decode's binary, row m of [C][router_n]
    launch(kc_(kk::pf_route_variant(), "kol_route"), 1, C_, 1,
           {PtrArg(s_.logits.ptr()), PtrArg(L.bias->ptr()), PtrArg(rt)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    const uint32_t tmax = pf_tiles(d_, C_);
    const std::string v = kk::pf_moe_variant();
    // kol_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax) - grid (1)
    launch(kc_(v, "kol_pf_sort"), 1, 1, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.tiles.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.pair_row.ptr()),
            arg_val(C_), arg_val(tmax)});
    prefill::profile_wait(cx_, Phase::kMoeSort);
    // kol_pf_gather(x, hdr, row_tok, xg) - grid (tmax x TM)
    launch(kc_(v, "kol_pf_gather"), tmax * kPfTm, 1, 1,
           {PtrArg(s_.x.ptr()), PtrArg(s_.hdr.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.xg.ptr())});
    prefill::profile_wait(cx_, Phase::kMoeGather);
    grouped("kol_pf_dequant_gu", L.gate_up->ptr(), L.shared_gate_up->ptr(), pf_batch_blocks_gu(d_), d_.hidden,
            2 * d_.moe_inter, true, s_.xg.ptr(), s_.h.ptr(), tmax);
    // down: A = h [rows][I], y = rne(down) into xg (the gathered A is consumed)
    grouped("kol_pf_dequant_dn", L.down->ptr(), L.shared_down->ptr(), pf_batch_blocks_dn(d_), d_.moe_inter, d_.hidden,
            false, s_.h.ptr(), s_.xg.ptr(), tmax);
    // kol_pf_moe_combine(route, hdr, pair_row, y, mo, C) - grid (hidden / 256, C)
    launch(kc_(v, "kol_pf_moe_combine"), d_.hidden / kk::kPfCombineWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.pair_row.ptr()), PtrArg(s_.xg.ptr()), PtrArg(s_.mo.ptr()),
            arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
  }

  void layer(uint32_t l) {
    const loader::KolLayer& L = part_.layer(l);
    require(L.qkv && L.oproj && L.norms && L.router && L.bias && L.gate_up && L.down && L.shared_gate_up && L.shared_down,
            "layer " + std::to_string(l) + " is not fully loaded");
    require(L.qkv->shape.N == d_.qkv_n() && L.oproj->shape.N == d_.hidden && L.oproj->shape.K == d_.q_n(),
            "layer " + std::to_string(l) + "'s attention rows");
    const l0::Mem& nw = *L.norms;
    // --- attention ---------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_input()));
    linear(*L.qkv, s_.x.ptr());                 // q || k || v -> partials [C][7168]
    attention(l, L);
    linear(*L.oproj, s_.attn_out.ptr());        // o_proj -> partials [C][2560]
    fold(s_.partials.ptr(), s_.a, s_.sumsq_a, kk::pf_fold_zero_variant());   // o_proj's own row a, Σa²
    post_add(s_.a, fat(nw, d_.norm_off_post_attn()));
    // --- the MoE block -------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_post_attention()));
    moe(l, L);
    fold(nullptr, s_.mo, s_.sumsq_a, kk::pf_fold_variant());                 // Σmo²
    post_add(s_.mo, fat(nw, d_.norm_off_post_ffn()));
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  KolibriPrefillScratch& s_;
  const loader::KolDevicePart& part_;
  const model::Kolibri1Desc& d_;
  KolibriBuffers& b_;
  const uint32_t pos_, C_;
  const bool eager_;
  size_t n_ = 0;
};

}  // namespace

KolibriPrefillScratch::KolibriPrefillScratch(l0::Context& ctx, const model::Kolibri1Desc& d)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d).ids),
      resid(ctx, l0::MemKind::Device, prefill_sizes(d).resid),
      x(ctx, l0::MemKind::Device, prefill_sizes(d).x),
      a(ctx, l0::MemKind::Device, prefill_sizes(d).a),
      mo(ctx, l0::MemKind::Device, prefill_sizes(d).mo),
      partials(ctx, l0::MemKind::Device, prefill_sizes(d).partials),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d).slab),
      sumsq_a(ctx, l0::MemKind::Device, prefill_sizes(d).sumsq_a),
      sumsq_r(ctx, l0::MemKind::Device, prefill_sizes(d).sumsq_r),
      attn_q(ctx, l0::MemKind::Device, prefill_sizes(d).attn_q),
      attn_out(ctx, l0::MemKind::Device, prefill_sizes(d).attn_out),
      logits(ctx, l0::MemKind::Device, prefill_sizes(d).logits),
      routes(ctx, l0::MemKind::Device, prefill_sizes(d).routes),
      hdr(ctx, l0::MemKind::Device, prefill_sizes(d).hdr),
      tiles(ctx, l0::MemKind::Device, prefill_sizes(d).tiles),
      row_tok(ctx, l0::MemKind::Device, prefill_sizes(d).row_tok),
      pair_row(ctx, l0::MemKind::Device, prefill_sizes(d).pair_row),
      xg(ctx, l0::MemKind::Device, prefill_sizes(d).xg),
      h(ctx, l0::MemKind::Device, prefill_sizes(d).h),
      w(ctx, l0::MemKind::Device, prefill_sizes(d).w),
      desc(d) {}

size_t KolibriPrefillScratch::bytes() const {
  return ids.size() + resid.size() + x.size() + a.size() + mo.size() + partials.size() + slab.size() + sumsq_a.size() +
         sumsq_r.size() + attn_q.size() + attn_out.size() + logits.size() + routes.size() + hdr.size() + tiles.size() +
         row_tok.size() + pair_row.size() + xg.size() + h.size() + w.size();
}

bool prefill_attn_eager() { return kolibri_attn() == KolAttn::Eager; }   // one parser, kolibri_sizes.cc

void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, KolibriPrefillScratch& s,
                   const loader::KolDevicePart& part, const model::Kolibri1Desc& d, KolibriBuffers& b, uint32_t pos,
                   uint32_t C, bool eager) {
  Walk(cx, kc, s, part, d, b, pos, C, eager).run();
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, KolibriPrefillScratch& s,
                  const loader::KolDevicePart& part, const model::Kolibri1Desc& d, KolibriBuffers& b,
                  uint32_t last_row) {
  if (last_row >= kPfC) throw std::runtime_error("runtime::kolibri::prefill_head: last_row past kPfC");
  if (!part.final_norm || !part.lm_head)
    throw std::runtime_error("runtime::kolibri::prefill_head: device " + std::to_string(part.device) +
                             " holds no head");
  uint8_t* row = at(s.resid, size_t(last_row) * d.hidden * 2);
  // decode's prep_res_fold (M 1, SP 0: nothing to fold, Σ row²) + kol_norm_finish (M 1) into b.x
  cx.launch(kc(kk::fold_variant(1), "prep_res_fold"), kk::kNormG, 1, 1,
            {PtrArg(b.partials.ptr()), PtrArg(row), PtrArg(b.sumsq_r.ptr())});
  cx.launch(kc(kk::norm_variant(1), "kol_norm_finish"), kk::kNormW, 1, 1,
            {PtrArg(b.sumsq_r.ptr()), PtrArg(row), PtrArg(part.final_norm->ptr()), PtrArg(b.x.ptr())});
  const loader::DeviceWeight& lm = *part.lm_head;
  if (lm.kind == model::WeightKind::Int8) {
    cx.launch(kc(kernels::gemv_i8w_variant(1, lm.shape.K, lm.shape.N), "gemv_i8w"), lm.shape.N / kernels::kGemvI8wCols,
              1, 1, {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    cx.launch(kc(kernels::gemv_bf16_variant(1, lm.shape.K, lm.shape.N, t), "gemv_bf16"), lm.shape.N / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(kk::argmax1_variant(1), "argmax_stage1"), kk::argmax_groups(), 1, 1,
            {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(kk::argmax2_variant(), "argmax_stage2"), 1, 1, 1, {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::kolibri
