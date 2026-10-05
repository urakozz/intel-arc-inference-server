#include "runtime/k2/k2_prefill.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/k2_kernels.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

// k2_prefill.cc - K2-Horizon's prefill chunk (k2_prefill.h has the walk in one table). Every
// binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/k2/{k2_pf_linear,k2_pf_moe,k2_pf_attn,k2_prep,k2_moe}.cl and the reused
// prefill sources (pf_embed.cl, pf_prep.cl, pf_gemv_bf16.cl, pf_gemm.cl, pf_moe_gemm.cl).
namespace runtime::k2 {
namespace {

namespace kk = kernels::k2;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants k2_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == kk::kPfC && kPfTm == kk::kPfTm && kPfSlab == kk::kPfSlab,
              "runtime/k2/k2_sizes.h and kernels/k2_kernels.h disagree on the prefill chunk");
static_assert(kk::kPfTm == kernels::pf_moe::kTileM, "the grouped GEMM's tile is pf_moe_gemm's TM");
static_assert(kRouteWords == kk::route::kWords, "the K2 route row's width moved");
static_assert(kk::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::k2::prefill_chunk: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, K2PrefillScratch& s,
       const loader::K2LoadedModel& m, K2Buffers& b, uint32_t pos, uint32_t C, bool eager)
      : cx_(cx), kc_(kc), s_(s), m_(m), d_(*m.desc), b_(b), pos_(pos), C_(C), eager_(eager) {}

  void run() {
    check();
    {   // pf_embed_gather(ids, embed, resid, m_count) - grid (1, C)
      l0::Kernel& k = kc_(kk::pf_embed_variant(d_.hidden, d_.vocab), "pf_embed_gather");
      launch(k, 1, C_, 1, {PtrArg(s_.ids.ptr()), PtrArg(m_.embed->ptr()), PtrArg(s_.resid.ptr()),
                            arg_val(C_)});
    }
    for (uint32_t l = 0; l < d_.layers; ++l) layer(l);
    require(n_ == prefill_chunk_launches(d_),
            "the chunk appended " + std::to_string(n_) + " launches, not the " +
                std::to_string(prefill_chunk_launches(d_)) + " runtime::k2::prefill_chunk_launches gives");
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len");
    require(&s_.desc == m_.desc && &b_.desc == m_.desc, "the scratch / buffers were sized for another K2");
    require(d_.head_dim == 128 && d_.q_n() % 256 == 0 && d_.kv_n() % 256 == 0,
            "k2_pf_attn.cl and the combines are written for head_dim 128 and whole 256-column groups");
    require(!d_.is_dense(d_.layers - 1), "the head folds S_PREV 0: K2's last layer is a MoE layer");
    const PrefillSizes ps = prefill_sizes(d_);
    require(s_.partials.size() == ps.partials && s_.xg.size() == ps.xg && s_.w.size() == ps.w &&
                s_.routes.size() == ps.routes && s_.attn_q.size() == ps.attn_q,
            "the prefill scratch is not runtime::k2::prefill_sizes'");
  }

  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<prefill::KernelArg> args) {
    cx_.launch(k, gx, gy, gz, args);
    ++n_;
  }

  // The grouped norm: pf_res_fold (stage A at runtime M, folding ONE slice of `partials` at
  // pitch hidden; m_count = kPfC so sumsq is [G][kPfC]) and k2_norm_finish at M = kPfC
  // (decode's stage B, the two groups of 1280 and the plain w) into xn.
  void fold_norm(uint32_t s_prev, const void* norm_w) {
    launch(kc_(kernels::pf_res_fold_variant(d_.hidden, s_prev, kk::kNormG), "pf_res_fold"),
           kk::kNormG, C_, 1,
           {PtrArg(s_.partials.ptr()), PtrArg(s_.resid.ptr()), PtrArg(s_.sumsq.ptr()), arg_val(kPfC)});
    launch(kc_(kk::pf_norm_variant(d_.hidden, kk::kNormG, kk::kNormW, d_.norm_groups), "k2_norm_finish"),
           kk::kNormW, C_, 1,
           {PtrArg(s_.sumsq.ptr()), PtrArg(s_.resid.ptr()), PtrArg(norm_w), PtrArg(s_.xn.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  void check_int4(const loader::DeviceWeight& w) {
    require(w.kind == model::WeightKind::Int4 && w.shape.layout == 0 && w.scales,
            "a K2 int4 linear is GPTQ layout 0 with its own scales");
    require(w.shape.K % 64 == 0 && w.shape.N % 16 == 0, "the linear is not whole k-groups / tiles");
    require(s_.slab.size() >= size_t(w.shape.K) * kPfSlab * 2, "the slab buffer is smaller than [K][1024]");
  }

  // One int4 linear: per slab k2_pf_dequant_slab(w, scales, slab, n0, ns) then pf_gemm
  // x [C][K] x slab [K][ns] into partials + n0 at pitch pf_ld(N) - slabs of 1024 and one
  // zero-padded tail (k2_pf_linear.cl).
  void linear(const loader::DeviceWeight& w, const uint16_t* x) {
    check_int4(w);
    const model::GemvShape& sh = w.shape;
    const uint32_t ld = pf_ld(sh.N);
    require(s_.partials.size() >= size_t(kPfC) * ld * 4, "partials is smaller than [kPfC][pf_ld(N)]");
    l0::Kernel& dq = kc_(kk::pf_slab_variant(sh.K, sh.N), "k2_pf_dequant_slab");
    for (uint32_t n0 = 0; n0 < sh.N;) {
      const uint32_t ns = pf_slab_width(sh.N, n0);
      launch(dq, ns / 16, sh.K / 64, 1,
             {PtrArg(w.mem.ptr()), PtrArg(w.scales->ptr()), PtrArg(s_.slab.ptr()), arg_val(n0),
              arg_val(ns)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, sh.K, ns, 1, sh.K, ns, ld, 0, 0, 0};
      prefill::gemm_l0(cx_, kc_, gb, x, s_.slab.as<uint16_t>(), s_.partials.as<float>() + n0, false);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
      n0 += ns;
    }
  }

  // The dense gate||up with SiLU x up fused in the GEMM's epilogue (pf_gemm_T0_SILU, the
  // same chain as decode's k2_silu_mul at S 1) into xi [C][dense_inter].
  void linear_silu(const loader::DeviceWeight& w, const uint16_t* x) {
    check_int4(w);
    const model::GemvShape& sh = w.shape;
    require(sh.N % kPfSlab == 0 && sh.N / 2 == d_.dense_inter, "gate||up is whole slabs of 2 x dense_inter");
    l0::Kernel& dq = kc_(kk::pf_slab_variant(sh.K, sh.N), "k2_pf_dequant_slab");
    for (uint32_t n0 = 0; n0 < sh.N; n0 += kPfSlab) {
      launch(dq, kPfSlab / 16, sh.K / 64, 1,
             {PtrArg(w.mem.ptr()), PtrArg(w.scales->ptr()), PtrArg(s_.slab.ptr()), arg_val(n0),
              arg_val(kPfSlab)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, sh.K, kPfSlab, 1, sh.K, kPfSlab, 0, 0, 0, 0};
      prefill::gemm_l0_silu(cx_, kc_, gb, x, s_.slab.as<uint16_t>(), s_.xi.as<uint16_t>() + n0 / 2,
                            d_.dense_inter);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
    }
  }

  // k2_attn_prep (decode's chain at M = kPfC, S 1) then the flash attention with the gate.
  void attention(uint32_t l, bool dense, uint32_t ld) {
    void* kvk = b_.kv_k_layer(l);
    void* kvv = b_.kv_v_layer(l);
    // k2_attn_prep(ctrl, partials, rope, attn_q, attn_gate, kv_k, kv_v) - grid (q + kv heads, C)
    launch(kc_(kk::pf_attn_prep_variant(ld, d_.q_heads, d_.kv_heads, dense), "k2_attn_prep"),
           d_.q_heads + d_.kv_heads, C_, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.partials.ptr()), PtrArg(m_.rope->ptr()),
            PtrArg(s_.attn_q.ptr()), PtrArg(s_.attn_gate.ptr()), PtrArg(kvk), PtrArg(kvv)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    // k2_pf_flash_attn(q, kv_k, kv_v, gate, out, pos, C) - grid (ceil(C / 8), kv heads, 1)
    launch(kc_(kk::pf_flash_variant(d_.q_heads, d_.kv_heads, eager_, true), "k2_pf_flash_attn"),
           (C_ + kk::kPfFlashRpw - 1) / kk::kPfFlashRpw, d_.kv_heads, 1,
           {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.attn_gate.ptr()),
            PtrArg(s_.attn_out.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
  }

  // One router's sort and gather: k2_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax)
  // and k2_pf_gather(xn, hdr, row_tok, xg) over the padded rows.
  void sort_gather(const std::string& v, const void* rt, uint32_t tmax) {
    launch(kc_(v, "k2_pf_sort"), 1, 1, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.tiles.ptr()), PtrArg(s_.row_tok.ptr()),
            PtrArg(s_.pair_row.ptr()), arg_val(C_), arg_val(tmax)});
    prefill::profile_wait(cx_, Phase::kMoeSort);
    launch(kc_(v, "k2_pf_gather"), tmax * kPfTm, 1, 1,
           {PtrArg(s_.xn.ptr()), PtrArg(s_.hdr.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.xg.ptr())});
    prefill::profile_wait(cx_, Phase::kMoeGather);
  }

  // One expert group's weight batches: `dequant` (k2_pf_dequant_*(w, hdr, out, b0), grid
  // (N / 16, K / 64, nb)) then pf_moe_gemm(tiles, A, -, B, -, out, b0, b1, 0) over the tile
  // table, grid (tmax, N / 256).
  void grouped(const std::string& v, const char* dequant, const void* blocks, uint32_t nblocks,
               uint32_t per, uint32_t K, uint32_t N, bool silu, const void* A, void* out,
               uint32_t tmax) {
    l0::Kernel& gemm = kc_(kernels::pf_moe_gemm_variant(K, N, false, silu), "pf_moe_gemm");
    for (uint32_t b0 = 0; b0 < nblocks; b0 += per) {
      const uint32_t nb = std::min(per, nblocks - b0);
      launch(kc_(v, dequant), N / 16, K / 64, nb,
             {PtrArg(blocks), PtrArg(s_.hdr.ptr()), PtrArg(s_.w.ptr()), arg_val(b0)});
      prefill::profile_wait(cx_, Phase::kMoeWeights);
      launch(gemm, tmax, N / kernels::pf_moe::kGemmWgN, 1,
             {PtrArg(s_.tiles.ptr()), PtrArg(A), PtrArg(nullptr), PtrArg(s_.w.ptr()), PtrArg(nullptr),
              PtrArg(out), arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
      prefill::profile_wait(cx_, Phase::kMoeGemm);
    }
  }

  // MoVA: route all C tokens over the fused row's v_router columns (decode's k2_route, grid
  // (1, C)), group by value expert, the grouped GEMM (2560 -> 1024 per expert), the combine
  // (SiLU, ascending id) straight into this layer's V cache at pos + t.
  void mova(uint32_t l, uint32_t ld) {
    const loader::K2Layer& L = m_.layers[l];
    void* rt = at(s_.routes, pf_route_at(l, 0));
    launch(kc_(kk::pf_route_variant(d_.value_experts, d_.value_top_k, ld, d_.v_off()), "k2_route"),
           1, C_, 1, {PtrArg(s_.partials.ptr()), PtrArg(at(*L.route, d_.route_off_mova())), PtrArg(rt)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    const uint32_t tmax = pf_mova_tiles(d_, C_);
    const std::string v = kk::pf_mova_variant(d_.value_experts, d_.value_top_k, d_.hidden, d_.kv_n());
    sort_gather(v, rt, tmax);
    grouped(v, "k2_pf_dequant_v", L.value->ptr(), d_.value_experts, pf_batch_blocks(d_, PfGroup::Value),
            d_.hidden, d_.kv_n(), false, s_.xg.ptr(), s_.h.ptr(), tmax);
    // k2_pf_mova_combine(route, pair_row, y, kv_v, pos, C) - grid (kv_n / 256, C)
    launch(kc_(v, "k2_pf_mova_combine"), d_.kv_n() / kk::kPfCombineWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.pair_row.ptr()), PtrArg(s_.h.ptr()), PtrArg(b_.kv_v_layer(l)),
            arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
  }

  // The MoE block: the router GEMV over the chunk (pf_gemv_bf16 at decode's {16, 16}), decode's
  // k2_route binary (grid (1, C)), sort, gather, gate||up (SiLU epilogue) and down grouped
  // GEMMs, the combine (ascending id, the ungated shared expert, the residual fold).
  void moe(uint32_t l) {
    const loader::K2Layer& L = m_.layers[l];
    require(L.router->shape.N == d_.router_n() && L.router->shape.K == d_.hidden &&
                L.router->kind == model::WeightKind::Bf16,
            "the MoE router is not bf16 [router_n][hidden]");
    void* rt = at(s_.routes, pf_route_at(l, 1));
    // pf_ab_proj(w, x, out, m_count) - grid (router_n / 16, ceil(C / 8))
    launch(kc_(kernels::pf_moe_router_variant(d_.hidden, d_.router_n()), "pf_ab_proj"),
           d_.router_n() / 16, (C_ + 7) / 8, 1,
           {PtrArg(L.router->mem.ptr()), PtrArg(s_.xn.ptr()), PtrArg(s_.logits.ptr()), arg_val(C_)});
    // k2_route(logits, bias, route) - decode's binary, row m of [C][router_n]
    launch(kc_(kk::pf_route_variant(d_.experts, d_.top_k, d_.router_n(), 0), "k2_route"), 1, C_, 1,
           {PtrArg(s_.logits.ptr()), PtrArg(at(*L.route, d_.route_off_moe())), PtrArg(rt)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    const uint32_t tmax = pf_moe_tiles(d_, C_);
    const std::string v = kk::pf_moe_variant(d_.experts, d_.top_k, d_.hidden, d_.moe_inter);
    sort_gather(v, rt, tmax);
    grouped(v, "k2_pf_dequant_gu", L.gate_up->ptr(), d_.moe_blocks(), pf_batch_blocks(d_, PfGroup::GateUp),
            d_.hidden, 2 * d_.moe_inter, true, s_.xg.ptr(), s_.h.ptr(), tmax);
    // down: A = h [rows][I], y = rne(down) into xg (the gathered A is consumed)
    grouped(v, "k2_pf_dequant_dn", L.down->ptr(), d_.moe_blocks(), pf_batch_blocks(d_, PfGroup::Down),
            d_.moe_inter, d_.hidden, false, s_.h.ptr(), s_.xg.ptr(), tmax);
    // k2_pf_moe_combine(route, hdr, pair_row, y, resid, C) - grid (hidden / 256, C)
    launch(kc_(v, "k2_pf_moe_combine"), d_.hidden / kk::kPfCombineWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.pair_row.ptr()), PtrArg(s_.xg.ptr()),
            PtrArg(s_.resid.ptr()), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
  }

  void layer(uint32_t l) {
    const loader::K2Layer& L = m_.layers[l];
    const bool dense = d_.is_dense(l);
    require(L.linears.size() == d_.layer_linears(l).size() && L.norms, "layer " + std::to_string(l) +
                                                                            " is not the descriptor's");
    // Nothing to fold at layer 0 (the embedding wrote resid) and after a MoE layer (its
    // combine folded itself); a dense layer's down leaves its one slice in partials.
    const uint32_t s_prev = l == 0 || !d_.is_dense(l - 1) ? 0 : 1;
    fold_norm(s_prev, at(*L.norms, d_.norms_off_input()));
    linear(L.linears[0], s_.xn.as<uint16_t>());   // q || k || gate || (v | v_router)
    const uint32_t ld = pf_ld(L.linears[0].shape.N);
    if (!dense) {
      require(L.route && L.router && L.value && L.gate_up && L.down,
              "MoVA/MoE layer " + std::to_string(l) + " lacks its router or expert blocks");
      mova(l, ld);   // v -> the V cache before the attention reads it
    }
    attention(l, dense, ld);
    linear(L.linears[1], s_.attn_out.as<uint16_t>());   // o_proj -> partials [C][hidden]
    fold_norm(1, at(*L.norms, d_.norms_off_post()));
    if (dense) {
      linear_silu(L.linears[2], s_.xn.as<uint16_t>());   // gate || up -> xi
      linear(L.linears[3], s_.xi.as<uint16_t>());        // down: its slice folds into the next layer
      return;
    }
    moe(l);
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  K2PrefillScratch& s_;
  const loader::K2LoadedModel& m_;
  const model::K2Desc& d_;
  K2Buffers& b_;
  const uint32_t pos_, C_;
  const bool eager_;
  size_t n_ = 0;
};

}  // namespace

K2PrefillScratch::K2PrefillScratch(l0::Context& ctx, const model::K2Desc& d)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d).ids),
      resid(ctx, l0::MemKind::Device, prefill_sizes(d).resid),
      xn(ctx, l0::MemKind::Device, prefill_sizes(d).xn),
      xi(ctx, l0::MemKind::Device, prefill_sizes(d).xi),
      partials(ctx, l0::MemKind::Device, prefill_sizes(d).partials),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d).slab),
      sumsq(ctx, l0::MemKind::Device, prefill_sizes(d).sumsq),
      attn_q(ctx, l0::MemKind::Device, prefill_sizes(d).attn_q),
      attn_gate(ctx, l0::MemKind::Device, prefill_sizes(d).attn_gate),
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

size_t K2PrefillScratch::bytes() const {
  return ids.size() + resid.size() + xn.size() + xi.size() + partials.size() + slab.size() +
         sumsq.size() + attn_q.size() + attn_gate.size() + attn_out.size() + logits.size() +
         routes.size() + hdr.size() + tiles.size() + row_tok.size() + pair_row.size() + xg.size() +
         h.size() + w.size();
}

bool prefill_attn_eager() { return k2_attn() == K2Attn::Eager; }   // one parser, k2_sizes.cc

void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, K2PrefillScratch& s,
                   const loader::K2LoadedModel& m, K2Buffers& b, uint32_t pos, uint32_t C,
                   bool eager) {
  Walk(cx, kc, s, m, b, pos, C, eager).run();
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, K2PrefillScratch& s,
                  const loader::K2LoadedModel& m, K2Buffers& b, uint32_t last_row) {
  const model::K2Desc& d = *m.desc;
  if (last_row >= kPfC) throw std::runtime_error("runtime::k2::prefill_head: last_row past kPfC");
  if (d.is_dense(d.layers - 1))
    throw std::runtime_error("runtime::k2::prefill_head: the final fold is S_PREV 0 (a MoE last layer)");
  uint8_t* row = at(s.resid, size_t(last_row) * d.hidden * 2);
  // decode's prep_res_fold (M 1, SP 0: nothing to fold) + k2_norm_finish (M 1) into b.x
  cx.launch(kc(kernels::prep_res_fold_variant(1, d.hidden, 0, kk::kNormG), "prep_res_fold"), kk::kNormG,
            1, 1, {PtrArg(b.partials.ptr()), PtrArg(row), PtrArg(b.norm_sumsq.ptr())});
  cx.launch(kc(kk::norm_variant(1, d.hidden, kk::kNormG, kk::kNormW, d.norm_groups), "k2_norm_finish"),
            kk::kNormW, 1, 1,
            {PtrArg(b.norm_sumsq.ptr()), PtrArg(row), PtrArg(m.final_norm->ptr()), PtrArg(b.x.ptr())});
  const loader::DeviceWeight& lm = *m.lm_head;
  if (lm.kind == model::WeightKind::Int8) {
    cx.launch(kc(kernels::gemv_i8w_variant(1, lm.shape.K, lm.shape.N), "gemv_i8w"),
              lm.shape.N / kernels::kGemvI8wCols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    cx.launch(kc(kernels::gemv_bf16_variant(1, lm.shape.K, lm.shape.N, t), "gemv_bf16"),
              lm.shape.N / t.cols, 1, 1, {PtrArg(lm.mem.ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(kk::argmax1_variant(1, d.vocab, d.vocab_used), "argmax_stage1"), kk::argmax_groups(d.vocab),
            1, 1, {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(kk::argmax2_variant(d.vocab), "argmax_stage2"), 1, 1, 1,
            {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::k2
