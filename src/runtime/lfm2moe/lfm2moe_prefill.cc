#include "runtime/lfm2moe/lfm2moe_prefill.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/lfm2moe_kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "loader/lfm2moe_layout.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

// lfm2moe_prefill.cc - LFM2.5-24B-A2B's prefill chunk on one device (lfm2moe_prefill.h has the walk in one table).
// Every binding site names the kernel it honours; argument orders are each __kernel's, in
// src/kernels/lfm2moe/{lfm_pf_conv,lfm_pf_moe,lfm_pf_attn,lfm_pf_prep,lfm_prep,lfm_moe}.cl and the reused sources
// (prefill/pf_dequant_slab.cl, pf_gemm.cl, pf_moe_gemm.cl).
namespace runtime::lfm2moe {
namespace {

namespace kl = kernels::lfm2moe;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants lfm2moe_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == kl::kPfC && kPfTm == kl::kPfTm && kPfSlab == kl::kPfSlab && kPfBatchBytes == kl::kPfBatchBytes,
              "runtime/lfm2moe/lfm2moe_sizes.h and kernels/lfm2moe_kernels.h disagree on the prefill chunk");
static_assert(kl::kPfTm == kernels::pf_moe::kTileM, "the grouped GEMM's tile is pf_moe_gemm's TM");
static_assert(kl::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(kPfHdrWords == kl::pf_hdr::words(), "the sort's header width moved");
static_assert(kRouteWords == kl::route::kWords, "the LFM2 route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(kl::kPfConvRows >= 2, "lfm_pf_conv's first row of a work-group is 0 or past the seed's two rows");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::lfm2moe::prefill_chunk: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, Lfm2MoePrefillScratch& s, const loader::LfmDevicePart& part,
       const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b, uint32_t pos, uint32_t C, bool eager, bool conv_tiles,
       PrefillResid in, l0::Mem* tap)
      : cx_(cx), kc_(kc), s_(s), part_(part), d_(d), b_(b), pos_(pos), C_(C), eager_(eager), tiles_(conv_tiles),
        r_(in), tap_(tap) {}

  PrefillResid run() {
    check();
    if (part_.device == 0) {
      // lfm_pf_embed_gather(ids, tiles, h, m_count) - grid (1, C): the chunk's rows become h0, nothing pending
      launch(kc_(kl::pf_embed_variant(), "lfm_pf_embed_gather"), 1, C_, 1,
             {PtrArg(s_.ids.ptr()), PtrArg(part_.embed_tiles->ptr()), PtrArg(s_.h0.ptr()), arg_val(C_)});
      prefill::profile_wait(cx_, Phase::kNorm);
      r_ = PrefillResid{};
    } else if (r_.pending && tap_) {
      pending_tap_ = int(part_.first) - 1;   // device 0 could not tap the dense cut: this device's first fold does
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) layer(l);
    const size_t want = prefill_device_launches(d_, b_.placement, part_.device);
    require(n_ == want, "device " + std::to_string(part_.device) + "'s chunk appended " + std::to_string(n_) +
                            " launches, not the " + std::to_string(want) +
                            " runtime::lfm2moe::prefill_device_launches gives");
    return r_;
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len");
    require(d_.hidden == kl::kHidden && d_.q_heads == kl::kQHeads && d_.kv_heads == kl::kKvHeads &&
                d_.head_dim == kl::kHd && d_.experts == kl::kExperts && d_.top_k == kl::kTopK &&
                d_.moe_inter == kl::kInter && d_.dense_inter == kl::kDenseInter && d_.vocab == kl::kVocab &&
                d_.conv_taps == 3 && loader::lfm::kConvRing == kl::kConvRing,
            "the descriptor is not LFM2.5-24B-A2B's shape (" + d_.name + "); the kernels bake it");
    require(&s_.desc == &d_ || s_.desc.layers == d_.layers, "the prefill scratch was sized for another descriptor");
    require(part_.device == b_.device && part_.first == b_.placement.first(b_.device) &&
                part_.end == b_.placement.end(b_.device),
            "the loaded part and the buffers are not the same device's");
    if (part_.device == 0) require(bool(part_.embed_tiles), "device 0 holds no embedding tiles");
    if (loader::lfm::device_has_attention(d_, b_.placement, part_.device))
      require(part_.rope && part_.rope->size() >= d_.rope_table_bytes(b_.max_len), "the RoPE table is shorter than max_len");
    const PrefillSizes ps = prefill_sizes(d_);
    require(s_.partials.size() == ps.partials && s_.slab.size() == ps.slab && s_.xg.size() == ps.xg &&
                s_.w.size() == ps.w && s_.routes.size() == ps.routes && s_.s_mlp.size() == ps.s_mlp &&
                s_.h0.size() == ps.h0 && s_.h1.size() == ps.h1,
            "the prefill scratch is not runtime::lfm2moe::prefill_sizes'");
    if (tap_) require(tap_->size() >= tap_bytes(d_), "the tap is smaller than [layers][hidden] bf16");
    require(r_.cur == 0 || r_.cur == 1, "the residual's buffer is h0 or h1");
  }

  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz, std::initializer_list<prefill::KernelArg> args) {
    cx_.launch(k, gx, gy, gz, args);
    ++n_;
  }
  const float* small(uint32_t l, size_t off) const {
    return reinterpret_cast<const float*>(at(*part_.layer(l).small, off));
  }
  void tap_row(uint32_t l) {
    if (!tap_) return;
    cx_.copy(at(*tap_, size_t(l) * d_.hidden * 2), at(s_.h(r_.cur), size_t(C_ - 1) * d_.hidden * 2), size_t(d_.hidden) * 2);
  }

  // lfm_res_norm(h_in, slices, w, h_out, x) at M kPfC - grid (kNormWgs, C): _S1 folds the pending rows of `partials`
  // (pitch hidden) from h[cur] into h[1 - cur] (then cur flips), _NF only normalises.
  void norm(const float* w) {
    const uint32_t S = r_.pending;
    const int in = r_.cur, out = S ? 1 - r_.cur : r_.cur;
    launch(kc_(kl::pf_norm_variant(S != 0), "lfm_res_norm"), kl::kNormWgs, C_, 1,
           {PtrArg(s_.h(in).ptr()), PtrArg(s_.partials.ptr()), PtrArg(w), PtrArg(s_.h(out).ptr()), PtrArg(s_.x.ptr())});
    require(!S || &s_.h(out) != &s_.h(in), "a fold site would rewrite the buffer its work-groups read");
    r_.cur = out;
    r_.pending = 0;
    if (S && pending_tap_ >= 0) {   // the folded h IS the dense layer's output
      tap_row(uint32_t(pending_tap_));
      pending_tap_ = -1;
    }
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  // One int4 linear at ITS layout (the descriptor's {K, N, layout}): per 1024-column slab
  //   pf_dequant_slab(w, scales, slab, n0)             grid (1024 / 16, K / 64)
  // then pf_gemm_T0: A [C][K] (pitch K) x slab [K][1024] into `out` + n0 (pitch N) - or, `silu_out`, the dense
  // gate||up's pf_gemm_T0_SILU into s_mlp + n0 / 2 (pitch dense_inter: no fp32 partials).
  void linear(const loader::DeviceWeight& w, const void* A, bool silu = false) {
    const model::GemvShape& sh = w.shape;
    require(w.kind == model::WeightKind::Int4, "an LFM2 linear is int4 (spec 24 §5)");
    require(sh.layout == 1 || bool(w.scales), "a layout-0 linear has no scales allocation");
    require(sh.K % 64 == 0 && sh.N % kPfSlab == 0, "the linear is not whole k-groups / 1024-column slabs");
    require(s_.slab.size() >= size_t(sh.K) * kPfSlab * 2, "the slab buffer is smaller than [K][1024]");
    if (silu)
      require(sh.N == 2 * d_.dense_inter && s_.s_mlp.size() >= size_t(kPfC) * d_.dense_inter * 2,
              "the SiLU epilogue writes s_mlp [kPfC][dense_inter]");
    else
      require(s_.partials.size() >= size_t(kPfC) * sh.N * 4, "partials is smaller than [kPfC][N]");
    l0::Kernel& sk = kc_(kl::pf_slab_variant(sh), "pf_dequant_slab");
    const void* scales = sh.layout == 0 ? w.scales->ptr() : w.mem.ptr();   // layout 1: inline, not read
    for (uint32_t n0 = 0; n0 < sh.N; n0 += kPfSlab) {
      launch(sk, kPfSlab / 16, sh.K / 64, 1, {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(s_.slab.ptr()), arg_val(n0)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      if (silu) {
        const prefill::GemmBatch gb{C_, sh.K, kPfSlab, 1, sh.K, kPfSlab, 0, 0, 0, 0};
        prefill::gemm_l0_silu(cx_, kc_, gb, static_cast<const uint16_t*>(A), s_.slab.as<uint16_t>(),
                              s_.s_mlp.as<uint16_t>() + n0 / 2, d_.dense_inter);
      } else {
        const prefill::GemmBatch gb{C_, sh.K, kPfSlab, 1, sh.K, kPfSlab, sh.N, 0, 0, 0};
        prefill::gemm_l0(cx_, kc_, gb, static_cast<const uint16_t*>(A), s_.slab.as<uint16_t>(),
                         s_.partials.as<float>() + n0, false);
      }
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
    }
  }

  // lfm_attn_prep at M kPfC / S 1 (q / k norms, RoPE, every row's K / V at pos + m), then the flash attention over
  // the growing cache (causal: row t sees keys [0, pos + t]).
  void attention(uint32_t l, const loader::LfmLayer& L, const loader::lfm::SmallBlock& sb) {
    void* kvk = b_.k_layer(l);
    void* kvv = b_.v_layer(l);
    require(sb.k_norm == sb.q_norm + size_t(d_.head_dim) * 4, "q_norm / k_norm are not adjacent");
    require(L.qkv->shape.N == d_.qkv_n(), "q||k||v's width");
    // lfm_attn_prep(ctrl, slices, qkn, rope, attn_q, kv_k, kv_v) - grid (q + 2 kv heads, C)
    launch(kc_(kl::pf_attn_prep_variant(), "lfm_attn_prep"), d_.q_heads + 2 * d_.kv_heads, C_, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.partials.ptr()), PtrArg(small(l, sb.q_norm)), PtrArg(part_.rope->ptr()),
            PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    // lfm_pf_attn(q, kv_k, kv_v, out, pos, C) - grid (ceil(C / RPW), kv heads, GQA / HPW)
    launch(kc_(kl::pf_attn_variant(eager_), "lfm_pf_attn"), (C_ + kl::kPfRpw - 1) / kl::kPfRpw, d_.kv_heads,
           d_.gqa() / kl::kPfHpw,
           {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.y.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
  }

  // The short conv over the chunk: the seed (the ring's slots of p0 - 2, p0 - 1 - a launch BEFORE the conv), then the
  // conv over in_proj's rows (y into s.y; the chunk's last two Bx into the ring).
  void conv(uint32_t l, const loader::lfm::SmallBlock& sb) {
    const std::string v = kl::pf_conv_variant(tiles_);
    void* ring = b_.ring_layer(l);
    // lfm_pf_conv_seed(ctrl, ring, seed) - grid (hidden / 256)
    launch(kc_(v, "lfm_pf_conv_seed"), d_.hidden / kl::kPfConvWg, 1, 1,
           {PtrArg(b_.control.ptr()), PtrArg(ring), PtrArg(s_.seed.ptr())});
    prefill::profile_wait(cx_, Phase::kGdnSeed);
    // lfm_pf_conv(ctrl, inproj, seed, taps, ring, y) - grid (hidden / 256, ceil(C / 16))
    launch(kc_(v, "lfm_pf_conv"), d_.hidden / kl::kPfConvWg, (C_ + kl::kPfConvRows - 1) / kl::kPfConvRows, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.partials.ptr()), PtrArg(s_.seed.ptr()), PtrArg(small(l, sb.conv_taps)),
            PtrArg(ring), PtrArg(s_.y.ptr())});
    prefill::profile_wait(cx_, Phase::kGdnConv);
  }

  // One weight batch group: `dequant` (lfm_pf_dequant_*(w, hdr, out, b0, b1), grid (N / 16, K / 64, b1 - b0)) then
  // pf_moe_gemm(tiles, A, -, B, -, out, b0, b1, 0) over the tile table, grid (tmax, N / 256).
  void grouped(const char* dequant, const void* blocks, uint32_t per, uint32_t K, uint32_t N, bool silu, const void* A,
               void* out, uint32_t tmax) {
    l0::Kernel& gemm = kc_(silu ? kl::pf_gemm_gu_variant() : kl::pf_gemm_dn_variant(), "pf_moe_gemm");
    l0::Kernel& dq = kc_(kl::pf_moe_variant(), dequant);
    require(size_t(per) * K * N * 2 <= s_.w.size(), "a weight batch overflows w");
    for (uint32_t b0 = 0; b0 < d_.experts; b0 += per) {
      const uint32_t b1 = std::min(d_.experts, b0 + per);
      launch(dq, N / 16, K / 64, b1 - b0,
             {PtrArg(blocks), PtrArg(s_.hdr.ptr()), PtrArg(s_.w.ptr()), arg_val(b0), arg_val(b1)});
      prefill::profile_wait(cx_, Phase::kMoeWeights);
      launch(gemm, tmax, N / kernels::pf_moe::kGemmWgN, 1,
             {PtrArg(s_.tiles.ptr()), PtrArg(A), PtrArg(nullptr), PtrArg(s_.w.ptr()), PtrArg(nullptr), PtrArg(out),
              arg_val(b0), arg_val(b1), arg_val(0u)});
      prefill::profile_wait(cx_, Phase::kMoeGemm);
    }
  }

  // The MoE block: the router over the chunk (lfm_router at M kPfC), the route (grid (1, C)), sort, gather, gate||up
  // (SiLU epilogue) and down grouped GEMMs, the combine folding into h[cur]. Only rows [0, C) are routed: the sort
  // reads C route rows, so no padding row ever reaches an expert.
  void moe(uint32_t l, const loader::LfmLayer& L, const loader::lfm::SmallBlock& sb) {
    require(L.router && L.router->shape.N == d_.experts && L.router->shape.K == d_.hidden &&
                L.router->kind == model::WeightKind::Bf16,
            "the router is not bf16 [experts][hidden] tiles");
    require(L.experts_gate_up && L.experts_down, "layer " + std::to_string(l) + "'s experts");
    void* rt = at(s_.routes, pf_route_at(l));
    // lfm_router(x, w, logits) - grid (E / 16, C)
    launch(kc_(kl::pf_router_variant(), "lfm_router"), d_.experts / 16, C_, 1,
           {PtrArg(s_.x.ptr()), PtrArg(L.router->mem.ptr()), PtrArg(s_.logits.ptr())});
    // lfm_route(logits, bias, route) - grid (1, C): row m of [C][64]
    launch(kc_(kl::pf_route_variant(), "lfm_route"), 1, C_, 1,
           {PtrArg(s_.logits.ptr()), PtrArg(small(l, sb.expert_bias)), PtrArg(rt)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    const uint32_t tmax = pf_tiles(d_, C_);
    const std::string v = kl::pf_moe_variant();
    // lfm_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax) - grid (1)
    launch(kc_(v, "lfm_pf_sort"), 1, 1, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.tiles.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.pair_row.ptr()),
            arg_val(C_), arg_val(tmax)});
    prefill::profile_wait(cx_, Phase::kMoeSort);
    // lfm_pf_gather(x, hdr, row_tok, xg) - grid (tmax x TM)
    launch(kc_(v, "lfm_pf_gather"), tmax * kPfTm, 1, 1,
           {PtrArg(s_.x.ptr()), PtrArg(s_.hdr.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.xg.ptr())});
    prefill::profile_wait(cx_, Phase::kMoeGather);
    grouped("lfm_pf_dequant_gu", L.experts_gate_up->ptr(), pf_batch_blocks_gu(d_), d_.hidden, 2 * d_.moe_inter, true,
            s_.xg.ptr(), s_.moe_h.ptr(), tmax);
    // down: A = moe_h [rows][I], y = rne(down) into xg (the gathered A is consumed)
    grouped("lfm_pf_dequant_dn", L.experts_down->ptr(), pf_batch_blocks_dn(d_), d_.moe_inter, d_.hidden, false,
            s_.moe_h.ptr(), s_.xg.ptr(), tmax);
    // lfm_pf_combine(route, pair_row, y, h, C) - grid (hidden / 256, C): folds into h[cur] in place
    launch(kc_(v, "lfm_pf_combine"), d_.hidden / kl::kPfCombineWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.pair_row.ptr()), PtrArg(s_.xg.ptr()), PtrArg(s_.h(r_.cur).ptr()), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
  }

  void layer(uint32_t l) {
    const loader::LfmLayer& L = part_.layer(l);
    const loader::lfm::SmallBlock sb = loader::lfm::small_block(d_, l);
    require(L.small && L.small->size() == sb.bytes, "layer " + std::to_string(l) + "'s small block is not small_block's");
    const bool attn = d_.is_attention(l), dense = d_.is_dense(l);
    // --- the mixer ------------------------------------------------------------------------------------------
    norm(small(l, sb.operator_norm));
    if (attn) {
      require(L.qkv && L.o_proj && L.o_proj->shape.N == d_.hidden, "layer " + std::to_string(l) + "'s attention linears");
      linear(*L.qkv, s_.x.ptr());            // q || k || v -> partials [C][3072]
      attention(l, L, sb);                    // -> y [C][2048]
      linear(*L.o_proj, s_.y.ptr());          // -> partials [C][2048]
    } else {
      require(L.in_proj && L.out_proj && L.in_proj->shape.N == 3 * d_.hidden && L.out_proj->shape.N == d_.hidden,
              "layer " + std::to_string(l) + "'s conv linears");
      linear(*L.in_proj, s_.x.ptr());         // B | C | x -> partials [C][6144]
      conv(l, sb);                            // -> y [C][2048], the ring's last two slots
      linear(*L.out_proj, s_.y.ptr());        // -> partials [C][2048]
    }
    r_.pending = 1;
    // --- the FFN --------------------------------------------------------------------------------------------
    norm(small(l, sb.ffn_norm));              // folds out_proj's / o_proj's rows
    if (dense) {
      require(L.dense_gate_up && L.dense_down, "layer " + std::to_string(l) + "'s dense FFN");
      linear(*L.dense_gate_up, s_.x.ptr(), true);   // SiLU(gate) x up -> s_mlp [C][11776]
      linear(*L.dense_down, s_.s_mlp.ptr());        // -> partials [C][2048]: the next norm folds it
      r_.pending = 1;
      if (tap_) pending_tap_ = int(l);
      return;
    }
    moe(l, L, sb);
    tap_row(l);
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  Lfm2MoePrefillScratch& s_;
  const loader::LfmDevicePart& part_;
  const model::Lfm2MoeDesc& d_;
  Lfm2MoeBuffers& b_;
  const uint32_t pos_, C_;
  const bool eager_, tiles_;
  PrefillResid r_;
  l0::Mem* tap_;
  int pending_tap_ = -1;
  size_t n_ = 0;
};

}  // namespace

Lfm2MoePrefillScratch::Lfm2MoePrefillScratch(l0::Context& ctx, const model::Lfm2MoeDesc& d)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d).ids),
      h0(ctx, l0::MemKind::Device, prefill_sizes(d).h0),
      h1(ctx, l0::MemKind::Device, prefill_sizes(d).h1),
      x(ctx, l0::MemKind::Device, prefill_sizes(d).x),
      y(ctx, l0::MemKind::Device, prefill_sizes(d).y),
      partials(ctx, l0::MemKind::Device, prefill_sizes(d).partials),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d).slab),
      seed(ctx, l0::MemKind::Device, prefill_sizes(d).seed),
      attn_q(ctx, l0::MemKind::Device, prefill_sizes(d).attn_q),
      logits(ctx, l0::MemKind::Device, prefill_sizes(d).logits),
      routes(ctx, l0::MemKind::Device, prefill_sizes(d).routes),
      hdr(ctx, l0::MemKind::Device, prefill_sizes(d).hdr),
      tiles(ctx, l0::MemKind::Device, prefill_sizes(d).tiles),
      row_tok(ctx, l0::MemKind::Device, prefill_sizes(d).row_tok),
      pair_row(ctx, l0::MemKind::Device, prefill_sizes(d).pair_row),
      xg(ctx, l0::MemKind::Device, prefill_sizes(d).xg),
      moe_h(ctx, l0::MemKind::Device, prefill_sizes(d).moe_h),
      s_mlp(ctx, l0::MemKind::Device, prefill_sizes(d).s_mlp),
      w(ctx, l0::MemKind::Device, prefill_sizes(d).w),
      desc(d) {}

size_t Lfm2MoePrefillScratch::bytes() const {
  return ids.size() + h0.size() + h1.size() + x.size() + y.size() + partials.size() + slab.size() + seed.size() +
         attn_q.size() + logits.size() + routes.size() + hdr.size() + tiles.size() + row_tok.size() + pair_row.size() +
         xg.size() + moe_h.size() + s_mlp.size() + w.size();
}

PrefillResid prefill_resid_in(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  if (dev == 0 || p.devices < 2) return PrefillResid{};
  return PrefillResid{0, d.is_dense(p.split - 1) ? 1u : 0u};
}
PrefillResid prefill_resid_out(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  PrefillResid r = prefill_resid_in(d, p, dev);
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    if (r.pending) r.cur = 1 - r.cur;   // the operator norm folds a dense layer's down rows
    r.cur = 1 - r.cur;                  // ffn_norm folds the mixer's out_proj / o_proj rows
    r.pending = d.is_dense(l) ? 1u : 0u;
  }
  return r;
}

PrefillResid prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, Lfm2MoePrefillScratch& s,
                           const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b,
                           uint32_t pos, uint32_t C, bool eager, bool conv_tiles, PrefillResid in, l0::Mem* tap) {
  if (!(in == prefill_resid_in(d, b.placement, part.device)))
    throw std::runtime_error("runtime::lfm2moe::prefill_chunk: the residual handed in is not the placement's");
  const PrefillResid out = Walk(cx, kc, s, part, d, b, pos, C, eager, conv_tiles, in, tap).run();
  if (!(out == prefill_resid_out(d, b.placement, part.device)))
    throw std::runtime_error("runtime::lfm2moe::prefill_chunk: the walk left h elsewhere than prefill_resid_out says");
  return out;
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, Lfm2MoePrefillScratch& s,
                  const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b, uint32_t last_row,
                  PrefillResid r, l0::Mem* tap) {
  if (last_row >= kPfC) throw std::runtime_error("runtime::lfm2moe::prefill_head: last_row past kPfC");
  if (!part.final_norm || (!part.head_int8 && !part.embed_tiles))
    throw std::runtime_error("runtime::lfm2moe::prefill_head: device " + std::to_string(part.device) + " holds no head");
  const size_t H = d.hidden;
  // decode's lfm_res_norm (M 1) over the last row with embedding_norm: _NF, or _S1 folding a dense last layer's row
  // (its pending down row in partials, pitch hidden) into the decode buffers' h1
  cx.launch(kc(kl::res_norm_variant(1, r.pending, false), "lfm_res_norm"), kl::kNormWgs, 1, 1,
            {PtrArg(at(s.h(r.cur), size_t(last_row) * H * 2)), PtrArg(s.partials.as<float>() + size_t(last_row) * H),
             PtrArg(part.final_norm->ptr()), PtrArg(b.h1.ptr()), PtrArg(b.x.ptr())});
  if (r.pending && tap)   // the folded row IS the dense last layer's output
    cx.copy(at(*tap, size_t(d.layers - 1) * H * 2), b.h1.ptr(), H * 2);
  if (part.head_int8) {   // gemv_i8w(w, scales, x, out), grid (N / 64)
    const loader::DeviceWeight& lm = *part.head_int8;
    cx.launch(kc(kl::head_variant(true), "gemv_i8w"), lm.shape.N / kernels::kGemvI8wCols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {   // gemv_bf16(w, x, out) over the embedding tiles (tied)
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(d.vocab);
    cx.launch(kc(kl::head_variant(false), "gemv_bf16"), d.vocab / t.cols, 1, 1,
              {PtrArg(part.embed_tiles->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(kl::argmax1_variant(1), "argmax_stage1"), kl::argmax_groups(), 1, 1,
            {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(kl::argmax2_variant(), "argmax_stage2"), 1, 1, 1, {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::lfm2moe
