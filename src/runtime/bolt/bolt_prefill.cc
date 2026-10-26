#include "runtime/bolt/bolt_prefill.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "kernels/bolt_kernels.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "loader/bolt_layout.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

// bolt_prefill.cc - Nemotron 3.5 Bolt's prefill chunk on one device (bolt_prefill.h has the walk in one table). Every binding
// site names the kernel it honours; argument orders are each __kernel's, in src/kernels/bolt/{nb_pf_mamba,nb_pf_moe,
// nb_pf_moe_gemm,nb_pf_prep,nb_prep,nb_moe}.cl and the reused sources (prefill/pf_dequant_slab.cl, k2/k2_pf_linear.cl,
// prefill/pf_gemm.cl, bolt/mg_pf_attn.cl).
namespace runtime::bolt {
namespace {

namespace kb = kernels::bolt;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants bolt_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == kb::kPfC && kPfSub == kb::kPfSub && kPfTm == kb::kPfTm && kPfSlab == kb::kPfSlab &&
                  kPfInLd == kb::kPfInLd && kPfBatchBytes == kb::kPfBatchBytes && kPfHdrWords == kb::pf_hdr::words() &&
                  kPfMaxSub == kb::kPfMaxSub && kPfTopK == kb::kTopK,
              "runtime/bolt/bolt_sizes.h and kernels/bolt_kernels.h disagree on the prefill chunk");
static_assert(kb::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(kRouteWords == kb::route::kWords, "the Bolt route row's width moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::bolt::prefill_chunk: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, BoltPrefillScratch& s, const loader::BoltDevicePart& part,
       const model::BoltDesc& d, BoltBuffers& b, uint32_t pos, uint32_t C, bool eager, PrefillResid in, l0::Mem* tap)
      : cx_(cx), kc_(kc), s_(s), part_(part), d_(d), b_(b), pos_(pos), C_(C), eager_(eager), r_(in), tap_(tap) {}

  PrefillResid run() {
    check();
    if (part_.device == 0) {
      // nb_pf_embed_gather(ids, tiles, h, m_count) - grid (1, C): the chunk's rows become h0, nothing pending
      launch(kc_(kb::pf_embed_variant(), "nb_pf_embed_gather"), 1, C_, 1,
             {PtrArg(s_.ids.ptr()), PtrArg(part_.embed_tiles->ptr()), PtrArg(s_.h0.ptr()), arg_val(C_)});
      prefill::profile_wait(cx_, Phase::kNorm);
      r_ = PrefillResid{};
    } else if (r_.pending && tap_) {
      pending_tap_ = int(part_.first) - 1;   // device 0 could not tap its last block: this device's first fold does
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) block(l);
    const size_t want = prefill_device_launches(d_, b_.placement, part_.device);
    require(n_ == want, "device " + std::to_string(part_.device) + "'s chunk appended " + std::to_string(n_) +
                            " launches, not the " + std::to_string(want) + " runtime::bolt::prefill_device_launches gives");
    return r_;
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len");
    require(d_.hidden == kb::kHidden && d_.m_heads == kb::kMHeads && d_.m_head_dim == kb::kMHeadDim &&
                d_.m_groups == kb::kMGroups && d_.m_state == kb::kMState && d_.m_conv == kb::kMConv &&
                d_.m_inproj == kb::kInProjN && d_.m_conv_dim == kb::kXbc && d_.m_inner == kb::kInner &&
                d_.q_heads == kb::kQHeads && d_.kv_heads == kb::kKvHeads && d_.head_dim == kb::kHd &&
                d_.latent == kb::kLatent && d_.shared_inter == kb::kShared && d_.top_k == kb::kTopK &&
                d_.vocab == kb::kVocab && d_.max_experts() <= kb::kMaxExperts && loader::bolt::kConvRing == kb::kConvRing,
            "the descriptor is not Nemotron 3.5 Bolt's shape (" + d_.name + "); the kernels bake it");
    require(&s_.desc == &d_ || s_.desc.layers == d_.layers, "the prefill scratch was sized for another descriptor");
    require(part_.device == b_.device && part_.first == b_.placement.first(b_.device) &&
                part_.end == b_.placement.end(b_.device),
            "the loaded part and the buffers are not the same device's");
    if (part_.device == 0) require(bool(part_.embed_tiles), "device 0 holds no embedding tiles");
    const PrefillSizes ps = prefill_sizes(d_);
    require(s_.pa.size() == ps.pa && s_.pb.size() == ps.pb && s_.slab.size() == ps.slab && s_.xg.size() == ps.xg &&
                s_.moe_h.size() == ps.moe_h && s_.w.size() == ps.w && s_.routes.size() == ps.routes &&
                s_.cb.size() == ps.cb && s_.h0.size() == ps.h0 && s_.h1.size() == ps.h1,
            "the prefill scratch is not runtime::bolt::prefill_sizes'");
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

  // nb_res_norm(h_in, slices, slices2, w, h_out, x) at M 2048 - grid (kNormWgs, C): _S1 folds pa's rows (pitch hidden),
  // _S1_T1 pa's and pb's (fc2 then shared_down: rne(h + rne(rne(fc2) + rne(sh)))), h[cur] -> h[1 - cur] (then cur flips);
  // _NF only normalises. The folded h is the previous block's output: its tap row.
  void norm(const float* w) {
    const uint32_t P = r_.pending;
    const int in = r_.cur, out = P ? 1 - r_.cur : r_.cur;
    launch(kc_(kb::pf_norm_variant(P ? 1u : 0u, P == 2 ? 1u : 0u), "nb_res_norm"), kb::kNormWgs, C_, 1,
           {PtrArg(s_.h(in).ptr()), PtrArg(s_.pa.ptr()), PtrArg(s_.pb.ptr()), PtrArg(w), PtrArg(s_.h(out).ptr()),
            PtrArg(s_.x.ptr())});
    require(!P || &s_.h(out) != &s_.h(in), "a fold site would rewrite the buffer its work-groups read");
    r_.cur = out;
    r_.pending = 0;
    if (P && pending_tap_ >= 0) {   // the folded h IS that block's output
      tap_row(uint32_t(pending_tap_));
      pending_tap_ = -1;
    }
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  // One int4 linear at ITS layout (the descriptor's {K, N, layout}) into `out` at pitch pad256(N): per 1024-column slab
  //   pf_dequant_slab(w, scales, slab, n0)                 grid (1024 / 16, K / 64)      whole slabs, either layout
  //   k2_pf_dequant_slab(w, scales, slab, n0, ns)          grid (ns / 16, K / 64)        layout 0 with a tail: the last
  //                                                                                       slab pad256(N - n0) wide, zeros past N
  // then pf_gemm_T0: A [C][K] (pitch K) x slab [K][ns] into out + n0.
  void linear(const loader::DeviceWeight& w, const void* A, const l0::Mem& out) {
    const model::GemvShape& sh = w.shape;
    require(w.kind == model::WeightKind::Int4, "a Bolt linear is int4 (spec 25 §5)");
    require(sh.layout == 1 || bool(w.scales), "a layout-0 linear has no scales allocation");
    const std::string v = kb::pf_slab_variant(sh);
    require(!v.empty(), "no prefill slab covers a layout-1 linear of N " + std::to_string(sh.N) +
                            " (not whole 1024-column slabs): a P0 re-pick to that cell needs its slab kernel first");
    require(sh.K % 64 == 0 && sh.N % 16 == 0, "the linear is not whole k-groups / tiles");
    require(s_.slab.size() >= size_t(sh.K) * kPfSlab * 2, "the slab buffer is smaller than [K][1024]");
    const uint32_t ld = pf_ld(sh.N);
    require(out.size() >= size_t(kPfC) * ld * 4, "the rows' buffer is smaller than [kPfC][pad256(N)]");
    const bool tail = kb::pf_slab_tail(sh);
    l0::Kernel& sk = kc_(v, kb::pf_slab_entry(sh));
    const void* scales = sh.layout == 0 ? w.scales->ptr() : w.mem.ptr();   // layout 1: inline, not read
    for (uint32_t n0 = 0; n0 < sh.N;) {
      const uint32_t ns = sh.N - n0 >= kPfSlab ? kPfSlab : prefill::pad256(sh.N - n0);
      if (tail)
        launch(sk, ns / 16, sh.K / 64, 1,
               {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      else
        launch(sk, kPfSlab / 16, sh.K / 64, 1, {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(s_.slab.ptr()), arg_val(n0)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, sh.K, ns, 1, sh.K, ns, ld, 0, 0, 0};
      prefill::gemm_l0(cx_, kc_, gb, static_cast<const uint16_t*>(A), s_.slab.as<uint16_t>(), out.as<float>() + n0, false);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
      n0 += ns;
    }
  }

  void block(uint32_t l) {
    const loader::BoltLayer& L = part_.layer(l);
    const loader::bolt::SmallBlock sb = loader::bolt::small_block(d_, l);
    require(L.small && L.small->size() == sb.bytes, "block " + std::to_string(l) + "'s small block is not small_block's");
    norm(small(l, sb.norm));
    if (d_.is_mamba(l))
      mamba(l, L, sb);
    else if (d_.is_attention(l))
      attention(l, L);
    else
      moe(l, L, sb);
    pending_tap_ = int(l);   // its output is folded by the next norm site (or the head)
  }

  // The Mamba-2 mixer over the chunk (nb_pf_mamba.cl): the seed BEFORE the conv; the state in the decode buffers.
  void mamba(uint32_t l, const loader::BoltLayer& L, const loader::bolt::SmallBlock& sb) {
    require(L.in_proj && L.out_proj && L.in_proj->shape.N == d_.m_inproj && L.out_proj->shape.K == d_.m_inner &&
                pf_ld(L.in_proj->shape.N) == kPfInLd,
            "block " + std::to_string(l) + "'s Mamba linears");
    linear(*L.in_proj, s_.x.ptr(), s_.pa);   // z | xBC | dt -> pa [C][18688]
    const std::string v = kb::pf_mamba_variant(b_.state == BoltState::Bf16);
    const uint32_t ns = kb::pf_nsub(pos_, C_);
    void* ring = b_.ring_layer(l);
    // nb_pf_seed(ring, seed, pos) - grid (10240 / 256)
    launch(kc_(v, "nb_pf_seed"), d_.m_conv_dim / kb::kPfWg, 1, 1, {PtrArg(ring), PtrArg(s_.seed.ptr()), arg_val(pos_)});
    prefill::profile_wait(cx_, Phase::kGdnSeed);
    // nb_pf_conv(P, seed, cw, cb, ring, xc, pos, C) - grid (10240 / 256, ceil(C / CONV_ROWS))
    launch(kc_(v, "nb_pf_conv"), d_.m_conv_dim / kb::kPfWg, (C_ + kb::kPfConvRows - 1) / kb::kPfConvRows, 1,
           {PtrArg(s_.pa.ptr()), PtrArg(s_.seed.ptr()), PtrArg(small(l, sb.conv_w)), PtrArg(small(l, sb.conv_b)),
            PtrArg(ring), PtrArg(s_.xc.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kGdnConv);
    // nb_pf_dt(P, A, dt_bias, dtv, cum, pos, C) - grid (1, nsub)
    launch(kc_(v, "nb_pf_dt"), 1, ns, 1,
           {PtrArg(s_.pa.ptr()), PtrArg(small(l, sb.A)), PtrArg(small(l, sb.dt_bias)), PtrArg(s_.dt.ptr()),
            PtrArg(s_.cum.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kGdnGate);
    // nb_pf_cb(xc, cbo, pos, C) - DPAS, grid (nsub, 8)
    launch(kc_(kb::pf_cb_dpas_variant(), "nb_pf_cb"), ns, d_.m_groups, 1,
           {PtrArg(s_.xc.ptr()), PtrArg(s_.cb.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kGdnA);
    // nb_pf_ssd(xc, dtv, cum, cbo, D, state, y, pos, C) - grid (128 x 64 / SSD_W)
    launch(kc_(v, "nb_pf_ssd"), d_.m_heads * (d_.m_head_dim / kb::kPfSsdW), 1, 1,
           {PtrArg(s_.xc.ptr()), PtrArg(s_.dt.ptr()), PtrArg(s_.cum.ptr()), PtrArg(s_.cb.ptr()), PtrArg(small(l, sb.D)),
            PtrArg(b_.ssm_layer(l)), PtrArg(s_.y.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kGdnScan);
    // nb_pf_gnorm(P, y, w, out) - grid (8, C)
    launch(kc_(v, "nb_pf_gnorm"), d_.m_inner / kb::kGnormGroup, C_, 1,
           {PtrArg(s_.pa.ptr()), PtrArg(s_.y.ptr()), PtrArg(small(l, sb.gnorm)), PtrArg(s_.g.ptr())});
    prefill::profile_wait(cx_, Phase::kGdnHead);
    linear(*L.out_proj, s_.g.ptr(), s_.pa);   // -> pa [C][4096]: the next norm folds it
    r_.pending = 1;
  }

  // NoPE attention (Decision 15): nb_attn_prep at M 2048 (K / V rows pos ..), spec 23d's mg_pf_flash_attn over the cache.
  void attention(uint32_t l, const loader::BoltLayer& L) {
    require(L.qkv && L.o_proj && L.qkv->shape.N == d_.qkv_n() && L.o_proj->shape.N == d_.hidden,
            "block " + std::to_string(l) + "'s attention linears");
    void* kvk = b_.k_layer(l);
    void* kvv = b_.v_layer(l);
    linear(*L.qkv, s_.x.ptr(), s_.pa);   // q || k || v -> pa [C][4608]
    // nb_attn_prep(ctrl, slices, attn_q, kv_k, kv_v) - grid (ceil(4608 / 256), C)
    launch(kc_(kb::pf_attn_prep_variant(), "nb_attn_prep"), (d_.qkv_n() + kb::kPrepWg - 1) / kb::kPrepWg, C_, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.pa.ptr()), PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    // mg_pf_flash_attn(q, kv_k, kv_v, out, pos, C): flash grid (ceil(C / RPW), kv heads, GQA / HPW); eager (q heads, C)
    l0::Kernel& fk = kc_(kb::pf_flash_variant(eager_), "mg_pf_flash_attn");
    if (eager_)
      launch(fk, d_.q_heads, C_, 1,
             {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.y.ptr()), arg_val(pos_), arg_val(C_)});
    else
      launch(fk, (C_ + kb::kPfFlashRpw - 1) / kb::kPfFlashRpw, d_.kv_heads, d_.q_heads / d_.kv_heads / kb::kPfFlashHpw,
             {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.y.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
    linear(*L.o_proj, s_.y.ptr(), s_.pa);   // -> pa [C][4096]
    r_.pending = 1;
  }

  // One weight batch loop: dequant (experts [b0, b1) into w) then the grouped GEMM over the tile table.
  void grouped(const std::string& moe_v, const char* dequant, const std::string& gemm_v, uint32_t E, uint32_t per,
               uint32_t gx_dq, uint32_t gy_dq, uint32_t n_cols, const void* blocks, const void* A, void* out, uint32_t T) {
    l0::Kernel& dq = kc_(moe_v, dequant);
    l0::Kernel& gemm = kc_(gemm_v, "nb_pf_moe_gemm");
    for (uint32_t b0 = 0; b0 < E; b0 += per) {
      const uint32_t b1 = std::min(E, b0 + per);
      // nb_pf_dequant_*(w, hdr, out, b0, b1, n_exp) - grid (N / 16, K / 64, b1 - b0)
      launch(dq, gx_dq, gy_dq, b1 - b0,
             {PtrArg(blocks), PtrArg(s_.hdr.ptr()), PtrArg(s_.w.ptr()), arg_val(b0), arg_val(b1), arg_val(E)});
      prefill::profile_wait(cx_, Phase::kMoeWeights);
      // nb_pf_moe_gemm(tiles, A, B, out, b0, b1) - grid (tmax, N / 256)
      launch(gemm, T, n_cols / 256, 1,
             {PtrArg(s_.tiles.ptr()), PtrArg(A), PtrArg(s_.w.ptr()), PtrArg(out), arg_val(b0), arg_val(b1)});
      prefill::profile_wait(cx_, Phase::kMoeGemm);
    }
  }

  // The latent MoE over the chunk's C REAL rows (no padding row routed: the sort reads route rows [0, C) only), then the
  // two dense linears whose rows the next norm folds (fc2 into pa, shared_down into pb).
  void moe(uint32_t l, const loader::BoltLayer& L, const loader::bolt::SmallBlock& sb) {
    const uint32_t E = d_.experts.at(l), I = d_.inter.at(l);
    require(L.router && L.experts && L.fc1_shared_up && L.fc2 && L.shared_down && L.router->shape.N == E &&
                L.router->shape.K == d_.hidden && L.router->kind == model::WeightKind::Bf16 &&
                L.fc1_shared_up->shape.N == d_.latent + d_.shared_inter && L.fc2->shape.N == d_.hidden &&
                L.shared_down->shape.N == d_.hidden &&
                L.experts->size() == loader::bolt::layer_expert_bytes(d_.latent, E, I),
            "block " + std::to_string(l) + "'s MoE block");
    require(E % 16 == 0 && E <= kb::kMaxExperts && E > kb::kTopK && I % 256 == 0,
            "block " + std::to_string(l) + "'s E " + std::to_string(E) + " / I " + std::to_string(I));
    void* rt = at(s_.routes, pf_route_at(l));
    // nb_router(x, w, logits, n_exp) at M 2048 - grid (E / 16, C)
    launch(kc_(kb::pf_router_variant(), "nb_router"), E / 16, C_, 1,
           {PtrArg(s_.x.ptr()), PtrArg(L.router->mem.ptr()), PtrArg(s_.logits.ptr()), arg_val(E)});
    // nb_route(logits, bias, route, n_exp) at M 2048 - grid (1, C): row m of [C][48]
    launch(kc_(kb::pf_route_variant(), "nb_route"), 1, C_, 1,
           {PtrArg(s_.logits.ptr()), PtrArg(small(l, sb.e_bias)), PtrArg(rt), arg_val(E)});
    prefill::profile_wait(cx_, Phase::kMoeRoute);
    linear(*L.fc1_shared_up, s_.x.ptr(), s_.pa);   // fc1_latent || shared_up -> pa [C][6400]
    // nb_sh_act(slices, lat, sh) at M 2048, S 1 - grid (ceil(6400 / 256), C)
    launch(kc_(kb::pf_sh_act_variant(), "nb_sh_act"), (kb::kFc1N + kb::kActWg - 1) / kb::kActWg, C_, 1,
           {PtrArg(s_.pa.ptr()), PtrArg(s_.lat.ptr()), PtrArg(s_.sh.ptr())});
    prefill::profile_wait(cx_, Phase::kSilu);
    const uint32_t T = pf_tiles(C_, E);
    const std::string v = kb::pf_moe_variant(I);
    // nb_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax, n_exp) - grid (1)
    launch(kc_(v, "nb_pf_sort"), 1, 1, 1,
           {PtrArg(rt), PtrArg(s_.hdr.ptr()), PtrArg(s_.tiles.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.pair_row.ptr()),
            arg_val(C_), arg_val(T), arg_val(E)});
    prefill::profile_wait(cx_, Phase::kMoeSort);
    // nb_pf_gather(lat, hdr, row_tok, xg) - grid (tmax x 32)
    launch(kc_(v, "nb_pf_gather"), T * kPfTm, 1, 1,
           {PtrArg(s_.lat.ptr()), PtrArg(s_.hdr.ptr()), PtrArg(s_.row_tok.ptr()), PtrArg(s_.xg.ptr())});
    prefill::profile_wait(cx_, Phase::kMoeGather);
    const uint32_t per = pf_batch_experts(I);
    require(size_t(per) * d_.latent * I * 2 <= s_.w.size(), "a weight batch overflows w");
    // up: A = xg [rows][1024] -> moe_h [rows][I] (relu2); down: A = moe_h -> xg [rows][1024] (the gathered A is consumed)
    grouped(v, "nb_pf_dequant_up", kb::pf_up_gemm_variant(I), E, per, I / 16, d_.latent / 64, I, L.experts->ptr(),
            s_.xg.ptr(), s_.moe_h.ptr(), T);
    grouped(v, "nb_pf_dequant_dn", kb::pf_down_gemm_variant(I), E, per, d_.latent / 16, I / 64, d_.latent,
            L.experts->ptr(), s_.moe_h.ptr(), s_.xg.ptr(), T);
    // nb_pf_combine(route, pair_row, y, r, C) - grid (1024 / 256, C)
    launch(kc_(v, "nb_pf_combine"), d_.latent / kb::kPfWg, C_, 1,
           {PtrArg(rt), PtrArg(s_.pair_row.ptr()), PtrArg(s_.xg.ptr()), PtrArg(s_.r.ptr()), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kMoeCombine);
    linear(*L.fc2, s_.r.ptr(), s_.pa);              // fc2_latent -> pa [C][4096]
    linear(*L.shared_down, s_.sh.ptr(), s_.pb);     // shared_experts.down -> pb [C][4096]
    r_.pending = 2;
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  BoltPrefillScratch& s_;
  const loader::BoltDevicePart& part_;
  const model::BoltDesc& d_;
  BoltBuffers& b_;
  const uint32_t pos_, C_;
  const bool eager_;
  PrefillResid r_;
  l0::Mem* tap_;
  int pending_tap_ = -1;
  size_t n_ = 0;
};

}  // namespace

BoltPrefillScratch::BoltPrefillScratch(l0::Context& ctx, const model::BoltDesc& d)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d).ids),
      h0(ctx, l0::MemKind::Device, prefill_sizes(d).h0),
      h1(ctx, l0::MemKind::Device, prefill_sizes(d).h1),
      x(ctx, l0::MemKind::Device, prefill_sizes(d).x),
      pa(ctx, l0::MemKind::Device, prefill_sizes(d).pa),
      pb(ctx, l0::MemKind::Device, prefill_sizes(d).pb),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d).slab),
      seed(ctx, l0::MemKind::Device, prefill_sizes(d).seed),
      xc(ctx, l0::MemKind::Device, prefill_sizes(d).xc),
      dt(ctx, l0::MemKind::Device, prefill_sizes(d).dt),
      cum(ctx, l0::MemKind::Device, prefill_sizes(d).cum),
      cb(ctx, l0::MemKind::Device, prefill_sizes(d).cb),
      y(ctx, l0::MemKind::Device, prefill_sizes(d).y),
      g(ctx, l0::MemKind::Device, prefill_sizes(d).g),
      attn_q(ctx, l0::MemKind::Device, prefill_sizes(d).attn_q),
      logits(ctx, l0::MemKind::Device, prefill_sizes(d).logits),
      routes(ctx, l0::MemKind::Device, prefill_sizes(d).routes),
      lat(ctx, l0::MemKind::Device, prefill_sizes(d).lat),
      sh(ctx, l0::MemKind::Device, prefill_sizes(d).sh),
      r(ctx, l0::MemKind::Device, prefill_sizes(d).r),
      hdr(ctx, l0::MemKind::Device, prefill_sizes(d).hdr),
      tiles(ctx, l0::MemKind::Device, prefill_sizes(d).tiles),
      row_tok(ctx, l0::MemKind::Device, prefill_sizes(d).row_tok),
      pair_row(ctx, l0::MemKind::Device, prefill_sizes(d).pair_row),
      xg(ctx, l0::MemKind::Device, prefill_sizes(d).xg),
      moe_h(ctx, l0::MemKind::Device, prefill_sizes(d).moe_h),
      w(ctx, l0::MemKind::Device, prefill_sizes(d).w),
      desc(d) {}

size_t BoltPrefillScratch::bytes() const {
  return ids.size() + h0.size() + h1.size() + x.size() + pa.size() + pb.size() + slab.size() + seed.size() + xc.size() +
         dt.size() + cum.size() + cb.size() + y.size() + g.size() + attn_q.size() + logits.size() + routes.size() +
         lat.size() + sh.size() + r.size() + hdr.size() + tiles.size() + row_tok.size() + pair_row.size() + xg.size() +
         moe_h.size() + w.size();
}

PrefillResid prefill_resid_in(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev) {
  if (dev == 0 || p.devices < 2) return PrefillResid{};
  return PrefillResid{0, d.is_moe(p.split - 1) ? 2u : 1u};
}
PrefillResid prefill_resid_out(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev) {
  PrefillResid r = prefill_resid_in(d, p, dev);
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    if (r.pending) r.cur = 1 - r.cur;   // the block's norm folds the previous block's rows
    r.pending = d.is_moe(l) ? 2u : 1u;
  }
  return r;
}

PrefillResid prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, BoltPrefillScratch& s,
                           const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b, uint32_t pos,
                           uint32_t C, bool eager, PrefillResid in, l0::Mem* tap) {
  if (!(in == prefill_resid_in(d, b.placement, part.device)))
    throw std::runtime_error("runtime::bolt::prefill_chunk: the residual handed in is not the placement's");
  const PrefillResid out = Walk(cx, kc, s, part, d, b, pos, C, eager, in, tap).run();
  if (!(out == prefill_resid_out(d, b.placement, part.device)))
    throw std::runtime_error("runtime::bolt::prefill_chunk: the walk left h elsewhere than prefill_resid_out says");
  return out;
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, BoltPrefillScratch& s, const loader::BoltDevicePart& part,
                  const model::BoltDesc& d, BoltBuffers& b, uint32_t last_row, PrefillResid r, bool int8_head, l0::Mem* tap) {
  if (last_row >= kPfC) throw std::runtime_error("runtime::bolt::prefill_head: last_row past kPfC");
  if (!part.final_norm || !part.head)
    throw std::runtime_error("runtime::bolt::prefill_head: device " + std::to_string(part.device) + " holds no head");
  if (r.pending == 0) throw std::runtime_error("runtime::bolt::prefill_head: no block's rows to fold");
  const size_t H = d.hidden;
  // decode's nb_res_norm (M 1) over the last row with norm_f: _S1 (o_proj's / out_proj's row) or _S1_T1 (fc2's and
  // shared_down's) into the decode buffers' h1 / x
  cx.launch(kc(kb::res_norm_variant(1, 1, r.pending == 2 ? 1u : 0u, false), "nb_res_norm"), kb::kNormWgs, 1, 1,
            {PtrArg(at(s.h(r.cur), size_t(last_row) * H * 2)), PtrArg(s.pa.as<float>() + size_t(last_row) * H),
             PtrArg((r.pending == 2 ? s.pb : s.pa).as<float>() + size_t(last_row) * H), PtrArg(part.final_norm->ptr()),
             PtrArg(b.h1.ptr()),
             PtrArg(b.x.ptr())});
  if (tap) cx.copy(at(*tap, size_t(d.layers - 1) * H * 2), b.h1.ptr(), H * 2);   // the folded row IS the last block's output
  const loader::DeviceWeight& lm = *part.head;
  if (int8_head) {   // gemv_i8w(w, scales, x, out), grid (N / 64)
    if (lm.kind != model::WeightKind::Int8 || !lm.scales)
      throw std::runtime_error("runtime::bolt::prefill_head: the loaded head is not int8 with scales");
    cx.launch(kc(kb::head_variant(true), "gemv_i8w"), lm.shape.N / kernels::kGemvI8wCols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {   // gemv_bf16(w, x, out) over lm_head's own tiles (untied)
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(d.vocab);
    cx.launch(kc(kb::head_variant(false), "gemv_bf16"), d.vocab / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(kb::argmax1_variant(1), "argmax_stage1"), kb::argmax_groups(), 1, 1,
            {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(kb::argmax2_variant(), "argmax_stage2"), 1, 1, 1, {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::bolt
