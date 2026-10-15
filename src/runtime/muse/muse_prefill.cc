#include "runtime/muse/muse_prefill.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/control.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/profile.h"

// muse_prefill.cc - Muse-Glimmer's prefill chunk on one device (muse_prefill.h has the walk in one table). Every binding
// site names the kernel it honours; argument orders are each __kernel's, in src/kernels/muse/{mg_pf_attn,mg_pf_prep,
// mg_prep}.cl and the reused sources (prefill/pf_prep.cl, prefill/pf_gemm.cl, k2/k2_pf_linear.cl, gemv_i8w.cl /
// gemv_bf16.cl, argmax.cl).
namespace runtime::muse {
namespace {

namespace km = kernels::muse;
using prefill::arg_val;
using prefill::Phase;
using prefill::PtrArg;

// The constants muse_sizes.h spells device-free, held to the kernels' host mirror here.
static_assert(kPfC == km::kPfC && kPfSlab == km::kPfSlab, "runtime/muse/muse_sizes.h and kernels/muse_kernels.h disagree");
static_assert(km::kPfSlab == kernels::kPfSlabWidth, "a whole slab is pf_gemm's 1024 columns");
static_assert(kNormG == km::kNormG, "the Σ² chunks moved");
static_assert(offsetof(Control, pos) == kernels::ctrl_index::kPos * 4 &&
                  offsetof(Control, n_active) == kernels::ctrl_index::kNActive * 4,
              "runtime::Control and CTRL_DEFINES disagree");
static_assert(model::MuseGlimmerDesc::kRing >= kPfC + km::kWindow - 1,
              "a chunk's own keys and the window before it must fit the ring without colliding");

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::muse::prefill_chunk: " + what);
}
const float* fat(const l0::Mem& m, size_t floats) { return static_cast<const float*>(m.ptr()) + floats; }

class Walk {
 public:
  Walk(prefill::Context& cx, prefill::KernelCache& kc, MusePrefillScratch& s, const loader::MuseDevicePart& part,
       const model::MuseGlimmerDesc& d, MuseBuffers& b, uint32_t pos, uint32_t C, bool eager)
      : cx_(cx), kc_(kc), s_(s), part_(part), d_(d), b_(b), pos_(pos), C_(C), eager_(eager) {}

  void run() {
    check();
    if (part_.device == 0) {
      // mg_pf_embed(ids, embed, resid, sumsq) - grid (1, C): the residual rows and layer 0's Σ²
      launch(kc_(km::pf_embed_variant(), "mg_pf_embed"), 1, C_, 1,
             {PtrArg(s_.ids.ptr()), PtrArg(part_.embed->ptr()), PtrArg(s_.resid.ptr()), PtrArg(s_.sumsq_r.ptr())});
      prefill::profile_wait(cx_, Phase::kNorm);
    }
    for (uint32_t l = part_.first; l < part_.end; ++l) layer(l);
    const size_t want = prefill_device_launches(d_, b_.placement, part_.device);
    require(n_ == want, "device " + std::to_string(part_.device) + "'s chunk appended " + std::to_string(n_) +
                            " launches, not the " + std::to_string(want) + " runtime::muse::prefill_device_launches gives");
  }

 private:
  void check() {
    require(C_ > 0 && C_ <= kPfC, "C = " + std::to_string(C_) + " is outside (0, kPfC]");
    require(size_t(pos_) + C_ <= b_.max_len, "pos + C exceeds max_len (the full layers' KV and the RoPE table)");
    require(d_.hidden == km::kHidden && d_.q_heads == km::kQHeads && d_.kv_heads == km::kKvHeads &&
                d_.head_dim == km::kHd && d_.qgkv_rows() == km::kQgkvN && d_.inter == km::kInter &&
                d_.window == km::kWindow && model::MuseGlimmerDesc::kRing == km::kRing,
            "the descriptor is not Muse-Glimmer-30B's shape; the kernels bake it");
    require(s_.desc.layers == d_.layers, "the prefill scratch was sized for another descriptor");
    require(part_.device == b_.device && part_.first == b_.placement.first(b_.device) &&
                part_.end == b_.placement.end(b_.device),
            "the loaded part and the buffers are not the same device's");
    if (part_.device == 0) require(bool(part_.embed), "device 0 holds no embedding");
    const PrefillSizes ps = prefill_sizes(d_);
    require(s_.partials.size() == ps.partials && s_.slab.size() == ps.slab && s_.attn_q.size() == ps.attn_q &&
                s_.h.size() == ps.h && s_.sumsq_r.size() == ps.sumsq_r,
            "the prefill scratch is not runtime::muse::prefill_sizes'");
  }

  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz, std::initializer_list<prefill::KernelArg> args) {
    cx_.launch(k, gx, gy, gz, args);
    ++n_;
  }

  // mg_norm(sumsq, resid, w, x) at M kPfC - grid (kNormW, C).
  void norm(const float* w) {
    launch(kc_(km::pf_norm_variant(), "mg_norm"), km::kNormW, C_, 1,
           {PtrArg(s_.sumsq_r.ptr()), PtrArg(s_.resid.ptr()), PtrArg(w), PtrArg(s_.x.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }
  // pf_res_fold _Z(partials, a, sumsq_a, m_count) - grid (kNormG, C): the sub-block's ONE slice -> its own bf16 row a
  // and Σa² ([kNormG][kPfC]: m_count is the pitch, kPfC, as mg_post_add's M).
  void fold_zero() {
    launch(kc_(km::pf_fold_zero_variant(), "pf_res_fold"), km::kNormG, C_, 1,
           {PtrArg(s_.partials.ptr()), PtrArg(s_.a.ptr()), PtrArg(s_.sumsq_a.ptr()), arg_val(kPfC)});
    prefill::profile_wait(cx_, Phase::kNorm);
  }
  // mg_post_add(sumsq_sub, sub, w_post, resid, sumsq_out) at M kPfC - grid (kNormG, C).
  void post_add(const float* w) {
    launch(kc_(km::pf_post_add_variant(), "mg_post_add"), km::kNormG, C_, 1,
           {PtrArg(s_.sumsq_a.ptr()), PtrArg(s_.a.ptr()), PtrArg(w), PtrArg(s_.resid.ptr()), PtrArg(s_.sumsq_r.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
  }

  // One int4 linear: per slab k2_pf_dequant_slab(w, scales, slab, n0, ns) (grid (ns / 16, K / 64)) then pf_gemm_T0
  // x [C][K] x slab [K][ns] into partials + n0 at pitch pf_ld(N) - slabs of 1024 and one zero-padded tail.
  void linear(const loader::DeviceWeight& w, const void* x) {
    const model::GemvShape& sh = w.shape;
    require(w.kind == model::WeightKind::Int4 && sh.layout == 0 && bool(w.scales),
            "a Muse linear is int4 GPTQ layout 0 with its scales");
    require(sh.K % 64 == 0 && sh.N % 16 == 0, "the linear is not whole k-groups / tiles");
    require(s_.slab.size() >= size_t(sh.K) * kPfSlab * 2, "the slab buffer is smaller than [K][1024]");
    const uint32_t ld = pf_ld(sh.N);
    require(s_.partials.size() >= size_t(kPfC) * ld * 4, "partials is smaller than [kPfC][pf_ld(N)]");
    l0::Kernel& sk = kc_(km::pf_slab_variant(sh.K, sh.N), "k2_pf_dequant_slab");
    for (uint32_t n0 = 0; n0 < sh.N;) {
      const uint32_t ns = pf_slab_width(sh.N, n0);
      launch(sk, ns / 16, sh.K / 64, 1,
             {PtrArg(w.mem.ptr()), PtrArg(w.scales->ptr()), PtrArg(s_.slab.ptr()), arg_val(n0), arg_val(ns)});
      prefill::profile_wait(cx_, Phase::kSlabDequant);
      const prefill::GemmBatch gb{C_, sh.K, ns, 1, sh.K, ns, ld, 0, 0, 0};
      prefill::gemm_l0(cx_, kc_, gb, static_cast<const uint16_t*>(x), s_.slab.as<uint16_t>(),
                       s_.partials.as<float>() + n0, false);
      ++n_;
      prefill::profile_wait(cx_, Phase::kSlabGemm);
      n0 += ns;
    }
  }

  // mg_attn_prep at M kPfC / S 1, the attention over the ring or the full cache, the gate.
  void attention(uint32_t l) {
    const bool sliding = !d_.is_full(l);
    void* kvk = b_.k_layer(l);
    void* kvv = b_.v_layer(l);
    // mg_attn_prep(ctrl, partials, rope, attn_q, gate, kv_k, kv_v) - grid (2 q + 2 kv heads, C)
    launch(kc_(km::pf_attn_prep_variant(sliding), "mg_attn_prep"), km::kPrepGroups, C_, 1,
           {PtrArg(b_.control.ptr()), PtrArg(s_.partials.ptr()), PtrArg(b_.rope.ptr()), PtrArg(s_.attn_q.ptr()),
            PtrArg(s_.gate.ptr()), PtrArg(kvk), PtrArg(kvv)});
    prefill::profile_wait(cx_, Phase::kAttnPrep);
    // mg_pf_flash_attn(q, kv_k, kv_v, out, pos, C): flash grid (ceil(C / RPW), kv heads, GQA / HPW); eager (q heads, C)
    l0::Kernel& fk = kc_(km::pf_flash_variant(sliding, eager_), "mg_pf_flash_attn");
    if (eager_)
      launch(fk, d_.q_heads, C_, 1,
             {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.attn_out.ptr()), arg_val(pos_), arg_val(C_)});
    else
      launch(fk, (C_ + km::kPfRpw - 1) / km::kPfRpw, d_.kv_heads, d_.q_heads / d_.kv_heads / km::kPfHpw,
             {PtrArg(s_.attn_q.ptr()), PtrArg(kvk), PtrArg(kvv), PtrArg(s_.attn_out.ptr()), arg_val(pos_), arg_val(C_)});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
    // mg_pf_attn_gate(gate, attn) - grid (q_n / 256, C): rne(rne(o) x bf16(sigmoid(g))) in place
    launch(kc_(km::pf_gate_variant(), "mg_pf_attn_gate"), km::kPfGateGroups, C_, 1,
           {PtrArg(s_.gate.ptr()), PtrArg(s_.attn_out.ptr())});
    prefill::profile_wait(cx_, Phase::kAttnFlash);
  }

  void layer(uint32_t l) {
    const loader::MuseLayer& L = part_.layer(l);
    require(L.qgkv && L.oproj && L.gate_up && L.down && L.norms, "layer " + std::to_string(l) + " is not fully loaded");
    require(L.qgkv->shape.N == d_.qgkv_rows() && L.oproj->shape.N == d_.hidden && L.oproj->shape.K == d_.q_n() &&
                L.gate_up->shape.N == 2 * d_.inter && L.down->shape.K == d_.inter,
            "layer " + std::to_string(l) + "'s linears are not Muse's shapes");
    const l0::Mem& nw = *L.norms;
    // --- attention -------------------------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_input()));
    linear(*L.qgkv, s_.x.ptr());         // q || gate || k || v -> partials [C][8704]
    attention(l);
    linear(*L.oproj, s_.attn_out.ptr());   // o_proj -> partials [C][6656]
    fold_zero();
    post_add(fat(nw, d_.norm_off_post_attn()));
    // --- the MLP ---------------------------------------------------------------------------------------------------
    norm(fat(nw, d_.norm_off_pre_ff()));
    linear(*L.gate_up, s_.x.ptr());      // gate || up (concatenated) -> partials [C][39936]
    // mg_silu_mul(partials, h) - grid (ceil(I / 256), C)
    launch(kc_(km::pf_silu_variant(), "mg_silu_mul"), km::kSiluGroups, C_, 1,
           {PtrArg(s_.partials.ptr()), PtrArg(s_.h.ptr())});
    prefill::profile_wait(cx_, Phase::kNorm);
    linear(*L.down, s_.h.ptr());         // down -> partials [C][6656]
    fold_zero();
    post_add(fat(nw, d_.norm_off_post_ff()));
  }

  prefill::Context& cx_;
  prefill::KernelCache& kc_;
  MusePrefillScratch& s_;
  const loader::MuseDevicePart& part_;
  const model::MuseGlimmerDesc& d_;
  MuseBuffers& b_;
  const uint32_t pos_, C_;
  const bool eager_;
  size_t n_ = 0;
};

}  // namespace

MusePrefillScratch::MusePrefillScratch(l0::Context& ctx, const model::MuseGlimmerDesc& d)
    : ids(ctx, l0::MemKind::Host, prefill_sizes(d).ids),
      resid(ctx, l0::MemKind::Device, prefill_sizes(d).resid),
      x(ctx, l0::MemKind::Device, prefill_sizes(d).x),
      a(ctx, l0::MemKind::Device, prefill_sizes(d).a),
      partials(ctx, l0::MemKind::Device, prefill_sizes(d).partials),
      slab(ctx, l0::MemKind::Device, prefill_sizes(d).slab),
      sumsq_a(ctx, l0::MemKind::Device, prefill_sizes(d).sumsq_a),
      sumsq_r(ctx, l0::MemKind::Device, prefill_sizes(d).sumsq_r),
      attn_q(ctx, l0::MemKind::Device, prefill_sizes(d).attn_q),
      gate(ctx, l0::MemKind::Device, prefill_sizes(d).gate),
      attn_out(ctx, l0::MemKind::Device, prefill_sizes(d).attn_out),
      h(ctx, l0::MemKind::Device, prefill_sizes(d).h),
      desc(d) {}

size_t MusePrefillScratch::bytes() const {
  return ids.size() + resid.size() + x.size() + a.size() + partials.size() + slab.size() + sumsq_a.size() +
         sumsq_r.size() + attn_q.size() + gate.size() + attn_out.size() + h.size();
}

void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, MusePrefillScratch& s,
                   const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b, uint32_t pos,
                   uint32_t C, bool eager) {
  Walk(cx, kc, s, part, d, b, pos, C, eager).run();
}

void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, MusePrefillScratch& s,
                  const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b,
                  uint32_t last_row) {
  if (last_row >= kPfC) throw std::runtime_error("runtime::muse::prefill_head: last_row past kPfC");
  if (!part.final_norm || !part.lm_head)
    throw std::runtime_error("runtime::muse::prefill_head: device " + std::to_string(part.device) + " holds no head");
  // mg_final_norm(sumsq, resid, w, x) from the M kPfC binary at grid (kNormW, 1): row 0 of the offsets IS the last
  // row - sumsq[g][kPfC] at + last_row, resid at + last_row x hidden - so the chain is decode's M 1 mg_final_norm's
  // (plain w, pow(ms + 1e-5, -0.5)), into the decode buffers' x.
  const uint16_t* row = s.resid.as<uint16_t>() + size_t(last_row) * d.hidden;
  const float* sums = s.sumsq_r.as<float>() + last_row;
  cx.launch(kc(km::pf_norm_variant(), "mg_final_norm"), km::kNormW, 1, 1,
            {PtrArg(sums), PtrArg(row), PtrArg(part.final_norm->ptr()), PtrArg(b.x.ptr())});
  const loader::DeviceWeight& lm = *part.lm_head;
  if (lm.kind == model::WeightKind::Int8) {
    cx.launch(kc(km::lm_head_variant(true), "gemv_i8w"), lm.shape.N / kernels::kGemvI8wCols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    cx.launch(kc(km::lm_head_variant(false), "gemv_bf16"), lm.shape.N / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(b.x.ptr()), PtrArg(b.logits.ptr())});
  }
  cx.launch(kc(km::argmax1_variant(1), "argmax_stage1"), km::argmax_groups(), 1, 1,
            {PtrArg(b.logits.ptr()), PtrArg(b.argmax_part.ptr())});
  cx.launch(kc(km::argmax2_variant(), "argmax_stage2"), 1, 1, 1, {PtrArg(b.control.ptr()), PtrArg(b.argmax_part.ptr())});
  prefill::profile_wait(cx, Phase::kHead);
}

}  // namespace runtime::muse
