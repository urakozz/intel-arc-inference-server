#include "runtime/buffers.h"

#include <cstddef>
#include "l0/cmdlist.h"
#include "runtime/control.h"

// Every size here comes from runtime/buffer_sizes.cc - the formulas, the model
// constants they read and the arithmetic tests/runtime/buffers_test.cc pins all live
// there (spec 6 §10: the memory planner adds up the same functions). What stays here
// is what is not a size: the memory kind of each allocation and the zero fills.
namespace runtime {

// --- PersistentBuffers -------------------------------------------------------

PersistentBuffers::PersistentBuffers(l0::Context& ctx, uint32_t max_len,
                                     const model::ModelDesc& desc, KvCache kv)
    : PersistentBuffers(ctx, max_len, sizes(max_len, desc, kv),
                        kv_layout(max_len, desc, desc.fa_layers, kv)) {}

PersistentBuffers::PersistentBuffers(l0::Context& ctx, uint32_t max_len,
                                     const model::ModelDesc& desc, KvCache kv,
                                     uint32_t gdn_layers, uint32_t fa_layers)
    : PersistentBuffers(ctx, max_len, stage_sizes(max_len, desc, kv, gdn_layers, fa_layers),
                        kv_layout(max_len, desc, fa_layers, kv)) {}

PersistentBuffers::PersistentBuffers(l0::Context& ctx, uint32_t max_len, const PersistentSizes& s,
                                     const KvLayout& kv)
    : control(ctx, l0::MemKind::Shared, s.control),
      gdn_state(ctx, l0::MemKind::Device, s.gdn_state),
      conv_ring(ctx, l0::MemKind::Device, s.conv_ring),
      kv_k(ctx, l0::MemKind::Device, s.kv_k),
      kv_v(ctx, l0::MemKind::Device, s.kv_v),
      max_len(max_len),
      kv_lay(kv) {
  // A run starts from an empty GDN state, an empty conv ring, an empty KV
  // cache and pos = 0. Scratch is written before it is read every token, so
  // it is left alone. The immediate list is synchronous: the fills have
  // landed when the constructor returns.
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

void PersistentBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &gdn_state, &conv_ring, &kv_k, &kv_v})
    imm.fill(m->ptr(), 0u, m->size());
}

size_t PersistentBuffers::bytes() const {
  return control.size() + gdn_state.size() + conv_ring.size() + kv_k.size() + kv_v.size();
}

// --- DecodeScratch -----------------------------------------------------------

DecodeScratch::DecodeScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc)
    : DecodeScratch(ctx, sizes(max_len, desc)) {}

DecodeScratch::DecodeScratch(l0::Context& ctx, const DecodeScratchSizes& s)
    : resid(ctx, l0::MemKind::Device, s.resid),
      x(ctx, l0::MemKind::Device, s.x),
      partials(ctx, l0::MemKind::Device, s.partials),
      ab_out(ctx, l0::MemKind::Device, s.ab_out),
      norm_sumsq(ctx, l0::MemKind::Device, s.norm_sumsq),
      gdn_o(ctx, l0::MemKind::Device, s.gdn_o),
      attn_q(ctx, l0::MemKind::Device, s.attn_q),
      attn_gate(ctx, l0::MemKind::Device, s.attn_gate),
      attn_part(ctx, l0::MemKind::Device, s.attn_part),
      attn_out(ctx, l0::MemKind::Device, s.attn_out),
      logits(ctx, l0::MemKind::Device, s.logits),
      argmax_part(ctx, l0::MemKind::Device, s.argmax_part),
      // Spec 15c: sized by moe_scratch_layout (0 bytes, so no allocation, when dense).
      moe(s.moe ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, s.moe) : nullptr) {}

size_t DecodeScratch::bytes() const {
  return resid.size() + x.size() + partials.size() + ab_out.size() + norm_sumsq.size() +
         gdn_o.size() + attn_q.size() + attn_gate.size() + attn_part.size() + attn_out.size() +
         logits.size() + argmax_part.size() + (moe ? moe->size() : 0);
}

// --- PrefillScratch ----------------------------------------------------------

PrefillScratch::PrefillScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc)
    : PrefillScratch(ctx, max_len, desc, sizes(max_len, desc)) {}

PrefillScratch::PrefillScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                               const PrefillScratchSizes& s)
    // `ids` is the one Host allocation: the host writes C ids per chunk and
    // the device gathers embeddings from them.
    : ids(ctx, l0::MemKind::Host, s.ids),
      resid(ctx, l0::MemKind::Device, s.resid),
      x(ctx, l0::MemKind::Device, s.x),
      partials(ctx, l0::MemKind::Device, s.partials),
      ab_out(ctx, l0::MemKind::Device, s.ab_out),
      norm_sumsq(ctx, l0::MemKind::Device, s.norm_sumsq),
      gdn_o(ctx, l0::MemKind::Device, s.gdn_o),
      mixer_out(ctx, l0::MemKind::Device, s.mixer_out),
      logits(ctx, l0::MemKind::Device, s.logits),
      argmax_part(ctx, l0::MemKind::Device, s.argmax_part),
      gdn_xb(ctx, l0::MemKind::Device, s.gdn_xb),
      gdn_seed(ctx, l0::MemKind::Device, s.gdn_seed),
      gdn_g(ctx, l0::MemKind::Device, s.gdn_g),
      gdn_beta(ctx, l0::MemKind::Device, s.gdn_beta),
      gdn_A(ctx, l0::MemKind::Device, s.gdn_A),
      gdn_A2(ctx, l0::MemKind::Device, s.gdn_A2),
      gdn_w(ctx, l0::MemKind::Device, s.gdn_w),
      gdn_u(ctx, l0::MemKind::Device, s.gdn_u),
      pf_q(ctx, l0::MemKind::Device, s.pf_q),
      pf_attn(ctx, l0::MemKind::Device, s.pf_attn),
      pf_o(ctx, l0::MemKind::Device, s.pf_o),
      pf_rowsum(ctx, l0::MemKind::Device, s.pf_rowsum),
      max_len(max_len),
      ctx_(&ctx),
      desc_(&desc) {
  // Nothing is zero-filled: no prefill kernel reads scratch it has not first
  // written, and the prefill determinism gate is the standing proof of that -
  // the same rule, and the same reason, Engine::reset() gives for decode.
  // Spec 15d: a MoE model's prefill MoE scratch (runtime::moe_prefill_layout), one
  // allocation, eager like the rest; none on a dense model.
  if (s.moe) moe = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, s.moe);
}

size_t PrefillScratch::bytes() const {
  return ids.size() + resid.size() + x.size() + partials.size() + ab_out.size() +
         norm_sumsq.size() + gdn_o.size() + mixer_out.size() + logits.size() +
         argmax_part.size() + gdn_xb.size() + gdn_seed.size() + gdn_g.size() +
         gdn_beta.size() + gdn_A.size() + gdn_A2.size() + gdn_w.size() + gdn_u.size() +
         pf_q.size() + pf_attn.size() + pf_o.size() +
         pf_rowsum.size() + (moe ? moe->size() : 0);
}

l0::Mem& PrefillScratch::dequant_buffer() {
  if (!dequant_)
    dequant_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device,
                                         sizes(max_len, *desc_).dequant);
  return *dequant_;
}
l0::Mem& PrefillScratch::slab_buffer() {
  if (!slab_)
    slab_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device, sizes(max_len, *desc_).slab);
  return *slab_;
}
l0::Mem& PrefillScratch::pf_s_buffer() {
  if (!pf_s_)
    pf_s_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device, sizes(max_len, *desc_).pf_s);
  return *pf_s_;
}
l0::Mem& PrefillScratch::pf_p_buffer() {
  if (!pf_p_)
    pf_p_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device, sizes(max_len, *desc_).pf_p);
  return *pf_p_;
}
size_t PrefillScratch::lazy_bytes() const {
  return (dequant_ ? dequant_->size() : 0) + (slab_ ? slab_->size() : 0) +
         (pf_s_ ? pf_s_->size() : 0) + (pf_p_ ? pf_p_->size() : 0);
}

// --- MtpBuffers (spec 8) -----------------------------------------------------

MtpBuffers::MtpBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                       uint32_t draft_vocab_size, KvCache kv)
    : MtpBuffers(ctx, max_len, sizes(max_len, desc, draft_vocab_size, kv), draft_vocab_size,
                 kv_layout(max_len, desc, 1, kv)) {}

MtpBuffers::MtpBuffers(l0::Context& ctx, uint32_t max_len, const MtpSizes& s,
                       uint32_t draft_vocab_size, const KvLayout& kv)
    : hctl(ctx, l0::MemKind::Shared, s.hctl),
      gdn_spec(ctx, l0::MemKind::Device, s.gdn_spec),
      kv_k(ctx, l0::MemKind::Device, s.kv_k),
      kv_v(ctx, l0::MemKind::Device, s.kv_v),
      hh(ctx, l0::MemKind::Device, s.hh),
      dh(ctx, l0::MemKind::Device, s.dh),
      logits(ctx, l0::MemKind::Device, s.logits),
      // Spec 8 §11: sized by MtpDims::sizes (0 bytes, so no allocation, when off).
      dv_logits(s.dv_logits ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, s.dv_logits)
                            : nullptr),
      max_len(max_len),
      draft_vocab(draft_vocab_size),
      kv_lay(kv) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

void MtpBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&hctl, &gdn_spec, &kv_k, &kv_v, &hh, &dh})
    imm.fill(m->ptr(), 0u, m->size());
  imm.fill(logits.ptr(), draft_vocab ? kNegInfBits : 0u, logits.size());   // spec 8 §11
  if (dv_logits) imm.fill(dv_logits->ptr(), 0u, dv_logits->size());
}

size_t MtpBuffers::bytes() const {
  return hctl.size() + gdn_spec.size() + kv_k.size() + kv_v.size() + hh.size() + dh.size() +
         logits.size() + (dv_logits ? dv_logits->size() : 0);
}

// --- DecodeBuffers, the view -------------------------------------------------

DecodeBuffers::DecodeBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                             KvCache kv)
    : own_p_(new PersistentBuffers(ctx, max_len, desc, kv)),
      own_s_(new DecodeScratch(ctx, max_len, desc)),
      control(own_p_->control),
      gdn_state(own_p_->gdn_state),
      conv_ring(own_p_->conv_ring),
      kv_k(own_p_->kv_k),
      kv_v(own_p_->kv_v),
      resid(own_s_->resid),
      x(own_s_->x),
      partials(own_s_->partials),
      ab_out(own_s_->ab_out),
      norm_sumsq(own_s_->norm_sumsq),
      gdn_o(own_s_->gdn_o),
      attn_q(own_s_->attn_q),
      attn_gate(own_s_->attn_gate),
      attn_part(own_s_->attn_part),
      attn_out(own_s_->attn_out),
      logits(own_s_->logits),
      argmax_part(own_s_->argmax_part),
      moe(own_s_->moe.get()),
      max_len(max_len),
      kv_lay(own_p_->kv_lay) {}

DecodeBuffers::DecodeBuffers(PersistentBuffers& p, DecodeScratch& s)
    : control(p.control),
      gdn_state(p.gdn_state),
      conv_ring(p.conv_ring),
      kv_k(p.kv_k),
      kv_v(p.kv_v),
      resid(s.resid),
      x(s.x),
      partials(s.partials),
      ab_out(s.ab_out),
      norm_sumsq(s.norm_sumsq),
      gdn_o(s.gdn_o),
      attn_q(s.attn_q),
      attn_gate(s.attn_gate),
      attn_part(s.attn_part),
      attn_out(s.attn_out),
      logits(s.logits),
      argmax_part(s.argmax_part),
      moe(s.moe.get()),
      max_len(p.max_len),
      kv_lay(p.kv_lay) {}

size_t DecodeBuffers::persistent_bytes() const {
  return control.size() + gdn_state.size() + conv_ring.size() + kv_k.size() + kv_v.size();
}

size_t DecodeBuffers::scratch_bytes() const {
  return resid.size() + x.size() + partials.size() + ab_out.size() + norm_sumsq.size() +
         gdn_o.size() + attn_q.size() + attn_gate.size() + attn_part.size() + attn_out.size() +
         logits.size() + argmax_part.size() + (moe ? moe->size() : 0);
}
}  // namespace runtime
