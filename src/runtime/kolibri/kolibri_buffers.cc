#include "runtime/kolibri/kolibri_buffers.h"

#include <stdexcept>
#include <string>

#include "l0/cmdlist.h"

namespace runtime::kolibri {

KolibriBuffers::KolibriBuffers(l0::Context& ctx, const model::Kolibri1Desc& d, const model::KolPlacement& p,
                               uint32_t dev, uint32_t len, KolAttn at)
    : control(ctx, l0::MemKind::Shared, persistent_sizes(d, p, dev, len).control),
      full_k(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).full_k),
      full_v(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).full_v),
      ring_k(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).ring_k),
      ring_v(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).ring_v),
      resid(ctx, l0::MemKind::Device, scratch_sizes(d).resid),
      x(ctx, l0::MemKind::Device, scratch_sizes(d).x),
      a(ctx, l0::MemKind::Device, scratch_sizes(d).a),
      mo(ctx, l0::MemKind::Device, scratch_sizes(d).mo),
      partials(ctx, l0::MemKind::Device, scratch_sizes(d).partials),
      sumsq_a(ctx, l0::MemKind::Device, scratch_sizes(d).sumsq_a),
      sumsq_r(ctx, l0::MemKind::Device, scratch_sizes(d).sumsq_r),
      attn_q(ctx, l0::MemKind::Device, scratch_sizes(d).attn_q),
      attn_part(ctx, l0::MemKind::Device, scratch_sizes(d).attn_part),
      attn_out(ctx, l0::MemKind::Device, scratch_sizes(d).attn_out),
      logits_r(ctx, l0::MemKind::Device, scratch_sizes(d).logits_r),
      routes(ctx, l0::MemKind::Device, scratch_sizes(d).routes),
      moe_h(ctx, l0::MemKind::Device, scratch_sizes(d).moe_h),
      logits(ctx, l0::MemKind::Device, scratch_sizes(d).logits),
      argmax_part(ctx, l0::MemKind::Device, scratch_sizes(d).argmax_part),
      attn_scores(at == KolAttn::Eager
                      ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, attn_scores_bytes(d, len, at))
                      : nullptr),
      desc(d),
      placement(p),
      device(dev),
      max_len(len),
      attn(at) {
  model::validate(p, d);
}

namespace {
// Layer `layer`'s index among this device's full (or sliding) layers.
uint32_t kind_index(const model::Kolibri1Desc& d, uint32_t first, uint32_t layer) {
  const bool sliding = d.is_sliding(layer);
  uint32_t n = 0;
  for (uint32_t l = first; l < layer; ++l) n += d.is_sliding(l) == sliding ? 1 : 0;
  return n;
}
}  // namespace

void* KolibriBuffers::k_layer(uint32_t layer) const {
  if (!holds(layer))
    throw std::logic_error("KolibriBuffers: layer " + std::to_string(layer) + " is not on device " + std::to_string(device));
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return desc.is_sliding(layer) ? static_cast<uint8_t*>(ring_k.ptr()) + size_t(i) * ring_layer_rows_bytes(desc)
                                : static_cast<uint8_t*>(full_k.ptr()) + size_t(i) * full_layer_rows_bytes(desc, max_len);
}
void* KolibriBuffers::v_layer(uint32_t layer) const {
  if (!holds(layer))
    throw std::logic_error("KolibriBuffers: layer " + std::to_string(layer) + " is not on device " + std::to_string(device));
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return desc.is_sliding(layer) ? static_cast<uint8_t*>(ring_v.ptr()) + size_t(i) * ring_layer_rows_bytes(desc)
                                : static_cast<uint8_t*>(full_v.ptr()) + size_t(i) * full_layer_rows_bytes(desc, max_len);
}

size_t KolibriBuffers::scratch_bytes() const {
  return resid.size() + x.size() + a.size() + mo.size() + partials.size() + sumsq_a.size() + sumsq_r.size() +
         attn_q.size() + attn_part.size() + attn_out.size() + logits_r.size() + routes.size() + moe_h.size() +
         logits.size() + argmax_part.size() + (attn_scores ? attn_scores->size() : 0);
}

void KolibriBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &full_k, &full_v, &ring_k, &ring_v}) imm.fill(m->ptr(), 0u, m->size());
}

}  // namespace runtime::kolibri
