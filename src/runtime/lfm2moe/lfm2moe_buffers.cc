#include "runtime/lfm2moe/lfm2moe_buffers.h"

#include <stdexcept>
#include <string>

#include "l0/cmdlist.h"

namespace runtime::lfm2moe {

Lfm2MoeBuffers::Lfm2MoeBuffers(l0::Context& ctx, const model::Lfm2MoeDesc& d, const model::LfmPlacement& p,
                               uint32_t dev, uint32_t len, LfmAttn at)
    : control(ctx, l0::MemKind::Shared, persistent_sizes(d, p, dev, len).control),
      kv_k(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).kv_k),
      kv_v(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).kv_v),
      conv_ring(ctx, l0::MemKind::Device, persistent_sizes(d, p, dev, len).conv_ring),
      h0(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).h0),
      h1(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).h1),
      x(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).x),
      slices(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).slices),
      conv_y(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).conv_y),
      logits_r(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).logits_r),
      routes(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).routes),
      moe_h(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).moe_h),
      attn_q(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).attn_q),
      attn_part(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).attn_part),
      attn_out(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).attn_out),
      s_mlp(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).s_mlp),
      logits(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).logits),
      argmax_part(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).argmax_part),
      attn_scores(at == LfmAttn::Eager
                      ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).attn_scores)
                      : nullptr),
      desc(d),
      placement(p),
      device(dev),
      max_len(len),
      attn(at) {
  model::validate(p, d);
}

namespace {
// Layer `layer`'s index among this device's attention (or conv) layers.
uint32_t kind_index(const model::Lfm2MoeDesc& d, uint32_t first, uint32_t layer) {
  const bool attn = d.is_attention(layer);
  uint32_t n = 0;
  for (uint32_t l = first; l < layer; ++l) n += d.is_attention(l) == attn ? 1 : 0;
  return n;
}
[[noreturn]] void not_here(uint32_t layer, uint32_t dev, const char* what) {
  throw std::logic_error("Lfm2MoeBuffers: layer " + std::to_string(layer) + " is not " + what + " on device " +
                         std::to_string(dev));
}
}  // namespace

void* Lfm2MoeBuffers::k_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_attention(layer)) not_here(layer, device, "an attention layer");
  return static_cast<uint8_t*>(kv_k.ptr()) + size_t(kind_index(desc, placement.first(device), layer)) *
                                                  kv_layer_bytes(desc, max_len);
}
void* Lfm2MoeBuffers::v_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_attention(layer)) not_here(layer, device, "an attention layer");
  return static_cast<uint8_t*>(kv_v.ptr()) + size_t(kind_index(desc, placement.first(device), layer)) *
                                                  kv_layer_bytes(desc, max_len);
}
void* Lfm2MoeBuffers::ring_layer(uint32_t layer) const {
  if (!holds(layer) || desc.is_attention(layer)) not_here(layer, device, "a conv layer");
  return static_cast<uint8_t*>(conv_ring.ptr()) + size_t(kind_index(desc, placement.first(device), layer)) *
                                                       ring_layer_bytes(desc);
}

size_t Lfm2MoeBuffers::scratch_bytes() const {
  return h0.size() + h1.size() + x.size() + slices.size() + conv_y.size() + logits_r.size() + routes.size() +
         moe_h.size() + attn_q.size() + attn_part.size() + attn_out.size() + s_mlp.size() + logits.size() +
         argmax_part.size() + (attn_scores ? attn_scores->size() : 0);
}

void Lfm2MoeBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &kv_k, &kv_v, &conv_ring}) imm.fill(m->ptr(), 0u, m->size());
}

}  // namespace runtime::lfm2moe
