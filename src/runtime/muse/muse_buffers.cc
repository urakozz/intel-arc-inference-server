#include "runtime/muse/muse_buffers.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "loader/muse_layout.h"
#include "runtime/muse/muse_rope.h"

namespace runtime::muse {
namespace {

std::unique_ptr<l0::Mem> group(l0::Context& ctx, size_t bytes) {
  return bytes ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, bytes) : nullptr;
}
// The K (or V) half of a group: a full layer's [max_len][kv_n] or a ring's [4096][kv_n] rows.
size_t half(size_t kv_and_v) { return kv_and_v / 2; }
// Layer `layer`'s index among this device's full (or sliding) layers.
uint32_t kind_index(const model::MuseGlimmerDesc& d, uint32_t first, uint32_t layer) {
  const bool full = d.is_full(layer);
  uint32_t n = 0;
  for (uint32_t l = first; l < layer; ++l) n += d.is_full(l) == full ? 1 : 0;
  return n;
}

}  // namespace

MuseBuffers::MuseBuffers(l0::Context& ctx, const model::MuseGlimmerDesc& d, const model::MusePlacement& p,
                         uint32_t dev, uint32_t len, MuseAttn at, uint32_t rows_)
    : control(ctx, l0::MemKind::Shared, scratch_sizes(d, len, rows_, at).control),
      full_k(group(ctx, half(device_kv_bytes(d, p, dev, len)))),
      full_v(group(ctx, half(device_kv_bytes(d, p, dev, len)))),
      ring_k(group(ctx, half(device_ring_bytes(d, p, dev)))),
      ring_v(group(ctx, half(device_ring_bytes(d, p, dev)))),
      resid(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).resid),
      x(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).x),
      a(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).a),
      partials(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).partials),
      sumsq_a(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).sumsq_a),
      sumsq_r(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).sumsq_r),
      attn_q(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).attn_q),
      gate(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).gate),
      attn_part(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).attn_part),
      attn_out(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).attn_out),
      h(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).h),
      logits(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).logits),
      argmax_part(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).argmax_part),
      rope(ctx, l0::MemKind::Device, scratch_sizes(d, len, rows_, at).rope),
      attn_s(group(ctx, scratch_sizes(d, len, rows_, at).attn_s)),
      desc(d),
      placement(p),
      device(dev),
      max_len(len),
      attn(at),
      rows(rows_) {
  model::validate(p, d);
  if (rows_ == 0 || rows_ > kVerifyMaxRows)
    throw std::invalid_argument("MuseBuffers: rows " + std::to_string(rows_) + " outside [1, 16] (spec 23f)");
  if (len == 0 || len > d.max_pos)
    throw std::invalid_argument("MuseBuffers: max_len " + std::to_string(len) + " outside [1, " + std::to_string(d.max_pos) +
                                "] (spec 23 decision 6)");
  const std::vector<float> table = rope_table(d.head_dim, d.rope_theta, len);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(rope.ptr(), table.data(), table.size() * 4);
}

void* MuseBuffers::k_layer(uint32_t layer) const {
  if (!holds(layer))
    throw std::logic_error("MuseBuffers: layer " + std::to_string(layer) + " is not on device " + std::to_string(device));
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return desc.is_full(layer)
             ? static_cast<uint8_t*>(full_k->ptr()) + size_t(i) * half(loader::muse_full_layer_kv_bytes(desc, max_len))
             : static_cast<uint8_t*>(ring_k->ptr()) + size_t(i) * half(loader::muse_ring_bytes_per_layer(desc));
}
void* MuseBuffers::v_layer(uint32_t layer) const {
  if (!holds(layer))
    throw std::logic_error("MuseBuffers: layer " + std::to_string(layer) + " is not on device " + std::to_string(device));
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return desc.is_full(layer)
             ? static_cast<uint8_t*>(full_v->ptr()) + size_t(i) * half(loader::muse_full_layer_kv_bytes(desc, max_len))
             : static_cast<uint8_t*>(ring_v->ptr()) + size_t(i) * half(loader::muse_ring_bytes_per_layer(desc));
}

size_t MuseBuffers::kv_bytes() const { return (full_k ? full_k->size() : 0) + (full_v ? full_v->size() : 0); }
size_t MuseBuffers::ring_bytes() const { return (ring_k ? ring_k->size() : 0) + (ring_v ? ring_v->size() : 0); }
size_t MuseBuffers::scratch_bytes() const {
  return control.size() + resid.size() + x.size() + a.size() + partials.size() + sumsq_a.size() + sumsq_r.size() +
         attn_q.size() + gate.size() + attn_part.size() + attn_out.size() + h.size() + logits.size() + argmax_part.size() +
         rope.size() + (attn_s ? attn_s->size() : 0);
}

void MuseBuffers::zero(l0::CmdList& imm) {
  imm.fill(control.ptr(), 0u, control.size());
  for (const std::unique_ptr<l0::Mem>* m : {&full_k, &full_v, &ring_k, &ring_v})
    if (*m) imm.fill((*m)->ptr(), 0u, (*m)->size());
}

}  // namespace runtime::muse
