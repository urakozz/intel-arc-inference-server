#include "runtime/bolt/bolt_buffers.h"

#include <stdexcept>
#include <string>

#include "l0/cmdlist.h"
#include "runtime/control.h"

namespace runtime::bolt {

static_assert(sizeof(Control) == 128, "persistent_sizes' control term is runtime::Control's size");

namespace {
l0::Mem dev_mem(l0::Context& ctx, size_t bytes) { return l0::Mem(ctx, l0::MemKind::Device, bytes); }
}  // namespace

BoltBuffers::BoltBuffers(l0::Context& ctx, const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev,
                         uint32_t len, BoltAttn at, BoltState st)
    : control(ctx, l0::MemKind::Shared, persistent_sizes(d, p, dev, len, st).control),
      kv_k(dev_mem(ctx, persistent_sizes(d, p, dev, len, st).kv_k)),
      kv_v(dev_mem(ctx, persistent_sizes(d, p, dev, len, st).kv_v)),
      conv_ring(dev_mem(ctx, persistent_sizes(d, p, dev, len, st).conv_ring)),
      ssm(dev_mem(ctx, persistent_sizes(d, p, dev, len, st).ssm)),
      h0(dev_mem(ctx, scratch_sizes(d, len, at).h0)),
      h1(dev_mem(ctx, scratch_sizes(d, len, at).h1)),
      x(dev_mem(ctx, scratch_sizes(d, len, at).x)),
      slices(dev_mem(ctx, scratch_sizes(d, len, at).slices)),
      xc(dev_mem(ctx, scratch_sizes(d, len, at).xc)),
      y(dev_mem(ctx, scratch_sizes(d, len, at).y)),
      g(dev_mem(ctx, scratch_sizes(d, len, at).g)),
      logits_r(dev_mem(ctx, scratch_sizes(d, len, at).logits_r)),
      routes(dev_mem(ctx, scratch_sizes(d, len, at).routes)),
      lat(dev_mem(ctx, scratch_sizes(d, len, at).lat)),
      sh(dev_mem(ctx, scratch_sizes(d, len, at).sh)),
      moe_a(dev_mem(ctx, scratch_sizes(d, len, at).moe_a)),
      moe_r(dev_mem(ctx, scratch_sizes(d, len, at).moe_r)),
      attn_q(dev_mem(ctx, scratch_sizes(d, len, at).attn_q)),
      attn_part(dev_mem(ctx, scratch_sizes(d, len, at).attn_part)),
      attn_out(dev_mem(ctx, scratch_sizes(d, len, at).attn_out)),
      ones(dev_mem(ctx, scratch_sizes(d, len, at).ones)),
      logits(dev_mem(ctx, scratch_sizes(d, len, at).logits)),
      argmax_part(dev_mem(ctx, scratch_sizes(d, len, at).argmax_part)),
      attn_scores(scratch_sizes(d, len, at).attn_scores
                      ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, scratch_sizes(d, len, at).attn_scores)
                      : nullptr),
      desc(d),
      placement(p),
      device(dev),
      max_len(len),
      attn(at),
      state(st) {
  model::validate(p, d);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.fill(ones.ptr(), 0x3F803F80u, ones.size());   // bf16 1.0 pairs: mg_attn's gate, exactly rne(o) x 1
}

namespace {
// Block `layer`'s index among this device's blocks of its kind.
uint32_t kind_index(const model::BoltDesc& d, uint32_t first, uint32_t layer) {
  uint32_t n = 0;
  for (uint32_t l = first; l < layer; ++l) n += d.block.at(l) == d.block.at(layer) ? 1 : 0;
  return n;
}
[[noreturn]] void not_here(uint32_t layer, uint32_t dev, const char* what) {
  throw std::logic_error("BoltBuffers: block " + std::to_string(layer) + " is not " + what + " on device " +
                         std::to_string(dev));
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }
}  // namespace

void* BoltBuffers::k_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_attention(layer)) not_here(layer, device, "an attention block");
  return at(kv_k, size_t(kind_index(desc, placement.first(device), layer)) * kv_layer_bytes(desc, max_len));
}
void* BoltBuffers::v_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_attention(layer)) not_here(layer, device, "an attention block");
  return at(kv_v, size_t(kind_index(desc, placement.first(device), layer)) * kv_layer_bytes(desc, max_len));
}
void* BoltBuffers::ring_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_mamba(layer)) not_here(layer, device, "a Mamba block");
  return at(conv_ring, size_t(kind_index(desc, placement.first(device), layer)) * ring_layer_bytes(desc));
}
void* BoltBuffers::ssm_layer(uint32_t layer) const {
  if (!holds(layer) || !desc.is_mamba(layer)) not_here(layer, device, "a Mamba block");
  return at(ssm, size_t(kind_index(desc, placement.first(device), layer)) * ssm_layer_bytes(desc, state));
}

size_t BoltBuffers::scratch_bytes() const {
  return h0.size() + h1.size() + x.size() + slices.size() + xc.size() + y.size() + g.size() + logits_r.size() +
         routes.size() + lat.size() + sh.size() + moe_a.size() + moe_r.size() + attn_q.size() + attn_part.size() +
         attn_out.size() + ones.size() + logits.size() + argmax_part.size() + (attn_scores ? attn_scores->size() : 0);
}

void BoltBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &kv_k, &kv_v, &conv_ring, &ssm}) imm.fill(m->ptr(), 0u, m->size());
}

}  // namespace runtime::bolt
