#include "runtime/k2/k2_buffers.h"

#include "l0/cmdlist.h"

namespace runtime::k2 {

K2Buffers::K2Buffers(l0::Context& ctx, const model::K2Desc& d, uint32_t len)
    : control(ctx, l0::MemKind::Shared, persistent_sizes(d, len).control),
      kv_k(ctx, l0::MemKind::Device, persistent_sizes(d, len).kv_k),
      kv_v(ctx, l0::MemKind::Device, persistent_sizes(d, len).kv_v),
      resid(ctx, l0::MemKind::Device, scratch_sizes(d).resid),
      x(ctx, l0::MemKind::Device, scratch_sizes(d).x),
      partials(ctx, l0::MemKind::Device, scratch_sizes(d).partials),
      norm_sumsq(ctx, l0::MemKind::Device, scratch_sizes(d).norm_sumsq),
      attn_q(ctx, l0::MemKind::Device, scratch_sizes(d).attn_q),
      attn_gate(ctx, l0::MemKind::Device, scratch_sizes(d).attn_gate),
      attn_part(ctx, l0::MemKind::Device, scratch_sizes(d).attn_part),
      attn_out(ctx, l0::MemKind::Device, scratch_sizes(d).attn_out),
      router(ctx, l0::MemKind::Device, scratch_sizes(d).router),
      routes(ctx, l0::MemKind::Device, scratch_sizes(d).routes),
      moe_h(ctx, l0::MemKind::Device, scratch_sizes(d).moe_h),
      logits(ctx, l0::MemKind::Device, scratch_sizes(d).logits),
      argmax_part(ctx, l0::MemKind::Device, scratch_sizes(d).argmax_part),
      desc(d),
      max_len(len) {}

void* K2Buffers::kv_k_layer(uint32_t layer) const {
  return static_cast<uint8_t*>(kv_k.ptr()) + size_t(layer) * kv_layer_bytes(desc, max_len);
}
void* K2Buffers::kv_v_layer(uint32_t layer) const {
  return static_cast<uint8_t*>(kv_v.ptr()) + size_t(layer) * kv_layer_bytes(desc, max_len);
}

size_t K2Buffers::scratch_bytes() const {
  return resid.size() + x.size() + partials.size() + norm_sumsq.size() + attn_q.size() +
         attn_gate.size() + attn_part.size() + attn_out.size() + router.size() + routes.size() +
         moe_h.size() + logits.size() + argmax_part.size();
}

void K2Buffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &kv_k, &kv_v}) imm.fill(m->ptr(), 0u, m->size());
}

}  // namespace runtime::k2
