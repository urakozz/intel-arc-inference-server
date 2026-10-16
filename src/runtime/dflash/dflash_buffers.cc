#include "runtime/dflash/dflash_buffers.h"

#include <cmath>
#include <cstring>

namespace runtime::dflash {

namespace {
l0::Mem dev(l0::Context& ctx, size_t bytes) { return l0::Mem(ctx, l0::MemKind::Device, bytes); }
}  // namespace

DFlashBuffers::DFlashBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t th, uint32_t rows, uint32_t ml)
    : desc(d),
      target_hidden(th),
      head_rows(rows),
      max_len(ml),
      sz(sizes(d, th, rows, ml)),
      dctl(ctx, l0::MemKind::Shared, sz.dctl),
      out(ctx, l0::MemKind::Shared, sz.out),
      rope(dev(ctx, sz.rope)),
      ring_k(dev(ctx, sz.ring_k)),
      ring_v(dev(ctx, sz.ring_v)),
      taps(dev(ctx, sz.taps)),
      resid(dev(ctx, sz.resid)),
      h(dev(ctx, sz.h)),
      x(dev(ctx, sz.x)),
      xa(dev(ctx, sz.xa)),
      gout(dev(ctx, sz.gout)),
      q(dev(ctx, sz.q)),
      kb(dev(ctx, sz.kb)),
      vb(dev(ctx, sz.vb)),
      part(dev(ctx, sz.part)),
      kv(dev(ctx, sz.kv)),
      logits(dev(ctx, sz.logits)),
      tpart(dev(ctx, sz.tpart)),
      cand(dev(ctx, sz.cand)),
      unary(dev(ctx, sz.unary)) {
  if (sz.co) co = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sz.co);
  if (sz.hid) hid = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sz.hid);
  if (sz.vids) vids = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, sz.vids);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

size_t DFlashBuffers::bytes() const {
  size_t b = 0;
  for (const l0::Mem* m : {&dctl, &out, &rope, &ring_k, &ring_v, &taps, &resid, &h, &x, &xa, &gout, &q, &kb, &vb,
                           &part, &kv, &logits, &tpart, &cand, &unary})
    b += m->size();
  if (co) b += co->size();
  if (hid) b += hid->size();
  if (vids) b += vids->size();
  return b;
}

void DFlashBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&ring_k, &ring_v, &taps, &resid, &h, &x, &xa, &gout, &q, &kb, &vb, &part, &kv, &logits, &tpart,
                     &cand, &unary})
    imm.fill(m->ptr(), 0u, m->size());
  if (co) imm.fill(co->ptr(), 0u, co->size());
  if (hid) imm.fill(hid->ptr(), 0u, hid->size());
  std::memset(dctl.ptr(), 0, dctl.size());
  std::memset(out.ptr(), 0, out.size());
  if (vids) std::memset(vids->ptr(), 0, vids->size());
  const std::vector<float> t = rope_table(desc.theta, desc.head_dim, max_len + list_rows(desc));
  imm.copy(rope.ptr(), t.data(), t.size() * 4);
}

}  // namespace runtime::dflash
