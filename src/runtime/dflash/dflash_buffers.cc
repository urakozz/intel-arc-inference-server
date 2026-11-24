#include "runtime/dflash/dflash_buffers.h"

#include <cmath>
#include <cstring>

#include "kernels/dflash_dims.h"

namespace runtime::dflash {

namespace {
l0::Mem dev(l0::Context& ctx, size_t bytes) { return l0::Mem(ctx, l0::MemKind::Device, bytes); }
// A zero_on_reset buffer's fill: shared memory from the host, device memory on the list.
void fill(l0::CmdList& imm, l0::Mem& m) {
  if (m.kind() == l0::MemKind::Shared)
    std::memset(m.ptr(), 0, m.size());
  else
    imm.fill(m.ptr(), 0u, m.size());
}
}  // namespace

DFlashBuffers::DFlashBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t th, uint32_t rows, uint32_t ml,
                             loader::DraftFormat format, bool candidate_head, bool host_candidates, uint32_t dr)
    : desc(d),
      target_hidden(th),
      head_rows(rows),
      max_len(ml),
      draft_rows(dr),
      sz(sizes(d, th, rows, ml, format, candidate_head, dr)),
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
      cand(ctx, host_candidates ? l0::MemKind::Shared : l0::MemKind::Device, sz.cand),
      unary(ctx, host_candidates ? l0::MemKind::Shared : l0::MemKind::Device, sz.unary) {
  if (sz.co) co = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sz.co);
  if (sz.candidate) candidate = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sz.candidate);
  if (sz.hid) hid = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sz.hid);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
  const std::vector<float> t = rope_table(desc.theta, desc.head_dim, max_len + kernels::dflash::kM);
  imm.copy(rope.ptr(), t.data(), t.size() * 4);
}

DFlashSession::DFlashSession(l0::Context& ctx, const DFlashSizes& sz)
    : dctl(ctx, l0::MemKind::Shared, sz.dctl),
      out(ctx, l0::MemKind::Shared, sz.out),
      ring_k(dev(ctx, sz.ring_k)),
      ring_v(dev(ctx, sz.ring_v)),
      taps(dev(ctx, sz.taps)) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

size_t DFlashSession::bytes() const { return gs::buffer_bytes<l0::Mem, SessionSizes>(*this); }

void DFlashSession::zero(l0::CmdList& imm) {
  gs::zero<l0::Mem, SessionSizes>(*this, [&](l0::Mem& m) { fill(imm, m); });
}

void DFlashBuffers::add_sessions(l0::Context& ctx, uint32_t n) {
  while (sessions.size() + 1 < n) sessions.push_back(std::make_unique<DFlashSession>(ctx, sz));
  if (n > 1 && !seqs) {
    const RowsSizes r = rows_sizes(sz, n);
    seqs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, r.seqs);
    proj_seqs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, r.proj_seqs);
    rows_taps = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, r.rows_taps);
    std::memset(seqs->ptr(), 0, r.seqs);
    std::memset(proj_seqs->ptr(), 0, r.proj_seqs);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    imm.fill(rows_taps->ptr(), 0u, rows_taps->size());
  }
}

DFlashBuffers::Session DFlashBuffers::session(uint32_t s) {
  if (s == 0) return {&dctl, &out, &ring_k, &ring_v, &taps};
  DFlashSession& x = *sessions.at(s - 1);
  return {&x.dctl, &x.out, &x.ring_k, &x.ring_v, &x.taps};
}

void DFlashBuffers::zero_session(l0::CmdList& imm, uint32_t s) {
  if (s == 0)
    gs::zero<l0::Mem, SessionSizes>(*this, [&](l0::Mem& m) { fill(imm, m); });
  else
    sessions.at(s - 1)->zero(imm);
}

size_t DFlashBuffers::bytes() const {
  size_t b = gs::buffer_bytes<l0::Mem, SessionSizes, DFlashSizes, RowsSizes>(*this);
  for (const auto& s : sessions) b += s->bytes();
  return b;
}

// Every zero_on_reset buffer: the first session's and the draft list's scratch (not the RoPE table, a constant).
void DFlashBuffers::zero(l0::CmdList& imm) {
  gs::zero<l0::Mem, SessionSizes, DFlashSizes, RowsSizes>(*this, [&](l0::Mem& m) { fill(imm, m); });
}

}  // namespace runtime::dflash
