#include "runtime/dflash/muse_drafter.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/dflash_dims.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::dflash {

namespace {
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash::MuseDrafter: " + what);
}
}  // namespace

const char* muse_q_name(MuseQ q) { return q == MuseQ::Capped ? "capped" : "raw"; }
bool parse_muse_q(const std::string& s, MuseQ& out) {
  if (s == "raw") return out = MuseQ::Raw, true;
  if (s == "capped") return out = MuseQ::Capped, true;
  return false;
}

MuseDrafter::MuseDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t max_len, uint32_t max_k, MuseQ q)
    : w_(std::move(w)),
      b_(ctx, w_.desc, w_.target_hidden, w_.desc.vocab, max_len),
      max_k_(max_k),
      q_(q),
      project_(build_dflash_project(ctx, w_, b_)) {
  require(w_.desc.muse(), "the weights are not a Muse-Glimmer assistant's (model::DFlashForm::Muse)");
  require(max_k >= 1 && max_k <= w_.desc.max_k() && max_k <= kernels::dflash::kMuseR,
          "max_k " + std::to_string(max_k) + " is outside 1..15 (the block of 16)");
  std::vector<uint8_t> norms(w_.hidden_norm->size() + w_.final_norm->size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(norms.data(), w_.hidden_norm->ptr(), w_.hidden_norm->size());
  imm.copy(norms.data() + w_.hidden_norm->size(), w_.final_norm->ptr(), w_.final_norm->size());
  key_ = drafter_key(w_.desc, w_.format, norms.data(), norms.size());
}

void MuseDrafter::attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head,
                         const l0::Mem& control) {
  require(draft_ == nullptr, "attach() twice");
  DFlashTarget t;
  t.embed = &embed;
  t.head = &head;
  t.vocab_rows = head.shape.N;
  t.vocab_used = head.shape.N;   // Muse's tokenizer defines every one of the 202048 rows
  if (q_ == MuseQ::Capped) {
    t.q_mult = kCapMult;
    t.q_softcap = kCapSoftcap;
  } else {
    t.q_mult = 1.0f;
    t.q_softcap = 0.0f;
  }
  draft_ = std::make_unique<CapturedStep>(build_dflash_draft(ctx, w_, t, b_, control));
}

const MuseDraftOut& MuseDrafter::draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k) {
  require(draft_ != nullptr, "draft before attach");
  require(k >= 1 && k <= max_k_, "k " + std::to_string(k) + " is outside 1.." + std::to_string(max_k_));
  set_draft_params(*b_.ctl(), w_.desc, k, 0.0f, 0);
  replay(draft_->list);
  const MuseDraftOut& out = *b_.muse_readback();
  require(out.k == k, "the draft list drafted " + std::to_string(out.k) + " ids, not " + std::to_string(k));
  return out;
}

void MuseDrafter::set_draft(uint32_t j, uint32_t id) {
  MuseDraftOut& o = *b_.muse_readback();
  require(j < o.k, "draft " + std::to_string(j) + " is past the last block's " + std::to_string(o.k));
  for (uint32_t c = 0; c < MuseDraftOut::kTopK; ++c)
    if (o.cand[j][c] == id) {
      o.ids[j] = id;
      o.chosen[j] = c;
      b_.vids->as<uint32_t>()[1 + j] = id;
      return;
    }
  require(false, "id " + std::to_string(id) + " is not a candidate of draft row " + std::to_string(j));
}

l0::CmdList& MuseDrafter::projection(uint32_t pos, uint32_t n) {
  const uint32_t M = kernels::dflash::kMuseM;
  require(n >= 1 && n <= M, "a projection takes 1.." + std::to_string(M) + " rows, not " + std::to_string(n));
  require(size_t(pos) + n <= size_t(b_.max_len) + M,
          "positions " + std::to_string(pos) + " + " + std::to_string(n) + " are past the RoPE table");
  DFlashCtl& c = *b_.ctl();
  c.proj_pos = pos;
  c.proj_n = n;
  return project_.list;
}

void MuseDrafter::save_ring(l0::CmdList& imm, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  imm.copy(h, b_.ring_k.ptr(), b_.ring_k.size());
  imm.copy(h + b_.ring_k.size(), b_.ring_v.ptr(), b_.ring_v.size());
}

void MuseDrafter::load_ring(l0::CmdList& imm, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  imm.copy(b_.ring_k.ptr(), h, b_.ring_k.size());
  imm.copy(b_.ring_v.ptr(), h + b_.ring_k.size(), b_.ring_v.size());
}

}  // namespace runtime::dflash
