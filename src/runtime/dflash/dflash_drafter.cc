#include "runtime/dflash/dflash_drafter.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/dflash_dims.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::dflash {

namespace {
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::dflash::DFlashDrafter: " + what);
}
uint32_t checked_max_k(const model::DFlashDesc& d, uint32_t k) {
  require(k >= 1 && k <= d.max_k() && k <= kernels::dflash::kR,
          "max_k " + std::to_string(k) + " is outside 1.." + std::to_string(std::min(d.max_k(), kernels::dflash::kR)) +
              " (the drafter's block of " + std::to_string(d.block) + ")");
  return k;
}
}  // namespace

DFlashDrafter::DFlashDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t head_rows, uint32_t max_len,
                             uint32_t max_k)
    : w_(std::move(w)),
      b_(ctx, w_.desc, w_.target_hidden, head_rows, max_len),
      max_k_(checked_max_k(w_.desc, max_k)),
      project_(build_dflash_project(ctx, w_, b_)) {
  // Spec 19d: the store's name for this drafter - its shapes, format and the two norms' weights as loaded.
  std::vector<uint8_t> norms(w_.hidden_norm->size() + w_.final_norm->size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(norms.data(), w_.hidden_norm->ptr(), w_.hidden_norm->size());
  imm.copy(norms.data() + w_.hidden_norm->size(), w_.final_norm->ptr(), w_.final_norm->size());
  key_ = drafter_key(w_.desc, w_.format, norms.data(), norms.size());
}

void DFlashDrafter::save_ring(l0::CmdList& imm, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  imm.copy(h, b_.ring_k.ptr(), b_.ring_k.size());
  imm.copy(h + b_.ring_k.size(), b_.ring_v.ptr(), b_.ring_v.size());
}

void DFlashDrafter::load_ring(l0::CmdList& imm, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  imm.copy(b_.ring_k.ptr(), h, b_.ring_k.size());
  imm.copy(b_.ring_v.ptr(), h + b_.ring_k.size(), b_.ring_v.size());
}

void DFlashDrafter::attach(Engine& e, bool draft_vocab) {
  require(draft_ == nullptr, "attach() twice");
  require(e.tap_drafter() == this, "the engine was not built with this drafter (Engine(..., &drafter))");
  attach_target(e.context(), e.model(), e.buffers().control, e.max_len(), draft_vocab);
  engine_ = &e;
}

void DFlashDrafter::attach_target(l0::Context& ctx, const loader::LoadedModel& target, const l0::Mem& control,
                                  uint32_t max_len, bool draft_vocab) {
  require(draft_ == nullptr, "attach() twice");
  require(max_len == b_.max_len, "the target's max_len " + std::to_string(max_len) +
                                     " is not the drafter's buffers' " + std::to_string(b_.max_len));
  draft_ = std::make_unique<CapturedStep>(
      build_dflash_draft(ctx, w_, dflash_target(target, draft_vocab), b_, control));
}

const DFlashDraftOut& DFlashDrafter::draft(Engine& e, uint32_t k, float temperature, uint64_t seed) {
  require(engine_ == &e, "draft() before attach() to this engine");
  require(k <= e.max_verify_k(),
          "k " + std::to_string(k) + " is outside 1.." + std::to_string(std::min(max_k_, e.max_verify_k())) +
              " at pos " + std::to_string(e.pos()));
  return draft_with([&e](l0::CmdList& list) { e.replay_aux(list); }, k, temperature, seed);
}

const DFlashDraftOut& DFlashDrafter::draft_with(const std::function<void(l0::CmdList&)>& replay, uint32_t k,
                                                float temperature, uint64_t seed) {
  require(draft_ != nullptr, "draft before attach");
  require(k >= 1 && k <= max_k_, "k " + std::to_string(k) + " is outside 1.." + std::to_string(max_k_));
  set_draft_params(*b_.ctl(), w_.desc, k, temperature, seed);
  replay(draft_->list);
  const DFlashDraftOut& out = *b_.readback();
  require(out.k == k, "the draft list drafted " + std::to_string(out.k) + " ids, not " + std::to_string(k));
  return out;
}

uint32_t DFlashDrafter::window() const {
  uint32_t w = 0;
  for (uint32_t l = 0; l < w_.desc.layers; ++l)
    w = std::max(w, w_.desc.window[l] ? w_.desc.window[l] : std::numeric_limits<uint32_t>::max());
  return w;
}

l0::CmdList& DFlashDrafter::projection(uint32_t pos, uint32_t n) {
  require(n >= 1 && n <= kernels::dflash::kM, "a projection takes 1.." + std::to_string(kernels::dflash::kM) +
                                                   " rows, not " + std::to_string(n));
  require(size_t(pos) + n <= size_t(b_.max_len) + kernels::dflash::kM,
          "positions " + std::to_string(pos) + " + " + std::to_string(n) + " are past the RoPE table");
  DFlashCtl& c = *b_.ctl();
  c.proj_pos = pos;
  c.proj_n = n;
  return project_.list;
}

}  // namespace runtime::dflash
