#include "runtime/dflash/dflash_drafter.h"

#include <algorithm>
#include <cstring>
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
      b_(ctx, w_.desc, w_.target_hidden, head_rows, max_len, w_.format, bool(w_.candidate_head)),
      max_k_(checked_max_k(w_.desc, max_k)),
      project_(build_dflash_project(ctx, w_, b_)) {
  // Spec 19d: the store's name for this drafter - its shapes, format and the two norms' weights as loaded.
  std::vector<uint8_t> norms(w_.hidden_norm->size() + w_.final_norm->size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(norms.data(), w_.hidden_norm->ptr(), w_.hidden_norm->size());
  imm.copy(norms.data() + w_.hidden_norm->size(), w_.final_norm->ptr(), w_.final_norm->size());
  key_ = drafter_key(w_.desc, w_.format, norms.data(), norms.size(), w_.packed);
}

void DFlashDrafter::save_ring(l0::CmdList& imm, void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  const DFlashBuffers::Session x = ses();
  imm.copy(h, x.ring_k->ptr(), x.ring_k->size());
  imm.copy(h + x.ring_k->size(), x.ring_v->ptr(), x.ring_v->size());
}

void DFlashDrafter::load_ring(l0::CmdList& imm, const void* host) {
  const auto* h = static_cast<const uint8_t*>(host);
  const DFlashBuffers::Session x = ses();
  imm.copy(x.ring_k->ptr(), h, x.ring_k->size());
  imm.copy(x.ring_v->ptr(), h + x.ring_k->size(), x.ring_v->size());
}

void DFlashDrafter::reset(l0::CmdList& imm) {
  if (b_.sessions.empty())
    b_.zero(imm);
  else
    b_.zero_session(imm, cur_);   // plan 19v3b: the selected slot's session; the shared scratch is written before read
}

void DFlashDrafter::add_sessions(l0::Context& ctx, uint32_t n) {
  require(draft_ == nullptr && b_.sessions.empty(), "add_sessions() after attach() or twice");
  b_.add_sessions(ctx, n);
  for (uint32_t q = 1; q < n; ++q) projects_.push_back(build_dflash_project(ctx, w_, b_, {}, q));
  if (n > 1) rows_project_ = std::make_unique<CapturedStep>(build_dflash_project_rows(ctx, w_, b_, n));
}

l0::CmdList& DFlashDrafter::rows_projection(const std::vector<RowsCommit>& commits) {
  require(rows_project_ != nullptr, "rows_projection() without sessions");
  namespace kd = kernels::dflash;
  uint32_t* t = b_.proj_seqs->as<uint32_t>();
  std::fill(t, t + kd::kM + 8 * kd::kSeqs, 0u);
  std::fill(t, t + kd::kM, 0xFFFFFFFFu);
  for (const RowsCommit& c : commits) {
    require(c.slot < sessions() && c.n >= 1 && c.row0 + c.n <= kd::kM,
            "a rows commit of slot " + std::to_string(c.slot) + " outside the rows");
    for (uint32_t r = c.row0; r < c.row0 + c.n; ++r) t[r] = c.slot;
    t[kd::kM + 8 * c.slot] = c.row0;
    t[kd::kM + 8 * c.slot + 1] = c.n;
    t[kd::kM + 8 * c.slot + 2] = c.pos;
  }
  return rows_project_->list;
}

void DFlashDrafter::select(uint32_t slot) {
  require(slot < sessions(), "no session " + std::to_string(slot) + " (the drafter has " + std::to_string(sessions()) +
                                 ")");
  cur_ = slot;
}

void DFlashDrafter::attach(Engine& e, bool draft_vocab) {
  require(draft_ == nullptr, "attach() twice");
  require(e.tap_drafter() == this, "the engine was not built with this drafter (Engine(..., &drafter))");
  require(e.slots() == sessions(), "the engine has " + std::to_string(e.slots()) + " slots, the drafter " +
                                       std::to_string(sessions()) + " sessions");
  attach_target(e.context(), e.model(), e.buffers().control, e.buffers().max_len, draft_vocab);
  std::vector<const void*> ctrls{e.buffers().control.ptr()};
  for (uint32_t q = 1; q < e.slots(); ++q) {
    drafts_.push_back(build_dflash_draft(e.context(), w_, dflash_target(e.model(), draft_vocab), b_,
                                         e.slot_view(q).control, q));
    ctrls.push_back(e.slot_view(q).control);
  }
  if (e.slots() > 1)
    seqs_draft_ = std::make_unique<CapturedStep>(
        build_dflash_draft_seqs(e.context(), w_, dflash_target(e.model(), draft_vocab), b_, ctrls));
  engine_ = &e;
}

void DFlashDrafter::draft_rows(Engine& e, const std::vector<DraftRun>& runs) {
  require(engine_ == &e && seqs_draft_, "draft_rows() before attach() to an engine of several slots");
  namespace kd = kernels::dflash;
  uint32_t* t = b_.seqs->as<uint32_t>();
  std::fill(t, t + kd::kM + 8 * kd::kSeqs, 0u);
  std::fill(t, t + kd::kM, 0xFFFFFFFFu);
  uint32_t row = 0;
  for (const DraftRun& r : runs) {
    require(r.slot < sessions() && t[kd::kM + 8 * r.slot + 1] == 0, "a run's slot is outside the engine or twice");
    const Engine::SlotView v = e.slot_view(r.slot);
    require(r.k >= 1 && r.k <= max_k_ && r.k + 1 <= v.states && v.control->pos + r.k + 1 <= v.len &&
                row + r.k + 1 <= kd::kM,
            "slot " + std::to_string(r.slot) + ": k " + std::to_string(r.k) + " at pos " +
                std::to_string(v.control->pos) + " (rows so far " + std::to_string(row) + ")");
    DFlashCtl c{};   // the same rules and encoding as one slot's draft (set_draft_params)
    set_draft_params(c, w_.desc, r.k, r.temperature, r.seed);
    for (uint32_t i = 0; i <= r.k; ++i) t[row + i] = r.slot;
    uint32_t* f = t + kd::kM + 8 * r.slot;
    f[0] = row;
    f[1] = r.k + 1;
    f[2] = v.control->pos;
    f[3] = c.k;
    f[4] = v.control->cur_token[0];
    std::memcpy(&f[5], &c.temperature, 4);
    f[6] = c.seed_lo;
    f[7] = c.seed_hi;
    row += r.k + 1;
  }
  e.replay_aux(seqs_draft_->list);
  for (const DraftRun& r : runs)
    require(b_.session(r.slot).readback()->k == r.k, "slot " + std::to_string(r.slot) + "'s batched draft drafted " +
                                                         std::to_string(b_.session(r.slot).readback()->k) + " ids");
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
  require(e.selected() == cur_, "the drafter's session is not the engine's selected slot");
  require(k <= e.max_verify_k(),
          "k " + std::to_string(k) + " is outside 1.." + std::to_string(std::min(max_k_, e.max_verify_k())) +
              " at pos " + std::to_string(e.pos()));
  return draft_with([&e](l0::CmdList& list) { e.replay_aux(list); }, k, temperature, seed);
}

const DFlashDraftOut& DFlashDrafter::draft_with(const std::function<void(l0::CmdList&)>& replay, uint32_t k,
                                                float temperature, uint64_t seed) {
  require(draft_ != nullptr, "draft before attach");
  require(k >= 1 && k <= max_k_, "k " + std::to_string(k) + " is outside 1.." + std::to_string(max_k_));
  set_draft_params(*ses().ctl(), w_.desc, k, temperature, seed);
  replay(cur_ == 0 ? draft_->list : drafts_.at(cur_ - 1).list);
  const DFlashDraftOut& out = *ses().readback();
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
  DFlashCtl& c = *ses().ctl();
  c.proj_pos = pos;
  c.proj_n = n;
  return cur_ == 0 ? project_.list : projects_.at(cur_ - 1).list;
}

bool DFlashDrafter::project_wide(prefill::Context& cx, prefill::KernelCache& kc, PrefillScratch& s, const void* taps,
                                 uint32_t pos, uint32_t n) {
  return dflash::project_wide(cx, kc, s, w_, b_, taps, pos, n, cur_);
}

}  // namespace runtime::dflash
