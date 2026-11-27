#include "runtime/inkling2/inkling2_tp_engine.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "runtime/dflash/dflash_control.h"
#include "runtime/tp_protocol.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;

// The tensors both cards hold whole: card 1's copy is card 0's (the drafter's caches are each card's heads, its last
// tap row whole).
bool replicated(SnapTensor t) {
  return t == SnapTensor::ConvAttn || t == SnapTensor::ConvMlp || t == SnapTensor::DraftTap;
}
}  // namespace

TpEngine::TpEngine(l0::Context& card0, l0::Context& card1, const loader::Inkling2Checkpoint& checkpoint,
                   uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8, bool int8_pass, uint32_t timeout_ms,
                   uint32_t prefill_timeout_ms)
    : prefill_rows_(prefill_rows), timeout_ns_(uint64_t(timeout_ms) * 1000000),
      prefill_timeout_ns_(uint64_t(prefill_timeout_ms) * 1000000) {
  if (card1.handle() != card0.handle())
    throw std::invalid_argument("Inkling tp: card 1 is not a view of card 0's Level Zero context");
  if (!card0.can_access_peer(card1) || !card1.can_access_peer(card0))
    throw std::runtime_error("--parallel tp: the cards cannot write each other's memory (zeDeviceCanAccessPeer is "
                             "false), and every all-reduce does. Peer access needs the P2P-capable kernel and both "
                             "cards under one root complex (docs/01-hardware.md)");
  const std::array<l0::Context*, 2> ctx{&card0, &card1};
  for (uint32_t r = 0; r < 2; ++r) {
    split_[r] = std::make_unique<TpWeights>(tp_weights(checkpoint.desc(), checkpoint.layout(), r));
    cards_[r].reset(new Engine(*ctx[r], checkpoint, max_len, draft, prefill_rows, kv8, int8_pass, Stage{},
                               split_[r]->layout, Tp{r, 2}));
  }
  const std::array<std::pair<l0::Context*, Buffers*>, 2> both{{{&card0, &cards_[0]->buffers_},
                                                               {&card1, &cards_[1]->buffers_}}};
  const std::array<const TpWeights*, 2> parts{split_[0].get(), split_[1].get()};
  Buffers::upload(both, parts, checkpoint);
  for (uint32_t r = 0; r < 2; ++r) {
    cards_[r]->peer_landing_ = cards_[1 - r]->buffers_.tp.tp_landing->ptr();
    cards_[r]->spin_limit_ = tp::kDefaultSpinLimit;
  }
  for (auto& e : cards_) e->loaded();
  for (uint32_t r = 0; r < 2; ++r) fences_[r] = std::make_unique<l0::Fence>(cards_[r]->queue_);
}

TpEngine::~TpEngine() {   // never free buffers under a running list (a crossing's waits are bounded)
  for (auto& e : cards_) e->queue_.synchronize();
  for (auto& l : compute_)
    if (l) l->host_synchronize();
  for (auto& l : copy_)
    if (l) l->host_synchronize();
}

MemoryComponents TpEngine::memory(uint32_t i) const {
  const Engine& e = *cards_[i];
  return sizes(e.desc_, e.weights_, e.max_len_, e.buffers_.weights.draft != nullptr, e.prefill_rows_, e.kv8_,
               e.int8_pass_, e.stage_, e.tp_).memory();
}

void TpEngine::reset() {
  for (auto& e : cards_) e->reset();   // both idle: the crossing's counters and landings restart together
  prefill_logits_ = broken_ = false;
}

void TpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("Inkling tp: " + what + ". The session's state is not valid; reset first.");
}

void TpEngine::set_position(uint32_t committed, uint32_t pending) {
  for (auto& e : cards_) {
    e->committed_ = committed;
    e->pending_ = pending;
    e->ctrl_->pos = committed;
  }
}

void TpEngine::check_crossings(const char* what) {
  for (uint32_t r = 0; r < 2; ++r) {
    // the crossing's state words (shared memory, read after the fences)
    const auto* w = static_cast<const uint32_t*>(cards_[r]->buffers_.tp.tp_state->ptr());
    if (w[tp::kStatus] == tp::kOk) continue;
    fail(std::string(what) + ": card " + std::to_string(r) + "'s all-reduce " +
         (w[tp::kStatus] == tp::kTimeout ? "waited out its bound" : "read a chunk without its data") + " (chunk " +
         std::to_string(w[tp::kChunk]) + ", call " + std::to_string(w[tp::kWant]) + ", flag " +
         std::to_string(w[tp::kFlag]) + ", " + std::to_string(w[tp::kFailed]) + " chunks failed)");
  }
}

void TpEngine::run(std::array<l0::CmdList*, 2> lists, const char* what) {
  if (broken_) throw std::runtime_error("Inkling tp: an all-reduce failed earlier in this session; reset first");
  for (uint32_t r = 0; r < 2; ++r) cards_[r]->queue_.execute(*lists[r], fences_[r].get());
  for (uint32_t r = 0; r < 2; ++r)
    if (!fences_[r]->wait_for(timeout_ns_))
      fail(std::string(what) + " on card " + std::to_string(r) + " did not finish within " +
           std::to_string(timeout_ns_ / 1000000) + " ms");
  check_crossings(what);
}

std::span<const uint32_t> TpEngine::verify(std::span<const uint32_t> ids) {
  if (pending()) throw std::logic_error("Commit pending Inkling rows before verifying again");
  if (ids.empty() || ids.size() > kRows || ids.size() > max_len() - position())
    throw std::invalid_argument("Inkling input rows exceed the context or verify width");
  for (const auto id : ids)
    if (id >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  const auto m = uint32_t(ids.size());
  for (uint32_t r = 0; r < 2; ++r) {
    Engine& e = *cards_[r];
    auto& list = target_[r][m - 1];
    if (list) continue;
    list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
    const auto first = e.kernels_.size();
    e.record_target(*list, m);
    if (m == 1) e.record_commit(*list, m, 1);
    e.target_launches_[m - 1] = e.kernels_.size() - first;
    list->close();
  }
  const uint32_t pos = position();
  for (auto& e : cards_) {
    std::copy(ids.begin(), ids.end(), e->ctrl_->cur_token);
    e->ctrl_->pos = pos;
    e->ctrl_->n_active = m;
    e->written_ = std::max(e->written_, pos + m);
    if (m == 1) e->set_commit_control(1);
  }
  run({target_[0][m - 1].get(), target_[1][m - 1].get()}, "a verify step");
  if (!std::equal(lead().ctrl_->out_token, lead().ctrl_->out_token + m, cards_[1]->ctrl_->out_token))
    fail("the two cards' predictions differ after a verify step");
  set_position(pos, m);
  prefill_logits_ = false;
  return {lead().ctrl_->out_token, m};
}

void TpEngine::commit(uint32_t rows) {
  const uint32_t m = pending();
  if (rows == 0 || rows > m) throw std::invalid_argument("Inkling commit must consume a prefix of the pending rows");
  if (m > 1) {
    for (uint32_t r = 0; r < 2; ++r) {
      Engine& e = *cards_[r];
      auto& list = commits_[r][m - 1][rows - 1];
      if (list) continue;
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
      const auto first = e.kernels_.size();
      e.record_commit(*list, m, rows);
      e.commit_launches_[m - 1][rows - 1] = e.kernels_.size() - first;
      list->close();
    }
    for (auto& e : cards_) e->set_commit_control(rows);
    run({commits_[0][m - 1][rows - 1].get(), commits_[1][m - 1][rows - 1].get()}, "a commit");
  }
  set_position(position() + rows, 0);
}

void TpEngine::read_logits(uint32_t row, std::span<float> out) {
  if (row >= pending() || out.size() != D::vocab_used)
    throw std::invalid_argument("Inkling logit read requires a pending row and the live vocabulary");
  lead().copy_logits(row, out);
}

void TpEngine::read_prefill_logits(std::span<float> out) { lead().copy_logits(0, out); }

// Both cards run the draft pass, its all-reduces crossing.
std::span<const uint32_t> TpEngine::propose(uint32_t anchor, uint32_t count, float temperature, uint64_t seed) {
  Engine& e0 = lead();
  if (!e0.buffers_.weights.draft || pending() || !position() || anchor >= D::vocab || count >= kRows ||
      count + 1 > max_len() - position())
    throw std::invalid_argument("Inkling draft is outside the committed context or the draft width");
  for (auto& e : cards_) {
    if (!e->draft_list_) {
      e->draft_list_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(e->ctx_));
      e->record_draft(*e->draft_list_);
      e->draft_list_->close();
    }
    e->ctrl_->cur_token[0] = anchor;
    e->ctrl_->pos = position();
    auto* dctl = static_cast<dflash::DFlashCtl*>(e->buffers_.draft.draft_control->ptr());
    dctl->k = kRows - 1;
    dctl->temperature = temperature;
    dctl->seed_lo = uint32_t(seed);
    dctl->seed_hi = uint32_t(seed >> 32);
  }
  run({e0.draft_list_.get(), cards_[1]->draft_list_.get()}, "a draft pass");
  if (!std::equal(e0.ctrl_->cur_token + 1, e0.ctrl_->cur_token + 1 + count, cards_[1]->ctrl_->cur_token + 1))
    fail("the two cards' drafts differ after a draft pass");
  return {e0.ctrl_->cur_token + 1, count};
}

void TpEngine::profile_target(std::span<const uint32_t> ids) {
  // card 1's step beside card 0's profiled one, as profile_draft runs them
  Engine& e1 = *cards_[1];
  const auto m = uint32_t(ids.size());
  auto list = l0::CmdList::regular(e1.ctx_);
  e1.record_target(list, m, true);
  list.close();
  std::copy(ids.begin(), ids.end(), e1.ctrl_->cur_token);
  e1.ctrl_->pos = position();
  e1.ctrl_->n_active = m;
  e1.queue_.execute(list, fences_[1].get());
  lead().profile_target(ids);
  if (!fences_[1]->wait_for(timeout_ns_)) fail("card 1's profiled target step did not finish");
  check_crossings("a profiled target step");
}

void TpEngine::profile_draft(uint32_t anchor) {
  Engine& e0 = lead();
  // card 1's pass beside card 0's profiled one (their all-reduces meet), then card 1 joined and the crossings read
  Engine& e1 = *cards_[1];
  if (!e1.draft_list_) {
    e1.draft_list_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(e1.ctx_));
    e1.record_draft(*e1.draft_list_);
    e1.draft_list_->close();
  }
  e1.ctrl_->cur_token[0] = anchor;
  e1.ctrl_->pos = position();
  static_cast<dflash::DFlashCtl*>(e1.buffers_.draft.draft_control->ptr())->k = kRows - 1;
  e1.queue_.execute(*e1.draft_list_, fences_[1].get());
  e0.profile_draft(anchor);
  if (!fences_[1]->wait_for(timeout_ns_)) fail("card 1's draft pass did not finish");
  check_crossings("a profiled draft pass");
}

uint32_t TpEngine::step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output) {
  Engine& e0 = lead();
  if (max_drafts > 0 && max_len() - position() > 1 && e0.buffers_.weights.draft && position()) {
    if (pending()) throw std::logic_error("Commit pending Inkling rows before a drafted step");
    if (anchor >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
    // Engine::step_drafted's one submission a card: the draft, the verify, the commit by ink_accept's count.
    const uint32_t k = std::min({max_drafts, kRows - 1, max_len() - position() - 1});
    for (uint32_t r = 0; r < 2; ++r) {
      Engine& e = *cards_[r];
      auto& list = rounds_[r][k - 1];
      if (list) continue;
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
      e.record_step_inputs(*list);
      e.record_draft(*list);
      e.record_target(*list, k + 1, true);
      e.record_commit(*list, k + 1, 0, true);
      e.record_step_outputs(*list, true);
      list->close();
    }
    for (auto& e : cards_) {
      auto* io = e->step_io();
      io->cur_token[0] = anchor;
      io->pos = io->proj_pos = position();
      io->n_active = k + 1;
      io->k = kRows - 1;
      io->proj_n = 0;
      e->written_ = std::max(e->written_, position() + k + 1);
    }
    run({rounds_[0][k - 1].get(), rounds_[1][k - 1].get()}, "a drafted step");
    const auto* io = e0.step_io();
    const auto* io1 = cards_[1]->step_io();
    const uint32_t accepted = io->accepted;
    if (accepted != io1->accepted || !std::equal(io->drafts + 1, io->drafts + accepted, io1->drafts + 1) ||
        !std::equal(io->out_token, io->out_token + accepted, io1->out_token))
      fail("the two cards' drafts or predictions differ after a drafted step");
    uint32_t host_accepted = 1;
    while (host_accepted <= k && io->drafts[host_accepted] == io->out_token[host_accepted - 1]) ++host_accepted;
    contract_assert(accepted == host_accepted);
    // drafts[0] is not the anchor by now: argmax_stage2 wrote the last row's prediction there (Engine::step_drafted's)
    output.push_back(anchor);
    output.insert(output.end(), io->drafts + 1, io->drafts + accepted);
    set_position(position() + accepted, 0);
    prefill_logits_ = false;
    return io->out_token[accepted - 1];
  }
  output.push_back(anchor);
  return append(anchor);
}

uint32_t TpEngine::append(uint32_t token) {
  if (pending()) throw std::logic_error("Commit pending Inkling rows before appending");
  if (position() >= max_len()) throw std::invalid_argument("Inkling input rows exceed the context");
  if (token >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  for (uint32_t r = 0; r < 2; ++r) {
    Engine& e = *cards_[r];
    auto& list = plain_[r];
    if (list) continue;
    list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
    e.record_step_inputs(*list);
    e.record_target(*list, 1, true);
    e.record_commit(*list, 1, 1);
    e.record_step_outputs(*list, false);
    list->close();
  }
  for (auto& e : cards_) {
    auto* io = e->step_io();
    io->pos = io->proj_pos = position();
    io->n_active = io->proj_n = 1;
    io->cur_token[0] = token;
    io->k = kRows - 1;
    e->written_ = std::max(e->written_, position() + 1);
  }
  run({plain_[0].get(), plain_[1].get()}, "a decode step");
  const uint32_t next = lead().step_io()->out_token[0];
  if (next != cards_[1]->step_io()->out_token[0]) fail("the two cards' predictions differ after a decode step");
  set_position(position() + 1, 0);
  prefill_logits_ = false;
  return next;
}

template <class F>
void TpEngine::in_host_order(const std::array<std::vector<Engine::SnapRun>, 2>& runs, bool both, F f) const {
  for (uint32_t t = 0; t <= uint32_t(SnapTensor::DraftTap); ++t) {
    const auto tensor = SnapTensor(t);
    std::array<const l0::Mem*, 2> mem{cards_[0]->snapshot_buffer(tensor), cards_[1]->snapshot_buffer(tensor)};
    if (!mem[0]) continue;
    std::vector<const Engine::SnapRun*> own;   // card 0's runs of the tensor
    for (const auto& r : runs[0])
      if (r.mem == mem[0]) own.push_back(&r);
    if (replicated(tensor)) {
      // the same runs on card 1 (the same buffer sizes): card 0's bytes, written to both on a load
      std::vector<const Engine::SnapRun*> peer;
      for (const auto& r : runs[1])
        if (r.mem == mem[1]) peer.push_back(&r);
      contract_assert(peer.size() == own.size());
      for (size_t i = 0; i < own.size(); ++i) {
        f(0u, tensor, *own[i], false);
        if (both) f(1u, tensor, *peer[i], true);
      }
      continue;
    }
    for (const auto* r : own) f(0u, tensor, *r, false);
    for (const auto& r : runs[1])
      if (r.mem == mem[1]) f(1u, tensor, r, false);
  }
}

size_t TpEngine::state_bytes() const {
  size_t n = 0;
  in_host_order({lead().state_runs(0), cards_[1]->state_runs(0)}, false,
                [&](uint32_t, SnapTensor, const Engine::SnapRun& r, bool) { n += r.bytes; });
  return n;
}

size_t TpEngine::kv_bytes(uint32_t positions) const {
  size_t n = 0;
  in_host_order({lead().kv_runs(0, positions), cards_[1]->kv_runs(0, positions)}, false,
                [&](uint32_t, SnapTensor, const Engine::SnapRun& r, bool) { n += r.bytes; });
  return n;
}

void TpEngine::save_state(void* host) {
  for (auto& e : cards_) e->queue_.synchronize();
  // The copy engines, as Engine::save_state's (the rings' device-to-host runs at ~2/3 of the compute engine's time).
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_copy(cards_[0]->ctx_),
                                   l0::CmdList::immediate_copy(cards_[1]->ctx_)};
  auto* h = static_cast<std::byte*>(host);
  const uint32_t pos = shadow_ >= 0 ? shadow_pos_ : position();
  in_host_order({lead().state_runs(pos), cards_[1]->state_runs(pos)}, false,
                [&](uint32_t c, SnapTensor t, const Engine::SnapRun& r, bool) {
                  const Engine& e = *cards_[c];
                  const std::byte* from = static_cast<const std::byte*>(r.mem->ptr()) + r.offset;
                  if (shadow_ >= 0) {   // inside a mid-prompt hook: the block end's image (record_visit_shadow)
                    const auto& shadow = *e.buffers_.handoff.snapshot_shadow;
                    from = static_cast<const std::byte*>(shadow.ptr()) + shadow_ * (shadow.size() / 2) +
                           e.shadow_image(t) + r.offset;
                  }
                  if (r.zero) std::memset(h, 0, r.bytes);   // positions before 0: never read
                  else lists[c].copy(h, from, r.bytes);
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
}

void TpEngine::load_state(const void* host, uint32_t pos) {
  for (auto& e : cards_) e->queue_.synchronize();
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards_[0]->ctx_),
                                   l0::CmdList::immediate_async(cards_[1]->ctx_)};
  const auto* h = static_cast<const std::byte*>(host);
  in_host_order({lead().state_runs(pos), cards_[1]->state_runs(pos)}, true,
                [&](uint32_t c, SnapTensor, const Engine::SnapRun& r, bool again) {
                  if (again) h -= r.bytes;   // card 0's bytes of a replicated run, once more for card 1
                  lists[c].copy(static_cast<std::byte*>(r.mem->ptr()) + r.offset, h, r.bytes);
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
  set_position(pos, 0);
  for (auto& e : cards_) {
    e->ctrl_->n_active = 0;
    e->written_ = std::max(e->written_, pos);
  }
  prefill_logits_ = false;
}

void TpEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  // Inside a mid-prompt hook the positions are final: the copies run on each card's copy engine beside the next
  // chunks (PERF-89, Engine::kv_store_) and are joined at the prompt's end.
  std::array<l0::CmdList*, 2> lists{};
  std::array<std::optional<l0::CmdList>, 2> own;
  for (uint32_t c = 0; c < 2; ++c) {
    Engine& e = *cards_[c];
    if (overlap_kv_) {
      if (!e.kv_store_) e.kv_store_ = std::make_unique<l0::CmdList>(l0::CmdList::immediate_copy(e.ctx_));
      lists[c] = e.kv_store_.get();
    } else {
      e.queue_.synchronize();
      lists[c] = &own[c].emplace(l0::CmdList::immediate_async(e.ctx_));
    }
  }
  auto* h = static_cast<std::byte*>(host);
  in_host_order({lead().kv_runs(begin, end), cards_[1]->kv_runs(begin, end)}, false,
                [&](uint32_t c, SnapTensor, const Engine::SnapRun& r, bool) {
                  lists[c]->copy(h, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
                  h += r.bytes;
                });
  if (!overlap_kv_)
    for (auto& l : lists) l->host_synchronize();
}

void TpEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  for (auto& e : cards_) e->queue_.synchronize();
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards_[0]->ctx_),
                                   l0::CmdList::immediate_async(cards_[1]->ctx_)};
  const auto* h = static_cast<const std::byte*>(host);
  in_host_order({lead().kv_runs(begin, end), cards_[1]->kv_runs(begin, end)}, true,
                [&](uint32_t c, SnapTensor, const Engine::SnapRun& r, bool again) {
                  if (again) h -= r.bytes;
                  lists[c].copy(static_cast<std::byte*>(r.mem->ptr()) + r.offset, h, r.bytes);
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
  for (auto& e : cards_) e->written_ = std::max(e->written_, end);
}

}  // namespace runtime::inkling2
