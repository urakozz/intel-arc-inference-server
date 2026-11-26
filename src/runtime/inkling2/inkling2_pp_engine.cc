#include "runtime/inkling2/inkling2_pp_engine.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "runtime/dflash/dflash_control.h"
#include "runtime/prefill_chunks.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
}  // namespace

PpEngine::PpEngine(l0::Context& card0, l0::Context& card1, const loader::Inkling2Checkpoint& checkpoint,
                   uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8, bool int8_pass, uint32_t timeout_ms,
                   uint32_t prefill_timeout_ms)
    : prefill_rows_(prefill_rows), timeout_ns_(uint64_t(timeout_ms) * 1000000),
      prefill_timeout_ns_(uint64_t(prefill_timeout_ms) * 1000000) {
  if (card1.handle() != card0.handle())
    throw std::invalid_argument("Inkling pp: card 1 is not a view of card 0's Level Zero context");
  if (!card0.can_access_peer(card1))
    throw std::runtime_error("--parallel pp: card 0 (" + card0.name() + ") cannot access card 1's memory "
                             "(zeDeviceCanAccessPeer is false), and the hand-off writes it. Peer access needs the "
                             "P2P-capable kernel and both cards under one root complex (docs/01-hardware.md)");
  const auto stages = pp_stages(draft);
  cards_[0].reset(new Engine(card0, checkpoint, max_len, false, prefill_rows, kv8, int8_pass, stages[0],
                             checkpoint.layout()));
  cards_[1].reset(new Engine(card1, checkpoint, max_len, draft, prefill_rows, kv8, int8_pass, stages[1],
                             checkpoint.layout()));
  const std::array<std::pair<l0::Context*, Buffers*>, 2> both{{{&card0, &cards_[0]->buffers_},
                                                               {&card1, &cards_[1]->buffers_}}};
  Buffers::upload(both, checkpoint);
  for (auto& e : cards_) e->loaded();
  for (uint32_t i = 0; i < 2; ++i) fences_[i] = std::make_unique<l0::Fence>(cards_[i]->queue_);
  const std::vector<const l0::Context*> views{&card0, &card1};
  handoff_ = std::make_unique<l0::SyncEvent>(card0, views);
  drafted_ = std::make_unique<l0::SyncEvent>(card0, views);
  accepted_ = std::make_unique<l0::SyncEvent>(card0, views);
  if (prefill_rows)
    prompt_lists_ = {std::make_unique<l0::CmdList>(l0::CmdList::immediate_async(card0)),
                     std::make_unique<l0::CmdList>(l0::CmdList::immediate_copy(card0)),
                     std::make_unique<l0::CmdList>(l0::CmdList::immediate_async(card1))};
}

PpEngine::~PpEngine() {   // never free buffers under a running list
  for (auto* e : {handoff_.get(), drafted_.get(), accepted_.get()}) e->host_signal();
  for (auto& e : cards_) e->queue_.synchronize();
}

MemoryComponents PpEngine::memory(uint32_t i) const {
  const Engine& e = *cards_[i];
  return sizes(e.desc_, e.weights_, e.max_len_, e.buffers_.weights.draft != nullptr, e.prefill_rows_, e.kv8_,
               e.int8_pass_, e.stage_).memory();
}

void PpEngine::reset() {
  for (auto& e : cards_) e->reset();
  for (auto* e : {handoff_.get(), drafted_.get(), accepted_.get()}) e->host_reset();
  prefill_logits_ = broken_ = false;
}

void PpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("Inkling pp hand-off: " + what + ". The session's state is not valid; reset first.");
}

void PpEngine::set_position(uint32_t committed, uint32_t pending) {
  for (auto& e : cards_) {
    e->committed_ = committed;
    e->pending_ = pending;
    e->ctrl_->pos = committed;
  }
}

void PpEngine::pack(l0::CmdList& list, uint32_t card, uint32_t rows, bool prompt, std::byte* buffer, bool out) {
  // Adjacent rows (the two blocks, residuals 3 and 4) go as one copy.
  const auto regions = cards_[card]->handoff_rows(rows, prompt);
  for (size_t i = 0; i < regions.size();) {
    auto* ptr = static_cast<std::byte*>(regions[i].first);
    size_t bytes = 0;
    for (; i < regions.size() && regions[i].second && regions[i].first == ptr + bytes; ++i) bytes += regions[i].second;
    if (!bytes) {
      ++i;
      continue;
    }
    if (out) list.copy(buffer, ptr, bytes);
    else list.copy(ptr, buffer, bytes);
    buffer += bytes;
  }
}

void PpEngine::run(l0::CmdList& list0, l0::CmdList& list1, const char* what) {
  if (broken_) throw std::runtime_error("Inkling pp: a hand-off failed earlier in this session; reset first");
  const auto events = {handoff_.get(), drafted_.get(), accepted_.get()};
  for (auto* e : events) e->host_reset();
  cards_[0]->queue_.execute(list0, fences_[0].get());
  cards_[1]->queue_.execute(list1, fences_[1].get());
  if (!fences_[1]->wait_for(timeout_ns_)) {
    // Card 1 still waits (on rows card 0 never sent, or a hung list): release both cards so their queues drain.
    for (auto* e : events) e->host_signal();
    const bool drained = fences_[1]->wait_for(timeout_ns_);
    const bool first = fences_[0]->wait_for(timeout_ns_);
    fail(std::string(what) + " on card 1 did not finish within " + std::to_string(timeout_ns_ / 1000000) + " ms (" +
         (first ? "card 0 finished but its rows never released card 1" : "card 0 did not finish either") +
         "; card 1 " + (drained ? "released" : "did not drain") + ")");
  }
  if (!fences_[0]->wait_for(timeout_ns_)) {
    for (auto* e : events) e->host_signal();
    fences_[0]->wait_for(timeout_ns_);
    fail(std::string(what) + " on card 0 did not finish within " + std::to_string(timeout_ns_ / 1000000) + " ms");
  }
}

std::span<const uint32_t> PpEngine::verify(std::span<const uint32_t> ids, bool candidates) {
  if (pending()) throw std::logic_error("Commit pending Inkling rows before verifying again");
  if (ids.empty() || ids.size() > kRows || ids.size() > max_len() - position())
    throw std::invalid_argument("Inkling input rows exceed the context or verify width");
  for (const auto id : ids)
    if (id >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  const auto m = uint32_t(ids.size());
  auto* landing = static_cast<std::byte*>(head().buffers_.handoff.landing->ptr());
  const bool deep = Engine::deep_verify(m, position());
  auto& targets = target_[candidates][deep];
  if (!targets[0][m - 1]) {
    size_t launches = 0;
    for (uint32_t i = 0; i < 2; ++i) {
      Engine& e = *cards_[i];
      auto& list = targets[i][m - 1];
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
      const auto first = e.kernels_.size();
      if (i == 1) {
        list->wait_event(handoff_->handle());
        pack(*list, 1, m, false, landing, false);
      }
      e.record_target(*list, m, candidates, deep);
      if (i == 0) {
        pack(*list, 0, m, false, landing, true);
        list->barrier_signal(handoff_->handle());
      }
      if (m == 1) e.record_commit(*list, m, 1);
      launches += e.kernels_.size() - first;
      list->close();
    }
    if (!deep && !candidates) target_launches_[m - 1] = launches;
  }
  const uint32_t pos = position();
  for (auto& e : cards_) {
    std::copy(ids.begin(), ids.end(), e->ctrl_->cur_token);
    e->ctrl_->pos = pos;
    e->ctrl_->n_active = m;
  }
  for (auto& e : cards_) e->written_ = std::max(e->written_, pos + m);
  if (m == 1) head().set_commit_control(1);
  run(*targets[0][m - 1], *targets[1][m - 1], "a verify step");
  set_position(pos, m);
  prefill_logits_ = false;
  return {head().ctrl_->out_token, m};
}

void PpEngine::commit(uint32_t rows) {
  const uint32_t m = pending();
  if (rows == 0 || rows > m) throw std::invalid_argument("Inkling commit must consume a prefix of the pending rows");
  if (m > 1) {
    if (!commits_[0][m - 1][rows - 1]) {
      size_t launches = 0;
      for (uint32_t i = 0; i < 2; ++i) {
        Engine& e = *cards_[i];
        auto& list = commits_[i][m - 1][rows - 1];
        list = std::make_unique<l0::CmdList>(l0::CmdList::regular(e.ctx_));
        const auto first = e.kernels_.size();
        e.record_commit(*list, m, rows);
        launches += e.kernels_.size() - first;
        list->close();
      }
      commit_launches_[m - 1][rows - 1] = launches;
    }
    head().set_commit_control(rows);
    for (uint32_t i = 0; i < 2; ++i) cards_[i]->queue_.execute(*commits_[i][m - 1][rows - 1], fences_[i].get());
    for (uint32_t i = 0; i < 2; ++i)
      if (!fences_[i]->wait_for(timeout_ns_)) fail("a commit on card " + std::to_string(i) + " did not finish");
  }
  set_position(position() + rows, 0);
}

size_t PpEngine::launch_count(uint32_t rows, uint32_t accepted) const {
  return target_launches_[rows - 1] + commit_launches_[rows - 1][accepted - 1];
}

void PpEngine::read_logits(uint32_t row, std::span<float> out) {
  if (row >= pending() || out.size() != D::vocab_used)
    throw std::invalid_argument("Inkling logit read requires a pending row and the live vocabulary");
  head().copy_logits(row, out);
}

void PpEngine::read_prefill_logits(std::span<float> out) { head().copy_logits(0, out); }

std::span<const uint32_t> PpEngine::propose(uint32_t anchor, uint32_t count) { return head().propose(anchor, count); }

uint32_t PpEngine::step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output) {
  const uint32_t room = max_len() - position();
  if (max_drafts > 0 && room > 1 && head().buffers_.weights.draft && position()) {
    if (pending()) throw std::logic_error("Commit pending Inkling rows before a drafted step");
    if (anchor >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
    const uint32_t k = std::min({max_drafts, kRows - 1, room - 1}), m = k + 1, pos = position();
    Engine& e0 = *cards_[0];
    Engine& e1 = head();
    auto* io0 = e0.step_io();
    auto* io1 = e1.step_io();
    const bool deep = Engine::deep_verify(m, pos);
    auto& lists = rounds_[deep];
    if (!lists[0][k - 1]) {
      auto* landing = static_cast<std::byte*>(e1.buffers_.handoff.landing->ptr());
      auto& list0 = *(lists[0][k - 1] = std::make_unique<l0::CmdList>(l0::CmdList::regular(e0.ctx_)));
      list0.wait_event(drafted_->handle());
      e0.record_step_inputs(list0);
      e0.record_target(list0, m, true, deep);
      pack(list0, 0, m, false, landing, true);
      list0.barrier_signal(handoff_->handle());
      list0.wait_event(accepted_->handle());
      e0.record_commit(list0, m, 0, false, &io0->accepted);
      list0.close();
      auto& list1 = *(lists[1][k - 1] = std::make_unique<l0::CmdList>(l0::CmdList::regular(e1.ctx_)));
      e1.record_step_inputs(list1);
      e1.record_draft(list1);
      list1.copy(io0->cur_token, e1.ctrl_->cur_token, sizeof(io0->cur_token));
      list1.barrier_signal(drafted_->handle());
      list1.wait_event(handoff_->handle());
      pack(list1, 1, m, false, landing, false);
      e1.record_target(list1, m, true, deep);
      e1.record_commit(list1, m, 0, true);
      list1.copy(&io0->accepted, &static_cast<dflash::DFlashCtl*>(e1.buffers_.draft.draft_control->ptr())->proj_n,
                 sizeof(uint32_t));
      list1.barrier_signal(accepted_->handle());
      e1.record_step_outputs(list1, true);
      list1.close();
    }
    io0->pos = pos;
    io0->n_active = m;
    io1->cur_token[0] = anchor;
    io1->pos = io1->proj_pos = pos;
    io1->n_active = m;
    io1->k = kRows - 1;
    io1->proj_n = 0;
    for (auto& e : cards_) e->written_ = std::max(e->written_, pos + m);
    run(*lists[0][k - 1], *lists[1][k - 1], "a drafted round");
    const uint32_t accepted = io1->accepted;
    uint32_t host_accepted = 1;
    while (host_accepted <= k && io1->drafts[host_accepted] == io1->out_token[host_accepted - 1]) ++host_accepted;
    contract_assert(accepted == host_accepted);
    // drafts[0] is not the anchor by now: argmax_stage2 wrote the last row's prediction there (the next step's input).
    output.push_back(anchor);
    output.insert(output.end(), io1->drafts + 1, io1->drafts + accepted);
    for (auto& e : cards_) e->committed_ = pos + accepted;
    prefill_logits_ = false;
    return io1->out_token[accepted - 1];
  }
  output.push_back(anchor);
  return append(anchor);
}

uint32_t PpEngine::append(uint32_t token) {
  const uint32_t next = verify({&token, 1}, true)[0];
  commit(1);
  return next;
}

template <class F>
void PpEngine::in_host_order(const std::array<Engine*, 2>& cards,
                             const std::array<std::vector<Engine::SnapRun>, 2>& runs, F f) {
  std::array<std::vector<size_t>, 2> offset;
  for (uint32_t c = 0; c < 2; ++c) {
    size_t at = 0;
    for (const auto& r : runs[c]) {
      offset[c].push_back(at);
      at += r.bytes;
    }
  }
  for (uint32_t t = 0; t <= uint32_t(SnapTensor::DraftTap); ++t)
    for (uint32_t c = 0; c < 2; ++c) {
      const l0::Mem* mem = cards[c]->snapshot_buffer(SnapTensor(t));
      for (size_t i = 0; i < runs[c].size(); ++i)
        if (mem && runs[c][i].mem == mem) f(c, runs[c][i], offset[c][i]);
    }
}

void PpEngine::save_state(void* host) {
  const std::array<Engine*, 2> cards{cards_[0].get(), cards_[1].get()};
  const uint32_t pos = shadow_ >= 0 ? shadow_pos_ : position();
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards[0]->ctx_),
                                   l0::CmdList::immediate_async(cards[1]->ctx_)};
  auto* h = static_cast<std::byte*>(host);
  in_host_order(cards, {cards[0]->state_runs(pos), cards[1]->state_runs(pos)},
                [&](uint32_t c, const Engine::SnapRun& r, size_t own) {
                  if (r.zero) {
                    std::memset(h, 0, r.bytes);
                  } else if (shadow_ >= 0) {
                    const auto& shadow = *cards[c]->buffers_.handoff.snapshot_shadow;
                    lists[c].copy(h, static_cast<const std::byte*>(shadow.ptr()) + shadow_ * shadow.size() / 2 + own,
                                  r.bytes);
                  } else {
                    lists[c].copy(h, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
                  }
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
}

void PpEngine::load_state(const void* host, uint32_t pos) {
  const std::array<Engine*, 2> cards{cards_[0].get(), cards_[1].get()};
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards[0]->ctx_),
                                   l0::CmdList::immediate_async(cards[1]->ctx_)};
  const auto* h = static_cast<const std::byte*>(host);
  in_host_order(cards, {cards[0]->state_runs(pos), cards[1]->state_runs(pos)},
                [&](uint32_t c, const Engine::SnapRun& r, size_t) {
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

void PpEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  const std::array<Engine*, 2> cards{cards_[0].get(), cards_[1].get()};
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards[0]->ctx_),
                                   l0::CmdList::immediate_async(cards[1]->ctx_)};
  auto* h = static_cast<std::byte*>(host);
  in_host_order(cards, {cards[0]->kv_runs(begin, end), cards[1]->kv_runs(begin, end)},
                [&](uint32_t c, const Engine::SnapRun& r, size_t) {
                  lists[c].copy(h, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
}

void PpEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  const std::array<Engine*, 2> cards{cards_[0].get(), cards_[1].get()};
  std::array<l0::CmdList, 2> lists{l0::CmdList::immediate_async(cards[0]->ctx_),
                                   l0::CmdList::immediate_async(cards[1]->ctx_)};
  const auto* h = static_cast<const std::byte*>(host);
  in_host_order(cards, {cards[0]->kv_runs(begin, end), cards[1]->kv_runs(begin, end)},
                [&](uint32_t c, const Engine::SnapRun& r, size_t) {
                  lists[c].copy(static_cast<std::byte*>(r.mem->ptr()) + r.offset, h, r.bytes);
                  h += r.bytes;
                });
  for (auto& l : lists) l.host_synchronize();
  for (auto& e : cards_) e->written_ = std::max(e->written_, end);
}

uint32_t PpEngine::prefill(std::span<const uint32_t> ids, uint32_t cap) {
  if (broken_) throw std::runtime_error("Inkling pp: a hand-off failed earlier in this session; reset first");
  Engine& e0 = *cards_[0];
  Engine& e1 = head();
  for (const auto id : ids) contract_assert(id < D::vocab);
  auto* staging = static_cast<uint32_t*>(e0.buffers_.prefill.ids->ptr());
  std::copy(ids.begin(), ids.end(), staging);
  const uint32_t base = position(), end = base + uint32_t(ids.size());
  for (auto& e : cards_) {
    e->written_ = std::max(e->written_, end);
    e->prefill_end_ = end;   // the DFlash 2 drafter's in-window rows (card 1)
    e->ctrl_->pos = end - 1;
    e->ctrl_->n_active = 1;
  }
  const std::array<size_t, 2> first{e0.kernels_.size(), e1.kernels_.size()};
  auto& cx0 = *prompt_lists_[0];
  auto& copy = *prompt_lists_[1];
  auto& cx1 = *prompt_lists_[2];
  auto* outbox = static_cast<std::byte*>(e0.buffers_.handoff.prefill_outbox->ptr());
  auto* landing = static_cast<std::byte*>(e1.buffers_.handoff.prefill_landing->ptr());
  const size_t row_bytes = handoff_row_bytes(e0.buffers_.prefill.draft_taps != nullptr);
  // With the prefix cache's hook every chunk ends at a snapshot point or the prompt's end (Engine::prefill's rule).
  std::vector<PrefillChunk> chunks;
  if (block_hook_) {
    for (uint32_t pos = base; pos < end;) {
      const uint32_t rows = prefill_chunk_rows(pos, end - pos, cap, true, kBlock, /*wide_stride=*/true);
      chunks.push_back({pos, rows, e0.desc_.visits});
      pos += rows;
    }
  } else {
    chunks = prefill_chunks(e0.desc_, base, uint32_t(ids.size()), cap);
  }
  std::vector<uint32_t> hooked;   // the mid-prompt block ends
  for (const auto& c : chunks)
    if (block_hook_ && c.position + c.rows < end && (c.position + c.rows) % kBlock == 0) hooked.push_back(c.position + c.rows);
  // Events: ready / sent / taken a chunk, then card 0's and card 1's `shadowed` a hook.
  const auto n = uint32_t(chunks.size()), events = 3 * n + 2 * uint32_t(hooked.size());
  const uint64_t bound = prefill_timeout_ns_ * n;   // the host waits for up to the whole prompt, a bound a chunk
  if (!chunk_events_ || chunk_events_->size() < events)
    chunk_events_ = std::make_unique<l0::SyncEventPool>(e0.ctx_, std::vector<const l0::Context*>{&e0.ctx_, &e1.ctx_},
                                                        events);
  auto& pool = *chunk_events_;
  for (uint32_t i = 0; i < events; ++i) pool.host_reset(i);
  const auto ready = [&](uint32_t j) { return pool.handle(3 * j); };
  const auto sent = [&](uint32_t j) { return pool.handle(3 * j + 1); };
  const auto taken = [&](uint32_t j) { return pool.handle(3 * j + 2); };
  const auto shadowed = [&](uint32_t h, uint32_t c) { return 3 * n + 2 * h + c; };
  // Card c's state at block end h into shadow h % 2. A list never waits on an event the host signals (the driver may
  // resolve such a wait on the host while appending): the host appends block end h's copies only after it stored
  // block end h - 2 out of that shadow.
  const auto shadow = [&](l0::CmdList& list, uint32_t c, uint32_t h) {
    Engine& e = *cards_[c];
    const auto& mem = *e.buffers_.handoff.snapshot_shadow;
    auto* to = static_cast<std::byte*>(mem.ptr()) + (h % 2) * (mem.size() / 2);
    for (const auto& r : e.state_runs(hooked[h])) {
      if (!r.zero) list.copy(to, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
      to += r.bytes;
    }
    contract_assert(to <= static_cast<std::byte*>(mem.ptr()) + (h % 2 + 1) * (mem.size() / 2));
    list.barrier_signal(pool.handle(shadowed(h, c)));
  };
  const auto drain = [&] {   // release every list waiting on an event that will not come
    for (uint32_t i = 0; i < events; ++i) pool.host_signal(i);
    cx0.host_synchronize(prefill_timeout_ns_);
    copy.host_synchronize(prefill_timeout_ns_);
    cx1.host_synchronize(prefill_timeout_ns_);
  };
  // Mid-prompt hook h, while the next chunks run: save_state reads the shadow, save_kv the final rows.
  uint32_t stored = 0;   // the hooks called
  const auto store = [&] {
    const uint32_t h = stored++;
    if (!pool.host_wait(shadowed(h, 0), bound) || !pool.host_wait(shadowed(h, 1), bound)) {
      drain();
      fail("the prompt chunk ending at " + std::to_string(hooked[h]) + " did not finish");
    }
    shadow_ = int(h % 2);
    shadow_pos_ = hooked[h];
    for (auto& e : cards_) e->committed_ = hooked[h];   // what the hook stores
    try {
      block_hook_(hooked[h], true);
    } catch (...) {
      shadow_ = -1;
      drain();
      broken_ = true;
      throw;
    }
    shadow_ = -1;
    for (auto& e : cards_) e->committed_ = base;
  };
  for (uint32_t j = 0, h = 0; j < n; ++j) {
    const auto& chunk = chunks[j];
    const bool hook = h < hooked.size() && hooked[h] == chunk.position + chunk.rows;
    while (hook && stored + 2 <= h) store();   // shadow h % 2 is free again
    e0.record_prefill_chunk(cx0, chunk, staging + (chunk.position - base), false);
    if (hook) shadow(cx0, 0, h);
    if (j) cx0.wait_event(sent(j - 1));   // the outbox is free again
    pack(cx0, 0, chunk.rows, true, outbox, true);
    cx0.barrier_signal(ready(j));
    copy.wait_event(ready(j));
    if (j) copy.wait_event(taken(j - 1));   // card 1 took the previous chunk out of the landing
    copy.copy(landing, outbox, size_t(chunk.rows) * row_bytes);
    copy.barrier_signal(sent(j));
    cx1.wait_event(sent(j));
    pack(cx1, 1, chunk.rows, true, landing, false);
    cx1.barrier_signal(taken(j));
    e1.record_prefill_chunk(cx1, chunk, nullptr, chunk.position + chunk.rows == end);
    if (hook) shadow(cx1, 1, h++);
  }
  while (stored < hooked.size()) store();
  if (!cx1.host_synchronize(bound) || !copy.host_synchronize(bound) ||
      !cx0.host_synchronize(bound)) {
    drain();
    fail("the prompt pass did not finish within " + std::to_string(bound / 1000000) + " ms");
  }
  for (uint32_t i = 0; i < 2; ++i)
    while (cards_[i]->kernels_.size() > first[i]) cards_[i]->kernels_.pop_back();
  set_position(end, 0);
  prefill_logits_ = true;
  if (block_hook_) block_hook_(end, end % kBlock == 0);   // the prompt's end: its first id chosen
  return e1.ctrl_->out_token[0];
}

}  // namespace runtime::inkling2
