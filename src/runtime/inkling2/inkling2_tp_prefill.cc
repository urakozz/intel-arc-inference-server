// Plan 41j Task 5: the tp prompt pass, two chunks in flight with the all-reduces on the copy engine.
//
// A chunk's walk is recorded in units (Engine::record_prefill_part): unit 0 runs up to the first all-reduce's part,
// unit i (1..96) starts with all-reduce i - 1's sum and runs to the next part (or, the last, to the chunk's end and
// the head). Two visits' crossings a visit, 48 visits: 96 a chunk. Per crossing and card: the compute list signals
// `ready`; the copy list waits on it, copies the card's part into the other card's landing of the chunk's lane
// (two slots, crossing i in slot i % 2) and signals `sent`; the next unit on each card waits for both cards' `sent`
// (the other's rows are in, its own part may be rewritten) and adds the two parts, card 0's first.
//
// Two lanes (chunk j's scratch is lane j % 2): the lagging chunk's unit i is recorded after the leading chunk's unit
// i + 2, so its visit v starts after the leader finished visit v (its caches, rings and convolution histories); each
// unit's crossing then has the other chunk's unit to hide behind. Both cards record the same units in the same order.
// With the prefix cache's hook a chunk that ends at a mid-prompt block end copies each visit's state slots into a
// shadow right after the visit (the next chunk rewrites them a visit later) and the host stores that block end while
// the next chunks run, as under pp: the copies of block end h are recorded only after the hook stored h - 2.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/inkling2_kernels.h"
#include "runtime/inkling2/inkling2_tp_engine.h"
#include "runtime/prefill_chunks.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
namespace k = kernels::inkling2;
constexpr uint32_t kCrossings = 2 * D::visit_count;   // a chunk's all-reduces
}  // namespace

uint32_t TpEngine::prefill(std::span<const uint32_t> ids, uint32_t cap) {
  if (broken_) throw std::runtime_error("Inkling tp: an all-reduce failed earlier in this session; reset first");
  for (const auto id : ids) contract_assert(id < D::vocab);
  cap = std::min(cap, chunk_rows(ids.size()));
  const uint32_t base = position(), end = base + uint32_t(ids.size());
  std::array<uint32_t*, 2> staging{};
  for (uint32_t r = 0; r < 2; ++r) {
    Engine& e = *cards_[r];
    staging[r] = static_cast<uint32_t*>(e.buffers_.prefill.ids->ptr());
    std::copy(ids.begin(), ids.end(), staging[r]);
    e.written_ = std::max(e.written_, end);
    e.prefill_end_ = end;   // the DFlash 2 drafter's in-window rows
    e.ctrl_->pos = end - 1;
    e.ctrl_->n_active = 1;
  }
  const std::array<size_t, 2> first{cards_[0]->kernels_.size(), cards_[1]->kernels_.size()};
  // B70_PREFILL_PROFILE=2: every launch timed on each card, by its stage; a card's device span against the time inside
  // its kernels is what it waited (for the other card's rows, its copy engine, the chunk ahead).
  const char* profile = std::getenv("B70_PREFILL_PROFILE");
  if (profile && std::strcmp(profile, "2") == 0)
    for (auto& e : cards_) {
      constexpr uint32_t kEvents = 1u << 18;
      e->profile_pool_ = std::make_unique<l0::EventPool>(e->ctx_, kEvents);
      e->profile_events_.reserve(kEvents);
    }
  const auto start = std::chrono::steady_clock::now();
  // With the prefix cache's hook every chunk ends at a snapshot point or the prompt's end (Engine::prefill's rule).
  std::vector<PrefillChunk> chunks;
  if (block_hook_) {
    for (uint32_t pos = base; pos < end;) {
      const uint32_t rows = prefill_chunk_rows(pos, end - pos, cap, true, kBlock, /*wide_stride=*/true);
      chunks.push_back({pos, rows, lead().desc_.visits});
      pos += rows;
    }
  } else {
    chunks = prefill_chunks(lead().desc_, base, uint32_t(ids.size()), cap);
  }
  const auto n = uint32_t(chunks.size());
  std::vector<int> hook_of(n, -1);   // the chunk's mid-prompt block end, by its index
  std::vector<uint32_t> hooked;
  for (uint32_t j = 0; j < n; ++j) {
    const uint32_t at = chunks[j].position + chunks[j].rows;
    if (block_hook_ && at < end && at % kBlock == 0) {
      hook_of[j] = int(hooked.size());
      hooked.push_back(at);
    }
  }
  // Events: ready / sent a crossing and card, then each card's `shadowed` a hook.
  const uint32_t events = 4 * kCrossings * n + 2 * uint32_t(hooked.size());
  Engine& e0 = lead();
  if (!chunk_events_ || chunk_events_->size() < events)
    chunk_events_ = std::make_unique<l0::SyncEventPool>(
        e0.ctx_, std::vector<const l0::Context*>{&cards_[0]->ctx_, &cards_[1]->ctx_}, events);
  auto& pool = *chunk_events_;
  for (uint32_t i = 0; i < events; ++i) pool.host_reset(i);
  const auto ready = [&](uint32_t j, uint32_t c, uint32_t r) { return pool.handle(4 * (kCrossings * j + c) + r); };
  const auto sent = [&](uint32_t j, uint32_t c, uint32_t r) { return pool.handle(4 * (kCrossings * j + c) + 2 + r); };
  const auto shadowed = [&](uint32_t h, uint32_t r) { return 4 * kCrossings * n + 2 * h + r; };
  for (uint32_t r = 0; r < 2; ++r) {
    if (!compute_[r]) compute_[r] = std::make_unique<l0::CmdList>(l0::CmdList::immediate_async(cards_[r]->ctx_));
    if (!copy_[r]) copy_[r] = std::make_unique<l0::CmdList>(l0::CmdList::immediate_copy(cards_[r]->ctx_));
  }
  const auto drain = [&] {   // release every list waiting on an event that will not come
    for (uint32_t i = 0; i < events; ++i) pool.host_signal(i);
    for (uint32_t r = 0; r < 2; ++r) {
      compute_[r]->host_synchronize(prefill_timeout_ns_);
      copy_[r]->host_synchronize(prefill_timeout_ns_);
    }
  };

  // Each chunk's walk on each card, its units recorded so far, the crossing its last unit left.
  struct Lane {
    std::array<Engine::PrefillWalk, 2> walk;
    std::array<Engine::Crossing, 2> crossing{};
    uint32_t units = 0;
    bool done = false;
  };
  std::vector<Lane> lanes;
  lanes.reserve(n);
  for (uint32_t j = 0; j < n; ++j) {
    const bool last = j + 1 == n;
    std::array<Engine::PrefillWalk, 2> walk{
        cards_[0]->prefill_walk(chunks[j], staging[0] + (chunks[j].position - base), last,
                                j % 2 ? cards_[0]->buffers_.prefill_b : cards_[0]->buffers_.prefill),
        cards_[1]->prefill_walk(chunks[j], staging[1] + (chunks[j].position - base), last,
                                j % 2 ? cards_[1]->buffers_.prefill_b : cards_[1]->buffers_.prefill)};
    lanes.push_back({walk});
  }
  uint32_t stored = 0;   // the hooks called
  const auto store = [&] {
    const uint32_t h = stored++;
    if (!pool.host_wait(shadowed(h, 0), prefill_timeout_ns_) || !pool.host_wait(shadowed(h, 1), prefill_timeout_ns_)) {
      drain();
      fail("the prompt chunk ending at " + std::to_string(hooked[h]) + " did not finish");
    }
    shadow_ = int(h % 2);
    shadow_pos_ = hooked[h];
    for (auto& e : cards_) e->committed_ = hooked[h];   // what the hook stores
    overlap_kv_ = true;
    try {
      block_hook_(hooked[h], true);
    } catch (...) {
      shadow_ = -1;
      overlap_kv_ = false;
      drain();
      broken_ = true;
      throw;
    }
    shadow_ = -1;
    overlap_kv_ = false;
    for (auto& e : cards_) e->committed_ = base;
  };
  // One unit of chunk j on both cards.
  const auto unit = [&](uint32_t j) {
    Lane& lane = lanes[j];
    const uint32_t u = lane.units++;
    const int hook = hook_of[j];
    if (hook >= 0 && u == 2)   // its first shadow copy follows: shadow hook % 2 must be stored
      while (stored + 2 <= uint32_t(hook)) store();
    for (uint32_t r = 0; r < 2; ++r) {
      Engine& e = *cards_[r];
      auto& list = *compute_[r];
      auto& walk = lane.walk[r];
      if (u) {   // crossing u - 1: both cards' rows in, card 0's first
        const uint32_t c = u - 1;
        const auto& x = lane.crossing[r];
        list.wait_event(sent(j, c, 1 - r));
        list.wait_event(sent(j, c, r));
        const auto* landing = static_cast<const std::byte*>(walk.s->tp_landing->ptr()) +
                              (c % 2) * (walk.s->tp_landing->size() / 2);
        const void* mine = x.part;
        e.launch(list, k::tp_prefill_glue(e.bf16_glue_), "ink_tp_sum", 128, D::hidden / 128, x.rows,
                 r == 0 ? mine : landing, r == 0 ? static_cast<const void*>(landing) : mine, x.sum);
      }
      const auto crossing = e.record_prefill_part(list, walk);
      // The visit before this unit's attention just ended (its experts' convolution and commit): its state slots
      // into the block end's shadow; after the last unit, the last visit's and the drafter's.
      if (hook >= 0 && u >= 2 && u % 2 == 0) e.record_visit_shadow(list, u / 2 - 1, uint32_t(hook) % 2);
      if (crossing) {
        lane.crossing[r] = *crossing;
        const uint32_t c = u;
        Engine& peer = *cards_[1 - r];
        const auto& landing = *(j % 2 ? peer.buffers_.prefill_b : peer.buffers_.prefill).tp_landing;
        list.barrier_signal(ready(j, c, r));
        copy_[r]->wait_event(ready(j, c, r));
        copy_[r]->copy(static_cast<std::byte*>(landing.ptr()) + (c % 2) * (landing.size() / 2), crossing->part,
                       size_t(crossing->rows) * D::hidden * sizeof(float));
        copy_[r]->barrier_signal(sent(j, c, r));
      } else {
        contract_assert(walk.done && u == kCrossings);
        if (hook >= 0) {
          e.record_visit_shadow(list, D::visit_count, uint32_t(hook) % 2);
          list.barrier_signal(pool.handle(shadowed(uint32_t(hook), r)));
        }
      }
    }
    lane.done = lane.walk[0].done;
  };
  // The leading chunk and the one behind it; the next chunk takes the leader's lane once the leader is done.
  uint32_t lead_j = 0, lag_j = 1;
  while (lead_j < n) {
    Lane& lead_lane = lanes[lead_j];
    if (!lead_lane.done) unit(lead_j);
    if (lag_j < n) {
      Lane& lag = lanes[lag_j];
      if (!lag.done && (lead_lane.done || lag.units + 3 <= lead_lane.units)) unit(lag_j);
    }
    if (lead_lane.done) {
      lead_j = lag_j;
      lag_j = lag_j + 1;
    }
  }
  while (stored < hooked.size()) store();
  for (uint32_t r = 0; r < 2; ++r)
    if (!compute_[r]->host_synchronize(prefill_timeout_ns_) || !copy_[r]->host_synchronize(prefill_timeout_ns_)) {
      drain();
      fail("the prompt pass did not finish within " + std::to_string(prefill_timeout_ns_ / 1000000) + " ms");
    }
  for (auto& e : cards_)
    if (e->kv_store_) e->kv_store_->host_synchronize();   // the mid-prompt KV stores, beside the chunks after them
  const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  for (uint32_t r = 0; r < 2; ++r)
    if (cards_[r]->profile_pool_)
      cards_[r]->report_profile(
          "tp card " + std::to_string(r) + " prompt profile: " + std::to_string(ids.size()) + " ids", wall);
  for (uint32_t r = 0; r < 2; ++r)
    while (cards_[r]->kernels_.size() > first[r]) cards_[r]->kernels_.pop_back();
  set_position(end, 0);
  prefill_logits_ = true;
  if (lead().ctrl_->out_token[0] != cards_[1]->ctrl_->out_token[0])
    fail("the two cards' first ids differ after the prompt pass");
  if (block_hook_) block_hook_(end, end % kBlock == 0);   // the prompt's end: its first id chosen
  return lead().ctrl_->out_token[0];
}

}  // namespace runtime::inkling2
