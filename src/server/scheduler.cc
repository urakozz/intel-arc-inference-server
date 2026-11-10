#include "server/scheduler.h"

#include "server/prompt_lookup.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

namespace server {

// ---- PrefixSlots: spec 7 per slot ---------------------------------------------------------

PrefixSlots::PrefixSlots(BatchEngineIface& engine, size_t budget_bytes, HostAlloc& alloc)
    : engine_(engine),
      cache_(budget_bytes, engine.state_bytes(), PrefixCache::KvBytesFn([this](uint32_t n) {
               return engine_.kv_bytes(n);
             }),
             engine.block(), alloc, /*hash=*/{}, store_root(engine.kv_form(), engine.drafter_key())),
      slots_(engine.slots()) {}

uint32_t PrefixSlots::match(uint32_t slot, const std::vector<uint32_t>& prompt) {
  const Slot& s = slots_.at(slot);
  if (!s.valid) return 0;
  const size_t n = std::min(prompt.size(), s.resident.size());
  uint32_t lcp = 0;
  while (lcp < n && prompt[lcp] == s.resident[lcp]) ++lcp;
  return lcp;
}

uint32_t PrefixSlots::begin(uint32_t slot, const std::vector<uint32_t>& prompt) {
  Slot& s = slots_.at(slot);
  const PrefixCache::Plan plan = cache_.plan(prompt, s.resident, s.valid);
  s.valid = false;
  try {
    if (plan.kind == PrefixCache::Plan::Cold) {
      engine_.reset(slot);
    } else if (plan.kind == PrefixCache::Plan::Restore) {
      engine_.load_state(slot, plan.state, plan.restart);
      for (const auto& run : plan.kv) engine_.load_kv(slot, run.begin, run.end, run.host);
    }
  } catch (...) {
    cache_.release();
    throw;
  }
  cache_.release();
  s.kind = plan.kind;
  s.resident.assign(prompt.begin(), prompt.begin() + plan.restart);
  s.prompt = prompt;
  s.valid = true;
  return plan.restart;
}

void PrefixSlots::prefilled(uint32_t slot, uint32_t end, bool last) {
  Slot& s = slots_.at(slot);
  if (!s.valid) return;
  s.resident.assign(s.prompt.begin(), s.prompt.begin() + end);
  if (last || end % cache_.block() == 0) store(slot, end);
}

void PrefixSlots::fed(uint32_t slot, uint32_t id) {
  Slot& s = slots_.at(slot);
  if (s.valid) s.resident.push_back(id);
}

void PrefixSlots::end(uint32_t slot) {
  Slot& s = slots_.at(slot);
  if (!s.valid) return;
  const uint32_t pos = engine_.pos(slot);   // the engine's pos is the truth
  if (s.resident.size() > pos) s.resident.resize(pos);
  if (s.resident.size() != pos) {
    s.valid = false;
    return;
  }
  store(slot, pos);
}

void PrefixSlots::fail(uint32_t slot) { slots_.at(slot).valid = false; }

uint32_t PrefixSlots::held(uint32_t slot) {
  const Slot& s = slots_.at(slot);
  return s.valid ? static_cast<uint32_t>(s.resident.size()) : 0;
}

void PrefixSlots::store(uint32_t slot, uint32_t end) {
  Slot& s = slots_.at(slot);
  try {
    const PrefixCache::Slot r = cache_.reserve(s.resident, end, end % cache_.block() == 0);
    if (!r.empty()) {
      if (r.state) engine_.save_state(slot, r.state);
      for (const auto& p : r.kv) engine_.save_kv(slot, p.begin, p.end, p.host);
      cache_.commit(r);
    }
  } catch (...) {
    s.valid = false;
    cache_.release();
    throw;
  }
  cache_.release();
}

// ---- Scheduler -----------------------------------------------------------------------------

struct Scheduler::Impl {
  mutable std::mutex mu;
  std::condition_variable cv;
  std::deque<std::shared_ptr<Job>> waiting;   // mu
  size_t active = 0;                          // mu: slots held
  uint64_t next_order = 0;                    // mu
  uint64_t wake = 0;                          // mu: bumped by submit and cancel
  bool stopping = false;                      // mu
  std::thread thread;
  // Tick thread only.
  std::vector<std::shared_ptr<Job>> slots;
  std::set<uint32_t> eos;
  uint32_t n_slots = 0, max_len = 0;
  std::vector<uint32_t> slot_len;             // plan 13b: slot_max_len per slot
  bool spec = false;                          // step_spec rounds (a speculative engine or lookup)
  bool lookup = false;                        // plan 13b: prompt lookup per request
  uint32_t cap = 0;                           // drafts per row at most (spec_k or verify_k)
  std::vector<std::unique_ptr<PromptLookup>> matchers;            // plan 13b: one per slot
  std::vector<std::deque<std::vector<uint32_t>>> history;        // per slot (lookup_history)
  size_t inflight = 0;                        // plan 13b: chunks prefill_begin'd, not yet joined
  // Replica x slots: overlaps_prefill(), each slot's card (BatchEngineIface::card_of) and the number of cards.
  bool overlap = false;
  std::vector<uint32_t> slot_card;
  uint32_t n_cards = 0;
};

struct Scheduler::PrefillCall {
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<BatchEngineIface::Chunk> chunks;
};

Scheduler::Scheduler(BatchEngineIface& engine, SchedulerOptions opts, SlotCacheIface* cache)
    : engine_(engine), opts_(std::move(opts)), cache_(cache), impl_(std::make_unique<Impl>()) {
  impl_->n_slots = engine_.slots();
  if (impl_->n_slots == 0) throw std::invalid_argument("Scheduler: the engine has no slots");
  impl_->max_len = 0;
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    impl_->slot_len.push_back(engine_.slot_max_len(s));
    impl_->max_len = std::max(impl_->max_len, impl_->slot_len.back());
  }
  impl_->slots.resize(impl_->n_slots);
  impl_->overlap = engine_.overlaps_prefill();
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    impl_->slot_card.push_back(impl_->overlap ? engine_.card_of(s) : 0);
    impl_->n_cards = std::max(impl_->n_cards, impl_->slot_card.back() + 1);
  }
  impl_->eos.insert(opts_.eos_ids.begin(), opts_.eos_ids.end());
  if (opts_.lookup) {   // plan 13b: server.cc's refusals for --spec lookup, per engine
    if (engine_.verify_k() == 0)
      throw std::invalid_argument("--spec lookup: this engine cannot verify external drafts");
    if (engine_.spec_k() != 0)
      throw std::invalid_argument("--spec lookup and --spec mtp are one proposer each; pick one");
    PromptLookup probe;
    if (opts_.lookup_min_match < PromptLookup::kMinMatch ||
        opts_.lookup_min_match > probe.options().max_match)
      throw std::invalid_argument("the lookup minimum match must be in [2, 64]");
    impl_->lookup = true;
    for (uint32_t s = 0; s < impl_->n_slots; ++s) impl_->matchers.push_back(std::make_unique<PromptLookup>());
    impl_->history.resize(impl_->n_slots);
  }
  impl_->spec = engine_.spec_k() > 0 || impl_->lookup;
  impl_->cap = impl_->lookup ? engine_.verify_k() : engine_.spec_k();
}

// Requests still queued or holding a slot are dropped without on_done: the owner of the
// scheduler drains it (or cancels and ticks) before destroying it.
Scheduler::~Scheduler() { stop(); }

double Scheduler::now() const {
  if (opts_.clock) return opts_.clock();
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::shared_ptr<Job> Scheduler::submit(SchedRequest r) {
  if (r.prompt.empty()) throw std::invalid_argument("prompt encodes to zero tokens");
  if (r.prompt.size() + 1 > impl_->max_len) {
    throw std::invalid_argument("prompt is " + std::to_string(r.prompt.size()) +
                                " tokens; max_model_len is " + std::to_string(impl_->max_len));
  }
  auto job = std::make_shared<Job>();
  // The room to generate is the granted slot's (plan 13b: per-slot lengths); clamped at the grant.
  job->max_tokens = r.max_tokens;
  job->rng.seed(r.sampling.has_seed ? r.sampling.seed : std::random_device{}());
  job->fin.prompt_tokens = static_cast<uint32_t>(r.prompt.size());
  job->req = std::move(r);
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    const size_t free = impl_->n_slots - impl_->active;
    if (impl_->waiting.size() >= free + opts_.queue_depth) return nullptr;
    job->arrived = now();
    job->order = impl_->next_order++;
    impl_->waiting.push_back(job);
    ++impl_->wake;
  }
  impl_->cv.notify_all();
  return job;
}

void Scheduler::cancel(const std::shared_ptr<Job>& job) {
  if (!job) return;
  job->cancelled_ = true;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    ++impl_->wake;
  }
  impl_->cv.notify_all();
}

size_t Scheduler::waiting() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->waiting.size();
}

size_t Scheduler::active() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->active;
}

double Scheduler::hold_remaining() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  const size_t free = impl_->n_slots - impl_->active;
  if (opts_.admission != Admission::Batched || impl_->waiting.empty() || free == 0 ||
      impl_->waiting.size() >= free)
    return -1;
  return std::max(0.0, opts_.burst_hold_s - (now() - impl_->waiting.front()->arrived));
}

// Replica x slots: a card is busy while one of its slots has a chunk in flight that has not finished - any call on
// the card would wait behind it, so the tick grants and cancels nothing there (its steps: chunk_on_card).
bool Scheduler::card_busy(uint32_t card) const {
  if (!impl_->overlap) return false;
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    const std::shared_ptr<Job>& j = impl_->slots[s];
    if (impl_->slot_card[s] == card && j && j->inflight && !engine_.prefill_ready(s)) return true;
  }
  return false;
}

// Review Focus 5: the free slot the prompt fits whose resident session matches the most of
// it (affinity: the card that holds the request's prefix); among equal matches (plan 13b) the
// one whose session holds least - a new conversation does not evict a long one from a card
// while another card holds nothing. Replica x slots: with no match anywhere, the card with the
// most free slots first (then the least held there). One slot per card (plan 13b) or one card
// (13c): every card has one free slot or the same count, so the rule is the earlier one. The
// chosen slot's card busy (a chunk running there): the request waits for the next tick.
int Scheduler::pick_slot(const Job& job) const {
  int best = -1;
  uint32_t best_match = 0, best_held = 0, best_free = 0;
  const size_t need = job.req.prompt.size() + 1;
  std::vector<uint32_t> free_on(impl_->n_cards, 0);
  for (uint32_t s = 0; s < impl_->n_slots; ++s)
    if (!impl_->slots[s]) ++free_on[impl_->slot_card[s]];
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    if (impl_->slots[s] || impl_->slot_len[s] < need) continue;
    const uint32_t m = cache_ ? cache_->match(s, job.req.prompt) : 0;
    const uint32_t h = cache_ ? cache_->held(s) : 0;
    const uint32_t f = free_on[impl_->slot_card[s]];
    const bool better = best < 0 || m > best_match ||
                        (m == best_match && (m == 0 && f != best_free ? f > best_free : h < best_held));
    if (better) {
      best = static_cast<int>(s);
      best_match = m;
      best_held = h;
      best_free = f;
    }
  }
  if (best >= 0 && card_busy(impl_->slot_card[static_cast<uint32_t>(best)])) return -1;
  return best;
}

// Spec 19e's lookup_begin (server.cc), on the slot's matcher.
void Scheduler::lookup_begin(uint32_t slot, const std::vector<uint32_t>& prompt) {
  PromptLookup& lk = *impl_->matchers[slot];
  if (opts_.lookup_history == 0) {
    const std::vector<uint32_t>& held = lk.ids();
    size_t common = 0;
    const size_t n = std::min(held.size(), prompt.size());
    while (common < n && held[common] == prompt[common]) ++common;
    lk.truncate(common);
    for (size_t i = common; i < prompt.size(); ++i) lk.append(prompt[i]);
    return;
  }
  lk.clear();
  for (const std::vector<uint32_t>& earlier : impl_->history[slot]) {
    lk.append(earlier);
    lk.boundary();
  }
  lk.append(prompt);
}

void Scheduler::lookup_end(uint32_t slot, const std::vector<uint32_t>& out) {
  if (opts_.lookup_history == 0) return;
  auto& h = impl_->history[slot];
  h.push_back(out);
  while (h.size() > opts_.lookup_history) h.pop_front();
}

void Scheduler::grant(const std::shared_ptr<Job>& job, uint32_t slot) {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    ++impl_->active;
  }
  impl_->slots[slot] = job;
  job->slot = static_cast<int>(slot);
  job->fin.slot = job->slot;
  job->state = Job::State::Prefilling;
  job->max_tokens = std::min(job->max_tokens,
                             impl_->slot_len[slot] - static_cast<uint32_t>(job->req.prompt.size()));
  job->split = job->req.split_last && cache_ != nullptr;
  if (impl_->spec && (impl_->lookup || opts_.spec_fixed_k == 0)) {   // the request's own K policy
    AdaptiveKOptions o = impl_->lookup ? opts_.lookup_adaptive : opts_.spec;
    o.max_k = std::min(o.max_k, impl_->cap);
    job->policy = std::make_unique<AdaptiveK>(o);
  }
  if (impl_->lookup) {
    Job* raw = job.get();
    job->propose = [this, raw](uint32_t pending) {
      if (raw->head_indexed) throw std::logic_error("BatchEngineIface::step_spec: propose twice");
      PromptLookup& m = *impl_->matchers[static_cast<uint32_t>(raw->slot)];
      m.append(pending);
      raw->head_indexed = true;
      std::vector<uint32_t> d =
          raw->iter_k > 0 ? m.propose(raw->iter_k, opts_.lookup_min_match) : std::vector<uint32_t>{};
      raw->proposed = static_cast<uint32_t>(d.size());
      return d;
    };
  }
  try {
    // Review Focus 5: the prefix plan is taken here, at the grant, against what the slot
    // and the host store hold now - not when the request arrived.
    uint32_t restart = 0;
    if (cache_) restart = cache_->begin(slot, job->req.prompt);
    else engine_.reset(slot);
    job->prefilled = restart;
    job->fin.cached_tokens = restart;
    if (impl_->lookup) lookup_begin(slot, job->req.prompt);
  } catch (const std::exception& error) {
    finish(job, Finish::Error, error.what());
  }
}

// Never throws (plan 13b review): a request ends here whatever its callbacks or the cache do.
void Scheduler::finish(const std::shared_ptr<Job>& job, Finish::Reason reason, std::string error) {
  if (job->finished_) return;
  job->fin.reason = reason;
  job->fin.error = std::move(error);
  if (job->slot >= 0) {
    const uint32_t slot = static_cast<uint32_t>(job->slot);
    if (job->inflight) {   // a chunk still on the card: join it first (the tick only gets here once it is ready)
      try {
        engine_.prefill_end(slot);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "scheduler: slot %u's prefill failed at the request's end: %s\n", slot, e.what());
      }
      job->inflight = false;
      --impl_->inflight;
      if (cache_) cache_->fail(slot);   // the chunk's positions are not in the cache's view of the slot
    }
    if (cache_) {
      // Review Focus 2: the slot's final state goes to the host store now, before the
      // slot can be granted again.
      try {
        if (reason == Finish::Error) cache_->fail(slot);
        else cache_->end(slot);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "scheduler: request-end snapshot on slot %u failed: %s\n", slot,
                     e.what());
        cache_->fail(slot);
      }
    }
    if (impl_->lookup && (reason == Finish::Stop || reason == Finish::Length)) {
      try {
        lookup_end(slot, job->out);
      } catch (const std::exception&) {
      }
    }
    impl_->slots[slot] = nullptr;
    std::lock_guard<std::mutex> lock(impl_->mu);
    --impl_->active;
  }
  job->finished_ = true;
  if (job->req.on_done) {
    try {
      job->req.on_done(job->fin);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "scheduler: a request's on_done threw: %s\n", e.what());
    } catch (...) {
      std::fprintf(stderr, "scheduler: a request's on_done threw\n");
    }
  }
}

// The next prefill call's chunk for a prefilling request: up to len (or len - 1 under
// split-last, then the replay of the last id), cut by the engine's rule when it has one,
// else at `chunk` when `cut` (Interleave while rows decode).
BatchEngineIface::Chunk Scheduler::next_chunk(const Job& j, bool cut) {
  const uint32_t len = static_cast<uint32_t>(j.req.prompt.size());
  const uint32_t slot = static_cast<uint32_t>(j.slot);
  const uint32_t end = j.split ? len - 1 : len;
  if (j.prefilled >= end) return {slot, j.req.prompt.data() + len - 1, 1, true, true};
  uint32_t n = end - j.prefilled;
  const uint32_t engine_cut = engine_.prefill_cut(slot, j.prefilled, n);
  if (engine_cut != 0) n = std::min(n, engine_cut);
  else if (cut && opts_.chunk != 0) n = std::min(n, opts_.chunk);
  return {slot, j.req.prompt.data() + j.prefilled, n, !j.split && j.prefilled + n == len, false};
}

bool Scheduler::prefill_plan(PrefillCall& call) {
  bool decoding = false;
  for (const auto& j : impl_->slots) {
    if (!j) continue;
    if (j->state == Job::State::Decoding) decoding = true;
    if (j->state == Job::State::Prefilling) call.jobs.push_back(j);
  }
  if (call.jobs.empty()) return false;
  std::sort(call.jobs.begin(), call.jobs.end(), [](const auto& a, const auto& b) { return a->order < b->order; });
  if (opts_.admission == Admission::Interleave) call.jobs.resize(1);   // the oldest, one chunk
  for (const auto& j : call.jobs)
    call.chunks.push_back(next_chunk(*j, opts_.admission == Admission::Interleave && decoding));
  return true;
}

// After the prefill call: `error` null when every chunk went in; a SlotsFailed fails only the
// chunks it names, any other error every chunk of the call.
void Scheduler::prefill_done(PrefillCall& call, std::exception_ptr error) {
  std::vector<std::string> failed(call.jobs.size());
  if (error) {
    try {
      std::rethrow_exception(error);
    } catch (const SlotsFailed& f) {
      for (size_t i = 0; i < failed.size(); ++i)
        failed[i] = i < f.errors.size() ? f.errors[i] : std::string(f.what());
    } catch (const std::exception& e) {
      for (auto& f : failed) f = e.what();
    }
  }
  for (size_t i = 0; i < call.jobs.size(); ++i) {
    const auto& j = call.jobs[i];
    const BatchEngineIface::Chunk& c = call.chunks[i];
    if (!failed[i].empty()) {
      finish(j, Finish::Error, failed[i]);
      continue;
    }
    const uint32_t len = static_cast<uint32_t>(j->req.prompt.size());
    try {
      if (c.replay) {   // split-last: the last prompt id went through the decode replay
        j->prefilled = len;
        if (cache_) cache_->fed(c.slot, c.ids[0]);
      } else {
        j->prefilled += c.n;
        // The prompt-end snapshot: at len, or at len - 1 under split-last.
        const bool prompt_end = c.last || (j->split && j->prefilled == len - 1);
        if (cache_) cache_->prefilled(c.slot, j->prefilled, prompt_end);
      }
    } catch (const std::exception& e) {
      finish(j, Finish::Error, e.what());
      continue;
    }
    if (!c.last) continue;
    j->state = Job::State::Decoding;
    if (j->max_tokens == 0) finish(j, Finish::Length);
  }
}

// Spec 19d: one speculative iteration over every decoding slot; each row's run goes through the per-id rules in
// order, and the ids past the one that ends a request are handed back before its request-end snapshot. Plan 13b:
// with prompt lookup each row carries its request's proposer; a SlotsFailed fails only the rows it names.
bool Scheduler::spec_round() {
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<BatchEngineIface::SpecRow> rows;
  const uint32_t cap = impl_->cap;
  std::vector<uint32_t> ks;
  const std::vector<bool> busy = chunk_on_card();
  for (const auto& j : impl_->slots) {
    if (!j || j->state != Job::State::Decoding || busy[impl_->slot_card[static_cast<uint32_t>(j->slot)]]) continue;
    jobs.push_back(j);
    const uint32_t k = j->policy ? j->policy->next() : std::min(opts_.spec_fixed_k, cap);
    ks.push_back(std::min(k, cap));
  }
  // Spec 13c: the engine's row budget for this many rows (0: none) - every row keeps its pending id, and the drafts
  // left are dealt one at a time round the rows, the longest-running request first, each up to its own K (measured,
  // docs/probes/probe-batching-2026-10-23.md section 8: K 1 each at four requests beats K 3 for the oldest by 15 %,
  // K 2 / 2 / 1 at three beats 3 / 2 / 0 by 15 %).
  // Replica x slots (overlaps_prefill()): each card's rows against that card's own budget.
  const auto deal = [&](std::vector<size_t> by_age, uint32_t budget) {
    std::sort(by_age.begin(), by_age.end(), [&](size_t a, size_t b) { return jobs[a]->order < jobs[b]->order; });
    uint32_t left = budget > by_age.size() ? budget - static_cast<uint32_t>(by_age.size()) : 0;
    std::vector<uint32_t> want = ks;
    for (size_t i : by_age) ks[i] = 0;
    for (bool dealt = true; left > 0 && dealt;) {
      dealt = false;
      for (size_t i : by_age)
        if (left > 0 && ks[i] < want[i]) {
          ++ks[i];
          --left;
          dealt = true;
        }
    }
  };
  if (impl_->overlap) {
    std::vector<std::vector<size_t>> on_card(impl_->n_cards);
    for (size_t i = 0; i < jobs.size(); ++i)
      on_card[impl_->slot_card[static_cast<uint32_t>(jobs[i]->slot)]].push_back(i);
    for (uint32_t c = 0; c < impl_->n_cards; ++c) {
      if (on_card[c].empty()) continue;
      if (const uint32_t budget = engine_.card_row_budget(c, static_cast<uint32_t>(on_card[c].size())); budget != 0)
        deal(on_card[c], budget);
    }
  } else if (const uint32_t budget = engine_.row_budget(static_cast<uint32_t>(jobs.size())); budget != 0) {
    std::vector<size_t> all(jobs.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = i;
    deal(all, budget);
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& j = jobs[i];
    BatchEngineIface::SpecRow row{static_cast<uint32_t>(j->slot), &j->req.sampling, &j->rng, ks[i], nullptr};
    if (impl_->lookup) {
      j->iter_k = row.k;
      j->proposed = 0;
      j->head_indexed = false;
      row.propose = &j->propose;
    }
    rows.push_back(row);
  }
  if (rows.empty()) return false;
  std::vector<std::vector<uint32_t>> runs;
  std::vector<std::string> failed(rows.size());
  try {
    try {
      runs = engine_.step_spec(rows);
    } catch (const SlotsFailed& f) {
      runs = f.runs;
      runs.resize(rows.size());
      for (size_t i = 0; i < rows.size(); ++i) failed[i] = i < f.errors.size() ? f.errors[i] : std::string(f.what());
    }
    if (runs.size() != rows.size())
      throw std::logic_error("BatchEngineIface::step_spec returned " + std::to_string(runs.size()) +
                             " runs for " + std::to_string(rows.size()) + " rows");
    for (size_t i = 0; i < runs.size(); ++i) {
      if (!failed[i].empty()) continue;
      const uint32_t most = impl_->lookup ? jobs[i]->proposed : rows[i].k;
      if (runs[i].empty() || runs[i].size() > size_t(most) + 1)
        throw std::logic_error("BatchEngineIface::step_spec returned " + std::to_string(runs[i].size()) +
                               " ids for a row of " + std::to_string(most) + " drafts");
    }
  } catch (const std::exception& error) {
    for (const auto& j : jobs) finish(j, Finish::Error, error.what());
    return true;
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& j = jobs[i];
    const uint32_t slot = rows[i].slot, k = rows[i].k;
    if (!failed[i].empty()) {
      finish(j, Finish::Error, failed[i]);
      continue;
    }
    try {   // plan 13b review: a throwing callback ends this request only; the next rows still get their ids
      const std::vector<uint32_t>& run = runs[i];
      const uint32_t kept_drafts = static_cast<uint32_t>(run.size() - 1);
      if (impl_->lookup) {
        if (j->policy) j->policy->observe(j->proposed, kept_drafts);   // server.cc's next_lookup
      } else if (j->policy) {
        j->policy->observe(k, std::min(k, kept_drafts));
      }
      ++j->fin.spec_iters;
      j->fin.spec_drafted += impl_->lookup ? j->proposed : k;
      // server.cc's per-id rules over the run; `end` = the reason, at the id that ends the request.
      bool ended = false;
      Finish::Reason reason = Finish::Length;
      size_t used = 0;
      for (; used < run.size() && !ended;) {
        const uint32_t id = run[used++];
        if (impl_->lookup && (used > 1 || !j->head_indexed)) impl_->matchers[slot]->append(id);
        if (cache_) cache_->fed(slot, id);
        ++j->steps;
        const bool is_eos = impl_->eos.count(id) != 0;
        if (is_eos && !j->req.ignore_eos && j->fin.completion_tokens >= j->req.min_tokens) {
          ended = true;
          reason = Finish::Stop;
          break;
        }
        ++j->fin.completion_tokens;
        if (impl_->lookup && opts_.lookup_history != 0) j->out.push_back(id);
        if (j->req.on_token && !j->req.on_token(id)) {
          ended = true;
          reason = Finish::Stop;
          break;
        }
        if (j->steps >= j->max_tokens) {
          ended = true;
          reason = Finish::Length;
          break;
        }
      }
      j->fin.spec_accepted += static_cast<uint32_t>(used - 1);
      if (!ended) continue;
      if (used < run.size()) {
        try {   // the slot keeps exactly the ids consumed; then the request-end snapshot (finish)
          engine_.truncate_to(slot, engine_.pos(slot) - static_cast<uint32_t>(run.size() - used));
        } catch (const std::exception& error) {
          if (cache_) cache_->fail(slot);
          finish(j, Finish::Error, error.what());
          continue;
        }
      }
      finish(j, reason);
    } catch (const std::exception& error) {
      if (cache_) cache_->fail(slot);   // the ids after the throw were never consumed by the request
      finish(j, Finish::Error, error.what());
    }
  }
  return true;
}

bool Scheduler::decode_round() {
  if (impl_->spec) return spec_round();   // spec 19d; plan 13b: prompt lookup
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<BatchEngineIface::Row> rows;
  const std::vector<bool> busy = chunk_on_card();
  for (const auto& j : impl_->slots) {
    if (!j || j->state != Job::State::Decoding || busy[impl_->slot_card[static_cast<uint32_t>(j->slot)]]) continue;
    jobs.push_back(j);
    rows.push_back({static_cast<uint32_t>(j->slot), &j->req.sampling, &j->rng});
  }
  if (rows.empty()) return false;
  std::vector<uint32_t> ids;
  std::vector<std::string> failed(rows.size());
  try {
    try {
      ids = engine_.step_batch(rows);
    } catch (const SlotsFailed& f) {   // plan 13b: the rows it names fail, the others stand
      ids.assign(rows.size(), 0);
      for (size_t i = 0; i < rows.size(); ++i) {
        failed[i] = i < f.errors.size() ? f.errors[i] : std::string(f.what());
        if (failed[i].empty() && (i >= f.runs.size() || f.runs[i].size() != 1))
          throw std::logic_error("BatchEngineIface::step_batch: SlotsFailed without a completed row's id");
        if (failed[i].empty()) ids[i] = f.runs[i][0];
      }
    }
    if (ids.size() != rows.size())
      throw std::logic_error("BatchEngineIface::step_batch returned " + std::to_string(ids.size()) +
                             " ids for " + std::to_string(rows.size()) + " rows");
  } catch (const std::exception& error) {
    for (const auto& j : jobs) finish(j, Finish::Error, error.what());
    return true;
  }
  // server.cc's per-id rules, per row: the id is consumed (fed) whatever happens next.
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& j = jobs[i];
    if (!failed[i].empty()) {
      finish(j, Finish::Error, failed[i]);
      continue;
    }
    const uint32_t id = ids[i];
    try {   // plan 13b review: a throwing on_token ends this request only; the next rows still get their ids
      if (cache_) cache_->fed(rows[i].slot, id);
      ++j->steps;
      const bool is_eos = impl_->eos.count(id) != 0;
      if (is_eos && !j->req.ignore_eos && j->fin.completion_tokens >= j->req.min_tokens) {
        finish(j, Finish::Stop);
        continue;
      }
      ++j->fin.completion_tokens;
      if (j->req.on_token && !j->req.on_token(id)) {
        finish(j, Finish::Stop);
        continue;
      }
      if (j->steps >= j->max_tokens) finish(j, Finish::Length);
    } catch (const std::exception& error) {
      finish(j, Finish::Error, error.what());
    }
  }
  return true;
}

// Replica x slots: the cards with a chunk in flight (started, not yet joined - finished or not). Their rows sit out
// the step: the step would wait behind a running chunk, and one that finished is joined (with the cache's hooks) at
// the next tick before the card steps - so a card runs chunk, step, chunk, step. One slot per card: a card with a
// chunk has no decoding row, nothing changes.
std::vector<bool> Scheduler::chunk_on_card() const {
  std::vector<bool> on(impl_->n_cards, false);
  for (uint32_t s = 0; s < impl_->n_slots; ++s)
    if (impl_->slots[s] && impl_->slots[s]->inflight) on[impl_->slot_card[s]] = true;
  return on;
}

// Plan 13b: the overlapping engine's prefills. Joins every slot whose chunk is done (its
// prefill_done: the cache hooks, Decoding when the prompt is in), then starts the next chunk of
// every prefilling slot with none in flight - all of them in one prefill_begin, so two prompts
// prefill on their two cards at once. Returns true when it joined or started anything.
// Replica x slots: at most one chunk in flight per card - the oldest prefilling request's there.
// Before the step (!after_step) the joins and the cards with no decoding row; after the step the
// other cards, so a card with rows decoding runs chunk, step, chunk, step (13c's interleave on
// one card) and its rows sit out only the ticks its chunk runs. One slot per card (plan 13b): a
// prefilling card has no decoding row, so everything starts before the step, as before.
bool Scheduler::overlapped_prefills(bool after_step) {
  bool did = false;
  for (uint32_t s = 0; s < impl_->n_slots && !after_step; ++s) {
    const std::shared_ptr<Job> j = impl_->slots[s];
    if (!j || !j->inflight || !engine_.prefill_ready(s)) continue;
    std::exception_ptr error;
    try {
      engine_.prefill_end(s);
    } catch (...) {
      error = std::current_exception();
    }
    j->inflight = false;
    --impl_->inflight;
    PrefillCall call;
    call.jobs.push_back(j);
    call.chunks.push_back(j->chunk);
    prefill_done(call, error);
    did = true;
  }
  std::vector<bool> taken(impl_->n_cards, false), decoding(impl_->n_cards, false);
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    const std::shared_ptr<Job>& j = impl_->slots[s];
    if (!j) continue;
    if (j->inflight) taken[impl_->slot_card[s]] = true;
    if (j->state == Job::State::Decoding) decoding[impl_->slot_card[s]] = true;
  }
  std::vector<std::shared_ptr<Job>> oldest(impl_->n_cards);
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    const std::shared_ptr<Job>& j = impl_->slots[s];
    const uint32_t c = impl_->slot_card[s];
    if (!j || j->inflight || j->state != Job::State::Prefilling || j->cancelled_) continue;
    if (taken[c] || (!after_step && decoding[c])) continue;
    if (!oldest[c] || j->order < oldest[c]->order) oldest[c] = j;
  }
  PrefillCall start;
  for (const std::shared_ptr<Job>& j : oldest) {
    if (!j) continue;
    start.jobs.push_back(j);
    start.chunks.push_back(next_chunk(*j, true));
  }
  if (start.jobs.empty()) return did;
  try {
    engine_.prefill_begin(start.chunks);
  } catch (const std::exception& error) {   // nothing was started (the engine checks before it posts)
    for (const auto& j : start.jobs) finish(j, Finish::Error, error.what());
    return true;
  }
  for (size_t i = 0; i < start.jobs.size(); ++i) {
    start.jobs[i]->inflight = true;
    start.jobs[i]->chunk = start.chunks[i];
    ++impl_->inflight;
  }
  return true;
}

bool Scheduler::tick() {
  bool did = false;
  // 1. Cancellations: queued ones leave the queue, granted ones free their slot.
  std::vector<std::shared_ptr<Job>> dropped;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto& q = impl_->waiting;
    for (auto it = q.begin(); it != q.end();) {
      if ((*it)->cancelled_) {
        dropped.push_back(*it);
        it = q.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (const auto& j : dropped) finish(j, Finish::Cancelled);
  did = !dropped.empty();
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    const std::shared_ptr<Job> j = impl_->slots[s];
    if (!j || !j->cancelled_) continue;
    if (card_busy(impl_->slot_card[s])) continue;   // plan 13b: its card finishes the chunk first
    finish(j, Finish::Cancelled);
    did = true;
  }

  // 2. Grants, in arrival order; a request no free slot fits (plan 13b: per-slot lengths)
  // goes back to the head of the queue.
  std::vector<std::shared_ptr<Job>> granted;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto& q = impl_->waiting;
    const size_t free = impl_->n_slots - impl_->active;
    if (free != 0 && !q.empty()) {
      const bool go = opts_.admission == Admission::Interleave || q.size() >= free ||
                      now() - q.front()->arrived >= opts_.burst_hold_s;
      while (go && granted.size() < free && !q.empty()) {
        granted.push_back(q.front());
        q.pop_front();
      }
    }
  }
  std::vector<std::shared_ptr<Job>> back;
  for (const auto& j : granted) {
    const int slot = pick_slot(*j);
    if (slot < 0) {
      back.push_back(j);
      continue;
    }
    grant(j, static_cast<uint32_t>(slot));
    did = true;
  }
  if (!back.empty()) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    for (auto it = back.rbegin(); it != back.rend(); ++it) impl_->waiting.push_front(*it);
  }

  // 3. At most one prefill call; 4. one decode step. Plan 13b: with overlaps_prefill() (one card
  // per slot) every prefilling slot has a chunk in flight on its card while the others step - the
  // step never waits for a prefill. Replica x slots: a card's rows skip the step while its chunk
  // runs, and a card with rows decoding starts its next chunk after the step.
  if (impl_->overlap) {
    if (overlapped_prefills(false)) did = true;
    if (decode_round()) did = true;
    if (overlapped_prefills(true)) did = true;
  } else {
    PrefillCall call;
    const bool prefill = prefill_plan(call);
    if (prefill) {
      std::exception_ptr error;
      try {
        engine_.prefill(call.chunks);
      } catch (...) {
        error = std::current_exception();
      }
      prefill_done(call, error);
      did = true;
    }
    if (decode_round()) did = true;
  }
  return did;
}

void Scheduler::start() {
  if (impl_->thread.joinable()) return;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stopping = false;
  }
  impl_->thread = std::thread([this] {
    for (;;) {
      uint64_t seen = 0;
      {
        std::lock_guard<std::mutex> lock(impl_->mu);
        if (impl_->stopping) return;
        seen = impl_->wake;
      }
      bool did = false;
      try {
        did = tick();
      } catch (const std::exception& error) {
        std::fprintf(stderr, "scheduler: tick failed: %s\n", error.what());
      }
      if (did) continue;
      const double hold = hold_remaining();
      std::unique_lock<std::mutex> lock(impl_->mu);
      const auto woken = [&] { return impl_->stopping || impl_->wake != seen; };
      if (impl_->inflight != 0)   // plan 13b: a card's chunk is running - poll it (tick thread's own count)
        impl_->cv.wait_for(lock, std::chrono::microseconds(200), woken);
      else if (hold >= 0) impl_->cv.wait_for(lock, std::chrono::duration<double>(hold), woken);
      else impl_->cv.wait(lock, woken);
    }
  });
}

void Scheduler::stop() {
  if (!impl_ || !impl_->thread.joinable()) return;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stopping = true;
  }
  impl_->cv.notify_all();
  impl_->thread.join();
}

}  // namespace server
