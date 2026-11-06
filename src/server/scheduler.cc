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

// Review Focus 5: the free slot the prompt fits whose resident session matches the most of
// it; among equal matches (plan 13b) the one whose session holds least - a new conversation
// does not evict a long one from a card while another card holds nothing.
int Scheduler::pick_slot(const Job& job) const {
  int best = -1;
  uint32_t best_match = 0, best_held = 0;
  const size_t need = job.req.prompt.size() + 1;
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    if (impl_->slots[s] || impl_->slot_len[s] < need) continue;
    const uint32_t m = cache_ ? cache_->match(s, job.req.prompt) : 0;
    const uint32_t h = cache_ ? cache_->held(s) : 0;
    if (best < 0 || m > best_match || (m == best_match && h < best_held)) {
      best = static_cast<int>(s);
      best_match = m;
      best_held = h;
    }
  }
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

void Scheduler::finish(const std::shared_ptr<Job>& job, Finish::Reason reason, std::string error) {
  if (job->finished_) return;
  job->fin.reason = reason;
  job->fin.error = std::move(error);
  if (job->slot >= 0) {
    const uint32_t slot = static_cast<uint32_t>(job->slot);
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
    if (impl_->lookup && (reason == Finish::Stop || reason == Finish::Length)) lookup_end(slot, job->out);
    impl_->slots[slot] = nullptr;
    std::lock_guard<std::mutex> lock(impl_->mu);
    --impl_->active;
  }
  job->finished_ = true;
  if (job->req.on_done) job->req.on_done(job->fin);
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
  for (const auto& j : impl_->slots) {
    if (!j || j->state != Job::State::Decoding) continue;
    jobs.push_back(j);
    const uint32_t k = j->policy ? j->policy->next() : std::min(opts_.spec_fixed_k, cap);
    BatchEngineIface::SpecRow row{static_cast<uint32_t>(j->slot), &j->req.sampling, &j->rng, std::min(k, cap), nullptr};
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
  }
  return true;
}

bool Scheduler::decode_round() {
  if (impl_->spec) return spec_round();   // spec 19d; plan 13b: prompt lookup
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<BatchEngineIface::Row> rows;
  for (const auto& j : impl_->slots) {
    if (!j || j->state != Job::State::Decoding) continue;
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

  // 3. At most one prefill call; 4. one decode step. Plan 13b: with overlaps_prefill() the
  // prefill call runs beside the step (other slots, other cards) and is joined after it.
  PrefillCall call;
  const bool prefill = prefill_plan(call);
  if (engine_.overlaps_prefill()) {
    std::exception_ptr error;
    bool started = false;
    if (prefill) {
      try {
        engine_.prefill_begin(call.chunks);
        started = true;
      } catch (...) {
        error = std::current_exception();
      }
    }
    if (decode_round()) did = true;
    if (prefill) {
      if (started) {
        try {
          engine_.prefill_end();
        } catch (...) {
          error = std::current_exception();
        }
      }
      prefill_done(call, error);
      did = true;
    }
  } else {
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
      if (hold >= 0) impl_->cv.wait_for(lock, std::chrono::duration<double>(hold), woken);
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
