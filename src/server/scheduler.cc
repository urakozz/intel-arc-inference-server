#include "server/scheduler.h"

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
      cache_(budget_bytes, engine.state_bytes(), engine.kv_bytes(1), engine.block(), alloc,
             /*hash=*/{}, engine.kv_form()),
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
};

Scheduler::Scheduler(BatchEngineIface& engine, SchedulerOptions opts, SlotCacheIface* cache)
    : engine_(engine), opts_(std::move(opts)), cache_(cache), impl_(std::make_unique<Impl>()) {
  impl_->n_slots = engine_.slots();
  impl_->max_len = engine_.max_len();
  if (impl_->n_slots == 0) throw std::invalid_argument("Scheduler: the engine has no slots");
  impl_->slots.resize(impl_->n_slots);
  impl_->eos.insert(opts_.eos_ids.begin(), opts_.eos_ids.end());
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
  const uint32_t budget = impl_->max_len - static_cast<uint32_t>(r.prompt.size());
  job->max_tokens = std::min(r.max_tokens, budget);
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

uint32_t Scheduler::pick_slot(const Job& job) const {
  int best = -1;
  uint32_t best_match = 0;
  for (uint32_t s = 0; s < impl_->n_slots; ++s) {
    if (impl_->slots[s]) continue;
    const uint32_t m = cache_ ? cache_->match(s, job.req.prompt) : 0;
    if (best < 0 || m > best_match) {
      best = static_cast<int>(s);
      best_match = m;
    }
  }
  if (best < 0) throw std::logic_error("Scheduler: no free slot to grant");
  return static_cast<uint32_t>(best);
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
  try {
    // Review Focus 5: the prefix plan is taken here, at the grant, against what the slot
    // and the host store hold now - not when the request arrived.
    uint32_t restart = 0;
    if (cache_) restart = cache_->begin(slot, job->req.prompt);
    else engine_.reset(slot);
    job->prefilled = restart;
    job->fin.cached_tokens = restart;
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
    impl_->slots[slot] = nullptr;
    std::lock_guard<std::mutex> lock(impl_->mu);
    --impl_->active;
  }
  job->finished_ = true;
  if (job->req.on_done) job->req.on_done(job->fin);
}

bool Scheduler::prefill_round() {
  std::vector<std::shared_ptr<Job>> jobs;
  bool decoding = false;
  for (const auto& j : impl_->slots) {
    if (!j) continue;
    if (j->state == Job::State::Decoding) decoding = true;
    if (j->state == Job::State::Prefilling) jobs.push_back(j);
  }
  if (jobs.empty()) return false;
  std::sort(jobs.begin(), jobs.end(), [](const auto& a, const auto& b) { return a->order < b->order; });
  if (opts_.admission == Admission::Interleave) jobs.resize(1);   // the oldest, one chunk

  std::vector<BatchEngineIface::Chunk> chunks;
  for (const auto& j : jobs) {
    const uint32_t len = static_cast<uint32_t>(j->req.prompt.size());
    uint32_t n = len - j->prefilled;
    if (opts_.admission == Admission::Interleave && decoding && opts_.chunk != 0)
      n = std::min(n, opts_.chunk);
    chunks.push_back({static_cast<uint32_t>(j->slot), j->req.prompt.data() + j->prefilled, n,
                      j->prefilled + n == len});
  }
  try {
    engine_.prefill(chunks);
  } catch (const std::exception& error) {
    for (const auto& j : jobs) finish(j, Finish::Error, error.what());
    return true;
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& j = jobs[i];
    j->prefilled += chunks[i].n;
    try {
      if (cache_) cache_->prefilled(chunks[i].slot, j->prefilled, chunks[i].last);
    } catch (const std::exception& error) {
      finish(j, Finish::Error, error.what());
      continue;
    }
    if (!chunks[i].last) continue;
    j->state = Job::State::Decoding;
    if (j->max_tokens == 0) finish(j, Finish::Length);
  }
  return true;
}

bool Scheduler::decode_round() {
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<BatchEngineIface::Row> rows;
  for (const auto& j : impl_->slots) {
    if (!j || j->state != Job::State::Decoding) continue;
    jobs.push_back(j);
    rows.push_back({static_cast<uint32_t>(j->slot), &j->req.sampling, &j->rng});
  }
  if (rows.empty()) return false;
  std::vector<uint32_t> ids;
  try {
    ids = engine_.step_batch(rows);
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

  // 2. Grants, in arrival order.
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
  for (const auto& j : granted) {
    grant(j, pick_slot(*j));
    did = true;
  }

  // 3. At most one prefill call; 4. one decode step.
  if (prefill_round()) did = true;
  if (decode_round()) did = true;
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
