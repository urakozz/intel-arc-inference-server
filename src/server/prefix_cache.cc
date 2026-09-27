#include "server/prefix_cache.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace server {

struct PrefixCache::Entry {
  bool is_block = true;
  Entry* parent = nullptr;               // nullptr only for the root
  uint64_t hash = 0;                     // the chain: hash(parent->hash, ids)
  uint32_t begin = 0, end = 0;           // KV [begin, end); a snapshot's pos is `end`
  std::vector<uint32_t> ids;             // ids [begin, end), exact
  void* state = nullptr;                 // snapshot only
  void* kv = nullptr;                    // KV [begin, end) in save_kv's layout (may be null)
  size_t bytes = 0;                      // one allocation: state then KV
  std::vector<Entry*> children;
  uint64_t lru = 0;
  int pins = 0;
  bool ready = false;
};

namespace {
uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

bool same_ids(const std::vector<uint32_t>& a, const uint32_t* b) {
  return std::memcmp(a.data(), b, a.size() * sizeof(uint32_t)) == 0;
}
}  // namespace

const char* plan_kind_name(PrefixCache::Plan::Kind k) {
  switch (k) {
    case PrefixCache::Plan::Cold: return "cold";
    case PrefixCache::Plan::Continue: return "continue";
    case PrefixCache::Plan::Restore: return "restore";
  }
  return "?";
}

PrefixCache::PrefixCache(size_t budget_bytes, size_t state_bytes, size_t kv_bytes_per_pos,
                         uint32_t block, HostAlloc& alloc, HashFn hash)
    : budget_(budget_bytes),
      state_bytes_(state_bytes),
      kv_per_pos_(kv_bytes_per_pos),
      block_(block),
      alloc_(alloc),
      hash_(std::move(hash)),
      root_(std::make_unique<Entry>()) {
  if (block_ == 0) throw std::invalid_argument("PrefixCache: block must be > 0");
  root_->ready = true;
}

PrefixCache::~PrefixCache() {
  for (auto& e : all_)
    if (e->bytes) alloc_.free(e->state ? e->state : e->kv, e->bytes);
}

size_t PrefixCache::entries() const { return all_.size(); }

uint64_t PrefixCache::hash(uint64_t parent, const uint32_t* ids, size_t n) const {
  if (hash_) return hash_(parent, ids, n);
  uint64_t h = splitmix(parent ^ 0x5851F42D4C957F2Dull);
  for (size_t i = 0; i < n; ++i) h = splitmix(h ^ ids[i]);
  return splitmix(h ^ n);
}

PrefixCache::Entry* PrefixCache::find_block(Entry* parent, const uint32_t* ids) const {
  const uint64_t h = hash(parent->hash, ids, block_);
  for (Entry* c : parent->children)
    if (c->is_block && c->ready && c->hash == h && same_ids(c->ids, ids)) return c;
  return nullptr;
}

PrefixCache::Entry* PrefixCache::find_snap(Entry* parent, const uint32_t* ids,
                                           uint32_t end) const {
  const uint32_t n = end - parent->end;
  const uint64_t h = hash(parent->hash, ids, n);
  for (Entry* c : parent->children)
    if (!c->is_block && c->end == end && c->hash == h && same_ids(c->ids, ids)) return c;
  return nullptr;
}

void PrefixCache::pin(Entry* e) {
  ++e->pins;
  pinned_.push_back(e);
}

void PrefixCache::remove(Entry* e) {
  Entry* p = e->parent;
  p->children.erase(std::find(p->children.begin(), p->children.end(), e));
  if (e->bytes) alloc_.free(e->state ? e->state : e->kv, e->bytes);
  used_ -= e->bytes;
  auto it = std::find_if(all_.begin(), all_.end(), [e](const auto& u) { return u.get() == e; });
  all_.erase(it);
}

bool PrefixCache::evict_one() {
  Entry* victim = nullptr;
  for (auto& u : all_) {
    Entry* e = u.get();
    if (!e->children.empty() || e->pins > 0 || !e->ready) continue;
    if (victim == nullptr || e->lru < victim->lru) victim = e;
  }
  if (victim == nullptr) return false;
  remove(victim);
  return true;
}

bool PrefixCache::make_room(size_t bytes) {
  if (bytes > budget_) return false;
  while (used_ + bytes > budget_)
    if (!evict_one()) return false;
  return true;
}

void* PrefixCache::alloc(size_t bytes) {
  if (!make_room(bytes)) return nullptr;
  for (;;) {
    if (void* p = alloc_.alloc(bytes)) return p;
    if (!evict_one()) return nullptr;   // fragmentation: free more and retry
  }
}

PrefixCache::Slot PrefixCache::reserve(const std::vector<uint32_t>& ids, uint32_t end,
                                       bool is_block_end) {
  Slot slot;
  if (end == 0 || ids.size() < end) return slot;
  if (is_block_end != (end % block_ == 0))
    throw std::invalid_argument("PrefixCache::reserve: is_block_end disagrees with end");
  const uint32_t base = end / block_ * block_;
  // The chain of present blocks, pinned so that making room cannot evict it.
  Entry* node = root_.get();
  uint32_t at = 0;
  while (at < base) {
    Entry* b = find_block(node, ids.data() + at);
    if (b == nullptr) break;
    b->lru = ++clock_;
    pin(b);
    node = b;
    at += block_;
  }
  // The missing blocks, in order; a block that cannot be stored ends the chain.
  while (at < base) {
    const size_t bytes = kv_per_pos_ * block_;
    void* p = alloc(bytes);
    if (p == nullptr) return slot;
    auto e = std::make_unique<Entry>();
    e->parent = node;
    e->begin = at;
    e->end = at + block_;
    e->ids.assign(ids.begin() + at, ids.begin() + at + block_);
    e->hash = hash(node->hash, e->ids.data(), block_);
    e->kv = p;
    e->bytes = bytes;
    e->lru = ++clock_;
    used_ += bytes;
    node->children.push_back(e.get());
    pin(e.get());
    slot.kv.push_back({at, at + block_, p});
    slot.entries.push_back(e.get());
    node = e.get();
    all_.push_back(std::move(e));
    at += block_;
  }
  // The snapshot at `end`, under the block that ends at base.
  if (Entry* s = find_snap(node, ids.data() + base, end)) {
    s->lru = ++clock_;
    return slot;
  }
  const size_t kv = kv_per_pos_ * (end - base);
  const size_t bytes = state_bytes_ + kv;
  void* p = alloc(bytes);
  if (p == nullptr) return slot;
  auto e = std::make_unique<Entry>();
  e->is_block = false;
  e->parent = node;
  e->begin = base;
  e->end = end;
  e->ids.assign(ids.begin() + base, ids.begin() + end);
  e->hash = hash(node->hash, e->ids.data(), end - base);
  e->state = p;
  e->kv = kv ? static_cast<uint8_t*>(p) + state_bytes_ : nullptr;
  e->bytes = bytes;
  e->lru = ++clock_;
  used_ += bytes;
  node->children.push_back(e.get());
  pin(e.get());
  slot.state = p;
  if (kv) slot.kv.push_back({base, end, e->kv});
  slot.entries.push_back(e.get());
  all_.push_back(std::move(e));
  return slot;
}

void PrefixCache::commit(const Slot& slot) {
  for (Entry* e : slot.entries) e->ready = true;
}

void PrefixCache::release() {
  for (Entry* e : pinned_) --e->pins;
  pinned_.clear();
  // Reserved and never committed (the engine threw while filling): drop, deepest first.
  for (bool again = true; again;) {
    again = false;
    for (auto& u : all_) {
      Entry* e = u.get();
      if (!e->ready && e->children.empty()) {
        remove(e);
        again = true;
        break;
      }
    }
  }
}

PrefixCache::Plan PrefixCache::plan(const std::vector<uint32_t>& prompt,
                                    const std::vector<uint32_t>& resident, bool resident_valid) {
  Plan out;
  const uint32_t len = uint32_t(prompt.size());
  if (len == 0) return out;
  uint32_t lcp = 0;
  if (resident_valid) {
    const size_t n = std::min(prompt.size(), resident.size());
    while (lcp < n && prompt[lcp] == resident[lcp]) ++lcp;
  }
  const bool cont = resident_valid && !resident.empty() && resident.size() <= len - 1 &&
                    lcp == resident.size();

  // The deepest snapshot whose ids are prompt[0:p], p <= len - 1.
  std::vector<Entry*> chain;             // matched blocks, root excluded
  Entry* node = root_.get();
  Entry* best = nullptr;
  for (;;) {
    for (Entry* c : node->children) {
      if (c->is_block || !c->ready) continue;
      const uint32_t e = c->end;
      if (e > len - 1) continue;
      if (best != nullptr && e <= best->end) continue;
      if (c->hash != hash(node->hash, prompt.data() + node->end, e - node->end)) continue;
      if (!same_ids(c->ids, prompt.data() + node->end)) continue;
      best = c;
    }
    if (node->end + block_ > len - 1) break;
    Entry* b = find_block(node, prompt.data() + node->end);
    if (b == nullptr) break;
    chain.push_back(b);
    node = b;
  }

  if (best != nullptr && (!cont || best->end > resident.size())) {
    out.kind = Plan::Restore;
    out.restart = best->end;
    out.state = best->state;
    const uint32_t d = std::min(lcp, best->end);
    out.kv_from = best->end;
    for (Entry* b : chain) {
      if (b->end > best->begin) break;   // blocks below the snapshot's base only
      if (b->end > d) {
        out.kv.push_back({b->begin, b->end, b->kv});
        out.kv_from = std::min(out.kv_from, b->begin);
      }
      b->lru = ++clock_;
      pin(b);
    }
    if (best->end > best->begin && best->end > d) {
      out.kv.push_back({best->begin, best->end, best->kv});
      out.kv_from = std::min(out.kv_from, best->begin);
    }
    best->lru = ++clock_;
    pin(best);
    return out;
  }
  if (cont) {
    out.kind = Plan::Continue;
    out.restart = uint32_t(resident.size());
    out.kv_from = out.restart;
  }
  return out;
}

// ---- the request path --------------------------------------------------------------

namespace {
double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

PrefixSession::PrefixSession(EngineIface& engine, size_t budget_bytes, HostAlloc* alloc)
    : engine_(engine) {
  if (budget_bytes == 0) return;
  if (alloc == nullptr) throw std::invalid_argument("PrefixSession: a budget needs an allocator");
  cache_ = std::make_unique<PrefixCache>(budget_bytes, engine.state_bytes(), engine.kv_bytes(1),
                                         engine.block(), *alloc);
  engine_.set_block_hook([this](uint32_t end, bool is_block_end) {
    if (prompt_ != nullptr) store(*prompt_, end, is_block_end);
  });
}

PrefixSession::~PrefixSession() {
  if (cache_) engine_.set_block_hook({});
}

void PrefixSession::store(const std::vector<uint32_t>& ids, uint32_t end, bool is_block_end) {
  const auto t0 = std::chrono::steady_clock::now();
  const PrefixCache::Slot slot = cache_->reserve(ids, end, is_block_end);
  if (!slot.empty()) {
    if (slot.state) engine_.save_state(slot.state);
    for (const auto& p : slot.kv) engine_.save_kv(p.begin, p.end, p.host);
    cache_->commit(slot);
  }
  store_ms_ += ms_since(t0);
}

PrefixSession::Report PrefixSession::begin(const std::vector<uint32_t>& prompt) {
  Report r;
  if (!cache_) {
    engine_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    engine_.prefill(prompt);
    r.prefill_ms = ms_since(t0);
    return r;
  }
  const PrefixCache::Plan plan = cache_->plan(prompt, resident_, valid_);
  valid_ = false;
  store_ms_ = 0;
  try {
    const auto t0 = std::chrono::steady_clock::now();
    if (plan.kind == PrefixCache::Plan::Cold) {
      engine_.reset();
    } else if (plan.kind == PrefixCache::Plan::Restore) {
      engine_.load_state(plan.state, plan.restart);
      for (const auto& run : plan.kv) {
        engine_.load_kv(run.begin, run.end, run.host);
        r.kv_bytes += engine_.kv_bytes(run.end - run.begin);
      }
    }
    r.restore_ms = ms_since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    // The tail to len - 1 (its last hook call is the prompt-end snapshot, at len - 1),
    // then the last id through one decode replay: pos == len, the first generated id
    // pending. plan() keeps restart <= len - 1.
    const size_t last = prompt.size() - 1;
    if (plan.restart < last) {
      prompt_ = &prompt;
      engine_.prefill(std::vector<uint32_t>(prompt.begin() + plan.restart, prompt.begin() + last));
      prompt_ = nullptr;
    }
    engine_.ingest({prompt[last]});
    r.prefill_ms = ms_since(t1) - store_ms_;
  } catch (...) {
    prompt_ = nullptr;
    cache_->release();
    throw;
  }
  r.kind = plan.kind;
  r.restart = plan.restart;
  r.store_ms = store_ms_;
  resident_ = prompt;
  valid_ = true;
  cache_->release();
  return r;
}

void PrefixSession::fed(uint32_t id) {
  if (cache_ && valid_) resident_.push_back(id);
}

double PrefixSession::end() {
  if (!cache_ || !valid_) return 0;
  // The engine's pos is the truth: every id step() returned has had its KV written.
  const uint32_t pos = engine_.pos();
  if (resident_.size() > pos) resident_.resize(pos);
  if (resident_.size() != pos) {
    valid_ = false;
    return 0;
  }
  store_ms_ = 0;
  try {
    store(resident_, pos, pos % cache_->block() == 0);
  } catch (...) {
    valid_ = false;
    cache_->release();
    throw;
  }
  cache_->release();
  return store_ms_;
}

}  // namespace server
