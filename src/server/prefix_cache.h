#pragma once
// Spec 7 §3.1 and §3.3: the prefix cache, host side. Pure host code: no device calls,
// testable without a GPU (tests/server/prefix_cache_test.cc, C4).
//
// The store holds two kinds of entry in pinned host memory, in a tree:
//   * a KV block: positions [b*block, (b+1)*block), K and V of every FA layer, in the
//     engine's save_kv host layout. Its parent is the block before it (the root for b = 0).
//   * a snapshot at position `end`: the recurrent state (save_state) plus the KV of
//     [base, end), base = the last block boundary <= end. Its parent is the block that ends
//     at base (the root when base = 0).
// Every entry stores the exact ids it covers and a 64-bit hash chain over them; a lookup
// matches the hash and then compares the ids, so a collision never produces a wrong hit.
// Eviction is LRU over leaves only (a block with a child is never evicted, so no snapshot
// outlives its blocks), and entries in use by the current request are pinned until
// release().
//
// PrefixSession is the request path over server::EngineIface: plan, restore, tail
// prefill with the block hook writing through, and the request-end snapshot. With
// `split_last` (b70-serve --prefix-split-last, chat only; off by default) it prefills the
// tail up to len - 1
// and feeds the last prompt id through one decode replay, so the prompt-end snapshot sits
// at len - 1: the next turn's history usually diverges exactly there (the template's
// "<think>\n" becomes "<think>\n\n</think>" when the client does not send the reasoning
// back), and the same prompt again restores at len - 1 (Review Focus 5). The price: the
// last prompt id goes through decode's kernels, not prefill's, so a near-tie first token
// can differ from a cold run's, and on the synthetic C3 replay two such differences fail
// the tie rule (docs/probe-prefix-cache-2026-09-27.md §9). Without it (the default) the
// tail is one prefill and the prompt-end snapshot is at len; a restore at a block end then
// reproduces a cold run bitwise.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "server/deps.h"

namespace server {

struct HostAlloc {                       // pinned host memory in b70_serve, malloc in tests
  virtual ~HostAlloc() = default;
  virtual void* alloc(size_t bytes) = 0;  // nullptr when it cannot
  virtual void free(void* p, size_t bytes) = 0;
};

class PrefixCache {
 public:
  // hash(parent_hash, ids, n): the hash chain step. Empty = the default (splitmix over the
  // ids); the tests inject a constant to force collisions.
  using HashFn = std::function<uint64_t(uint64_t, const uint32_t*, size_t)>;
  // `kv_form` (spec 12b): the engine's KV cache form (EngineIface::kv_form, 0 = bf16), the
  // root of every hash chain - an entry's bytes are save_kv's host layout IN THAT FORM, so
  // the same ids under another form are another key and never a hit. 0 leaves every hash
  // what it was.
  PrefixCache(size_t budget_bytes, size_t state_bytes, size_t kv_bytes_per_pos, uint32_t block,
              HostAlloc& alloc, HashFn hash = {}, uint64_t kv_form = 0);
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  struct Entry;                          // opaque
  // One KV range to fill: engine.save_kv(begin, end, host).
  struct Piece {
    uint32_t begin, end;
    void* host;
  };
  // What a store call must copy: engine.save_state(state) if non-null, then every piece.
  // Empty (state == nullptr, kv empty) when nothing is to be stored.
  struct Slot {
    void* state = nullptr;
    std::vector<Piece> kv;
    std::vector<Entry*> entries;         // for commit()
    bool empty() const { return state == nullptr && kv.empty(); }
  };
  // The block hook's store call. `ids` are the session's ids, at least `end` of them; the
  // card holds exactly ids[0:end). Stores every block of ids[0:floor(end/block)*block) the
  // store lacks (as many as the budget fits, in order) and a snapshot at `end` (if it is
  // not present and its blocks are). Entries reserved here are pinned until release().
  Slot reserve(const std::vector<uint32_t>& ids, uint32_t end, bool is_block_end);
  void commit(const Slot& slot);         // after the engine has filled it

  struct Plan {
    enum Kind { Cold, Continue, Restore } kind = Cold;
    uint32_t restart = 0;                // positions [0, restart) are reused
    uint32_t kv_from = 0;                // Restore: KV [kv_from, restart) comes from the host
    const void* state = nullptr;         // Restore
    struct Run {
      uint32_t begin, end;
      const void* host;
    };
    std::vector<Run> kv;                 // Restore: load_kv runs covering [kv_from, restart)
  };
  // `resident` = the ids whose state and KV the card holds (resident.size() == pos),
  // `resident_valid` false after a failure. Pins the entries it returns until release().
  Plan plan(const std::vector<uint32_t>& prompt, const std::vector<uint32_t>& resident,
            bool resident_valid);
  // Unpins everything plan() and reserve() pinned; drops reserved entries never committed.
  void release();

  size_t bytes_used() const { return used_; }
  size_t budget() const { return budget_; }
  size_t entries() const;
  uint32_t block() const { return block_; }

 private:
  uint64_t hash(uint64_t parent, const uint32_t* ids, size_t n) const;
  Entry* find_block(Entry* parent, const uint32_t* ids) const;
  Entry* find_snap(Entry* parent, const uint32_t* ids, uint32_t end) const;
  bool make_room(size_t bytes);          // evicts unpinned LRU leaves
  bool evict_one();
  void* alloc(size_t bytes);             // alloc with eviction retries
  void remove(Entry* e);
  void pin(Entry* e);

  size_t budget_, state_bytes_, kv_per_pos_;
  uint32_t block_;
  HostAlloc& alloc_;
  HashFn hash_;
  std::unique_ptr<Entry> root_;
  std::vector<std::unique_ptr<Entry>> all_;
  std::vector<Entry*> pinned_;
  size_t used_ = 0;
  uint64_t clock_ = 0;
};

// The request path (spec 7 §3.3) with the cache on; with budget 0 it is today's server:
// reset() and a full prefill.
class PrefixSession {
 public:
  PrefixSession(EngineIface& engine, size_t budget_bytes, HostAlloc* alloc);
  ~PrefixSession();
  bool enabled() const { return cache_ != nullptr; }

  struct Report {
    PrefixCache::Plan::Kind kind = PrefixCache::Plan::Cold;
    uint32_t restart = 0;                // usage.prompt_tokens_details.cached_tokens
    size_t kv_bytes = 0;                 // host to device for the restore
    double restore_ms = 0, prefill_ms = 0, store_ms = 0;   // prefill_ms: the tail to
                                                            // len - 1 and the last id's replay
  };
  // Leaves the engine at pos == prompt.size() with the first generated id pending.
  // Throws what the engine throws; the resident session is then unknown.
  Report begin(const std::vector<uint32_t>& prompt, bool split_last = false);
  void fed(uint32_t id);                 // every id step() returned: its KV is written
  // After the last frame: the request-end snapshot. Returns the store time in ms.
  double end();
  void fail() { valid_ = false; }        // the request failed after begin()

  PrefixCache* cache() { return cache_.get(); }
  const std::vector<uint32_t>& resident() const { return resident_; }
  bool resident_valid() const { return valid_; }

 private:
  void store(const std::vector<uint32_t>& ids, uint32_t end, bool is_block_end);

  EngineIface& engine_;
  std::unique_ptr<PrefixCache> cache_;
  std::vector<uint32_t> resident_;
  bool valid_ = false;
  const std::vector<uint32_t>* prompt_ = nullptr;   // the prompt being prefilled
  double store_ms_ = 0;
};

const char* plan_kind_name(PrefixCache::Plan::Kind k);

}  // namespace server
