#pragma once
// Spec 31 §5.4 (plan 31 Task 3): the slots' KV page pool on the host. A card's KV is `total` pages of kKvPage
// positions (runtime/buffer_sizes.h); `seqs` sequences each own a table of the pages that hold their positions 0,
// kKvPage, 2 kKvPage, ... - the host mirror of the device page table Engine binds (Engine::kv_page_table). A
// sequence takes pages before a step writes them and gives them all back when it is reset or evicted.
//
// Every page has a reference count: 0 on the free list, 1 in one table. Above 1 is a page shared on purpose
// (read-only prefix pages, spec 31 §9: share, make_private before a write); take and release keep the count, so a
// shared page returns to the free list only when its last holder releases it, then zeroed whole. A release names the rows to zero on the pages it frees - every row below the
// sequence's written bound, none above (PERF-90's written-range reset, per page): the engine zeroes those before a
// page is handed out again, so a fresh sequence's pages are as zero as a fresh slot's.
//
// The free list is a stack: a fresh pool hands out pages 0, 1, 2, ... in order, and a released page is the next one
// taken. Device-free and header-only.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/buffer_sizes.h"

namespace runtime {

class KvPool {
 public:
  // `max_pages`: the longest table a sequence may hold (the per-request cap in pages).
  KvPool(uint32_t total, uint32_t seqs, uint32_t max_pages)
    pre(total != 0 && seqs != 0 && max_pages != 0)
      : refs_(total, 0), whole_(total, false), tables_(seqs), max_pages_(max_pages) {
    free_.reserve(total);
    for (uint32_t p = total; p-- > 0;) free_.push_back(p);
  }

  // Pages until `seq`'s table covers positions [0, upto_pos). False - and nothing changed - when the free pages do
  // not cover it. Past the per-sequence cap is the caller's error (a contract violation).
  bool take(uint32_t seq, uint32_t upto_pos)
    pre(kv_pages(upto_pos) <= max_pages()) {
    std::vector<uint32_t>& t = tables_.at(seq);
    const uint32_t want = kv_pages(upto_pos);
    if (want <= t.size()) return true;
    const size_t more = want - t.size();
    if (more > free_.size()) return false;
    for (size_t i = 0; i < more; ++i) {
      const uint32_t p = free_.back();
      free_.pop_back();
      refs_[p] = 1;
      t.push_back(p);
    }
    return true;
  }

  // A page and its rows [0, rows) to zero.
  struct Zero {
    uint32_t page = 0, rows = 0;
  };
  // Each of `seq`'s pages loses a reference; at none it returns to the free list. The table is left empty. `written`:
  // the sequence's written bound (its positions [0, written) may hold KV). Returns, for each page that returns to the
  // free list with rows below the bound, those rows: whole pages below it, the bound's page up to written % kKvPage,
  // none above. A bound past the sequence's pages is the caller's error (a contract violation).
  std::vector<Zero> release(uint32_t seq, uint32_t written)
    pre(uint64_t(written) <= uint64_t(pages_of(seq).size()) * kKvPage) {
    std::vector<uint32_t>& t = tables_.at(seq);
    std::vector<Zero> zero;
    for (size_t i = t.size(); i-- > 0;) {
      if (--refs_[t[i]] != 0) continue;
      free_.push_back(t[i]);
      const uint64_t first = uint64_t(i) * kKvPage;
      if (whole_[t[i]])   // a page that was shared: its every row was written (by its first holder)
        zero.push_back({t[i], kKvPage});
      else if (written > first)
        zero.push_back({t[i], static_cast<uint32_t>(std::min<uint64_t>(written - first, kKvPage))});
      whole_[t[i]] = false;
    }
    t.clear();
    std::reverse(zero.begin(), zero.end());   // in table order
    return zero;
  }

  // Spec 31 §9 (stage 2): `seq`'s next table entry becomes `from`'s page `index` - the positions [index kKvPage,
  // (index + 1) kKvPage) both hold, read-only while both do (the page's count says how many). `seq` must hold exactly
  // `index` pages and `from` more than `index`; throws std::invalid_argument otherwise. Returns the page.
  uint32_t share(uint32_t seq, uint32_t index, uint32_t from) {
    std::vector<uint32_t>& t = tables_.at(seq);
    const std::vector<uint32_t>& f = tables_.at(from);
    if (seq == from || t.size() != index || f.size() <= index || index + 1 > max_pages_)
      throw std::invalid_argument("KvPool::share: page " + std::to_string(index) + " of a sequence holding " +
                                  std::to_string(f.size()) + " into one holding " + std::to_string(t.size()));
    const uint32_t p = f[index];
    ++refs_[p];
    whole_[p] = true;
    t.push_back(p);
    return p;
  }
  // Before `seq` writes into its page `index`: when the page is shared, `seq` gets a fresh page in its place (the
  // others keep the old one); `old` and `fresh` say which (equal when it was its own). False - and nothing changed -
  // when no page is free.
  bool make_private(uint32_t seq, uint32_t index, uint32_t* old, uint32_t* fresh) {
    std::vector<uint32_t>& t = tables_.at(seq);
    *old = *fresh = t.at(index);
    if (refs_[*old] == 1) return true;
    if (free_.empty()) return false;
    *fresh = free_.back();
    free_.pop_back();
    refs_[*fresh] = 1;
    --refs_[*old];
    t[index] = *fresh;
    return true;
  }

  uint32_t pages_free() const { return static_cast<uint32_t>(free_.size()); }
  uint32_t pages_total() const { return static_cast<uint32_t>(refs_.size()); }
  uint32_t max_pages() const { return max_pages_; }
  uint32_t seqs() const { return static_cast<uint32_t>(tables_.size()); }
  // `seq`'s table: entry i is the page holding its positions [i kKvPage, (i + 1) kKvPage).
  const std::vector<uint32_t>& pages_of(uint32_t seq) const { return tables_.at(seq); }
  uint32_t refs(uint32_t page) const { return refs_.at(page); }

 private:
  std::vector<uint32_t> refs_;                 // per page
  std::vector<bool> whole_;                    // per page: shared once, so all its rows were written
  std::vector<uint32_t> free_;                 // a stack; back() is taken next
  std::vector<std::vector<uint32_t>> tables_;  // per sequence
  uint32_t max_pages_;
};

}  // namespace runtime
