#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/qwen4exp_ple_hash.h"   // the hash in exact integers, Q4PleScale (header-only)
#include "loader/safetensors.h"
#include "model/qwen4exp.h"

// Spec 21b Task 4: Qwen3.8-Flash-Next's PLE n-gram table on the host - the int8 file (tools/quantize/
// qwen4exp/ple_int8.py, decision 7's proposal as built) read by mmap, checked against the descriptor, and
// copied into 16 + 16 host-USM ranges the device reads zero-copy (spec 21 §4.3: 21c's q4_ple_gather
// dereferences them through the device pointer table `ptrs`, by an index it computes itself).
//
// **Why per head.** One 51.8 GB allocation would exceed the largest pinned allocation ever measured (48 GiB,
// docs/probe-prefix-cache-2026-09-27.md §1); head h's rows [offset_h, offset_h + prime_h) are contiguous in
// the hash's id space, so each head is one ~3.2 GB range (+ its scales), the "PLE table's shards" spec 22's
// P0.5 measures - and the allocation, the pointer passing and the load-time checks below are the mechanism
// spec 22's expert mirror reuses.
//
// The file (one safetensors set, an index json): per head h `ple.h<h>.q` I8 [prime_h][dim] and
// `ple.h<h>.s` F32 or BF16 [prime_h] (one dtype for all heads), `ple.layer_multipliers` I64 [3],
// `ple.ngram_heads_vocab_sizes` / `ple.ngram_heads_offsets` I64 [heads]. Row r of head h dequantises to
// bf16(float(q) x float(s)) (spec 9's row rule; qwen4exp_ref.PleTable's int8 reader).
namespace loader {

// The int8 file, mmapped (a SafetensorsSet over its directory). Throws naming the tensor when a head's
// q / s is missing or misshapen, the scale dtypes differ, or the I64 tensors are absent.
struct Q4PleHost {
  explicit Q4PleHost(const std::string& dir);
  Q4PleScale scale() const { return scale_; }
  uint32_t heads() const { return uint32_t(rows_.size()); }
  uint32_t dim() const { return dim_; }
  uint64_t rows(uint32_t h) const { return rows_.at(h); }
  const int8_t* q(uint32_t h) const { return q_.at(h); }
  const void* s(uint32_t h) const { return s_.at(h); }
  size_t q_bytes(uint32_t h) const { return size_t(rows(h)) * dim_; }
  size_t s_bytes(uint32_t h) const { return size_t(rows(h)) * q4_ple_scale_bytes(scale_); }
  size_t bytes() const;   // every head's q + s
  // Row r of head h as bf16(float(q) x float(s)) - the reference's dequant (the test's and 21c's check).
  void dequant_row(uint32_t h, uint64_t r, uint16_t* out) const;
  std::array<uint64_t, 3> multipliers{};   // the file's I64 tensors
  std::vector<uint64_t> sizes, offsets;
  std::string dir;

 private:
  std::unique_ptr<SafetensorsSet> set_;
  Q4PleScale scale_ = Q4PleScale::F32;
  uint32_t dim_ = 0;
  std::vector<uint64_t> rows_;
  std::vector<const int8_t*> q_;
  std::vector<const void*> s_;
};

// The file against the descriptor, before a byte is pinned: heads = d.ple_heads, dim = d.ple_dim, the I64
// tensors = the formula for (d.vocab, d.ngram, ple index 0, d.ple_seed) and d.ple_base, each head's rows =
// its size. Throws naming the tensor and both values.
void check_q4_ple(const model::Qwen4ExpDesc& d, const Q4PleHost& h);

// The host-memory rule (the measured pinned cap's): the table must fit in MemAvailable less a 16 GiB
// margin. Throws naming both numbers; `mem_available` 0 = unknown (no /proc/meminfo): passes.
inline constexpr size_t kQ4PleHostMargin = size_t(16) << 30;
void check_q4_ple_host_fit(size_t table_bytes, size_t mem_available);
// MemAvailable from /proc/meminfo in bytes; 0 when unreadable (not Linux).
size_t q4_mem_available();

// "<snapshot>-ple-int8/" (the snapshot path without its trailing slash), or $B70_Q4_PLE when set. Throws,
// naming the directory and ple_int8.py's command line, when it holds no model.safetensors.index.json.
std::string q4_ple_dir(const std::string& snapshot_dir);

// The tag the loader writes at the start of every 2 MiB page of every host range before the data lands
// (the alias check: a page two ranges share reads back the later tag), and the page size.
inline constexpr size_t kQ4PlePage = size_t(2) << 20;
inline uint64_t q4_ple_tag(uint32_t range, uint64_t page) {
  return 0x5134504C45000000ull ^ (uint64_t(range) << 40) ^ page;   // "Q4PLE" | range | page
}

struct Q4PleTable {                        // 16 host-USM ranges (q) + 16 (scales); device-visible
  std::vector<std::unique_ptr<l0::Mem>> q, s;
  std::unique_ptr<l0::Mem> ptrs;           // device u64 [32]: q[0..15], s[0..15] - the kernel's table
  size_t bytes = 0;
  Q4PleScale scale = Q4PleScale::F32;
  uint64_t tag_checksum = 0;               // host-side: the sum of the tags written and read back
  // The data's first u64 of every 2 MiB page of every range, range-major (q[0..15] then s[0..15]): what
  // 21c's q4_ple_check reads through `ptrs` on the device (the loader read them back on the host).
  std::vector<uint64_t> page_words;
  size_t rows_checked = 0;                 // sampled rows compared with the file after the copy
  size_t mem_before = 0, mem_after = 0;    // MemAvailable around the pinning (0: unknown)
  double seconds = 0;
};
// Checks the file against the descriptor (check_q4_ple), refuses when the bytes do not fit the host
// (check_q4_ple_host_fit), then per range: allocates host USM (`zeMemAllocHost` through `ctx`), writes and
// reads back a tag per 2 MiB page (all ranges before any data), copies the head's rows, compares sampled rows
// with the file and records each page's first word; finally the device pointer table on `ctx`'s device.
Q4PleTable load_q4_ple(l0::Context& ctx, const model::Qwen4ExpDesc& d, const std::string& ple_dir);

}  // namespace loader
