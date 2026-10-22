#pragma once
// Spec 13 §9 (plan 13a): the row table - a decode step's rows as (slot, position, GDN state index) - and the host
// arithmetic around it: the rules a table must keep, the slots' KV bases, and the memory of a layout by spec 13
// §9.1's method. Device-free and header-only. Plan 13a's probe (tools/probe/probe_batch_step, runtime::build_rows
// under B70_BATCH_PROBE) is its only user so far; plan 13b promotes it into the engine and the planner.
//
// **The device layout.** BatchRowTable is 256 B of shared memory the host writes before a replay; the kernels'
// BATCH_ROWS builds (gdn_step.cl, attn.cl, attn_v2.cl, kv8.cl) read row m's words at m * kBatchRowWords with the
// indices of src/kernels/CMakeLists.txt's BROW_DEFINES (kernels::batch_row, src/kernels/batch_kernels.h - the host
// mirror, static_asserted against this struct there). The row count is the list's Control::n_active, as for every
// kernel of the list.
//
//   pos      the row's position in its sequence (RoPE, the causal bound, the conv ring's slot)
//   kv_base  the slot's first position in the KV allocation: the row's K / V row is kv_base + pos
//   src      the GDN state index read at the slot's FIRST row (index 0 = PersistentBuffers::gdn_state, i >= 1 the
//            pool's slot i - 1); later rows of the slot carry the state in registers
//   dst      the GDN state index the state after THIS row is written to
//   slot     the slot (its conv ring; rows of one slot are contiguous - a run)
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"

namespace runtime {

struct BatchRow {
  uint32_t pos = 0, kv_base = 0, src = 0, dst = 0, slot = 0;
  uint32_t pad[3] = {0, 0, 0};
};
inline constexpr uint32_t kBatchRowWords = 8;
static_assert(sizeof(BatchRow) == kBatchRowWords * 4, "a row is eight words");
struct BatchRowTable {
  BatchRow row[DecodeScratchDims::kM];
};
static_assert(sizeof(BatchRowTable) == 256, "the table is 8 rows of 32 B");

// One slot of a layout: its positions and whether it carries the MTP head's state (spec 13 §9.2: the engine's MTP
// setting holds for every slot of a served layout; the planner's arithmetic below takes it per slot so the
// alternatives §9.2 rules out can be priced too).
struct SlotSpec {
  uint32_t max_len = 0;
  bool mtp = true;
};

// The slots' KV ranges: one allocation at the slots' sum, slot s at the positions after slots 0..s-1.
struct BatchLayout {
  std::vector<SlotSpec> slots;
  uint32_t total() const {
    uint64_t t = 0;
    for (const SlotSpec& s : slots) t += s.max_len;
    if (t > 0xFFFFFFFFull) throw std::invalid_argument("BatchLayout: the slots' positions overflow 32 bits");
    return static_cast<uint32_t>(t);
  }
  uint32_t kv_base(uint32_t slot) const {
    if (slot >= slots.size()) throw std::out_of_range("BatchLayout: no slot " + std::to_string(slot));
    uint32_t b = 0;
    for (uint32_t s = 0; s < slot; ++s) b += slots[s].max_len;
    return b;
  }
};

// One slot's rows in a step: `rows` consecutive positions from `pos`, the state read from `src` at the first and
// the state after row j written to dst[j].
struct RowRun {
  uint32_t slot = 0, pos = 0, rows = 1, src = 0;
  std::vector<uint32_t> dst;
};

// The rules a table keeps - the kernels assume them and check none:
//   1..kM rows; a slot < the layout's, a row's position < its slot's max_len, kv_base the slot's;
//   a slot's rows contiguous (one run per slot), at consecutive positions, with one src;
//   every dst distinct, and no dst another run's src (every work-group walks the runs in order, so a later run would
//   read a state an earlier run of the same step had overwritten; its OWN src it may write - the tile is read first).
inline void check_rows(const std::vector<BatchRow>& rows, const BatchLayout& lay) {
  auto fail = [](const std::string& w) { throw std::invalid_argument("runtime::check_rows: " + w); };
  if (rows.empty() || rows.size() > DecodeScratchDims::kM)
    fail(std::to_string(rows.size()) + " rows; a step carries 1.." + std::to_string(DecodeScratchDims::kM));
  std::vector<bool> seen(lay.slots.size(), false);
  std::vector<uint32_t> srcs;
  for (size_t m = 0; m < rows.size(); ++m) {
    const BatchRow& r = rows[m];
    if (r.slot >= lay.slots.size()) fail("row " + std::to_string(m) + " names slot " + std::to_string(r.slot));
    if (r.pos >= lay.slots[r.slot].max_len)
      fail("row " + std::to_string(m) + " at position " + std::to_string(r.pos) + " is past slot " +
           std::to_string(r.slot) + "'s " + std::to_string(lay.slots[r.slot].max_len));
    if (r.kv_base != lay.kv_base(r.slot)) fail("row " + std::to_string(m) + "'s kv_base is not its slot's");
    const bool first = m == 0 || rows[m - 1].slot != r.slot;
    if (first) {
      if (seen[r.slot]) fail("slot " + std::to_string(r.slot) + "'s rows are split (a slot is one run)");
      seen[r.slot] = true;
      srcs.push_back(r.src);
    } else {
      if (r.pos != rows[m - 1].pos + 1) fail("row " + std::to_string(m) + " does not follow its run's position");
      if (r.src != rows[m - 1].src) fail("row " + std::to_string(m) + " changes its run's src");
    }
  }
  for (size_t i = 0; i < srcs.size(); ++i)
    for (size_t j = i + 1; j < srcs.size(); ++j)
      if (srcs[i] == srcs[j]) fail("two runs read one src " + std::to_string(srcs[i]));
  for (size_t m = 0; m < rows.size(); ++m) {
    for (size_t n = m + 1; n < rows.size(); ++n)
      if (rows[m].dst == rows[n].dst) fail("rows " + std::to_string(m) + " and " + std::to_string(n) + " write one dst");
    for (size_t n = 0; n < rows.size(); ++n)
      if (rows[n].slot != rows[m].slot && rows[m].dst == rows[n].src)
        fail("row " + std::to_string(m) + " writes another run's src " + std::to_string(rows[m].dst));
  }
}

// The runs as a table, checked.
inline std::vector<BatchRow> make_rows(const BatchLayout& lay, const std::vector<RowRun>& runs) {
  std::vector<BatchRow> t;
  for (const RowRun& run : runs) {
    if (run.rows == 0 || run.dst.size() != run.rows)
      throw std::invalid_argument("runtime::make_rows: slot " + std::to_string(run.slot) + "'s run has " +
                                  std::to_string(run.rows) + " rows and " + std::to_string(run.dst.size()) + " dsts");
    for (uint32_t j = 0; j < run.rows; ++j) {
      BatchRow r;
      r.pos = run.pos + j;
      r.kv_base = run.slot < lay.slots.size() ? lay.kv_base(run.slot) : 0;
      r.src = run.src;
      r.dst = run.dst[j];
      r.slot = run.slot;
      t.push_back(r);
    }
  }
  check_rows(t, lay);
  return t;
}

// The table's shape: the run lengths in row order (what a captured list's per-run copies depend on).
inline std::vector<uint32_t> run_lengths(const std::vector<BatchRow>& rows) {
  std::vector<uint32_t> n;
  for (size_t m = 0; m < rows.size(); ++m)
    if (m == 0 || rows[m - 1].slot != rows[m].slot)
      n.push_back(1);
    else
      ++n.back();
  return n;
}

// --- the memory of a layout (spec 13 §9.1's method) --------------------------------------------------------------
//
// The code's own formulas (buffer_sizes.h, memory_plan.h) per slot and shared:
//   kv       per slot, K and V at the slot's max_len over the FA layers + the MTP head's layer when the slot has MTP
//   state    per slot, the GDN states (MtpDims::kSlots with MTP, 1 without) and one conv ring
//   weights  the loaded weights in the serving form (the int8 head), the MTP head's when any slot has MTP, the RoPE
//            table at the longest slot
//   scratch  runtime::plan's decode state without the persistent state and the head's KV / spec states, + its prefill
//            scratch + int8 (l0-int8, the default) - shared by the slots, one prefill at a time
// `weights` excludes the MTP head and the RoPE table: memory_plan_test's kQwenInt8Weights / kAgnesInt8Weights.
struct BatchMemory {
  size_t kv = 0, state = 0, weights = 0, scratch = 0;
  size_t total() const { return kv + state + weights + scratch; }
};
inline BatchMemory batch_memory(const model::ModelDesc& d, const std::vector<SlotSpec>& slots, KvCache kv,
                                size_t weights, size_t mtp_head) {
  BatchMemory m;
  uint32_t longest = 0;
  bool any_mtp = false;
  for (const SlotSpec& s : slots) {
    const PersistentSizes ps = PersistentDims::sizes(s.max_len, d, kv);
    m.kv += ps.kv_k + ps.kv_v;
    m.state += ps.gdn_state * (s.mtp ? MtpDims::kSlots : 1) + ps.conv_ring;
    if (s.mtp) {
      const MtpSizes ms = MtpDims::sizes(s.max_len, d, 0, kv);
      m.kv += ms.kv_k + ms.kv_v;
    }
    longest = s.max_len > longest ? s.max_len : longest;
    any_mtp = any_mtp || s.mtp;
  }
  m.weights = weights + (any_mtp ? mtp_head : 0) + model::Qwen35::rope_table_bytes(longest);
  const MemoryPlan p = plan(d, longest, any_mtp, 0, PrefillPath{}, DraftVocabPlan{}, kv);
  const PersistentSizes ps = PersistentDims::sizes(longest, d, kv);
  m.scratch = p.decode_state - ps.gdn_state - ps.conv_ring + p.prefill_scratch + p.int8;
  if (any_mtp) {
    const MtpSizes ms = MtpDims::sizes(longest, d, 0, kv);
    m.scratch -= ms.gdn_spec + ms.kv_k + ms.kv_v;
  }
  return m;
}

// One long slot and `n - 1` short ones, every slot's MTP as given (spec 13 §9.1's layouts).
inline std::vector<SlotSpec> main_and_short(uint32_t n, uint32_t main_len, uint32_t short_len, bool mtp) {
  std::vector<SlotSpec> v{{main_len, mtp}};
  for (uint32_t i = 1; i < n; ++i) v.push_back({short_len, mtp});
  return v;
}

// The largest n <= max_n whose main_and_short layout fits `budget` bytes (the device less the reserve); 0 if none
// (spec 13 §9.2: the number the refusal names).
inline uint32_t largest_batch(const model::ModelDesc& d, uint32_t max_n, uint32_t main_len, uint32_t short_len,
                              bool mtp, KvCache kv, size_t weights, size_t mtp_head, size_t budget) {
  for (uint32_t n = max_n; n >= 1; --n)
    if (batch_memory(d, main_and_short(n, main_len, short_len, mtp), kv, weights, mtp_head).total() <= budget)
      return n;
  return 0;
}

}  // namespace runtime
