#include "runtime/memory_plan.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>
#include "loader/moe_layout.h"
#include "runtime/buffer_sizes.h"

namespace runtime {

std::string format_memory(const char* label, const MemoryComponents& c, size_t device_bytes) {
  const double gb = 1e9;
  std::string dr;
  if (c.drafter) dr = std::format(" drafter {:.3f} GB,", c.drafter / gb);   // spec 19b
  std::string buf;
  buf = std::format(
    "{}: model {:.3f} GB, kv {:.3f} GB, decode state {:.3f} GB, prefill scratch {:.3f} GB,"
    " int8 {:.3f} GB,{} total {:.3f} GB of {:.3f} GB",
    label, c.model / gb, c.kv / gb, c.decode_state / gb, c.prefill_scratch / gb,
    c.int8 / gb, dr, c.total() / gb, device_bytes / gb);
  return buf;
}

MemoryPlan plan(const model::ModelDesc& desc, uint32_t max_len, bool mtp, size_t model_bytes,
                const PrefillPath& path, KvCache kv, const DrafterPlan& drafter) {
  MemoryPlan p;
  p.drafter = drafter.bytes(max_len);   // spec 19b: 0 without a drafter
  p.max_len = max_len;
  p.mtp = mtp;
  p.kv_cache = kv;
  p.chunk_cap = path.chunk_cap;
  p.rope = model::Qwen35::rope_table_bytes(max_len);
  p.model = model_bytes + p.rope;

  const PersistentSizes ps = PersistentDims::sizes(max_len, desc, kv);
  p.kv = ps.kv_k + ps.kv_v;
  // Spec 32e: the decode scratch at the deep verify lists' 64 rows only where they can be built.
  p.decode_state = ps.total() - p.kv +
                   DecodeScratchDims::sizes(max_len, desc, DecodeScratchDims::rows(desc, mtp, kv, 1)).total();
  // Spec 15c: the MoE split (both already counted above - model_bytes is the loaded total,
  // DecodeScratchSizes::moe is in decode_state).
  p.moe_weights = loader::moe_bytes(desc);
  p.moe_scratch = moe_scratch_layout(desc).total;
  if (mtp) {
    p.mtp_buffers = MtpDims::sizes(max_len, desc, kv).total();
    // Allocated on the first prefill of an engine with the head (engine_prefill.cc).
    p.mtp_hidden = path.prefill ? engine_prefill_sizes(desc, path.chunk_cap, true, 0).mtp_hid : 0;
    p.decode_state += p.mtp_buffers + p.mtp_hidden;
  }

  if (path.prefill) {
    const PrefillScratchSizes s = PrefillScratchDims::sizes(max_len, desc, path.chunk_cap);
    // linear_l0 walks the slab for every int4 linear; on l0-int8 a MoE model's mixer linears (open-work 16,
    // l0_int8_bf16_linear), a dense model's short chunks (PERF-88) and the MTP head's bf16 linears do (step_mtp_kv).
    p.prefill_lazy += s.slab;
    // Phase 3 Task 2: a MoE model's router / a||b B slabs, made by Engine::prepare_prefill (both backends).
    p.prefill_lazy += route_slab_bytes(desc, 0, desc.layers);
    if (path.dense_bf16) p.prefill_lazy += dense_slab_bytes(desc);   // open-work 38, 0 when dense
    if (kv == KvCache::Int8) p.prefill_lazy += s.kv8_raw;
    p.prefill_scratch = s.eager() + p.prefill_lazy;
    // Engine::prepare_prefill: Int8State for the MLP intermediate (down's K), then every
    // h8 linear's rotated column scales, all before the first chunk.
    if (path.backend == PrefillBackend::L0Int8)
      p.int8 = int8_scratch_sizes(prefill_int8_max_k(desc), path.chunk_cap).total() + int8_scale_bytes(desc);
    p.moe_prefill = s.moe;   // spec 15d: inside s.eager(), 0 when dense
  }
  // Spec 32: the deferred GDN rows (with either proposer: a drafter's too - spec 39 found them unplanned there) and
  // the deep lists' chain scratch - Engine's own (EngineSizes), as the engine allocates them.
  const EngineSizes es = EngineDims::sizes(desc, kv, 1, mtp, drafter.on());
  p.deep = es.gdn_pend + es.gdn_x + es.conv_x;
  p.decode_state += p.deep + es.head_gate;   // PERF-123: and the verify lists' head choice word
  // Spec 19c: the engine's side of a drafter - 7 more GDN slots, and the prefill's tap staging.
  if (drafter.on()) {
    p.drafter_slots = es.df_slots;
    p.decode_state += p.drafter_slots;
    if (path.prefill) {
      p.drafter_taps = engine_prefill_sizes(desc, path.chunk_cap, false, drafter.n_taps).df_taps;
      p.prefill_scratch += p.drafter_taps;
    }
  }
  return p;
}

namespace {
// The largest multiple of kMaxLenQuantum in [min(kMinAutoMaxLen, cap), cap] that `fits` (non-decreasing in the
// length: the lengths that fit are a prefix), by bisection; 0 when not even the lowest fits.
template <class Fits>
uint32_t largest_that_fits(uint32_t cap, const Fits& fits) {
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum;   // in quanta
  uint32_t hi = cap / kMaxLenQuantum;
  if (!fits(lo * kMaxLenQuantum)) return 0;
  while (lo < hi) {   // invariant: lo fits
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid * kMaxLenQuantum))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}
}  // namespace

uint32_t max_len_that_fits(const model::ModelDesc& desc, bool mtp, size_t model_bytes,
                           size_t device_bytes, size_t reserve_bytes, uint32_t cap,
                           const PrefillPath& path, KvCache kv,
                           const DrafterPlan& drafter) {
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("max_len_that_fits: the cap " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) +
                                "-position quantum");
  // Every term of the plan is non-decreasing in max_len (the KV, attn_part's block
  // count, the RoPE table, the MTP head's KV, the composed pf_s / pf_p), so the
  // quanta that fit are a prefix and a bisection finds its end.
  return largest_that_fits(cap, [&](uint32_t len) {
    return plan(desc, len, mtp, model_bytes, path, kv, drafter).total() + reserve_bytes <= device_bytes;
  });
}

uint32_t prefill_chunk_cap_that_fits(const model::ModelDesc& desc, bool mtp, size_t model_bytes,
                                     size_t device_bytes, size_t reserve_bytes, uint32_t max_len,
                                     const PrefillPath& path, KvCache kv,
                                     const DrafterPlan& drafter) {
  PrefillPath wide = path;
  wide.chunk_cap = PrefillScratchDims::kCMax;
  try {
    check_prefill_chunk_cap(desc, wide.chunk_cap);
  } catch (const std::invalid_argument&) {
    return PrefillScratchDims::kCDefault;   // a model that refuses 4096-row chunks
  }
  return plan(desc, max_len, mtp, model_bytes, wide, kv, drafter).total() + reserve_bytes <= device_bytes
             ? PrefillScratchDims::kCMax
             : PrefillScratchDims::kCDefault;
}

namespace {
// plan_pool's body: plan() at `positions` (the one KV allocation: the main cache's and the head's
// layers, and the RoPE table) with slot 0's drafter at `reach` (the furthest position a sequence writes), int8 prefill's
// widening scratch at `longest` (one sequence prefills at a time), and the further sequences' state - SlotDims' at
// `table_pages` page table entries, their drafter sessions at `session_lens`, and with MTP their GDN states and hh rows.
MemoryPlan plan_sequences(const model::ModelDesc& desc, uint32_t positions, uint32_t reach, uint32_t longest,
                          const std::vector<uint32_t>& session_lens, uint32_t table_pages, bool mtp, size_t model_bytes,
                          const PrefillPath& path, KvCache kv, const DrafterPlan& drafter) {
  const uint32_t seqs = static_cast<uint32_t>(session_lens.size());
  MemoryPlan p = plan(desc, positions, mtp, model_bytes, path, kv, drafter);
  // Spec 31: the pool's drafter reaches one request's cap; the RoPE table stays at the pool's positions (the engine's
  // max_len, the loader's table - one number, runtime::Engine).
  if (reach != positions) p.drafter = drafter.bytes(reach);
  // Spec 32e: several slots build no deep verify list - their decode scratch keeps kWideM rows, not plan()'s; plan 40a
  // Task 3: kWideRowsM beside a proposer on Qwen3.8 (the 32-row lists across sequences).
  p.decode_state -=
      DecodeScratchDims::sizes(positions, desc, DecodeScratchDims::rows(desc, mtp, kv, 1)).total() -
      DecodeScratchDims::sizes(positions, desc, DecodeScratchDims::rows(desc, mtp || drafter.on(), kv, seqs)).total();
  // 3j: int8 prefill's widening scratch holds one sequence's rows (one prefills at a time, slot-relative), so the
  // engine sizes it at the longest slot (PrefillScratch::kv8_len), not at the sum plan() counted.
  if (path.prefill && kv == KvCache::Int8) {
    const size_t extra = PrefillScratchDims::sizes(p.max_len, desc, path.chunk_cap).kv8_raw -
                         PrefillScratchDims::sizes(longest, desc, path.chunk_cap).kv8_raw;
    p.prefill_lazy -= extra;
    p.prefill_scratch -= extra;
  }
  p.slot_state = SlotDims::sizes(desc, seqs, mtp, table_pages).total();
  if (seqs > 1 && p.drafter_slots != 0) {   // plan 40a Task 2: several sequences beside a drafter hold no verify slots
    p.decode_state -= p.drafter_slots;
    p.drafter_slots = 0;
  }
  for (uint32_t q = 1; q < seqs; ++q) p.slot_state += drafter.session(session_lens[q]);   // plan 19v3b: their sessions
  if (seqs > 1) p.slot_state += drafter.rows;   // spec 39: and the row tables and row taps (unplanned before)
  if (seqs > 1) {   // spec 32: several slots build no deep list; plan 31 Task 13 / 40a: each its deferred GDN rows
    const EngineSizes es = EngineDims::sizes(desc, kv, seqs, mtp, drafter.on());
    p.decode_state -= p.deep;
    p.deep = es.gdn_pend + es.gdn_x + es.conv_x;
    p.decode_state += p.deep;
  }
  if (mtp) {   // the further slots' GDN states and hh rows: MtpBuffers at `seqs`
    // Spec 32: may be negative - one sequence's hh rows (the deep lists' 65) outweigh the slots'.
    const int64_t more = int64_t(MtpDims::sizes(p.max_len, desc, kv, MtpDims::kSlots, seqs).total()) -
                         int64_t(MtpDims::sizes(p.max_len, desc, kv).total());
    p.slot_state = size_t(int64_t(p.slot_state) + more);
    p.mtp_buffers = size_t(int64_t(p.mtp_buffers) + more);
  }
  p.decode_state += p.slot_state;   // plan() counted MtpBuffers at one sequence
  return p;
}
}  // namespace

uint32_t pool_table_pages(const PoolShape& s) { return s.seqs * kv_pages(s.cap); }

MemoryPlan plan_pool(const model::ModelDesc& desc, const PoolShape& s, bool mtp, size_t model_bytes,
                     const PrefillPath& path, KvCache kv, const DrafterPlan& drafter) {
  if (s.seqs < 2 || s.seqs > SlotDims::kMaxSeqs)
    throw std::invalid_argument("runtime::plan_pool: " + std::to_string(s.seqs) + " sequences (2.." +
                                std::to_string(SlotDims::kMaxSeqs) + ")");
  if (s.cap == 0 || s.cap % kMaxLenQuantum != 0 || uint64_t(s.cap) > s.positions())
    throw std::invalid_argument("runtime::plan_pool: a request's cap " + std::to_string(s.cap) + " in a pool of " +
                                std::to_string(s.positions()) + " positions");
  const std::vector<uint32_t> reach(s.seqs, s.cap);   // every sequence may grow to the cap
  MemoryPlan p = plan_sequences(desc, static_cast<uint32_t>(s.positions()), s.cap, s.cap, reach, pool_table_pages(s),
                                mtp, model_bytes, path, kv, drafter);
  p.slot_lens = reach;
  p.pool_pages = s.pages;
  return p;
}

PoolShape pool_that_fits(const model::ModelDesc& desc, uint32_t seqs, bool mtp, size_t model_bytes, size_t device_bytes,
                         size_t reserve_bytes, uint32_t limit, const PrefillPath& path, KvCache kv,
                         const DrafterPlan& drafter) {
  const uint32_t top = limit / kMaxLenQuantum * kMaxLenQuantum;
  if (top == 0)
    throw std::invalid_argument("pool_that_fits: the cap " + std::to_string(limit) + " is below one " +
                                std::to_string(kMaxLenQuantum) + "-position quantum");
  const auto shape = [&](uint32_t pages) { return PoolShape{seqs, pages, std::min(pages * kKvPage, top)}; };
  const auto fits = [&](uint32_t pages) {
    return plan_pool(desc, shape(pages), mtp, model_bytes, path, kv, drafter).total() + reserve_bytes <= device_bytes;
  };
  // Every term is non-decreasing in the pages (the KV, and the cap-sized terms while the cap grows with them), so
  // the counts that fit are a prefix: bisect it. No more than `seqs` requests at the cap can use.
  uint32_t lo = kv_pages(std::min(kMinAutoMaxLen, top)), hi = seqs * kv_pages(top);
  if (!fits(lo)) return {seqs, 0, 0};
  while (lo < hi) {   // invariant: lo fits
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid))
      lo = mid;
    else
      hi = mid - 1;
  }
  return shape(lo);
}

std::string describe(const MemoryPlan& p, size_t device_bytes, size_t reserve_bytes) {
  const double gb = 1e9;
  const std::string label = "plan at max_len " + std::to_string(p.max_len);
  std::string s = format_memory(label.c_str(), p, device_bytes);
  std::string buf;
  buf = std::format("; + reserve {:.3f} GB = {:.3f} GB (RoPE {:.3f} GB in model",
     reserve_bytes / gb, (p.total() + reserve_bytes) / gb, p.rope / gb);
  s += buf;
  if (p.mtp) {
    buf = std::format(", MTP {:.3f} GB in decode state", (p.mtp_buffers + p.mtp_hidden + p.deep) / gb);
    s += buf;
  }
  if (p.moe_weights != 0) {
    buf = std::format(", MoE experts {:.3f} GB in model", p.moe_weights / gb);
    s += buf;
  }
  if (p.moe_prefill != 0) {
    buf = std::format(", MoE prefill {:.3f} GB in prefill scratch", p.moe_prefill / gb);
    s += buf;
  }
  if (p.prefill_lazy != 0) {
    buf = std::format(", lazy {:.3f} GB in prefill scratch", p.prefill_lazy / gb);
    s += buf;
  }
  if (p.drafter_slots != 0) {   // spec 19c
    buf = std::format(", drafter's verify slots {:.3f} GB in decode state", p.drafter_slots / gb);
    s += buf;
  }
  if (p.drafter_taps != 0) {
    buf = std::format(", drafter's prefill taps {:.3f} GB in prefill scratch", p.drafter_taps / gb);
    s += buf;
  }
  if (p.pool_pages != 0) {   // spec 31
    buf = std::format(", {:d} requests sharing a KV pool of {:d} pages of {:d}, one up to {:d}, their state {:.3f} GB in"
                      " decode state", p.slot_lens.size(), p.pool_pages, kKvPage, p.slot_lens[0], p.slot_state / gb);
    s += buf;
  }
  if (p.kv_cache == KvCache::Int8) s += ", int8 KV cache";   // spec 12b; bf16 says nothing
  if (p.chunk_cap != PrefillScratchDims::kCDefault)   // plan 5d; 2048 says nothing
    s += ", prefill chunk cap " + std::to_string(p.chunk_cap);
  return s + ")";
}

}  // namespace runtime
