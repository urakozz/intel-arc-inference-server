#include "runtime/pipeline_prefill_plan.h"

#include <stdexcept>
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"
#include "runtime/prefill_chunks.h"

namespace runtime {
namespace {
size_t round_up(size_t n, size_t q) { return (n + q - 1) / q * q; }
constexpr size_t kTimestampBytes = size_t(kPpPfDepth) * 2 * 8;   // [slot][start, end] uint64
}  // namespace

std::vector<PpChunk> pp_prefill_chunks(uint32_t base, size_t n, uint32_t chunk, bool hooked,
                                       uint32_t block) {
  if (n == 0) throw std::invalid_argument("runtime::pp_prefill_chunks: no ids");
  if (chunk == 0) throw std::invalid_argument("runtime::pp_prefill_chunks: a chunk of 0 rows");
  if (hooked && block == 0) throw std::invalid_argument("runtime::pp_prefill_chunks: block 0");
  std::vector<PpChunk> out;
  for (size_t off = 0; off < n;) {
    PpChunk c;
    c.pos = base + uint32_t(off);
    c.rows = prefill_chunk_rows(c.pos, n - off, chunk, hooked, block);
    off += c.rows;
    // Engine::prefill: a hooked chunk that is not the last and ends on a block end.
    c.hook = hooked && off < n && c.end() % block == 0;
    out.push_back(c);
  }
  return out;
}

const char* pp_pf_op_name(PpPfOp op) {
  switch (op) {
    case PpPfOp::Run0: return "Run0";
    case PpPfOp::Run1: return "Run1";
    case PpPfOp::Wait0: return "Wait0";
    case PpPfOp::Wait1: return "Wait1";
    case PpPfOp::Hook: return "Hook";
    case PpPfOp::Head: return "Head";
    case PpPfOp::Drain: return "Drain";
  }
  return "?";
}

std::vector<PpPfStep> pp_prefill_schedule(const std::vector<PpChunk>& chunks) {
  const uint32_t n = uint32_t(chunks.size());
  if (n == 0) throw std::invalid_argument("runtime::pp_prefill_schedule: no chunks");
  std::vector<PpPfStep> s;
  s.reserve(size_t(n) * 5 + 4);
  const auto retire = [&](uint32_t j) {   // both devices done with chunk j, then its hook
    s.push_back({PpPfOp::Wait0, j});
    s.push_back({PpPfOp::Wait1, j});
    if (chunks[j].hook) s.push_back({PpPfOp::Hook, j});
  };
  for (uint32_t j = 0; j < n; ++j) {
    if (j >= kPpPfDepth) retire(j - kPpPfDepth);
    s.push_back({PpPfOp::Run0, j});
    s.push_back({PpPfOp::Run1, j});
  }
  // The tail: chunk n - 2's hook still reads the shadows (chunk n - 1 may be running); the
  // head goes on device 1's list after its last chunk, once that hook is done with the
  // Control blocks; then the last chunk's waits and the drain.
  if (n >= kPpPfDepth) retire(n - kPpPfDepth);
  s.push_back({PpPfOp::Head, n - 1});
  s.push_back({PpPfOp::Wait0, n - 1});
  s.push_back({PpPfOp::Wait1, n - 1});
  s.push_back({PpPfOp::Drain, n});
  return s;
}

std::string pp_prefill_check(const std::vector<PpPfStep>& steps,
                             const std::vector<PpChunk>& chunks) {
  const uint32_t n = uint32_t(chunks.size());
  constexpr size_t kNone = size_t(-1);
  // When (the step index) each event happened; kNone = not yet.
  std::vector<size_t> run[2], wait[2], hook;
  for (auto* v : {&run[0], &run[1], &wait[0], &wait[1], &hook}) v->assign(n, kNone);
  size_t head = kNone, drain = kNone;
  const auto next_hooked = [&](uint32_t from) {
    while (from < n && !chunks[from].hook) ++from;
    return from;
  };
  uint32_t next_run[2] = {0, 0}, next_wait[2] = {0, 0}, next_hook = next_hooked(0);
  const auto at = [](size_t i, const PpPfStep& s) {
    return "step " + std::to_string(i) + " (" + pp_pf_op_name(s.op) + " " + std::to_string(s.chunk) +
           "): ";
  };
  for (size_t i = 0; i < steps.size(); ++i) {
    const PpPfStep& s = steps[i];
    if (drain != kNone) return at(i, s) + "after the Drain";
    const uint32_t j = s.chunk;
    switch (s.op) {
      case PpPfOp::Run0:
      case PpPfOp::Run1: {
        const uint32_t dev = s.op == PpPfOp::Run0 ? 0 : 1;
        if (j != next_run[dev]) return at(i, s) + "device " + std::to_string(dev) + "'s chunks out of order";
        if (j >= n) return at(i, s) + "past the last chunk";
        if (head != kNone) return at(i, s) + "after the Head";
        if (dev == 1 && run[0][j] == kNone) return at(i, s) + "device 1 runs a chunk device 0 has not";
        if (j >= kPpPfDepth) {
          const uint32_t k = j - kPpPfDepth;
          if (wait[0][k] == kNone || wait[1][k] == kNone)
            return at(i, s) + "chunk " + std::to_string(k) +
                   " (the same slot, Control, events and shadow) is not yet waited for on both "
                   "devices - the back-pressure rule";
          if (chunks[k].hook && hook[k] == kNone)
            return at(i, s) + "chunk " + std::to_string(k) + "'s hook has not run and this chunk"
                   " overwrites its shadow";
        }
        run[dev][j] = i;
        ++next_run[dev];
        break;
      }
      case PpPfOp::Wait0:
      case PpPfOp::Wait1: {
        const uint32_t dev = s.op == PpPfOp::Wait0 ? 0 : 1;
        if (j != next_wait[dev]) return at(i, s) + "device " + std::to_string(dev) + "'s waits out of order";
        if (j >= n || run[dev][j] == kNone) return at(i, s) + "waits for a chunk not run";
        wait[dev][j] = i;
        ++next_wait[dev];
        break;
      }
      case PpPfOp::Hook: {
        if (j >= n || !chunks[j].hook) return at(i, s) + "a hook for a chunk that has none";
        if (j != next_hook) return at(i, s) + "hooks out of order";
        if (wait[0][j] == kNone || wait[1][j] == kNone)
          return at(i, s) + "the hook runs before both devices have finished the chunk";
        if (head != kNone) return at(i, s) + "the hook runs after the Head";
        hook[j] = i;
        next_hook = next_hooked(j + 1);
        break;
      }
      case PpPfOp::Head:
        if (head != kNone) return at(i, s) + "a second Head";
        if (j + 1 != n || run[1][n - 1] == kNone) return at(i, s) + "the Head before device 1's last chunk";
        head = i;
        break;
      case PpPfOp::Drain:
        drain = i;
        break;
    }
  }
  for (uint32_t j = 0; j < n; ++j) {
    for (uint32_t dev = 0; dev < 2; ++dev) {
      if (run[dev][j] == kNone) return "device " + std::to_string(dev) + " never runs chunk " + std::to_string(j);
      if (wait[dev][j] == kNone) return "chunk " + std::to_string(j) + " is never waited for on device " + std::to_string(dev);
    }
    if (chunks[j].hook && hook[j] == kNone) return "chunk " + std::to_string(j) + "'s hook never runs";
  }
  if (head == kNone) return "no Head";
  if (drain == kNone) return "no Drain";
  return "";
}

PpPfLinkLayout pp_prefill_link_layout(const model::ModelDesc& d) {
  PpPfLinkLayout l;
  const PrefillScratchSizes s = PrefillScratchDims::sizes(kMinAutoMaxLen, d);   // rows and sums: max_len-free
  l.resid_bytes = s.resid;
  l.sumsq_bytes = s.norm_sumsq;
  l.sumsq_off = round_up(l.resid_bytes, kPpPage);
  l.stamp_off = l.sumsq_off + l.sumsq_bytes;
  l.flag_off = round_up(l.stamp_off + 4, kPpPage);
  l.slot_bytes = round_up(l.flag_off + kPpPage, kPpLandingAlign);
  l.total = l.slot_bytes * kPpPfDepth;
  return l;
}

size_t pp_prefill_link_bytes(const model::ModelDesc& d, uint32_t device) {
  const size_t common = size_t(kPpPfDepth) * sizeof(Control) + kTimestampBytes;
  const size_t peer_words = size_t(kPpPfDepth) * kPpStateWords * 4;
  return device == 0 ? common + peer_words : pp_prefill_link_layout(d).total + common + peer_words;
}

size_t pp_prefill_shadow_bytes(const model::ModelDesc& d, const PpStage& st) {
  const PersistentSizes p = PersistentDims::stage_sizes(kMinAutoMaxLen, d, KvCache::Bf16, st.gdn, st.fa);
  return size_t(kPpPfDepth) * (p.gdn_state + p.conv_ring);
}

}  // namespace runtime
