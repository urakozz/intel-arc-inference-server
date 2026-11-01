#pragma once
// Spec 17a: the TP all-reduce's crossing protocol, as host code over 32-bit words - the mirror of
// tools/probe/probe_tp.cl's tp_send / tp_wait (line for line) and the CPU-sum step probe_tp's host thread runs for
// routes (H) host-bar and (H') host-read. tests/runtime/tp_crossing_protocol_test.cc runs it on host threads (two
// "cards" and the CPU); probe_tp uses cpu_sum_chunk itself on the box, with an sfence for the write-combined BAR.
// A change to one side is a change to both: probe_tp.cl's header points here.
//
// The message: one call's fp32 rows (M x hidden words), cut into `chunk`-word chunks, one work-group each. Every
// destination buffer holds TWO slots (call n uses slot n & 1), so a writer at call n + 1 never overwrites the row a
// reader of call n - 1 is still reading (it cannot get there before it has read the reader's call n row, which the
// reader wrote after finishing call n - 1). Per slot and chunk: the data, a trailer stamp written WITH the data, and
// a flag on its own 64-byte line, all flags on a page of their own:
//
//   [slot 0 data | slot 1 data | trailers [2][G] | pad to 4 KiB | flags [2][G] x 16 words | pad to 4 KiB]
//
// Per chunk j at call n (n = 1, 2, ... from a counter the kernel owns, one per work-group, so replays with frozen
// arguments progress - spec 16 §2):
//   send:  data stores, trailer[slot][j] = n, a system-scope release fence, flag[slot][j] = n (release store);
//   wait:  spin (bounded) on flag[slot][j] == n with acquire loads; trailer[slot][j] must be n, else the flag came
//          without its data - STALE (the "silent hand-off" spec 16 §2 warns about); the data read with
//          system-scope loads; the first failure recorded in state, the list allowed to finish.
// The cross-card add is in ONE fixed order on every route: card 0's word + card 1's (combine). Routes (H) / (H'):
// the CPU waits on both mailboxes, adds in that order, and sends the sum on with the same send; a failure it saw
// is forwarded POISONED (trailer = n | kPoison) so the cards catch it too, never consume it.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tp_protocol {

enum : uint32_t { kOk = 0, kTimeout = 1, kStale = 2 };
enum : uint32_t { kInjNone = 0, kInjSkipData = 1, kInjSkipAll = 2 };
// state words: [kStatus] 0 = every call so far good, else the first failure's status; then the flag and trailer read
// at it, the call expected and the chunk.
enum : uint32_t { kStatus = 0, kFlag = 1, kStamp = 2, kWant = 3, kChunk = 4, kStateWords = 8 };
constexpr uint32_t kSlots = 2;
constexpr uint32_t kFlagStride = 16;   // words: one 64-byte line a flag
constexpr uint32_t kPageWords = 1024;  // 4 KiB
constexpr uint32_t kPoison = 0x80000000u;

enum class Route { P2pPush, P2pPull, HostMbox, HostBar, HostRead };
inline const char* route_name(Route r) {
  switch (r) {
    case Route::P2pPush: return "p2p-push";
    case Route::P2pPull: return "p2p-pull";
    case Route::HostMbox: return "host-mbox";
    case Route::HostBar: return "host-bar";
    case Route::HostRead: return "host-read";
  }
  return "?";
}
inline Route route_from_name(const std::string& s) {
  for (Route r : {Route::P2pPush, Route::P2pPull, Route::HostMbox, Route::HostBar, Route::HostRead})
    if (s == route_name(r)) return r;
  throw std::runtime_error("unknown route '" + s + "' (p2p-push|p2p-pull|host-mbox|host-bar|host-read)");
}

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct Layout {
  uint32_t words = 0;   // fp32 words a message: M x hidden
  uint32_t chunk = 0;   // words a chunk (one work-group)
  uint32_t chunks() const { return words / chunk; }
  size_t data(uint32_t slot) const { return size_t(slot) * words; }
  size_t trailer(uint32_t slot, uint32_t j) const { return size_t(kSlots) * words + size_t(slot) * chunks() + j; }
  size_t flag_base() const { return round_up(size_t(kSlots) * words + size_t(kSlots) * chunks(), kPageWords); }
  size_t flag(uint32_t slot, uint32_t j) const {
    return flag_base() + (size_t(slot) * chunks() + j) * kFlagStride;
  }
  size_t total_words() const {
    return round_up(flag_base() + size_t(kSlots) * chunks() * kFlagStride, kPageWords);
  }
  size_t total_bytes() const { return total_words() * 4; }
  static Layout make(uint32_t words, uint32_t chunk) {
    if (chunk == 0 || words == 0 || words % chunk != 0)
      throw std::runtime_error("tp_protocol: " + std::to_string(words) + " words do not cut into chunks of " +
                               std::to_string(chunk));
    Layout L;
    L.words = words;
    L.chunk = chunk;
    return L;
  }
};
inline uint32_t slot_of(uint32_t n) { return n & 1u; }

inline float bits_to_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t float_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// The per-call term each card adds to its folded row - probe_tp.cl's tp_gen, bit for bit: a hash of (n, rank, e)
// into [1, 2), minus 1.5 (exact). Only integer ops and one exact subtraction, so host and card agree.
inline uint32_t gen_bits(uint32_t n, uint32_t rank, uint32_t e) {
  uint32_t h = n * 0x9E3779B1u ^ (rank + 1u) * 0x85EBCA77u ^ e * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  h *= 0x297A2D39u;
  h ^= h >> 15;
  return h;
}
inline float gen(uint32_t n, uint32_t rank, uint32_t e) {
  return bits_to_float((gen_bits(n, rank, e) & 0x007FFFFFu) | 0x3F800000u) - 1.5f;
}
// A card's folded row at call n: its S split-K slices summed in slice order, then the call's term - probe_tp.cl's
// tp_send (fp32 adds only, no contraction possible).
inline float fold(const float* P, uint32_t S, uint32_t words, uint32_t e, uint32_t n, uint32_t rank) {
  float v = P[e];
  for (uint32_t s = 1; s < S; ++s) v += P[size_t(s) * words + e];
  return v + gen(n, rank, e);
}
// The cross-card add, the fixed order on both cards: card 0's word, then card 1's.
inline float combine(float own, float peer, uint32_t rank) { return rank == 0 ? own + peer : peer + own; }
// The output hash's term (tp_wait's): summed with wrap-around over a message, so the order work-groups finish in
// does not matter, and a flipped bit changes it.
inline uint32_t mix(uint32_t bits, uint32_t e) {
  uint32_t h = bits ^ (e * 0x9E3779B1u);
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return h;
}

// --- word access: the C++ memory model's stand-ins for the kernel's system-scope atomics ---------------------------
inline uint32_t load_acq(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline uint32_t load_rlx(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
inline void store_rel(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline void store_rlx(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }

// tp_send for one chunk: `src` is chunk j's `chunk` words; `ctr` the chunk's counter.
// `fence` orders the data and trailer before the flag (the kernel's system-scope release fence; on the CPU into a
// write-combined BAR mapping it must be an sfence).
template <class Fence>
void send_chunk(uint32_t* buf, const Layout& L, uint32_t j, uint32_t& ctr, const uint32_t* src, uint32_t inject,
                Fence&& fence, uint32_t trailer_or = 0u) {
  const uint32_t n = ctr + 1u;
  const uint32_t slot = slot_of(n);
  ctr = n;
  if (inject == kInjSkipAll) return;
  if (inject != kInjSkipData) {
    uint32_t* d = buf + L.data(slot) + size_t(j) * L.chunk;
    for (uint32_t i = 0; i < L.chunk; ++i) store_rlx(d + i, src[i]);
    store_rlx(buf + L.trailer(slot, j), n | trailer_or);
  }
  fence();
  store_rel(buf + L.flag(slot, j), n);
}

// tp_wait for one chunk: returns this call's status; copies chunk j's words into out (at j * chunk) whatever the
// status (the kernel lets the list finish and the host throws after the fence); keeps the first failure in state.
inline uint32_t wait_chunk(const uint32_t* buf, const Layout& L, uint32_t j, uint32_t& ctr, uint32_t spin_limit,
                           uint32_t* state, uint32_t* out) {
  const uint32_t want = ctr + 1u;
  const uint32_t slot = slot_of(want);
  ctr = want;
  uint32_t got = 0u, k = 0u;
  do {
    got = load_acq(buf + L.flag(slot, j));
  } while (got != want && ++k < spin_limit);
  const uint32_t stamp = load_acq(buf + L.trailer(slot, j));
  const uint32_t status = got != want ? kTimeout : (stamp != want ? kStale : kOk);
  if (status != kOk && state[kStatus] == kOk) {
    state[kStatus] = status;
    state[kFlag] = got;
    state[kStamp] = stamp;
    state[kWant] = want;
    state[kChunk] = j;
  }
  const uint32_t* d = buf + L.data(slot) + size_t(j) * L.chunk;
  uint32_t* o = out + size_t(j) * L.chunk;
  for (uint32_t i = 0; i < L.chunk; ++i) o[i] = load_rlx(d + i);
  return status;
}

// The CPU's step for chunk j of routes (H) / (H'): wait on card 0's mailbox, then card 1's; sum word by word in the
// fixed order (card 0's + card 1's) in fp32; send the sum into every destination (each card's BAR-mapped VRAM for
// (H), one pinned host buffer for (H')). A failure on either input is forwarded poisoned. `scratch`: chunk words.
template <class Fence>
uint32_t cpu_sum_chunk(const uint32_t* mbox0, const uint32_t* mbox1, uint32_t* const* dsts, uint32_t ndst,
                       const Layout& L, uint32_t j, uint32_t& ctr, uint32_t spin_limit, uint32_t* state,
                       uint32_t* scratch, Fence&& fence) {
  const uint32_t want = ctr + 1u;
  const uint32_t slot = slot_of(want);
  const uint32_t* in[2] = {mbox0, mbox1};
  uint32_t status = kOk;
  for (uint32_t r = 0; r < 2 && status == kOk; ++r) {
    uint32_t got = 0u, k = 0u;
    do {
      got = load_acq(in[r] + L.flag(slot, j));
    } while (got != want && ++k < spin_limit);
    const uint32_t stamp = load_acq(in[r] + L.trailer(slot, j));
    status = got != want ? kTimeout : (stamp != want ? kStale : kOk);
    if (status != kOk && state[kStatus] == kOk) {
      state[kStatus] = status;
      state[kFlag] = got;
      state[kStamp] = stamp;
      state[kWant] = want;
      state[kChunk] = j;
    }
  }
  const uint32_t* a = mbox0 + L.data(slot) + size_t(j) * L.chunk;
  const uint32_t* b = mbox1 + L.data(slot) + size_t(j) * L.chunk;
  for (uint32_t i = 0; i < L.chunk; ++i)
    scratch[i] = float_bits(bits_to_float(load_rlx(a + i)) + bits_to_float(load_rlx(b + i)));
  for (uint32_t d = 0; d < ndst; ++d) {
    uint32_t c = ctr;
    send_chunk(dsts[d], L, j, c, scratch, kInjNone, fence, status == kOk ? 0u : kPoison);
  }
  ctr = want;
  return status;
}

}  // namespace tp_protocol
