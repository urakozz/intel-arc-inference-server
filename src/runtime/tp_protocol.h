#pragma once
// Spec 17b: the tensor-parallel crossing as host code - the mirror of src/kernels/tp.cl's tp_send / tp_res_fold, line
// for line (a change to one is a change to both; spec 17a's tools/probe/tp_protocol.h, promoted with the slices form,
// the per-row counters and the skip-after-failure rule). tests/runtime/tp_protocol_test.cc runs it on two host
// threads; runtime::TpEngine sizes the landing buffers and reads the state words with it.
//
// One landing buffer a direction, in the READER's memory (spec 17a §3.3: a card sees flags only in its own VRAM), of
// 32-bit words:
//
//   [slot 0 data | slot 1 data | trailers [2][J] | pad to 4 KiB | flags [2][J] x 16 words | pad to 4 KiB]
//
// data = s_msg x kMRows x hidden words (a call at M rows uses [s_msg][M][hidden]); J = fold_g x kMRows chunks, chunk
// j = g x kMRows + m (the K range [g chunk, + chunk) of row m, every slice). Call n uses slot n & 1. Per chunk:
//   send  the part's words, trailer = n, a release fence, flag = n (a plain store, then a fence);
//   fold  spin (bounded) on flag == n with acquire loads unless a failure is already recorded; trailer must be n
//         (else STALE: the flag came without its data); the data read; the parts summed from 0 in one fixed order -
//         card 0's part first, so both cards compute the same bits (and, with slices, one card's chain).
// Counters: one per chunk, per kernel (send's and fold's), in the card's memory, advanced by the kernels; chunk
// (g, 0) also sets rows M .. kMRows - 1 to n.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model_desc.h"

namespace runtime {
namespace tp {

enum : uint32_t { kOk = 0, kTimeout = 1, kStale = 2 };
// state words (shared memory, read by the host after the fences)
enum : uint32_t { kStatus = 0, kFlag = 1, kStamp = 2, kWant = 3, kChunk = 4, kFailed = 5, kStateWords = 8 };
constexpr uint32_t kSlots = 2, kFlagStride = 16, kPageWords = 1024;
constexpr uint32_t kMRows = 8;     // tp.cl TP_MROWS: DecodeScratchDims::kM
constexpr uint32_t kFoldG = 20;    // the fold's work-groups: DecodeScratchDims::kNormGroups
constexpr uint32_t kDefaultSpinLimit = 1u << 22;

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct Layout {
  uint32_t hidden = 0, s_msg = 0, fold_g = kFoldG;
  uint32_t chunks() const { return fold_g * kMRows; }
  uint32_t chunk_words() const { return (hidden + fold_g - 1) / fold_g; }   // tp.cl CHUNK
  size_t data_words() const { return size_t(s_msg) * kMRows * hidden; }
  size_t data(uint32_t slot) const { return size_t(slot) * data_words(); }
  size_t trailer(uint32_t slot, uint32_t j) const { return kSlots * data_words() + size_t(slot) * chunks() + j; }
  size_t flag_base() const { return round_up(kSlots * data_words() + size_t(kSlots) * chunks(), kPageWords); }
  size_t flag(uint32_t slot, uint32_t j) const { return flag_base() + (size_t(slot) * chunks() + j) * kFlagStride; }
  size_t total_words() const { return round_up(flag_base() + size_t(kSlots) * chunks() * kFlagStride, kPageWords); }
  size_t total_bytes() const { return total_words() * 4; }
  size_t counter_bytes() const { return size_t(chunks()) * 4; }
};

// The crossing a per-card descriptor runs: the slices form sends the card's split-K slices of the row-parallel
// linears (one-card S / 2 - every row-parallel linear of the card has that S), the folded form one row.
inline Layout layout(const model::ModelDesc& card, model::TpCrossing c) {
  using model::LinearId;
  const uint32_t s = card.shape(LinearId::Down).S;
  if (card.shape(LinearId::OutProj).S != s || card.shape(LinearId::OProj).S != s)
    throw std::logic_error("runtime::tp::layout: the row-parallel linears of " + card.name + " differ in S");
  return Layout{card.hidden, c == model::TpCrossing::Slices ? s : 1u, kFoldG};
}

// --- the host mirror (tp_protocol_test): one "card" is a thread over plain words -------------------------------

inline uint32_t load_acq(const uint32_t* p) {
  return reinterpret_cast<const std::atomic<uint32_t>*>(p)->load(std::memory_order_acquire);
}
inline void store_rel(uint32_t* p, uint32_t v) {
  reinterpret_cast<std::atomic<uint32_t>*>(p)->store(v, std::memory_order_release);
}

inline uint32_t next(uint32_t* ctr, uint32_t g, uint32_t m, uint32_t M) {
  const uint32_t j = g * kMRows + m;
  const uint32_t n = ctr[j] + 1;
  ctr[j] = n;
  if (m == 0)
    for (uint32_t r = M; r < kMRows; ++r) ctr[g * kMRows + r] = n;
  return n;
}

// tp_send's chunk (g, m): `part` is the card's message [s_msg][M][hidden] as floats (slices, or the folded row);
// `drop` skips everything (a lost send), `skip_data` publishes the flag without data or trailer (a stale chunk).
inline void send_chunk(const Layout& L, uint32_t* land, uint32_t* ctr, const float* part, uint32_t M, uint32_t g,
                       uint32_t m, bool drop = false, bool skip_data = false) {
  const uint32_t n = next(ctr, g, m, M), slot = n & 1u, j = g * kMRows + m;
  if (drop) return;
  const uint32_t k0 = g * L.chunk_words(), k1 = std::min(L.hidden, k0 + L.chunk_words());
  if (!skip_data) {
    for (uint32_t s = 0; s < L.s_msg; ++s)
      for (uint32_t k = k0; k < k1; ++k) {
        const size_t e = (size_t(s) * M + m) * L.hidden + k;
        std::memcpy(&land[L.data(slot) + e], &part[e], 4);
      }
    land[L.trailer(slot, j)] = n;
  }
  std::atomic_thread_fence(std::memory_order_release);
  store_rel(&land[L.flag(slot, j)], n);
}

// tp_res_fold's wait and chain for chunk (g, m) of card `rank`: `mine` its own part, the peer's in `land`; `out`
// receives the fp32 sum per element (the value the kernel rounds into the residual). Returns the status.
inline uint32_t fold_chunk(const Layout& L, const uint32_t* land, uint32_t* ctr, uint32_t* state, const float* mine,
                           uint32_t rank, uint32_t M, uint32_t g, uint32_t m, uint32_t spin_limit, float* out) {
  const uint32_t want = next(ctr, g, m, M), slot = want & 1u, j = g * kMRows + m;
  uint32_t status = kOk;
  if (reinterpret_cast<std::atomic<uint32_t>*>(state)->load(std::memory_order_relaxed) == 0) {
    uint32_t got = 0, i = 0;
    do {
      got = load_acq(&land[L.flag(slot, j)]);
    } while (got != want && ++i < spin_limit);
    const uint32_t stamp = load_acq(&land[L.trailer(slot, j)]);
    status = got != want ? kTimeout : (stamp != want ? kStale : kOk);
    if (status != kOk) {
      reinterpret_cast<std::atomic<uint32_t>*>(state + kFailed)->fetch_add(1);
      uint32_t zero = 0;
      if (reinterpret_cast<std::atomic<uint32_t>*>(state)->compare_exchange_strong(zero, status)) {
        state[kFlag] = got;
        state[kStamp] = stamp;
        state[kWant] = want;
        state[kChunk] = j;
      }
    }
  } else {
    status = state[kStatus];
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  const uint32_t k0 = g * L.chunk_words(), k1 = std::min(L.hidden, k0 + L.chunk_words());
  for (uint32_t k = k0; k < k1; ++k) {
    float acc = 0.f;
    for (uint32_t half = 0; half < 2; ++half) {
      const bool local = (half == 0) == (rank == 0);
      for (uint32_t s = 0; s < L.s_msg; ++s) {
        const size_t e = (size_t(s) * M + m) * L.hidden + k;
        float v;
        if (local) {
          v = mine[e];
        } else {
          std::memcpy(&v, &land[L.data(slot) + e], 4);
        }
        acc += v;
      }
    }
    out[size_t(m) * L.hidden + k] = acc;
  }
  return status;
}

}  // namespace tp
}  // namespace runtime
