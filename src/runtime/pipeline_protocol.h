#pragma once
// Spec 16b: the peer hand-off's protocol (src/kernels/pp_handoff.cl: pp_send, pp_recv) as
// host code over std::atomic words, line for line, so tests/runtime/pp_protocol_test.cc can
// run it on two host threads standing in for the two devices: sequence numbers advanced by
// the "kernels" in "device" memory (replayable with frozen arguments), the stamp after the
// norm sums, the bounded spin and the first-failure record. The memory-model claims are the
// C++ ones here (acquire / release on the flag and the stamp, relaxed data words) and the
// OpenCL ones there (memory_scope_all_svm_devices); what only the box shows is that the
// card's system-scope atomics really see a PCIe peer write (spec 16a Review Focus 1).
//
// A change to one side is a change to both: the kernel's comment points here.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace runtime::pp_protocol {

// pp_recv's state words (pp_handoff.cl): [0] the last sequence consumed, [1] status,
// [2] the flag read at the first failure, [3] the stamp read, [4] the sequence expected.
enum : uint32_t { kSeq = 0, kStatus = 1, kFlag = 2, kStamp = 3, kWant = 4, kStateWords = 5 };
enum : uint32_t { kOk = 0, kTimeout = 1, kStale = 2 };

// Device 1's landing buffer: the residual words, the norm sums and the stamp right after
// them, the flag on its own (page). Every word is an atomic: the receiver reads all of them
// with "system-scope" loads, as the kernel does.
struct Landing {
  Landing(size_t resid_words, size_t sumsq_words)
      : resid_n(resid_words),
        sumsq_n(sumsq_words),
        resid(new std::atomic<uint32_t>[resid_words]()),
        sumsq(new std::atomic<uint32_t>[sumsq_words + 1]()) {}   // + the stamp
  size_t resid_n, sumsq_n;
  std::unique_ptr<std::atomic<uint32_t>[]> resid, sumsq;
  std::atomic<uint32_t> flag{0};
  void zero() {
    for (size_t i = 0; i < resid_n; ++i) resid[i].store(0, std::memory_order_relaxed);
    for (size_t i = 0; i <= sumsq_n; ++i) sumsq[i].store(0, std::memory_order_relaxed);
    flag.store(0, std::memory_order_release);
  }
};

// pp_send: device 0's last kernel. `seq` is device 0's counter.
inline void send(const uint32_t* resid, const uint32_t* sumsq, Landing& l, uint32_t& seq) {
  const uint32_t next = seq + 1u;
  for (size_t i = 0; i < l.resid_n; ++i) l.resid[i].store(resid[i], std::memory_order_relaxed);
  for (size_t i = 0; i < l.sumsq_n; ++i) l.sumsq[i].store(sumsq[i], std::memory_order_relaxed);
  l.sumsq[l.sumsq_n].store(next, std::memory_order_relaxed);   // the stamp
  // atomic_work_item_fence(release, all devices) + the barrier, then lane 0's release store.
  seq = next;
  l.flag.store(next, std::memory_order_release);
}

// pp_recv: device 1's first kernel. `state` is device 1's [kStateWords]. Returns the status
// of THIS hand-off (state[kStatus] keeps the first failure). Copies the landing buffer into
// `resid` / `sumsq` whatever the status - the kernel lets the list finish and the host
// throws - and counts the flag loads into `*loads` when asked.
inline uint32_t recv(const Landing& l, uint32_t* state, uint32_t* resid, uint32_t* sumsq,
                     uint32_t spin_limit, uint32_t* loads = nullptr) {
  const uint32_t want = state[kSeq] + 1u;
  uint32_t got = 0u, n = 0u;
  do {
    got = l.flag.load(std::memory_order_acquire);
  } while (got != want && ++n < spin_limit);
  const uint32_t stamp = l.sumsq[l.sumsq_n].load(std::memory_order_acquire);
  const uint32_t status = got != want ? kTimeout : (stamp != want ? kStale : kOk);
  state[kSeq] = want;
  if (status != kOk && state[kStatus] == kOk) {
    state[kStatus] = status;
    state[kFlag] = got;
    state[kStamp] = stamp;
    state[kWant] = want;
  }
  for (size_t i = 0; i < l.resid_n; ++i) resid[i] = l.resid[i].load(std::memory_order_relaxed);
  for (size_t i = 0; i < l.sumsq_n; ++i) sumsq[i] = l.sumsq[i].load(std::memory_order_relaxed);
  if (loads) *loads = got == want ? n + 1u : n;   // a timeout made exactly spin_limit loads
  return status;
}

}  // namespace runtime::pp_protocol
