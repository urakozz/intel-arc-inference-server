// Spec 17a (plan 17a Task 1): the TP all-reduce's protocol (tools/probe/tp_protocol.h, the host mirror of
// tools/probe/probe_tp.cl's tp_send / tp_wait and probe_tp's CPU-sum thread) on host threads, beside
// pp_protocol_test. Two "card" threads and, for routes (H) / (H'), the CPU-sum thread. What it proves, on every
// route (P push, P pull, host-mbox, H host-bar, H' host-read):
//   - the per-chunk counters advanced by the "kernels" keep thousands of calls in step (replays with frozen
//     arguments progress), the two-slot landing never overwrites a row the peer has not read;
//   - both cards end every call with the same bits, equal to the fixed-order fp32 sum (card 0's row + card 1's);
//   - a skipped data write (flag without its data) is caught as stale - never consumed silently - and on (H) the CPU
//     forwards it poisoned so the cards catch it too; a dropped write (no flag) is a bounded timeout;
//   - the first failure is the one kept.
// What it does not: the card's memory model over PCIe or through the BAR (the box: probe_tp ar / bar).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "tp_protocol.h"

namespace {
namespace tpp = tp_protocol;

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kChunk = 1024;
constexpr uint32_t kS = 4;

std::vector<float> partials(uint32_t words, uint32_t seed) {
  std::vector<float> p(size_t(kS) * words);
  uint32_t st = seed;
  for (float& v : p) {
    st = st * 1664525u + 1013904223u;
    v = tpp::bits_to_float((st >> 9) | 0x3e000000u) - 0.25f;   // small, normal
  }
  return p;
}

struct Buffers {
  explicit Buffers(const tpp::Layout& L) : L(L) {
    for (auto* v : {&land0, &land1, &mbox0, &mbox1, &bar0, &bar1, &hsum}) v->assign(L.total_words(), 0u);
  }
  tpp::Layout L;
  std::vector<uint32_t> land0, land1, mbox0, mbox1, bar0, bar1, hsum;
};

struct RouteBufs {
  uint32_t* dst[2];
  uint32_t* src[2];
  bool cpu = false;        // a CPU-sum thread between the cards
  bool take = false;       // the card takes the sum (H / H') instead of adding the peer's row
  uint32_t* cpu_dst[2] = {nullptr, nullptr};
  uint32_t cpu_ndst = 0;
};

RouteBufs route(tpp::Route r, Buffers& b) {
  RouteBufs x{};
  switch (r) {
    case tpp::Route::P2pPush:   // each card writes into the PEER's landing, reads its own
      x.dst[0] = b.land1.data(); x.dst[1] = b.land0.data();
      x.src[0] = b.land0.data(); x.src[1] = b.land1.data();
      break;
    case tpp::Route::P2pPull:   // each card writes its own exposed buffer, reads the peer's
      x.dst[0] = b.land0.data(); x.dst[1] = b.land1.data();
      x.src[0] = b.land1.data(); x.src[1] = b.land0.data();
      break;
    case tpp::Route::HostMbox:  // each card writes its host mailbox, reads the peer's
      x.dst[0] = b.mbox0.data(); x.dst[1] = b.mbox1.data();
      x.src[0] = b.mbox1.data(); x.src[1] = b.mbox0.data();
      break;
    case tpp::Route::HostBar:   // mailboxes -> CPU sum -> each card's own VRAM (BAR)
      x.dst[0] = b.mbox0.data(); x.dst[1] = b.mbox1.data();
      x.src[0] = b.bar0.data(); x.src[1] = b.bar1.data();
      x.cpu = true; x.take = true;
      x.cpu_dst[0] = b.bar0.data(); x.cpu_dst[1] = b.bar1.data(); x.cpu_ndst = 2;
      break;
    case tpp::Route::HostRead:  // mailboxes -> CPU sum -> one pinned host buffer both cards read
      x.dst[0] = b.mbox0.data(); x.dst[1] = b.mbox1.data();
      x.src[0] = b.hsum.data(); x.src[1] = b.hsum.data();
      x.cpu = true; x.take = true;
      x.cpu_dst[0] = b.hsum.data(); x.cpu_ndst = 1;
      break;
  }
  return x;
}

struct Card {
  std::vector<uint32_t> send_ctr, wait_ctr;   // per chunk ("per work-group"), advanced by the kernels
  uint32_t state[tpp::kStateWords] = {};
  std::vector<float> own, peer, out;
};

void rel_fence() { std::atomic_thread_fence(std::memory_order_release); }

// One call on one card: fold + send every chunk, then wait + combine every chunk.
void card_call(uint32_t rank, Card& c, const RouteBufs& rb, const tpp::Layout& L, const std::vector<float>& P,
               uint32_t inject_at, uint32_t inject, uint32_t spin) {
  const uint32_t G = L.chunks();
  for (uint32_t j = 0; j < G; ++j) {
    const uint32_t n = c.send_ctr[j] + 1;
    for (uint32_t e = j * L.chunk; e < (j + 1) * L.chunk; ++e) c.own[e] = tpp::fold(P.data(), kS, L.words, e, n, rank);
    const uint32_t inj = (rank == 1 && n == inject_at) ? inject : tpp::kInjNone;
    tpp::send_chunk(rb.dst[rank], L, j, c.send_ctr[j],
                    reinterpret_cast<const uint32_t*>(c.own.data()) + size_t(j) * L.chunk, inj, rel_fence);
  }
  for (uint32_t j = 0; j < G; ++j) {
    tpp::wait_chunk(rb.src[rank], L, j, c.wait_ctr[j], spin, c.state, reinterpret_cast<uint32_t*>(c.peer.data()));
    for (uint32_t e = j * L.chunk; e < (j + 1) * L.chunk; ++e)
      c.out[e] = rb.take ? c.peer[e] : tpp::combine(c.own[e], c.peer[e], rank);
  }
}

struct Result {
  uint32_t state0[tpp::kStateWords], state1[tpp::kStateWords], cpu[tpp::kStateWords];
  uint32_t bad_calls = 0;     // calls where a card's output differed from the expected fixed-order sum
  uint32_t diverged = 0;      // calls where the two cards' outputs differed
};

// `calls` calls on two card threads (+ the CPU thread on H / H'); the expected sum checked every call.
Result run(tpp::Route r, uint32_t M, uint32_t calls, uint32_t inject_at = 0, uint32_t inject = tpp::kInjNone,
           uint32_t spin = 1u << 28) {
  const tpp::Layout L = tpp::Layout::make(M * kHidden, kChunk);
  Buffers b(L);
  const RouteBufs rb = route(r, b);
  const std::vector<float> P = partials(L.words, 12345);   // the same partials on both cards (the probe's choice)
  Card c[2];
  for (Card& x : c) {
    x.send_ctr.assign(L.chunks(), 0);
    x.wait_ctr.assign(L.chunks(), 0);
    x.own.assign(L.words, 0.f);
    x.peer.assign(L.words, 0.f);
    x.out.assign(L.words, 0.f);
  }
  Result res{};
  std::vector<uint32_t> cpu_ctr(L.chunks(), 0);
  std::vector<uint32_t> scratch(L.chunk);
  std::thread cpu;
  if (rb.cpu)
    cpu = std::thread([&] {
      for (uint32_t n = 1; n <= calls; ++n)
        for (uint32_t j = 0; j < L.chunks(); ++j)
          tpp::cpu_sum_chunk(rb.dst[0], rb.dst[1], rb.cpu_dst, rb.cpu_ndst, L, j, cpu_ctr[j], spin, res.cpu,
                             scratch.data(), rel_fence);
    });
  std::vector<float> want(L.words);
  for (uint32_t n = 1; n <= calls; ++n) {
    std::thread t1([&] {
      if (n % 5 == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
      card_call(1, c[1], rb, L, P, inject_at, inject, spin);
    });
    if (n % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
    card_call(0, c[0], rb, L, P, inject_at, inject, spin);
    t1.join();
    for (uint32_t e = 0; e < L.words; ++e)
      want[e] = tpp::fold(P.data(), kS, L.words, e, n, 0) + tpp::fold(P.data(), kS, L.words, e, n, 1);
    if (std::memcmp(c[0].out.data(), c[1].out.data(), L.words * 4) != 0) ++res.diverged;
    if (std::memcmp(c[0].out.data(), want.data(), L.words * 4) != 0 ||
        std::memcmp(c[1].out.data(), want.data(), L.words * 4) != 0)
      ++res.bad_calls;
  }
  if (cpu.joinable()) cpu.join();
  std::memcpy(res.state0, c[0].state, sizeof res.state0);
  std::memcpy(res.state1, c[1].state, sizeof res.state1);
  return res;
}

const tpp::Route kRoutes[] = {tpp::Route::P2pPush, tpp::Route::P2pPull, tpp::Route::HostMbox, tpp::Route::HostBar,
                              tpp::Route::HostRead};

void check_layout() {
  const tpp::Layout L = tpp::Layout::make(5120, 1024);
  CHECK_EQ(L.chunks(), 5u);
  CHECK_EQ(L.data(1), size_t(5120));
  CHECK_EQ(L.trailer(0, 0), size_t(2 * 5120));
  CHECK_EQ(L.flag_base() % 1024, size_t(0));                   // the flags start on a page of their own
  CHECK(L.flag_base() >= L.trailer(1, 4) + 1);
  CHECK_EQ(L.flag(0, 1) - L.flag(0, 0), size_t(tpp::kFlagStride));   // one 64-byte line a flag
  CHECK_EQ(L.total_words() % 1024, size_t(0));
  CHECK(L.total_words() >= L.flag(1, 4) + 1);
  const tpp::Layout L8 = tpp::Layout::make(8 * 5120, 1024);
  CHECK_EQ(L8.chunks(), 40u);
  bool threw = false;
  try { tpp::Layout::make(5000, 1024); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
  CHECK_EQ(tpp::slot_of(1), 1u);
  CHECK_EQ(tpp::slot_of(2), 0u);
  CHECK_EQ(std::string(tpp::route_name(tpp::Route::HostBar)), std::string("host-bar"));
  CHECK(tpp::route_from_name("p2p-pull") == tpp::Route::P2pPull);
  CHECK(tpp::route_from_name("host-mbox") == tpp::Route::HostMbox);
}

// The data generator is deterministic, depends on call, rank and element, and the fold's order is fixed.
void check_gen() {
  CHECK_EQ(tpp::gen_bits(1, 0, 0), tpp::gen_bits(1, 0, 0));
  CHECK(tpp::gen_bits(1, 0, 0) != tpp::gen_bits(2, 0, 0));
  CHECK(tpp::gen_bits(1, 0, 0) != tpp::gen_bits(1, 1, 0));
  CHECK(tpp::gen_bits(1, 0, 0) != tpp::gen_bits(1, 0, 1));
  const float g = tpp::gen(7, 1, 99);
  CHECK(g >= -0.5f && g < 0.5f);
  // combine is the fixed order card 0 + card 1 on both ranks
  CHECK_EQ(tpp::float_bits(tpp::combine(1.5f, 2.25f, 0)), tpp::float_bits(1.5f + 2.25f));
  CHECK_EQ(tpp::float_bits(tpp::combine(2.25f, 1.5f, 1)), tpp::float_bits(1.5f + 2.25f));
  // the stamp hash over a row is order-independent across work-groups and catches a flipped bit
  std::vector<uint32_t> row(64);
  for (uint32_t i = 0; i < 64; ++i) row[i] = i * 2654435761u;
  uint32_t h = 0;
  for (uint32_t i = 0; i < 64; ++i) h += tpp::mix(row[i], i);
  uint32_t h2 = 0;
  for (uint32_t i = 64; i-- > 0;) h2 += tpp::mix(row[i], i);
  CHECK_EQ(h, h2);
  row[17] ^= 1u;
  uint32_t h3 = 0;
  for (uint32_t i = 0; i < 64; ++i) h3 += tpp::mix(row[i], i);
  CHECK(h3 != h);
}

void check_routes_bitwise() {
  for (tpp::Route r : kRoutes)
    for (uint32_t M : {1u, 8u}) {
      const uint32_t calls = M == 1 ? 2000 : 300;
      const Result res = run(r, M, calls);
      std::printf("  %-9s M=%u: %u calls, bad %u, diverged %u, status %u/%u/%u\n", tpp::route_name(r), M, calls,
                  res.bad_calls, res.diverged, res.state0[tpp::kStatus], res.state1[tpp::kStatus],
                  res.cpu[tpp::kStatus]);
      CHECK_EQ(res.bad_calls, 0u);
      CHECK_EQ(res.diverged, 0u);
      CHECK_EQ(res.state0[tpp::kStatus], uint32_t{tpp::kOk});
      CHECK_EQ(res.state1[tpp::kStatus], uint32_t{tpp::kOk});
      CHECK_EQ(res.cpu[tpp::kStatus], uint32_t{tpp::kOk});
    }
}

// Card 1 skips its data write at call 37 (the flag goes out without the data): the reader catches it as stale -
// card 0 directly, or the CPU on (H) / (H') which then forwards the sum poisoned, so both cards catch it too.
void check_skipped_data_caught() {
  for (tpp::Route r : kRoutes) {
    const Result res = run(r, 1, 60, 37, tpp::kInjSkipData);
    const bool cpu = r == tpp::Route::HostBar || r == tpp::Route::HostRead;
    std::printf("  %-9s skip-data at 37: status %u/%u/%u want %u/%u/%u\n", tpp::route_name(r),
                res.state0[tpp::kStatus], res.state1[tpp::kStatus], res.cpu[tpp::kStatus], res.state0[tpp::kWant],
                res.state1[tpp::kWant], res.cpu[tpp::kWant]);
    CHECK(res.bad_calls >= 1);   // that call's data really was wrong ...
    CHECK_EQ(res.state0[tpp::kStatus], uint32_t{tpp::kStale});   // ... and it was caught, not consumed silently
    CHECK_EQ(res.state0[tpp::kWant], 37u);
    if (cpu) {
      CHECK_EQ(res.cpu[tpp::kStatus], uint32_t{tpp::kStale});
      CHECK_EQ(res.cpu[tpp::kWant], 37u);
      CHECK_EQ(res.state1[tpp::kStatus], uint32_t{tpp::kStale});   // the poisoned forward reaches card 1 as well
    } else {
      CHECK_EQ(res.state1[tpp::kStatus], uint32_t{tpp::kOk});      // card 0's row to card 1 was fine
    }
  }
}

// Card 1 drops its write entirely at call 5 (no flag): a bounded timeout, recorded with the value it saw.
void check_dropped_write_times_out() {
  for (tpp::Route r : {tpp::Route::P2pPush, tpp::Route::HostMbox, tpp::Route::HostBar}) {
    const auto t0 = std::chrono::steady_clock::now();
    const Result res = run(r, 1, 6, 5, tpp::kInjSkipAll, 200000);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %-9s skip-all at 5: status %u/%u/%u in %.2f s\n", tpp::route_name(r), res.state0[tpp::kStatus],
                res.state1[tpp::kStatus], res.cpu[tpp::kStatus], s);
    CHECK(s < 30.0);
    if (r == tpp::Route::HostBar) {
      CHECK_EQ(res.cpu[tpp::kStatus], uint32_t{tpp::kTimeout});
      CHECK_EQ(res.cpu[tpp::kWant], 5u);
    } else {
      CHECK_EQ(res.state0[tpp::kStatus], uint32_t{tpp::kTimeout});
      CHECK_EQ(res.state0[tpp::kWant], 5u);
      CHECK_EQ(res.state0[tpp::kFlag], 3u);   // the slot's previous stamp (call 3), not 5
    }
  }
}

// The first failure is kept: a later good call does not clear the state.
void check_first_failure_kept() {
  const tpp::Layout L = tpp::Layout::make(kHidden, kChunk);
  std::vector<uint32_t> buf(L.total_words(), 0u);
  std::vector<uint32_t> row(L.words, 0x3f800000u), got(L.words);
  uint32_t sctr = 0, wctr = 0, state[tpp::kStateWords] = {};
  // call 1: skipped data -> stale
  tpp::send_chunk(buf.data(), L, 0, sctr, row.data(), tpp::kInjSkipData, rel_fence);
  CHECK_EQ(tpp::wait_chunk(buf.data(), L, 0, wctr, 1000, state, got.data()), uint32_t{tpp::kStale});
  // call 2: good
  tpp::send_chunk(buf.data(), L, 0, sctr, row.data(), tpp::kInjNone, rel_fence);
  CHECK_EQ(tpp::wait_chunk(buf.data(), L, 0, wctr, 1000, state, got.data()), uint32_t{tpp::kOk});
  CHECK_EQ(state[tpp::kStatus], uint32_t{tpp::kStale});
  CHECK_EQ(state[tpp::kWant], 1u);
  CHECK_EQ(state[tpp::kChunk], 0u);
  CHECK_EQ(sctr, 2u);
  CHECK_EQ(wctr, 2u);
}

}  // namespace

int main() {
  check_layout();
  check_gen();
  check_first_failure_kept();
  check_routes_bitwise();
  check_skipped_data_caught();
  check_dropped_write_times_out();
  std::printf("tp_crossing_protocol_test: PASS\n");
  return 0;
}
