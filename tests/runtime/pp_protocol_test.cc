// Spec 16b: the peer hand-off's protocol (pp_handoff.cl) on two host threads - the host
// mirror in runtime/pipeline_protocol.h. What it proves: sequence numbers advanced by the
// "kernels" keep 1000 replays with frozen arguments in step; every hand-off delivers that
// step's words; a receiver that starts first waits for the sender; a missing sender is a
// bounded timeout with the flag and the expected number recorded; a flag without its data
// (the stamp did not travel) is caught as stale; a sender that ran ahead is caught; the
// first failure is the one kept; a reset restarts both counters. What it does not: the
// card's memory model over PCIe (the box, spec 16a Review Focus 1).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "check.h"
#include "runtime/pipeline_protocol.h"

namespace {
namespace pp = runtime::pp_protocol;

constexpr size_t kResid = 2560;   // Qwen3.8's one bf16 row, in words
constexpr size_t kSumsq = 160;    // norm_sumsq's 640 B

std::vector<uint32_t> words(size_t n, uint32_t step, uint32_t salt) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = step * 2654435761u + uint32_t(i) * 40503u + salt;
  return v;
}

// 1000 steps on two threads, the receiver started first each time (device 1's list is
// submitted with device 0's and spins until the residual arrives).
void check_threaded_replay() {
  pp::Landing land(kResid, kSumsq);
  uint32_t send_seq = 0;
  uint32_t state[pp::kStateWords] = {};
  std::vector<uint32_t> out_r(kResid), out_s(kSumsq);
  for (uint32_t step = 1; step <= 1000; ++step) {
    const std::vector<uint32_t> r = words(kResid, step, 1), s = words(kSumsq, step, 7);
    uint32_t status = 99;
    std::thread rx([&] {
      status = pp::recv(land, state, out_r.data(), out_s.data(), 1u << 30);
    });
    std::thread tx([&] {
      if (step % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
      pp::send(r.data(), s.data(), land, send_seq);
    });
    tx.join();
    rx.join();
    CHECK_EQ(status, uint32_t{pp::kOk});
    CHECK(out_r == r);
    CHECK(out_s == s);
    CHECK_EQ(send_seq, step);
    CHECK_EQ(state[pp::kSeq], step);
    CHECK_EQ(land.flag.load(), step);
  }
  CHECK_EQ(state[pp::kStatus], uint32_t{pp::kOk});
}

void check_timeout() {
  pp::Landing land(kResid, kSumsq);
  uint32_t state[pp::kStateWords] = {};
  std::vector<uint32_t> out_r(kResid), out_s(kSumsq);
  uint32_t loads = 0;
  const auto t0 = std::chrono::steady_clock::now();
  // No sender: the spin gives up after exactly spin_limit loads, records why, and the
  // sequence still advances (the host throws and resets).
  CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 100000, &loads),
           uint32_t{pp::kTimeout});
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
  CHECK_EQ(loads, 100000u);
  CHECK_EQ(state[pp::kSeq], 1u);
  CHECK_EQ(state[pp::kStatus], uint32_t{pp::kTimeout});
  CHECK_EQ(state[pp::kFlag], 0u);
  CHECK_EQ(state[pp::kWant], 1u);
  // The first failure is kept: a later good hand-off does not clear it (only reset does).
  uint32_t seq = 1;   // device 0 is now one behind device 1's expectation: send 2
  const std::vector<uint32_t> r = words(kResid, 2, 1), s = words(kSumsq, 2, 7);
  pp::send(r.data(), s.data(), land, seq);
  CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kOk});
  CHECK_EQ(state[pp::kStatus], uint32_t{pp::kTimeout});
  CHECK_EQ(state[pp::kWant], 1u);
}

void check_stale_and_ahead() {
  std::vector<uint32_t> out_r(kResid), out_s(kSumsq);
  {
    // The flag arrives, the data does not: the stamp still holds the previous step's number.
    pp::Landing land(kResid, kSumsq);
    uint32_t seq = 0, state[pp::kStateWords] = {};
    const std::vector<uint32_t> r = words(kResid, 1, 1), s = words(kSumsq, 1, 7);
    pp::send(r.data(), s.data(), land, seq);
    CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kOk});
    land.flag.store(2, std::memory_order_release);   // a "transport" that moved only the flag
    CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kStale});
    CHECK_EQ(state[pp::kStatus], uint32_t{pp::kStale});
    CHECK_EQ(state[pp::kStamp], 1u);
    CHECK_EQ(state[pp::kWant], 2u);
  }
  {
    // Device 0 ran a step device 1 never consumed (lockstep broken): the flag reads 2 where 1
    // is expected - a bounded timeout, never a silent pass.
    pp::Landing land(kResid, kSumsq);
    uint32_t seq = 0, state[pp::kStateWords] = {};
    const std::vector<uint32_t> r = words(kResid, 1, 1), s = words(kSumsq, 1, 7);
    pp::send(r.data(), s.data(), land, seq);
    pp::send(r.data(), s.data(), land, seq);
    CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kTimeout});
    CHECK_EQ(state[pp::kFlag], 2u);
    CHECK_EQ(state[pp::kWant], 1u);
  }
}

void check_reset() {
  pp::Landing land(kResid, kSumsq);
  uint32_t seq = 0, state[pp::kStateWords] = {};
  std::vector<uint32_t> out_r(kResid), out_s(kSumsq);
  for (uint32_t step = 1; step <= 3; ++step) {
    const std::vector<uint32_t> r = words(kResid, step, 1), s = words(kSumsq, step, 7);
    pp::send(r.data(), s.data(), land, seq);
    CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kOk});
  }
  // PipelineEngine::reset(): the landing buffer, the flag and both counters to zero.
  land.zero();
  seq = 0;
  for (uint32_t& w : state) w = 0;
  const std::vector<uint32_t> r = words(kResid, 9, 1), s = words(kSumsq, 9, 7);
  pp::send(r.data(), s.data(), land, seq);
  CHECK_EQ(pp::recv(land, state, out_r.data(), out_s.data(), 1000), uint32_t{pp::kOk});
  CHECK_EQ(seq, 1u);
  CHECK_EQ(state[pp::kSeq], 1u);
  CHECK(out_r == r);
}
}  // namespace

int main() {
  check_threaded_replay();
  check_timeout();
  check_stale_and_ahead();
  check_reset();
  std::printf("pp_protocol_test: OK\n");
  return 0;
}
