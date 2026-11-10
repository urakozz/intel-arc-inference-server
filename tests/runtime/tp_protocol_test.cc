// Spec 17b: the tensor-parallel crossing's protocol (runtime/tp_protocol.h, the host mirror of src/kernels/tp.cl's
// tp_send / tp_res_fold) on two host threads standing in for the cards. Both forms:
//   slices  each "card" sends its S_LOC split-K slices; both fold all 2 S_LOC in one-card order - every fold BITWISE
//           the one-card chain ((0 + p0) + p1) + p2) + p3 and equal on both cards;
//   folded  each sends its slices' sum; both add card 0's row then card 1's - equal on both cards, and the host's
//           (p0 + p1) + (p2 + p3).
// Calls: 2000 at M = 1, then lists alternating M = 1 / 8 / 3 (the per-row counters stay level across M). A send whose
// data is skipped is caught STALE, a dropped send a bounded TIMEOUT, and after the first failure every later wait
// skips its spin (the state keeps the first failure).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "check.h"
#include "runtime/tp_protocol.h"

namespace {
using namespace runtime::tp;
constexpr uint32_t kHidden = 5120;

float gen(uint32_t call, uint32_t slice, uint32_t m, uint32_t k) {
  uint32_t h = call * 0x9E3779B1u ^ (slice + 1) * 0x85EBCA77u ^ (m * kHidden + k) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 13;
  return float(int32_t(h & 0xFFFFF) - 0x80000) * 1e-4f;
}

struct Card {
  std::vector<uint32_t> land;   // written by the other card
  std::vector<uint32_t> send_ctr, fold_ctr, state;
  explicit Card(const Layout& L)
      : land(L.total_words(), 0), send_ctr(L.chunks(), 0), fold_ctr(L.chunks(), 0), state(kStateWords, 0) {}
};

struct Plan {
  uint32_t calls;
  uint32_t M;
};

// One card's list: per call, its send of every chunk, then its fold of every chunk; out[call] the fold's sums.
void card_run(const Layout& L, bool folded, uint32_t s_loc, uint32_t rank, Card& me, Card& peer,
              const std::vector<Plan>& plan, std::vector<std::vector<float>>& outs, uint32_t spin, uint32_t drop_at,
              uint32_t stale_at, std::vector<uint32_t>& statuses) {
  uint32_t call = 0;
  for (const Plan& p : plan)
    for (uint32_t c = 0; c < p.calls; ++c) {
      ++call;
      // The card's part: its s_loc slices (rank 0: one-card slices 0 .. s_loc - 1), or their sum.
      std::vector<float> part(size_t(L.s_msg) * p.M * kHidden);
      for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t k = 0; k < kHidden; ++k) {
          float acc = 0.f;
          for (uint32_t s = 0; s < s_loc; ++s) {
            const float v = gen(call, rank * s_loc + s, m, k);
            if (folded)
              acc += v;
            else
              part[(size_t(s) * p.M + m) * kHidden + k] = v;
          }
          if (folded) part[size_t(m) * kHidden + k] = acc;
        }
      for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t g = 0; g < L.fold_g; ++g)
          send_chunk(L, peer.land.data(), me.send_ctr.data(), part.data(), p.M, g, m, call == drop_at,
                     call == stale_at);
      std::vector<float> out(size_t(p.M) * kHidden);
      uint32_t worst = kOk;
      for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t g = 0; g < L.fold_g; ++g) {
          const uint32_t s = fold_chunk(L, me.land.data(), me.fold_ctr.data(), me.state.data(), part.data(), rank,
                                        p.M, g, m, spin, out.data());
          if (s != kOk) worst = s;
        }
      outs.push_back(std::move(out));
      statuses.push_back(worst);
    }
}

void run(bool folded) {
  const uint32_t s_loc = folded ? 4 : 2;
  const Layout L{kHidden, folded ? 1u : s_loc, kFoldG};
  Card c0(L), c1(L);
  const std::vector<Plan> plan = {{2000, 1}, {50, 8}, {50, 1}, {50, 3}, {50, 8}};
  std::vector<std::vector<float>> o0, o1;
  std::vector<uint32_t> st0, st1;
  std::thread t0([&] { card_run(L, folded, s_loc, 0, c0, c1, plan, o0, 1u << 30, 0, 0, st0); });
  card_run(L, folded, s_loc, 1, c1, c0, plan, o1, 1u << 30, 0, 0, st1);
  t0.join();
  CHECK_EQ(c0.state[kStatus], uint32_t(kOk));
  CHECK_EQ(c1.state[kStatus], uint32_t(kOk));
  CHECK_EQ(o0.size(), o1.size());
  uint32_t call = 0;
  for (const Plan& p : plan)
    for (uint32_t c = 0; c < p.calls; ++c, ++call) {
      CHECK(std::memcmp(o0[call].data(), o1[call].data(), o0[call].size() * 4) == 0);   // both cards, bitwise
      for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t k = 0; k < kHidden; k += 97) {
          float want = 0.f;
          if (folded) {
            float a = 0.f, b = 0.f;
            for (uint32_t s = 0; s < s_loc; ++s) a += gen(call + 1, s, m, k);
            for (uint32_t s = 0; s < s_loc; ++s) b += gen(call + 1, s_loc + s, m, k);
            want = (0.f + a) + b;
          } else {
            for (uint32_t s = 0; s < 2 * s_loc; ++s) want += gen(call + 1, s, m, k);   // one card's chain
          }
          float got = o0[call][size_t(m) * kHidden + k];
          CHECK(std::memcmp(&want, &got, 4) == 0);
        }
    }
  std::printf("%s: %u calls (M 1 / 8 / 3), both cards bitwise, %s\n", folded ? "folded" : "slices", call,
              folded ? "card 0's row + card 1's" : "the one-card slice chain");
}

void run_failure(bool drop) {
  const Layout L{kHidden, 2, kFoldG};
  Card c0(L), c1(L);
  const std::vector<Plan> plan = {{10, 1}};
  std::vector<std::vector<float>> o0, o1;
  std::vector<uint32_t> st0, st1;
  // card 1 drops (or skips the data of) its send at call 4: card 0's call-4 wait fails, later ones skip their spin.
  std::thread t0([&] { card_run(L, false, 2, 0, c0, c1, plan, o0, 1u << 26, 0, 0, st0); });
  card_run(L, false, 2, 1, c1, c0, plan, o1, 1u << 26, drop ? 4 : 0, drop ? 0 : 4, st1);
  t0.join();
  CHECK_EQ(c0.state[kStatus], drop ? uint32_t(kTimeout) : uint32_t(kStale));
  CHECK_EQ(c0.state[kWant], 4u);
  for (uint32_t i = 0; i < 3; ++i) CHECK_EQ(st0[i], uint32_t(kOk));
  for (uint32_t i = 3; i < 10; ++i) CHECK(st0[i] != uint32_t(kOk));   // the first failure, then skipped waits
  // Card 1 sent its part of call 5 on time, but card 0's part of call 5 comes only after card 0's call-4 wait has
  // spun out its whole bound - card 1's call-5 wait may time out as well (a lost part cascades, bounded): never
  // earlier than call 5, never stale.
  CHECK(c1.state[kStatus] == uint32_t(kOk) || (c1.state[kStatus] == uint32_t(kTimeout) && c1.state[kWant] >= 5));
  std::printf("%s at call 4: card 0 recorded %s (flag %u, trailer %u), %u failed chunks\n",
              drop ? "a dropped send" : "a send without its data", drop ? "TIMEOUT" : "STALE", c0.state[kFlag],
              c0.state[kStamp], c0.state[kFailed]);
}
}  // namespace

int main() {
  // The layout: data, trailers, flags on their own page, everything in whole pages.
  const Layout L{kHidden, 2, kFoldG};
  CHECK_EQ(L.chunks(), 160u);
  CHECK_EQ(L.chunk_words(), 256u);
  CHECK(L.flag_base() % kPageWords == 0 && L.flag_base() >= L.trailer(1, L.chunks() - 1) + 1);
  CHECK(L.total_words() % kPageWords == 0 && L.total_words() > L.flag(1, L.chunks() - 1));
  run(false);
  run(true);
  run_failure(true);
  run_failure(false);
  std::printf("tp_protocol_test: OK\n");
  return 0;
}
