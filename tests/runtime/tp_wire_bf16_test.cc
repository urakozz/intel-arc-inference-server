// The bf16 wire of tp.cl's TP_WIRE_BF16 (Inkling tp's verify rows, M > 1) as a host twin of the kernel arithmetic on
// random data: tp_send packs the part's fp32 words as bf16 pairs, two a 32-bit word (one lane a pair, low half the
// even k), tp_sum reads the containing word and takes the half k selects, then adds in the fixed order (card 0's part
// first), both parts rounded to bf16 (the own part in the fold, the peer's on the wire) so both cards hold the same
// bits. Checked: both cards' sums are bitwise equal, each equals the directly computed bf16(card 0) + bf16(card 1),
// and the packed wire occupies half the slot's data words, clear of the trailers and flags (the layout is the fp32
// wire's, so a list may mix the forms). The protocol itself (flags, stale, slots) is tp_protocol_test's.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "runtime/tp_protocol.h"

namespace {
using namespace runtime::tp;
constexpr uint32_t kHidden = 2304, kWireFoldG = 9;

float gen(uint32_t call, uint32_t m, uint32_t k) {
  uint32_t h = call * 0x9E3779B1u ^ (m + 1) * 0x85EBCA77u ^ (m * kHidden + k) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 13;
  return float(int32_t(h & 0xFFFFF) - 0x80000) * 1e-4f;
}

// tp_send's packed store, one pair a lane: the word's low half is the even k's bf16, the high half the odd k's.
uint32_t pack(float even, float odd) {
  return uint32_t(common::f32_to_bf16(odd)) << 16 | common::f32_to_bf16(even);
}

// tp_sum's packed read of element e's bf16.
float unpack(uint32_t w, uint32_t k) {
  return common::bf16_to_f32(uint16_t((k & 1u) ? (w >> 16) : (w & 0xFFFFu)));
}
}  // namespace

int main() {
  const Layout L{kHidden, 1, kWireFoldG};   // the landing as the host sizes it: the fp32 word capacity
  for (const uint32_t M : {2u, 3u, 8u}) {
    for (uint32_t call = 0; call < 4; ++call) {
      const uint32_t slot = call & 1u;
      std::vector<float> part[2];
      for (uint32_t r = 0; r < 2; ++r) {
        part[r].resize(size_t(M) * kHidden);
        for (uint32_t m = 0; m < M; ++m)
          for (uint32_t k = 0; k < kHidden; ++k) part[r][size_t(m) * kHidden + k] = gen(call + r * 7, m, k);
      }
      // Each card's send into the other's landing slot: the packed words, then (as the kernel publishes them) the
      // trailer line and the flag - the twin only needs the data's words here.
      std::vector<uint32_t> wire[2];
      for (uint32_t r = 0; r < 2; ++r) {
        wire[r].assign(L.data_words(), 0);   // the slot keeps its fp32 capacity; the wire uses the low half
        for (uint32_t m = 0; m < M; ++m)
          for (uint32_t k = 0; k + 1 < kHidden; k += 2) {
            const size_t e = size_t(m) * kHidden + k;
            wire[r][e >> 1] = pack(part[r][e], part[r][e + 1]);
          }
      }
      // tp_sum on each card: its own part rounded to bf16 and the peer's packed words, card 0's part first - so both
      // cards sum the same two bf16 values in the same order.
      std::vector<float> sum[2];
      for (uint32_t rank = 0; rank < 2; ++rank) {
        const auto& mine = part[rank];
        const auto& remote_wire = wire[1 - rank];
        sum[rank].resize(size_t(M) * kHidden);
        for (uint32_t m = 0; m < M; ++m)
          for (uint32_t k = 0; k < kHidden; ++k) {
            const size_t e = size_t(m) * kHidden + k;
            const float local = common::bf16_to_f32(common::f32_to_bf16(mine[e]));
            const float remote = unpack(remote_wire[e >> 1], k);
            float acc = 0.f;
            if (rank == 0)
              acc = local + remote;
            else
              acc = remote + local;
            sum[rank][e] = acc;
          }
      }
      CHECK_EQ(sum[0].size(), sum[1].size());
      CHECK(std::memcmp(sum[0].data(), sum[1].data(), sum[0].size() * 4) == 0);   // equal on both cards
      for (uint32_t r = 0; r < 2; ++r)
        for (size_t e = 0; e < sum[r].size(); ++e) {
          const float expected = common::bf16_to_f32(common::f32_to_bf16(part[0][e])) +
                                 common::bf16_to_f32(common::f32_to_bf16(part[1][e]));
          uint32_t a, b;
          std::memcpy(&a, &sum[r][e], 4);
          std::memcpy(&b, &expected, 4);
          CHECK_EQ(a, b);   // the packed path is the direct bf16(card 0) + bf16(card 1), exactly
        }
      // The wire's words and the layout: half the slot's data words, below the first trailer.
      const size_t wire_words = size_t(M) * kHidden / 2;
      CHECK_EQ(wire_words * 2, size_t(M) * kHidden);
      CHECK(size_t(slot) * L.data_words() + wire_words <= L.trailer(0, 0));
      CHECK_EQ(L.data_words(), size_t(kMRows) * kHidden);   // the fp32 capacity the host allocates
    }
    std::printf("M=%u: the bf16 wire's sums equal on both cards and match the direct twin; %zu of %zu data words\n", M,
                size_t(M) * kHidden / 2, L.data_words());
  }
  std::printf("tp_wire_bf16_test PASS\n");
  return 0;
}
