#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "runtime/inkling2_prefill_test_access.h"

int main(int argc, char** argv) {
  CHECK(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "repeat"));
  namespace ink = runtime::inkling2;
  using D = model::Inkling2Desc;
  l0::Context ctx(0);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  ink::Engine engine(ctx, checkpoint, 5056, true, ink::kPrefillRows);
  auto copy = l0::CmdList::immediate(ctx);
  l0::Mem staging(ctx, l0::MemKind::Host, 32ul << 20);
  const auto download = [&](const l0::Mem& src, std::span<std::byte> dst) {
    CHECK_EQ(src.size(), dst.size());
    for (size_t off = 0; off < dst.size(); off += staging.size()) {
      const size_t bytes = std::min(staging.size(), dst.size() - off);
      copy.copy(staging.ptr(), static_cast<const std::byte*>(src.ptr()) + off, bytes);
      std::memcpy(dst.data() + off, staging.ptr(), bytes);
    }
  };
  const auto mem = ink::PrefillTestAccess::state(engine);
  constexpr std::array names{"full k", "full v", "ring k", "ring v", "conv k", "conv v", "conv attention",
                             "conv MoE", "drafter full k", "drafter full v", "drafter ring k", "drafter ring v"};
  std::array<std::vector<std::byte>, names.size()> baseline;
  for (size_t i = 0; i < mem.size(); ++i) baseline[i].resize(mem[i]->size());
  std::vector<std::byte> actual;
  std::vector<float> logits(D::vocab_used), split(D::vocab_used);
  for (auto [length, cap] : {std::pair{1500u, 1500u}, std::pair{5000u, 512u}}) {
    if (argc == 3 && length != 1500) break;
    std::vector<uint32_t> ids(length);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = 20 + (i * 131) % 509;
    engine.reset();
    std::fprintf(stderr, "Checking %u IDs with chunks of %u\n", length, ink::kPrefillRows);
    const auto expected = engine.prefill(ids, ink::kPrefillRows);
    engine.read_prefill_logits(logits);
    for (size_t i = 0; i < mem.size(); ++i)
      download(*mem[i], baseline[i]);

    engine.reset();
    std::fprintf(stderr, "Checking %u IDs with chunks of %u\n", length, cap);
    const auto next = engine.prefill(ids, cap);
    engine.read_prefill_logits(split);
    // Find the first persistent divergence before checking derived logits.
    bool same = true;
    for (size_t i = 0; i < mem.size(); ++i) {
      actual.resize(mem[i]->size());
      download(*mem[i], actual);
      const auto mismatch = std::mismatch(baseline[i].begin(), baseline[i].end(), actual.begin());
      if (mismatch.first == baseline[i].end()) continue;
      same = false;
      const size_t byte = size_t(mismatch.first - baseline[i].begin());
      if (i < 4) {
        constexpr size_t row_bytes = D::kv_n * sizeof(uint16_t);
        const uint32_t stride = i < 2 ? engine.max_len() : ink::kRing;
        const size_t slot = byte / (stride * row_bytes);
        const auto visit = std::find_if(checkpoint.desc().visits.begin(), checkpoint.desc().visits.end(),
            [&](const auto& v) { return v.kv_slot == slot && bool(v.window) == (i >= 2); });
        uint16_t expected_word, actual_word;
        std::memcpy(&expected_word, baseline[i].data() + (byte & ~size_t(1)), sizeof(expected_word));
        std::memcpy(&actual_word, actual.data() + (byte & ~size_t(1)), sizeof(actual_word));
        std::fprintf(stderr, "%s differs at visit %u, row %zu, column %zu (%04x vs %04x)\n",
                     names[i], visit->conv_slot, byte / row_bytes % stride, byte % row_bytes / 2,
                     unsigned(expected_word), unsigned(actual_word));
      } else {
        std::fprintf(stderr, "%s differs at byte %zu\n", names[i], byte);
      }
    }
    CHECK(same);
    for (size_t i = 0; i < logits.size(); ++i)
      CHECK_EQ(std::bit_cast<uint32_t>(logits[i]), std::bit_cast<uint32_t>(split[i]));
    CHECK_EQ(next, expected);
    CHECK_EQ(engine.position(), length);
    std::printf("%u tokens: chunks %u and %u agree bitwise on state and logits\n", length, cap, ink::kPrefillRows);
  }
}
